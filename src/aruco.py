"""ArUco detection and the plate coordinate frame.

The plate carries a 3x3 block of markers in one corner.  Those markers define a
frame fixed to the plate, so if somebody nudges the plate the robot can find it
again instead of driving to where it used to be.

Two ways to use the markers, both here:

  `PlateBoard.homography` - image pixels -> plate millimetres, planar, needs no
      camera calibration at all.  This is the one to use today.
  `PlateBoard.pose`       - full 6-DoF board pose via solvePnP, needs camera
      intrinsics from `calibrate_camera.py`.  Use it if the camera looks at the
      plate from a steep angle or you want the plate's tilt.

OpenCV renamed most of the aruco API in 4.7; this module uses the new
ArucoDetector class and works on 4.7+.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"

# Every predefined dictionary, ordered so the common ones are tried first.
DICTS: dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def make_detector(dict_name: str = "DICT_4X4_50") -> cv2.aruco.ArucoDetector:
    """Detector tuned a little for printed-on-plastic markers."""
    if dict_name not in DICTS:
        raise KeyError(f"unknown dictionary {dict_name!r}; one of {list(DICTS)}")
    d = cv2.aruco.getPredefinedDictionary(DICTS[dict_name])
    p = cv2.aruco.DetectorParameters()
    # Subpixel corners matter: a half-pixel corner error at 1 m is ~0.5 mm of
    # robot error, and it is free.
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    # Glossy plastic under room lighting tends to blow out the white quiet zone;
    # a wider adaptive-threshold sweep recovers markers a fixed threshold loses.
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 43
    p.adaptiveThreshWinSizeStep = 8
    p.minMarkerPerimeterRate = 0.01
    return cv2.aruco.ArucoDetector(d, p)


def detect(frame: np.ndarray, detector: cv2.aruco.ArucoDetector
           ) -> dict[int, np.ndarray]:
    """``{marker_id: (4, 2) corner array}``, corners clockwise from top-left."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}


def identify_dictionary(frame: np.ndarray, min_markers: int = 1
                        ) -> list[tuple[str, int, list[int]]]:
    """Try every dictionary and report which ones see markers.

    Run this first on an unknown plate - guessing wrong just gives you an empty
    detection with no error message.  Returns ``(name, count, ids)`` sorted by
    count.  A dictionary nested inside a bigger one (4X4_50 vs 4X4_1000) will
    match too; prefer the smallest that finds all your markers.
    """
    out = []
    for name in DICTS:
        found = detect(frame, make_detector(name))
        if len(found) >= min_markers:
            out.append((name, len(found), sorted(found)))
    return sorted(out, key=lambda t: -t[1])


def centers(found: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Marker id -> centre pixel."""
    return {i: c.mean(axis=0) for i, c in found.items()}


@dataclass
class PlateBoard:
    """A 3x3 grid of markers glued to the plate, and the frame it defines.

    The plate frame has its origin at the top-left corner of marker `ids[0]`,
    +X running along the top row and +Y down the first column, in millimetres.
    Every ArUco-derived measurement in this project lands in that frame.

    marker_mm  side length of one black marker square
    gap_mm     white gap between neighbouring markers
    ids        the nine ids, row-major, top-left first
    """
    marker_mm: float = 20.0
    gap_mm: float = 5.0
    ids: list[int] = field(default_factory=lambda: list(range(9)))
    dict_name: str = "DICT_4X4_50"

    def __post_init__(self) -> None:
        if len(self.ids) != 9:
            raise ValueError(f"expected 9 marker ids, got {len(self.ids)}")
        self.detector = make_detector(self.dict_name)

    @property
    def pitch(self) -> float:
        """Centre-to-centre spacing between neighbouring markers, mm."""
        return self.marker_mm + self.gap_mm

    def object_points(self) -> dict[int, np.ndarray]:
        """``{id: (4, 2) corner positions in plate mm}``, same corner order as
        the detector returns (top-left, top-right, bottom-right, bottom-left)."""
        out = {}
        m, p = self.marker_mm, self.pitch
        for k, mid in enumerate(self.ids):
            ox, oy = (k % 3) * p, (k // 3) * p
            out[mid] = np.array([[ox, oy], [ox + m, oy],
                                 [ox + m, oy + m], [ox, oy + m]], float)
        return out

    # -- planar mapping (no camera calibration needed) ---------------------- #
    def homography(self, frame: np.ndarray, min_markers: int = 2
                   ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """``(H, detections)`` mapping image pixels -> plate mm.

        Uses every visible corner, so a partly occluded grid still works as long
        as two markers survive.  Raises if too few are visible.
        """
        found = detect(frame, self.detector)
        obj = self.object_points()
        usable = {i: c for i, c in found.items() if i in obj}
        if len(usable) < min_markers:
            raise RuntimeError(
                f"only {len(usable)} of the board's markers visible "
                f"(saw ids {sorted(found)}, board expects {sorted(obj)}). "
                "Check lighting, focus, and that dict_name matches the plate.")
        src = np.vstack([usable[i] for i in sorted(usable)])
        dst = np.vstack([obj[i] for i in sorted(usable)])
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            raise RuntimeError("homography failed - markers may be collinear")
        return H, usable

    @staticmethod
    def apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Push ``(N, 2)`` points through a homography."""
        pts = np.asarray(pts, float).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    # -- full 6-DoF pose (needs intrinsics) --------------------------------- #
    def pose(self, frame: np.ndarray, K: np.ndarray, dist: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
        """``(rvec, tvec, detections)`` of the plate frame in camera coordinates."""
        found = detect(frame, self.detector)
        obj = self.object_points()
        usable = {i: c for i, c in found.items() if i in obj}
        if len(usable) < 1:
            raise RuntimeError("no board markers visible")
        img_pts = np.vstack([usable[i] for i in sorted(usable)]).astype(np.float32)
        obj_pts = np.vstack([np.column_stack([obj[i], np.zeros(4)])
                             for i in sorted(usable)]).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("solvePnP failed")
        return rvec, tvec, usable

    def draw(self, frame: np.ndarray, found: dict[int, np.ndarray],
             stale: bool = False) -> np.ndarray:
        """Outline detections and label them. Returns the same array.

        Pass ``stale=True`` when the detections came from a cache rather than
        this frame, so a hidden marker block cannot be mistaken for a live one -
        green means "seen right now", amber means "remembered".
        """
        for mid, c in found.items():
            pts = c.astype(int)
            known = mid in self.ids
            if stale:
                colour = (0, 190, 255)
            else:
                colour = (0, 220, 0) if known else (0, 140, 255)
            cv2.polylines(frame, [pts], True, colour, 1 if stale else 2)
            cv2.circle(frame, tuple(pts[0]), 4, (0, 0, 255), -1)   # corner 0
            # One pass only: putText scales letter spacing with thickness, so
            # the usual thick-black-then-thin-colour outline ghosts sideways.
            ctr = pts.mean(axis=0).astype(int)
            cv2.putText(frame, str(mid), tuple(ctr - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, colour, 2, cv2.LINE_AA)
        return frame

    # -- persistence -------------------------------------------------------- #
    def save(self, path: Path | str = CALIB_DIR / "board.json") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"marker_mm": self.marker_mm, "gap_mm": self.gap_mm,
             "ids": self.ids, "dict_name": self.dict_name}, indent=2))
        return path

    @classmethod
    def load(cls, path: Path | str = CALIB_DIR / "board.json") -> "PlateBoard":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text()))


class PlateTracker:
    """Keeps the last good homography so the arm can block the markers.

    With a fixed overhead camera this is nearly free accuracy.  The camera does
    not move, so a homography computed two seconds ago is still correct - but
    the Dobot reaching over the plate *will* cover the marker block or throw a
    shadow across it, and a bare `PlateBoard.homography` call raises the moment
    that happens.  Caching turns a hard failure into a brief stale reading.

    The cache is invalidated by age, not by movement, so if somebody does slide
    the plate while the arm is parked on top of the markers you get a stale
    frame for up to `max_age_s` before it starts refusing.  That is the right
    trade for a workbench: the plate moves rarely, the arm occludes constantly.
    """

    def __init__(self, board: "PlateBoard", max_age_s: float = 5.0):
        self.board = board
        self.max_age_s = max_age_s
        self._H: np.ndarray | None = None
        self._found: dict[int, np.ndarray] = {}
        self._stamp: float = 0.0
        self._fresh = False

    def update(self, frame: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """``(H, detections)``, from this frame if possible, else the cache.

        Raises `RuntimeError` only when there is no usable homography at all -
        nothing detected now and nothing recent enough to fall back on.
        """
        try:
            self._H, self._found = self.board.homography(frame)
            self._stamp = time.time()
            self._fresh = True
        except RuntimeError:
            self._fresh = False
            if self._H is None:
                raise
            if self.age > self.max_age_s:
                raise RuntimeError(
                    f"plate not visible for {self.age:.1f} s and the cached "
                    "homography has expired - move the arm out of the camera's "
                    "view of the markers") from None
        return self._H, self._found

    @property
    def age(self) -> float:
        """Seconds since the homography was last computed from a live frame."""
        return float("inf") if self._stamp == 0.0 else time.time() - self._stamp

    @property
    def fresh(self) -> bool:
        """True if the most recent `update` saw the markers itself."""
        return self._fresh

    @property
    def status(self) -> str:
        """One-line state, for drawing on a live view."""
        if self._H is None:
            return "no plate lock"
        if self._fresh:
            return f"{len(self._found)}/9 markers"
        return f"markers hidden - cached {self.age:.1f}s ago"
