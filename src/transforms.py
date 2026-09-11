"""The coordinate chain: image pixels -> plate mm -> robot mm.

    pixel  --H (ArUco, recomputed every frame)-->  plate mm
    plate  --T (touch calibration, fixed)------->  robot base mm

Keeping the two stages separate is what makes the system robust: H is re-derived
from the markers on every frame, so bumping the plate or the camera costs you
nothing, while T only depends on where the robot's base is relative to the
plate's own frame - which does not change when the plate slides around.

Fitting a *similarity* (rotation + translation + one scale) rather than a full
affine is deliberate.  Both frames are already in millimetres, so the true
transform is rigid; the fitted scale should come out at 1.000 +- 0.005 and is a
free error check.  A full affine would happily absorb a bad homography into a
shear and hide the problem.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"


def fit_similarity(src: np.ndarray, dst: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares 2D similarity (Umeyama): ``dst ~= s * R @ src + t``.

    Returns ``(R, t, s)``.  Needs at least two non-coincident point pairs; three
    or more spread across the working area is what you actually want.
    """
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != dst.shape or src.shape[0] < 2:
        raise ValueError("need matching point sets with at least 2 pairs")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = (d_c.T @ s_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:      # forbid a mirror solution
        S[1, 1] = -1
    R = U @ S @ Vt
    var = (s_c ** 2).sum() / len(src)
    scale = float((D * np.diag(S)).sum() / var) if var > 0 else 1.0
    t = mu_d - scale * R @ mu_s
    return R, t, scale


@dataclass
class PlateToRobot:
    """Rigid map from the plate frame into the robot's base frame.

    plate_z is the robot Z at which the tool touches the plate surface - the
    number every pick and place is measured down from.
    """
    R: np.ndarray
    t: np.ndarray
    scale: float = 1.0
    plate_z: float = 0.0
    rms_mm: float = 0.0
    n_points: int = 0

    def __call__(self, xy) -> np.ndarray:
        """Plate mm -> robot mm. Accepts one point or an ``(N, 2)`` array."""
        p = np.atleast_2d(np.asarray(xy, float))
        out = (self.scale * (self.R @ p.T)).T + self.t
        return out[0] if np.ndim(xy) == 1 else out

    def to_xyz(self, xy, z_above: float = 0.0) -> np.ndarray:
        """Plate mm -> robot ``[x, y, z]``, `z_above` mm over the plate surface."""
        x, y = self(np.asarray(xy, float).ravel()[:2])
        return np.array([x, y, self.plate_z + z_above])

    def inverse(self, xy) -> np.ndarray:
        """Robot mm -> plate mm."""
        p = np.atleast_2d(np.asarray(xy, float))
        out = (self.R.T @ (p - self.t).T).T / self.scale
        return out[0] if np.ndim(xy) == 1 else out

    @property
    def rotation_deg(self) -> float:
        """How far the plate's X axis is rotated from the robot's, in degrees."""
        return float(np.degrees(np.arctan2(self.R[1, 0], self.R[0, 0])))

    def save(self, path: Path | str = CALIB_DIR / "plate_to_robot.json") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "R": self.R.tolist(), "t": self.t.tolist(), "scale": self.scale,
            "plate_z": self.plate_z, "rms_mm": self.rms_mm,
            "n_points": self.n_points, "rotation_deg": self.rotation_deg,
        }, indent=2))
        return path

    @classmethod
    def load(cls, path: Path | str = CALIB_DIR / "plate_to_robot.json"
             ) -> "PlateToRobot":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run `python -m src.calibrate_plate` first")
        d = json.loads(path.read_text())
        return cls(R=np.array(d["R"]), t=np.array(d["t"]), scale=d["scale"],
                   plate_z=d["plate_z"], rms_mm=d.get("rms_mm", 0.0),
                   n_points=d.get("n_points", 0))

    @classmethod
    def from_points(cls, plate_xy, robot_xy, plate_z: float = 0.0
                    ) -> "PlateToRobot":
        """Fit from matched point lists and record the residual."""
        plate_xy, robot_xy = np.asarray(plate_xy, float), np.asarray(robot_xy, float)
        R, t, s = fit_similarity(plate_xy, robot_xy)
        pred = (s * (R @ plate_xy.T)).T + t
        err = np.linalg.norm(pred - robot_xy, axis=1)
        return cls(R=R, t=t, scale=s, plate_z=plate_z,
                   rms_mm=float(np.sqrt(np.mean(err ** 2))),
                   n_points=len(plate_xy))


def pixel_to_robot(H: np.ndarray, tf: PlateToRobot, pixels,
                   z_above: float = 0.0) -> np.ndarray:
    """The whole chain in one call: image pixels -> robot ``[x, y, z]``."""
    import cv2
    pts = np.asarray(pixels, float).reshape(-1, 1, 2)
    plate = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    xy = tf(plate)
    z = np.full((len(xy), 1), tf.plate_z + z_above)
    out = np.hstack([xy, z])
    return out[0] if np.ndim(pixels) == 1 else out
