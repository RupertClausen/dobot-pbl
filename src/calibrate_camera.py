"""Camera intrinsics from a printed chessboard.

You only need this for `PlateBoard.pose` (full 6-DoF) or if you want to
undistort frames.  The pixel -> plate -> robot chain used by the demos is
purely planar and works fine without it, so skip this unless you need it.

Print a chessboard, tape it flat, then:

    python -m src.calibrate_camera --cols 9 --rows 6 --square-mm 25

`--cols`/`--rows` count INNER corners, not squares: a board with 10x7 squares
has 9x6 inner corners.  Capture 15-20 views with the board tilted differently
each time and filling different parts of the frame; all-frontal views give a
confident-looking fit with a badly wrong focal length.

Keys:  space = capture   d = drop last   c = calibrate   q = quit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"
INTRINSICS = CALIB_DIR / "intrinsics.json"


def load_intrinsics(path: Path | str = INTRINSICS) -> tuple[np.ndarray, np.ndarray]:
    """``(K, dist)``. Raises if the camera has not been calibrated."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m src.calibrate_camera` first")
    d = json.loads(path.read_text())
    return np.array(d["K"]), np.array(d["dist"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=9, help="inner corners across")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--camera", type=int, default=None)
    args = ap.parse_args()

    pattern = (args.cols, args.rows)
    # One canonical board in its own frame; scaling by square size puts the
    # resulting translations in millimetres.
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_mm

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    with Camera(args.camera) as cam:
        size = cam.size
        cv2.namedWindow("chessboard", cv2.WINDOW_NORMAL)
        while True:
            frame = cam.read(flush=1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ok, corners = cv2.findChessboardCorners(
                gray, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                + cv2.CALIB_CB_FAST_CHECK)
            vis = frame.copy()
            if ok:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           criteria)
                cv2.drawChessboardCorners(vis, pattern, corners, ok)

            cv2.putText(vis, f"captured {len(obj_points)}   "
                             f"{'BOARD FOUND' if ok else 'searching...'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 0) if ok else (0, 0, 255), 2)
            cv2.putText(vis, "space=capture  d=drop  c=calibrate  q=quit",
                        (10, vis.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            cv2.imshow("chessboard", vis)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            if key == ord(" ") and ok:
                obj_points.append(objp.copy())
                img_points.append(corners)
                print(f"captured {len(obj_points)}")
            if key == ord("d") and obj_points:
                obj_points.pop(), img_points.pop()
                print(f"dropped, {len(obj_points)} left")
            if key == ord("c"):
                if len(obj_points) < 8:
                    print(f"want at least 8 views, have {len(obj_points)}")
                    continue
                break
        cv2.destroyAllWindows()

    print(f"\ncalibrating on {len(obj_points)} views...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, size, None, None)

    # Reprojection error per view: one bad view can drag the whole fit.
    per_view = []
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        per_view.append(float(cv2.norm(img_points[i], proj, cv2.NORM_L2)
                              / len(proj)))

    print(f"  overall rms   : {rms:.4f} px")
    print(f"  worst view    : {max(per_view):.4f} px")
    print(f"  focal length  : fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f} px")
    print(f"  principal pt  : ({K[0, 2]:.1f}, {K[1, 2]:.1f}) px "
          f"(image centre is ({size[0] / 2:.0f}, {size[1] / 2:.0f}))")
    if rms > 1.0:
        print("  WARNING: rms above 1 px. Drop the worst views and recapture "
              "with the board tilted more.")

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    INTRINSICS.write_text(json.dumps({
        "K": K.tolist(), "dist": dist.tolist(), "rms_px": float(rms),
        "image_size": list(size), "n_views": len(obj_points),
        "per_view_px": per_view,
    }, indent=2))
    print(f"\nwrote {INTRINSICS}")


if __name__ == "__main__":
    main()
