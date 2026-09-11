"""Teach the robot where the plate is: the plate -> robot base transform.

How it works, per sample:

  1. you drag the arm (hold the Unlock button) until the tool tip touches a
     spot on the plate - anywhere, but spread them out;
  2. you click that same spot in the camera window;
  3. the script reads the robot's own XYZ, converts your click to plate
     millimetres through the live ArUco homography, and stores the pair.

Four samples is the minimum, six to eight spread across the whole plate is
better.  Corners matter more than the middle - a transform fitted from points
clustered in one area extrapolates badly.

    python -m src.calibrate_plate

Keys:  click = record   u = undo last   f = finish & fit   q = quit
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.aruco import PlateBoard
from src.camera import Camera
from src.dobot_arm import connect
from src.overlay import footer, hud
from src.transforms import PlateToRobot

WINDOW = "plate calibration - click the tool tip, 'f' to fit"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default=None,
                    help="camera index, or a name matched under "
                         "/dev/v4l/by-id (e.g. C270). Defaults to "
                         "$CAMERA_INDEX.")
    ap.add_argument("--min-samples", type=int, default=4)
    args = ap.parse_args()

    board = PlateBoard.load()
    print(f"board: {board.dict_name}, ids {board.ids}, "
          f"{board.marker_mm} mm markers, {board.gap_mm} mm gaps")

    samples: list[dict] = []
    click: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            click.append((x, y))

    with Camera(args.camera) as cam, connect() as arm:
        if arm.simulate:
            print("\nWARNING: running in simulation - the recorded robot "
                  "positions will be made up. Plug the arm in for a real "
                  "calibration.\n")
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, on_mouse)

        while True:
            frame = cam.read()
            vis = frame.copy()
            try:
                H, found = board.homography(frame)
                board.draw(vis, found)
                status, colour = f"{len(found)} markers", (0, 200, 0)
            except RuntimeError as exc:
                H = None
                status, colour = str(exc)[:60], (0, 0, 255)

            while click:
                px = click.pop()
                if H is None:
                    print("  ignored: no homography this frame")
                    continue
                plate = board.apply(H, np.array(px, float))[0]
                xyz = arm.position
                samples.append({"pixel": px, "plate": plate, "robot": xyz.copy()})
                print(f"  #{len(samples)}  px{px} -> plate "
                      f"({plate[0]:7.2f}, {plate[1]:7.2f}) mm   robot "
                      f"({xyz[0]:7.2f}, {xyz[1]:7.2f}, {xyz[2]:7.2f}) mm")

            for k, s in enumerate(samples, 1):
                cv2.drawMarker(vis, tuple(map(int, s["pixel"])), (255, 0, 255),
                               cv2.MARKER_CROSS, 18, 2)
                cv2.putText(vis, str(k),
                            (int(s["pixel"][0]) + 10, int(s["pixel"][1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            hud(vis, [status, f"samples: {len(samples)}"], colours=[colour])
            footer(vis, "click = record   u = undo   f = fit   q = quit")
            cv2.imshow(WINDOW, vis)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                print("aborted, nothing written")
                break
            if key == ord("u") and samples:
                print(f"  undid #{len(samples)}")
                samples.pop()
            if key == ord("f"):
                if len(samples) < args.min_samples:
                    print(f"  need at least {args.min_samples} samples, "
                          f"have {len(samples)}")
                    continue
                fit_and_report(samples)
                break

        cv2.destroyAllWindows()


def fit_and_report(samples: list[dict]) -> None:
    plate = np.array([s["plate"] for s in samples])
    robot = np.array([s["robot"][:2] for s in samples])
    z = np.array([s["robot"][2] for s in samples])

    tf = PlateToRobot.from_points(plate, robot, plate_z=float(z.mean()))
    pred = tf(plate)
    err = np.linalg.norm(pred - robot, axis=1)

    print("\n--- fit ---")
    print(f"  points        : {tf.n_points}")
    print(f"  rotation      : {tf.rotation_deg:+.2f} deg")
    print(f"  scale         : {tf.scale:.4f}   (should be ~1.000)")
    print(f"  plate z       : {tf.plate_z:.2f} mm  "
          f"(spread {z.max() - z.min():.2f} mm)")
    print(f"  residual rms  : {tf.rms_mm:.2f} mm,  max {err.max():.2f} mm")
    print("\n  per point:")
    for i, (e, s) in enumerate(zip(err, samples), 1):
        print(f"    #{i}  {e:5.2f} mm" + ("   <-- worst" if e == err.max() else ""))

    if abs(tf.scale - 1.0) > 0.02:
        print("\n  WARNING: scale is off by more than 2%. Usually means "
              "marker_mm in calib/board.json does not match the real plate. "
              "Re-measure a marker and re-run identify_plate.")
    if z.max() - z.min() > 3.0:
        print("\n  WARNING: the recorded Z values vary by more than 3 mm, so "
              "the tip was not touching down consistently. plate_z will be off.")
    if tf.rms_mm > 3.0:
        print("\n  WARNING: residual is large. Most likely a click that missed "
              "the tip, or samples bunched in one corner. Check the worst "
              "point above and redo it.")

    print(f"\nwrote {tf.save()}")


if __name__ == "__main__":
    main()
