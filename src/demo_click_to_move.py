"""Click a spot on the plate; the robot goes there.

This is the payoff demo - it exercises every piece at once: ArUco gives the
plate frame, the touch calibration maps that into robot coordinates, IK checks
the point is reachable, and the arm moves.  If this works, the whole chain is
sound.

    python -m src.demo_click_to_move                 # hover 20 mm above
    python -m src.demo_click_to_move --touch         # actually go down to it
    python -m src.demo_click_to_move --simulate      # no robot needed

Keys:  click = move there   h = home   q = quit
Safety: the arm sweeps a wide arc on 'h'. Keep hands and coffee out of the way.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.aruco import PlateBoard
from src.camera import Camera
from src.dobot_arm import connect
from src.kinematics import UnreachableError
from src.transforms import PlateToRobot

WINDOW = "click to move - q to quit"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hover", type=float, default=20.0,
                    help="mm above the plate surface to stop at")
    ap.add_argument("--touch", action="store_true",
                    help="go all the way down to the plate instead of hovering")
    ap.add_argument("--speed", type=float, default=40.0, help="percent")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()

    board = PlateBoard.load()
    tf = PlateToRobot.load()
    print(f"plate transform: {tf.rotation_deg:+.2f} deg, scale {tf.scale:.4f}, "
          f"z {tf.plate_z:.1f} mm, rms {tf.rms_mm:.2f} mm")

    z_above = 0.0 if args.touch else args.hover
    click: list[tuple[int, int]] = []
    last: dict = {}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            click.append((x, y))

    with Camera(args.camera) as cam, connect(simulate=args.simulate or None) as arm:
        arm.speed(args.speed, args.speed)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, on_mouse)

        while True:
            frame = cam.read(flush=1)
            vis = frame.copy()
            try:
                H, found = board.homography(frame)
                board.draw(vis, found)
                msg, colour = f"{len(found)} markers - click anywhere", (0, 200, 0)
            except RuntimeError as exc:
                H = None
                msg, colour = str(exc)[:60], (0, 0, 255)

            while click:
                px = click.pop()
                if H is None:
                    print("  no plate visible, ignoring click")
                    continue
                plate = board.apply(H, np.array(px, float))[0]
                target = tf.to_xyz(plate, z_above=z_above)
                try:
                    arm.move_to(*target)
                    last = {"px": px, "plate": plate, "target": target, "ok": True}
                    print(f"  px{px} -> plate ({plate[0]:6.1f}, {plate[1]:6.1f}) "
                          f"-> robot ({target[0]:6.1f}, {target[1]:6.1f}, "
                          f"{target[2]:6.1f})")
                except UnreachableError as exc:
                    last = {"px": px, "plate": plate, "target": target,
                            "ok": False, "why": str(exc)}
                    print(f"  px{px} unreachable: {exc}")

            if last:
                c = (0, 255, 0) if last["ok"] else (0, 0, 255)
                cv2.drawMarker(vis, tuple(map(int, last["px"])), c,
                               cv2.MARKER_CROSS, 24, 2)
                t = last["target"]
                cv2.putText(vis, f"({t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f}) mm",
                            (int(last["px"][0]) + 14, int(last["px"][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
                if not last["ok"]:
                    cv2.putText(vis, last["why"][:58], (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            mode = "TOUCH" if args.touch else f"hover {args.hover:.0f} mm"
            cv2.putText(vis, f"{msg}   [{mode}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.imshow(WINDOW, vis)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                print("  homing - stand clear")
                arm.home()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
