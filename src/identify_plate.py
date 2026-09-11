"""First thing to run with the plate under the camera.

Answers the three questions you need before anything else works:

  1. which ArUco dictionary is printed on the plate,
  2. which marker ids are there and how they are arranged,
  3. how big the markers and the gaps are, in millimetres.

(3) comes from you measuring ONE marker with a ruler; the script works out the
gap from the detected geometry.  It then writes calib/board.json, which every
other script loads.

    python -m src.identify_plate --marker-mm 20
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.aruco import PlateBoard, centers, detect, identify_dictionary, make_detector
from src.camera import Camera


def guess_layout(found: dict[int, np.ndarray]) -> tuple[list[int], float]:
    """Sort markers into row-major order and estimate centre-to-centre pitch.

    Rows are grouped by Y with a tolerance of half the typical marker size, so a
    slightly rotated or tilted plate still groups correctly.
    """
    ctr = centers(found)
    ids = list(ctr)
    pts = np.array([ctr[i] for i in ids])
    side = np.mean([np.linalg.norm(c[0] - c[1]) for c in found.values()])

    order = np.argsort(pts[:, 1])
    rows, current = [], [order[0]]
    for prev, cur in zip(order, order[1:]):
        if abs(pts[cur, 1] - pts[prev, 1]) > side * 0.5:
            rows.append(current)
            current = []
        current.append(cur)
    rows.append(current)

    ordered = []
    for row in rows:
        ordered += [ids[k] for k in sorted(row, key=lambda k: pts[k, 0])]

    # Pitch = median nearest-neighbour distance, in pixels.
    d = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return ordered, float(np.median(d.min(axis=1)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--marker-mm", type=float, default=None,
                    help="measured side of one BLACK marker square, in mm")
    ap.add_argument("--camera", default=None,
                    help="camera index, or a name matched under "
                         "/dev/v4l/by-id (e.g. C270). Defaults to "
                         "$CAMERA_INDEX.")
    ap.add_argument("--image", type=str, default=None,
                    help="analyse a saved image instead of the live camera")
    ap.add_argument("--save", action="store_true",
                    help="write calib/board.json (needs --marker-mm)")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"could not read {args.image}")
    else:
        with Camera(args.camera) as cam:
            # Say which camera this is. Silently capturing from the laptop's
            # built-in webcam instead of the overhead one looks exactly like
            # "no markers detected", and there is otherwise nothing to see.
            print(f"capturing from camera index {cam.index} at "
                  f"{cam.size[0]}x{cam.size[1]}")
            frame = cam.read()
        cv2.imwrite("data/plate_snapshot.png", frame)
        print("snapshot -> data/plate_snapshot.png")

    print("\n=== scanning every ArUco dictionary ===")
    results = identify_dictionary(frame)
    if not results:
        raise SystemExit(
            "No markers detected in any dictionary.\n"
            "  - is the plate actually in frame and in focus?\n"
            "  - glare on glossy plastic is the usual culprit: tilt the plate,\n"
            "    or move the light off the specular angle\n"
            "  - check data/plate_snapshot.png to see what the camera saw\n"
            "  - and confirm the index above is the overhead camera, not the\n"
            "    laptop's built-in one: `python -m src.camera` lists them")

    for name, count, ids in results:
        print(f"  {name:<24} {count} marker(s)  ids={ids}")

    best_name, best_count, _ = results[0]
    # Smaller dictionaries are nested in larger ones; the smallest that finds
    # everything is the one actually printed on the plate.
    same = [r for r in results if r[1] == best_count]
    best_name = min(same, key=lambda r: DICT_SIZE_KEY(r[0]))[0]
    print(f"\n-> using {best_name}")

    found = detect(frame, make_detector(best_name))
    ids, pitch_px = guess_layout(found)
    side_px = np.mean([np.linalg.norm(c[0] - c[1]) for c in found.values()])
    print(f"   ids row-major : {ids}")
    print(f"   marker side   : {side_px:.1f} px")
    print(f"   centre pitch  : {pitch_px:.1f} px")
    print(f"   gap / marker  : {(pitch_px - side_px) / side_px:.3f}")

    if len(found) != 9:
        print(f"\nNOTE: saw {len(found)} markers, expected 9. The board helpers "
              "still work with a partial view, but calibrate with all nine in "
              "frame if you can.")

    if args.marker_mm:
        mm_per_px = args.marker_mm / side_px
        gap_mm = (pitch_px - side_px) * mm_per_px
        print(f"\n   scale         : {mm_per_px:.4f} mm/px at this distance")
        print(f"   gap           : {gap_mm:.1f} mm")
        board = PlateBoard(marker_mm=args.marker_mm, gap_mm=round(gap_mm, 1),
                           ids=ids[:9] if len(ids) >= 9 else ids + [-1] * (9 - len(ids)),
                           dict_name=best_name)
        if args.save:
            print(f"\nwrote {board.save()}")
        else:
            print("\n(pass --save to write calib/board.json)")
    else:
        print("\nMeasure one black marker square with a ruler and re-run with "
              "--marker-mm <value> --save")

    vis = frame.copy()
    PlateBoard(dict_name=best_name, ids=ids[:9] if len(ids) >= 9
               else list(range(9))).draw(vis, found)
    cv2.imwrite("data/plate_detected.png", vis)
    print("annotated -> data/plate_detected.png")


def DICT_SIZE_KEY(name: str) -> int:
    """Sort key preferring small dictionaries and real ArUco over AprilTag."""
    for n in (50, 100, 250, 1000):
        if name.endswith(str(n)):
            return n
    return 5000


if __name__ == "__main__":
    main()
