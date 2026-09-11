"""Generate a printable 3x3 ArUco board.

Handy as a spare: print it at 100% scale (NOT 'fit to page', which silently
rescales and ruins the millimetre figures), measure one marker to confirm, and
you have a replacement plate.

    python -m src.make_board --marker-mm 20 --gap-mm 5 --dpi 300
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.aruco import DICTS, PlateBoard


def render(board: PlateBoard, dpi: int = 300, margin_mm: float = 10.0
           ) -> np.ndarray:
    px_per_mm = dpi / 25.4
    m = int(round(board.marker_mm * px_per_mm))
    p = int(round(board.pitch * px_per_mm))
    edge = int(round(margin_mm * px_per_mm))
    side = 2 * edge + 2 * p + m

    canvas = np.full((side, side), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(DICTS[board.dict_name])
    for k, mid in enumerate(board.ids):
        img = cv2.aruco.generateImageMarker(d, mid, m)
        y, x = edge + (k // 3) * p, edge + (k % 3) * p
        canvas[y:y + m, x:x + m] = img

    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    label = (f"{board.dict_name}  ids {board.ids}  "
             f"marker {board.marker_mm}mm  gap {board.gap_mm}mm  "
             f"print at 100%")
    cv2.putText(canvas, label, (edge, side - edge // 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4 * dpi / 150, (0, 0, 0),
                max(1, dpi // 200), cv2.LINE_AA)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--marker-mm", type=float, default=20.0)
    ap.add_argument("--gap-mm", type=float, default=5.0)
    ap.add_argument("--dict", default="DICT_4X4_50", choices=sorted(DICTS))
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="data/aruco_board_3x3.png")
    args = ap.parse_args()

    board = PlateBoard(marker_mm=args.marker_mm, gap_mm=args.gap_mm,
                       ids=list(range(args.start_id, args.start_id + 9)),
                       dict_name=args.dict)
    img = render(board, args.dpi)
    cv2.imwrite(args.out, img)
    size_mm = 2 * board.pitch + board.marker_mm + 20
    print(f"wrote {args.out}  ({img.shape[1]}x{img.shape[0]} px at {args.dpi} dpi "
          f"= {size_mm:.0f}x{size_mm:.0f} mm)")
    print(f"board spec -> {board.save()}")


if __name__ == "__main__":
    main()
