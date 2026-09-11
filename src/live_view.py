"""Live camera view with ArUco overlay.

Two ways to watch, because X11 out of a container is the flakiest part of this
whole setup:

    python -m src.live_view              # OpenCV window on your desktop
    python -m src.live_view --stream     # http://localhost:5000 in a browser

The browser mode is worth knowing about for the workshop: everyone can open the
same URL on their own laptop and watch what the robot sees, and it works when
X11 does not.

Keys in window mode:  s = save frame   r = reset FPS   q = quit
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.aruco import PlateBoard, PlateTracker
from src.camera import Camera
from src.overlay import hud
from src.transforms import PlateToRobot

DATA = Path(__file__).resolve().parent.parent / "data"


def annotate(frame: np.ndarray, tracker: PlateTracker,
             tf: PlateToRobot | None, fps: float) -> np.ndarray:
    """Draw markers, the plate axes and a live readout. Returns a new image."""
    board = tracker.board
    vis = frame.copy()
    try:
        H, found = tracker.update(frame)
        board.draw(vis, found, stale=not tracker.fresh)

        # Plate axes, 40 mm long, projected back into the image.
        Hi = np.linalg.inv(H)
        axes = cv2.perspectiveTransform(
            np.array([[[0, 0]], [[40, 0]], [[0, 40]]], float), Hi).reshape(-1, 2)
        o, ax, ay = axes.astype(int)
        cv2.arrowedLine(vis, tuple(o), tuple(ax), (0, 0, 255), 3, tipLength=0.2)
        cv2.arrowedLine(vis, tuple(o), tuple(ay), (0, 255, 0), 3, tipLength=0.2)
        cv2.putText(vis, "X", tuple(ax + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        cv2.putText(vis, "Y", tuple(ay + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        lines = [tracker.status]
        if tf is not None:
            c = board.apply(H, np.array([[vis.shape[1] / 2, vis.shape[0] / 2]]))[0]
            r = tf(c)
            lines.append(f"centre: plate ({c[0]:6.1f}, {c[1]:6.1f}) mm")
            lines.append(f"        robot ({r[0]:6.1f}, {r[1]:6.1f}) mm")
        colour = (0, 255, 0) if tracker.fresh else (0, 190, 255)
    except RuntimeError as exc:
        lines, colour = [str(exc)[:64]], (0, 0, 255)

    lines.append(f"{fps:4.1f} fps")
    return hud(vis, lines, colours=[colour])


def run_window(cam: Camera, tracker: PlateTracker,
               tf: PlateToRobot | None) -> None:
    cv2.namedWindow("dobot live", cv2.WINDOW_NORMAL)
    times: deque[float] = deque(maxlen=30)
    while True:
        t0 = time.time()
        vis = annotate(cam.read(flush=1), tracker, tf,
                       len(times) / sum(times) if times else 0.0)
        cv2.imshow("dobot live", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            DATA.mkdir(exist_ok=True)
            p = DATA / f"frame_{int(time.time())}.png"
            cv2.imwrite(str(p), vis)
            print(f"saved {p}")
        if key == ord("r"):
            times.clear()
        times.append(time.time() - t0)
    cv2.destroyAllWindows()


def run_stream(cam: Camera, tracker: PlateTracker, tf: PlateToRobot | None,
               port: int = 5000) -> None:
    from flask import Flask, Response

    app = Flask(__name__)

    def frames():
        times: deque[float] = deque(maxlen=30)
        while True:
            t0 = time.time()
            vis = annotate(cam.read(flush=1), tracker, tf,
                           len(times) / sum(times) if times else 0.0)
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
            times.append(time.time() - t0)

    @app.route("/")
    def index():
        return ('<body style="margin:0;background:#111;text-align:center">'
                '<img src="/stream" style="max-width:100%;height:auto"></body>')

    @app.route("/stream")
    def stream():
        return Response(frames(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    print(f"\n  open http://localhost:{port} in a browser (Ctrl-C to stop)\n")
    app.run(host="0.0.0.0", port=port, threaded=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stream", action="store_true",
                    help="serve MJPEG over HTTP instead of opening a window")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--camera", type=int, default=None)
    args = ap.parse_args()

    tracker = PlateTracker(PlateBoard.load())
    try:
        tf = PlateToRobot.load()
        print(f"plate transform loaded (rms {tf.rms_mm:.2f} mm)")
    except FileNotFoundError:
        tf = None
        print("no plate transform yet - showing plate mm only")

    with Camera(args.camera) as cam:
        print(f"camera {cam.index} at {cam.size}")
        if args.stream:
            run_stream(cam, tracker, tf, args.port)
        else:
            run_window(cam, tracker, tf)


if __name__ == "__main__":
    main()
