"""Camera helpers: opening the right device and keeping frames fresh.

The one trap worth knowing about: V4L2 buffers a few frames, so a naive
`cap.read()` inside a slow loop hands you an image from a second ago.  For a
robot that means grabbing at where the marker *was*.  `Camera.read()` flushes
the buffer first.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np


def list_cameras(max_index: int = 8) -> list[int]:
    """Indices that actually deliver a frame.

    On a laptop /dev/video0..3 often all exist but only some are real capture
    nodes - the rest are metadata streams that open fine and then return
    nothing.  Probing only existing device nodes keeps V4L2 from logging a wall
    of warnings about indices that were never there.
    """
    found = []
    for i in range(max_index):
        if not os.path.exists(f"/dev/video{i}"):
            continue
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened() and cap.read()[0]:
            found.append(i)
        cap.release()
    return found


class Camera:
    """A V4L2 capture device with sane defaults for marker detection."""

    def __init__(self, index: int | None = None, width: int = 1280,
                 height: int = 720, fps: int = 30, autofocus: bool = False):
        if index is None:
            index = int(os.environ.get("CAMERA_INDEX", "0"))
        self.index = index
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {index}. Available: {list_cameras()}. "
                "If this list is empty inside the container, check that /dev is "
                "mounted and the camera is not already in use on the host.")
        # MJPG first: at 720p+ the YUYV fallback is capped around 5 fps.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Autofocus hunting changes the intrinsics mid-session and silently
        # invalidates your calibration, so it is off unless you ask for it.
        if not autofocus:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        time.sleep(0.3)                       # let auto-exposure settle
        for _ in range(5):
            self.cap.read()

    @property
    def size(self) -> tuple[int, int]:
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def read(self, flush: int = 2) -> np.ndarray:
        """Return the freshest frame, discarding `flush` stale buffered ones."""
        for _ in range(flush):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed - was the cable pulled?")
        return frame

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


if __name__ == "__main__":
    print("cameras that deliver frames:", list_cameras())
    with Camera() as cam:
        print(f"camera {cam.index} at {cam.size}, frame {cam.read().shape}")
