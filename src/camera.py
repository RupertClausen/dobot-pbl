"""Camera helpers: opening the right device and keeping frames fresh.

The one trap worth knowing about: V4L2 buffers a few frames, so a naive
`cap.read()` inside a slow loop hands you an image from a second ago.  For a
robot that means grabbing at where the marker *was*.  `Camera.read()` flushes
the buffer first.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np


def resolve_index(name: str) -> int:
    """Camera index for a device matched by name, e.g. ``"C270"``.

    V4L2 index numbers shuffle on replug and between boots - the overhead
    camera can be 4 today and 2 tomorrow, which silently points every script at
    the laptop's built-in webcam instead. `/dev/v4l/by-id/` symlinks are stable
    per physical device, so match on those.
    """
    by_id = Path("/dev/v4l/by-id")
    if not by_id.is_dir():
        raise RuntimeError("/dev/v4l/by-id is not available; use a numeric index")
    hits = sorted(l for l in by_id.iterdir()
                  if name.lower() in l.name.lower() and l.name.endswith("index0"))
    if not hits:
        have = sorted(l.name for l in by_id.iterdir() if l.name.endswith("index0"))
        raise RuntimeError(f"no camera matching {name!r}. Present: {have}")
    if len(hits) > 1:
        raise RuntimeError(f"{name!r} matches more than one camera: "
                           f"{[h.name for h in hits]}")
    return int(str(hits[0].resolve()).rsplit("video", 1)[1])


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

    def __init__(self, index: int | str | None = None, width: int = 1280,
                 height: int = 720, fps: int = 30, autofocus: bool = False):
        """`index` may be a V4L2 number, or a name to match under
        /dev/v4l/by-id (``Camera("C270")``). `CAMERA_INDEX` accepts either."""
        if index is None:
            index = os.environ.get("CAMERA_INDEX", "0")
        if isinstance(index, str) and not index.lstrip("-").isdigit():
            index = resolve_index(index)
        index = int(index)
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
    by_id = Path("/dev/v4l/by-id")
    if by_id.is_dir():
        print("stable names (use these, indices move on replug):")
        for link in sorted(by_id.iterdir()):
            if link.name.endswith("index0"):
                print(f"  {link.name}  ->  {link.resolve().name}")
    with Camera() as cam:
        print(f"\nopened camera {cam.index} at {cam.size}, "
              f"frame {cam.read().shape}")
