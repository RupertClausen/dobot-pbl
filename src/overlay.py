"""Readable text on top of a camera frame.

The obvious way to make text readable over a busy image is to draw it twice -
thick black, then thin colour on top. That does not work in OpenCV: `putText`
scales the glyph *advance* with `thickness`, not just the stroke width, so the
two passes have different letter spacing and the outline drifts sideways,
leaving a visible ghost by the end of a long line.

So instead: one translucent panel, one pass of text.
"""
from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def hud(img: np.ndarray, lines: list[str], colours: list[tuple] | None = None,
        origin: tuple[int, int] = (10, 10), scale: float = 0.65,
        pad: int = 8, alpha: float = 0.55) -> np.ndarray:
    """Draw `lines` in a translucent panel. Modifies and returns `img`.

    `colours` is one BGR tuple per line; anything missing falls back to white.
    """
    if not lines:
        return img
    colours = (colours or []) + [(255, 255, 255)] * (len(lines) - len(colours or []))

    sizes = [cv2.getTextSize(t, FONT, scale, 1)[0] for t in lines]
    step = max(h for _, h in sizes) + 12
    w = max(wd for wd, _ in sizes)
    x, y = origin

    panel = img[y:y + step * len(lines) + pad, x:x + w + 2 * pad]
    if panel.size:                      # skip if the origin is off-frame
        panel[:] = cv2.addWeighted(panel, 1 - alpha,
                                   np.zeros_like(panel), alpha, 0)

    for i, (text, colour) in enumerate(zip(lines, colours)):
        cv2.putText(img, text, (x + pad, y + pad + step * i + sizes[i][1]),
                    FONT, scale, colour, 1, cv2.LINE_AA)
    return img


def footer(img: np.ndarray, text: str, scale: float = 0.6) -> np.ndarray:
    """One line of help text along the bottom edge."""
    (w, h), _ = cv2.getTextSize(text, FONT, scale, 1)
    y = img.shape[0] - h - 16
    return hud(img, [text], origin=(10, y), scale=scale)
