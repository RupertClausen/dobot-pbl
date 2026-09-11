"""End-to-end check of pixel -> plate -> robot, on a synthetically rendered board.

Rendering a board we already know the geometry of means the expected answer is
known exactly, so this catches a sign flip or a transposed corner order that
the unit tests would let through.

Run with: python -m pytest tests -q
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.aruco import PlateBoard, detect, identify_dictionary, make_detector
from src.identify_plate import guess_layout
from src.make_board import render
from src.transforms import PlateToRobot, pixel_to_robot


@pytest.fixture(scope="module")
def board() -> PlateBoard:
    return PlateBoard(marker_mm=20.0, gap_mm=5.0, ids=list(range(9)),
                      dict_name="DICT_4X4_50")


@pytest.fixture(scope="module")
def image(board: PlateBoard) -> np.ndarray:
    return render(board, dpi=200, margin_mm=10.0)


def test_dictionary_is_identified(image):
    top_name, top_count, ids = identify_dictionary(image)[0]
    assert top_count == 9
    assert top_name.startswith("DICT_4X4")
    assert ids == list(range(9))


def test_layout_comes_back_row_major(image, board):
    found = detect(image, make_detector(board.dict_name))
    ids, pitch_px = guess_layout(found)
    assert ids == list(range(9))
    # 200 dpi, 25 mm pitch -> 25 / 25.4 * 200 px
    assert pitch_px == pytest.approx(25.0 / 25.4 * 200, rel=0.02)


def test_homography_recovers_plate_millimetres(image, board):
    H, usable = board.homography(image)
    obj = board.object_points()
    assert len(usable) == 9
    err = np.concatenate([
        np.linalg.norm(board.apply(H, c) - obj[mid], axis=1)
        for mid, c in usable.items()])
    assert err.max() < 0.5, f"worst corner off by {err.max():.3f} mm"


def test_full_chain_lands_on_the_right_millimetre(image, board):
    """Push a known marker centre all the way through to robot coordinates."""
    H, usable = board.homography(image)

    angle = np.radians(25.0)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    offset = np.array([200.0, -30.0])
    plate_pts = np.array([[0, 0], [70, 0], [70, 70], [0, 70], [35, 35]], float)
    tf = PlateToRobot.from_points(plate_pts, (R @ plate_pts.T).T + offset,
                                  plate_z=-8.0)

    got = pixel_to_robot(H, tf, usable[4].mean(axis=0), z_above=10.0)
    truth_plate = board.object_points()[4].mean(axis=0)
    truth = np.array([*(R @ truth_plate + offset), -8.0 + 10.0])
    assert np.linalg.norm(got - truth) < 0.5


def test_homography_fails_loudly_when_the_plate_is_hidden(board):
    """A blank frame must raise a useful message, not return a garbage matrix."""
    with pytest.raises(RuntimeError, match="visible"):
        board.homography(np.zeros((400, 400, 3), np.uint8))


def test_wrong_dictionary_finds_nothing(image):
    """Guessing the dictionary wrong gives silence - which is why we scan."""
    assert detect(image, make_detector("DICT_6X6_250")) == {}
