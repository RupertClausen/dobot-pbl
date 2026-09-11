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


# --------------------------------------------------------------------------- #
# overhead camera specifics
# --------------------------------------------------------------------------- #
def test_tracker_survives_the_arm_covering_the_markers(image, board):
    """A blocked frame must return the cached homography, not raise."""
    from src.aruco import PlateTracker

    tracker = PlateTracker(board, max_age_s=5.0)
    H_good, _ = tracker.update(image)
    assert tracker.fresh

    blocked = np.zeros_like(image)                  # arm covers everything
    H_cached, _ = tracker.update(blocked)
    assert not tracker.fresh
    assert np.allclose(H_cached, H_good)
    assert "cached" in tracker.status


def test_tracker_gives_up_once_the_cache_is_stale(image, board):
    from src.aruco import PlateTracker

    tracker = PlateTracker(board, max_age_s=0.0)    # expire immediately
    tracker.update(image)
    with pytest.raises(RuntimeError, match="expired"):
        tracker.update(np.zeros_like(image))


def test_tracker_raises_when_it_never_had_a_lock(board):
    from src.aruco import PlateTracker

    with pytest.raises(RuntimeError, match="visible"):
        PlateTracker(board).update(np.zeros((400, 400, 3), np.uint8))


def test_parallax_correction_inverts_the_projection():
    """Project a known base position outward, then correct it back."""
    from src.transforms import parallax_correct

    nadir = np.array([100.0, 80.0])
    H_cam, h_obj = 500.0, 20.0
    true_base = np.array([[250.0, 80.0], [100.0, 80.0], [40.0, 10.0]])

    # Forward model: where the top face appears on the plate plane.
    observed = nadir + (true_base - nadir) * H_cam / (H_cam - h_obj)
    assert np.allclose(parallax_correct(observed, nadir, H_cam, h_obj), true_base)


def test_parallax_is_zero_directly_under_the_camera():
    from src.transforms import parallax_correct

    nadir = np.array([100.0, 80.0])
    assert np.allclose(parallax_correct(nadir, nadir, 500.0, 30.0), nadir)


def test_parallax_of_a_flat_object_changes_nothing():
    from src.transforms import parallax_correct

    pts = np.array([[10.0, 20.0], [200.0, 150.0]])
    assert np.allclose(parallax_correct(pts, [100.0, 80.0], 500.0, 0.0), pts)


def test_parallax_error_has_the_documented_magnitude():
    """The README quotes ~6 mm for a 20 mm block 150 mm off-nadir at 500 mm."""
    from src.transforms import parallax_correct

    nadir = np.array([0.0, 0.0])
    observed = np.array([150.0, 0.0])
    corrected = parallax_correct(observed, nadir, 500.0, 20.0)
    assert np.linalg.norm(observed - corrected) == pytest.approx(6.0, abs=0.1)


def test_parallax_rejects_an_object_taller_than_the_camera():
    from src.transforms import parallax_correct

    with pytest.raises(ValueError, match="must exceed"):
        parallax_correct([10.0, 10.0], [0.0, 0.0], 50.0, 60.0)
