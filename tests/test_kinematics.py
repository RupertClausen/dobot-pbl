"""FK/IK consistency checks. Run with: python -m pytest tests -q"""
from __future__ import annotations

import numpy as np
import pytest

from src.kinematics import (DEFAULT_LIMITS, Geometry, UnreachableError, forward,
                            forward_many, inverse, jacobian, interpolate_line)
from src.transforms import PlateToRobot, fit_similarity

GEOM = Geometry()


@pytest.mark.parametrize("joints", [
    (0, 0, 0), (0, 30, 30), (45, 45, 10), (-30, 60, 45),
    (120, 10, 80), (-90, 80, 90), (0, 85, 95), (135, 0, -10),
])
def test_ik_inverts_fk(joints):
    """IK must return exactly the joints FK was given - it is a closed form."""
    xyz = forward(*joints, geom=GEOM)
    back = inverse(*xyz, geom=GEOM, limits=None)
    assert np.allclose(back, joints, atol=1e-6), f"{back} != {joints}"


def test_fk_matches_hand_computation():
    """Guards the angle convention against an accidental sign flip."""
    # J2 = 0 (rear arm straight up), J3 = 0 (forearm horizontal forward).
    xyz = forward(0, 0, 0, geom=GEOM)
    assert xyz[0] == pytest.approx(GEOM.L2 + GEOM.Ltool)      # r = 147 + 61
    assert xyz[1] == pytest.approx(0.0)
    assert xyz[2] == pytest.approx(GEOM.L0 + GEOM.L1)         # z = 138 + 135


def test_forward_many_matches_forward():
    joints = np.array([[0, 10, 20], [45, 30, 40], [-60, 50, 5]], float)
    assert np.allclose(forward_many(joints, GEOM),
                       [forward(*j, geom=GEOM) for j in joints])


def test_base_rotation_is_pure_yaw():
    """Spinning J1 must sweep a circle at constant radius and height."""
    poses = forward_many(np.column_stack(
        [np.linspace(-135, 135, 25), np.full(25, 30.0), np.full(25, 30.0)]), GEOM)
    r = np.hypot(poses[:, 0], poses[:, 1])
    assert np.allclose(r, r[0])
    assert np.allclose(poses[:, 2], poses[0, 2])


def test_out_of_reach_raises():
    with pytest.raises(UnreachableError, match="beyond"):
        inverse(900.0, 0.0, 0.0, geom=GEOM)


def test_joint_limit_enforced():
    """Directly above the base: geometrically fine, but J1/J2 cannot get there."""
    with pytest.raises(UnreachableError):
        inverse(0.0, 0.0, 400.0, geom=GEOM, limits=DEFAULT_LIMITS)


def test_limits_can_be_disabled():
    xyz = forward(0, 85, 95, geom=GEOM)
    inverse(*xyz, geom=GEOM, limits=None)        # must not raise


def test_jacobian_matches_finite_difference():
    j = np.array([30.0, 40.0, 25.0])
    analytic = jacobian(*j, geom=GEOM)
    h = 1e-6
    numeric = np.empty((3, 3))
    for k in range(3):
        step = np.zeros(3)
        step[k] = np.degrees(h)                  # perturb by h radians
        numeric[:, k] = (forward(*(j + step), geom=GEOM)
                         - forward(*(j - step), geom=GEOM)) / (2 * h)
    assert np.allclose(analytic, numeric, atol=1e-3)


def test_interpolate_line_hits_both_ends():
    pts = interpolate_line([0, 0, 0], [10, 0, 0], step_mm=2.0)
    assert np.allclose(pts[0], [0, 0, 0])
    assert np.allclose(pts[-1], [10, 0, 0])
    assert len(pts) == 6
    spacing = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert np.allclose(spacing, spacing[0])


def test_geometry_roundtrips_through_json(tmp_path):
    g = Geometry(L0=140.5, L1=134.2, L2=148.1, Ltool=59.3, Ztool=-2.5)
    assert Geometry.load(g.save(tmp_path / "g.json")) == g


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def test_similarity_recovers_a_known_transform():
    angle = np.radians(37.0)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    t = np.array([210.0, -45.0])
    src = np.array([[0, 0], [100, 0], [100, 100], [0, 100], [50, 20]], float)
    dst = (R @ src.T).T + t

    R_fit, t_fit, s_fit = fit_similarity(src, dst)
    assert s_fit == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(R_fit, R)
    assert np.allclose(t_fit, t)


def test_plate_transform_roundtrips():
    src = np.array([[0, 0], [120, 0], [120, 120], [0, 120]], float)
    angle = np.radians(-15.0)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    dst = (R @ src.T).T + np.array([180.0, 30.0])

    tf = PlateToRobot.from_points(src, dst, plate_z=-12.0)
    assert tf.rms_mm < 1e-9
    assert tf.rotation_deg == pytest.approx(-15.0)
    assert np.allclose(tf.inverse(tf(src)), src)
    assert tf.to_xyz([0, 0], z_above=5.0)[2] == pytest.approx(-7.0)


def test_similarity_refuses_a_mirror():
    """A reflected point set must come back as a rotation, never a flip."""
    src = np.array([[0, 0], [10, 0], [0, 10]], float)
    dst = np.array([[0, 0], [0, 10], [10, 0]], float)      # mirrored
    R, _, _ = fit_similarity(src, dst)
    assert np.linalg.det(R) == pytest.approx(1.0)


@pytest.mark.parametrize("joints, reported", [
    # folded rest pose, arm parked
    ([0.00, -10.15, 100.65], [8.75, 0.00, -11.58]),
    # unfolded, well out into the workspace and off-axis in J1
    ([67.45, 20.59, 62.59], [67.06, 161.47, -4.12]),
])
def test_fk_matches_the_real_robot(joints, reported):
    """Locked against poses read off an actual Magician over serial.

    Joints and TCP were captured together from the arm's own encoders. These
    pin L0 = 0: the robot reports Z from its shoulder axis, and an earlier
    default of 138 put every height a full base-height out.

    Two poses, deliberately far apart in configuration space. One point can be
    fitted by any number of wrong parameter sets; agreeing at both a folded and
    an extended pose, and off the J1 = 0 plane, is what makes it convincing.
    """
    assert np.allclose(forward(*joints, geom=GEOM), reported, atol=0.05)


def test_the_bench_is_reachable():
    """A plate on the bench sits near z = -138. The arm must be able to get there.

    With L0 = 138 this failed, which is exactly how the bug showed up: the model
    claimed the workspace stopped a few mm above z = 0.
    """
    assert inverse(200.0, 0.0, -120.0, geom=GEOM) is not None
