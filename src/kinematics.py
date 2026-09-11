"""Forward / inverse kinematics for the Dobot Magician.

The Magician is a parallel-linkage arm: the four-bar keeps the end-effector
mounting plate horizontal no matter how the arm is folded, so the tool never
tilts and the whole problem collapses to a planar 2-link arm rotated about the
base.  That is why there is no orientation term here - only J4 (tool yaw).

Angle convention (the same one Dobot Studio shows):

    J1  base yaw,  0 = +X axis, positive = CCW seen from above
    J2  rear arm,  0 = straight UP,       positive = leaning forward
    J3  forearm,   0 = horizontal FORWARD, positive = tipping down
    J4  tool yaw (does not affect TCP position)

                       (elbow)
                    o---------____
                   /  L2          ----o  <- J3 axis
             L1   /                   | Ltool
                 /                    o  <- TCP
                o  (shoulder, J2 axis)  ... z = 0 IS HERE
                |
                | ~138 mm, not part of the model
             ===+===  bench, at about z = -138

So, with r the horizontal distance from the base axis to the tool:

    r = L1*sin(J2) + L2*cos(J3) + Ltool
    z = L0 + L1*cos(J2) - L2*sin(J3) + Ztool
    x = r*cos(J1),  y = r*sin(J1)

The link lengths below are the nominal published figures.  They are close but
not exact for any individual arm (and Ltool changes with the end effector:
suction cup, gripper and pen all differ).  Run `python -m src.fit_kinematics`
to fit them to YOUR robot from a handful of recorded poses - it typically takes
the residual from several millimetres down to a few tenths.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

CALIB_DIR = Path(__file__).resolve().parent.parent / "calib"


@dataclass
class Geometry:
    """Link lengths in millimetres.

    `L0` is 0 and that is not a placeholder. The Magician reports Z measured
    from its **shoulder (J2) axis**, not from the bench it stands on - so in the
    robot's own coordinates the shoulder is the origin and the bench is about
    -138 mm. The published "138 mm base height" describes where the shoulder sits
    physically; it is not an offset in the reported numbers. Verified against a
    real arm to 0.01 mm (see `test_fk_matches_the_real_robot`).

    Put 138 here and every Z is a whole base-height too high: the model decides
    it cannot reach the bench at all, which is nonsense but looks like a joint
    limit problem.
    """
    L0: float = 0.0       # shoulder (J2) axis -> Z origin; the robot's own datum
    L1: float = 135.0     # rear arm:  shoulder -> elbow
    L2: float = 147.0     # forearm:   elbow -> J3 axis
    Ltool: float = 59.7   # J3 axis -> TCP, horizontal (bare mounting plate)
    Ztool: float = 0.0    # J3 axis -> TCP, vertical (negative = tool hangs below)

    def save(self, path: Path | str = CALIB_DIR / "geometry.json") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: Path | str = CALIB_DIR / "geometry.json") -> "Geometry":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text()))


@dataclass(frozen=True)
class Limits:
    """Joint limits in degrees, from the Magician spec sheet.

    `elbow_min/max` bound (J3 - J2), the physical angle between the two links;
    the linkage jams before either joint reaches its own limit at some poses.
    Verify against your arm in Dobot Studio before trusting them near the edge.
    """
    j1: tuple[float, float] = (-135.0, 135.0)
    j2: tuple[float, float] = (0.0, 85.0)
    j3: tuple[float, float] = (-10.0, 95.0)
    j4: tuple[float, float] = (-145.0, 145.0)
    elbow: tuple[float, float] = (-90.0, 65.0)

    # A parked arm can read outside these - the folded rest pose is set by hand
    # with the servos off, so it is not a commandable pose. Do not widen these
    # to accommodate a reading taken while the arm was folded up.


DEFAULT_LIMITS = Limits()


class UnreachableError(ValueError):
    """Target is outside the workspace or violates a joint limit."""


# --------------------------------------------------------------------------- #
# forward kinematics
# --------------------------------------------------------------------------- #
def forward(j1: float, j2: float, j3: float, geom: Geometry | None = None
            ) -> np.ndarray:
    """Joint angles (degrees) -> TCP position ``[x, y, z]`` in mm."""
    g = geom or Geometry()
    a1, a2, a3 = np.radians([j1, j2, j3])
    r = g.L1 * np.sin(a2) + g.L2 * np.cos(a3) + g.Ltool
    z = g.L0 + g.L1 * np.cos(a2) - g.L2 * np.sin(a3) + g.Ztool
    return np.array([r * np.cos(a1), r * np.sin(a1), z])


def forward_many(joints: np.ndarray, geom: Geometry | None = None) -> np.ndarray:
    """Vectorised `forward` over an ``(N, 3)`` array of J1/J2/J3 rows."""
    g = geom or Geometry()
    a = np.radians(np.asarray(joints, dtype=float))
    a1, a2, a3 = a[:, 0], a[:, 1], a[:, 2]
    r = g.L1 * np.sin(a2) + g.L2 * np.cos(a3) + g.Ltool
    z = g.L0 + g.L1 * np.cos(a2) - g.L2 * np.sin(a3) + g.Ztool
    return np.column_stack([r * np.cos(a1), r * np.sin(a1), z])


# --------------------------------------------------------------------------- #
# inverse kinematics
# --------------------------------------------------------------------------- #
def inverse(x: float, y: float, z: float,
            geom: Geometry | None = None,
            limits: Limits | None = DEFAULT_LIMITS,
            elbow_up: bool = True) -> np.ndarray:
    """TCP position (mm) -> ``[J1, J2, J3]`` in degrees. Closed form.

    Raises `UnreachableError` if the point is outside the reachable annulus or
    the solution breaks a joint limit.  Set ``limits=None`` to skip the check
    (useful for plotting the unconstrained workspace, never for driving a real
    arm).
    """
    g = geom or Geometry()

    j1 = np.degrees(np.arctan2(y, x))

    # Strip the base rotation and the fixed tool offset to get a plain 2-link
    # planar problem in (a, b): reach out along the arm, and height above the
    # shoulder.
    a = np.hypot(x, y) - g.Ltool
    b = z - g.L0 - g.Ztool
    rho = np.hypot(a, b)

    if rho > g.L1 + g.L2:
        raise UnreachableError(
            f"({x:.1f}, {y:.1f}, {z:.1f}) is {rho - g.L1 - g.L2:.1f} mm beyond "
            f"the arm's {g.L1 + g.L2:.0f} mm reach")
    if rho < abs(g.L1 - g.L2):
        raise UnreachableError(
            f"({x:.1f}, {y:.1f}, {z:.1f}) is inside the {abs(g.L1 - g.L2):.1f} mm "
            "dead zone right above the base")

    # Law of cosines for the interior elbow angle.
    cos_gamma = (rho ** 2 - g.L1 ** 2 - g.L2 ** 2) / (2 * g.L1 * g.L2)
    gamma = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
    gamma = -gamma if elbow_up else gamma      # the Magician folds elbow-up

    # Absolute link angles measured from the +a axis.
    theta1 = np.arctan2(b, a) - np.arctan2(g.L2 * np.sin(gamma),
                                           g.L1 + g.L2 * np.cos(gamma))
    theta2 = theta1 + gamma

    # ...and back into Dobot's convention (J2 from vertical, J3 from horizontal).
    j2 = 90.0 - np.degrees(theta1)
    j3 = -np.degrees(theta2)

    sol = np.array([j1, j2, j3])
    if limits is not None:
        check_limits(*sol, limits=limits)
    return sol


def check_limits(j1: float, j2: float, j3: float, j4: float = 0.0,
                 limits: Limits = DEFAULT_LIMITS) -> None:
    """Raise `UnreachableError` naming the first joint that is out of range."""
    for name, val, (lo, hi) in (("J1", j1, limits.j1), ("J2", j2, limits.j2),
                                ("J3", j3, limits.j3), ("J4", j4, limits.j4)):
        if not (lo - 1e-6 <= val <= hi + 1e-6):
            raise UnreachableError(
                f"{name} = {val:.1f} deg is outside its limit [{lo}, {hi}]")
    elbow = j3 - j2
    lo, hi = limits.elbow
    if not (lo - 1e-6 <= elbow <= hi + 1e-6):
        raise UnreachableError(
            f"elbow angle J3-J2 = {elbow:.1f} deg is outside [{lo}, {hi}] - the "
            "linkage would jam")


def reachable(x: float, y: float, z: float, geom: Geometry | None = None,
              limits: Limits = DEFAULT_LIMITS) -> bool:
    """`inverse` without the exception - handy for filtering candidate points."""
    try:
        inverse(x, y, z, geom=geom, limits=limits)
        return True
    except UnreachableError:
        return False


# --------------------------------------------------------------------------- #
# differential kinematics
# --------------------------------------------------------------------------- #
def jacobian(j1: float, j2: float, j3: float,
             geom: Geometry | None = None) -> np.ndarray:
    """3x3 analytic Jacobian: d[x,y,z] / d[J1,J2,J3], mm per *radian*.

    Its determinant goes to zero at singularities - straight-out (elbow fully
    extended) and on the base axis where J1 stops doing anything.
    """
    g = geom or Geometry()
    a1, a2, a3 = np.radians([j1, j2, j3])
    r = g.L1 * np.sin(a2) + g.L2 * np.cos(a3) + g.Ltool
    dr_d2, dr_d3 = g.L1 * np.cos(a2), -g.L2 * np.sin(a3)
    c1, s1 = np.cos(a1), np.sin(a1)
    return np.array([
        [-r * s1, c1 * dr_d2,        c1 * dr_d3],
        [ r * c1, s1 * dr_d2,        s1 * dr_d3],
        [    0.0, -g.L1 * np.sin(a2), -g.L2 * np.cos(a3)],
    ])


def manipulability(j1: float, j2: float, j3: float,
                   geom: Geometry | None = None) -> float:
    """Yoshikawa's measure, sqrt(det(J J^T)). Near 0 => near a singularity."""
    j = jacobian(j1, j2, j3, geom)
    return float(np.sqrt(max(np.linalg.det(j @ j.T), 0.0)))


# --------------------------------------------------------------------------- #
# path helpers
# --------------------------------------------------------------------------- #
def interpolate_line(start, end, step_mm: float = 2.0) -> np.ndarray:
    """Straight line in Cartesian space, sampled every `step_mm`.

    Use this when the *path* matters (drawing, dragging along the plate).  A
    plain move_to only guarantees the endpoints - in joint mode the tool bulges
    sideways in between.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    n = max(int(np.ceil(np.linalg.norm(end - start) / step_mm)), 1)
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return start + t * (end - start)


def check_path(points, geom: Geometry | None = None,
               limits: Limits = DEFAULT_LIMITS) -> None:
    """Validate every waypoint before sending any of them to the arm.

    Cheap insurance: it turns a mid-motion stall into an error message you get
    while the robot is still parked.
    """
    for i, p in enumerate(np.asarray(points, float)):
        try:
            inverse(*p, geom=geom, limits=limits)
        except UnreachableError as exc:
            raise UnreachableError(f"waypoint {i} at {np.round(p, 1)}: {exc}") from exc


if __name__ == "__main__":
    g = Geometry.load()
    print(f"geometry: {g}\n")
    for joints in ([0, 0, 0], [0, 20, 20], [45, 45, 10], [-30, 60, 45]):
        xyz = forward(*joints, geom=g)
        back = inverse(*xyz, geom=g, limits=None)
        print(f"J{joints} -> xyz {np.round(xyz, 2)} -> J{np.round(back, 4)}  "
              f"(residual {np.linalg.norm(back - joints):.2e} deg)")
