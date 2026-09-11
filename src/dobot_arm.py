"""Safe-ish wrapper around pydobot, plus a simulator for working without hardware.

Two things this adds over calling pydobot directly:

  * every target is checked against the kinematic model and a Z floor *before*
    it is sent, so a typo produces a Python exception instead of the arm
    grinding into the plate;
  * `Arm` is a context manager, so the serial port is always released - the
    single most common way to end up with "port busy" halfway through a session.

Use `connect(simulate=True)` to get an identical API with no robot attached.
"""
from __future__ import annotations

import glob
import os
import time
from contextlib import contextmanager

import numpy as np

from src.kinematics import (Geometry, Limits, DEFAULT_LIMITS, UnreachableError,
                            forward, inverse, interpolate_line, check_path)


def find_port() -> str | None:
    """First plausible Dobot serial port, or None.

    The Magician shows up as a CP210x USB-UART (/dev/ttyUSB*); a few revisions
    enumerate as CDC-ACM instead.
    """
    env = os.environ.get("DOBOT_PORT")
    if env and os.path.exists(env):
        return env
    ports = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    return ports[0] if ports else None


class Arm:
    """A Dobot Magician, or a pretend one.

    Parameters
    ----------
    port        serial device; auto-detected when None.
    z_floor     hard lower bound on Z in mm.  Nothing this class does will ever
                command the tool below it.  Raise it while testing.
    simulate    skip the hardware and track an internal pose instead.
    """

    def __init__(self, port: str | None = None, *, z_floor: float = -40.0,
                 geom: Geometry | None = None, limits: Limits = DEFAULT_LIMITS,
                 simulate: bool = False, verbose: bool = False):
        self.geom = geom or Geometry.load()
        self.limits = limits
        self.z_floor = z_floor
        self.simulate = simulate
        self._dev = None
        self._sim_pose = np.array([200.0, 0.0, 50.0, 0.0])   # x, y, z, r
        self._sim_suction = False

        if simulate:
            print("[arm] simulation mode - no hardware will be touched")
            return

        port = port or find_port()
        if port is None:
            raise RuntimeError(
                "No Dobot serial port found. Check that the arm is powered on "
                "and the USB cable is in, then look for /dev/ttyUSB0 on the "
                "host. Pass simulate=True to work without the robot.")
        from pydobot import Dobot           # imported late so sim mode needs no pydobot
        self._dev = Dobot(port=port, verbose=verbose)
        self.port = port
        print(f"[arm] connected on {port}")

    # -- lifecycle ---------------------------------------------------------- #
    def __enter__(self) -> "Arm":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._dev is not None:
            try:
                self._dev.close()
            finally:
                self._dev = None
                print("[arm] port closed")

    # -- state -------------------------------------------------------------- #
    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """``(xyzr, joints)`` as reported by the robot - TCP mm, joints degrees."""
        if self.simulate:
            j = inverse(*self._sim_pose[:3], geom=self.geom, limits=None)
            return self._sim_pose.copy(), np.array([*j, self._sim_pose[3]])
        p = self._dev.pose()
        return np.array(p[:4], float), np.array(p[4:8], float)

    @property
    def position(self) -> np.ndarray:
        """TCP ``[x, y, z]`` in mm."""
        return self.pose()[0][:3]

    def speed(self, velocity: float = 100.0, acceleration: float = 100.0) -> None:
        """Velocity/acceleration as a percentage. Keep it near 30 while testing."""
        if not self.simulate:
            self._dev.speed(velocity, acceleration)

    def home(self) -> None:
        """Run the built-in homing routine. The arm sweeps - keep hands clear."""
        if self.simulate:
            self._sim_pose = np.array([200.0, 0.0, 50.0, 0.0])
            return
        if hasattr(self._dev, "home"):
            self._dev.home()
        else:                                  # older pydobot builds
            self._dev._set_home_cmd()
        time.sleep(0.5)

    # -- motion ------------------------------------------------------------- #
    def _validate(self, x: float, y: float, z: float, r: float = 0.0) -> None:
        if z < self.z_floor:
            raise UnreachableError(
                f"z = {z:.1f} mm is below the {self.z_floor:.1f} mm floor. Raise "
                "the target, or lower z_floor if you really mean to go there.")
        j = inverse(x, y, z, geom=self.geom, limits=self.limits)
        from src.kinematics import check_limits
        check_limits(*j, r, limits=self.limits)

    def move_to(self, x: float, y: float, z: float, r: float = 0.0,
                wait: bool = True) -> None:
        """Joint-space move to a Cartesian target. Endpoints exact, path curved."""
        self._validate(x, y, z, r)
        if self.simulate:
            self._sim_pose = np.array([x, y, z, r], float)
            return
        self._dev.move_to(x, y, z, r, wait=wait)

    def move_linear(self, x: float, y: float, z: float, r: float = 0.0,
                    step_mm: float = 2.0) -> None:
        """Straight line in Cartesian space, via interpolated waypoints.

        Slower than `move_to` but the tool actually travels in a straight line -
        what you want for drawing, or for moving just above the plate.
        """
        pts = interpolate_line(self.position, (x, y, z), step_mm=step_mm)
        check_path(pts, geom=self.geom, limits=self.limits)
        for p in pts:
            self.move_to(*p, r, wait=True)

    def move_joints(self, j1: float, j2: float, j3: float, j4: float = 0.0) -> None:
        """Command joint angles directly, by converting through FK."""
        from src.kinematics import check_limits
        check_limits(j1, j2, j3, j4, limits=self.limits)
        self.move_to(*forward(j1, j2, j3, geom=self.geom), j4)

    def move_relative(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                      dr: float = 0.0) -> None:
        """Jog by a delta from wherever the tool is now."""
        x, y, z, r = self.pose()[0]
        self.move_to(x + dx, y + dy, z + dz, r + dr)

    # -- end effector ------------------------------------------------------- #
    def suction(self, on: bool) -> None:
        self._sim_suction = on
        if not self.simulate:
            self._dev.suck(on)
        time.sleep(0.4)          # the pump needs a moment to build/release vacuum

    def gripper(self, closed: bool) -> None:
        if not self.simulate:
            self._dev.grip(closed)
        time.sleep(0.6)

    # -- composite ---------------------------------------------------------- #
    def pick(self, x: float, y: float, z: float, r: float = 0.0,
             approach: float = 30.0, tool: str = "suction") -> None:
        """Approach from `approach` mm above, grab, retreat straight back up."""
        self.move_to(x, y, z + approach, r)
        self.move_linear(x, y, z, r)
        self.suction(True) if tool == "suction" else self.gripper(True)
        self.move_linear(x, y, z + approach, r)

    def place(self, x: float, y: float, z: float, r: float = 0.0,
              approach: float = 30.0, tool: str = "suction") -> None:
        """Mirror of `pick`."""
        self.move_to(x, y, z + approach, r)
        self.move_linear(x, y, z, r)
        self.suction(False) if tool == "suction" else self.gripper(False)
        self.move_linear(x, y, z + approach, r)


@contextmanager
def connect(port: str | None = None, *, simulate: bool | None = None, **kw):
    """Open an `Arm`, always closing the port on the way out.

    With ``simulate=None`` (the default) it falls back to simulation when no
    serial port is present, so demo scripts still run with the robot unplugged.
    """
    if simulate is None:
        simulate = find_port() is None
        if simulate:
            print("[arm] no serial port found - falling back to simulation")
    arm = Arm(port, simulate=simulate, **kw)
    try:
        yield arm
    finally:
        arm.close()


if __name__ == "__main__":
    with connect() as arm:
        arm.speed(30, 30)
        xyz, j = arm.pose()
        print(f"TCP    {np.round(xyz, 2)}")
        print(f"joints {np.round(j, 2)}")
        print(f"FK says {np.round(forward(*j[:3], geom=arm.geom), 2)}")
