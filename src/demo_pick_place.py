"""Pick something up off the plate and put it somewhere else.

Works in plate coordinates throughout, so the sequence keeps working after the
plate is moved - the ArUco markers re-establish the frame each run.

    python -m src.demo_pick_place --simulate                 # dry run first
    python -m src.demo_pick_place --from 40 40 --to 90 40    # plate mm

Run it with --simulate at least once before letting it near the real plate.
"""
from __future__ import annotations

import argparse

import numpy as np

from src.dobot_arm import connect
from src.kinematics import UnreachableError
from src.transforms import PlateToRobot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", type=float, nargs=2, default=[40, 40],
                    metavar=("X", "Y"), help="pick point, plate mm")
    ap.add_argument("--to", dest="dst", type=float, nargs=2, default=[90, 40],
                    metavar=("X", "Y"), help="place point, plate mm")
    ap.add_argument("--grip-z", type=float, default=2.0,
                    help="mm above the plate at which the tool grips")
    ap.add_argument("--approach", type=float, default=35.0,
                    help="mm of clearance on the way in and out")
    ap.add_argument("--tool", choices=["suction", "gripper"], default="suction")
    ap.add_argument("--speed", type=float, default=35.0)
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()

    tf = PlateToRobot.load()
    pick = tf.to_xyz(args.src, z_above=args.grip_z)
    place = tf.to_xyz(args.dst, z_above=args.grip_z)

    print(f"pick  plate {args.src} -> robot {np.round(pick, 1)}")
    print(f"place plate {args.dst} -> robot {np.round(place, 1)}")

    with connect(simulate=args.simulate or None) as arm:
        # Validate both ends before moving anything, so an unreachable place
        # point does not strand a part halfway.
        for name, p in (("pick", pick), ("place", place)):
            for label, z in (("approach", p[2] + args.approach), ("grip", p[2])):
                try:
                    arm.move_to  # noqa: B018 - just proving the attribute exists
                    from src.kinematics import inverse
                    inverse(p[0], p[1], z, geom=arm.geom, limits=arm.limits)
                except UnreachableError as exc:
                    raise SystemExit(f"{name} {label} point is unreachable: {exc}")

        arm.speed(args.speed, args.speed)
        print("\npicking...")
        arm.pick(*pick, approach=args.approach, tool=args.tool)
        print("placing...")
        arm.place(*place, approach=args.approach, tool=args.tool)
        print("done")


if __name__ == "__main__":
    main()
