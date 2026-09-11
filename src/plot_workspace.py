"""Plot where the arm can actually reach, and whether the plate fits inside it.

Two views, both worth looking at before planning any layout:

  left   a vertical slice through the arm's plane - the reachable (r, z) region,
         which is the annulus between the two links, clipped by joint limits;
  right  a top-down map at one height, with the plate footprint drawn on if the
         plate has been calibrated.

Sampling joint space rather than Cartesian space is what makes the boundary
come out right: every sampled joint triple is reachable by construction, so the
limits show up exactly instead of being approximated by an IK failure test.

    python -m src.plot_workspace --z 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # container has no Tk; save to PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.kinematics import DEFAULT_LIMITS, Geometry, forward_many, reachable
from src.transforms import PlateToRobot

DATA = Path(__file__).resolve().parent.parent / "data"


def sample_joint_space(geom: Geometry, n: int = 90) -> pd.DataFrame:
    """Dense sweep of (J2, J3) within limits -> a tidy frame of reachable poses."""
    lim = DEFAULT_LIMITS
    j2 = np.linspace(*lim.j2, n)
    j3 = np.linspace(*lim.j3, n)
    g2, g3 = np.meshgrid(j2, j3)
    joints = np.column_stack([np.zeros(g2.size), g2.ravel(), g3.ravel()])

    elbow = joints[:, 2] - joints[:, 1]
    ok = (elbow >= lim.elbow[0]) & (elbow <= lim.elbow[1])
    joints = joints[ok]

    xyz = forward_many(joints, geom)
    return pd.DataFrame({
        "j2": joints[:, 1], "j3": joints[:, 2],
        "r": np.hypot(xyz[:, 0], xyz[:, 1]), "z": xyz[:, 2],
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", type=float, default=None,
                    help="height for the top-down slice, mm "
                         "(default: mid-height of the reachable range)")
    ap.add_argument("--plate-mm", type=float, default=200.0,
                    help="plate size to outline, mm")
    ap.add_argument("--out", default=str(DATA / "workspace.png"))
    args = ap.parse_args()

    geom = Geometry.load()
    df = sample_joint_space(geom)
    print(f"geometry: {geom}")
    print(f"reachable r: {df.r.min():.0f} to {df.r.max():.0f} mm")
    print(f"reachable z: {df.z.min():.0f} to {df.z.max():.0f} mm")

    z_slice = args.z if args.z is not None else float(np.median(df.z))
    if not (df.z.min() <= z_slice <= df.z.max()):
        print(f"\nNOTE: z = {z_slice:.0f} mm is outside the model's reachable "
              f"range, so the top view will be empty.")
        if geom.Ztool == 0.0:
            print("      Ztool is still 0, which assumes the tool tip sits exactly\n"
                  "      on the J3 axis. A suction cup or gripper hangs below it, so\n"
                  "      the real Ztool is negative and the arm reaches lower than\n"
                  "      this model thinks. Run `python -m src.fit_kinematics --auto`.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    sc = ax1.scatter(df.r, df.z, c=df.j2, s=3, cmap="viridis")
    ax1.axhline(z_slice, color="crimson", ls="--", lw=1.2,
                label=f"slice at z = {z_slice:.0f} mm")
    ax1.set_xlabel("horizontal reach r [mm]")
    ax1.set_ylabel("height z [mm]")
    ax1.set_title("Side view: reachable envelope")
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")
    fig.colorbar(sc, ax=ax1, label="J2 [deg]")

    # Top-down: test a grid through the real IK so joint limits on J1 show too.
    span = np.arange(-320, 321, 4.0)
    gx, gy = np.meshgrid(span, span)
    mask = np.array([reachable(x, y, z_slice, geom)
                     for x, y in zip(gx.ravel(), gy.ravel())]).reshape(gx.shape)
    ax2.pcolormesh(gx, gy, mask, cmap="Greens", alpha=0.6, shading="auto")
    ax2.plot(0, 0, "ks", ms=9, label="robot base")

    try:
        tf = PlateToRobot.load()
        s = args.plate_mm
        corners = tf(np.array([[0, 0], [s, 0], [s, s], [0, s], [0, 0]], float))
        ax2.plot(corners[:, 0], corners[:, 1], "b-", lw=2, label="plate")
        origin = tf(np.array([0.0, 0.0]))
        ax2.plot(*origin, "bo", ms=7)
        ax2.annotate("plate origin\n(marker 0 corner)", origin,
                     textcoords="offset points", xytext=(8, 8), color="b")
        inside = mask.ravel()[
            np.argmin(np.linalg.norm(
                np.column_stack([gx.ravel(), gy.ravel()]) - origin, axis=1))]
        print(f"plate origin at robot {np.round(origin, 1)} - "
              f"{'reachable' if inside else 'NOT reachable'}")
    except FileNotFoundError:
        print("no plate calibration yet - plate outline omitted")

    ax2.set_xlabel("robot X [mm]")
    ax2.set_ylabel("robot Y [mm]")
    ax2.set_title(f"Top view at z = {z_slice:.0f} mm"
                  + ("  (nothing reachable here)" if not mask.any() else ""))
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right")

    fig.suptitle("Dobot Magician workspace", fontsize=14)
    fig.tight_layout()
    DATA.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
