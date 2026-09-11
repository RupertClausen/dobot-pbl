# dobot-pbl

Docker workspace for programming a **Dobot Magician** with an **OpenCV + ArUco**
camera on top: forward/inverse kinematics, a live camera view, and the
calibration that ties what the camera sees to where the arm can reach.

Built for the joint PBL exchange week. The target setup is a white plastic plate
with a 3x3 block of ArUco markers in one corner; the markers define a coordinate
frame fixed to the plate, so the robot keeps working after somebody slides the
plate across the bench.

---

## Quick start

```bash
dobotpbl --devices     # is the arm and camera actually visible?
dobotpbl               # shell inside the container
dobotpbl --test        # 26 tests, no hardware needed
```

`dobotpbl` builds the image on first use and keeps one container alive. The
project directory is bind-mounted at `/work`, so **edit on the host, run inside**
— nothing needs rebuilding for a code change, and deleting the container never
loses work.

| command | what it does |
|---|---|
| `dobotpbl` | shell in the container |
| `dobotpbl <cmd...>` | run one command inside |
| `dobotpbl --lab` | JupyterLab on <http://localhost:8888> |
| `dobotpbl --stream` | live ArUco view on <http://localhost:5000> |
| `dobotpbl --test` | run the test suite |
| `dobotpbl --devices` | what robot/camera the container can see |
| `dobotpbl --status` / `--stop` / `--restart` | container lifecycle |
| `dobotpbl --rebuild` | rebuild the image after changing `requirements.txt` |

Repo: <https://github.com/RupertClausen/dobot-pbl>

### On another machine (the lab computer)

```bash
git clone git@github.com:RupertClausen/dobot-pbl.git
cd dobot-pbl && ./install.sh
```

That installs the `dobotpbl` launcher there and builds the image. If the clone
is not at `~/Software/Coding/dobot-lab`, `install.sh` prints the
`DOBOTPBL_PROJECT` line to add to your `~/.bashrc`.

Calibration lives in `calib/` and **is** committed, so `git pull` carries the
lab robot's geometry across. `plate_to_robot.json` is the one that is specific
to a physical setup — if the two machines drive different robots, keep an eye on
which one you last committed.

---

## The order to do things in

Roughly 30 minutes end to end, first time.

**1. Check the plumbing.**

```bash
dobotpbl --devices
```

The Dobot should appear as `/dev/ttyUSB0`. If it does not, it is a host problem,
not a container problem — check power and the cable first.

**2. Work out what is printed on the plate.** Put the plate in view, measure one
black marker square with a ruler, then:

```bash
dobotpbl python -m src.identify_plate --marker-mm 20 --save
```

This scans *every* ArUco dictionary, so you do not have to guess which one the
plate uses — guessing wrong just returns nothing with no error. It writes
`calib/board.json` and drops annotated images in `data/`.

**3. Fit the arm's real link lengths.**

```bash
dobotpbl python -m src.fit_kinematics --auto
```

> **Do not skip this.** The nominal dimensions in `kinematics.py` assume the tool
> tip sits exactly on the J3 axis. A suction cup or gripper hangs *below* it, so
> `Ztool` is really negative and the uncalibrated model gets Z wrong by centimetres
> — it will not let you reach the plate at all. The fit takes about two minutes
> and typically pulls the error down to a few tenths of a millimetre.

**4. Tell the robot where the plate is.**

```bash
dobotpbl python -m src.calibrate_plate
```

Drag the arm (hold the **Unlock** button) so the tool tip touches a spot on the
plate, then click that same spot in the camera window. Six to eight samples
**spread across the whole plate** — corners especially. Points bunched in one
area fit beautifully and extrapolate terribly.

Check the printed `scale` afterwards: it should be `1.000 ± 0.005`. If it is not,
`marker_mm` in `board.json` does not match the real plate.

**5. Prove it works.**

```bash
dobotpbl python -m src.demo_click_to_move          # hover 20 mm above
dobotpbl python -m src.demo_click_to_move --touch  # go down and touch it
```

Click anywhere on the plate and the arm goes there. This exercises the entire
chain at once — if this is accurate, everything is wired up correctly.

---

## Coordinate frames

Three frames, two transforms. Keeping them separate is what makes the system
survive a knocked plate:

```
  image pixels  ──H──▶  plate mm  ──T──▶  robot base mm
                 ▲                 ▲
                 │                 └─ calibrate_plate.py, fixed until the
                 │                    ROBOT moves relative to the plate
                 └─ recomputed from the ArUco markers on EVERY frame, so
                    moving the plate or the camera costs nothing
```

- **Plate frame** — origin at the top-left corner of the first marker, +X along
  the top row, +Y down the first column, millimetres.
- **Robot frame** — the Magician's own base frame, the numbers Dobot Studio shows.

`T` is fitted as a *similarity* (rotation + translation + one scale), not a full
affine. Both sides are already in millimetres, so the true transform is rigid and
the fitted scale coming out at 1.000 is a free correctness check. A full affine
would quietly absorb a bad homography into a shear and hide the error.

---

## Kinematics

The Magician's parallel linkage keeps the end effector horizontal at all poses,
so the problem reduces to a planar 2-link arm rotated about the base — which is
why the IK here is closed-form and exact, not iterative.

```
J1  base yaw,  0 = +X axis, positive CCW from above
J2  rear arm,  0 = straight UP,        positive = leaning forward
J3  forearm,   0 = horizontal FORWARD, positive = tipping down
J4  tool yaw (does not move the TCP)
```

```python
from src.kinematics import forward, inverse, jacobian

xyz    = forward(0, 20, 20)          # joints (deg) -> [x, y, z] mm
joints = inverse(*xyz)               # and exactly back again
J      = jacobian(0, 20, 20)         # mm per radian, for singularity checks
```

`inverse` raises `UnreachableError` with a message saying *why* — out of reach,
inside the dead zone above the base, or which joint limit it broke. `Arm.move_to`
runs that check plus a hard Z floor before anything is sent over serial, so a
typo becomes a Python traceback instead of the arm leaning into the plate.

Plot the reachable envelope, with your plate drawn on it:

```bash
dobotpbl python -m src.plot_workspace --z 0    # -> data/workspace.png
```

---

## No robot? No problem

Everything runs without hardware, so students can develop while the arm is busy:

```python
from src.dobot_arm import connect

with connect(simulate=True) as arm:      # identical API, no serial port
    arm.move_to(200, 0, 50)
    print(arm.position)
```

`connect()` with no arguments falls back to simulation automatically when no
serial port is present.

---

## Files

```
dobotpbl                    the launcher (also installed to ~/.local/bin)
Dockerfile                  Python 3.11 + OpenCV 5 (Qt GUI) + the science stack
src/
  kinematics.py             FK, IK, Jacobian, joint limits, path interpolation
  dobot_arm.py              safe pydobot wrapper + simulator
  fit_kinematics.py         fit link lengths to YOUR arm            [calibration]
  camera.py                 V4L2 capture with stale-buffer flushing
  aruco.py                  marker detection, plate frame, homography
  identify_plate.py         work out the plate's dictionary & geometry [calibration]
  calibrate_camera.py       chessboard intrinsics (optional)        [calibration]
  calibrate_plate.py        plate -> robot transform                [calibration]
  transforms.py             the coordinate chain
  live_view.py              live overlay: X11 window or browser stream
  make_board.py             generate a printable 3x3 board
  demo_click_to_move.py     click the plate, arm goes there
  demo_pick_place.py        pick and place in plate coordinates
  plot_workspace.py         matplotlib workspace envelope
tests/                      26 tests, hardware-free
calib/                      calibration output (committed — see calib/README.md)
data/                       snapshots and plots (gitignored)
```

---

## Troubleshooting

**No `/dev/ttyUSB0`.** Check the host first with `dobotpbl --devices`. If the host
cannot see it, the container never will. The Magician needs its power brick on,
not just USB. `/dev` is mounted whole, so replugging works without restarting the
container.

**`cv2.imshow` window never appears.** The launcher runs `xhost +local:docker`
already. On Wayland the window goes through XWayland, which is normally fine. If
it still fails, use the browser instead — it works everywhere:

```bash
dobotpbl --stream        # then open http://localhost:5000
```

**OpenCV windows are tiny.** This is the 2880x1800 @200% display. Bump the scale:

```bash
QT_SCALE_FACTOR=2 dobotpbl --rebuild
```

**No markers detected.** Glare on glossy plastic is the usual cause — tilt the
plate or move the light off the specular angle. Check `data/plate_snapshot.png`
to see what the camera actually saw. Confirm the dictionary with
`identify_plate`; a wrong dictionary returns silence, not an error.

**The arm reaches the wrong height.** You skipped step 3. Run `fit_kinematics`.

**`plate_to_robot.json` not found.** Run `calibrate_plate` (step 4).

---

## Safety

- Keep `arm.speed(30, 30)` while testing. It is quite strong enough to break the
  plate, and itself.
- `home()` sweeps a wide arc. Clear the bench first.
- `Arm` enforces `z_floor` (default -40 mm). Raise it while experimenting —
  `connect(z_floor=0)` will not let anything touch the plate at all.
- Always dry-run a new sequence with `--simulate` before letting it near hardware.
