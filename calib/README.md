# Calibration files

These are committed on purpose: they are what makes the lab machine and your
laptop agree about where things are, and they are small.

| file | what it is | when to redo it |
|---|---|---|
| `geometry.json` | fitted link lengths for **this** arm and end effector | after swapping the end effector, or on a different Dobot |
| `board.json` | the plate's ArUco dictionary, ids, marker size and gap | only if the plate changes |
| `plate_to_robot.json` | plate frame -> robot base frame, plus the plate's Z height | whenever the **robot** is moved relative to the plate |
| `intrinsics.json` | camera matrix and distortion (optional) | after changing camera or resolution |

`plate_to_robot.json` is the one that goes stale. Moving the *plate* is fine -
the markers re-establish the frame every frame. Moving the *robot*, or bumping
it hard enough to shift the base, means redoing `calibrate_plate`.

`fk_samples.csv` is gitignored: it is raw data, and `geometry.json` is the
result you actually need.
