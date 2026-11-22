# Panda pick-and-place regression

This is the video-first RoboLens demo. It runs Robosuite `PickPlaceCan` with a
Panda arm under real MuJoCo physics:

- **Before:** a `0.06 m` action-frame calibration error makes the gripper miss the can.
- **After:** the offset is removed, so the gripper grasps, transports, places,
  and retreats.
- **Gate:** the can remains in its target bin for `>= 0.5 s`.

The 12-second 1280x720 video makes the physical outcome visible. Exact simulator
state, not a video estimate, supplies the automatic metric.

## Run it

From the repository root:

```bash
./scripts/run_mujoco_grasp_demo.sh /tmp/robolens-grasp-demo
```

The script uses a pinned Python 3.11 / Robosuite 1.5.2 / MuJoCo 3.7.0 image with
CPU OSMesa rendering. A successful run ends with:

```text
PASS — PASS: all required cases satisfy the configured gate
verified=/tmp/robolens-grasp-demo/check_<id>
visualize=uv run --no-sync robolens visualize /tmp/robolens-grasp-demo/check_<id>
```

Install the optional local viewer once, then run the printed command:

```bash
uv sync --extra visualize
uv run robolens visualize /tmp/robolens-grasp-demo/check_<id>
```

The default Rerun layout is deliberately simple: synchronized failed/fixed
videos occupy about 80% of the screen. Exact object-to-target XY distance and
placement success sit below on the same timeline.

## Evidence

```text
check_<id>/
  result.json
  evidence.html
  baseline/cases/pick_place_offset/repeats/1000/
    episode.mp4
    trajectory.json
  candidate/cases/pick_place_offset/repeats/1000/
    episode.mp4
    trajectory.json
  visualizations/pick_place_offset/1000.rrd
```

The one-command verifier refuses to print `verified=` unless the gate is
`PASS`, the comparison is `FIXED`, both 240-frame 1280x720 videos decode, both
trajectory artifacts are confined and non-empty, model and initial-state
identities match, and the exact placement outcomes satisfy the expected failure/fix.
