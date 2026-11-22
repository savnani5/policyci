# ROS 2 Twist replay example

This is deterministic component replay, not closed-loop replay of a full robot. The input MCAP contains a stop, drive, stop `/vehicle/cmd_vel` profile. The subject under test computes fresh `/user/cmd_vel` output. The adapter compares the output sequence with the input sequence and returns `within_tolerance`, absolute-error metrics, and message count.

The fixed baseline reproduces the complete profile. The candidate behaves correctly through the initial stop and drive segments, then introduces a `0.4 m/s` bias when the final stop begins. The suite-owned tolerance is `0.05 m/s`; the candidate cannot change it. RoboLens therefore marks the comparison `REGRESSED`, blocks the gate, and identifies the first violating sample.

Run inside a sourced ROS 2 environment with MCAP support:

```bash
robolens check \
  --suite suites/ros2-twist-replay.yaml \
  --baseline subjects/twist-fixed.yaml \
  --candidate subjects/twist-stop-regression.yaml
```

The replay runner verifies the input topic, starts the subject, waits until the output publisher and recorder subscriptions are discoverable, then plays only the declared input topic. Missing, stale, empty, partially finalized, or otherwise unsound capture returns `NO_VERDICT`. Subject processes receive an explicit ROS/runtime environment allowlist rather than host CI credentials.

If ROS 2 is not installed locally, run the pinned container workflow from the repository root:

```bash
./scripts/run_ros2_replay_demo.sh
```

The script builds `examples/ros2_replay/Dockerfile` and writes the portable check directory below `runs/ros2-replay-demo/`. Docker Desktop or another running Docker daemon is required. It prints the exact command for opening the verified result in Rerun.

Install the optional local viewer once, then open that result:

```bash
uv sync --extra visualize
uv run robolens visualize /absolute/path/to/check_<id>
```

The default dashboard shows the automated verdict, expected/baseline/candidate command traces, absolute errors, the immutable tolerance, and the first candidate violation. The paired MCAP files are passed to Rerun as additional recordings for raw ROS topic inspection. Use `--no-open` to generate the durable `visualizations/<case>/<repeat>.rrd` without launching a GUI.

Rerun is an evidence viewer, not a second judge. RoboLens computes metrics and gates during the replay check; the dashboard visualizes the exact samples used for that decision.
