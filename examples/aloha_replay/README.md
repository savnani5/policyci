# ALOHA sensor-rich replay fixture

This demo converts one real episode from `lerobot/aloha_static_coffee` into a
ROS 2 MCAP containing three RGB cameras, joint position and effort, reference
actions, and the task label. The immutable source revision and episode are in
`source.json`.

The generated MCAP is intentionally not checked into Git. Prepare it once with:

```bash
./scripts/prepare_aloha_replay_fixture.sh
```

Then run the offline conference workflow:

```bash
./scripts/run_aloha_replay_demo.sh runs/aloha-replay-demo
```

The script runs both subjects in ROS 2, requires the baseline to pass, requires
the candidate with a 300 ms gripper-command delay to fail, verifies a `BLOCK`
gate, renders the paired Rerun dashboard, and prints the exact `.rrd` path.

The input MCAP must contain all three cameras, joint states, effort, reference
actions, and the task label. Only `/episode/reference_action` is injected into
this action-processing component. The other topics remain hash-verified,
immutable context in each side's portable evidence.

This is an open-loop component regression. The synchronized recorded cameras
and robot state explain when the command fault occurs, but do not claim a
counterfactual physical outcome. Closed-loop behavioral testing belongs in a
simulator or on hardware.

The source dataset card declares the data MIT licensed. Preserve its attribution
and the generated fixture manifest with any redistributed artifact.
