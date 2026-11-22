# RoboLens check

`robolens check` answers one release question: did the candidate regress a protected robot behavior relative to the baseline?

## Inputs

The command consumes three versioned files:

```bash
robolens check \
  --suite suites/manipulation-release.yaml \
  --baseline subjects/smolvla-baseline.yaml \
  --candidate subjects/smolvla-candidate.yaml
```

The suite owns cases, exact seeds or replay inputs, judge configuration, metric, and gate threshold. A subject may identify a checkpoint, image, code SHA, and inference configuration, but the selected runner must explicitly declare every field it applies. Unapplied identity fields and unknown capabilities are rejected before execution rather than displayed as if they were tested.

The built-in LIBERO suite format is:

```yaml
version: 1
name: manipulation-release
runner:
  type: libero
  suite: libero_object
execution:
  episodes: 5
  seeds: [1000, 1001, 1002, 1003, 1004]
gate:
  metric: success_rate
  mode: max_drop
  max_drop_pp: 10
  min_valid_episodes: 5
cases:
  - id: soup_to_basket
    task: 0
```

Pose and language perturbations are intentionally rejected by this check format until the live LIBERO runner applies them to the environment rather than merely recording metadata.

For replay adapters, behavior thresholds belong to the case, not the subject. For example, the Twist fixture places `tolerance: 0.05` under `cases[].judge`; neither baseline nor candidate can loosen its own judge.

## Decision semantics

Execution soundness and behavioral status are distinct:

- `VALID`: the subject ran on the declared units and required artifacts exist.
- `NO_VERDICT`: preflight, execution, evidence, or comparability is unsound.
- `REGRESSED`, `FIXED`, or `UNCHANGED`: behavior classifications made only after both sides are valid and paired.
- `NOT_EVALUATED`: the comparison cannot support a behavioral conclusion.

Overall precedence is:

1. any valid required regression -> `BLOCK` (exit 1);
2. otherwise any required `NOT_EVALUATED` case -> `NO_VERDICT` (exit 2);
3. otherwise -> `PASS` (exit 0).

Malformed configuration or a harness failure before a result can be formed exits 3.

## Output

```text
runs/<check_id>/
  check.json
  result.json
  evidence.html
  baseline/
    manifest.json
    cases/<case_id>/
      execution.json
      episodes/<seed>.json
      artifacts/...
  candidate/
    manifest.json
    cases/<case_id>/...
  pairs/<case_id>/<seed>/pair.json
```

All artifact references are relative to the check directory. The tree can remain in a customer VPC and be uploaded as one CI artifact without a RoboLens service or database.

`result.json` and `evidence.html` retain the suite version and immutable per-case judge configuration. The report also shows the declared subject diff, exact gate rule, per-case validity and metric delta, and baseline/candidate evidence side by side for the same case and seed. Dense sensor data stays in tools such as Rerun or Foxglove and can be linked as an artifact.

### Physics-backed manipulation visualization

The checked-in MuJoCo demo protects one concrete behavior: a Panda must grasp a
can, transport it across the workspace, place it in the target bin, and retreat.
The can must remain placed for at least `0.5 s`. Its baseline injects a `0.06 m`
action-frame offset; its candidate removes that calibration error.

```bash
./scripts/run_mujoco_grasp_demo.sh /tmp/robolens-grasp-demo
uv sync --extra visualize
uv run robolens visualize /tmp/robolens-grasp-demo/check_<id>
```

Each side emits a synchronized `episode.mp4` and `trajectory.json`. The latter
contains exact simulator time, object, target, and end-effector poses, gripper
width, contact bits, and task-success state. RoboLens derives the automatic
metric only from that exact state. The pixels are human evidence, never a proxy
measurement.

Before rendering, RoboLens re-derives every repeat's metric and rejects evidence
if baseline and candidate differ in seed, model hash, initial-state hash, sample
times, or video duration. The Rerun dashboard puts the failed and fixed videos
side by side, with exact target XY distance and placement success below on the same
simulation timeline. The synchronized videos occupy about 80% of the dashboard.

### Rerun replay visualization

ROS replay adapters can attach a portable `trace.json` containing the exact finite samples used to compute their metrics. The Twist example emits expected, observed, and absolute-error signals for every repeat. Existing metric-only adapters remain valid but cannot produce the derived dashboard.

Replay cases may declare `playback_topics` as a subset of `required_topics`.
`required_topics` defines the complete evidence contract and is checked against
the input bag. `playback_topics` defines only what is injected into the subject.
This lets the ALOHA action demo retain three cameras, joint state, effort, task,
and reference actions while replaying only the action topic through the tested
component.

Install the optional local viewer and open any completed replay or MuJoCo check:

```bash
uv sync --extra visualize
uv run robolens visualize /absolute/path/to/check_<id>
```

RoboLens selects the first `REGRESSED` or `FIXED` case and first paired repeat by default. Use `--case`, `--repeat`, and `--no-open` for explicit or headless operation. Before rendering, RoboLens verifies the stored error samples and re-derives supported aggregate metrics from every paired trace; the dashboard distinguishes the aggregate automatic metric from the selected repeat's result. The command writes `visualizations/<case>/<repeat>.rrd`, then opens it with the paired raw MCAP recordings. Rerun displays the evidence; it does not decide or alter the stored gate verdict.

## Real versus synthetic execution

The `check` command does not expose a synthetic demo flag. Unit tests use explicitly labeled fake executions. Conference or customer evidence must come from the real runner and contain the resulting videos or replay captures.

For LIBERO, configure Modal and the Hugging Face secret first:

```bash
uv sync --extra dev
modal secret create huggingface-secret HF_TOKEN=<token>
```

The machine that materializes a LIBERO check must also provide `ffprobe`. RoboLens counts frames in every declared video and returns `NO_VERDICT` for missing, empty, or unreadable evidence.

The example candidate changes `n_action_steps`. A convincing regression experiment requires completed real rollouts on both sides; an offline test result is not evidence that this change regresses a policy.
