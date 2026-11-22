#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${1:-${REPO_ROOT}/runs/mujoco-grasp-demo}"
IMAGE_NAME="robolens-mujoco-grasp:local"
DOCKER_CONFIG_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${DOCKER_CONFIG_DIR}"
}
trap cleanup EXIT

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

# The demo uses only public images. An isolated config avoids coupling the
# build to a desktop credential helper or registry login on the host.
docker --config "${DOCKER_CONFIG_DIR}" build \
  --file "${REPO_ROOT}/examples/mujoco_grasp/Dockerfile" \
  --tag "${IMAGE_NAME}" \
  "${REPO_ROOT}"

docker run --rm \
  --env MUJOCO_GL=osmesa \
  --env PYOPENGL_PLATFORM=osmesa \
  --volume "${OUTPUT_ROOT}:/output" \
  "${IMAGE_NAME}" \
  python -m policyci.cli check \
    --suite suites/mujoco-grasp-regression.yaml \
    --baseline subjects/grasp-offset-bad.yaml \
    --candidate subjects/grasp-offset-fixed.yaml \
    --runs-dir /output

PYTHONPATH="${REPO_ROOT}/src" python3 - "${OUTPUT_ROOT}" <<'PY'
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from policyci.examples.mujoco_grasp.sensors import (
    gripper_closing_delta_m,
    validate_sensor_evidence_sample,
)


def fail(message: str) -> None:
    raise SystemExit(message)


def confined_file(root: Path, relative: str, *, kind: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        fail(f"{kind} artifact escapes the check directory: {relative}")
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"{kind} artifact is missing or empty: {relative}")
    return path


def longest_interval(times: list[float], active: list[bool]) -> float:
    longest = 0.0
    start = None
    previous = times[0]
    for time_s, enabled in zip(times, active):
        if enabled and start is None:
            start = time_s
        elif not enabled and start is not None:
            longest = max(longest, previous - start)
            start = None
        previous = time_s
    if start is not None:
        longest = max(longest, times[-1] - start)
    return longest


def analyze_trace(path: Path) -> dict:
    trace = json.loads(path.read_text())
    if trace.get("version") != 1 or trace.get("timeline") != "sim_time_s":
        fail(f"unexpected trajectory contract: {path}")
    if trace.get("seed") != 1000 or trace.get("sample_rate_hz") != 20.0:
        fail(f"unexpected seed or sample rate: {path}")
    samples = trace.get("samples")
    if not isinstance(samples, list) or not samples:
        fail(f"trajectory contains no samples: {path}")
    times = [float(sample["time_s"]) for sample in samples]
    if any(not math.isfinite(value) for value in times):
        fail(f"trajectory contains a non-finite time: {path}")
    if any(current <= previous for previous, current in zip(times, times[1:])):
        fail(f"trajectory times are not strictly increasing: {path}")
    if len(samples) != 240 or times[-1] - times[0] > 12.0:
        fail(f"trajectory does not match the 240-frame story: {path}")
    commands = []
    gripper_widths = []
    left_normal_forces = []
    right_normal_forces = []
    wrist_force_magnitudes = []
    for sample in samples:
        try:
            command, width, left_force, right_force, wrist_force = (
                validate_sensor_evidence_sample(sample)
            )
        except ValueError as exc:
            fail(f"trajectory contains invalid sensor evidence ({exc}): {path}")
        commands.append(command)
        gripper_widths.append(width)
        left_normal_forces.append(left_force)
        right_normal_forces.append(right_force)
        wrist_force_magnitudes.append(math.sqrt(sum(value**2 for value in wrist_force)))
    initial_z = float(samples[0]["object_position_m"][2])
    heights = [float(sample["object_position_m"][2]) - initial_z for sample in samples]
    task_success = [sample.get("task_success") is True for sample in samples]
    targets = [sample.get("target_position_m") for sample in samples]
    if any(not isinstance(target, list) or len(target) != 3 for target in targets):
        fail(f"trajectory is missing exact target coordinates: {path}")
    if any(target != targets[0] for target in targets[1:]):
        fail(f"target coordinates changed during the episode: {path}")
    distances = [
        math.dist(sample["object_position_m"][:2], target[:2])
        for sample, target in zip(samples, targets)
    ]
    return {
        "times": times,
        "max_lift": max(heights),
        "placement_success": longest_interval(times, task_success) >= 0.5,
        "placement_duration": longest_interval(times, task_success),
        "min_target_distance": min(distances),
        "final_target_distance": distances[-1],
        "target": targets[0],
        "initial_hash": trace.get("initial_state_sha256"),
        "model_hash": trace.get("model_sha256"),
        "command_range": [min(commands), max(commands)],
        "gripper_closing_delta_m": gripper_closing_delta_m(
            commands=commands,
            widths_m=gripper_widths,
        ),
        "max_left_normal_force": max(left_normal_forces),
        "max_right_normal_force": max(right_normal_forces),
        "max_wrist_force": max(wrist_force_magnitudes),
    }


def verify_video(path: Path) -> None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        fail("ffprobe is required to verify the demo videos")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,width,height",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        fail(f"video does not decode: {path}: {completed.stderr.strip()}")
    try:
        streams = json.loads(completed.stdout)["streams"]
        frames = int(streams[0]["nb_read_frames"])
        dimensions = (int(streams[0]["width"]), int(streams[0]["height"]))
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        frames = 0
        dimensions = (0, 0)
    if frames != 240 or dimensions != (1280, 720):
        fail(
            f"video must be 240 frames at 1280x720, got "
            f"frames={frames}, dimensions={dimensions}: {path}"
        )


checks = sorted(Path(sys.argv[1]).glob("check_*"), key=lambda path: path.stat().st_mtime)
if not checks:
    fail("the container did not produce a check directory")
check_root = checks[-1].resolve()
result_path = check_root / "result.json"
if not result_path.is_file():
    fail("the check did not produce result.json")
result = json.loads(result_path.read_text())
if result.get("runner") != "mujoco":
    fail(f"expected mujoco runner, got {result.get('runner')!r}")
if result.get("gate", {}).get("verdict") != "pass":
    fail(f"expected PASS gate, got {result.get('gate', {}).get('verdict')!r}")
comparisons = result.get("comparisons", [])
if len(comparisons) != 1:
    fail(f"expected exactly one comparison, got {len(comparisons)}")
comparison = comparisons[0]
if comparison.get("case_id") != "pick_place_offset" or comparison.get("status") != "fixed":
    fail(f"expected pick_place_offset FIXED comparison, got {comparison}")
if comparison.get("metric") != "placement_success":
    fail(f"unexpected automatic metric: {comparison.get('metric')!r}")
if comparison.get("baseline_value") is not False or comparison.get("candidate_value") is not True:
    fail("baseline/candidate placement metrics do not show false -> true")
judge = result.get("case_judges", {}).get("pick_place_offset")
if judge != {"placement_hold_duration_s": 0.5}:
    fail(f"unexpected immutable judge: {judge!r}")
pairs = comparison.get("evidence_pairs", [])
if len(pairs) != 1 or pairs[0].get("unit_id") != "1000":
    fail(f"expected exactly one paired repeat for seed 1000, got {pairs}")

analyses = {}
for side in ("baseline", "candidate"):
    artifacts = pairs[0].get(f"{side}_artifacts", [])
    kinds = [artifact.get("kind") for artifact in artifacts]
    if sorted(kinds) != ["trajectory", "video"]:
        fail(f"{side} evidence must contain exactly one video and trajectory: {kinds}")
    paths = {
        artifact["kind"]: confined_file(
            check_root,
            artifact["path"],
            kind=artifact["kind"],
        )
        for artifact in artifacts
    }
    verify_video(paths["video"])
    analyses[side] = analyze_trace(paths["trajectory"])

baseline = analyses["baseline"]
candidate = analyses["candidate"]
if baseline["times"] != candidate["times"]:
    fail("baseline and candidate sample times differ")
if baseline["initial_hash"] != candidate["initial_hash"]:
    fail("baseline and candidate initial-state hashes differ")
if baseline["model_hash"] != candidate["model_hash"]:
    fail("baseline and candidate model hashes differ")
if not isinstance(baseline["initial_hash"], str) or len(baseline["initial_hash"]) != 64:
    fail("initial-state hash is malformed")
if not isinstance(baseline["model_hash"], str) or len(baseline["model_hash"]) != 64:
    fail("model hash is malformed")
if baseline["target"] != candidate["target"]:
    fail("baseline and candidate target coordinates differ")
if baseline["placement_success"] is not False or baseline["final_target_distance"] < 0.15:
    fail(f"baseline did not visibly fail placement: {baseline}")
if (
    candidate["placement_success"] is not True
    or candidate["placement_duration"] < 0.5
    or candidate["min_target_distance"] > 0.08
    or candidate["max_lift"] < 0.12
    or candidate["command_range"][0] >= 0.0
    or candidate["command_range"][1] <= 0.0
    or candidate["gripper_closing_delta_m"] < 0.001
    or candidate["max_left_normal_force"] <= 0.0
    or candidate["max_right_normal_force"] <= 0.0
    or candidate["max_wrist_force"] <= 0.0
):
    fail(f"candidate did not complete grasp, transport, and placement: {candidate}")

print(f"verified={check_root}")
print(f"visualize=uv run --no-sync robolens visualize {check_root}")
PY
