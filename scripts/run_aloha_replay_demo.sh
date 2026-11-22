#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${1:-${REPO_ROOT}/runs/aloha-replay-demo}"
IMAGE_NAME="robolens-aloha-replay:local"
INPUT_MCAP="${REPO_ROOT}/examples/aloha_replay/fixtures/aloha_coffee_episode_000000.mcap"
INPUT_MANIFEST="${INPUT_MCAP}.manifest.json"

if [[ ! -s "${INPUT_MCAP}" ]] || [[ ! -s "${INPUT_MANIFEST}" ]]; then
  echo "prepared ALOHA fixture is missing; run ./scripts/prepare_aloha_replay_fixture.sh" >&2
  exit 3
fi
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "demo image is missing; run ./scripts/prepare_aloha_replay_fixture.sh" >&2
  exit 3
fi
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

set +e
docker run --rm \
  --network none \
  --volume "${OUTPUT_ROOT}:/output" \
  --volume "${REPO_ROOT}/src:/workspace/src:ro" \
  --volume "${REPO_ROOT}/suites:/workspace/suites:ro" \
  --volume "${REPO_ROOT}/subjects:/workspace/subjects:ro" \
  --volume "${REPO_ROOT}/examples/aloha_replay/fixtures:/workspace/examples/aloha_replay/fixtures:ro" \
  --env PYTHONPATH=/workspace/src \
  --env ROS_LOCALHOST_ONLY=1 \
  "${IMAGE_NAME}" \
  python3 -m policyci.cli check \
    --suite suites/aloha-coffee-replay.yaml \
    --baseline subjects/aloha-action-baseline.yaml \
    --candidate subjects/aloha-gripper-delay.yaml \
    --runs-dir /output
CHECK_EXIT=$?
set -e

if [[ "${CHECK_EXIT}" -ne 1 ]]; then
  echo "expected the ALOHA regression gate to exit 1, got ${CHECK_EXIT}" >&2
  exit 3
fi

CHECK_ROOT="$(python3 - "${OUTPUT_ROOT}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path


checks = sorted(Path(sys.argv[1]).glob("check_*"), key=lambda path: path.stat().st_mtime)
if not checks:
    raise SystemExit("the container did not produce a check directory")
check_root = checks[-1].resolve()
result = json.loads((check_root / "result.json").read_text())
if result["gate"]["verdict"] != "block" or result["gate"]["exit_code"] != 1:
    raise SystemExit(f"expected BLOCK/1 gate, got {result['gate']}")
comparison = result["comparisons"][0]
if comparison["case_id"] != "aloha_coffee_replay":
    raise SystemExit(f"unexpected case: {comparison['case_id']}")
if comparison["status"] != "regressed":
    raise SystemExit(f"expected REGRESSED comparison, got {comparison['status']}")
if comparison["baseline_value"] is not True:
    raise SystemExit("baseline did not pass the automatic tolerance gate")
if comparison["candidate_value"] is not False:
    raise SystemExit("delayed candidate did not fail the automatic tolerance gate")
pairs = comparison["evidence_pairs"]
if len(pairs) != 1:
    raise SystemExit(f"expected one paired replay repeat, got {len(pairs)}")

input_hashes = []
traces = {}
for side in ("baseline", "candidate"):
    artifacts = pairs[0][f"{side}_artifacts"]
    kinds = {artifact["kind"] for artifact in artifacts}
    if not {"input_mcap", "mcap", "trace"}.issubset(kinds):
        raise SystemExit(f"{side} evidence lacks input/output/trace artifacts: {kinds}")
    for artifact in artifacts:
        path = (check_root / artifact["path"]).resolve()
        if not path.is_relative_to(check_root):
            raise SystemExit(f"artifact escapes the check directory: {artifact['path']}")
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"artifact is missing or empty: {artifact['path']}")
        if artifact["kind"] == "input_mcap":
            input_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        if artifact["kind"] == "trace":
            traces[side] = json.loads(path.read_text())
if len(set(input_hashes)) != 1:
    raise SystemExit("baseline and candidate portable input MCAP hashes differ")
if any(trace.get("trace_type") != "aloha_action" for trace in traces.values()):
    raise SystemExit("paired traces are not typed ALOHA action evidence")

baseline_error = traces["baseline"]["series"]["max_gripper_error"]["values"]
candidate_error = traces["candidate"]["series"]["max_gripper_error"]["values"]
candidate_arm_error = traces["candidate"]["series"]["max_arm_error"]["values"]
tolerance = result["case_judges"]["aloha_coffee_replay"]["gripper_tolerance"]
if max(baseline_error) > tolerance:
    raise SystemExit("baseline trace exceeds the immutable gripper tolerance")
if not math.isclose(max(candidate_arm_error), 0.0, abs_tol=1e-12):
    raise SystemExit("demo fault changed arm channels; expected gripper-only delay")
first_violation = next(
    (index for index, error in enumerate(candidate_error) if error > tolerance),
    None,
)
if first_violation != 4:
    raise SystemExit(f"expected first candidate violation at sample 4, got {first_violation}")
print(check_root)
PY
)"

CHECK_NAME="$(basename "${CHECK_ROOT}")"
CONTAINER_CHECK="/output/${CHECK_NAME}"
CONTAINER_RRD="${CONTAINER_CHECK}/visualizations/aloha_coffee_replay/repeat-000.rrd"

docker run --rm \
  --network none \
  --volume "${OUTPUT_ROOT}:/output" \
  --volume "${REPO_ROOT}/src:/workspace/src:ro" \
  --env PYTHONPATH=/workspace/src \
  "${IMAGE_NAME}" \
  python3 -m policyci.cli visualize "${CONTAINER_CHECK}" --no-open

docker run --rm \
  --network none \
  --volume "${OUTPUT_ROOT}:/output:ro" \
  "${IMAGE_NAME}" \
  rerun rrd verify "${CONTAINER_RRD}"

RRD_PATH="${CHECK_ROOT}/visualizations/aloha_coffee_replay/repeat-000.rrd"
echo "verified open-loop component regression: baseline PASS, candidate FAIL, gate BLOCK"
echo "check=${CHECK_ROOT}"
echo "input_mcap=${INPUT_MCAP}"
echo "rerun=${RRD_PATH}"
echo "open=uv run --no-sync rerun ${RRD_PATH}"
