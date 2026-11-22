#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${1:-${REPO_ROOT}/runs/ros2-replay-demo}"
IMAGE_NAME="robolens-ros2-replay:local"

mkdir -p "${OUTPUT_ROOT}"

if [[ "${ROBOLENS_REBUILD_IMAGE:-0}" == "1" ]] \
  || ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  docker build \
    --file "${REPO_ROOT}/examples/ros2_replay/Dockerfile" \
    --tag "${IMAGE_NAME}" \
    "${REPO_ROOT}"
fi

set +e
docker run --rm \
  --volume "${OUTPUT_ROOT}:/output" \
  --volume "${REPO_ROOT}/src:/workspace/src:ro" \
  --volume "${REPO_ROOT}/suites:/workspace/suites:ro" \
  --volume "${REPO_ROOT}/subjects:/workspace/subjects:ro" \
  --volume "${REPO_ROOT}/examples/ros2_replay/fixtures:/workspace/examples/ros2_replay/fixtures:ro" \
  --env PYTHONPATH=/workspace/src \
  "${IMAGE_NAME}" \
  python3 -m policyci.cli check \
    --suite suites/ros2-twist-replay.yaml \
    --baseline subjects/twist-fixed.yaml \
    --candidate subjects/twist-stop-regression.yaml \
    --runs-dir /output
CHECK_EXIT=$?
set -e

if [[ "${CHECK_EXIT}" -ne 1 ]]; then
  echo "expected the regression gate to exit 1, got ${CHECK_EXIT}" >&2
  exit 3
fi

CHECK_ROOT="$(python3 - "${OUTPUT_ROOT}" <<'PY'
import json
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
if comparison["case_id"] != "stop_command_regression":
    raise SystemExit(f"unexpected case: {comparison['case_id']}")
if comparison["status"] != "regressed":
    raise SystemExit(f"expected REGRESSED comparison, got {comparison['status']}")
if comparison["baseline_value"] is not True or comparison["candidate_value"] is not False:
    raise SystemExit("fixed/regressed metrics did not distinguish the subjects")
evidence_pairs = comparison["evidence_pairs"]
if len(evidence_pairs) != 3:
    raise SystemExit(f"expected exactly 3 replay evidence pairs, got {len(evidence_pairs)}")
for pair in evidence_pairs:
    for side in ("baseline_artifacts", "candidate_artifacts"):
        kinds = {artifact["kind"] for artifact in pair[side]}
        if not {"mcap", "trace"}.issubset(kinds):
            raise SystemExit(f"paired replay evidence lacks MCAP/trace artifacts: {side}")
        for artifact in pair[side]:
            path = (check_root / artifact["path"]).resolve()
            if not path.is_relative_to(check_root):
                raise SystemExit(f"artifact escapes the check directory: {artifact['path']}")
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f"artifact is missing or empty: {artifact['path']}")

first_pair = evidence_pairs[0]
traces = {}
for side in ("baseline", "candidate"):
    artifacts = first_pair[f"{side}_artifacts"]
    trace_ref = next(artifact for artifact in artifacts if artifact["kind"] == "trace")
    traces[side] = json.loads((check_root / trace_ref["path"]).read_text())
expected = traces["baseline"]["series"]["expected"]["values"]
baseline = traces["baseline"]["series"]["observed"]["values"]
candidate = traces["candidate"]["series"]["observed"]["values"]
candidate_error = traces["candidate"]["series"]["absolute_error"]["values"]
if expected != [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]:
    raise SystemExit(f"unexpected stop-drive-stop profile: {expected}")
if baseline != expected:
    raise SystemExit(f"baseline did not reproduce the command profile: {baseline}")
if candidate[:7] != expected[:7] or candidate[7:] != [0.4, 0.4, 0.4]:
    raise SystemExit(f"candidate regression did not begin at final stop: {candidate}")
first_violation = next(
    (index for index, error in enumerate(candidate_error) if error > 0.05),
    None,
)
if first_violation != 7:
    raise SystemExit(f"expected first candidate violation at sample 7, got {first_violation}")
print(check_root)
PY
)"

PYTHONPATH="${REPO_ROOT}/src" \
  uv run --no-sync robolens visualize "${CHECK_ROOT}" --no-open

RRD_PATH="${CHECK_ROOT}/visualizations/stop_command_regression/repeat-000.rrd"
uv run --no-sync rerun rrd verify "${RRD_PATH}"

echo "verified regression: baseline PASS, candidate FAIL, gate BLOCK (check exit 1)"
echo "check=${CHECK_ROOT}"
echo "rerun=${RRD_PATH}"
echo "open=uv run --no-sync robolens visualize ${CHECK_ROOT}"
