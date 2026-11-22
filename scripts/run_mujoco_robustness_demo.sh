#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${1:-${REPO_ROOT}/runs/mujoco-robustness-demo}"
IMAGE_NAME="robolens-mujoco-grasp:local"
DOCKER_CONFIG_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${DOCKER_CONFIG_DIR}"
}
trap cleanup EXIT

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"

if [[ "${ROBOLENS_REBUILD_IMAGE:-0}" == "1" ]] \
  || ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  docker --config "${DOCKER_CONFIG_DIR}" build \
    --file "${REPO_ROOT}/examples/mujoco_grasp/Dockerfile" \
    --tag "${IMAGE_NAME}" \
    "${REPO_ROOT}"
fi

set +e
docker run --rm \
  --env MUJOCO_GL=osmesa \
  --env PYOPENGL_PLATFORM=osmesa \
  --env PYTHONPATH=/workspace/src \
  --volume "${OUTPUT_ROOT}:/output" \
  --volume "${REPO_ROOT}/src:/workspace/src:ro" \
  --volume "${REPO_ROOT}/suites:/workspace/suites:ro" \
  --volume "${REPO_ROOT}/subjects:/workspace/subjects:ro" \
  "${IMAGE_NAME}" \
  python -m policyci.cli robustness-check \
    --suite suites/mujoco-pick-place-robustness.yaml \
    --subject subjects/grasp-offset-fixed.yaml \
    --runs-dir /output
GATE_EXIT=$?
set -e

if [[ "${GATE_EXIT}" -ne 1 ]]; then
  echo "expected robustness gate exit 1, got ${GATE_EXIT}" >&2
  exit 3
fi

CHECK_ROOT="$(python3 - "${OUTPUT_ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
checks = sorted(root.glob("check_*/result.json"), key=lambda path: path.stat().st_mtime)
if not checks:
    raise SystemExit("robustness check did not produce result.json")
print(checks[-1].parent)
PY
)"

PYTHONPATH="${REPO_ROOT}/src" uv run --no-sync robolens robustness-visualize \
  "${CHECK_ROOT}" \
  --no-open

RERUN_PATH="$(python3 - "${CHECK_ROOT}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
recordings = sorted(root.glob("visualizations/robustness/*.rrd"))
if len(recordings) != 1:
    raise SystemExit(f"expected one robustness RRD, found {len(recordings)}")
print(recordings[0])
PY
)"

uv run --no-sync rerun rrd verify "${RERUN_PATH}"
echo "verified=${CHECK_ROOT}"
echo "rerun=${RERUN_PATH}"
echo "open=uv run --no-sync robolens robustness-visualize ${CHECK_ROOT}"
exit "${GATE_EXIT}"
