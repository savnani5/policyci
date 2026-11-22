#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="robolens-aloha-replay:local"
CACHE_DIR="${REPO_ROOT}/.cache/aloha-hf-cache"
FIXTURE_DIR="${REPO_ROOT}/examples/aloha_replay/fixtures"
OUTPUT_NAME="aloha_coffee_episode_000000.mcap"

mkdir -p "${CACHE_DIR}" "${FIXTURE_DIR}"

if [[ "${ROBOLENS_REBUILD_IMAGE:-0}" == "1" ]] \
  || ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  docker build \
    --file "${REPO_ROOT}/examples/ros2_replay/Dockerfile" \
    --tag "${IMAGE_NAME}" \
    "${REPO_ROOT}"
fi

docker run --rm \
  --volume "${REPO_ROOT}/examples/aloha_replay/source.json:/workspace/source.json:ro" \
  --volume "${CACHE_DIR}:/cache" \
  --volume "${FIXTURE_DIR}:/fixtures" \
  "${IMAGE_NAME}" \
  python3 -m policyci.examples.aloha_replay.fixture \
    --manifest /workspace/source.json \
    --cache-dir /cache \
    --output "/fixtures/${OUTPUT_NAME}"

echo "fixture=${FIXTURE_DIR}/${OUTPUT_NAME}"
echo "manifest=${FIXTURE_DIR}/${OUTPUT_NAME}.manifest.json"
