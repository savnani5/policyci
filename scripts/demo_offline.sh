#!/usr/bin/env bash
# Offline showcase (no Modal): flat suite eval + CI gate + legacy sweep.
set -euo pipefail
cd "$(dirname "$0")/.."

POLICY_V1="${POLICY_V1:-HuggingFaceVLA/smolvla_libero}"
POLICY_V2="${POLICY_V2:-checkpoints/smolvla_libero_degraded}"

echo "==> flat suite eval dry-run"
uv run policyci eval \
  -c suites/soup_basket.yaml \
  -p "$POLICY_V1" \
  --dry-run \
  --report runs/eval_report.html

echo "==> flat suite CI dry-run (expect FAIL / exit 1)"
set +e
uv run policyci ci \
  -c suites/soup_basket.yaml \
  --v1 "$POLICY_V1" \
  --v2 "$POLICY_V2" \
  --dry-run \
  --report runs/ci_report.html
ci_ec=$?
set -e
echo "ci exit_code=$ci_ec (expected 1)"

echo "==> legacy axes sweep dry-run (robustness cliff)"
uv run policyci sweep \
  --policy "$POLICY_V1" \
  --suite-yaml suites/libero_object_robustness.yaml \
  --dry-run \
  --report runs/sweep_report.html

echo "==> reports"
ls -la runs/eval_report.html runs/ci_report.html runs/sweep_report.html
echo "DONE — open runs/ci_report.html and runs/sweep_report.html"
