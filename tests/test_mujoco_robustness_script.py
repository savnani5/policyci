"""Executable contract for the conference demo wrapper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_demo_script_treats_unexpected_gate_pass_as_harness_error(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
if [[ "$1" == "image" && "$2" == "inspect" ]]; then
  exit 0
fi
if [[ "$1" == "run" ]]; then
  exit 0
fi
exit 9
"""
    )
    docker.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"

    completed = subprocess.run(
        [
            str(repo / "scripts/run_mujoco_robustness_demo.sh"),
            str(tmp_path / "output"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert completed.returncode == 3
    assert "expected robustness gate exit 1, got 0" in completed.stderr
