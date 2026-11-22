#!/usr/bin/env python3
"""Save a derisk eval_info.json (or synthetic stand-in) as the canonical fixture.

Usage:
  uv run python scripts/save_derisk_fixture.py path/to/eval_info.json
  uv run python scripts/save_derisk_fixture.py --from-modal-result runs/derisk_result.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source", nargs="?", help="Path to eval_info.json")
    p.add_argument("--from-modal-result", type=Path, help="shard_result / derisk JSON with eval_info key")
    args = p.parse_args()

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    dest = FIXTURE_DIR / "eval_info_day1.json"

    if args.from_modal_result:
        data = json.loads(args.from_modal_result.read_text())
        eval_info = data.get("eval_info")
        if not eval_info:
            raise SystemExit("No eval_info in modal result")
        dest.write_text(json.dumps(eval_info, indent=2))
        # Also dump full result for debugging
        (FIXTURE_DIR / "derisk_day1_result.json").write_text(
            json.dumps({k: v for k, v in data.items() if k != "eval_info"}, indent=2, default=str)
        )
    elif args.source:
        shutil.copy2(args.source, dest)
    else:
        raise SystemExit("Provide source path or --from-modal-result")

    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
