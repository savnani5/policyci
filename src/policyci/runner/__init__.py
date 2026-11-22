"""Cloud execution — the only package that knows Modal exists."""

from policyci.runner.client import run_eval, run_eval_fanout

__all__ = ["run_eval", "run_eval_fanout"]
