"""Small runner contract shared by simulation and replay implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from policyci.checkspec import CaseSpec
from policyci.schemas import CaseExecution, ExecutionValidity, SubjectRef


class PreflightResult(BaseModel):
    validity: ExecutionValidity
    reasons: list[str] = Field(default_factory=list)

    @classmethod
    def valid(cls) -> "PreflightResult":
        return cls(validity=ExecutionValidity.VALID)

    @classmethod
    def no_verdict(cls, *reasons: str) -> "PreflightResult":
        return cls(validity=ExecutionValidity.NO_VERDICT, reasons=list(reasons))


class Runner(Protocol):
    def preflight(self, subject: SubjectRef, case: CaseSpec) -> PreflightResult: ...

    def run_case(
        self,
        subject: SubjectRef,
        case: CaseSpec,
        output_dir: Path,
    ) -> CaseExecution: ...
