#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..problems import BenchmarkProblem, Capability


@dataclass(frozen=True)
class Tolerances:
    relative: float = 1e-8
    absolute: float = 1e-10
    max_steps: int = 500

    def as_dict(self) -> dict[str, float | int]:
        return {
            "relative": self.relative,
            "absolute": self.absolute,
            "max_steps": self.max_steps,
        }


@dataclass(frozen=True)
class CaseSpec:
    name: str
    problem: BenchmarkProblem
    tolerances: Tolerances = field(default_factory=Tolerances)
    solver_mode: Literal["default", "dense", "matrix-free", "sparse"] = "default"

    def __post_init__(self) -> None:
        if self.solver_mode != "default" and self.problem.capability != "nonlinear.root":
            raise ValueError("non-default solver modes require a nonlinear root problem")

    @property
    def capability(self) -> Capability:
        return self.problem.capability


@dataclass(frozen=True)
class Availability:
    available: bool
    capability: str
    dependency: str
    dependency_version: str | None
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "capability": self.capability,
            "dependency": self.dependency,
            "dependency_version": self.dependency_version,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Implementation:
    adapter: str
    backend: str
    method: str
    preconditioner: str
    versions: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "backend": self.backend,
            "method": self.method,
            "preconditioner": self.preconditioner,
            "versions": dict(self.versions),
        }


@dataclass(frozen=True)
class SolveResult:
    solution: Any
    auxiliary: dict[str, Any]
    converged: Any
    message: str
    operations: dict[str, Any | None]


@dataclass(frozen=True)
class RefreshEvidence:
    applicable: bool
    symbolic_reused: bool | None
    numeric_refreshed: bool | None
    symbolic_refresh_count: int
    numeric_refresh_count: int
    evidence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "symbolic_reused": self.symbolic_reused,
            "numeric_refreshed": self.numeric_refreshed,
            "symbolic_refresh_count": self.symbolic_refresh_count,
            "numeric_refresh_count": self.numeric_refresh_count,
            "evidence": self.evidence,
        }


NOT_APPLICABLE_REFRESH = RefreshEvidence(
    applicable=False,
    symbolic_reused=None,
    numeric_refreshed=None,
    symbolic_refresh_count=0,
    numeric_refresh_count=0,
    evidence="adapter has no refresh operation for this capability",
)


@dataclass(frozen=True)
class TransferEvidence:
    input_origin: str
    host_to_device_bytes: int | None
    host_to_device_timing_phase: str | None
    device_to_host_bytes: int | None
    device_to_host_timing_phase: str | None
    evidence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_origin": self.input_origin,
            "host_to_device_bytes": self.host_to_device_bytes,
            "host_to_device_timing_phase": self.host_to_device_timing_phase,
            "device_to_host_bytes": self.device_to_host_bytes,
            "device_to_host_timing_phase": self.device_to_host_timing_phase,
            "evidence": self.evidence,
        }


SKIPPED_TRANSFERS = TransferEvidence(
    input_origin="numpy-host",
    host_to_device_bytes=None,
    host_to_device_timing_phase=None,
    device_to_host_bytes=None,
    device_to_host_timing_phase=None,
    evidence="not measured because the row was skipped",
)


class BenchmarkAdapter:
    """Common phase boundary implemented by every benchmark adapter."""

    name: str
    dependency: str
    capabilities: frozenset[str]

    def availability(self, capability: str, /) -> Availability:
        raise NotImplementedError

    def implementation(self, spec: CaseSpec, /) -> Implementation:
        raise NotImplementedError

    def setup(self, spec: CaseSpec, /) -> Any:
        raise NotImplementedError

    def compilation_applicable(self, setup_state: Any, /) -> bool:
        return False

    def compilation_after_preparation(self, setup_state: Any, /) -> bool:
        return False

    def compile(self, setup_state: Any, /) -> Any:
        return setup_state

    def preparation_applicable(self, compiled_state: Any, /) -> bool:
        return False

    def prepare(self, compiled_state: Any, /) -> Any:
        return compiled_state

    def solve(self, prepared_state: Any, /) -> SolveResult:
        raise NotImplementedError

    def differentiation_applicable(self, prepared_state: Any, /) -> bool:
        return False

    def compile_differentiation(self, prepared_state: Any, /) -> Any:
        return prepared_state

    def differentiate(self, prepared_state: Any, /) -> Any:
        raise NotImplementedError

    def refresh_applicable(self, prepared_state: Any, /) -> bool:
        return False

    def refresh(self, prepared_state: Any, /) -> tuple[Any, RefreshEvidence]:
        return prepared_state, NOT_APPLICABLE_REFRESH

    def certificate_problem(self, prepared_state: Any, /) -> BenchmarkProblem:
        return prepared_state.spec.problem

    def materialize_result(
        self,
        prepared_state: Any,
        result: SolveResult,
        /,
    ) -> tuple[Any, dict[str, Any], Any, dict[str, Any | None]]:
        del prepared_state
        import jax

        return jax.device_get(
            (
                result.solution,
                result.auxiliary,
                result.converged,
                result.operations,
            )
        )

    def release(self, prepared_state: Any, /) -> None:
        del prepared_state

    def memory(self, prepared_state: Any, result: SolveResult, /) -> dict[str, Any]:
        raise NotImplementedError

    def transfers(
        self,
        prepared_state: Any,
        result: SolveResult,
        /,
        *,
        device_to_host_bytes: int,
    ) -> TransferEvidence:
        if device_to_host_bytes != 0:
            raise ValueError(
                f"CPU adapter {self.name!r} returned device arrays without "
                "declaring transfer evidence"
            )
        return TransferEvidence(
            input_origin="numpy-host",
            host_to_device_bytes=0,
            host_to_device_timing_phase=None,
            device_to_host_bytes=0,
            device_to_host_timing_phase=None,
            evidence=(
                "canonical NumPy inputs and returned arrays remained host-resident "
                "through setup, solve, and verification"
            ),
        )


__all__ = [
    "Availability",
    "BenchmarkAdapter",
    "CaseSpec",
    "Implementation",
    "NOT_APPLICABLE_REFRESH",
    "SKIPPED_TRANSFERS",
    "RefreshEvidence",
    "SolveResult",
    "Tolerances",
    "TransferEvidence",
]
