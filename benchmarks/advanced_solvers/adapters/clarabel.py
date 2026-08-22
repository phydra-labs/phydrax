#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ..problems import MathematicalProgramProblem
from ._availability import probe_modules
from .base import Availability, CaseSpec, Implementation
from .phydrax import _PhydraxState, PhydraxAdapter


_CAPABILITIES = frozenset(
    {
        "optimization.linear-program",
        "optimization.quadratic-program",
        "optimization.conic-program",
    }
)


class ClarabelAdapter(PhydraxAdapter):
    """Clarabel quadratic-conic methods through Phydrax's independent audit contract."""

    name = "clarabel"
    dependency = "clarabel"
    capabilities = _CAPABILITIES

    def availability(self, capability: str, /) -> Availability:
        return probe_modules(
            adapter=self.name,
            dependency=self.dependency,
            capability=capability,
            supported=self.capabilities,
            modules=("phydrax", "clarabel"),
            distribution="clarabel",
        )

    def implementation(self, spec: CaseSpec, /) -> Implementation:
        return Implementation(
            adapter=self.name,
            backend="clarabel-rust-host",
            method="clarabel-interior-point",
            preconditioner="sparse-direct-kkt",
            versions={},
        )

    def setup(self, spec: CaseSpec, /) -> _PhydraxState:
        if not isinstance(spec.problem, MathematicalProgramProblem):
            raise TypeError("Clarabel adapter requires a mathematical-program problem")
        state = super().setup(spec)
        optim = state.phx.optim
        state.policy = optim.ConvexSolvePolicy(
            optim.ClarabelInteriorPoint(presolve=False),
            termination=optim.ConvexTermination(
                absolute=spec.tolerances.absolute,
                relative=spec.tolerances.relative,
                maximum_steps=spec.tolerances.max_steps,
            ),
        )
        return state


__all__ = ["ClarabelAdapter"]
