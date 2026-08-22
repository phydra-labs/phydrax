#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ..problems import MathematicalProgramProblem
from ._availability import probe_modules
from .base import Availability, CaseSpec, Implementation
from .phydrax import _PhydraxState, PhydraxAdapter


_CAPABILITIES = frozenset(
    {"optimization.linear-program", "optimization.quadratic-program"}
)


class MPAXAdapter(PhydraxAdapter):
    """MPAX LP/QP methods through Phydrax's independent audit contract."""

    name = "mpax"
    dependency = "mpax"
    capabilities = _CAPABILITIES

    def availability(self, capability: str, /) -> Availability:
        return probe_modules(
            adapter=self.name,
            dependency=self.dependency,
            capability=capability,
            supported=self.capabilities,
            modules=("jax", "phydrax", "mpax"),
            distribution="mpax",
        )

    def implementation(self, spec: CaseSpec, /) -> Implementation:
        method = "mpax-rapdhg"
        return Implementation(
            adapter=self.name,
            backend="mpax-jax-device",
            method=method,
            preconditioner="ruiz-pock-chambolle-scaling",
            versions={},
        )

    def setup(self, spec: CaseSpec, /) -> _PhydraxState:
        if not isinstance(spec.problem, MathematicalProgramProblem):
            raise TypeError("MPAX adapter requires a mathematical-program problem")
        state = super().setup(spec)
        optim = state.phx.optim
        state.policy = optim.ConvexSolvePolicy(
            optim.MPAXraPDHG(iteration_limit=spec.tolerances.max_steps),
            termination=optim.ConvexTermination(
                absolute=spec.tolerances.absolute,
                relative=spec.tolerances.relative,
                maximum_steps=spec.tolerances.max_steps,
            ),
        )
        return state


__all__ = ["MPAXAdapter"]
