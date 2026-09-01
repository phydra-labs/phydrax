#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._execution import (
    _realize_lattice_boltzmann,
    lattice_boltzmann_equivalence,
    LatticeBoltzmannRealizationResult,
    ReferenceLatticeBoltzmannExecutionPlan,
    with_lattice_boltzmann_equivalence,
)
from ._lattice import LatticeBoltzmannVelocitySet


FusedLatticeBoltzmannImplementation: TypeAlias = Literal["jax-jit-scan"]


class FusedLatticeBoltzmannExecutionPlan(StrictModule, NonTrainableState):
    """Whole-rollout JIT realization with optional reference qualification.

    The fixed collide/stream/boundary schedule is traced once as one XLA program.
    Qualification compares states, failures, work, and diagnostics to the unfused
    reference. Production calls may skip the duplicate reference realization after
    that evidence has been established.
    """

    reference: ReferenceLatticeBoltzmannExecutionPlan
    implementation: FusedLatticeBoltzmannImplementation = eqx.field(static=True)
    accelerated: bool = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference: ReferenceLatticeBoltzmannExecutionPlan,
        /,
        *,
        backend: str = "jax",
    ):
        if not isinstance(reference, ReferenceLatticeBoltzmannExecutionPlan):
            raise TypeError("reference must be a reference LBM execution plan.")
        if backend != "jax":
            raise ValueError("Fused LBM execution supports only the JAX backend.")
        self.reference = reference
        self.implementation = "jax-jit-scan"
        self.accelerated = True
        self.backend = "jax"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fused-lattice-boltzmann-execution",
                "reference": reference.plan_id,
                "implementation": "jax-jit-scan",
                "accelerated": True,
                "backend": "jax",
            }
        )

    @property
    def velocity_set(self) -> LatticeBoltzmannVelocitySet:
        return self.reference.velocity_set

    @property
    def step_id(self) -> str:
        return self.reference.step_id

    def realize(
        self,
        initial_populations: Array,
        /,
        *,
        step_count: int,
        step_size: Any,
        args: Any = None,
        t0: Any = 0.0,
        rtol: float = 0.0,
        atol: float = 0.0,
        verify_equivalence: bool = True,
    ) -> LatticeBoltzmannRealizationResult:
        reference = (
            self.reference.realize(
                initial_populations,
                step_count=step_count,
                step_size=step_size,
                args=args,
                t0=t0,
            )
            if verify_equivalence
            else None
        )

        def fused_kernel(populations, size, runtime_args, initial_time):
            return _realize_lattice_boltzmann(
                self.reference.step,
                self.reference.velocity_set,
                populations,
                step_count=step_count,
                step_size=size,
                args=runtime_args,
                t0=initial_time,
                plan_id=self.plan_id,
                step_id=self.reference.step_id,
                execution_kind="fused",
            )

        candidate = eqx.filter_jit(fused_kernel)(
            initial_populations,
            step_size,
            args,
            t0,
        )
        if reference is None:
            return candidate
        evidence = lattice_boltzmann_equivalence(
            reference,
            candidate,
            rtol=rtol,
            atol=atol,
        )
        return with_lattice_boltzmann_equivalence(candidate, evidence)


__all__ = [
    "FusedLatticeBoltzmannExecutionPlan",
    "FusedLatticeBoltzmannImplementation",
]
