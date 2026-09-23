#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from ..discretization.discrete_velocity import (
    CompressibleKineticRuntimePlan,
    CompressibleKineticRuntimeState,
)
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult


class CompressibleKineticFixedStepMethod(AbstractFixedStepMethod):
    runtime: CompressibleKineticRuntimePlan
    method_id: str = eqx.field(static=True)

    def __init__(self, runtime: CompressibleKineticRuntimePlan, /):
        if not isinstance(runtime, CompressibleKineticRuntimePlan):
            raise TypeError("runtime must be CompressibleKineticRuntimePlan.")
        self.runtime = runtime
        self.method_id = canonical_fingerprint(
            {"kind": "compressible-kinetic-fixed-step", "runtime": runtime.runtime_id}
        )

    @property
    def required_step_size(self) -> float:
        return self.runtime.time_step

    @property
    def allows_step_reduction(self) -> bool:
        return False

    def step(
        self,
        step_index: Array,
        time: Array,
        state: CompressibleKineticRuntimeState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index, time
        checked_step = eqx.error_if(
            step_size,
            jnp.abs(step_size - self.runtime.time_step)
            > 64.0 * jnp.finfo(jnp.asarray(step_size).dtype).eps,
            "Compressible kinetic runtime requires its exact lattice time step.",
        )
        del checked_step
        result = self.runtime.advance(state, args)
        residual = jnp.maximum(
            jnp.abs(result.evidence.collision.conservation.mass_defect),
            jnp.maximum(
                result.evidence.collision.conservation.momentum_defect,
                jnp.abs(result.evidence.collision.conservation.energy_defect),
            ),
        )
        return FixedStepResult(
            result.candidate,
            result.accepted,
            result.successful,
            jnp.max(residual),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((), dtype=state.kinetic.populations[0].dtype),
        )


__all__ = ["CompressibleKineticFixedStepMethod"]
