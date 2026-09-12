#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from ..discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleKineticState,
)
from ..discretization.discrete_velocity._spatial import (
    PreparedSmoothCompressibleD2V17SpatialDynamics,
)
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult


class OracleSmoothCompressibleD2V17FixedStepMethod(AbstractFixedStepMethod):
    """Fixed-step adapter for oracle-energy D2V17 collide-stream dynamics."""

    dynamics: PreparedSmoothCompressibleD2V17SpatialDynamics
    method_id: str = eqx.field(static=True)

    def __init__(self, dynamics: PreparedSmoothCompressibleD2V17SpatialDynamics, /):
        if not isinstance(dynamics, PreparedSmoothCompressibleD2V17SpatialDynamics):
            raise TypeError(
                "dynamics must be PreparedSmoothCompressibleD2V17SpatialDynamics."
            )
        self.dynamics = dynamics
        self.method_id = canonical_fingerprint(
            {
                "kind": "oracle-smooth-compressible-d2v17-fixed-step",
                "dynamics": dynamics.prepared_id,
            }
        )

    @property
    def required_step_size(self) -> float:
        return self.dynamics.required_step_size

    @property
    def allows_step_reduction(self) -> bool:
        return False

    def step(
        self,
        step_index: Array,
        time: Array,
        state: SmoothCompressibleKineticState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        del step_index, time
        result, _ = self.dynamics.step_oracle(state, step_size, args)
        return FixedStepResult(
            result.candidate_state,
            result.accepted_state,
            result.successful,
            result.residual,
            jnp.asarray(1, dtype=jnp.int32),
            result.work,
            jnp.asarray(False),
            jnp.zeros((), dtype=state.particle_populations.dtype),
        )


__all__ = ["OracleSmoothCompressibleD2V17FixedStepMethod"]
