#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from ..discretization.lattice_boltzmann import PreparedLatticeBoltzmannDynamics
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult


class LatticeBoltzmannFixedStepMethod(AbstractFixedStepMethod):
    """Fixed-step adapter for one prepared collide-and-route LBM dynamics."""

    dynamics: PreparedLatticeBoltzmannDynamics
    method_id: str = eqx.field(static=True)

    def __init__(self, dynamics: PreparedLatticeBoltzmannDynamics, /):
        if not isinstance(dynamics, PreparedLatticeBoltzmannDynamics):
            raise TypeError("dynamics must be PreparedLatticeBoltzmannDynamics.")
        self.dynamics = dynamics
        self.method_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-fixed-step",
                "dynamics": dynamics.prepared_id,
            }
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        result = self.dynamics.step_detailed(step_index, time, state, step_size, args)
        return FixedStepResult(
            result.candidate_state,
            result.accepted_state,
            result.successful,
            result.residual,
            jnp.asarray(1, dtype=jnp.int32),
            result.work,
            jnp.asarray(False),
            jnp.zeros((), dtype=state.dtype),
        )


__all__ = ["LatticeBoltzmannFixedStepMethod"]
