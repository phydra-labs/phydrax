#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._numerics._ssp_runge_kutta import ssprk33_step
from .._trainable import NonTrainableState
from ..discretization.particle import (
    PreparedDFSPH,
    PreparedIISPH,
    PreparedSoftSphereDEMDynamics,
    PreparedTransportVelocityDynamics,
)
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult


class TransportVelocityFixedStepMethod(AbstractFixedStepMethod, NonTrainableState):
    dynamics: PreparedTransportVelocityDynamics
    method_id: str = eqx.field(static=True)

    def __init__(self, dynamics: PreparedTransportVelocityDynamics, /):
        self.dynamics = dynamics
        self.method_id = canonical_fingerprint(
            {
                "kind": "transport-velocity-fixed-step",
                "dynamics": dynamics.prepared_id,
                "integrator": "ssprk33-refresh",
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
        del step_index
        candidate = ssprk33_step(self.dynamics, time, state, step_size, args)
        refreshed = self.dynamics.refresh_transport_velocity(
            time + step_size, candidate, step_size, args
        )
        correction = jnp.sqrt(jnp.sum((refreshed - candidate) ** 2))
        successful = jnp.all(jnp.isfinite(refreshed))
        accepted = jnp.where(successful, refreshed, state)
        return FixedStepResult(
            candidate,
            accepted,
            successful,
            correction,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(4, dtype=jnp.int32),
            jnp.asarray(True),
            correction,
        )


class IISPHFixedStepMethod(AbstractFixedStepMethod, NonTrainableState):
    dynamics: PreparedIISPH
    method_id: str = eqx.field(static=True)

    def __init__(self, dynamics: PreparedIISPH, /):
        self.dynamics = dynamics
        self.method_id = canonical_fingerprint(
            {"kind": "iisph-fixed-step", "dynamics": dynamics.prepared_id}
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
        del step_index
        result = self.dynamics.step_detailed(time, state, step_size, args)
        return FixedStepResult(
            result.candidate_state,
            result.accepted_state,
            result.successful,
            result.residual,
            result.iterations,
            result.iterations,
            jnp.asarray(False),
            jnp.zeros((), dtype=state.dtype),
        )


class DFSPHFixedStepMethod(AbstractFixedStepMethod, NonTrainableState):
    dynamics: PreparedDFSPH
    method_id: str = eqx.field(static=True)

    def __init__(self, dynamics: PreparedDFSPH, /):
        self.dynamics = dynamics
        self.method_id = canonical_fingerprint(
            {"kind": "dfsph-fixed-step", "dynamics": dynamics.prepared_id}
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
        del step_index
        result = self.dynamics.step_detailed(time, state, step_size, args)
        residual = jnp.maximum(result.divergence_residual, result.density_residual)
        iterations = result.divergence_iterations + result.density_iterations
        return FixedStepResult(
            result.candidate_state,
            result.accepted_state,
            result.successful,
            residual,
            iterations,
            iterations,
            jnp.asarray(False),
            jnp.zeros((), dtype=state.dtype),
        )


class DEMFixedStepMethod(AbstractFixedStepMethod, NonTrainableState):
    """Fixed-step adapter for prepared soft-sphere DEM dynamics."""

    dynamics: PreparedSoftSphereDEMDynamics
    method_id: str = eqx.field(static=True)

    def __init__(self, dynamics: PreparedSoftSphereDEMDynamics, /):
        if not isinstance(dynamics, PreparedSoftSphereDEMDynamics):
            raise TypeError("dynamics must be PreparedSoftSphereDEMDynamics.")
        self.dynamics = dynamics
        self.method_id = canonical_fingerprint(
            {
                "kind": "dem-fixed-step",
                "dynamics": dynamics.prepared_id,
                "integrator": "kick-drift-contact-kick",
            }
        )

    def step(
        self,
        step_index: Array,
        time: Array,
        state,
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
            jnp.zeros((), dtype=state.kinematics.position.dtype),
        )


__all__ = [
    "DEMFixedStepMethod",
    "DFSPHFixedStepMethod",
    "IISPHFixedStepMethod",
    "TransportVelocityFixedStepMethod",
]
