#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_allfinite, tree_where
from ..discretization.particle import DEMRuntimeState, RigidSphereKinematics
from ..equations._cfd_dem import (
    CFDEMCouplingEvaluation,
    evaluate_unresolved_cfd_dem,
    UnresolvedCFDEMCouplingPlan,
)


class CFDEMCouplingSchedulePlan(StrictModule, NonTrainableState):
    dem_substeps: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(self, dem_substeps: int, /):
        count = int(dem_substeps)
        if count <= 0:
            raise ValueError("dem_substeps must be positive.")
        self.dem_substeps = count
        self.schedule_id = canonical_fingerprint(
            {"kind": "cfd-dem-coupling-schedule", "dem_substeps": count}
        )


class CFDEMCouplingState(StrictModule):
    dem_state: DEMRuntimeState
    fluid_state: Any
    cumulative_particle_impulse: Array
    cumulative_fluid_impulse: Array
    cumulative_hydrodynamic_work: Array
    accepted_windows: Array


class CFDEMMacroStepResult(StrictModule):
    candidate_state: CFDEMCouplingState
    accepted_state: CFDEMCouplingState
    last_evaluation: CFDEMCouplingEvaluation
    momentum_residual: Array
    successful: Array
    schedule_id: str = eqx.field(static=True)


def _hydrodynamic_kick(
    plan: UnresolvedCFDEMCouplingPlan,
    state: DEMRuntimeState,
    force: Array,
    scale: Array,
    /,
) -> DEMRuntimeState:
    mobile = (
        plan.dynamics.bodies.particles.active_mask & ~plan.dynamics.bodies.fixed_mask
    )[:, None]
    velocity = state.kinematics.velocity + scale * (
        plan.dynamics.bodies.inverse_masses[:, None] * force
    )
    velocity = jnp.where(mobile, velocity, 0.0)
    kinematics = RigidSphereKinematics(
        state.kinematics.position,
        velocity,
        state.kinematics.angular_velocity,
    )
    return eqx.tree_at(lambda value: value.kinematics, state, kinematics)


def advance_cfd_dem_window(
    coupling: UnresolvedCFDEMCouplingPlan,
    schedule: CFDEMCouplingSchedulePlan,
    state: CFDEMCouplingState,
    fluid_velocity: ArrayLike,
    fluid_density: ArrayLike,
    dynamic_viscosity: ArrayLike,
    pressure_gradient: ArrayLike,
    particle_volume: ArrayLike,
    time: ArrayLike,
    fluid_step_size: ArrayLike,
    fluid_update: Callable[[Any, Array, Array], Any],
    /,
    *,
    args: Any = None,
) -> CFDEMMacroStepResult:
    """Advance one atomic frozen-fluid explicit CFD--DEM coupling window."""

    if not isinstance(coupling, UnresolvedCFDEMCouplingPlan):
        raise TypeError("coupling must be UnresolvedCFDEMCouplingPlan.")
    if not isinstance(schedule, CFDEMCouplingSchedulePlan):
        raise TypeError("schedule must be CFDEMCouplingSchedulePlan.")
    if not isinstance(state, CFDEMCouplingState):
        raise TypeError("state must be CFDEMCouplingState.")
    if not callable(fluid_update):
        raise TypeError("fluid_update must be callable.")
    macro_dt = jnp.asarray(
        fluid_step_size, dtype=state.dem_state.kinematics.position.dtype
    )
    if not np.isfinite(float(fluid_step_size)) or float(fluid_step_size) <= 0.0:
        raise ValueError("fluid_step_size must be finite and positive.")
    dem_dt = macro_dt / schedule.dem_substeps
    indices = jnp.arange(schedule.dem_substeps, dtype=jnp.int32)

    def substep(carry, index):
        dem_state, particle_impulse, fluid_impulse, work, prior_success = carry
        subtime = jnp.asarray(time, dtype=macro_dt.dtype) + index * dem_dt
        first = evaluate_unresolved_cfd_dem(
            coupling,
            dem_state,
            fluid_velocity,
            fluid_density,
            dynamic_viscosity,
            pressure_gradient,
            particle_volume,
            dem_dt,
        )
        pre = _hydrodynamic_kick(coupling, dem_state, first.particle_force, 0.5 * dem_dt)
        detail = coupling.dynamics.step_detailed(index, subtime, pre, dem_dt, args)
        second = evaluate_unresolved_cfd_dem(
            coupling,
            detail.accepted_state,
            fluid_velocity,
            fluid_density,
            dynamic_viscosity,
            pressure_gradient,
            particle_volume,
            dem_dt,
        )
        post = _hydrodynamic_kick(
            coupling,
            detail.accepted_state,
            second.particle_force,
            0.5 * dem_dt,
        )
        successful = (
            prior_success & first.successful & detail.successful & second.successful
        )
        accepted_dem = tree_where(successful, post, dem_state)
        particle_increment = 0.5 * dem_dt * (first.particle_force + second.particle_force)
        fluid_increment = (
            0.5
            * dem_dt
            * (first.fluid_momentum_source_rate + second.fluid_momentum_source_rate)
        )
        average_velocity = 0.5 * (
            dem_state.kinematics.velocity + post.kinematics.velocity
        )
        work_increment = jnp.sum(particle_increment * average_velocity)
        payload = second
        return (
            accepted_dem,
            particle_impulse + jnp.sum(particle_increment, axis=0),
            fluid_impulse + jnp.sum(fluid_increment, axis=0),
            work + work_increment,
            successful,
        ), payload

    dimension = coupling.dynamics.bodies.ambient_dimension
    initial_carry = (
        state.dem_state,
        jnp.zeros((dimension,), dtype=macro_dt.dtype),
        jnp.zeros((dimension,), dtype=macro_dt.dtype),
        jnp.zeros((), dtype=macro_dt.dtype),
        jnp.asarray(True),
    )
    (dem_state, particle_impulse, fluid_impulse, work, successful), evaluations = (
        jax.lax.scan(substep, initial_carry, indices)
    )
    fluid_candidate = fluid_update(state.fluid_state, fluid_impulse, macro_dt)
    fluid_successful = tree_allfinite(fluid_candidate)
    successful = successful & fluid_successful
    candidate = CFDEMCouplingState(
        dem_state,
        fluid_candidate,
        state.cumulative_particle_impulse + particle_impulse,
        state.cumulative_fluid_impulse + fluid_impulse,
        state.cumulative_hydrodynamic_work + work,
        state.accepted_windows + jnp.asarray(1, dtype=jnp.int32),
    )
    accepted = tree_where(successful, candidate, state)
    residual = particle_impulse + fluid_impulse
    last = jax.tree.map(lambda value: value[-1], evaluations)
    return CFDEMMacroStepResult(
        candidate,
        accepted,
        last,
        residual,
        successful,
        schedule.schedule_id,
    )


__all__ = [
    "CFDEMCouplingSchedulePlan",
    "CFDEMCouplingState",
    "CFDEMMacroStepResult",
    "advance_cfd_dem_window",
]
