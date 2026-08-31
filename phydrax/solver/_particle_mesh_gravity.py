#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.splatting import PreparedParticleGridSplat
from ._self_gravity import PreparedNewtonianSelfGravity


class ParticleMeshGravityState(StrictModule):
    position: Array
    momentum: Array


class ParticleMeshGravityDiagnostics(StrictModule):
    initial_acceleration: Array
    final_acceleration: Array
    initial_potential: Array
    final_potential: Array
    mass_balance_defect: Array
    net_force: Array
    successful: Array


class ParticleMeshGravityStepResult(StrictModule):
    state: ParticleMeshGravityState
    diagnostics: ParticleMeshGravityDiagnostics
    successful: Array


class ParticleMeshGravityPlan(StrictModule, NonTrainableState):
    """Differentiable kick-drift-kick particle-mesh gravity coupling."""

    gravity: PreparedNewtonianSelfGravity
    transfer: PreparedParticleGridSplat
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gravity: PreparedNewtonianSelfGravity,
        transfer: PreparedParticleGridSplat,
        /,
    ):
        if not isinstance(gravity, PreparedNewtonianSelfGravity):
            raise TypeError("gravity must be PreparedNewtonianSelfGravity.")
        if not isinstance(transfer, PreparedParticleGridSplat):
            raise TypeError("transfer must be PreparedParticleGridSplat.")
        if (
            transfer.plan.target.prepared_id
            != gravity.dynamics.discretization.grid.prepared_id
        ):
            raise ValueError("Particle transfer and gravity must share one tensor grid.")
        if transfer.target_shape != gravity.cell_shape:
            raise ValueError("Particle transfer target shape must match gravity cells.")
        if transfer.particles.ambient_dimension != len(gravity.cell_shape):
            raise ValueError("Particle and gravity dimensions must match.")
        self.gravity = gravity
        self.transfer = transfer
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-mesh-gravity",
                "gravity": gravity.process_id,
                "transfer": transfer.prepared_id,
                "integrator": "kick-drift-kick",
            }
        )

    def initialize(
        self,
        position: ArrayLike,
        /,
        *,
        velocity: ArrayLike | None = None,
        momentum: ArrayLike | None = None,
    ) -> ParticleMeshGravityState:
        if (velocity is None) == (momentum is None):
            raise ValueError("Supply exactly one of velocity or momentum.")
        position_ = jnp.asarray(position)
        expected = (
            self.transfer.particles.capacity,
            self.transfer.particles.ambient_dimension,
        )
        if position_.shape != expected:
            raise ValueError(f"Particle position must have shape {expected}.")
        masses = self.transfer.particles.safe_masses.astype(position_.dtype)
        momentum_ = (
            jnp.asarray(momentum, dtype=position_.dtype)
            if momentum is not None
            else masses[:, None] * jnp.asarray(velocity, dtype=position_.dtype)
        )
        if momentum_.shape != expected:
            raise ValueError(f"Particle momentum must have shape {expected}.")
        active = self.transfer.particles.active_mask[:, None]
        return ParticleMeshGravityState(
            position=jnp.where(active, position_, 0.0),
            momentum=jnp.where(active, momentum_, 0.0),
        )

    def density(self, position: ArrayLike, /):
        routes = self.transfer.build(position)
        deposited = self.transfer.deposit_content(routes, self.transfer.particles.masses)
        return deposited, routes

    def acceleration(
        self,
        position: ArrayLike,
        args: Any = None,
        /,
        *,
        background_density: ArrayLike | None = None,
    ):
        deposited, routes = self.density(position)
        density = deposited.density
        if background_density is not None:
            background = jnp.asarray(background_density, dtype=density.dtype)
            if background.shape != density.shape:
                raise ValueError("background_density must match the gravity grid.")
            density = density + background
        potential, _, cell_acceleration, solved = self.gravity.solve_density(
            density, args
        )
        gathered = self.transfer.gather(routes, cell_acceleration)
        active = self.transfer.particles.active_mask[:, None]
        acceleration = jnp.where(active, gathered.values, 0.0)
        successful = (
            deposited.successful
            & solved.converged
            & jnp.all(gathered.support | ~self.transfer.particles.active_mask)
            & jnp.all(jnp.isfinite(acceleration))
        )
        return acceleration, potential, deposited, successful

    def step(
        self,
        state: ParticleMeshGravityState,
        start_time: ArrayLike,
        end_time: ArrayLike,
        args: Any = None,
        /,
        *,
        background_density: ArrayLike | None = None,
    ) -> ParticleMeshGravityStepResult:
        if not isinstance(state, ParticleMeshGravityState):
            raise TypeError("state must be ParticleMeshGravityState.")
        start = jnp.asarray(start_time)
        end = jnp.asarray(end_time, dtype=start.dtype)
        step = end - start
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0.0),
            "Particle-mesh step interval must be finite and increasing.",
        )
        masses = self.transfer.particles.safe_masses.astype(state.position.dtype)
        active = self.transfer.particles.active_mask[:, None]
        acceleration_0, potential_0, deposited_0, success_0 = self.acceleration(
            state.position, args, background_density=background_density
        )
        half_momentum = state.momentum + 0.5 * step * masses[:, None] * acceleration_0
        next_position = state.position + step * half_momentum / masses[:, None]
        next_position = jnp.where(active, next_position, 0.0)
        acceleration_1, potential_1, deposited_1, success_1 = self.acceleration(
            next_position, args, background_density=background_density
        )
        next_momentum = half_momentum + 0.5 * step * masses[:, None] * acceleration_1
        next_momentum = jnp.where(active, next_momentum, 0.0)
        successful = (
            success_0
            & success_1
            & jnp.all(jnp.isfinite(next_position))
            & jnp.all(jnp.isfinite(next_momentum))
        )
        candidate = ParticleMeshGravityState(next_position, next_momentum)
        accepted = jax.lax.cond(
            successful,
            lambda _: candidate,
            lambda _: state,
            operand=None,
        )
        physical_masses = self.transfer.particles.masses.astype(state.position.dtype)
        net_force = jnp.sum(
            physical_masses[:, None] * acceleration_0,
            axis=0,
        )
        diagnostics = ParticleMeshGravityDiagnostics(
            initial_acceleration=acceleration_0,
            final_acceleration=acceleration_1,
            initial_potential=potential_0,
            final_potential=potential_1,
            mass_balance_defect=jnp.maximum(
                deposited_0.balance.maximum_absolute_balance_defect,
                deposited_1.balance.maximum_absolute_balance_defect,
            ),
            net_force=net_force,
            successful=successful,
        )
        return ParticleMeshGravityStepResult(accepted, diagnostics, successful)


__all__ = [
    "ParticleMeshGravityDiagnostics",
    "ParticleMeshGravityPlan",
    "ParticleMeshGravityState",
    "ParticleMeshGravityStepResult",
]
