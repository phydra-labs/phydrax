#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import ParticleMeshGravityPlan
from ._background import FLRWBackground
from ._particles import (
    CosmologicalKDKPlan,
    CosmologicalParticleState,
)


class CosmologicalParticleMeshDiagnostics(StrictModule):
    """Per-interval and aggregate evidence for one scale-factor PM rollout."""

    drift_factors: Array
    first_kick_factors: Array
    second_kick_factors: Array
    mass_balance_defects: Array
    net_forces: Array
    force_successful: Array
    accepted: Array
    initial_force_successful: Array
    completed: Array
    accepted_steps: Array
    maximum_mass_balance_defect: Array
    maximum_net_force_norm: Array
    first_failed_step: Array


class CosmologicalParticleMeshResult(StrictModule):
    state: CosmologicalParticleState
    diagnostics: CosmologicalParticleMeshDiagnostics
    successful: Array


class _CosmologicalParticleMeshInterval(StrictModule):
    state: CosmologicalParticleState
    acceleration: Array
    drift_factor: Array
    first_kick_factor: Array
    second_kick_factor: Array
    mass_balance_defect: Array
    net_force: Array
    force_successful: Array
    successful: Array


def _advance_particle_mesh_interval(
    kinematics: CosmologicalKDKPlan,
    gravity: ParticleMeshGravityPlan,
    background: FLRWBackground,
    state: CosmologicalParticleState,
    end_scale_factor: ArrayLike,
    acceleration_start: ArrayLike,
    args: Any,
    /,
) -> _CosmologicalParticleMeshInterval:
    """Execute the one authoritative PM interval transaction."""

    proposal = kinematics.propose(
        background,
        state,
        end_scale_factor,
        acceleration_start,
    )
    endpoint_force = gravity.acceleration(proposal.positions, args)
    candidate, kdk = kinematics.complete(
        state,
        proposal,
        endpoint_force.acceleration,
    )
    successful = endpoint_force.successful & kdk.successful
    accepted = CosmologicalParticleState(
        jnp.where(successful, candidate.positions, state.positions),
        jnp.where(
            successful,
            candidate.canonical_momenta,
            state.canonical_momenta,
        ),
        jnp.where(successful, candidate.scale_factor, state.scale_factor),
    )
    return _CosmologicalParticleMeshInterval(
        accepted,
        endpoint_force.acceleration,
        kdk.drift_factor,
        kdk.first_kick_factor,
        kdk.second_kick_factor,
        endpoint_force.deposited.balance.maximum_absolute_balance_defect,
        endpoint_force.net_force,
        endpoint_force.successful,
        successful,
    )


class CosmologicalParticleMeshPlan(StrictModule, NonTrainableState):
    """Compose Phydrax PM acceleration with canonical scale-factor KDK."""

    kinematics: CosmologicalKDKPlan
    gravity: ParticleMeshGravityPlan
    scale_factors: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kinematics: CosmologicalKDKPlan,
        gravity: ParticleMeshGravityPlan,
        scale_factors: ArrayLike,
        /,
    ):
        if not isinstance(kinematics, CosmologicalKDKPlan):
            raise TypeError("kinematics must be CosmologicalKDKPlan.")
        if not isinstance(gravity, ParticleMeshGravityPlan):
            raise TypeError("gravity must be ParticleMeshGravityPlan.")
        if kinematics.particles.prepared_id != gravity.transfer.particles.prepared_id:
            raise ValueError("Cosmological KDK and PM must share one particle support.")
        grid = gravity.transfer.plan.target
        if len(grid.axes) != len(kinematics.box_size):
            raise ValueError("Cosmological KDK and PM grid dimensions disagree.")
        for axis, length in zip(grid.axes, kinematics.box_size, strict=True):
            if not axis.periodic or axis.bounds is None:
                raise ValueError("Cosmological PM requires finite periodic grid axes.")
            bounds = np.asarray(axis.bounds, dtype=np.float64)
            if abs(bounds[0]) > 1.0e-12 or abs(bounds[1] - length) > 1.0e-12:
                raise ValueError("Cosmological PM grid bounds must match [0, box_size].")
        schedule_host = np.asarray(scale_factors, dtype=np.float64).reshape((-1,))
        if (
            schedule_host.size < 2
            or np.any(~np.isfinite(schedule_host))
            or np.any(schedule_host <= 0.0)
            or np.any(np.diff(schedule_host) <= 0.0)
        ):
            raise ValueError(
                "Cosmological PM scale factors must be finite, positive, and increasing."
            )
        self.kinematics = kinematics
        self.gravity = gravity
        self.scale_factors = jnp.asarray(schedule_host)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cosmological-particle-mesh-rollout",
                "kinematics": kinematics.plan_id,
                "gravity": gravity.plan_id,
                "scale_factors": schedule_host.tolist(),
            }
        )

    def rollout(
        self,
        background: FLRWBackground,
        state: CosmologicalParticleState,
        args: Any = None,
        /,
    ) -> CosmologicalParticleMeshResult:
        if not isinstance(background, FLRWBackground):
            raise TypeError("background must be FLRWBackground.")
        if not isinstance(state, CosmologicalParticleState):
            raise TypeError("state must be CosmologicalParticleState.")
        if background.scale.scale_id != self.kinematics.scale.scale_id:
            raise ValueError("Background and cosmological PM scales disagree.")
        initial_scale = self.scale_factors[0].astype(state.scale_factor.dtype)
        state_scale = eqx.error_if(
            state.scale_factor,
            jnp.abs(state.scale_factor - initial_scale) > 1.0e-12,
            "Cosmological PM state must start at the first scheduled scale factor.",
        )
        state_scale = background.require_flat(state_scale)
        state = CosmologicalParticleState(
            state.positions,
            state.canonical_momenta,
            state_scale,
        )
        initial_force = self.gravity.acceleration(state.positions, args)
        initial_mass_defect = (
            initial_force.deposited.balance.maximum_absolute_balance_defect
        )
        running = initial_force.successful
        accepted_count = jnp.asarray(0, dtype=jnp.int32)

        def step(carry, end_scale):
            (
                current,
                acceleration_start,
                previous_mass_defect,
                previous_net_force,
                active,
                count,
            ) = carry

            def attempt(_):
                interval = _advance_particle_mesh_interval(
                    self.kinematics,
                    self.gravity,
                    background,
                    current,
                    end_scale,
                    acceleration_start,
                    args,
                )
                successful = active & interval.successful
                mass_defect = jnp.maximum(
                    previous_mass_defect, interval.mass_balance_defect
                )
                next_acceleration = jnp.where(
                    successful,
                    interval.acceleration,
                    acceleration_start,
                )
                next_mass_defect = jnp.where(
                    successful,
                    interval.mass_balance_defect,
                    previous_mass_defect,
                )
                next_net_force = jnp.where(
                    successful,
                    interval.net_force,
                    previous_net_force,
                )
                diagnostics = (
                    interval.drift_factor,
                    interval.first_kick_factor,
                    interval.second_kick_factor,
                    mass_defect,
                    interval.net_force,
                    interval.force_successful,
                    successful,
                )
                next_carry = (
                    interval.state,
                    next_acceleration,
                    next_mass_defect,
                    next_net_force,
                    successful,
                    count + successful.astype(jnp.int32),
                )
                return next_carry, diagnostics

            def stopped(_):
                zero = jnp.asarray(0.0, dtype=current.positions.dtype)
                diagnostics = (
                    zero,
                    zero,
                    zero,
                    previous_mass_defect,
                    previous_net_force,
                    jnp.asarray(False),
                    jnp.asarray(False),
                )
                return carry, diagnostics

            return jax.lax.cond(active, attempt, stopped, operand=None)

        initial_carry = (
            state,
            initial_force.acceleration,
            initial_mass_defect,
            initial_force.net_force,
            running,
            accepted_count,
        )
        final_carry, recorded = jax.lax.scan(
            step,
            initial_carry,
            self.scale_factors[1:].astype(state.scale_factor.dtype),
        )
        final_state, _, _, _, completed, accepted_steps = final_carry
        (
            drift,
            first_kick,
            second_kick,
            mass_defect,
            net_force,
            force_successful,
            accepted,
        ) = recorded
        failed = ~accepted
        first_failed = jnp.where(
            jnp.any(failed),
            jnp.argmax(failed).astype(jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
        )
        diagnostics = CosmologicalParticleMeshDiagnostics(
            drift_factors=drift,
            first_kick_factors=first_kick,
            second_kick_factors=second_kick,
            mass_balance_defects=mass_defect,
            net_forces=net_force,
            force_successful=force_successful,
            accepted=accepted,
            initial_force_successful=initial_force.successful,
            completed=completed,
            accepted_steps=accepted_steps,
            maximum_mass_balance_defect=jnp.max(mass_defect),
            maximum_net_force_norm=jnp.max(jnp.sqrt(jnp.sum(net_force**2, axis=-1))),
            first_failed_step=first_failed,
        )
        return CosmologicalParticleMeshResult(final_state, diagnostics, completed)


__all__ = [
    "CosmologicalParticleMeshDiagnostics",
    "CosmologicalParticleMeshPlan",
    "CosmologicalParticleMeshResult",
]
