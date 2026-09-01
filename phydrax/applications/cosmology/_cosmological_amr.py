#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import PreparedNewtonianSelfGravity
from ._particles import CosmologicalParticleState


class TwoLevelAMRState(StrictModule):
    coarse_cell_average: Array
    fine_cell_average: Array
    refined_parent_mask: Array
    scale_factor: Array


class TwoLevelAMRPlan(StrictModule, NonTrainableState):
    coarse_shape: tuple[int, ...] = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    refinement_ratio: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, coarse_shape: tuple[int, ...], component_count: int, /):
        shape = tuple(int(value) for value in coarse_shape)
        components = int(component_count)
        if (
            not shape
            or len(shape) not in (1, 2, 3)
            or any(value <= 0 for value in shape)
            or components <= 0
        ):
            raise ValueError("Two-level AMR shape/components are invalid.")
        self.coarse_shape = shape
        self.component_count = components
        self.refinement_ratio = 2
        self.plan_id = canonical_fingerprint(
            {
                "kind": "two-level-ratio-two-amr",
                "coarse_shape": list(shape),
                "component_count": components,
            }
        )

    @property
    def fine_shape(self) -> tuple[int, ...]:
        return tuple(2 * value for value in self.coarse_shape)

    def initialize(
        self,
        coarse_cell_average: ArrayLike,
        refined_parent_mask: ArrayLike,
        scale_factor: ArrayLike,
        /,
    ) -> TwoLevelAMRState:
        coarse = jnp.asarray(coarse_cell_average)
        mask = jnp.asarray(refined_parent_mask, dtype=bool)
        if (
            coarse.shape != self.coarse_shape + (self.component_count,)
            or mask.shape != self.coarse_shape
        ):
            raise ValueError("Two-level AMR coarse state/mask shapes are invalid.")
        fine = self.prolong(coarse)
        return TwoLevelAMRState(
            coarse, fine, mask, jnp.asarray(scale_factor, dtype=coarse.dtype)
        )

    def prolong(self, coarse: ArrayLike, /) -> Array:
        values = jnp.asarray(coarse)
        result = values
        for axis in range(len(self.coarse_shape)):
            result = jnp.repeat(result, 2, axis=axis)
        return result

    def restrict(self, fine: ArrayLike, /) -> Array:
        values = jnp.asarray(fine)
        expected = self.fine_shape + (self.component_count,)
        if values.shape != expected:
            raise ValueError(f"Fine AMR values must have shape {expected}.")
        reshaped = values
        for axis in reversed(range(len(self.coarse_shape))):
            shape = reshaped.shape
            reshaped = reshaped.reshape(
                shape[:axis] + (self.coarse_shape[axis], 2) + shape[axis + 1 :]
            )
            reshaped = jnp.mean(reshaped, axis=axis + 1)
        return reshaped

    def average_down(self, state: TwoLevelAMRState, /) -> TwoLevelAMRState:
        restricted = self.restrict(state.fine_cell_average)
        coarse = jnp.where(
            state.refined_parent_mask[..., None],
            restricted,
            state.coarse_cell_average,
        )
        return TwoLevelAMRState(
            coarse, state.fine_cell_average, state.refined_parent_mask, state.scale_factor
        )


class CoarseFineFluxRegister(StrictModule):
    coarse_flux_integral: Array
    fine_flux_integral: Array
    mismatch: Array
    finite: Array

    def __init__(self, coarse_flux_integral: ArrayLike, fine_flux_integral: ArrayLike, /):
        coarse = jnp.asarray(coarse_flux_integral)
        fine = jnp.asarray(fine_flux_integral, dtype=coarse.dtype)
        if coarse.shape != fine.shape:
            raise ValueError("Coarse and fine flux integrals must share a face layout.")
        mismatch = coarse - fine
        self.coarse_flux_integral = coarse
        self.fine_flux_integral = fine
        self.mismatch = mismatch
        self.finite = jnp.all(jnp.isfinite(mismatch))

    def reflux(
        self,
        coarse_cell_average: ArrayLike,
        left_cell_indices: ArrayLike,
        right_cell_indices: ArrayLike,
        cell_volumes: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(coarse_cell_average)
        left = jnp.asarray(left_cell_indices, dtype=jnp.int32)
        right = jnp.asarray(right_cell_indices, dtype=jnp.int32)
        volume = jnp.asarray(cell_volumes, dtype=values.dtype)
        flat = values.reshape((-1, values.shape[-1]))
        flat = flat.at[left].add(-self.mismatch / volume[left, None])
        flat = flat.at[right].add(self.mismatch / volume[right, None])
        return flat.reshape(values.shape)


class TwoLevelGravityResult(StrictModule):
    fine_potential: Array
    fine_acceleration: Array
    coarse_acceleration: Array
    density_consistency: Array
    converged: Array
    successful: Array


class TwoLevelCompositeGravityPlan(StrictModule, NonTrainableState):
    amr: TwoLevelAMRPlan
    fine_gravity: PreparedNewtonianSelfGravity

    def __init__(
        self, amr: TwoLevelAMRPlan, fine_gravity: PreparedNewtonianSelfGravity, /
    ):
        self.amr = amr
        self.fine_gravity = fine_gravity

    def solve(
        self,
        coarse_density: ArrayLike,
        fine_density: ArrayLike,
        args=None,
        /,
    ) -> TwoLevelGravityResult:
        coarse = jnp.asarray(coarse_density)
        fine = jnp.asarray(fine_density, dtype=coarse.dtype)
        potential, _, fine_acceleration, evidence = self.fine_gravity.solve_density(
            fine, args
        )
        fine_acceleration_components = tuple(
            self.amr.restrict(fine_acceleration[..., component : component + 1])[..., 0]
            for component in range(len(self.amr.coarse_shape))
        )
        coarse_acceleration = jnp.stack(fine_acceleration_components, axis=-1)
        restricted_density = self.amr.restrict(fine[..., None])[..., 0]
        consistency = jnp.max(jnp.abs(restricted_density - coarse))
        successful = evidence.converged & jnp.all(jnp.isfinite(coarse_acceleration))
        return TwoLevelGravityResult(
            potential,
            fine_acceleration,
            coarse_acceleration,
            consistency,
            evidence.converged,
            successful,
        )


class AMRParticleLevelAssignment(StrictModule):
    levels: Array
    fine_indices: Array
    coarse_indices: Array
    routed: Array
    successful: Array


class TwoLevelParticleRoutingPlan(StrictModule, NonTrainableState):
    amr: TwoLevelAMRPlan
    box_size: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, amr: TwoLevelAMRPlan, box_size: tuple[float, ...], /):
        lengths = tuple(float(value) for value in box_size)
        if len(lengths) != len(amr.coarse_shape):
            raise ValueError("AMR particle box dimension disagrees with hierarchy.")
        self.amr = amr
        self.box_size = lengths

    def route(
        self, positions: ArrayLike, refined_parent_mask: ArrayLike, /
    ) -> AMRParticleLevelAssignment:
        position = jnp.asarray(positions)
        mask = jnp.asarray(refined_parent_mask, dtype=bool)
        coarse_shape = jnp.asarray(self.amr.coarse_shape)
        box = jnp.asarray(self.box_size, dtype=position.dtype)
        coarse_index = jnp.floor(position / box * coarse_shape).astype(jnp.int32)
        coarse_index = jnp.clip(coarse_index, 0, coarse_shape - 1)
        fine_index = jnp.floor(position / box * (2 * coarse_shape)).astype(jnp.int32)
        fine_index = jnp.clip(fine_index, 0, 2 * coarse_shape - 1)
        refined = mask[tuple(coarse_index[:, axis] for axis in range(position.shape[1]))]
        levels = refined.astype(jnp.int8)
        routed = jnp.all(jnp.isfinite(position))
        return AMRParticleLevelAssignment(
            levels, fine_index, coarse_index, routed, routed
        )


class AMREpochResult(StrictModule):
    state: TwoLevelAMRState
    particles: CosmologicalParticleState
    reflux_finite: Array
    gravity_successful: Array
    particle_routed: Array
    successful: Array


class TwoLevelAMREpochPlan(StrictModule, NonTrainableState):
    amr: TwoLevelAMRPlan
    routing: TwoLevelParticleRoutingPlan

    def __init__(self, amr: TwoLevelAMRPlan, routing: TwoLevelParticleRoutingPlan, /):
        self.amr = amr
        self.routing = routing

    def commit(
        self,
        previous_state: TwoLevelAMRState,
        candidate_state: TwoLevelAMRState,
        previous_particles: CosmologicalParticleState,
        candidate_particles: CosmologicalParticleState,
        flux_register: CoarseFineFluxRegister,
        gravity_successful: ArrayLike,
        /,
    ) -> AMREpochResult:
        averaged = self.amr.average_down(candidate_state)
        routed = self.routing.route(
            candidate_particles.positions, averaged.refined_parent_mask
        )
        success = (
            flux_register.finite & jnp.asarray(gravity_successful) & routed.successful
        )
        state = TwoLevelAMRState(
            jnp.where(
                success, averaged.coarse_cell_average, previous_state.coarse_cell_average
            ),
            jnp.where(
                success, averaged.fine_cell_average, previous_state.fine_cell_average
            ),
            previous_state.refined_parent_mask,
            jnp.where(success, averaged.scale_factor, previous_state.scale_factor),
        )
        particles = CosmologicalParticleState(
            jnp.where(
                success, candidate_particles.positions, previous_particles.positions
            ),
            jnp.where(
                success,
                candidate_particles.canonical_momenta,
                previous_particles.canonical_momenta,
            ),
            jnp.where(
                success, candidate_particles.scale_factor, previous_particles.scale_factor
            ),
        )
        return AMREpochResult(
            state,
            particles,
            flux_register.finite,
            jnp.asarray(gravity_successful),
            routed.successful,
            success,
        )


__all__ = [
    "AMREpochResult",
    "AMRParticleLevelAssignment",
    "CoarseFineFluxRegister",
    "TwoLevelAMREpochPlan",
    "TwoLevelAMRPlan",
    "TwoLevelAMRState",
    "TwoLevelCompositeGravityPlan",
    "TwoLevelGravityResult",
    "TwoLevelParticleRoutingPlan",
]
