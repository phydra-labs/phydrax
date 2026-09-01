#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import pi

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .particle._core import ParticleDiscretization, ParticleSetPlan
from .particle._pairwise import particle_pair_geometry, ParticlePairRelation
from .splatting import ParticleGridSplatPlan, PreparedParticleGridSplat


class ParticleContinuumFields(StrictModule):
    """Balance-audited continuum fields reconstructed from particles and pairs."""

    mass_content: Array
    mass_density: Array
    volume_content: Array
    volume_fraction: Array
    momentum_content: Array
    momentum_density: Array
    raw_momentum_flux_content: Array
    raw_momentum_flux_density: Array
    mean_velocity: Array
    kinetic_stress: Array
    contact_stress_content: Array
    contact_stress: Array
    external_force_content: Array
    external_force_density: Array
    partial_mass_density: Array
    partial_volume_fraction: Array
    partial_momentum_density: Array
    bulk_stress: Array
    supported: Array
    maximum_particle_balance_defect: Array
    contact_stress_balance_defect: Array
    successful: Array


class ParticleCoarseGrainingPlan(StrictModule, NonTrainableState):
    """Structured-grid particle coarse graining with line-integrated pair stress."""

    splat: ParticleGridSplatPlan
    quadrature_order: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        splat: ParticleGridSplatPlan,
        /,
        *,
        quadrature_order: int = 4,
        plan_id: str | None = None,
    ):
        if not isinstance(splat, ParticleGridSplatPlan):
            raise TypeError("splat must be a ParticleGridSplatPlan.")
        order = int(quadrature_order)
        if order < 2 or order > 16:
            raise ValueError("quadrature_order must lie in [2, 16].")
        generated = canonical_fingerprint(
            {
                "kind": "particle-coarse-graining",
                "splat": splat.plan_id,
                "quadrature_order": order,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.splat = splat
        self.quadrature_order = order
        self.plan_id = identifier

    def prepare(
        self, particles: ParticleDiscretization, pair_capacity: int, /
    ) -> PreparedParticleCoarseGraining:
        return PreparedParticleCoarseGraining(self, particles, pair_capacity)


class PreparedParticleCoarseGraining(StrictModule, NonTrainableState):
    plan: ParticleCoarseGrainingPlan
    particles: ParticleDiscretization
    particle_splat: PreparedParticleGridSplat
    segment_splat: PreparedParticleGridSplat
    quadrature_nodes: Array
    quadrature_weights: Array
    pair_capacity: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ParticleCoarseGrainingPlan,
        particles: ParticleDiscretization,
        pair_capacity: int,
        /,
    ):
        if not isinstance(plan, ParticleCoarseGrainingPlan):
            raise TypeError("plan must be a ParticleCoarseGrainingPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        capacity = int(pair_capacity)
        if capacity <= 0:
            raise ValueError("pair_capacity must be positive.")
        nodes, weights = np.polynomial.legendre.leggauss(plan.quadrature_order)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        segment_capacity = capacity * plan.quadrature_order
        segment_particles = ParticleSetPlan(
            np.arange(segment_capacity, dtype=np.int64),
            np.ones((segment_capacity,), dtype=np.dtype(particles.plan.coordinate_dtype)),
            ambient_dimension=particles.ambient_dimension,
            name="interaction-segment-quadrature",
            domain_labels=("interaction_segment", "quadrature_point"),
            coordinate_dtype=particles.plan.coordinate_dtype,
        ).prepare(numeric_version=particles.numeric_version)
        segment_plan = ParticleGridSplatPlan(
            plan.splat.target,
            location=plan.splat.location,
            assignment=plan.splat.assignment,
            boundary=plan.splat.boundary,
            execution=plan.splat.execution,
            precision=plan.splat.precision,
            budget=plan.splat.budget,
        )
        self.plan = plan
        self.particles = particles
        self.particle_splat = plan.splat.prepare(particles)
        self.segment_splat = segment_plan.prepare(segment_particles)
        self.quadrature_nodes = jnp.asarray(nodes, dtype=particles.safe_masses.dtype)
        self.quadrature_weights = jnp.asarray(weights, dtype=particles.safe_masses.dtype)
        self.pair_capacity = capacity
        self.ambient_dimension = particles.ambient_dimension
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-particle-coarse-graining",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "particle_splat": self.particle_splat.prepared_id,
                "segment_splat": self.segment_splat.prepared_id,
                "pair_capacity": capacity,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        velocities: ArrayLike,
        masses: ArrayLike,
        particle_volumes: ArrayLike,
        active_mask: ArrayLike,
        pair_relation: ParticlePairRelation,
        pair_displacement: ArrayLike,
        pair_force: ArrayLike,
        pair_active: ArrayLike,
        /,
        *,
        external_force: ArrayLike | None = None,
        constituent_weights: ArrayLike | None = None,
    ) -> ParticleContinuumFields:
        position = jnp.asarray(positions)
        velocity = jnp.asarray(velocities, dtype=position.dtype)
        mass = jnp.asarray(masses, dtype=position.dtype)
        volume = jnp.asarray(particle_volumes, dtype=position.dtype)
        active = jnp.asarray(active_mask, dtype=bool)
        particle_shape = (self.particles.capacity, self.ambient_dimension)
        if position.shape != particle_shape or velocity.shape != particle_shape:
            raise ValueError(
                f"Particle position and velocity must have shape {particle_shape}."
            )
        if mass.shape != particle_shape[:1] or volume.shape != mass.shape:
            raise ValueError(
                "Particle mass and volume must have particle-capacity shape."
            )
        if active.shape != mass.shape:
            raise ValueError("active_mask must have particle-capacity shape.")
        if not isinstance(pair_relation, ParticlePairRelation):
            raise TypeError("pair_relation must be a ParticlePairRelation.")
        if pair_relation.capacity != self.pair_capacity:
            raise ValueError(
                "Pair relation capacity does not match coarse-graining preparation."
            )
        displacement = jnp.asarray(pair_displacement, dtype=position.dtype)
        force = jnp.asarray(pair_force, dtype=position.dtype)
        interaction_active = jnp.asarray(pair_active, dtype=bool) & pair_relation.valid
        pair_shape = (self.pair_capacity, self.ambient_dimension)
        if displacement.shape != pair_shape or force.shape != pair_shape:
            raise ValueError(f"Pair displacement and force must have shape {pair_shape}.")
        if interaction_active.shape != pair_shape[:1]:
            raise ValueError("pair_active must have pair-capacity shape.")
        applied_external = (
            jnp.zeros_like(position)
            if external_force is None
            else jnp.asarray(external_force, dtype=position.dtype)
        )
        if applied_external.shape != particle_shape:
            raise ValueError("external_force must match particle vector shape.")
        constituents = (
            jnp.ones((self.particles.capacity, 1), dtype=position.dtype)
            if constituent_weights is None
            else jnp.asarray(constituent_weights, dtype=position.dtype)
        )
        if constituents.ndim != 2 or constituents.shape[0] != self.particles.capacity:
            raise ValueError(
                "constituent_weights must have shape (particles, constituents)."
            )

        particle_state = self.particle_splat.build(position, active_mask=active)
        mass_result = self.particle_splat.deposit_content(particle_state, mass)
        volume_result = self.particle_splat.deposit_content(particle_state, volume)
        momentum = mass[:, None] * velocity
        momentum_result = self.particle_splat.deposit_content(particle_state, momentum)
        velocity_outer = contract("pi,pj->pij", velocity, velocity)
        raw_flux_result = self.particle_splat.deposit_content(
            particle_state, mass[:, None, None] * velocity_outer
        )
        external_result = self.particle_splat.deposit_content(
            particle_state, applied_external
        )
        partial_mass_result = self.particle_splat.deposit_content(
            particle_state, mass[:, None] * constituents
        )
        partial_volume_result = self.particle_splat.deposit_content(
            particle_state, volume[:, None] * constituents
        )
        partial_momentum_result = self.particle_splat.deposit_content(
            particle_state,
            momentum[:, None, :] * constituents[:, :, None],
        )

        right = pair_relation.right_indices
        segment_origin = position[right]
        nodes = self.quadrature_nodes.astype(position.dtype)
        weights = self.quadrature_weights.astype(position.dtype)
        segment_positions = (
            segment_origin[:, None, :] + nodes[None, :, None] * displacement[:, None, :]
        ).reshape((-1, self.ambient_dimension))
        segment_active = jnp.broadcast_to(
            interaction_active[:, None],
            (self.pair_capacity, self.plan.quadrature_order),
        ).reshape((-1,))
        segment_state = self.segment_splat.build(
            segment_positions, active_mask=segment_active
        )
        pair_virial = -contract("pi,pj->pij", force, displacement)
        segment_content = (
            pair_virial[:, None, :, :] * weights[None, :, None, None]
        ).reshape((-1, self.ambient_dimension, self.ambient_dimension))
        contact_result = self.segment_splat.deposit_content(
            segment_state, segment_content
        )

        density = mass_result.density
        supported = density > 64.0 * jnp.finfo(density.dtype).eps
        safe_density = jnp.where(supported, density, 1.0)
        mean_velocity = jnp.where(
            supported[..., None],
            momentum_result.density / safe_density[..., None],
            0.0,
        )
        advective_flux = contract(
            "...i,...j->...ij", momentum_result.density, mean_velocity
        )
        kinetic_stress = -(raw_flux_result.density - advective_flux)
        target_measure = self.particle_splat.target_measure.weights.reshape(
            self.particle_splat.target_shape
        ).astype(position.dtype)
        total_measure = jnp.sum(target_measure)
        total_stress_content = (
            contact_result.content + kinetic_stress * target_measure[..., None, None]
        )
        bulk_stress = (
            jnp.sum(
                total_stress_content.reshape(
                    (-1, self.ambient_dimension, self.ambient_dimension)
                ),
                axis=0,
            )
            / total_measure
        )
        particle_defects = jnp.stack(
            (
                mass_result.balance.maximum_absolute_balance_defect,
                volume_result.balance.maximum_absolute_balance_defect,
                momentum_result.balance.maximum_absolute_balance_defect,
                raw_flux_result.balance.maximum_absolute_balance_defect,
                external_result.balance.maximum_absolute_balance_defect,
                partial_mass_result.balance.maximum_absolute_balance_defect,
                partial_volume_result.balance.maximum_absolute_balance_defect,
                partial_momentum_result.balance.maximum_absolute_balance_defect,
            )
        )
        successful = (
            mass_result.successful
            & volume_result.successful
            & momentum_result.successful
            & raw_flux_result.successful
            & external_result.successful
            & partial_mass_result.successful
            & partial_volume_result.successful
            & partial_momentum_result.successful
            & contact_result.successful
            & jnp.all(jnp.isfinite(bulk_stress))
        )
        return ParticleContinuumFields(
            mass_result.content,
            mass_result.density,
            volume_result.content,
            volume_result.density,
            momentum_result.content,
            momentum_result.density,
            raw_flux_result.content,
            raw_flux_result.density,
            mean_velocity,
            kinetic_stress,
            contact_result.content,
            contact_result.density,
            external_result.content,
            external_result.density,
            partial_mass_result.density,
            partial_volume_result.density,
            partial_momentum_result.density,
            bulk_stress,
            supported,
            jnp.max(particle_defects),
            contact_result.balance.maximum_absolute_balance_defect,
            successful,
        )

    def evaluate_dem(self, dynamics, state, evaluation, /) -> ParticleContinuumFields:
        from .particle._dem import (
            DEMEvaluation,
            DEMRuntimeState,
            PreparedSoftSphereDEMDynamics,
        )

        if not isinstance(dynamics, PreparedSoftSphereDEMDynamics):
            raise TypeError("dynamics must be PreparedSoftSphereDEMDynamics.")
        if not isinstance(state, DEMRuntimeState):
            raise TypeError("state must be DEMRuntimeState.")
        if not isinstance(evaluation, DEMEvaluation):
            raise TypeError("evaluation must be DEMEvaluation.")
        pairs = evaluation.neighborhood.pair_relation
        geometry = particle_pair_geometry(
            state.kinematics.position,
            pairs,
            box=evaluation.neighborhood.box,
            cell_vectors=(
                None if state.periodic_cell is None else state.periodic_cell.vectors
            ),
        )
        radii = state.body_properties.radii
        particle_volume = (
            pi * radii**2 if self.ambient_dimension == 2 else (4.0 * pi / 3.0) * radii**3
        )
        noncontact_force = (
            evaluation.loads.total.force - evaluation.loads.particle_contact.force
        )
        return self.evaluate(
            state.kinematics.position,
            state.kinematics.velocity,
            state.body_properties.masses,
            particle_volume,
            state.body_properties.active,
            pairs,
            geometry.displacement,
            evaluation.particle_contact.pair_force,
            evaluation.particle_contact.active,
            external_force=noncontact_force,
        )


__all__ = [
    "ParticleCoarseGrainingPlan",
    "ParticleContinuumFields",
    "PreparedParticleCoarseGraining",
]
