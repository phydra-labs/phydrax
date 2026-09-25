#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle import (
    AbstractPreparedParticleNeighborhood,
    AbstractSPHSmoothingKernel,
    adaptive_smoothing_state,
    CoupledSummationSmoothingLengthPlan,
    particle_pair_geometry,
    ParticleBox,
    ParticleExecutionPolicy,
    ParticlePairRelation,
    scatter_elastic_pairs,
)
from ._background import FLRWBackground
from ._dark_sector_species import DarkSectorSpeciesPlan
from ._distances import FLRWDistancePlan
from ._particle_mesh import (
    _advance_particle_mesh_interval,
    CosmologicalParticleMeshDiagnostics,
    CosmologicalParticleMeshPlan,
)
from ._particles import CosmologicalParticleState
from ._sidm_kernels import (
    angles_from_direction,
    directions_from_angles,
    TwoBodyDifferentialKernelPlan,
)


class SIDMCrossSectionPlan(StrictModule, NonTrainableState):
    """Constant isotropic elastic SIDM cross section per physical mass."""

    cross_section_per_mass: float = eqx.field(static=True)
    cross_section_per_mass_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    kernel: TwoBodyDifferentialKernelPlan

    def __init__(self, cross_section_per_mass: float, /):
        value = float(cross_section_per_mass)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "SIDM cross_section_per_mass must be finite and nonnegative."
            )
        self.cross_section_per_mass = value
        self.cross_section_per_mass_unit = "physical-area/physical-mass"
        reference_species = DarkSectorSpeciesPlan(
            "constant-isotropic-sidm-reference",
            1.0,
            mass_unit="physical-mass",
        )
        self.kernel = TwoBodyDifferentialKernelPlan.constant_isotropic(
            reference_species,
            value * reference_species.mass,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "constant-isotropic-elastic-sidm-cross-section",
                "cross_section_per_mass": value,
            }
        )


class SIDMCollisionPolicy(StrictModule, NonTrainableState):
    """Validity envelope for the rare, pairwise SIDM realization."""

    maximum_pair_probability: float = eqx.field(static=True)
    maximum_particle_probability: float = eqx.field(static=True)
    minimum_knudsen_number: float = eqx.field(static=True)
    maximum_events_per_half_step: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_pair_probability: float = 0.1,
        maximum_particle_probability: float = 0.25,
        minimum_knudsen_number: float = 1.0,
        maximum_events_per_half_step: int,
    ):
        maximum = float(maximum_pair_probability)
        maximum_particle = float(maximum_particle_probability)
        minimum_knudsen = float(minimum_knudsen_number)
        event_capacity = int(maximum_events_per_half_step)
        if not np.isfinite(maximum) or not 0.0 < maximum <= 1.0:
            raise ValueError("maximum_pair_probability must lie in (0,1].")
        if not np.isfinite(maximum_particle) or not 0.0 < maximum_particle <= 1.0:
            raise ValueError("maximum_particle_probability must lie in (0,1].")
        if not np.isfinite(minimum_knudsen) or minimum_knudsen <= 0.0:
            raise ValueError("minimum_knudsen_number must be finite and positive.")
        if event_capacity < 0:
            raise ValueError("maximum_events_per_half_step must be nonnegative.")
        self.maximum_pair_probability = maximum
        self.maximum_particle_probability = maximum_particle
        self.minimum_knudsen_number = minimum_knudsen
        self.maximum_events_per_half_step = event_capacity
        self.policy_id = canonical_fingerprint(
            {
                "kind": "rare-sidm-collision-policy",
                "maximum_pair_probability": maximum,
                "maximum_particle_probability": maximum_particle,
                "minimum_knudsen_number": minimum_knudsen,
                "maximum_events_per_half_step": event_capacity,
            }
        )


class SIDMCollisionDiagnostics(StrictModule, NonTrainableState):
    """Fixed-shape physical evidence for one SIDM collision half-step."""

    physical_time_step: Array
    smoothing_length_comoving: Array
    density_comoving: Array
    density_physical: Array
    kernel_weight_comoving: Array
    kernel_weight_physical: Array
    relative_speed_physical: Array
    pair_probability: Array
    kernel_total_cross_section: Array
    kernel_total_cross_section_per_mass: Array
    kernel_supported: Array
    sampled_cosine: Array
    sampled_azimuth: Array
    angular_sample_successful: Array
    particle_aggregate_probability: Array
    random_uniform: Array
    proposed_pairs: Array
    selected_pairs: Array
    accepted_pairs: Array
    event_count: Array
    mean_free_path_physical: Array
    support_radius_physical: Array
    knudsen_number: Array
    pair_momentum_defect: Array
    pair_kinetic_energy_defect: Array
    total_momentum_defect: Array
    total_kinetic_energy_defect: Array
    neighborhood_successful: Array
    smoothing_converged: Array
    smoothing_within_bounds: Array
    equal_active_mass: Array
    probability_valid: Array
    aggregate_probability_valid: Array
    capacity_valid: Array
    knudsen_valid: Array
    endpoint_disjoint: Array
    inactive_preserved: Array
    conservative: Array
    finite: Array
    successful: Array


class SIDMCollisionResult(StrictModule, NonTrainableState):
    """Candidate and atomically accepted state for one SIDM half-step."""

    candidate_state: CosmologicalParticleState
    accepted_state: CosmologicalParticleState
    pairs: ParticlePairRelation
    diagnostics: SIDMCollisionDiagnostics
    successful: Array


class CosmologicalSIDMDiagnostics(StrictModule, NonTrainableState):
    """PM and split-collision evidence for a complete SIDM rollout."""

    particle_mesh: CosmologicalParticleMeshDiagnostics
    first_half_collisions: SIDMCollisionDiagnostics
    second_half_collisions: SIDMCollisionDiagnostics
    accepted: Array
    completed: Array
    accepted_steps: Array
    first_failed_step: Array


class CosmologicalSIDMResult(StrictModule, NonTrainableState):
    state: CosmologicalParticleState
    diagnostics: CosmologicalSIDMDiagnostics
    successful: Array


def _identity_key(key: Array, identity: Array, /) -> Array:
    identity_ = jnp.asarray(identity, dtype=jnp.int64)
    low = identity_.astype(jnp.uint32)
    high = jnp.right_shift(identity_, 32).astype(jnp.uint32)
    return jr.fold_in(jr.fold_in(key, low), high)


def _pair_keys(key: Array, pairs: ParticlePairRelation, epoch: ArrayLike, /) -> Array:
    epoch_ = jnp.asarray(epoch, dtype=jnp.int64).reshape(())
    root = _identity_key(key, epoch_)
    return jax.vmap(lambda left, right: _identity_key(_identity_key(root, left), right))(
        pairs.left_particle_ids, pairs.right_particle_ids
    )


def _isotropic_directions(keys: Array, dtype, /) -> Array:
    samples = jax.vmap(lambda key: jr.normal(key, (3,), dtype=dtype))(keys)
    norms = jnp.sqrt(ein.contract("...i,...i->...", samples, samples))
    fallback = jnp.asarray((1.0, 0.0, 0.0), dtype=dtype)
    safe_norms = jnp.where(norms > 0.0, norms, 1.0)
    return jnp.where(
        (norms > 0.0)[:, None],
        samples / safe_norms[:, None],
        fallback,
    )


def _select_endpoint_disjoint(
    proposed: Array,
    priority: Array,
    pairs: ParticlePairRelation,
    particle_capacity: int,
    /,
) -> Array:
    order = jnp.lexsort((pairs.right_particle_ids, pairs.left_particle_ids, priority))
    selected = jnp.zeros(proposed.shape, dtype=jnp.bool_)
    used = jnp.zeros((particle_capacity,), dtype=jnp.bool_)

    def body(index, carry):
        accepted, occupied = carry
        route = order[index]
        left = pairs.left_indices[route]
        right = pairs.right_indices[route]
        take = proposed[route] & ~occupied[left] & ~occupied[right]
        accepted = accepted.at[route].set(take)
        occupied = occupied.at[left].set(occupied[left] | take)
        occupied = occupied.at[right].set(occupied[right] | take)
        return accepted, occupied

    return jax.lax.fori_loop(0, proposed.size, body, (selected, used))[0]


class CosmologicalSIDMPlan(StrictModule):
    """Rare elastic SIDM split around canonical particle-mesh intervals."""

    particle_mesh: CosmologicalParticleMeshPlan
    neighborhood: AbstractPreparedParticleNeighborhood
    smoothing: CoupledSummationSmoothingLengthPlan
    kernel: AbstractSPHSmoothingKernel
    execution: ParticleExecutionPolicy
    time: FLRWDistancePlan
    cross_section: SIDMCrossSectionPlan | TwoBodyDifferentialKernelPlan
    scattering_kernel: TwoBodyDifferentialKernelPlan
    policy: SIDMCollisionPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_mesh: CosmologicalParticleMeshPlan,
        neighborhood: AbstractPreparedParticleNeighborhood,
        smoothing: CoupledSummationSmoothingLengthPlan,
        kernel: AbstractSPHSmoothingKernel,
        cross_section: SIDMCrossSectionPlan | TwoBodyDifferentialKernelPlan,
        policy: SIDMCollisionPolicy,
        /,
        *,
        execution: ParticleExecutionPolicy | None = None,
        time: FLRWDistancePlan | None = None,
    ):
        if not isinstance(particle_mesh, CosmologicalParticleMeshPlan):
            raise TypeError("particle_mesh must be CosmologicalParticleMeshPlan.")
        if not isinstance(neighborhood, AbstractPreparedParticleNeighborhood):
            raise TypeError("neighborhood must be a prepared particle neighborhood.")
        if not isinstance(smoothing, CoupledSummationSmoothingLengthPlan):
            raise TypeError("smoothing must be CoupledSummationSmoothingLengthPlan.")
        if not isinstance(kernel, AbstractSPHSmoothingKernel):
            raise TypeError("kernel must be an AbstractSPHSmoothingKernel.")
        if not isinstance(
            cross_section, (SIDMCrossSectionPlan, TwoBodyDifferentialKernelPlan)
        ):
            raise TypeError(
                "cross_section must be SIDMCrossSectionPlan or TwoBodyDifferentialKernelPlan."
            )
        scattering_kernel = (
            cross_section.kernel
            if isinstance(cross_section, SIDMCrossSectionPlan)
            else cross_section
        )
        if (
            scattering_kernel.first_species.species_plan_id
            != scattering_kernel.second_species.species_plan_id
        ):
            raise ValueError(
                "Cosmological elastic SIDM requires one microscopic incoming species."
            )
        if not isinstance(policy, SIDMCollisionPolicy):
            raise TypeError("policy must be SIDMCollisionPolicy.")
        particles = particle_mesh.kinematics.particles
        if neighborhood.particle_discretization_id != particles.prepared_id:
            raise ValueError("SIDM neighborhood and PM must share one particle support.")
        if particles.ambient_dimension != 3 or kernel.dimension != 3:
            raise ValueError(
                "Cosmological SIDM requires three-dimensional particle support."
            )
        if not isinstance(neighborhood.box, ParticleBox):
            raise ValueError(
                "Cosmological SIDM requires a bounded axis-aligned ParticleBox."
            )
        lengths = np.asarray(neighborhood.box.lengths)
        if tuple(neighborhood.box.periodic_axes) != (True, True, True) or not np.allclose(
            lengths, np.asarray(particle_mesh.kinematics.box_size)
        ):
            raise ValueError("SIDM neighborhood must share the periodic PM box.")
        execution_ = ParticleExecutionPolicy() if execution is None else execution
        time_ = FLRWDistancePlan() if time is None else time
        if not isinstance(time_, FLRWDistancePlan):
            raise TypeError("time must be FLRWDistancePlan.")
        if not isinstance(execution_, ParticleExecutionPolicy):
            raise TypeError("execution must be ParticleExecutionPolicy.")
        self.particle_mesh = particle_mesh
        self.neighborhood = neighborhood
        self.smoothing = smoothing
        self.kernel = kernel
        self.time = time_
        self.execution = execution_
        self.cross_section = cross_section
        self.scattering_kernel = scattering_kernel
        self.policy = policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": (
                    "cosmological-rare-isotropic-sidm"
                    if isinstance(cross_section, SIDMCrossSectionPlan)
                    else "cosmological-rare-differential-sidm"
                ),
                "particle_mesh": particle_mesh.plan_id,
                "neighborhood": neighborhood.prepared_id,
                "smoothing": smoothing.plan_id,
                "kernel": kernel.kernel_id,
                "execution": execution_.policy_id,
                "time": time_.plan_id,
                "cross_section": (
                    cross_section.plan_id
                    if isinstance(cross_section, SIDMCrossSectionPlan)
                    else cross_section.kernel_id
                ),
                "policy": policy.policy_id,
            }
        )

    def physical_time_between(
        self,
        background: FLRWBackground,
        start_scale_factor: ArrayLike,
        end_scale_factor: ArrayLike,
        /,
    ) -> Array:
        """Integrate dt = da / (a H(a)) with the owned FLRW quadrature."""

        return self.time.cosmic_time_between(
            background, start_scale_factor, end_scale_factor
        )

    def collide(
        self,
        state: CosmologicalParticleState,
        key: Array,
        epoch: ArrayLike,
        physical_time_step: ArrayLike,
        /,
    ) -> SIDMCollisionResult:
        """Apply one atomic collision stage at the state's fixed scale factor."""

        if not isinstance(state, CosmologicalParticleState):
            raise TypeError("state must be CosmologicalParticleState.")
        particles = self.particle_mesh.kinematics.particles
        dtype = state.positions.dtype
        active = particles.active_mask
        active_column = active[:, None]
        state_finite = (
            jnp.all(jnp.isfinite(state.positions) | ~active_column)
            & jnp.all(jnp.isfinite(state.canonical_momenta) | ~active_column)
            & jnp.isfinite(state.scale_factor)
            & (state.scale_factor > 0.0)
        )
        safe_positions = jnp.where(jnp.isfinite(state.positions), state.positions, 0.0)
        safe_momenta = jnp.where(
            jnp.isfinite(state.canonical_momenta),
            state.canonical_momenta,
            0.0,
        )
        safe_scale = jnp.where(
            jnp.isfinite(state.scale_factor) & (state.scale_factor > 0.0),
            state.scale_factor,
            1.0,
        )
        dt = jnp.asarray(physical_time_step, dtype=dtype).reshape(())
        time_valid = jnp.isfinite(dt) & (dt >= 0.0)
        safe_dt = jnp.where(time_valid, dt, 0.0)

        neighborhood = self.neighborhood.build(safe_positions, active_mask=active)
        pairs = neighborhood.pair_relation
        geometry = particle_pair_geometry(safe_positions, pairs, box=neighborhood.box)
        adaptive = adaptive_smoothing_state(
            self.smoothing,
            particles,
            pairs,
            geometry,
            self.kernel,
            self.execution,
        )
        h = adaptive.smoothing_length.astype(dtype)
        left = pairs.left_indices
        right = pairs.right_indices
        support_pair = pairs.valid & (
            geometry.distance
            < self.kernel.support_factor * jnp.maximum(h[left], h[right])
        )
        left_kernel = self.kernel.value(geometry.distance, h[left])
        right_kernel = self.kernel.value(geometry.distance, h[right])
        kernel_comoving = jnp.where(support_pair, 0.5 * (left_kernel + right_kernel), 0.0)
        kernel_physical = kernel_comoving / safe_scale**3

        masses = particles.safe_masses.astype(dtype)
        reference_index = jnp.argmax(active.astype(jnp.int32))
        reference_mass = masses[reference_index]
        equal_mass = jnp.all(~active | (masses == reference_mass))
        velocities = safe_momenta / (masses[:, None] * safe_scale)
        relative = velocities[left] - velocities[right]
        relative_speed = jnp.sqrt(ein.contract("...i,...i->...", relative, relative))
        pair_mass = 0.5 * (masses[left] + masses[right])
        kernel_moments = self.scattering_kernel.moments(relative_speed)
        kernel_total = kernel_moments.total.astype(dtype)
        kernel_supported = (
            kernel_moments.supported & jnp.isfinite(kernel_total) & (kernel_total >= 0.0)
        )
        if isinstance(self.cross_section, SIDMCrossSectionPlan):
            cross_section_per_mass = jnp.full(
                relative_speed.shape,
                self.cross_section.cross_section_per_mass,
                dtype=dtype,
            )
        else:
            microscopic_mass = jnp.asarray(
                self.scattering_kernel.first_species.mass, dtype=dtype
            )
            cross_section_per_mass = kernel_total / microscopic_mass
        safe_cross_section_per_mass = jnp.where(
            kernel_supported, cross_section_per_mass, 0.0
        )
        probability = (
            safe_cross_section_per_mass
            * pair_mass
            * relative_speed
            * safe_dt
            * kernel_physical
        )
        probability = jnp.where(support_pair, probability, 0.0)
        probability_valid = jnp.all(
            ~support_pair
            | (
                jnp.isfinite(probability)
                & (probability >= 0.0)
                & (probability <= self.policy.maximum_pair_probability)
            )
        )
        particle_probability = jnp.zeros((particles.capacity,), dtype=dtype)
        particle_probability = particle_probability.at[left].add(probability)
        particle_probability = particle_probability.at[right].add(probability)
        aggregate_probability_valid = jnp.all(
            ~active
            | (
                jnp.isfinite(particle_probability)
                & (particle_probability <= self.policy.maximum_particle_probability)
            )
        )

        keys = _pair_keys(key, pairs, epoch)
        uniforms = jax.vmap(
            lambda local: jr.uniform(jr.fold_in(local, 0), (), dtype=dtype)
        )(keys)
        priorities = jax.vmap(
            lambda local: jr.uniform(jr.fold_in(local, 1), (), dtype=dtype)
        )(keys)
        direction_keys = jax.vmap(lambda local: jr.fold_in(local, 2))(keys)
        if (
            isinstance(self.cross_section, SIDMCrossSectionPlan)
            or self.scattering_kernel.isotropic_specialization
        ):
            directions = _isotropic_directions(direction_keys, dtype)
            sampled_cosine, sampled_azimuth = angles_from_direction(relative, directions)
            angular_sample_successful = jnp.ones(relative_speed.shape, dtype=jnp.bool_)
        else:
            angular_samples = jax.vmap(self.scattering_kernel.sample_angles)(
                direction_keys, relative_speed
            )
            sampled_cosine = angular_samples.cosine
            sampled_azimuth = angular_samples.azimuth
            angular_sample_successful = (
                angular_samples.supported
                & jnp.isfinite(angular_samples.normalization_residual)
                & (
                    angular_samples.normalization_residual
                    <= self.scattering_kernel.normalization_tolerance
                )
            )
            directions = directions_from_angles(relative, sampled_cosine, sampled_azimuth)
        requires_angular_sample = (
            support_pair & (relative_speed > 0.0) & (kernel_total > 0.0)
        )
        kernel_domain_valid = jnp.all(
            ~support_pair | (relative_speed == 0.0) | kernel_supported
        )
        angular_sampling_valid = jnp.all(
            ~requires_angular_sample | angular_sample_successful
        )
        proposed = support_pair & (relative_speed > 0.0) & (uniforms < probability)
        selected = _select_endpoint_disjoint(
            proposed, priorities, pairs, particles.capacity
        )
        event_count = jnp.sum(selected, dtype=jnp.int32)
        capacity_valid = event_count <= self.policy.maximum_events_per_half_step

        scattering = scatter_elastic_pairs(
            velocities[left],
            velocities[right],
            masses[left],
            masses[right],
            directions,
            mask=selected,
        )
        first_delta = jnp.where(
            selected[:, None], scattering.first_velocity - velocities[left], 0.0
        )
        second_delta = jnp.where(
            selected[:, None], scattering.second_velocity - velocities[right], 0.0
        )
        first_momentum_delta = first_delta * masses[left, None] * safe_scale
        second_momentum_delta = second_delta * masses[right, None] * safe_scale
        candidate_momenta = safe_momenta.at[left].add(first_momentum_delta)
        candidate_momenta = candidate_momenta.at[right].add(second_momentum_delta)
        candidate_momenta = jnp.where(
            active_column, candidate_momenta, state.canonical_momenta
        )
        candidate_state = CosmologicalParticleState(
            state.positions, candidate_momenta, state.scale_factor
        )

        density_comoving = adaptive.density.astype(dtype)
        density_physical = density_comoving / safe_scale**3
        if isinstance(self.cross_section, SIDMCrossSectionPlan):
            sigma = jnp.full(
                (particles.capacity,),
                self.cross_section.cross_section_per_mass,
                dtype=dtype,
            )
        else:
            pair_sigma = jnp.where(
                support_pair & kernel_supported, safe_cross_section_per_mass, 0.0
            )
            sigma = jnp.zeros((particles.capacity,), dtype=dtype)
            sigma = sigma.at[left].max(pair_sigma)
            sigma = sigma.at[right].max(pair_sigma)
        inverse_mean_free_path = density_physical * sigma
        mean_free_path = jnp.where(
            inverse_mean_free_path > 0.0,
            1.0 / inverse_mean_free_path,
            jnp.inf,
        )
        support_radius = safe_scale * self.kernel.support_factor * h
        knudsen = mean_free_path / support_radius
        knudsen_valid = jnp.all(
            ~active
            | (sigma == 0.0)
            | (jnp.isfinite(knudsen) & (knudsen >= self.policy.minimum_knudsen_number))
        )
        endpoint_count = jnp.zeros((particles.capacity,), dtype=jnp.int32)
        endpoint_count = endpoint_count.at[left].add(selected.astype(jnp.int32))
        endpoint_count = endpoint_count.at[right].add(selected.astype(jnp.int32))
        endpoint_disjoint = jnp.all(endpoint_count <= 1)
        inactive_preserved = jnp.all(
            jnp.where(
                active_column,
                True,
                candidate_momenta == state.canonical_momenta,
            )
        )
        total_momentum_defect = jnp.sum(scattering.momentum_defect, axis=0)
        total_energy_defect = jnp.sum(scattering.kinetic_energy_defect)
        particle_momentum = masses[:, None] * velocities
        energy_before = 0.5 * jnp.sum(
            masses * ein.contract("...i,...i->...", velocities, velocities)
        )
        epsilon = jnp.finfo(dtype).eps
        tiny = jnp.finfo(dtype).tiny
        momentum_scale = jnp.maximum(
            jnp.sum(
                jnp.sqrt(
                    ein.contract(
                        "...i,...i->...",
                        particle_momentum,
                        particle_momentum,
                    )
                )
            ),
            tiny,
        )
        energy_scale = jnp.maximum(jnp.abs(energy_before), tiny)
        conservative = (
            jnp.sqrt(ein.contract("i,i->", total_momentum_defect, total_momentum_defect))
            <= 1024.0 * epsilon * momentum_scale
        ) & (jnp.abs(total_energy_defect) <= 1024.0 * epsilon * energy_scale)
        smoothing_within_bounds = ~jnp.any(active & adaptive.bound_active)
        finite = (
            state_finite
            & time_valid
            & jnp.all(jnp.isfinite(h) | ~active)
            & jnp.all(jnp.isfinite(density_comoving) | ~active)
            & jnp.all(jnp.isfinite(kernel_comoving))
            & jnp.all(jnp.isfinite(relative_speed))
            & jnp.all(jnp.isfinite(kernel_total))
            & jnp.all(jnp.isfinite(sampled_cosine))
            & jnp.all(jnp.isfinite(sampled_azimuth))
            & jnp.all(scattering.finite)
            & jnp.all(jnp.isfinite(candidate_momenta) | ~active_column)
            & jnp.all(jnp.isfinite(total_momentum_defect))
            & jnp.isfinite(total_energy_defect)
        )
        successful = (
            neighborhood.successful
            & adaptive.converged
            & smoothing_within_bounds
            & equal_mass
            & kernel_domain_valid
            & angular_sampling_valid
            & probability_valid
            & aggregate_probability_valid
            & capacity_valid
            & knudsen_valid
            & endpoint_disjoint
            & inactive_preserved
            & conservative
            & finite
            & jnp.all(scattering.successful)
        )
        accepted_pairs = selected & successful
        accepted_state = CosmologicalParticleState(
            jnp.where(successful, candidate_state.positions, state.positions),
            jnp.where(
                successful,
                candidate_state.canonical_momenta,
                state.canonical_momenta,
            ),
            jnp.where(successful, candidate_state.scale_factor, state.scale_factor),
        )
        diagnostics = SIDMCollisionDiagnostics(
            dt,
            h,
            density_comoving,
            density_physical,
            kernel_comoving,
            kernel_physical,
            relative_speed,
            probability,
            kernel_total,
            safe_cross_section_per_mass,
            kernel_supported,
            sampled_cosine,
            sampled_azimuth,
            angular_sample_successful,
            particle_probability,
            uniforms,
            proposed,
            selected,
            accepted_pairs,
            event_count,
            mean_free_path,
            support_radius,
            knudsen,
            scattering.momentum_defect,
            scattering.kinetic_energy_defect,
            total_momentum_defect,
            total_energy_defect,
            neighborhood.successful,
            adaptive.converged,
            smoothing_within_bounds,
            equal_mass,
            probability_valid,
            aggregate_probability_valid,
            capacity_valid,
            knudsen_valid,
            endpoint_disjoint,
            inactive_preserved,
            conservative,
            finite,
            successful,
        )
        return SIDMCollisionResult(
            candidate_state, accepted_state, pairs, diagnostics, successful
        )

    def rollout(
        self,
        background: FLRWBackground,
        state: CosmologicalParticleState,
        key: Array,
        args: Any = None,
        /,
    ) -> CosmologicalSIDMResult:
        """Apply collision/PM/collision Strang splitting over the PM schedule."""

        if not isinstance(background, FLRWBackground):
            raise TypeError("background must be FLRWBackground.")
        if not isinstance(state, CosmologicalParticleState):
            raise TypeError("state must be CosmologicalParticleState.")
        if background.scale.scale_id != self.particle_mesh.kinematics.scale.scale_id:
            raise ValueError("Background and cosmological SIDM scales disagree.")
        initial_scale = self.particle_mesh.scale_factors[0].astype(
            state.scale_factor.dtype
        )
        state_scale = eqx.error_if(
            state.scale_factor,
            jnp.abs(state.scale_factor - initial_scale) > 1.0e-12,
            "Cosmological SIDM state must start at the first scheduled scale factor.",
        )
        state_scale = background.require_flat(state_scale)
        state = CosmologicalParticleState(
            state.positions, state.canonical_momenta, state_scale
        )
        initial_force = self.particle_mesh.gravity.acceleration(state.positions, args)
        initial_mass_defect = (
            initial_force.deposited.balance.maximum_absolute_balance_defect
        )
        running = initial_force.successful
        accepted_count = jnp.asarray(0, dtype=jnp.int32)
        indices = jnp.arange(
            self.particle_mesh.scale_factors.shape[0] - 1, dtype=jnp.int32
        )
        end_scales = self.particle_mesh.scale_factors[1:].astype(state.scale_factor.dtype)

        def step(carry, schedule):
            (
                current,
                acceleration_start,
                previous_mass_defect,
                previous_net_force,
                active_rollout,
                count,
            ) = carry
            interval_index, end_scale = schedule
            physical_dt = self.physical_time_between(
                background, current.scale_factor, end_scale
            )
            first = self.collide(
                current,
                key,
                2 * interval_index,
                0.5 * physical_dt,
            )
            pm = _advance_particle_mesh_interval(
                self.particle_mesh.kinematics,
                self.particle_mesh.gravity,
                background,
                first.accepted_state,
                end_scale,
                acceleration_start,
                args,
            )
            second = self.collide(
                pm.state,
                key,
                2 * interval_index + 1,
                0.5 * physical_dt,
            )
            successful = (
                active_rollout & first.successful & pm.successful & second.successful
            )
            accepted_state = CosmologicalParticleState(
                jnp.where(
                    successful,
                    second.accepted_state.positions,
                    current.positions,
                ),
                jnp.where(
                    successful,
                    second.accepted_state.canonical_momenta,
                    current.canonical_momenta,
                ),
                jnp.where(
                    successful,
                    second.accepted_state.scale_factor,
                    current.scale_factor,
                ),
            )
            next_acceleration = jnp.where(successful, pm.acceleration, acceleration_start)
            next_mass_defect = jnp.where(
                successful, pm.mass_balance_defect, previous_mass_defect
            )
            next_net_force = jnp.where(successful, pm.net_force, previous_net_force)
            mass_defect = jnp.maximum(previous_mass_defect, pm.mass_balance_defect)
            pm_record = (
                pm.drift_factor,
                pm.first_kick_factor,
                pm.second_kick_factor,
                mass_defect,
                pm.net_force,
                pm.force_successful,
            )
            next_carry = (
                accepted_state,
                next_acceleration,
                next_mass_defect,
                next_net_force,
                successful,
                count + successful.astype(jnp.int32),
            )
            return next_carry, (
                pm_record,
                first.diagnostics,
                second.diagnostics,
                successful,
            )

        initial_carry = (
            state,
            initial_force.acceleration,
            initial_mass_defect,
            initial_force.net_force,
            running,
            accepted_count,
        )
        final_carry, recorded = jax.lax.scan(step, initial_carry, (indices, end_scales))
        final_state, _, _, _, completed, accepted_steps = final_carry
        pm_record, first_half, second_half, accepted = recorded
        (
            drift,
            first_kick,
            second_kick,
            mass_defect,
            net_force,
            force_successful,
        ) = pm_record
        failed = ~accepted
        first_failed = jnp.where(
            jnp.any(failed),
            jnp.argmax(failed).astype(jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
        )
        pm_diagnostics = CosmologicalParticleMeshDiagnostics(
            drift,
            first_kick,
            second_kick,
            mass_defect,
            net_force,
            force_successful,
            accepted,
            initial_force.successful,
            completed,
            accepted_steps,
            jnp.max(mass_defect),
            jnp.max(jnp.sqrt(jnp.sum(net_force**2, axis=-1))),
            first_failed,
        )
        diagnostics = CosmologicalSIDMDiagnostics(
            pm_diagnostics,
            first_half,
            second_half,
            accepted,
            completed,
            accepted_steps,
            first_failed,
        )
        return CosmologicalSIDMResult(final_state, diagnostics, completed)


__all__ = [
    "CosmologicalSIDMDiagnostics",
    "CosmologicalSIDMPlan",
    "CosmologicalSIDMResult",
    "SIDMCollisionDiagnostics",
    "SIDMCollisionPolicy",
    "SIDMCollisionResult",
    "SIDMCrossSectionPlan",
]
