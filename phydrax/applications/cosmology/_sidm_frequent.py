#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

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
    particle_pair_geometry,
    ParticlePairRelation,
)
from ._sidm import _pair_keys
from ._sidm_kernels import directions_from_angles, SmallAngleSplitPlan
from ._sidm_weighted import (
    _finite_state,
    _mass_relation,
    _shape_check,
    _state_where,
    WeightedSIDMPacketState,
)


class FrequentSmallAngleSIDMDiagnostics(StrictModule):
    physical_time_step: Array
    smoothing_length_comoving: Array
    kernel_weight_physical: Array
    relative_speed_physical: Array
    encounter_factor: Array
    small_transfer_cross_section: Array
    small_viscosity_cross_section: Array
    target_drag_fraction: Array
    particle_aggregate_drag_fraction: Array
    particle_aggregate_transverse_variance: Array
    achieved_transverse_variance: Array
    target_transverse_variance: Array
    transverse_variance_defect: Array
    pair_diffusion_covariance: Array
    random_azimuth: Array
    matching_priority: Array
    supported_pairs: Array
    selected_pair_fraction: Array
    selected_pairs: Array
    accepted_pairs: Array
    pair_canonical_impulse: Array
    pair_momentum_defect: Array
    pair_kinetic_energy_defect: Array
    total_momentum_defect: Array
    total_kinetic_energy_defect: Array
    maximum_drag_fraction: Array
    maximum_transverse_variance: Array
    split_reconstruction_error: Array
    neighborhood_successful: Array
    kernel_supported: Array
    diffusion_psd: Array
    timestep_valid: Array
    all_pairs_covered: Array
    support_identity_valid: Array
    equal_pair_weights: Array
    mass_relation_valid: Array
    conservative: Array
    finite: Array
    successful: Array


class FrequentSmallAngleSIDMResult(StrictModule):
    candidate_state: WeightedSIDMPacketState
    accepted_state: WeightedSIDMPacketState
    pairs: ParticlePairRelation
    diagnostics: FrequentSmallAngleSIDMDiagnostics
    successful: Array


def _transverse_direction(relative: Array, azimuth: Array, /) -> tuple[Array, Array]:
    speed = jnp.sqrt(ein.contract("ni,ni->n", relative, relative))
    direction = relative / jnp.where(speed > 0.0, speed, 1.0)[:, None]
    transverse = directions_from_angles(
        relative, jnp.zeros(azimuth.shape, dtype=relative.dtype), azimuth
    )
    return direction, transverse


class FrequentSmallAngleSIDMPlan(StrictModule, NonTrainableState):
    """Pair-owned frequent small-angle SIDM on weighted packets.

    The plan consumes only the small-angle portion authored by
    :class:`SmallAngleSplitPlan`. It applies no rare/frequent regime switch.
    Every supported edge is applied once in an epoch-keyed sequential schedule;
    endpoint aggregate drag and diffusion bounds gate the physical timestep.
    This narrow closure requires equal endpoint packet weights and an
    axisymmetric kernel. Every pair receives an antisymmetric drag impulse and
    paired transverse random impulse. The finite step remains exactly elastic;
    its first and transverse second Kramers--Moyal moments and finite-step
    defect are retained as evidence.
    """

    neighborhood: AbstractPreparedParticleNeighborhood
    spatial_kernel: AbstractSPHSmoothingKernel
    split: SmallAngleSplitPlan
    smoothing_length_comoving: float = eqx.field(static=True)
    maximum_drag_fraction_per_step: float = eqx.field(static=True)
    maximum_transverse_variance_per_step: float = eqx.field(static=True)
    moment_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        neighborhood: AbstractPreparedParticleNeighborhood,
        spatial_kernel: AbstractSPHSmoothingKernel,
        split: SmallAngleSplitPlan,
        /,
        *,
        smoothing_length_comoving: float,
        maximum_drag_fraction_per_step: float = 0.1,
        maximum_transverse_variance_per_step: float = 0.2,
        moment_tolerance: float = 0.05,
    ):
        if not isinstance(neighborhood, AbstractPreparedParticleNeighborhood):
            raise TypeError("neighborhood must be a prepared particle neighborhood.")
        if not isinstance(spatial_kernel, AbstractSPHSmoothingKernel):
            raise TypeError("spatial_kernel must be an AbstractSPHSmoothingKernel.")
        if not isinstance(split, SmallAngleSplitPlan):
            raise TypeError("split must be SmallAngleSplitPlan.")
        if spatial_kernel.dimension != 3:
            raise ValueError("Frequent small-angle SIDM requires three dimensions.")
        first_species = split.kernel.first_species
        second_species = split.kernel.second_species
        if first_species.species_plan_id != second_species.species_plan_id:
            raise ValueError(
                "Frequent small-angle SIDM supports one elastic species only."
            )
        if split.kernel.azimuths is not None:
            raise ValueError(
                "Frequent small-angle SIDM requires an axisymmetric kernel; "
                "azimuth-dependent drift and diffusion tensors are unsupported."
            )
        smoothing = float(smoothing_length_comoving)
        maximum_drag = float(maximum_drag_fraction_per_step)
        maximum_variance = float(maximum_transverse_variance_per_step)
        tolerance = float(moment_tolerance)
        if not np.isfinite(smoothing) or smoothing <= 0.0:
            raise ValueError("smoothing_length_comoving must be finite and positive.")
        if not np.isfinite(maximum_drag) or not 0.0 < maximum_drag < 1.0:
            raise ValueError("maximum_drag_fraction_per_step must lie in (0,1).")
        if not np.isfinite(maximum_variance) or not 0.0 < maximum_variance < 1.0:
            raise ValueError("maximum_transverse_variance_per_step must lie in (0,1).")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("moment_tolerance must be finite and nonnegative.")
        self.neighborhood = neighborhood
        self.spatial_kernel = spatial_kernel
        self.split = split
        self.smoothing_length_comoving = smoothing
        self.maximum_drag_fraction_per_step = maximum_drag
        self.maximum_transverse_variance_per_step = maximum_variance
        self.moment_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "frequent-small-angle-sidm",
                "neighborhood": neighborhood.prepared_id,
                "spatial_kernel": spatial_kernel.kernel_id,
                "split": split.split_id,
                "smoothing_length_comoving": smoothing,
                "maximum_drag_fraction_per_step": maximum_drag,
                "maximum_transverse_variance_per_step": maximum_variance,
                "moment_tolerance": tolerance,
                "closure": "pair-owned-elastic-transverse-diffusion",
            }
        )

    def apply(
        self,
        state: WeightedSIDMPacketState,
        key: Array,
        epoch: ArrayLike,
        physical_time_step: ArrayLike,
        /,
    ) -> FrequentSmallAngleSIDMResult:
        capacity, dimension = state.positions.shape
        _shape_check(state, capacity, dimension)
        if dimension != 3:
            raise ValueError("Frequent small-angle SIDM state must be three-dimensional.")
        dtype = state.positions.dtype
        active = state.active_mask
        safe_scale = jnp.where(
            jnp.isfinite(state.scale_factor) & (state.scale_factor > 0.0),
            state.scale_factor,
            1.0,
        )
        safe_mass = jnp.where(active, state.gravitational_masses, 1.0)
        safe_weight = jnp.where(active, state.weights, 0.0)
        safe_momentum = jnp.where(
            jnp.isfinite(state.canonical_momenta), state.canonical_momenta, 0.0
        )
        safe_positions = jnp.where(jnp.isfinite(state.positions), state.positions, 0.0)
        velocity = safe_momentum / (safe_mass[:, None] * safe_scale)
        dt = jnp.asarray(physical_time_step, dtype=dtype).reshape(())
        time_valid = jnp.isfinite(dt) & (dt >= 0.0)
        safe_dt = jnp.where(time_valid, dt, 0.0)
        neighborhood = self.neighborhood.build(safe_positions, active_mask=active)
        pairs = neighborhood.pair_relation
        if (
            pairs.relation.source_size != capacity
            or pairs.relation.target_size != capacity
        ):
            raise ValueError("Frequent SIDM relation must align with packet capacity.")
        geometry = particle_pair_geometry(safe_positions, pairs, box=neighborhood.box)
        left = pairs.left_indices
        right = pairs.right_indices
        smoothing = jnp.asarray(self.smoothing_length_comoving, dtype=dtype)
        support = (
            pairs.valid
            & active[left]
            & active[right]
            & (geometry.distance < self.spatial_kernel.support_factor * smoothing)
        )
        kernel_comoving = jnp.where(
            support, self.spatial_kernel.value(geometry.distance, smoothing), 0.0
        )
        kernel_physical = kernel_comoving / safe_scale**3
        relative = velocity[left] - velocity[right]
        speed = jnp.sqrt(ein.contract("ni,ni->n", relative, relative))
        moments = self.split.moments(speed)
        encounter = (
            0.5
            * (safe_weight[left] + safe_weight[right])
            * speed
            * safe_dt
            * kernel_physical
        )
        encounter = jnp.where(support & moments.small.supported, encounter, 0.0)
        drag = encounter * moments.small.transfer.astype(dtype)
        target_transverse = encounter * moments.small.viscosity.astype(dtype)
        achieved_transverse = jnp.maximum(2.0 * drag - drag**2, 0.0)
        variance_defect = achieved_transverse - target_transverse
        pair_keys = _pair_keys(key, pairs, epoch)
        matching_priority = jax.vmap(
            lambda local: jr.uniform(jr.fold_in(local, 30), (), dtype=dtype)
        )(pair_keys)
        eligible = support & moments.small.supported & (speed > 0.0)
        selected = eligible
        all_pairs_covered = jnp.all(~eligible | selected)
        supported_count = jnp.sum(eligible, dtype=jnp.int32)
        selected_fraction = jnp.sum(selected, dtype=dtype) / jnp.maximum(
            supported_count.astype(dtype), 1.0
        )
        azimuth = jax.vmap(
            lambda local: jr.uniform(
                jr.fold_in(local, 31),
                (),
                minval=0.0,
                maxval=2.0 * jnp.pi,
                dtype=dtype,
            )
        )(pair_keys)
        order = jnp.argsort(matching_priority)
        zero_vector = jnp.zeros(relative.shape, dtype=dtype)
        zero_scalar = jnp.zeros(speed.shape, dtype=dtype)

        def apply_pair(index, carry):
            momentum_values, impulse_values, momentum_defects, energy_defects = carry
            route = order[index]
            left_index = left[route]
            right_index = right[route]
            left_velocity = momentum_values[left_index] / (
                safe_mass[left_index] * safe_scale
            )
            right_velocity = momentum_values[right_index] / (
                safe_mass[right_index] * safe_scale
            )
            relative_velocity = left_velocity - right_velocity
            current_speed = jnp.sqrt(
                ein.contract("i,i->", relative_velocity, relative_velocity)
            )
            current_direction = relative_velocity / jnp.where(
                current_speed > 0.0, current_speed, 1.0
            )
            transverse_direction = directions_from_angles(
                relative_velocity,
                jnp.asarray(0.0, dtype=dtype),
                azimuth[route],
            )
            outgoing_relative = current_speed * (
                (1.0 - drag[route]) * current_direction
                + jnp.sqrt(achieved_transverse[route]) * transverse_direction
            )
            pair_mass = safe_mass[left_index] + safe_mass[right_index]
            center_velocity = (
                safe_mass[left_index] * left_velocity
                + safe_mass[right_index] * right_velocity
            ) / pair_mass
            next_left_velocity = (
                center_velocity + safe_mass[right_index] * outgoing_relative / pair_mass
            )
            next_right_velocity = (
                center_velocity - safe_mass[left_index] * outgoing_relative / pair_mass
            )
            take = selected[route]
            left_delta = jnp.where(
                take,
                safe_mass[left_index] * safe_scale * (next_left_velocity - left_velocity),
                0.0,
            )
            right_delta = jnp.where(
                take,
                safe_mass[right_index]
                * safe_scale
                * (next_right_velocity - right_velocity),
                0.0,
            )
            before_energy = 0.5 * (
                safe_mass[left_index]
                * ein.contract("i,i->", left_velocity, left_velocity)
                + safe_mass[right_index]
                * ein.contract("i,i->", right_velocity, right_velocity)
            )
            after_energy = 0.5 * (
                safe_mass[left_index]
                * ein.contract("i,i->", next_left_velocity, next_left_velocity)
                + safe_mass[right_index]
                * ein.contract("i,i->", next_right_velocity, next_right_velocity)
            )
            momentum_values = momentum_values.at[left_index].add(left_delta)
            momentum_values = momentum_values.at[right_index].add(right_delta)
            impulse_values = impulse_values.at[route].set(left_delta)
            momentum_defects = momentum_defects.at[route].set(left_delta + right_delta)
            energy_defects = energy_defects.at[route].set(
                jnp.where(take, after_energy - before_energy, 0.0)
            )
            return (
                momentum_values,
                impulse_values,
                momentum_defects,
                energy_defects,
            )

        (
            candidate_momentum,
            first_delta,
            pair_momentum_defect,
            pair_energy_defect,
        ) = jax.lax.fori_loop(
            0,
            pairs.capacity,
            apply_pair,
            (safe_momentum, zero_vector, zero_vector, zero_scalar),
        )
        candidate = WeightedSIDMPacketState(
            state.positions,
            state.microscopic_masses,
            state.weights,
            state.gravitational_masses,
            jnp.where(active[:, None], candidate_momentum, state.canonical_momenta),
            state.active_mask,
            state.packet_ids,
            state.parent_packet_ids,
            state.lineage_depth,
            state.scale_factor,
        )
        total_momentum_defect = jnp.sum(pair_momentum_defect, axis=0)
        total_energy_defect = jnp.sum(pair_energy_defect)
        before_pair_energy = 0.5 * (
            safe_mass[left] * ein.contract("ni,ni->n", velocity[left], velocity[left])
            + safe_mass[right]
            * ein.contract("ni,ni->n", velocity[right], velocity[right])
        )
        direction, _ = _transverse_direction(relative, azimuth)
        identity = jnp.eye(3, dtype=dtype)
        projector = identity - ein.contract("ni,nj->nij", direction, direction)
        diffusion_covariance = (
            0.5
            * achieved_transverse[:, None, None]
            * speed[:, None, None] ** 2
            * projector
        )
        projector_defect = jnp.max(
            jnp.abs(ein.contract("nij,njk->nik", projector, projector) - projector),
            axis=(-2, -1),
        )
        diffusion_psd = jnp.all(
            ~selected
            | (
                jnp.isfinite(achieved_transverse)
                & (achieved_transverse >= 0.0)
                & (projector_defect <= 256.0 * jnp.finfo(dtype).eps)
            )
        )
        aggregate_drag = jnp.zeros((capacity,), dtype=dtype)
        aggregate_drag = aggregate_drag.at[left].add(jnp.where(selected, drag, 0.0))
        aggregate_drag = aggregate_drag.at[right].add(jnp.where(selected, drag, 0.0))
        aggregate_variance = jnp.zeros((capacity,), dtype=dtype)
        aggregate_variance = aggregate_variance.at[left].add(
            jnp.where(selected, achieved_transverse, 0.0)
        )
        aggregate_variance = aggregate_variance.at[right].add(
            jnp.where(selected, achieved_transverse, 0.0)
        )
        maximum_drag = jnp.max(jnp.where(selected, drag, 0.0), initial=0.0)
        maximum_variance = jnp.max(
            jnp.where(selected, achieved_transverse, 0.0), initial=0.0
        )
        timestep_valid = (
            time_valid
            & jnp.all(
                ~selected
                | (
                    jnp.isfinite(drag)
                    & (drag >= 0.0)
                    & (drag <= self.maximum_drag_fraction_per_step)
                    & jnp.isfinite(achieved_transverse)
                    & (achieved_transverse <= self.maximum_transverse_variance_per_step)
                )
            )
            & jnp.all(
                ~active
                | (
                    (aggregate_drag <= self.maximum_drag_fraction_per_step)
                    & (aggregate_variance <= self.maximum_transverse_variance_per_step)
                )
            )
        )
        moment_scale = jnp.maximum(
            jnp.maximum(jnp.abs(target_transverse), jnp.abs(achieved_transverse)),
            jnp.finfo(dtype).tiny,
        )
        finite_step_allowance = drag**2 + self.moment_tolerance * moment_scale
        moment_valid = jnp.all(
            ~selected | (jnp.abs(variance_defect) <= finite_step_allowance)
        )
        packet_momentum_norm = jnp.sqrt(
            ein.contract("ni,ni->n", safe_momentum, safe_momentum)
        )
        momentum_scale = jnp.maximum(
            jnp.sum(jnp.where(active, packet_momentum_norm, 0.0)),
            jnp.finfo(dtype).tiny,
        )
        energy_scale = jnp.maximum(
            jnp.sum(jnp.abs(before_pair_energy)), jnp.finfo(dtype).tiny
        )
        epsilon = jnp.finfo(dtype).eps
        conservative = (
            jnp.sqrt(ein.contract("i,i->", total_momentum_defect, total_momentum_defect))
            <= 2048.0 * epsilon * momentum_scale
        ) & (jnp.abs(total_energy_defect) <= 4096.0 * epsilon * energy_scale)
        microscopic_mass = jnp.asarray(self.split.kernel.first_species.mass, dtype=dtype)
        species_scale = jnp.maximum(
            jnp.maximum(jnp.abs(state.microscopic_masses), jnp.abs(microscopic_mass)),
            jnp.finfo(dtype).tiny,
        )
        species_valid = jnp.all(
            ~active
            | (
                jnp.abs(state.microscopic_masses - microscopic_mass)
                <= 64.0 * epsilon * species_scale
            )
        )
        mass_valid = _mass_relation(state) & _mass_relation(candidate) & species_valid
        identity_valid = jnp.all(
            ~pairs.valid
            | (
                (state.packet_ids[left] == pairs.left_particle_ids)
                & (state.packet_ids[right] == pairs.right_particle_ids)
            )
        )
        pair_weight_scale = jnp.maximum(
            jnp.maximum(jnp.abs(safe_weight[left]), jnp.abs(safe_weight[right])),
            jnp.finfo(dtype).tiny,
        )
        equal_pair_weights = jnp.all(
            ~support
            | (
                jnp.abs(safe_weight[left] - safe_weight[right])
                <= 64.0 * epsilon * pair_weight_scale
            )
        )
        kernel_supported = jnp.all(~support | moments.supported) & jnp.all(
            ~support | moments.successful
        )
        finite = (
            _finite_state(state)
            & _finite_state(candidate)
            & jnp.all(jnp.isfinite(encounter))
            & jnp.all(jnp.isfinite(drag))
            & jnp.all(jnp.isfinite(target_transverse))
            & jnp.all(jnp.isfinite(variance_defect))
            & jnp.all(jnp.isfinite(pair_momentum_defect))
            & jnp.all(jnp.isfinite(pair_energy_defect))
            & jnp.all(jnp.isfinite(diffusion_covariance))
        )
        successful = (
            neighborhood.successful
            & kernel_supported
            & diffusion_psd
            & timestep_valid
            & moment_valid
            & all_pairs_covered
            & mass_valid
            & identity_valid
            & equal_pair_weights
            & conservative
            & finite
        )
        accepted = _state_where(successful, candidate, state)
        diagnostics = FrequentSmallAngleSIDMDiagnostics(
            dt,
            smoothing,
            kernel_physical,
            speed,
            encounter,
            moments.small.transfer,
            moments.small.viscosity,
            drag,
            aggregate_drag,
            aggregate_variance,
            achieved_transverse,
            target_transverse,
            variance_defect,
            diffusion_covariance,
            azimuth,
            matching_priority,
            support,
            selected_fraction,
            selected,
            selected & successful,
            first_delta,
            pair_momentum_defect,
            pair_energy_defect,
            total_momentum_defect,
            total_energy_defect,
            maximum_drag,
            maximum_variance,
            moments.reconstruction_error,
            neighborhood.successful,
            kernel_supported,
            diffusion_psd,
            timestep_valid & moment_valid,
            all_pairs_covered,
            identity_valid,
            equal_pair_weights,
            mass_valid,
            conservative,
            finite,
            successful,
        )
        return FrequentSmallAngleSIDMResult(
            candidate, accepted, pairs, diagnostics, successful
        )


__all__ = [
    "FrequentSmallAngleSIDMDiagnostics",
    "FrequentSmallAngleSIDMPlan",
    "FrequentSmallAngleSIDMResult",
]
