#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein
from phydrax.linalg import HermitianSpectrum

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle import (
    AbstractPreparedParticleNeighborhood,
    AbstractSPHSmoothingKernel,
    particle_pair_geometry,
    ParticleBox,
    ParticlePairRelation,
)
from ...discretization.splatting import ParticleGridSplatState, SplatDepositResult
from ...solver import KDKCoefficients, KDKTransactionPlan
from ._background import FLRWBackground
from ._distances import FLRWDistancePlan
from ._particle_mesh import CosmologicalParticleMeshPlan
from ._sidm import _pair_keys, _select_endpoint_disjoint, SIDMCollisionPolicy
from ._sidm_kernels import (
    directions_from_angles,
    SmallAngleSplitPlan,
    TwoBodyDifferentialKernelPlan,
)


class WeightedSIDMPacketState(StrictModule):
    """Fixed-capacity weighted packets at one cosmological time level.

    ``positions`` are comoving. ``microscopic_masses`` are physical masses per
    represented quantum, ``weights`` are dimensionless multiplicities, and
    ``gravitational_masses`` are physical macro masses constrained to
    ``microscopic_masses * weights``. The canonical macro momentum is
    ``p = M a² dx/dt = M a u_pec``. Packet IDs are the immutable stable IDs of
    the prepared particle support; child creation changes only lineage leaves.
    """

    positions: Array
    microscopic_masses: Array
    weights: Array
    gravitational_masses: Array
    canonical_momenta: Array
    active_mask: Array
    packet_ids: Array
    parent_packet_ids: Array
    lineage_depth: Array
    scale_factor: Array

    def __init__(
        self,
        positions: ArrayLike,
        microscopic_masses: ArrayLike,
        weights: ArrayLike,
        gravitational_masses: ArrayLike,
        canonical_momenta: ArrayLike,
        active_mask: ArrayLike,
        packet_ids: ArrayLike,
        parent_packet_ids: ArrayLike,
        lineage_depth: ArrayLike,
        scale_factor: ArrayLike,
        /,
    ):
        position = jnp.asarray(positions)
        self.positions = position
        self.microscopic_masses = jnp.asarray(microscopic_masses, dtype=position.dtype)
        self.weights = jnp.asarray(weights, dtype=position.dtype)
        self.gravitational_masses = jnp.asarray(
            gravitational_masses, dtype=position.dtype
        )
        self.canonical_momenta = jnp.asarray(canonical_momenta, dtype=position.dtype)
        self.active_mask = jnp.asarray(active_mask, dtype=bool)
        self.packet_ids = jnp.asarray(packet_ids, dtype=jnp.int64)
        self.parent_packet_ids = jnp.asarray(parent_packet_ids, dtype=jnp.int64)
        self.lineage_depth = jnp.asarray(lineage_depth, dtype=jnp.int32)
        self.scale_factor = jnp.asarray(scale_factor, dtype=position.dtype).reshape(())


class WeightedPacketLedger(StrictModule):
    mass: Array
    canonical_momentum: Array
    kinetic_energy: Array


class WeightedPMForceResult(StrictModule):
    acceleration: Array
    potential: Array
    deposited: SplatDepositResult
    routes: ParticleGridSplatState
    net_force: Array
    converged: Array
    support_complete: Array
    mass_relation_valid: Array
    support_identity_valid: Array
    successful: Array


class WeightedPMIntervalDiagnostics(StrictModule):
    drift_factor: Array
    first_kick_factor: Array
    second_kick_factor: Array
    mass_balance_defect: Array
    net_force: Array
    successful: Array


class WeightedPMIntervalResult(StrictModule):
    candidate_state: WeightedSIDMPacketState
    accepted_state: WeightedSIDMPacketState
    acceleration: Array
    diagnostics: WeightedPMIntervalDiagnostics
    successful: Array


class WeightedSIDMCollisionDiagnostics(StrictModule):
    physical_time_step: Array
    smoothing_length_comoving: Array
    kernel_weight_physical: Array
    relative_speed_physical: Array
    total_cross_section: Array
    pair_probability: Array
    particle_aggregate_probability: Array
    exchanged_weight: Array
    proposed_pairs: Array
    selected_pairs: Array
    accepted_pairs: Array
    child_required: Array
    child_slots_used: Array
    event_count: Array
    number_density_physical: Array
    mean_free_path_physical: Array
    support_radius_physical: Array
    knudsen_number: Array
    mass_defect: Array
    momentum_defect: Array
    kinetic_energy_defect: Array
    neighborhood_successful: Array
    kernel_supported: Array
    angular_split_successful: Array
    angular_sampling_residual: Array
    angular_sampling_valid: Array
    probability_valid: Array
    aggregate_probability_valid: Array
    capacity_valid: Array
    knudsen_valid: Array
    endpoint_disjoint: Array
    mass_relation_valid: Array
    lineage_valid: Array
    conservative: Array
    finite: Array
    successful: Array


class WeightedSIDMCollisionResult(StrictModule):
    candidate_state: WeightedSIDMPacketState
    accepted_state: WeightedSIDMPacketState
    pairs: ParticlePairRelation
    diagnostics: WeightedSIDMCollisionDiagnostics
    successful: Array
    profile_id: str = eqx.field(static=True)


class WeightedSIDMRolloutDiagnostics(StrictModule):
    first_half_collisions: WeightedSIDMCollisionDiagnostics
    particle_mesh: WeightedPMIntervalDiagnostics
    second_half_collisions: WeightedSIDMCollisionDiagnostics
    accepted: Array
    completed: Array
    accepted_steps: Array
    first_failed_step: Array


class WeightedSIDMRolloutResult(StrictModule):
    state: WeightedSIDMPacketState
    diagnostics: WeightedSIDMRolloutDiagnostics
    successful: Array
    profile_id: str = eqx.field(static=True)


class WeightedPacketResamplingDiagnostics(StrictModule):
    active_before: Array
    active_after: Array
    mass_defect: Array
    centroid_defect: Array
    momentum_defect: Array
    kinetic_energy_defect: Array
    velocity_covariance_defect: Array
    position_covariance_loss: Array
    velocity_third_moment_loss: Array
    accepted_boundary: Array
    common_microscopic_mass: Array
    centroid_defined: Array
    identity_valid: Array
    lineage_valid: Array
    capacity_valid: Array
    nonnegative: Array
    bounded: Array
    moment_closed: Array
    finite: Array
    successful: Array


class WeightedPacketResamplingResult(StrictModule):
    candidate_state: WeightedSIDMPacketState
    accepted_state: WeightedSIDMPacketState
    diagnostics: WeightedPacketResamplingDiagnostics
    successful: Array


def _shape_check(state: WeightedSIDMPacketState, capacity: int, dimension: int) -> None:
    vector = (capacity, dimension)
    scalar = (capacity,)
    if state.positions.shape != vector or state.canonical_momenta.shape != vector:
        raise ValueError(f"Weighted packet vectors must have shape {vector}.")
    scalar_arrays = (
        state.microscopic_masses,
        state.weights,
        state.gravitational_masses,
        state.active_mask,
        state.packet_ids,
        state.parent_packet_ids,
        state.lineage_depth,
    )
    if any(value.shape != scalar for value in scalar_arrays):
        raise ValueError(f"Weighted packet scalar fields must have shape {scalar}.")
    if state.scale_factor.shape != ():
        raise ValueError("Weighted packet scale_factor must be scalar.")


def _state_where(
    accepted: Array, candidate: WeightedSIDMPacketState, original: WeightedSIDMPacketState
) -> WeightedSIDMPacketState:
    def choose(new, old):
        condition = accepted.reshape((1,) * new.ndim) if new.ndim else accepted
        return jnp.where(condition, new, old)

    return jax.tree.map(choose, candidate, original)


def _mass_relation(state: WeightedSIDMPacketState) -> Array:
    active = state.active_mask
    expected = state.microscopic_masses * state.weights
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(expected), jnp.abs(state.gravitational_masses)),
        jnp.finfo(state.positions.dtype).tiny,
    )
    tolerance = 64.0 * jnp.finfo(state.positions.dtype).eps * scale
    return jnp.all(
        ~active
        | (
            jnp.isfinite(state.microscopic_masses)
            & (state.microscopic_masses > 0.0)
            & jnp.isfinite(state.weights)
            & (state.weights > 0.0)
            & jnp.isfinite(state.gravitational_masses)
            & (state.gravitational_masses > 0.0)
            & (jnp.abs(state.gravitational_masses - expected) <= tolerance)
        )
    )


def _finite_state(state: WeightedSIDMPacketState) -> Array:
    active = state.active_mask
    active_column = active[:, None]
    return (
        jnp.isfinite(state.scale_factor)
        & (state.scale_factor > 0.0)
        & jnp.all(jnp.isfinite(state.positions) | ~active_column)
        & jnp.all(jnp.isfinite(state.canonical_momenta) | ~active_column)
        & jnp.all(jnp.isfinite(state.microscopic_masses) | ~active)
        & jnp.all(jnp.isfinite(state.weights) | ~active)
        & jnp.all(jnp.isfinite(state.gravitational_masses) | ~active)
    )


def _ledger(state: WeightedSIDMPacketState) -> WeightedPacketLedger:
    active = state.active_mask
    mass = jnp.where(active, state.gravitational_masses, 0.0)
    safe_mass = jnp.where(active, state.gravitational_masses, 1.0)
    momentum = jnp.where(active[:, None], state.canonical_momenta, 0.0)
    peculiar = momentum / (safe_mass[:, None] * state.scale_factor)
    kinetic = 0.5 * jnp.sum(mass * ein.contract("ni,ni->n", peculiar, peculiar))
    return WeightedPacketLedger(jnp.sum(mass), jnp.sum(momentum, axis=0), kinetic)


def _scattered_velocities(
    first_velocity: Array,
    second_velocity: Array,
    first_mass: Array,
    second_mass: Array,
    cosine: Array,
    azimuth: Array,
    /,
) -> tuple[Array, Array]:
    relative = first_velocity - second_velocity
    speed = jnp.sqrt(ein.contract("ni,ni->n", relative, relative))
    outgoing = speed[:, None] * directions_from_angles(relative, cosine, azimuth)
    total_mass = first_mass + second_mass
    center = (
        first_mass[:, None] * first_velocity + second_mass[:, None] * second_velocity
    ) / total_mass[:, None]
    first = center + second_mass[:, None] * outgoing / total_mass[:, None]
    second = center - first_mass[:, None] * outgoing / total_mass[:, None]
    return first, second


class WeightedSIDMPlan(StrictModule, NonTrainableState):
    """Fixed-capacity weighted rare SIDM with runtime-mass PM coupling.

    This is an explicitly rare, packet-splitting profile. It never switches to
    the frequent or gravothermal closures. Topology changes and sampled events
    are nondifferentiable; continuous PM and collision algebra remains JAX
    compatible for a fixed accepted topology.
    """

    particle_mesh: CosmologicalParticleMeshPlan
    neighborhood: AbstractPreparedParticleNeighborhood
    spatial_kernel: AbstractSPHSmoothingKernel
    differential_kernel: TwoBodyDifferentialKernelPlan
    angular_split: SmallAngleSplitPlan | None
    policy: SIDMCollisionPolicy
    time: FLRWDistancePlan
    smoothing_length_comoving: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_mesh: CosmologicalParticleMeshPlan,
        neighborhood: AbstractPreparedParticleNeighborhood,
        spatial_kernel: AbstractSPHSmoothingKernel,
        differential_kernel: TwoBodyDifferentialKernelPlan,
        policy: SIDMCollisionPolicy,
        /,
        *,
        smoothing_length_comoving: float,
        angular_split: SmallAngleSplitPlan | None = None,
        time: FLRWDistancePlan | None = None,
    ):
        if not isinstance(particle_mesh, CosmologicalParticleMeshPlan):
            raise TypeError("particle_mesh must be CosmologicalParticleMeshPlan.")
        if not isinstance(neighborhood, AbstractPreparedParticleNeighborhood):
            raise TypeError("neighborhood must be a prepared particle neighborhood.")
        if not isinstance(spatial_kernel, AbstractSPHSmoothingKernel):
            raise TypeError("spatial_kernel must be an AbstractSPHSmoothingKernel.")
        if not isinstance(differential_kernel, TwoBodyDifferentialKernelPlan):
            raise TypeError("differential_kernel must be TwoBodyDifferentialKernelPlan.")
        if angular_split is not None and not isinstance(
            angular_split, SmallAngleSplitPlan
        ):
            raise TypeError("angular_split must be SmallAngleSplitPlan or None.")
        if (
            angular_split is not None
            and angular_split.kernel.kernel_id != differential_kernel.kernel_id
        ):
            raise ValueError("angular_split must partition differential_kernel.")
        if not isinstance(policy, SIDMCollisionPolicy):
            raise TypeError("policy must be SIDMCollisionPolicy.")
        particles = particle_mesh.kinematics.particles
        if neighborhood.particle_discretization_id != particles.prepared_id:
            raise ValueError("Weighted SIDM neighborhood and PM must share support.")
        if np.any(np.asarray(particles.particle_ids, dtype=np.int64) < 0):
            raise ValueError("Weighted SIDM requires nonnegative stable packet IDs.")
        if particles.ambient_dimension != 3 or spatial_kernel.dimension != 3:
            raise ValueError("Weighted SIDM requires three-dimensional support.")
        if not isinstance(neighborhood.box, ParticleBox):
            raise ValueError("Weighted SIDM requires a bounded ParticleBox.")
        lengths = np.asarray(neighborhood.box.lengths)
        if tuple(neighborhood.box.periodic_axes) != (True, True, True) or not np.allclose(
            lengths, np.asarray(particle_mesh.kinematics.box_size)
        ):
            raise ValueError("Weighted SIDM neighborhood must share the periodic PM box.")
        if (
            differential_kernel.first_species.species_plan_id
            != differential_kernel.second_species.species_plan_id
        ):
            raise ValueError("Weighted SIDM currently supports one elastic species only.")
        smoothing_length = float(smoothing_length_comoving)
        if not np.isfinite(smoothing_length) or smoothing_length <= 0.0:
            raise ValueError("smoothing_length_comoving must be finite and positive.")
        time_ = FLRWDistancePlan() if time is None else time
        if not isinstance(time_, FLRWDistancePlan):
            raise TypeError("time must be FLRWDistancePlan.")
        self.particle_mesh = particle_mesh
        self.neighborhood = neighborhood
        self.spatial_kernel = spatial_kernel
        self.differential_kernel = differential_kernel
        self.angular_split = angular_split
        self.policy = policy
        self.time = time_
        self.smoothing_length_comoving = smoothing_length
        self.plan_id = canonical_fingerprint(
            {
                "kind": "weighted-rare-sidm",
                "particle_mesh": particle_mesh.plan_id,
                "neighborhood": neighborhood.prepared_id,
                "spatial_kernel": spatial_kernel.kernel_id,
                "differential_kernel": differential_kernel.kernel_id,
                "angular_split": (
                    None if angular_split is None else angular_split.split_id
                ),
                "policy": policy.policy_id,
                "time": time_.plan_id,
                "smoothing_length_comoving": smoothing_length,
            }
        )

    @property
    def particles(self):
        return self.particle_mesh.kinematics.particles

    def initialize(
        self,
        positions: ArrayLike,
        microscopic_masses: ArrayLike,
        weights: ArrayLike,
        canonical_momenta: ArrayLike,
        scale_factor: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ) -> WeightedSIDMPacketState:
        position = jnp.asarray(positions)
        capacity = self.particles.capacity
        dimension = self.particles.ambient_dimension
        expected = (capacity, dimension)
        if position.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        microscopic = jnp.asarray(microscopic_masses, dtype=position.dtype)
        weight = jnp.asarray(weights, dtype=position.dtype)
        momentum = jnp.asarray(canonical_momenta, dtype=position.dtype)
        if microscopic.shape != (capacity,) or weight.shape != (capacity,):
            raise ValueError("microscopic_masses and weights must have capacity shape.")
        if momentum.shape != expected:
            raise ValueError(f"canonical_momenta must have shape {expected}.")
        active = (
            self.particles.active_mask
            if active_mask is None
            else jnp.asarray(active_mask, dtype=bool)
        )
        if active.shape != (capacity,):
            raise ValueError("active_mask must have particle-capacity shape.")
        active = active & self.particles.active_mask
        macro = microscopic * weight
        ids = self.particles.particle_ids
        state = WeightedSIDMPacketState(
            jnp.where(active[:, None], position, 0.0),
            jnp.where(active, microscopic, 0.0),
            jnp.where(active, weight, 0.0),
            jnp.where(active, macro, 0.0),
            jnp.where(active[:, None], momentum, 0.0),
            active,
            ids,
            jnp.full((capacity,), -1, dtype=jnp.int64),
            jnp.where(active, 0, -1).astype(jnp.int32),
            scale_factor,
        )
        species_mass = jnp.asarray(
            self.differential_kernel.first_species.mass, dtype=position.dtype
        )
        species_scale = jnp.maximum(
            jnp.maximum(jnp.abs(microscopic), jnp.abs(species_mass)),
            jnp.finfo(position.dtype).tiny,
        )
        species_valid = jnp.all(
            ~active
            | (
                jnp.abs(microscopic - species_mass)
                <= 64.0 * jnp.finfo(position.dtype).eps * species_scale
            )
        )
        valid = (
            jnp.any(active)
            & jnp.array_equal(ids, self.particles.particle_ids)
            & species_valid
            & _mass_relation(state)
            & _finite_state(state)
        )
        leaves = eqx.filter(state, eqx.is_array)
        first = jax.tree.leaves(leaves)[0]
        first = eqx.error_if(
            first,
            ~valid,
            "Weighted SIDM initial state must match the kernel species and have "
            "finite, positive, mass-consistent packet fields.",
        )
        return eqx.tree_at(lambda value: value.positions, state, first)

    def density(
        self, state: WeightedSIDMPacketState, /
    ) -> tuple[SplatDepositResult, ParticleGridSplatState]:
        """Deposit runtime gravitational macro masses without mutating support."""

        _shape_check(state, self.particles.capacity, self.particles.ambient_dimension)
        routes = self.particle_mesh.gravity.transfer.build(
            state.positions, active_mask=state.active_mask
        )
        deposited = self.particle_mesh.gravity.transfer.deposit_content(
            routes, state.gravitational_masses
        )
        return deposited, routes

    def acceleration(
        self,
        state: WeightedSIDMPacketState,
        args: Any = None,
        /,
        *,
        background_density: ArrayLike | None = None,
    ) -> WeightedPMForceResult:
        deposited, routes = self.density(state)
        density = deposited.density
        if background_density is not None:
            background = jnp.asarray(background_density, dtype=density.dtype)
            if background.shape != density.shape:
                raise ValueError("background_density must match the gravity grid.")
            density = density + background
        potential, _, cell_acceleration, solved = (
            self.particle_mesh.gravity.gravity.solve_density(density, args)
        )
        gathered = self.particle_mesh.gravity.transfer.gather(routes, cell_acceleration)
        active = state.active_mask
        acceleration = jnp.where(active[:, None], gathered.values, 0.0)
        support_complete = jnp.all(gathered.support | ~active)
        mass_valid = _mass_relation(state)
        identity_valid = jnp.array_equal(state.packet_ids, self.particles.particle_ids)
        net_force = jnp.sum(state.gravitational_masses[:, None] * acceleration, axis=0)
        successful = (
            deposited.successful
            & solved.converged
            & support_complete
            & mass_valid
            & identity_valid
            & _finite_state(state)
            & jnp.all(jnp.isfinite(acceleration))
        )
        return WeightedPMForceResult(
            acceleration,
            potential,
            deposited,
            routes,
            net_force,
            solved.converged,
            support_complete,
            mass_valid,
            identity_valid,
            successful,
        )

    def advance_particle_mesh(
        self,
        background: FLRWBackground,
        state: WeightedSIDMPacketState,
        end_scale_factor: ArrayLike,
        acceleration_start: ArrayLike,
        args: Any = None,
        /,
    ) -> WeightedPMIntervalResult:
        if not isinstance(background, FLRWBackground):
            raise TypeError("background must be FLRWBackground.")
        if background.scale.scale_id != self.particle_mesh.kinematics.scale.scale_id:
            raise ValueError("Background and weighted PM scale contracts disagree.")
        _shape_check(state, self.particles.capacity, self.particles.ambient_dimension)
        state = eqx.tree_at(
            lambda value: value.scale_factor,
            state,
            background.require_flat(state.scale_factor),
        )
        acceleration_0 = jnp.asarray(acceleration_start, dtype=state.positions.dtype)
        if acceleration_0.shape != state.positions.shape:
            raise ValueError("acceleration_start must align with packet positions.")
        end = jnp.asarray(end_scale_factor, dtype=state.scale_factor.dtype).reshape(())
        interval_valid = jnp.isfinite(end) & (end > state.scale_factor)
        safe_end = jnp.where(
            interval_valid,
            end,
            state.scale_factor * (1.0 + jnp.finfo(end.dtype).eps),
        )
        midpoint = 0.5 * (state.scale_factor + safe_end)
        coefficients = KDKCoefficients(
            background.kick_factor(state.scale_factor, midpoint),
            background.drift_factor(state.scale_factor, safe_end),
            background.kick_factor(midpoint, safe_end),
        )
        active = state.active_mask
        safe_mass = jnp.where(active, state.gravitational_masses, 1.0)
        proposal = KDKTransactionPlan(self.particle_mesh.kinematics.box_size).propose(
            state.positions,
            state.canonical_momenta,
            safe_mass,
            acceleration_0,
            coefficients,
        )
        proposed_state = WeightedSIDMPacketState(
            jnp.where(active[:, None], proposal.positions, 0.0),
            state.microscopic_masses,
            state.weights,
            state.gravitational_masses,
            jnp.where(active[:, None], proposal.half_momenta, 0.0),
            active,
            state.packet_ids,
            state.parent_packet_ids,
            state.lineage_depth,
            safe_end,
        )
        endpoint = self.acceleration(proposed_state, args)
        completion = KDKTransactionPlan(self.particle_mesh.kinematics.box_size).complete(
            proposal, safe_mass, endpoint.acceleration
        )
        candidate = WeightedSIDMPacketState(
            jnp.where(active[:, None], completion.positions, 0.0),
            state.microscopic_masses,
            state.weights,
            state.gravitational_masses,
            jnp.where(active[:, None], completion.momenta, 0.0),
            active,
            state.packet_ids,
            state.parent_packet_ids,
            state.lineage_depth,
            safe_end,
        )
        successful = (
            interval_valid
            & proposal.successful
            & endpoint.successful
            & completion.successful
            & _mass_relation(candidate)
            & _finite_state(candidate)
        )
        accepted = _state_where(successful, candidate, state)
        diagnostics = WeightedPMIntervalDiagnostics(
            coefficients.drift,
            coefficients.first_kick,
            coefficients.second_kick,
            endpoint.deposited.balance.maximum_absolute_balance_defect,
            endpoint.net_force,
            successful,
        )
        return WeightedPMIntervalResult(
            candidate, accepted, endpoint.acceleration, diagnostics, successful
        )

    def collide(
        self,
        state: WeightedSIDMPacketState,
        key: Array,
        epoch: ArrayLike,
        physical_time_step: ArrayLike,
        /,
    ) -> WeightedSIDMCollisionResult:
        _shape_check(state, self.particles.capacity, self.particles.ambient_dimension)
        dtype = state.positions.dtype
        active = state.active_mask
        active_column = active[:, None]
        safe_positions = jnp.where(jnp.isfinite(state.positions), state.positions, 0.0)
        safe_scale = jnp.where(
            jnp.isfinite(state.scale_factor) & (state.scale_factor > 0.0),
            state.scale_factor,
            1.0,
        )
        safe_mass = jnp.where(active, state.gravitational_masses, 1.0)
        safe_micro = jnp.where(active, state.microscopic_masses, 1.0)
        safe_weight = jnp.where(active, state.weights, 0.0)
        safe_momentum = jnp.where(
            jnp.isfinite(state.canonical_momenta), state.canonical_momenta, 0.0
        )
        dt = jnp.asarray(physical_time_step, dtype=dtype).reshape(())
        time_valid = jnp.isfinite(dt) & (dt >= 0.0)
        safe_dt = jnp.where(time_valid, dt, 0.0)
        neighborhood = self.neighborhood.build(safe_positions, active_mask=active)
        pairs = neighborhood.pair_relation
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
        velocity = safe_momentum / (safe_mass[:, None] * safe_scale)
        relative = velocity[left] - velocity[right]
        speed = jnp.sqrt(ein.contract("ni,ni->n", relative, relative))
        if self.angular_split is None:
            moments = self.differential_kernel.moments(speed)
            split_successful = jnp.ones(speed.shape, dtype=bool)
        else:
            split_evaluation = self.angular_split.moments(speed)
            moments = split_evaluation.rare
            split_successful = split_evaluation.successful
        sigma_total = moments.total.astype(dtype)
        probability = (
            jnp.maximum(safe_weight[left], safe_weight[right])
            * sigma_total
            * speed
            * safe_dt
            * kernel_physical
        )
        probability = jnp.where(support & moments.supported, probability, 0.0)
        probability_valid = jnp.all(
            ~support
            | (
                moments.supported
                & jnp.isfinite(probability)
                & (probability >= 0.0)
                & (probability <= self.policy.maximum_pair_probability)
            )
        )
        particle_probability = jnp.zeros((self.particles.capacity,), dtype=dtype)
        particle_probability = particle_probability.at[left].add(probability)
        particle_probability = particle_probability.at[right].add(probability)
        aggregate_valid = jnp.all(
            ~active
            | (
                jnp.isfinite(particle_probability)
                & (particle_probability <= self.policy.maximum_particle_probability)
            )
        )
        pair_keys = _pair_keys(key, pairs, epoch)
        uniforms = jax.vmap(
            lambda local: jr.uniform(jr.fold_in(local, 0), (), dtype=dtype)
        )(pair_keys)
        priorities = jax.vmap(
            lambda local: jr.uniform(jr.fold_in(local, 1), (), dtype=dtype)
        )(pair_keys)
        proposed = support & moments.supported & (speed > 0.0) & (uniforms < probability)
        selected = _select_endpoint_disjoint(
            proposed, priorities, pairs, self.particles.capacity
        )
        event_count = jnp.sum(selected, dtype=jnp.int32)
        angle_keys = jax.vmap(lambda local: jr.fold_in(local, 2))(pair_keys)
        if self.angular_split is None:
            sampled = jax.vmap(self.differential_kernel.sample_angles)(angle_keys, speed)
        else:
            sampled = jax.vmap(self.angular_split.sample_rare_angles)(angle_keys, speed)
        first_scattered, second_scattered = _scattered_velocities(
            velocity[left],
            velocity[right],
            safe_micro[left],
            safe_micro[right],
            sampled.cosine,
            sampled.azimuth,
        )
        q = jnp.where(
            selected,
            jnp.minimum(safe_weight[left], safe_weight[right]),
            0.0,
        )
        epsilon = jnp.finfo(dtype).eps
        split_left = selected & (safe_weight[left] > q)
        split_right = selected & (safe_weight[right] > q)
        child_required = jnp.sum(split_left, dtype=jnp.int32) + jnp.sum(
            split_right, dtype=jnp.int32
        )
        available_count = jnp.sum(~active & self.particles.active_mask, dtype=jnp.int32)
        capacity_valid = (event_count <= self.policy.maximum_events_per_half_step) & (
            child_required <= available_count
        )
        left_remaining_mass = safe_micro[left] * (safe_weight[left] - q)
        right_remaining_mass = safe_micro[right] * (safe_weight[right] - q)
        left_scattered_mass = safe_micro[left] * q
        right_scattered_mass = safe_micro[right] * q
        left_target_momentum = jnp.where(
            split_left[:, None],
            left_remaining_mass[:, None] * safe_scale * velocity[left],
            left_scattered_mass[:, None] * safe_scale * first_scattered,
        )
        right_target_momentum = jnp.where(
            split_right[:, None],
            right_remaining_mass[:, None] * safe_scale * velocity[right],
            right_scattered_mass[:, None] * safe_scale * second_scattered,
        )
        momentum = safe_momentum.at[left].add(
            jnp.where(selected[:, None], left_target_momentum - safe_momentum[left], 0.0)
        )
        momentum = momentum.at[right].add(
            jnp.where(
                selected[:, None], right_target_momentum - safe_momentum[right], 0.0
            )
        )
        weights = safe_weight.at[left].add(jnp.where(split_left, -q, 0.0))
        weights = weights.at[right].add(jnp.where(split_right, -q, 0.0))
        macro = state.gravitational_masses.at[left].add(
            jnp.where(split_left, -left_scattered_mass, 0.0)
        )
        macro = macro.at[right].add(jnp.where(split_right, -right_scattered_mass, 0.0))
        positions = state.positions
        microscopic = state.microscopic_masses
        child_active = active
        parents = state.parent_packet_ids
        depth = state.lineage_depth
        inactive_priority = jnp.where(
            ~active & self.particles.active_mask,
            state.packet_ids,
            jnp.iinfo(jnp.int64).max,
        )
        available_slots = jnp.argsort(inactive_priority)
        request_mask = jnp.concatenate((split_left, split_right))
        request_parent = jnp.concatenate((left, right))
        request_position = jnp.concatenate(
            (state.positions[left], state.positions[right])
        )
        request_micro = jnp.concatenate((safe_micro[left], safe_micro[right]))
        request_weight = jnp.concatenate((q, q))
        request_mass = request_micro * request_weight
        request_velocity = jnp.concatenate((first_scattered, second_scattered))
        request_momentum = request_mass[:, None] * safe_scale * request_velocity
        parent_ids = state.packet_ids[request_parent]
        request_side = jnp.concatenate((jnp.zeros_like(left), jnp.ones_like(right)))
        request_order = jnp.lexsort((request_side, parent_ids))

        def assign_child(index, carry):
            (
                next_slot,
                position_values,
                micro_values,
                weight_values,
                mass_values,
                momentum_values,
                active_values,
                parent_values,
                depth_values,
            ) = carry
            request = request_order[index]
            take = request_mask[request] & (next_slot < available_count)
            slot = available_slots[jnp.minimum(next_slot, self.particles.capacity - 1)]
            position_values = position_values.at[slot].set(
                jnp.where(take, request_position[request], position_values[slot])
            )
            micro_values = micro_values.at[slot].set(
                jnp.where(take, request_micro[request], micro_values[slot])
            )
            weight_values = weight_values.at[slot].set(
                jnp.where(take, request_weight[request], weight_values[slot])
            )
            mass_values = mass_values.at[slot].set(
                jnp.where(take, request_mass[request], mass_values[slot])
            )
            momentum_values = momentum_values.at[slot].set(
                jnp.where(take, request_momentum[request], momentum_values[slot])
            )
            active_values = active_values.at[slot].set(active_values[slot] | take)
            parent = request_parent[request]
            parent_values = parent_values.at[slot].set(
                jnp.where(take, state.packet_ids[parent], parent_values[slot])
            )
            depth_values = depth_values.at[slot].set(
                jnp.where(take, state.lineage_depth[parent] + 1, depth_values[slot])
            )
            return (
                next_slot + take.astype(jnp.int32),
                position_values,
                micro_values,
                weight_values,
                mass_values,
                momentum_values,
                active_values,
                parent_values,
                depth_values,
            )

        assigned = jax.lax.fori_loop(
            0,
            request_mask.size,
            assign_child,
            (
                jnp.asarray(0, dtype=jnp.int32),
                positions,
                microscopic,
                weights,
                macro,
                momentum,
                child_active,
                parents,
                depth,
            ),
        )
        child_slots_used = assigned[0]
        candidate = WeightedSIDMPacketState(
            assigned[1],
            assigned[2],
            assigned[3],
            assigned[4],
            assigned[5],
            assigned[6],
            state.packet_ids,
            assigned[7],
            assigned[8],
            state.scale_factor,
        )
        self_weight = safe_weight * self.spatial_kernel.value(0.0, smoothing)
        number_density = self_weight.at[left].add(
            jnp.where(support, safe_weight[right] * kernel_comoving, 0.0)
        )
        number_density = number_density.at[right].add(
            jnp.where(support, safe_weight[left] * kernel_comoving, 0.0)
        )
        number_density_physical = number_density / safe_scale**3
        transfer_cross_section = moments.transfer.astype(dtype)
        maximum_transfer = jnp.zeros((self.particles.capacity,), dtype=dtype)
        maximum_transfer = maximum_transfer.at[left].max(
            jnp.where(support, transfer_cross_section, 0.0)
        )
        maximum_transfer = maximum_transfer.at[right].max(
            jnp.where(support, transfer_cross_section, 0.0)
        )
        inverse_mfp = number_density_physical * maximum_transfer
        mean_free_path = jnp.where(inverse_mfp > 0.0, 1.0 / inverse_mfp, jnp.inf)
        support_radius = safe_scale * self.spatial_kernel.support_factor * smoothing
        knudsen = mean_free_path / support_radius
        knudsen_valid = jnp.all(
            ~active
            | (maximum_transfer == 0.0)
            | (jnp.isfinite(knudsen) & (knudsen >= self.policy.minimum_knudsen_number))
        )
        endpoint_count = jnp.zeros((self.particles.capacity,), dtype=jnp.int32)
        endpoint_count = endpoint_count.at[left].add(selected.astype(jnp.int32))
        endpoint_count = endpoint_count.at[right].add(selected.astype(jnp.int32))
        endpoint_disjoint = jnp.all(endpoint_count <= 1)
        before = _ledger(state)
        after = _ledger(candidate)
        mass_defect = after.mass - before.mass
        momentum_defect = after.canonical_momentum - before.canonical_momentum
        energy_defect = after.kinetic_energy - before.kinetic_energy
        packet_momentum_norm = jnp.sqrt(
            ein.contract("ni,ni->n", state.canonical_momenta, state.canonical_momenta)
        )
        momentum_scale = jnp.maximum(
            jnp.sum(jnp.where(active, packet_momentum_norm, 0.0)),
            jnp.finfo(dtype).tiny,
        )
        energy_scale = jnp.maximum(jnp.abs(before.kinetic_energy), jnp.finfo(dtype).tiny)
        mass_scale = jnp.maximum(jnp.abs(before.mass), jnp.finfo(dtype).tiny)
        conservative = (
            (jnp.abs(mass_defect) <= 2048.0 * epsilon * mass_scale)
            & (
                jnp.sqrt(ein.contract("i,i->", momentum_defect, momentum_defect))
                <= 2048.0 * epsilon * momentum_scale
            )
            & (jnp.abs(energy_defect) <= 4096.0 * epsilon * energy_scale)
        )
        identity_order = jnp.argsort(candidate.packet_ids)
        ordered_ids = candidate.packet_ids[identity_order]
        identity_valid = jnp.all(ordered_ids >= 0) & jnp.all(
            ordered_ids[1:] > ordered_ids[:-1]
        )
        parent_location = jnp.searchsorted(
            ordered_ids, candidate.parent_packet_ids, side="left"
        )
        safe_parent_location = jnp.minimum(parent_location, self.particles.capacity - 1)
        parent_slot = identity_order[safe_parent_location]
        parent_known = (
            (candidate.parent_packet_ids >= 0)
            & (parent_location < self.particles.capacity)
            & (ordered_ids[safe_parent_location] == candidate.parent_packet_ids)
        )
        parent_depth = candidate.lineage_depth[parent_slot]
        lineage_valid = (
            identity_valid
            & jnp.array_equal(candidate.packet_ids, self.particles.particle_ids)
            & jnp.all(
                jnp.where(
                    candidate.active_mask,
                    (
                        (
                            (candidate.lineage_depth == 0)
                            & (candidate.parent_packet_ids == -1)
                        )
                        | (
                            (candidate.lineage_depth > parent_depth)
                            & parent_known
                            & candidate.active_mask[parent_slot]
                        )
                    ),
                    (candidate.lineage_depth == -1) & (candidate.parent_packet_ids == -1),
                )
            )
            & (child_slots_used == child_required)
        )
        species_mass = jnp.asarray(
            self.differential_kernel.first_species.mass, dtype=dtype
        )
        species_scale = jnp.maximum(
            jnp.maximum(jnp.abs(state.microscopic_masses), jnp.abs(species_mass)),
            jnp.finfo(dtype).tiny,
        )
        species_valid = jnp.all(
            ~active
            | (
                jnp.abs(state.microscopic_masses - species_mass)
                <= 64.0 * epsilon * species_scale
            )
        )
        mass_valid = _mass_relation(state) & _mass_relation(candidate) & species_valid
        finite = (
            _finite_state(state)
            & _finite_state(candidate)
            & time_valid
            & jnp.all(jnp.isfinite(kernel_physical))
            & jnp.all(jnp.isfinite(speed))
            & jnp.all(jnp.isfinite(probability))
            & jnp.isfinite(mass_defect)
            & jnp.all(jnp.isfinite(momentum_defect))
            & jnp.isfinite(energy_defect)
        )
        angular_sampling_valid = jnp.all(
            ~selected
            | (
                sampled.supported
                & jnp.isfinite(sampled.normalization_residual)
                & (
                    sampled.normalization_residual
                    <= self.differential_kernel.normalization_tolerance
                )
            )
        )
        kernel_supported = (
            jnp.all(~support | moments.supported)
            & jnp.all(~support | split_successful)
            & angular_sampling_valid
        )
        successful = (
            neighborhood.successful
            & kernel_supported
            & probability_valid
            & aggregate_valid
            & capacity_valid
            & knudsen_valid
            & endpoint_disjoint
            & mass_valid
            & lineage_valid
            & conservative
            & finite
        )
        accepted = _state_where(successful, candidate, state)
        diagnostics = WeightedSIDMCollisionDiagnostics(
            dt,
            smoothing,
            kernel_physical,
            speed,
            sigma_total,
            probability,
            particle_probability,
            q,
            proposed,
            selected,
            selected & successful,
            child_required,
            child_slots_used,
            event_count,
            number_density_physical,
            mean_free_path,
            support_radius,
            knudsen,
            mass_defect,
            momentum_defect,
            energy_defect,
            neighborhood.successful,
            kernel_supported,
            jnp.all(~support | split_successful),
            sampled.normalization_residual,
            angular_sampling_valid,
            probability_valid,
            aggregate_valid,
            capacity_valid,
            knudsen_valid,
            endpoint_disjoint,
            mass_valid,
            lineage_valid,
            conservative,
            finite,
            successful,
        )
        return WeightedSIDMCollisionResult(
            candidate, accepted, pairs, diagnostics, successful, self.plan_id
        )

    def rollout(
        self,
        background: FLRWBackground,
        state: WeightedSIDMPacketState,
        key: Array,
        args: Any = None,
        /,
    ) -> WeightedSIDMRolloutResult:
        if not isinstance(background, FLRWBackground):
            raise TypeError("background must be FLRWBackground.")
        if background.scale.scale_id != self.particle_mesh.kinematics.scale.scale_id:
            raise ValueError("Background and weighted SIDM scale contracts disagree.")
        state = eqx.tree_at(
            lambda value: value.scale_factor,
            state,
            background.require_flat(state.scale_factor),
        )
        initial_scale = self.particle_mesh.scale_factors[0].astype(
            state.scale_factor.dtype
        )
        scale_valid = jnp.abs(state.scale_factor - initial_scale) <= 1.0e-12
        initial_force = self.acceleration(state, args)
        running = scale_valid & initial_force.successful
        accepted_count = jnp.asarray(0, dtype=jnp.int32)

        def step(carry, item):
            current, acceleration, active_run, accepted_steps = carry
            index, end_scale = item
            dt = self.time.cosmic_time_between(
                background, current.scale_factor, end_scale
            )
            first = self.collide(current, jr.fold_in(key, 2 * index), 2 * index, 0.5 * dt)
            force_after_collision = self.acceleration(first.accepted_state, args)
            mesh = self.advance_particle_mesh(
                background,
                first.accepted_state,
                end_scale,
                force_after_collision.acceleration,
                args,
            )
            second = self.collide(
                mesh.accepted_state,
                jr.fold_in(key, 2 * index + 1),
                2 * index + 1,
                0.5 * dt,
            )
            accepted = (
                active_run
                & first.successful
                & force_after_collision.successful
                & mesh.successful
                & second.successful
            )
            next_state = _state_where(accepted, second.accepted_state, current)
            next_acceleration = jnp.where(accepted, mesh.acceleration, acceleration)
            return (
                next_state,
                next_acceleration,
                accepted,
                accepted_steps + accepted.astype(jnp.int32),
            ), (
                first.diagnostics,
                mesh.diagnostics,
                second.diagnostics,
                accepted,
            )

        indices = jnp.arange(self.particle_mesh.scale_factors.size - 1, dtype=jnp.int32)
        carry, outputs = jax.lax.scan(
            step,
            (state, initial_force.acceleration, running, accepted_count),
            (indices, self.particle_mesh.scale_factors[1:]),
        )
        final_state, _, completed, accepted_steps = carry
        accepted = outputs[3]
        failure_index = jnp.min(
            jnp.where(~accepted, indices, indices.size), initial=indices.size
        )
        first_failed = jnp.where(completed, -1, failure_index)
        diagnostics = WeightedSIDMRolloutDiagnostics(
            outputs[0],
            outputs[1],
            outputs[2],
            accepted,
            completed,
            accepted_steps,
            first_failed,
        )
        return WeightedSIDMRolloutResult(
            final_state, diagnostics, completed, self.plan_id
        )


class WeightedPacketResamplingPlan(StrictModule, NonTrainableState):
    """Accepted-boundary deterministic packet closure at fixed capacity.

    Mass, centroid, and canonical momentum are always declared preserved. The
    final velocity moment is either scalar kinetic energy or the full velocity
    covariance. Position covariance and velocity moments above the declared
    order are explicitly reported as lost, never silently claimed.
    """

    target_active_count: int = eqx.field(static=True)
    periodic_box_size: tuple[float, ...] = eqx.field(static=True)
    velocity_moment: Literal["kinetic_energy", "covariance"] = eqx.field(static=True)
    maximum_packet_weight: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        target_active_count: int,
        /,
        *,
        periodic_box_size: tuple[float, ...],
        velocity_moment: Literal["kinetic_energy", "covariance"] = "kinetic_energy",
        maximum_packet_weight: float = np.finfo(np.float64).max,
        tolerance: float = 1.0e-10,
    ):
        count = int(target_active_count)
        maximum = float(maximum_packet_weight)
        tolerance_ = float(tolerance)
        box_size = tuple(float(value) for value in periodic_box_size)
        if count <= 0:
            raise ValueError("target_active_count must be positive.")
        if velocity_moment not in ("kinetic_energy", "covariance"):
            raise ValueError("velocity_moment must be 'kinetic_energy' or 'covariance'.")
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_packet_weight must be finite and positive.")
        if not np.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and nonnegative.")
        if not box_size or any(
            not np.isfinite(value) or value <= 0.0 for value in box_size
        ):
            raise ValueError("periodic_box_size must contain finite positive lengths.")
        self.target_active_count = count
        self.periodic_box_size = box_size
        self.velocity_moment = velocity_moment
        self.maximum_packet_weight = maximum
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "weighted-packet-resampling",
                "target_active_count": count,
                "periodic_box_size": list(box_size),
                "velocity_moment": velocity_moment,
                "maximum_packet_weight": maximum,
                "tolerance": tolerance_,
            }
        )

    def apply(
        self,
        state: WeightedSIDMPacketState,
        accepted_boundary: ArrayLike,
        /,
    ) -> WeightedPacketResamplingResult:
        capacity, dimension = state.positions.shape
        _shape_check(state, capacity, dimension)
        if len(self.periodic_box_size) != dimension:
            raise ValueError("periodic_box_size must match packet position dimension.")
        if self.target_active_count > capacity:
            raise ValueError("target_active_count exceeds fixed packet capacity.")
        if (
            self.velocity_moment == "covariance"
            and self.target_active_count < 2 * dimension
        ):
            raise ValueError(
                "Covariance closure requires at least twice the ambient dimension."
            )
        accepted_boundary_ = jnp.asarray(accepted_boundary, dtype=bool).reshape(())
        ordered_packet_ids = jnp.sort(state.packet_ids)
        identity_valid = jnp.all(ordered_packet_ids >= 0) & jnp.all(
            ordered_packet_ids[1:] > ordered_packet_ids[:-1]
        )
        active = state.active_mask
        active_column = active[:, None]
        safe_positions = jnp.where(active_column, state.positions, 0.0)
        safe_momenta = jnp.where(active_column, state.canonical_momenta, 0.0)
        mass = jnp.where(active, state.gravitational_masses, 0.0)
        total_mass = jnp.sum(mass)
        safe_total = jnp.where(total_mass > 0.0, total_mass, 1.0)
        normalized = mass / safe_total
        safe_macro = jnp.where(active, state.gravitational_masses, 1.0)
        velocity = safe_momenta / (safe_macro[:, None] * state.scale_factor)
        box_size = jnp.asarray(self.periodic_box_size, dtype=state.positions.dtype)
        phase = 2.0 * jnp.pi * safe_positions / box_size
        phase_cosine = ein.contract("n,ni->i", normalized, jnp.cos(phase))
        phase_sine = ein.contract("n,ni->i", normalized, jnp.sin(phase))
        phase_resultant = jnp.sqrt(phase_cosine**2 + phase_sine**2)
        centroid_defined = jnp.all(phase_resultant > self.tolerance)
        centroid_phase = jnp.mod(jnp.arctan2(phase_sine, phase_cosine), 2.0 * jnp.pi)
        centroid = box_size * centroid_phase / (2.0 * jnp.pi)
        mean_velocity = ein.contract("n,ni->i", normalized, velocity)
        centered_position = (
            jnp.mod(safe_positions - centroid + 0.5 * box_size, box_size) - 0.5 * box_size
        )
        centered_velocity = velocity - mean_velocity
        position_covariance = ein.contract(
            "n,ni,nj->ij", normalized, centered_position, centered_position
        )
        velocity_covariance = ein.contract(
            "n,ni,nj->ij", normalized, centered_velocity, centered_velocity
        )
        kinetic_before = 0.5 * jnp.sum(
            mass * ein.contract("ni,ni->n", velocity, velocity)
        )
        third_before = ein.contract(
            "n,ni,nj,nk->ijk",
            normalized,
            centered_velocity,
            centered_velocity,
            centered_velocity,
        )
        first_active = jnp.argmax(active.astype(jnp.int32))
        microscopic_mass = state.microscopic_masses[first_active]
        common_micro_scale = jnp.maximum(
            jnp.maximum(
                jnp.abs(state.microscopic_masses),
                jnp.abs(microscopic_mass),
            ),
            jnp.finfo(state.positions.dtype).tiny,
        )
        common_micro = jnp.all(
            ~active
            | (
                jnp.abs(state.microscopic_masses - microscopic_mass)
                <= self.tolerance * common_micro_scale
            )
        )
        target_count = self.target_active_count
        target_mass = total_mass / target_count
        target_weight = target_mass / jnp.where(
            microscopic_mass > 0.0, microscopic_mass, 1.0
        )
        slot_order = jnp.argsort(state.packet_ids)
        target_slots = slot_order[:target_count]
        rank = (
            jnp.zeros((capacity,), dtype=jnp.int32)
            .at[target_slots]
            .set(jnp.arange(target_count, dtype=jnp.int32))
        )
        target_active = jnp.zeros((capacity,), dtype=bool).at[target_slots].set(True)
        local_rank = rank
        trace = jnp.trace(velocity_covariance)
        if self.velocity_moment == "kinetic_energy":
            nonzero_dispersion = trace > self.tolerance
            count_valid = (target_count >= 2) | ~nonzero_dispersion
            amplitude = jnp.sqrt(jnp.maximum(trace / max(target_count - 1, 1), 0.0))
            scalar = jnp.where(
                local_rank < target_count - 1,
                amplitude,
                -(target_count - 1) * amplitude,
            )
            deviation = jnp.zeros((capacity, dimension), dtype=state.positions.dtype)
            deviation = deviation.at[:, 0].set(scalar)
        else:
            spectrum = HermitianSpectrum(velocity_covariance, tolerance=self.tolerance)
            eigenvalues = jnp.maximum(spectrum.eigenvalues, 0.0)
            eigenvectors = spectrum.eigenvectors
            mode = local_rank // 2
            sign = jnp.where(local_rank % 2 == 0, 1.0, -1.0)
            safe_mode = jnp.minimum(mode, dimension - 1)
            amplitude = jnp.sqrt(0.5 * target_count * eigenvalues[safe_mode])
            sigma_deviation = (
                sign[:, None] * amplitude[:, None] * eigenvectors[:, safe_mode].T
            )
            deviation = jnp.where(
                (local_rank < 2 * dimension)[:, None], sigma_deviation, 0.0
            )
            count_valid = spectrum.valid & (
                spectrum.minimum_eigenvalue
                >= -self.tolerance
                * jnp.maximum(
                    jnp.max(jnp.abs(eigenvalues)),
                    jnp.finfo(state.positions.dtype).tiny,
                )
            )
        new_velocity = mean_velocity + deviation
        positions = jnp.where(target_active[:, None], centroid, 0.0)
        microscopic = jnp.where(target_active, microscopic_mass, 0.0)
        weights = jnp.where(target_active, target_weight, 0.0)
        gravitational = jnp.where(target_active, target_mass, 0.0)
        momentum = jnp.where(
            target_active[:, None],
            target_mass * state.scale_factor * new_velocity,
            0.0,
        )
        lineage_root_slot = target_slots[0]
        lineage_root_id = state.packet_ids[lineage_root_slot]
        is_lineage_root = jnp.arange(capacity) == lineage_root_slot
        parent_ids = jnp.where(
            target_active & ~is_lineage_root, lineage_root_id, -1
        ).astype(jnp.int64)
        depths = jnp.where(target_active, jnp.where(is_lineage_root, 0, 1), -1).astype(
            jnp.int32
        )
        candidate = WeightedSIDMPacketState(
            positions,
            microscopic,
            weights,
            gravitational,
            momentum,
            target_active,
            state.packet_ids,
            parent_ids,
            depths,
            state.scale_factor,
        )
        lineage_valid = (
            jnp.sum(
                candidate.active_mask
                & (candidate.parent_packet_ids == -1)
                & (candidate.lineage_depth == 0),
                dtype=jnp.int32,
            )
            == 1
        ) & jnp.all(
            jnp.where(
                candidate.active_mask,
                (
                    ((candidate.parent_packet_ids == -1) & (candidate.lineage_depth == 0))
                    | (
                        (candidate.parent_packet_ids == lineage_root_id)
                        & (candidate.packet_ids != lineage_root_id)
                        & (candidate.lineage_depth == 1)
                    )
                ),
                (candidate.parent_packet_ids == -1) & (candidate.lineage_depth == -1),
            )
        )
        after_mass = jnp.sum(candidate.gravitational_masses)
        after_normalized = candidate.gravitational_masses / jnp.where(
            after_mass > 0.0, after_mass, 1.0
        )
        after_phase = 2.0 * jnp.pi * candidate.positions / box_size
        after_phase_cosine = ein.contract(
            "n,ni->i", after_normalized, jnp.cos(after_phase)
        )
        after_phase_sine = ein.contract("n,ni->i", after_normalized, jnp.sin(after_phase))
        after_centroid = (
            box_size
            * jnp.mod(
                jnp.arctan2(after_phase_sine, after_phase_cosine),
                2.0 * jnp.pi,
            )
            / (2.0 * jnp.pi)
        )
        after_velocity = candidate.canonical_momenta / jnp.where(
            candidate.active_mask[:, None],
            candidate.gravitational_masses[:, None] * candidate.scale_factor,
            1.0,
        )
        centroid_defect = (
            jnp.mod(after_centroid - centroid + 0.5 * box_size, box_size) - 0.5 * box_size
        )
        after_mean = ein.contract("n,ni->i", after_normalized, after_velocity)
        after_centered = after_velocity - after_mean
        covariance_after = ein.contract(
            "n,ni,nj->ij", after_normalized, after_centered, after_centered
        )
        kinetic_after = 0.5 * jnp.sum(
            candidate.gravitational_masses
            * ein.contract("ni,ni->n", after_velocity, after_velocity)
        )
        third_after = ein.contract(
            "n,ni,nj,nk->ijk",
            after_normalized,
            after_centered,
            after_centered,
            after_centered,
        )
        mass_defect = after_mass - total_mass
        momentum_defect = jnp.sum(candidate.canonical_momenta, axis=0) - jnp.sum(
            jnp.where(active[:, None], state.canonical_momenta, 0.0), axis=0
        )
        kinetic_defect = kinetic_after - kinetic_before
        covariance_defect = covariance_after - velocity_covariance
        position_loss = jnp.sqrt(jnp.sum(position_covariance**2))
        third_loss = jnp.sqrt(jnp.sum((third_after - third_before) ** 2))
        mass_tolerance = self.tolerance * jnp.maximum(
            jnp.abs(total_mass), jnp.finfo(state.positions.dtype).tiny
        )
        centroid_tolerance = self.tolerance * jnp.maximum(
            jnp.sqrt(jnp.sum(box_size**2)),
            jnp.finfo(state.positions.dtype).tiny,
        )
        input_momentum_norm = jnp.sqrt(
            ein.contract("ni,ni->n", safe_momenta, safe_momenta)
        )
        momentum_tolerance = self.tolerance * jnp.maximum(
            jnp.sum(input_momentum_norm),
            jnp.finfo(state.positions.dtype).tiny,
        )
        energy_tolerance = self.tolerance * jnp.maximum(
            jnp.abs(kinetic_before),
            jnp.finfo(state.positions.dtype).tiny,
        )
        covariance_tolerance = self.tolerance * jnp.maximum(
            jnp.sqrt(jnp.sum(velocity_covariance**2)),
            jnp.finfo(state.positions.dtype).tiny,
        )
        if self.velocity_moment == "kinetic_energy":
            declared_closed = jnp.abs(kinetic_defect) <= energy_tolerance
        else:
            declared_closed = (
                jnp.sqrt(jnp.sum(covariance_defect**2)) <= covariance_tolerance
            )
        moment_closed = (
            (jnp.abs(mass_defect) <= mass_tolerance)
            & (jnp.sqrt(jnp.sum(centroid_defect**2)) <= centroid_tolerance)
            & (jnp.sqrt(jnp.sum(momentum_defect**2)) <= momentum_tolerance)
            & declared_closed
        )
        capacity_valid = count_valid & (target_count <= capacity)
        nonnegative = jnp.all(candidate.weights >= 0.0) & jnp.all(
            candidate.gravitational_masses >= 0.0
        )
        bounded = jnp.all(
            ~candidate.active_mask | (candidate.weights <= self.maximum_packet_weight)
        )
        finite = _finite_state(state) & _finite_state(candidate)
        successful = (
            accepted_boundary_
            & common_micro
            & centroid_defined
            & identity_valid
            & lineage_valid
            & capacity_valid
            & nonnegative
            & bounded
            & moment_closed
            & _mass_relation(state)
            & _mass_relation(candidate)
            & finite
        )
        accepted = _state_where(successful, candidate, state)
        diagnostics = WeightedPacketResamplingDiagnostics(
            jnp.sum(active, dtype=jnp.int32),
            jnp.sum(candidate.active_mask, dtype=jnp.int32),
            mass_defect,
            centroid_defect,
            momentum_defect,
            kinetic_defect,
            covariance_defect,
            position_loss,
            third_loss,
            accepted_boundary_,
            common_micro,
            centroid_defined,
            identity_valid,
            lineage_valid,
            capacity_valid,
            nonnegative,
            bounded,
            moment_closed,
            finite,
            successful,
        )
        return WeightedPacketResamplingResult(
            candidate, accepted, diagnostics, successful
        )


__all__ = [
    "WeightedPacketLedger",
    "WeightedPacketResamplingDiagnostics",
    "WeightedPacketResamplingPlan",
    "WeightedPacketResamplingResult",
    "WeightedPMForceResult",
    "WeightedPMIntervalDiagnostics",
    "WeightedPMIntervalResult",
    "WeightedSIDMCollisionDiagnostics",
    "WeightedSIDMCollisionResult",
    "WeightedSIDMPacketState",
    "WeightedSIDMPlan",
    "WeightedSIDMRolloutDiagnostics",
    "WeightedSIDMRolloutResult",
]
