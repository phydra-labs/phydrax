#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._particle_epoch import ParticleCapacityRequest
from ._population import (
    ParticleAllocationRequest,
    ParticlePopulationPlan,
    ParticlePopulationState,
)


SPHDensityInitialization: TypeAlias = Literal["summation", "continuity"]


class SPHParticleSourceState(StrictModule):
    unmaterialized_mass: Array
    cumulative_requested_mass: Array
    cumulative_injected_mass: Array
    cumulative_injected_momentum: Array
    cumulative_barotropic_energy: Array
    next_event_id: Array


class SPHEmissionRequest(StrictModule):
    mass_flux: Array
    source_velocity: Array
    source_density: Array
    barotropic_specific_energy: Array
    valid: Array


class SPHRuntimeState(StrictModule):
    population: ParticlePopulationState
    position: Array
    velocity: Array
    evolved_density: Array
    source: SPHParticleSourceState


class SPHEmissionEvidence(StrictModule):
    mass_residual: Array
    momentum_residual: Array
    energy_residual: Array
    minimum_domain_margin: Array
    minimum_wall_margin: Array
    minimum_particle_clearance: Array
    allocation_tie_margin: Array
    finite: Array
    derivative_valid: Array
    successful: Array


class SPHEmissionResult(StrictModule):
    candidate_state: SPHRuntimeState
    accepted_state: SPHRuntimeState
    slots: Array
    incarnations: Array
    requested_count: Array
    inserted_count: Array
    capacity_growth_required: Array
    capacity_request: ParticleCapacityRequest
    event_tape: Any
    evidence: SPHEmissionEvidence
    successful: Array


class SPHParticleSourcePlan(StrictModule, NonTrainableState):
    """Deterministic fixed-site accepted-boundary SPH emission transaction."""

    population: ParticlePopulationPlan
    source_sites: Array
    normals: Array
    quadrature_area: Array
    candidate_offsets: Array
    particle_mass: Array
    domain_lower: Array
    domain_upper: Array
    minimum_source_clearance: float = eqx.field(static=True)
    minimum_particle_clearance: float = eqx.field(static=True)
    minimum_wall_clearance: float = eqx.field(static=True)
    maximum_emissions_per_site: int = eqx.field(static=True)
    density_initialization: SPHDensityInitialization = eqx.field(static=True)
    replay_policy: Any
    schedule_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        population: ParticlePopulationPlan,
        source_sites: ArrayLike,
        normals: ArrayLike,
        quadrature_area: ArrayLike,
        candidate_offsets: ArrayLike,
        particle_mass: ArrayLike,
        /,
        *,
        domain_lower: ArrayLike,
        domain_upper: ArrayLike,
        minimum_source_clearance: float,
        minimum_particle_clearance: float,
        minimum_wall_clearance: float,
        maximum_emissions_per_site: int,
        density_initialization: SPHDensityInitialization,
        replay_policy: Any,
        schedule_id: str = "sph-emission",
    ):
        if not isinstance(population, ParticlePopulationPlan):
            raise TypeError("population must be ParticlePopulationPlan.")
        sites = np.asarray(source_sites, dtype=float)
        normal = np.asarray(normals, dtype=float)
        area = np.asarray(quadrature_area, dtype=float)
        offsets = np.asarray(candidate_offsets, dtype=float)
        mass = np.asarray(particle_mass, dtype=float)
        lower = np.asarray(domain_lower, dtype=float)
        upper = np.asarray(domain_upper, dtype=float)
        maximum = int(maximum_emissions_per_site)
        dimension = population.particles.ambient_dimension
        if sites.ndim != 2 or sites.shape[1] != dimension or sites.shape[0] == 0:
            raise ValueError("SPH source sites have an invalid shape.")
        site_count = sites.shape[0]
        if normal.shape != sites.shape or area.shape != (site_count,):
            raise ValueError("SPH source normals/areas do not match sites.")
        if offsets.shape != (site_count, maximum, dimension):
            raise ValueError("SPH candidate offsets do not match source/maximum shape.")
        if mass.ndim == 0:
            mass = np.full((site_count,), float(mass))
        if mass.shape != (site_count,):
            raise ValueError("SPH particle mass must be scalar or source-site shaped.")
        if lower.shape != (dimension,) or upper.shape != (dimension,):
            raise ValueError("SPH source domain bounds have the wrong dimension.")
        normal_norm = np.linalg.norm(normal, axis=-1)
        if (
            maximum <= 0
            or np.any(~np.isfinite(sites))
            or np.any(~np.isfinite(normal))
            or np.any(normal_norm <= 0.0)
            or np.any(~np.isfinite(area))
            or np.any(area <= 0.0)
            or np.any(~np.isfinite(offsets))
            or np.any(~np.isfinite(mass))
            or np.any(mass <= 0.0)
            or np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(upper <= lower)
        ):
            raise ValueError("SPH source geometry/mass/domain data are invalid.")
        if site_count * maximum > population.allocation_capacity:
            raise ValueError(
                "SPH emission capacity exceeds ParticlePopulationPlan allocation capacity."
            )
        clearances = (
            float(minimum_source_clearance),
            float(minimum_particle_clearance),
            float(minimum_wall_clearance),
        )
        if any(not np.isfinite(value) or value < 0.0 for value in clearances):
            raise ValueError("SPH source clearances must be finite and nonnegative.")
        if density_initialization not in ("summation", "continuity"):
            raise ValueError(
                "density_initialization must be 'summation' or 'continuity'."
            )
        if replay_policy.maximum_events < 1:
            raise ValueError("SPH emission requires at least one hybrid tape event slot.")
        identifier = str(schedule_id)
        if not identifier:
            raise ValueError("schedule_id must be nonempty.")
        self.population = population
        self.source_sites = jnp.asarray(sites)
        self.normals = jnp.asarray(normal / normal_norm[:, None])
        self.quadrature_area = jnp.asarray(area)
        self.candidate_offsets = jnp.asarray(offsets)
        self.particle_mass = jnp.asarray(mass)
        self.domain_lower = jnp.asarray(lower)
        self.domain_upper = jnp.asarray(upper)
        self.minimum_source_clearance = clearances[0]
        self.minimum_particle_clearance = clearances[1]
        self.minimum_wall_clearance = clearances[2]
        self.maximum_emissions_per_site = maximum
        self.density_initialization = density_initialization
        self.replay_policy = replay_policy
        self.schedule_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sph-particle-source",
                "population": population.plan_id,
                "sites": array_tree_fingerprint(sites),
                "normals": array_tree_fingerprint(normal / normal_norm[:, None]),
                "area": array_tree_fingerprint(area),
                "offsets": array_tree_fingerprint(offsets),
                "particle_mass": array_tree_fingerprint(mass),
                "domain": (
                    array_tree_fingerprint(lower),
                    array_tree_fingerprint(upper),
                ),
                "clearances": clearances,
                "maximum_emissions_per_site": maximum,
                "density_initialization": density_initialization,
                "replay_policy": replay_policy.policy_id,
                "schedule_id": identifier,
            }
        )

    @property
    def site_count(self) -> int:
        return int(self.source_sites.shape[0])

    @property
    def emission_capacity(self) -> int:
        return self.site_count * self.maximum_emissions_per_site

    def initialize_source_state(self, dtype=float, /) -> SPHParticleSourceState:
        return SPHParticleSourceState(
            jnp.zeros((self.site_count,), dtype=dtype),
            jnp.zeros((), dtype=dtype),
            jnp.zeros((), dtype=dtype),
            jnp.zeros((self.population.particles.ambient_dimension,), dtype=dtype),
            jnp.zeros((), dtype=dtype),
            jnp.asarray(0, dtype=jnp.int64),
        )

    def initialize_runtime(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        /,
        *,
        population: ParticlePopulationState | None = None,
        evolved_density: ArrayLike | None = None,
    ) -> SPHRuntimeState:
        population_ = self.population.initialize() if population is None else population
        if not isinstance(population_, ParticlePopulationState):
            raise TypeError("population must be ParticlePopulationState or None.")
        position_ = jnp.asarray(position)
        velocity_ = jnp.asarray(velocity, dtype=position_.dtype)
        expected = (
            self.population.particles.capacity,
            self.population.particles.ambient_dimension,
        )
        if position_.shape != expected or velocity_.shape != expected:
            raise ValueError("SPH runtime position/velocity shapes are invalid.")
        density = (
            jnp.zeros((expected[0],), dtype=position_.dtype)
            if evolved_density is None
            else jnp.asarray(evolved_density, dtype=position_.dtype)
        )
        if density.shape != (expected[0],):
            raise ValueError("SPH runtime evolved density has invalid shape.")
        return SPHRuntimeState(
            population_,
            jnp.where(population_.active[:, None], position_, 0.0),
            jnp.where(population_.active[:, None], velocity_, 0.0),
            jnp.where(population_.active, density, 0.0),
            self.initialize_source_state(position_.dtype),
        )


def emit_sph_particles(
    plan: SPHParticleSourcePlan,
    state: SPHRuntimeState,
    request: SPHEmissionRequest,
    step_size: ArrayLike,
    /,
    *,
    event_time: ArrayLike = 0.0,
) -> SPHEmissionResult:
    """Materialize one complete fixed-capacity emission batch atomically."""

    if not isinstance(plan, SPHParticleSourcePlan):
        raise TypeError("plan must be SPHParticleSourcePlan.")
    if not isinstance(state, SPHRuntimeState):
        raise TypeError("state must be SPHRuntimeState.")
    if not isinstance(request, SPHEmissionRequest):
        raise TypeError("request must be SPHEmissionRequest.")
    flux = jnp.asarray(request.mass_flux, dtype=state.position.dtype)
    source_velocity = jnp.asarray(request.source_velocity, dtype=state.position.dtype)
    source_density = jnp.asarray(request.source_density, dtype=state.position.dtype)
    specific_energy = jnp.asarray(
        request.barotropic_specific_energy, dtype=state.position.dtype
    )
    valid_site = jnp.asarray(request.valid, dtype=bool)
    dt = jnp.asarray(step_size, dtype=state.position.dtype)
    site_count = plan.site_count
    dimension = plan.population.particles.ambient_dimension
    if (
        flux.shape != (site_count,)
        or source_velocity.shape != (site_count, dimension)
        or source_density.shape != (site_count,)
        or specific_energy.shape != (site_count,)
        or valid_site.shape != (site_count,)
        or dt.shape != ()
    ):
        raise ValueError("SPH emission request arrays have invalid fixed shapes.")
    requested_mass_by_site = jnp.where(
        valid_site,
        flux * plan.quadrature_area.astype(flux.dtype) * dt,
        0.0,
    )
    accumulated = state.source.unmaterialized_mass + requested_mass_by_site
    raw_count = jnp.floor(
        accumulated / plan.particle_mass.astype(accumulated.dtype)
    ).astype(jnp.int32)
    count = jnp.clip(raw_count, 0, plan.maximum_emissions_per_site)
    overflow = jnp.any(raw_count > plan.maximum_emissions_per_site)
    ordinal = jnp.arange(plan.maximum_emissions_per_site, dtype=jnp.int32)
    candidate_valid = valid_site[:, None] & (ordinal[None, :] < count[:, None])
    candidate_position = plan.source_sites[:, None, :].astype(
        state.position.dtype
    ) + plan.candidate_offsets.astype(state.position.dtype)
    flat_valid = candidate_valid.reshape((-1,))
    flat_position = candidate_position.reshape((-1, dimension))
    flat_velocity = jnp.broadcast_to(
        source_velocity[:, None, :], candidate_position.shape
    ).reshape((-1, dimension))
    flat_density = jnp.broadcast_to(
        source_density[:, None], candidate_valid.shape
    ).reshape((-1,))
    flat_energy = jnp.broadcast_to(
        specific_energy[:, None], candidate_valid.shape
    ).reshape((-1,))
    flat_mass = (
        jnp.broadcast_to(plan.particle_mass[:, None], candidate_valid.shape)
        .reshape((-1,))
        .astype(state.position.dtype)
    )
    event_ids = state.source.next_event_id + jnp.arange(
        plan.emission_capacity, dtype=jnp.int64
    )
    allocation = plan.population.allocate(
        state.population,
        ParticleAllocationRequest(event_ids, flat_mass, flat_valid),
    )
    slots = allocation.slots
    safe_slots = jnp.maximum(slots, 0)
    allocated = allocation.allocated
    domain_margin = jnp.min(
        jnp.minimum(
            flat_position - plan.domain_lower.astype(state.position.dtype),
            plan.domain_upper.astype(state.position.dtype) - flat_position,
        ),
        axis=-1,
    )
    repeated_normal = jnp.broadcast_to(
        plan.normals[:, None, :], candidate_position.shape
    ).reshape((-1, dimension))
    repeated_offset = plan.candidate_offsets.reshape((-1, dimension)).astype(
        state.position.dtype
    )
    wall_margin = (
        oe.contract("ed,ed->e", repeated_offset, repeated_normal, backend="jax")
        - plan.minimum_wall_clearance
    )
    active_delta = flat_position[:, None, :] - state.position[None, :, :]
    active_distance = jnp.sqrt(jnp.sum(active_delta * active_delta, axis=-1))
    active_distance = jnp.where(
        state.population.active[None, :], active_distance, jnp.inf
    )
    active_clearance = jnp.min(active_distance, axis=-1)
    candidate_delta = flat_position[:, None, :] - flat_position[None, :, :]
    candidate_distance = jnp.sqrt(jnp.sum(candidate_delta * candidate_delta, axis=-1))
    candidate_pair = flat_valid[:, None] & flat_valid[None, :]
    candidate_pair = candidate_pair & ~jnp.eye(plan.emission_capacity, dtype=bool)
    candidate_clearance = jnp.min(
        jnp.where(candidate_pair, candidate_distance, jnp.inf), axis=-1
    )
    minimum_clearance = jnp.minimum(active_clearance, candidate_clearance)
    geometry_valid = jnp.all(
        (~flat_valid)
        | (
            (domain_margin >= plan.minimum_source_clearance)
            & (wall_margin >= 0.0)
            & (minimum_clearance >= plan.minimum_particle_clearance)
        )
    )
    request_finite = (
        jnp.all(jnp.where(valid_site, jnp.isfinite(flux) & (flux >= 0.0), True))
        & jnp.all(jnp.where(valid_site[:, None], jnp.isfinite(source_velocity), True))
        & jnp.all(
            jnp.where(
                valid_site,
                jnp.isfinite(source_density) & (source_density > 0.0),
                True,
            )
        )
        & jnp.all(jnp.where(valid_site, jnp.isfinite(specific_energy), True))
        & jnp.isfinite(dt)
        & (dt > 0.0)
    )
    density_valid = request_finite & jnp.all(
        jnp.where(valid_site, source_density > 0.0, True)
    )
    successful = allocation.successful & ~overflow & geometry_valid & density_valid
    candidate_population = allocation.candidate_state
    candidate_position_state = state.position.at[safe_slots].set(
        jnp.where(allocated[:, None], flat_position, state.position[safe_slots])
    )
    candidate_velocity_state = state.velocity.at[safe_slots].set(
        jnp.where(allocated[:, None], flat_velocity, state.velocity[safe_slots])
    )
    initialized_density = (
        flat_density
        if plan.density_initialization == "continuity"
        else jnp.zeros_like(flat_density)
    )
    candidate_density_state = state.evolved_density.at[safe_slots].set(
        jnp.where(allocated, initialized_density, state.evolved_density[safe_slots])
    )
    inserted_by_site = count.astype(accumulated.dtype) * plan.particle_mass.astype(
        accumulated.dtype
    )
    remainder = accumulated - inserted_by_site
    requested_mass = jnp.sum(requested_mass_by_site)
    inserted_mass = jnp.sum(jnp.where(flat_valid, flat_mass, 0.0))
    inserted_momentum = jnp.sum(
        jnp.where(flat_valid[:, None], flat_mass[:, None] * flat_velocity, 0.0),
        axis=0,
    )
    inserted_energy = jnp.sum(jnp.where(flat_valid, flat_mass * flat_energy, 0.0))
    candidate_source = SPHParticleSourceState(
        remainder,
        state.source.cumulative_requested_mass + requested_mass,
        state.source.cumulative_injected_mass + inserted_mass,
        state.source.cumulative_injected_momentum + inserted_momentum,
        state.source.cumulative_barotropic_energy + inserted_energy,
        state.source.next_event_id + jnp.sum(flat_valid, dtype=jnp.int64),
    )
    mass_residual = (
        inserted_mass
        + jnp.sum(remainder)
        - jnp.sum(state.source.unmaterialized_mass)
        - requested_mass
    )
    momentum_residual = jnp.zeros_like(inserted_momentum)
    energy_residual = jnp.zeros_like(inserted_energy)
    mass_scale = jnp.maximum(jnp.abs(requested_mass), 1.0)
    tolerance = 256.0 * jnp.finfo(state.position.dtype).eps * mass_scale
    threshold_distance = jnp.min(
        jnp.abs(
            accumulated / plan.particle_mass.astype(accumulated.dtype)
            - jnp.round(accumulated / plan.particle_mass.astype(accumulated.dtype))
        )
    )
    successful = (
        successful
        & (jnp.abs(mass_residual) <= tolerance)
        & jnp.all(jnp.abs(momentum_residual) <= tolerance)
        & (jnp.abs(energy_residual) <= tolerance)
    )
    derivative_valid = successful & (threshold_distance > tolerance)
    candidate = SPHRuntimeState(
        candidate_population,
        candidate_position_state,
        candidate_velocity_state,
        candidate_density_state,
        candidate_source,
    )
    accepted = SPHRuntimeState(
        ParticlePopulationState(
            jnp.where(successful, candidate.population.active, state.population.active),
            jnp.where(successful, candidate.population.mass, state.population.mass),
            jnp.where(
                successful, candidate.population.incarnation, state.population.incarnation
            ),
            jnp.where(
                successful,
                candidate.population.ever_occupied,
                state.population.ever_occupied,
            ),
            jnp.where(successful, candidate.population.retired, state.population.retired),
        ),
        jnp.where(successful, candidate.position, state.position),
        jnp.where(successful, candidate.velocity, state.velocity),
        jnp.where(successful, candidate.evolved_density, state.evolved_density),
        SPHParticleSourceState(
            jnp.where(
                successful,
                candidate.source.unmaterialized_mass,
                state.source.unmaterialized_mass,
            ),
            jnp.where(
                successful,
                candidate.source.cumulative_requested_mass,
                state.source.cumulative_requested_mass,
            ),
            jnp.where(
                successful,
                candidate.source.cumulative_injected_mass,
                state.source.cumulative_injected_mass,
            ),
            jnp.where(
                successful,
                candidate.source.cumulative_injected_momentum,
                state.source.cumulative_injected_momentum,
            ),
            jnp.where(
                successful,
                candidate.source.cumulative_barotropic_energy,
                state.source.cumulative_barotropic_energy,
            ),
            jnp.where(
                successful, candidate.source.next_event_id, state.source.next_event_id
            ),
        ),
    )
    ledger_before = jnp.concatenate(
        (
            state.source.cumulative_injected_mass[None],
            state.source.cumulative_injected_momentum,
            state.source.cumulative_barotropic_energy[None],
        )
    )
    ledger_after = jnp.concatenate(
        (
            accepted.source.cumulative_injected_mass[None],
            accepted.source.cumulative_injected_momentum,
            accepted.source.cumulative_barotropic_energy[None],
        )
    )
    event_active = successful & (allocation.allocated_count > 0)
    tape_capacity = plan.replay_policy.maximum_events
    tape_active = jnp.zeros((tape_capacity,), dtype=bool).at[0].set(event_active)
    from ...solver._hybrid_event import HybridEventTape

    tape = HybridEventTape(
        event_indices=jnp.full((tape_capacity,), -1, dtype=jnp.int32)
        .at[0]
        .set(jnp.where(event_active, 0, -1)),
        event_times=jnp.zeros((tape_capacity,), dtype=state.position.dtype)
        .at[0]
        .set(jnp.asarray(event_time, dtype=state.position.dtype)),
        states_before=jnp.zeros(
            (tape_capacity,) + ledger_before.shape, dtype=ledger_before.dtype
        )
        .at[0]
        .set(ledger_before),
        states_after=jnp.zeros(
            (tape_capacity,) + ledger_after.shape, dtype=ledger_after.dtype
        )
        .at[0]
        .set(ledger_after),
        guard_residuals=jnp.zeros((tape_capacity,), dtype=state.position.dtype)
        .at[0]
        .set(threshold_distance),
        transversality=jnp.zeros((tape_capacity,), dtype=state.position.dtype)
        .at[0]
        .set(threshold_distance),
        saltation_valid=jnp.zeros((tape_capacity,), dtype=bool)
        .at[0]
        .set(derivative_valid & event_active),
        determinant_signs=jnp.ones((tape_capacity,), dtype=state.position.dtype),
        log_abs_determinants=jnp.zeros((tape_capacity,), dtype=state.position.dtype),
        log_jacobian_valid=jnp.zeros((tape_capacity,), dtype=bool),
        active=tape_active,
        event_count=jnp.where(event_active, 1, 0).astype(jnp.int32),
        terminal=jnp.asarray(False),
        capacity_exceeded=jnp.asarray(False),
        status=jnp.where(successful, 0, plan.replay_policy.failure).astype(jnp.int32),
        policy_id=plan.replay_policy.policy_id,
        schedule_id=plan.schedule_id,
    )
    capacity_growth_required = allocation.requested_count > allocation.allocated_count
    capacity_request = ParticleCapacityRequest(
        plan.population.particles.capacity + plan.emission_capacity,
        required_pair_capacity=(
            plan.population.particles.capacity + plan.emission_capacity
        )
        * (plan.population.particles.capacity + plan.emission_capacity - 1)
        // 2,
        reason="sph_emission",
    )
    evidence = SPHEmissionEvidence(
        mass_residual,
        momentum_residual,
        energy_residual,
        jnp.min(jnp.where(flat_valid, domain_margin, jnp.inf), initial=jnp.inf),
        jnp.min(jnp.where(flat_valid, wall_margin, jnp.inf), initial=jnp.inf),
        jnp.min(jnp.where(flat_valid, minimum_clearance, jnp.inf), initial=jnp.inf),
        threshold_distance,
        request_finite,
        derivative_valid,
        successful,
    )
    return SPHEmissionResult(
        candidate,
        accepted,
        slots,
        candidate_population.incarnation[safe_slots],
        allocation.requested_count,
        allocation.allocated_count,
        capacity_growth_required,
        capacity_request,
        tape,
        evidence,
        successful,
    )


__all__ = [
    "SPHDensityInitialization",
    "SPHEmissionEvidence",
    "SPHEmissionRequest",
    "SPHEmissionResult",
    "SPHParticleSourcePlan",
    "SPHParticleSourceState",
    "SPHRuntimeState",
    "emit_sph_particles",
]
