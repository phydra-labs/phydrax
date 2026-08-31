#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_where
from ..finite_volume import UnstructuredAMRFluxRegister, UnstructuredAMRHierarchyPlan


class ParticleInternalAdaptationPolicy(StrictModule, NonTrainableState):
    refine_threshold: float = eqx.field(static=True)
    coarsen_threshold: float = eqx.field(static=True)
    minimum_dwell_windows: int = eqx.field(static=True)
    balance_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        refine_threshold: float,
        coarsen_threshold: float,
        /,
        *,
        minimum_dwell_windows: int = 1,
        balance_tolerance: float = 1.0e-10,
    ):
        refine = float(refine_threshold)
        coarsen = float(coarsen_threshold)
        dwell = int(minimum_dwell_windows)
        tolerance = float(balance_tolerance)
        if (
            not np.isfinite(refine)
            or not np.isfinite(coarsen)
            or coarsen < 0.0
            or refine <= coarsen
            or dwell < 0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Particle internal adaptation controls are invalid.")
        self.refine_threshold = refine
        self.coarsen_threshold = coarsen
        self.minimum_dwell_windows = dwell
        self.balance_tolerance = tolerance
        self.policy_id = canonical_fingerprint(
            {
                "kind": "particle-internal-adaptation-policy",
                "refine": refine,
                "coarsen": coarsen,
                "dwell": dwell,
                "balance_tolerance": tolerance,
            }
        )


class ParticleInternalAMRState(StrictModule):
    coarse_internal_energy: Array
    fine_internal_energy: Array
    coarse_species_amount: Array
    fine_species_amount: Array
    coarse_pore_volume: Array
    fine_pore_volume: Array
    coarse_surface_area: Array
    fine_surface_area: Array
    coarse_reaction_progress: Array
    fine_reaction_progress: Array
    coarse_refined: Array
    fine_active: Array
    dwell_windows: Array
    outer_scale: Array
    particle_active: Array
    hierarchy_id: str = eqx.field(static=True)


class ParticleInternalAdaptationEvidence(StrictModule):
    selected_count: Array
    eligible_count: Array
    overflow_count: Array
    energy_residual: Array
    species_residual: Array
    pore_volume_residual: Array
    surface_area_residual: Array
    route_digest: Array
    successful: Array
    policy_id: str = eqx.field(static=True)
    hierarchy_id: str = eqx.field(static=True)


class ParticleInternalAdaptationResult(StrictModule):
    candidate_state: ParticleInternalAMRState
    accepted_state: ParticleInternalAMRState
    evidence: ParticleInternalAdaptationEvidence
    growth_required: Array
    required_additional_cells: Array
    successful: Array


def initialize_particle_internal_amr(
    hierarchy: UnstructuredAMRHierarchyPlan,
    internal_energy: ArrayLike,
    species_amount: ArrayLike,
    porosity: ArrayLike,
    internal_surface_area: ArrayLike,
    reaction_progress: ArrayLike,
    outer_scale: ArrayLike,
    particle_active: ArrayLike,
    /,
) -> ParticleInternalAMRState:
    if not isinstance(hierarchy, UnstructuredAMRHierarchyPlan):
        raise TypeError("hierarchy must be UnstructuredAMRHierarchyPlan.")
    energy = jnp.asarray(internal_energy)
    species = jnp.asarray(species_amount, dtype=energy.dtype)
    pore = jnp.asarray(porosity, dtype=energy.dtype)
    area = jnp.asarray(internal_surface_area, dtype=energy.dtype)
    progress = jnp.asarray(reaction_progress, dtype=energy.dtype)
    scale = jnp.asarray(outer_scale, dtype=energy.dtype)
    active = jnp.asarray(particle_active, dtype=bool)
    particle_count = energy.shape[0]
    coarse_count = hierarchy.coarse.cell_count
    if energy.shape != (particle_count, coarse_count):
        raise ValueError("internal_energy must have particle-coarse-cell shape.")
    if species.ndim != 3 or species.shape[:2] != energy.shape:
        raise ValueError("species_amount must have particle-coarse-cell-species shape.")
    if pore.shape != energy.shape or area.shape != energy.shape:
        raise ValueError("Porosity and surface area must match coarse cells.")
    if progress.ndim != 3 or progress.shape[:2] != energy.shape:
        raise ValueError("reaction_progress must have particle-coarse-cell-front shape.")
    if scale.shape != (particle_count,) or active.shape != (particle_count,):
        raise ValueError("AMR scale/activity must have particle shape.")
    dimension = hierarchy.coarse.cell_dimension
    coarse_volume = scale[:, None] ** dimension * hierarchy.coarse.cell_volumes[None, :]
    fine_shape = (particle_count, hierarchy.fine.cell_count)
    fine_energy = jnp.zeros(fine_shape, dtype=energy.dtype)
    fine_species = jnp.zeros(fine_shape + species.shape[2:], dtype=species.dtype)
    fine_pore = jnp.zeros(fine_shape, dtype=pore.dtype)
    fine_area = jnp.zeros(fine_shape, dtype=area.dtype)
    fine_progress = jnp.zeros(fine_shape + progress.shape[2:], dtype=progress.dtype)
    return ParticleInternalAMRState(
        energy,
        fine_energy,
        species,
        fine_species,
        pore * coarse_volume,
        fine_pore,
        area,
        fine_area,
        progress,
        fine_progress,
        jnp.zeros((particle_count, coarse_count), dtype=bool),
        jnp.zeros(fine_shape, dtype=bool),
        jnp.zeros((particle_count, coarse_count), dtype=jnp.int32),
        scale,
        active,
        hierarchy.plan_id,
    )


def _composite_content(hierarchy, coarse, fine, refined, fine_active):
    coarse_mask = ~refined
    coarse_total = jnp.sum(
        jnp.where(
            coarse_mask.reshape(coarse_mask.shape + (1,) * (coarse.ndim - 2)),
            coarse,
            0.0,
        ),
        axis=1,
    )
    fine_total = jnp.sum(
        jnp.where(
            fine_active.reshape(fine_active.shape + (1,) * (fine.ndim - 2)),
            fine,
            0.0,
        ),
        axis=1,
    )
    return coarse_total + fine_total


def adapt_particle_internal_mesh(
    hierarchy: UnstructuredAMRHierarchyPlan,
    policy: ParticleInternalAdaptationPolicy,
    state: ParticleInternalAMRState,
    indicator: ArrayLike,
    /,
) -> ParticleInternalAdaptationResult:
    if not isinstance(hierarchy, UnstructuredAMRHierarchyPlan):
        raise TypeError("hierarchy must be UnstructuredAMRHierarchyPlan.")
    if not isinstance(policy, ParticleInternalAdaptationPolicy):
        raise TypeError("policy must be ParticleInternalAdaptationPolicy.")
    if not isinstance(state, ParticleInternalAMRState):
        raise TypeError("state must be ParticleInternalAMRState.")
    if state.hierarchy_id != hierarchy.plan_id:
        raise ValueError("AMR state belongs to another hierarchy.")
    values = jnp.asarray(indicator, dtype=state.coarse_internal_energy.dtype)
    particle_count, coarse_count = state.coarse_refined.shape
    if values.shape != (particle_count, coarse_count):
        raise ValueError("indicator must have particle-coarse-cell shape.")

    def select_one(value, active):
        return hierarchy.select(
            value,
            policy.refine_threshold,
            active_mask=jnp.full((coarse_count,), active),
        )

    proposed = jax.vmap(select_one)(values, state.particle_active)
    dwell_ok = state.dwell_windows >= policy.minimum_dwell_windows
    retain = state.coarse_refined & ((values >= policy.coarsen_threshold) | ~dwell_ok)
    refined = proposed.coarse_refined | retain
    fine_active = refined[:, hierarchy.fine_parent_cells]
    newly_refined = refined & ~state.coarse_refined
    newly_coarsened = state.coarse_refined & ~refined

    old_energy = _composite_content(
        hierarchy,
        state.coarse_internal_energy,
        state.fine_internal_energy,
        state.coarse_refined,
        state.fine_active,
    )
    old_species = _composite_content(
        hierarchy,
        state.coarse_species_amount,
        state.fine_species_amount,
        state.coarse_refined,
        state.fine_active,
    )
    old_pore = _composite_content(
        hierarchy,
        state.coarse_pore_volume,
        state.fine_pore_volume,
        state.coarse_refined,
        state.fine_active,
    )
    old_area = _composite_content(
        hierarchy,
        state.coarse_surface_area,
        state.fine_surface_area,
        state.coarse_refined,
        state.fine_active,
    )

    def remap_one(
        coarse_energy,
        fine_energy,
        coarse_species,
        fine_species,
        coarse_pore,
        fine_pore,
        coarse_area,
        fine_area,
        coarse_progress,
        fine_progress,
        previous_refined,
        previous_fine_active,
        next_refined,
        next_fine_active,
        refine_mask,
        coarsen_mask,
    ):
        restricted_energy = hierarchy.restrict_content(
            fine_energy, fine_active_mask=previous_fine_active
        )
        restricted_species = hierarchy.restrict_content(
            fine_species, fine_active_mask=previous_fine_active
        )
        restricted_pore = hierarchy.restrict_content(
            fine_pore, fine_active_mask=previous_fine_active
        )
        restricted_area = hierarchy.restrict_content(
            fine_area, fine_active_mask=previous_fine_active
        )
        restricted_progress = hierarchy.restrict(
            fine_progress,
            fine_active_mask=previous_fine_active,
            bounded=True,
            lower=0.0,
            upper=1.0,
        )
        coarse_energy = jnp.where(coarsen_mask, restricted_energy, coarse_energy)
        coarse_species = jnp.where(
            coarsen_mask[:, None], restricted_species, coarse_species
        )
        coarse_pore = jnp.where(coarsen_mask, restricted_pore, coarse_pore)
        coarse_area = jnp.where(coarsen_mask, restricted_area, coarse_area)
        coarse_progress = jnp.where(
            coarsen_mask[:, None], restricted_progress, coarse_progress
        )
        prolonged_energy = hierarchy.prolong_content(
            coarse_energy, fine_active_mask=next_fine_active
        )
        prolonged_species = hierarchy.prolong_content(
            coarse_species, fine_active_mask=next_fine_active
        )
        prolonged_pore = hierarchy.prolong_content(
            coarse_pore, fine_active_mask=next_fine_active
        )
        prolonged_area = hierarchy.prolong_content(
            coarse_area, fine_active_mask=next_fine_active
        )
        prolonged_progress = hierarchy.prolong(
            coarse_progress,
            fine_active_mask=next_fine_active,
            bounded=True,
            lower=0.0,
            upper=1.0,
        )
        child_new = refine_mask[hierarchy.fine_parent_cells]
        fine_energy = jnp.where(child_new, prolonged_energy, fine_energy)
        fine_species = jnp.where(child_new[:, None], prolonged_species, fine_species)
        fine_pore = jnp.where(child_new, prolonged_pore, fine_pore)
        fine_area = jnp.where(child_new, prolonged_area, fine_area)
        fine_progress = jnp.where(child_new[:, None], prolonged_progress, fine_progress)
        fine_energy = jnp.where(next_fine_active, fine_energy, 0.0)
        fine_species = jnp.where(next_fine_active[:, None], fine_species, 0.0)
        fine_pore = jnp.where(next_fine_active, fine_pore, 0.0)
        fine_area = jnp.where(next_fine_active, fine_area, 0.0)
        fine_progress = jnp.where(next_fine_active[:, None], fine_progress, 0.0)
        return (
            coarse_energy,
            fine_energy,
            coarse_species,
            fine_species,
            coarse_pore,
            fine_pore,
            coarse_area,
            fine_area,
            coarse_progress,
            fine_progress,
        )

    values_out = jax.vmap(remap_one)(
        state.coarse_internal_energy,
        state.fine_internal_energy,
        state.coarse_species_amount,
        state.fine_species_amount,
        state.coarse_pore_volume,
        state.fine_pore_volume,
        state.coarse_surface_area,
        state.fine_surface_area,
        state.coarse_reaction_progress,
        state.fine_reaction_progress,
        state.coarse_refined,
        state.fine_active,
        refined,
        fine_active,
        newly_refined,
        newly_coarsened,
    )
    dwell = jnp.where(
        refined == state.coarse_refined,
        state.dwell_windows + 1,
        jnp.zeros_like(state.dwell_windows),
    )
    candidate = ParticleInternalAMRState(
        *values_out,
        refined,
        fine_active,
        dwell,
        state.outer_scale,
        state.particle_active,
        state.hierarchy_id,
    )
    new_energy = _composite_content(
        hierarchy,
        candidate.coarse_internal_energy,
        candidate.fine_internal_energy,
        refined,
        fine_active,
    )
    new_species = _composite_content(
        hierarchy,
        candidate.coarse_species_amount,
        candidate.fine_species_amount,
        refined,
        fine_active,
    )
    new_pore = _composite_content(
        hierarchy,
        candidate.coarse_pore_volume,
        candidate.fine_pore_volume,
        refined,
        fine_active,
    )
    new_area = _composite_content(
        hierarchy,
        candidate.coarse_surface_area,
        candidate.fine_surface_area,
        refined,
        fine_active,
    )
    energy_residual = new_energy - old_energy
    species_residual = new_species - old_species
    pore_residual = new_pore - old_pore
    area_residual = new_area - old_area
    overflow = jnp.any(proposed.capacity_overflow)
    tolerance = policy.balance_tolerance
    successful = (
        ~overflow
        & jnp.all(
            jnp.abs(energy_residual) <= tolerance * jnp.maximum(jnp.abs(old_energy), 1.0)
        )
        & jnp.all(
            jnp.abs(species_residual)
            <= tolerance * jnp.maximum(jnp.abs(old_species), 1.0)
        )
        & jnp.all(
            jnp.abs(pore_residual) <= tolerance * jnp.maximum(jnp.abs(old_pore), 1.0)
        )
        & jnp.all(
            jnp.abs(area_residual) <= tolerance * jnp.maximum(jnp.abs(old_area), 1.0)
        )
        & jnp.all(jnp.isfinite(new_energy))
    )
    route_weights = jnp.arange(coarse_count, dtype=jnp.int64) + 1
    route_digest = jnp.sum(refined.astype(jnp.int64) * route_weights[None, :])
    evidence = ParticleInternalAdaptationEvidence(
        jnp.sum(proposed.selected_count),
        jnp.sum(proposed.eligible_count),
        jnp.sum(proposed.overflow_count),
        energy_residual,
        species_residual,
        pore_residual,
        area_residual,
        route_digest,
        successful,
        policy.policy_id,
        hierarchy.plan_id,
    )
    accepted = tree_where(successful, candidate, state)
    return ParticleInternalAdaptationResult(
        candidate,
        accepted,
        evidence,
        overflow,
        jnp.sum(proposed.overflow_count),
        successful,
    )


def apply_particle_internal_flux_correction(
    hierarchy: UnstructuredAMRHierarchyPlan,
    coarse_cell_average: ArrayLike,
    register: UnstructuredAMRFluxRegister,
    /,
    *,
    active_mask: ArrayLike | None = None,
) -> Array:
    return hierarchy.reflux(
        coarse_cell_average,
        register,
        coarse_active_mask=active_mask,
    )


__all__ = [
    "ParticleInternalAMRState",
    "ParticleInternalAdaptationEvidence",
    "ParticleInternalAdaptationPolicy",
    "ParticleInternalAdaptationResult",
    "adapt_particle_internal_mesh",
    "apply_particle_internal_flux_correction",
    "initialize_particle_internal_amr",
]
