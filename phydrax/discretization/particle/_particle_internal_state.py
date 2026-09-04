#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._tree_math import tree_allfinite
from ...metrix._state_geometry import AbstractStateGeometry
from ._particle_internal_mesh import PreparedParticleInternalBatch


class ParticleInternalBatchState(StrictModule):
    internal_energy: Array
    species_amount: Array
    porosity: Array
    internal_surface_area: Array
    outer_scale: Array
    reaction_front: Array
    active: Array
    batch_id: str = eqx.field(static=True)


class ParticleConversionLedger(StrictModule):
    initial_internal_energy: Array
    initial_species_amount: tuple[Array, ...]
    cumulative_boundary_heat: Array
    cumulative_contact_heat: Array
    cumulative_radiative_heat: Array
    cumulative_species_exchange: tuple[Array, ...]
    cumulative_reaction_energy: Array
    cumulative_phase_change_energy: Array
    accepted_steps: Array


class ParticleConversionState(StrictModule):
    batches: tuple[ParticleInternalBatchState, ...]
    ledger: ParticleConversionLedger
    state_id: str = eqx.field(static=True)


class ParticleSurfaceState(StrictModule):
    internal_energy_density: Array
    species_concentration: Array
    porosity: Array
    surface_measure: Array
    outer_scale: Array
    active: Array
    successful: Array
    batch_id: str = eqx.field(static=True)


class ParticleConversionDiagnostics(StrictModule):
    total_internal_energy: Array
    total_species_amount: tuple[Array, ...]
    minimum_species_margin: Array
    minimum_porosity_margin: Array
    minimum_scale_margin: Array
    successful: Array


def initialize_particle_internal_batch(
    batch: PreparedParticleInternalBatch,
    internal_energy: ArrayLike,
    species_amount: ArrayLike,
    porosity: ArrayLike,
    internal_surface_area: ArrayLike,
    outer_scale: ArrayLike,
    /,
    *,
    reaction_front: ArrayLike | None = None,
) -> ParticleInternalBatchState:
    if not isinstance(batch, PreparedParticleInternalBatch):
        raise TypeError("batch must be a PreparedParticleInternalBatch.")
    energy = jnp.asarray(internal_energy)
    species = jnp.asarray(species_amount, dtype=energy.dtype)
    pore = jnp.asarray(porosity, dtype=energy.dtype)
    area = jnp.asarray(internal_surface_area, dtype=energy.dtype)
    scale = jnp.asarray(outer_scale, dtype=energy.dtype)
    expected_cells = (batch.particle_count, batch.cell_capacity)
    if energy.shape != expected_cells:
        raise ValueError("internal_energy must have particle-cell shape.")
    if species.shape != expected_cells + (batch.species_count,):
        raise ValueError("species_amount must have particle-cell-species shape.")
    if pore.shape != expected_cells or area.shape != expected_cells:
        raise ValueError("Porosity and internal area must have particle-cell shape.")
    if scale.shape != (batch.particle_count,):
        raise ValueError("outer_scale must have internal particle shape.")
    front = (
        jnp.zeros((batch.particle_count, batch.front_count), dtype=energy.dtype)
        if reaction_front is None
        else jnp.asarray(reaction_front, dtype=energy.dtype)
    )
    if front.shape != (batch.particle_count, batch.front_count):
        raise ValueError("reaction_front must have particle-front shape.")
    active = batch.active
    valid = (
        jnp.all(~active[:, None] | jnp.isfinite(energy))
        & jnp.all(~active[:, None, None] | (jnp.isfinite(species) & (species >= 0.0)))
        & jnp.all(~active[:, None] | (jnp.isfinite(pore) & (pore >= 0.0) & (pore < 1.0)))
        & jnp.all(~active[:, None] | (jnp.isfinite(area) & (area >= 0.0)))
        & jnp.all(~active | (jnp.isfinite(scale) & (scale > 0.0)))
        & jnp.all(
            ~active[:, None] | (jnp.isfinite(front) & (front >= 0.0) & (front <= 1.0))
        )
    )
    energy = eqx.error_if(
        energy, ~valid, "Particle-internal initial state is inadmissible."
    )
    return ParticleInternalBatchState(
        jnp.where(active[:, None], energy, 0.0),
        jnp.where(active[:, None, None], species, 0.0),
        jnp.where(active[:, None], pore, 0.0),
        jnp.where(active[:, None], area, 0.0),
        jnp.where(active, scale, 1.0),
        jnp.where(active[:, None], front, 0.0),
        active,
        batch.prepared_id,
    )


def initialize_particle_conversion_state(
    batches: Sequence[ParticleInternalBatchState],
    /,
    *,
    state_id: str | None = None,
) -> ParticleConversionState:
    values = tuple(batches)
    if not values or any(
        not isinstance(value, ParticleInternalBatchState) for value in values
    ):
        raise TypeError("batches must contain ParticleInternalBatchState values.")
    identifiers = tuple(value.batch_id for value in values)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Particle conversion batch IDs must be unique.")
    dtype = values[0].internal_energy.dtype
    zero = jnp.zeros((), dtype=dtype)
    initial_species = tuple(
        jnp.sum(value.species_amount, axis=(0, 1)) for value in values
    )
    ledger = ParticleConversionLedger(
        jnp.sum(jnp.stack(tuple(jnp.sum(value.internal_energy) for value in values))),
        initial_species,
        zero,
        zero,
        zero,
        tuple(jnp.zeros_like(value) for value in initial_species),
        zero,
        zero,
        jnp.zeros((), dtype=jnp.int32),
    )
    generated = canonical_fingerprint(
        {"kind": "particle-conversion-state", "batches": list(identifiers)}
    )
    identifier = generated if state_id is None else str(state_id)
    if not identifier:
        raise ValueError("state_id must be nonempty.")
    return ParticleConversionState(values, ledger, identifier)


def particle_surface_state(
    batch: PreparedParticleInternalBatch,
    state: ParticleInternalBatchState,
    /,
) -> ParticleSurfaceState:
    if batch.prepared_id != state.batch_id:
        raise ValueError("Particle internal state does not match prepared batch.")
    metrics = batch.mesh.metrics(state.outer_scale)
    outer_cell_measure = metrics.cell_measures[:, -1]
    energy_density = state.internal_energy[:, -1] / outer_cell_measure
    concentration = state.species_amount[:, -1, :] / outer_cell_measure[:, None]
    successful = (
        metrics.successful
        & jnp.all(jnp.isfinite(energy_density))
        & jnp.all(jnp.isfinite(concentration) & (concentration >= 0.0))
    )
    return ParticleSurfaceState(
        jnp.where(state.active, energy_density, 0.0),
        jnp.where(state.active[:, None], concentration, 0.0),
        state.porosity[:, -1],
        metrics.surface_measure,
        state.outer_scale,
        state.active,
        successful,
        state.batch_id,
    )


def particle_conversion_diagnostics(
    state: ParticleConversionState, /
) -> ParticleConversionDiagnostics:
    energy = jnp.sum(
        jnp.stack(tuple(jnp.sum(value.internal_energy) for value in state.batches))
    )
    species = tuple(jnp.sum(value.species_amount, axis=(0, 1)) for value in state.batches)
    species_margin = jnp.min(
        jnp.stack(
            tuple(
                jnp.min(
                    jnp.where(
                        value.active[:, None, None],
                        value.species_amount,
                        jnp.inf,
                    )
                )
                for value in state.batches
            )
        )
    )
    porosity_margin = jnp.min(
        jnp.stack(
            tuple(
                jnp.min(
                    jnp.where(
                        value.active[:, None],
                        jnp.minimum(value.porosity, 1.0 - value.porosity),
                        jnp.inf,
                    )
                )
                for value in state.batches
            )
        )
    )
    scale_margin = jnp.min(
        jnp.stack(
            tuple(
                jnp.min(jnp.where(value.active, value.outer_scale, jnp.inf))
                for value in state.batches
            )
        )
    )
    successful = conversion_state_admissible(state)
    return ParticleConversionDiagnostics(
        energy,
        species,
        species_margin,
        porosity_margin,
        scale_margin,
        successful,
    )


def conversion_state_admissible(state: ParticleConversionState, /) -> Array:
    valid = tree_allfinite(state)
    for value in state.batches:
        valid = (
            valid
            & jnp.all(jnp.isfinite(value.internal_energy))
            & jnp.all(value.species_amount >= 0.0)
            & jnp.all((value.porosity >= 0.0) & (value.porosity < 1.0))
            & jnp.all(value.internal_surface_area >= 0.0)
            & jnp.all(value.outer_scale > 0.0)
            & jnp.all((value.reaction_front >= 0.0) & (value.reaction_front <= 1.0))
        )
    return valid


class ParticleConversionStateGeometry(AbstractStateGeometry):
    """Additive four-space geometry with frozen particle-route leaves."""

    state_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_inverse: bool = eqx.field(static=True)
    supports_exact_differential: bool = eqx.field(static=True)
    supports_transport: bool = eqx.field(static=True)
    supports_isometric_transport: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(self, state_id: str, /):
        identifier = str(state_id)
        if not identifier:
            raise ValueError("state_id must be nonempty.")
        self.state_id = identifier
        self.geometry_id = f"state-geometry:particle-conversion:{identifier}"
        self.retraction_method = "conservative-extensive-addition"
        self.trivial = True
        self.supports_exact_inverse = True
        self.supports_exact_differential = True
        self.supports_transport = True
        self.supports_isometric_transport = True
        self.supports_commutator_free = True

    def contains(self, state, /):
        return conversion_state_admissible(state)

    def project_tangent(self, state, vector, /):
        return _continuous_tangent(state, vector)

    def retract(self, state, local_tangent, /):
        local = _continuous_tangent(state, local_tangent)
        return jax.tree.map(
            lambda base, tangent: base + tangent if eqx.is_inexact_array(base) else base,
            state,
            local,
        )

    def inverse_retract(self, state, point, /):
        target = _require_frozen_compatible(state, point)
        return jax.tree.map(
            lambda base, value: (
                value - base if eqx.is_inexact_array(base) else jnp.zeros_like(base)
            ),
            state,
            target,
        )

    def retraction_jvp(self, state, local_tangent, local_velocity, /):
        _continuous_tangent(state, local_tangent)
        return _continuous_tangent(state, local_velocity)

    def retraction_inverse_jvp(self, state, point, tangent, /):
        result = _continuous_tangent(point, tangent)
        return _require_frozen_compatible(state, point, result)

    def retraction_vjp(self, state, local_tangent, cotangent, /):
        _continuous_tangent(state, local_tangent)
        return _continuous_tangent(state, cotangent)

    def transport_tangent(self, state, point, tangent, /):
        result = _continuous_tangent(state, tangent)
        return _require_frozen_compatible(state, point, result)

    def transport_cotangent_pullback(self, state, point, cotangent, /):
        result = _continuous_tangent(state, cotangent)
        return _require_frozen_compatible(state, point, result)

    def cut_locus_margin(self, state, point, /):
        dtype = next(
            leaf.dtype for leaf in jax.tree.leaves(state) if eqx.is_inexact_array(leaf)
        )
        margin = jnp.asarray(1.0, dtype=dtype)
        return _require_frozen_compatible(state, point, margin)


def _matching_tree_leaves(reference, value, name):
    reference_leaves, reference_structure = jax.tree.flatten(reference)
    value_leaves, value_structure = jax.tree.flatten(value)
    if reference_structure != value_structure:
        raise ValueError(f"{name} must preserve the particle-conversion structure.")
    for reference_leaf, value_leaf in zip(reference_leaves, value_leaves, strict=True):
        if jnp.shape(reference_leaf) != jnp.shape(value_leaf):
            raise ValueError(f"{name} leaf shapes must match the state.")
    return reference_leaves, value_leaves


def _require_frozen_compatible(state, point, value=None):
    state_leaves, point_leaves = _matching_tree_leaves(state, point, "Point")
    compatible = jnp.asarray(True)
    for base, target in zip(state_leaves, point_leaves, strict=True):
        if not eqx.is_inexact_array(base):
            compatible = compatible & jnp.all(jnp.asarray(base) == jnp.asarray(target))
    checked = point if value is None else value
    return jax.tree.map(
        lambda leaf: eqx.error_if(
            leaf,
            ~compatible,
            "Particle-conversion points have incompatible frozen route leaves.",
        ),
        checked,
    )


def _continuous_tangent(state, vector):
    _matching_tree_leaves(state, vector, "Tangent")
    return jax.tree.map(
        lambda base, tangent: (
            tangent if eqx.is_inexact_array(base) else jnp.zeros_like(base)
        ),
        state,
        vector,
    )


__all__ = [
    "ParticleConversionDiagnostics",
    "ParticleConversionLedger",
    "ParticleConversionState",
    "ParticleConversionStateGeometry",
    "ParticleInternalBatchState",
    "ParticleSurfaceState",
    "conversion_state_admissible",
    "initialize_particle_conversion_state",
    "initialize_particle_internal_batch",
    "particle_conversion_diagnostics",
    "particle_surface_state",
]
