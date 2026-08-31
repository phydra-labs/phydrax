#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..stochastic._categorical_diffusion import CategoricalDiffusionSchedule
from ..stochastic._gaussian_diffusion import AbstractGaussianDiffusion
from ._types import AtomisticBatch


def _expanded_mask(batch: AtomisticBatch) -> Array:
    return batch.atom_mask[..., None]


def _center_positions(batch: AtomisticBatch, positions: ArrayLike, /) -> tuple[Array, Array]:
    value = jnp.asarray(positions, dtype=batch.positions.dtype)
    mask = _expanded_mask(batch)
    mass = jnp.where(batch.atom_mask, batch.masses, 0.0)
    total = jnp.sum(mass, axis=1, keepdims=True)
    center = jnp.sum(mass[..., None] * jnp.where(mask, value, 0.0), axis=1) / total
    centered = jnp.where(mask, value - center[:, None, :], 0.0)
    return centered, center


class AtomisticCoordinateDiffusion(StrictModule):
    """Mass-centered continuous diffusion on one fixed-capacity atomistic batch."""

    template: AtomisticBatch
    process: AbstractGaussianDiffusion
    center_weights: Array
    center_weight_squared_norm: Array
    active_count: Array
    process_id: str = eqx.field(static=True)

    def __init__(self, template, process, /, *, process_id: str | None = None):
        if not isinstance(template, AtomisticBatch):
            raise TypeError("template must be an AtomisticBatch.")
        if not isinstance(process, AbstractGaussianDiffusion):
            raise TypeError("process must implement AbstractGaussianDiffusion.")
        if template.has_periodic_metadata:
            raise ValueError("Atomistic coordinate diffusion initially excludes periodic cells.")
        if process.state_shape != (int(template.positions.size),):
            raise ValueError("Coordinate diffusion dimension must equal padded position size.")
        mass = jnp.where(template.atom_mask, template.masses, 0.0)
        center_weights = mass / jnp.sum(mass, axis=1, keepdims=True)
        self.template = template
        self.process = process
        self.center_weights = center_weights
        self.center_weight_squared_norm = jnp.sum(
            center_weights**2, axis=1, keepdims=True
        )
        self.active_count = jnp.sum(
            template.atom_mask, axis=1, keepdims=True, dtype=template.positions.dtype
        )
        self.process_id = process_id or canonical_fingerprint(
            {
                "kind": "atomistic-coordinate-diffusion",
                "batch_topology_id": template.candidate_topology_id,
                "process_id": process.process_id,
            }
        )

    def _require_batch(self, batch: AtomisticBatch, /) -> None:
        if not isinstance(batch, AtomisticBatch):
            raise TypeError("batch must be an AtomisticBatch.")
        if batch.candidate_topology_id != self.template.candidate_topology_id:
            raise ValueError("Atomistic diffusion requires the template candidate topology.")
        if not jnp.array_equal(batch.atomic_numbers, self.template.atomic_numbers):
            raise ValueError("Coordinate diffusion requires fixed atom species.")
        if not (
            jnp.array_equal(batch.atom_mask, self.template.atom_mask)
            and jnp.array_equal(batch.particle_ids, self.template.particle_ids)
            and jnp.array_equal(batch.masses, self.template.masses)
            and batch.scale.scale_id == self.template.scale.scale_id
        ):
            raise ValueError("Coordinate diffusion requires fixed masks, masses, and scale.")

    def perturb(self, batch: AtomisticBatch, key: Key[Array, ""], /, *, time) -> AtomisticBatch:
        self._require_batch(batch)
        centered, _ = _center_positions(batch, batch.positions)
        perturbed = self.process.perturb(key, centered.reshape((-1,)), t1=time).reshape(
            centered.shape
        )
        perturbed, _ = _center_positions(batch, perturbed)
        perturbed = jnp.where(_expanded_mask(batch), perturbed, batch.positions)
        return batch.with_positions(perturbed)

    def conditional_score(self, perturbed: AtomisticBatch, clean: AtomisticBatch, /, *, time):
        self._require_batch(perturbed)
        self._require_batch(clean)
        noisy, _ = _center_positions(perturbed, perturbed.positions)
        source, _ = _center_positions(clean, clean.positions)
        unconstrained = self.process.conditional_score(
            noisy.reshape((-1,)), source.reshape((-1,)), t1=time
        ).reshape(noisy.shape)
        unconstrained = jnp.where(_expanded_mask(clean), unconstrained, 0.0)
        summed = jnp.sum(unconstrained, axis=1, keepdims=True)
        denominator = self.active_count * self.center_weight_squared_norm
        direction = (
            self.center_weight_squared_norm * clean.atom_mask - self.center_weights
        )
        return unconstrained - summed / denominator[..., None] * direction[..., None]


class AtomisticHybridDiffusion(StrictModule):
    """Coupled continuous coordinates and categorical species corruption."""

    coordinate: AtomisticCoordinateDiffusion
    species_schedule: CategoricalDiffusionSchedule
    species: tuple[int, ...] = eqx.field(static=True)
    species_to_index: Array
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate: AtomisticCoordinateDiffusion,
        species_schedule: CategoricalDiffusionSchedule,
        species: Sequence[int],
        /,
    ):
        if not isinstance(coordinate, AtomisticCoordinateDiffusion):
            raise TypeError("coordinate must be an AtomisticCoordinateDiffusion.")
        if not isinstance(species_schedule, CategoricalDiffusionSchedule):
            raise TypeError("species_schedule must be categorical diffusion.")
        values = tuple(int(number) for number in species)
        if len(values) != species_schedule.num_classes or len(set(values)) != len(values):
            raise ValueError("species must uniquely map every categorical class.")
        if any(number <= 0 for number in values):
            raise ValueError("Atomistic species vocabulary must contain positive numbers.")
        maximum = max(values)
        mapping = jnp.full((maximum + 1,), -1, dtype=jnp.int32)
        mapping = mapping.at[jnp.asarray(values)].set(
            jnp.arange(len(values), dtype=jnp.int32)
        )
        active_numbers = coordinate.template.atomic_numbers[coordinate.template.atom_mask]
        if bool(jnp.any(active_numbers > maximum)) or bool(
            jnp.any(mapping[active_numbers] < 0)
        ):
            raise ValueError("Template contains atomic numbers outside species vocabulary.")
        self.coordinate = coordinate
        self.species_schedule = species_schedule
        self.species = values
        self.species_to_index = mapping
        self.process_id = canonical_fingerprint(
            {
                "kind": "atomistic-hybrid-diffusion",
                "coordinate_process_id": coordinate.process_id,
                "species_schedule_id": species_schedule.schedule_id,
                "species": values,
            }
        )

    def perturb(
        self,
        batch: AtomisticBatch,
        coordinate_key: Key[Array, ""],
        species_key: Key[Array, ""],
        /,
        *,
        continuous_time,
        discrete_timestep,
    ) -> tuple[AtomisticBatch, Array]:
        coordinates = self.coordinate.perturb(batch, coordinate_key, time=continuous_time)
        indices = jnp.where(
            batch.atom_mask,
            self.species_to_index[batch.atomic_numbers],
            0,
        )
        noised = self.species_schedule.corrupt(indices, discrete_timestep, species_key)
        noised = jnp.where(batch.atom_mask, noised, -1)
        return coordinates, noised

    def decode_species(self, indices: ArrayLike, mask: ArrayLike, /) -> Array:
        raw = jnp.asarray(indices)
        if not jnp.issubdtype(raw.dtype, jnp.integer):
            raise TypeError("Species indices must use an integer dtype.")
        value = raw.astype(jnp.int32)
        active = jnp.asarray(mask, dtype=bool)
        vocabulary = jnp.asarray(self.species, dtype=jnp.int32)
        if value.shape != active.shape:
            raise ValueError("Species indices and mask must have identical shapes.")
        value = eqx.error_if(
            value,
            jnp.any(active & ((value < 0) | (value >= len(self.species)))),
            "Active species index lies outside the declared vocabulary.",
        )
        safe = jnp.where(active, value, 0)
        return jnp.where(active, vocabulary[safe], 0)


class AtomisticEquivarianceReport(StrictModule):
    rotation_defect: Array
    translation_residual: Array
    permutation_defect: Array
    valid: Array


def atomistic_score_equivariance(
    score_model: Any,
    batch: AtomisticBatch,
    time: ArrayLike,
    rotation: ArrayLike,
    permutation: ArrayLike,
    /,
) -> AtomisticEquivarianceReport:
    raw_matrix = jnp.asarray(rotation)
    raw_order = jnp.asarray(permutation)
    if jnp.iscomplexobj(raw_matrix):
        raise TypeError("rotation must be real-valued.")
    if not jnp.issubdtype(raw_order.dtype, jnp.integer):
        raise TypeError("permutation must use an integer dtype.")
    matrix = raw_matrix.astype(batch.positions.dtype)
    order = raw_order.astype(jnp.int32)
    if matrix.shape != (3, 3) or order.shape != (batch.atom_capacity,):
        raise ValueError("rotation/permutation shapes are invalid.")
    orthogonality = matrix.T @ matrix
    if not bool(
        jnp.allclose(
            orthogonality,
            jnp.eye(3, dtype=matrix.dtype),
            atol=1e-7,
            rtol=1e-7,
        )
        & jnp.isclose(jnp.linalg.det(matrix), 1.0, atol=1e-7, rtol=1e-7)
    ):
        raise ValueError("rotation must be a proper orthogonal matrix.")
    if not bool(jnp.array_equal(jnp.sort(order), jnp.arange(batch.atom_capacity))):
        raise ValueError("permutation must contain every atom-axis index exactly once.")
    base = jnp.asarray(score_model(batch, time))
    if base.shape != batch.positions.shape:
        raise ValueError("Atomistic score model must return one vector per padded atom.")
    rotated_batch = batch.with_positions(batch.positions @ matrix.T)
    rotated = jnp.asarray(score_model(rotated_batch, time))
    if rotated.shape != base.shape:
        raise ValueError("Rotated atomistic score changed shape.")
    rotation_defect = jnp.max(jnp.abs(rotated - base @ matrix.T))
    permuted = AtomisticBatch(
        batch.atomic_numbers[:, order],
        batch.positions[:, order],
        batch.masses[:, order],
        batch.scale,
        particle_ids=batch.particle_ids[:, order],
        atom_mask=batch.atom_mask[:, order],
        cells=batch.cells,
        periodic_axes=batch.periodic_axes,
        structure_ids=batch.structure_ids,
        coordinate_dtype=batch.positions.dtype,
    )
    permuted_score = jnp.asarray(score_model(permuted, time))
    if permuted_score.shape != base.shape:
        raise ValueError("Permuted atomistic score changed shape.")
    permutation_defect = jnp.max(jnp.abs(permuted_score - base[:, order]))
    mass = jnp.where(batch.atom_mask, batch.masses, 0.0)
    translation = jnp.sum(mass[..., None] * base, axis=1)
    translation_residual = jnp.max(jnp.abs(translation))
    finite = jnp.all(
        jnp.isfinite(
            jnp.asarray([rotation_defect, permutation_defect, translation_residual])
        )
    )
    return AtomisticEquivarianceReport(
        rotation_defect,
        translation_residual,
        permutation_defect,
        finite,
    )


__all__ = [
    "AtomisticCoordinateDiffusion",
    "AtomisticEquivarianceReport",
    "AtomisticHybridDiffusion",
    "atomistic_score_equivariance",
]
