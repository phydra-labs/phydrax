#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._precision import real_precision_dtype_name
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import ParticleDiscretization, ParticleSetPlan


class AtomisticStatus(IntEnum):
    """Portable status for a finite-molecule prediction or training run."""

    SUCCESS = 0
    NEIGHBOR_OVERFLOW = 1
    NONFINITE = 2
    STOPPED_EARLY = 3


class AtomisticScaleContract(StrictModule, NonTrainableState):
    """Explicit identity and reference conversion for molecular length and energy."""

    length_unit: str = eqx.field(static=True)
    energy_unit: str = eqx.field(static=True)
    length_to_reference: float = eqx.field(static=True)
    energy_to_reference: float = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)

    def __init__(
        self,
        length_unit: str,
        energy_unit: str,
        /,
        *,
        length_to_reference: float = 1.0,
        energy_to_reference: float = 1.0,
    ):
        length = str(length_unit).strip()
        energy = str(energy_unit).strip()
        length_factor = float(length_to_reference)
        energy_factor = float(energy_to_reference)
        if not length or not energy:
            raise ValueError("length_unit and energy_unit must be non-empty.")
        if (
            not np.isfinite(length_factor)
            or length_factor <= 0.0
            or not np.isfinite(energy_factor)
            or energy_factor <= 0.0
        ):
            raise ValueError("Reference conversion factors must be finite and positive.")
        self.length_unit = length
        self.energy_unit = energy
        self.length_to_reference = length_factor
        self.energy_to_reference = energy_factor
        self.scale_id = canonical_fingerprint(
            {
                "kind": "atomistic-scale-contract",
                "length_unit": length,
                "energy_unit": energy,
                "length_to_reference": length_factor,
                "energy_to_reference": energy_factor,
            }
        )

    @property
    def force_unit(self) -> str:
        return f"{self.energy_unit}/{self.length_unit}"


class AtomisticPrecisionPolicy(StrictModule, NonTrainableState):
    """Coordinate, model-compute, reduction, and output precision contract."""

    coordinate_dtype: str = eqx.field(static=True)
    compute_dtype: str = eqx.field(static=True)
    reduction_dtype: str = eqx.field(static=True)
    output_dtype: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        coordinate_dtype: Any = "float64",
        compute_dtype: Any = "float64",
        reduction_dtype: Any = "float64",
        output_dtype: Any = "float64",
    ):
        coordinate = real_precision_dtype_name(coordinate_dtype)
        compute = real_precision_dtype_name(compute_dtype)
        reduction = real_precision_dtype_name(reduction_dtype)
        output = real_precision_dtype_name(output_dtype)
        self.coordinate_dtype = coordinate
        self.compute_dtype = compute
        self.reduction_dtype = reduction
        self.output_dtype = output
        self.policy_id = canonical_fingerprint(
            {
                "kind": "atomistic-precision-policy",
                "coordinate": coordinate,
                "compute": compute,
                "reduction": reduction,
                "output": output,
            }
        )


class AtomicStructure(StrictModule, NonTrainableState):
    """One fixed-capacity finite molecular structure with stable material atoms."""

    atomic_numbers: Array
    positions: Array
    particle_ids: Array
    masses: Array
    active_mask: Array
    particles: ParticleDiscretization
    scale: AtomisticScaleContract
    cell: Array | None
    periodic_axes: Array | None
    name: str = eqx.field(static=True)
    axis_names: tuple[str, str] = eqx.field(static=True)
    has_periodic_metadata: bool = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        atomic_numbers: ArrayLike,
        positions: ArrayLike,
        masses: ArrayLike,
        scale: AtomisticScaleContract,
        /,
        *,
        particle_ids: ArrayLike | None = None,
        active_mask: ArrayLike | None = None,
        cell: ArrayLike | None = None,
        periodic_axes: ArrayLike | None = None,
        name: str = "molecule",
        coordinate_dtype: Any | None = None,
        numeric_version: str = "0",
    ):
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        numbers = np.asarray(atomic_numbers)
        if numbers.ndim != 1 or numbers.size == 0:
            raise ValueError("atomic_numbers must be a non-empty rank-1 array.")
        if not np.issubdtype(numbers.dtype, np.integer):
            raise TypeError("atomic_numbers must contain integers.")
        numbers = numbers.astype(np.int32, copy=False)
        position_host = np.asarray(positions)
        if position_host.shape != (numbers.size, 3):
            raise ValueError("positions must have shape (atom_capacity, 3).")
        inferred_dtype = (
            position_host.dtype
            if np.issubdtype(position_host.dtype, np.inexact)
            else "float64"
        )
        dtype = real_precision_dtype_name(
            inferred_dtype if coordinate_dtype is None else coordinate_dtype
        )
        position_host = position_host.astype(dtype, copy=False)
        mass_host = np.asarray(masses)
        if mass_host.shape != numbers.shape:
            raise ValueError("masses must have the atomic_numbers shape.")
        mass_host = mass_host.astype(dtype, copy=False)
        active = (
            np.ones(numbers.shape, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active.shape != numbers.shape:
            raise ValueError("active_mask must have the atomic_numbers shape.")
        if not np.any(active):
            raise ValueError("An atomic structure requires at least one active atom.")
        if np.any(numbers[active] <= 0):
            raise ValueError("Active atomic numbers must be positive; zero is padding only.")
        if np.any(numbers[~active] != 0):
            raise ValueError("Inactive padded atoms must have atomic number zero.")
        if np.any(~np.isfinite(position_host[active])):
            raise ValueError("Active atom positions must be finite.")
        if np.any(~np.isfinite(mass_host[active])) or np.any(mass_host[active] <= 0.0):
            raise ValueError("Active atom masses must be finite and positive.")
        ids = (
            np.arange(numbers.size, dtype=np.int64)
            if particle_ids is None
            else np.asarray(particle_ids)
        )
        if ids.shape != numbers.shape or not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("particle_ids must be an integer vector matching atomic_numbers.")
        ids = ids.astype(np.int64, copy=False)
        molecule_name = str(name).strip()
        if not molecule_name:
            raise ValueError("name must be non-empty.")

        if cell is None:
            cell_host = None
        else:
            cell_host = np.asarray(cell, dtype=dtype)
            if cell_host.shape != (3, 3) or np.any(~np.isfinite(cell_host)):
                raise ValueError("cell must be a finite (3, 3) array when provided.")
        if periodic_axes is None:
            periodic_host = None
        else:
            periodic_host = np.asarray(periodic_axes, dtype=bool)
            if periodic_host.shape != (3,):
                raise ValueError("periodic_axes must have shape (3,) when provided.")
        if periodic_host is not None and np.any(periodic_host) and cell_host is None:
            raise ValueError("Periodic axes require preserved cell metadata.")

        particle_plan = ParticleSetPlan(
            ids,
            mass_host,
            ambient_dimension=3,
            active_mask=active,
            name=f"{molecule_name}-atoms",
            domain_labels=("atom",),
            coordinate_dtype=dtype,
        )
        particles = particle_plan.prepare(numeric_version=numeric_version)
        self.atomic_numbers = jnp.asarray(numbers, dtype=jnp.int32)
        self.positions = jnp.asarray(position_host, dtype=dtype)
        self.particle_ids = particles.particle_ids
        self.masses = jnp.asarray(mass_host, dtype=dtype)
        self.active_mask = particles.active_mask
        self.particles = particles
        self.scale = scale
        self.cell = None if cell_host is None else jnp.asarray(cell_host, dtype=dtype)
        self.periodic_axes = (
            None if periodic_host is None else jnp.asarray(periodic_host, dtype=bool)
        )
        self.name = molecule_name
        self.axis_names = ("atom", "cartesian")
        self.has_periodic_metadata = cell_host is not None or periodic_host is not None
        self.structure_id = canonical_fingerprint(
            {
                "kind": "atomic-structure",
                "scale": scale.scale_id,
                "particles": particles.prepared_id,
                "arrays": array_tree_fingerprint(
                    {
                        "atomic_numbers": numbers,
                        "positions": position_host,
                        "cell": cell_host,
                        "periodic_axes": periodic_host,
                    }
                ),
            }
        )

    @property
    def atom_capacity(self) -> int:
        return int(self.atomic_numbers.shape[0])

    @property
    def atom_count(self) -> int:
        return int(np.count_nonzero(np.asarray(self.active_mask)))


def _atomistic_batch_id(
    *,
    scale_id: str,
    topology_id: str,
    structure_ids: tuple[str, ...],
    atomic_numbers: Any,
    positions: Any,
    masses: Any,
    mask: Any,
    cells: Any,
    periodic_axes: Any,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "atomistic-batch",
            "scale": scale_id,
            "topology": topology_id,
            "structure_ids": list(structure_ids),
            "arrays": array_tree_fingerprint(
                {
                    "atomic_numbers": atomic_numbers,
                    "positions": positions,
                    "masses": masses,
                    "mask": mask,
                    "cells": cells,
                    "periodic_axes": periodic_axes,
                }
            ),
        }
    )


class AtomisticBatch(StrictModule, NonTrainableState):
    """Case-isolated fixed-capacity batch of finite molecular structures."""

    atomic_numbers: Array
    positions: Array
    particle_ids: Array
    masses: Array
    atom_mask: Array
    particles: tuple[ParticleDiscretization, ...]
    scale: AtomisticScaleContract
    cells: Array | None
    periodic_axes: Array | None
    atom_cases: Array
    structure_ids: tuple[str, ...] = eqx.field(static=True)
    axis_names: tuple[str, str, str] = eqx.field(static=True)
    has_periodic_metadata: bool = eqx.field(static=True)
    candidate_topology_id: str = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    def __init__(
        self,
        atomic_numbers: ArrayLike,
        positions: ArrayLike,
        masses: ArrayLike,
        scale: AtomisticScaleContract,
        /,
        *,
        particle_ids: ArrayLike | None = None,
        atom_mask: ArrayLike | None = None,
        cells: ArrayLike | None = None,
        periodic_axes: ArrayLike | None = None,
        structure_ids: Sequence[str] | None = None,
        coordinate_dtype: Any | None = None,
        numeric_version: str = "0",
    ):
        if not isinstance(scale, AtomisticScaleContract):
            raise TypeError("scale must be an AtomisticScaleContract.")
        numbers = np.asarray(atomic_numbers)
        position_host = np.asarray(positions)
        if numbers.ndim != 2 or numbers.shape[0] == 0 or numbers.shape[1] == 0:
            raise ValueError("atomic_numbers must have non-empty shape (case, atom).")
        if not np.issubdtype(numbers.dtype, np.integer):
            raise TypeError("atomic_numbers must contain integers.")
        numbers = numbers.astype(np.int32, copy=False)
        if position_host.shape != numbers.shape + (3,):
            raise ValueError("positions must have shape (case, atom, 3).")
        inferred_dtype = (
            position_host.dtype
            if np.issubdtype(position_host.dtype, np.inexact)
            else "float64"
        )
        dtype = real_precision_dtype_name(
            inferred_dtype if coordinate_dtype is None else coordinate_dtype
        )
        position_host = position_host.astype(dtype, copy=False)
        mass_host = np.asarray(masses, dtype=dtype)
        if mass_host.shape != numbers.shape:
            raise ValueError("masses must have shape (case, atom).")
        mask = numbers > 0 if atom_mask is None else np.asarray(atom_mask, dtype=bool)
        if mask.shape != numbers.shape:
            raise ValueError("atom_mask must have shape (case, atom).")
        if np.any(np.sum(mask, axis=1) == 0):
            raise ValueError("Every batch case requires at least one active atom.")
        if np.any(numbers[mask] <= 0):
            raise ValueError("Active atomic numbers must be positive; zero is padding only.")
        if np.any(numbers[~mask] != 0):
            raise ValueError("Inactive padded atoms must have atomic number zero.")
        if np.any(~np.isfinite(position_host[mask])):
            raise ValueError("Active atom positions must be finite.")
        if np.any(~np.isfinite(mass_host[mask])) or np.any(mass_host[mask] <= 0.0):
            raise ValueError("Active atom masses must be finite and positive.")
        case_count, atom_capacity = numbers.shape
        ids = (
            np.broadcast_to(
                np.arange(atom_capacity, dtype=np.int64), (case_count, atom_capacity)
            ).copy()
            if particle_ids is None
            else np.asarray(particle_ids)
        )
        if ids.shape != numbers.shape or not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("particle_ids must be integer values shaped (case, atom).")
        ids = ids.astype(np.int64, copy=False)
        ids_host = (
            tuple(f"case-{index}" for index in range(case_count))
            if structure_ids is None
            else tuple(str(value) for value in structure_ids)
        )
        if len(ids_host) != case_count or any(not value for value in ids_host):
            raise ValueError("structure_ids must name every case with non-empty values.")

        if cells is None:
            cell_host = None
        else:
            cell_host = np.asarray(cells, dtype=dtype)
            if cell_host.shape != (case_count, 3, 3) or np.any(~np.isfinite(cell_host)):
                raise ValueError("cells must have finite shape (case, 3, 3).")
        if periodic_axes is None:
            periodic_host = None
        else:
            periodic_host = np.asarray(periodic_axes, dtype=bool)
            if periodic_host.shape != (case_count, 3):
                raise ValueError("periodic_axes must have shape (case, 3).")
        if periodic_host is not None and np.any(periodic_host) and cell_host is None:
            raise ValueError("Periodic axes require preserved cell metadata.")

        particles = tuple(
            ParticleSetPlan(
                ids[index],
                mass_host[index],
                ambient_dimension=3,
                active_mask=mask[index],
                name=f"{ids_host[index]}-atoms",
                domain_labels=("atom",),
                coordinate_dtype=dtype,
            ).prepare(numeric_version=numeric_version)
            for index in range(case_count)
        )
        atom_cases = np.repeat(np.arange(case_count, dtype=np.int32), atom_capacity)
        topology_id = canonical_fingerprint(
            {
                "kind": "dense-directed-atomistic-candidates",
                "case_count": case_count,
                "atom_capacity": atom_capacity,
                "structure_particles": [value.prepared_id for value in particles],
            }
        )
        self.atomic_numbers = jnp.asarray(numbers, dtype=jnp.int32)
        self.positions = jnp.asarray(position_host, dtype=dtype)
        self.particle_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.masses = jnp.asarray(mass_host, dtype=dtype)
        self.atom_mask = jnp.asarray(mask, dtype=bool)
        self.particles = particles
        self.scale = scale
        self.cells = None if cell_host is None else jnp.asarray(cell_host, dtype=dtype)
        self.periodic_axes = (
            None if periodic_host is None else jnp.asarray(periodic_host, dtype=bool)
        )
        self.atom_cases = jnp.asarray(atom_cases, dtype=jnp.int32)
        self.structure_ids = ids_host
        self.axis_names = ("case", "atom", "cartesian")
        self.has_periodic_metadata = cell_host is not None or periodic_host is not None
        self.candidate_topology_id = topology_id
        self.batch_id = _atomistic_batch_id(
            scale_id=scale.scale_id,
            topology_id=topology_id,
            structure_ids=ids_host,
            atomic_numbers=numbers,
            positions=position_host,
            masses=mass_host,
            mask=mask,
            cells=cell_host,
            periodic_axes=periodic_host,
        )

    @classmethod
    def from_structures(
        cls,
        structures: Sequence[AtomicStructure],
        /,
        *,
        atom_capacity: int | None = None,
    ) -> "AtomisticBatch":
        values = tuple(structures)
        if not values or any(not isinstance(value, AtomicStructure) for value in values):
            raise TypeError("structures must be a non-empty sequence of AtomicStructure.")
        scale = values[0].scale
        if any(value.scale.scale_id != scale.scale_id for value in values[1:]):
            raise ValueError("All structures in a batch must share one exact scale contract.")
        maximum = max(value.atom_capacity for value in values)
        capacity = maximum if atom_capacity is None else int(atom_capacity)
        if capacity < maximum:
            raise ValueError("atom_capacity cannot truncate a structure.")
        if capacity <= 0:
            raise ValueError("atom_capacity must be positive.")
        dtype = values[0].positions.dtype
        count = len(values)
        numbers = np.zeros((count, capacity), dtype=np.int32)
        positions = np.zeros((count, capacity, 3), dtype=np.dtype(dtype))
        masses = np.ones((count, capacity), dtype=np.dtype(dtype))
        ids = np.zeros((count, capacity), dtype=np.int64)
        mask = np.zeros((count, capacity), dtype=bool)
        any_metadata = any(value.has_periodic_metadata for value in values)
        cells = np.zeros((count, 3, 3), dtype=np.dtype(dtype)) if any_metadata else None
        periodic = np.zeros((count, 3), dtype=bool) if any_metadata else None
        for index, structure in enumerate(values):
            size = structure.atom_capacity
            numbers[index, :size] = np.asarray(structure.atomic_numbers)
            positions[index, :size] = np.asarray(structure.positions)
            masses[index, :size] = np.asarray(structure.masses)
            ids[index, :size] = np.asarray(structure.particle_ids)
            mask[index, :size] = np.asarray(structure.active_mask)
            if cells is not None and structure.cell is not None:
                cells[index] = np.asarray(structure.cell)
            if periodic is not None and structure.periodic_axes is not None:
                periodic[index] = np.asarray(structure.periodic_axes)
        return cls(
            numbers,
            positions,
            masses,
            scale,
            particle_ids=ids,
            atom_mask=mask,
            cells=cells,
            periodic_axes=periodic,
            structure_ids=tuple(value.structure_id for value in values),
            coordinate_dtype=dtype,
        )

    @classmethod
    def from_structure(cls, structure: AtomicStructure, /) -> "AtomisticBatch":
        return cls.from_structures((structure,))

    @property
    def case_count(self) -> int:
        return int(self.atomic_numbers.shape[0])

    @property
    def atom_capacity(self) -> int:
        return int(self.atomic_numbers.shape[1])

    @property
    def atom_counts(self) -> Array:
        return jnp.sum(self.atom_mask, axis=1, dtype=jnp.int32)

    def with_positions(self, positions: ArrayLike, /) -> "AtomisticBatch":
        value = jnp.asarray(positions, dtype=self.positions.dtype)
        if value.shape != self.positions.shape:
            raise ValueError("Replacement positions must have the batch position shape.")
        host = np.asarray(value)
        active = np.asarray(self.atom_mask, dtype=bool)
        if np.any(~np.isfinite(host[active])):
            raise ValueError("Replacement active atom positions must be finite.")
        updated = eqx.tree_at(lambda batch: batch.positions, self, value)
        batch_id = _atomistic_batch_id(
            scale_id=self.scale.scale_id,
            topology_id=self.candidate_topology_id,
            structure_ids=self.structure_ids,
            atomic_numbers=np.asarray(self.atomic_numbers),
            positions=host,
            masses=np.asarray(self.masses),
            mask=active,
            cells=None if self.cells is None else np.asarray(self.cells),
            periodic_axes=(
                None if self.periodic_axes is None else np.asarray(self.periodic_axes)
            ),
        )
        object.__setattr__(updated, "batch_id", batch_id)
        return updated


__all__ = [
    "AtomisticBatch",
    "AtomisticPrecisionPolicy",
    "AtomisticScaleContract",
    "AtomisticStatus",
    "AtomicStructure",
]
