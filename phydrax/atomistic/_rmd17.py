#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._precision import real_precision_dtype_name
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..units import (
    ANGSTROM,
    conversion_factor,
    DALTON,
    ELECTRONVOLT,
    KILOCALORIE_PER_MOLE,
    UnitDefinition,
)
from ._types import AtomisticBatch, AtomisticScaleContract
from ._units import molar_energy_to_single_system_factor


class RMD17Dataset(StrictModule, NonTrainableState):
    """Validated in-memory view of one local revised-MD17 NPZ archive."""

    atomic_numbers: Array
    positions: Array
    energies: Array
    forces: Array
    masses: Array
    sample_ids: Array
    scale: AtomisticScaleContract
    source_length_unit: UnitDefinition
    source_energy_unit: UnitDefinition
    source_mass_unit: UnitDefinition
    avogadro_constant_set_id: str = eqx.field(static=True)
    source_path: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)

    @property
    def sample_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def atom_count(self) -> int:
        return int(self.atomic_numbers.shape[0])

    def take(self, indices: ArrayLike, /) -> tuple[AtomisticBatch, Array, Array]:
        index = np.asarray(indices)
        if index.ndim != 1 or not np.issubdtype(index.dtype, np.integer):
            raise TypeError("indices must be a rank-1 integer array.")
        index = index.astype(np.int64, copy=False)
        if np.any(index < 0) or np.any(index >= self.sample_count):
            raise IndexError("rMD17 sample index is out of range.")
        if np.unique(index).size != index.size:
            raise ValueError("rMD17 batch indices must be unique.")
        count = int(index.size)
        numbers = np.broadcast_to(
            np.asarray(self.atomic_numbers)[None, :], (count, self.atom_count)
        )
        masses = np.broadcast_to(
            np.asarray(self.masses)[None, :], (count, self.atom_count)
        )
        sample_ids = np.asarray(self.sample_ids)[index]
        batch = AtomisticBatch(
            numbers,
            np.asarray(self.positions)[index],
            masses,
            self.scale,
            structure_ids=tuple(
                f"{self.dataset_id}/sample-{int(sample_id)}" for sample_id in sample_ids
            ),
            coordinate_dtype=self.positions.dtype,
        )
        return batch, self.energies[index], self.forces[index]


class RMD17Split(StrictModule, NonTrainableState):
    """Disjoint deterministic train/validation/test index identity."""

    train_indices: Array
    validation_indices: Array
    test_indices: Array
    seed: int = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    split_id: str = eqx.field(static=True)


def _archive_value(archive: Any, names: tuple[str, ...], /) -> np.ndarray:
    present = tuple(name for name in names if name in archive.files)
    if len(present) != 1:
        raise ValueError(
            f"rMD17 archive must contain exactly one of {names!r}; found {present!r}."
        )
    return np.asarray(archive[present[0]])


def _rmd17_masses(atomic_numbers: np.ndarray, /) -> np.ndarray:
    supported_numbers = np.asarray((1, 6, 7, 8), dtype=np.int32)
    supported_masses = np.asarray((1.00784, 12.011, 14.007, 15.999), dtype=np.float64)
    match = atomic_numbers[:, None] == supported_numbers[None, :]
    if np.any(np.sum(match, axis=1) != 1):
        unsupported = np.unique(atomic_numbers[np.sum(match, axis=1) != 1])
        raise ValueError(
            "The local rMD17 parser has no authoritative mass for atomic numbers "
            f"{unsupported.tolist()}; pass explicit masses."
        )
    return np.sum(match * supported_masses[None, :], axis=1)


def load_rmd17_npz(
    path: str | Path,
    /,
    *,
    scale: AtomisticScaleContract | None = None,
    masses: ArrayLike | None = None,
    dtype: Any = "float64",
) -> RMD17Dataset:
    """Load rMD17 and explicitly convert kcal/mol source energies to one-system energy."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Local rMD17 NPZ does not exist: {source}")
    if source.suffix.lower() != ".npz":
        raise ValueError("rMD17 input must be a local .npz archive.")
    precision = real_precision_dtype_name(dtype)
    archive = np.load(source, allow_pickle=False)
    numbers = _archive_value(archive, ("nuclear_charges", "atomic_numbers", "z", "Z"))
    positions = _archive_value(archive, ("coords", "positions", "R"))
    energies = _archive_value(archive, ("energies", "energy", "E"))
    forces = _archive_value(archive, ("forces", "force", "F"))
    sample_ids = (
        np.asarray(archive["old_indices"])
        if "old_indices" in archive.files
        else np.arange(positions.shape[0], dtype=np.int64)
    )
    archive.close()
    if numbers.ndim != 1 or not np.issubdtype(numbers.dtype, np.integer):
        raise TypeError("rMD17 nuclear charges must be a rank-1 integer array.")
    numbers = numbers.astype(np.int32, copy=False)
    if np.any(numbers <= 0):
        raise ValueError("rMD17 nuclear charges must be positive.")
    if positions.ndim != 3 or positions.shape[1:] != (numbers.size, 3):
        raise ValueError("rMD17 positions must have shape (sample, atom, 3).")
    energies = np.asarray(energies).reshape((-1,))
    if energies.shape != (positions.shape[0],):
        raise ValueError("rMD17 energies must contain one scalar per sample.")
    if forces.shape != positions.shape:
        raise ValueError("rMD17 forces must have the position shape.")
    if sample_ids.shape != (positions.shape[0],) or not np.issubdtype(
        sample_ids.dtype, np.integer
    ):
        raise TypeError("rMD17 old_indices must be one integer per sample.")
    if np.unique(sample_ids).size != sample_ids.size:
        raise ValueError("rMD17 sample identities must be unique.")
    scale_ = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT) if scale is None else scale
    if not isinstance(scale_, AtomisticScaleContract):
        raise TypeError("scale must be an AtomisticScaleContract or None.")
    constant_set_id = "codata-2018"
    length_factor = float(conversion_factor(ANGSTROM, scale_.length_unit))
    energy_factor = molar_energy_to_single_system_factor(
        KILOCALORIE_PER_MOLE,
        scale_.energy_unit,
        constant_set_id=constant_set_id,
    )
    position_values = (positions * length_factor).astype(precision, copy=False)
    energy_values = (energies * energy_factor).astype(precision, copy=False)
    force_values = (forces * energy_factor / length_factor).astype(precision, copy=False)
    if (
        np.any(~np.isfinite(position_values))
        or np.any(~np.isfinite(energy_values))
        or np.any(~np.isfinite(force_values))
    ):
        raise ValueError("rMD17 coordinates, energies, and forces must be finite.")
    if masses is None:
        mass_values = _rmd17_masses(numbers).astype(precision)
    else:
        mass_values = np.asarray(masses, dtype=precision)
        if mass_values.shape != numbers.shape:
            raise ValueError("Explicit masses must have the nuclear-charge shape.")
        if np.any(~np.isfinite(mass_values)) or np.any(mass_values <= 0.0):
            raise ValueError("Explicit masses must be finite and positive.")
    dataset_id = canonical_fingerprint(
        {
            "kind": "local-rmd17-dataset",
            "scale": scale_.scale_id,
            "source_length_unit": ANGSTROM.unit_id,
            "source_energy_unit": KILOCALORIE_PER_MOLE.unit_id,
            "source_mass_unit": DALTON.unit_id,
            "avogadro_constant_set": constant_set_id,
            "arrays": array_tree_fingerprint(
                {
                    "atomic_numbers": numbers,
                    "positions": position_values,
                    "energies": energy_values,
                    "forces": force_values,
                    "masses": mass_values,
                    "sample_ids": sample_ids,
                }
            ),
        }
    )
    return RMD17Dataset(
        atomic_numbers=jnp.asarray(numbers, dtype=jnp.int32),
        positions=jnp.asarray(position_values, dtype=precision),
        energies=jnp.asarray(energy_values, dtype=precision),
        forces=jnp.asarray(force_values, dtype=precision),
        masses=jnp.asarray(mass_values, dtype=precision),
        sample_ids=jnp.asarray(sample_ids, dtype=jnp.int64),
        scale=scale_,
        source_length_unit=ANGSTROM,
        source_energy_unit=KILOCALORIE_PER_MOLE,
        source_mass_unit=DALTON,
        avogadro_constant_set_id=constant_set_id,
        source_path=str(source),
        dataset_id=dataset_id,
    )


def split_rmd17(
    dataset: RMD17Dataset,
    /,
    *,
    train_size: int = 950,
    validation_size: int = 50,
    test_size: int = 1000,
    seed: int = 0,
) -> RMD17Split:
    """Create one deterministic canonical-size disjoint rMD17 split."""

    if not isinstance(dataset, RMD17Dataset):
        raise TypeError("dataset must be an RMD17Dataset.")
    train_count = int(train_size)
    validation_count = int(validation_size)
    test_count = int(test_size)
    seed_value = int(seed)
    if min(train_count, validation_count, test_count) < 0:
        raise ValueError("rMD17 split sizes must be non-negative.")
    selected_count = train_count + validation_count + test_count
    if selected_count > dataset.sample_count:
        raise ValueError(
            f"Requested {selected_count} disjoint samples from {dataset.sample_count}."
        )
    permutation = np.random.default_rng(seed_value).permutation(dataset.sample_count)
    train = permutation[:train_count]
    validation = permutation[train_count : train_count + validation_count]
    test = permutation[
        train_count + validation_count : train_count + validation_count + test_count
    ]
    split_id = canonical_fingerprint(
        {
            "kind": "rmd17-split",
            "dataset": dataset.dataset_id,
            "seed": seed_value,
            "train": array_tree_fingerprint(train),
            "validation": array_tree_fingerprint(validation),
            "test": array_tree_fingerprint(test),
        }
    )
    return RMD17Split(
        train_indices=jnp.asarray(train, dtype=jnp.int32),
        validation_indices=jnp.asarray(validation, dtype=jnp.int32),
        test_indices=jnp.asarray(test, dtype=jnp.int32),
        seed=seed_value,
        dataset_id=dataset.dataset_id,
        split_id=split_id,
    )


__all__ = ["RMD17Dataset", "RMD17Split", "load_rmd17_npz", "split_rmd17"]
