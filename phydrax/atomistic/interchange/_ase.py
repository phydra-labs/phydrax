#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...interchange import AdapterError, AdapterLoss, AdapterReport, AdapterStatus
from .._types import AtomicStructure, AtomisticScaleContract


ASE_PARTICLE_ID_ARRAY = "phydrax_particle_id"
ASE_SOURCE_ID_INFO = "phydrax_source_id"

_STANDARD_ARRAYS = frozenset(("numbers", "positions", "masses"))
_OCCUPANCY_NAMES = frozenset(
    ("occupancy", "occupancies", "partial_occupancy", "disorder")
)
_TOPOLOGY_NAMES = frozenset(("bond", "bonds", "bond_indices", "connectivity", "topology"))
_SPIN_NAMES = frozenset(("initial_magmoms", "magmom", "magmoms", "spin", "spins"))
_UNIT_NAMES = frozenset(("unit", "units", "length_unit", "energy_unit", "mass_unit"))


def is_ase_available() -> bool:
    """Return whether ASE can be imported, without importing it."""

    return importlib.util.find_spec("ase") is not None


def require_ase():
    """Import ASE only at the host-side interoperability boundary."""

    if not is_ase_available():
        raise AdapterError(
            AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE,
            "ASE conversion requires optional dependency 'ase'.",
        )
    return importlib.import_module("ase")


def from_ase_atoms(
    atoms: Any,
    scale: AtomisticScaleContract,
    /,
    *,
    source_id: str | None = None,
    name: str = "ase-atoms",
) -> tuple[AtomicStructure, AdapterReport]:
    """Copy one ``ase.Atoms`` into an identity-bearing native structure.

    ASE coordinates and cells are interpreted in angstrom, energies in electronvolt,
    and masses in dalton. The caller must therefore provide the exact matching scale
    contract. Unsupported external state is either rejected or enumerated in the
    returned adapter report; no ASE or calculator object is retained.
    """

    ase = require_ase()
    if not isinstance(atoms, ase.Atoms):
        raise TypeError("atoms must be an ase.Atoms instance.")
    _require_ase_scale(scale)

    array_names = _string_keys(atoms.arrays, "ASE array")
    info_names = _string_keys(atoms.info, "ASE info")
    _reject_required_semantics(array_names, info_names)

    numbers = np.array(atoms.get_atomic_numbers(), copy=True)
    positions = np.array(atoms.get_positions(wrap=False), copy=True)
    masses = np.array(atoms.get_masses(), copy=True)
    cell, periodic_axes = _validated_cell(atoms)
    _validate_atoms_arrays(numbers, positions, masses)

    particle_ids: np.ndarray | None
    losses: list[AdapterLoss] = []
    if ASE_PARTICLE_ID_ARRAY in array_names:
        particle_ids = np.array(atoms.arrays[ASE_PARTICLE_ID_ARRAY], copy=True)
        _validate_particle_ids(particle_ids, numbers.size)
    else:
        particle_ids = None
        losses.append(
            AdapterLoss(
                f"arrays.{ASE_PARTICLE_ID_ARRAY}",
                "import",
                "synthesized",
                "ASE supplied no stable particle IDs; AtomicStructure assigned its "
                "deterministic order-based IDs.",
                changes_interpretation=False,
            )
        )

    losses.extend(_array_losses(array_names))
    if atoms.constraints:
        losses.append(
            AdapterLoss(
                "constraints",
                "import",
                "dropped",
                "AtomicStructure represents structure, not ASE motion constraints.",
                changes_interpretation=True,
            )
        )
    if atoms.calc is not None:
        losses.append(
            AdapterLoss(
                "calculator",
                "import",
                "unsupported",
                "ASE calculator implementation and cached state were not inspected "
                "or retained.",
                changes_interpretation=True,
            )
        )
    losses.extend(_info_losses(info_names))

    resolved_source_id = _source_id(
        atoms,
        source_id,
        numbers=numbers,
        positions=positions,
        masses=masses,
        cell=cell,
        periodic_axes=periodic_axes,
        particle_ids=particle_ids,
        loss_paths=tuple(loss.path for loss in losses),
    )
    structure = AtomicStructure(
        numbers,
        positions,
        masses,
        scale,
        particle_ids=particle_ids,
        cell=cell,
        periodic_axes=periodic_axes,
        name=name,
    )
    assumptions = [
        "ASE native length and energy units are angstrom and electronvolt.",
        "ASE masses and AtomicStructure masses are interpreted in dalton.",
    ]
    if "masses" not in array_names:
        assumptions.append(
            "ASE elemental default masses were materialized because no explicit "
            "mass array was present."
        )
    if cell is None:
        assumptions.append(
            "ASE's zero cell with no periodic axes was preserved as absent native "
            "periodic metadata."
        )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        "ase.Atoms",
        "AtomicStructure",
        source_id=resolved_source_id,
        target_id=structure.structure_id,
        coordinate_mapping=(
            "positions[:, xyz] -> positions[:, xyz]",
            "cell[vector, xyz] -> cell[vector, xyz]",
            "pbc[axis] -> periodic_axes[axis]",
        ),
        preserved_fields=(
            "atomic_numbers",
            "positions",
            "masses",
            "cell",
            "periodic_axes",
            "particle_ids",
            "source_id",
        ),
        assumptions=assumptions,
        losses=losses,
    )
    return structure, report


def to_ase_atoms(
    structure: AtomicStructure,
    /,
) -> tuple[Any, AdapterReport]:
    """Copy a non-padded native structure into a detached ``ase.Atoms`` value."""

    ase = require_ase()
    if not isinstance(structure, AtomicStructure):
        raise TypeError("structure must be an AtomicStructure.")
    _require_ase_scale(structure.scale)

    active = np.asarray(structure.active_mask, dtype=bool)
    if not np.all(active):
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "ASE export does not discard inactive AtomicStructure padding.",
        )
    numbers = np.array(structure.atomic_numbers, copy=True)
    positions = np.array(structure.positions, copy=True)
    masses = np.array(structure.masses, copy=True)
    particle_ids = np.array(structure.particle_ids, dtype=np.int64, copy=True)
    cell = (
        np.zeros((3, 3), dtype=positions.dtype)
        if structure.cell is None
        else np.array(structure.cell, copy=True)
    )
    periodic_axes = (
        np.zeros((3,), dtype=bool)
        if structure.periodic_axes is None
        else np.array(structure.periodic_axes, dtype=bool, copy=True)
    )
    _validate_cell(cell, periodic_axes)

    atoms = ase.Atoms(
        numbers=numbers,
        positions=positions,
        masses=masses,
        cell=cell,
        pbc=periodic_axes,
        info={ASE_SOURCE_ID_INFO: structure.structure_id},
    )
    atoms.new_array(ASE_PARTICLE_ID_ARRAY, particle_ids)
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "AtomicStructure",
        "ase.Atoms",
        source_id=structure.structure_id,
        target_id=structure.structure_id,
        coordinate_mapping=(
            "positions[:, xyz] -> positions[:, xyz]",
            "cell[vector, xyz] -> cell[vector, xyz]",
            "periodic_axes[axis] -> pbc[axis]",
        ),
        preserved_fields=(
            "atomic_numbers",
            "positions",
            "masses",
            "cell",
            "periodic_axes",
            "particle_ids",
            "structure_id",
        ),
        assumptions=(
            "ASE native length and energy units are angstrom and electronvolt.",
            "ASE masses and AtomicStructure masses are interpreted in dalton.",
        ),
    )
    return atoms, report


def _require_ase_scale(scale: AtomisticScaleContract, /) -> None:
    if not isinstance(scale, AtomisticScaleContract):
        raise TypeError("scale must be an AtomisticScaleContract.")
    if (
        scale.length_unit.strip().lower() != "angstrom"
        or scale.energy_unit.strip().lower() != "electronvolt"
        or scale.length_to_reference != 1.0
        or scale.energy_to_reference != 1.0
    ):
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "ASE interchange requires the exact angstrom/electronvolt scale contract.",
        )


def _string_keys(mapping: Any, owner: str, /) -> frozenset[str]:
    names = tuple(mapping.keys())
    if not all(isinstance(key, str) and key for key in names):
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE,
            f"{owner} names must be non-empty strings.",
        )
    return frozenset(names)


def _reject_required_semantics(
    array_names: frozenset[str], info_names: frozenset[str], /
) -> None:
    normalized_arrays = {name.lower() for name in array_names}
    normalized_info = {name.lower() for name in info_names}
    for label, names in (
        ("partial occupancy or disorder", _OCCUPANCY_NAMES),
        ("atomistic topology", _TOPOLOGY_NAMES),
        ("spin state", _SPIN_NAMES),
    ):
        present = sorted(
            normalized_arrays.intersection(names) | normalized_info.intersection(names)
        )
        if present:
            raise AdapterError(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                f"ASE {label} is not representable by AtomicStructure: "
                + ", ".join(present),
            )
    unit_fields = sorted(normalized_info.intersection(_UNIT_NAMES))
    if unit_fields:
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "ASE uses fixed native units; competing unit metadata is ambiguous: "
            + ", ".join(unit_fields),
        )


def _validated_cell(atoms: Any, /) -> tuple[np.ndarray | None, np.ndarray | None]:
    cell = np.array(atoms.cell.array, copy=True)
    periodic_axes = np.array(atoms.pbc, dtype=bool, copy=True)
    _validate_cell(cell, periodic_axes)
    if not np.any(periodic_axes) and not np.any(cell):
        return None, None
    return cell, periodic_axes


def _validate_cell(cell: np.ndarray, periodic_axes: np.ndarray, /) -> None:
    if cell.shape != (3, 3) or not np.issubdtype(cell.dtype, np.number):
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE,
            "ASE cell must be a numeric (3, 3) array.",
        )
    if np.any(~np.isfinite(cell)):
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE,
            "ASE cell entries must be finite.",
        )
    if periodic_axes.shape != (3,):
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE,
            "ASE pbc must have shape (3,).",
        )
    periodic_count = int(np.count_nonzero(periodic_axes))
    if periodic_count and np.linalg.matrix_rank(cell[periodic_axes]) != periodic_count:
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE,
            "ASE periodic cell vectors must be nonzero and linearly independent.",
        )


def _validate_atoms_arrays(
    numbers: np.ndarray, positions: np.ndarray, masses: np.ndarray, /
) -> None:
    if numbers.ndim != 1 or numbers.size == 0:
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE,
            "ASE atomic numbers must be a non-empty rank-1 array.",
        )
    if not np.issubdtype(numbers.dtype, np.integer) or np.any(numbers <= 0):
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "AtomicStructure requires positive integer atomic numbers; ASE dummy "
            "atoms are unsupported.",
        )
    if positions.shape != (numbers.size, 3) or np.any(~np.isfinite(positions)):
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE,
            "ASE positions must be a finite (atom_count, 3) array.",
        )
    if (
        masses.shape != numbers.shape
        or np.any(~np.isfinite(masses))
        or np.any(masses <= 0)
    ):
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE,
            "ASE masses must be finite and positive with shape (atom_count,).",
        )


def _validate_particle_ids(particle_ids: np.ndarray, count: int, /) -> None:
    if particle_ids.shape != (count,) or not np.issubdtype(
        particle_ids.dtype, np.integer
    ):
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE,
            f"ASE {ASE_PARTICLE_ID_ARRAY!r} must contain one integer per atom.",
        )
    int64 = np.iinfo(np.int64)
    if (
        int(np.min(particle_ids)) < int64.min
        or int(np.max(particle_ids)) > int64.max
        or np.unique(particle_ids).size != count
    ):
        raise AdapterError(
            AdapterStatus.MALFORMED_SOURCE,
            f"ASE {ASE_PARTICLE_ID_ARRAY!r} values must be unique signed 64-bit "
            "integers.",
        )


def _array_losses(array_names: frozenset[str], /) -> list[AdapterLoss]:
    losses = []
    for array_name in sorted(array_names - _STANDARD_ARRAYS - {ASE_PARTICLE_ID_ARRAY}):
        normalized = array_name.lower()
        if normalized in ("momenta", "velocities"):
            rationale = (
                "AtomicStructure has no velocity or momentum state; the ASE array "
                "was not retained."
            )
            changes_interpretation = False
        elif normalized in ("charge", "charges", "initial_charges"):
            rationale = (
                "AtomicStructure has no per-atom charge field; the ASE charge array "
                "was not retained."
            )
            changes_interpretation = True
        else:
            rationale = (
                "AtomicStructure has no declared field for this ASE array; it was "
                "not retained."
            )
            changes_interpretation = True
        losses.append(
            AdapterLoss(
                f"arrays.{array_name}",
                "import",
                "dropped",
                rationale,
                changes_interpretation=changes_interpretation,
            )
        )
    return losses


def _info_losses(info_names: frozenset[str], /) -> list[AdapterLoss]:
    return [
        AdapterLoss(
            f"info.{name}",
            "import",
            "dropped",
            "AtomicStructure has no field for arbitrary ASE info metadata; the "
            "value was not inspected or retained.",
            changes_interpretation=True,
        )
        for name in sorted(info_names - {ASE_SOURCE_ID_INFO})
    ]


def _source_id(
    atoms: Any,
    explicit_source_id: str | None,
    /,
    *,
    numbers: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    cell: np.ndarray | None,
    periodic_axes: np.ndarray | None,
    particle_ids: np.ndarray | None,
    loss_paths: tuple[str, ...],
) -> str:
    embedded_source_id: str | None = None
    if ASE_SOURCE_ID_INFO in atoms.info:
        embedded = atoms.info[ASE_SOURCE_ID_INFO]
        if not isinstance(embedded, str) or not embedded.strip():
            raise AdapterError(
                AdapterStatus.MALFORMED_SOURCE,
                f"ASE info {ASE_SOURCE_ID_INFO!r} must be a non-empty string.",
            )
        embedded_source_id = embedded.strip()
    supplied_source_id = None
    if explicit_source_id is not None:
        if not isinstance(explicit_source_id, str):
            raise TypeError("source_id must be a string when provided.")
        supplied_source_id = explicit_source_id.strip()
        if not supplied_source_id:
            raise ValueError("source_id must be non-empty when provided.")
    if (
        supplied_source_id is not None
        and embedded_source_id is not None
        and supplied_source_id != embedded_source_id
    ):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "Explicit source_id conflicts with ASE embedded source provenance.",
        )
    if supplied_source_id is not None:
        return supplied_source_id
    if embedded_source_id is not None:
        return embedded_source_id
    return canonical_fingerprint(
        {
            "kind": "ase-atoms-source",
            "arrays": array_tree_fingerprint(
                {
                    "atomic_numbers": numbers,
                    "positions": positions,
                    "masses": masses,
                    "cell": cell,
                    "periodic_axes": periodic_axes,
                    "particle_ids": particle_ids,
                }
            ),
            "explicit_particle_ids": particle_ids is not None,
            "loss_paths": list(loss_paths),
        }
    )


__all__ = [
    "ASE_PARTICLE_ID_ARRAY",
    "ASE_SOURCE_ID_INFO",
    "from_ase_atoms",
    "is_ase_available",
    "require_ase",
    "to_ase_atoms",
]
