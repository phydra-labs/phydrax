#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class MacromolecularStructure(StrictModule, NonTrainableState):
    """Fixed numeric macromolecular topology with model-indexed coordinates."""

    atomic_numbers: Array
    formal_charges: Array
    positions: Array
    occupancies: Array
    b_factors: Array
    present_mask: Array
    model_numbers: Array
    atom_to_residue: Array
    atom_name_codes: Array
    atom_altloc_choice: Array
    residue_to_chain: Array
    residue_component_codes: Array
    residue_anchor_atoms: Array
    chain_to_entity: Array
    bond_indices: Array
    bond_orders: Array
    bond_aromatic: Array
    connection_kinds: Array
    altloc_choice_residue: Array
    assembly_ids: Array
    assembly_operation_indices: Array
    assembly_chain_indices: Array
    assembly_rotations: Array
    assembly_translations: Array
    missing_residue_chain_indices: Array
    missing_residue_label_seq_ids: Array
    missing_residue_auth_seq_ids: Array
    missing_residue_model_numbers: Array
    missing_atom_residue_indices: Array
    missing_atom_name_codes: Array
    missing_atom_model_numbers: Array
    atom_capacity: int = eqx.field(static=True)
    residue_capacity: int = eqx.field(static=True)
    chain_capacity: int = eqx.field(static=True)
    model_capacity: int = eqx.field(static=True)
    assembly_application_capacity: int = eqx.field(static=True)
    length_unit: str = eqx.field(static=True)
    source_record_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        atomic_numbers: ArrayLike,
        positions: ArrayLike,
        present_mask: ArrayLike,
        atom_to_residue: ArrayLike,
        residue_to_chain: ArrayLike,
        chain_to_entity: ArrayLike,
        /,
        *,
        formal_charges: ArrayLike | None = None,
        occupancies: ArrayLike | None = None,
        b_factors: ArrayLike | None = None,
        model_numbers: ArrayLike | None = None,
        atom_name_codes: ArrayLike | None = None,
        atom_altloc_choice: ArrayLike | None = None,
        residue_component_codes: ArrayLike | None = None,
        residue_anchor_atoms: ArrayLike | None = None,
        bond_indices: ArrayLike | None = None,
        bond_orders: ArrayLike | None = None,
        bond_aromatic: ArrayLike | None = None,
        connection_kinds: ArrayLike | None = None,
        altloc_choice_residue: ArrayLike | None = None,
        assembly_ids: ArrayLike | None = None,
        assembly_operation_indices: ArrayLike | None = None,
        assembly_chain_indices: ArrayLike | None = None,
        assembly_rotations: ArrayLike | None = None,
        assembly_translations: ArrayLike | None = None,
        missing_residue_chain_indices: ArrayLike | None = None,
        missing_residue_label_seq_ids: ArrayLike | None = None,
        missing_residue_auth_seq_ids: ArrayLike | None = None,
        missing_residue_model_numbers: ArrayLike | None = None,
        missing_atom_residue_indices: ArrayLike | None = None,
        missing_atom_name_codes: ArrayLike | None = None,
        missing_atom_model_numbers: ArrayLike | None = None,
        length_unit: str = "angstrom",
        source_record_id: str,
    ):
        numbers = np.asarray(atomic_numbers)
        position = np.asarray(positions)
        present = np.asarray(present_mask, dtype=bool)
        atom_residue = np.asarray(atom_to_residue)
        residue_chain = np.asarray(residue_to_chain)
        chain_entity = np.asarray(chain_to_entity)
        if (
            numbers.ndim != 1
            or numbers.size == 0
            or not np.issubdtype(numbers.dtype, np.integer)
        ):
            raise TypeError("atomic_numbers must be a non-empty integer vector.")
        atom_count = int(numbers.size)
        if position.ndim != 3 or position.shape[1:] != (atom_count, 3):
            raise ValueError(
                "positions must have shape (model_capacity, atom_capacity, 3)."
            )
        model_count = int(position.shape[0])
        if model_count == 0 or present.shape != (model_count, atom_count):
            raise ValueError("present_mask must match the model and atom capacities.")
        if not np.issubdtype(position.dtype, np.inexact):
            position = position.astype(np.float64)
        if np.any(~np.isfinite(position[present])):
            raise ValueError("Present atom coordinates must be finite.")
        if atom_residue.shape != (atom_count,) or not np.issubdtype(
            atom_residue.dtype, np.integer
        ):
            raise TypeError("atom_to_residue must be an integer atom vector.")
        if residue_chain.ndim != 1 or not np.issubdtype(residue_chain.dtype, np.integer):
            raise TypeError("residue_to_chain must be an integer vector.")
        residue_count = int(residue_chain.size)
        if (
            residue_count == 0
            or np.any(atom_residue < 0)
            or np.any(atom_residue >= residue_count)
        ):
            raise ValueError("atom_to_residue contains an out-of-range residue.")
        if chain_entity.ndim != 1 or not np.issubdtype(chain_entity.dtype, np.integer):
            raise TypeError("chain_to_entity must be an integer vector.")
        chain_count = int(chain_entity.size)
        if (
            chain_count == 0
            or np.any(residue_chain < 0)
            or np.any(residue_chain >= chain_count)
        ):
            raise ValueError("residue_to_chain contains an out-of-range chain.")
        if np.any(chain_entity < 0):
            raise ValueError("chain_to_entity must contain non-negative entity codes.")
        numbers = numbers.astype(np.int32, copy=False)
        if np.any(numbers <= 0):
            raise ValueError("Compiled atoms require resolved positive atomic numbers.")

        def vector(
            name: str, value: ArrayLike | None, length: int, dtype: Any, default: Any
        ) -> np.ndarray:
            result = (
                np.full((length,), default, dtype=dtype)
                if value is None
                else np.asarray(value, dtype=dtype)
            )
            if result.shape != (length,):
                raise ValueError(f"{name} must have shape ({length},).")
            return result

        charges = vector("formal_charges", formal_charges, atom_count, np.int32, 0)
        occupancy = (
            np.where(present, 1.0, 0.0).astype(position.dtype)
            if occupancies is None
            else np.asarray(occupancies, dtype=position.dtype)
        )
        bfactor = (
            np.zeros((model_count, atom_count), dtype=position.dtype)
            if b_factors is None
            else np.asarray(b_factors, dtype=position.dtype)
        )
        if occupancy.shape != present.shape or bfactor.shape != present.shape:
            raise ValueError("occupancies and b_factors must match present_mask.")
        if np.any(~np.isfinite(occupancy)) or np.any(
            (occupancy < 0.0) | (occupancy > 1.0)
        ):
            raise ValueError("occupancies must be finite and lie in [0, 1].")
        if np.any(~np.isfinite(bfactor)) or np.any(bfactor < 0.0):
            raise ValueError("b_factors must be finite and non-negative.")
        models = vector("model_numbers", model_numbers, model_count, np.int32, 0)
        if model_numbers is None:
            models = np.arange(1, model_count + 1, dtype=np.int32)
        if np.any(models <= 0) or np.unique(models).size != model_count:
            raise ValueError("model_numbers must be distinct and positive.")
        atom_names = vector("atom_name_codes", atom_name_codes, atom_count, np.int32, 0)
        alt_choice = vector(
            "atom_altloc_choice", atom_altloc_choice, atom_count, np.int32, 0
        )
        component = vector(
            "residue_component_codes", residue_component_codes, residue_count, np.int32, 0
        )
        anchor = vector(
            "residue_anchor_atoms", residue_anchor_atoms, residue_count, np.int32, -1
        )
        if np.any((anchor < -1) | (anchor >= atom_count)):
            raise ValueError("residue_anchor_atoms contains an out-of-range atom.")
        if np.any(
            (anchor >= 0)
            & (atom_residue[np.maximum(anchor, 0)] != np.arange(residue_count))
        ):
            raise ValueError("Every residue anchor must belong to its residue.")

        bonds = (
            np.zeros((0, 2), dtype=np.int32)
            if bond_indices is None
            else np.asarray(bond_indices)
        )
        if (
            bonds.ndim != 2
            or bonds.shape[1] != 2
            or not np.issubdtype(bonds.dtype, np.integer)
        ):
            raise TypeError(
                "bond_indices must have shape (bond_count, 2) and integer dtype."
            )
        bonds = bonds.astype(np.int32, copy=False)
        if bonds.size and (
            np.any(bonds < 0)
            or np.any(bonds >= atom_count)
            or np.any(bonds[:, 0] == bonds[:, 1])
        ):
            raise ValueError("bond_indices contains an invalid endpoint.")
        canonical_bonds = np.sort(bonds, axis=1)
        if canonical_bonds.shape[0] != np.unique(canonical_bonds, axis=0).shape[0]:
            raise ValueError("bond_indices contains duplicate unordered pairs.")
        bond_count = int(bonds.shape[0])
        orders = vector("bond_orders", bond_orders, bond_count, np.int32, 0)
        aromatic = vector("bond_aromatic", bond_aromatic, bond_count, bool, False)
        connections = vector(
            "connection_kinds", connection_kinds, bond_count, np.int32, 0
        )

        choice_residue = (
            np.zeros((0,), dtype=np.int32)
            if altloc_choice_residue is None
            else np.asarray(altloc_choice_residue)
        )
        if choice_residue.ndim != 1 or not np.issubdtype(
            choice_residue.dtype, np.integer
        ):
            raise TypeError("altloc_choice_residue must be an integer vector.")
        choice_residue = choice_residue.astype(np.int32, copy=False)
        if np.any(choice_residue < 0) or np.any(choice_residue >= residue_count):
            raise ValueError("altloc_choice_residue contains an out-of-range residue.")
        if np.any(alt_choice < 0) or np.any(alt_choice > choice_residue.size):
            raise ValueError("atom_altloc_choice contains an out-of-range choice code.")
        nonzero = alt_choice > 0
        if np.any(nonzero) and np.any(
            choice_residue[alt_choice[nonzero] - 1] != atom_residue[nonzero]
        ):
            raise ValueError(
                "Alternate-location choices must belong to each atom's residue."
            )

        rotations = (
            np.zeros((0, 3, 3), dtype=position.dtype)
            if assembly_rotations is None
            else np.asarray(assembly_rotations, dtype=position.dtype)
        )
        translations = (
            np.zeros((rotations.shape[0], 3), dtype=position.dtype)
            if assembly_translations is None
            else np.asarray(assembly_translations, dtype=position.dtype)
        )
        if (
            rotations.ndim != 3
            or rotations.shape[1:] != (3, 3)
            or translations.shape != (rotations.shape[0], 3)
        ):
            raise ValueError(
                "Assembly rotations/translations must have shapes (O,3,3)/(O,3)."
            )
        operation_count = int(rotations.shape[0])
        assembly_id = (
            np.zeros((0,), dtype=np.int32)
            if assembly_ids is None
            else np.asarray(assembly_ids)
        )
        operation_index = (
            np.zeros((0,), dtype=np.int32)
            if assembly_operation_indices is None
            else np.asarray(assembly_operation_indices)
        )
        assembly_chain = (
            np.zeros((0,), dtype=np.int32)
            if assembly_chain_indices is None
            else np.asarray(assembly_chain_indices)
        )
        if not (
            assembly_id.ndim == operation_index.ndim == assembly_chain.ndim == 1
            and assembly_id.shape == operation_index.shape == assembly_chain.shape
        ):
            raise ValueError(
                "Assembly application vectors must be aligned rank-1 arrays."
            )
        if not all(
            np.issubdtype(value.dtype, np.integer)
            for value in (assembly_id, operation_index, assembly_chain)
        ):
            raise TypeError("Assembly application vectors must contain integers.")
        if operation_index.size and (
            np.any(operation_index < 0) or np.any(operation_index >= operation_count)
        ):
            raise ValueError(
                "assembly_operation_indices contains an out-of-range operation."
            )
        if assembly_chain.size and (
            np.any(assembly_chain < 0) or np.any(assembly_chain >= chain_count)
        ):
            raise ValueError("assembly_chain_indices contains an out-of-range chain.")

        def aligned_optional(
            prefix: str, values: tuple[ArrayLike | None, ...], dtypes: tuple[Any, ...]
        ) -> tuple[np.ndarray, ...]:
            supplied = [
                np.asarray(value, dtype=dtype)
                for value, dtype in zip(values, dtypes, strict=True)
                if value is not None
            ]
            length = int(supplied[0].size) if supplied else 0
            result = tuple(
                np.zeros((length,), dtype=dtype)
                if value is None
                else np.asarray(value, dtype=dtype)
                for value, dtype in zip(values, dtypes, strict=True)
            )
            if any(array.shape != (length,) for array in result):
                raise ValueError(f"{prefix} arrays must be aligned rank-1 vectors.")
            return result

        missing_residue = aligned_optional(
            "missing residue",
            (
                missing_residue_chain_indices,
                missing_residue_label_seq_ids,
                missing_residue_auth_seq_ids,
                missing_residue_model_numbers,
            ),
            (np.int32, np.int32, np.int32, np.int32),
        )
        missing_atom = aligned_optional(
            "missing atom",
            (
                missing_atom_residue_indices,
                missing_atom_name_codes,
                missing_atom_model_numbers,
            ),
            (np.int32, np.int32, np.int32),
        )
        if missing_residue[0].size and (
            np.any(missing_residue[0] < 0) or np.any(missing_residue[0] >= chain_count)
        ):
            raise ValueError("Missing residues contain an out-of-range chain.")
        if missing_atom[0].size and (
            np.any(missing_atom[0] < 0) or np.any(missing_atom[0] >= residue_count)
        ):
            raise ValueError("Missing atoms contain an out-of-range residue.")

        arrays = {
            "atomic_numbers": numbers,
            "formal_charges": charges,
            "positions": position,
            "occupancies": occupancy,
            "b_factors": bfactor,
            "present_mask": present,
            "model_numbers": models,
            "atom_to_residue": atom_residue.astype(np.int32, copy=False),
            "atom_name_codes": atom_names,
            "atom_altloc_choice": alt_choice,
            "residue_to_chain": residue_chain.astype(np.int32, copy=False),
            "residue_component_codes": component,
            "residue_anchor_atoms": anchor,
            "chain_to_entity": chain_entity.astype(np.int32, copy=False),
            "bond_indices": canonical_bonds,
            "bond_orders": orders,
            "bond_aromatic": aromatic,
            "connection_kinds": connections,
            "altloc_choice_residue": choice_residue,
            "assembly_ids": assembly_id.astype(np.int32, copy=False),
            "assembly_operation_indices": operation_index.astype(np.int32, copy=False),
            "assembly_chain_indices": assembly_chain.astype(np.int32, copy=False),
            "assembly_rotations": rotations,
            "assembly_translations": translations,
            "missing_residue_chain_indices": missing_residue[0],
            "missing_residue_label_seq_ids": missing_residue[1],
            "missing_residue_auth_seq_ids": missing_residue[2],
            "missing_residue_model_numbers": missing_residue[3],
            "missing_atom_residue_indices": missing_atom[0],
            "missing_atom_name_codes": missing_atom[1],
            "missing_atom_model_numbers": missing_atom[2],
        }
        unit = str(length_unit).strip()
        source = str(source_record_id).strip()
        if not unit or not source:
            raise ValueError("length_unit and source_record_id must be non-empty.")
        for name, value in arrays.items():
            setattr(self, name, jnp.asarray(value))
        self.atom_capacity = atom_count
        self.residue_capacity = residue_count
        self.chain_capacity = chain_count
        self.model_capacity = model_count
        self.assembly_application_capacity = int(assembly_id.size)
        self.length_unit = unit
        self.source_record_id = source
        self.structure_id = canonical_fingerprint(
            {
                "kind": "compiled-macromolecular-structure",
                "source": source,
                "length_unit": unit,
                "arrays": array_tree_fingerprint(arrays),
            }
        )

    def model_index(self, model_number: int) -> int:
        """Resolve a model number outside compiled execution."""

        matches = np.flatnonzero(np.asarray(self.model_numbers) == int(model_number))
        if matches.size != 1:
            raise KeyError(f"Unknown model number {model_number}.")
        return int(matches[0])

    def altloc_mask(self, model_index: int = 0) -> Array:
        """Select one occupancy-coupled conformer per residue plus shared atoms."""

        if model_index < 0 or model_index >= self.model_capacity:
            raise IndexError("model_index is outside the compiled model capacity.")
        present = self.present_mask[model_index]
        choices = self.atom_altloc_choice
        choice_count = int(self.altloc_choice_residue.shape[0])
        if choice_count == 0:
            return present
        choice_index = jnp.maximum(choices - 1, 0)
        weighted = jnp.where((choices > 0) & present, self.occupancies[model_index], 0.0)
        counts = jax.ops.segment_sum(
            jnp.where((choices > 0) & present, 1.0, 0.0),
            choice_index,
            num_segments=choice_count,
        )
        totals = jax.ops.segment_sum(weighted, choice_index, num_segments=choice_count)
        scores = jnp.where(counts > 0.0, totals / counts, -jnp.inf)
        residue_best = jax.ops.segment_max(
            scores,
            self.altloc_choice_residue,
            num_segments=self.residue_capacity,
        )
        tied = scores == residue_best[self.altloc_choice_residue]
        sentinel = jnp.asarray(choice_count, dtype=jnp.int32)
        best_choice = jax.ops.segment_min(
            jnp.where(tied, jnp.arange(choice_count, dtype=jnp.int32), sentinel),
            self.altloc_choice_residue,
            num_segments=self.residue_capacity,
        )
        selected = best_choice[self.atom_to_residue] + 1
        return present & ((choices == 0) | (choices == selected))

    def assembly_application(
        self, application_index: int, model_index: int = 0
    ) -> tuple[Array, Array]:
        """Return transformed coordinates and atom mask for one assembly application."""

        if (
            application_index < 0
            or application_index >= self.assembly_application_capacity
        ):
            raise IndexError("application_index is outside the assembly capacity.")
        if model_index < 0 or model_index >= self.model_capacity:
            raise IndexError("model_index is outside the model capacity.")
        operation = self.assembly_operation_indices[application_index]
        chain = self.assembly_chain_indices[application_index]
        rotation = self.assembly_rotations[operation]
        translation = self.assembly_translations[operation]
        transformed = self.positions[model_index] @ rotation.T + translation
        atom_chains = self.residue_to_chain[self.atom_to_residue]
        mask = self.altloc_mask(model_index) & (atom_chains == chain)
        return transformed, mask


__all__ = ["MacromolecularStructure"]
