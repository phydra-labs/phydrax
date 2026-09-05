# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Explicit biological atom identities compiled into an existing numeric support."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import PreparedAtomisticSystem
from ._construct import NucleicAcidConstruct, NucleotideKey


@dataclass(frozen=True, slots=True)
class NucleotideAtomMapping:
    construct: NucleicAcidConstruct
    atom_ids: tuple[int, ...]
    nucleotide_keys: tuple[NucleotideKey, ...]
    atom_names: tuple[str, ...]

    def __post_init__(self):
        if any(
            not isinstance(x, tuple)
            for x in (self.atom_ids, self.nucleotide_keys, self.atom_names)
        ):
            raise TypeError("Atom mapping columns must be tuples.")
        if (
            not self.atom_ids
            or len(self.atom_ids) != len(self.nucleotide_keys)
            or len(self.atom_ids) != len(self.atom_names)
        ):
            raise ValueError("Atom mapping columns must be nonempty and aligned.")
        if len(set(self.atom_ids)) != len(self.atom_ids) or any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < 2**63
            for i in self.atom_ids
        ):
            raise ValueError("Atom IDs must be unique nonnegative int64 values.")
        keys = set(self.construct.nucleotide_keys)
        if any(key not in keys for key in self.nucleotide_keys):
            raise ValueError("Atom mapping contains a nucleotide outside the construct.")
        if any(not name or name != name.strip() for name in self.atom_names):
            raise ValueError(
                "Atom names must be explicit canonical source-resolved names."
            )
        if len(set(zip(self.nucleotide_keys, self.atom_names, strict=True))) != len(
            self.atom_ids
        ):
            raise ValueError(
                "Alternate atom locations must be selected explicitly before binding."
            )

    def fingerprint(self) -> str:
        rows = sorted(
            (i, key.strand_id, key.position, name)
            for i, key, name in zip(
                self.atom_ids, self.nucleotide_keys, self.atom_names, strict=True
            )
        )
        return canonical_fingerprint(
            {"construct": self.construct.fingerprint(), "atoms": rows}
        )


class PreparedNucleotideBinding(StrictModule, NonTrainableState):
    construct: NucleicAcidConstruct = eqx.field(static=True)
    mapping: NucleotideAtomMapping = eqx.field(static=True)
    support_atom_ids: Array
    atom_indices: Array
    atom_mask: Array
    ring_indices: Array
    ring_mask: Array
    binding_id: str = eqx.field(static=True)
    support_size: int = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)


def prepare_nucleotide_binding(
    mapping, support_atom_ids, *, coordinate_mask=None
) -> PreparedNucleotideBinding:
    """Prepare without relabeling IDs; missing coordinates are not inactive padding.

    support_atom_ids may be a PreparedAtomisticSystem; mapped atoms must belong to
    active chemical material. coordinate_mask is independent coordinate coverage.
    """
    if not isinstance(mapping, NucleotideAtomMapping):
        raise TypeError("mapping must be NucleotideAtomMapping.")
    system = (
        support_atom_ids
        if isinstance(support_atom_ids, PreparedAtomisticSystem)
        else None
    )
    ids = np.asarray(system.plan.particle_ids if system is not None else support_atom_ids)
    if (
        ids.ndim != 1
        or not np.issubdtype(ids.dtype, np.integer)
        or len(set(ids.tolist())) != ids.size
        or np.any(ids < 0)
    ):
        raise ValueError("Support atom IDs must be unique nonnegative rank-one integers.")
    if ids.size == 0:
        raise ValueError("Coordinate support cannot be empty.")
    mask = (
        np.ones(ids.size, bool)
        if coordinate_mask is None
        else np.asarray(coordinate_mask, bool)
    )
    if mask.shape != ids.shape:
        raise ValueError("Coordinate coverage must align with the existing atom support.")
    lookup = {int(atom): i for i, atom in enumerate(ids)}
    if any(atom not in lookup for atom in mapping.atom_ids):
        raise ValueError("Mapped chemically present atoms cannot disappear as padding.")
    indices = np.array([lookup[atom] for atom in mapping.atom_ids], dtype=np.int64)
    if system is not None and not np.all(np.asarray(system.active_mask)[indices]):
        raise ValueError(
            "Mapped atoms must belong to active chemical support, not padding."
        )
    atom_lookup = {
        (key, name): row
        for key, name, row in zip(
            mapping.nucleotide_keys, mapping.atom_names, indices, strict=True
        )
    }
    ring, covered = [], []
    for key, base in zip(
        mapping.construct.nucleotide_keys, mapping.construct.bases, strict=True
    ):
        # Published purine/pyrimidine handedness, not source row order.
        names = ("C2", "C6", "C4") if base in "AG" else ("C2", "C4", "C6")
        rows = [atom_lookup.get((key, name), -1) for name in names]
        ring.append([max(row, 0) for row in rows])
        covered.append([row >= 0 and mask[row] for row in rows])
    periodic = system is not None and system.cell is not None
    identity = canonical_fingerprint(
        {
            "mapping": mapping.fingerprint(),
            "support": ids.tolist(),
            "coverage": mask.tolist(),
            "periodic": periodic,
        }
    )
    return PreparedNucleotideBinding(
        mapping.construct,
        mapping,
        jnp.asarray(ids, dtype=jnp.int64),
        jnp.asarray(indices),
        jnp.asarray(mask[indices]),
        jnp.asarray(ring, dtype=jnp.int64),
        jnp.asarray(covered),
        identity,
        ids.size,
        periodic,
    )


__all__ = [
    "NucleotideAtomMapping",
    "PreparedNucleotideBinding",
    "prepare_nucleotide_binding",
]
