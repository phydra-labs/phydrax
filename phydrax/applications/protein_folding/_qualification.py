# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...atomistic.sampling import CollectiveVariableKind, CollectiveVariablePlan
from ...units import ANGSTROM, conversion_factor
from ._binding import PreparedProteinBinding
from ._construct import ProteinAtomKey


@dataclass(frozen=True, slots=True)
class ProteinTorsionCriterion:
    """Caller-declared periodic torsion support in radians."""

    atom_keys: tuple[ProteinAtomKey, ProteinAtomKey, ProteinAtomKey, ProteinAtomKey]
    lower: float
    upper: float
    criterion_id: str

    def __post_init__(self):
        if (
            not isinstance(self.atom_keys, tuple)
            or len(self.atom_keys) != 4
            or len(set(self.atom_keys)) != 4
            or any(not isinstance(key, ProteinAtomKey) for key in self.atom_keys)
        ):
            raise ValueError("A torsion criterion requires four distinct protein atoms.")
        if (
            not self.criterion_id
            or not np.isfinite((self.lower, self.upper)).all()
            or not -np.pi <= self.lower <= np.pi
            or not -np.pi <= self.upper <= np.pi
            or self.lower == self.upper
        ):
            raise ValueError(
                "Torsion support requires an identity and distinct bounds in [-pi, pi]."
            )


class ProteinGeometryEvidence(StrictModule):
    bond_lengths: Array
    covalent_valid: Array
    chiral_volumes: Array
    chirality_valid: Array
    clash_distances: Array
    clash_free: Array
    peptide_angles: Array
    peptide_planar: Array
    torsion_angles: Array
    torsion_valid: Array
    finite: Array
    successful: Array
    qualification_id: str = eqx.field(static=True)


class PreparedProteinQualification(StrictModule):
    """Fixed-support geometry evidence, not an experimental quality score."""

    bond_indices: Array
    bond_lower: Array
    bond_upper: Array
    chirality_indices: Array
    clash_indices: Array
    peptide_variables: tuple
    torsion_variables: tuple
    torsion_lower: Array
    torsion_upper: Array
    torsion_wraps: Array
    torsion_criterion_ids: tuple[str, ...] = eqx.field(static=True)
    active_indices: Array
    minimum_chiral_volume: float = eqx.field(static=True)
    clash_distance: float = eqx.field(static=True)
    peptide_tolerance: float = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)

    def __init__(
        self,
        binding: PreparedProteinBinding,
        *,
        bond_bounds,
        bounds_unit=ANGSTROM,
        clash_distance=0.8,
        minimum_chiral_volume=0.1,
        peptide_tolerance=0.35,
        maximum_clash_pairs=100_000,
        torsion_criteria=(),
    ):
        """Prepare explicit per-native-bond bounds and conservative clash screening.

        ``bond_bounds`` is (native bond count, 2), in ``bounds_unit``. The
        chirality threshold is in its cubic length and clash threshold in its
        length. Peptide tolerance is radians to either cis or trans planarity.
        Side-chain/torsion support is evaluated only when explicit periodic
        ``torsion_criteria`` are supplied; it is never inferred from a template.
        """
        system = binding.force_field.system
        factor = float(
            conversion_factor(bounds_unit, system.plan.units.scale.length_unit)
        )
        bounds = np.asarray(bond_bounds, dtype=float)
        bonds = np.asarray(system.topology.bond_indices)
        if (
            bounds.shape != (len(bonds), 2)
            or not np.all(np.isfinite(bounds))
            or np.any(bounds[:, 0] <= 0)
            or np.any(bounds[:, 1] <= bounds[:, 0])
        ):
            raise ValueError(
                "Every native covalent bond needs finite positive ordered geometry bounds."
            )
        if (
            not np.isfinite(clash_distance)
            or clash_distance <= 0
            or not np.isfinite(minimum_chiral_volume)
            or minimum_chiral_volume <= 0
            or not 0 < peptide_tolerance < np.pi / 2
        ):
            raise ValueError(
                "Geometry thresholds must be positive and peptide tolerance below pi/2."
            )
        index = dict(zip(binding.atom_keys, binding.atom_indices, strict=True))
        residues = binding.chemistry.construct.residue_keys
        chiral = []
        for residue, letter in zip(
            residues, binding.chemistry.construct.sequences[0], strict=True
        ):
            if letter != "G":
                chiral.append(
                    [
                        index[ProteinAtomKey(residue, atom)]
                        for atom in ("CA", "N", "C", "CB")
                    ]
                )
        peptides = tuple(
            CollectiveVariablePlan(
                CollectiveVariableKind.TORSION,
                [
                    index[ProteinAtomKey(left, "CA")],
                    index[ProteinAtomKey(left, "C")],
                    index[ProteinAtomKey(right, "N")],
                    index[ProteinAtomKey(right, "CA")],
                ],
            ).prepare(system)
            for left, right in zip(residues[:-1], residues[1:], strict=True)
        )
        criteria = tuple(torsion_criteria)
        if any(not isinstance(value, ProteinTorsionCriterion) for value in criteria):
            raise TypeError(
                "torsion_criteria must contain explicit ProteinTorsionCriterion records."
            )
        if len({value.criterion_id for value in criteria}) != len(criteria):
            raise ValueError("Protein torsion criterion identities must be unique.")
        torsions = tuple(
            CollectiveVariablePlan(
                CollectiveVariableKind.TORSION,
                [index[key] for key in criterion.atom_keys],
            ).prepare(system)
            for criterion in criteria
        )
        adjacency = {i: set() for i in binding.atom_indices}
        for a, b in bonds:
            adjacency[int(a)].add(int(b))
            adjacency[int(b)].add(int(a))
        pairs = []
        for a, b in combinations(binding.atom_indices, 2):
            if b not in adjacency[a] and not adjacency[a].intersection(adjacency[b]):
                pairs.append((a, b))
                if len(pairs) > maximum_clash_pairs:
                    raise ValueError(
                        "Clash-pair capacity exceeded; prepare a smaller explicit qualification support."
                    )
        self.bond_indices = jnp.asarray(bonds, dtype=jnp.int32)
        self.bond_lower = jnp.asarray(bounds[:, 0] * factor)
        self.bond_upper = jnp.asarray(bounds[:, 1] * factor)
        self.chirality_indices = jnp.asarray(
            np.asarray(chiral, dtype=np.int32).reshape((-1, 4))
        )
        self.clash_indices = jnp.asarray(
            np.asarray(pairs, dtype=np.int32).reshape((-1, 2))
        )
        self.peptide_variables = peptides
        self.torsion_variables = torsions
        self.torsion_lower = jnp.asarray(
            tuple(value.lower for value in criteria),
            dtype=binding.realized_positions.dtype,
        )
        self.torsion_upper = jnp.asarray(
            tuple(value.upper for value in criteria),
            dtype=binding.realized_positions.dtype,
        )
        self.torsion_wraps = self.torsion_lower > self.torsion_upper
        self.torsion_criterion_ids = tuple(value.criterion_id for value in criteria)
        self.active_indices = jnp.asarray(binding.atom_indices, dtype=jnp.int32)
        self.minimum_chiral_volume = float(minimum_chiral_volume * factor**3)
        self.clash_distance = float(clash_distance * factor)
        self.peptide_tolerance = float(peptide_tolerance)
        self.binding_id = binding.binding_id
        self.qualification_id = canonical_fingerprint(
            {
                "kind": "protein-geometry-qualification",
                "binding": binding.binding_id,
                "bounds": bounds.tolist(),
                "unit": bounds_unit.unit_id,
                "clash": clash_distance,
                "chirality": minimum_chiral_volume,
                "peptide": peptide_tolerance,
                "torsions": [
                    {
                        "id": value.criterion_id,
                        "atoms": [key.record() for key in value.atom_keys],
                        "bounds": (value.lower, value.upper),
                    }
                    for value in criteria
                ],
            }
        )

    def evaluate(self, positions):
        x = jnp.asarray(positions)
        bonds = x[self.bond_indices[:, 0]] - x[self.bond_indices[:, 1]]
        lengths = jnp.sqrt(jnp.sum(bonds * bonds, axis=-1))
        covalent = (lengths >= self.bond_lower) & (lengths <= self.bond_upper)
        center = x[self.chirality_indices[:, 0]]
        n, c, cb = (x[self.chirality_indices[:, i]] - center for i in (1, 2, 3))
        volumes = jnp.sum(jnp.cross(n, c) * cb, axis=-1)
        chirality = volumes > self.minimum_chiral_volume
        delta = x[self.clash_indices[:, 0]] - x[self.clash_indices[:, 1]]
        distances = jnp.sqrt(jnp.sum(delta * delta, axis=-1))
        clash_free = distances >= self.clash_distance
        peptide = tuple(variable.evaluate(x) for variable in self.peptide_variables)
        angles = (
            jnp.stack(tuple(value.value for value in peptide))
            if peptide
            else jnp.zeros((0,), dtype=x.dtype)
        )
        peptide_valid = (
            jnp.stack(tuple(value.successful for value in peptide))
            if peptide
            else jnp.ones((0,), dtype=bool)
        )
        planar = peptide_valid & (
            jnp.abs(jnp.sin(angles)) <= jnp.sin(self.peptide_tolerance)
        )
        torsion = tuple(variable.evaluate(x) for variable in self.torsion_variables)
        torsion_angles = (
            jnp.stack(tuple(value.value for value in torsion))
            if torsion
            else jnp.zeros((0,), dtype=x.dtype)
        )
        torsion_numeric = (
            jnp.stack(tuple(value.successful for value in torsion))
            if torsion
            else jnp.ones((0,), dtype=bool)
        )
        torsion_valid = torsion_numeric & jnp.where(
            self.torsion_wraps,
            (torsion_angles >= self.torsion_lower)
            | (torsion_angles <= self.torsion_upper),
            (torsion_angles >= self.torsion_lower)
            & (torsion_angles <= self.torsion_upper),
        )
        finite = jnp.all(jnp.isfinite(x[self.active_indices]))
        success = (
            finite
            & jnp.all(covalent)
            & jnp.all(chirality)
            & jnp.all(clash_free)
            & jnp.all(planar)
            & jnp.all(torsion_valid)
        )
        return ProteinGeometryEvidence(
            lengths,
            covalent,
            volumes,
            chirality,
            distances,
            clash_free,
            angles,
            planar,
            torsion_angles,
            torsion_valid,
            finite,
            success,
            self.qualification_id,
        )


__all__ = [
    "PreparedProteinQualification",
    "ProteinGeometryEvidence",
    "ProteinTorsionCriterion",
]
