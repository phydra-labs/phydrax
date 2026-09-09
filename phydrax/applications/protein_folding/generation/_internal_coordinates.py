# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Topology-bound protein internal coordinates for fixed chemical constructs."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ..._coordinate_generation._decoder import (
    AbstractCoordinateDecoder,
    CoordinateEncoding,
)
from ..._coordinate_generation._support import PreparedCoordinateSupport
from .._binding import PreparedProteinBinding
from .._construct import ProteinAtomKey, ResidueKey


_STANDARD_SIDECHAIN_BONDS = {
    "A": (("CA", "CB"),),
    "R": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "NE"),
        ("NE", "CZ"),
        ("CZ", "NH1"),
        ("CZ", "NH2"),
    ),
    "N": (("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")),
    "D": (("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")),
    "C": (("CA", "CB"), ("CB", "SG")),
    "E": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "OE1"),
        ("CD", "OE2"),
    ),
    "Q": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "OE1"),
        ("CD", "NE2"),
    ),
    "G": (),
    "H": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "ND1"),
        ("ND1", "CE1"),
        ("CE1", "NE2"),
        ("NE2", "CD2"),
        ("CD2", "CG"),
    ),
    "I": (
        ("CA", "CB"),
        ("CB", "CG1"),
        ("CB", "CG2"),
        ("CG1", "CD1"),
    ),
    "L": (("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")),
    "K": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "CE"),
        ("CE", "NZ"),
    ),
    "M": (("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")),
    "F": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
    ),
    "P": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "N"),
    ),
    "S": (("CA", "CB"), ("CB", "OG")),
    "T": (("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")),
    "W": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "NE1"),
        ("NE1", "CE2"),
        ("CE2", "CD2"),
        ("CD2", "CE3"),
        ("CE3", "CZ3"),
        ("CZ3", "CH2"),
        ("CH2", "CZ2"),
        ("CZ2", "CE2"),
    ),
    "Y": (
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
        ("CZ", "OH"),
    ),
    "V": (("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")),
}


@dataclass(frozen=True, slots=True)
class ProteinClosureGroup:
    """A rigid standard-residue ring whose exact topology remains evaluated."""

    group_id: str
    kind: str
    residue: ResidueKey
    atom_indices: tuple[int, ...]
    bond_indices: tuple[tuple[int, int], ...]
    target_lengths: tuple[float, ...]
    tolerance: float

    def __post_init__(self):
        if self.kind not in ("proline-ring", "aromatic-ring") or not self.group_id:
            raise ValueError(
                "Closure groups must identify a supported rigid protein ring."
            )
        if (
            not isinstance(self.residue, ResidueKey)
            or not self.atom_indices
            or len(set(self.atom_indices)) != len(self.atom_indices)
            or len(self.bond_indices) != len(self.target_lengths)
            or not self.bond_indices
        ):
            raise ValueError("Closure groups require exact atoms, bonds, and targets.")
        if (
            not np.isfinite(self.target_lengths).all()
            or any(value <= 0 for value in self.target_lengths)
            or not np.isfinite(self.tolerance)
            or self.tolerance <= 0
        ):
            raise ValueError("Closure targets and tolerance must be finite and positive.")


class ProteinInternalCoordinatePlan(StrictModule, NonTrainableState):
    """Immutable kinematic plan bound to one resolved topology and reference gauge."""

    atom_order: tuple[ProteinAtomKey, ...] = eqx.field(static=True)
    atom_output_indices: tuple[int, ...] = eqx.field(static=True)
    reference_indices: Array
    bond_indices: Array
    bond_lengths: Array
    angle_indices: Array
    bond_angles: Array
    torsion_bond_indices: Array
    torsion_slots: Array
    closure_groups: tuple[ProteinClosureGroup, ...] = eqx.field(static=True)
    closure_bond_indices: Array
    closure_target_lengths: Array
    closure_tolerances: Array
    fixed_mask: Array
    rotation_masks: Array
    reference_torsions: Array
    reference_positions: Array
    active_mask: Array
    variable_torsion_rows: tuple[int, ...] = eqx.field(static=True)
    peptide_modes: tuple[str, ...] = eqx.field(static=True)
    chemistry_profile_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    _support_id: str = eqx.field(static=True)
    _representation_id: str = eqx.field(static=True)
    _coordinate_size: int = eqx.field(static=True)
    minimum_periodic_norm: float = eqx.field(static=True)
    construction_length_tolerance: float = eqx.field(static=True)
    construction_angle_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    guaranteed_invariants: tuple[str, ...] = eqx.field(static=True)
    evaluated_invariants: tuple[str, ...] = eqx.field(static=True)
    scientific_scope: str = eqx.field(static=True)

    @property
    def coordinate_size(self):
        return self._coordinate_size

    @property
    def support_id(self):
        return self._support_id

    @property
    def representation_id(self):
        return self._representation_id


class ProteinClosureEvidence(eqx.Module):
    residuals: Array
    valid: Array
    finite: Array
    group_ids: tuple[str, ...] = eqx.field(static=True)


class ProteinDecodedCoordinates(eqx.Module):
    """Every decoded case plus construction and closure evidence."""

    positions: Array
    periodic_coordinates: Array
    periodic_norms: Array
    periodic_valid: Array
    bond_residuals: Array
    angle_residuals: Array
    closure: ProteinClosureEvidence
    finite: Array
    valid: Array
    plan_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    guaranteed_invariants: tuple[str, ...] = eqx.field(static=True)

    @property
    def residuals(self):
        return jnp.concatenate(
            (
                self.bond_residuals,
                self.angle_residuals,
                self.closure.residuals,
            ),
            axis=-1,
        )


class PreparedProteinCoordinateDecoder(AbstractCoordinateDecoder):
    """Differentiable periodic-torsion decoder; no minimization or resampling."""

    plan: ProteinInternalCoordinatePlan

    @property
    def coordinate_size(self):
        return self.plan.coordinate_size

    @property
    def support_id(self):
        return self.plan.support_id

    @property
    def representation_id(self):
        return self.plan.representation_id

    def project(self, coordinates):
        values = jnp.asarray(coordinates)
        if values.shape != (self.coordinate_size,):
            raise ValueError(
                "Protein model state must have the plan's fixed periodic coordinate shape."
            )
        return values

    def reference_periodic_coordinates(self):
        rows = jnp.asarray(self.plan.variable_torsion_rows, dtype=jnp.int32)
        angles = self.plan.reference_torsions[rows]
        return jnp.stack((jnp.sin(angles), jnp.cos(angles)), axis=-1).reshape(
            (self.coordinate_size,)
        )

    def reconstruct(self, positions):
        """No-learning fixed-geometry reconstruction with all evidence retained."""
        encoding = self.encode(positions)
        decoded = self.decode(encoding.coordinates)
        return encoding, decoded

    def encode(self, positions):
        values = jnp.asarray(positions)
        capacity = self.plan.reference_positions.shape[0]
        if values.shape[-2:] != (capacity, 3):
            raise ValueError(
                "Protein encoder input must end in the plan's (atom_capacity, 3)."
            )
        local = values[..., jnp.asarray(self.plan.atom_output_indices), :]
        rows = jnp.asarray(self.plan.variable_torsion_rows, dtype=jnp.int32)
        references = self.plan.reference_indices[rows]
        angles, torsion_valid = _dihedral_jax(local, references)
        coordinates = jnp.stack((jnp.sin(angles), jnp.cos(angles)), axis=-1)
        encoded = coordinates.reshape(values.shape[:-2] + (self.coordinate_size,))
        finite = jnp.all(
            jnp.isfinite(jnp.where(self.plan.active_mask[:, None], values, 0.0)),
            axis=(-2, -1),
        )
        return CoordinateEncoding(
            encoded,
            finite & jnp.all(torsion_valid, axis=-1),
            self.representation_id,
        )

    def decode(self, periodic_coordinates):
        raw = jnp.asarray(periodic_coordinates)
        if raw.shape[-1:] != (self.coordinate_size,):
            raise ValueError(
                "Protein periodic coordinates must end in twice the torsion count."
            )
        leading = raw.shape[:-1]
        count = len(self.plan.variable_torsion_rows)
        pairs = raw.reshape(leading + (count, 2))
        norms = jnp.sqrt(jnp.sum(pairs * pairs, axis=-1))
        periodic_valid = jnp.all(jnp.isfinite(pairs), axis=-1) & (
            norms > self.plan.minimum_periodic_norm
        )
        angles = jnp.arctan2(pairs[..., 0], pairs[..., 1])
        reference = self.plan.reference_torsions[
            jnp.asarray(self.plan.variable_torsion_rows, dtype=jnp.int32)
        ]
        deltas = _wrap_angle(angles - reference)
        output_indices = jnp.asarray(self.plan.atom_output_indices, dtype=jnp.int32)
        local = jnp.broadcast_to(
            self.plan.reference_positions[output_indices],
            leading + (len(self.plan.atom_order), 3),
        )
        for slot, row in enumerate(self.plan.variable_torsion_rows):
            parent, child = self.plan.torsion_bond_indices[row]
            origin = local[..., parent, :]
            endpoint = local[..., child, :]
            axis = endpoint - origin
            axis_norm = jnp.sqrt(jnp.sum(axis * axis, axis=-1, keepdims=True))
            unit = axis / axis_norm
            relative = local - origin[..., None, :]
            angle = deltas[..., slot]
            sine = jnp.sin(angle)[..., None, None]
            cosine = jnp.cos(angle)[..., None, None]
            direction = unit[..., None, :]
            rotated = (
                relative * cosine
                + jnp.cross(direction, relative) * sine
                + direction
                * jnp.sum(direction * relative, axis=-1, keepdims=True)
                * (1.0 - cosine)
            )
            candidate = origin[..., None, :] + rotated
            mask = self.plan.rotation_masks[row]
            local = jnp.where(mask[:, None], candidate, local)
        positions = jnp.broadcast_to(
            jnp.zeros_like(self.plan.reference_positions),
            leading + self.plan.reference_positions.shape,
        )
        positions = positions.at[..., output_indices, :].set(local)

        bonds = self.plan.bond_indices
        bond_delta = local[..., bonds[:, 0], :] - local[..., bonds[:, 1], :]
        lengths = jnp.sqrt(jnp.sum(bond_delta * bond_delta, axis=-1))
        bond_residuals = lengths - self.plan.bond_lengths
        angle_values, angle_valid = _bond_angle_jax(local, self.plan.angle_indices)
        angle_residuals = _wrap_angle(angle_values - self.plan.bond_angles)
        closure_pairs = self.plan.closure_bond_indices
        closure_delta = (
            local[..., closure_pairs[:, 0], :] - local[..., closure_pairs[:, 1], :]
        )
        closure_lengths = jnp.sqrt(jnp.sum(closure_delta * closure_delta, axis=-1))
        closure_residuals = closure_lengths - self.plan.closure_target_lengths
        closure_finite = jnp.all(jnp.isfinite(closure_residuals), axis=-1)
        closure_valid = jnp.isfinite(closure_residuals) & (
            jnp.abs(closure_residuals) <= self.plan.closure_tolerances
        )
        closure = ProteinClosureEvidence(
            closure_residuals,
            closure_valid,
            closure_finite,
            tuple(
                group.group_id
                for group in self.plan.closure_groups
                for _ in group.bond_indices
            ),
        )
        finite = jnp.all(
            jnp.isfinite(jnp.where(self.plan.active_mask[:, None], positions, 0.0)),
            axis=(-2, -1),
        )
        bond_valid = jnp.all(
            jnp.abs(bond_residuals) <= self.plan.construction_length_tolerance,
            axis=-1,
        )
        angle_residual_valid = jnp.all(
            angle_valid
            & (jnp.abs(angle_residuals) <= self.plan.construction_angle_tolerance),
            axis=-1,
        )
        valid = (
            finite
            & jnp.all(periodic_valid, axis=-1)
            & bond_valid
            & angle_residual_valid
            & jnp.all(closure_valid, axis=-1)
        )
        return ProteinDecodedCoordinates(
            positions,
            raw,
            norms,
            periodic_valid,
            bond_residuals,
            angle_residuals,
            closure,
            finite,
            valid,
            self.plan.plan_id,
            self.representation_id,
            self.plan.guaranteed_invariants,
        )


def _wrap_angle(value):
    return jnp.arctan2(jnp.sin(value), jnp.cos(value))


def _dihedral_jax(positions, indices):
    points = tuple(positions[..., indices[:, i], :] for i in range(4))
    first = points[1] - points[0]
    axis = points[2] - points[1]
    last = points[3] - points[2]
    axis_norm = jnp.sqrt(jnp.sum(axis * axis, axis=-1, keepdims=True))
    unit = axis / axis_norm
    first_plane = first - jnp.sum(first * unit, axis=-1, keepdims=True) * unit
    last_plane = last - jnp.sum(last * unit, axis=-1, keepdims=True) * unit
    first_norm = jnp.sqrt(jnp.sum(first_plane * first_plane, axis=-1))
    last_norm = jnp.sqrt(jnp.sum(last_plane * last_plane, axis=-1))
    x = jnp.sum(first_plane * last_plane, axis=-1)
    y = jnp.sum(jnp.cross(unit, first_plane) * last_plane, axis=-1)
    valid = (
        jnp.isfinite(x)
        & jnp.isfinite(y)
        & (axis_norm[..., 0] > 0)
        & (first_norm > 0)
        & (last_norm > 0)
    )
    return jnp.arctan2(y, x), valid


def _bond_angle_jax(positions, indices):
    first = positions[..., indices[:, 0], :] - positions[..., indices[:, 1], :]
    last = positions[..., indices[:, 2], :] - positions[..., indices[:, 1], :]
    cross = jnp.sqrt(jnp.sum(jnp.cross(first, last) ** 2, axis=-1))
    dot = jnp.sum(first * last, axis=-1)
    first_norm = jnp.sqrt(jnp.sum(first * first, axis=-1))
    last_norm = jnp.sqrt(jnp.sum(last * last, axis=-1))
    valid = jnp.isfinite(cross) & jnp.isfinite(dot) & (first_norm > 0) & (last_norm > 0)
    return jnp.arctan2(cross, dot), valid


def _dihedral_numpy(positions, indices):
    values, valid = _dihedral_jax(jnp.asarray(positions), jnp.asarray([indices]))
    if not bool(valid[0]):
        raise ValueError("Reference topology contains a degenerate torsion frame.")
    return float(values[0])


def _bond_angle_numpy(positions, indices):
    values, valid = _bond_angle_jax(jnp.asarray(positions), jnp.asarray([indices]))
    if not bool(valid[0]):
        raise ValueError("Reference topology contains a degenerate bond angle.")
    return float(values[0])


def _component_without_edge(adjacency, start, edge):
    seen, pending = set(), [start]
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(
            neighbor
            for neighbor in adjacency[node]
            if frozenset((node, neighbor)) != edge and neighbor not in seen
        )
    return seen


def _cycle_components(nonbridges):
    adjacency = {}
    for edge in nonbridges:
        left, right = tuple(edge)
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    groups = []
    remaining = set(adjacency)
    while remaining:
        start = min(remaining)
        seen, pending = set(), [start]
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            pending.extend(adjacency[node] - seen)
        groups.append(tuple(sorted(seen)))
        remaining -= seen
    return tuple(groups)


def prepare_protein_internal_coordinate_plan(
    binding,
    support,
    *,
    cis_peptide_indices=(),
    closure_tolerance=1e-5,
    construction_length_tolerance=1e-5,
    construction_angle_tolerance=1e-5,
    minimum_periodic_norm=1e-8,
):
    """Compile exact standard chemistry into rigid-ring torsion kinematics.

    Only the resolved uncapped canonical-L single-chain profile is accepted.
    Missing atoms, disulfides, cyclic/crosslinked chains, nonstandard rings, and
    undeclared cis peptide geometry are refused at preparation. Ring and proline
    geometry is retained as rigid template geometry and independently evidenced.
    """
    if not isinstance(binding, PreparedProteinBinding) or not isinstance(
        support, PreparedCoordinateSupport
    ):
        raise TypeError(
            "Protein internal coordinates require a bound protein and coordinate support."
        )
    chemistry = binding.chemistry
    if chemistry.profile != "canonical-L-single-chain-explicit":
        raise ValueError("Internal coordinates do not support this protein chemistry.")
    if support.construct_id != chemistry.construct.fingerprint():
        raise ValueError("Decoder support and resolved protein construct differ.")
    tolerances = (
        closure_tolerance,
        construction_length_tolerance,
        construction_angle_tolerance,
        minimum_periodic_norm,
    )
    if not np.isfinite(tolerances).all() or any(value <= 0 for value in tolerances):
        raise ValueError("Decoder tolerances must be finite and positive.")

    system = binding.force_field.system
    system_ids = np.asarray(system.plan.particle_ids)
    system_active = np.asarray(system.active_mask)
    support_ids = np.asarray(support.template.particle_ids[0])
    support_active = np.asarray(support.template.atom_mask[0])
    if (
        system.capacity != support.template.atom_capacity
        or not np.array_equal(system_ids, support_ids)
        or not np.array_equal(system_active, support_active)
    ):
        raise ValueError(
            "Decoder, full qualification, and sparse support require identical atom order."
        )
    if tuple(binding.atom_ids) != tuple(
        int(value) for value in system_ids[list(binding.atom_indices)]
    ):
        raise ValueError(
            "Binding stable atom identities do not match its native topology."
        )

    atom_order = binding.atom_keys
    local_by_system = {
        system_index: local for local, system_index in enumerate(binding.atom_indices)
    }
    output_indices = tuple(binding.atom_indices)
    bonds_system = np.asarray(system.topology.bond_indices, dtype=np.int32)
    if any(
        int(left) not in local_by_system or int(right) not in local_by_system
        for left, right in bonds_system
    ):
        raise ValueError("Protein topology contains nonmaterial or unresolved atoms.")
    bonds = tuple(
        sorted(
            {
                tuple(sorted((local_by_system[int(left)], local_by_system[int(right)])))
                for left, right in bonds_system
            }
        )
    )
    if len(bonds) != len(bonds_system):
        raise ValueError("Protein topology contains duplicate covalent edges.")
    adjacency = {index: set() for index in range(len(atom_order))}
    for left, right in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)

    residues = chemistry.construct.residue_keys
    sequence = chemistry.construct.sequences[0]
    if set(atom_order) != set(chemistry.atom_keys):
        raise ValueError(
            "Binding must materialize every atom in the resolved protein chemistry exactly."
        )
    atom_index = {key: index for index, key in enumerate(atom_order)}
    atomic_number_by_key = dict(
        zip(chemistry.atom_keys, chemistry.atomic_numbers, strict=True)
    )
    atomic_numbers = tuple(atomic_number_by_key[key] for key in atom_order)
    residue_position = {residue: index for index, residue in enumerate(residues)}
    expected_peptides = {
        frozenset(
            (
                atom_index[ProteinAtomKey(left, "C")],
                atom_index[ProteinAtomKey(right, "N")],
            )
        )
        for left, right in zip(residues[:-1], residues[1:], strict=True)
    }
    peptide_edges = set()
    for left, right in bonds:
        first, second = atom_order[left], atom_order[right]
        if first.residue != second.residue:
            edge = frozenset((left, right))
            if edge not in expected_peptides:
                if first.atom_name == "SG" and second.atom_name == "SG":
                    raise ValueError("Disulfide closure is unsupported by this decoder.")
                raise ValueError(
                    "Cyclic, crosslinked, or non-peptide inter-residue topology is unsupported."
                )
            peptide_edges.add(edge)
    if peptide_edges != expected_peptides:
        raise ValueError("Resolved topology lacks a required peptide bond.")

    expected_heavy = set(expected_peptides)
    for residue, letter in zip(residues, sequence, strict=True):
        standard_bonds = (
            ("N", "CA"),
            ("CA", "C"),
            ("C", "O"),
            *_STANDARD_SIDECHAIN_BONDS[letter],
        )
        if residue == residues[-1]:
            standard_bonds = (*standard_bonds, ("C", "OXT"))
        expected_heavy.update(
            frozenset(
                (
                    atom_index[ProteinAtomKey(residue, first)],
                    atom_index[ProteinAtomKey(residue, second)],
                )
            )
            for first, second in standard_bonds
        )
    actual_heavy = {
        frozenset((left, right))
        for left, right in bonds
        if atomic_numbers[left] != 1 and atomic_numbers[right] != 1
    }
    if actual_heavy != expected_heavy:
        raise ValueError(
            "Protein topology must match the explicit standard heavy-atom covalent graph."
        )
    for index, number in enumerate(atomic_numbers):
        if number != 1:
            continue
        if len(adjacency[index]) != 1:
            raise ValueError(
                "Explicit protein hydrogens must be terminal atoms with one covalent bond."
            )
        neighbor = next(iter(adjacency[index]))
        if (
            atomic_numbers[neighbor] == 1
            or atom_order[neighbor].residue != atom_order[index].residue
        ):
            raise ValueError(
                "Explicit protein hydrogens must bind one heavy atom in their own residue."
            )

    bridges = set()
    for left, right in bonds:
        edge = frozenset((left, right))
        if right not in _component_without_edge(adjacency, left, edge):
            bridges.add(edge)
    nonbridges = {frozenset(edge) for edge in bonds} - bridges
    reference_support, gauge_valid = support.canonicalize(binding.realized_positions)
    if not bool(gauge_valid):
        raise ValueError("Bound reference coordinates have a degenerate proper gauge.")
    reference_support = np.asarray(reference_support)
    reference = reference_support[np.asarray(output_indices)]

    closure_groups = []
    for component in _cycle_components(nonbridges):
        ring_residues = {atom_order[index].residue for index in component}
        if len(ring_residues) != 1:
            raise ValueError("Inter-residue or fused nonstandard closure is unsupported.")
        residue = next(iter(ring_residues))
        letter = sequence[residue_position[residue]]
        if letter == "P":
            kind = "proline-ring"
        elif letter in "FHWY":
            kind = "aromatic-ring"
        else:
            raise ValueError(
                "Only proline and standard aromatic rigid rings are supported."
            )
        component_set = set(component)
        ring_bonds = tuple(
            edge
            for edge in bonds
            if edge[0] in component_set
            and edge[1] in component_set
            and frozenset(edge) in nonbridges
        )
        targets = tuple(
            float(np.linalg.norm(reference[a] - reference[b])) for a, b in ring_bonds
        )
        group_id = canonical_fingerprint(
            {
                "kind": kind,
                "residue": (residue.chain_id, residue.position),
                "atoms": [atom_order[index].record() for index in component],
                "bonds": ring_bonds,
                "targets": targets,
                "tolerance": closure_tolerance,
            }
        )
        closure_groups.append(
            ProteinClosureGroup(
                group_id,
                kind,
                residue,
                component,
                ring_bonds,
                targets,
                float(closure_tolerance),
            )
        )

    declared_cis = tuple(cis_peptide_indices)
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in declared_cis
        )
        or len(set(declared_cis)) != len(declared_cis)
        or any(not 0 <= value < len(residues) - 1 for value in declared_cis)
    ):
        raise ValueError(
            "Cis peptide declarations require unique zero-based bond indices."
        )
    observed_cis = []
    peptide_modes = []
    for index, (left, right) in enumerate(zip(residues[:-1], residues[1:], strict=True)):
        torsion_keys = (
            ProteinAtomKey(left, "CA"),
            ProteinAtomKey(left, "C"),
            ProteinAtomKey(right, "N"),
            ProteinAtomKey(right, "CA"),
        )
        torsion_indices = tuple(atom_order.index(key) for key in torsion_keys)
        angle = _dihedral_numpy(reference, torsion_indices)
        distance_cis = abs(float(np.arctan2(np.sin(angle), np.cos(angle))))
        distance_trans = abs(abs(angle) - np.pi)
        if distance_cis <= 0.35:
            observed_cis.append(index)
            peptide_modes.append("cis")
        elif distance_trans <= 0.35:
            peptide_modes.append("trans")
        else:
            raise ValueError(
                "Reference peptide geometry is neither declared cis nor trans planar."
            )
    if set(observed_cis) != set(declared_cis):
        raise ValueError(
            "Cis peptide geometry must be declared exactly; trans is never imposed silently."
        )

    root = atom_order.index(ProteinAtomKey(residues[0], "N"))
    parent = {root: -1}
    depth = {root: 0}
    pending = [root]
    while pending:
        node = pending.pop(0)
        for neighbor in sorted(adjacency[node]):
            if neighbor not in parent:
                parent[neighbor] = node
                depth[neighbor] = depth[node] + 1
                pending.append(neighbor)
    if len(parent) != len(atom_order):
        raise ValueError("Protein topology must be one connected fixed construct.")

    candidates = []
    for edge in bridges:
        first, second = tuple(edge)
        parent_atom, child_atom = (
            (first, second) if depth[first] < depth[second] else (second, first)
        )
        parent_neighbors = adjacency[parent_atom] - {child_atom}
        child_neighbors = adjacency[child_atom] - {parent_atom}
        if not parent_neighbors or not child_neighbors:
            continue
        first_reference = (
            parent[parent_atom]
            if parent[parent_atom] in parent_neighbors
            else min(parent_neighbors)
        )
        last_reference = min(child_neighbors, key=lambda value: (depth[value], value))
        descendants = _component_without_edge(adjacency, child_atom, edge)
        if root in descendants:
            raise ValueError(
                "Internal torsion orientation failed to preserve the root gauge."
            )
        candidates.append(
            (
                depth[child_atom],
                parent_atom,
                child_atom,
                first_reference,
                last_reference,
                tuple(sorted(descendants)),
                edge in peptide_edges,
            )
        )
    candidates.sort()
    reference_indices = tuple(
        (first_ref, parent_atom, child_atom, last_ref)
        for _, parent_atom, child_atom, first_ref, last_ref, _, _ in candidates
    )
    reference_torsions = tuple(
        _dihedral_numpy(reference, indices) for indices in reference_indices
    )
    fixed_mask = tuple(item[-1] for item in candidates)
    variable_rows = tuple(index for index, fixed in enumerate(fixed_mask) if not fixed)
    slots = []
    next_slot = 0
    for fixed in fixed_mask:
        slots.append(-1 if fixed else next_slot)
        next_slot += int(not fixed)
    rotation_masks = np.zeros((len(candidates), len(atom_order)), dtype=bool)
    for row, item in enumerate(candidates):
        rotation_masks[row, list(item[-2])] = True

    bond_lengths = tuple(
        float(np.linalg.norm(reference[a] - reference[b])) for a, b in bonds
    )
    angles = []
    for center in range(len(atom_order)):
        neighbors = sorted(adjacency[center])
        for first_index, first in enumerate(neighbors):
            for last in neighbors[first_index + 1 :]:
                angles.append((first, center, last))
    bond_angles = tuple(_bond_angle_numpy(reference, indices) for indices in angles)
    closure_pairs = tuple(pair for group in closure_groups for pair in group.bond_indices)
    closure_targets = tuple(
        target for group in closure_groups for target in group.target_lengths
    )
    closure_tolerances = tuple(
        group.tolerance for group in closure_groups for _ in group.bond_indices
    )
    dtype = support.template.positions.dtype
    active_mask = np.asarray(support.template.atom_mask[0])
    plan_record = {
        "kind": "protein-internal-coordinate-plan-v1",
        "binding": binding.binding_id,
        "support": support.support_id,
        "chemistry": chemistry.fingerprint(),
        "chemistry_support": (
            "canonical-standard-heavy-graph; explicit terminal intra-residue hydrogens"
        ),
        "topology": system.topology.topology_id,
        "atoms": [key.record() for key in atom_order],
        "output_indices": output_indices,
        "bonds": bonds,
        "bond_lengths": bond_lengths,
        "angles": angles,
        "bond_angles": bond_angles,
        "references": reference_indices,
        "torsions": reference_torsions,
        "fixed": fixed_mask,
        "closures": [group.group_id for group in closure_groups],
        "peptides": peptide_modes,
        "reference": array_tree_fingerprint(reference_support),
        "tolerances": tolerances,
    }
    plan_id = canonical_fingerprint(plan_record)
    representation_id = "protein-periodic-internal:" + plan_id
    guaranteed = (
        "reference topology bond lengths",
        "reference bonded angles",
        "proper-rotation chirality",
        "rigid proline and aromatic ring geometry",
        "declared cis/trans peptide geometry",
        "proper-rotation reference frame before workflow canonicalization",
    )
    evaluated = (
        "ring closure residuals",
        "all covalent bounds",
        "non-glycine alpha chirality",
        "nonlocal clashes",
        "peptide planarity",
        "declared torsion support",
    )
    return ProteinInternalCoordinatePlan(
        atom_order,
        output_indices,
        jnp.asarray(reference_indices, dtype=jnp.int32).reshape((-1, 4)),
        jnp.asarray(bonds, dtype=jnp.int32).reshape((-1, 2)),
        jnp.asarray(bond_lengths, dtype=dtype),
        jnp.asarray(angles, dtype=jnp.int32).reshape((-1, 3)),
        jnp.asarray(bond_angles, dtype=dtype),
        jnp.asarray(
            tuple((item[1], item[2]) for item in candidates), dtype=jnp.int32
        ).reshape((-1, 2)),
        jnp.asarray(slots, dtype=jnp.int32),
        tuple(closure_groups),
        jnp.asarray(closure_pairs, dtype=jnp.int32).reshape((-1, 2)),
        jnp.asarray(closure_targets, dtype=dtype),
        jnp.asarray(closure_tolerances, dtype=dtype),
        jnp.asarray(fixed_mask, dtype=bool),
        jnp.asarray(rotation_masks),
        jnp.asarray(reference_torsions, dtype=dtype),
        jnp.asarray(reference_support, dtype=dtype),
        jnp.asarray(active_mask),
        variable_rows,
        tuple(peptide_modes),
        chemistry.fingerprint(),
        system.topology.topology_id,
        binding.binding_id,
        support.support_id,
        representation_id,
        2 * len(variable_rows),
        float(minimum_periodic_norm),
        float(construction_length_tolerance),
        float(construction_angle_tolerance),
        plan_id,
        guaranteed,
        evaluated,
        (
            "fixed-construct proposal mechanics only; non-equilibrium and not "
            "sequence-general; generated frequency is not physical population"
        ),
    )


def prepare_protein_coordinate_decoder(binding, support, **kwargs):
    """Prepare the numeric decoder for one exact binding and support."""
    return PreparedProteinCoordinateDecoder(
        prepare_protein_internal_coordinate_plan(binding, support, **kwargs)
    )


__all__ = [
    "PreparedProteinCoordinateDecoder",
    "ProteinClosureEvidence",
    "ProteinClosureGroup",
    "ProteinDecodedCoordinates",
    "ProteinInternalCoordinatePlan",
    "prepare_protein_coordinate_decoder",
    "prepare_protein_internal_coordinate_plan",
]
