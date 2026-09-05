# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...artifacts import ScientificArtifactEnvelope
from ...atomistic import PreparedAtomisticForceField
from ...qualification import ReferenceArtifactManifest
from ...units import conversion_factor, UnitDefinition
from ._chemical_state import ResolvedProteinChemistry
from ._construct import ProteinAtomKey
from ._hypotheses import ProteinStructureHypothesis


@dataclass(frozen=True, slots=True)
class ProteinMappingCoverage:
    required_count: int
    observed_count: int
    missing_atoms: tuple[ProteinAtomKey, ...]
    unexpected_atoms: tuple[ProteinAtomKey, ...]

    @property
    def complete(self):
        return not self.missing_atoms and not self.unexpected_atoms


def protein_mapping_coverage(hypothesis, chemistry) -> ProteinMappingCoverage:
    if hypothesis.construct.fingerprint() != chemistry.construct.fingerprint():
        raise ValueError(
            "Hypothesis and chemical realization describe different ordered constructs."
        )
    observed = {row.atom_key for row in hypothesis.source_atoms}
    expected = set(chemistry.atom_keys)
    return ProteinMappingCoverage(
        len(expected),
        len(observed),
        tuple(key for key in chemistry.atom_keys if key not in observed),
        tuple(
            row.atom_key
            for row in hypothesis.source_atoms
            if row.atom_key not in expected
        ),
    )


@dataclass(frozen=True, slots=True)
class PreparedProteinBinding:
    """Host binding; numeric energy/force execution remains native atomistic code."""

    hypothesis: ProteinStructureHypothesis
    chemistry: ResolvedProteinChemistry
    force_field: PreparedAtomisticForceField
    atom_keys: tuple[ProteinAtomKey, ...]
    atom_ids: tuple[int, ...]
    atom_indices: tuple[int, ...]
    source_indices: tuple[int, ...]
    realized_positions: Array
    coverage: ProteinMappingCoverage
    artifact: ScientificArtifactEnvelope
    rights: tuple[ReferenceArtifactManifest, ...]
    binding_id: str

    def evaluate(self, neighborhood, positions=None):
        """Conservative energy and full active forces; fixed atoms retain reactions.

        ``neighborhood`` is an already realized native ParticleNeighborhoodState.
        The fixed-shape numeric call is differentiable/JIT-compatible. Host
        preparation and lineage construction are intentionally outside JIT.
        """
        return self.force_field.potential.evaluate(
            self.realized_positions if positions is None else positions, neighborhood
        )

    def require_rights(self, **requested_use):
        return tuple(manifest.require_rights(**requested_use) for manifest in self.rights)


def bind_protein(
    hypothesis: ProteinStructureHypothesis,
    chemistry: ResolvedProteinChemistry,
    force_field: PreparedAtomisticForceField,
    atom_ids: Mapping[ProteinAtomKey, int],
    *,
    parameter_energy_unit: UnitDefinition,
    parameter_rights: tuple[ReferenceArtifactManifest, ...],
    commercial_use=False,
    redistribution=False,
    training_use=False,
    export=False,
) -> PreparedProteinBinding:
    """Bind a complete user-parameterized isolated protein, without atom completion.

    The caller certifies the force-field numeric coefficients already use
    ``parameter_energy_unit``. A mismatching scale is refused, never relabelled.
    This first chemistry profile excludes solvent/ions/caps and virtual DOFs;
    externally solvated ensembles remain valid inputs to thermodynamic estimators.
    """
    if (
        not isinstance(hypothesis, ProteinStructureHypothesis)
        or not isinstance(chemistry, ResolvedProteinChemistry)
        or not isinstance(force_field, PreparedAtomisticForceField)
    ):
        raise TypeError(
            "A hypothesis, explicit chemistry and prepared native force field are required."
        )
    coverage = protein_mapping_coverage(hypothesis, chemistry)
    if not coverage.complete:
        raise ValueError(
            f"Incomplete coordinate coverage: missing={coverage.missing_atoms}, "
            f"unexpected={coverage.unexpected_atoms}; missing atoms are not padding."
        )
    system = force_field.system
    if parameter_energy_unit.unit_id != system.plan.units.scale.energy_unit.unit_id:
        raise ValueError(
            "Force-field coefficients must already use the exact system energy scale."
        )
    factor = float(
        conversion_factor(hypothesis.length_unit, system.plan.units.scale.length_unit)
    )
    if system.cell is not None:
        raise ValueError(
            "The first isolated-protein binding profile is nonperiodic; "
            "periodic unwrapping needs a separate qualified profile."
        )
    expected = set(chemistry.atom_keys)
    if set(atom_ids) != expected:
        raise ValueError(
            "Stable-ID binding must cover the entire chemical inventory exactly."
        )
    ids = tuple(atom_ids[key] for key in chemistry.atom_keys)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not np.iinfo(np.int64).min <= value <= np.iinfo(np.int64).max
        for value in ids
    ):
        raise ValueError("Atom bindings require explicit int64-compatible stable IDs.")
    ids = tuple(map(int, ids))
    if len(set(ids)) != len(ids):
        raise ValueError("Distinct chemical atoms cannot alias one stable ID.")
    system_ids = np.asarray(system.plan.particle_ids)
    active = np.asarray(system.active_mask)
    if set(map(int, system_ids[active])) != set(ids):
        raise ValueError(
            "Chemically present atoms must match active material support exactly; "
            "missing atoms cannot be inactive padding."
        )
    if not np.all(np.asarray(system.plan.element_mask)[active]):
        raise ValueError("The all-atom protein profile requires elemental DOFs.")
    lookup = {int(value): index for index, value in enumerate(system_ids)}
    indices = tuple(lookup[value] for value in ids)
    source_lookup = {
        row.atom_key: index for index, row in enumerate(hypothesis.source_atoms)
    }
    source_indices = tuple(source_lookup[key] for key in chemistry.atom_keys)
    source_numbers = tuple(hypothesis.source_atoms[i].element for i in source_indices)
    if (
        source_numbers != chemistry.atomic_numbers
        or tuple(map(int, np.asarray(system.plan.atomic_numbers)[list(indices)]))
        != chemistry.atomic_numbers
    ):
        raise ValueError(
            "Source, chemical inventory and force field disagree on elements."
        )
    edges = {frozenset(map(int, pair)) for pair in np.asarray(system.plan.topology.bonds)}
    id_map = dict(zip(chemistry.atom_keys, ids, strict=True))
    required_edges = []
    residues = chemistry.construct.residue_keys
    for residue in residues:
        required_edges.extend(
            (ProteinAtomKey(residue, a), ProteinAtomKey(residue, b))
            for a, b in (("N", "CA"), ("CA", "C"), ("C", "O"))
        )
    required_edges.extend(
        (ProteinAtomKey(left, "C"), ProteinAtomKey(right, "N"))
        for left, right in zip(residues[:-1], residues[1:], strict=True)
    )
    if any(frozenset((id_map[a], id_map[b])) not in edges for a, b in required_edges):
        raise ValueError("Native topology lacks declared backbone/peptide connectivity.")
    adjacency = {value: set() for value in ids}
    for edge in edges:
        left, right = tuple(edge)
        if left not in adjacency or right not in adjacency:
            raise ValueError("Protein topology contains nonmaterial atoms.")
        adjacency[left].add(right)
        adjacency[right].add(left)
    visited, pending = set(), [ids[0]]
    while pending:
        value = pending.pop()
        if value not in visited:
            visited.add(value)
            pending.extend(adjacency[value] - visited)
    if visited != set(ids):
        raise ValueError(
            "Every chemically present atom must belong to the connected protein topology."
        )
    parameters = tuple(parameter_rights)
    if not parameters or any(
        not isinstance(value, ReferenceArtifactManifest) for value in parameters
    ):
        raise TypeError("Explicit parameter artifact rights are required.")
    rights = hypothesis.rights + parameters
    for manifest in rights:
        manifest.require_rights(
            commercial_use=commercial_use,
            redistribution=redistribution,
            training_use=training_use,
            export=export,
        )
    coordinates = np.zeros((system.capacity, 3), dtype=system.plan.coordinate_dtype)
    coordinates[list(indices)] = (
        np.asarray(hypothesis.positions)[list(source_indices)] * factor
    )
    identity = canonical_fingerprint(
        {
            "kind": "protein-native-binding",
            "hypothesis": hypothesis.hypothesis_id,
            "chemistry": chemistry.fingerprint(),
            "force_field": force_field.prepared_id,
            "mapping": [
                (key.record(), value)
                for key, value in zip(chemistry.atom_keys, ids, strict=True)
            ],
            "coordinates": array_tree_fingerprint(coordinates),
            "rights": [manifest.manifest_id for manifest in rights],
        }
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind="parameterized-protein-realization",
        content_digest=identity,
        producer="phydrax.protein_folding.bind_protein",
        producer_version="native",
        build_id=force_field.prepared_id,
        license_id="inherited-see-parent-manifests",
        resource_id=system.prepared_id,
        status="complete",
        parent_artifact_ids=(
            hypothesis.source.artifact_id,
            hypothesis.hypothesis_id,
            chemistry.source_id,
            *(manifest.manifest_id for manifest in rights),
        ),
    )
    return PreparedProteinBinding(
        hypothesis,
        chemistry,
        force_field,
        chemistry.atom_keys,
        ids,
        indices,
        source_indices,
        jnp.asarray(coordinates),
        coverage,
        artifact,
        rights,
        identity,
    )


__all__ = [
    "ProteinMappingCoverage",
    "PreparedProteinBinding",
    "protein_mapping_coverage",
    "bind_protein",
]
