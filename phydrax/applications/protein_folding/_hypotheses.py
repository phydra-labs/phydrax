# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...artifacts import ScientificArtifactEnvelope
from ...qualification import ReferenceArtifactManifest
from ...units import LENGTH, UnitDefinition
from ._construct import _identifier, ProteinAtomKey, ProteinConstruct


@dataclass(frozen=True, slots=True)
class ProteinSourceAtom:
    """Lossless source-row identity for one explicitly selected atom conformer."""

    record_id: str
    atom_key: ProteinAtomKey
    model_id: str
    author_chain_id: str
    author_residue_number: str
    insertion_code: str
    alternate_location: str
    occupancy: float
    element: int
    label_chain_id: str = ""
    label_residue_number: str = ""

    def __post_init__(self):
        for value in (self.record_id, self.model_id, self.author_residue_number):
            _identifier(value, "source identity")
        if not isinstance(self.atom_key, ProteinAtomKey):
            raise TypeError("atom_key must be a ProteinAtomKey.")
        if not np.isfinite(self.occupancy) or not 0 < self.occupancy <= 1:
            raise ValueError("Selected source atoms require occupancy in (0, 1].")
        if (
            isinstance(self.element, bool)
            or not isinstance(self.element, int)
            or self.element <= 0
        ):
            raise ValueError("element must be an explicit positive atomic number.")

    def record(self):
        return (
            self.record_id,
            self.atom_key.record(),
            self.model_id,
            self.author_chain_id,
            self.author_residue_number,
            self.insertion_code,
            self.alternate_location,
            self.occupancy,
            self.element,
            self.label_chain_id,
            self.label_residue_number,
        )


@dataclass(frozen=True, slots=True, init=False)
class ProteinStructureHypothesis:
    """Static coordinate proposal, not an equilibrium ensemble or a trajectory."""

    construct: ProteinConstruct
    source_atoms: tuple[ProteinSourceAtom, ...]
    positions: Array
    length_unit: UnitDefinition
    source: ScientificArtifactEnvelope
    rights: tuple[ReferenceArtifactManifest, ...]
    provider: str
    confidence: tuple[tuple[str, float], ...]
    hypothesis_id: str

    def __init__(
        self,
        construct,
        source_atoms,
        positions,
        length_unit,
        source,
        rights,
        *,
        provider="user-supplied",
        confidence=(),
    ):
        if not isinstance(construct, ProteinConstruct) or not isinstance(
            source, ScientificArtifactEnvelope
        ):
            raise TypeError("A construct and original scientific artifact are required.")
        if source.status != "complete":
            raise ValueError("Failed source artifacts cannot become admitted hypotheses.")
        if not isinstance(length_unit, UnitDefinition) or length_unit.dimension != LENGTH:
            raise ValueError("Coordinates need an exact length unit.")
        rows = tuple(source_atoms)
        coordinates = np.asarray(positions, dtype=float)
        if not rows or any(not isinstance(row, ProteinSourceAtom) for row in rows):
            raise ValueError("Explicit source atom records are required.")
        if coordinates.shape != (len(rows), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError(
                "Coordinates must be finite with one Cartesian vector per source atom."
            )
        if len({row.record_id for row in rows}) != len(rows) or len(
            {row.atom_key for row in rows}
        ) != len(rows):
            raise ValueError(
                "Duplicate source or chemical identity; select alternate locations explicitly."
            )
        if len({row.model_id for row in rows}) != 1:
            raise ValueError("Each source model is a separate hypothesis.")
        if any(row.atom_key.residue not in construct.residue_keys for row in rows):
            raise ValueError("Source atoms reference residues outside the construct.")
        manifests = tuple(rights)
        if not manifests or any(
            not isinstance(value, ReferenceArtifactManifest) for value in manifests
        ):
            raise TypeError("Source and inherited rights manifests are required.")
        _identifier(provider, "provider")
        confidence_ = tuple((str(name), float(value)) for name, value in confidence)
        if any(not name or not np.isfinite(value) for name, value in confidence_):
            raise ValueError("Provider-specific confidence must be finite and named.")
        identity = canonical_fingerprint(
            {
                "kind": "protein-structure-hypothesis",
                "construct": construct.fingerprint(),
                "rows": [row.record() for row in rows],
                "coordinates": array_tree_fingerprint(coordinates),
                "unit": length_unit.unit_id,
                "source": source.artifact_id,
                "rights": [value.manifest_id for value in manifests],
                "provider": provider,
                "confidence": confidence_,
            }
        )
        for name, value in (
            ("construct", construct),
            ("source_atoms", rows),
            ("positions", jnp.asarray(coordinates)),
            ("length_unit", length_unit),
            ("source", source),
            ("rights", manifests),
            ("provider", provider),
            ("confidence", confidence_),
            ("hypothesis_id", identity),
        ):
            object.__setattr__(self, name, value)

    def require_rights(
        self,
        *,
        commercial_use=False,
        redistribution=False,
        training_use=False,
        export=False,
    ):
        return tuple(
            manifest.require_rights(
                commercial_use=commercial_use,
                redistribution=redistribution,
                training_use=training_use,
                export=export,
            )
            for manifest in self.rights
        )


@dataclass(frozen=True, slots=True)
class ProteinHypothesisView:
    """Selection retains every original hypothesis and a separate policy identity."""

    hypotheses: tuple[ProteinStructureHypothesis, ...]
    selected_indices: tuple[int, ...]
    policy_id: str

    def __post_init__(self):
        _identifier(self.policy_id, "selection policy")
        if (
            not self.hypotheses
            or not self.selected_indices
            or len(set(self.selected_indices)) != len(self.selected_indices)
        ):
            raise ValueError("Retain all hypotheses and select unique indices.")
        if any(i < 0 or i >= len(self.hypotheses) for i in self.selected_indices):
            raise ValueError("Selection index lies outside the retained hypotheses.")
        if len({h.construct.fingerprint() for h in self.hypotheses}) != 1:
            raise ValueError("Hypotheses must describe one ordered construct.")


__all__ = ["ProteinSourceAtom", "ProteinStructureHypothesis", "ProteinHypothesisView"]
