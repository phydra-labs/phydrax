# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Already-resolved coordinate hypotheses; no lossy PDB/mmCIF parser."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...artifacts import ScientificArtifactEnvelope
from ...interchange import AdapterLoss, AdapterReport, AdapterStatus
from ...qualification import ReferenceArtifactManifest
from ...units import ANGSTROM, conversion_factor, UnitDefinition
from ._binding import NucleotideAtomMapping, prepare_nucleotide_binding


@dataclass(frozen=True, slots=True, init=False)
class NucleicStructureHypothesis:
    mapping: NucleotideAtomMapping
    positions: object
    length_unit: UnitDefinition
    source: ScientificArtifactEnvelope
    rights: tuple[ReferenceArtifactManifest, ...]
    coordinate_mask: object
    image_policy: str
    hypothesis_id: str
    parent: NucleicStructureHypothesis | None

    def __init__(
        self,
        mapping,
        positions,
        length_unit,
        source,
        rights,
        *,
        coordinate_mask=None,
        image_policy="nonperiodic",
        requested_use=None,
        parent=None,
    ):
        conversion_factor(length_unit, ANGSTROM)
        rights = (
            (rights,) if isinstance(rights, ReferenceArtifactManifest) else tuple(rights)
        )
        if (
            not isinstance(source, ScientificArtifactEnvelope)
            or not rights
            or any(not isinstance(item, ReferenceArtifactManifest) for item in rights)
        ):
            raise TypeError(
                "Hypotheses require the native source envelope and rights manifests."
            )
        for item in rights:
            item.require_rights(**({} if requested_use is None else requested_use))
        if source.license_id != rights[0].license_id:
            raise ValueError("Source and reference rights must agree.")
        if parent is not None:
            if (
                tuple(item.manifest_id for item in rights)
                != tuple(item.manifest_id for item in parent.rights)
                or parent.source.artifact_id not in source.parent_artifact_ids
            ):
                raise ValueError(
                    "Derived hypotheses must retain parent rights and lineage."
                )
        if image_policy not in ("nonperiodic", "unwrapped"):
            raise ValueError(
                "A hypothesis needs explicit nonperiodic or externally unwrapped coordinates."
            )
        coordinates = np.asarray(positions, dtype=float)
        mask = (
            np.ones(len(mapping.atom_ids), bool)
            if coordinate_mask is None
            else np.asarray(coordinate_mask, bool)
        )
        if (
            coordinates.shape != (len(mapping.atom_ids), 3)
            or mask.shape != (len(mapping.atom_ids),)
            or np.any(~np.isfinite(coordinates[mask]))
        ):
            raise ValueError(
                "Positions and finite coordinate coverage must align with the source atom map."
            )
        # Nonfinite unobserved entries remain in raw data; numeric binding masks
        # them rather than pretending they are absent chemical material.
        fields = dict(
            mapping=mapping,
            positions=jnp.asarray(coordinates),
            length_unit=length_unit,
            source=source,
            rights=rights,
            coordinate_mask=jnp.asarray(mask),
            image_policy=image_policy,
            parent=parent,
        )
        fields["hypothesis_id"] = canonical_fingerprint(
            {
                "mapping": mapping.fingerprint(),
                "source": source.artifact_id,
                "rights": tuple(item.manifest_id for item in rights),
                "unit": length_unit.unit_id,
                "positions": np.where(mask[:, None], coordinates, 0.0).tolist(),
                "coverage": mask.tolist(),
                "images": image_policy,
            }
        )
        for name, value in fields.items():
            object.__setattr__(self, name, value)

    def prepare_binding(self):
        return prepare_nucleotide_binding(
            self.mapping, self.mapping.atom_ids, coordinate_mask=self.coordinate_mask
        )

    def require_rights(self, requested_use=None):
        return tuple(
            item.require_rights(**({} if requested_use is None else requested_use))
            for item in self.rights
        )


@dataclass(frozen=True, slots=True)
class NormalizedNucleicHypothesis:
    raw: NucleicStructureHypothesis
    normalized: NucleicStructureHypothesis
    report: AdapterReport


def normalize_nucleic_hypothesis(
    hypothesis, *, length_unit, atom_order=None, requested_use=None
) -> NormalizedNucleicHypothesis:
    """Only explicit unit conversion/order normalization; never imputation.

    The raw hypothesis, confidence provider and restrictions remain unchanged.
    Normalized coordinates are a distinct derived artifact, not a trajectory.
    """
    hypothesis.require_rights(requested_use)
    source_order = hypothesis.mapping.atom_ids
    order = tuple(sorted(source_order)) if atom_order is None else tuple(atom_order)
    if len(order) != len(source_order) or set(order) != set(source_order):
        raise ValueError("Normalization must preserve every atom exactly once.")
    rows = [source_order.index(atom) for atom in order]
    factor = float(conversion_factor(hypothesis.length_unit, length_unit))
    mapping = NucleotideAtomMapping(
        hypothesis.mapping.construct,
        order,
        tuple(hypothesis.mapping.nucleotide_keys[i] for i in rows),
        tuple(hypothesis.mapping.atom_names[i] for i in rows),
    )
    positions = hypothesis.positions[jnp.asarray(rows)] * factor
    mask = hypothesis.coordinate_mask[jnp.asarray(rows)]
    digest = canonical_fingerprint(
        {"parent": hypothesis.hypothesis_id, "order": order, "unit": length_unit.unit_id}
    )
    source = ScientificArtifactEnvelope(
        artifact_kind="normalized-nucleic-structure",
        content_digest=digest,
        producer="phydrax.nucleic-acid-biophysics",
        producer_version="native",
        build_id="explicit-coordinate-normalization",
        license_id=hypothesis.source.license_id,
        resource_id=hypothesis.source.resource_id,
        status="complete",
        parent_artifact_ids=(hypothesis.source.artifact_id,),
    )
    normalized = NucleicStructureHypothesis(
        mapping,
        positions,
        length_unit,
        source,
        hypothesis.rights,
        coordinate_mask=mask,
        image_policy=hypothesis.image_policy,
        requested_use=requested_use,
        parent=hypothesis,
    )
    changes = []
    if order != source_order:
        changes.append(
            AdapterLoss(
                "atoms",
                "import",
                "transformed",
                "Explicit stable-ID row permutation; reverse correspondence retained.",
                changes_interpretation=False,
            )
        )
    if hypothesis.length_unit.unit_id != length_unit.unit_id:
        changes.append(
            AdapterLoss(
                "positions",
                "import",
                "transformed",
                "Exact-unit scale converted into target floating representation.",
                changes_interpretation=False,
            )
        )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if changes else AdapterStatus.LOSSLESS,
        "resolved-nucleotide-records",
        "normalized-nucleotide-records",
        source_id=hypothesis.hypothesis_id,
        target_id=normalized.hypothesis_id,
        coordinate_mapping=tuple(
            f"{atom}:{i}->{order.index(atom)}" for i, atom in enumerate(source_order)
        ),
        preserved_fields=(
            "construct",
            "atom-identity",
            "chemistry",
            "coverage",
            "source",
            "rights",
            "raw-coordinates",
        ),
        losses=tuple(changes),
        stage="nucleic-normalization",
    )
    return NormalizedNucleicHypothesis(hypothesis, normalized, report)


__all__ = [
    "NucleicStructureHypothesis",
    "NormalizedNucleicHypothesis",
    "normalize_nucleic_hypothesis",
]
