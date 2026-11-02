# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact source-copy ancestry for a published surface bridge."""

from __future__ import annotations

from ..._fingerprint import canonical_fingerprint
from .._cad_revision import (
    AssociationCoverageEvidence,
    AssociationGraph,
    OccurrenceCorrespondence,
    OccurrenceCorrespondenceTransaction,
)
from ._bridge_publication import BridgePublication
from ._model import BRepModel
from ._partition import _entity_occurrence_id, cad_revision_from_brep_model


def bridge_ancestry(
    models: tuple[BRepModel, BRepModel], publication: BridgePublication, /
) -> tuple[AssociationGraph, AssociationGraph]:
    """Associate each source with its explicitly copied output entities.

    The publication maps source definition indices to output definition indices;
    they do not prove ancestry for assembly occurrences or generated topology.
    Unmapped occurrences remain unresolved in each graph, including entities
    copied from the other source. No orientation correspondence is asserted.
    """
    if len(models) != 2 or not all(isinstance(model, BRepModel) for model in models):
        raise ValueError("Bridge ancestry requires exactly two BRepModel sources.")
    if not isinstance(publication, BridgePublication):
        raise TypeError("publication must be a BridgePublication.")
    target = publication.model
    target_revision = cad_revision_from_brep_model(target)
    target_ids = {occurrence.occurrence_id for occurrence in target_revision.occurrences}
    maps = (
        ("vertex_ids", publication.source_vertex_indices),
        ("edge_ids", publication.source_edge_indices),
        ("face_ids", publication.source_face_indices),
    )
    if any(len(indices) != 2 for _, indices in maps):
        raise ValueError("Bridge publication must provide a map for each source.")

    graphs = []
    for source_index, model in enumerate(models):
        source_revision = cad_revision_from_brep_model(model)
        source_ids = {
            occurrence.occurrence_id for occurrence in source_revision.occurrences
        }
        pairs = []
        for attribute, indices in maps:
            source_entities = getattr(model, attribute)
            target_entities = getattr(target, attribute)
            mapping = indices[source_index]
            if len(mapping) != len(source_entities):
                raise ValueError(
                    f"Bridge publication {attribute} map has the wrong length."
                )
            for source_entity, target_index in zip(source_entities, mapping, strict=True):
                if isinstance(target_index, bool) or not isinstance(target_index, int):
                    raise TypeError("Bridge publication indices must be integers.")
                if target_index < 0 or target_index >= len(target_entities):
                    raise ValueError(
                        "Bridge publication index is outside the target inventory."
                    )
                source_id = _entity_occurrence_id(source_entity)
                target_id = _entity_occurrence_id(target_entities[target_index])
                if source_id not in source_ids or target_id not in target_ids:
                    raise ValueError(
                        "Bridge copy map does not identify qualified inventory occurrences."
                    )
                pairs.append((source_id, target_id))
        if len(set(pairs)) != len(pairs):
            raise ValueError("Bridge publication repeats an ancestry correspondence.")
        certificate = canonical_fingerprint(
            {
                "kind": "surface-bridge-source-copy-ancestry",
                "source_revision": source_revision.revision_id,
                "source_geometry": source_revision.provenance_id,
                "target_revision": target_revision.revision_id,
                "target_geometry": target_revision.provenance_id,
                "correspondences": sorted(pairs),
            }
        )
        correspondences = tuple(
            OccurrenceCorrespondence(
                source_id, target_id, certificate, orientation_transform=0
            )
            for source_id, target_id in pairs
        )
        transaction = OccurrenceCorrespondenceTransaction(
            certificate,
            source_revision.revision_id,
            target_revision.revision_id,
            correspondences,
            frozenset(source_ids - {source_id for source_id, _ in pairs}),
            frozenset(target_ids - {target_id for _, target_id in pairs}),
            AssociationCoverageEvidence(
                False, False, certificate, "native-surface-bridge-copy"
            ),
        )
        graphs.append(AssociationGraph(source_revision, target_revision, transaction))
    return graphs[0], graphs[1]
