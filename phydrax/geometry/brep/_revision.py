#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Occurrence-qualified CAD revision identities for a BRep model."""

from __future__ import annotations

from ...typing import checked
from .._cad_revision import CADOccurrence, CADRevision
from ._model import BRepEntityId, BRepModel
from ._projection_contracts import brep_entity_id, BRepEntityDimension


def edge_occurrence_id(face_index: int, edge_index: int) -> str:
    return f"face:{face_index}/edge:{edge_index}"


def _face_occurrence_id(solid_index: int, face_index: int) -> str:
    return f"solid:{solid_index}/face:{face_index}"


def _entity_text(entity: BRepEntityId) -> str:
    match entity.kind:
        case "vertex":
            dimension = BRepEntityDimension.VERTEX
        case "edge":
            dimension = BRepEntityDimension.EDGE
        case "face":
            dimension = BRepEntityDimension.FACE
        case "solid":
            dimension = BRepEntityDimension.SOLID
        case "coedge" | "shell":
            raise ValueError(
                f"CADRevision.entity.kind must identify a spatial stratum; got {entity.kind!r}"
            )
        case _:
            raise ValueError(
                "BRepEntityId.kind must be vertex, edge, face, or solid; "
                f"got {entity.kind!r}"
            )
    return brep_entity_id(
        entity.source_revision,
        dimension,
        entity.index,
        occurrence_path=entity.occurrence_path,
    )


def _solid_occurrence_id(index: int) -> str:
    return f"solid:{index}"


@checked
def cad_revision_from_brep_model(model: BRepModel, /) -> CADRevision:
    """Create the region and boundary occurrence inventory of a B-Rep."""

    occurrences: list[CADOccurrence] = []
    if model.topology.num_solids:
        for solid_index, (face_indices, orientations) in enumerate(
            zip(
                model.topology.solid_faces,
                model.topology.solid_face_orientations,
                strict=True,
            )
        ):
            solid_occurrence_id = _solid_occurrence_id(solid_index)
            occurrences.append(
                CADOccurrence(
                    model.source_revision,
                    solid_occurrence_id,
                    _entity_text(model.solid_ids[solid_index]),
                    "solid",
                    (solid_occurrence_id,),
                )
            )
            for face_index, orientation in zip(face_indices, orientations, strict=True):
                face_occurrence_id = _face_occurrence_id(solid_index, face_index)
                occurrences.append(
                    CADOccurrence(
                        model.source_revision,
                        face_occurrence_id,
                        _entity_text(model.face_ids[face_index]),
                        "face",
                        (solid_occurrence_id, face_occurrence_id),
                        solid_occurrence_id,
                        orientation,
                    )
                )
    else:
        for face_index, wires in enumerate(model.topology.face_wires):
            face_occurrence_id = f"face:{face_index}"
            occurrences.append(
                CADOccurrence(
                    model.source_revision,
                    face_occurrence_id,
                    _entity_text(model.face_ids[face_index]),
                    "face",
                    (face_occurrence_id,),
                )
            )
            edge_orientations: dict[int, int] = {}
            for wire in wires:
                for signed_edge_index in wire:
                    edge_index = abs(signed_edge_index) - 1
                    if edge_index in edge_orientations:
                        raise ValueError(
                            "BRepModel.topology.face_wires must use each edge once per face; "
                            f"got repeated edge {edge_index} in face {face_index}"
                        )
                    edge_orientations[edge_index] = 1 if signed_edge_index > 0 else -1
            if set(edge_orientations) != set(model.topology.face_edges[face_index]):
                raise ValueError(
                    "BRepModel.topology.face_edges must match the face wire edges; "
                    f"got {model.topology.face_edges[face_index]!r} for face {face_index} "
                    f"with wire edge orientations {edge_orientations!r}"
                )
            for edge_index in model.topology.face_edges[face_index]:
                edge_occurrence_id_bank = edge_occurrence_id(face_index, edge_index)
                occurrences.append(
                    CADOccurrence(
                        model.source_revision,
                        edge_occurrence_id_bank,
                        _entity_text(model.edge_ids[edge_index]),
                        "edge",
                        (face_occurrence_id, edge_occurrence_id_bank),
                        face_occurrence_id,
                        edge_orientations[edge_index],
                    )
                )
    return CADRevision(
        model.source_revision,
        model.source_id,
        tuple(occurrences),
        model.model_id,
    )
