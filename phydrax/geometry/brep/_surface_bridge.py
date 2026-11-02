# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Prescribed-boundary parametric spline surface bridges.

This operation does not construct a rolling-ball fillet.  It admits full native
isoparametric boundaries only, and certifies C1 or C2 in the explicitly supplied
affine collar charts; it does not solve general G1/G2 reparameterization.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from ...discretization._coordinate_enclosure import ExactScalar
from .._cad_revision import AssociationGraph
from ._bridge_ancestry import bridge_ancestry
from ._bridge_embedding_proof import BridgeEmbeddingEvidence, BridgeSeparation
from ._hermite_bridge import HermiteBridgeEvidence, Side, build_hermite_bridge_certified
from ._model import BRepEntityId, BRepModel
from ._patches import BSplineSurfacePatch
from ._bridge_publication import BridgePublicationRefusal, publish_bridge


@dataclass(frozen=True, slots=True)
class BRepSurfaceBridgeEvidence:
    """Source-bound whole-seam construction and full-patch embedding evidence."""

    source_revisions: tuple[str, str]
    source_faces: tuple[BRepEntityId, BRepEntityId]
    source_edges: tuple[BRepEntityId, BRepEntityId]
    sides: tuple[Side, Side]
    along: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    collars: tuple[Fraction, Fraction]
    continuity: int
    hermite: HermiteBridgeEvidence
    embedding: BridgeEmbeddingEvidence


@dataclass(frozen=True, slots=True)
class BRepSurfaceBridgeResult:
    """A certified open model, or a refusal with no published model."""

    status: Literal["certified", "unsupported", "unresolved"]
    reason: str
    model: BRepModel | None
    evidence: BRepSurfaceBridgeEvidence | None
    ancestry: tuple[AssociationGraph, AssociationGraph] | None = None


def brep_surface_bridge(
    left: BRepModel,
    right: BRepModel,
    left_face: BRepEntityId,
    left_edge: BRepEntityId,
    right_face: BRepEntityId,
    right_edge: BRepEntityId,
    /,
    *,
    left_side: Side,
    right_side: Side,
    left_along: tuple[Fraction, Fraction],
    right_along: tuple[Fraction, Fraction],
    left_collar: Fraction,
    right_collar: Fraction,
    continuity: int,
    projection: tuple[
        tuple[ExactScalar, ExactScalar, ExactScalar],
        tuple[ExactScalar, ExactScalar, ExactScalar],
    ],
    source_projections: tuple[
        tuple[
            tuple[ExactScalar, ExactScalar, ExactScalar],
            tuple[ExactScalar, ExactScalar, ExactScalar],
        ],
        tuple[
            tuple[ExactScalar, ExactScalar, ExactScalar],
            tuple[ExactScalar, ExactScalar, ExactScalar],
        ],
    ],
    seams: tuple[BridgeSeparation, BridgeSeparation],
    source_separator: tuple[tuple[ExactScalar, ExactScalar, ExactScalar], ExactScalar],
    source_id: str = "native-surface-bridge",
) -> BRepSurfaceBridgeResult:
    """Bridge two explicit source edges with certified parametric C1 or C2.

    The first bridge coordinate is the common seam coordinate on [0, 1].
    Each along pair maps its endpoints to the source's varying parameter;
    collars are positive exact inward source-parameter scales.  At the left
    boundary the bridge transverse jet points out of its source, at the right
    boundary it points into its source.  Knot degree and expanded knot vectors
    must agree after these maps; insufficient knot continuity and refinement
    are refused. Positive homogeneous bridge weights, source regularity,
    globally injective charts and source separation are proved exactly.
    Only the explicitly admitted full rectangular open-face profile is supported;
    arbitrary trims, corner networks, G3 and rolling-ball contacts are not.
    """
    if continuity not in (1, 2) or isinstance(continuity, bool):
        raise ValueError("continuity must be 1 or 2")
    if not source_id:
        raise ValueError("source_id must be nonempty")
    models = (left, right)
    faces = (left_face, right_face)
    edges = (left_edge, right_edge)
    for model, face, edge in zip(models, faces, edges, strict=True):
        if not isinstance(model, BRepModel):
            raise TypeError("bridge sources must be BRepModel instances")
        model.require_current()
        if face not in model.face_ids or edge not in model.edge_ids:
            raise ValueError(
                "source face and edge must belong to the supplied model revision"
            )
    patches = tuple(
        model.patches[face.index] for model, face in zip(models, faces, strict=True)
    )
    if not all(isinstance(patch, BSplineSurfacePatch) for patch in patches):
        return BRepSurfaceBridgeResult(
            "unsupported", "source faces must be rational B-spline surfaces", None, None
        )
    left_patch, right_patch = patches
    if not isinstance(left_patch, BSplineSurfacePatch) or not isinstance(
        right_patch, BSplineSurfacePatch
    ):
        raise RuntimeError("validated source patch type changed")
    try:
        bridge, hermite = build_hermite_bridge_certified(
            left_patch,
            right_patch,
            left_side=left_side,
            right_side=right_side,
            left_along=left_along,
            right_along=right_along,
            left_collar=left_collar,
            right_collar=right_collar,
            continuity=continuity,
        )
    except (ValueError, ArithmeticError) as error:
        return BRepSurfaceBridgeResult("unsupported", str(error), None, None)
    publication = publish_bridge(
        models,
        faces,
        edges,
        (left_side, right_side),
        (left_along, right_along),
        bridge,
        projection=projection,
        source_projections=source_projections,
        seams=seams,
        source_separator=source_separator,
        source_id=source_id,
    )
    if isinstance(publication, BridgePublicationRefusal):
        return BRepSurfaceBridgeResult(publication.status, publication.reason, None, None)
    evidence = BRepSurfaceBridgeEvidence(
        (left.source_revision, right.source_revision),
        faces,
        edges,
        (left_side, right_side),
        (left_along, right_along),
        (left_collar, right_collar),
        continuity,
        hermite,
        publication.embedding_evidence,
    )
    return BRepSurfaceBridgeResult(
        "certified", "", publication.model, evidence, bridge_ancestry(models, publication)
    )
