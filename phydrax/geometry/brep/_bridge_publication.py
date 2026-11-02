# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Private, exact topology publication for a constructed Hermite bridge.

The caller constructs the Hermite surface from these very source carriers.  This
module replays the embedding and edge-lift proofs; it does not certify jets of
an independently supplied surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np

from ...discretization._exact_algebraic_geometry import ExactScalar
from ._bridge_embedding_proof import (
    BridgeEmbeddingEvidence,
    BridgeEmbeddingRefusal,
    BridgeSeparation,
    certify_bridge_embedding,
)
from ._constructors import _Builder, assemble_brep_model
from ._correspondence import exact_source_lift
from ._intersection_curve import SurfaceRegion, exact_curve_parameter_point
from ._model import BRepEntityId, BRepModel
from ._patches import BSplineCurve, BSplineSurfacePatch, ExactLineCurve

type _Projection = tuple[
    tuple[ExactScalar, ExactScalar, ExactScalar],
    tuple[ExactScalar, ExactScalar, ExactScalar],
]


@dataclass(frozen=True, slots=True)
class BridgePublication:
    model: BRepModel
    source_vertex_indices: tuple[tuple[int, ...], tuple[int, ...]]
    source_edge_indices: tuple[tuple[int, ...], tuple[int, ...]]
    source_face_indices: tuple[tuple[int, ...], tuple[int, ...]]
    embedding_evidence: BridgeEmbeddingEvidence


@dataclass(frozen=True, slots=True)
class BridgePublicationRefusal:
    reason: str
    status: Literal["unsupported", "unresolved"]


class _Refuse(ValueError):
    pass


class _Unresolved(_Refuse):
    pass


def _rectangle(
    model: BRepModel,
    face_id: BRepEntityId,
    edge_id: BRepEntityId,
    side: str,
    along: tuple[Fraction, Fraction],
) -> tuple[BSplineSurfacePatch, tuple[int, int], int, int, int]:
    if not isinstance(model, BRepModel):
        raise _Refuse("source must be a BRepModel")
    model.require_current()
    g = model.geometry
    if len(model.patches) != 1 or face_id != model.face_ids[0]:
        raise _Refuse("each source must explicitly select its only face")
    if not isinstance(edge_id, BRepEntityId) or edge_id not in model.edge_ids:
        raise _Refuse("selected edge does not belong to the source revision")
    if g.shell_faces or g.solid_shells or g.assembly_graph.solid_count:
        raise _Refuse("shells, solids and assembly occurrences are unsupported")
    patch = model.patches[0]
    if not isinstance(patch, BSplineSurfacePatch):
        raise _Refuse("source faces must be B-spline patches")
    if side not in ("u0", "u1", "v0", "v1"):
        raise _Refuse("source boundary side must be explicit")
    if len(along) != 2 or any(not isinstance(x, Fraction) for x in along):
        raise _Refuse("along map endpoints must be exact Fractions")
    knots = (patch.exact_u_knots, patch.exact_v_knots)
    degrees = (patch.u_degree, patch.v_degree)
    counts = (len(patch.exact_control_points), len(patch.exact_control_points[0]))
    domain = tuple((knots[i][degrees[i]], knots[i][counts[i]]) for i in range(2))
    bounds = np.asarray(model.parameter_bounds[0])
    if any(
        Fraction.from_float(float(bounds[j, i])) != domain[i][j]
        for i in range(2)
        for j in range(2)
    ):
        raise _Refuse("source trim box must be the exact full patch domain")
    trim = model.trim_domains[0]
    if trim is not None:
        if trim.numeric_only or trim.outer is None or trim.holes or trim.curve_loops:
            raise _Refuse("source trim domain is not an exact full rectangle")
        polygon = np.asarray(trim.outer)
        corners = {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}
        if polygon.shape != (4, 2) or {tuple(row) for row in polygon} != corners:
            raise _Refuse("source trim domain is not the full normalized rectangle")
        if any(sum(abs(polygon[i] - polygon[(i + 1) % 4])) != 1 for i in range(4)):
            raise _Refuse("source trim rectangle has crossed boundary")
    loops = g.face_loops[0]
    if (
        len(loops) != 1
        or len(loops[0]) != 4
        or len(g.coedge_edges) != 4
        or len(g.edge_vertices) != 4
    ):
        raise _Refuse("source must have one four-edge rectangular boundary")
    region = SurfaceRegion(patch, bounds)
    ranges = np.asarray(g.edge_ranges)
    found = {}
    for coedge in loops[0]:
        edge = g.coedge_edges[coedge]
        ci = g.edge_curves[edge]
        pc = g.pcurves[coedge]
        if (
            ci < 0
            or not isinstance(pc, ExactLineCurve)
            or tuple(ranges[edge]) != (0.0, 1.0)
        ):
            raise _Refuse(
                "rectangular trims require unit-range curves and exact affine p-curves"
            )
        a, b = pc.first_exact, pc.last_exact
        axes = [i for i in range(2) if a[i] == b[i] and a[i] in domain[i]]
        if len(axes) != 1:
            raise _Refuse("trim is not a full rectangular domain side")
        axis = axes[0]
        other = 1 - axis
        if {a[other], b[other]} != set(domain[other]):
            raise _Refuse("trim does not cover its complete domain side")
        key = (axis, int(a[axis] == domain[axis][1]))
        if key in found:
            raise _Refuse("rectangular side occurs more than once")
        curve = g.curves[ci]
        if not exact_source_lift(curve, pc, region, 0.0, 1.0):
            raise _Unresolved("source edge lift from its p-curve was not proved")
        for endpoint, vertex in enumerate(g.edge_vertices[edge]):
            point = g.vertex_exact_points[vertex]
            if point is None:
                raise _Refuse("source vertex lacks exact rational coordinates")
            endpoint_point = exact_curve_parameter_point(curve, Fraction(endpoint))
            if endpoint_point is None:
                raise _Unresolved("exact source edge endpoint evaluation unavailable")
            if endpoint_point != point:
                raise _Refuse("source edge endpoint does not match its vertex")
        found[key] = (edge, coedge, a, b)
    key = (0 if side[0] == "u" else 1, int(side[1]))
    edge, coedge, a, b = found[key]
    if edge != edge_id.index:
        raise _Refuse("selected edge is not the explicitly selected source side")
    if (a[1 - key[0]], b[1 - key[0]]) == along:
        direction = 1
    elif (b[1 - key[0]], a[1 - key[0]]) == along:
        direction = -1
    else:
        raise _Refuse("along map does not match the source edge p-curve")
    return patch, key, edge, coedge, direction


def _copy(
    builder: _Builder, model: BRepModel
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    g = model.geometry
    vertices = tuple(
        builder.vertex(
            np.asarray(g.vertex_points[i]),
            exact_point=g.vertex_exact_points[i],
            root=g.vertex_roots[i],
        )
        for i in range(len(g.vertex_points))
    )
    edges = []
    for i, (a, b) in enumerate(g.edge_vertices):
        ci = g.edge_curves[i]
        lo, hi = map(float, np.asarray(g.edge_ranges[i]))
        if ci < 0:
            edges.append(
                builder.degenerate(
                    vertices[a], lo, hi, endpoint_roots=g.edge_endpoint_roots[i]
                )
            )
        else:
            edges.append(
                builder.edge(
                    g.curves[ci],
                    lo,
                    hi,
                    vertices[a],
                    vertices[b],
                    endpoint_roots=g.edge_endpoint_roots[i],
                )
            )
    coedges = tuple(
        builder.coedge(
            edges[e],
            g.coedge_senses[i],
            g.pcurves[i],
            endpoint_roots=g.coedge_endpoint_roots[i],
        )
        for i, e in enumerate(g.coedge_edges)
    )
    faces = tuple(
        builder.face(
            patch,
            np.asarray(model.parameter_bounds[i]),
            int(model.orientation[i]),
            model.physical_tags[i],
            [[coedges[c] for c in loop] for loop in g.face_loops[i]],
        )
        for i, patch in enumerate(model.patches)
    )
    return vertices, tuple(edges), faces


def _transverse_curve(patch: BSplineSurfacePatch, end: int) -> BSplineCurve:
    i = -1 if end else 0
    points = tuple(patch.exact_control_points[i])
    weights = tuple(patch.exact_weights[i])
    knots = tuple(patch.exact_v_knots)
    return BSplineCurve(
        np.asarray(points, dtype=float),
        np.asarray(weights, dtype=float),
        np.asarray(knots, dtype=float),
        patch.v_degree,
        exact_control_points=points,
        exact_weights=weights,
        exact_knots=knots,
    )


def publish_bridge(
    models: tuple[BRepModel, BRepModel],
    faces: tuple[BRepEntityId, BRepEntityId],
    edges: tuple[BRepEntityId, BRepEntityId],
    sides: tuple[str, str],
    along: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    bridge: BSplineSurfacePatch,
    *,
    projection: _Projection,
    source_projections: tuple[_Projection, _Projection],
    seams: tuple[BridgeSeparation, BridgeSeparation],
    source_separator: tuple[tuple[ExactScalar, ExactScalar, ExactScalar], ExactScalar],
    source_id: str,
) -> BridgePublication | BridgePublicationRefusal:
    """Publish a private Hermite operation result, or return a typed refusal.

    Source identities and seam selectors are explicit; source topology is never
    matched by coordinates.  The caller must supply the just-constructed Hermite
    patch, not an arbitrary surface claiming continuity.
    """
    try:
        if any(len(x) != 2 for x in (models, faces, edges, sides, along, seams)):
            raise _Refuse("exactly two source selections are required")
        if (
            models[0] is models[1]
            or models[0].source_revision == models[1].source_revision
        ):
            raise _Refuse("two distinct source definitions are required")
        if (
            models[0].coordinate_contract.spatial_id
            != models[1].coordinate_contract.spatial_id
        ):
            raise _Refuse("source coordinate contracts differ")
        selected = tuple(
            _rectangle(*values)
            for values in zip(models, faces, edges, sides, along, strict=True)
        )
        if not isinstance(bridge, BSplineSurfacePatch):
            raise _Refuse("bridge must be a B-spline patch")
        for i in range(2):
            if seams[i].bridge_edge != (1, i) or seams[i].source_edge != selected[i][1]:
                raise _Refuse(
                    "embedding seam selectors differ from the selected boundaries"
                )
        proof = certify_bridge_embedding(
            bridge,
            selected[0][0],
            selected[1][0],
            projection=projection,
            source_projections=source_projections,
            seams=seams,
            source_separator=source_separator,
        )
        if isinstance(proof, BridgeEmbeddingRefusal):
            raise _Unresolved(proof.reason)
        domain = (
            (
                bridge.exact_u_knots[bridge.u_degree],
                bridge.exact_u_knots[len(bridge.exact_control_points)],
            ),
            (
                bridge.exact_v_knots[bridge.v_degree],
                bridge.exact_v_knots[len(bridge.exact_control_points[0])],
            ),
        )
        if domain != ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(1))):
            raise _Refuse("bridge must have the unit-square domain")
        region = SurfaceRegion(bridge, np.asarray(((0.0, 0.0), (1.0, 1.0))))
        builder = _Builder()
        copies = tuple(_copy(builder, model) for model in models)
        bridge_edges = []
        corners = []
        orientations = []
        for i, (patch, key, edge, coedge, direction) in enumerate(selected):
            g = models[i].geometry
            pc = ExactLineCurve(
                (Fraction(0) if direction == 1 else Fraction(1), Fraction(i)),
                (Fraction(1) if direction == 1 else Fraction(0), Fraction(i)),
            )
            curve = g.curves[g.edge_curves[edge]]
            if not exact_source_lift(curve, pc, region, 0.0, 1.0):
                raise _Unresolved("source edge lift to the bridge seam was not proved")
            seam = builder.coedge(
                copies[i][1][edge], direction * (1 if i == 0 else -1), pc
            )
            bridge_edges.append(seam)
            a, b = g.edge_vertices[edge]
            corners.append(
                (copies[i][0][a], copies[i][0][b])
                if direction == 1
                else (copies[i][0][b], copies[i][0][a])
            )
            orientations.append(
                -int(models[i].orientation[0])
                * g.coedge_senses[coedge]
                * direction
                * (1 if i == 0 else -1)
            )
        if orientations[0] != orientations[1]:
            raise _Refuse("source face orientations are incompatible across the bridge")
        cross = []
        for i in range(2):
            curve = _transverse_curve(bridge, i)
            pc = ExactLineCurve((Fraction(i), Fraction(0)), (Fraction(i), Fraction(1)))
            if not exact_source_lift(curve, pc, region, 0.0, 1.0):
                raise _Unresolved("bridge transverse boundary source lift was not proved")
            for endpoint in range(2):
                point = builder.vertex_exact_points[corners[endpoint][i]]
                endpoint_point = exact_curve_parameter_point(curve, Fraction(endpoint))
                if endpoint_point is None:
                    raise _Unresolved("exact bridge corner evaluation unavailable")
                if endpoint_point != point:
                    raise _Refuse("bridge corner does not equal the source vertex")
            edge = builder.edge(curve, 0.0, 1.0, corners[0][i], corners[1][i])
            cross.append(builder.coedge(edge, 1 if i else -1, pc))
        builder.face(
            bridge,
            ((0.0, 0.0), (1.0, 1.0)),
            orientations[0],
            "bridge",
            [[bridge_edges[0], cross[1], bridge_edges[1], cross[0]]],
        )
        geometry, patches, bounds, signs, tags, trims = builder.geometry(solid=False)
        model = assemble_brep_model(
            geometry,
            patches,
            bounds,
            signs,
            tags,
            coordinate_contract=models[0].coordinate_contract,
            source_id=source_id,
            trim_domains=(models[0].trim_domains[0], models[1].trim_domains[0], trims[2]),
        )
        return BridgePublication(
            model,
            (copies[0][0], copies[1][0]),
            (copies[0][1], copies[1][1]),
            (copies[0][2], copies[1][2]),
            proof,
        )
    except _Unresolved as error:
        return BridgePublicationRefusal(str(error), "unresolved")
    except (ArithmeticError, AttributeError, IndexError, TypeError, ValueError) as error:
        return BridgePublicationRefusal(str(error), "unsupported")
