#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact circular B-Rep rim occurrences and their native parameter traces.

This does not by itself prove the interior regularity or all shared traces of
any material mesh. The disk and edge proofs are separately reusable premises.
"""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple

import numpy as np

from ..discretization._exact_algebraic_geometry import coefficient_interval, ExactScalar
from ._domain_boundary import exact_plane_chart, ExactPlaneChart
from ._meshing_domain import (
    _PlacedSurfacePatch,
    MeshingDomain,
    PatchCurveUse,
    PatchPoleUse,
)
from ._surface_source_support import (
    _cad_root_edges,
    SurfaceSourceCharts,
    validate_cad_curve_ancestry,
)
from .brep._assembly import BRepAffinePlacement
from .brep._boolean_core import chart_add, chart_scale, ChartScalar
from .brep._carrier_pieces import carrier_content_id
from .brep._correspondence import (
    _exact_brep_pole_collapse,
    _exact_vertex_root_source,
    exact_source_lift,
)
from .brep._intersection import NativePeriodEndpoint
from .brep._intersection_curve import SurfaceRegion
from .brep._patches import (
    _validate_source_shadows,
    AbstractCurve,
    AbstractSurfacePatch,
    CircleCurve,
)
from .brep._source_embedding import certify_brep_trim_sheet


class CircleRimEdge(NamedTuple):
    """Canonical mesh-edge orientation and its exact source-circle phase."""

    curve: CircleCurve
    curve_row: int
    first: ChartScalar
    last: ChartScalar
    cap_cell: int
    side_cell: int
    cap_slot: int
    side_slot: int


def prove_circle_cap_disk(
    domain: MeshingDomain, patch: int, curve_row: int, curve: CircleCurve, /
) -> ExactPlaneChart:
    """Prove the actual trimmed cap is exactly this circle's closed disk.

    The Gram identity uses the retained exact coefficients, including non-unit
    source frame shadows; it does not assume that a floating-point frame is
    exactly orthonormal.
    """
    domain.require_current()
    plane = exact_plane_chart(domain, patch)
    sheet = certify_brep_trim_sheet(domain, patch)
    if (
        plane is None
        or sheet is None
        or sheet.embedding.family != "revolution-plane"
        or len(plane.loops) != 1
        or len(plane.loops[0]) != 1
        or plane.loops[0][0].curve != curve_row
        or not isinstance(plane.loops[0][0].pcurve, CircleCurve)
        or sum(
            isinstance(use, PatchPoleUse)
            for loop in domain.patches[patch].loops
            for use in loop
        )
        != 1
    ):
        raise ValueError("The source face is not a proved singly bounded disk.")
    model = domain.brep_authority
    if model is None:
        raise ValueError("A circular cap needs current B-Rep authority.")
    pole_loop, pole_index, pole = next(
        (loop_index, use_index, use)
        for loop_index, loop in enumerate(domain.patches[patch].loops)
        for use_index, use in enumerate(loop)
        if isinstance(use, PatchPoleUse)
    )
    face = int(domain.source_indices[2][patch])
    geometry = model.geometry
    coedge = geometry.face_loops[face][pole_loop][pole_index]
    edge = geometry.coedge_edges[coedge]
    vertex, other = geometry.edge_vertices[edge]
    if (
        geometry.edge_curves[edge] >= 0
        or vertex != other
        or int(domain.source_indices[0][pole.corner]) != vertex
    ):
        raise ValueError("The cap pole changes its exact B-Rep occurrence.")
    first, last = (float(value) for value in np.asarray(geometry.edge_ranges[edge]))
    exact_vertex = geometry.vertex_exact_points[vertex]
    if exact_vertex is None:
        exact_vertex, _ = _exact_vertex_root_source(geometry.vertex_roots[vertex])
    if exact_vertex != tuple(curve.center_exact) or not _exact_brep_pole_collapse(
        None,
        model.patches[face],
        geometry.pcurves[coedge],
        first,
        last,
        geometry.vertex_exact_points[vertex],
        geometry.vertex_roots[vertex],
        geometry.coedge_endpoint_roots[coedge],
    ):
        raise ValueError("The cap pole is not proved to collapse to the circle center.")
    pole_start, pole_stop = _phase_roots(
        geometry.coedge_endpoint_roots[coedge], geometry.pcurves[coedge]
    )
    if (
        pole_start.patch is None
        or pole_start.axis != 0
        or carrier_content_id(pole_start.patch) != carrier_content_id(model.patches[face])
        or pole_start.rational != 0
        or pole_start.turns != 0
        or pole_stop.turns != 1
    ):
        raise ValueError("The cap pole does not retain its complete native turn.")
    _validate_source_shadows(curve, set())
    radius = plane.loops[0][0].pcurve.radius_exact
    if tuple(curve.center_exact) != plane.origin:
        raise ValueError("The rim circle changes the exact cap center.")
    for i in range(3):
        for j in range(3):
            rim = (
                curve.radius_exact
                * curve.radius_exact
                * (
                    curve.first_axis_exact[i] * curve.first_axis_exact[j]
                    + curve.second_axis_exact[i] * curve.second_axis_exact[j]
                )
            )
            disk = (
                radius
                * radius
                * (
                    plane.first_axis[i] * plane.first_axis[j]
                    + plane.second_axis[i] * plane.second_axis[j]
                )
            )
            if rim != disk:
                raise ValueError("The source rim changes the exact trimmed disk frame.")
    return plane


def require_circle_disk_point(
    curve: CircleCurve, point: tuple[ExactScalar, ExactScalar, ExactScalar], /
) -> None:
    """Prove an exact point is in the closed disk bounded by ``curve``."""
    if len(point) != 3:
        raise ValueError("A disk point must be three-dimensional.")
    first, second = curve.first_axis_exact, curve.second_axis_exact
    offset = tuple(point[i] - curve.center_exact[i] for i in range(3))

    def dot(a: tuple[ExactScalar, ...], b: tuple[ExactScalar, ...], /) -> ExactScalar:
        return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))

    aa, ab, bb = dot(first, first), dot(first, second), dot(second, second)
    determinant = aa * bb - ab * ab
    lower, _ = coefficient_interval(determinant, bits=128)
    if lower <= 0:
        raise ValueError("The exact rim frame has no proved planar rank.")
    ap, bp = dot(first, offset), dot(second, offset)
    u = (bb * ap - ab * bp) / determinant
    v = (aa * bp - ab * ap) / determinant
    if any(offset[i] != u * first[i] + v * second[i] for i in range(3)):
        raise ValueError("A cap material point is not in the exact rim plane.")
    margin = curve.radius_exact * curve.radius_exact - u * u - v * v
    if margin != 0 and coefficient_interval(margin, bits=128)[0] <= 0:
        raise ValueError("A cap material point is not proved inside the trimmed disk.")


def _source_patch(surface: AbstractSurfacePatch, /) -> AbstractSurfacePatch:
    if isinstance(surface, _PlacedSurfacePatch):
        if (
            surface.placement.identity_payload()
            != BRepAffinePlacement.identity().identity_payload()
        ):
            raise ValueError("A placed rim needs an exact placement transport.")
        return surface.source
    return surface


def _phase_roots(
    roots: object, carrier: AbstractCurve, /
) -> tuple[NativePeriodEndpoint, NativePeriodEndpoint]:
    if (
        not isinstance(roots, tuple)
        or len(roots) != 2
        or any(not isinstance(root, NativePeriodEndpoint) for root in roots)
    ):
        raise ValueError("A source rim requires two native period endpoint roots.")
    first, last = roots
    if not isinstance(first, NativePeriodEndpoint) or not isinstance(
        last, NativePeriodEndpoint
    ):
        raise ValueError("A source rim requires two native period endpoint roots.")
    if (
        carrier_content_id(first.curve) != carrier_content_id(carrier)
        or carrier_content_id(last.curve) != carrier_content_id(carrier)
        or first.rational != last.rational
        or last.turns - first.turns != 1
        or first.axis != last.axis
        or ((first.patch is None) != (last.patch is None))
    ):
        raise ValueError("A source rim changes its exact native phase or carrier.")
    if first.patch is not None:
        if (
            last.patch is None
            or first.axis is None
            or carrier_content_id(first.patch) != carrier_content_id(last.patch)
            or first.patch.native_period_symbols()[first.axis] != "two_pi"
        ):
            raise ValueError("A source rim changes its native period surface.")
    elif first.curve.native_period_symbol() != "two_pi":
        raise ValueError("A source rim has no native period.")
    return first, last


def prove_circle_rim_edges(
    charts: SurfaceSourceCharts, tolerance: float, /
) -> dict[int, CircleRimEdge]:
    """Replay exact coedge lifts and return directed phases for all cap rims.

    Parameters are aligned to the canonical mesh-edge vertex order.  The two
    source faces must independently lift the very same spatial circle and the
    same native phase; endpoint sampling or rounded values are not a lift proof.
    This function does not authorize any cell map by itself.
    """
    charts.require_current()
    model = charts.domain.brep_authority
    if model is None:
        raise ValueError("Circular rims require current B-Rep authority.")
    model.require_current()
    bindings = validate_cad_curve_ancestry(charts, tolerance)
    _, adjacency = _cad_root_edges(charts)
    domain, geometry = charts.domain, model.geometry
    rows = {cell: row for row, cell in enumerate(charts.cell_global_ids)}
    output: dict[int, CircleRimEdge] = {}
    planes: dict[int, bool] = {}
    for edge_id, uses in bindings.items():
        if len(uses) != 2 or len(adjacency.get(edge_id, ())) != 2:
            continue
        patch_ids = tuple(
            int(charts.root_patches[rows[use.root_cell_id]]) for use in uses
        )
        if patch_ids[0] == patch_ids[1]:
            continue
        for patch in patch_ids:
            if patch not in planes:
                planes[patch] = exact_plane_chart(domain, patch) is not None
        if sum(planes[patch] for patch in patch_ids) != 1:
            continue
        cap_index = 0 if planes[patch_ids[0]] else 1
        side_index = 1 - cap_index
        if (
            uses[0].curve_row != uses[1].curve_row
            or uses[0].parameters != uses[1].parameters
        ):
            raise ValueError(
                "Both rim faces must retain the same oriented source interval."
            )
        curve_row = uses[0].curve_row
        source_edge = int(domain.source_indices[1][curve_row])
        curve_index = geometry.edge_curves[source_edge]
        if curve_index < 0:
            raise ValueError("A cap rim must retain its exact spatial circle carrier.")
        curve = geometry.curves[curve_index]
        if not isinstance(curve, CircleCurve):
            raise ValueError("A cap rim must retain its exact spatial circle carrier.")
        _validate_source_shadows(curve, set())
        start, stop = _phase_roots(geometry.edge_endpoint_roots[source_edge], curve)
        interval = np.asarray(geometry.edge_ranges[source_edge], dtype=np.float64)
        if tuple(float(value) for value in interval) != (start.parameter, stop.parameter):
            raise ValueError(
                "The rim range is not the projection of its native endpoints."
            )
        owner_patch, owner_loop, owner_use = (
            int(value) for value in domain.curve_owners[curve_row]
        )
        owner = domain.patches[owner_patch].loops[owner_loop][owner_use]
        if not isinstance(owner, PatchCurveUse) or (owner.first, owner.last) != tuple(
            interval
        ):
            raise ValueError("The rim owner changes the native edge parameter gauge.")
        source_coedges = []
        for use, patch in zip(uses, patch_ids, strict=True):
            face = int(domain.source_indices[2][patch])
            face_surface = model.patches[face]
            if carrier_content_id(
                _source_patch(domain.patches[patch].surface)
            ) != carrier_content_id(face_surface):
                raise ValueError("A rim face changes its retained surface occurrence.")
            coedge = geometry.face_loops[face][use.loop_index][use.use_index]
            if geometry.coedge_edges[coedge] != source_edge:
                raise ValueError("A rim changes its authored coedge occurrence.")
            occurrence = domain.patches[patch].loops[use.loop_index][use.use_index]
            if not isinstance(occurrence, PatchCurveUse) or occurrence.curve != curve_row:
                raise ValueError("A rim coedge does not bind the source curve row.")
            pcurve = geometry.pcurves[coedge]
            if carrier_content_id(occurrence.pcurve) != carrier_content_id(pcurve):
                raise ValueError("A rim coedge changes its source p-curve.")
            pc_start, pc_stop = _phase_roots(
                geometry.coedge_endpoint_roots[coedge], pcurve
            )
            if pc_start.patch is None or carrier_content_id(
                pc_start.patch
            ) != carrier_content_id(face_surface):
                raise ValueError("A rim phase root changes its source face.")
            if (
                pc_start.exact_parameter != start.exact_parameter
                or pc_stop.exact_parameter != stop.exact_parameter
            ):
                raise ValueError("A rim p-curve changes the native phase parameter.")
            if not exact_source_lift(
                curve,
                pcurve,
                SurfaceRegion(face_surface, model.parameter_bounds[face]),
                float(interval[0]),
                float(interval[1]),
            ):
                raise ValueError(
                    "A rim face lacks an exact full-range spatial source lift."
                )
            source_coedges.append(coedge)
        if (
            geometry.coedge_senses[source_coedges[0]]
            != -geometry.coedge_senses[source_coedges[1]]
        ):
            raise ValueError("A source rim must have opposed oriented coedge uses.")
        delta = chart_add(
            stop.exact_parameter, chart_scale(start.exact_parameter, Fraction(-1))
        )
        first, last = tuple(
            chart_add(start.exact_parameter, chart_scale(delta, value))
            for value in uses[0].parameters
        )
        slots = {cell: slot for cell, slot, _, _ in adjacency[edge_id]}
        output[edge_id] = CircleRimEdge(
            curve,
            curve_row,
            first,
            last,
            uses[cap_index].root_cell_id,
            uses[side_index].root_cell_id,
            slots[uses[cap_index].root_cell_id],
            slots[uses[side_index].root_cell_id],
        )
    return output
