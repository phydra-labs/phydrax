#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Proof primitives for explicit, occurrence-preserving face completion."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import cast

from ...discretization._exact_algebraic_geometry import ExactScalar
from ._boolean_core import PeriodicPCurve
from ._carrier_pieces import carrier_payload
from ._intersection import ConicParameterEndpoint, NativePeriodEndpoint
from ._model import BRepGeometry, BRepModel
from ._patches import (
    CircleCurve,
    CylinderPatch,
    PlaneCylinderSectionPCurve,
    PlanePatch,
    TorusPatch,
    TorusRootedCirclePCurve,
)
from ._source_repair_trim import (
    annulus_enclosure_bounds,
    AnnulusBoundary,
    completed_profile_enclosure,
    cylinder_enclosure_bounds,
    DirectedFaceBounds,
    Interval,
    phase_interval,
    scalar_interval,
    StripBoundary,
    torus_enclosure_bounds,
)
from ._trim_helpers import period_offset, period_scalar_bounds


def _circle_winding(model: BRepModel, occurrences: tuple[int, ...], /) -> int:
    """Prove that a closed sequence of exact circle arcs winds once simply.

    Exact target vertex incidence establishes integral winding; strictly
    positive, consistently oriented arcs and an interval enclosing their total
    sweep strictly between zero and two turns force its absolute degree to one.
    """
    from ._source_repair_completion import _endpoint_interval, _vertex_point
    from ._source_repair_roots import _circle_data

    geometry = model.geometry
    if not occurrences or len(set(occurrences)) != len(occurrences):
        raise ValueError("A circle boundary needs distinct coedge occurrences.")
    edges = [geometry.coedge_edges[coedge] for coedge in occurrences]
    if len(set(edges)) != len(edges):
        raise ValueError("A simple circle cannot repeat an edge occurrence.")
    curves = [geometry.curves[geometry.edge_curves[edge]] for edge in edges]
    if any(type(curve) is not CircleCurve for curve in curves):
        raise ValueError("Circle boundary contains a non-circular edge.")
    _circle_data(cast(CircleCurve, curves[0]))
    carrier = carrier_payload(curves[0])
    if any(carrier_payload(curve) != carrier for curve in curves[1:]):
        raise ValueError("Circle boundary arcs do not share an exact carrier.")
    senses = [geometry.coedge_senses[coedge] for coedge in occurrences]
    if any(sense not in (-1, 1) for sense in senses) or len(set(senses)) != 1:
        raise ValueError("Circle boundary does not have a consistent traversal.")
    vertices = [geometry.edge_vertices[edge] for edge in edges]
    oriented = [
        pair if sense == 1 else pair[::-1]
        for pair, sense in zip(vertices, senses, strict=True)
    ]
    if any(
        oriented[i][1] != oriented[(i + 1) % len(oriented)][0]
        for i in range(len(oriented))
    ):
        raise ValueError("Circle boundary is not topologically closed.")
    lower = upper = Fraction(0)
    for edge in edges:
        curve = cast(CircleCurve, geometry.curves[geometry.edge_curves[edge]])
        for endpoint, vertex in enumerate(geometry.edge_vertices[edge]):
            root = geometry.edge_endpoint_roots[edge][endpoint]
            point = _vertex_point(model, vertex)
            if isinstance(root, ConicParameterEndpoint):
                if root.point_exact != point:
                    raise ValueError("Circle endpoint does not bind its target vertex.")
            elif isinstance(root, NativePeriodEndpoint):
                if root.rational != 0 or root.turns.denominator != 1:
                    raise ValueError(
                        "Native circle endpoint is not an exact cardinal phase."
                    )
                expected = tuple(
                    a + curve.radius_exact * b
                    for a, b in zip(
                        curve.center_exact, curve.first_axis_exact, strict=True
                    )
                )
                if point != expected:
                    raise ValueError(
                        "Native circle endpoint does not bind its target vertex."
                    )
            else:
                raise ValueError("Circle endpoint lacks an exact incidence witness.")
        first = _endpoint_interval(model, edge, 0, repaired=True)
        last = _endpoint_interval(model, edge, 1, repaired=True)
        arc_lower, arc_upper = last[0] - first[1], last[1] - first[0]
        if arc_lower <= 0:
            raise ValueError("Circle arc parameter order is not proved.")
        lower += arc_lower
        upper += arc_upper
    period_lower, _ = period_scalar_bounds(period_offset(Fraction(0), Fraction(1)))
    if lower <= 0 or upper >= 2 * Fraction.from_float(float(period_lower)):
        raise ValueError("Circle boundary winding is not uniquely one turn.")
    return senses[0]


def _strip_rings(
    geometry: BRepGeometry, face: int, /
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Extract the two closed ring runs and the oppositely used chart seam."""
    loops = geometry.face_loops[face]
    if len(loops) != 1 or not loops[0]:
        raise ValueError("A strip requires one nonempty authored loop.")
    loop = tuple(loops[0])
    edges = tuple(geometry.coedge_edges[coedge] for coedge in loop)
    repeated = {edge for edge in edges if edges.count(edge) > 1}
    if len(repeated) != 1:
        raise ValueError("A strip requires exactly one repeated seam edge.")
    seam = next(iter(repeated))
    positions = [i for i, edge in enumerate(edges) if edge == seam]
    if len(positions) != 2:
        raise ValueError("A strip seam must have exactly two occurrences.")
    first, last = positions
    if geometry.coedge_senses[loop[first]] != -geometry.coedge_senses[loop[last]]:
        raise ValueError("The two seam occurrences must have opposite senses.")
    first_ring = loop[first + 1 : last]
    second_ring = loop[last + 1 :] + loop[:first]
    if not first_ring or not second_ring:
        raise ValueError("A strip must have two nonempty ring boundaries.")
    return first_ring, second_ring, seam


def _annulus_rings(
    geometry: BRepGeometry, face: int, /
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Extract the two distinct complete annular boundary loops."""
    loops = geometry.face_loops[face]
    if len(loops) != 2 or any(not loop for loop in loops):
        raise ValueError("An annulus requires exactly two nonempty loops.")
    first, second = tuple(loops[0]), tuple(loops[1])
    if set(first) & set(second):
        raise ValueError("Annulus loops cannot share an occurrence.")
    return first, second


def _strip_level(
    geometry: BRepGeometry, ring: tuple[int, ...], /, *, torus: bool
) -> Interval:
    """Read a common exact latitude from every target ring occurrence."""
    values = []
    for coedge in ring:
        chart = geometry.pcurves[coedge]
        if isinstance(chart, PeriodicPCurve):
            if chart.period_shifts[1]:
                raise ValueError(
                    "A shifted profile level requires explicit root sheet evidence."
                )
            chart = chart.source_curve
        if torus:
            if not isinstance(chart, TorusRootedCirclePCurve):
                raise ValueError("Torus ring lacks an exact latitude root.")
            root = chart.root_line.root
            key = (root.cosine_exact, root.sine_exact, root.turns)
            interval = phase_interval(root)
        else:
            if (
                not isinstance(chart, PlaneCylinderSectionPCurve)
                or chart.linear_exact[1] != 0
                or chart.cosine_exact[1] != 0
                or chart.sine_exact[1] != 0
            ):
                raise ValueError("Cylinder ring does not have constant exact height.")
            key = chart.offset_exact[1]
            interval = scalar_interval(key)
        values.append((key, interval))
    if any(key != values[0][0] for key, _ in values[1:]):
        raise ValueError("A ring's target occurrences do not share one profile level.")
    return values[0][1]


@dataclass(frozen=True, slots=True)
class FaceCompletion:
    source_to_target: Fraction
    target_to_source: Fraction
    connector_bound: Fraction
    payload: dict[str, object]


def _cross(
    a: tuple[ExactScalar, ...], b: tuple[ExactScalar, ...]
) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple[ExactScalar, ...], b: tuple[ExactScalar, ...]) -> ExactScalar:
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _ring_data(
    model: BRepModel, ring: tuple[int, ...], axis: tuple[ExactScalar, ...]
) -> tuple[CircleCurve, ExactScalar, frozenset[int]]:
    geometry = model.geometry
    winding = _circle_winding(model, ring)
    edge = geometry.coedge_edges[ring[0]]
    curve = cast(CircleCurve, geometry.curves[geometry.edge_curves[edge]])
    normal = _cross(curve.first_axis_exact, curve.second_axis_exact)
    alignment = _dot(normal, axis)
    if alignment not in (-1, 1) or normal != tuple(alignment * x for x in axis):
        raise ValueError("A boundary circle is not exactly normal to the common axis.")
    return curve, winding * alignment, frozenset(geometry.coedge_edges[c] for c in ring)


def _payload_fraction(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _payload_interval(interval: Interval) -> list[list[int]]:
    return [_payload_fraction(value) for value in interval]


def _strip_face(
    source: BRepModel, target: BRepModel, face: int
) -> tuple[DirectedFaceBounds, dict[str, object]]:
    first, second = source.patches[face], target.patches[face]
    torus = isinstance(first, TorusPatch) and isinstance(second, TorusPatch)
    if not torus and not (
        isinstance(first, CylinderPatch) and isinstance(second, CylinderPatch)
    ):
        raise ValueError("Source and target do not have corresponding strip supports.")
    geometry = target.geometry
    rings = _strip_rings(geometry, face)
    seam = rings[2]
    loop = geometry.face_loops[face][0]
    seam_uses = [c for c in loop if geometry.coedge_edges[c] == seam]
    charts = [geometry.pcurves[c] for c in seam_uses]
    if any(not isinstance(chart, PeriodicPCurve) for chart in charts):
        raise ValueError(
            "The target seam requires two explicit periodic chart occurrences."
        )
    charts = [cast(PeriodicPCurve, chart) for chart in charts]
    if carrier_payload(charts[0].source_curve) != carrier_payload(charts[1].source_curve):
        raise ValueError("Target seam occurrences do not share the same chart carrier.")
    shifts = [chart.period_shifts for chart in charts]
    if shifts[0][1] != shifts[1][1] or abs(shifts[0][0] - shifts[1][0]) != 1:
        raise ValueError("Target seam occurrences must differ by one longitude turn.")
    axis = second.axis_exact
    data = [
        (_strip_level(geometry, ring, torus=torus), _ring_data(target, ring, axis), ring)
        for ring in rings[:2]
    ]
    data.sort(key=lambda item: item[0][0])
    (low, low_data, low_ring), (high, high_data, high_ring) = data
    if low[1] >= high[0] or low_data[1] != -high_data[1]:
        raise ValueError("Strip boundary order or opposite winding is not proved.")
    # The target seam cuts the full circular product; seam phase is immaterial
    # to its image. It is not used to identify a rounded chart value as zero.
    boundary = StripBoundary(
        low, high, (Fraction(0), Fraction(0)), low_data[2], high_data[2], seam
    )
    enclosure = completed_profile_enclosure(low, high, periodic=torus)
    source_frame = (first.first_axis_exact, first.second_axis_exact, first.axis_exact)
    target_frame = (second.first_axis_exact, second.second_axis_exact, second.axis_exact)
    if torus:
        first_torus, second_torus = first, second
        result = torus_enclosure_bounds(
            first_torus.center_exact,
            second_torus.center_exact,
            source_frame,
            target_frame,
            first_torus.major_radius_exact,
            second_torus.major_radius_exact,
            first_torus.minor_radius_exact,
            second_torus.minor_radius_exact,
            enclosure,
            boundary,
        )
    else:
        first_cylinder = cast(CylinderPatch, first)
        second_cylinder = cast(CylinderPatch, second)
        result = cylinder_enclosure_bounds(
            first_cylinder.origin_exact,
            second_cylinder.origin_exact,
            source_frame,
            target_frame,
            first_cylinder.radius_exact,
            second_cylinder.radius_exact,
            enclosure,
            boundary,
        )
    return result, {
        "kind": "full-torus-strip" if torus else "full-cylinder-strip",
        "lower_occurrences": list(low_ring),
        "upper_occurrences": list(high_ring),
        "seam_edge": seam,
        "seam_occurrences": seam_uses,
        "seam_shifts": [list(shift) for shift in shifts],
        "lower_profile": _payload_interval(low),
        "upper_profile": _payload_interval(high),
        "windings": [int(cast(Fraction, low_data[1])), int(cast(Fraction, high_data[1]))],
    }


def _annulus_face(
    source: BRepModel, target: BRepModel, face: int
) -> tuple[DirectedFaceBounds, dict[str, object]]:
    first, second = source.patches[face], target.patches[face]
    if not isinstance(first, PlanePatch) or not isinstance(second, PlanePatch):
        raise ValueError("Source and target do not have corresponding plane supports.")
    if carrier_payload(first) != carrier_payload(second):
        raise ValueError(
            "A planar completion requires the same exact source and target plane."
        )
    rings = _annulus_rings(target.geometry, face)
    geometry = target.geometry
    seed = geometry.curves[geometry.edge_curves[geometry.coedge_edges[rings[0][0]]]]
    if type(seed) is not CircleCurve:
        raise ValueError("Annulus boundary is not circular.")
    axis = _cross(seed.first_axis_exact, seed.second_axis_exact)
    data = [(_ring_data(target, ring, axis), ring) for ring in rings]
    data.sort(key=lambda item: scalar_interval(item[0][0].radius_exact)[0])
    (inner, inner_ring), (outer, outer_ring) = data
    if inner[0].center_exact != outer[0].center_exact or inner[1] != -outer[1]:
        raise ValueError("Annulus boundaries are not concentric with opposite winding.")
    inner_radius, outer_radius = (
        scalar_interval(inner[0].radius_exact),
        scalar_interval(outer[0].radius_exact),
    )
    boundary = AnnulusBoundary(
        inner[0].radius_exact, outer[0].radius_exact, inner[2], outer[2]
    )
    enclosure = completed_profile_enclosure(inner_radius, outer_radius, radial=True)
    frame = (seed.first_axis_exact, seed.second_axis_exact, axis)
    center = seed.center_exact
    result = annulus_enclosure_bounds(center, center, frame, frame, enclosure, boundary)
    return result, {
        "kind": "full-planar-annulus",
        "inner_occurrences": list(inner_ring),
        "outer_occurrences": list(outer_ring),
        "inner_radius": _payload_interval(inner_radius),
        "outer_radius": _payload_interval(outer_radius),
        "windings": [int(cast(Fraction, inner[1])), int(cast(Fraction, outer[1]))],
    }


def _face_incidence(geometry: BRepGeometry, face: int) -> set[int]:
    seen = set()
    for loop in geometry.face_loops[face]:
        if not loop:
            raise ValueError("An authored face loop cannot be empty.")
        oriented = []
        for coedge in loop:
            if coedge in seen or not 0 <= coedge < len(geometry.coedge_edges):
                raise ValueError("An authored coedge occurrence is invalid or repeated.")
            seen.add(coedge)
            edge = geometry.coedge_edges[coedge]
            if not 0 <= edge < len(geometry.edge_vertices):
                raise ValueError("An authored coedge refers to an invalid edge.")
            sense = geometry.coedge_senses[coedge]
            if sense not in (-1, 1):
                raise ValueError("An authored coedge has an invalid sense.")
            pair = geometry.edge_vertices[edge]
            oriented.append(pair if sense == 1 else pair[::-1])
        if any(
            oriented[i][1] != oriented[(i + 1) % len(oriented)][0]
            for i in range(len(oriented))
        ):
            raise ValueError("An authored loop is not topologically closed.")
    return seen


def certify_face_completions(
    source: BRepModel, target: BRepModel, /, *, tolerance: Fraction
) -> tuple[FaceCompletion, ...]:
    """Certify explicit full-product completions, not the original gapped fill."""
    from ._source_repair import _require_exact_closed
    from ._source_repair_chart_completion import certify_chart_completion
    from ._source_repair_completion import _require_same_incidence

    if not isinstance(source, BRepModel) or not isinstance(target, BRepModel):
        raise TypeError("Face completion requires two BRepModel values.")
    if not isinstance(tolerance, Fraction) or tolerance <= 0:
        raise ValueError("Face completion tolerance must be a positive Fraction.")
    source.require_current()
    target.require_current()
    _require_same_incidence(source, target)
    _require_exact_closed(target)
    seen = set()
    records = []
    for face, patch in enumerate(source.patches):
        domain = target.trim_domains[face]
        if domain is None or not domain.curve_loops:
            raise ValueError(
                "A repaired face requires source-bound exact curve trim loops."
            )
        occurrences = _face_incidence(source.geometry, face)
        if seen & occurrences:
            raise ValueError("A coedge is used by more than one face.")
        seen.update(occurrences)
        if isinstance(patch, PlanePatch):
            bounds, detail = _annulus_face(source, target, face)
        elif isinstance(patch, (CylinderPatch, TorusPatch)):
            bounds, detail = _strip_face(source, target, face)
        else:
            raise ValueError("Unsupported source face completion support.")
        connector, chart_payload = certify_chart_completion(
            source,
            target,
            face,
            tolerance=tolerance,
        )
        if not isinstance(connector, Fraction) or connector < 0:
            raise ValueError(
                "Chart completion did not return an exact nonnegative connector bound."
            )
        if max(connector, bounds.source_to_target, bounds.target_to_source) > tolerance:
            raise ValueError("Face completion exceeds the declared tolerance.")
        detail.update(
            {
                "face": face,
                "interpretation": "target-chart-product-pulled-back-to-source-support",
                "original_source_region": "not-asserted",
                "source_to_target": _payload_fraction(bounds.source_to_target),
                "target_to_source": _payload_fraction(bounds.target_to_source),
                "connector_bound": _payload_fraction(connector),
                "chart_correspondence": chart_payload,
            }
        )
        records.append(
            FaceCompletion(
                bounds.source_to_target, bounds.target_to_source, connector, detail
            )
        )
    if seen != set(range(len(source.geometry.coedge_edges))):
        raise ValueError("Every authored coedge must belong to exactly one face.")
    return tuple(records)
