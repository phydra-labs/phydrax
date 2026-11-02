#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Occurrence-preserving exact endpoint reconstruction for source repair.

The source coordinates and parameter ranges are selection hints, not equality
claims.  Every returned point belongs to the rebuilt carriers exactly; the
caller must separately certify a bound on the resulting source-to-target map.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import atan2, isfinite, tau
from typing import cast

import numpy as np
import numpy.typing as npt

from ._intersection import ConicParameterEndpoint, NativePeriodEndpoint
from ._model import BRepGeometry, BRepModel
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    CircleCurve,
    CylinderPatch,
    LineCurve,
    PlanePatch,
    TorusPatch,
)
from ._root_bindings import BRepVertexRoot
from ._source_repair_roots import (
    _circle_data,
    _point_on_circle,
    circle_circle_points,
    circle_line_points,
    conic_binding,
    endpoint_on_carrier,
    ExactPoint,
    line_parameter,
)
from ._trim_helpers import period_offset, period_scalar_bounds


type RepairEndpoint = ConicParameterEndpoint | NativePeriodEndpoint | None
type RepairEndpointPair = tuple[RepairEndpoint, RepairEndpoint]


@dataclass(frozen=True, slots=True)
class TargetTopology:
    vertex_points: npt.NDArray[np.float64]
    vertex_exact_points: tuple[tuple[Fraction, Fraction, Fraction] | None, ...]
    vertex_roots: tuple[BRepVertexRoot | None, ...]
    edge_endpoint_roots: tuple[RepairEndpointPair, ...]
    coedge_endpoint_roots: tuple[RepairEndpointPair, ...]
    edge_ranges: npt.NDArray[np.float64]
    parameter_bounds: npt.NDArray[np.float64]


def _angle(circle: CircleCurve, point: ExactPoint, hint: float) -> float:
    center, first, second, radius = _circle_data(circle)
    delta = tuple(point[i] - center[i] for i in range(3))
    cosine = sum((delta[i] * first[i] for i in range(3)), Fraction(0)) / radius
    sine = sum((delta[i] * second[i] for i in range(3)), Fraction(0)) / radius
    angle = atan2(float(sine), float(cosine)) % tau
    return angle + round((hint - angle) / tau) * tau


def _line_intersection(first: LineCurve, second: LineCurve) -> ExactPoint | None:
    a, u, b, v = (
        first.origin_exact,
        first.direction_exact,
        second.origin_exact,
        second.direction_exact,
    )
    if not all(isinstance(x, Fraction) for x in (*a, *u, *b, *v)):
        raise ValueError("Target line incidence requires rational coefficients.")
    for i in range(3):
        for j in range(i + 1, 3):
            determinant = u[j] * v[i] - u[i] * v[j]
            if determinant:
                t = ((b[j] - a[j]) * v[i] - (b[i] - a[i]) * v[j]) / determinant
                point = cast(ExactPoint, tuple(a[k] + t * u[k] for k in range(3)))
                line_parameter(second, point)
                return point
    return None


def _incidence(
    geometry: BRepGeometry, patches: tuple[AbstractSurfacePatch, ...]
) -> list[list[tuple[int, int, int]]]:
    edges = geometry.edge_vertices
    uses = [[] for _ in edges]
    seen = set()
    for face, loops in enumerate(geometry.face_loops):
        if (
            not isinstance(patches[face], (PlanePatch, CylinderPatch, TorusPatch))
            or not loops
        ):
            raise ValueError("Unsupported target face or empty face boundary.")
        for loop in loops:
            if not loop:
                raise ValueError("Empty source loop.")
            oriented = []
            for coedge in loop:
                if coedge in seen or not 0 <= coedge < len(geometry.coedge_edges):
                    raise ValueError("Source coedge occurrence is duplicated or invalid.")
                seen.add(coedge)
                edge = geometry.coedge_edges[coedge]
                if not 0 <= edge < len(edges):
                    raise ValueError("Invalid source edge occurrence.")
                sense = geometry.coedge_senses[coedge]
                if sense not in (-1, 1):
                    raise ValueError("Invalid coedge orientation.")
                a, b = edges[edge]
                oriented.append((a, b) if sense == 1 else (b, a))
                uses[edge].append((face, coedge, sense))
            if any(
                oriented[i][1] != oriented[(i + 1) % len(oriented)][0]
                for i in range(len(oriented))
            ):
                raise ValueError("Source coedge loop is not topologically closed.")
    if len(seen) != len(geometry.coedge_edges):
        raise ValueError("A source coedge has no face occurrence.")
    for edge_uses in uses:
        if len(edge_uses) != 2:
            raise ValueError("Source edge is not a two-use manifold edge.")
        if edge_uses[0][0] == edge_uses[1][0]:
            if edge_uses[0][2] == edge_uses[1][2] or isinstance(
                patches[edge_uses[0][0]], PlanePatch
            ):
                raise ValueError("Unsupported or inconsistently oriented source seam.")
    return uses


def target_topology(
    model: BRepModel,
    curves: tuple[AbstractCurve, ...],
    patches: tuple[AbstractSurfacePatch, ...],
    pcurves: tuple[AbstractCurve, ...],
    /,
) -> TargetTopology:
    """Rebuild target vertices, endpoint witnesses and chart bounds in source order.

    Curves are indexed by source edge, patches by face and pcurves by coedge.
    Coincident carriers do not isolate a vertex: another incident carrier or a
    native cardinal endpoint is required. Ambiguous nearest choices are refused.
    """
    geometry = model.geometry
    if (
        geometry is None
        or len(curves) != len(geometry.edge_vertices)
        or len(patches) != len(geometry.face_loops)
        or len(pcurves) != len(geometry.coedge_edges)
    ):
        raise ValueError("Target carrier tables do not match source topology.")
    _incidence(geometry, patches)
    if any(
        not isinstance(curve, (CircleCurve, LineCurve)) or curve.ambient_dimension != 3
        for curve in curves
    ):
        raise ValueError("Unsupported target edge carrier.")
    incident = [[] for _ in geometry.vertex_points]
    for edge, (a, b) in enumerate(geometry.edge_vertices):
        if not (0 <= a < len(incident) and 0 <= b < len(incident)):
            raise ValueError("Invalid source edge vertex.")
        incident[a].append(edge)
        if b != a:
            incident[b].append(edge)
    source_points = np.asarray(geometry.vertex_points, dtype=float)
    source_ranges = np.asarray(geometry.edge_ranges, dtype=float)
    exact_points, vertex_roots, points = [], [], []
    for vertex, edges in enumerate(incident):
        if not edges:
            raise ValueError("Unreferenced source vertex is unsupported.")
        circles = [curves[e] for e in edges if isinstance(curves[e], CircleCurve)]
        lines = [curves[e] for e in edges if isinstance(curves[e], LineCurve)]
        candidates = []
        for i, circle in enumerate(circles):
            for other in circles[i + 1 :]:
                try:
                    candidates.extend(circle_circle_points(circle, other))
                except ValueError as error:
                    if "Coincident circle carriers" not in str(error):
                        raise
            for line in lines:
                candidates.extend(circle_line_points(circle, line))
        if not candidates and circles:
            # The completion policy explicitly replaces a rootless chart phase
            # by its unique nearest native quarter-turn branch. This is not a
            # claim that the rounded source phase or point was cardinal.
            branches = []
            for edge in edges:
                circle = curves[edge]
                if not isinstance(circle, CircleCurve):
                    continue
                center, first, second, radius = _circle_data(circle)
                for endpoint, source_vertex in enumerate(geometry.edge_vertices[edge]):
                    if source_vertex != vertex:
                        continue
                    phase = float(source_ranges[edge, endpoint])
                    quarter = int(round(phase / (tau / 4)))
                    _, lower_high = period_scalar_bounds(
                        period_offset(Fraction(0), Fraction(2 * quarter - 1, 8))
                    )
                    upper_low, _ = period_scalar_bounds(
                        period_offset(Fraction(0), Fraction(2 * quarter + 1, 8))
                    )
                    phase_exact = Fraction.from_float(phase)
                    if (
                        not Fraction.from_float(float(lower_high))
                        < phase_exact
                        < Fraction.from_float(float(upper_low))
                    ):
                        raise ValueError("Source phase has an ambiguous cardinal branch.")
                    branch = quarter % 4
                    c, s = ((1, 0), (0, 1), (-1, 0), (0, -1))[branch]
                    branches.append(
                        cast(
                            ExactPoint,
                            tuple(
                                center[i] + radius * (c * first[i] + s * second[i])
                                for i in range(3)
                            ),
                        )
                    )
            if not branches or any(point != branches[0] for point in branches[1:]):
                raise ValueError(
                    "Source occurrences do not select one cardinal target branch."
                )
            candidates.append(branches[0])
        if not candidates and lines:
            for i, line in enumerate(lines):
                for other in lines[i + 1 :]:
                    point = _line_intersection(line, other)
                    if point is not None:
                        candidates.append(point)
            if not candidates:
                edge = edges[0]
                endpoint = 0 if geometry.edge_vertices[edge][0] == vertex else 1
                parameter = Fraction.from_float(float(source_ranges[edge, endpoint]))
                candidates.append(
                    tuple(
                        x + parameter * d
                        for x, d in zip(
                            lines[0].origin_exact, lines[0].direction_exact, strict=True
                        )
                    )
                )
        valid = []
        for point in candidates:
            if point in valid or not all(_point_on_circle(point, c) for c in circles):
                continue
            try:
                for line in lines:
                    line_parameter(line, point)
            except ValueError:
                continue
            valid.append(point)
        if not valid:
            raise ValueError(
                "Incident target carriers have no supported common exact vertex."
            )
        source = tuple(Fraction.from_float(float(x)) for x in source_points[vertex])
        ranked = sorted(
            (sum(((p[i] - source[i]) ** 2 for i in range(3)), Fraction(0)), n)
            for n, p in enumerate(valid)
        )
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            raise ValueError("Source vertex does not select a unique target root.")
        point = valid[ranked[0][1]]
        binding = None
        if circles:
            edge = next(e for e in edges if curves[e] is circles[0])
            endpoint = 0 if geometry.edge_vertices[edge][0] == vertex else 1
            binding, _ = conic_binding(
                circles[0],
                point,
                _angle(circles[0], point, source_ranges[edge, endpoint]),
            )
        exact_points.append(
            point if all(isinstance(x, Fraction) for x in point) else None
        )
        vertex_roots.append(binding)
        points.append(tuple(float(x) for x in point))
    ranges = np.empty_like(source_ranges)
    edge_roots = []
    for edge, (a, b) in enumerate(geometry.edge_vertices):
        curve = curves[edge]
        pair = []
        for end, vertex in enumerate((a, b)):
            binding = vertex_roots[vertex]
            point = (
                binding.primary.root.exact_point()
                if binding is not None
                else exact_points[vertex]
            )
            if isinstance(curve, CircleCurve):
                _, endpoint = conic_binding(
                    curve, point, _angle(curve, point, source_ranges[edge, end])
                )
                parameter = endpoint.root.parameter
            else:
                parameter_exact = line_parameter(cast(LineCurve, curve), point)
                parameter = float(parameter_exact)
                endpoint = (
                    None
                    if binding is None
                    else endpoint_on_carrier(
                        binding.primary.root, curve, parameter_exact, point
                    )
                )
            pair.append(endpoint)
            ranges[edge, end] = parameter
        if isinstance(curve, CircleCurve) and a == b:
            if ranges[edge, 1] <= ranges[edge, 0]:
                _, pair[1] = conic_binding(
                    curve, point, _angle(curve, point, ranges[edge, 0] + tau)
                )
                ranges[edge, 1] = pair[1].root.parameter
            root = cast(ConicParameterEndpoint, pair[0]).root
            if root.cosine_exact == 1 and root.sine_exact == 0 and root.turns == 0:
                pair = [
                    NativePeriodEndpoint(curve, turns=0),
                    NativePeriodEndpoint(curve, turns=1),
                ]
                ranges[edge] = (0.0, tau)
        if (
            not all(isfinite(x) for x in ranges[edge])
            or ranges[edge, 0] >= ranges[edge, 1]
        ):
            raise ValueError("Rebuilt edge parameters are not strictly ordered.")
        edge_roots.append(cast(RepairEndpointPair, tuple(pair)))
    from ._boolean_core import PeriodicPCurve

    coedge_faces = {}
    for face, loops in enumerate(geometry.face_loops):
        for loop in loops:
            for coedge in loop:
                coedge_faces[coedge] = face
    coedge_roots = []
    for coedge, edge in enumerate(geometry.coedge_edges):
        pair = []
        for root in edge_roots[edge]:
            if isinstance(root, NativePeriodEndpoint):
                pcurve = pcurves[coedge]
                if pcurve.native_period_symbol() == "two_pi":
                    pair.append(
                        NativePeriodEndpoint(
                            pcurve, rational=root.rational, turns=root.turns
                        )
                    )
                else:
                    # The endpoint parameter is the physical edge parameter t.
                    # The patch axis supplies only a native-period authority;
                    # its chart coordinate may have a separate exact phase and
                    # deck shift, which the pcurve binding must compose.
                    patch = patches[coedge_faces[coedge]]
                    axis = 0
                    base = pcurve
                    while isinstance(base, PeriodicPCurve):
                        base = base.source_curve
                    if isinstance(base, LineCurve):
                        directions = base.direction_exact
                        varying = [
                            i for i, direction in enumerate(directions) if direction != 0
                        ]
                        if len(varying) != 1:
                            raise ValueError("Native seam chart direction is ambiguous.")
                        axis = varying[0]
                    pair.append(
                        NativePeriodEndpoint(
                            pcurve, patch, axis, rational=root.rational, turns=root.turns
                        )
                    )
            elif isinstance(root, ConicParameterEndpoint):
                pair.append(
                    ConicParameterEndpoint(
                        root.root, pcurves[coedge], root.parameter_exact, root.point_exact
                    )
                )
            else:
                pair.append(None)
        coedge_roots.append(cast(RepairEndpointPair, tuple(pair)))
    bounds = []
    for loops in geometry.face_loops:
        boxes = []
        for loop in loops:
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                lo, hi = ranges[edge]
                for root in coedge_roots[coedge]:
                    if root is not None:
                        rlo, rhi = root.parameter_enclosure()
                        lo, hi = min(lo, rlo), max(hi, rhi)
                box = np.asarray(pcurves[coedge].bounding_box(lo, hi), dtype=float)
                if box.shape != (2, 2) or not np.all(np.isfinite(box)):
                    raise ValueError("Target pcurve has no finite chart enclosure.")
                boxes.append(box)
        bounds.append(
            np.stack(
                (
                    np.min([b[0] for b in boxes], axis=0),
                    np.max([b[1] for b in boxes], axis=0),
                )
            )
        )
    return TargetTopology(
        np.asarray(points, dtype=float),
        tuple(exact_points),
        tuple(vertex_roots),
        tuple(edge_roots),
        tuple(coedge_roots),
        ranges,
        np.asarray(bounds),
    )
