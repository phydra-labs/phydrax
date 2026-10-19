#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Identity-preserving sharp offsets of rational planar solids."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import isqrt

import numpy as np

from ...discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from ...linalg._small_batched import prepare_exact_small_linear_actions
from .._planar_primitives import reserve_fraction_work
from .._rational import fraction_endpoint, rational_cross, rational_vector
from ._analytic_cylinder_closest import sqrt_bounds
from ._constructors import _Face, BRepTessellationPolicy, Builder, finish
from ._edge_classification import brep_edge_classification
from ._model import BRepCurve, BRepEntityId, BRepGeometry, BRepModel
from ._offset_result import BRepOffsetResult
from ._offset_rounding import offset_rounding_evidence, publish_endpoint_rounding
from ._parameter_range import (
    attach_parameter_range,
    CertifiedParameterRange,
    ParameterRangeConstraint,
    ParameterRangeEndStatus,
    unknown_parameter_range,
)
from ._patches import (
    AbstractSurfacePatch,
    BSplineCurve,
    ExtrusionSurface,
    LineCurve,
    PlanePatch,
)
from ._request_preparations import TrimRequestPreparations


def planar_carrier(patch: AbstractSurfacePatch, /) -> PlanePatch | None:
    """Recognize a plane independently of an affine sweep's chart scale."""
    if isinstance(patch, PlanePatch):
        return patch
    if isinstance(patch, ExtrusionSurface):
        curve = patch.curve
        if isinstance(curve, BSplineCurve) and curve.degree == 1:
            controls, weights = (
                np.asarray(curve.control_points),
                np.asarray(curve.weights),
            )
            if controls.shape == (2, 3) and weights[0] == weights[1]:
                return PlanePatch(controls[0], controls[1] - controls[0], patch.direction)
    return None


def _dot(a: tuple[Fraction, ...], b: tuple[Fraction, ...], /) -> Fraction:
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction())


def _normal(
    patch: PlanePatch, orientation: int, /, *, face: int | None = None
) -> tuple[Fraction, ...]:
    normal = rational_cross(
        rational_vector(np.asarray(patch.first_axis)),
        rational_vector(np.asarray(patch.second_axis)),
    )
    squared = _dot(normal, normal)
    numerator, denominator = isqrt(squared.numerator), isqrt(squared.denominator)
    if (
        numerator * numerator != squared.numerator
        or denominator * denominator != squared.denominator
        or numerator == 0
    ):
        raise ValueError(
            f"brep_offset.planar_normal must have a rational unit normal; got unavailable face {face}"
        )
    length = Fraction(numerator, denominator)
    return tuple(orientation * value / length for value in normal)


def _solve(
    matrix: tuple[tuple[Fraction, ...], ...],
    right: tuple[Fraction, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Fraction, ...]:
    proof = prepare_exact_small_linear_actions(
        matrix, tuple((value,) for value in right), budget=budget
    )
    if proof.actions is None:
        raise ValueError(
            "brep_offset.planar_topology must have independent incident planes; got a singular system"
        )
    return tuple(row[0] for row in proof.actions)


def planar_offset(
    model: BRepModel,
    distance: float,
    /,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    include_parameter_range: bool = False,
) -> BRepOffsetResult:
    """Intersect translated planes, proving every source incidence unchanged.

    Incident-plane equality and strict edge-direction signs certify the local
    face lattice. Rational nonadjacent-face separation bounds certify the global
    collision-free isotopy, including concave shells and pocket boundaries.
    """
    geometry = model.require_geometry()
    classifications = brep_edge_classification(
        model, geometry, maximum_cells=1, coordinate_budget=budget
    )
    if any(record.status not in ("convex", "concave") for record in classifications):
        raise ValueError(
            "brep_offset.planar_edges must have certified convexity; got an unresolved classification"
        )
    planes = []
    carriers = tuple(planar_carrier(patch) for patch in model.patches)
    for face, patch in enumerate(carriers):
        if patch is None:
            raise ValueError(
                f"brep_offset.planar_faces must have planar carriers; got curved sharp face {face} requiring blend faces"
            )
        normal = _normal(patch, int(np.asarray(model.orientation)[face]), face=face)
        origin = rational_vector(np.asarray(patch.origin))
        planes.append((normal, _dot(normal, origin)))
    delta = Fraction(distance)
    vertices = _offset_vertices(geometry, tuple(planes), delta, budget=budget)
    try:
        _certify_planar_lattice(geometry, tuple(vertices), delta, budget=budget)
    except CoordinateEnclosureResourceError:
        raise
    except ValueError as error:
        raise attach_parameter_range(
            error, planar_parameter_range(model, tuple(planes), budget=budget)
        )
    builder = Builder()
    builder.vertices = [
        np.asarray([float(value) for value in point], dtype=np.float64)
        for point in vertices
    ]
    for first, last in geometry.edge_vertices:
        if vertices[first] == vertices[last]:
            raise ValueError(
                "brep_offset.planar_topology must preserve nonvanishing faces; got a face vanishes at the requested distance"
            )
        direction = tuple(
            b - a for a, b in zip(vertices[first], vertices[last], strict=True)
        )
        builder.edge(
            LineCurve(
                builder.vertices[first], tuple(float(value) for value in direction)
            ),
            0.0,
            1.0,
            first,
            last,
        )
    for face, patch in enumerate(carriers):
        if patch is None:
            raise RuntimeError(
                "brep_offset.face must retain its plane carrier; got another patch family"
            )
        _stage_planar_face(
            builder,
            model,
            face,
            patch,
            tuple(vertices),
            preparations=preparations,
            budget=budget,
        )
    publish_endpoint_rounding(builder, budget=budget)
    evidence, rounding_margin = offset_rounding_evidence(
        builder,
        geometry,
        tuple(vertices),
        tuple(normal for normal, _ in planes),
        tuple(int(value) for value in np.asarray(model.orientation)),
        budget=budget,
    )
    if rounding_margin > 0.0:
        try:
            _certify_planar_lattice(
                geometry,
                tuple(vertices),
                delta,
                rounding_margin=Fraction(rounding_margin),
                budget=budget,
            )
        except CoordinateEnclosureResourceError:
            raise
        except ValueError as error:
            raise attach_parameter_range(
                error, planar_parameter_range(model, tuple(planes), budget=budget)
            )
    published = finish(
        builder,
        solid=True,
        coordinate_contract=model.coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        shells=(geometry.shell_faces, geometry.shell_orientations, geometry.solid_shells),
        preparations=preparations,
        budget=budget,
    )
    return BRepOffsetResult(
        published,
        evidence,
        parameter_range=(
            planar_parameter_range(model, tuple(planes), budget=budget)
            if include_parameter_range
            else None
        ),
    )


def planar_parameter_range(
    model: BRepModel,
    planes: tuple[tuple[tuple[Fraction, ...], Fraction], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CertifiedParameterRange:
    """Solve affine edge-sign roots and the lattice's global separation bounds."""
    geometry = model.require_geometry()
    original = tuple(
        rational_vector(point) for point in np.asarray(geometry.vertex_points)
    )
    zero = _offset_vertices(geometry, planes, Fraction(), budget=budget)
    if zero != original:
        return unknown_parameter_range(
            "distance", "source vertices differ from exact incident-plane intersections"
        )
    one = _offset_vertices(geometry, planes, Fraction(1), budget=budget)
    velocity = tuple(
        tuple(b - a for a, b in zip(first, last, strict=True))
        for first, last in zip(zero, one, strict=True)
    )
    reserve_fraction_work(
        (*original, *velocity),
        32 * len(geometry.edge_vertices),
        16 * len(geometry.face_loops) ** 2,
        64,
        budget=budget,
    )
    candidates = _edge_range_candidates(model, zero, velocity)
    separation = _separation_range_candidates(model, original)
    if separation is None:
        return unknown_parameter_range(
            "distance",
            "nonadjacent source face boxes have no strict separation certificate",
        )
    return _publish_planar_range(model, candidates, separation)


type _RangeCandidate = tuple[
    Fraction, ParameterRangeConstraint, tuple[BRepEntityId, ...], ParameterRangeEndStatus
]


def _publish_planar_range(
    model: BRepModel,
    candidates: tuple[_RangeCandidate, ...],
    separation: tuple[_RangeCandidate, ...],
    /,
) -> CertifiedParameterRange:
    """Reduce binding certificates and round the selected endpoints inward."""
    limits: list[Fraction | None] = [None, None]
    constraints: list[ParameterRangeConstraint] = ["unknown", "unknown"]
    statuses: list[ParameterRangeEndStatus] = ["attained_exactly", "attained_exactly"]
    entities: list[tuple[BRepEntityId, ...]] = [(), ()]
    for root, constraint, ids, status in (*candidates, *separation):
        end = 0 if root < 0 else 1
        current = limits[end]
        if current is None or abs(root) < abs(current):
            limits[end] = root
            constraints[end] = constraint
            entities[end] = ids
            statuses[end] = status
    interval = []
    for end, limit in enumerate(limits):
        value = (
            (-np.inf if end == 0 else np.inf)
            if limit is None
            else fraction_endpoint(limit, upper=end == 0, saturate=True)
        )
        if limit is not None and Fraction(value) != limit:
            statuses[end] = "certified_inner_bound"
        interval.append(value)
    if interval[0] >= interval[1]:
        return unknown_parameter_range(
            "distance", "no nonempty binary64 inner interval is representable"
        )
    return CertifiedParameterRange(
        "distance",
        (interval[0], interval[1]),
        end_status=(statuses[0], statuses[1]),
        constraints=(constraints[0], constraints[1]),
        entity_ids=(entities[0], entities[1]),
        evidence=(
            "exact affine incident-plane solves; positive source-edge dot products; "
            "nonadjacent source face-box separation exceeds twice absolute distance; "
            "zero is the unchanged source, excluded by the construction's nonzero-distance API",
            f"source_model_id={model.model_id}; rational lower={limits[0]}; rational upper={limits[1]}",
            *(
                f"collision separation pair {ids!r}: absolute distance < {abs(root)}"
                for root, _, ids, _ in separation
                if root < 0
            ),
        ),
    )


def _planar_face_boxes(
    geometry: BRepGeometry,
    original: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[
    tuple[frozenset[int], ...],
    tuple[tuple[tuple[Fraction, Fraction], ...], ...],
]:
    """Keep face vertex inventories and their rational coordinate boxes aligned."""
    face_vertices = tuple(
        frozenset(
            vertex
            for loop in loops
            for coedge in loop
            for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
        )
        for loops in geometry.face_loops
    )
    boxes = tuple(
        tuple(
            (
                min(original[index][axis] for index in indices),
                max(original[index][axis] for index in indices),
            )
            for axis in range(3)
        )
        for indices in face_vertices
    )
    return face_vertices, boxes


def _edge_range_candidates(
    model: BRepModel,
    zero: tuple[tuple[Fraction, ...], ...],
    velocity: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[_RangeCandidate, ...]:
    """Locate exact zeros of each unchanged edge-direction certificate."""
    candidates: list[_RangeCandidate] = []
    for edge, (first, last) in enumerate(model.require_geometry().edge_vertices):
        direction = tuple(b - a for a, b in zip(zero[first], zero[last], strict=True))
        derivative = tuple(
            b - a for a, b in zip(velocity[first], velocity[last], strict=True)
        )
        slope = _dot(direction, derivative)
        if slope != 0:
            candidates.append(
                (
                    -_dot(direction, direction) / slope,
                    "trim_vanishes",
                    (model.edge_ids[edge],),
                    "attained_exactly",
                )
            )
    return tuple(candidates)


def _separation_range_candidates(
    model: BRepModel,
    original: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[_RangeCandidate, ...] | None:
    """Bound the absolute distance with the same disjoint-face boxes as construction."""
    geometry = model.require_geometry()
    face_vertices, boxes = _planar_face_boxes(geometry, original)
    candidates: list[_RangeCandidate] = []
    for first, second in combinations(range(len(face_vertices)), 2):
        if face_vertices[first] & face_vertices[second]:
            continue
        squared = sum(
            (
                max(Fraction(), b[0] - a[1], a[0] - b[1]) ** 2
                for a, b in zip(boxes[first], boxes[second], strict=True)
            ),
            Fraction(),
        )
        bound = sqrt_bounds(squared)[0] / 2
        if bound == 0:
            return None
        ids = (model.face_ids[first], model.face_ids[second])
        candidates.extend(
            (
                (-bound, "edges_collide", ids, "certified_inner_bound"),
                (bound, "edges_collide", ids, "certified_inner_bound"),
            )
        )
    return tuple(candidates)


def _certify_planar_lattice(
    geometry: BRepGeometry,
    vertices: tuple[tuple[Fraction, ...], ...],
    delta: Fraction,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    rounding_margin: Fraction = Fraction(),
) -> None:
    original = tuple(
        rational_vector(np.asarray(point)) for point in geometry.vertex_points
    )
    reserve_fraction_work(
        (*original, *vertices, (delta,)),
        24 * len(geometry.edge_vertices),
        12 * len(geometry.face_loops) ** 2,
        16,
        budget=budget,
    )
    reversed_edge = False
    for first, last in geometry.edge_vertices:
        old = tuple(b - a for a, b in zip(original[first], original[last], strict=True))
        new = tuple(b - a for a, b in zip(vertices[first], vertices[last], strict=True))
        sign = _dot(old, new)
        if sign == 0:
            raise ValueError(
                "brep_offset.planar_topology must preserve nonvanishing faces; got a face vanishes at the requested distance"
            )
        reversed_edge = reversed_edge or sign < 0
        if sign > 0 and sign * sign <= _dot(old, old) * (2 * rounding_margin) ** 2:
            raise ValueError(
                "brep_offset.edge_direction must exceed its coefficient-rounding margin; got an unresolved direction"
            )
    face_vertices, boxes = _planar_face_boxes(geometry, original)
    for first, second in combinations(range(len(face_vertices)), 2):
        if face_vertices[first] & face_vertices[second]:
            continue
        # Disjoint coordinate boxes provide a rational lower bound on the
        # polygon-polygon distance, independent of convexity and face holes.
        squared = sum(
            (
                max(Fraction(), b[0] - a[1], a[0] - b[1]) ** 2
                for a, b in zip(boxes[first], boxes[second], strict=True)
            ),
            Fraction(),
        )
        if squared <= (2 * abs(delta) + 2 * rounding_margin) ** 2:
            raise ValueError(
                f"brep_offset.planar_topology must preserve nonadjacent separation; got collision pair ({first}, {second}) without a certificate"
            )
    if reversed_edge:
        raise ValueError(
            "brep_offset.planar_topology must preserve edge direction; got a reversed edge at the requested distance"
        )


def _stage_planar_face(
    builder: Builder,
    model: BRepModel,
    face: int,
    patch: PlanePatch,
    vertices: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> None:
    geometry = model.require_geometry()
    anchor = geometry.edge_vertices[
        geometry.coedge_edges[geometry.face_loops[face][0][0]]
    ][0]
    origin = vertices[anchor]
    first_axis = rational_vector(np.asarray(patch.first_axis))
    second_axis = rational_vector(np.asarray(patch.second_axis))
    translated = PlanePatch(
        tuple(float(value) for value in origin), patch.first_axis, patch.second_axis
    )
    gram = (
        (_dot(first_axis, first_axis), _dot(first_axis, second_axis)),
        (_dot(second_axis, first_axis), _dot(second_axis, second_axis)),
    )
    coordinates: dict[int, tuple[Fraction, ...]] = {}
    for source_loop in geometry.face_loops[face]:
        for coedge in source_loop:
            for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]:
                if vertex not in coordinates:
                    displacement = tuple(
                        value - center
                        for value, center in zip(vertices[vertex], origin, strict=True)
                    )
                    coordinates[vertex] = _solve(
                        gram,
                        (_dot(first_axis, displacement), _dot(second_axis, displacement)),
                        budget=budget,
                    )
    loops = []
    for source_loop in geometry.face_loops[face]:
        loop = []
        for coedge in source_loop:
            edge = geometry.coedge_edges[coedge]
            start, end = geometry.edge_vertices[edge]
            uv = coordinates[start]
            direction = tuple(b - a for a, b in zip(uv, coordinates[end], strict=True))
            loop.append(
                builder.coedge(
                    edge,
                    geometry.coedge_senses[coedge],
                    LineCurve(
                        tuple(float(value) for value in uv),
                        tuple(float(value) for value in direction),
                    ),
                    preparations=preparations,
                    budget=budget,
                )
            )
        loops.append(loop)
    points = np.asarray(
        [[float(value) for value in uv] for uv in coordinates.values()], dtype=np.float64
    )
    box = np.stack((points.min(axis=0), points.max(axis=0)))
    builder.faces.append(
        _Face(
            translated,
            box,
            loops,
            int(np.asarray(model.orientation)[face]),
            "planar-offset",
        )
    )


def _affine_edge(curve: BRepCurve, /) -> bool:
    match curve:
        case LineCurve():
            return True
        case BSplineCurve(degree=1):
            return (
                curve.control_points.shape[0] == 2
                and np.asarray(curve.weights)[0] == np.asarray(curve.weights)[1]
            )
        case _:
            return False


def _offset_vertices(
    geometry: BRepGeometry,
    planes: tuple[tuple[tuple[Fraction, ...], Fraction], ...],
    delta: Fraction,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Fraction, ...], ...]:
    """Solve each incident plane system and certify retained vertex incidence."""
    incident: list[set[int]] = [set() for _ in geometry.vertex_points]
    for face, loops in enumerate(geometry.face_loops):
        for coedge in (coedge for loop in loops for coedge in loop):
            edge = geometry.coedge_edges[coedge]
            if not _affine_edge(geometry.curves[geometry.edge_curves[edge]]):
                raise ValueError(
                    "brep_offset.planar_edges must be affine lines; got another curve family"
                )
            for vertex in geometry.edge_vertices[edge]:
                incident[vertex].add(face)
    vertices = []
    for faces in incident:
        candidate = None
        for triple in combinations(sorted(faces), 3):
            proof = prepare_exact_small_linear_actions(
                tuple(planes[face][0] for face in triple),
                tuple((planes[face][1] + delta,) for face in triple),
                budget=budget,
            )
            if proof.actions is not None:
                candidate = tuple(row[0] for row in proof.actions)
                break
        if candidate is None:
            raise ValueError(
                "brep_offset.planar_topology must have three independent incident planes; got an unsupported vertex"
            )
        for face in faces:
            normal, height = planes[face]
            if _dot(normal, candidate) != height + delta:
                raise ValueError(
                    "brep_offset.planar_topology must preserve vertex incidence; got a nonsimple changed vertex"
                )
        vertices.append(candidate)
    return tuple(vertices)
