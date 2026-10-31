#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Identity-preserving sharp offsets of rational planar solids."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import isqrt

import numpy as np

from ...linalg._exact import determinant_exact, solve_exact
from .._rational import fraction_endpoint, rational_cross, rational_vector
from ._constructors import (
    _Builder as Builder,
    assemble_brep_model,
    BRepTessellationPolicy,
)
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
    ExactLineCurve,
    ExtrusionSurface,
    LineCurve,
    PlanePatch,
)
from ._planar_arrangement import inside_loop, on_segment

type _Vector = tuple[Fraction, ...]
type _Plane = tuple[_Vector, Fraction]
type _Box = tuple[tuple[Fraction, Fraction], ...]
type _Contact = tuple[Fraction, BRepEntityId, BRepEntityId]


def planar_carrier(patch: AbstractSurfacePatch, /) -> PlanePatch | None:
    """Recognize the supported native representations of a plane."""
    if isinstance(patch, PlanePatch):
        return patch
    if not isinstance(patch, ExtrusionSurface):
        return None
    curve = patch.curve
    if isinstance(curve, ExactLineCurve):
        direction = tuple(
            last - first
            for first, last in zip(curve.first_exact, curve.last_exact, strict=True)
        )
        return PlanePatch(
            curve.first,
            np.asarray(direction, dtype=np.float64),
            patch.direction,
        )
    if isinstance(curve, LineCurve):
        return PlanePatch(curve.origin, curve.direction, patch.direction)
    if isinstance(curve, BSplineCurve) and curve.degree == 1:
        controls = np.asarray(curve.control_points)
        weights = np.asarray(curve.weights)
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
    matrix: tuple[tuple[Fraction, ...], ...], right: tuple[Fraction, ...], /
) -> tuple[Fraction, ...]:
    try:
        actions = solve_exact(matrix, tuple((value,) for value in right))
    except ValueError as error:
        raise ValueError(
            "brep_offset.planar_topology must have independent incident planes; got a singular system"
        ) from error
    return tuple(row[0] for row in actions)


def planar_offset(
    model: BRepModel,
    distance: float,
    /,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
    include_parameter_range: bool = False,
) -> BRepOffsetResult:
    """Translate planar face planes with a complete local and global proof.

    Exact incident-plane equality and edge-direction signs preserve the local
    face lattice. Exact contact roots and closed-feature membership preserve
    non-incident separation throughout the requested path, including concave
    shells and pockets.
    """
    geometry = model.geometry
    classifications = brep_edge_classification(model, geometry)
    if any(record.status not in ("convex", "concave") for record in classifications):
        raise ValueError(
            "brep_offset.planar_edges must have certified convexity; got an unresolved classification"
        )
    carriers = tuple(planar_carrier(patch) for patch in model.patches)
    planes = _offset_planes(model, carriers)
    delta = Fraction(distance)
    zero = _offset_vertices(geometry, planes, Fraction())
    vertices = _offset_vertices(geometry, planes, delta)
    velocity = tuple(
        tuple((b - a) / delta for a, b in zip(start, end, strict=True))
        for start, end in zip(zero, vertices, strict=True)
    )
    motion = _planar_motion(model, planes, zero, velocity)
    try:
        _certify_planar_lattice(geometry, motion, delta)
    except ValueError as error:
        raise attach_parameter_range(error, planar_parameter_range(model, planes))

    builder = Builder()
    for point in vertices:
        builder.vertex(
            tuple(float(value) for value in point),
            exact_point=(point[0], point[1], point[2]),
        )
    for first, last in geometry.edge_vertices:
        if vertices[first] == vertices[last]:
            raise ValueError(
                "brep_offset.planar_topology must preserve nonvanishing faces; got a face vanishes at the requested distance"
            )
        builder.edge(
            ExactLineCurve(vertices[first], vertices[last]),
            0.0,
            1.0,
            first,
            last,
        )
    staged_faces = []
    pcurves: dict[int, LineCurve] = {}
    for face, patch in enumerate(carriers):
        if patch is None:
            raise RuntimeError(
                "brep_offset.face must retain its plane carrier; got another patch family"
            )
        translated, box, face_pcurves = _stage_planar_face(model, face, patch, vertices)
        staged_faces.append((translated, box))
        pcurves.update(face_pcurves)
    # Coedges keep their source indices so every face loop images identically.
    for coedge, edge in enumerate(geometry.coedge_edges):
        builder.coedge(edge, geometry.coedge_senses[coedge], pcurves[coedge])
    for face, (translated, box) in enumerate(staged_faces):
        builder.face(
            translated,
            box,
            int(np.asarray(model.orientation)[face]),
            "planar-offset",
            [list(loop) for loop in geometry.face_loops[face]],
        )

    edge_vertex_gap_bounds = publish_endpoint_rounding(builder)
    evidence, rounding_margin, edge_face_gap_bounds = offset_rounding_evidence(
        builder,
        geometry,
        vertices,
        tuple(normal for normal, _ in planes),
        tuple(int(value) for value in np.asarray(model.orientation)),
    )
    if rounding_margin > 0.0:
        try:
            _certify_planar_lattice(
                geometry,
                motion,
                delta,
                rounding_margin=Fraction(rounding_margin),
            )
        except ValueError as error:
            raise attach_parameter_range(error, planar_parameter_range(model, planes))

    (
        staged_geometry,
        staged_patches,
        staged_bounds,
        staged_orientation,
        staged_tags,
        staged_trims,
    ) = builder.geometry(
        solid=True,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        assembly_graph=geometry.assembly_graph,
        edge_vertex_gap_bounds=edge_vertex_gap_bounds,
        edge_face_gap_bounds=edge_face_gap_bounds,
    )
    published = assemble_brep_model(
        staged_geometry,
        staged_patches,
        staged_bounds,
        staged_orientation,
        staged_tags,
        coordinate_contract=model.coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        trim_domains=staged_trims,
    )
    return BRepOffsetResult(
        model,
        published,
        evidence,
        parameter_range=(
            planar_parameter_range(model, planes) if include_parameter_range else None
        ),
    )


def _offset_planes(
    model: BRepModel, carriers: tuple[PlanePatch | None, ...], /
) -> tuple[_Plane, ...]:
    """Construct rational, oriented unit-normal face planes."""
    patches: list[PlanePatch] = []
    for face, patch in enumerate(carriers):
        if patch is None:
            raise ValueError(
                f"brep_offset.planar_faces must have planar carriers; got curved sharp face {face} requiring blend faces"
            )
        patches.append(patch)
    orientation = np.asarray(model.orientation)
    planes = []
    for face, patch in enumerate(patches):
        normal = _normal(patch, int(orientation[face]), face=face)
        planes.append((normal, _dot(normal, rational_vector(np.asarray(patch.origin)))))
    return tuple(planes)


def planar_parameter_range(
    model: BRepModel, planes: tuple[_Plane, ...], /
) -> CertifiedParameterRange:
    """Solve affine edge-sign roots and the nearest verified feature contacts."""
    geometry = model.geometry
    original = tuple(
        exact if exact is not None else rational_vector(point)
        for exact, point in zip(
            geometry.vertex_exact_points,
            np.asarray(geometry.vertex_points),
            strict=True,
        )
    )
    zero = _offset_vertices(geometry, planes, Fraction())
    if zero != original:
        return unknown_parameter_range(
            "distance", "source vertices differ from exact incident-plane intersections"
        )
    one = _offset_vertices(geometry, planes, Fraction(1))
    velocity = tuple(
        tuple(b - a for a, b in zip(first, last, strict=True))
        for first, last in zip(zero, one, strict=True)
    )
    candidates = _edge_range_candidates(model, zero, velocity)
    lower = max(
        (candidate[0] for candidate in candidates if candidate[0] < 0),
        default=None,
    )
    upper = min(
        (candidate[0] for candidate in candidates if candidate[0] > 0),
        default=None,
    )
    motion = _planar_motion(model, planes, zero, velocity)
    if _clearance_violation(motion, Fraction(), Fraction()) is not None:
        return unknown_parameter_range(
            "distance",
            "non-incident source boundary features have no strict clearance certificate",
        )
    contacts: list[_RangeCandidate] = []
    for contact in _first_contacts(motion, lower, upper):
        if contact is not None:
            candidate: _RangeCandidate = (
                contact[0],
                "edges_collide",
                (contact[1], contact[2]),
                "attained_exactly",
            )
            contacts.append(candidate)
    return _publish_planar_range(model, candidates, tuple(contacts))


type _RangeCandidate = tuple[
    Fraction,
    ParameterRangeConstraint,
    tuple[BRepEntityId, ...],
    ParameterRangeEndStatus,
]


def _publish_planar_range(
    model: BRepModel,
    candidates: tuple[_RangeCandidate, ...],
    contacts: tuple[_RangeCandidate, ...],
    /,
) -> CertifiedParameterRange:
    """Reduce binding certificates and round selected endpoints inward.

    Trim roots precede contacts, so a contact at the same distance as a
    vanishing trim leaves the trim as the binding constraint.
    """
    limits: list[Fraction | None] = [None, None]
    constraints: list[ParameterRangeConstraint] = ["unknown", "unknown"]
    statuses: list[ParameterRangeEndStatus] = ["attained_exactly", "attained_exactly"]
    entities: list[tuple[BRepEntityId, ...]] = [(), ()]
    for root, constraint, ids, status in (*candidates, *contacts):
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
    evidence = (
        "exact affine incident-plane solves; positive source-edge dot products; "
        "strictly separated non-incident source vertex-face, vertex-edge and "
        "edge-edge pairs; nearest affine vertex-plane, collinear vertex-vertex "
        "and skew edge-line contact roots verified by exact closed-feature membership",
        f"source_model_id={model.model_id}; rational lower={limits[0]}; rational upper={limits[1]}",
        *(
            f"first contact pair {ids!r}: rational distance {root}"
            for root, _, ids, _ in contacts
        ),
    )
    return CertifiedParameterRange(
        "distance",
        (interval[0], interval[1]),
        end_status=(statuses[0], statuses[1]),
        constraints=(constraints[0], constraints[1]),
        entity_ids=(entities[0], entities[1]),
        evidence=evidence,
    )


def _edge_range_candidates(
    model: BRepModel,
    zero: tuple[_Vector, ...],
    velocity: tuple[_Vector, ...],
    /,
) -> tuple[_RangeCandidate, ...]:
    """Locate exact zeros of each unchanged edge-direction certificate."""
    candidates: list[_RangeCandidate] = []
    for edge, (first, last) in enumerate(model.geometry.edge_vertices):
        direction = _sub(zero[last], zero[first])
        derivative = _sub(velocity[last], velocity[first])
        slope = _dot3(direction, derivative)
        if slope != 0:
            candidates.append(
                (
                    -_dot3(direction, direction) / slope,
                    "trim_vanishes",
                    (model.edge_ids[edge],),
                    "attained_exactly",
                )
            )
    return tuple(candidates)


def _certify_planar_lattice(
    geometry: BRepGeometry,
    motion: _PlanarMotion,
    delta: Fraction,
    /,
    *,
    rounding_margin: Fraction = Fraction(),
) -> None:
    """Certify edge signs and the contact-free distance path.

    A positive rounding margin re-checks edge signs and the unrounded clearance of
    the reached boundary, so every coefficient-rounding perturbation of it stays
    separated; the path itself was certified without a margin.
    """
    original = tuple(
        (
            exact if exact is not None else rational_vector(point)
            for exact, point in zip(
                geometry.vertex_exact_points,
                np.asarray(geometry.vertex_points),
                strict=True,
            )
        )
    )
    reached = motion.points(delta)
    threshold = 2 * rounding_margin * (2 * rounding_margin)
    margined = rounding_margin > 0
    reversed_edge = False
    for first, last in geometry.edge_vertices:
        old = _sub(original[last], original[first])
        new = _sub(reached[last], reached[first])
        sign = _dot3(old, new)
        if sign == 0:
            raise ValueError(
                "brep_offset.planar_topology must preserve nonvanishing faces; got a face vanishes at the requested distance"
            )
        reversed_edge = reversed_edge or sign < 0
        if sign > 0 and sign * sign <= _dot3(old, old) * threshold:
            raise ValueError(
                "brep_offset.edge_direction must exceed its coefficient-rounding margin; got an unresolved direction"
            )
    if margined:
        pair = _clearance_violation(motion, delta, rounding_margin)
        if pair is not None:
            raise ValueError(
                f"brep_offset.planar_clearance must exceed twice its coefficient-rounding margin; got {_pair_name(*pair)} within it"
            )
    else:
        _certify_contact_free(motion, delta)
    if reversed_edge:
        raise ValueError(
            "brep_offset.planar_topology must preserve edge direction; got a reversed edge at the requested distance"
        )


def _certify_contact_free(motion: _PlanarMotion, delta: Fraction, /) -> None:
    """Require separated source features and no verified contact up to ``delta``."""
    pair = _clearance_violation(motion, Fraction(), Fraction())
    if pair is not None:
        raise ValueError(
            f"brep_offset.source_boundary must separate non-incident features; got touching {_pair_name(*pair)}"
        )
    lower, upper = (delta, Fraction()) if delta.numerator < 0 else (Fraction(), delta)
    side = int(delta.numerator > 0)
    contact = _first_contacts(motion, lower, upper)[side]
    if contact is not None:
        raise ValueError(
            f"brep_offset.planar_topology must preserve non-incident separation; got collision pair {_pair_name(contact[1], contact[2])} at distance {contact[0]}"
        )


def _pair_name(first: BRepEntityId, second: BRepEntityId, /) -> str:
    return f"{first.kind} {first.index} and {second.kind} {second.index}"


@dataclass(frozen=True, slots=True)
class _PlanarMotion:
    """Rational affine vertex trajectories ``origin + distance * velocity``.

    Every face plane translates along its fixed unit normal and every edge line
    keeps its direction, so each vertex star is a translate of the source star
    while its edges stay nonvanishing. Every vertex-plane and edge-line contact
    between non-incident features is therefore the root of an affine function.
    """

    origin: tuple[_Vector, ...]
    velocity: tuple[_Vector, ...]
    planes: tuple[_Plane, ...]
    edges: tuple[tuple[int, int], ...]
    edge_faces: tuple[frozenset[int], ...]
    face_vertices: tuple[frozenset[int], ...]
    face_loops: tuple[tuple[tuple[int, ...], ...], ...]
    vertex_ids: tuple[BRepEntityId, ...]
    edge_ids: tuple[BRepEntityId, ...]
    face_ids: tuple[BRepEntityId, ...]

    def point(self, vertex: int, distance: Fraction, /) -> _Vector:
        return tuple(
            (
                a + distance * b
                for a, b in zip(self.origin[vertex], self.velocity[vertex], strict=True)
            )
        )

    def points(self, distance: Fraction, /) -> tuple[_Vector, ...]:
        return tuple((self.point(vertex, distance) for vertex in range(len(self.origin))))


def _planar_motion(
    model: BRepModel,
    planes: tuple[_Plane, ...],
    origin: tuple[_Vector, ...],
    velocity: tuple[_Vector, ...],
    /,
) -> _PlanarMotion:
    """Index ordered trim corners and edge-face incidence once per certificate."""
    geometry = model.geometry
    edge_faces: list[set[int]] = [set() for _ in geometry.edge_vertices]
    face_loops: list[tuple[tuple[int, ...], ...]] = []
    for face, loops in enumerate(geometry.face_loops):
        ordered: list[tuple[int, ...]] = []
        for loop in loops:
            corners: list[int] = []
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                edge_faces[edge].add(face)
                corners.append(
                    geometry.edge_vertices[edge][
                        0 if geometry.coedge_senses[coedge] == 1 else 1
                    ]
                )
            ordered.append(tuple(corners))
        face_loops.append(tuple(ordered))
    return _PlanarMotion(
        origin,
        velocity,
        planes,
        tuple(geometry.edge_vertices),
        tuple((frozenset(faces) for faces in edge_faces)),
        tuple(
            (
                frozenset((vertex for loop in loops for vertex in loop))
                for loops in face_loops
            )
        ),
        tuple(face_loops),
        model.vertex_ids,
        model.edge_ids,
        model.face_ids,
    )


def _sub(first: _Vector, second: _Vector, /) -> _Vector:
    return tuple((a - b for a, b in zip(first, second, strict=True)))


def _dot3(first: _Vector, second: _Vector, /) -> Fraction:
    """Compute an exact three-dimensional dot product."""
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _loop_segments[T](loop: tuple[T, ...], /) -> Iterator[tuple[T, T]]:
    return zip(loop, (*loop[1:], loop[0]), strict=True)


def _box(points: tuple[_Vector, ...], /) -> _Box:
    """Return existing coordinate extrema; ``n`` points take 6(n - 1) comparisons."""
    return tuple(
        (
            (
                min((point[axis] for point in points)),
                max((point[axis] for point in points)),
            )
            for axis in range(3)
        )
    )


def _boxes_within(first: _Box, second: _Box, limit: Fraction, /) -> bool:
    """Decide whether two closed coordinate boxes are within a squared distance."""
    squared = Fraction()
    for a, b in zip(first, second, strict=True):
        if b[0] > a[1]:
            gap = b[0] - a[1]
        elif a[0] > b[1]:
            gap = a[0] - b[1]
        else:
            continue
        squared += gap * gap
    return squared <= limit


def _on_every_edge_face(
    sets: tuple[frozenset[int], ...], motion: _PlanarMotion, edge: int, /
) -> frozenset[int]:
    common = frozenset(range(len(motion.origin)))
    for face in motion.edge_faces[edge]:
        common &= sets[face]
    return common


def _edge_pairs(motion: _PlanarMotion, /) -> Iterator[tuple[int, int]]:
    """Yield edge pairs without a shared endpoint in deterministic order."""
    for first, second in combinations(range(len(motion.edges)), 2):
        start, end = motion.edges[first]
        if start not in motion.edges[second] and end not in motion.edges[second]:
            yield (first, second)


def _closed_face_contains(
    normal: _Vector, loops: tuple[tuple[_Vector, ...], ...], point: _Vector, /
) -> bool:
    """Decide closed trim membership on a face plane, including its boundary and holes."""
    magnitudes = tuple((abs(value) for value in normal))
    dropped = magnitudes.index(max(magnitudes))
    kept = tuple((axis for axis in range(3) if axis != dropped))
    target = (point[kept[0]], point[kept[1]])
    projected = tuple(
        (tuple(((corner[kept[0]], corner[kept[1]]) for corner in loop)) for loop in loops)
    )
    if any(
        (
            on_segment(target, segment)
            for loop in projected
            for segment in _loop_segments(loop)
        )
    ):
        return True
    return sum((inside_loop(target, loop) for loop in projected)) % 2 == 1


def _point_segment_within(
    point: _Vector, first: _Vector, last: _Vector, limit: Fraction, /
) -> bool:
    """Decide a point-to-closed-segment distance bound without division."""
    direction, offset = (_sub(last, first), _sub(point, first))
    along = _dot3(offset, direction)
    if along <= 0:
        return _dot3(offset, offset) <= limit
    length = _dot3(direction, direction)
    if along >= length:
        rest = _sub(point, last)
        return _dot3(rest, rest) <= limit
    return _dot3(offset, offset) * length - along * along <= limit * length


def _segments_within(
    start: _Vector, end: _Vector, other: _Vector, finish: _Vector, limit: Fraction, /
) -> bool:
    """Decide whether two closed segments are within a squared distance."""
    along, across = (_sub(end, start), _sub(finish, other))
    offset = _sub(start, other)
    aa, ab, bb = (_dot3(along, along), _dot3(along, across), _dot3(across, across))
    determinant = aa * bb - ab * ab
    if determinant:
        ao, bo = (_dot3(along, offset), _dot3(across, offset))
        first = ab * bo - bb * ao
        second = aa * bo - ab * ao
        if 0 <= first <= determinant and 0 <= second <= determinant:
            height = _dot3(offset, rational_cross(along, across))
            return height * height <= limit * determinant
    return (
        _point_segment_within(start, other, finish, limit)
        or _point_segment_within(end, other, finish, limit)
        or _point_segment_within(other, start, end, limit)
        or _point_segment_within(finish, start, end, limit)
    )


def _point_face_within(
    plane: _Plane,
    distance: Fraction,
    loops: tuple[tuple[_Vector, ...], ...],
    point: _Vector,
    limit: Fraction,
    /,
) -> bool:
    """Bound distance to the face projection or its closed trim boundary.

    The orthogonal foot is the nearest point when it lies in the closed face;
    otherwise the minimum is attained on one of the boundary segments.
    """
    normal, height = plane
    elevation = _dot3(normal, point) - (height + distance)
    foot = tuple((a - elevation * b for a, b in zip(point, normal, strict=True)))
    if _closed_face_contains(normal, loops, foot):
        return elevation * elevation <= limit
    return any(
        (
            _point_segment_within(point, first, last, limit)
            for loop in loops
            for first, last in _loop_segments(loop)
        )
    )


def _clearance_violation(
    motion: _PlanarMotion, distance: Fraction, rounding_margin: Fraction, /
) -> tuple[BRepEntityId, BRepEntityId] | None:
    """Find a non-incident pair within twice the rounding margin."""
    points = motion.points(distance)
    limit = 4 * (rounding_margin * rounding_margin)
    return _violating_pair(motion, points, distance, limit)


def _violating_pair(
    motion: _PlanarMotion,
    points: tuple[_Vector, ...],
    distance: Fraction,
    limit: Fraction,
    /,
) -> tuple[BRepEntityId, BRepEntityId] | None:
    face_boxes = tuple(
        (
            _box(tuple((points[vertex] for vertex in sorted(corners))))
            for corners in motion.face_vertices
        )
    )
    edge_boxes = tuple((_box((points[a], points[b])) for a, b in motion.edges))
    for face, plane in enumerate(motion.planes):
        loops = tuple(
            (
                tuple((points[corner] for corner in loop))
                for loop in motion.face_loops[face]
            )
        )
        for vertex, point in enumerate(points):
            if (
                vertex not in motion.face_vertices[face]
                and _boxes_within(_box((point,)), face_boxes[face], limit)
                and _point_face_within(plane, distance, loops, point, limit)
            ):
                return (motion.vertex_ids[vertex], motion.face_ids[face])
    for edge, (first, last) in enumerate(motion.edges):
        shared = _on_every_edge_face(motion.face_vertices, motion, edge)
        for vertex in sorted(shared - {first, last}):
            if _point_segment_within(points[vertex], points[first], points[last], limit):
                return (motion.vertex_ids[vertex], motion.edge_ids[edge])
    for first, second in _edge_pairs(motion):
        (a, b), (c, d) = (motion.edges[first], motion.edges[second])
        if _boxes_within(
            edge_boxes[first], edge_boxes[second], limit
        ) and _segments_within(points[a], points[b], points[c], points[d], limit):
            return (motion.edge_ids[first], motion.edge_ids[second])
    return None


def _admissible(
    nearest: list[_Contact | None],
    bounds: tuple[Fraction | None, Fraction | None],
    root: Fraction,
    /,
) -> int | None:
    """Return the side on which a nonzero contact root can bind."""
    if root == 0:
        return None
    side = int(root > 0)
    current = nearest[side]
    if current is not None:
        return side if abs(root) < abs(current[0]) else None
    bound = bounds[side]
    return side if bound is None or abs(root) <= abs(bound) else None


def _vertex_face_touches(
    motion: _PlanarMotion, face: int, vertex: int, root: Fraction, /
) -> bool:
    """Verify closed face membership at an exact contact root."""
    loops = tuple(
        (
            tuple((motion.point(corner, root) for corner in loop))
            for loop in motion.face_loops[face]
        )
    )
    return _closed_face_contains(
        motion.planes[face][0], loops, motion.point(vertex, root)
    )


def _vertex_face_contacts(
    motion: _PlanarMotion,
    nearest: list[_Contact | None],
    bounds: tuple[Fraction | None, Fraction | None],
    /,
) -> tuple[frozenset[int], ...]:
    """Stream vertex-plane roots and collect identically coplanar vertices.

    A vertex on a face plane at every distance enters that face only through a
    trim, where another face's root or an endpoint coincidence detects it.
    """
    on_plane: list[frozenset[int]] = []
    for face, (normal, height) in enumerate(motion.planes):
        coplanar = set(motion.face_vertices[face])
        for vertex, (point, rate) in enumerate(
            zip(motion.origin, motion.velocity, strict=True)
        ):
            if vertex in motion.face_vertices[face]:
                continue
            gap = _dot3(normal, point) - height
            closing = _dot3(normal, rate) - 1
            if not closing:
                if not gap:
                    coplanar.add(vertex)
                continue
            root = -gap / closing
            side = _admissible(nearest, bounds, root)
            if side is not None and _vertex_face_touches(motion, face, vertex, root):
                nearest[side] = (root, motion.vertex_ids[vertex], motion.face_ids[face])
        on_plane.append(frozenset(coplanar))
    return tuple(on_plane)


def _coincidence_root(
    motion: _PlanarMotion, first: int, second: int, /
) -> Fraction | None:
    """Return the unique distance at which two vertex trajectories meet."""
    root: Fraction | None = None
    for gap, closing in zip(
        _sub(motion.origin[second], motion.origin[first]),
        _sub(motion.velocity[first], motion.velocity[second]),
        strict=True,
    ):
        if closing:
            value = gap / closing
            if root is not None and root != value:
                return None
            root = value
        elif gap:
            return None
    return root


def _collinear_contacts(
    motion: _PlanarMotion,
    nearest: list[_Contact | None],
    bounds: tuple[Fraction | None, Fraction | None],
    on_plane: tuple[frozenset[int], ...],
    /,
) -> None:
    """Stream endpoint coincidences of vertices identically on an edge line."""
    for edge, (first, last) in enumerate(motion.edges):
        for vertex in sorted(_on_every_edge_face(on_plane, motion, edge) - {first, last}):
            for end in (first, last):
                root = _coincidence_root(motion, vertex, end)
                if root is None:
                    continue
                side = _admissible(nearest, bounds, root)
                if side is not None and motion.point(vertex, root) == motion.point(
                    end, root
                ):
                    ids = motion.vertex_ids
                    nearest[side] = (root, ids[vertex], ids[end])


def _meet_scale(
    offset: _Vector,
    drift: _Vector,
    direction: _Vector,
    normal: _Vector,
    gap: Fraction,
    closing: Fraction,
    /,
) -> Fraction:
    """Return ``closing * ((offset + t drift) x direction) . normal`` at the root."""
    return (
        _dot3(rational_cross(offset, direction), normal) * closing
        - _dot3(rational_cross(drift, direction), normal) * gap
    )


def _edge_edge_touches(
    motion: _PlanarMotion,
    directions: tuple[_Vector, ...],
    pair: tuple[int, int],
    normal: _Vector,
    gap: Fraction,
    closing: Fraction,
    /,
) -> bool:
    """Verify that coplanar lines meet both closed edge segments at the root."""
    (a, b), (c, d) = (motion.edges[pair[0]], motion.edges[pair[1]])
    along, across = (directions[pair[0]], directions[pair[1]])
    offset = _sub(motion.origin[c], motion.origin[a])
    drift = _sub(motion.velocity[c], motion.velocity[a])
    sigma = _meet_scale(offset, drift, across, normal, gap, closing)
    rho = _meet_scale(offset, drift, along, normal, gap, closing)
    along_squared, across_squared = (_dot3(along, along), _dot3(across, across))
    first_reach = (
        along_squared * closing
        - _dot3(_sub(motion.velocity[b], motion.velocity[a]), along) * gap
    )
    second_reach = (
        across_squared * closing
        - _dot3(_sub(motion.velocity[d], motion.velocity[c]), across) * gap
    )
    squared = _dot3(normal, normal)
    if closing < 0:
        sigma, rho, first_reach, second_reach = (
            -sigma,
            -rho,
            -first_reach,
            -second_reach,
        )
    return (
        sigma >= 0
        and sigma * along_squared <= first_reach * squared
        and (rho >= 0)
        and (rho * across_squared <= second_reach * squared)
    )


def _edge_edge_contacts(
    motion: _PlanarMotion,
    nearest: list[_Contact | None],
    bounds: tuple[Fraction | None, Fraction | None],
    /,
) -> None:
    """Stream coplanarity roots of skew non-incident edge lines.

    Parallel or always-coplanar lines first touch at an endpoint, which the
    vertex families detect.
    """
    directions = tuple(
        (_sub(motion.origin[last], motion.origin[first]) for first, last in motion.edges)
    )
    for pair in _edge_pairs(motion):
        normal = rational_cross(directions[pair[0]], directions[pair[1]])
        start, other = (motion.edges[pair[0]][0], motion.edges[pair[1]][0])
        closing = _dot3(normal, _sub(motion.velocity[other], motion.velocity[start]))
        if not closing:
            continue
        gap = _dot3(normal, _sub(motion.origin[other], motion.origin[start]))
        root = -gap / closing
        side = _admissible(nearest, bounds, root)
        if side is not None and _edge_edge_touches(
            motion, directions, pair, normal, gap, closing
        ):
            nearest[side] = (root, motion.edge_ids[pair[0]], motion.edge_ids[pair[1]])


def _first_contacts(
    motion: _PlanarMotion, lower: Fraction | None, upper: Fraction | None, /
) -> tuple[_Contact | None, _Contact | None]:
    """Find the nearest verified non-incident contact on either side of zero."""
    nearest: list[_Contact | None] = [None, None]
    bounds = (lower, upper)
    on_plane = _vertex_face_contacts(motion, nearest, bounds)
    _collinear_contacts(motion, nearest, bounds, on_plane)
    _edge_edge_contacts(motion, nearest, bounds)
    return (nearest[0], nearest[1])


def _stage_planar_face(
    model: BRepModel,
    face: int,
    patch: PlanePatch,
    vertices: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[PlanePatch, np.ndarray, dict[int, LineCurve]]:
    geometry = model.geometry
    anchor = geometry.edge_vertices[
        geometry.coedge_edges[geometry.face_loops[face][0][0]]
    ][0]
    origin = vertices[anchor]
    first_axis = rational_vector(np.asarray(patch.first_axis))
    second_axis = rational_vector(np.asarray(patch.second_axis))
    translated = PlanePatch(
        tuple((float(value) for value in origin)), patch.first_axis, patch.second_axis
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
                        (
                            value - center
                            for value, center in zip(
                                vertices[vertex], origin, strict=True
                            )
                        )
                    )
                    coordinates[vertex] = _solve(
                        gram,
                        (_dot(first_axis, displacement), _dot(second_axis, displacement)),
                    )
    pcurves: dict[int, LineCurve] = {}
    for source_loop in geometry.face_loops[face]:
        for coedge in source_loop:
            start, end = geometry.edge_vertices[geometry.coedge_edges[coedge]]
            uv = coordinates[start]
            direction = tuple((b - a for a, b in zip(uv, coordinates[end], strict=True)))
            pcurves[coedge] = LineCurve(
                tuple((float(value) for value in uv)),
                tuple((float(value) for value in direction)),
            )
    points = np.asarray(
        [[float(value) for value in uv] for uv in coordinates.values()], dtype=np.float64
    )
    box = np.stack((points.min(axis=0), points.max(axis=0)))
    return translated, box, pcurves


def _affine_edge(curve: BRepCurve, /) -> bool:
    match curve:
        case LineCurve() | ExactLineCurve():
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
    shifted = tuple((height + delta for _, height in planes))
    vertices = []
    for faces in incident:
        candidate = None
        for triple in combinations(sorted(faces), 3):
            matrix = tuple(planes[face][0] for face in triple)
            if determinant_exact(matrix) != 0:
                candidate = _solve(matrix, tuple(shifted[face] for face in triple))
                break
        if candidate is None:
            raise ValueError(
                "brep_offset.planar_topology must have three independent incident planes; got an unsupported vertex"
            )
        for face in faces:
            if _dot3(planes[face][0], candidate) != shifted[face]:
                raise ValueError(
                    "brep_offset.planar_topology must preserve vertex incidence; got a nonsimple changed vertex"
                )
        vertices.append(candidate)
    return tuple(vertices)
