#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact axis-aligned box arrangements for native B-Rep partitioning."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...discretization._exact_algebraic_geometry import coefficient_payload
from .._cad_revision import (
    AssociationCoverageEvidence,
    AssociationGraph,
    CADOccurrence,
    CADRevision,
    OccurrenceCorrespondence,
    OccurrenceCorrespondenceTransaction,
)
from ._assembly import BRepAffinePlacement
from ._constructors import assemble_brep_model, BRepTessellationPolicy
from ._model import BRepEntityId, BRepGeometry, BRepModel
from ._patches import ExactLineCurve, ExtrusionSurface, LineCurve, PlanePatch


if TYPE_CHECKING:
    from ._partition import (
        BRepPartitionOperand,
        BRepPartitionPlan,
        BRepPartitionResult,
    )


Point = tuple[Fraction, Fraction, Fraction]
FaceKey = tuple[int, Fraction, Fraction, Fraction, Fraction, Fraction]
Box = tuple[Point, Point]


@dataclass(frozen=True, slots=True)
class _Atom:
    bounds: Box
    owner: str


def _fraction_point(value: np.ndarray, /) -> Point:
    return (
        Fraction.from_float(float(value[0])),
        Fraction.from_float(float(value[1])),
        Fraction.from_float(float(value[2])),
    )


def _line_point(
    curve: LineCurve | ExactLineCurve, parameter: Fraction, /
) -> tuple[Fraction, ...]:
    if isinstance(curve, ExactLineCurve):
        return curve.evaluate_exact(parameter)
    origin = tuple(
        Fraction.from_float(float(value)) for value in np.asarray(curve.origin)
    )
    direction = tuple(
        Fraction.from_float(float(value)) for value in np.asarray(curve.direction)
    )
    return tuple(
        origin[axis] + parameter * direction[axis] for axis in range(len(origin))
    )


def _source_vertex(geometry: BRepGeometry, index: int, /) -> Point:
    exact = geometry.vertex_exact_points[index]
    if exact is not None:
        return exact
    return _fraction_point(np.asarray(geometry.vertex_points[index]))


def _patch_point(patch: object, parameters: tuple[Fraction, Fraction], /) -> Point:
    first, second = parameters
    if isinstance(patch, PlanePatch):
        origin = _fraction_point(np.asarray(patch.origin))
        first_axis = _fraction_point(np.asarray(patch.first_axis))
        second_axis = _fraction_point(np.asarray(patch.second_axis))
        return (
            origin[0] + first * first_axis[0] + second * second_axis[0],
            origin[1] + first * first_axis[1] + second * second_axis[1],
            origin[2] + first * first_axis[2] + second * second_axis[2],
        )
    if isinstance(patch, ExtrusionSurface) and isinstance(
        patch.curve, (LineCurve, ExactLineCurve)
    ):
        origin = _line_point(patch.curve, first)
        direction = _fraction_point(np.asarray(patch.direction))
        return (
            origin[0] + second * direction[0],
            origin[1] + second * direction[1],
            origin[2] + second * direction[2],
        )
    raise NotImplementedError(
        "Native box partition requires plane or line-extrusion source faces."
    )


def _cross(
    first: tuple[Fraction, Fraction, Fraction],
    second: tuple[Fraction, Fraction, Fraction],
    /,
) -> Point:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _box_bounds(model: BRepModel, /) -> Box:
    geometry = model.geometry
    occurrence_iterator = geometry.assembly_graph.iter_occurrences()
    occurrence = next(occurrence_iterator, None)
    identity = BRepAffinePlacement.identity().matrix
    single_unplaced = (
        occurrence is not None
        and occurrence.solid_index == 0
        and occurrence.placement.matrix == identity
        and next(occurrence_iterator, None) is None
    )
    if (
        model.report.curve_surface_tolerance != 0.0
        or model.topology.num_solids != 1
        or model.topology.num_faces != 6
        or model.topology.num_edges != 12
        or model.topology.num_vertices != 8
        or geometry.solid_shells != ((0,),)
        or len(geometry.shell_faces) != 1
        or not single_unplaced
    ):
        raise NotImplementedError(
            "Native exact partition requires one unplaced exact six-face, twelve-edge, eight-vertex solid."
        )
    source_points = tuple(
        _source_vertex(geometry, index) for index in range(geometry.topology.num_vertices)
    )
    source_vertices = set(source_points)
    lower = (
        min(point[0] for point in source_points),
        min(point[1] for point in source_points),
        min(point[2] for point in source_points),
    )
    upper = (
        max(point[0] for point in source_points),
        max(point[1] for point in source_points),
        max(point[2] for point in source_points),
    )
    if any(a >= b for a, b in zip(lower, upper, strict=True)):
        raise NotImplementedError("Native box partition requires a nondegenerate box.")
    corners = {
        (x, y, z)
        for x, y, z in product(*(tuple(sorted({lower[i], upper[i]})) for i in range(3)))
    }
    if len(source_vertices) != 8 or source_vertices != corners:
        raise NotImplementedError(
            "Native exact box partition needs all eight distinct corners."
        )
    edge_uses = [0] * geometry.topology.num_edges
    edge_balances = [0] * geometry.topology.num_edges
    face_keys: set[FaceKey] = set()
    orientations = np.asarray(model.orientation, dtype=np.int8)
    solid_faces = model.topology.solid_faces[0]
    solid_face_orientations = model.topology.solid_face_orientations[0]
    if set(solid_faces) != set(range(model.topology.num_faces)):
        raise NotImplementedError(
            "Native box partition requires every face in its solid shell."
        )
    for face_index, shell_orientation in zip(
        solid_faces, solid_face_orientations, strict=True
    ):
        patch = model.patches[face_index]
        loops = geometry.face_loops[face_index]
        if len(loops) != 1 or len(loops[0]) != 4:
            raise NotImplementedError(
                "Native box partition requires four-edge planar faces."
            )
        face_vertices: set[int] = set()
        for coedge in loops[0]:
            edge = geometry.coedge_edges[coedge]
            curve_index = geometry.edge_curves[edge]
            if curve_index < 0:
                raise NotImplementedError(
                    "Native box partition requires straight source edges."
                )
            curve = geometry.curves[curve_index]
            if not isinstance(curve, (LineCurve, ExactLineCurve)):
                raise NotImplementedError(
                    "Native box partition requires straight source edges."
                )
            first, last = (
                Fraction.from_float(float(value))
                for value in np.asarray(geometry.edge_ranges)[edge]
            )
            edge_start, edge_end = geometry.edge_vertices[edge]
            source_start = _source_vertex(geometry, edge_start)
            source_end = _source_vertex(geometry, edge_end)
            if (
                _line_point(curve, first) != source_start
                or _line_point(curve, last) != source_end
            ):
                raise NotImplementedError(
                    "Native box edges must retain exact represented endpoints."
                )
            if sum(a != b for a, b in zip(source_start, source_end, strict=True)) != 1:
                raise NotImplementedError(
                    "Native box edges must be axis-aligned source segments."
                )
            pcurve = geometry.pcurves[coedge]
            if not isinstance(pcurve, (LineCurve, ExactLineCurve)):
                raise NotImplementedError(
                    "Native box faces require straight source p-curves."
                )
            for parameter in (first, last):
                uv = _line_point(pcurve, parameter)
                if len(uv) != 2 or _patch_point(patch, (uv[0], uv[1])) != _line_point(
                    curve, parameter
                ):
                    raise NotImplementedError(
                        "Native box coedges must have exact coupled curve and p-curve endpoints."
                    )
            face_vertices.update((edge_start, edge_end))
            edge_uses[edge] += 1
            edge_balances[edge] += (
                int(geometry.coedge_senses[coedge])
                * int(orientations[face_index])
                * int(shell_orientation)
            )
        if len(face_vertices) != 4:
            raise NotImplementedError(
                "Native box faces require four distinct source corners."
            )
        face_key = _face_key(
            tuple(_source_vertex(geometry, index) for index in face_vertices)
        )
        axis, fixed, lower_first, upper_first, lower_second, upper_second = face_key
        if fixed not in (lower[axis], upper[axis]):
            raise NotImplementedError(
                "Native box faces must lie on an exact outer coordinate plane."
            )
        other_axes = tuple(value for value in range(3) if value != axis)
        if (lower_first, upper_first) != (lower[other_axes[0]], upper[other_axes[0]]) or (
            lower_second,
            upper_second,
        ) != (lower[other_axes[1]], upper[other_axes[1]]):
            raise NotImplementedError(
                "Native box faces must cover their exact rectangular sides."
            )
        if isinstance(patch, PlanePatch):
            first_axis = _fraction_point(np.asarray(patch.first_axis))
            second_axis = _fraction_point(np.asarray(patch.second_axis))
        elif isinstance(patch, ExtrusionSurface) and isinstance(
            patch.curve, (LineCurve, ExactLineCurve)
        ):
            if isinstance(patch.curve, LineCurve):
                first_axis = _fraction_point(np.asarray(patch.curve.direction))
            else:
                first_axis = (
                    patch.curve.last_exact[0] - patch.curve.first_exact[0],
                    patch.curve.last_exact[1] - patch.curve.first_exact[1],
                    patch.curve.last_exact[2] - patch.curve.first_exact[2],
                )
            second_axis = _fraction_point(np.asarray(patch.direction))
        else:
            raise NotImplementedError(
                "Native box faces require exact plane or line-extrusion carriers."
            )
        normal = _cross(first_axis, second_axis)
        if any(normal[index] != 0 for index in range(3) if index != axis):
            raise NotImplementedError(
                "Native box face normals must align with a coordinate axis."
            )
        outward_sign = 1 if fixed == upper[axis] else -1
        oriented_normal = (
            normal[axis] * int(orientations[face_index]) * int(shell_orientation)
        )
        if oriented_normal * outward_sign <= 0:
            raise NotImplementedError(
                "Native box shell faces must have exact outward orientation."
            )
        face_keys.add(face_key)
    expected_faces = {
        (
            axis,
            fixed,
            lower[next_axis],
            upper[next_axis],
            lower[last_axis],
            upper[last_axis],
        )
        for axis in range(3)
        for fixed in (lower[axis], upper[axis])
        for next_axis, last_axis in (tuple(value for value in range(3) if value != axis),)
    }
    if face_keys != expected_faces:
        raise NotImplementedError(
            "Native box face topology must exhaust all six oriented sides."
        )
    if any(uses != 2 for uses in edge_uses) or any(edge_balances):
        raise NotImplementedError(
            "Native box source must have a closed, consistently oriented edge shell."
        )
    return lower, upper


def _face_key(points: tuple[Point, ...], /) -> FaceKey:
    lower = tuple(min(point[axis] for point in points) for axis in range(3))
    upper = tuple(max(point[axis] for point in points) for axis in range(3))
    fixed = tuple(axis for axis in range(3) if lower[axis] == upper[axis])
    varying = tuple(axis for axis in range(3) if lower[axis] < upper[axis])
    if len(fixed) != 1 or len(varying) != 2 or len(set(points)) != 4:
        raise ValueError("Axis-aligned box face has invalid exact corner incidence.")
    axis = fixed[0]
    first, second = varying
    return axis, lower[axis], lower[first], upper[first], lower[second], upper[second]


def _face_axes(axis: int, sign: int, /) -> tuple[int, int]:
    match axis, sign:
        case 0, 1:
            return 1, 2
        case 0, -1:
            return 2, 1
        case 1, 1:
            return 2, 0
        case 1, -1:
            return 0, 2
        case 2, 1:
            return 0, 1
        case 2, -1:
            return 1, 0
        case _:
            raise ValueError("A box face normal requires one axis and an oriented sign.")


def _source_face_key(model: BRepModel, face_index: int, /) -> FaceKey:
    geometry = model.geometry
    vertex_ids = {
        vertex
        for loop in geometry.face_loops[face_index]
        for coedge in loop
        for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
    }
    points = tuple(
        _fraction_point(np.asarray(geometry.vertex_points[index])) for index in vertex_ids
    )
    return _face_key(points)


def _contained_face(inner: FaceKey, outer: FaceKey, /) -> bool:
    if inner[:2] != outer[:2]:
        return False
    return all(inner[index] >= outer[index] for index in (2, 4)) and all(
        inner[index] <= outer[index] for index in (3, 5)
    )


def _composite_revision(
    plan: BRepPartitionPlan, operands: tuple[BRepPartitionOperand, ...], /
) -> tuple[CADRevision, dict[tuple[str, str], str]]:
    from ._partition import (
        _assembly_namespace_occurrence_id,
        _entity_occurrence_id,
        _entity_text,
        _operand_selection,
        cad_revision_from_brep_model,
    )

    plan_id = plan.plan_id
    source_id = f"native-brep-partition:{plan_id}"
    revision_id = canonical_fingerprint(
        {
            "kind": "native-brep-partition-source-revision",
            "plan": plan_id,
            "sources": sorted(
                (
                    value.operand_id,
                    value.model.source_revision,
                    _operand_selection(value).selection_id,
                )
                for value in operands
            ),
        }
    )
    occurrences: list[CADOccurrence] = []
    source_occurrence_ids: dict[tuple[str, str], str] = {}
    for operand in sorted(operands, key=lambda value: value.operand_id):
        model = operand.model
        geometry = model.geometry
        # Each operand keeps its exact assembly forest, rescoped to the operand, so
        # source namespaces correspond to the result namespaces of their cells.
        assembly_paths: dict[str, tuple[str, ...]] = {}
        for item in cad_revision_from_brep_model(model).occurrences:
            if item.kind != "assembly":
                continue
            assembly_path = tuple(
                f"operand:{operand.operand_id}:assembly:{value}" for value in item.path
            )
            assembly_paths[item.occurrence_id] = assembly_path
            occurrences.append(
                CADOccurrence(
                    revision_id,
                    assembly_path[-1],
                    item.entity_id,
                    "assembly",
                    assembly_path,
                    assembly_path[-2] if len(assembly_path) > 1 else None,
                )
            )
        for source_occurrence in geometry.iter_occurrences():
            solid = source_occurrence.solid_index
            solid_entity = BRepEntityId(
                model.source_id,
                model.source_revision,
                "solid",
                solid,
                source_occurrence.path,
            )
            local_id = _entity_occurrence_id(solid_entity)
            occurrence_id = f"operand:{operand.operand_id}:solid:{solid}"
            parent_path = assembly_paths.get(
                _assembly_namespace_occurrence_id(
                    model.source_id, model.source_revision, source_occurrence.path
                ),
                (),
            )
            solid_path = (*parent_path, occurrence_id)
            occurrences.append(
                CADOccurrence(
                    revision_id,
                    occurrence_id,
                    _entity_text(solid_entity),
                    "solid",
                    solid_path,
                    parent_path[-1] if parent_path else None,
                )
            )
            source_occurrence_ids[(operand.operand_id, local_id)] = occurrence_id
            for face, orientation in zip(
                geometry.topology.solid_faces[solid],
                geometry.topology.solid_face_orientations[solid],
                strict=True,
            ):
                face_entity = BRepEntityId(
                    model.source_id,
                    model.source_revision,
                    "face",
                    face,
                    source_occurrence.path,
                )
                face_local_id = _entity_occurrence_id(face_entity)
                face_occurrence_id = f"{occurrence_id}/face:{face}"
                occurrences.append(
                    CADOccurrence(
                        revision_id,
                        face_occurrence_id,
                        _entity_text(face_entity),
                        "face",
                        (*solid_path, face_occurrence_id),
                        occurrence_id,
                        orientation,
                    )
                )
                source_occurrence_ids[(operand.operand_id, face_local_id)] = (
                    face_occurrence_id
                )
    return (
        CADRevision(revision_id, source_id, tuple(occurrences), plan_id),
        source_occurrence_ids,
    )


def _arrangement(plan: BRepPartitionPlan, /) -> tuple[tuple[_Atom, ...], dict[str, Box]]:
    from ._partition import _entity_text, _operand_selection

    operand_boxes: dict[str, Box] = {}
    for operand in plan.operands:
        selection = _operand_selection(operand)
        if len(selection.selectors) != 1 or operand.model.topology.num_solids != 1:
            raise NotImplementedError(
                "Native exact box partition requires one selected solid per operand."
            )
        selected = selection.selectors[0]
        occurrences = operand.model.geometry.iter_occurrences()
        occurrence = next(occurrences, None)
        extra_occurrence = next(occurrences, None)
        if (
            selected.kind != "solid"
            or occurrence is None
            or extra_occurrence is not None
            or occurrence.solid_index != 0
            or selected.entity_id
            != _entity_text(
                BRepEntityId(
                    operand.model.source_id,
                    operand.model.source_revision,
                    "solid",
                    0,
                    occurrence.path,
                )
            )
        ):
            raise NotImplementedError(
                "Native exact box partition requires its sole source-solid occurrence."
            )
        operand_boxes[operand.operand_id] = _box_bounds(operand.model)
    axes = (
        tuple(
            sorted(
                {
                    point[0]
                    for lower, upper in operand_boxes.values()
                    for point in (lower, upper)
                }
            )
        ),
        tuple(
            sorted(
                {
                    point[1]
                    for lower, upper in operand_boxes.values()
                    for point in (lower, upper)
                }
            )
        ),
        tuple(
            sorted(
                {
                    point[2]
                    for lower, upper in operand_boxes.values()
                    for point in (lower, upper)
                }
            )
        ),
    )
    atoms: list[_Atom] = []
    voids = tuple(value for value in plan.operands if value.role.value == "void")
    regions = {
        value.operand_id: value for value in plan.operands if value.role.value == "region"
    }
    for cell_index in product(*(range(len(values) - 1) for values in axes)):
        lower: Point = (
            axes[0][cell_index[0]],
            axes[1][cell_index[1]],
            axes[2][cell_index[2]],
        )
        upper: Point = (
            axes[0][cell_index[0] + 1],
            axes[1][cell_index[1] + 1],
            axes[2][cell_index[2] + 1],
        )
        memberships = {
            operand_id
            for operand_id, (box_lower, box_upper) in operand_boxes.items()
            if all(
                a >= b and c <= d
                for a, c, b, d in zip(lower, upper, box_lower, box_upper, strict=True)
            )
        }
        owner = next(
            (name for name in plan.policy.region_precedence if name in memberships), None
        )
        if owner is None or any(
            owner in void.target_region_ids and void.operand_id in memberships
            for void in voids
        ):
            continue
        atoms.append(_Atom((lower, upper), owner))
    if not atoms:
        raise ValueError("Native exact partition removes every region cell.")
    if {atom.owner for atom in atoms} != set(regions):
        raise ValueError(
            "Every declared partition region must retain a positive-volume cell."
        )
    atoms.sort(
        key=lambda value: (
            value.bounds[0],
            value.bounds[1],
            plan.policy.region_precedence.index(value.owner),
        )
    )
    return tuple(atoms), operand_boxes


def _replace(point: Point, axis: int, value: Fraction, /) -> Point:
    values = list(point)
    values[axis] = value
    return values[0], values[1], values[2]


def _cell_key(lower: Point, upper: Point, axis: int, /) -> FaceKey:
    first, second = (value for value in range(3) if value != axis)
    return axis, lower[axis], lower[first], upper[first], lower[second], upper[second]


def _point_in_face(point: Point, key: FaceKey, /) -> bool:
    axis, fixed, lower_first, upper_first, lower_second, upper_second = key
    first, second = (value for value in range(3) if value != axis)
    return (
        point[axis] == fixed
        and lower_first <= point[first] <= upper_first
        and lower_second <= point[second] <= upper_second
    )


def _segment_axis(segment: tuple[Point, Point], /) -> int:
    start, end = segment
    return next(axis for axis in range(3) if start[axis] != end[axis])


def _union_find(count: int, pairs: list[tuple[int, int]], /) -> list[int]:
    parent = list(range(count))

    def find(index: int, /) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first, second in pairs:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)
    return [find(index) for index in range(count)]


@dataclass(frozen=True, slots=True)
class _RegionFace:
    """One maximal planar face separating two different region solids."""

    axis: int
    sign: int
    first_axis: int
    second_axis: int
    coordinate: Fraction
    negative_solid: int | None
    positive_solid: int | None
    cells: tuple[FaceKey, ...]
    loops: tuple[tuple[tuple[Point, Point], ...], ...]
    origin: Point


@dataclass(frozen=True, slots=True)
class _PartitionLayout:
    """Exact atom-to-region-solid and grid-cell-to-face lineage of one partition."""

    solid_atoms: tuple[tuple[int, ...], ...]
    solid_owners: tuple[str, ...]
    face_cells: tuple[tuple[FaceKey, ...], ...]
    edge_segments: tuple[tuple[tuple[Point, Point], ...], ...]


def _region_solids(
    atoms: tuple[_Atom, ...], rank: dict[str, int], /
) -> tuple[tuple[int, ...], ...]:
    """Group atoms into face-connected components of one owning region."""
    by_lower = {atom.bounds[0]: index for index, atom in enumerate(atoms)}
    pairs: list[tuple[int, int]] = []
    for index, atom in enumerate(atoms):
        lower, upper = atom.bounds
        for axis in range(3):
            neighbor = by_lower.get(_replace(lower, axis, upper[axis]))
            if neighbor is not None and atoms[neighbor].owner == atom.owner:
                pairs.append((index, neighbor))
    roots = _union_find(len(atoms), pairs)
    components: dict[int, list[int]] = {}
    for index, root in enumerate(roots):
        components.setdefault(root, []).append(index)
    return tuple(
        sorted(
            (tuple(indices) for indices in components.values()),
            key=lambda indices: (
                rank[atoms[indices[0]].owner],
                min(atoms[index].bounds for index in indices),
            ),
        )
    )


def _face_loops(
    cells: tuple[tuple[Point, Point], ...],
    first_axis: int,
    second_axis: int,
    /,
) -> tuple[tuple[tuple[Point, Point], ...], ...]:
    """Trace the cancelled directed boundary of edge-connected grid rectangles."""
    directed: set[tuple[Point, Point]] = set()
    for lower, upper in cells:
        first_corner = _replace(lower, first_axis, upper[first_axis])
        corners = (
            lower,
            first_corner,
            _replace(first_corner, second_axis, upper[second_axis]),
            _replace(lower, second_axis, upper[second_axis]),
        )
        for index, start in enumerate(corners):
            end = corners[(index + 1) % 4]
            if (end, start) in directed:
                directed.remove((end, start))
            else:
                directed.add((start, end))
    successor: dict[Point, Point] = {}
    for start, end in directed:
        if start in successor:
            raise NotImplementedError(
                "Native box partition region faces must not pinch at a vertex."
            )
        successor[start] = end
    loops: list[tuple[tuple[Point, Point], ...]] = []
    areas: list[Fraction] = []
    while successor:
        start = min(successor)
        point = start
        loop: list[tuple[Point, Point]] = []
        area = Fraction(0)
        while True:
            following = successor.pop(point)
            loop.append((point, following))
            area += (
                point[first_axis] * following[second_axis]
                - following[first_axis] * point[second_axis]
            )
            point = following
            if point == start:
                break
        loops.append(tuple(loop))
        areas.append(area)
    outer = [index for index, area in enumerate(areas) if area > 0]
    if len(outer) != 1 or any(area == 0 for area in areas):
        raise NotImplementedError(
            "Native box partition region faces need one outer loop and inner holes."
        )
    return (loops[outer[0]],) + tuple(
        loop for index, loop in enumerate(loops) if index != outer[0]
    )


def _region_faces(
    atoms: tuple[_Atom, ...],
    solid_atoms: tuple[tuple[int, ...], ...],
    /,
) -> tuple[_RegionFace, ...]:
    """Merge differently owned grid rectangles into maximal planar region faces."""
    solid_of = {
        atom_index: solid
        for solid, indices in enumerate(solid_atoms)
        for atom_index in indices
    }
    by_lower = {atom.bounds[0]: index for index, atom in enumerate(atoms)}
    by_upper = {atom.bounds[1]: index for index, atom in enumerate(atoms)}
    groups: dict[
        tuple[int, Fraction, int | None, int | None], set[tuple[Point, Point]]
    ] = {}
    for index, atom in enumerate(atoms):
        lower, upper = atom.bounds
        solid = solid_of[index]
        for axis in range(3):
            for sign in (-1, 1):
                if sign > 0:
                    coordinate = upper[axis]
                    neighbor = by_lower.get(_replace(lower, axis, coordinate))
                else:
                    coordinate = lower[axis]
                    neighbor = by_upper.get(_replace(upper, axis, coordinate))
                neighbor_solid = None if neighbor is None else solid_of[neighbor]
                if neighbor_solid == solid:
                    continue
                negative, positive = (
                    (solid, neighbor_solid) if sign > 0 else (neighbor_solid, solid)
                )
                groups.setdefault((axis, coordinate, negative, positive), set()).add(
                    (_replace(lower, axis, coordinate), _replace(upper, axis, coordinate))
                )
    faces: list[_RegionFace] = []
    for (axis, coordinate, negative, positive), cell_set in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            -1 if item[0][2] is None else item[0][2],
            -1 if item[0][3] is None else item[0][3],
        ),
    ):
        cells = tuple(sorted(cell_set))
        cell_by_lower = {lower: index for index, (lower, _) in enumerate(cells)}
        in_plane = tuple(value for value in range(3) if value != axis)
        pairs = [
            (index, neighbor)
            for index, (lower, upper) in enumerate(cells)
            for planar_axis in in_plane
            if (
                neighbor := cell_by_lower.get(
                    _replace(lower, planar_axis, upper[planar_axis])
                )
            )
            is not None
        ]
        roots = _union_find(len(cells), pairs)
        components: dict[int, list[tuple[Point, Point]]] = {}
        for index, root in enumerate(roots):
            components.setdefault(root, []).append(cells[index])
        sign = 1 if negative is not None else -1
        first_axis, second_axis = _face_axes(axis, sign)
        for component in sorted(components.values()):
            component_cells = tuple(component)
            origin = _replace(
                _replace(
                    component_cells[0][0],
                    first_axis,
                    min(lower[first_axis] for lower, _ in component_cells),
                ),
                second_axis,
                min(lower[second_axis] for lower, _ in component_cells),
            )
            faces.append(
                _RegionFace(
                    axis,
                    sign,
                    first_axis,
                    second_axis,
                    coordinate,
                    negative,
                    positive,
                    tuple(_cell_key(lower, upper, axis) for lower, upper in component),
                    _face_loops(component_cells, first_axis, second_axis),
                    origin,
                )
            )
    return tuple(faces)


def _build_model(
    atoms: tuple[_Atom, ...],
    plan: BRepPartitionPlan,
    tessellation: BRepTessellationPolicy | None,
    /,
) -> tuple[BRepModel, _PartitionLayout]:
    rank = {name: index for index, name in enumerate(plan.policy.region_precedence)}
    solid_atoms = _region_solids(atoms, rank)
    faces = _region_faces(atoms, solid_atoms)

    segment_faces: dict[tuple[Point, Point], set[int]] = {}
    point_segments: dict[Point, set[tuple[Point, Point]]] = {}
    for face_id, face in enumerate(faces):
        for loop in face.loops:
            for start, end in loop:
                key = (start, end) if start < end else (end, start)
                segment_faces.setdefault(key, set()).add(face_id)
                point_segments.setdefault(start, set()).add(key)
                point_segments.setdefault(end, set()).add(key)

    def is_vertex(point: Point, /) -> bool:
        # Collinear grid segments merge into one edge only through a degree-two
        # point whose two segments bound exactly the same faces.
        segments = point_segments[point]
        if len(segments) != 2:
            return True
        first, second = segments
        return (
            _segment_axis(first) != _segment_axis(second)
            or segment_faces[first] != segment_faces[second]
        )

    vertices: list[Point] = []
    vertex_index: dict[Point, int] = {}
    curves: list[ExactLineCurve] = []
    edge_vertices: list[tuple[int, int]] = []
    edge_segments: list[tuple[tuple[Point, Point], ...]] = []
    edge_index: dict[tuple[Point, Point], int] = {}
    pcurves: list[ExactLineCurve] = []
    coedge_edges: list[int] = []
    coedge_senses: list[int] = []
    face_loops: list[tuple[tuple[int, ...], ...]] = []
    face_edges: list[set[int]] = []
    face_bounds: list[np.ndarray] = []
    patches: list[PlanePatch] = []
    tags: list[str] = []

    def vertex(point: Point, /) -> int:
        index = vertex_index.get(point)
        if index is None:
            index = len(vertices)
            vertex_index[point] = index
            vertices.append(point)
        return index

    for face in faces:
        first_axis, second_axis = face.first_axis, face.second_axis
        origin_first = face.origin[first_axis]
        origin_second = face.origin[second_axis]
        loops: list[tuple[int, ...]] = []
        edges: set[int] = set()
        for loop in face.loops:
            offset = next(
                index for index, (start, _) in enumerate(loop) if is_vertex(start)
            )
            chains: list[list[tuple[Point, Point]]] = []
            for segment in loop[offset:] + loop[:offset]:
                if is_vertex(segment[0]):
                    chains.append([])
                chains[-1].append(segment)
            coedges: list[int] = []
            for chain in chains:
                start, end = chain[0][0], chain[-1][1]
                edge_key = (start, end) if start < end else (end, start)
                global_edge = edge_index.get(edge_key)
                if global_edge is None:
                    global_edge = len(curves)
                    edge_index[edge_key] = global_edge
                    curves.append(ExactLineCurve(*edge_key))
                    edge_vertices.append((vertex(edge_key[0]), vertex(edge_key[1])))
                    edge_segments.append(
                        tuple(
                            (first, last) if first < last else (last, first)
                            for first, last in chain
                        )
                    )
                coedge = len(coedge_edges)
                coedge_edges.append(global_edge)
                coedge_senses.append(1 if start == edge_key[0] else -1)
                pcurves.append(
                    ExactLineCurve(
                        *(
                            (
                                point[first_axis] - origin_first,
                                point[second_axis] - origin_second,
                            )
                            for point in edge_key
                        )
                    )
                )
                coedges.append(coedge)
                edges.add(global_edge)
            loops.append(tuple(coedges))
        face_loops.append(tuple(loops))
        face_edges.append(edges)
        patches.append(
            PlanePatch(
                np.asarray([float(value) for value in face.origin], dtype=np.float64),
                np.eye(3, dtype=np.float64)[first_axis],
                np.eye(3, dtype=np.float64)[second_axis],
            )
        )
        face_bounds.append(
            np.asarray(
                (
                    (0.0, 0.0),
                    (
                        float(
                            max(start[first_axis] for start, _ in face.loops[0])
                            - origin_first
                        ),
                        float(
                            max(start[second_axis] for start, _ in face.loops[0])
                            - origin_second
                        ),
                    ),
                ),
                dtype=np.float64,
            )
        )
        tags.append(f"native-box-face:{face.axis}:{face.sign}")

    shell_faces: list[tuple[int, ...]] = []
    shell_orientations: list[tuple[int, ...]] = []
    solid_shells: list[tuple[int, ...]] = []
    for solid in range(len(solid_atoms)):
        entries = tuple(
            (face_id, face.sign if face.negative_solid == solid else -face.sign)
            for face_id, face in enumerate(faces)
            if solid in (face.negative_solid, face.positive_solid)
        )
        edge_uses: dict[int, list[int]] = {}
        for position, (face_id, _) in enumerate(entries):
            for edge in face_edges[face_id]:
                edge_uses.setdefault(edge, []).append(position)
        if any(len(uses) != 2 for uses in edge_uses.values()):
            raise NotImplementedError(
                "Native box partition region solids must have manifold edges."
            )
        roots = _union_find(
            len(entries), [(uses[0], uses[1]) for uses in edge_uses.values()]
        )
        shells: dict[int, list[int]] = {}
        for position, root in enumerate(roots):
            shells.setdefault(root, []).append(position)

        def lowest_point(
            positions: list[int], /, entries: tuple[tuple[int, int], ...] = entries
        ) -> Point:
            return min(
                start
                for position in positions
                for loop in faces[entries[position][0]].loops
                for start, _ in loop
            )

        solid_shell_ids: list[int] = []
        # The outer shell carries the solid's lexicographically lowest point.
        for positions in sorted(shells.values(), key=lowest_point):
            solid_shell_ids.append(len(shell_faces))
            shell_faces.append(tuple(entries[position][0] for position in positions))
            shell_orientations.append(
                tuple(entries[position][1] for position in positions)
            )
        solid_shells.append(tuple(solid_shell_ids))
    geometry = BRepGeometry(
        vertex_points=np.asarray(
            [[float(value) for value in point] for point in vertices], dtype=np.float64
        ),
        vertex_exact_points=tuple(vertices),
        curves=tuple(curves),
        edge_curves=tuple(range(len(curves))),
        edge_ranges=np.tile(np.asarray([0.0, 1.0], dtype=np.float64), (len(curves), 1)),
        edge_vertices=tuple(edge_vertices),
        pcurves=tuple(pcurves),
        coedge_edges=tuple(coedge_edges),
        coedge_senses=tuple(coedge_senses),
        face_loops=tuple(face_loops),
        shell_faces=tuple(shell_faces),
        shell_orientations=tuple(shell_orientations),
        solid_shells=tuple(solid_shells),
    )
    source_digest = canonical_fingerprint(
        {
            "kind": "native-axis-box-partition-result",
            "plan": plan.plan_id,
            "solids": tuple(
                (
                    atoms[indices[0]].owner,
                    tuple(
                        tuple(
                            tuple(coefficient_payload(value) for value in point)
                            for point in atoms[index].bounds
                        )
                        for index in indices
                    ),
                )
                for indices in solid_atoms
            ),
        }
    )
    model = assemble_brep_model(
        geometry,
        tuple(patches),
        np.asarray(face_bounds),
        np.ones(len(patches), dtype=np.float64),
        tuple(tags),
        coordinate_contract=plan.coordinate_contract,
        source_id=f"native-brep-partition:{plan.plan_id}",
        source_digest=source_digest,
        curve_surface_tolerance=0.0,
        tessellation=tessellation,
    )
    layout = _PartitionLayout(
        solid_atoms,
        tuple(atoms[indices[0]].owner for indices in solid_atoms),
        tuple(face.cells for face in faces),
        tuple(edge_segments),
    )
    return model, layout


def partition_native_boxes(
    plan: BRepPartitionPlan,
    /,
    *,
    tessellation: BRepTessellationPolicy | None = None,
) -> BRepPartitionResult:
    """Partition exact native ``brep_box`` operands without external CAD engines."""
    from ._partition import (
        _entity_occurrence_id,
        _operand_selection,
        BRepPartitionPatch,
        BRepPartitionRegion,
        BRepPartitionReport,
        BRepPartitionResult,
        cad_revision_from_brep_model,
    )

    atoms, operand_boxes = _arrangement(plan)
    model, layout = _build_model(atoms, plan, tessellation)
    source_revision, source_occurrence_ids = _composite_revision(plan, plan.operands)
    target_revision = cad_revision_from_brep_model(model)
    rank = {name: index for index, name in enumerate(plan.policy.region_precedence)}
    solid_owner = layout.solid_owners
    source_occurrence_by_operand: dict[str, tuple[str, ...]] = {}
    source_face_keys: dict[str, tuple[tuple[str, FaceKey], ...]] = {}
    for operand in plan.operands:
        geometry = operand.model.geometry
        selector = _operand_selection(operand).selectors[0]
        source_occurrence_by_operand[operand.operand_id] = (
            source_occurrence_ids[(operand.operand_id, selector.occurrence_id)],
        )
        source_face_keys[operand.operand_id] = tuple(
            (
                source_occurrence_ids[
                    (
                        operand.operand_id,
                        _entity_occurrence_id(
                            BRepEntityId(
                                operand.model.source_id,
                                operand.model.source_revision,
                                "face",
                                face,
                                occurrence.path,
                            )
                        ),
                    )
                ],
                _source_face_key(operand.model, face),
            )
            for occurrence in geometry.iter_occurrences()
            for face in geometry.topology.solid_faces[occurrence.solid_index]
        )
    target_solid_occurrences: dict[int, tuple[str, tuple[str, ...]]] = {}
    target_face_occurrences: dict[tuple[int, int], str] = {}
    for occurrence in model.geometry.iter_occurrences():
        solid = occurrence.solid_index
        path = occurrence.path
        target_solid_occurrences[solid] = (
            _entity_occurrence_id(
                BRepEntityId(model.source_id, model.source_revision, "solid", solid, path)
            ),
            path,
        )
        for face in model.topology.solid_faces[solid]:
            target_face_occurrences[(solid, face)] = _entity_occurrence_id(
                BRepEntityId(model.source_id, model.source_revision, "face", face, path)
            )

    def assembly_chain(revision: CADRevision, occurrence_id: str, /) -> tuple[str, ...]:
        chain: list[str] = []
        parent = revision.occurrence(occurrence_id).parent_occurrence_id
        while parent is not None:
            chain.append(parent)
            parent = revision.occurrence(parent).parent_occurrence_id
        return tuple(chain)

    correspondence_pairs: set[tuple[str, str]] = set()
    for solid_index, atom_indices in enumerate(layout.solid_atoms):
        target_solid_id, occurrence_path = target_solid_occurrences[solid_index]
        target_assemblies = assembly_chain(target_revision, target_solid_id)
        for operand_id, (lower, upper) in operand_boxes.items():
            # A region solid descends from every operand containing one of its atoms.
            if any(
                all(
                    a >= b and c <= d
                    for a, c, b, d in zip(
                        atoms[index].bounds[0],
                        atoms[index].bounds[1],
                        lower,
                        upper,
                        strict=True,
                    )
                )
                for index in atom_indices
            ):
                source_id = source_occurrence_by_operand[operand_id][0]
                correspondence_pairs.add((source_id, target_solid_id))
                source_assemblies = assembly_chain(source_revision, source_id)
                if source_assemblies and not target_assemblies:
                    raise RuntimeError(
                        "Native box partition cells must retain their source assembly occurrence."
                    )
                for depth, source_assembly in enumerate(source_assemblies):
                    correspondence_pairs.add(
                        (
                            source_assembly,
                            target_assemblies[min(depth, len(target_assemblies) - 1)],
                        )
                    )
        for face_index in model.topology.solid_faces[solid_index]:
            target_face_id = target_face_occurrences[(solid_index, face_index)]
            for face_sources in source_face_keys.values():
                for source_id, source_key in face_sources:
                    # Faces, edges and vertices descend from a source face through
                    # positive-area grid cells, positive-length grid segments and
                    # exact incident points of its closed rectangle.
                    if not any(
                        _contained_face(cell, source_key)
                        for cell in layout.face_cells[face_index]
                    ):
                        continue
                    correspondence_pairs.add((source_id, target_face_id))
                    for edge in model.topology.face_edges[face_index]:
                        if any(
                            _point_in_face(start, source_key)
                            and _point_in_face(end, source_key)
                            for start, end in layout.edge_segments[edge]
                        ):
                            correspondence_pairs.add(
                                (
                                    source_id,
                                    _entity_occurrence_id(
                                        BRepEntityId(
                                            model.source_id,
                                            model.source_revision,
                                            "edge",
                                            edge,
                                            occurrence_path,
                                        )
                                    ),
                                )
                            )
                        for vertex in model.geometry.edge_vertices[edge]:
                            exact_point = model.geometry.vertex_exact_points[vertex]
                            if exact_point is None or not _point_in_face(
                                exact_point, source_key
                            ):
                                continue
                            correspondence_pairs.add(
                                (
                                    source_id,
                                    _entity_occurrence_id(
                                        BRepEntityId(
                                            model.source_id,
                                            model.source_revision,
                                            "vertex",
                                            vertex,
                                            occurrence_path,
                                        )
                                    ),
                                )
                            )
    certificate_id = canonical_fingerprint(
        {
            "kind": "native-axis-box-partition-history",
            "plan": plan.plan_id,
            "source_revision": source_revision.revision_id,
            "target_revision": target_revision.revision_id,
            "correspondences": tuple(sorted(correspondence_pairs)),
        }
    )
    correspondences = tuple(
        OccurrenceCorrespondence(source_id, target_id, certificate_id)
        for source_id, target_id in sorted(correspondence_pairs)
    )
    coverage = AssociationCoverageEvidence(
        True, True, certificate_id, "native-axis-box-arrangement"
    )
    transaction = OccurrenceCorrespondenceTransaction(
        certificate_id,
        source_revision.revision_id,
        target_revision.revision_id,
        tuple(correspondences),
        frozenset(),
        frozenset(),
        coverage,
    )
    graph = AssociationGraph(source_revision, target_revision, transaction)
    regions = tuple(
        BRepPartitionRegion(
            name,
            tuple(
                model.solid_ids[index]
                for index, owner in enumerate(solid_owner)
                if owner == name
            ),
        )
        for name in plan.policy.region_precedence
    )
    region_for_solid = dict(enumerate(solid_owner))
    grouped_faces: dict[tuple[str, ...], list[BRepEntityId]] = {}
    for face_index, solids in enumerate(model.topology.face_solids):
        adjacent = tuple(
            sorted((region_for_solid[index] for index in solids), key=rank.__getitem__)
        )
        if len(adjacent) not in (1, 2):
            raise RuntimeError(
                "Native box arrangement produced a nonmanifold partition face."
            )
        grouped_faces.setdefault(adjacent, []).append(model.face_ids[face_index])
    patches = tuple(
        BRepPartitionPatch(
            "interface:" + ":".join(adjacent),
            tuple(face_ids),
            adjacent,
        )
        for adjacent, face_ids in sorted(
            grouped_faces.items(),
            key=lambda item: tuple(rank[value] for value in item[0]),
        )
    )
    source_edge_ids = {edge.source_occurrence_id for edge in correspondences}
    target_edge_ids = {edge.target_occurrence_id for edge in correspondences}
    report = BRepPartitionReport(
        plan.plan_id,
        model.model_id,
        source_revision.revision_id,
        target_revision.revision_id,
        certificate_id,
        sum(value.kind == "solid" for value in source_revision.occurrences),
        sum(value.kind == "face" for value in source_revision.occurrences),
        model.topology.num_solids,
        model.topology.num_faces,
        sum(
            value.occurrence_id not in source_edge_ids
            for value in source_revision.occurrences
        ),
        sum(
            value.occurrence_id not in target_edge_ids
            for value in target_revision.occurrences
        ),
    )
    return BRepPartitionResult(
        model,
        target_revision,
        graph,
        regions,
        patches,
        report,
    )


__all__ = ["partition_native_boxes"]
