#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Capacity-bounded 3-D multivalued cut-cell topology over block-AMR leaves.

The geometry is exact relative to the piecewise-affine map and piecewise-linear
level-set field on the declared conforming subcell tetrahedralization. Smooth maps
and level sets are therefore approximated by that declared finite geometry.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import permutations
from math import prod
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cell_mesh import CellMesh
from ._canonical import (
    BlockAMRResourcePlan,
    canonicalize_patch_hierarchy,
    CanonicalPatchHierarchy,
)
from ._core import BlockHierarchyTopology
from ._mapped_geometry import PatchCoordinateMapSet
from ._variable import VariablePatchHierarchyTopology


if TYPE_CHECKING:
    from ..finite_volume._unstructured import UnstructuredFiniteVolumePlan


CoordinateMap = Callable[[Array, Array, Any], ArrayLike]
LevelSet = Callable[[Array, Array, Any], ArrayLike]

_FACE_INTERNAL = 0
_FACE_PHYSICAL = 1
_FACE_EMBEDDED = 2


@dataclass(frozen=True)
class _Vertex:
    key: tuple[Any, ...]
    reference: np.ndarray
    point: np.ndarray
    phi: float
    body_tag: int


@dataclass(frozen=True)
class _Face:
    vertices: tuple[_Vertex, ...]
    kind: int
    axis: int
    side: int
    body_tag: int


@dataclass(frozen=True)
class _Fragment:
    faces: tuple[_Face, ...]
    volume: float
    centroid: np.ndarray


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def root(self, value: int) -> int:
        current = int(value)
        while self.parent[current] != current:
            self.parent[current] = self.parent[self.parent[current]]
            current = self.parent[current]
        return current

    def join(self, left: int, right: int) -> None:
        first = self.root(left)
        second = self.root(right)
        if first != second:
            self.parent[max(first, second)] = min(first, second)


class EmbeddedLevelSetBody(StrictModule, NonTrainableState):
    """One negative-inside solid body with a stable physical boundary tag."""

    level_set: LevelSet = eqx.field(static=True)
    field_id: str = eqx.field(static=True)
    body_tag: int = eqx.field(static=True)
    body_id: str = eqx.field(static=True)

    def __init__(self, level_set: LevelSet, field_id: str, body_tag: int, /):
        field = str(field_id)
        tag = int(body_tag)
        if not callable(level_set) or not field or tag < 0:
            raise ValueError(
                "Embedded bodies require a callable, field ID, and body tag."
            )
        self.level_set = level_set
        self.field_id = field
        self.body_tag = tag
        self.body_id = canonical_fingerprint(
            {"kind": "embedded-level-set-body", "field": field, "tag": tag}
        )


class EmbeddedLevelSetBodySet(StrictModule, NonTrainableState):
    """Tagged solid CSG over positive-fluid level-set operands."""

    bodies: tuple[EmbeddedLevelSetBody, ...]
    operation: str = eqx.field(static=True)
    body_signs: tuple[int, ...] = eqx.field(static=True)
    body_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        bodies: Sequence[EmbeddedLevelSetBody],
        /,
        *,
        operation: str = "union",
        body_signs: Sequence[int] | None = None,
    ):
        bodies_ = tuple(sorted(bodies, key=lambda body: body.body_tag))
        operation_ = str(operation)
        signs = (
            (1,) * len(bodies_)
            if body_signs is None
            else tuple(int(value) for value in body_signs)
        )
        if (
            not bodies_
            or not all(isinstance(body, EmbeddedLevelSetBody) for body in bodies_)
            or len({body.body_tag for body in bodies_}) != len(bodies_)
            or operation_ not in ("union", "intersection")
            or len(signs) != len(bodies_)
            or any(value not in (-1, 1) for value in signs)
        ):
            raise ValueError(
                "Embedded body CSG requires unique tags, union/intersection, and ±1 signs."
            )
        self.bodies = bodies_
        self.operation = operation_
        self.body_signs = signs
        self.body_set_id = canonical_fingerprint(
            {
                "kind": "embedded-level-set-body-csg",
                "operation": operation_,
                "bodies": [body.body_id for body in bodies_],
                "body_signs": signs,
            }
        )

    def evaluate(
        self,
        points: Array,
        time: Array,
        args: Any,
        /,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.stack(
            [
                np.asarray(body.level_set(points, time, args), dtype=float)
                for body in self.bodies
            ],
            axis=0,
        )
        if (
            values.ndim != 2
            or values.shape[1] != points.shape[0]
            or np.any(~np.isfinite(values))
        ):
            raise ValueError(
                "Every embedded level set must return one finite scalar per point."
            )
        signed = np.asarray(self.body_signs, dtype=float)[:, None] * values
        if self.operation == "union":
            active = np.argmin(signed, axis=0)
            composite = np.min(signed, axis=0)
        else:
            active = np.argmax(signed, axis=0)
            composite = np.max(signed, axis=0)
        tags = np.asarray(
            [self.bodies[int(index)].body_tag for index in active], dtype=np.int32
        )
        return composite, tags


class CutCellSignTopology(StrictModule, NonTrainableState):
    """Sample-lattice sign topology used for event localization."""

    topology_id: str = eqx.field(static=True)
    negative_sample_count: int = eqx.field(static=True)
    minimum_margin: float = eqx.field(static=True)


class MultivaluedCutCellEvidence(StrictModule, NonTrainableState):
    """Host-certified topology, capacity, and geometric closure evidence."""

    leaf_cell_count: int = eqx.field(static=True)
    regular_cell_count: int = eqx.field(static=True)
    covered_cell_count: int = eqx.field(static=True)
    cut_cell_count: int = eqx.field(static=True)
    multivalued_cell_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    maximum_components_in_cell: int = eqx.field(static=True)
    minimum_predicate_margin: float = eqx.field(static=True)
    maximum_volume_closure_defect: float = eqx.field(static=True)
    maximum_face_closure_defect: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    valid: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        leaf_cell_count: int,
        regular_cell_count: int,
        covered_cell_count: int,
        cut_cell_count: int,
        multivalued_cell_count: int,
        component_count: int,
        face_count: int,
        maximum_components_in_cell: int,
        minimum_predicate_margin: float,
        maximum_volume_closure_defect: float,
        maximum_face_closure_defect: float,
        tolerance: float,
    ):
        counts = (
            leaf_cell_count,
            regular_cell_count,
            covered_cell_count,
            cut_cell_count,
            multivalued_cell_count,
            component_count,
            face_count,
            maximum_components_in_cell,
        )
        margin = float(minimum_predicate_margin)
        volume_defect = float(maximum_volume_closure_defect)
        face_defect = float(maximum_face_closure_defect)
        tolerance_ = float(tolerance)
        if any(int(value) < 0 for value in counts) or any(
            not np.isfinite(value) or value < 0.0
            for value in (margin, volume_defect, face_defect, tolerance_)
        ):
            raise ValueError(
                "Cut-cell evidence counts and defects must be finite/nonnegative."
            )
        valid = volume_defect <= tolerance_ and face_defect <= tolerance_
        (
            self.leaf_cell_count,
            self.regular_cell_count,
            self.covered_cell_count,
            self.cut_cell_count,
            self.multivalued_cell_count,
            self.component_count,
            self.face_count,
            self.maximum_components_in_cell,
        ) = tuple(int(value) for value in counts)
        self.minimum_predicate_margin = margin
        self.maximum_volume_closure_defect = volume_defect
        self.maximum_face_closure_defect = face_defect
        self.tolerance = tolerance_
        self.valid = valid
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "multivalued-cut-cell-evidence",
                "counts": counts,
                "minimum_predicate_margin": margin,
                "volume_closure": volume_defect,
                "face_closure": face_defect,
                "tolerance": tolerance_,
                "valid": valid,
            }
        )


class MultivaluedCutCellComplex(StrictModule, NonTrainableState):
    """Padded component/face graph plus exact active polyhedral mesh."""

    hierarchy: CanonicalPatchHierarchy
    mesh: CellMesh
    component_active: Array
    component_levels: Array
    component_cell_coordinates: Array
    component_slots: Array
    component_volumes: Array
    component_centers: Array
    component_volume_fractions: Array
    component_tetrahedra: tuple[
        tuple[tuple[tuple[float, float, float], ...], ...], ...
    ] = eqx.field(static=True)
    face_active: Array
    face_owner_components: Array
    face_neighbour_components: Array
    face_mesh_indices: Array
    face_kinds: Array
    face_body_tags: Array
    face_axes: Array
    face_sides: Array
    face_centers: Array
    face_area_vectors: Array
    face_measures: Array
    evidence: MultivaluedCutCellEvidence
    body_set_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    component_capacity: int = eqx.field(static=True)
    face_capacity: int = eqx.field(static=True)
    active_component_count: int = eqx.field(static=True)
    active_face_count: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        hierarchy: CanonicalPatchHierarchy,
        mesh: CellMesh,
        component_active: ArrayLike,
        component_levels: ArrayLike,
        component_cell_coordinates: ArrayLike,
        component_slots: ArrayLike,
        component_volumes: ArrayLike,
        component_centers: ArrayLike,
        component_volume_fractions: ArrayLike,
        component_tetrahedra: Sequence[Sequence[ArrayLike]],
        face_active: ArrayLike,
        face_owner_components: ArrayLike,
        face_neighbour_components: ArrayLike,
        face_mesh_indices: ArrayLike,
        face_kinds: ArrayLike,
        face_body_tags: ArrayLike,
        face_axes: ArrayLike,
        face_sides: ArrayLike,
        face_centers: ArrayLike,
        face_area_vectors: ArrayLike,
        face_measures: ArrayLike,
        evidence: MultivaluedCutCellEvidence,
        body_set_id: str,
    ):
        if not isinstance(hierarchy, CanonicalPatchHierarchy) or not isinstance(
            mesh, CellMesh
        ):
            raise TypeError("Cut-cell complex requires canonical hierarchy and CellMesh.")
        active = np.asarray(component_active, dtype=bool)
        levels = np.asarray(component_levels)
        coordinates = np.asarray(component_cell_coordinates)
        slots = np.asarray(component_slots)
        volumes = np.asarray(component_volumes)
        centers = np.asarray(component_centers)
        fractions = np.asarray(component_volume_fractions)
        tetrahedra = tuple(
            tuple(
                tuple(
                    tuple(float(value) for value in point) for point in np.asarray(tetra)
                )
                for tetra in component
            )
            for component in component_tetrahedra
        )
        face_active_ = np.asarray(face_active, dtype=bool)
        owners = np.asarray(face_owner_components)
        neighbours = np.asarray(face_neighbour_components)
        mesh_indices = np.asarray(face_mesh_indices)
        kinds = np.asarray(face_kinds)
        tags = np.asarray(face_body_tags)
        axes = np.asarray(face_axes)
        sides = np.asarray(face_sides)
        face_centers_ = np.asarray(face_centers)
        area_vectors = np.asarray(face_area_vectors)
        measures = np.asarray(face_measures)
        component_capacity = active.size
        face_capacity = face_active_.size
        dimension = hierarchy.dimension
        if (
            active.ndim != 1
            or levels.shape != active.shape
            or coordinates.shape != (component_capacity, dimension)
            or slots.shape != active.shape
            or volumes.shape != active.shape
            or centers.shape != (component_capacity, dimension)
            or fractions.shape != active.shape
            or owners.shape != (face_capacity,)
            or neighbours.shape != (face_capacity,)
            or kinds.shape != (face_capacity,)
            or mesh_indices.shape != (face_capacity,)
            or tags.shape != (face_capacity,)
            or axes.shape != (face_capacity,)
            or sides.shape != (face_capacity,)
            or face_centers_.shape != (face_capacity, dimension)
            or area_vectors.shape != (face_capacity, dimension)
            or measures.shape != (face_capacity,)
        ):
            raise ValueError(
                "Cut-cell component or face arrays have incompatible shapes."
            )
        if len(tetrahedra) != int(np.count_nonzero(active)) or any(
            not component or any(len(tetra) != 4 for tetra in component)
            for component in tetrahedra
        ):
            raise ValueError("Every active cut component requires affine tetrahedra.")
        if int(np.count_nonzero(active)) != mesh.connectivity.cell_count:
            raise ValueError(
                "Active cut components must match the polyhedral mesh cells."
            )
        if np.any(face_active_ & ((owners < 0) | (owners >= component_capacity))):
            raise ValueError("Active cut faces require in-range owner components.")
        safe_neighbours = np.maximum(neighbours, 0)
        if np.any(
            face_active_
            & (neighbours >= 0)
            & ((safe_neighbours >= component_capacity) | ~active[safe_neighbours])
        ):
            raise ValueError("Active interior cut faces require active neighbours.")
        self.hierarchy = hierarchy
        self.mesh = mesh
        self.component_active = jnp.asarray(active)
        self.component_levels = jnp.asarray(levels, dtype=jnp.int32)
        self.component_cell_coordinates = jnp.asarray(coordinates, dtype=jnp.int32)
        self.component_slots = jnp.asarray(slots, dtype=jnp.int32)
        self.component_volumes = jnp.asarray(volumes)
        self.component_centers = jnp.asarray(centers)
        self.component_volume_fractions = jnp.asarray(fractions)
        self.face_active = jnp.asarray(face_active_)
        self.face_owner_components = jnp.asarray(owners, dtype=jnp.int32)
        self.face_mesh_indices = jnp.asarray(mesh_indices, dtype=jnp.int32)
        self.face_neighbour_components = jnp.asarray(neighbours, dtype=jnp.int32)
        self.component_tetrahedra = tetrahedra
        self.face_kinds = jnp.asarray(kinds, dtype=jnp.int32)
        self.face_body_tags = jnp.asarray(tags, dtype=jnp.int32)
        self.face_axes = jnp.asarray(axes, dtype=jnp.int32)
        self.face_sides = jnp.asarray(sides, dtype=jnp.int32)
        self.face_centers = jnp.asarray(face_centers_)
        self.face_area_vectors = jnp.asarray(area_vectors)
        self.face_measures = jnp.asarray(measures)
        self.evidence = evidence
        self.body_set_id = str(body_set_id)
        self.component_capacity = component_capacity
        self.face_capacity = face_capacity
        self.active_component_count = int(np.count_nonzero(active))
        self.active_face_count = int(np.count_nonzero(face_active_))
        self.topology_id = canonical_fingerprint(
            {
                "kind": "multivalued-cut-cell-complex",
                "hierarchy": hierarchy.topology_id,
                "mesh": mesh.topology_id,
                "component_levels": array_tree_fingerprint(levels[active]),
                "component_cells": array_tree_fingerprint(coordinates[active]),
                "component_slots": array_tree_fingerprint(slots[active]),
                "face_routes": array_tree_fingerprint(
                    np.stack((owners[face_active_], neighbours[face_active_]), axis=-1)
                ),
                "face_kinds": array_tree_fingerprint(kinds[face_active_]),
                "body_set": body_set_id,
            }
        )
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "multivalued-cut-cell-geometry",
                "topology": self.topology_id,
                "mesh": mesh.geometry_id,
                "volumes": array_tree_fingerprint(volumes[active]),
                "faces": array_tree_fingerprint(area_vectors[face_active_]),
            }
        )

    @property
    def component_count(self) -> int:
        return self.active_component_count

    @property
    def face_count(self) -> int:
        return self.active_face_count

    def finite_volume_plan(
        self,
        /,
        *,
        field_name: str = "state",
        component_names: Sequence[str] = ("value",),
    ) -> UnstructuredFiniteVolumePlan:
        """Lower the active component complex to the canonical polyhedral FV plan."""
        from ..finite_volume._unstructured import UnstructuredFiniteVolumePlan

        active = np.asarray(self.face_active, dtype=bool)
        boundary = active & (np.asarray(self.face_neighbour_components) < 0)
        mesh_indices = np.asarray(self.face_mesh_indices, dtype=np.int32)
        kinds = np.asarray(self.face_kinds, dtype=np.int32)
        axes = np.asarray(self.face_axes, dtype=np.int32)
        sides = np.asarray(self.face_sides, dtype=np.int32)
        tags = np.asarray(self.face_body_tags, dtype=np.int32)
        groups: dict[str, list[int]] = {}
        for face in np.flatnonzero(boundary):
            if kinds[face] == _FACE_EMBEDDED:
                name = f"embedded-{int(tags[face])}"
            elif kinds[face] == _FACE_PHYSICAL:
                name = f"physical-{int(axes[face])}-{int(sides[face])}"
            else:
                raise ValueError("Boundary cut face has no physical ownership kind.")
            groups.setdefault(name, []).append(int(mesh_indices[face]))
        return UnstructuredFiniteVolumePlan.from_cell_mesh(
            self.mesh,
            field_name=field_name,
            component_names=component_names,
            boundary_face_groups={
                name: np.asarray(indices, dtype=np.int32)
                for name, indices in groups.items()
            },
        )


def _oriented_tetra_faces(
    vertices: tuple[_Vertex, ...],
) -> tuple[tuple[_Vertex, ...], ...]:
    result = []
    for omitted in range(4):
        face = [vertices[index] for index in range(4) if index != omitted]
        points = np.stack([vertex.point for vertex in face])
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        opposite = vertices[omitted].point
        if float(np.dot(normal, opposite - points[0])) > 0.0:
            face[1], face[2] = face[2], face[1]
        result.append(tuple(face))
    return tuple(result)


def _edge_intersection(left: _Vertex, right: _Vertex) -> _Vertex:
    denominator = left.phi - right.phi
    fraction = left.phi / denominator
    reference = left.reference + fraction * (right.reference - left.reference)
    point = left.point + fraction * (right.point - left.point)
    if left.key <= right.key:
        first_key, second_key, canonical_fraction = left.key, right.key, fraction
    else:
        first_key, second_key, canonical_fraction = right.key, left.key, 1.0 - fraction
    tag = left.body_tag if abs(left.phi) <= abs(right.phi) else right.body_tag
    return _Vertex(
        ("edge", first_key, second_key, round(float(canonical_fraction), 15)),
        reference,
        point,
        0.0,
        tag,
    )


def _clip_face_positive(face: tuple[_Vertex, ...]) -> tuple[_Vertex, ...]:
    output: list[_Vertex] = []
    for left, right in zip(face, face[1:] + face[:1], strict=True):
        left_inside = left.phi > 0.0
        right_inside = right.phi > 0.0
        if left_inside:
            output.append(left)
        if left_inside != right_inside:
            output.append(_edge_intersection(left, right))
    unique: list[_Vertex] = []
    for vertex in output:
        if not unique or vertex.key != unique[-1].key:
            unique.append(vertex)
    if len(unique) > 1 and unique[0].key == unique[-1].key:
        unique.pop()
    return tuple(unique)


def _polygon_area_vector(vertices: tuple[_Vertex, ...]) -> np.ndarray:
    points = np.stack([vertex.point for vertex in vertices])
    origin = np.mean(points, axis=0)
    area = np.zeros((3,), dtype=float)
    for left, right in zip(points, np.roll(points, -1, axis=0), strict=True):
        area += 0.5 * np.cross(left - origin, right - origin)
    return area


def _cap_face(vertices: tuple[_Vertex, ...]) -> tuple[_Vertex, ...]:
    intersections: dict[tuple[Any, ...], _Vertex] = {}
    for left_index in range(4):
        for right_index in range(left_index + 1, 4):
            left = vertices[left_index]
            right = vertices[right_index]
            if (left.phi > 0.0) != (right.phi > 0.0):
                value = _edge_intersection(left, right)
                intersections[value.key] = value
    values = tuple(intersections[key] for key in sorted(intersections, key=repr))
    if len(values) < 3:
        return ()
    points = np.stack([vertex.point for vertex in values])
    matrix = np.stack(
        (
            vertices[1].point - vertices[0].point,
            vertices[2].point - vertices[0].point,
            vertices[3].point - vertices[0].point,
        ),
        axis=0,
    )
    gradient = np.linalg.solve(
        matrix,
        np.asarray([vertex.phi - vertices[0].phi for vertex in vertices[1:]]),
    )
    outward = -gradient / np.linalg.norm(gradient)
    center = np.mean(points, axis=0)
    first = points[0] - center
    first = first / np.linalg.norm(first)
    second = np.cross(outward, first)
    order = np.argsort(
        np.asarray(
            [
                np.arctan2(
                    float(np.dot(point - center, second)),
                    float(np.dot(point - center, first)),
                )
                for point in points
            ]
        ),
        kind="stable",
    )
    ordered = tuple(values[int(index)] for index in order)
    if float(np.dot(_polygon_area_vector(ordered), outward)) < 0.0:
        ordered = tuple(reversed(ordered))
    return ordered


def _face_classification(
    face: tuple[_Vertex, ...],
    reference_lower: np.ndarray,
    reference_upper: np.ndarray,
    tolerance: float,
) -> tuple[int, int, int]:
    references = np.stack([vertex.reference for vertex in face])
    for axis in range(3):
        if np.all(np.abs(references[:, axis] - reference_lower[axis]) <= tolerance):
            return _FACE_PHYSICAL, axis, 0
        if np.all(np.abs(references[:, axis] - reference_upper[axis]) <= tolerance):
            return _FACE_PHYSICAL, axis, 1
    return _FACE_INTERNAL, -1, -1


def _polyhedron_moments(
    faces: tuple[_Face, ...],
    tolerance: float,
) -> tuple[float, np.ndarray]:
    unique = {vertex.key: vertex.point for face in faces for vertex in face.vertices}
    reference = np.mean(np.stack(tuple(unique.values())), axis=0)
    volume = 0.0
    first_moment = np.zeros((3,), dtype=float)
    for face in faces:
        points = np.stack([vertex.point for vertex in face.vertices])
        for index in range(1, points.shape[0] - 1):
            first, second, third = points[0], points[index], points[index + 1]
            tetra_volume = float(
                np.dot(first - reference, np.cross(second - reference, third - reference))
                / 6.0
            )
            volume += tetra_volume
            first_moment += tetra_volume * (reference + first + second + third) / 4.0
    if not np.isfinite(volume) or volume <= tolerance:
        raise ValueError(
            "Clipped tetrahedron has nonpositive or unresolved fluid volume."
        )
    return volume, first_moment / volume


def _clip_tetrahedron(
    vertices: tuple[_Vertex, ...],
    reference_lower: np.ndarray,
    reference_upper: np.ndarray,
    tolerance: float,
) -> _Fragment | None:
    positive = sum(vertex.phi > 0.0 for vertex in vertices)
    if positive == 0:
        return None
    faces: list[_Face] = []
    for original in _oriented_tetra_faces(vertices):
        clipped = _clip_face_positive(original)
        if len(clipped) < 3:
            continue
        area = float(np.linalg.norm(_polygon_area_vector(clipped)))
        if area <= tolerance:
            continue
        kind, axis, side = _face_classification(
            clipped, reference_lower, reference_upper, tolerance
        )
        faces.append(_Face(clipped, kind, axis, side, -1))
    if positive != 4:
        cap = _cap_face(vertices)
        if len(cap) >= 3:
            tags = {vertex.body_tag for vertex in cap}
            if len(tags) != 1:
                raise ValueError(
                    "Intersecting body tags within one cut simplex require explicit CSG resolution."
                )
            faces.append(_Face(cap, _FACE_EMBEDDED, -1, -1, next(iter(tags))))
    if len(faces) < 4:
        raise ValueError("Clipped tetrahedron does not form a closed fluid polyhedron.")
    volume, centroid = _polyhedron_moments(tuple(faces), tolerance)
    return _Fragment(tuple(faces), volume, centroid)


def _tetrahedra_in_subcube(
    lower: tuple[int, int, int],
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    result = []
    for ordering in permutations(range(3)):
        point = np.asarray(lower, dtype=np.int64)
        vertices = [tuple(int(value) for value in point)]
        for axis in ordering:
            point = point.copy()
            point[axis] += 1
            vertices.append(tuple(int(value) for value in point))
        result.append(tuple(vertices))
    return tuple(result)


def _leaf_cells(
    hierarchy: CanonicalPatchHierarchy,
) -> tuple[tuple[int, tuple[int, ...], str], ...]:
    cells = []
    for level in hierarchy.levels:
        for bucket in level.buckets:
            leaf = np.asarray(bucket.leaf_active, dtype=bool)
            for lane, box in enumerate(bucket.boxes):
                if box is None:
                    continue
                for local in np.argwhere(leaf[lane]):
                    coordinate = tuple(
                        int(start) + int(offset)
                        for start, offset in zip(box.lower, local, strict=True)
                    )
                    cells.append((level.level, coordinate, box.box_id))
    return tuple(sorted(cells, key=lambda item: (item[0], item[1], item[2])))


def _fine_scale(hierarchy: CanonicalPatchHierarchy, level: int) -> int:
    return prod(
        hierarchy.levels[index].refinement_ratio
        for index in range(level, len(hierarchy.levels) - 1)
    )


def _fragment_face_key(face: _Face) -> tuple[tuple[Any, ...], ...]:
    return tuple(sorted((vertex.key for vertex in face.vertices), key=repr))


def _cell_components(
    fragments: tuple[_Fragment, ...],
) -> tuple[tuple[int, ...], ...]:
    if not fragments:
        return ()
    union = _UnionFind(len(fragments))
    incidents: dict[tuple[tuple[Any, ...], ...], list[int]] = {}
    for fragment_index, fragment in enumerate(fragments):
        for face in fragment.faces:
            if face.kind == _FACE_INTERNAL:
                incidents.setdefault(_fragment_face_key(face), []).append(fragment_index)
    for values in incidents.values():
        if len(values) > 2:
            raise ValueError("Cut-simplex internal faces are nonmanifold.")
        if len(values) == 2:
            union.join(values[0], values[1])
    groups: dict[int, list[int]] = {}
    for index in range(len(fragments)):
        groups.setdefault(union.root(index), []).append(index)
    return tuple(tuple(groups[key]) for key in sorted(groups))


class MultivaluedCutCellPlan(StrictModule, NonTrainableState):
    """Prepare a bounded 3-D piecewise-linear multivalued cut complex."""

    hierarchy: CanonicalPatchHierarchy
    coordinate_map: CoordinateMap | PatchCoordinateMapSet = eqx.field(static=True)
    coordinate_map_id: str = eqx.field(static=True)
    bodies: EmbeddedLevelSetBodySet
    resources: BlockAMRResourcePlan
    subdivision: int = eqx.field(static=True)
    predicate_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: BlockHierarchyTopology | VariablePatchHierarchyTopology,
        coordinate_map: CoordinateMap | PatchCoordinateMapSet,
        coordinate_map_id: str,
        bodies: EmbeddedLevelSetBodySet,
        resources: BlockAMRResourcePlan,
        /,
        *,
        subdivision: int = 1,
        predicate_tolerance: float = 1.0e-12,
    ):
        hierarchy = canonicalize_patch_hierarchy(topology)
        map_id = str(coordinate_map_id)
        subdivision_ = int(subdivision)
        tolerance = float(predicate_tolerance)
        if hierarchy.dimension != 3:
            raise ValueError("Multivalued cut-cell preparation currently requires 3-D.")
        if not isinstance(coordinate_map, PatchCoordinateMapSet) and not callable(
            coordinate_map
        ):
            raise ValueError("Cut-cell preparation requires a coordinate map.")
        if not map_id:
            raise ValueError("Cut-cell preparation requires a stable coordinate-map ID.")
        if not isinstance(bodies, EmbeddedLevelSetBodySet) or not isinstance(
            resources, BlockAMRResourcePlan
        ):
            raise TypeError("Cut-cell preparation requires body and resource plans.")
        if subdivision_ <= 0 or not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                "Cut-cell subdivision and predicate tolerance must be positive."
            )
        self.hierarchy = hierarchy
        self.coordinate_map = coordinate_map
        self.coordinate_map_id = map_id
        self.bodies = bodies
        self.resources = resources
        self.subdivision = subdivision_
        self.predicate_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multivalued-cut-cell-plan",
                "hierarchy": hierarchy.hierarchy_id,
                "coordinate_map": map_id,
                "bodies": bodies.body_set_id,
                "resources": resources.resource_id,
                "subdivision": subdivision_,
                "predicate_tolerance": tolerance,
            }
        )

    def sign_topology(
        self,
        time: ArrayLike = 0.0,
        args: Any = None,
        /,
    ) -> CutCellSignTopology:
        """Fingerprint finite sample signs without constructing near-event polyhedra."""

        time_array = jnp.asarray(time)
        if time_array.shape != ():
            raise ValueError("Cut-cell sign-topology time must be scalar.")
        lower_bounds = np.asarray(
            [
                axis.bounds[0]
                for axis in self.hierarchy.topology.plan.grid.structured_axes
            ],
            dtype=float,
        )
        records = []
        minimum_margin = np.inf
        negative_count = 0
        for level_index, cell_coordinate, patch_id in _leaf_cells(self.hierarchy):
            level = self.hierarchy.levels[level_index]
            divisions = self.subdivision * _fine_scale(self.hierarchy, level_index)
            spacing = np.asarray(level.spacing, dtype=float)
            cell_lower = lower_bounds + spacing * np.asarray(cell_coordinate, dtype=float)
            indices = tuple(np.ndindex((divisions + 1,) * 3))
            references = np.stack(
                [
                    cell_lower + spacing * np.asarray(index, dtype=float) / divisions
                    for index in indices
                ]
            )
            mapped = (
                self.coordinate_map.map_for(patch_id)
                if isinstance(self.coordinate_map, PatchCoordinateMapSet)
                else self.coordinate_map
            )
            points = np.asarray(
                mapped(jnp.asarray(references), time_array, args), dtype=float
            )
            values, tags = self.bodies.evaluate(jnp.asarray(points), time_array, args)
            if values.shape != (len(indices),) or np.any(~np.isfinite(values)):
                raise ValueError("Cut-cell sign topology requires finite sample values.")
            signs = values < 0.0
            minimum_margin = min(
                minimum_margin,
                float(np.min(np.abs(values), initial=np.inf)),
            )
            negative_count += int(np.count_nonzero(signs))
            records.append(
                {
                    "level": level_index,
                    "cell": cell_coordinate,
                    "signs": signs.tolist(),
                    "body_tags": np.asarray(tags, dtype=np.int32).tolist(),
                }
            )
        return CutCellSignTopology(
            topology_id=canonical_fingerprint(
                {
                    "kind": "cut-cell-sample-sign-topology",
                    "plan": self.plan_id,
                    "records": records,
                }
            ),
            negative_sample_count=negative_count,
            minimum_margin=(0.0 if not np.isfinite(minimum_margin) else minimum_margin),
        )

    def prepare(
        self, time: ArrayLike = 0.0, args: Any = None, /
    ) -> MultivaluedCutCellComplex:
        time_array = jnp.asarray(time)
        if time_array.shape != ():
            raise ValueError("Cut-cell preparation time must be scalar.")
        hierarchy = self.hierarchy
        leaf_cells = _leaf_cells(hierarchy)
        lower_bounds = np.asarray(
            [axis.bounds[0] for axis in hierarchy.topology.plan.grid.structured_axes],
            dtype=float,
        )
        global_vertices: dict[tuple[Any, ...], int] = {}
        global_points: list[np.ndarray] = []
        component_faces: list[tuple[tuple[int, ...], ...]] = []
        component_levels: list[int] = []
        component_coordinates: list[tuple[int, ...]] = []
        component_slots: list[int] = []
        component_volumes: list[float] = []
        component_centers: list[np.ndarray] = []
        component_tetrahedra: list[list[np.ndarray]] = []
        component_fractions: list[float] = []
        component_shells: list[tuple[_Face, ...]] = []
        regular_cells = 0
        covered_cells = 0
        cut_cells = 0
        multivalued_cells = 0
        minimum_margin = np.inf
        volume_defects: list[float] = []

        def global_vertex(vertex: _Vertex) -> int:
            if vertex.key in global_vertices:
                index = global_vertices[vertex.key]
                distance = float(np.linalg.norm(global_points[index] - vertex.point))
                scale = max(1.0, float(np.linalg.norm(vertex.point)))
                if distance > self.predicate_tolerance * scale:
                    raise ValueError(
                        "Patch maps disagree at one canonical cut-complex vertex."
                    )
                return index
            index = len(global_points)
            global_vertices[vertex.key] = index
            global_points.append(vertex.point)
            return index

        for level_index, cell_coordinate, patch_id in leaf_cells:
            level = hierarchy.levels[level_index]
            fine_scale = _fine_scale(hierarchy, level_index)
            divisions = self.subdivision * fine_scale
            spacing = np.asarray(level.spacing, dtype=float)
            cell_lower = lower_bounds + spacing * np.asarray(cell_coordinate, dtype=float)
            cell_upper = cell_lower + spacing
            sample_shape = (divisions + 1,) * 3
            sample_indices = tuple(np.ndindex(sample_shape))
            references = np.stack(
                [
                    cell_lower + spacing * np.asarray(index, dtype=float) / divisions
                    for index in sample_indices
                ]
            )
            mapped = (
                self.coordinate_map.map_for(patch_id)
                if isinstance(self.coordinate_map, PatchCoordinateMapSet)
                else self.coordinate_map
            )
            points = np.asarray(
                mapped(jnp.asarray(references), time_array, args),
                dtype=float,
            )
            if points.shape != references.shape or np.any(~np.isfinite(points)):
                raise ValueError(
                    "Coordinate map must return one finite 3-D point per sample."
                )
            phi, tags = self.bodies.evaluate(jnp.asarray(points), time_array, args)
            scale = max(1.0, float(np.max(np.linalg.norm(points, axis=1))))
            tolerance = self.predicate_tolerance * scale
            margin = float(np.min(np.abs(phi), initial=np.inf))
            minimum_margin = min(minimum_margin, margin)
            if np.any(np.abs(phi) <= tolerance):
                raise ValueError(
                    "Cut topology is unresolved because a sampled predicate is within tolerance."
                )
            sample_lookup = {index: offset for offset, index in enumerate(sample_indices)}
            sample_keys = {
                index: (
                    "grid",
                    *tuple(
                        int(cell_coordinate[axis]) * divisions + int(index[axis])
                        for axis in range(3)
                    ),
                    len(hierarchy.levels) - 1,
                    self.subdivision,
                )
                for index in sample_indices
            }
            vertices = {
                index: _Vertex(
                    sample_keys[index],
                    references[offset],
                    points[offset],
                    float(phi[offset]),
                    int(tags[offset]),
                )
                for index, offset in sample_lookup.items()
            }
            fragments: list[_Fragment] = []
            full_volume = 0.0
            for subcube in np.ndindex((divisions,) * 3):
                for tetra_indices in _tetrahedra_in_subcube(subcube):
                    tetra = tuple(vertices[index] for index in tetra_indices)
                    matrix = np.stack(
                        (
                            tetra[1].point - tetra[0].point,
                            tetra[2].point - tetra[0].point,
                            tetra[3].point - tetra[0].point,
                        ),
                        axis=1,
                    )
                    determinant = float(np.linalg.det(matrix))
                    if not np.isfinite(determinant) or abs(determinant) <= tolerance**3:
                        raise ValueError(
                            "Mapped cut-cell tetrahedralization is degenerate."
                        )
                    if determinant < 0.0:
                        tetra = (tetra[0], tetra[2], tetra[1], tetra[3])
                        determinant = -determinant
                    full_volume += determinant / 6.0
                    fragment = _clip_tetrahedron(
                        tetra,
                        cell_lower,
                        cell_upper,
                        tolerance,
                    )
                    if fragment is not None:
                        fragments.append(fragment)
            fragment_tuple = tuple(fragments)
            groups = _cell_components(fragment_tuple)
            fluid_volume = sum(fragment.volume for fragment in fragment_tuple)
            defect = abs(fluid_volume + (full_volume - fluid_volume) - full_volume)
            volume_defects.append(defect)
            if not groups:
                covered_cells += 1
                continue
            cut = fluid_volume < full_volume - tolerance
            if cut:
                cut_cells += 1
            else:
                regular_cells += 1
            if len(groups) > self.resources.maximum_components_per_cell:
                raise ValueError("Cut cell exceeds maximum_components_per_cell.")
            if len(groups) > 1:
                multivalued_cells += 1
            for slot, group in enumerate(groups):
                group_set = set(group)
                face_incidents: dict[
                    tuple[tuple[Any, ...], ...], list[tuple[int, _Face]]
                ] = {}
                volume = 0.0
                moment = np.zeros((3,), dtype=float)
                for fragment_index in group:
                    fragment = fragment_tuple[fragment_index]
                    volume += fragment.volume
                    moment += fragment.volume * fragment.centroid
                    for face in fragment.faces:
                        face_incidents.setdefault(_fragment_face_key(face), []).append(
                            (fragment_index, face)
                        )
                shell: list[_Face] = []
                for incidents in face_incidents.values():
                    local = [value for value in incidents if value[0] in group_set]
                    if len(local) == 1:
                        shell.append(local[0][1])
                    elif len(local) != 2:
                        raise ValueError("Cut component shell is nonmanifold.")
                if any(face.kind == _FACE_INTERNAL for face in shell):
                    raise ValueError(
                        "Cut component has an unresolved internal simplex face."
                    )
                if len(shell) < 4 or volume <= tolerance:
                    raise ValueError("Cut component shell is incomplete or zero measure.")
                loops = tuple(
                    tuple(global_vertex(vertex) for vertex in face.vertices)
                    for face in shell
                )
                component_faces.append(loops)
                component_levels.append(level_index)
                component_coordinates.append(cell_coordinate)
                component_slots.append(slot)
                component_volumes.append(volume)
                component_centers.append(moment / volume)
                component_fractions.append(volume / full_volume)
                tetrahedra: list[np.ndarray] = []
                for fragment_index in group:
                    fragment = fragment_tuple[fragment_index]
                    for face in fragment.faces:
                        points = np.stack([vertex.point for vertex in face.vertices])
                        for face_index in range(1, points.shape[0] - 1):
                            tetra = np.stack(
                                (
                                    fragment.centroid,
                                    points[0],
                                    points[face_index],
                                    points[face_index + 1],
                                )
                            )
                            determinant = float(
                                np.linalg.det(
                                    np.stack(
                                        (
                                            tetra[1] - tetra[0],
                                            tetra[2] - tetra[0],
                                            tetra[3] - tetra[0],
                                        ),
                                        axis=1,
                                    )
                                )
                            )
                            if determinant < 0.0:
                                tetra[[2, 3]] = tetra[[3, 2]]
                                determinant = -determinant
                            if determinant > tolerance**3:
                                tetrahedra.append(tetra)
                if not tetrahedra:
                    raise ValueError("Cut component tetrahedralization is empty.")
                component_tetrahedra.append(tetrahedra)
                component_shells.append(tuple(shell))

        if not component_faces:
            raise ValueError("Cut-cell hierarchy contains no active fluid component.")
        cell_ids = np.arange(len(component_faces), dtype=np.int64)
        mesh = CellMesh.from_polyhedra(
            np.stack(global_points),
            component_faces,
            cell_global_ids=cell_ids,
            numeric_version="block-amr-cut-complex",
        )
        mesh_ids = np.asarray(mesh.connectivity.cell_global_ids, dtype=np.int64)
        component_order = mesh_ids.astype(np.int64, copy=False)
        inverse_order = np.empty_like(component_order)
        inverse_order[component_order] = np.arange(component_order.size)
        ordered_component_tetrahedra = tuple(
            tuple(component_tetrahedra[int(index)]) for index in component_order
        )

        face_incidents: dict[tuple[tuple[Any, ...], ...], list[tuple[int, _Face]]] = {}
        for component, shell in enumerate(component_shells):
            for face in shell:
                face_incidents.setdefault(_fragment_face_key(face), []).append(
                    (int(inverse_order[component]), face)
                )
        face_records: list[tuple[int, int, _Face]] = []
        for key in sorted(face_incidents, key=repr):
            incidents = face_incidents[key]
            if len(incidents) == 1:
                owner, face = incidents[0]
                if face.kind == _FACE_INTERNAL:
                    raise ValueError("Unpaired internal face remains in cut complex.")
                face_records.append((owner, -1, face))
            elif len(incidents) == 2:
                (first_owner, first_face), (second_owner, second_face) = incidents
                if first_owner == second_owner:
                    raise ValueError("One cut component repeats a global face.")
                if first_owner < second_owner:
                    face_records.append((first_owner, second_owner, first_face))
                else:
                    face_records.append((second_owner, first_owner, second_face))
            else:
                raise ValueError(
                    "Cut-complex face has more than two incident components."
                )
        face_offsets = np.asarray(mesh.connectivity.face_vertex_offsets, dtype=np.int32)
        face_vertices = np.asarray(mesh.connectivity.face_vertex_values, dtype=np.int32)
        mesh_face_lookup = {
            tuple(
                sorted(
                    int(value)
                    for value in face_vertices[
                        int(face_offsets[index]) : int(face_offsets[index + 1])
                    ]
                )
            ): index
            for index in range(mesh.connectivity.face_count)
        }
        face_mesh_records = []
        for _, _, face in face_records:
            key = tuple(sorted(global_vertex(vertex) for vertex in face.vertices))
            if key not in mesh_face_lookup:
                raise RuntimeError("Cut-complex face is absent from its canonical mesh.")
            face_mesh_records.append(mesh_face_lookup[key])

        active_component_count = len(component_faces)
        component_capacity = len(leaf_cells) * self.resources.maximum_components_per_cell
        face_capacity = len(leaf_cells) * (
            6 * self.resources.maximum_apertures_per_face
            + self.resources.maximum_embedded_faces_per_cell
        )
        if active_component_count > component_capacity:
            raise ValueError("Cut-complex component capacity is exceeded.")
        if len(face_records) > face_capacity:
            raise ValueError("Cut-complex face capacity is exceeded.")

        component_active = np.zeros((component_capacity,), dtype=bool)
        component_active[:active_component_count] = True
        component_levels_array = np.full((component_capacity,), -1, dtype=np.int32)
        component_coordinates_array = np.full((component_capacity, 3), -1, dtype=np.int32)
        component_slots_array = np.full((component_capacity,), -1, dtype=np.int32)
        component_volumes_array = np.zeros((component_capacity,), dtype=float)
        component_centers_array = np.zeros((component_capacity, 3), dtype=float)
        component_fractions_array = np.zeros((component_capacity,), dtype=float)
        component_levels_array[:active_component_count] = np.asarray(component_levels)[
            component_order
        ]
        component_coordinates_array[:active_component_count] = np.asarray(
            component_coordinates
        )[component_order]
        component_slots_array[:active_component_count] = np.asarray(component_slots)[
            component_order
        ]
        component_volumes_array[:active_component_count] = np.asarray(component_volumes)[
            component_order
        ]
        component_centers_array[:active_component_count] = np.asarray(component_centers)[
            component_order
        ]
        component_fractions_array[:active_component_count] = np.asarray(
            component_fractions
        )[component_order]

        face_active = np.zeros((face_capacity,), dtype=bool)
        face_owner = np.zeros((face_capacity,), dtype=np.int32)
        face_neighbour = np.full((face_capacity,), -1, dtype=np.int32)
        face_kind = np.zeros((face_capacity,), dtype=np.int32)
        face_tag = np.full((face_capacity,), -1, dtype=np.int32)
        face_axis = np.full((face_capacity,), -1, dtype=np.int32)
        face_side = np.full((face_capacity,), -1, dtype=np.int32)
        face_mesh_index = np.full((face_capacity,), -1, dtype=np.int32)
        face_centers = np.zeros((face_capacity, 3), dtype=float)
        face_area = np.zeros((face_capacity, 3), dtype=float)
        face_measure = np.zeros((face_capacity,), dtype=float)
        closure = np.zeros((active_component_count, 3), dtype=float)
        embedded_counts: dict[tuple[int, tuple[int, ...]], int] = {}
        aperture_counts: dict[tuple[int, tuple[int, ...], int, int], int] = {}
        for index, (owner, neighbour, face) in enumerate(face_records):
            points = np.stack([vertex.point for vertex in face.vertices])
            area = _polygon_area_vector(face.vertices)
            center = np.mean(points, axis=0)
            measure = float(np.linalg.norm(area))
            if not np.isfinite(measure) or measure <= self.predicate_tolerance:
                raise ValueError("Cut-complex face has unresolved measure.")
            if neighbour >= 0:
                direction = (
                    component_centers_array[neighbour] - component_centers_array[owner]
                )
                if float(np.dot(area, direction)) < 0.0:
                    area = -area
            else:
                direction = center - component_centers_array[owner]
                if float(np.dot(area, direction)) < 0.0:
                    area = -area
            face_mesh_index[index] = face_mesh_records[index]
            face_active[index] = True
            face_owner[index] = owner
            face_neighbour[index] = neighbour
            face_kind[index] = face.kind if neighbour < 0 else _FACE_INTERNAL
            face_tag[index] = face.body_tag
            face_axis[index] = face.axis
            face_side[index] = face.side
            face_centers[index] = center
            face_area[index] = area
            face_measure[index] = measure
            closure[owner] += area
            if neighbour >= 0:
                closure[neighbour] -= area
            owner_key = (
                int(component_levels_array[owner]),
                tuple(int(value) for value in component_coordinates_array[owner]),
            )
            if face.kind == _FACE_EMBEDDED:
                embedded_counts[owner_key] = embedded_counts.get(owner_key, 0) + 1
            elif face.kind == _FACE_PHYSICAL:
                aperture_key = owner_key + (face.axis, face.side)
                aperture_counts[aperture_key] = aperture_counts.get(aperture_key, 0) + 1
        if (
            embedded_counts
            and max(embedded_counts.values())
            > self.resources.maximum_embedded_faces_per_cell
        ):
            raise ValueError("Cut cell exceeds maximum_embedded_faces_per_cell.")
        if (
            aperture_counts
            and max(aperture_counts.values()) > self.resources.maximum_apertures_per_face
        ):
            raise ValueError("Cut face exceeds maximum_apertures_per_face.")
        closure_defect = np.linalg.norm(closure, axis=1)
        geometry_scale = max(1.0, float(np.max(face_measure[: len(face_records)])))
        closure_tolerance = 512.0 * np.finfo(float).eps * geometry_scale
        evidence = MultivaluedCutCellEvidence(
            leaf_cell_count=len(leaf_cells),
            regular_cell_count=regular_cells,
            covered_cell_count=covered_cells,
            cut_cell_count=cut_cells,
            multivalued_cell_count=multivalued_cells,
            component_count=active_component_count,
            face_count=len(face_records),
            maximum_components_in_cell=max(component_slots, default=-1) + 1,
            minimum_predicate_margin=(
                0.0 if not np.isfinite(minimum_margin) else minimum_margin
            ),
            maximum_volume_closure_defect=max(volume_defects, default=0.0),
            maximum_face_closure_defect=float(np.max(closure_defect, initial=0.0)),
            tolerance=closure_tolerance,
        )
        if not evidence.valid:
            raise ValueError("Cut-cell volume or face closure certification failed.")
        return MultivaluedCutCellComplex(
            hierarchy=hierarchy,
            mesh=mesh,
            component_active=component_active,
            component_levels=component_levels_array,
            component_cell_coordinates=component_coordinates_array,
            component_slots=component_slots_array,
            component_volumes=component_volumes_array,
            component_centers=component_centers_array,
            component_volume_fractions=component_fractions_array,
            face_active=face_active,
            face_owner_components=face_owner,
            face_neighbour_components=face_neighbour,
            face_kinds=face_kind,
            face_mesh_indices=face_mesh_index,
            face_body_tags=face_tag,
            face_axes=face_axis,
            face_sides=face_side,
            face_centers=face_centers,
            face_area_vectors=face_area,
            face_measures=face_measure,
            evidence=evidence,
            component_tetrahedra=ordered_component_tetrahedra,
            body_set_id=self.bodies.body_set_id,
        )


__all__ = [
    "CutCellSignTopology",
    "EmbeddedLevelSetBody",
    "EmbeddedLevelSetBodySet",
    "MultivaluedCutCellComplex",
    "MultivaluedCutCellEvidence",
    "MultivaluedCutCellPlan",
]
