#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact piecewise-linear 2-D multivalued cut components and FV execution."""

from __future__ import annotations

from collections.abc import Mapping
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
from .._conservation_boundary import AbstractConservationBoundary
from ._canonical import (
    BlockAMRResourcePlan,
    canonicalize_patch_hierarchy,
    CanonicalPatchHierarchy,
)
from ._core import BlockHierarchyTopology
from ._cut_complex import EmbeddedLevelSetBodySet
from ._mapped_geometry import PatchCoordinateMapSet
from ._variable import VariablePatchHierarchyTopology


if TYPE_CHECKING:
    from ..finite_volume._riemann import AbstractArbitraryNormalNumericalFluxPlan


_FACE_INTERNAL = 0
_FACE_PHYSICAL = 1
_FACE_EMBEDDED = 2


@dataclass(frozen=True)
class _Vertex2D:
    key: tuple[Any, ...]
    reference: np.ndarray
    point: np.ndarray
    phi: float
    body_tag: int


@dataclass(frozen=True)
class _Edge2D:
    start: _Vertex2D
    stop: _Vertex2D
    kind: int
    axis: int
    side: int
    body_tag: int


@dataclass(frozen=True)
class _Fragment2D:
    edges: tuple[_Edge2D, ...]
    area: float
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


class MultivaluedCutCell2DEvidence(StrictModule, NonTrainableState):
    """Component count and independent area/edge closure evidence."""

    leaf_cell_count: int = eqx.field(static=True)
    cut_cell_count: int = eqx.field(static=True)
    multivalued_cell_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    minimum_predicate_margin: float = eqx.field(static=True)
    maximum_face_closure_defect: float = eqx.field(static=True)
    valid: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class MultivaluedCutCell2DComplex(StrictModule, NonTrainableState):
    """Padded connected-component state and owner-oriented edge graph."""

    hierarchy: CanonicalPatchHierarchy
    component_active: Array
    component_levels: Array
    component_cell_coordinates: Array
    component_slots: Array
    component_areas: Array
    component_centers: Array
    component_area_fractions: Array
    component_triangles: tuple[tuple[tuple[tuple[float, float], ...], ...], ...] = (
        eqx.field(static=True)
    )
    face_active: Array
    face_owner_components: Array
    face_neighbour_components: Array
    face_kinds: Array
    face_body_tags: Array
    face_axes: Array
    face_sides: Array
    face_centers: Array
    face_area_vectors: Array
    face_measures: Array
    face_boundary_names: tuple[str | None, ...] = eqx.field(static=True)
    face_boundary_axes: tuple[int, ...] = eqx.field(static=True)
    evidence: MultivaluedCutCell2DEvidence
    component_capacity: int = eqx.field(static=True)
    face_capacity: int = eqx.field(static=True)
    active_component_count: int = eqx.field(static=True)
    active_face_count: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    @property
    def component_count(self) -> int:
        return self.active_component_count

    @property
    def face_count(self) -> int:
        return self.active_face_count

    def ssprk33_step(
        self,
        system: Any,
        numerical_flux: AbstractArbitraryNormalNumericalFluxPlan,
        boundaries: Mapping[str, AbstractConservationBoundary],
        state: ArrayLike,
        time: ArrayLike,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        """Advance component averages with one conservative SSPRK(3,3) step."""
        from ..finite_volume._riemann import AbstractArbitraryNormalNumericalFluxPlan

        if not isinstance(numerical_flux, AbstractArbitraryNormalNumericalFluxPlan):
            raise TypeError("2-D cut execution requires arbitrary-normal numerical flux.")
        value = jnp.asarray(state)
        time_ = jnp.asarray(time)
        step = jnp.asarray(step_size, dtype=value.dtype)
        expected = (self.component_capacity, system.component_count)
        if value.shape != expected or time_.shape != () or step.shape != ():
            raise ValueError("2-D cut state, time, or step shape is invalid.")
        value = eqx.error_if(
            value,
            ~jnp.isfinite(step) | (step <= 0.0),
            "2-D cut SSPRK step must be positive and finite.",
        )

        def rhs(stage_time, stage_state):
            count = self.component_count
            owner = self.face_owner_components
            neighbour = self.face_neighbour_components
            safe_neighbour = jnp.maximum(neighbour, 0)
            left = stage_state[owner]
            right = stage_state[safe_neighbour]
            normal = (
                self.face_area_vectors
                / jnp.where(
                    self.face_active,
                    self.face_measures,
                    1.0,
                )[:, None]
            )
            for face, name in enumerate(self.face_boundary_names):
                if name is None:
                    continue
                if name not in boundaries or not isinstance(
                    boundaries[name], AbstractConservationBoundary
                ):
                    raise ValueError(f"Missing 2-D cut boundary policy {name!r}.")
                exterior = boundaries[name].exterior_state(
                    system,
                    stage_time,
                    left[face],
                    self.face_centers[face],
                    normal[face],
                    self.face_boundary_axes[face],
                    args,
                )
                right = right.at[face].set(exterior)
            flux = numerical_flux.normal_face_flux(
                system,
                left,
                right,
                normal,
                args,
            ).normal_flux
            integrated = flux * self.face_measures[:, None]
            integrated = jnp.where(
                self.face_active[:, None], integrated, jnp.zeros_like(integrated)
            )
            content_rate = jnp.zeros(
                (self.component_capacity, system.component_count), dtype=value.dtype
            )
            content_rate = content_rate.at[owner].add(-integrated)
            internal = self.face_active & (neighbour >= 0)
            neighbour_contribution = jnp.where(
                internal[:, None], integrated, jnp.zeros_like(integrated)
            )
            content_rate = content_rate.at[safe_neighbour].add(neighbour_contribution)
            average_rate = content_rate[:count] / self.component_areas[:count, None]
            padded = jnp.zeros_like(stage_state)
            return padded.at[:count].set(average_rate)

        first = value + step * rhs(time_, value)
        second = 0.75 * value + 0.25 * (first + step * rhs(time_ + step, first))
        final = (1.0 / 3.0) * value + (2.0 / 3.0) * (
            second + step * rhs(time_ + 0.5 * step, second)
        )
        return jnp.where(self.component_active[:, None], final, jnp.zeros_like(final))


def _edge_intersection(left: _Vertex2D, right: _Vertex2D) -> _Vertex2D:
    fraction = left.phi / (left.phi - right.phi)
    if left.key <= right.key:
        first, second, canonical_fraction = left.key, right.key, fraction
    else:
        first, second, canonical_fraction = right.key, left.key, 1.0 - fraction
    tag = left.body_tag if abs(left.phi) <= abs(right.phi) else right.body_tag
    return _Vertex2D(
        ("edge", first, second, round(float(canonical_fraction), 15)),
        left.reference + fraction * (right.reference - left.reference),
        left.point + fraction * (right.point - left.point),
        0.0,
        tag,
    )


def _clip_triangle(vertices: tuple[_Vertex2D, ...]) -> tuple[_Vertex2D, ...]:
    output = []
    for left, right in zip(vertices, vertices[1:] + vertices[:1], strict=True):
        left_inside = left.phi > 0.0
        right_inside = right.phi > 0.0
        if left_inside:
            output.append(left)
        if left_inside != right_inside:
            output.append(_edge_intersection(left, right))
    return tuple(output)


def _polygon_moments(vertices: tuple[_Vertex2D, ...]) -> tuple[float, np.ndarray]:
    points = np.stack([vertex.point for vertex in vertices])
    cross = (
        points[:, 0] * np.roll(points[:, 1], -1)
        - np.roll(points[:, 0], -1) * points[:, 1]
    )
    signed_area = 0.5 * np.sum(cross)
    if signed_area < 0.0:
        vertices = tuple(reversed(vertices))
        points = np.stack([vertex.point for vertex in vertices])
        cross = (
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
        signed_area = 0.5 * np.sum(cross)
    centroid = np.asarray(
        (
            np.sum((points[:, 0] + np.roll(points[:, 0], -1)) * cross),
            np.sum((points[:, 1] + np.roll(points[:, 1], -1)) * cross),
        )
    ) / (6.0 * signed_area)
    return float(signed_area), centroid


def _edge_kind(
    start: _Vertex2D,
    stop: _Vertex2D,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float,
) -> tuple[int, int, int, int]:
    for axis in range(2):
        if (
            abs(start.reference[axis] - lower[axis]) <= tolerance
            and abs(stop.reference[axis] - lower[axis]) <= tolerance
        ):
            return _FACE_PHYSICAL, axis, 0, -1
        if (
            abs(start.reference[axis] - upper[axis]) <= tolerance
            and abs(stop.reference[axis] - upper[axis]) <= tolerance
        ):
            return _FACE_PHYSICAL, axis, 1, -1
    if start.phi == 0.0 and stop.phi == 0.0:
        tags = {start.body_tag, stop.body_tag}
        if len(tags) != 1:
            raise ValueError("2-D cut segment has unresolved multiple body tags.")
        return _FACE_EMBEDDED, -1, -1, next(iter(tags))
    return _FACE_INTERNAL, -1, -1, -1


def _leaf_cells(hierarchy: CanonicalPatchHierarchy):
    cells = []
    for level in hierarchy.levels:
        for bucket in level.buckets:
            active = np.asarray(bucket.leaf_active, dtype=bool)
            for lane, box in enumerate(bucket.boxes):
                if box is None:
                    continue
                for local in np.argwhere(active[lane]):
                    cells.append(
                        (
                            level.level,
                            tuple(
                                int(start) + int(offset)
                                for start, offset in zip(box.lower, local, strict=True)
                            ),
                            box.box_id,
                        )
                    )
    return tuple(sorted(cells))


def _fine_scale(hierarchy: CanonicalPatchHierarchy, level: int) -> int:
    return prod(
        hierarchy.levels[index].refinement_ratio
        for index in range(level, len(hierarchy.levels) - 1)
    )


class MultivaluedCutCell2DPlan(StrictModule, NonTrainableState):
    """Prepare disconnected and hole-bearing 2-D component face graphs."""

    hierarchy: CanonicalPatchHierarchy
    coordinate_map: Any = eqx.field(static=True)
    coordinate_map_id: str = eqx.field(static=True)
    bodies: EmbeddedLevelSetBodySet
    resources: BlockAMRResourcePlan
    subdivision: int = eqx.field(static=True)
    predicate_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: BlockHierarchyTopology | VariablePatchHierarchyTopology,
        coordinate_map: Any,
        coordinate_map_id: str,
        bodies: EmbeddedLevelSetBodySet,
        resources: BlockAMRResourcePlan,
        /,
        *,
        subdivision: int = 1,
        predicate_tolerance: float = 1.0e-12,
    ):
        hierarchy = canonicalize_patch_hierarchy(topology)
        if hierarchy.dimension != 2:
            raise ValueError("MultivaluedCutCell2DPlan requires a 2-D hierarchy.")
        if not callable(coordinate_map) and not isinstance(
            coordinate_map, PatchCoordinateMapSet
        ):
            raise TypeError("2-D cut geometry requires a coordinate map.")
        if not isinstance(bodies, EmbeddedLevelSetBodySet) or not isinstance(
            resources, BlockAMRResourcePlan
        ):
            raise TypeError("2-D cut geometry requires body and resource plans.")
        subdivision_ = int(subdivision)
        tolerance = float(predicate_tolerance)
        if subdivision_ <= 0 or not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("2-D cut subdivision/tolerance must be positive.")
        self.hierarchy = hierarchy
        self.coordinate_map = coordinate_map
        self.coordinate_map_id = str(coordinate_map_id)
        self.bodies = bodies
        self.resources = resources
        self.subdivision = subdivision_
        self.predicate_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multivalued-cut-cell-2d-plan",
                "hierarchy": hierarchy.hierarchy_id,
                "map": self.coordinate_map_id,
                "bodies": bodies.body_set_id,
                "resources": resources.resource_id,
                "subdivision": subdivision_,
                "tolerance": tolerance,
            }
        )

    def prepare(
        self, time: ArrayLike = 0.0, args: Any = None, /
    ) -> MultivaluedCutCell2DComplex:
        time_ = jnp.asarray(time)
        lower_bounds = np.asarray(
            [
                axis.bounds[0]
                for axis in self.hierarchy.topology.plan.grid.structured_axes
            ],
            dtype=float,
        )
        cells = _leaf_cells(self.hierarchy)
        global_vertices: dict[tuple[Any, ...], int] = {}
        points: list[np.ndarray] = []
        component_edges = []
        component_triangles = []
        levels = []
        coordinates = []
        slots = []
        areas = []
        centers = []
        fractions = []
        cut_count = 0
        multivalued_count = 0
        minimum_margin = np.inf

        def vertex_id(vertex: _Vertex2D) -> int:
            if vertex.key not in global_vertices:
                global_vertices[vertex.key] = len(points)
                points.append(vertex.point)
            return global_vertices[vertex.key]

        for level_index, cell_coordinate, patch_id in cells:
            level = self.hierarchy.levels[level_index]
            divisions = self.subdivision * _fine_scale(self.hierarchy, level_index)
            spacing = np.asarray(level.spacing, dtype=float)
            lower = lower_bounds + spacing * np.asarray(cell_coordinate)
            upper = lower + spacing
            indices = tuple(np.ndindex((divisions + 1, divisions + 1)))
            references = np.stack(
                [lower + spacing * np.asarray(index) / divisions for index in indices]
            )
            mapped = (
                self.coordinate_map.map_for(patch_id)
                if isinstance(self.coordinate_map, PatchCoordinateMapSet)
                else self.coordinate_map
            )
            physical = np.asarray(
                mapped(jnp.asarray(references), time_, args), dtype=float
            )
            phi, tags = self.bodies.evaluate(jnp.asarray(physical), time_, args)
            scale = max(1.0, float(np.max(np.linalg.norm(physical, axis=1))))
            tolerance = self.predicate_tolerance * scale
            minimum_margin = min(minimum_margin, float(np.min(np.abs(phi))))
            if np.any(np.abs(phi) <= tolerance):
                raise ValueError("2-D cut predicate lies within its declared tolerance.")
            lookup = {index: position for position, index in enumerate(indices)}
            sample = {
                index: _Vertex2D(
                    (
                        "grid",
                        int(cell_coordinate[0]) * divisions + index[0],
                        int(cell_coordinate[1]) * divisions + index[1],
                        len(self.hierarchy.levels) - 1,
                        self.subdivision,
                    ),
                    references[position],
                    physical[position],
                    float(phi[position]),
                    int(tags[position]),
                )
                for index, position in lookup.items()
            }
            fragments = []
            full_area = 0.0
            for subcell in np.ndindex((divisions, divisions)):
                for ordering in permutations(range(2)):
                    point = np.asarray(subcell)
                    triangle_indices = [tuple(int(value) for value in point)]
                    for axis in ordering:
                        point = point.copy()
                        point[axis] += 1
                        triangle_indices.append(tuple(int(value) for value in point))
                    triangle = tuple(sample[index] for index in triangle_indices)
                    first_edge = triangle[1].point - triangle[0].point
                    second_edge = triangle[2].point - triangle[0].point
                    full_triangle_area = (
                        abs(
                            first_edge[0] * second_edge[1]
                            - first_edge[1] * second_edge[0]
                        )
                        / 2.0
                    )
                    full_area += float(full_triangle_area)
                    polygon = _clip_triangle(triangle)
                    if len(polygon) < 3:
                        continue
                    area, centroid = _polygon_moments(polygon)
                    if area <= tolerance**2:
                        continue
                    edges = []
                    for start, stop in zip(
                        polygon, polygon[1:] + polygon[:1], strict=True
                    ):
                        kind, axis, side, tag = _edge_kind(
                            start, stop, lower, upper, tolerance
                        )
                        edges.append(_Edge2D(start, stop, kind, axis, side, tag))
                    fragments.append(_Fragment2D(tuple(edges), area, centroid))
            union = _UnionFind(len(fragments))
            incidents: dict[tuple[Any, ...], list[int]] = {}
            for fragment_index, fragment in enumerate(fragments):
                for edge in fragment.edges:
                    if edge.kind == _FACE_INTERNAL:
                        key = tuple(sorted((edge.start.key, edge.stop.key), key=repr))
                        incidents.setdefault(key, []).append(fragment_index)
            for values in incidents.values():
                if len(values) == 2:
                    union.join(values[0], values[1])
                elif len(values) != 1:
                    raise ValueError("2-D cut simplex edge is nonmanifold.")
            groups: dict[int, list[int]] = {}
            for index in range(len(fragments)):
                groups.setdefault(union.root(index), []).append(index)
            if len(groups) > self.resources.maximum_components_per_cell:
                raise ValueError("2-D cut cell exceeds component capacity.")
            fluid_area = sum(fragment.area for fragment in fragments)
            if fluid_area < full_area - tolerance**2 and fluid_area > tolerance**2:
                cut_count += 1
            if len(groups) > 1:
                multivalued_count += 1
            for slot, group in enumerate(tuple(groups[key] for key in sorted(groups))):
                edge_incidents: dict[tuple[Any, ...], list[_Edge2D]] = {}
                area = 0.0
                moment = np.zeros((2,), dtype=float)
                triangles = []
                for fragment_index in group:
                    fragment = fragments[fragment_index]
                    area += fragment.area
                    moment += fragment.area * fragment.centroid
                    polygon_vertices = tuple(
                        vertex_id(edge.start) for edge in fragment.edges
                    )
                    anchor = polygon_vertices[0]
                    for index in range(1, len(polygon_vertices) - 1):
                        triangles.append(
                            (
                                tuple(float(value) for value in points[anchor]),
                                tuple(
                                    float(value)
                                    for value in points[polygon_vertices[index]]
                                ),
                                tuple(
                                    float(value)
                                    for value in points[polygon_vertices[index + 1]]
                                ),
                            )
                        )
                    for edge in fragment.edges:
                        key = tuple(sorted((edge.start.key, edge.stop.key), key=repr))
                        edge_incidents.setdefault(key, []).append(edge)
                shell = [
                    values[0] for values in edge_incidents.values() if len(values) == 1
                ]
                if any(edge.kind == _FACE_INTERNAL for edge in shell):
                    raise ValueError("2-D cut component has unresolved internal edge.")
                component_edges.append(tuple(shell))
                component_triangles.append(tuple(triangles))
                levels.append(level_index)
                coordinates.append(cell_coordinate)
                slots.append(slot)
                areas.append(area)
                centers.append(moment / area)
                fractions.append(area / full_area)

        active_count = len(component_edges)
        capacity = len(cells) * self.resources.maximum_components_per_cell
        face_capacity = len(cells) * (
            4 * self.resources.maximum_apertures_per_face
            + self.resources.maximum_embedded_faces_per_cell
        )
        if active_count == 0 or active_count > capacity:
            raise ValueError("2-D cut component capacity is empty or exceeded.")
        face_incidents: dict[tuple[Any, ...], list[tuple[int, _Edge2D]]] = {}
        for component, edges in enumerate(component_edges):
            for edge in edges:
                key = tuple(sorted((edge.start.key, edge.stop.key), key=repr))
                face_incidents.setdefault(key, []).append((component, edge))
        records = []
        for key in sorted(face_incidents, key=repr):
            values = face_incidents[key]
            if len(values) == 1:
                records.append((values[0][0], -1, values[0][1]))
            elif len(values) == 2:
                first, second = values
                records.append(
                    (first[0], second[0], first[1])
                    if first[0] < second[0]
                    else (second[0], first[0], second[1])
                )
            else:
                raise ValueError("2-D cut face has more than two incidents.")
        if len(records) > face_capacity:
            raise ValueError("2-D cut face capacity is exceeded.")
        component_active = np.zeros((capacity,), dtype=bool)
        component_active[:active_count] = True
        level_array = np.full((capacity,), -1, dtype=np.int32)
        coordinate_array = np.full((capacity, 2), -1, dtype=np.int32)
        slot_array = np.full((capacity,), -1, dtype=np.int32)
        area_array = np.zeros((capacity,), dtype=float)
        center_array = np.zeros((capacity, 2), dtype=float)
        fraction_array = np.zeros((capacity,), dtype=float)
        level_array[:active_count] = levels
        coordinate_array[:active_count] = coordinates
        slot_array[:active_count] = slots
        area_array[:active_count] = areas
        center_array[:active_count] = centers
        fraction_array[:active_count] = fractions
        face_active = np.zeros((face_capacity,), dtype=bool)
        owner = np.zeros((face_capacity,), dtype=np.int32)
        neighbour = np.full((face_capacity,), -1, dtype=np.int32)
        kinds = np.zeros((face_capacity,), dtype=np.int32)
        tags = np.full((face_capacity,), -1, dtype=np.int32)
        axes = np.full((face_capacity,), -1, dtype=np.int32)
        sides = np.full((face_capacity,), -1, dtype=np.int32)
        face_centers = np.zeros((face_capacity, 2), dtype=float)
        area_vectors = np.zeros((face_capacity, 2), dtype=float)
        measures = np.zeros((face_capacity,), dtype=float)
        closure = np.zeros((active_count, 2), dtype=float)
        for face, (left, right, edge) in enumerate(records):
            vector = edge.stop.point - edge.start.point
            area_vector = np.asarray((vector[1], -vector[0]))
            center = 0.5 * (edge.start.point + edge.stop.point)
            if right >= 0:
                if np.dot(area_vector, center_array[right] - center_array[left]) < 0.0:
                    area_vector = -area_vector
            elif np.dot(area_vector, center - center_array[left]) < 0.0:
                area_vector = -area_vector
            face_active[face] = True
            owner[face] = left
            neighbour[face] = right
            kinds[face] = _FACE_INTERNAL if right >= 0 else edge.kind
            tags[face] = edge.body_tag
            axes[face] = edge.axis
            sides[face] = edge.side
            face_centers[face] = center
            area_vectors[face] = area_vector
            measures[face] = np.linalg.norm(area_vector)
            closure[left] += area_vector
            if right >= 0:
                closure[right] -= area_vector
        closure_defect = float(np.max(np.linalg.norm(closure, axis=1), initial=0.0))
        tolerance = 512.0 * np.finfo(float).eps * max(1.0, float(np.max(measures)))
        valid = bool(closure_defect <= tolerance)
        boundary_names = tuple(
            (
                None
                if right >= 0
                else f"embedded-{edge.body_tag}"
                if edge.kind == _FACE_EMBEDDED
                else f"physical-{edge.axis}-{edge.side}"
            )
            for _, right, edge in records
        )
        boundary_axes = tuple(max(0, edge.axis) for _, _, edge in records)
        evidence = MultivaluedCutCell2DEvidence(
            leaf_cell_count=len(cells),
            cut_cell_count=cut_count,
            multivalued_cell_count=multivalued_count,
            component_count=active_count,
            face_count=len(records),
            minimum_predicate_margin=float(minimum_margin),
            maximum_face_closure_defect=closure_defect,
            valid=valid,
            evidence_id=canonical_fingerprint(
                {
                    "kind": "multivalued-cut-cell-2d-evidence",
                    "components": active_count,
                    "faces": len(records),
                    "closure": closure_defect,
                }
            ),
        )
        if not valid:
            raise ValueError("2-D cut component face closure failed.")
        return MultivaluedCutCell2DComplex(
            hierarchy=self.hierarchy,
            component_active=jnp.asarray(component_active),
            component_levels=jnp.asarray(level_array),
            component_cell_coordinates=jnp.asarray(coordinate_array),
            component_slots=jnp.asarray(slot_array),
            component_areas=jnp.asarray(area_array),
            component_centers=jnp.asarray(center_array),
            component_area_fractions=jnp.asarray(fraction_array),
            component_triangles=tuple(component_triangles),
            face_active=jnp.asarray(face_active),
            face_owner_components=jnp.asarray(owner),
            face_neighbour_components=jnp.asarray(neighbour),
            face_kinds=jnp.asarray(kinds),
            face_body_tags=jnp.asarray(tags),
            face_axes=jnp.asarray(axes),
            face_sides=jnp.asarray(sides),
            face_centers=jnp.asarray(face_centers),
            face_area_vectors=jnp.asarray(area_vectors),
            face_measures=jnp.asarray(measures),
            face_boundary_names=boundary_names,
            face_boundary_axes=boundary_axes,
            evidence=evidence,
            component_capacity=capacity,
            face_capacity=face_capacity,
            active_component_count=active_count,
            active_face_count=len(records),
            topology_id=canonical_fingerprint(
                {
                    "kind": "multivalued-cut-cell-2d-topology",
                    "hierarchy": self.hierarchy.topology_id,
                    "components": [levels, coordinates, slots],
                    "routes": array_tree_fingerprint(
                        np.stack(
                            (owner[: len(records)], neighbour[: len(records)]), axis=-1
                        )
                    ),
                }
            ),
            geometry_id=canonical_fingerprint(
                {
                    "kind": "multivalued-cut-cell-2d-geometry",
                    "plan": self.plan_id,
                    "areas": array_tree_fingerprint(area_array[:active_count]),
                    "faces": array_tree_fingerprint(area_vectors[: len(records)]),
                }
            ),
        )


__all__ = [
    "MultivaluedCutCell2DComplex",
    "MultivaluedCutCell2DEvidence",
    "MultivaluedCutCell2DPlan",
]
