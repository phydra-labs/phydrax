#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact transverse partitions of accepted triangular layer columns."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import canonical_fingerprint
from .._live_proofs import live_content_fingerprint, LiveProofLedger
from ..discretization import CellMesh
from ..discretization._exact_algebraic_geometry import rational_coefficient
from ..discretization._exact_plc_geometry import ExactPlcCellGeometrySource
from ..discretization._periodic_exact import exact_periodic_element
from ..geometry._rational import fraction_down, fraction_up


if TYPE_CHECKING:
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..discretization._nested_reference_types import ReferenceCellPartition
    from ._contracts import VolumeMeshingSpec
    from ._layer_core_topology import LayerCoreConstruction
    from ._result import CellMeshingResult
    from .providers._native_sources import NativeLayerCoreSource


_StencilKey = tuple[tuple[int, Fraction], ...]


@dataclass(frozen=True, slots=True)
class OverlapSourceStencils:
    """Exact boundary-point stencils bound to one original source and target."""

    source_binding_id: str
    target_mesh_id: str
    # A term is (root bank, source vertex ID, exact nonnegative weight).
    rows: tuple[tuple[int, tuple[tuple[str, int, Fraction], ...]], ...]
    witness_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.source_binding_id or not self.target_mesh_id:
            raise ValueError("Overlap stencils require source and target identities.")
        if not isinstance(self.rows, tuple) or any(
            not isinstance(row, tuple)
            or len(row) != 2
            or not isinstance(row[1], tuple)
            or any(not isinstance(term, tuple) or len(term) != 3 for term in row[1])
            for row in self.rows
        ):
            raise TypeError("Overlap stencil rows and terms must be immutable tuples.")
        identifiers = tuple(identifier for identifier, _ in self.rows)
        if identifiers != tuple(sorted(set(identifiers))) or any(
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier < 0
            for identifier in identifiers
        ):
            raise ValueError(
                "Overlap stencil target identities must be sorted and unique."
            )
        payload = []
        for identifier, terms in self.rows:
            keys = tuple((kind, root) for kind, root, _ in terms)
            if (
                not terms
                or keys != tuple(sorted(set(keys)))
                or any(
                    kind not in ("layer", "plc")
                    or isinstance(root, bool)
                    or not isinstance(root, int)
                    or root < 0
                    or not isinstance(weight, Fraction)
                    or weight <= 0
                    for kind, root, weight in terms
                )
                or sum((weight for _, _, weight in terms), Fraction(0)) != 1
            ):
                raise ValueError(
                    "Overlap stencils need canonical convex source-root weights."
                )
            payload.append(
                (
                    identifier,
                    tuple(
                        (kind, root, weight.numerator, weight.denominator)
                        for kind, root, weight in terms
                    ),
                )
            )
        object.__setattr__(
            self,
            "witness_id",
            canonical_fingerprint(
                {
                    "kind": "layer-core-overlap-source-stencils",
                    "source_binding": self.source_binding_id,
                    "target_mesh": self.target_mesh_id,
                    "rows": tuple(payload),
                }
            ),
        )


def _require_overlap_provenance(
    source: object,
    mesh: CellMesh,
    carrier: object,
    stencils: object,
    required: set[int],
    /,
) -> None:
    """Recompute exact target points from original roots and deck actions."""
    from .providers._native_sources import NativeLayerCoreSource

    if not isinstance(source, NativeLayerCoreSource):
        raise TypeError("Overlap provenance requires the original layer/core source.")
    source.validate_restored()
    if (
        not isinstance(mesh, CellMesh)
        or not isinstance(carrier, ExactPlcCellGeometrySource)
        or not isinstance(stencils, OverlapSourceStencils)
        or source.mapped_domain is not None
        or source.complex.geometry is not None
        or stencils.source_binding_id != source.binding_id
        or stencils.target_mesh_id != mesh.mesh_id
        or (carrier.source_id, carrier.source_revision)
        != (source.source_id, source.source_revision)
    ):
        raise ValueError(
            "Overlap source stencils are not bound to the original source and target."
        )
    rebuilt = OverlapSourceStencils(
        stencils.source_binding_id, stencils.target_mesh_id, stencils.rows
    )
    if rebuilt.witness_id != stencils.witness_id:
        raise ValueError("Overlap stencil witness identity is stale.")
    target_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if not np.array_equal(np.asarray(carrier.vertex_global_ids), target_ids):
        raise ValueError("Overlap exact coordinates do not name the target vertex bank.")
    if not np.array_equal(
        np.asarray(carrier.rounded_vertices), np.asarray(mesh.coordinates)
    ):
        raise ValueError("Overlap exact coordinates do not round to the target mesh.")
    roots: dict[tuple[str, int], tuple[Fraction, ...]] = {}
    layer_coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.asarray(
        source.layers.mesh.coordinates, dtype=np.float64
    )
    for identifier, point in zip(
        np.asarray(source.layers.mesh.vertex_global_ids, dtype=np.int64),
        layer_coordinates,
        strict=True,
    ):
        roots[("layer", int(identifier))] = tuple(
            Fraction(float(value)) for value in point
        )
    plc_coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.asarray(
        source.complex.vertices, dtype=np.float64
    )
    for identifier, point in zip(
        source.complex.vertex_entity_ids,
        plc_coordinates,
        strict=True,
    ):
        roots[("plc", int(identifier))] = tuple(Fraction(float(value)) for value in point)
    positions = {int(identifier): row for row, identifier in enumerate(target_ids)}
    checked: set[int] = set()
    for identifier, terms in stencils.rows:
        if identifier in checked or identifier not in positions or not terms:
            raise ValueError("Overlap stencil rows must identify unique target vertices.")
        checked.add(identifier)
        keys = tuple((kind, root) for kind, root, _ in terms)
        if (
            keys != tuple(sorted(set(keys)))
            or any(key not in roots for key in keys)
            or any(
                not isinstance(weight, Fraction) or weight < 0 for _, _, weight in terms
            )
            or sum((weight for _, _, weight in terms), Fraction(0)) != 1
        ):
            raise ValueError(
                "Overlap stencil coefficients or original source roots are invalid."
            )
        exact = tuple(
            sum(
                (weight * roots[(kind, root)][axis] for kind, root, weight in terms),
                Fraction(0),
            )
            for axis in range(3)
        )
        if exact != tuple(carrier.exact_coordinates[positions[identifier]]):
            raise ValueError("Overlap stencil does not reproduce its exact target point.")
    boundary = np.asarray(mesh.connectivity.boundary_vertices, dtype=np.bool_)
    required = required | {int(value) for value in target_ids[boundary]}
    if not required.issubset(checked):
        raise ValueError("Overlap source stencils omit a target boundary vertex.")
    periodic = mesh.periodic_topology
    if periodic is not None:
        representatives = np.asarray(periodic.vertex_representatives, dtype=np.int64)
        shifts = np.asarray(periodic.vertex_shifts, dtype=np.int64)
        for identifier in required:
            row = positions[identifier]
            representative = int(representatives[row])
            if int(target_ids[representative]) not in checked:
                raise ValueError(
                    "Overlap periodic representative lacks source provenance."
                )
            matrix = exact_periodic_element(periodic.cell, shifts[row])
            original = carrier.exact_coordinates[representative]
            image = tuple(
                sum(
                    (matrix[axis][column] * original[column] for column in range(3)),
                    matrix[axis][3],
                )
                for axis in range(3)
            )
            if image != tuple(carrier.exact_coordinates[row]):
                raise ValueError(
                    "Overlap exact coordinates violate the authored periodic action."
                )


def require_overlap_source_provenance(
    source: NativeLayerCoreSource, construction: LayerCoreConstruction, /
) -> None:
    """Reprove source-bound boundary points on an in-flight construction."""
    mesh = construction.mesh
    target_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    core_faces = np.asarray(construction.core.boundary_faces, dtype=np.int64)
    core_map = np.asarray(construction.core_vertex_map, dtype=np.int64)
    required = {int(value) for value in target_ids[core_map[core_faces].reshape(-1)]}
    if construction.layer_core_column_certification is not None:
        required.update(
            vertex
            for column in construction.layer_core_column_certification.columns
            for vertex in column.target_vertices
        )
    _require_overlap_provenance(
        source,
        mesh,
        construction.overlap_exact_geometry,
        construction.overlap_source_stencils,
        required,
    )


@dataclass(frozen=True, slots=True)
class TransverseLayerPartition:
    """A target prism partition and its exact accepted-layer ancestry.

    Stencils use original layer vertex global IDs, not coordinates or generated
    source identities. They are in target coordinate-row order. Each child
    records both original prism IDs whose union it partitions; a child may cross
    their internal diagonal. No layer-normal interval is subdivided.
    """

    coordinates: np.ndarray
    vertex_ids: np.ndarray
    prism_vertices: np.ndarray
    prism_ids: np.ndarray
    source_pair_cell_ids: np.ndarray
    stencils: tuple[tuple[tuple[int, Fraction], ...], ...]
    exact_coordinates: tuple[tuple[Fraction, Fraction, Fraction], ...]


def subdivide_parallel_prism_pairs(
    mesh: CellMesh,
    layer_indices: np.ndarray,
    layer_regions: np.ndarray,
    divisions: int,
    /,
) -> TransverseLayerPartition:
    """Split coplanar prism pairs only along proved equal-displacement columns.

    Each pair must form a parallelogram at both interval endpoints. The two
    endpoints of each split edge must have exactly the same interval displacement.
    Thus every generated column has the original physical displacement, not an
    average of unequal column vectors. Source-prism overlap must be certified
    separately because the new transverse triangles may cross their old diagonal.
    """
    if (
        not isinstance(mesh, CellMesh)
        or mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
    ):
        raise TypeError("An accepted three-dimensional CellMesh is required.")
    if (
        isinstance(divisions, bool)
        or not isinstance(divisions, Integral)
        or divisions < 1
    ):
        raise ValueError("Transverse divisions must be a positive integer.")
    divisions = int(divisions)
    if any(block.cell_kind != "prism" for block in mesh.blocks):
        raise ValueError("Transverse layer subdivision requires prism-only layers.")
    source_points: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.asarray(
        mesh.coordinates, dtype=np.float64
    )
    source_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if len(set(int(value) for value in source_ids)) != source_ids.size:
        raise ValueError("Accepted layer vertex IDs must be unique.")
    exact_points = {
        int(identifier): tuple(Fraction(float(value)) for value in point)
        for identifier, point in zip(source_ids, source_points, strict=True)
    }
    source_rows = {int(identifier): row for row, identifier in enumerate(source_ids)}
    key_type = tuple[tuple[int, Fraction], ...]

    def point(row: int) -> tuple[Fraction, ...]:
        return exact_points[int(source_ids[row])]

    def edge_key(first: int, last: int, index: int) -> key_type:
        weights = ((first, divisions - index), (last, index))
        return tuple(
            sorted(
                (int(source_ids[row]), Fraction(weight, divisions))
                for row, weight in weights
                if weight
            )
        )

    cells: list[tuple[int, np.ndarray]] = []
    for block in mesh.blocks:
        vertices: np.ndarray[tuple[int, int], np.dtype[np.int64]] = np.asarray(
            block.vertices, dtype=np.int64
        )
        cells.extend(
            (int(identifier), row)
            for identifier, row in zip(block.global_ids, vertices, strict=True)
        )
    cells.sort(key=lambda item: item[0])
    if len({identifier for identifier, _ in cells}) != len(cells):
        raise ValueError("Accepted layer cell IDs must be unique.")
    intervals = np.asarray(layer_indices)
    regions = np.asarray(layer_regions)
    if (
        intervals.shape != (len(cells),)
        or regions.shape != (len(cells),)
        or intervals.dtype.kind not in "iu"
        or regions.dtype.kind not in "iu"
    ):
        raise ValueError("Layer intervals and regions must align with sorted cell IDs.")
    by_edge: dict[tuple[int, int, int], list[int]] = {}
    for row, (_, vertices) in enumerate(cells):
        if vertices.shape != (6,):
            raise ValueError("A prism must have six reference-ordered corners.")
        for index in range(3):
            a, b = int(vertices[index]), int(vertices[(index + 1) % 3])
            by_edge.setdefault((int(intervals[row]), min(a, b), max(a, b)), []).append(
                row
            )
    new_keys: set[key_type] = set()
    children: list[tuple[tuple[int, int], tuple[key_type, ...]]] = []
    interfaces: dict[tuple[int, ...], list[frozenset[tuple[key_type, ...]]]] = {}
    paired: set[int] = set()
    for first, (parent, vertices) in enumerate(cells):
        if first in paired:
            continue
        choices = []
        for index in range(3):
            a, b = int(vertices[index]), int(vertices[(index + 1) % 3])
            for second in by_edge[(int(intervals[first]), min(a, b), max(a, b))]:
                if (
                    second == first
                    or second in paired
                    or regions[first] != regions[second]
                ):
                    continue
                other = cells[second][1]
                top = {
                    int(bottom): int(upper)
                    for cell in (vertices, other)
                    for bottom, upper in zip(cell[:3], cell[3:], strict=True)
                }
                if any(
                    top[int(bottom)] != int(upper)
                    for cell in (vertices, other)
                    for bottom, upper in zip(cell[:3], cell[3:], strict=True)
                ):
                    continue
                directed = [
                    (int(cell[i]), int(cell[(i + 1) % 3]))
                    for cell in (vertices, other)
                    for i in range(3)
                ]
                boundary = [(u, v) for u, v in directed if (v, u) not in directed]
                if len(boundary) != 4:
                    continue
                successor = dict(boundary)
                start = min(successor, key=lambda row: int(source_ids[row]))
                cycle = (
                    start,
                    successor[start],
                    successor[successor[start]],
                    successor[successor[successor[start]]],
                )
                if len(set(cycle)) != 4 or successor.get(cycle[3]) != start:
                    continue
                for rotation in (0, 1):
                    q = cycle[rotation:] + cycle[:rotation]
                    t = tuple(top[row] for row in q)
                    if any(
                        point(q[0])[axis] + point(q[2])[axis]
                        != point(q[1])[axis] + point(q[3])[axis]
                        or point(t[0])[axis] + point(t[2])[axis]
                        != point(t[1])[axis] + point(t[3])[axis]
                        for axis in range(3)
                    ):
                        continue
                    if any(
                        point(t[u])[axis] - point(q[u])[axis]
                        != point(t[v])[axis] - point(q[v])[axis]
                        for u, v in ((0, 1), (3, 2))
                        for axis in range(3)
                    ):
                        continue
                    length = sum(
                        (point(q[1])[axis] - point(q[0])[axis]) ** 2 for axis in range(3)
                    )
                    choices.append((length, -cells[second][0], q, t, second))
        if not choices:
            raise ValueError("No exact equal-displacement prism-pair subdivision exists.")
        _, _, q, t, second = max(choices)
        paired.update((first, second))
        parents = (min(parent, cells[second][0]), max(parent, cells[second][0]))
        child_start = len(children)
        for i in range(divisions):
            lower = (
                edge_key(q[0], q[1], i),
                edge_key(q[0], q[1], i + 1),
                edge_key(q[3], q[2], i + 1),
                edge_key(q[3], q[2], i),
            )
            upper = (
                edge_key(t[0], t[1], i),
                edge_key(t[0], t[1], i + 1),
                edge_key(t[3], t[2], i + 1),
                edge_key(t[3], t[2], i),
            )
            for triangle in ((0, 1, 2), (0, 2, 3)):
                child = tuple(lower[k] for k in triangle) + tuple(
                    upper[k] for k in triangle
                )
                children.append((parents, child))
                new_keys.update(
                    item for item in child if len(item) != 1 or item[0][1] != 1
                )
        faces = ((0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5))
        source_faces = [q, t]
        source_faces.extend(
            (q[i], q[(i + 1) % 4], t[(i + 1) % 4], t[i]) for i in range(4)
        )
        for source_face in source_faces:
            source_keys = tuple(sorted(int(source_ids[row]) for row in source_face))
            support = set(source_keys)
            pieces = frozenset(
                tuple(sorted(child[index] for index in face))
                for _, child in children[child_start:]
                for face in faces
                if all(root in support for index in face for root, _ in child[index])
            )
            if not pieces:
                raise ValueError("Layer subdivision lost an original pair boundary.")
            interfaces.setdefault(source_keys, []).append(pieces)
    if any(
        len(parts) > 2 or (len(parts) == 2 and parts[0] != parts[1])
        for parts in interfaces.values()
    ):
        raise ValueError(
            "Directed layer partitions do not conform across original pair boundaries."
        )
    ordered_keys = tuple(sorted(new_keys))
    max_vertex = max((int(value) for value in source_ids), default=-1)
    max_cell = max((identifier for identifier, _ in cells), default=-1)
    if (
        max_vertex + len(ordered_keys) > np.iinfo(np.int64).max
        or max_cell + len(children) > np.iinfo(np.int64).max
    ):
        raise OverflowError("Transverse layer subdivision exhausts entity IDs.")
    new_vertex_ids = np.asarray(
        [max_vertex + 1 + row for row in range(len(ordered_keys))], dtype=np.int64
    )
    vertex_ids = np.concatenate((source_ids, new_vertex_ids))
    stencils = (
        tuple(((int(identifier), Fraction(1)),) for identifier in source_ids)
        + ordered_keys
    )
    target_rows: dict[_StencilKey, int] = {
        ((identifier, Fraction(1)),): row for identifier, row in source_rows.items()
    }
    target_rows.update(
        (item, source_ids.size + row) for row, item in enumerate(ordered_keys)
    )
    exact_coordinates = tuple(
        (values[0], values[1], values[2])
        for stencil in stencils
        for values in (
            tuple(
                sum(
                    (
                        weight * exact_points[identifier][axis]
                        for identifier, weight in stencil
                    ),
                    Fraction(0),
                )
                for axis in range(3)
            ),
        )
    )
    coordinates = np.asarray(
        [[float(value) for value in point] for point in exact_coordinates],
        dtype=np.float64,
    )
    # Copy original shadows, including signed zero, rather than reconstructing them.
    coordinates[: source_ids.size] = source_points
    prism_vertices = np.asarray(
        [[target_rows[item] for item in child] for _, child in children], dtype=np.int64
    ).reshape(-1, 6)
    prism_ids = np.asarray(
        [max_cell + 1 + row for row in range(len(children))], dtype=np.int64
    )
    parents = np.asarray([parent for parent, _ in children], dtype=np.int64).reshape(
        -1, 2
    )
    return TransverseLayerPartition(
        coordinates,
        vertex_ids,
        prism_vertices,
        prism_ids,
        parents,
        stencils,
        exact_coordinates,
    )


@dataclass(frozen=True, slots=True)
class NestedLayerPartition:
    """A transverse prism restriction with exact original-cell ancestry."""

    coordinates: np.ndarray
    vertex_ids: np.ndarray
    prism_vertices: np.ndarray
    prism_ids: np.ndarray
    parent_cell_ids: np.ndarray
    stencils: tuple[tuple[tuple[int, Fraction], ...], ...]
    exact_coordinates: tuple[tuple[Fraction, Fraction, Fraction], ...]
    child_reference_triangles: tuple[tuple[tuple[Fraction, Fraction], ...], ...]


def nested_prism_reference_partition(
    partition: NestedLayerPartition, /
) -> ReferenceCellPartition:
    """Validate the complete exact target-prism partition of original prisms."""
    from ..discretization._nested_reference_types import (
        ReferenceCellMapWitnesses,
        ReferenceCellPartition,
        validate_reference_cell_partition,
    )

    exact = tuple(
        tuple((u, v, Fraction(level)) for level in (0, 1) for u, v in triangle)
        for triangle in partition.child_reference_triangles
    )
    witness = ReferenceCellMapWitnesses(
        partition.prism_ids,
        partition.parent_cell_ids,
        "prism",
        "prism",
        np.asarray(
            [[[float(value) for value in point] for point in cell] for cell in exact]
        ),
        exact,
    )
    result = ReferenceCellPartition((witness,))
    validate_reference_cell_partition(result)
    return result


def source_derived_nested_divisions(source: NativeLayerCoreSource, /) -> int:
    """Choose a dyadic cap partition from exact original lateral quad scales.

    Each cap boundary edge is paired with its lateral coplanar quad. The
    selected power of two makes cap subedges no longer than twice the smallest
    exact quad altitude; completed quality and controls remain independently
    certified against the unchanged request.
    """
    complex_ = source.complex
    polygons = np.asarray(complex_.polygon_vertices, dtype=np.int64).reshape(-1, 3)
    cap = {int(row) for row in np.asarray(source.cap_polygon_ids, dtype=np.int64)}
    cap_vertices = {int(vertex) for row in cap for vertex in polygons[row]}
    cap_uses: dict[tuple[int, int], int] = {}
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for row, triangle in enumerate(polygons):
        for i in range(3):
            edge = (
                min(int(triangle[i]), int(triangle[(i + 1) % 3])),
                max(int(triangle[i]), int(triangle[(i + 1) % 3])),
            )
            edge_faces.setdefault(edge, []).append(row)
            if row in cap:
                cap_uses[edge] = cap_uses.get(edge, 0) + 1
    points = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(complex_.vertices, dtype=np.float64)
    )

    def subtract(first: int, second: int) -> tuple[Fraction, ...]:
        return tuple(points[first][i] - points[second][i] for i in range(3))

    def dot(first: tuple[Fraction, ...], second: tuple[Fraction, ...]) -> Fraction:
        return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))

    def cross(
        first: tuple[Fraction, ...], second: tuple[Fraction, ...]
    ) -> tuple[Fraction, ...]:
        return (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )

    facet_rows = np.asarray(complex_.polygon_facets, dtype=np.int64)
    sides = np.asarray(complex_.facet_regions, dtype=np.int64)
    divisions = 1
    for edge, count in sorted(cap_uses.items()):
        if count != 1:
            continue
        adjacent = [row for row in edge_faces[edge] if row not in cap]
        if len(adjacent) != 1:
            raise ValueError(
                "A cap boundary edge must meet exactly one lateral source triangle."
            )
        first = adjacent[0]
        first_vertices = set(int(value) for value in polygons[first])
        candidates = []
        for row, triangle in enumerate(polygons):
            if row in cap or row == first:
                continue
            second_vertices = set(int(value) for value in triangle)
            shared = first_vertices & second_vertices
            union = first_vertices | second_vertices
            if (
                len(shared) != 2
                or len(union) != 4
                or not np.array_equal(sides[facet_rows[first]], sides[facet_rows[row]])
                or len(union - set(edge)) != 2
                or bool((union - set(edge)) & cap_vertices)
            ):
                continue
            a, b, c = (int(value) for value in polygons[first])
            normal = cross(subtract(b, a), subtract(c, a))
            other = next(iter(second_vertices - first_vertices))
            if dot(normal, subtract(other, a)) == 0:
                candidates.append(tuple(sorted(union - set(edge))))
        if not candidates:
            raise ValueError(
                "A cap boundary edge requires an exact coplanar lateral source quad."
            )
        direction = subtract(edge[1], edge[0])
        length_squared = dot(direction, direction)
        heights = []
        for upper in {vertex for pair in candidates for vertex in pair}:
            normal = cross(direction, subtract(upper, edge[0]))
            heights.append(dot(normal, normal))
        if length_squared <= 0 or min(heights) <= 0:
            raise ValueError("A lateral source quad has zero exact edge or altitude.")
        ratio_squared = length_squared * length_squared / (4 * min(heights))
        quotient = ratio_squared.numerator // ratio_squared.denominator
        required = math.isqrt(quotient)
        if required * required * ratio_squared.denominator < ratio_squared.numerator:
            required += 1
        while divisions < required:
            divisions *= 2
    return divisions


def subdivide_nested_prism_columns(
    mesh: CellMesh, divisions: int, /
) -> NestedLayerPartition:
    """Restrict each accepted prism by a conforming rational triangle partition.

    This keeps the accepted geometry, wall and cap exactly; callers must certify
    actual new-column wall distances and growth against the original control.
    """
    if not isinstance(mesh, CellMesh) or (
        mesh.topological_dimension,
        mesh.ambient_dimension,
    ) != (3, 3):
        raise TypeError("An accepted three-dimensional CellMesh is required.")
    if (
        isinstance(divisions, bool)
        or not isinstance(divisions, Integral)
        or divisions < 1
    ):
        raise ValueError("Transverse divisions must be a positive integer.")
    divisions = int(divisions)
    if any(block.cell_kind != "prism" for block in mesh.blocks):
        raise ValueError("Nested transverse subdivision requires prism-only layers.")
    source_points: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.asarray(
        mesh.coordinates, dtype=np.float64
    )
    source_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if np.unique(source_ids).size != source_ids.size:
        raise ValueError("Accepted layer vertex IDs must be unique.")
    exact_points = {
        int(identifier): tuple(Fraction(float(value)) for value in point)
        for identifier, point in zip(source_ids, source_points, strict=True)
    }
    key_type = tuple[tuple[int, Fraction], ...]

    def key(vertices: np.ndarray, bary: tuple[int, int, int]) -> key_type:
        return tuple(
            sorted(
                (int(source_ids[int(vertex)]), Fraction(weight, divisions))
                for vertex, weight in zip(vertices, bary, strict=True)
                if weight
            )
        )

    def barycentric(i: int, j: int) -> tuple[int, int, int]:
        return divisions - i - j, i, j

    triangles = []
    for i in range(divisions):
        for j in range(divisions - i):
            triangles.append(
                (barycentric(i, j), barycentric(i + 1, j), barycentric(i, j + 1))
            )
            if i + j + 1 < divisions:
                triangles.append(
                    (
                        barycentric(i + 1, j),
                        barycentric(i + 1, j + 1),
                        barycentric(i, j + 1),
                    )
                )
    cells = sorted(
        (
            (int(identifier), np.asarray(row, dtype=np.int64))
            for block in mesh.blocks
            for identifier, row in zip(block.global_ids, block.vertices, strict=True)
        ),
        key=lambda item: item[0],
    )
    if len({identifier for identifier, _ in cells}) != len(cells):
        raise ValueError("Accepted layer cell IDs must be unique.")
    new_keys: set[key_type] = set()
    children = []
    reference = []
    for parent, vertices in cells:
        if vertices.shape != (6,):
            raise ValueError("A prism must have six reference-ordered corners.")
        for triangle in triangles:
            child = tuple(key(vertices[:3], bary) for bary in triangle) + tuple(
                key(vertices[3:], bary) for bary in triangle
            )
            children.append((parent, child))
            reference.append(
                tuple(
                    (Fraction(bary[1], divisions), Fraction(bary[2], divisions))
                    for bary in triangle
                )
            )
            new_keys.update(item for item in child if len(item) != 1 or item[0][1] != 1)
    ordered_keys = tuple(sorted(new_keys))
    max_vertex = max((int(value) for value in source_ids), default=-1)
    max_cell = max((identifier for identifier, _ in cells), default=-1)
    if max_vertex + len(ordered_keys) > np.iinfo(np.int64).max or (
        divisions > 1 and max_cell + len(children) > np.iinfo(np.int64).max
    ):
        raise OverflowError("Nested layer subdivision exhausts entity IDs.")
    vertex_ids = np.concatenate(
        (
            source_ids,
            np.asarray(
                [max_vertex + 1 + row for row in range(len(ordered_keys))], dtype=np.int64
            ),
        )
    )
    stencils = (
        tuple(((int(identifier), Fraction(1)),) for identifier in source_ids)
        + ordered_keys
    )
    target_rows: dict[_StencilKey, int] = {
        ((int(identifier), Fraction(1)),): row
        for row, identifier in enumerate(source_ids)
    }
    target_rows.update(
        (item, source_ids.size + row) for row, item in enumerate(ordered_keys)
    )
    exact_coordinates = tuple(
        (values[0], values[1], values[2])
        for stencil in stencils
        for values in (
            tuple(
                sum(
                    (
                        weight * exact_points[identifier][axis]
                        for identifier, weight in stencil
                    ),
                    Fraction(0),
                )
                for axis in range(3)
            ),
        )
    )
    coordinates = np.asarray(
        [[float(value) for value in point] for point in exact_coordinates]
    )
    coordinates[: source_ids.size] = source_points
    prism_vertices = np.asarray(
        [[target_rows[item] for item in child] for _, child in children], dtype=np.int64
    ).reshape(-1, 6)
    parents = np.asarray([parent for parent, _ in children], dtype=np.int64)
    prism_ids = (
        parents.copy()
        if divisions == 1
        else np.asarray(
            [max_cell + 1 + row for row in range(len(children))], dtype=np.int64
        )
    )
    return NestedLayerPartition(
        coordinates,
        vertex_ids,
        prism_vertices,
        prism_ids,
        parents,
        stencils,
        exact_coordinates,
        tuple(reference),
    )


@dataclass(frozen=True, slots=True)
class NearestWallPointEvidence:
    """Exact nearest-wall squared distance and all co-nearest oriented supports."""

    squared_distance: Fraction
    # Each support is (authored wall face ID, exact barycentrics, signed normal dot).
    supports: tuple[tuple[int, tuple[Fraction, Fraction, Fraction], Fraction], ...]


@dataclass(frozen=True, slots=True)
class RefinedColumnEvidence:
    """Original-column stencil and exact physical wall-distance evidence."""

    source_columns: tuple[tuple[int, Fraction], ...]
    target_vertices: tuple[int, ...]
    wall_distances: tuple[NearestWallPointEvidence, ...]
    maximum_stretch_squared: Fraction


@dataclass(frozen=True, slots=True, weakref_slot=True)
class RefinedColumnCertification:
    """Exact refined-column evidence bound to its unmodified authoring request."""

    source_binding_id: str
    specification_id: str
    control_id: str
    policy_id: str
    columns: tuple[RefinedColumnEvidence, ...]
    certification_id: str = field(init=False)

    def __post_init__(self) -> None:
        def rational(value: Fraction) -> tuple[int, int]:
            return value.numerator, value.denominator

        payload = tuple(
            (
                tuple(
                    (identifier, *rational(weight))
                    for identifier, weight in column.source_columns
                ),
                column.target_vertices,
                tuple(
                    (
                        rational(distance.squared_distance),
                        tuple(
                            (
                                face,
                                tuple(rational(value) for value in bary),
                                rational(signed),
                            )
                            for face, bary, signed in distance.supports
                        ),
                    )
                    for distance in column.wall_distances
                ),
                rational(column.maximum_stretch_squared),
            )
            for column in self.columns
        )
        object.__setattr__(
            self,
            "certification_id",
            canonical_fingerprint(
                {
                    "kind": "refined-layer-column-certification",
                    "source_binding": self.source_binding_id,
                    "specification": self.specification_id,
                    "control": self.control_id,
                    "policy": self.policy_id,
                    "columns": payload,
                }
            ),
        )


def _nearest_wall_point(
    point: tuple[Fraction, ...],
    triangles: tuple[tuple[int, tuple[tuple[Fraction, ...], ...]], ...],
    /,
) -> NearestWallPointEvidence:
    def dot(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> Fraction:
        return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))

    best: Fraction | None = None
    supports: list[tuple[int, tuple[Fraction, Fraction, Fraction], Fraction]] = []
    for identifier, corners in triangles:
        a, b, c = corners
        u = tuple(b[i] - a[i] for i in range(3))
        v = tuple(c[i] - a[i] for i in range(3))
        w = tuple(point[i] - a[i] for i in range(3))
        uu, uv, vv = dot(u, u), dot(u, v), dot(v, v)
        denominator = uu * vv - uv * uv
        if denominator <= 0:
            raise ValueError("Original wall triangles must be exactly nondegenerate.")
        wu, wv = dot(w, u), dot(w, v)
        first = (wu * vv - wv * uv) / denominator
        second = (wv * uu - wu * uv) / denominator
        candidates: set[tuple[Fraction, Fraction, Fraction]] = set()
        if first >= 0 and second >= 0 and first + second <= 1:
            candidates.add((1 - first - second, first, second))
        for start, end in ((0, 1), (1, 2), (2, 0)):
            direction = tuple(corners[end][i] - corners[start][i] for i in range(3))
            relative = tuple(point[i] - corners[start][i] for i in range(3))
            parameter = max(
                Fraction(0),
                min(Fraction(1), dot(relative, direction) / dot(direction, direction)),
            )
            weights = [Fraction(0), Fraction(0), Fraction(0)]
            weights[start], weights[end] = 1 - parameter, parameter
            candidates.add((weights[0], weights[1], weights[2]))
        normal = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        signed = dot(w, normal)
        for weights in candidates:
            projected = tuple(
                sum((weights[k] * corners[k][axis] for k in range(3)), Fraction(0))
                for axis in range(3)
            )
            delta = tuple(point[i] - projected[i] for i in range(3))
            squared = dot(delta, delta)
            if best is None or squared < best:
                best, supports = squared, [(identifier, weights, signed)]
            elif squared == best:
                supports.append((identifier, weights, signed))
    if best is None:
        raise ValueError("The original layer control has no selected wall triangles.")
    return NearestWallPointEvidence(best, tuple(sorted(set(supports))))


class _PreparedWallTriangles:
    """Operation-local conservative boxes for exact nearest-wall queries."""

    def __init__(
        self,
        triangles: tuple[tuple[int, tuple[tuple[Fraction, ...], ...]], ...],
        /,
    ) -> None:
        self.triangles = triangles
        lower = []
        upper = []
        vertices = []
        for _, corners in triangles:
            a, b, c = corners
            u = tuple(b[i] - a[i] for i in range(3))
            v = tuple(c[i] - a[i] for i in range(3))
            uu = sum((value * value for value in u), Fraction(0))
            uv = sum((u[i] * v[i] for i in range(3)), Fraction(0))
            vv = sum((value * value for value in v), Fraction(0))
            if uu * vv - uv * uv <= 0:
                raise ValueError("Original wall triangles must be exactly nondegenerate.")
            lower.append(
                tuple(
                    fraction_down(min(corner[i] for corner in corners)) for i in range(3)
                )
            )
            upper.append(
                tuple(fraction_up(max(corner[i] for corner in corners)) for i in range(3))
            )
            vertices.extend(corners)
        self.vertices = tuple(vertices)
        lower_boxes = np.asarray(lower, dtype=np.float64).reshape(-1, 3)
        upper_boxes = np.asarray(upper, dtype=np.float64).reshape(-1, 3)
        approximate_vertices = np.asarray(
            [tuple(fraction_down(value) for value in vertex) for vertex in vertices],
            dtype=np.float64,
        ).reshape(-1, 3)
        # Float32 BVH storage is explicitly outward-rounded even when JAX
        # x64 is disabled. Extreme coordinates use the exhaustive owner.
        limit = np.finfo(np.float32).max / 4
        if (
            triangles
            and np.all(np.isfinite(lower_boxes))
            and np.all(np.isfinite(upper_boxes))
            and np.all(np.isfinite(approximate_vertices))
            and np.all(np.abs(lower_boxes) <= limit)
            and np.all(np.abs(upper_boxes) <= limit)
        ):
            self.wall_bvh = prepare_bvh(lower_boxes, upper_boxes, dtype=np.float32)
            self.vertex_tree = cKDTree(approximate_vertices)
        else:
            self.wall_bvh = None
            self.vertex_tree = None

    def candidates(
        self, point: tuple[Fraction, ...], /
    ) -> tuple[tuple[int, tuple[tuple[Fraction, ...], ...]], ...]:
        if self.wall_bvh is None or self.vertex_tree is None:
            return self.triangles
        # The seed is an actual wall vertex; its approximate selection has no
        # bearing on correctness. Its exact L1 distance bounds the true
        # Euclidean distance, so every co-nearest point lies in this cube.
        approximate = np.asarray([fraction_down(value) for value in point])
        limit = np.finfo(np.float32).max / 4
        if not np.all(np.isfinite(approximate)) or np.any(np.abs(approximate) > limit):
            return self.triangles
        _, index = self.vertex_tree.query(approximate, k=1)
        vertex = self.vertices[int(index)]
        radius = sum((abs(point[i] - vertex[i]) for i in range(3)), Fraction(0))
        lower = np.asarray([fraction_down(point[i] - radius) for i in range(3)])
        upper = np.asarray([fraction_up(point[i] + radius) for i in range(3)])
        if (
            not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(np.abs(lower) > limit)
            or np.any(np.abs(upper) > limit)
        ):
            return self.triangles
        query_bvh = prepare_bvh(lower[None, :], upper[None, :], dtype=np.float32)
        selected = sorted(
            int(index)
            for indices, _ in bvh_overlap_pair_blocks(
                self.wall_bvh, query_bvh, include_touching=True, item_block=256
            )
            for index in indices
        )
        return tuple(self.triangles[index] for index in selected)


def _distance_increment_at_least(
    upper: Fraction, lower: Fraction, minimum: Fraction, /
) -> bool:
    """Decide sqrt(upper)-sqrt(lower) >= minimum without rounding."""
    difference = upper - lower - minimum * minimum
    return difference >= 0 and difference * difference >= 4 * minimum * minimum * lower


def _growth_difference_sign(
    upper: Fraction, middle: Fraction, lower: Fraction, factor: Fraction, /
) -> int:
    """Sign of sqrt(upper)+factor*sqrt(lower)-(1+factor)*sqrt(middle)."""
    rhs = (1 + factor) ** 2 * middle - upper - factor * factor * lower
    if rhs < 0:
        return 1
    if rhs == 0:
        return int(factor > 0 and upper > 0 and lower > 0)
    difference = 4 * factor * factor * upper * lower - rhs * rhs
    return (difference > 0) - (difference < 0)


def certify_nested_layer_columns(
    source: NativeLayerCoreSource,
    partition: NestedLayerPartition,
    specification: VolumeMeshingSpec,
    /,
) -> RefinedColumnCertification:
    """Prove actual new-column nearest-wall increments against original controls."""
    layers = source.layers
    mesh = layers.mesh
    source_cells = {
        int(identifier): np.asarray(row, dtype=np.int64)
        for block in mesh.blocks
        for identifier, row in zip(block.global_ids, block.vertices, strict=True)
    }
    source_cell_ids = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    intervals = dict(
        zip(
            source_cell_ids.tolist(),
            np.asarray(layers.layer_index, dtype=np.int64).tolist(),
            strict=True,
        )
    )
    vertex_columns = np.asarray(layers.vertex_column_ids, dtype=np.int64)
    columns: dict[tuple[tuple[int, Fraction], ...], dict[int, int]] = {}
    for child, parent_id, reference in zip(
        partition.prism_vertices,
        partition.parent_cell_ids,
        partition.child_reference_triangles,
        strict=True,
    ):
        parent = source_cells[int(parent_id)]
        interval = intervals[int(parent_id)]
        for corner, (u, v) in enumerate(reference):
            bary = (1 - u - v, u, v)
            key = tuple(
                sorted(
                    (int(vertex_columns[int(parent[3 + index])]), weight)
                    for index, weight in enumerate(bary)
                    if weight
                )
            )
            if any(identifier < 0 for identifier, _ in key):
                raise ValueError("Original layer prism lacks source-column ancestry.")
            levels = columns.setdefault(key, {})
            for level, vertex in (
                (interval, int(child[corner])),
                (interval + 1, int(child[corner + 3])),
            ):
                if level in levels and levels[level] != vertex:
                    raise ValueError(
                        "A refined source column has inconsistent level vertices."
                    )
                levels[level] = vertex
    return _certify_column_bank(
        source, specification, columns, partition.vertex_ids, partition.exact_coordinates
    )


def _certify_column_bank(
    source: NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    columns: dict[tuple[tuple[int, Fraction], ...], dict[int, int]],
    vertex_ids: np.ndarray,
    exact_coordinates: tuple[tuple[Fraction, ...], ...],
    /,
) -> RefinedColumnCertification:
    control = source.layer_control
    layers = source.layers
    if (
        control is None
        or source.layer_policy is None
        or control.control_id != layers.control_id
        or source.layer_policy.policy_id != layers.policy_id
    ):
        raise ValueError("New layer columns require their original control and policy.")
    wall = layers.source_wall
    if control.wall_scope.entity_set_id != wall.entity_set(2).entity_set_id:
        raise ValueError("Original wall control must resolve in the retained wall mesh.")
    selected = {int(value) for value in np.asarray(control.wall_scope.entity_ids)}
    wall_points = np.asarray(wall.coordinates, dtype=np.float64)
    triangles = []
    for block in wall.blocks:
        if block.cell_kind != "triangle":
            raise ValueError(
                "Exact refined-column certification requires triangular wall cells."
            )
        for identifier, row in zip(block.global_ids, block.vertices, strict=True):
            if int(identifier) in selected:
                triangles.append(
                    (
                        int(identifier),
                        tuple(
                            tuple(
                                Fraction(float(value))
                                for value in wall_points[int(vertex)]
                            )
                            for vertex in row
                        ),
                    )
                )
    if {identifier for identifier, _ in triangles} != selected:
        raise ValueError("The original wall scope does not resolve every selected face.")
    triangles = tuple(triangles)
    face_normals = {}
    for identifier, (a, b, c) in triangles:
        u = tuple(b[i] - a[i] for i in range(3))
        v = tuple(c[i] - a[i] for i in range(3))
        face_normals[identifier] = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
    original_ids = np.asarray(layers.column_vertex_ids, dtype=np.int64)
    face_offsets = np.asarray(layers.column_source_face_offsets, dtype=np.int64)
    face_ids = np.asarray(layers.column_source_face_ids, dtype=np.int64)
    incident = {
        int(identifier): set(
            int(value) for value in face_ids[face_offsets[row] : face_offsets[row + 1]]
        )
        for row, identifier in enumerate(original_ids)
    }
    schedule = tuple(Fraction(float(value)) for value in control.schedule.thicknesses)
    compliance = specification.size_compliance
    absolute = Fraction(float(compliance.absolute_tolerance))
    relative = Fraction(float(compliance.relative_tolerance))
    fraction = Fraction(float(control.minimum_thickness_fraction))
    min_fraction = max(Fraction(0), fraction - absolute - relative * fraction)
    max_stretch = Fraction(float(control.maximum_corner_stretch))
    max_stretch += absolute + relative * max_stretch
    evidence = []
    prepared_wall = None
    for key, levels in sorted(columns.items()):
        face_sets = [incident.get(identifier, set()) & selected for identifier, _ in key]
        supporting_faces = set.intersection(*face_sets) if face_sets else set()
        if not supporting_faces:
            raise ValueError("A refined column has no common original wall-face support.")
        if set(levels) != set(range(len(schedule) + 1)):
            raise ValueError(
                "A refined source column does not cover the original schedule."
            )
        rows = tuple(levels[level] for level in range(len(schedule) + 1))
        points = tuple(exact_coordinates[row] for row in rows)
        if prepared_wall is None:
            prepared_wall = _PreparedWallTriangles(triangles)
        distances = tuple(
            _nearest_wall_point(point, prepared_wall.candidates(point))
            for point in points
        )
        if distances[0].squared_distance != 0:
            raise ValueError("A refined column starts away from its original wall.")
        stretch_squared = Fraction(1)
        for index, requested in enumerate(schedule, start=1):
            current, previous = distances[index], distances[index - 1]
            if any(signed <= 0 for _, _, signed in current.supports):
                raise ValueError(
                    "A refined layer column is not on the original inward wall side."
                )
            if not _distance_increment_at_least(
                current.squared_distance,
                previous.squared_distance,
                requested * min_fraction,
            ):
                raise ValueError(
                    "A refined layer column violates minimum wall thickness."
                )
            vector = tuple(points[index][axis] - points[0][axis] for axis in range(3))
            squared_length = sum((value * value for value in vector), Fraction(0))
            for face in supporting_faces:
                normal = face_normals[face]
                signed = sum((vector[i] * normal[i] for i in range(3)), Fraction(0))
                normal_squared = sum((value * value for value in normal), Fraction(0))
                if (
                    signed <= 0
                    or squared_length * normal_squared
                    > max_stretch * max_stretch * signed * signed
                ):
                    raise ValueError(
                        "A refined column violates an original incident-face stretch bound."
                    )
                stretch_squared = max(
                    stretch_squared, squared_length * normal_squared / (signed * signed)
                )
        if control.growth_rate_bounds is not None:
            lower, upper = (
                Fraction(float(value)) for value in control.growth_rate_bounds
            )
            tolerance = absolute + relative * max(abs(lower), abs(upper))
            lower = max(Fraction(0), lower - tolerance)
            upper += tolerance
            for index in range(1, len(schedule)):
                a, b, c = (
                    distances[index + 1].squared_distance,
                    distances[index].squared_distance,
                    distances[index - 1].squared_distance,
                )
                if (
                    _growth_difference_sign(a, b, c, lower) < 0
                    or _growth_difference_sign(a, b, c, upper) > 0
                ):
                    raise ValueError(
                        "A refined layer column violates original growth bounds."
                    )
        evidence.append(
            RefinedColumnEvidence(
                key,
                tuple(int(vertex_ids[row]) for row in rows),
                distances,
                stretch_squared,
            )
        )
    return RefinedColumnCertification(
        source.binding_id,
        specification.specification_id,
        control.control_id,
        source.layer_policy.policy_id,
        tuple(evidence),
    )


def _layer_exact_by_id(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> dict[int, tuple[Fraction, ...]]:
    """Evaluate and reconcile exact prism corner maps in the actual target mesh."""
    from ..discretization import reference_cell_topology

    target_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    exact_by_id: dict[int, tuple[Fraction, ...]] = {}
    for block in mesh.blocks:
        if block.cell_kind != "prism":
            continue
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        corners = tuple(
            tuple(Fraction(float(value)) for value in point)
            for point in reference_cell_topology(block.cell_kind).vertices
        )
        evaluated = geometry.evaluate_exact(
            mesh, identifiers, tuple(corners for _ in identifiers)
        )
        block_vertices: np.ndarray[tuple[int, int], np.dtype[np.int64]] = np.asarray(
            block.vertices, dtype=np.int64
        )
        for rows, points in zip(block_vertices, evaluated, strict=True):
            for row, point in zip(rows, points, strict=True):
                identifier = int(target_ids[int(row)])
                exact = tuple(point)
                if any(not isinstance(value, Fraction) for value in exact):
                    raise ValueError(
                        "Refined columns require rational source restrictions."
                    )
                exact = tuple(value for value in exact if isinstance(value, Fraction))
                if exact_by_id.setdefault(identifier, exact) != exact:
                    raise ValueError("Adjacent layer maps disagree at a refined vertex.")
                if tuple(float(value) for value in exact) != tuple(
                    float(value) for value in np.asarray(mesh.coordinates)[int(row)]
                ):
                    raise ValueError(
                        "Exact refined-column coordinates do not round to the mesh."
                    )
    return exact_by_id


# Column replays bound to the exact live source, mesh, map, request and column
# evidence they proved (see require_refined_column_certification).
_PROVEN_COLUMN_CERTIFICATIONS: LiveProofLedger[bool] = LiveProofLedger()


def require_refined_column_certification(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    specification: VolumeMeshingSpec,
    certification: RefinedColumnCertification | None,
    /,
) -> None:
    """Reprove accepted column stencils and wall-distance evidence from source.

    The replay is made once per exact live operand set: a later call on the same
    unchanged objects reuses it, and any copy, rebuild or changed content
    replays in full.
    """
    if not isinstance(certification, RefinedColumnCertification):
        raise TypeError("Accepted conforming layers need their column certification.")
    operands = (source, mesh, geometry, specification, certification)
    if _PROVEN_COLUMN_CERTIFICATIONS.find(operands) is not None:
        return
    fingerprint = live_content_fingerprint(operands)
    _replay_refined_column_certification(
        source, mesh, geometry, specification, certification
    )
    _PROVEN_COLUMN_CERTIFICATIONS.record(operands, fingerprint, True)


def _replay_refined_column_certification(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    specification: VolumeMeshingSpec,
    certification: RefinedColumnCertification,
    /,
) -> None:
    target_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    exact_by_id = _layer_exact_by_id(mesh, geometry)
    original = source.layers.mesh
    original_points = np.asarray(original.coordinates, dtype=np.float64)
    source_columns = np.asarray(source.layers.vertex_column_ids, dtype=np.int64)
    source_ids = np.sort(np.asarray(original.entity_set(3).entity_ids, dtype=np.int64))
    source_intervals = dict(
        zip(
            source_ids.tolist(),
            np.asarray(source.layers.layer_index, dtype=np.int64).tolist(),
            strict=True,
        )
    )
    roots: dict[tuple[int, int], tuple[Fraction, ...]] = {}
    for block in original.blocks:
        if block.cell_kind != "prism":
            raise ValueError("Refined-column replay requires original prism layers.")
        for identifier, cell in zip(block.global_ids, block.vertices, strict=True):
            interval = source_intervals[int(identifier)]
            for corner in range(3):
                column = int(source_columns[int(cell[corner + 3])])
                if column < 0:
                    raise ValueError("Original prism lacks its exact column identity.")
                for level, vertex in (
                    (interval, int(cell[corner])),
                    (interval + 1, int(cell[corner + 3])),
                ):
                    point = tuple(
                        Fraction(float(value)) for value in original_points[vertex]
                    )
                    old = roots.setdefault((column, level), point)
                    if old != point:
                        raise ValueError(
                            "Original column level has conflicting source coordinates."
                        )
    positions = {int(identifier): row for row, identifier in enumerate(target_ids)}
    columns: dict[tuple[tuple[int, Fraction], ...], dict[int, int]] = {}
    for column in certification.columns:
        key = column.source_columns
        if (
            not key
            or key != tuple(sorted(set(key)))
            or any(not isinstance(weight, Fraction) or weight <= 0 for _, weight in key)
            or sum((weight for _, weight in key), Fraction(0)) != 1
            or key in columns
        ):
            raise ValueError("Refined source-column stencils are not canonical.")
        levels = {}
        for level, identifier in enumerate(column.target_vertices):
            if identifier not in positions or any(
                (root, level) not in roots for root, _ in key
            ):
                raise ValueError(
                    "A refined column references an absent target or source root."
                )
            row = positions[identifier]
            expected = tuple(
                sum(
                    (weight * roots[(root, level)][axis] for root, weight in key),
                    Fraction(0),
                )
                for axis in range(3)
            )
            if expected != exact_by_id.get(identifier):
                raise ValueError(
                    "A refined column changed its original exact layer geometry."
                )
            levels[level] = row
        columns[key] = levels
    referenced = {
        int(identifier)
        for column in certification.columns
        for identifier in column.target_vertices
    }
    prism_vertices = {
        int(target_ids[int(row)])
        for block in mesh.blocks
        if block.cell_kind == "prism"
        for cell in np.asarray(block.vertices, dtype=np.int64)
        for row in cell
    }
    if referenced != prism_vertices:
        raise ValueError(
            "Refined-column evidence does not cover every accepted layer vertex."
        )
    exact_coordinates = tuple(
        exact_by_id.get(int(identifier), (Fraction(0), Fraction(0), Fraction(0)))
        for identifier in target_ids
    )
    rebuilt = _certify_column_bank(
        source, specification, columns, target_ids, exact_coordinates
    )
    if (
        rebuilt != certification
        or rebuilt.certification_id != certification.certification_id
    ):
        raise ValueError(
            "Refined-column evidence is stale or differs from original controls."
        )


def renew_refined_column_certification(
    source: NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    source_result: CellMeshingResult,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    vertex_stencil: object,
    geometry_transition: object,
    /,
) -> RefinedColumnCertification:
    """Compose a mixed transition's exact stencils and reprove current columns.

    Layer vertices must remain on their original schedule level; an axial split,
    a missing source-column term, or an unproved coordinate-map transition is
    refused. The returned evidence describes the current target, not the root.
    """
    from ..discretization._cell_geometry_transition_types import CellGeometryTransition
    from ._lineage import VertexInterpolationStencil

    previous = source_result.layer_core_column_certification
    require_refined_column_certification(
        source, source_result.mesh, source_result.geometry, specification, previous
    )
    if not isinstance(previous, RefinedColumnCertification):
        raise TypeError("Accepted conforming layers need their column certification.")
    if not isinstance(vertex_stencil, VertexInterpolationStencil):
        raise TypeError("Refined columns need the exact mixed vertex stencil.")
    if not isinstance(geometry_transition, CellGeometryTransition):
        raise TypeError("Refined columns need an exact mixed geometry transition.")
    geometry_transition = CellGeometryTransition.from_stored_fields(geometry_transition)
    geometry_transition.require_bound(
        source_result.mesh, source_result.geometry, target_mesh, target_geometry
    )
    if geometry_transition.evidence.kind != "nested_restriction":
        raise ValueError("Refined columns require exact nested geometry ancestry.")
    if (
        vertex_stencil.source_entity_set_id
        != source_result.mesh.entity_set(0).entity_set_id
        or vertex_stencil.target_entity_set_id != target_mesh.entity_set(0).entity_set_id
    ):
        raise ValueError(
            "Mixed vertex stencils do not bind the actual transition endpoints."
        )
    old_vertices: dict[int, tuple[tuple[tuple[int, Fraction], ...], int]] = {}
    for column in previous.columns:
        for level, identifier in enumerate(column.target_vertices):
            value = (column.source_columns, level)
            if old_vertices.setdefault(identifier, value) != value:
                raise ValueError(
                    "A source vertex has conflicting exact column identities."
                )
    target_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    positions = {int(identifier): row for row, identifier in enumerate(target_ids)}
    required = {
        int(target_ids[int(row)])
        for block in target_mesh.blocks
        if block.cell_kind == "prism"
        for cell in np.asarray(block.vertices, dtype=np.int64)
        for row in cell
    }
    target_rows = np.asarray(vertex_stencil.target_global_ids, dtype=np.int64)
    stencil_sources: np.ndarray[tuple[int, int], np.dtype[np.int64]] = np.asarray(
        vertex_stencil.source_global_ids, dtype=np.int64
    )
    valid: np.ndarray[tuple[int, int], np.dtype[np.bool_]] = np.asarray(
        vertex_stencil.valid, dtype=np.bool_
    )
    if np.unique(target_rows).size != target_rows.size:
        raise ValueError("Mixed vertex stencil repeats a target vertex.")
    columns: dict[tuple[tuple[int, Fraction], ...], dict[int, int]] = {}
    seen: set[int] = set()
    for identifier, roots, flags, weights in zip(
        target_rows.tolist(),
        stencil_sources,
        valid,
        vertex_stencil.exact_weights,
        strict=True,
    ):
        if identifier not in required:
            continue
        if identifier not in positions:
            raise ValueError("Mixed stencil references an absent target layer vertex.")
        active = [
            (
                int(root),
                rational_coefficient(weight, role="Mixed layer vertex stencil weight"),
            )
            for root, flag, weight in zip(roots, flags, weights, strict=True)
            if flag and weight
        ]
        if not active or any(
            weight < 0 or root not in old_vertices for root, weight in active
        ):
            raise ValueError("A target layer vertex lacks convex source-column ancestry.")
        if sum((weight for _, weight in active), Fraction(0)) != 1:
            raise ValueError("Mixed layer vertex weights do not preserve constants.")
        levels = {old_vertices[root][1] for root, _ in active}
        if len(levels) != 1:
            raise ValueError(
                "Mixed layer adaptation changed an original schedule interval."
            )
        level = next(iter(levels))
        combined: dict[int, Fraction] = {}
        for root, weight in active:
            for column_id, coefficient in old_vertices[root][0]:
                combined[column_id] = (
                    combined.get(column_id, Fraction(0)) + weight * coefficient
                )
        key = tuple(
            sorted(
                (column_id, weight) for column_id, weight in combined.items() if weight
            )
        )
        row = positions[identifier]
        target_levels = columns.setdefault(key, {})
        if level in target_levels and target_levels[level] != row:
            raise ValueError("Mixed adaptation duplicated a source-column level.")
        target_levels[level] = row
        seen.add(identifier)
    if seen != required:
        raise ValueError("Mixed vertex stencils omit accepted layer vertices.")
    exact_by_id = _layer_exact_by_id(target_mesh, target_geometry)
    exact = tuple(
        exact_by_id.get(int(identifier), (Fraction(0), Fraction(0), Fraction(0)))
        for identifier in target_ids
    )
    renewed = _certify_column_bank(source, specification, columns, target_ids, exact)
    require_refined_column_certification(
        source, target_mesh, target_geometry, specification, renewed
    )
    return renewed


def require_descendant_overlap_source_provenance(
    source: NativeLayerCoreSource, result: CellMeshingResult, /
) -> None:
    """Reprove every retained root stencil against the actual descendant maps."""
    from ..discretization import reference_cell_topology
    from .providers._native_sources import NativeLayerCoreSource

    if not isinstance(source, NativeLayerCoreSource) or not source.conforming_execution:
        raise ValueError("Descendant overlap requires its authored conforming source.")
    source.validate_restored()

    mesh = result.mesh
    geometry = result.geometry
    witness = result.layer_core_overlap_stencils
    if not isinstance(witness, OverlapSourceStencils):
        raise ValueError(
            "A conforming descendant must retain its original source stencils."
        )
    rebuilt = OverlapSourceStencils(
        witness.source_binding_id, witness.target_mesh_id, witness.rows
    )
    if (
        rebuilt.witness_id != witness.witness_id
        or witness.source_binding_id != source.binding_id
    ):
        raise ValueError("Retained source stencils have stale original-source identity.")
    if mesh.mesh_id != witness.target_mesh_id and geometry.restriction_source is None:
        raise ValueError("A descendant must retain its exact root restriction authority.")
    ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    positions = {int(identifier): row for row, identifier in enumerate(ids)}
    retained = {identifier for identifier, _ in witness.rows}
    if not retained.issubset(positions):
        raise ValueError("A descendant dropped an original witnessed source vertex.")
    periodic = mesh.periodic_topology
    required = set(retained)
    if periodic is not None:
        representatives = np.asarray(periodic.vertex_representatives, dtype=np.int64)
        required.update(
            int(ids[int(representatives[positions[value]])]) for value in retained
        )
    exact: dict[int, tuple[Fraction, ...]] = {}
    coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.asarray(
        mesh.coordinates, dtype=np.float64
    )
    for block in mesh.blocks:
        cells: np.ndarray[tuple[int, int], np.dtype[np.int64]] = np.asarray(
            block.vertices, dtype=np.int64
        )
        active = np.asarray(
            [any(int(ids[int(row)]) in required for row in cell) for cell in cells],
            dtype=np.bool_,
        )
        if not np.any(active):
            continue
        local = tuple(
            tuple(Fraction(float(value)) for value in point)
            for point in reference_cell_topology(block.cell_kind).vertices
        )
        selected = np.asarray(block.global_ids, dtype=np.int64)[active]
        evaluated = geometry.evaluate_exact(
            mesh, selected, tuple(local for _ in selected)
        )
        selected_cells: np.ndarray[tuple[int, int], np.dtype[np.int64]] = cells[active]
        for cell, values in zip(selected_cells, evaluated, strict=True):
            for row, point in zip(cell, values, strict=True):
                identifier = int(ids[int(row)])
                if identifier not in required:
                    continue
                value = tuple(point)
                if any(not isinstance(component, Fraction) for component in value):
                    raise ValueError(
                        "Descendant overlap roots require rational exact geometry."
                    )
                value = tuple(
                    component for component in value if isinstance(component, Fraction)
                )
                if exact.setdefault(identifier, value) != value:
                    raise ValueError(
                        "Descendant maps disagree at a retained source vertex."
                    )
                if tuple(float(component) for component in value) != tuple(
                    coordinates[int(row)]
                ):
                    raise ValueError(
                        "Descendant exact source point does not round to its mesh."
                    )
    if set(exact) != required:
        raise ValueError(
            "Descendant geometry does not realize every retained source point."
        )
    roots: dict[tuple[str, int], tuple[Fraction, ...]] = {}
    layer_coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.asarray(
        source.layers.mesh.coordinates, dtype=np.float64
    )
    for identifier, point in zip(
        np.asarray(source.layers.mesh.vertex_global_ids, dtype=np.int64),
        layer_coordinates,
        strict=True,
    ):
        roots[("layer", int(identifier))] = tuple(
            Fraction(float(value)) for value in point
        )
    plc_coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.asarray(
        source.complex.vertices, dtype=np.float64
    )
    for identifier, point in zip(
        source.complex.vertex_entity_ids,
        plc_coordinates,
        strict=True,
    ):
        roots[("plc", int(identifier))] = tuple(Fraction(float(value)) for value in point)
    for identifier, terms in witness.rows:
        if any((kind, root) not in roots for kind, root, _ in terms):
            raise ValueError("A retained overlap stencil names an absent original root.")
        expected = tuple(
            sum(
                (weight * roots[(kind, root)][axis] for kind, root, weight in terms),
                Fraction(0),
            )
            for axis in range(3)
        )
        if exact[identifier] != expected:
            raise ValueError("A descendant moved an original source-stencil point.")
    if periodic is not None:
        shifts = np.asarray(periodic.vertex_shifts, dtype=np.int64)
        for identifier in retained:
            row = positions[identifier]
            root = int(ids[int(representatives[row])])
            matrix = exact_periodic_element(periodic.cell, shifts[row])
            image = tuple(
                sum(
                    (matrix[axis][column] * exact[root][column] for column in range(3)),
                    matrix[axis][3],
                )
                for axis in range(3)
            )
            if image != exact[identifier]:
                raise ValueError(
                    "A descendant changed the exact authored periodic action."
                )
