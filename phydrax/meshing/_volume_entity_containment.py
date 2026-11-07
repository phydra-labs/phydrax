#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact cross-dimensional entity containment for mixed volume transitions."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from ..discretization import CellMesh
from ..discretization._cell_geometry_transition_types import CellGeometryTransition
from ..discretization._coordinate_enclosure import Expression, expression_equal
from ..discretization._nested_reference_types import (
    _reference_cell_contains,
    exact_reference_cell_vertices,
    ReferenceCellMapWitnesses,
)
from ..discretization._reference_cell import reference_cell_topology
from ..geometry._exact_polyhedral_geometry import point_in_triangle3, point_on_segment3
from ._mixed_adaptation import (
    _affine_chart,
    _exact_child_to_parent_affine_map,
    _point_in_root_hull,
)
from ._topology_edit import entity_keys


type _CellEntities = dict[int, tuple[str, tuple[tuple[int, ...], ...]]]
type _PointKey = tuple[tuple[int, Fraction], ...]


class ContainedEntity(NamedTuple):
    """A fine entity contained in a coarse entity in one exact cell chart."""

    fine_dimension: int
    fine_id: int
    coarse_dimension: int
    coarse_id: int
    fine_cell_id: int
    coarse_cell_id: int


def _cell_entities(mesh: CellMesh, /) -> _CellEntities:
    """Resolve every local reference entity to its global mesh identity."""
    top = mesh.topological_dimension
    lookup: list[dict[tuple[int, ...], int]] = []
    for dimension in range(top):
        keys = entity_keys(mesh, dimension)
        ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
        lookup.append(
            {
                tuple(int(v) for v in key if v >= 0): int(identifier)
                for key, identifier in zip(keys, ids, strict=True)
            }
        )
    vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells: _CellEntities = {}
    for block in mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        for cell_id, rows in zip(
            np.asarray(block.global_ids, dtype=np.int64),
            np.asarray(block.vertices, dtype=np.int64),
            strict=True,
        ):
            global_vertices = vertices[rows]
            local: list[tuple[int, ...]] = []
            for dimension in range(top):
                local.append(
                    tuple(
                        lookup[dimension][
                            tuple(sorted(int(global_vertices[i]) for i in entity))
                        ]
                        for entity in topology.entities[dimension]
                    )
                )
            local.append((int(cell_id),))
            cells[int(cell_id)] = (block.cell_kind, tuple(local))
    return cells


def _point_key(point: tuple[Fraction, ...], /) -> _PointKey:
    return tuple((axis, value) for axis, value in enumerate(point) if value)


def _reference_subentity_contains(
    kind: str, dimension: int, index: int, point: tuple[Fraction, ...], /
) -> bool:
    """Exact containment in a canonical standard reference-cell subentity."""
    topology = reference_cell_topology(kind)
    vertices = tuple(
        tuple(Fraction(float(value)) for value in topology.vertices[vertex])
        for vertex in topology.entities[dimension][index]
    )
    if len(point) != 3 or any(len(vertex) != 3 for vertex in vertices):
        raise ValueError("Volume reference entities require three coordinates.")
    point3 = (point[0], point[1], point[2])
    vertices3 = tuple((vertex[0], vertex[1], vertex[2]) for vertex in vertices)
    if dimension == 0 and len(vertices) == 1:
        return point == vertices[0]
    if dimension == 1 and len(vertices) == 2:
        return point_on_segment3(point3, vertices3[0], vertices3[1])
    if dimension == 2 and len(vertices) == 3:
        return point_in_triangle3(point3, (vertices3[0], vertices3[1], vertices3[2]))
    if (
        dimension == 2
        and len(vertices) == 4
        and kind in ("hexahedron", "tensor:3", "prism", "pyramid")
    ):
        # Standard reference-cell facet loops are convex and cyclic. Both
        # triangles lie in the exact same reference plane.
        return point_in_triangle3(point3, (vertices3[0], vertices3[1], vertices3[2])) or (
            point_in_triangle3(point3, (vertices3[0], vertices3[2], vertices3[3]))
        )
    if dimension == 3 and kind in (
        "tetrahedron",
        "simplex:3",
        "hexahedron",
        "tensor:3",
        "prism",
        "pyramid",
    ):
        return _reference_cell_contains(kind, point)
    return _point_in_root_hull(
        _point_key(point), tuple(_point_key(vertex) for vertex in vertices)
    )


def reference_entity_containment(
    fine_mesh: CellMesh,
    coarse_mesh: CellMesh,
    groups: tuple[ReferenceCellMapWitnesses, ...],
    /,
) -> tuple[ContainedEntity, ...]:
    """Collect exact containment from affine fine-to-coarse reference maps."""
    if fine_mesh.topological_dimension != 3 or coarse_mesh.topological_dimension != 3:
        raise ValueError("Mixed volume containment requires three-dimensional cells.")
    fine_cells = _cell_entities(fine_mesh)
    coarse_cells = _cell_entities(coarse_mesh)
    contains = lru_cache(maxsize=8192)(_reference_subentity_contains)

    @lru_cache(maxsize=8192)
    def reference_affine(
        kind: str, corners: tuple[tuple[Fraction, ...], ...], /
    ) -> tuple[Expression, ...]:
        matrix, offset = _exact_child_to_parent_affine_map(kind, corners)
        return _affine_chart(matrix, offset)

    result: set[ContainedEntity] = set()
    for witness in groups:
        exact = exact_reference_cell_vertices(witness)
        if witness.source_reference_chart_arguments is None:
            raise ValueError(
                "Exact mixed containment requires explicit affine chart expressions."
            )
        for row, (fine_raw, coarse_raw) in enumerate(
            zip(witness.target_cell_ids, witness.source_cell_ids, strict=True)
        ):
            fine_id, coarse_id = int(fine_raw), int(coarse_raw)
            fine_kind, fine_entities = fine_cells[fine_id]
            coarse_kind, coarse_entities = coarse_cells[coarse_id]
            if (
                fine_kind != witness.target_cell_kind
                or coarse_kind != witness.source_cell_kind
            ):
                raise ValueError(
                    "Reference map cell kinds do not match their endpoint cells."
                )
            affine = reference_affine(fine_kind, exact[row])
            declared = witness.source_reference_chart_arguments[row]
            if len(declared) != len(affine) or not all(
                expression_equal(dict(left) if isinstance(left, Mapping) else left, right)
                for left, right in zip(declared, affine, strict=True)
            ):
                raise ValueError(
                    "Entity containment requires an exact affine reference map."
                )
            fine_topology = reference_cell_topology(fine_kind)
            coarse_topology = reference_cell_topology(coarse_kind)
            for fine_dim in range(4):
                for fine_index, fine_corners in enumerate(
                    fine_topology.entities[fine_dim]
                ):
                    points = tuple(exact[row][corner] for corner in fine_corners)
                    for coarse_dim in range(fine_dim, 4):
                        for coarse_index, _ in enumerate(
                            coarse_topology.entities[coarse_dim]
                        ):
                            if all(
                                contains(coarse_kind, coarse_dim, coarse_index, point)
                                for point in points
                            ):
                                result.add(
                                    ContainedEntity(
                                        fine_dim,
                                        fine_entities[fine_dim][fine_index],
                                        coarse_dim,
                                        coarse_entities[coarse_dim][coarse_index],
                                        fine_id,
                                        coarse_id,
                                    )
                                )
    return tuple(sorted(result))


def nested_volume_entity_containment(
    transition: CellGeometryTransition, /
) -> tuple[tuple[ContainedEntity, ...], tuple[ContainedEntity, ...]]:
    """Return exact fine/coarse containment for both transition directions."""
    if not isinstance(transition, CellGeometryTransition):
        raise TypeError("transition must be CellGeometryTransition.")
    transition.require_bound(
        transition.source_mesh,
        transition.source_geometry,
        transition.target_mesh,
        transition.target_geometry,
    )
    if transition.reference_map is None:
        direct = ()
    elif isinstance(transition.reference_map, ReferenceCellMapWitnesses):
        direct = (transition.reference_map,)
    else:
        direct = transition.reference_map
    partition = transition.coarsening_partition
    reverse = () if partition is None else partition.maps
    return (
        reference_entity_containment(
            transition.target_mesh, transition.source_mesh, direct
        ),
        reference_entity_containment(
            transition.source_mesh, transition.target_mesh, reverse
        ),
    )


__all__ = [
    "ContainedEntity",
    "nested_volume_entity_containment",
    "reference_entity_containment",
]
