#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ...discretization._cell_mesh import CellMesh
from ...discretization._periodic_exact import exact_periodic_element
from ...discretization._periodic_topology import _entity_vertex_rows, PeriodicMeshTopology
from ...typing import checked


def _exact_point(
    matrix: tuple[tuple[Fraction, ...], ...], point: np.ndarray, /
) -> tuple[Fraction, ...]:
    values = tuple(Fraction.from_float(float(item)) for item in point) + (Fraction(1),)
    dimension = point.size
    return tuple(
        sum(
            (matrix[row][column] * values[column] for column in range(dimension + 1)),
            Fraction(),
        )
        for row in range(dimension)
    )


@checked
def periodic_quotient_edge_squared_lengths(
    mesh: CellMesh,
    /,
) -> tuple[np.ndarray, np.ndarray, tuple[Fraction, ...]]:
    """Exact squared lengths of each quotient edge, preserving its winding action."""
    topology = mesh.periodic_topology
    if not isinstance(topology, PeriodicMeshTopology):
        raise ValueError("mesh must retain PeriodicMeshTopology.")
    topology.validate_restored()
    topology.require_lifted(mesh.topology, mesh.topology_id, mesh.blocks)
    topology.require_coordinates(mesh.coordinates)
    edge_rows = _entity_vertex_rows(mesh)[1]
    lifted_offsets = np.asarray(topology.lifted_offsets, dtype=np.int64)
    quotient_offsets = np.asarray(topology.quotient_offsets, dtype=np.int64)
    quotient_begin, quotient_end = map(int, quotient_offsets[1:3])
    representatives = np.asarray(topology.representatives, dtype=np.int64)[
        quotient_begin:quotient_end
    ]
    if representatives.size != quotient_end - quotient_begin:
        raise ValueError("Periodic quotient edge representative table is malformed.")
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    vertex_representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    vertex_shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    vertex_orbits = (
        np.asarray(topology.orbit_indices, dtype=np.int64)[lifted_offsets[0]:lifted_offsets[1]]
        - quotient_offsets[0]
    )
    vertex_ids = np.asarray(topology.quotient.entity_sets[0].entity_ids, dtype=np.int64)
    edge_ids = np.asarray(topology.quotient.entity_sets[1].entity_ids, dtype=np.int64)
    endpoints = np.empty((representatives.size, 2), dtype=np.int64)
    squared: list[Fraction] = []
    for row, lifted_edge in enumerate(representatives):
        if not 0 <= lifted_edge < len(edge_rows):
            raise ValueError("Periodic quotient edge representative is out of range.")
        vertices = np.asarray(edge_rows[int(lifted_edge)], dtype=np.int64)
        if vertices.shape != (2,):
            raise ValueError("Periodic quotient edges must have two vertices.")
        first, second = map(int, vertices)
        endpoints[row] = (
            vertex_ids[vertex_orbits[first]],
            vertex_ids[vertex_orbits[second]],
        )
        first_action = exact_periodic_element(topology.cell, vertex_shifts[first])
        second_action = exact_periodic_element(topology.cell, vertex_shifts[second])
        first_point = _exact_point(first_action, points[vertex_representatives[first]])
        second_point = _exact_point(second_action, points[vertex_representatives[second]])
        squared.append(
            sum(
                ((second_point[axis] - first_point[axis]) ** 2 for axis in range(points.shape[1])),
                Fraction(),
            )
        )
    if edge_ids.shape != (len(squared),):
        raise ValueError("Periodic quotient edge IDs do not align with representative rows.")
    return edge_ids, endpoints, tuple(squared)


__all__ = ["periodic_quotient_edge_squared_lengths"]
