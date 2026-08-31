#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np

from ._diagram import PersistenceDiagram
from ._filtration import CellFiltration


def compute_h0_persistence_union_find(
    filtration: CellFiltration,
    /,
) -> PersistenceDiagram:
    """Compute exact H0 persistence by elder-rule union-find on vertices and edges."""
    if filtration.max_degree < 1:
        births = np.asarray(filtration.values[0])[
            np.asarray(filtration.complex.masks[0], dtype=bool)
        ]
        entity_ids = np.asarray(filtration.complex.topology.entity_sets[0].entity_ids)[
            np.asarray(filtration.complex.masks[0], dtype=bool)
        ]
        return PersistenceDiagram(
            np.zeros((births.size,), dtype=np.int32),
            births,
            np.zeros_like(births),
            entity_ids,
            np.zeros_like(entity_ids),
            np.zeros((births.size,), dtype=bool),
            np.arange(births.size, dtype=np.int32),
            source_id=f"union-find:{filtration.filtration_id}",
        )
    vertex_slots = np.asarray(
        filtration.complex.layout.compact_to_ambient[0], dtype=np.int32
    )
    inverse = np.asarray(filtration.complex.layout.ambient_to_compact[0], dtype=np.int32)
    vertex_values = np.asarray(filtration.values[0])[vertex_slots]
    vertex_entities = np.asarray(filtration.complex.topology.entity_sets[0].entity_ids)[
        vertex_slots
    ]
    canonical_vertex = (
        vertex_values if filtration.direction == "sublevel" else -vertex_values
    )
    parent = np.arange(vertex_slots.size, dtype=np.int32)
    birth = canonical_vertex.copy()
    birth_entity = vertex_entities.copy()
    active = np.zeros((vertex_slots.size,), dtype=bool)

    def root(index):
        current = int(index)
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = int(parent[current])
        return current

    incidence = filtration.complex.topology.incidences[0]
    valid = np.asarray(incidence.relation.valid, dtype=bool)
    lower = np.asarray(incidence.relation.source_indices)[valid]
    edges = np.asarray(incidence.relation.target_indices)[valid]
    endpoints: dict[int, list[int]] = {}
    for vertex, edge in zip(lower, edges, strict=True):
        compact_vertex = int(inverse[int(vertex)])
        if compact_vertex >= 0:
            endpoints.setdefault(int(edge), []).append(compact_vertex)
    records = []
    order_degrees = np.asarray(filtration.order_degrees)
    order_ambient = np.asarray(filtration.order_ambient_indices)
    for degree, ambient in zip(order_degrees, order_ambient, strict=True):
        if degree == 0:
            compact_vertex = int(inverse[int(ambient)])
            if compact_vertex >= 0:
                active[compact_vertex] = True
            continue
        if degree != 1:
            continue
        vertices = endpoints.get(int(ambient), [])
        if len(vertices) != 2 or not all(active[index] for index in vertices):
            continue
        left = root(vertices[0])
        right = root(vertices[1])
        if left == right:
            continue
        older, younger = (
            (left, right)
            if (birth[left], int(birth_entity[left]))
            <= (birth[right], int(birth_entity[right]))
            else (right, left)
        )
        parent[younger] = older
        death_value = np.asarray(filtration.values[1])[int(ambient)]
        records.append(
            (
                vertex_values[younger],
                death_value,
                birth_entity[younger],
                int(
                    np.asarray(filtration.complex.topology.entity_sets[1].entity_ids)[
                        int(ambient)
                    ]
                ),
                True,
            )
        )
    essential_roots = sorted(
        {root(index) for index in range(vertex_slots.size) if active[index]}
    )
    for index in essential_roots:
        records.append((vertex_values[index], 0.0, birth_entity[index], 0, False))
    records.sort(key=lambda value: (float(value[0]), not value[4], int(value[2])))
    return PersistenceDiagram(
        np.zeros((len(records),), dtype=np.int32),
        np.asarray([value[0] for value in records]),
        np.asarray([value[1] for value in records]),
        np.asarray([value[2] for value in records], dtype=np.int64),
        np.asarray([value[3] for value in records], dtype=np.int64),
        np.asarray([value[4] for value in records], dtype=bool),
        np.arange(len(records), dtype=np.int32),
        source_id=f"union-find:{filtration.filtration_id}",
    )


__all__ = ["compute_h0_persistence_union_find"]
