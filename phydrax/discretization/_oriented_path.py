#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._topology import CellComplexTopology


def _edge_endpoints(topology: CellComplexTopology, /) -> tuple[np.ndarray, np.ndarray]:
    if topology.dimension < 1:
        raise ValueError("Oriented edge paths require a topology with edges.")
    incidence = topology.incidences[0]
    relation = incidence.relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    vertices = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    edges = np.asarray(relation.target_indices, dtype=np.int64)[valid]
    signs = np.asarray(incidence.signs)[valid]
    edge_count = topology.entities(1).count
    tails = np.full((edge_count,), -1, dtype=np.int32)
    heads = np.full((edge_count,), -1, dtype=np.int32)
    counts = np.zeros((edge_count,), dtype=np.int32)
    for vertex, edge, sign in zip(vertices, edges, signs, strict=True):
        counts[edge] += 1
        if sign == -1:
            if tails[edge] >= 0:
                raise ValueError("Every oriented edge requires exactly one tail.")
            tails[edge] = vertex
        elif sign == 1:
            if heads[edge] >= 0:
                raise ValueError("Every oriented edge requires exactly one head.")
            heads[edge] = vertex
        else:
            raise ValueError("Edge incidence signs must be plus or minus one.")
    if np.any(counts != 2) or np.any(tails < 0) or np.any(heads < 0):
        raise ValueError("Every oriented edge requires one tail and one head incidence.")
    return tails, heads


def oriented_edge_endpoints(topology: CellComplexTopology, /) -> tuple[Array, Array]:
    """Return canonical tail and head vertex indices for every oriented edge."""
    tails, heads = _edge_endpoints(topology)
    return jnp.asarray(tails, dtype=jnp.int32), jnp.asarray(heads, dtype=jnp.int32)


class OrientedEdgePathPlan(StrictModule, NonTrainableState):
    """Fixed-capacity ordered traversals through one cell-complex edge set."""

    topology: CellComplexTopology
    edge_indices: Array
    orientations: Array
    valid: Array
    start_vertices: Array
    end_vertices: Array
    path_names: tuple[str, ...] = eqx.field(static=True)
    require_closed: bool = eqx.field(static=True)
    num_paths: int = eqx.field(static=True)
    max_length: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    edge_entity_set_id: str = eqx.field(static=True)
    path_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: CellComplexTopology,
        edge_indices: ArrayLike,
        orientations: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        path_names: Sequence[str] | None = None,
        require_closed: bool = False,
    ):
        if not isinstance(topology, CellComplexTopology):
            raise TypeError("topology must be CellComplexTopology.")
        tails, heads = _edge_endpoints(topology)
        edges = np.asarray(edge_indices, dtype=np.int64)
        signs = np.asarray(orientations, dtype=np.int64)
        if edges.ndim != 2 or signs.shape != edges.shape or min(edges.shape) < 1:
            raise ValueError(
                "edge_indices and orientations must be non-empty rank-two arrays."
            )
        active = (
            np.ones(edges.shape, dtype=np.bool_)
            if valid is None
            else np.asarray(valid, dtype=np.bool_)
        )
        if active.shape != edges.shape:
            raise ValueError("valid must match the path array shape.")
        if np.any(active[:, 1:] & ~active[:, :-1]):
            raise ValueError("Every path validity mask must be a contiguous prefix.")
        if np.any(np.sum(active, axis=1) == 0):
            raise ValueError("Every path must contain at least one active edge.")
        if np.any(edges[active] < 0) or np.any(edges[active] >= tails.size):
            raise ValueError("Active path edges lie outside the topology edge set.")
        if np.any(np.abs(signs[active]) != 1):
            raise ValueError("Active path orientations must be plus or minus one.")
        names = (
            tuple(f"path_{index}" for index in range(edges.shape[0]))
            if path_names is None
            else tuple(str(value) for value in path_names)
        )
        if (
            len(names) != edges.shape[0]
            or any(not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("path_names must be distinct, non-empty, and path-aligned.")

        starts = np.zeros((edges.shape[0],), dtype=np.int32)
        ends = np.zeros((edges.shape[0],), dtype=np.int32)
        for path_index in range(edges.shape[0]):
            length = int(np.sum(active[path_index]))
            path_edges = edges[path_index, :length]
            path_signs = signs[path_index, :length]
            segment_starts = np.where(
                path_signs > 0, tails[path_edges], heads[path_edges]
            )
            segment_ends = np.where(path_signs > 0, heads[path_edges], tails[path_edges])
            if length > 1 and np.any(segment_ends[:-1] != segment_starts[1:]):
                raise ValueError("Consecutive oriented path edges must share a vertex.")
            starts[path_index] = segment_starts[0]
            ends[path_index] = segment_ends[-1]
            if require_closed and starts[path_index] != ends[path_index]:
                raise ValueError("A required closed path does not return to its start.")

        self.topology = topology
        self.edge_indices = jnp.asarray(edges, dtype=jnp.int32)
        self.orientations = jnp.asarray(signs, dtype=jnp.int32)
        self.valid = jnp.asarray(active)
        self.start_vertices = jnp.asarray(starts)
        self.end_vertices = jnp.asarray(ends)
        self.path_names = names
        self.require_closed = bool(require_closed)
        self.num_paths = edges.shape[0]
        self.max_length = edges.shape[1]
        self.topology_id = topology.topology_id
        self.edge_entity_set_id = topology.entities(1).entity_set_id
        self.path_plan_id = canonical_fingerprint(
            {
                "kind": "oriented-edge-path-plan",
                "topology": topology.topology_id,
                "edges": array_tree_fingerprint(edges),
                "orientations": array_tree_fingerprint(signs),
                "valid": array_tree_fingerprint(active),
                "names": list(names),
                "require_closed": bool(require_closed),
            }
        )


class CellBoundaryPathPlan(StrictModule, NonTrainableState):
    """Ordered closed paths certified against selected oriented two-cell boundaries."""

    paths: OrientedEdgePathPlan
    cell_indices: Array
    cell_entity_set_id: str = eqx.field(static=True)
    boundary_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        paths: OrientedEdgePathPlan,
        cell_indices: ArrayLike,
        /,
    ):
        if not isinstance(paths, OrientedEdgePathPlan):
            raise TypeError("paths must be OrientedEdgePathPlan.")
        if not paths.require_closed:
            raise ValueError("Cell boundary paths must require closed traversal.")
        topology = paths.topology
        if topology.dimension < 2:
            raise ValueError("Cell boundary paths require topology degree two.")
        cells = np.asarray(cell_indices, dtype=np.int64)
        if cells.shape != (paths.num_paths,):
            raise ValueError("cell_indices must provide one cell per path.")
        cell_count = topology.entities(2).count
        if np.any(cells < 0) or np.any(cells >= cell_count):
            raise ValueError("cell_indices lie outside the degree-two entity set.")
        if np.unique(cells).size != cells.size:
            raise ValueError("Cell boundary paths require distinct cell indices.")
        boundary = topology.incidences[1].scipy_boundary().toarray()
        path_edges = np.asarray(paths.edge_indices)
        path_signs = np.asarray(paths.orientations)
        path_valid = np.asarray(paths.valid)
        for path_index, cell in enumerate(cells):
            coefficients = np.zeros((topology.entities(1).count,), dtype=np.int64)
            np.add.at(
                coefficients,
                path_edges[path_index, path_valid[path_index]],
                path_signs[path_index, path_valid[path_index]],
            )
            if not np.array_equal(coefficients, boundary[:, cell]):
                raise ValueError(
                    "Ordered path coefficients do not reproduce the selected cell boundary."
                )
        self.paths = paths
        self.cell_indices = jnp.asarray(cells, dtype=jnp.int32)
        self.cell_entity_set_id = topology.entities(2).entity_set_id
        self.boundary_plan_id = canonical_fingerprint(
            {
                "kind": "cell-boundary-path-plan",
                "paths": paths.path_plan_id,
                "cells": array_tree_fingerprint(cells),
            }
        )


def prepare_cell_boundary_paths(
    topology: CellComplexTopology,
    /,
    *,
    cell_indices: ArrayLike | None = None,
) -> CellBoundaryPathPlan:
    """Recover deterministic ordered simple cycles from oriented two-cell incidence."""
    if not isinstance(topology, CellComplexTopology):
        raise TypeError("topology must be CellComplexTopology.")
    if topology.dimension < 2:
        raise ValueError("Cell boundary preparation requires degree-two topology.")
    tails, heads = _edge_endpoints(topology)
    cell_count = topology.entities(2).count
    cells = (
        np.arange(cell_count, dtype=np.int32)
        if cell_indices is None
        else np.asarray(cell_indices, dtype=np.int32)
    )
    if cells.ndim != 1 or cells.size == 0:
        raise ValueError("cell_indices must be a non-empty vector.")
    if np.any(cells < 0) or np.any(cells >= cell_count):
        raise ValueError("cell_indices lie outside the degree-two entity set.")
    if np.unique(cells).size != cells.size:
        raise ValueError("cell_indices must be distinct.")
    boundary = topology.incidences[1].scipy_boundary().toarray()
    ordered_edges: list[list[int]] = []
    ordered_signs: list[list[int]] = []
    for cell in cells:
        coefficients = boundary[:, cell]
        active_edges = np.flatnonzero(coefficients)
        if active_edges.size == 0 or np.any(np.abs(coefficients[active_edges]) != 1):
            raise ValueError(
                "Two-cell boundaries must be non-empty simple oriented cycles."
            )
        signs = coefficients[active_edges].astype("int64")
        starts = np.where(signs > 0, tails[active_edges], heads[active_edges])
        ends = np.where(signs > 0, heads[active_edges], tails[active_edges])
        first = int(np.lexsort((active_edges, starts))[0])
        remaining = set(range(active_edges.size))
        order = [first]
        remaining.remove(first)
        current = int(ends[first])
        while remaining:
            candidates = [index for index in remaining if int(starts[index]) == current]
            if len(candidates) != 1:
                raise ValueError("Two-cell boundary is not one simple directed cycle.")
            selected = candidates[0]
            order.append(selected)
            remaining.remove(selected)
            current = int(ends[selected])
        if current != int(starts[first]):
            raise ValueError("Two-cell boundary does not close.")
        ordered_edges.append([int(active_edges[index]) for index in order])
        ordered_signs.append([int(signs[index]) for index in order])
    capacity = max(len(value) for value in ordered_edges)
    edge_array = np.zeros((len(ordered_edges), capacity), dtype=np.int32)
    sign_array = np.zeros((len(ordered_edges), capacity), dtype=np.int32)
    valid_array = np.zeros((len(ordered_edges), capacity), dtype=np.bool_)
    for index, (edges, signs) in enumerate(
        zip(ordered_edges, ordered_signs, strict=True)
    ):
        length = len(edges)
        edge_array[index, :length] = edges
        sign_array[index, :length] = signs
        valid_array[index, :length] = True
    paths = OrientedEdgePathPlan(
        topology,
        edge_array,
        sign_array,
        valid=valid_array,
        path_names=tuple(f"cell_{int(cell)}" for cell in cells),
        require_closed=True,
    )
    return CellBoundaryPathPlan(paths, cells)


def reverse_oriented_paths(plan: OrientedEdgePathPlan, /) -> OrientedEdgePathPlan:
    """Return every traversal in reverse order and orientation."""
    if not isinstance(plan, OrientedEdgePathPlan):
        raise TypeError("plan must be OrientedEdgePathPlan.")
    edges = np.asarray(plan.edge_indices)
    signs = np.asarray(plan.orientations)
    valid = np.asarray(plan.valid)
    reversed_edges = np.zeros_like(edges)
    reversed_signs = np.zeros_like(signs)
    for index in range(plan.num_paths):
        length = int(np.sum(valid[index]))
        reversed_edges[index, :length] = edges[index, :length][::-1]
        reversed_signs[index, :length] = -signs[index, :length][::-1]
    return OrientedEdgePathPlan(
        plan.topology,
        reversed_edges,
        reversed_signs,
        valid=valid,
        path_names=tuple(f"{name}:reverse" for name in plan.path_names),
        require_closed=plan.require_closed,
    )


__all__ = [
    "CellBoundaryPathPlan",
    "OrientedEdgePathPlan",
    "oriented_edge_endpoints",
    "prepare_cell_boundary_paths",
    "reverse_oriented_paths",
]
