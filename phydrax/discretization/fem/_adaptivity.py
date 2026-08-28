#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cell_complex import PolygonalConnectivity
from .._cell_mesh import CellBlock, CellMesh
from .._transfer import FieldTransfer


FiniteElementTransferRole = Literal[
    "primal",
    "dual-residual",
    "coefficient",
    "material-state",
    "adjoint",
]


class FiniteElementRefinementMap(StrictModule, NonTrainableState):
    """Versioned parent/child and old/new entity lineage for one mesh update."""

    parent_cells: Array
    child_cells: Array
    old_vertex_to_new: Array
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    refinement_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_topology_id: str,
        target_topology_id: str,
        parent_cells: ArrayLike,
        child_cells: ArrayLike,
        old_vertex_to_new: ArrayLike,
        /,
    ):
        source = str(source_topology_id)
        target = str(target_topology_id)
        parents = np.asarray(parent_cells, dtype=np.int32)
        children = np.asarray(child_cells, dtype=np.int32)
        vertex_map = np.asarray(old_vertex_to_new, dtype=np.int32)
        if not source or not target or source == target:
            raise ValueError("Refinement topology IDs must be distinct and non-empty.")
        if parents.ndim != 1 or children.ndim != 2 or children.shape[0] != parents.size:
            raise ValueError("Refinement parent/child routes have incompatible shapes.")
        if vertex_map.ndim != 1 or np.any(vertex_map < 0):
            raise ValueError("old_vertex_to_new must be one non-negative rank-1 map.")
        self.parent_cells = jnp.asarray(parents)
        self.child_cells = jnp.asarray(children)
        self.old_vertex_to_new = jnp.asarray(vertex_map)
        self.source_topology_id = source
        self.target_topology_id = target
        self.refinement_id = canonical_fingerprint(
            {
                "kind": "finite-element-refinement-map",
                "source": source,
                "target": target,
                "parents": array_tree_fingerprint(parents),
                "children": array_tree_fingerprint(children),
                "vertices": array_tree_fingerprint(vertex_map),
            }
        )


class FiniteElementTransferPlan(StrictModule, NonTrainableState):
    """One existing FieldTransfer assigned an explicit FE scientific role."""

    transfer: FieldTransfer
    role: FiniteElementTransferRole = eqx.field(static=True)
    refinement_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: FieldTransfer,
        role: FiniteElementTransferRole,
        refinement_id: str,
        /,
    ):
        if not isinstance(transfer, FieldTransfer):
            raise TypeError("transfer must be FieldTransfer.")
        if role not in (
            "primal",
            "dual-residual",
            "coefficient",
            "material-state",
            "adjoint",
        ):
            raise ValueError("Unknown finite-element transfer role.")
        refinement = str(refinement_id)
        if not refinement:
            raise ValueError("refinement_id must be non-empty.")
        self.transfer = transfer
        self.role = role
        self.refinement_id = refinement
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-transfer-plan",
                "transfer": transfer.transfer_id,
                "role": role,
                "refinement": refinement,
            }
        )


class FiniteElementErrorEstimate(StrictModule):
    """Cell-local residual/jump or goal-oriented error evidence."""

    cell_indicators: Array
    global_estimate: Array
    estimator_id: str = eqx.field(static=True)

    def __init__(self, cell_indicators: ArrayLike, estimator_id: str, /):
        indicators = jnp.asarray(cell_indicators)
        if indicators.ndim != 1:
            raise ValueError("cell_indicators must be rank-1.")
        identifier = str(estimator_id)
        if not identifier:
            raise ValueError("estimator_id must be non-empty.")
        self.cell_indicators = indicators
        self.global_estimate = jnp.sqrt(jnp.sum(indicators**2))
        self.estimator_id = identifier


def residual_jump_estimate(
    cell_residual: ArrayLike,
    cell_measure: ArrayLike,
    facet_jump: ArrayLike,
    facet_measure: ArrayLike,
    facet_owner: ArrayLike,
    facet_neighbour: ArrayLike,
    /,
) -> FiniteElementErrorEstimate:
    residual = jnp.asarray(cell_residual)
    cells = jnp.asarray(cell_measure)
    jumps = jnp.asarray(facet_jump)
    facets = jnp.asarray(facet_measure)
    owner = jnp.asarray(facet_owner, dtype=jnp.int32)
    neighbour = jnp.asarray(facet_neighbour, dtype=jnp.int32)
    if residual.shape != cells.shape or jumps.shape != facets.shape:
        raise ValueError("Residual/jump values must match their measures.")
    indicators = cells * residual**2
    contributions = facets * jumps**2
    indicators = indicators.at[owner].add(0.5 * contributions)
    active_neighbour = neighbour >= 0
    safe_neighbour = jnp.where(active_neighbour, neighbour, 0)
    indicators = indicators.at[safe_neighbour].add(
        jnp.where(active_neighbour, 0.5 * contributions, 0.0)
    )
    return FiniteElementErrorEstimate(
        jnp.sqrt(jnp.maximum(indicators, 0.0)),
        "residual-jump",
    )


def dual_weighted_residual_estimate(
    residual: ArrayLike,
    dual_correction: ArrayLike,
    /,
) -> Array:
    residual_ = jnp.asarray(residual)
    correction = jnp.asarray(dual_correction)
    if residual_.shape != correction.shape:
        raise ValueError("Residual and dual correction shapes must match.")
    return jnp.real(jnp.vdot(correction.reshape((-1,)), residual_.reshape((-1,))))


def refine_triangles_uniform(
    mesh: CellMesh,
    /,
    *,
    numeric_version: str = "refined",
) -> tuple[CellMesh, FiniteElementRefinementMap]:
    """Uniformly split every T3 into four conforming children."""

    if not isinstance(mesh, CellMesh) or mesh.topological_dimension != 2:
        raise TypeError("Uniform T3 refinement requires a two-dimensional CellMesh.")
    if len(mesh.blocks) != 1 or mesh.blocks[0].cell_kind != "triangle":
        raise ValueError("Uniform T3 refinement currently requires one triangle block.")
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("Uniform T3 refinement requires polygonal connectivity.")
    vertices = np.asarray(mesh.coordinates)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int32)
    edges = np.asarray(connectivity.edges, dtype=np.int32)
    midpoints = 0.5 * (vertices[edges[:, 0]] + vertices[edges[:, 1]])
    midpoint_ids = np.arange(
        vertices.shape[0],
        vertices.shape[0] + edges.shape[0],
        dtype=np.int32,
    )
    cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)[:, :3]
    children = []
    child_map = np.empty((cells.shape[0], 4), dtype=np.int32)
    for cell_index, (cell, local_edges) in enumerate(zip(cells, cell_edges, strict=True)):
        v0, v1, v2 = (int(value) for value in cell)
        m01, m12, m20 = (int(midpoint_ids[int(edge)]) for edge in local_edges)
        local_children = (
            (v0, m01, m20),
            (m01, v1, m12),
            (m20, m12, v2),
            (m01, m12, m20),
        )
        offset = len(children)
        children.extend(local_children)
        child_map[cell_index] = np.arange(offset, offset + 4, dtype=np.int32)
    coordinates = np.concatenate((vertices, midpoints), axis=0)
    vertex_ids = np.concatenate(
        (
            np.asarray(mesh.vertex_global_ids, dtype=np.int64),
            np.arange(
                int(np.max(np.asarray(mesh.vertex_global_ids))) + 1,
                int(np.max(np.asarray(mesh.vertex_global_ids))) + 1 + edges.shape[0],
                dtype=np.int64,
            ),
        )
    )
    parent_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    child_ids = (parent_ids[:, None] * 4 + np.arange(4, dtype=np.int64)[None, :]).reshape(
        (-1,)
    )
    refined = CellMesh(
        coordinates,
        (
            CellBlock(
                mesh.blocks[0].name,
                "triangle",
                np.asarray(children, dtype=np.int32),
                global_ids=child_ids,
            ),
        ),
        vertex_global_ids=vertex_ids,
        numeric_version=numeric_version,
    )
    refinement = FiniteElementRefinementMap(
        mesh.topology_id,
        refined.topology_id,
        np.arange(cells.shape[0], dtype=np.int32),
        child_map,
        np.arange(vertices.shape[0], dtype=np.int32),
    )
    return refined, refinement


def adaptive_uniform_solve(
    mesh: CellMesh,
    solve,
    estimate,
    /,
    *,
    max_levels: int,
    target_error: float,
):
    """Execute solve/estimate/uniform-refine until target or bounded levels."""

    if not callable(solve) or not callable(estimate):
        raise TypeError("Adaptive solve and estimate callbacks must be callable.")
    levels = int(max_levels)
    target = float(target_error)
    if levels <= 0 or target <= 0.0:
        raise ValueError("Adaptive level bound and target error must be positive.")
    current = mesh
    records = []
    for level in range(levels):
        solution = solve(current)
        error = jnp.asarray(estimate(current, solution))
        if error.shape != ():
            raise ValueError("Adaptive estimator callback must return one scalar.")
        records.append((current, solution, error))
        if float(error) <= target or level + 1 == levels:
            break
        current, _ = refine_triangles_uniform(
            current,
            numeric_version=f"adaptive-level-{level + 1}",
        )
    return current, tuple(records)


__all__ = [
    "FiniteElementErrorEstimate",
    "FiniteElementRefinementMap",
    "FiniteElementTransferPlan",
    "FiniteElementTransferRole",
    "dual_weighted_residual_estimate",
    "refine_triangles_uniform",
    "residual_jump_estimate",
    "adaptive_uniform_solve",
]
