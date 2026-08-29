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


class FiniteElementAdaptationMap(StrictModule, NonTrainableState):
    """Bidirectional active-cell lineage for one local triangle update."""

    source_mesh: CellMesh
    target_mesh: CellMesh
    parent_cell_ids: Array
    child_cell_ids: Array
    child_valid: Array
    midpoint_vertex_ids: Array
    midpoint_parent_vertex_ids: Array
    adaptation_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_mesh: CellMesh,
        target_mesh: CellMesh,
        parent_cell_ids: ArrayLike,
        child_cell_ids: ArrayLike,
        child_valid: ArrayLike,
        midpoint_vertex_ids: ArrayLike,
        midpoint_parent_vertex_ids: ArrayLike,
        /,
    ):
        parents = np.asarray(parent_cell_ids, dtype=np.int64)
        children = np.asarray(child_cell_ids, dtype=np.int64)
        valid = np.asarray(child_valid, dtype=bool)
        midpoint_ids = np.asarray(midpoint_vertex_ids, dtype=np.int64)
        midpoint_parents = np.asarray(midpoint_parent_vertex_ids, dtype=np.int64)
        if not isinstance(source_mesh, CellMesh) or not isinstance(target_mesh, CellMesh):
            raise TypeError("Adaptation lineage requires source and target CellMesh.")
        if (
            parents.ndim != 1
            or children.ndim != 2
            or valid.shape != children.shape
            or children.shape[0] != parents.size
        ):
            raise ValueError("Adaptation parent/child arrays are incompatible.")
        if midpoint_ids.ndim != 1 or midpoint_parents.shape != (
            midpoint_ids.size,
            2,
        ):
            raise ValueError("Adaptation midpoint lineage is incompatible.")
        self.source_mesh = source_mesh
        self.target_mesh = target_mesh
        self.parent_cell_ids = jnp.asarray(parents)
        self.child_cell_ids = jnp.asarray(children)
        self.child_valid = jnp.asarray(valid)
        self.midpoint_vertex_ids = jnp.asarray(midpoint_ids)
        self.midpoint_parent_vertex_ids = jnp.asarray(midpoint_parents)
        self.adaptation_id = canonical_fingerprint(
            {
                "kind": "finite-element-local-adaptation",
                "source": source_mesh.topology_id,
                "target": target_mesh.topology_id,
                "parents": array_tree_fingerprint(parents),
                "children": array_tree_fingerprint(children),
                "valid": array_tree_fingerprint(valid),
                "midpoints": array_tree_fingerprint(midpoint_ids),
                "midpoint_parents": array_tree_fingerprint(midpoint_parents),
            }
        )


class FiniteElementTransferBundle(StrictModule, NonTrainableState):
    """Primal transfer and its raw-dual/pairing adjoint maps."""

    primal: Array
    dual_pullback: Array
    adjoint: Array
    adaptation_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(self, primal: ArrayLike, adaptation_id: str, /):
        primal_ = jnp.asarray(primal)
        if primal_.ndim != 2 or not jnp.issubdtype(primal_.dtype, jnp.inexact):
            raise ValueError("Primal adaptation transfer must be one inexact matrix.")
        adaptation = str(adaptation_id)
        if not adaptation:
            raise ValueError("adaptation_id must be non-empty.")
        self.primal = primal_
        self.dual_pullback = primal_.T
        self.adjoint = primal_.T
        self.adaptation_id = adaptation
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "finite-element-transfer-bundle",
                "adaptation": adaptation,
                "shape": list(primal_.shape),
            }
        )


def dorfler_mark(
    indicators: ArrayLike,
    theta: float,
    /,
    *,
    cell_global_ids: ArrayLike | None = None,
) -> Array:
    values = np.asarray(indicators, dtype=float)
    ids = (
        np.arange(values.size, dtype=np.int64)
        if cell_global_ids is None
        else np.asarray(cell_global_ids, dtype=np.int64)
    )
    fraction = float(theta)
    if (
        values.ndim != 1
        or ids.shape != values.shape
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or not 0.0 < fraction <= 1.0
    ):
        raise ValueError("Dörfler indicators, IDs, or theta are invalid.")
    squared = values**2
    total = float(np.sum(squared))
    if total == 0.0:
        return jnp.asarray(np.empty((0,), dtype=np.int64))
    order = np.lexsort((ids, -squared))
    count = int(
        np.searchsorted(
            np.cumsum(squared[order]),
            fraction * total,
            side="left",
        )
        + 1
    )
    return jnp.asarray(np.sort(ids[order[:count]]))


def maximum_mark(
    indicators: ArrayLike,
    fraction: float,
    /,
    *,
    cell_global_ids: ArrayLike | None = None,
) -> Array:
    values = np.asarray(indicators, dtype=float)
    ids = (
        np.arange(values.size, dtype=np.int64)
        if cell_global_ids is None
        else np.asarray(cell_global_ids, dtype=np.int64)
    )
    fraction_ = float(fraction)
    if (
        values.ndim != 1
        or ids.shape != values.shape
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or not 0.0 < fraction_ <= 1.0
    ):
        raise ValueError("Maximum indicators, IDs, or fraction are invalid.")
    count = max(1, int(np.ceil(fraction_ * values.size)))
    order = np.lexsort((ids, -values))
    return jnp.asarray(np.sort(ids[order[:count]]))


def _triangle_children(cell, marked_local_edges, midpoint_by_local):
    a, b, c = (int(value) for value in cell)
    marked = tuple(sorted(marked_local_edges))
    if marked == (0,):
        m = midpoint_by_local[0]
        return ((a, m, c), (m, b, c))
    if marked == (1,):
        m = midpoint_by_local[1]
        return ((b, m, a), (m, c, a))
    if marked == (2,):
        m = midpoint_by_local[2]
        return ((c, m, b), (m, a, b))
    if marked == (0, 1):
        m01, m12 = midpoint_by_local[0], midpoint_by_local[1]
        return ((a, m01, c), (m01, m12, c), (m01, b, m12))
    if marked == (1, 2):
        m12, m20 = midpoint_by_local[1], midpoint_by_local[2]
        return ((b, m12, a), (m12, m20, a), (m12, c, m20))
    if marked == (0, 2):
        m01, m20 = midpoint_by_local[0], midpoint_by_local[2]
        return ((c, m20, b), (m20, m01, b), (m20, a, m01))
    if marked == (0, 1, 2):
        m01, m12, m20 = (
            midpoint_by_local[0],
            midpoint_by_local[1],
            midpoint_by_local[2],
        )
        return (
            (a, m01, m20),
            (m01, b, m12),
            (m20, m12, c),
            (m01, m12, m20),
        )
    return (tuple(cell),)


def refine_triangles_local(
    mesh: CellMesh,
    marked_cell_ids: ArrayLike,
    /,
    *,
    numeric_version: str = "locally-refined",
) -> tuple[CellMesh, FiniteElementAdaptationMap, FiniteElementTransferBundle]:
    if (
        not isinstance(mesh, CellMesh)
        or len(mesh.blocks) != 1
        or mesh.blocks[0].cell_kind != "triangle"
    ):
        raise ValueError("Local refinement currently requires one T3 block.")
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise TypeError("Local T3 refinement requires polygonal connectivity.")
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int32)
    cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    coordinates = np.asarray(mesh.coordinates)
    marked_ids = np.asarray(marked_cell_ids, dtype=np.int64)
    if marked_ids.ndim != 1 or np.unique(marked_ids).size != marked_ids.size:
        raise ValueError("Marked cell IDs must be one unique rank-1 array.")
    unknown = set(marked_ids.tolist()) - set(cell_ids.tolist())
    if unknown:
        raise ValueError(f"Unknown marked cell IDs {sorted(unknown)!r}.")
    cell_by_id = {int(value): index for index, value in enumerate(cell_ids)}
    edges = np.asarray(connectivity.edges, dtype=np.int32)
    cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)[:, :3]
    marked_edges = set()
    for cell_id in sorted(marked_ids.tolist()):
        cell = cell_by_id[int(cell_id)]
        candidates = []
        for edge in cell_edges[cell]:
            vertices = edges[int(edge)]
            delta = coordinates[vertices[1]] - coordinates[vertices[0]]
            length = float(np.dot(delta, delta))
            key = tuple(sorted(vertex_ids[vertices].tolist()))
            candidates.append((-length, key, int(edge)))
        candidates.sort()
        marked_edges.add(candidates[0][2])
    sorted_edges = sorted(
        marked_edges,
        key=lambda edge: tuple(sorted(vertex_ids[edges[edge]].tolist())),
    )
    midpoint_local = {}
    midpoint_ids = []
    midpoint_parent_ids = []
    new_coordinates = [value for value in coordinates]
    new_vertex_ids = [int(value) for value in vertex_ids]
    next_vertex_id = int(np.max(vertex_ids)) + 1
    for edge in sorted_edges:
        vertices = edges[edge]
        midpoint_local[edge] = len(new_coordinates)
        new_coordinates.append(
            0.5 * (coordinates[vertices[0]] + coordinates[vertices[1]])
        )
        midpoint_ids.append(next_vertex_id)
        midpoint_parent_ids.append(vertex_ids[vertices])
        new_vertex_ids.append(next_vertex_id)
        next_vertex_id += 1
    children = []
    child_global_ids = []
    parent_ids = []
    family_ids = []
    family_valid = []
    next_cell_id = int(np.max(cell_ids)) + 1
    for cell_index, cell in enumerate(cells):
        local_marked = tuple(
            local
            for local, edge in enumerate(cell_edges[cell_index])
            if int(edge) in marked_edges
        )
        midpoint_by_local = {
            local: midpoint_local[int(cell_edges[cell_index, local])]
            for local in local_marked
        }
        local_children = _triangle_children(cell, local_marked, midpoint_by_local)
        if not local_marked:
            children.append(tuple(int(value) for value in cell))
            child_global_ids.append(int(cell_ids[cell_index]))
            continue
        ids = []
        for child in local_children:
            children.append(child)
            child_global_ids.append(next_cell_id)
            ids.append(next_cell_id)
            next_cell_id += 1
        parent_ids.append(int(cell_ids[cell_index]))
        padded = ids + [-1] * (4 - len(ids))
        family_ids.append(padded)
        family_valid.append([True] * len(ids) + [False] * (4 - len(ids)))
    refined = CellMesh(
        np.asarray(new_coordinates),
        (
            CellBlock(
                mesh.blocks[0].name,
                "triangle",
                np.asarray(children, dtype=np.int32),
                global_ids=np.asarray(child_global_ids, dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray(new_vertex_ids, dtype=np.int64),
        numeric_version=numeric_version,
    )
    adaptation = FiniteElementAdaptationMap(
        mesh,
        refined,
        np.asarray(parent_ids, dtype=np.int64),
        np.asarray(family_ids, dtype=np.int64).reshape((-1, 4)),
        np.asarray(family_valid, dtype=bool).reshape((-1, 4)),
        np.asarray(midpoint_ids, dtype=np.int64),
        np.asarray(midpoint_parent_ids, dtype=np.int64).reshape((-1, 2)),
    )
    primal = np.zeros(
        (refined.coordinates.shape[0], mesh.coordinates.shape[0]),
        dtype=coordinates.dtype,
    )
    source_vertex_by_id = {int(value): index for index, value in enumerate(vertex_ids)}
    for target, vertex_id in enumerate(new_vertex_ids):
        if vertex_id in source_vertex_by_id:
            primal[target, source_vertex_by_id[vertex_id]] = 1.0
    for midpoint_id, parents in zip(midpoint_ids, midpoint_parent_ids, strict=True):
        target = new_vertex_ids.index(midpoint_id)
        primal[target, source_vertex_by_id[int(parents[0])]] = 0.5
        primal[target, source_vertex_by_id[int(parents[1])]] = 0.5
    return (
        refined,
        adaptation,
        FiniteElementTransferBundle(primal, adaptation.adaptation_id),
    )


def coarsen_triangles_local(
    mesh: CellMesh,
    adaptation: FiniteElementAdaptationMap,
    marked_child_ids: ArrayLike,
    /,
    *,
    numeric_version: str = "locally-coarsened",
) -> CellMesh:
    if (
        not isinstance(adaptation, FiniteElementAdaptationMap)
        or mesh.topology_id != adaptation.target_mesh.topology_id
    ):
        raise ValueError("Coarsening requires the exact target adaptation mesh.")
    marked = set(np.asarray(marked_child_ids, dtype=np.int64).tolist())
    selected_parents = []
    for parent, children, valid in zip(
        np.asarray(adaptation.parent_cell_ids),
        np.asarray(adaptation.child_cell_ids),
        np.asarray(adaptation.child_valid),
        strict=True,
    ):
        family = set(children[valid].tolist())
        overlap = family & marked
        if overlap and overlap != family:
            raise ValueError("Coarsening requires a complete active sibling family.")
        if family and overlap == family:
            selected_parents.append(int(parent))
    if not selected_parents:
        raise ValueError("No complete sibling family was selected for coarsening.")
    if set(selected_parents) == set(np.asarray(adaptation.parent_cell_ids).tolist()):
        return CellMesh(
            adaptation.source_mesh.coordinates,
            adaptation.source_mesh.blocks,
            vertex_global_ids=adaptation.source_mesh.vertex_global_ids,
            numeric_version=numeric_version,
        )
    raise ValueError(
        "Partial coarsening is rejected until neighbour-balance closure selects all "
        "incident sibling families."
    )


class FiniteElementDWRIndicators(StrictModule):
    signed: Array
    absolute: Array
    global_estimate: Array

    def __init__(self, signed: ArrayLike, /):
        signed_ = jnp.asarray(signed)
        if signed_.ndim != 1:
            raise ValueError("DWR signed indicators must be rank-1.")
        self.signed = signed_
        self.absolute = jnp.abs(signed_)
        self.global_estimate = jnp.sum(signed_)


def local_dual_weighted_residual(
    cell_residual: ArrayLike,
    dual_correction: ArrayLike,
    /,
) -> FiniteElementDWRIndicators:
    residual = jnp.asarray(cell_residual)
    correction = jnp.asarray(dual_correction)
    if residual.shape != correction.shape or residual.ndim < 2:
        raise ValueError(
            "Local DWR residual/correction arrays must share cell-leading shape."
        )
    axes = tuple(range(1, residual.ndim))
    signed = jnp.sum(jnp.real(jnp.conj(correction) * residual), axis=axes)
    return FiniteElementDWRIndicators(signed)


__all__ = [
    "FiniteElementAdaptationMap",
    "FiniteElementErrorEstimate",
    "FiniteElementDWRIndicators",
    "FiniteElementRefinementMap",
    "FiniteElementTransferBundle",
    "FiniteElementTransferPlan",
    "FiniteElementTransferRole",
    "coarsen_triangles_local",
    "dorfler_mark",
    "dual_weighted_residual_estimate",
    "local_dual_weighted_residual",
    "maximum_mark",
    "refine_triangles_local",
    "refine_triangles_uniform",
    "residual_jump_estimate",
]
