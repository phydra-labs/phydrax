#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cell-complex and per-sheet manifold views of multiregion surfaces."""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization._topology import CellComplexTopology, EntitySet, OrientedIncidence
from ...sparse import EdgeRelation
from ...typing import Dim, Identifier, Int64
from ..simplicial._mesh import TriangleMesh
from ._geometry import PreparedMultiRegionSurface
from ._state import MultiRegionSurfaceState
from ._topology import MultiRegionSurfaceTopology


class _SheetVertexDim(Dim, minimum=3):
    """Vertices of one sheet view."""


class _SheetFaceDim(Dim, minimum=1):
    """Faces of one sheet view."""


def multiregion_cell_complex(topology: MultiRegionSurfaceTopology, /) -> CellComplexTopology:
    """Signed ``vertex -> edge -> face -> finite region`` cell complex.

    ``d(edge) = v1 - v0``, ``d(face) = sum s_fe edge`` with the face traversal
    signs, and ``d(region) = sum_f (+1 if left, -1 if right) face`` over finite
    regions only; boundary labels (the ambient) are not 3-cells. Entity ids are
    the stable vertex/face global ids, edge slots and finite-region slots. The
    boundary-of-boundary chain check of `CellComplexTopology` certifies that
    every finite region is a closed oriented 2-cycle.
    """
    if not isinstance(topology, MultiRegionSurfaceTopology):
        raise TypeError("topology must be a MultiRegionSurfaceTopology.")
    edges = np.asarray(topology.edges[: topology.edge_count], dtype=np.int64)
    face_edges = np.asarray(topology.face_edges[: topology.face_count], dtype=np.int64)
    face_signs = np.asarray(
        topology.face_edge_signs[: topology.face_count], dtype=np.float64
    )
    labels = topology.host_face_labels()
    vertices = EntitySet(
        "multiregion-vertices",
        0,
        np.asarray(topology.vertex_global_ids[: topology.vertex_count]),
    )
    edge_set = EntitySet("multiregion-edges", 1, np.arange(topology.edge_count))
    faces = EntitySet(
        "multiregion-faces", 2, np.asarray(topology.face_global_ids[: topology.face_count])
    )
    vertex_edge = OrientedIncidence(
        1,
        vertices,
        edge_set,
        EdgeRelation(
            edges.reshape(-1),
            np.repeat(np.arange(topology.edge_count), 2),
            source_size=topology.vertex_count,
            target_size=topology.edge_count,
        ),
        np.tile(np.asarray((-1.0, 1.0)), topology.edge_count),
    )
    edge_face = OrientedIncidence(
        2,
        edge_set,
        faces,
        EdgeRelation(
            face_edges.reshape(-1),
            np.repeat(np.arange(topology.face_count), 3),
            source_size=topology.edge_count,
            target_size=topology.face_count,
        ),
        face_signs.reshape(-1),
    )
    finite = topology.finite_region_indices
    if not finite:
        return CellComplexTopology(
            (vertices, edge_set, faces),
            (vertex_edge, edge_face),
            topology_id=f"{topology.topology_id}/cell-complex",
        )
    local = np.full((topology.region_count,), -1, dtype=np.int64)
    local[list(finite)] = np.arange(len(finite))
    rows = np.repeat(np.arange(topology.face_count), 2)
    regions = local[labels.reshape(-1)]
    signs = np.tile(np.asarray((1.0, -1.0)), topology.face_count)
    keep = regions >= 0
    region_set = EntitySet("multiregion-finite-regions", 3, np.asarray(finite))
    face_region = OrientedIncidence(
        3,
        faces,
        region_set,
        EdgeRelation(
            rows[keep],
            regions[keep],
            source_size=topology.face_count,
            target_size=len(finite),
        ),
        signs[keep],
    )
    return CellComplexTopology(
        (vertices, edge_set, faces, region_set),
        (vertex_edge, edge_face, face_region),
        topology_id=f"{topology.topology_id}/cell-complex",
    )


@final
class MultiRegionSheetView(StrictModule):
    """One region-pair sheet as an independent manifold triangle mesh.

    ``mesh`` holds the sheet's vertices (ascending global vertex slot) and faces
    oriented so that normals point out of ``region_ids[0]``, the first label of
    the canonical pair. ``vertex_indices`` map mesh vertices to surface vertex
    slots, ``slot_indices`` to flattened ``vertex * slot_width + slot`` sheet
    slots (the storage of sheet fields) and ``face_indices`` to surface face
    slots. Borders (wires and Plateau borders) are mesh boundary loops.
    """

    __strict_contract__ = True

    pair_index: int = eqx.field(static=True)
    region_ids: tuple[str, str] = eqx.field(static=True)
    vertex_indices: Int64[_SheetVertexDim]
    slot_indices: Int64[_SheetVertexDim]
    face_indices: Int64[_SheetFaceDim]
    mesh: TriangleMesh
    view_id: Identifier = eqx.field(static=True)


@final
class MultiRegionSheetViews(StrictModule):
    """Manifold sheet views plus the region pairs that are not manifold."""

    views: tuple[MultiRegionSheetView, ...]
    nonmanifold_pair_indices: tuple[int, ...] = eqx.field(static=True)
    views_id: str = eqx.field(static=True)


def _sheet_is_manifold(faces: np.ndarray, vertex_count: int, /) -> bool:
    """Edge-manifold, consistently oriented, regular borders and single vertex fans."""
    origin = faces.reshape(-1)
    destination = np.roll(faces, -1, axis=1).reshape(-1)
    keys = np.stack((np.minimum(origin, destination), np.maximum(origin, destination)), 1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    if np.any(counts > 2):
        return False
    direction = np.where(origin < destination, 1, -1)
    signed = np.zeros((counts.size,), dtype=np.int64)
    np.add.at(signed, inverse, direction)
    if np.any((counts == 2) & (signed != 0)):
        return False
    border = counts[inverse] == 1
    outgoing = np.bincount(origin[border], minlength=vertex_count)
    incoming = np.bincount(destination[border], minlength=vertex_count)
    if np.any(outgoing > 1) or np.any(outgoing != incoming):
        return False
    # Corners (face, vertex) are joined across every interior edge at both of the
    # edge's vertices; a manifold vertex owns exactly one connected fan.
    half = np.arange(origin.size)
    corner_origin = half
    corner_destination = (half // 3) * 3 + (half % 3 + 1) % 3
    interior = np.flatnonzero(counts[inverse] == 2)
    order = interior[np.argsort(inverse[interior], kind="stable")]
    first, second = order[0::2], order[1::2]
    rows = np.concatenate(
        (corner_origin[first], corner_destination[first])
    )
    cols = np.concatenate(
        (corner_destination[second], corner_origin[second])
    )
    graph = sp.coo_matrix(
        (np.ones(rows.size), (rows, cols)), shape=(origin.size, origin.size)
    )
    _, component = connected_components(graph, directed=False)
    fans = np.unique(np.stack((origin, component), axis=1), axis=0)
    return bool(np.all(np.bincount(fans[:, 0], minlength=vertex_count)[np.unique(origin)] == 1))


def multiregion_sheet_views(
    prepared: PreparedMultiRegionSurface, state: MultiRegionSurfaceState, /
) -> MultiRegionSheetViews:
    """Per-region-pair manifold views of the current geometry (host boundary)."""
    if not isinstance(prepared, PreparedMultiRegionSurface):
        raise TypeError("prepared must be a PreparedMultiRegionSurface.")
    topology = prepared.topology
    state.require_topology(topology)
    faces = topology.host_faces()
    face_pairs = np.asarray(topology.face_pairs[: topology.face_count], dtype=np.int64)
    pair_signs = np.asarray(
        topology.face_pair_signs[: topology.face_count], dtype=np.int64
    )
    pairs = np.asarray(topology.region_pairs[: topology.region_pair_count], dtype=np.int64)
    slot_table = np.asarray(topology.vertex_pair_slots, dtype=np.int64)
    points = np.asarray(state.positions, dtype=np.float64)
    views = []
    nonmanifold = []
    for pair in range(topology.region_pair_count):
        selected = np.flatnonzero(face_pairs == pair)
        oriented = faces[selected].copy()
        flip = pair_signs[selected] < 0
        oriented[flip] = oriented[flip][:, (0, 2, 1)]
        vertex_indices, local = np.unique(oriented, return_inverse=True)
        local_faces = local.reshape((-1, 3))
        if not _sheet_is_manifold(local_faces, vertex_indices.size):
            nonmanifold.append(pair)
            continue
        slots = np.argmax(slot_table[vertex_indices] == pair, axis=1)
        region_ids = (
            topology.region_ids[int(pairs[pair, 0])],
            topology.region_ids[int(pairs[pair, 1])],
        )
        view_id = canonical_fingerprint(
            {
                "kind": "multiregion-sheet-view",
                "topology": topology.topology_id,
                "pair": region_ids,
                "faces": array_tree_fingerprint(selected),
            }
        )
        views.append(
            MultiRegionSheetView(
                pair_index=pair,
                region_ids=region_ids,
                vertex_indices=jnp.asarray(vertex_indices, dtype=jnp.int64),
                slot_indices=jnp.asarray(
                    vertex_indices * topology.slot_width + slots, dtype=jnp.int64
                ),
                face_indices=jnp.asarray(selected, dtype=jnp.int64),
                mesh=TriangleMesh(points[vertex_indices], local_faces, source_id=view_id),
                view_id=view_id,
            )
        )
    return MultiRegionSheetViews(
        views=tuple(views),
        nonmanifold_pair_indices=tuple(nonmanifold),
        views_id=canonical_fingerprint(
            {
                "kind": "multiregion-sheet-views",
                "prepared": prepared.prepared_id,
                "views": [view.view_id for view in views],
                "nonmanifold": nonmanifold,
            }
        ),
    )


__all__ = [
    "MultiRegionSheetView",
    "MultiRegionSheetViews",
    "multiregion_cell_complex",
    "multiregion_sheet_views",
]
