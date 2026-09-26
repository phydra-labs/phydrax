from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation
from ._topology import (
    _has_duplicate_rows,
    _row_order,
    _row_run_starts,
    CellComplexTopology,
    EntitySet,
    EntitySubset,
    OrientedIncidence,
)


_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
_FACES = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


class HexahedralConnectivity(StrictModule, NonTrainableState):
    edges: Array
    faces: Array
    face_edges: Array
    face_edge_signs: Array
    cell_edges: Array
    cell_edge_signs: Array
    cell_faces: Array
    cell_face_signs: Array
    cell_face_vertex_permutations: Array
    face_cell_counts: Array
    boundary_vertices: Array
    boundary_edges: Array
    boundary_faces: Array
    vertex_count: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)

    def cell_edge_permutations(self, width: int, /) -> Array:
        """Map local edge positions to canonical edge positions."""

        width_ = int(width)
        if width_ <= 0:
            raise ValueError("Edge permutation width must be positive.")
        forward = np.arange(width_, dtype=np.int32)
        signs = np.asarray(self.cell_edge_signs)
        permutations = np.where(
            signs[..., None] > 0.0,
            forward,
            forward[::-1],
        )
        return jnp.asarray(permutations)

    def cell_face_permutations(self, width_u: int, width_v: int, /) -> Array:
        """Map local tensor-face positions to canonical flattened positions."""

        permutations = np.asarray(self.cell_face_vertex_permutations)
        vertex_routes = permutations.reshape((-1, 4))
        # Valid face permutations are the eight rotations and reflections of a
        # square, so every face gathers one of eight tensor routes.
        steps = np.where((vertex_routes[:, 1] - vertex_routes[:, 0]) % 4 == 3, -1, 1)
        valid = np.all(
            vertex_routes == (vertex_routes[:, :1] + steps[:, None] * np.arange(4)) % 4,
            axis=1,
        )
        # Validation runs in the order a per-face scan checks permutations and
        # widths; an invalid permutation raises its own validation error.
        if not valid[0]:
            _quadrilateral_tensor_permutation(vertex_routes[0], width_u, width_v)
        tensor_routes = np.stack(
            [
                _quadrilateral_tensor_permutation(
                    (start + step * np.arange(4)) % 4, width_u, width_v
                )
                for step in (1, -1)
                for start in range(4)
            ]
        )
        if not np.all(valid):
            _quadrilateral_tensor_permutation(
                vertex_routes[np.argmin(valid)], width_u, width_v
            )
        symmetry = vertex_routes[:, 0] + 4 * (steps < 0)
        return jnp.asarray(
            tensor_routes[symmetry].reshape((*permutations.shape[:2], -1)),
            dtype=jnp.int32,
        )


def _cycle_permutation(cycle, canonical):
    permutation = tuple(canonical.index(vertex) for vertex in cycle)
    if tuple(sorted(permutation)) != (0, 1, 2, 3):
        raise ValueError("Hex face does not contain its canonical vertices.")
    steps = tuple(
        (permutation[(position + 1) % 4] - permutation[position]) % 4
        for position in range(4)
    )
    if steps not in ((1, 1, 1, 1), (3, 3, 3, 3)):
        raise ValueError("Hex face orientation is inconsistent.")
    return permutation


def _quadrilateral_tensor_permutation(
    vertex_permutation,
    width_u: int,
    width_v: int,
    /,
):
    """Map one local C-order tensor grid to canonical face positions."""

    permutation = tuple(vertex_permutation)
    _cycle_permutation(list(range(4)), permutation)
    widths = (int(width_u), int(width_v))
    if any(width <= 0 for width in widths):
        raise ValueError("Face permutation widths must be positive.")
    corners = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.int32)
    origin = corners[permutation[0]]
    directions = (
        corners[permutation[1]] - origin,
        corners[permutation[3]] - origin,
    )
    canonical_widths = (
        widths[0] if directions[0][0] else widths[1],
        widths[0] if directions[0][1] else widths[1],
    )
    result = np.empty((widths[0] * widths[1],), dtype=np.int32)
    for local_u in range(widths[0]):
        for local_v in range(widths[1]):
            canonical = (
                origin * (np.asarray(canonical_widths) - 1)
                + directions[0] * local_u
                + directions[1] * local_v
            )
            result[local_u * widths[1] + local_v] = int(canonical[0]) * canonical_widths[
                1
            ] + int(canonical[1])
    return result


def hexahedral_connectivity(
    hexahedra: ArrayLike, vertex_count: int, /
) -> HexahedralConnectivity:
    vertices = int(vertex_count)
    cells = np.asarray(hexahedra, dtype=np.int32)
    if (
        vertices <= 0
        or cells.ndim != 2
        or cells.shape[1] != 8
        or cells.shape[0] == 0
        or np.any(cells < 0)
        or np.any(cells >= vertices)
    ):
        raise ValueError("hexahedra must have shape (n > 0,8) with valid vertices.")
    ordered_cells = np.sort(cells, axis=1)
    if np.any(ordered_cells[:, 1:] == ordered_cells[:, :-1]):
        raise ValueError("Each hexahedron must reference eight distinct vertices.")
    if _has_duplicate_rows(ordered_cells):
        raise ValueError("hexahedra cannot contain duplicate cells.")

    # Edges and faces are numbered by their sorted vertex keys.
    local_edges = cells[:, np.asarray(_EDGES)].reshape((-1, 2))
    edge_rows = np.sort(local_edges, axis=1)
    edge_order = _row_order(edge_rows)
    edge_starts = _row_run_starts(edge_rows[edge_order])
    edges = edge_rows[edge_order[edge_starts]]
    cell_edges = np.empty((edge_rows.shape[0],), dtype=np.int32)
    cell_edges[edge_order] = np.cumsum(edge_starts) - 1
    cell_edge_signs = np.where(local_edges[:, 0] < local_edges[:, 1], 1.0, -1.0)

    cycles = cells[:, np.asarray(_FACES)].reshape((-1, 4))
    occurrences = np.arange(cycles.shape[0], dtype=np.int64)
    # The lexicographically smallest rotation or reflection of a cycle of
    # distinct vertices starts at its minimum toward the smaller neighbor;
    # `directions` is +1 when that runs along the cycle and -1 against it.
    minimum = np.argmin(cycles, axis=1)
    directions = np.where(
        cycles[occurrences, (minimum + 1) % 4] < cycles[occurrences, (minimum - 1) % 4],
        1,
        -1,
    )
    positions = np.arange(4, dtype=np.int64)
    canonical = cycles[
        occurrences[:, None], (minimum[:, None] + directions[:, None] * positions) % 4
    ]
    # Local vertex i sits at canonical position directions * (i - minimum).
    permutations = (directions[:, None] * (positions - minimum[:, None])) % 4

    face_keys = np.sort(cycles, axis=1)
    face_order = _row_order(face_keys)
    face_starts = _row_run_starts(face_keys[face_order])
    first_positions = np.flatnonzero(face_starts)
    sorted_faces = np.cumsum(face_starts) - 1
    occurrence_faces = np.empty((cycles.shape[0],), dtype=np.int64)
    occurrence_faces[face_order] = sorted_faces
    occurrence_ranks = np.empty((cycles.shape[0],), dtype=np.int64)
    occurrence_ranks[face_order] = occurrences - first_positions[sorted_faces]
    # The stable order puts each face's earliest occurrence first.
    faces = canonical[face_order[first_positions]]
    incompatible = np.any(canonical != faces[occurrence_faces], axis=1)
    defective = incompatible | (occurrence_ranks >= 2)
    if np.any(defective):
        # A sequential scan reports the earliest defective occurrence.
        if incompatible[np.argmax(defective)]:
            raise ValueError("Shared hexahedral face cycles are incompatible.")
        raise ValueError("Non-manifold hexahedral face.")
    counts = np.bincount(occurrence_faces, minlength=faces.shape[0]).astype(np.int32)
    orientation = np.bincount(
        occurrence_faces, weights=directions, minlength=faces.shape[0]
    )
    if np.any((counts == 2) & (np.abs(orientation) == 2.0)):
        raise ValueError("Shared hexahedral faces must have opposite orientation.")

    face_sides = np.stack((faces, np.roll(faces, -1, axis=1)), axis=2).reshape((-1, 2))
    face_side_rows = np.sort(face_sides, axis=1).astype(np.int64)
    edge_keys = edges[:, 0].astype(np.int64) * vertices + edges[:, 1]
    face_edges = np.searchsorted(
        edge_keys, face_side_rows[:, 0] * vertices + face_side_rows[:, 1]
    ).astype(np.int32)
    face_edge_signs = np.where(face_sides[:, 0] < face_sides[:, 1], 1.0, -1.0)
    face_edges = face_edges.reshape((-1, 4))
    boundary_faces = counts == 1
    boundary_edges = np.zeros((edges.shape[0],), dtype=np.bool_)
    boundary_edges[face_edges[boundary_faces].reshape((-1,))] = True
    boundary_vertices = np.zeros((vertices,), dtype=np.bool_)
    boundary_vertices[faces[boundary_faces].reshape((-1,))] = True
    return HexahedralConnectivity(
        edges=jnp.asarray(edges),
        faces=jnp.asarray(faces),
        face_edges=jnp.asarray(face_edges),
        face_edge_signs=jnp.asarray(face_edge_signs.reshape((-1, 4))),
        cell_edges=jnp.asarray(cell_edges.reshape((-1, 12))),
        cell_edge_signs=jnp.asarray(cell_edge_signs.reshape((-1, 12))),
        cell_faces=jnp.asarray(occurrence_faces.reshape((-1, 6)).astype(np.int32)),
        cell_face_signs=jnp.asarray(directions.reshape((-1, 6)).astype(np.float64)),
        cell_face_vertex_permutations=jnp.asarray(
            permutations.reshape((-1, 6, 4)).astype(np.int32)
        ),
        face_cell_counts=jnp.asarray(counts),
        boundary_vertices=jnp.asarray(boundary_vertices),
        boundary_edges=jnp.asarray(boundary_edges),
        boundary_faces=jnp.asarray(boundary_faces),
        vertex_count=vertices,
        cell_count=cells.shape[0],
    )


def _hexahedral_complex(
    c: HexahedralConnectivity,
    /,
    *,
    vertex_global_ids,
    edge_global_ids,
    face_global_ids,
    cell_global_ids,
) -> CellComplexTopology:
    edges = np.asarray(c.edges)
    faces = np.asarray(c.faces)
    vids = (
        np.arange(c.vertex_count, dtype=np.int64)
        if vertex_global_ids is None
        else np.asarray(vertex_global_ids, dtype=np.int64)
    )
    cids = (
        np.arange(c.cell_count, dtype=np.int64)
        if cell_global_ids is None
        else np.asarray(cell_global_ids, dtype=np.int64)
    )
    v = EntitySet(
        "vertices", 0, vids, subsets=(EntitySubset("boundary", c.boundary_vertices),)
    )
    eids = (
        np.arange(len(edges), dtype=np.int64)
        if edge_global_ids is None
        else np.asarray(edge_global_ids, dtype=np.int64)
    )
    fids = (
        np.arange(len(faces), dtype=np.int64)
        if face_global_ids is None
        else np.asarray(face_global_ids, dtype=np.int64)
    )
    e = EntitySet(
        "edges",
        1,
        eids,
        subsets=(EntitySubset("boundary", c.boundary_edges),),
    )
    f = EntitySet(
        "faces",
        2,
        fids,
        subsets=(EntitySubset("boundary", c.boundary_faces),),
    )
    cells = EntitySet(
        "cells",
        3,
        cids,
        subsets=(EntitySubset("boundary", np.zeros(c.cell_count, np.bool_)),),
        entity_set_id=canonical_fingerprint(
            {
                "kind": "hexahedral-cell-entity-set",
                "cell_global_ids": np.sort(cids).tolist(),
            }
        ),
    )
    ve = OrientedIncidence(
        1,
        v,
        e,
        EdgeRelation(
            edges.reshape(-1),
            np.repeat(np.arange(len(edges)), 2),
            source_size=c.vertex_count,
            target_size=len(edges),
        ),
        np.tile(np.asarray([-1.0, 1.0]), len(edges)),
    )
    ef = OrientedIncidence(
        2,
        e,
        f,
        EdgeRelation(
            np.asarray(c.face_edges).reshape(-1),
            np.repeat(np.arange(len(faces)), 4),
            source_size=len(edges),
            target_size=len(faces),
        ),
        np.asarray(c.face_edge_signs).reshape(-1),
    )
    fc = OrientedIncidence(
        3,
        f,
        cells,
        EdgeRelation(
            np.asarray(c.cell_faces).reshape(-1),
            np.repeat(np.arange(c.cell_count), 6),
            source_size=len(faces),
            target_size=c.cell_count,
        ),
        np.asarray(c.cell_face_signs).reshape(-1),
    )
    return CellComplexTopology((v, e, f, cells), (ve, ef, fc))


__all__ = ["HexahedralConnectivity", "hexahedral_connectivity"]
