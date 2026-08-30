from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation
from ._topology import CellComplexTopology, EntitySet, EntitySubset, OrientedIncidence


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
        rows = [
            [
                _quadrilateral_tensor_permutation(permutation, width_u, width_v)
                for permutation in cell
            ]
            for cell in permutations
        ]
        return jnp.asarray(rows, dtype=jnp.int32)


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


def _canonical_cycle(cycle):
    rotations = tuple(tuple(cycle[shift:] + cycle[:shift]) for shift in range(len(cycle)))
    reversed_cycle = list(reversed(cycle))
    reflected = tuple(
        tuple(reversed_cycle[shift:] + reversed_cycle[:shift])
        for shift in range(len(cycle))
    )
    return min(rotations + reflected)


def _cycle_sign(cycle, canonical):
    permutation = _cycle_permutation(cycle, canonical)
    return 1.0 if (permutation[1] - permutation[0]) % 4 == 1 else -1.0


def _quadrilateral_tensor_permutation(
    vertex_permutation,
    width_u: int,
    width_v: int,
    /,
):
    """Map one local C-order tensor grid to canonical face positions."""

    permutation = tuple(int(value) for value in vertex_permutation)
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
    if np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
        raise ValueError("Each hexahedron must reference eight distinct vertices.")
    if np.unique(np.sort(cells, axis=1), axis=0).shape[0] != cells.shape[0]:
        raise ValueError("hexahedra cannot contain duplicate cells.")

    edge_keys = sorted(
        {
            tuple(sorted((int(cell[start]), int(cell[stop]))))
            for cell in cells
            for start, stop in _EDGES
        }
    )
    edge_map = {key: index for index, key in enumerate(edge_keys)}
    edges = np.asarray(edge_keys, dtype=np.int32)
    cell_edges = np.empty((len(cells), 12), dtype=np.int32)
    cell_edge_signs = np.empty((len(cells), 12), dtype=float)
    for cell_index, cell in enumerate(cells):
        for local_edge, (start, stop) in enumerate(_EDGES):
            pair = (int(cell[start]), int(cell[stop]))
            key = tuple(sorted(pair))
            cell_edges[cell_index, local_edge] = edge_map[key]
            cell_edge_signs[cell_index, local_edge] = 1.0 if pair == key else -1.0

    face_cycles = {}
    face_counts = {}
    face_incidents = []
    for cell in cells:
        cell_incidents = []
        for template in _FACES:
            cycle = [int(cell[index]) for index in template]
            key = tuple(sorted(cycle))
            canonical = _canonical_cycle(cycle)
            if key in face_cycles and face_cycles[key] != canonical:
                raise ValueError("Shared hexahedral face cycles are incompatible.")
            face_cycles[key] = canonical
            face_counts[key] = face_counts.get(key, 0) + 1
            if face_counts[key] > 2:
                raise ValueError("Non-manifold hexahedral face.")
            cell_incidents.append((key, cycle))
        face_incidents.append(cell_incidents)

    face_keys = sorted(face_cycles)
    face_map = {key: index for index, key in enumerate(face_keys)}
    faces = np.asarray([face_cycles[key] for key in face_keys], dtype=np.int32)
    counts = np.asarray([face_counts[key] for key in face_keys], dtype=np.int32)
    cell_faces = np.empty((len(cells), 6), dtype=np.int32)
    cell_face_signs = np.empty((len(cells), 6), dtype=float)
    cell_face_vertex_permutations = np.empty(
        (len(cells), 6, 4),
        dtype=np.int32,
    )
    incident_signs = [[] for _ in face_keys]
    for cell_index, incidents in enumerate(face_incidents):
        for local_face, (key, cycle) in enumerate(incidents):
            face = face_map[key]
            canonical = list(face_cycles[key])
            sign = _cycle_sign(cycle, canonical)
            cell_faces[cell_index, local_face] = face
            cell_face_signs[cell_index, local_face] = sign
            cell_face_vertex_permutations[cell_index, local_face] = _cycle_permutation(
                cycle, canonical
            )
            incident_signs[face].append(sign)
    if any(len(signs) == 2 and signs[0] == signs[1] for signs in incident_signs):
        raise ValueError("Shared hexahedral faces must have opposite orientation.")

    face_edges = np.empty((len(faces), 4), dtype=np.int32)
    face_edge_signs = np.empty((len(faces), 4), dtype=float)
    for face_index, face in enumerate(faces):
        for position in range(4):
            pair = (
                int(face[position]),
                int(face[(position + 1) % 4]),
            )
            key = tuple(sorted(pair))
            face_edges[face_index, position] = edge_map[key]
            face_edge_signs[face_index, position] = 1.0 if pair == key else -1.0
    boundary_faces = counts == 1
    boundary_edges = np.zeros(len(edges), dtype=bool)
    boundary_vertices = np.zeros(vertices, dtype=bool)
    if np.any(boundary_faces):
        boundary_edges[np.unique(face_edges[boundary_faces])] = True
        boundary_vertices[np.unique(faces[boundary_faces])] = True
    return HexahedralConnectivity(
        edges=jnp.asarray(edges),
        faces=jnp.asarray(faces),
        face_edges=jnp.asarray(face_edges),
        face_edge_signs=jnp.asarray(face_edge_signs),
        cell_edges=jnp.asarray(cell_edges),
        cell_edge_signs=jnp.asarray(cell_edge_signs),
        cell_faces=jnp.asarray(cell_faces),
        cell_face_signs=jnp.asarray(cell_face_signs),
        cell_face_vertex_permutations=jnp.asarray(cell_face_vertex_permutations),
        face_cell_counts=jnp.asarray(counts),
        boundary_vertices=jnp.asarray(boundary_vertices),
        boundary_edges=jnp.asarray(boundary_edges),
        boundary_faces=jnp.asarray(boundary_faces),
        vertex_count=vertices,
        cell_count=len(cells),
    )


def hexahedral_cell_complex(
    hexahedra: ArrayLike,
    vertex_count: int,
    /,
    *,
    vertex_global_ids=None,
    cell_global_ids=None,
) -> CellComplexTopology:
    c = hexahedral_connectivity(hexahedra, vertex_count)
    edges = np.asarray(c.edges)
    faces = np.asarray(c.faces)
    vids = (
        np.arange(vertex_count, dtype=np.int64)
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
    e = EntitySet(
        "edges",
        1,
        np.arange(len(edges)),
        subsets=(EntitySubset("boundary", c.boundary_edges),),
    )
    f = EntitySet(
        "faces",
        2,
        np.arange(len(faces)),
        subsets=(EntitySubset("boundary", c.boundary_faces),),
    )
    cells = EntitySet(
        "cells",
        3,
        cids,
        subsets=(EntitySubset("boundary", np.zeros(c.cell_count, bool)),),
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
            source_size=vertex_count,
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


__all__ = ["HexahedralConnectivity", "hexahedral_connectivity", "hexahedral_cell_complex"]
