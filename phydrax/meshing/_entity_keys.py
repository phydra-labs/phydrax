#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host entity keys and fixed-width device key lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from ..discretization import _cell_complex as cell_complex_owner
from ..discretization._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._topology_rows import facet_columns


if TYPE_CHECKING:
    from ..discretization import _cell_mesh as cell_mesh_owner
    from ..discretization._adaptive_simplex import AdaptiveSimplexState


def entity_keys(mesh: cell_mesh_owner.CellMesh, dimension: int, /) -> np.ndarray:
    """Keys aligned with ``mesh.entity_set(dimension)`` rows.

    Vertices and cells give ``(n, 1)`` global IDs; edges and faces give ascending
    vertex-global-ID rows, padded with ``-1`` after the IDs when faces of a mixed
    mesh have different arities.
    """
    # CellMesh construction uses key lookup through scope; defer this nominal mesh guard.
    from ..discretization import _cell_mesh as cell_mesh_owner

    if not isinstance(mesh, cell_mesh_owner.CellMesh):
        raise TypeError(f"entity_keys.mesh must be CellMesh; got {type(mesh).__name__}.")
    target = int(dimension)
    if target == 0:
        return np.asarray(mesh.vertex_global_ids, dtype=np.int64)[:, None]
    if target == mesh.topological_dimension:
        return np.asarray(mesh.entity_set(target).entity_ids, dtype=np.int64)[:, None]
    connectivity = mesh.connectivity
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if target == 1 and isinstance(
        connectivity,
        (
            PolygonalConnectivity,
            TetrahedralConnectivity,
            HexahedralConnectivity,
            cell_complex_owner.PolyhedralConnectivity,
        ),
    ):
        return np.sort(vertex_ids[np.asarray(connectivity.edges, dtype=np.int64)], axis=1)
    if target == 2 and isinstance(
        connectivity, (TetrahedralConnectivity, HexahedralConnectivity)
    ):
        return np.sort(vertex_ids[np.asarray(connectivity.faces, dtype=np.int64)], axis=1)
    if target == 2 and isinstance(
        connectivity, cell_complex_owner.PolyhedralConnectivity
    ):
        offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        values = vertex_ids[np.asarray(connectivity.face_vertex_values, dtype=np.int64)]
        arity = np.diff(offsets)
        width = int(np.max(arity, initial=0))
        column = np.arange(values.size, dtype=np.int64) - np.repeat(offsets[:-1], arity)
        padded = np.full((arity.size, width), np.iinfo(np.int64).max, dtype=np.int64)
        padded[np.repeat(np.arange(arity.size), arity), column] = values
        ordered = np.sort(padded, axis=1)
        return np.where(ordered == np.iinfo(np.int64).max, -1, ordered)
    raise ValueError(
        f"entity_keys.dimension must have an entity-key route; got {target!r}."
    )


def key_rows(table: np.ndarray, queries: np.ndarray, /) -> np.ndarray:
    """Row of each query key in ``table`` (unique rows), or -1 when absent."""

    table_ = np.asarray(table, dtype=np.int64)
    queries_ = np.asarray(queries, dtype=np.int64)
    if queries_.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if table_.shape[0] == 0:
        return np.full((queries_.shape[0],), -1, dtype=np.int64)
    width = table_.shape[1]
    if queries_.ndim != 2 or queries_.shape[1] != width:
        raise ValueError(
            f"key_rows.queries.shape must be (query_count, {width}); got {queries_.shape!r}."
        )
    combined = np.concatenate((table_, queries_), axis=0)
    _, inverse = np.unique(combined, axis=0, return_inverse=True)
    inverse = inverse.reshape((-1,))
    owner = np.full((int(np.max(inverse)) + 1,), -1, dtype=np.int64)
    owner[inverse[: table_.shape[0]]] = np.arange(table_.shape[0], dtype=np.int64)
    return owner[inverse[table_.shape[0] :]]


@jax.jit
def key_positions(table: jax.Array, queries: jax.Array, /) -> jax.Array:
    """Lexicographic device lookup of fixed-width global-ID entity keys."""

    low = jnp.zeros((queries.shape[0],), dtype=jnp.int64)
    high = jnp.full(low.shape, table.shape[0], dtype=jnp.int64)

    def search(
        _: int, bounds: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        left, right = bounds
        middle = (left + right) // 2
        row = table[jnp.minimum(middle, table.shape[0] - 1)]
        equal = row == queries
        earlier_equal = jnp.concatenate(
            (
                jnp.ones((queries.shape[0], 1), dtype=jnp.bool_),
                jnp.cumprod(equal[:, :-1], axis=1, dtype=jnp.int32).astype(jnp.bool_),
            ),
            axis=1,
        )
        less = jnp.any((row < queries) & earlier_equal, axis=1)
        return jnp.where(less, middle + 1, left), jnp.where(less, right, middle)

    low, _ = jax.lax.fori_loop(0, table.shape[0].bit_length() + 1, search, (low, high))
    clipped = jnp.minimum(low, table.shape[0] - 1)
    return jnp.where(jnp.all(table[clipped] == queries, axis=1), clipped, -1)


def facet_packets(state: AdaptiveSimplexState, /) -> tuple[jax.Array, jax.Array]:
    width = state.mesh.cells.shape[1]
    columns = jnp.asarray(facet_columns(width), dtype=jnp.int32)
    identifiers = state.mesh.vertex_ids[state.mesh.cells[:, columns]]
    signs = jnp.ones(identifiers.shape[:2], dtype=jnp.int32)
    for left in range(width - 1):
        for right in range(left + 1, width - 1):
            signs *= jnp.where(identifiers[..., left] > identifiers[..., right], -1, 1)
    signs *= jnp.where(jnp.arange(width) % 2 == 0, 1, -1)[None, :]
    active = jnp.repeat(state.mesh.cell_active, width)
    keys = jnp.sort(identifiers, axis=2).reshape((-1, width - 1))
    keys = jnp.where(active[:, None], keys, jnp.iinfo(jnp.int64).max)
    return keys, jnp.where(active, signs.reshape(-1), 0)
