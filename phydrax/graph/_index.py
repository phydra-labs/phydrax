from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

from ._kernels import segment_max, segment_mean, segment_min, segment_sum


Reduce = Literal["add", "mean", "max", "min"]


def maybe_num_nodes(edge_index: jnp.ndarray, num_nodes: int | None = None) -> int:
    if num_nodes is not None:
        return int(num_nodes)
    if int(edge_index.size) == 0:
        return 0
    return int(jnp.max(edge_index)) + 1


def _validate_edge_index(edge_index: jnp.ndarray) -> jnp.ndarray:
    edge_index = jnp.asarray(edge_index)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("`edge_index` must have shape (2, num_edges).")
    if not jnp.issubdtype(edge_index.dtype, jnp.integer):
        raise TypeError("`edge_index` must use integer dtype.")
    return edge_index.astype(jnp.int32)


def _reduce_edge_attr(
    edge_attr: jnp.ndarray,
    inverse: jnp.ndarray,
    n_unique: int,
    reduce: Reduce,
) -> jnp.ndarray:
    if reduce == "add":
        return segment_sum(edge_attr, inverse, n_unique)
    if reduce == "mean":
        return segment_mean(edge_attr, inverse, n_unique)
    if reduce == "max":
        return segment_max(edge_attr, inverse, n_unique)
    if reduce == "min":
        return segment_min(edge_attr, inverse, n_unique)
    raise ValueError(f"Unsupported reduce mode: {reduce!r}.")


def coalesce(
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray | None = None,
    *,
    num_nodes: int | None = None,
    reduce: Reduce = "add",
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    edge_index = _validate_edge_index(edge_index)
    if int(edge_index.shape[1]) == 0:
        return edge_index, edge_attr

    n_nodes = maybe_num_nodes(edge_index, num_nodes)
    row = edge_index[0]
    col = edge_index[1]
    keys = row * n_nodes + col

    uniq, inverse = jnp.unique(keys, return_inverse=True)
    row_u = uniq // n_nodes
    col_u = uniq % n_nodes
    edge_index_u = jnp.stack([row_u, col_u], axis=0)

    if edge_attr is None:
        return edge_index_u, None

    edge_attr = jnp.asarray(edge_attr)
    n_unique = int(uniq.shape[0])
    edge_attr_u = _reduce_edge_attr(edge_attr, inverse, n_unique, reduce)
    return edge_index_u, edge_attr_u


def to_undirected(
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray | None = None,
    *,
    num_nodes: int | None = None,
    reduce: Reduce = "add",
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    edge_index = _validate_edge_index(edge_index)

    rev = jnp.stack([edge_index[1], edge_index[0]], axis=0)
    undirected = jnp.concatenate([edge_index, rev], axis=1)

    attrs = None
    if edge_attr is not None:
        edge_attr = jnp.asarray(edge_attr)
        attrs = jnp.concatenate([edge_attr, edge_attr], axis=0)

    return coalesce(undirected, attrs, num_nodes=num_nodes, reduce=reduce)


def remove_self_loops(
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    edge_index = _validate_edge_index(edge_index)
    mask = edge_index[0] != edge_index[1]
    out_index = edge_index[:, mask]
    if edge_attr is None:
        return out_index, None
    return out_index, jnp.asarray(edge_attr)[mask]


def add_self_loops(
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray | None = None,
    *,
    fill_value: float = 1.0,
    num_nodes: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    edge_index = _validate_edge_index(edge_index)
    n_nodes = maybe_num_nodes(edge_index, num_nodes)

    loops = jnp.arange(n_nodes, dtype=jnp.int32)
    loop_index = jnp.stack([loops, loops], axis=0)
    out_index = jnp.concatenate([edge_index, loop_index], axis=1)

    if edge_attr is None:
        return out_index, None

    edge_attr = jnp.asarray(edge_attr)
    if edge_attr.ndim == 1:
        loop_attr = jnp.full((n_nodes,), fill_value, dtype=edge_attr.dtype)
    else:
        loop_attr = jnp.full((n_nodes,) + edge_attr.shape[1:], fill_value, dtype=edge_attr.dtype)
    out_attr = jnp.concatenate([edge_attr, loop_attr], axis=0)
    return out_index, out_attr


def add_remaining_self_loops(
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray | None = None,
    *,
    fill_value: float = 1.0,
    num_nodes: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    edge_index = _validate_edge_index(edge_index)
    n_nodes = maybe_num_nodes(edge_index, num_nodes)

    row = edge_index[0]
    col = edge_index[1]
    is_loop = row == col

    has_loop = jnp.zeros((n_nodes,), dtype=bool)
    if int(row.shape[0]) > 0:
        has_loop = has_loop.at[row[is_loop]].set(True)

    missing = jnp.where(~has_loop)[0]
    if int(missing.shape[0]) == 0:
        return edge_index, edge_attr

    loop_index = jnp.stack([missing, missing], axis=0)
    out_index = jnp.concatenate([edge_index, loop_index], axis=1)

    if edge_attr is None:
        return out_index, None

    edge_attr = jnp.asarray(edge_attr)
    if edge_attr.ndim == 1:
        loop_attr = jnp.full((missing.shape[0],), fill_value, dtype=edge_attr.dtype)
    else:
        loop_attr = jnp.full(
            (missing.shape[0],) + edge_attr.shape[1:],
            fill_value,
            dtype=edge_attr.dtype,
        )
    out_attr = jnp.concatenate([edge_attr, loop_attr], axis=0)
    return out_index, out_attr


def degree(
    index: jnp.ndarray,
    *,
    num_nodes: int | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    index = jnp.asarray(index, dtype=jnp.int32)
    n_nodes = maybe_num_nodes(jnp.stack([index, index], axis=0), num_nodes)
    ones = jnp.ones((index.shape[0],), dtype=dtype)
    return segment_sum(ones, index, n_nodes)


def in_degree(
    edge_index: jnp.ndarray,
    *,
    num_nodes: int | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    edge_index = _validate_edge_index(edge_index)
    return degree(edge_index[1], num_nodes=num_nodes, dtype=dtype)


def out_degree(
    edge_index: jnp.ndarray,
    *,
    num_nodes: int | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    edge_index = _validate_edge_index(edge_index)
    return degree(edge_index[0], num_nodes=num_nodes, dtype=dtype)


def to_dense_adj(
    edge_index: jnp.ndarray,
    edge_attr: jnp.ndarray | None = None,
    *,
    num_nodes: int | None = None,
) -> jnp.ndarray:
    edge_index = _validate_edge_index(edge_index)
    n_nodes = maybe_num_nodes(edge_index, num_nodes)
    row = edge_index[0]
    col = edge_index[1]

    if edge_attr is None:
        out = jnp.zeros((n_nodes, n_nodes), dtype=jnp.float32)
        return out.at[row, col].set(1.0)

    edge_attr = jnp.asarray(edge_attr)
    if edge_attr.ndim == 1:
        out = jnp.zeros((n_nodes, n_nodes), dtype=edge_attr.dtype)
        return out.at[row, col].set(edge_attr)

    out = jnp.zeros((n_nodes, n_nodes) + edge_attr.shape[1:], dtype=edge_attr.dtype)
    return out.at[row, col].set(edge_attr)


def to_edge_index(adj: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    adj = jnp.asarray(adj)
    if adj.ndim == 2:
        row, col = jnp.where(adj != 0)
        edge_index = jnp.stack([row, col], axis=0).astype(jnp.int32)
        edge_attr = adj[row, col]
        return edge_index, edge_attr

    if adj.ndim >= 3:
        mask = jnp.any(adj != 0, axis=tuple(range(2, adj.ndim)))
        row, col = jnp.where(mask)
        edge_index = jnp.stack([row, col], axis=0).astype(jnp.int32)
        edge_attr = adj[row, col]
        return edge_index, edge_attr

    raise ValueError("`adj` must be rank-2 or higher.")


__all__ = [
    "coalesce",
    "to_undirected",
    "remove_self_loops",
    "add_self_loops",
    "add_remaining_self_loops",
    "degree",
    "in_degree",
    "out_degree",
    "to_dense_adj",
    "to_edge_index",
    "maybe_num_nodes",
]
