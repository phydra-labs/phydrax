from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from .._index import add_self_loops, maybe_num_nodes
from .._kernels import scatter_max, scatter_mean, scatter_min, segment_sum
from .._mp import MessagePassing


def _apply_linear(linear: eqx.nn.Linear, x: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(linear)(x)


def _aggregate(
    messages: jnp.ndarray,
    index: jnp.ndarray,
    num_nodes: int,
    aggr: Literal["add", "mean", "max", "min"],
) -> jnp.ndarray:
    if aggr == "add":
        return segment_sum(messages, index, num_nodes)
    if aggr == "mean":
        return scatter_mean(messages, index, num_nodes)
    if aggr == "max":
        return scatter_max(messages, index, num_nodes)
    if aggr == "min":
        return scatter_min(messages, index, num_nodes)
    raise ValueError(f"Unsupported aggregation mode: {aggr!r}.")


class GCNConv(eqx.Module):
    """Graph Convolution layer with symmetric degree normalization."""

    linear: eqx.nn.Linear
    add_self_loops: bool = eqx.field(static=True)
    normalize: bool = eqx.field(static=True)
    improved: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: jax.Array,
        add_self_loops: bool = True,
        normalize: bool = True,
        improved: bool = False,
        use_bias: bool = True,
    ):
        self.linear = eqx.nn.Linear(
            in_features,
            out_features,
            use_bias=use_bias,
            key=key,
        )
        self.add_self_loops = bool(add_self_loops)
        self.normalize = bool(normalize)
        self.improved = bool(improved)

    def __call__(
        self,
        x: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_weight: jnp.ndarray | None = None,
        *,
        num_nodes: int | None = None,
    ) -> jnp.ndarray:
        edge_index = jnp.asarray(edge_index)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("`edge_index` must have shape (2, num_edges).")

        x = jnp.asarray(x)
        if x.ndim != 2:
            raise ValueError("`x` must be rank-2 with shape (num_nodes, num_features).")

        n_nodes = maybe_num_nodes(
            edge_index, num_nodes if num_nodes is not None else x.shape[0]
        )

        if edge_weight is None:
            edge_weight = jnp.ones((edge_index.shape[1],), dtype=x.dtype)
        else:
            edge_weight = jnp.asarray(edge_weight, dtype=x.dtype)

        if self.add_self_loops:
            fill = 2.0 if self.improved else 1.0
            edge_index, loop_weight = add_self_loops(
                edge_index,
                edge_weight,
                fill_value=fill,
                num_nodes=n_nodes,
            )
            if loop_weight is None:
                raise RuntimeError("Self-loop insertion dropped explicit edge weights.")
            edge_weight = loop_weight

        row = edge_index[0].astype(jnp.int32)
        col = edge_index[1].astype(jnp.int32)

        if self.normalize:
            deg = segment_sum(edge_weight, col, n_nodes)
            deg_inv_sqrt = jnp.where(deg > 0, jnp.power(deg, -0.5), 0.0)
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        else:
            norm = edge_weight

        x_proj = _apply_linear(self.linear, x)
        messages = x_proj[row] * norm[:, None]
        return segment_sum(messages, col, n_nodes)


class SAGEConv(eqx.Module):
    """GraphSAGE convolution with configurable neighborhood aggregation."""

    lin_neigh: eqx.nn.Linear
    lin_root: eqx.nn.Linear | None
    aggr: Literal["add", "mean", "max", "min"] = eqx.field(static=True)
    normalize_output: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int | tuple[int, int],
        out_features: int,
        *,
        key: jax.Array,
        aggr: Literal["add", "mean", "max", "min"] = "mean",
        root_weight: bool = True,
        normalize_output: bool = False,
        use_bias: bool = True,
    ):
        k1, k2 = jax.random.split(key)
        if isinstance(in_features, tuple):
            in_src, in_dst = in_features
        else:
            in_src = int(in_features)
            in_dst = int(in_features)

        self.lin_neigh = eqx.nn.Linear(in_src, out_features, use_bias=False, key=k1)
        self.lin_root = (
            eqx.nn.Linear(in_dst, out_features, use_bias=use_bias, key=k2)
            if root_weight
            else None
        )
        self.aggr = aggr
        self.normalize_output = bool(normalize_output)

    def __call__(
        self,
        x: jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray],
        edge_index: jnp.ndarray,
    ) -> jnp.ndarray:
        edge_index = jnp.asarray(edge_index)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("`edge_index` must have shape (2, num_edges).")

        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x
            x_dst = x

        row = edge_index[0].astype(jnp.int32)
        col = edge_index[1].astype(jnp.int32)

        src_proj = _apply_linear(self.lin_neigh, x_src)
        messages = src_proj[row]
        out = _aggregate(messages, col, int(x_dst.shape[0]), self.aggr)

        if self.lin_root is not None:
            out = out + _apply_linear(self.lin_root, x_dst)

        if self.normalize_output:
            norm = jnp.linalg.norm(out, axis=-1, keepdims=True)
            out = out / jnp.maximum(norm, 1e-12)

        return out


class GINConv(MessagePassing):
    """Graph Isomorphism Network convolution."""

    mlp: Callable[[jnp.ndarray], jnp.ndarray]
    eps: jnp.ndarray

    def __init__(
        self,
        mlp: Callable[[jnp.ndarray], jnp.ndarray],
        *,
        eps: float = 0.0,
    ):
        super().__init__(aggr="add")
        self.mlp = mlp
        self.eps = jnp.asarray(eps)

    def __call__(
        self,
        x: jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray],
        edge_index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
        size: tuple[int, int] | None = None,
    ) -> jnp.ndarray:
        del edge_attr
        edge_index = jnp.asarray(edge_index)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("`edge_index` must have shape (2, num_edges).")

        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x
            x_dst = x

        row = edge_index[0].astype(jnp.int32)
        col = edge_index[1].astype(jnp.int32)
        num_targets = int(x_dst.shape[0]) if size is None else int(size[1])
        neigh = segment_sum(x_src[row], col, num_targets)
        out = (1.0 + self.eps) * x_dst + neigh
        return jax.vmap(self.mlp)(out)


__all__ = ["GCNConv", "SAGEConv", "GINConv"]
