from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax._strict import StrictModule

from ...sparse import gather_routes, route_reduce
from .._index import add_self_loops, maybe_num_nodes
from .._mp import (
    _edge_index_relation,
    _route_reduction,
    MessageAggregation,
    MessagePassing,
)


def _apply_linear(linear: eqx.nn.Linear, x: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(linear)(x)


class GCNConv(StrictModule):
    """Graph Convolution layer with symmetric degree normalization.

    Degrees, gathers, and the neighborhood sum run over the `edge_index`
    `EdgeRelation` (after optional self-loop insertion).
    """

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

        relation = _edge_index_relation(
            edge_index, reverse=False, source_size=n_nodes, target_size=n_nodes
        )

        if self.normalize:
            deg = route_reduce(relation, edge_weight)
            deg_inv_sqrt = jnp.where(deg > 0, jnp.power(deg, -0.5), 0.0)
            norm = (
                gather_routes(relation, deg_inv_sqrt)
                * edge_weight
                * gather_routes(relation.transpose(), deg_inv_sqrt)
            )
        else:
            norm = edge_weight

        x_proj = _apply_linear(self.linear, x)
        messages = gather_routes(relation, x_proj) * norm[:, None]
        return route_reduce(relation, messages)


class SAGEConv(StrictModule):
    """GraphSAGE convolution with configurable neighborhood aggregation.

    Neighborhood aggregation reduces source projections over the `edge_index`
    `EdgeRelation`; `aggr="add"` is the route `"sum"` and empty neighborhoods
    aggregate to zero.
    """

    lin_neigh: eqx.nn.Linear
    lin_root: eqx.nn.Linear | None
    aggr: MessageAggregation = eqx.field(static=True)
    normalize_output: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int | tuple[int, int],
        out_features: int,
        *,
        key: jax.Array,
        aggr: MessageAggregation = "mean",
        root_weight: bool = True,
        normalize_output: bool = False,
        use_bias: bool = True,
    ):
        _route_reduction(aggr)
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
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x
            x_dst = x

        relation = _edge_index_relation(
            edge_index,
            reverse=False,
            source_size=x_src.shape[0],
            target_size=x_dst.shape[0],
        )
        src_proj = _apply_linear(self.lin_neigh, x_src)
        out = route_reduce(
            relation,
            gather_routes(relation, src_proj),
            reduction=_route_reduction(self.aggr),
        )

        if self.lin_root is not None:
            out = out + _apply_linear(self.lin_root, x_dst)

        if self.normalize_output:
            norm = jnp.linalg.norm(out, axis=-1, keepdims=True)
            out = out / jnp.maximum(norm, 1e-12)

        return out


class GINConv(StrictModule):
    """Graph Isomorphism Network convolution."""

    mlp: Callable[[jnp.ndarray], jnp.ndarray]
    eps: jnp.ndarray
    message_passing: MessagePassing = eqx.field(static=True)

    def __init__(
        self,
        mlp: Callable[[jnp.ndarray], jnp.ndarray],
        *,
        eps: float = 0.0,
    ):
        self.mlp = mlp
        self.eps = jnp.asarray(eps)
        self.message_passing = MessagePassing(aggr="add")

    def __call__(
        self,
        x: jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray],
        edge_index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
        size: tuple[int, int] | None = None,
    ) -> jnp.ndarray:
        neigh = self.message_passing(
            x,
            edge_index,
            edge_attr,
            size,
        )
        x_dst = x[1] if isinstance(x, tuple) else x
        out = (1.0 + self.eps) * x_dst + neigh
        return jax.vmap(self.mlp)(out)


__all__ = ["GCNConv", "SAGEConv", "GINConv"]
