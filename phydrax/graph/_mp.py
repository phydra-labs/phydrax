from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp

from phydrax._strict import StrictModule

from ..sparse import EdgeRelation, gather_routes, route_reduce, RouteReduction


MessageAggregation: TypeAlias = Literal["add", "mean", "max", "min"]


def _route_reduction(aggr: MessageAggregation, /) -> RouteReduction:
    match aggr:
        case "add":
            return "sum"
        case "mean" | "max" | "min":
            return aggr
        case _:
            raise ValueError(f"Unsupported aggregation mode: {aggr!r}.")


def _edge_index_relation(
    edge_index: jnp.ndarray,
    /,
    *,
    reverse: bool,
    source_size: int,
    target_size: int,
) -> EdgeRelation:
    """Return the fixed `(2, num_edges)` topology as source-to-target routes."""
    edge_index = jnp.asarray(edge_index)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("`edge_index` must have shape (2, num_edges).")
    if not jnp.issubdtype(edge_index.dtype, jnp.integer):
        raise TypeError("`edge_index` must use integer dtype.")
    source, target = (edge_index[1], edge_index[0]) if reverse else edge_index
    return EdgeRelation(source, target, source_size=source_size, target_size=target_size)


def _source_message(
    x_j: jnp.ndarray,
    x_i: jnp.ndarray | None,
    edge_attr: jnp.ndarray | None,
    /,
) -> jnp.ndarray:
    del x_i, edge_attr
    return x_j


def _aggregated_update(
    aggregated: jnp.ndarray,
    x: jnp.ndarray | None,
    /,
) -> jnp.ndarray:
    del x
    return aggregated


class MessagePassing(StrictModule):
    """Final configurable gather, message, aggregate, and update executor.

    `edge_index` is converted to a `phydrax.sparse.EdgeRelation`; source and
    target features are gathered along its routes and messages are reduced with
    `route_reduce` (`"add"` is the route `"sum"`; empty targets reduce to zero).
    """

    aggr: MessageAggregation = eqx.field(static=True)
    flow: Literal["source_to_target", "target_to_source"] = eqx.field(static=True)
    message_fn: Callable[
        [jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None],
        jnp.ndarray,
    ]
    update_fn: Callable[[jnp.ndarray, jnp.ndarray | None], jnp.ndarray]

    def __init__(
        self,
        *,
        aggr: MessageAggregation = "add",
        flow: Literal["source_to_target", "target_to_source"] = "source_to_target",
        message: Callable[
            [jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None],
            jnp.ndarray,
        ] = _source_message,
        update: Callable[
            [jnp.ndarray, jnp.ndarray | None],
            jnp.ndarray,
        ] = _aggregated_update,
    ):
        _route_reduction(aggr)
        if flow not in ("source_to_target", "target_to_source"):
            raise ValueError(f"Unsupported flow mode: {flow!r}.")
        self.aggr = aggr
        self.flow = flow
        if not callable(message) or not callable(update):
            raise TypeError("message and update must be callable.")
        self.message_fn = message
        self.update_fn = update

    def propagate(
        self,
        edge_index: jnp.ndarray,
        x: jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray],
        edge_attr: jnp.ndarray | None = None,
        size: tuple[int, int] | None = None,
    ) -> jnp.ndarray:
        if isinstance(x, tuple):
            x_src, x_dst = x
            if size is None:
                size = (x_src.shape[0], x_dst.shape[0])
        else:
            x_src = x
            x_dst = x
            if size is None:
                n = x.shape[0]
                size = (n, n)

        if self.flow == "source_to_target":
            relation = _edge_index_relation(
                edge_index, reverse=False, source_size=size[0], target_size=size[1]
            )
            x_base, x_source = x_dst, x_src
        else:
            relation = _edge_index_relation(
                edge_index, reverse=True, source_size=size[1], target_size=size[0]
            )
            x_base, x_source = x_src, x_dst
        x_j = gather_routes(relation, x_source)
        x_i = gather_routes(relation.transpose(), x_base)

        messages = self.message_fn(x_j, x_i, edge_attr)
        aggr_out = route_reduce(relation, messages, reduction=_route_reduction(self.aggr))
        return self.update_fn(aggr_out, x_base)

    def __call__(
        self,
        x: jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray],
        edge_index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
        size: tuple[int, int] | None = None,
    ) -> jnp.ndarray:
        return self.propagate(edge_index, x, edge_attr, size)


__all__ = ["MessagePassing"]
