from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax.numpy as jnp

from phydrax._strict import StrictModule

from ._kernels import scatter_add, scatter_max, scatter_mean, scatter_min


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
    """Final configurable gather, message, aggregate, and update executor."""

    aggr: Literal["add", "mean", "max", "min"] = eqx.field(static=True)
    flow: Literal["source_to_target", "target_to_source"] = eqx.field(static=True)
    message_fn: Callable[
        [jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None],
        jnp.ndarray,
    ]
    update_fn: Callable[[jnp.ndarray, jnp.ndarray | None], jnp.ndarray]

    def __init__(
        self,
        *,
        aggr: Literal["add", "mean", "max", "min"] = "add",
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
        if aggr not in ("add", "mean", "max", "min"):
            raise ValueError(f"Unsupported aggregation mode: {aggr!r}.")
        if flow not in ("source_to_target", "target_to_source"):
            raise ValueError(f"Unsupported flow mode: {flow!r}.")
        self.aggr = aggr
        self.flow = flow
        if not callable(message) or not callable(update):
            raise TypeError("message and update must be callable.")
        self.message_fn = message
        self.update_fn = update

    def aggregate(
        self,
        messages: jnp.ndarray,
        index: jnp.ndarray,
        dim_size: int,
    ) -> jnp.ndarray:
        match self.aggr:
            case "add":
                return scatter_add(messages, index, dim_size)
            case "mean":
                return scatter_mean(messages, index, dim_size)
            case "max":
                return scatter_max(messages, index, dim_size)
            case "min":
                return scatter_min(messages, index, dim_size)
            case _:
                raise AssertionError("Validated aggregation mode fell through.")

    def propagate(
        self,
        edge_index: jnp.ndarray,
        x: jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray],
        edge_attr: jnp.ndarray | None = None,
        size: tuple[int, int] | None = None,
    ) -> jnp.ndarray:
        edge_index = jnp.asarray(edge_index)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("`edge_index` must have shape (2, num_edges).")
        if not jnp.issubdtype(edge_index.dtype, jnp.integer):
            raise TypeError("`edge_index` must use integer dtype.")

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
            row = edge_index[0].astype(jnp.int32)
            col = edge_index[1].astype(jnp.int32)
            dim_size = int(size[1])
            x_base = x_dst
        else:
            row = edge_index[1].astype(jnp.int32)
            col = edge_index[0].astype(jnp.int32)
            dim_size = int(size[0])
            x_base = x_src

        x_j = jnp.take(x_src, row, axis=0)
        x_i = jnp.take(x_dst, col, axis=0)

        messages = self.message_fn(x_j, x_i, edge_attr)
        aggr_out = self.aggregate(messages, col, dim_size)
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
