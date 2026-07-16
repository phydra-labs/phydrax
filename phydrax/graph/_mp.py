from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp

from ._kernels import scatter_add, scatter_max, scatter_mean, scatter_min


class MessagePassing(eqx.Module):
    """Base class for message passing layers.

    Runtime flow:
    1. gather source/target node features per edge
    2. `message`
    3. `aggregate`
    4. `update`
    """

    aggr: Literal["add", "mean", "max", "min"] = eqx.field(static=True)
    flow: Literal["source_to_target", "target_to_source"] = eqx.field(static=True)

    def __init__(
        self,
        *,
        aggr: Literal["add", "mean", "max", "min"] = "add",
        flow: Literal["source_to_target", "target_to_source"] = "source_to_target",
    ):
        if aggr not in ("add", "mean", "max", "min"):
            raise ValueError(f"Unsupported aggregation mode: {aggr!r}.")
        if flow not in ("source_to_target", "target_to_source"):
            raise ValueError(f"Unsupported flow mode: {flow!r}.")
        self.aggr = aggr
        self.flow = flow

    def message(
        self,
        x_j: jnp.ndarray,
        x_i: jnp.ndarray | None = None,
        edge_attr: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        del x_i, edge_attr
        return x_j

    def aggregate(
        self,
        messages: jnp.ndarray,
        index: jnp.ndarray,
        dim_size: int,
    ) -> jnp.ndarray:
        if self.aggr == "add":
            return scatter_add(messages, index, dim_size)
        if self.aggr == "mean":
            return scatter_mean(messages, index, dim_size)
        if self.aggr == "max":
            return scatter_max(messages, index, dim_size)
        if self.aggr == "min":
            return scatter_min(messages, index, dim_size)
        raise ValueError(f"Unsupported aggregation mode: {self.aggr!r}.")

    def update(
        self,
        aggr_out: jnp.ndarray,
        x: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        del x
        return aggr_out

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
                size = (int(x_src.shape[0]), int(x_dst.shape[0]))
        else:
            x_src = x
            x_dst = x
            if size is None:
                n = int(x.shape[0])
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

        messages = self.message(x_j, x_i, edge_attr)
        aggr_out = self.aggregate(messages, col, dim_size)
        return self.update(aggr_out, x_base)

    def __call__(
        self,
        x: jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray],
        edge_index: jnp.ndarray,
        edge_attr: jnp.ndarray | None = None,
        size: tuple[int, int] | None = None,
    ) -> jnp.ndarray:
        return self.propagate(edge_index, x, edge_attr, size)


__all__ = ["MessagePassing"]
