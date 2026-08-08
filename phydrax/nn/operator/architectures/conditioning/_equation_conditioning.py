#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import sqrt

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, Key

from phydrax._strict import StrictModule
from phydrax.equations import (
    PDE_OPERATOR_VOCABULARY,
    PDE_TOKEN_ATTRIBUTES,
    PDE_TOKEN_KINDS,
    PDETokenBatch,
)
from phydrax.nn.layers._linear import Linear
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch


class PDEConditionEncoder(StrictModule):
    """Structure-aware encoder for canonical PDE IR token batches."""

    kind_embeddings: Array
    operator_embeddings: Array
    attribute_embeddings: Array
    depth_embeddings: Array
    scalar_projection: Linear
    slot_projection: Linear
    dimension_projection: Linear
    symbol_attention_biases: Array
    parent_projections: tuple[Linear, ...]
    query_projections: tuple[Linear, ...]
    key_projections: tuple[Linear, ...]
    value_projections: tuple[Linear, ...]
    output_projections: tuple[Linear, ...]
    feed_forward_in: tuple[Linear, ...]
    feed_forward_out: tuple[Linear, ...]
    width: int
    dimension_rank: int
    max_tree_depth: int

    def __init__(
        self,
        *,
        width: int = 128,
        depth: int = 3,
        dimension_rank: int = 7,
        max_tree_depth: int = 32,
        key: Key[Array, ""],
    ):
        if min(width, depth, max_tree_depth) <= 0 or dimension_rank < 0:
            raise ValueError("PDE encoder dimensions must be positive.")
        self.width = int(width)
        self.dimension_rank = int(dimension_rank)
        self.max_tree_depth = int(max_tree_depth)
        keys = iter(jr.split(key, 7 * int(depth) + 8))
        scale = 1.0 / sqrt(float(width))
        self.kind_embeddings = scale * jr.normal(
            next(keys), (len(PDE_TOKEN_KINDS), self.width)
        )
        self.operator_embeddings = scale * jr.normal(
            next(keys), (len(PDE_OPERATOR_VOCABULARY), self.width)
        )
        self.attribute_embeddings = scale * jr.normal(
            next(keys), (len(PDE_TOKEN_ATTRIBUTES), self.width)
        )
        self.depth_embeddings = scale * jr.normal(
            next(keys), (self.max_tree_depth + 1, self.width)
        )
        self.scalar_projection = Linear(
            in_size=1,
            out_size=self.width,
            activation=None,
            key=next(keys),
        )
        self.slot_projection = Linear(
            in_size=1,
            out_size=self.width,
            activation=None,
            key=next(keys),
        )
        self.dimension_projection = Linear(
            in_size=max(1, self.dimension_rank),
            out_size=self.width,
            activation=None,
            key=next(keys),
        )
        self.symbol_attention_biases = scale * jr.normal(
            next(keys),
            (int(depth),),
        )
        self.parent_projections = tuple(
            Linear(
                in_size=self.width,
                out_size=self.width,
                activation=None,
                key=next(keys),
            )
            for _ in range(int(depth))
        )
        self.query_projections = tuple(
            Linear(
                in_size=self.width,
                out_size=self.width,
                activation=None,
                key=next(keys),
            )
            for _ in range(int(depth))
        )
        self.key_projections = tuple(
            Linear(
                in_size=self.width,
                out_size=self.width,
                activation=None,
                key=next(keys),
            )
            for _ in range(int(depth))
        )
        self.value_projections = tuple(
            Linear(
                in_size=self.width,
                out_size=self.width,
                activation=None,
                key=next(keys),
            )
            for _ in range(int(depth))
        )
        self.output_projections = tuple(
            Linear(
                in_size=self.width,
                out_size=self.width,
                activation=None,
                key=next(keys),
            )
            for _ in range(int(depth))
        )
        self.feed_forward_in = tuple(
            Linear(
                in_size=self.width,
                out_size=2 * self.width,
                activation=jnn.gelu,
                key=next(keys),
            )
            for _ in range(int(depth))
        )
        self.feed_forward_out = tuple(
            Linear(
                in_size=2 * self.width,
                out_size=self.width,
                activation=None,
                key=next(keys),
            )
            for _ in range(int(depth))
        )

    @staticmethod
    def _normalize(values: Array, /) -> Array:
        return values * jax.lax.rsqrt(
            jnp.mean(jnp.square(values), axis=-1, keepdims=True) + 1e-6
        )

    def _encode_case(
        self,
        hidden: Array,
        parent: Array,
        symbol: Array,
        mask: Array,
    ) -> Array:
        token_count = hidden.shape[0]
        parent_index = jnp.clip(parent, 0, token_count - 1)
        parent_mask = (parent >= 0) & mask
        same_symbol = (
            (symbol[:, None] == symbol[None, :])
            & (symbol[:, None] != 0)
            & mask[:, None]
            & mask[None, :]
        )
        for (
            symbol_bias,
            parent_projection,
            query,
            key,
            value,
            output,
            ff_in,
            ff_out,
        ) in zip(
            self.symbol_attention_biases,
            self.parent_projections,
            self.query_projections,
            self.key_projections,
            self.value_projections,
            self.output_projections,
            self.feed_forward_in,
            self.feed_forward_out,
            strict=True,
        ):
            normalized = self._normalize(hidden)
            parent_hidden = normalized[parent_index] * parent_mask[:, None]
            hidden = hidden + parent_projection(parent_hidden)
            normalized = self._normalize(hidden)
            queries = query(normalized)
            keys = key(normalized)
            values = value(normalized)
            logits = oe.contract("id,jd->ij", queries, keys) / sqrt(float(self.width))
            logits = logits + symbol_bias * same_symbol
            logits = jnp.where(mask[None, :], logits, -jnp.inf)
            attention = jnn.softmax(logits, axis=-1)
            attended = oe.contract("ij,jd->id", attention, values)
            hidden = hidden + output(attended)
            hidden = hidden + ff_out(ff_in(self._normalize(hidden)))
            hidden = hidden * mask[:, None]
        weights = mask.astype(hidden.dtype)
        return jnp.sum(hidden * weights[:, None], axis=0) / jnp.maximum(
            jnp.sum(weights), 1.0
        )

    def __call__(self, tokens: PDETokenBatch, /) -> Array:
        if tokens.physical_dimension.shape[-1] != self.dimension_rank:
            raise ValueError(
                "PDE token physical dimension rank does not match the encoder."
            )
        case_shape = tokens.batch_shape
        token_count = tokens.max_tokens
        scalar = jnp.sign(tokens.scalar) * jnp.log1p(jnp.abs(tokens.scalar))
        slot = jnp.where(
            tokens.slot >= 0,
            jnp.log1p(tokens.slot.astype(scalar.dtype)) + 1.0,
            0.0,
        )
        dimension = tokens.physical_dimension
        if self.dimension_rank == 0:
            dimension = jnp.zeros(case_shape + (token_count, 1), dtype=scalar.dtype)
        hidden = (
            self.kind_embeddings[tokens.kind]
            + self.operator_embeddings[tokens.operator]
            + self.attribute_embeddings[tokens.attribute]
            + self.depth_embeddings[jnp.clip(tokens.depth, 0, self.max_tree_depth)]
            + self.scalar_projection(scalar[..., None])
            + self.slot_projection(slot[..., None])
            + self.dimension_projection(dimension)
        )
        cases = 1
        for size in case_shape:
            cases *= size
        encoded = jax.vmap(self._encode_case)(
            hidden.reshape((cases, token_count, self.width)),
            tokens.parent.reshape((cases, token_count)),
            tokens.symbol.reshape((cases, token_count)),
            tokens.mask.reshape((cases, token_count)),
        )
        return encoded.reshape(case_shape + (self.width,))


def attach_pde_condition(
    batch: OperatorBatch,
    tokens: PDETokenBatch,
    encoder: PDEConditionEncoder,
    /,
    *,
    input_name: str = "equation",
) -> OperatorBatch:
    """Attach an encoded PDE as a one-anchor global operator input branch."""
    if input_name in batch.inputs:
        raise ValueError(f"OperatorBatch already contains input {input_name!r}.")
    condition = encoder(tokens)
    if tokens.batch_shape == () and batch.case_shape:
        condition = jnp.broadcast_to(condition, batch.case_shape + condition.shape)
    elif tokens.batch_shape != batch.case_shape:
        raise ValueError(
            "PDE token batch shape must be scalar or match OperatorBatch case_shape."
        )
    values = condition[..., None, :]
    inputs = dict(batch.inputs)
    inputs[input_name] = FunctionSamples(
        values=values,
        coordinates=jnp.zeros((1, 1), dtype=values.dtype),
        quadrature_weights=jnp.ones((1,), dtype=values.dtype),
    )
    return OperatorBatch(
        inputs=inputs,
        queries=batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


__all__ = ["PDEConditionEncoder", "attach_pde_condition"]
