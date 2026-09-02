#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, prod
from typing import Literal

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

import phydrax.ein as ein
from phydrax._doc import DOC_KEY0
from phydrax._strict import StrictModule
from phydrax.nn._keys import EvalKey
from phydrax.nn._utils import _get_size
from phydrax.nn.layers._linear import Linear
from phydrax.nn.layers._measure_attention import (
    AttentionExecution,
    AttentionKernel,
    MeasureAwareAttention,
)
from phydrax.nn.operator.architectures.attention._upt import (
    _feature_norm,
    _flatten_function_values,
    _flatten_geometry,
    LatentTokenProcessor,
)
from phydrax.nn.operator.context import EncodedOperatorState, operator_context_fingerprint
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch
from phydrax.nn.operator.encoded import AbstractEncodedOperatorModel


class _PhysicsSliceTokenizer(StrictModule):
    """Learn sparse physical memberships and integrate points into slice tokens."""

    assignment: Linear
    value: Linear
    channels: int = eqx.field(static=True)
    num_slices: int = eqx.field(static=True)
    top_k: int = eqx.field(static=True)
    temperature: float = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        num_slices: int,
        /,
        *,
        top_k: int,
        temperature: float,
        key: Key[Array, ""],
    ):
        self.channels = int(channels)
        self.num_slices = int(num_slices)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        if self.channels <= 0 or self.num_slices <= 0:
            raise ValueError("Slice channels and num_slices must be positive.")
        if self.top_k <= 0 or self.top_k > self.num_slices:
            raise ValueError("slice_top_k must be between one and num_slices.")
        if not isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("slice_temperature must be finite and positive.")
        assignment_key, value_key = jr.split(key)
        self.assignment = Linear(
            in_size=self.channels,
            out_size=self.num_slices,
            activation=None,
            key=assignment_key,
        )
        self.value = Linear(
            in_size=self.channels,
            out_size=self.channels,
            activation=None,
            key=value_key,
        )

    def _memberships(self, features: Array, /) -> Array:
        logits = self.assignment(features) / self.temperature
        soft_membership = jnn.softmax(logits, axis=-1)
        if self.top_k == self.num_slices:
            return soft_membership

        _, indices = jax.lax.top_k(logits, self.top_k)
        selected = jnp.sum(
            jnn.one_hot(indices, self.num_slices, dtype=bool), axis=-2
        ).astype(bool)
        restricted_logits = jnp.where(
            selected,
            logits,
            jnp.asarray(-jnp.inf, dtype=logits.dtype),
        )
        restricted = jnn.softmax(restricted_logits, axis=-1)
        # The forward pass is an exact top-k partition. The dense softmax gradient
        # keeps the assignment head trainable for the top-k=1 Transolver path.
        return soft_membership + jax.lax.stop_gradient(restricted - soft_membership)

    def __call__(
        self,
        features: Array,
        quadrature: Array,
        mask: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        array = jnp.asarray(features)
        if array.ndim != 3 or int(array.shape[-1]) != self.channels:
            raise ValueError(
                "Physics slice tokenization expects (cases, points, channels)."
            )
        weights = jnp.asarray(quadrature)
        valid = jnp.asarray(mask, dtype=bool)
        if weights.shape != array.shape[:2] or valid.shape != array.shape[:2]:
            raise ValueError("Slice quadrature and masks must match physical points.")

        safe_features = jnp.where(valid[..., None], array, 0.0)
        memberships = self._memberships(safe_features)
        measure = jnp.where(valid & (weights > 0.0), weights, 0.0)
        weighted_memberships = memberships * measure[..., None]
        slice_measure = jnp.sum(weighted_memberships, axis=1)
        slice_mask = slice_measure > 0.0
        normalizer = jnp.where(slice_mask, slice_measure, 1.0)
        tokens = ein.contract(
            "bns,bnc->bsc", weighted_memberships, self.value(safe_features)
        )
        tokens = (tokens / normalizer[..., None]) * slice_mask[..., None]
        return tokens, slice_measure, slice_mask


class Transolver(AbstractEncodedOperatorModel):
    """Physics-attention operator with quadrature-aware physical slice tokens.

    ``slice_top_k=1`` gives the non-overlapping Transolver partition. Larger
    values use normalized top-k memberships, so each physical point contributes
    to several slices; setting it to ``num_slices`` gives fully soft overlap.
    """

    operator_architecture = "Transolver"

    source_lift: Linear
    tokenizer: _PhysicsSliceTokenizer
    processor: LatentTokenProcessor
    query_lift: Linear
    decoder_attention: MeasureAwareAttention
    decoder_norm: eqx.nn.RMSNorm
    projection: Linear
    source_key: str | None = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    coord_dim: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    num_slices: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    slice_top_k: int = eqx.field(static=True)
    slice_temperature: float = eqx.field(static=True)
    in_size: int | Literal["scalar"] = eqx.field(static=True)
    out_size: int | Literal["scalar"] = eqx.field(static=True)

    def __init__(
        self,
        *,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        coord_dim: int,
        num_slices: int = 64,
        width: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        head_dim: int | None = None,
        slice_top_k: int = 1,
        slice_temperature: float = 1.0,
        feed_forward_multiplier: float = 4.0,
        source_key: str | None = None,
        attention_kernel: AttentionKernel = "softmax",
        attention_execution: AttentionExecution = "auto",
        attention_block_size: int = 256,
        accumulation_dtype: str = "input",
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_channels = _get_size(in_channels)
        self.out_channels = _get_size(out_channels)
        self.coord_dim = int(coord_dim)
        self.width = int(width)
        self.num_slices = int(num_slices)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.slice_top_k = int(slice_top_k)
        self.slice_temperature = float(slice_temperature)
        self.source_key = None if source_key is None else str(source_key)
        self.in_size = in_channels
        self.out_size = out_channels
        if (
            min(
                self.in_channels,
                self.out_channels,
                self.coord_dim,
                self.width,
                self.num_slices,
                self.depth,
                self.num_heads,
            )
            <= 0
        ):
            raise ValueError("Transolver dimensions must be positive.")
        if self.slice_top_k <= 0 or self.slice_top_k > self.num_slices:
            raise ValueError("slice_top_k must be between one and num_slices.")
        if not isfinite(self.slice_temperature) or self.slice_temperature <= 0.0:
            raise ValueError("slice_temperature must be finite and positive.")
        if head_dim is None:
            if self.width % self.num_heads != 0:
                raise ValueError(
                    "Transolver width must be divisible by num_heads when "
                    "head_dim is not supplied."
                )
            resolved_head_dim = self.width // self.num_heads
        else:
            resolved_head_dim = int(head_dim)
        if resolved_head_dim <= 0:
            raise ValueError("head_dim must be positive.")
        self.head_dim = resolved_head_dim

        keys = jr.split(key, 6)
        self.source_lift = Linear(
            in_size=self.in_channels + self.coord_dim,
            out_size=self.width,
            activation=jnn.gelu,
            key=keys[0],
        )
        self.tokenizer = _PhysicsSliceTokenizer(
            self.width,
            self.num_slices,
            top_k=self.slice_top_k,
            temperature=self.slice_temperature,
            key=keys[1],
        )
        self.processor = LatentTokenProcessor(
            self.width,
            self.depth,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            feed_forward_multiplier=feed_forward_multiplier,
            kernel=attention_kernel,
            execution=attention_execution,
            block_size=attention_block_size,
            accumulation_dtype=accumulation_dtype,
            key=keys[2],
        )
        self.query_lift = Linear(
            in_size=self.coord_dim,
            out_size=self.width,
            activation=jnn.gelu,
            key=keys[3],
        )
        self.decoder_attention = MeasureAwareAttention(
            source_channels=self.width,
            query_channels=self.width,
            out_channels=self.width,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            kernel=attention_kernel,
            execution=attention_execution,
            block_size=attention_block_size,
            accumulation_dtype=accumulation_dtype,
            key=keys[4],
        )
        self.decoder_norm = eqx.nn.RMSNorm(self.width, eps=1e-6, use_bias=False)
        self.projection = Linear(
            in_size=self.width,
            out_size=self.out_channels,
            activation=None,
            key=keys[5],
        )

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError(
                "Transolver requires source_key when OperatorBatch has multiple inputs."
            )
        return next(iter(batch.inputs.values()))

    def encode_inputs(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> EncodedOperatorState:
        del key
        source = self._source(batch)
        values = _flatten_function_values(source, batch.case_shape, self.in_channels)
        coordinates, quadrature, source_mask = _flatten_geometry(source, batch.case_shape)
        if int(coordinates.shape[-1]) != self.coord_dim:
            raise ValueError(
                f"Transolver expected coordinate dimension {self.coord_dim}; "
                f"got {coordinates.shape[-1]}."
            )

        cases = prod(batch.case_shape) if batch.case_shape else 1
        point_count = prod(source.sample_shape)
        values = values.reshape((cases, point_count, self.in_channels))
        coordinates = coordinates.reshape((cases, point_count, self.coord_dim))
        quadrature = quadrature.reshape((cases, point_count))
        source_mask = source_mask.reshape((cases, point_count))
        safe_values = jnp.where(source_mask[..., None], values, 0.0)
        safe_coordinates = jnp.where(source_mask[..., None], coordinates, 0.0)
        source_features = (
            self.source_lift(jnp.concatenate((safe_values, safe_coordinates), axis=-1))
            * source_mask[..., None]
        )
        tokens, slice_measure, slice_mask = self.tokenizer(
            source_features, quadrature, source_mask
        )
        processed, layers = self.processor(
            tokens,
            slice_measure,
            slice_mask,
            return_layers=True,
        )

        token_shape = batch.case_shape + (self.num_slices,)
        layer_values = tuple(
            layer.reshape(token_shape + (self.width,)) for layer in layers
        )
        return EncodedOperatorState(
            kind="learned",
            values=processed.reshape(token_shape + (self.width,)),
            coordinates=None,
            weights=slice_measure.reshape(token_shape),
            mask=slice_mask.reshape(token_shape),
            case_shape=batch.case_shape,
            schema_fingerprint=operator_context_fingerprint(
                source, case_shape=batch.case_shape
            ),
            layer_values=layer_values,
        )

    def decode_query(
        self,
        state: EncodedOperatorState,
        query: FunctionSamples,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        del key
        coordinates = query.coordinates_array(case_shape=state.case_shape, flatten=True)
        if int(coordinates.shape[-1]) != self.coord_dim:
            raise ValueError(
                f"Transolver expected query coordinate dimension {self.coord_dim}; "
                f"got {coordinates.shape[-1]}."
            )
        cases = prod(state.case_shape) if state.case_shape else 1
        query_count = prod(query.sample_shape)
        query_mask = query.mask_array(case_shape=state.case_shape).reshape(
            (cases, query_count)
        )
        coordinates = coordinates.reshape((cases, query_count, self.coord_dim))
        safe_coordinates = jnp.where(query_mask[..., None], coordinates, 0.0)
        query_features = self.query_lift(safe_coordinates) * query_mask[..., None]
        source = state.values.reshape((cases, self.num_slices, self.width))
        decoded = self.decoder_attention(
            source,
            query_features,
            state.weights.reshape((cases, self.num_slices)),
            source_mask=state.mask.reshape((cases, self.num_slices)),
            query_mask=query_mask,
        )
        decoded = (
            _feature_norm(self.decoder_norm, query_features + decoded)
            * query_mask[..., None]
        )
        output = self.projection(decoded) * query_mask[..., None]
        shaped = output.reshape(
            state.case_shape + query.sample_shape + (self.out_channels,)
        )
        return shaped[..., 0] if self.out_size == "scalar" else shaped

    def __call__(
        self,
        x: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("Transolver requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = ["Transolver"]
