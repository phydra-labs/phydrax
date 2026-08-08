#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

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
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch
from phydrax.nn.operator.encoded import AbstractEncodedOperatorModel
from phydrax.nn.operator.prompt import OperatorPrompt, PromptedOperatorBatch


class OperatorPromptState(StrictModule):
    """Permutation-invariant encoded demonstration memory."""

    values: Array
    weights: Array
    mask: Array
    case_shape: tuple[int, ...]
    capacity: int
    tokens_per_example: int

    def __init__(
        self,
        *,
        values: Array,
        weights: Array,
        mask: Array,
        case_shape: tuple[int, ...],
        capacity: int,
        tokens_per_example: int,
    ):
        cases = tuple(int(size) for size in case_shape)
        values_ = jnp.asarray(values)
        if values_.ndim != len(cases) + 2:
            raise ValueError("Prompt state values require case/token/channel axes.")
        expected = cases + (int(values_.shape[-2]),)
        weights_ = jnp.asarray(weights)
        mask_ = jnp.asarray(mask, dtype=bool)
        if weights_.shape != expected or mask_.shape != expected:
            raise ValueError("Prompt state weights and mask must match its tokens.")
        self.values = values_
        self.weights = weights_
        self.mask = mask_
        self.case_shape = cases
        self.capacity = int(capacity)
        self.tokens_per_example = int(tokens_per_example)

    @property
    def num_tokens(self) -> int:
        return int(self.values.shape[-2])


class InContextOperatorState(StrictModule):
    """Query-conditioned latent state after attending to a prompt memory."""

    values: Array
    weights: Array
    mask: Array
    case_shape: tuple[int, ...]
    prompt_state: OperatorPromptState

    def __init__(
        self,
        *,
        values: Array,
        weights: Array,
        mask: Array,
        case_shape: tuple[int, ...],
        prompt_state: OperatorPromptState,
    ):
        self.values = jnp.asarray(values)
        self.weights = jnp.asarray(weights)
        self.mask = jnp.asarray(mask, dtype=bool)
        self.case_shape = tuple(int(size) for size in case_shape)
        self.prompt_state = prompt_state

    @property
    def num_tokens(self) -> int:
        return int(self.values.shape[-2])


class InContextOperator(AbstractEncodedOperatorModel):
    """ICON-style operator that conditions on supervised function demonstrations."""

    operator_architecture = "InContextOperator"

    source_lift: Linear
    target_lift: Linear
    token_bank: Array
    role_embeddings: Array
    prompt_source_attention: MeasureAwareAttention
    prompt_target_attention: MeasureAwareAttention
    prompt_processor: LatentTokenProcessor
    current_attention: MeasureAwareAttention
    context_attention: MeasureAwareAttention
    current_processor: LatentTokenProcessor
    query_lift: Linear
    decoder_attention: MeasureAwareAttention
    decoder_norm: eqx.nn.RMSNorm
    projection: Linear
    source_key: str | None
    in_channels: int
    out_channels: int
    coord_dim: int
    width: int
    num_tokens: int
    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]

    def __init__(
        self,
        *,
        in_channels: int | Literal["scalar"],
        out_channels: int | Literal["scalar"],
        coord_dim: int,
        width: int = 128,
        num_tokens: int = 32,
        prompt_depth: int = 2,
        processor_depth: int = 4,
        num_heads: int = 8,
        head_dim: int | None = None,
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
        self.num_tokens = int(num_tokens)
        self.source_key = source_key
        self.in_size = in_channels
        self.out_size = out_channels
        if (
            min(
                self.in_channels,
                self.out_channels,
                self.coord_dim,
                self.width,
                self.num_tokens,
                int(prompt_depth),
                int(processor_depth),
                int(num_heads),
            )
            <= 0
        ):
            raise ValueError("In-context operator dimensions must be positive.")
        resolved_head_dim = (
            self.width // int(num_heads) if head_dim is None else int(head_dim)
        )
        if head_dim is None and self.width % int(num_heads) != 0:
            raise ValueError("In-context width must be divisible by num_heads.")
        keys = jr.split(key, 13)
        self.source_lift = Linear(
            in_size=self.in_channels + self.coord_dim,
            out_size=self.width,
            activation=jnn.gelu,
            key=keys[0],
        )
        self.target_lift = Linear(
            in_size=self.out_channels + self.coord_dim,
            out_size=self.width,
            activation=jnn.gelu,
            key=keys[1],
        )
        self.token_bank = jr.normal(keys[2], (self.num_tokens, self.width)) / jnp.sqrt(
            float(self.width)
        )
        self.role_embeddings = jr.normal(keys[3], (3, self.width)) / jnp.sqrt(
            float(self.width)
        )
        attention_kwargs = dict(
            source_channels=self.width,
            query_channels=self.width,
            out_channels=self.width,
            num_heads=int(num_heads),
            head_dim=resolved_head_dim,
            kernel=attention_kernel,
            execution=attention_execution,
            block_size=attention_block_size,
            accumulation_dtype=accumulation_dtype,
        )
        self.prompt_source_attention = MeasureAwareAttention(
            key=keys[4], **attention_kwargs
        )
        self.prompt_target_attention = MeasureAwareAttention(
            key=keys[5], **attention_kwargs
        )
        self.prompt_processor = LatentTokenProcessor(
            self.width,
            int(prompt_depth),
            num_heads=int(num_heads),
            head_dim=resolved_head_dim,
            feed_forward_multiplier=feed_forward_multiplier,
            kernel=attention_kernel,
            execution=attention_execution,
            block_size=attention_block_size,
            accumulation_dtype=accumulation_dtype,
            key=keys[6],
        )
        self.current_attention = MeasureAwareAttention(key=keys[7], **attention_kwargs)
        self.context_attention = MeasureAwareAttention(key=keys[8], **attention_kwargs)
        self.current_processor = LatentTokenProcessor(
            self.width,
            int(processor_depth),
            num_heads=int(num_heads),
            head_dim=resolved_head_dim,
            feed_forward_multiplier=feed_forward_multiplier,
            kernel=attention_kernel,
            execution=attention_execution,
            block_size=attention_block_size,
            accumulation_dtype=accumulation_dtype,
            key=keys[9],
        )
        self.query_lift = Linear(
            in_size=self.coord_dim,
            out_size=self.width,
            activation=jnn.gelu,
            key=keys[10],
        )
        self.decoder_attention = MeasureAwareAttention(key=keys[11], **attention_kwargs)
        self.decoder_norm = eqx.nn.RMSNorm(self.width, eps=1e-6, use_bias=False)
        self.projection = Linear(
            in_size=self.width,
            out_size=self.out_channels,
            activation=None,
            key=keys[12],
        )

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError("InContextOperator requires source_key for multiple inputs.")
        return next(iter(batch.inputs.values()))

    def _physical_features(
        self,
        batch: OperatorBatch,
        /,
    ) -> tuple[Array, Array, Array]:
        source = self._source(batch)
        values = _flatten_function_values(source, batch.case_shape, self.in_channels)
        coordinates, weights, mask = _flatten_geometry(source, batch.case_shape)
        if int(coordinates.shape[-1]) != self.coord_dim:
            raise ValueError(
                f"Expected source coordinate dimension {self.coord_dim}; "
                f"got {coordinates.shape[-1]}."
            )
        features = self.source_lift(jnp.concatenate((values, coordinates), axis=-1))
        return features, weights, mask

    def _target_features(
        self,
        batch: OperatorBatch,
        targets: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        query = FunctionSamples(
            values=targets,
            axes=batch.require_single_query().axes,
            coordinates=batch.require_single_query().coordinates,
            quadrature_weights=batch.require_single_query().quadrature_weights,
            mask=batch.require_single_query().mask,
            topology=batch.require_single_query().topology,
        )
        values = _flatten_function_values(query, batch.case_shape, self.out_channels)
        coordinates, weights, mask = _flatten_geometry(query, batch.case_shape)
        if int(coordinates.shape[-1]) != self.coord_dim:
            raise ValueError(
                f"Expected target coordinate dimension {self.coord_dim}; "
                f"got {coordinates.shape[-1]}."
            )
        features = self.target_lift(jnp.concatenate((values, coordinates), axis=-1))
        return features, weights, mask

    def _tokens_from_samples(
        self,
        features: Array,
        weights: Array,
        sample_mask: Array,
        attention: MeasureAwareAttention,
        role_index: int,
        case_shape: tuple[int, ...],
        /,
    ) -> Array:
        cases = prod(case_shape) if case_shape else 1
        sample_count = int(features.shape[-2])
        source = features.reshape((cases, sample_count, self.width))
        tokens = jnp.broadcast_to(
            self.token_bank + self.role_embeddings[role_index],
            (cases, self.num_tokens, self.width),
        )
        return tokens + attention(
            source,
            tokens,
            weights.reshape((cases, sample_count)),
            source_mask=sample_mask.reshape((cases, sample_count)),
        )

    def encode_prompt(self, prompt: OperatorPrompt, /) -> OperatorPromptState:
        case_shape = prompt.case_shape
        cases = prod(case_shape) if case_shape else 1
        examples: list[Array] = []
        masks: list[Array] = []
        for index, example in enumerate(prompt.examples):
            source_features, source_weights, source_mask = self._physical_features(
                example.batch
            )
            target_features, target_weights, target_mask = self._target_features(
                example.batch, example.targets
            )
            source_tokens = self._tokens_from_samples(
                source_features,
                source_weights,
                source_mask,
                self.prompt_source_attention,
                0,
                case_shape,
            )
            target_tokens = self._tokens_from_samples(
                target_features,
                target_weights,
                target_mask,
                self.prompt_target_attention,
                1,
                case_shape,
            )
            combined = jnp.concatenate((source_tokens, target_tokens), axis=1)
            example_present = prompt.mask[..., index].reshape((cases,))
            token_mask = jnp.broadcast_to(
                example_present[:, None], (cases, 2 * self.num_tokens)
            )
            token_weights = jnp.ones(token_mask.shape, dtype=combined.dtype)
            combined = self.prompt_processor(
                combined * token_mask[..., None], token_weights, token_mask
            )
            examples.append(combined)
            masks.append(token_mask)
        stacked = jnp.stack(examples, axis=1)
        stacked_mask = jnp.stack(masks, axis=1)
        memory_count = prompt.capacity * 2 * self.num_tokens
        shape = case_shape + (memory_count,)
        return OperatorPromptState(
            values=stacked.reshape(shape + (self.width,)),
            weights=jnp.ones(shape, dtype=stacked.dtype),
            mask=stacked_mask.reshape(shape),
            case_shape=case_shape,
            capacity=prompt.capacity,
            tokens_per_example=2 * self.num_tokens,
        )

    def encode_with_prompt_state(
        self,
        batch: OperatorBatch,
        prompt_state: OperatorPromptState,
        /,
    ) -> InContextOperatorState:
        if batch.case_shape != prompt_state.case_shape:
            raise ValueError("Prompt state and query batch case shapes must match.")
        features, weights, source_mask = self._physical_features(batch)
        cases = prod(batch.case_shape) if batch.case_shape else 1
        source_count = int(features.shape[-2])
        tokens = jnp.broadcast_to(
            self.token_bank + self.role_embeddings[2],
            (cases, self.num_tokens, self.width),
        )
        token_mask = jnp.ones((cases, self.num_tokens), dtype=bool)
        tokens = tokens + self.current_attention(
            features.reshape((cases, source_count, self.width)),
            tokens,
            weights.reshape((cases, source_count)),
            source_mask=source_mask.reshape((cases, source_count)),
            query_mask=token_mask,
        )
        memory = prompt_state.values.reshape((cases, prompt_state.num_tokens, self.width))
        tokens = tokens + self.context_attention(
            memory,
            tokens,
            prompt_state.weights.reshape((cases, prompt_state.num_tokens)),
            source_mask=prompt_state.mask.reshape((cases, prompt_state.num_tokens)),
            query_mask=token_mask,
        )
        token_weights = jnp.ones(token_mask.shape, dtype=tokens.dtype)
        tokens = self.current_processor(tokens, token_weights, token_mask)
        shape = batch.case_shape + (self.num_tokens,)
        return InContextOperatorState(
            values=tokens.reshape(shape + (self.width,)),
            weights=token_weights.reshape(shape),
            mask=token_mask.reshape(shape),
            case_shape=batch.case_shape,
            prompt_state=prompt_state,
        )

    def encode_inputs(
        self,
        batch: PromptedOperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> InContextOperatorState:
        del key
        if not isinstance(batch, PromptedOperatorBatch):
            raise TypeError("InContextOperator requires a PromptedOperatorBatch.")
        return self.encode_with_prompt_state(
            batch.batch, self.encode_prompt(batch.prompt)
        )

    def decode_query(
        self,
        state: InContextOperatorState,
        query: FunctionSamples,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        del key
        coordinates = query.coordinates_array(case_shape=state.case_shape, flatten=True)
        if int(coordinates.shape[-1]) != self.coord_dim:
            raise ValueError(
                f"Expected query coordinate dimension {self.coord_dim}; "
                f"got {coordinates.shape[-1]}."
            )
        cases = prod(state.case_shape) if state.case_shape else 1
        query_count = prod(query.sample_shape)
        query_mask = query.mask_array(case_shape=state.case_shape).reshape(
            (cases, query_count)
        )
        query_features = self.query_lift(coordinates).reshape(
            (cases, query_count, self.width)
        )
        memory = state.values.reshape((cases, state.num_tokens, self.width))
        decoded = query_features + self.decoder_attention(
            memory,
            query_features,
            state.weights.reshape((cases, state.num_tokens)),
            source_mask=state.mask.reshape((cases, state.num_tokens)),
            query_mask=query_mask,
        )
        decoded = _feature_norm(self.decoder_norm, decoded)
        output = self.projection(decoded) * query_mask[..., None]
        shaped = output.reshape(
            state.case_shape + query.sample_shape + (self.out_channels,)
        )
        return shaped[..., 0] if self.out_size == "scalar" else shaped

    def __call__(
        self,
        x: PromptedOperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if not isinstance(x, PromptedOperatorBatch):
            raise TypeError("InContextOperator requires a PromptedOperatorBatch.")
        state = self.encode_inputs(x, key=key)
        return self.decode_query(state, x.batch.require_single_query(), key=key)


__all__ = [
    "InContextOperator",
    "InContextOperatorState",
    "OperatorPromptState",
]
