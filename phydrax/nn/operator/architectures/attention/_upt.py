#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import Any, ClassVar, Literal, overload

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._frozendict import frozendict
from phydrax._strict import StrictModule
from phydrax.nn._keys import EvalKey
from phydrax.nn._utils import _get_size
from phydrax.nn.layers._linear import Linear
from phydrax.nn.layers._measure_attention import (
    AttentionExecution,
    AttentionKernel,
    MeasureAwareAttention,
)
from phydrax.nn.operator.branches import (
    apply_branch_interactions,
    BranchedEncodedOperatorState,
    OperatorBranchGraph,
)
from phydrax.nn.operator.context import (
    EncodedOperatorState,
    operator_context_fingerprint,
)
from phydrax.nn.operator.data import (
    FunctionSamples,
    OperatorBatch,
    OperatorFieldBatch,
    OperatorOutputSpec,
    OperatorPrediction,
)
from phydrax.nn.operator.encoded import AbstractEncodedOperatorModel


def _feature_norm(norm: eqx.nn.RMSNorm, values: Array, /) -> Array:
    array = jnp.asarray(values)
    flattened = array.reshape((-1, int(array.shape[-1])))
    return jax.vmap(norm)(flattened).reshape(array.shape)


def _flatten_function_values(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    channels: int,
    /,
) -> Array:
    if samples.values is None:
        raise ValueError("Operator source samples require values.")
    sample_shape = samples.sample_shape
    sample_ndim = len(sample_shape)
    count = prod(sample_shape)
    array = samples.values
    scalar_shape = case_shape + sample_shape
    if tuple(int(size) for size in array.shape) == scalar_shape:
        values = array.reshape(case_shape + (count, 1))
    elif (
        array.ndim == len(case_shape) + sample_ndim + 1
        and tuple(int(size) for size in array.shape[: len(case_shape)]) == case_shape
        and tuple(int(size) for size in array.shape[len(case_shape) : -1]) == sample_shape
    ):
        values = array.reshape(case_shape + (count, int(array.shape[-1])))
    else:
        raise ValueError(
            "Operator source values must have case/sample axes and at most one "
            f"channel axis; got {array.shape}."
        )
    if int(values.shape[-1]) != int(channels):
        raise ValueError(
            f"Expected {channels} source channels across value leaves; "
            f"got {values.shape[-1]}."
        )
    return values


def _flatten_geometry(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    /,
) -> tuple[Array, Array, Array]:
    count = prod(samples.sample_shape)
    coordinates = samples.coordinates_array(case_shape=case_shape, flatten=True)
    weights = samples.quadrature(case_shape=case_shape).reshape(case_shape + (count,))
    mask = samples.mask_array(case_shape=case_shape).reshape(case_shape + (count,))
    return coordinates, weights, mask


class LatentTokenBlock(StrictModule):
    """Quadrature-aware pre-normalized transformer block for abstract tokens."""

    attention: MeasureAwareAttention
    attention_norm: eqx.nn.RMSNorm
    feed_forward_norm: eqx.nn.RMSNorm
    gate: Linear
    value: Linear
    output: Linear
    width: int

    def __init__(
        self,
        width: int,
        /,
        *,
        num_heads: int = 8,
        head_dim: int | None = None,
        feed_forward_multiplier: float = 4.0,
        kernel: AttentionKernel = "softmax",
        execution: AttentionExecution = "auto",
        block_size: int = 256,
        accumulation_dtype: str = "input",
        norm_eps: float = 1e-6,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.width = int(width)
        hidden = round(float(feed_forward_multiplier) * self.width)
        resolved_head_dim = (
            self.width // int(num_heads) if head_dim is None else int(head_dim)
        )
        if min(self.width, hidden, int(num_heads), resolved_head_dim) <= 0:
            raise ValueError("Token block dimensions must be positive.")
        if head_dim is None and self.width % int(num_heads) != 0:
            raise ValueError("Token width must be divisible by num_heads.")
        if float(norm_eps) <= 0.0:
            raise ValueError("norm_eps must be positive.")
        attention_key, gate_key, value_key, output_key = jr.split(key, 4)
        self.attention = MeasureAwareAttention(
            source_channels=self.width,
            query_channels=self.width,
            out_channels=self.width,
            num_heads=int(num_heads),
            head_dim=resolved_head_dim,
            kernel=kernel,
            execution=execution,
            block_size=block_size,
            accumulation_dtype=accumulation_dtype,
            key=attention_key,
        )
        self.attention_norm = eqx.nn.RMSNorm(
            self.width, eps=float(norm_eps), use_bias=False
        )
        self.feed_forward_norm = eqx.nn.RMSNorm(
            self.width, eps=float(norm_eps), use_bias=False
        )
        self.gate = Linear(
            in_size=self.width,
            out_size=hidden,
            activation=None,
            use_bias=False,
            key=gate_key,
        )
        self.value = Linear(
            in_size=self.width,
            out_size=hidden,
            activation=None,
            use_bias=False,
            key=value_key,
        )
        self.output = Linear(
            in_size=hidden,
            out_size=self.width,
            activation=None,
            use_bias=False,
            key=output_key,
        )

    def __call__(
        self,
        values: Array,
        weights: Array,
        mask: Array,
        /,
    ) -> Array:
        array = jnp.asarray(values)
        case_shape = tuple(int(size) for size in array.shape[:-2])
        cases = prod(case_shape) if case_shape else 1
        tokens = int(array.shape[-2])
        flattened = array.reshape((cases, tokens, self.width))
        flattened_weights = jnp.asarray(weights).reshape((cases, tokens))
        flattened_mask = jnp.asarray(mask, dtype=bool).reshape((cases, tokens))
        normalized = _feature_norm(self.attention_norm, flattened)
        attended = self.attention(
            normalized,
            normalized,
            flattened_weights,
            source_mask=flattened_mask,
            query_mask=flattened_mask,
        )
        updated = flattened + attended
        normalized = _feature_norm(self.feed_forward_norm, updated)
        feed_forward = self.output(
            jnn.silu(self.gate(normalized)) * self.value(normalized)
        )
        updated = (updated + feed_forward) * flattened_mask[..., None]
        return updated.reshape(array.shape)


class LatentTokenProcessor(StrictModule):
    """Shared fixed-cardinality latent processor used by UPT-style models."""

    blocks: tuple[LatentTokenBlock, ...]
    width: int
    depth: int

    def __init__(
        self,
        width: int,
        depth: int,
        /,
        *,
        num_heads: int = 8,
        head_dim: int | None = None,
        feed_forward_multiplier: float = 4.0,
        kernel: AttentionKernel = "softmax",
        execution: AttentionExecution = "auto",
        block_size: int = 256,
        accumulation_dtype: str = "input",
        norm_eps: float = 1e-6,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.width = int(width)
        self.depth = int(depth)
        if self.depth <= 0:
            raise ValueError("Latent token processor depth must be positive.")
        self.blocks = tuple(
            LatentTokenBlock(
                self.width,
                num_heads=num_heads,
                head_dim=head_dim,
                feed_forward_multiplier=feed_forward_multiplier,
                kernel=kernel,
                execution=execution,
                block_size=block_size,
                accumulation_dtype=accumulation_dtype,
                norm_eps=norm_eps,
                key=block_key,
            )
            for block_key in jr.split(key, self.depth)
        )

    @overload
    def __call__(
        self,
        values: Array,
        weights: Array,
        mask: Array,
        /,
        *,
        return_layers: Literal[False] = False,
    ) -> Array: ...

    @overload
    def __call__(
        self,
        values: Array,
        weights: Array,
        mask: Array,
        /,
        *,
        return_layers: Literal[True],
    ) -> tuple[Array, tuple[Array, ...]]: ...

    def __call__(
        self,
        values: Array,
        weights: Array,
        mask: Array,
        /,
        *,
        return_layers: bool = False,
    ) -> Array | tuple[Array, tuple[Array, ...]]:
        hidden = values
        layers: list[Array] = []
        for block in self.blocks:
            hidden = block(hidden, weights, mask)
            layers.append(hidden)
        if return_layers:
            return hidden, tuple(layers)
        return hidden


class UPT(AbstractEncodedOperatorModel):
    """Universal Physics Transformer with fixed latent tokens and query decoding."""

    operator_architecture = "UPT"

    source_lift: Linear
    latent_tokens: Array
    encoder_attention: MeasureAwareAttention
    processor: LatentTokenProcessor
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
        num_tokens: int = 64,
        depth: int = 4,
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
                int(depth),
                int(num_heads),
            )
            <= 0
        ):
            raise ValueError("UPT dimensions must be positive.")
        resolved_head_dim = (
            self.width // int(num_heads) if head_dim is None else int(head_dim)
        )
        if head_dim is None and self.width % int(num_heads) != 0:
            raise ValueError("UPT width must be divisible by num_heads.")
        keys = jr.split(key, 8)
        self.source_lift = Linear(
            in_size=self.in_channels + self.coord_dim,
            out_size=self.width,
            activation=jnn.gelu,
            key=keys[0],
        )
        self.latent_tokens = jr.normal(keys[1], (self.num_tokens, self.width)) / jnp.sqrt(
            float(self.width)
        )
        attention_kwargs = dict(
            num_heads=int(num_heads),
            head_dim=resolved_head_dim,
            kernel=attention_kernel,
            execution=attention_execution,
            block_size=attention_block_size,
            accumulation_dtype=accumulation_dtype,
        )
        self.encoder_attention = MeasureAwareAttention(
            source_channels=self.width,
            query_channels=self.width,
            out_channels=self.width,
            key=keys[2],
            **attention_kwargs,
        )
        self.processor = LatentTokenProcessor(
            self.width,
            int(depth),
            num_heads=int(num_heads),
            head_dim=resolved_head_dim,
            feed_forward_multiplier=feed_forward_multiplier,
            kernel=attention_kernel,
            execution=attention_execution,
            block_size=attention_block_size,
            accumulation_dtype=accumulation_dtype,
            key=keys[3],
        )
        self.query_lift = Linear(
            in_size=self.coord_dim,
            out_size=self.width,
            activation=jnn.gelu,
            key=keys[4],
        )
        self.decoder_attention = MeasureAwareAttention(
            source_channels=self.width,
            query_channels=self.width,
            out_channels=self.width,
            key=keys[5],
            **attention_kwargs,
        )
        self.decoder_norm = eqx.nn.RMSNorm(self.width, eps=1e-6, use_bias=False)
        self.projection = Linear(
            in_size=self.width,
            out_size=self.out_channels,
            activation=None,
            key=keys[6],
        )

    def _source(self, batch: OperatorBatch, /) -> FunctionSamples:
        if self.source_key is not None:
            return batch.input(self.source_key)
        if len(batch.inputs) != 1:
            raise ValueError(
                "UPT requires source_key when OperatorBatch has multiple inputs."
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
        coordinates, weights, source_mask = _flatten_geometry(source, batch.case_shape)
        if int(coordinates.shape[-1]) != self.coord_dim:
            raise ValueError(
                f"UPT expected coordinate dimension {self.coord_dim}; "
                f"got {coordinates.shape[-1]}."
            )
        source_features = self.source_lift(
            jnp.concatenate((values, coordinates), axis=-1)
        )
        cases = prod(batch.case_shape) if batch.case_shape else 1
        source_count = int(source_features.shape[-2])
        flattened_source = source_features.reshape((cases, source_count, self.width))
        tokens = jnp.broadcast_to(
            self.latent_tokens, (cases, self.num_tokens, self.width)
        )
        token_mask = jnp.ones((cases, self.num_tokens), dtype=bool)
        encoded = tokens + self.encoder_attention(
            flattened_source,
            tokens,
            weights.reshape((cases, source_count)),
            source_mask=source_mask.reshape((cases, source_count)),
            query_mask=token_mask,
        )
        token_weights = jnp.ones((cases, self.num_tokens), dtype=encoded.dtype)
        encoded, layers = self.processor(
            encoded,
            token_weights,
            token_mask,
            return_layers=True,
        )
        shape = batch.case_shape + (self.num_tokens,)
        layer_values = tuple(layer.reshape(shape + (self.width,)) for layer in layers)
        return EncodedOperatorState(
            kind="learned",
            values=encoded.reshape(shape + (self.width,)),
            coordinates=None,
            weights=token_weights.reshape(shape),
            mask=token_mask.reshape(shape),
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
                f"UPT expected query coordinate dimension {self.coord_dim}; "
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
        source = state.values.reshape((cases, state.num_tokens, self.width))
        attended = self.decoder_attention(
            source,
            query_features,
            state.weights.reshape((cases, state.num_tokens)),
            source_mask=state.mask.reshape((cases, state.num_tokens)),
            query_mask=query_mask,
        )
        decoded = _feature_norm(self.decoder_norm, query_features + attended)
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
            raise TypeError("UPT requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


def _predict_abupt(
    model: Any,
    batch: OperatorBatch,
    key: EvalKey,
    /,
) -> OperatorPrediction:
    state = model.encode_inputs(batch, key=key)
    queries = {}
    for name in model.prediction_names:
        branch = model.graph.branch(name)
        assert branch.query_name is not None
        queries[name] = batch.query(branch.query_name)
    values = model.decode_queries(state, queries)
    fields = {}
    for name in model.prediction_names:
        branch = model.graph.branch(name)
        assert branch.query_name is not None
        assert branch.output_spec is not None
        fields[name] = OperatorFieldBatch(
            values[name],
            query_name=branch.query_name,
            spec=branch.output_spec,
        )
    return OperatorPrediction(
        fields,
        batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


class ABUPT(AbstractEncodedOperatorModel):
    """Anchored-branched UPT with typed physical branches and cross-attention."""

    operator_architecture = "ABUPT"
    _operator_prediction_builder: ClassVar = staticmethod(_predict_abupt)

    graph: OperatorBranchGraph
    input_channels: frozendict[str, int]
    coord_dims: frozendict[str, int]
    anchor_counts: frozendict[str, int]
    source_lifts: tuple[Linear, ...]
    role_embeddings: Array
    self_processors: tuple[tuple[LatentTokenBlock, ...], ...]
    interaction_groups: tuple[str, ...]
    interaction_attention: tuple[MeasureAwareAttention, ...]
    query_lifts: tuple[Linear, ...]
    decoder_attention: tuple[MeasureAwareAttention, ...]
    decoder_norms: tuple[eqx.nn.RMSNorm, ...]
    projections: tuple[Linear, ...]
    conditioning_names: tuple[str, ...]
    prediction_names: tuple[str, ...]
    default_output_branch: str
    width: int
    depth: int
    in_size: tuple[int, ...]
    out_size: int | Literal["scalar"]

    def __init__(
        self,
        graph: OperatorBranchGraph,
        /,
        *,
        input_channels: Mapping[str, int | Literal["scalar"]],
        coord_dims: Mapping[str, int],
        anchor_counts: int | Mapping[str, int] = 64,
        default_output_branch: str | None = None,
        width: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        head_dim: int | None = None,
        feed_forward_multiplier: float = 4.0,
        attention_kernel: AttentionKernel = "softmax",
        attention_execution: AttentionExecution = "auto",
        attention_block_size: int = 256,
        accumulation_dtype: str = "input",
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.graph = graph
        self.conditioning_names = graph.conditioning_names
        self.prediction_names = graph.prediction_names
        if not self.conditioning_names:
            raise ValueError("ABUPT requires at least one conditioning branch.")
        if not self.prediction_names:
            raise ValueError("ABUPT requires at least one prediction branch.")
        if any(not graph.branch(name).conditions for name in self.prediction_names):
            raise ValueError(
                "ABUPT prediction branches must also condition so their anchors can decode."
            )
        if set(input_channels) != set(self.conditioning_names):
            raise ValueError(
                "input_channels must cover every conditioning branch exactly."
            )
        if set(coord_dims) != set(self.conditioning_names):
            raise ValueError("coord_dims must cover every conditioning branch exactly.")
        resolved_anchors = (
            {name: int(anchor_counts) for name in self.conditioning_names}
            if isinstance(anchor_counts, int)
            else {name: int(value) for name, value in anchor_counts.items()}
        )
        if set(resolved_anchors) != set(self.conditioning_names):
            raise ValueError(
                "anchor_counts must cover every conditioning branch exactly."
            )
        self.input_channels = frozendict(
            {name: _get_size(input_channels[name]) for name in self.conditioning_names}
        )
        self.coord_dims = frozendict(
            {name: int(coord_dims[name]) for name in self.conditioning_names}
        )
        self.anchor_counts = frozendict(resolved_anchors)
        self.width = int(width)
        self.depth = int(depth)
        if (
            min(
                self.width,
                self.depth,
                int(num_heads),
                *(self.input_channels.values()),
                *(self.coord_dims.values()),
                *(self.anchor_counts.values()),
            )
            <= 0
        ):
            raise ValueError("ABUPT dimensions must be positive.")
        chosen_output = (
            self.prediction_names[0]
            if default_output_branch is None
            else str(default_output_branch)
        )
        if chosen_output not in self.prediction_names:
            raise ValueError("default_output_branch must be a prediction branch.")
        self.default_output_branch = chosen_output
        self.in_size = tuple(
            self.input_channels[name] for name in self.conditioning_names
        )
        default_spec = graph.branch(chosen_output).output_spec
        assert default_spec is not None
        self.out_size = default_spec.channels
        for interaction in graph.interactions:
            if interaction.stage >= self.depth:
                raise ValueError("Branch interaction stages must be smaller than depth.")
        resolved_head_dim = (
            self.width // int(num_heads) if head_dim is None else int(head_dim)
        )
        if head_dim is None and self.width % int(num_heads) != 0:
            raise ValueError("ABUPT width must be divisible by num_heads.")
        groups = tuple(sorted({item.parameter_group for item in graph.interactions}))
        self.interaction_groups = groups
        key_count = (
            len(self.conditioning_names)
            + self.depth * len(self.conditioning_names)
            + len(groups)
            + 3 * len(self.prediction_names)
            + 1
        )
        keys = iter(jr.split(key, key_count))
        self.source_lifts = tuple(
            Linear(
                in_size=self.input_channels[name] + self.coord_dims[name],
                out_size=self.width,
                activation=jnn.gelu,
                key=next(keys),
            )
            for name in self.conditioning_names
        )
        self.role_embeddings = jr.normal(
            next(keys), (len(self.conditioning_names), self.width)
        ) / jnp.sqrt(float(self.width))
        self.self_processors = tuple(
            tuple(
                LatentTokenBlock(
                    self.width,
                    num_heads=num_heads,
                    head_dim=resolved_head_dim,
                    feed_forward_multiplier=feed_forward_multiplier,
                    kernel=attention_kernel,
                    execution=attention_execution,
                    block_size=attention_block_size,
                    accumulation_dtype=accumulation_dtype,
                    key=next(keys),
                )
                for _ in self.conditioning_names
            )
            for _ in range(self.depth)
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
        self.interaction_attention = tuple(
            MeasureAwareAttention(key=next(keys), **attention_kwargs) for _ in groups
        )
        self.query_lifts = tuple(
            Linear(
                in_size=self.coord_dims[name],
                out_size=self.width,
                activation=jnn.gelu,
                key=next(keys),
            )
            for name in self.prediction_names
        )
        self.decoder_attention = tuple(
            MeasureAwareAttention(key=next(keys), **attention_kwargs)
            for _ in self.prediction_names
        )
        self.decoder_norms = tuple(
            eqx.nn.RMSNorm(self.width, eps=1e-6, use_bias=False)
            for _ in self.prediction_names
        )

        def output_size(name: str, /) -> int:
            output_spec = graph.branch(name).output_spec
            assert output_spec is not None
            return _get_size(output_spec.channels)

        self.projections = tuple(
            Linear(
                in_size=self.width,
                out_size=output_size(name),
                activation=None,
                key=next(keys),
            )
            for name in self.prediction_names
        )

    @property
    def operator_output_specs(self) -> dict[str, OperatorOutputSpec]:
        specs: dict[str, OperatorOutputSpec] = {}
        for name in self.prediction_names:
            output_spec = self.graph.branch(name).output_spec
            assert output_spec is not None
            specs[name] = output_spec
        return specs

    def _initial_branch_state(
        self,
        batch: OperatorBatch,
        name: str,
        index: int,
        /,
    ) -> EncodedOperatorState:
        spec = self.graph.branch(name)
        assert spec.source_name is not None
        source = batch.input(spec.source_name)
        values = _flatten_function_values(
            source, batch.case_shape, self.input_channels[name]
        )
        coordinates, weights, mask = _flatten_geometry(source, batch.case_shape)
        if int(coordinates.shape[-1]) != self.coord_dims[name]:
            raise ValueError(
                f"ABUPT branch {name!r} expected coordinate dimension "
                f"{self.coord_dims[name]}; got {coordinates.shape[-1]}."
            )
        features = self.source_lifts[index](
            jnp.concatenate((values, coordinates), axis=-1)
        )
        count = int(features.shape[-2])
        anchors = self.anchor_counts[name]
        if anchors > count:
            raise ValueError(
                f"ABUPT branch {name!r} cannot choose {anchors} anchors from {count} points."
            )
        selected = jnp.rint(jnp.linspace(0, count - 1, anchors)).astype(jnp.int32)
        anchor_values = jnp.take(features, selected, axis=-2)
        anchor_values = anchor_values + self.role_embeddings[index]
        anchor_coordinates = jnp.take(coordinates, selected, axis=-2)
        anchor_weights = jnp.take(weights, selected, axis=-1)
        anchor_mask = jnp.take(mask, selected, axis=-1)
        return EncodedOperatorState(
            kind="sampled_anchor",
            values=anchor_values * anchor_mask[..., None],
            coordinates=anchor_coordinates,
            weights=anchor_weights,
            mask=anchor_mask,
            case_shape=batch.case_shape,
            schema_fingerprint=operator_context_fingerprint(
                source, case_shape=batch.case_shape
            ),
        )

    def encode_inputs(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> BranchedEncodedOperatorState:
        del key
        state = BranchedEncodedOperatorState(
            {
                name: self._initial_branch_state(batch, name, index)
                for index, name in enumerate(self.conditioning_names)
            }
        )
        attention = dict(
            zip(self.interaction_groups, self.interaction_attention, strict=True)
        )
        for stage, blocks in enumerate(self.self_processors):
            replacements: dict[str, EncodedOperatorState] = {}
            for name, block in zip(self.conditioning_names, blocks, strict=True):
                branch = state.branch(name)
                updated = block(branch.values, branch.weights, branch.mask)
                replacements[name] = branch.replace_layers(
                    branch.layer_values + (updated,)
                )
            state = state.replace(replacements)
            state = apply_branch_interactions(state, self.graph, attention, stage)
        return state

    def decode_branch(
        self,
        state: BranchedEncodedOperatorState,
        branch_name: str,
        query: FunctionSamples,
        /,
    ) -> Array:
        if branch_name not in self.prediction_names:
            raise KeyError(
                f"Unknown ABUPT prediction branch {branch_name!r}; "
                f"expected {self.prediction_names}."
            )
        index = self.prediction_names.index(branch_name)
        branch = state.branch(branch_name)
        coordinates = query.coordinates_array(case_shape=state.case_shape, flatten=True)
        if int(coordinates.shape[-1]) != self.coord_dims[branch_name]:
            raise ValueError(
                f"ABUPT branch {branch_name!r} expected query coordinate dimension "
                f"{self.coord_dims[branch_name]}; got {coordinates.shape[-1]}."
            )
        cases = prod(state.case_shape) if state.case_shape else 1
        query_count = prod(query.sample_shape)
        query_mask = query.mask_array(case_shape=state.case_shape).reshape(
            (cases, query_count)
        )
        query_features = self.query_lifts[index](coordinates).reshape(
            (cases, query_count, self.width)
        )
        source = branch.values.reshape((cases, branch.num_tokens, self.width))
        attended = self.decoder_attention[index](
            source,
            query_features,
            branch.weights.reshape((cases, branch.num_tokens)),
            source_mask=branch.mask.reshape((cases, branch.num_tokens)),
            query_mask=query_mask,
        )
        decoded = _feature_norm(self.decoder_norms[index], query_features + attended)
        output = self.projections[index](decoded) * query_mask[..., None]
        spec = self.graph.branch(branch_name).output_spec
        assert spec is not None
        channels = _get_size(spec.channels)
        shaped = output.reshape(state.case_shape + query.sample_shape + (channels,))
        return shaped[..., 0] if spec.channels == "scalar" else shaped

    def decode_queries(
        self,
        state: BranchedEncodedOperatorState,
        queries: Mapping[str, FunctionSamples],
        /,
    ) -> frozendict[str, Array]:
        if set(queries) != set(self.prediction_names):
            raise ValueError("queries must cover every ABUPT prediction branch exactly.")
        return frozendict(
            {
                name: self.decode_branch(state, name, queries[name])
                for name in self.prediction_names
            }
        )

    def decode_query(
        self,
        state: BranchedEncodedOperatorState,
        query: FunctionSamples,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        del key
        return self.decode_branch(state, self.default_output_branch, query)

    def __call__(
        self,
        x: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("ABUPT requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = [
    "ABUPT",
    "LatentTokenBlock",
    "LatentTokenProcessor",
    "UPT",
]
