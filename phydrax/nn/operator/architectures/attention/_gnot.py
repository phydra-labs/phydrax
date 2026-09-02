#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import cast, Literal

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

import phydrax.ein as ein
from phydrax._doc import DOC_KEY0
from phydrax.nn._keys import EvalKey
from phydrax.nn._utils import _get_size
from phydrax.nn.layers._linear import Linear
from phydrax.nn.layers._measure_attention import (
    AttentionExecution,
    AttentionKernel,
    MeasureAwareAttention,
)
from phydrax.nn.models._mlp import MLP
from phydrax.nn.operator.architectures.attention._upt import LatentTokenProcessor
from phydrax.nn.operator.architectures.geometric._geometry_operator import (
    _sample_coordinates,
    _sample_mask,
    _sample_values,
)
from phydrax.nn.operator.data import OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel


class GNOT(AbstractOperatorModel):
    """Heterogeneous geometry-aware neural operator transformer.

    Every named source has its own value/coordinate encoder and measure-aware
    cross-attention path to the query points. Branches are fused by learned
    query-local gates conditioned on measure-weighted source summaries, then
    refined by a quadrature-aware transformer over the query geometry.
    """

    operator_architecture = "GNOT"

    source_encoders: tuple[MLP, ...]
    source_attentions: tuple[MeasureAwareAttention, ...]
    fusion_gates: tuple[MLP, ...]
    query_encoder: MLP
    processor: LatentTokenProcessor
    projection: Linear
    in_size: tuple[int, ...]
    out_size: int | Literal["scalar"]
    source_keys: tuple[str, ...] = eqx.field(static=True)
    source_channels: tuple[int, ...] = eqx.field(static=True)
    coord_dim: int = eqx.field(static=True)
    query_channels: int = eqx.field(static=True)
    hidden_channels: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        in_channels: Mapping[str, int | Literal["scalar"]],
        out_channels: int | Literal["scalar"] = "scalar",
        coord_dim: int,
        query_channels: int = 0,
        hidden_channels: int = 64,
        encoder_width: int | None = None,
        encoder_depth: int = 2,
        fusion_width: int | None = None,
        fusion_depth: int = 2,
        transformer_depth: int = 4,
        num_heads: int = 8,
        head_dim: int | None = None,
        feed_forward_multiplier: float = 4.0,
        attention_kernel: AttentionKernel = "softmax",
        attention_execution: AttentionExecution = "auto",
        attention_block_size: int = 256,
        accumulation_dtype: str = "input",
        norm_eps: float = 1e-6,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(in_channels, Mapping) or not in_channels:
            raise ValueError("GNOT in_channels must be a non-empty named mapping.")
        source_items = tuple(
            sorted(
                (
                    (str(name), _get_size(channels))
                    for name, channels in cast(
                        Mapping[str, int | Literal["scalar"]], in_channels
                    ).items()
                ),
                key=lambda item: item[0],
            )
        )
        source_keys = tuple(name for name, _ in source_items)
        if len(set(source_keys)) != len(source_keys):
            raise ValueError("GNOT source names must be unique after string conversion.")
        source_channels = tuple(channels for _, channels in source_items)
        coordinate_width = int(coord_dim)
        query_width = int(query_channels)
        hidden_width = int(hidden_channels)
        encoder_hidden = hidden_width if encoder_width is None else int(encoder_width)
        fusion_hidden = hidden_width if fusion_width is None else int(fusion_width)
        if coordinate_width <= 0:
            raise ValueError("coord_dim must be positive.")
        if query_width < 0:
            raise ValueError("query_channels must be non-negative.")
        if min(hidden_width, encoder_hidden, fusion_hidden) <= 0:
            raise ValueError("GNOT hidden widths must be positive.")
        if min(int(encoder_depth), int(fusion_depth), int(transformer_depth)) <= 0:
            raise ValueError(
                "GNOT encoder, fusion, and transformer depths must be positive."
            )
        if int(num_heads) <= 0:
            raise ValueError("num_heads must be positive.")
        resolved_head_dim = (
            hidden_width // int(num_heads) if head_dim is None else int(head_dim)
        )
        if resolved_head_dim <= 0:
            raise ValueError("head_dim must be positive.")
        if head_dim is None and hidden_width % int(num_heads) != 0:
            raise ValueError(
                "hidden_channels must be divisible by num_heads when head_dim is omitted."
            )

        keys = iter(jr.split(key, 3 * len(source_items) + 3))
        self.source_encoders = tuple(
            MLP(
                in_size=channels + coordinate_width,
                out_size=hidden_width,
                width_size=encoder_hidden,
                depth=int(encoder_depth),
                key=next(keys),
            )
            for _, channels in source_items
        )
        self.source_attentions = tuple(
            MeasureAwareAttention(
                source_channels=hidden_width,
                query_channels=hidden_width,
                out_channels=hidden_width,
                num_heads=int(num_heads),
                head_dim=resolved_head_dim,
                kernel=attention_kernel,
                execution=attention_execution,
                block_size=int(attention_block_size),
                accumulation_dtype=accumulation_dtype,
                key=next(keys),
            )
            for _ in source_items
        )
        self.fusion_gates = tuple(
            MLP(
                in_size=2 * hidden_width,
                out_size="scalar",
                width_size=fusion_hidden,
                depth=int(fusion_depth),
                key=next(keys),
            )
            for _ in source_items
        )
        self.query_encoder = MLP(
            in_size=coordinate_width + query_width,
            out_size=hidden_width,
            width_size=encoder_hidden,
            depth=int(encoder_depth),
            key=next(keys),
        )
        self.processor = LatentTokenProcessor(
            hidden_width,
            int(transformer_depth),
            num_heads=int(num_heads),
            head_dim=resolved_head_dim,
            feed_forward_multiplier=float(feed_forward_multiplier),
            kernel=attention_kernel,
            execution=attention_execution,
            block_size=int(attention_block_size),
            accumulation_dtype=accumulation_dtype,
            norm_eps=float(norm_eps),
            key=next(keys),
        )
        self.projection = Linear(
            in_size=hidden_width,
            out_size=_get_size(out_channels),
            activation=None,
            key=next(keys),
        )
        self.in_size = source_channels
        self.out_size = out_channels
        self.source_keys = source_keys
        self.source_channels = source_channels
        self.coord_dim = coordinate_width
        self.query_channels = query_width
        self.hidden_channels = hidden_width

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        del key
        case_shape = batch.case_shape
        cases = prod(case_shape) if case_shape else 1
        query_count = prod(batch.require_single_query().sample_shape)
        query_coordinates = _sample_coordinates(batch.require_single_query(), case_shape)
        if int(query_coordinates.shape[-1]) != self.coord_dim:
            raise ValueError("GNOT query coordinate dimension does not match coord_dim.")
        query_mask = _sample_mask(batch.require_single_query(), case_shape)
        query_measure = (
            batch.require_single_query()
            .quadrature(case_shape=case_shape)
            .reshape((cases, query_count))
        )
        if self.query_channels == 0:
            query_inputs = query_coordinates
        else:
            query_values = _sample_values(
                batch.require_single_query(),
                case_shape,
                self.query_channels,
                name="query covariates",
            )
            query_inputs = jnp.concatenate((query_coordinates, query_values), axis=-1)
        query_features = self.query_encoder(query_inputs).reshape(
            (cases, query_count, self.hidden_channels)
        )
        flat_query_mask = query_mask.reshape((cases, query_count))

        branch_values: list[Array] = []
        branch_logits: list[Array] = []
        for name, channels, encoder, attention, gate in zip(
            self.source_keys,
            self.source_channels,
            self.source_encoders,
            self.source_attentions,
            self.fusion_gates,
            strict=True,
        ):
            source = batch.input(name)
            source_count = prod(source.sample_shape)
            source_coordinates = _sample_coordinates(source, case_shape)
            if int(source_coordinates.shape[-1]) != self.coord_dim:
                raise ValueError(
                    f"GNOT source {name!r} coordinate dimension does not match coord_dim."
                )
            source_values = _sample_values(
                source,
                case_shape,
                channels,
                name=f"source {name!r}",
            )
            source_features = encoder(
                jnp.concatenate((source_coordinates, source_values), axis=-1)
            ).reshape((cases, source_count, self.hidden_channels))
            source_mask = _sample_mask(source, case_shape).reshape((cases, source_count))
            source_measure = source.quadrature(case_shape=case_shape).reshape(
                (cases, source_count)
            )
            attended = attention(
                source_features,
                query_features,
                source_measure,
                source_mask=source_mask,
                query_mask=flat_query_mask,
            )
            normalized_weights = source.weights(
                normalized=True, case_shape=case_shape
            ).reshape((cases, source_count))
            summary = ein.contract("bs,bsw->bw", normalized_weights, source_features)
            summary = jnp.broadcast_to(
                summary[:, None, :],
                (cases, query_count, self.hidden_channels),
            )
            logits = gate(jnp.concatenate((query_features, summary), axis=-1))
            branch_values.append(attended)
            branch_logits.append(logits)

        stacked_values = jnp.stack(branch_values, axis=-2)
        gates = jnn.softmax(jnp.stack(branch_logits, axis=-1), axis=-1)
        fused = query_features + jnp.sum(stacked_values * gates[..., None], axis=-2)
        refined = jnp.asarray(self.processor(fused, query_measure, flat_query_mask))
        output = self.projection(refined) * flat_query_mask[..., None]
        shaped = output.reshape(
            case_shape
            + batch.require_single_query().sample_shape
            + (int(jnp.asarray(output).shape[-1]),)
        )
        if self.out_size == "scalar":
            return shaped[..., 0]
        return shaped

    def __call__(
        self,
        x: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("GNOT requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = ["GNOT"]
