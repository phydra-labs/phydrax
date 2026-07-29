#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from ....graph import (
    batched_knn_graph,
    GraphNeuralOperator,
    GraphProcessor,
    RepeatedGraphProcessor,
)
from ..architectures._mlp import MLP
from ..core._keys import EvalKey
from ._dropout import Dropout


def _apply_rms_norm(norm: eqx.nn.RMSNorm, values: Array, /) -> Array:
    values = jnp.asarray(values)
    flattened = values.reshape((-1, int(values.shape[-1])))
    return jax.vmap(norm)(flattened).reshape(values.shape)


class _RegionalMessage(eqx.Module):
    model: MLP
    source_norm: eqx.nn.RMSNorm
    target_norm: eqx.nn.RMSNorm

    def __init__(
        self,
        channels: int,
        coord_dim: int,
        width: int,
        depth: int,
        /,
        *,
        activation: Callable[[Array], Array],
        key: Array,
    ):
        self.model = MLP(
            in_size=2 * int(channels) + int(coord_dim) + 1,
            out_size=int(channels),
            width_size=int(width),
            depth=int(depth),
            activation=activation,
            key=key,
        )
        self.source_norm = eqx.nn.RMSNorm(channels, eps=1e-6)
        self.target_norm = eqx.nn.RMSNorm(channels, eps=1e-6)

    def __call__(
        self,
        edges: Any,
        sent: Array,
        received: Array,
        globals_: Any,
        /,
    ) -> Array:
        del globals_
        relative = jnp.asarray(edges["scaled_relative"])
        distance = jnp.asarray(edges["distance"])
        features = jnp.concatenate(
            (
                _apply_rms_norm(self.source_norm, sent),
                _apply_rms_norm(self.target_norm, received),
                relative,
                distance,
            ),
            axis=-1,
        )
        return self.model(features)


class _RegionalUpdate(eqx.Module):
    model: MLP
    node_norm: eqx.nn.RMSNorm
    message_norm: eqx.nn.RMSNorm
    residual_scale: float = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        width: int,
        depth: int,
        /,
        *,
        activation: Callable[[Array], Array],
        residual_scale: float,
        key: Array,
    ):
        if float(residual_scale) <= 0.0:
            raise ValueError("residual_scale must be positive.")
        self.model = MLP(
            in_size=2 * int(channels),
            out_size=int(channels),
            width_size=int(width),
            depth=int(depth),
            activation=activation,
            key=key,
        )
        self.node_norm = eqx.nn.RMSNorm(channels, eps=1e-6)
        self.message_norm = eqx.nn.RMSNorm(channels, eps=1e-6)
        self.residual_scale = float(residual_scale)

    def __call__(
        self,
        nodes: Array,
        messages: Array,
        globals_: Any,
        /,
    ) -> Array:
        del globals_
        update = self.model(
            jnp.concatenate(
                (
                    _apply_rms_norm(self.node_norm, nodes),
                    _apply_rms_norm(self.message_norm, messages),
                ),
                axis=-1,
            )
        )
        return nodes + self.residual_scale * update


def _ones_like(values: Array, /) -> Array:
    return jnp.ones_like(values)


def _regional_block(
    channels: int,
    coord_dim: int,
    width: int,
    mlp_depth: int,
    /,
    *,
    activation: Callable[[Array], Array],
    residual_scale: float,
    key: Array,
) -> GraphNeuralOperator:
    message_key, update_key = jr.split(key)
    return GraphNeuralOperator(
        _RegionalMessage(
            channels,
            coord_dim,
            width,
            mlp_depth,
            activation=activation,
            key=message_key,
        ),
        source_fn=_ones_like,
        update_node_fn=_RegionalUpdate(
            channels,
            width,
            mlp_depth,
            activation=activation,
            residual_scale=residual_scale,
            key=update_key,
        ),
        input_key="features",
        output_key="features",
        edge_weight_key=None,
        source_measure_key="quadrature_weight",
        normalize=True,
    )


class RegionalGraphProcessor(eqx.Module):
    """Measure-aware message passing on a regional latent point cloud.

    Coordinates and graph topology are rebuilt from each operator batch with a
    deterministic, case-local K-nearest-neighbor graph. Regional quadrature
    enters every aggregation, and normalized weighted reductions keep the
    processor insensitive to a uniform rescaling of the measure.
    """

    processor: GraphProcessor | RepeatedGraphProcessor
    edge_dropout: Dropout
    channels: int = eqx.field(static=True)
    coord_dim: int = eqx.field(static=True)
    neighbors: int = eqx.field(static=True)
    radius: float | None = eqx.field(static=True)
    include_self: bool = eqx.field(static=True)
    target_chunk_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        coord_dim: int,
        /,
        *,
        neighbors: int = 8,
        depth: int = 4,
        width: int = 64,
        mlp_depth: int = 2,
        radius: float | None = None,
        include_self: bool = False,
        target_chunk_size: int | None = None,
        shared: bool = True,
        edge_dropout: float = 0.0,
        residual_scale: float = 1.0,
        activation: Callable[[Array], Array] = jnp.tanh,
        key: Array,
    ):
        if int(channels) <= 0 or int(coord_dim) <= 0:
            raise ValueError("channels and coord_dim must be positive.")
        if int(neighbors) <= 0:
            raise ValueError("neighbors must be positive.")
        if int(depth) <= 0:
            raise ValueError("depth must be positive.")
        if int(width) <= 0 or int(mlp_depth) <= 0:
            raise ValueError("width and mlp_depth must be positive.")
        if radius is not None and float(radius) <= 0.0:
            raise ValueError("radius must be positive when supplied.")
        if target_chunk_size is not None and int(target_chunk_size) <= 0:
            raise ValueError("target_chunk_size must be positive when supplied.")

        if shared:
            block = _regional_block(
                channels,
                coord_dim,
                width,
                mlp_depth,
                activation=activation,
                residual_scale=residual_scale,
                key=key,
            )
            processor: GraphProcessor | RepeatedGraphProcessor = RepeatedGraphProcessor(
                block,
                steps=int(depth),
            )
        else:
            keys = jr.split(key, int(depth))
            processor = GraphProcessor(
                tuple(
                    _regional_block(
                        channels,
                        coord_dim,
                        width,
                        mlp_depth,
                        activation=activation,
                        residual_scale=residual_scale,
                        key=block_key,
                    )
                    for block_key in keys
                )
            )
        self.processor = processor
        self.edge_dropout = Dropout("scalar", p=float(edge_dropout), mode="elementwise")
        self.channels = int(channels)
        self.coord_dim = int(coord_dim)
        self.neighbors = int(neighbors)
        self.radius = None if radius is None else float(radius)
        self.include_self = bool(include_self)
        self.target_chunk_size = (
            None if target_chunk_size is None else int(target_chunk_size)
        )

    def __call__(
        self,
        values: Array,
        coordinates: Array,
        measure: Array,
        mask: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        values = jnp.asarray(values)
        coordinates = jnp.asarray(coordinates, dtype=float)
        measure = jnp.asarray(measure, dtype=float)
        mask = jnp.asarray(mask, dtype=bool)
        if values.ndim < 2 or int(values.shape[-1]) != self.channels:
            raise ValueError(
                f"Regional values must end in (num_points, {self.channels})."
            )
        if coordinates.shape[:-1] != values.shape[:-1]:
            raise ValueError("Regional coordinates and values must share point axes.")
        if int(coordinates.shape[-1]) != self.coord_dim:
            raise ValueError(
                f"Regional coordinates must have dimension {self.coord_dim}."
            )
        if measure.shape != values.shape[:-1] or mask.shape != values.shape[:-1]:
            raise ValueError("Regional measure and mask must match the value point axes.")
        if self.neighbors > int(values.shape[-2]):
            raise ValueError("neighbors cannot exceed the regional latent point count.")

        graph = batched_knn_graph(
            coordinates,
            k=self.neighbors,
            node_mask=mask,
            node_features=values,
            node_measure=measure,
            radius=self.radius,
            include_self=self.include_self,
            target_chunk_size=self.target_chunk_size,
            validate=False,
        )
        edge_mask = jnp.asarray(graph.edge_mask, dtype=bool)
        keep = self.edge_dropout(jnp.ones(edge_mask.shape), key=key) > 0.0
        graph = graph.replace(edge_mask=edge_mask & keep, validate=False)
        output = self.processor(graph)
        if not isinstance(output.nodes, dict):
            raise TypeError(
                "Regional graph processor requires mapping-valued graph nodes."
            )
        features = jnp.asarray(output.nodes["features"]).reshape(values.shape)
        return features * mask[..., None].astype(features.dtype)


__all__ = ["RegionalGraphProcessor"]
