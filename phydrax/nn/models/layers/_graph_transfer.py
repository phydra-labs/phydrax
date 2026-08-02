#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod, sqrt
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ....graph._multigraph import query_target_features
from ....graph._neural_operators import GraphAttentionOperator, GraphNeuralOperator
from ....graph._query_batch import (
    batched_knn_query_graph,
    query_neighbors,
    QueryNeighborhood,
)
from ..architectures._mlp import MLP
from ..core._keys import EvalKey, split_eval_key
from ._linear import Linear


TransferReduction = Literal["integral", "normalized"]
MultiscaleFusion = Literal["concat", "gated"]


def _feature_array(
    name: str,
    value: Any,
    coordinates: Array,
    channels: int,
    /,
) -> Array:
    array = jnp.asarray(value)
    point_shape = coordinates.shape[:-1]
    if array.shape == point_shape and int(channels) == 1:
        return array[..., None]
    expected = point_shape + (int(channels),)
    if array.shape != expected:
        raise ValueError(
            f"{name} must have shape {point_shape} or {expected}; got {array.shape}."
        )
    return array


def _optional_feature_array(
    name: str,
    value: Any | None,
    coordinates: Array,
    channels: int,
    /,
) -> Array:
    if int(channels) == 0:
        if value is not None:
            raise ValueError(
                f"{name} was supplied but this transfer expects no channels."
            )
        return jnp.zeros(coordinates.shape[:-1] + (0,), dtype=coordinates.dtype)
    if value is None:
        raise ValueError(
            f"{name} is required because this transfer expects {channels} channels."
        )
    return _feature_array(name, value, coordinates, channels)


def _mask_array(value: Any | None, coordinates: Array, /) -> Array:
    shape = coordinates.shape[:-1]
    if value is None:
        return jnp.ones(shape, dtype=bool)
    mask = jnp.asarray(value, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"Point mask must have shape {shape}; got {mask.shape}.")
    return mask


def _measure_array(value: Any | None, coordinates: Array, /) -> Array | None:
    if value is None:
        return None
    measure = jnp.asarray(value, dtype=float)
    if measure.shape != coordinates.shape[:-1]:
        raise ValueError(
            f"Source measure must have shape {coordinates.shape[:-1]}; got {measure.shape}."
        )
    return measure


def _apply_rows(model: Any, values: Array, key: EvalKey, /) -> Array:
    leading = values.shape[:-1]
    flattened = values.reshape((-1, int(values.shape[-1])))
    output = jax.vmap(lambda row: model(row, key=key))(flattened)
    return jnp.asarray(output).reshape(leading + (int(jnp.asarray(output).shape[-1]),))


class _EdgeKernel(eqx.Module):
    model: MLP

    def __call__(self, edges, sent_nodes, received_nodes, globals_):
        del sent_nodes, globals_
        if not isinstance(edges, Mapping):
            raise TypeError("Graph transfer edge kernels require mapping-valued edges.")
        relative = jnp.asarray(edges["scaled_relative"])
        distance = jnp.linalg.norm(relative, axis=-1, keepdims=True)
        features = jnp.concatenate((relative, distance, received_nodes), axis=-1)
        return _apply_rows(self.model, features, None)


class GraphKernelTransfer(eqx.Module):
    """Learned measure-aware graph integral between two point sets."""

    source_lift: MLP
    target_lift: MLP
    edge_kernel: _EdgeKernel
    in_channels: int = eqx.field(static=True)
    target_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    coord_dim: int = eqx.field(static=True)
    neighbors: int = eqx.field(static=True)
    radius: float | None = eqx.field(static=True)
    reduction: TransferReduction = eqx.field(static=True)
    coordinate_scale: float = eqx.field(static=True)
    target_chunk_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        in_channels: int,
        target_channels: int = 0,
        out_channels: int,
        coord_dim: int,
        neighbors: int,
        radius: float | None = None,
        reduction: TransferReduction = "integral",
        width: int = 32,
        depth: int = 2,
        coordinate_scale: float = 1.0,
        target_chunk_size: int | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if int(in_channels) <= 0 or int(out_channels) <= 0 or int(coord_dim) <= 0:
            raise ValueError("Transfer channel counts and coord_dim must be positive.")
        if int(target_channels) < 0:
            raise ValueError("target_channels must be non-negative.")
        if int(neighbors) <= 0:
            raise ValueError("neighbors must be positive.")
        if radius is not None and float(radius) <= 0.0:
            raise ValueError("radius must be positive when supplied.")
        if reduction not in ("integral", "normalized"):
            raise ValueError("reduction must be 'integral' or 'normalized'.")
        if float(coordinate_scale) <= 0.0:
            raise ValueError("coordinate_scale must be positive.")
        if target_chunk_size is not None and int(target_chunk_size) <= 0:
            raise ValueError("target_chunk_size must be positive when supplied.")
        source_key, target_key, kernel_key = jr.split(key, 3)
        self.source_lift = MLP(
            in_size=int(in_channels) + int(coord_dim),
            out_size=int(out_channels),
            width_size=int(width),
            depth=int(depth),
            key=source_key,
        )
        self.target_lift = MLP(
            in_size=int(target_channels) + int(coord_dim),
            out_size=int(out_channels),
            width_size=int(width),
            depth=int(depth),
            key=target_key,
        )
        self.edge_kernel = _EdgeKernel(
            MLP(
                in_size=int(coord_dim) + 1 + int(out_channels),
                out_size=int(out_channels),
                width_size=int(width),
                depth=int(depth),
                key=kernel_key,
            )
        )
        self.in_channels = int(in_channels)
        self.target_channels = int(target_channels)
        self.out_channels = int(out_channels)
        self.coord_dim = int(coord_dim)
        self.neighbors = int(neighbors)
        self.radius = None if radius is None else float(radius)
        self.reduction = reduction
        self.coordinate_scale = float(coordinate_scale)
        self.target_chunk_size = (
            None if target_chunk_size is None else int(target_chunk_size)
        )

    def __call__(
        self,
        source_values: Any,
        source_coordinates: Any,
        target_coordinates: Any,
        /,
        *,
        source_measure: Any | None,
        source_mask: Any | None = None,
        target_mask: Any | None = None,
        target_features: Any | None = None,
        key: EvalKey = None,
    ) -> Array:
        source = jnp.asarray(source_coordinates, dtype=float)
        target = jnp.asarray(target_coordinates, dtype=float)
        if (
            int(source.shape[-1]) != self.coord_dim
            or int(target.shape[-1]) != self.coord_dim
        ):
            raise ValueError(
                "Source and target coordinate dimensions must match coord_dim."
            )
        values = _feature_array("source_values", source_values, source, self.in_channels)
        target_values = _optional_feature_array(
            "target_features", target_features, target, self.target_channels
        )
        source_valid = _mask_array(source_mask, source)
        target_valid = _mask_array(target_mask, target)
        measure = _measure_array(source_measure, source)
        if self.reduction == "integral" and measure is None:
            raise ValueError("Integral graph transfer requires explicit source_measure.")
        source_key, target_key = split_eval_key(key, 2)
        lifted = _apply_rows(
            self.source_lift,
            jnp.concatenate((values, source / self.coordinate_scale), axis=-1),
            source_key,
        )
        target_lifted = _apply_rows(
            self.target_lift,
            jnp.concatenate(
                (target_values, target / self.coordinate_scale),
                axis=-1,
            ),
            target_key,
        )
        query = batched_knn_query_graph(
            source,
            target,
            k=self.neighbors,
            source_mask=source_valid,
            target_mask=target_valid,
            source_features=lifted,
            target_features=target_lifted,
            source_measure=measure,
            radius=self.radius,
            target_chunk_size=self.target_chunk_size,
            validate=False,
        )
        operator = GraphNeuralOperator(
            self.edge_kernel,
            input_key="features",
            output_key="transfer",
            edge_weight_key=None,
            source_measure_key="quadrature_weight" if measure is not None else None,
            normalize=self.reduction == "normalized",
            target_node_type=query.target_type,
        )
        output = query_target_features(operator(query.graph), query, "transfer")
        case_shape = tuple(int(size) for size in target.shape[:-2])
        return output.reshape(case_shape + (int(target.shape[-2]), self.out_channels))


class _MultiheadLogits(eqx.Module):
    edge_bias: MLP
    heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __call__(self, edges, keys, queries, globals_):
        del globals_
        if not isinstance(edges, Mapping):
            raise TypeError("Graph attention transfer requires mapping-valued edges.")
        key_heads = keys.reshape((keys.shape[0], self.heads, self.head_dim))
        query_heads = queries.reshape((queries.shape[0], self.heads, self.head_dim))
        logits = jnp.sum(key_heads * query_heads, axis=-1) / sqrt(float(self.head_dim))
        relative = jnp.asarray(edges["scaled_relative"])
        distance = jnp.linalg.norm(relative, axis=-1, keepdims=True)
        bias = _apply_rows(
            self.edge_bias,
            jnp.concatenate((relative, distance), axis=-1),
            None,
        )
        return logits + bias


class _AttentionOutput(eqx.Module):
    projection: Linear

    def __call__(self, nodes, aggregated, globals_):
        del nodes, globals_
        return self.projection(aggregated)


class GraphAttentionTransfer(eqx.Module):
    """Quadrature-aware multihead attention between point sets."""

    source_lift: MLP
    target_lift: MLP
    query_projection: Linear
    key_projection: Linear
    value_projection: Linear
    logits: _MultiheadLogits
    output: _AttentionOutput
    in_channels: int = eqx.field(static=True)
    target_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    coord_dim: int = eqx.field(static=True)
    node_width: int = eqx.field(static=True)
    heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    neighbors: int = eqx.field(static=True)
    radius: float | None = eqx.field(static=True)
    coordinate_scale: float = eqx.field(static=True)
    require_measure: bool = eqx.field(static=True)
    target_chunk_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        coord_dim: int,
        neighbors: int,
        target_channels: int = 0,
        radius: float | None = None,
        node_width: int = 32,
        heads: int = 4,
        head_dim: int | None = None,
        depth: int = 2,
        coordinate_scale: float = 1.0,
        require_measure: bool = True,
        target_chunk_size: int | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if (
            min(
                int(in_channels),
                int(out_channels),
                int(coord_dim),
                int(neighbors),
                int(node_width),
                int(heads),
            )
            <= 0
        ):
            raise ValueError("Attention transfer dimensions must be positive.")
        if int(target_channels) < 0:
            raise ValueError("target_channels must be non-negative.")
        if radius is not None and float(radius) <= 0.0:
            raise ValueError("radius must be positive when supplied.")
        resolved_head_dim = (
            max(1, int(node_width) // int(heads)) if head_dim is None else int(head_dim)
        )
        if resolved_head_dim <= 0 or float(coordinate_scale) <= 0.0:
            raise ValueError("head_dim and coordinate_scale must be positive.")
        keys = jr.split(key, 8)
        self.source_lift = MLP(
            in_size=int(in_channels) + int(coord_dim),
            out_size=int(node_width),
            width_size=int(node_width),
            depth=int(depth),
            key=keys[0],
        )
        self.target_lift = MLP(
            in_size=int(target_channels) + int(coord_dim),
            out_size=int(node_width),
            width_size=int(node_width),
            depth=int(depth),
            key=keys[1],
        )
        attention_width = int(heads) * resolved_head_dim
        self.query_projection = Linear(
            in_size=int(node_width), out_size=attention_width, key=keys[2]
        )
        self.key_projection = Linear(
            in_size=int(node_width), out_size=attention_width, key=keys[3]
        )
        self.value_projection = Linear(
            in_size=int(node_width), out_size=int(out_channels), key=keys[4]
        )
        self.logits = _MultiheadLogits(
            MLP(
                in_size=int(coord_dim) + 1,
                out_size=int(heads),
                width_size=int(node_width),
                depth=int(depth),
                key=keys[5],
            ),
            heads=int(heads),
            head_dim=resolved_head_dim,
        )
        self.output = _AttentionOutput(
            Linear(
                in_size=int(heads) * int(out_channels),
                out_size=int(out_channels),
                key=keys[6],
            )
        )
        self.in_channels = int(in_channels)
        self.target_channels = int(target_channels)
        self.out_channels = int(out_channels)
        self.coord_dim = int(coord_dim)
        self.node_width = int(node_width)
        self.heads = int(heads)
        self.head_dim = resolved_head_dim
        self.neighbors = int(neighbors)
        self.radius = None if radius is None else float(radius)
        self.coordinate_scale = float(coordinate_scale)
        self.require_measure = bool(require_measure)
        self.target_chunk_size = (
            None if target_chunk_size is None else int(target_chunk_size)
        )

    def __call__(
        self,
        source_values: Any,
        source_coordinates: Any,
        target_coordinates: Any,
        /,
        *,
        source_measure: Any | None,
        source_mask: Any | None = None,
        target_mask: Any | None = None,
        target_features: Any | None = None,
        key: EvalKey = None,
    ) -> Array:
        source = jnp.asarray(source_coordinates, dtype=float)
        target = jnp.asarray(target_coordinates, dtype=float)
        if (
            int(source.shape[-1]) != self.coord_dim
            or int(target.shape[-1]) != self.coord_dim
        ):
            raise ValueError(
                "Source and target coordinate dimensions must match coord_dim."
            )
        source_values_ = _feature_array(
            "source_values", source_values, source, self.in_channels
        )
        target_values = _optional_feature_array(
            "target_features", target_features, target, self.target_channels
        )
        source_valid = _mask_array(source_mask, source)
        target_valid = _mask_array(target_mask, target)
        measure = _measure_array(source_measure, source)
        if self.require_measure and measure is None:
            raise ValueError("Measure-aware attention requires explicit source_measure.")
        source_key, target_key = split_eval_key(key, 2)
        source_nodes = _apply_rows(
            self.source_lift,
            jnp.concatenate((source_values_, source / self.coordinate_scale), axis=-1),
            source_key,
        )
        target_nodes = _apply_rows(
            self.target_lift,
            jnp.concatenate((target_values, target / self.coordinate_scale), axis=-1),
            target_key,
        )
        query = batched_knn_query_graph(
            source,
            target,
            k=self.neighbors,
            source_mask=source_valid,
            target_mask=target_valid,
            source_features=source_nodes,
            target_features=target_nodes,
            source_measure=measure,
            radius=self.radius,
            target_chunk_size=self.target_chunk_size,
            validate=False,
        )
        operator = GraphAttentionOperator(
            query_fn=self.query_projection,
            key_fn=self.key_projection,
            value_fn=self.value_projection,
            logit_fn=self.logits,
            update_node_fn=self.output,
            input_key="features",
            output_key="transfer",
            source_measure_key="quadrature_weight" if measure is not None else None,
            target_node_type=query.target_type,
        )
        output = query_target_features(operator(query.graph), query, "transfer")
        case_shape = tuple(int(size) for size in target.shape[:-2])
        return output.reshape(case_shape + (int(target.shape[-2]), self.out_channels))


class GeometryMomentEmbedding(eqx.Module):
    """Measure-weighted, dimensionless local neighborhood statistics."""

    coord_dim: int = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    reference_measure: float = eqx.field(static=True)

    def __init__(
        self,
        coord_dim: int,
        radius: float,
        /,
        *,
        reference_measure: float = 1.0,
    ):
        if int(coord_dim) <= 0 or float(radius) <= 0.0:
            raise ValueError("coord_dim and radius must be positive.")
        if float(reference_measure) <= 0.0:
            raise ValueError("reference_measure must be positive.")
        self.coord_dim = int(coord_dim)
        self.radius = float(radius)
        self.reference_measure = float(reference_measure)

    @property
    def out_size(self) -> int:
        return 2 * self.coord_dim + 5

    def __call__(
        self,
        neighborhood: QueryNeighborhood,
        source_measure: Array,
        /,
    ) -> Array:
        measure = jnp.asarray(source_measure, dtype=float)
        if measure.ndim != 2:
            raise ValueError(
                "GeometryMomentEmbedding source_measure must have shape (case, source)."
            )
        cases, targets, neighbors = neighborhood.indices.shape
        expanded = jnp.broadcast_to(
            measure[:, None, :],
            (cases, targets, int(measure.shape[-1])),
        )
        selected = jnp.take_along_axis(expanded, neighborhood.indices, axis=2)
        selected = selected * neighborhood.mask.astype(selected.dtype)
        mass = jnp.sum(selected, axis=-1, keepdims=True)
        normalized = jnp.where(mass > 0.0, selected / mass, jnp.zeros_like(selected))
        relative = neighborhood.relative / self.radius
        mean = oe.contract("ctn,ctnd->ctd", normalized, relative)
        centered = relative - mean[..., None, :]
        covariance = oe.contract(
            "ctn,ctni,ctnj->ctij",
            normalized,
            centered,
            centered,
        )
        eigenvalues = jnp.linalg.eigvalsh(covariance)
        distance = neighborhood.distance / self.radius
        mean_distance = jnp.sum(normalized * distance, axis=-1, keepdims=True)
        rms_distance = jnp.sqrt(jnp.sum(normalized * distance**2, axis=-1, keepdims=True))
        minimum = jnp.min(
            jnp.where(neighborhood.mask, distance, jnp.inf),
            axis=-1,
            keepdims=True,
        )
        coverage = jnp.any(neighborhood.mask, axis=-1, keepdims=True)
        minimum = jnp.where(coverage, minimum, 0.0)
        return jnp.concatenate(
            (
                mass / self.reference_measure,
                mean,
                eigenvalues,
                minimum,
                mean_distance,
                rms_distance,
                coverage.astype(relative.dtype),
            ),
            axis=-1,
        )


class MultiscaleGraphTransfer(eqx.Module):
    """Fuse several graph transfers using multiscale geometry statistics."""

    transfers: tuple[GraphKernelTransfer | GraphAttentionTransfer, ...]
    embeddings: tuple[GeometryMomentEmbedding, ...]
    mixer: MLP | None
    gate: MLP | None
    fusion: MultiscaleFusion = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    target_channels: int = eqx.field(static=True)
    coord_dim: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(
        self,
        transfers: Sequence[GraphKernelTransfer | GraphAttentionTransfer],
        /,
        *,
        fusion: MultiscaleFusion = "concat",
        reference_measure: float = 1.0,
        width: int = 64,
        depth: int = 2,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        transfers_ = tuple(transfers)
        if not transfers_:
            raise ValueError("MultiscaleGraphTransfer requires at least one scale.")
        channels = transfers_[0].out_channels
        coord_dim = transfers_[0].coord_dim
        in_channels = transfers_[0].in_channels
        target_channels = transfers_[0].target_channels
        if any(transfer.in_channels != in_channels for transfer in transfers_[1:]):
            raise ValueError("All multiscale transfers must share in_channels.")
        if any(
            transfer.target_channels != target_channels for transfer in transfers_[1:]
        ):
            raise ValueError("All multiscale transfers must share target_channels.")
        if any(transfer.out_channels != channels for transfer in transfers_[1:]):
            raise ValueError("All multiscale transfers must share out_channels.")
        if any(transfer.coord_dim != coord_dim for transfer in transfers_[1:]):
            raise ValueError("All multiscale transfers must share coord_dim.")
        radii: list[float] = []
        for transfer in transfers_:
            if transfer.radius is None:
                raise ValueError("Every multiscale transfer requires an explicit radius.")
            radii.append(float(transfer.radius))
        if fusion not in ("concat", "gated"):
            raise ValueError("fusion must be 'concat' or 'gated'.")
        embeddings = tuple(
            GeometryMomentEmbedding(
                coord_dim,
                radius,
                reference_measure=reference_measure,
            )
            for radius in radii
        )
        geometry_size = sum(embedding.out_size for embedding in embeddings)
        value_key, gate_key = jr.split(key)
        if fusion == "concat":
            mixer = MLP(
                in_size=len(transfers_) * channels + geometry_size,
                out_size=channels,
                width_size=int(width),
                depth=int(depth),
                key=value_key,
            )
            gate = None
        else:
            mixer = None
            gate = MLP(
                in_size=geometry_size,
                out_size=len(transfers_),
                width_size=int(width),
                depth=int(depth),
                key=gate_key,
            )
        self.transfers = transfers_
        self.embeddings = embeddings
        self.mixer = mixer
        self.gate = gate
        self.fusion = fusion
        self.out_channels = channels
        self.in_channels = in_channels
        self.target_channels = target_channels
        self.coord_dim = coord_dim

    def _geometry(
        self,
        source_coordinates: Array,
        target_coordinates: Array,
        source_measure: Array,
        source_mask: Array,
        target_mask: Array,
        /,
    ) -> Array:
        cases = prod(source_coordinates.shape[:-2]) if source_coordinates.ndim > 2 else 1
        source = source_coordinates.reshape(
            (cases, int(source_coordinates.shape[-2]), int(source_coordinates.shape[-1]))
        )
        target = target_coordinates.reshape(
            (cases, int(target_coordinates.shape[-2]), int(target_coordinates.shape[-1]))
        )
        measure = source_measure.reshape((cases, int(source_measure.shape[-1])))
        source_valid = source_mask.reshape((cases, int(source_mask.shape[-1])))
        target_valid = target_mask.reshape((cases, int(target_mask.shape[-1])))
        features = []
        for transfer, embedding in zip(self.transfers, self.embeddings, strict=True):
            neighborhood = query_neighbors(
                source,
                target,
                source_mask=source_valid,
                target_mask=target_valid,
                max_neighbors=transfer.neighbors,
                radius=transfer.radius,
                target_chunk_size=transfer.target_chunk_size,
            )
            features.append(embedding(neighborhood, measure))
        return jnp.concatenate(features, axis=-1).reshape(
            target_coordinates.shape[:-1] + (-1,)
        )

    def scale_weights(
        self,
        source_coordinates: Any,
        target_coordinates: Any,
        /,
        *,
        source_measure: Any,
        source_mask: Any | None = None,
        target_mask: Any | None = None,
    ) -> Array:
        if self.gate is None:
            raise ValueError("scale_weights is only available for gated fusion.")
        source = jnp.asarray(source_coordinates, dtype=float)
        target = jnp.asarray(target_coordinates, dtype=float)
        measure = _measure_array(source_measure, source)
        if measure is None:
            raise ValueError("Multiscale geometry requires source_measure.")
        geometry = self._geometry(
            source,
            target,
            measure,
            _mask_array(source_mask, source),
            _mask_array(target_mask, target),
        )
        return jax.nn.softmax(_apply_rows(self.gate, geometry, None), axis=-1)

    def __call__(
        self,
        source_values: Any,
        source_coordinates: Any,
        target_coordinates: Any,
        /,
        *,
        source_measure: Any,
        source_mask: Any | None = None,
        target_mask: Any | None = None,
        target_features: Any | None = None,
        key: EvalKey = None,
    ) -> Array:
        source = jnp.asarray(source_coordinates, dtype=float)
        target = jnp.asarray(target_coordinates, dtype=float)
        measure = _measure_array(source_measure, source)
        if measure is None:
            raise ValueError("Multiscale graph transfer requires source_measure.")
        source_valid = _mask_array(source_mask, source)
        target_valid = _mask_array(target_mask, target)
        keys = split_eval_key(key, len(self.transfers) + 1)
        outputs = tuple(
            transfer(
                source_values,
                source,
                target,
                source_measure=measure,
                source_mask=source_valid,
                target_mask=target_valid,
                target_features=target_features,
                key=keys[index],
            )
            for index, transfer in enumerate(self.transfers)
        )
        geometry = self._geometry(
            source,
            target,
            measure,
            source_valid,
            target_valid,
        )
        if self.fusion == "concat":
            if self.mixer is None:
                raise RuntimeError("Concat multiscale transfer has no mixer.")
            return _apply_rows(
                self.mixer,
                jnp.concatenate((*outputs, geometry), axis=-1),
                keys[-1],
            )
        if self.gate is None:
            raise RuntimeError("Gated multiscale transfer has no gate.")
        weights = jax.nn.softmax(_apply_rows(self.gate, geometry, keys[-1]), axis=-1)
        stacked = jnp.stack(outputs, axis=-2)
        return jnp.sum(stacked * weights[..., None], axis=-2)


__all__ = [
    "GeometryMomentEmbedding",
    "GraphAttentionTransfer",
    "GraphKernelTransfer",
    "MultiscaleFusion",
    "MultiscaleGraphTransfer",
    "TransferReduction",
]
