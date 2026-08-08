#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from math import prod
from typing import Any, ClassVar, Literal

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from phydrax._doc import DOC_KEY0
from phydrax._frozendict import frozendict
from phydrax._strict import StrictModule
from phydrax.geometry.operator import TensorGridLatentGeometry
from phydrax.nn._keys import EvalKey
from phydrax.nn._utils import _get_size
from phydrax.nn.layers._linear import Linear
from phydrax.nn.layers._measure_attention import (
    AttentionExecution,
    AttentionKernel,
    MeasureAwareAttention,
)
from phydrax.nn.operator.architectures.spectral._fno import Factorization, SpectralConvND
from phydrax.nn.operator.data import (
    FunctionSamples,
    OperatorBatch,
    OperatorFieldBatch,
    OperatorOutputSpec,
    OperatorPrediction,
)
from phydrax.nn.operator.encoded import AbstractEncodedOperatorModel
from phydrax.nn.operator.field import OperatorFieldSpec
from phydrax.nn.operator.layers._attention import CodomainAttention


def _named_key(key: Key[Array, ""], label: str, /) -> Key[Array, ""]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return jr.fold_in(key, int.from_bytes(digest[:4], "little"))


def _feature_norm(norm: eqx.nn.RMSNorm, values: Array, /) -> Array:
    array = jnp.asarray(values)
    flattened = array.reshape((-1, int(array.shape[-1])))
    return jax.vmap(norm)(flattened).reshape(array.shape)


def _field_values(
    field: OperatorFieldSpec,
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    /,
) -> Array:
    if samples.values is None:
        raise ValueError(f"CoDA-NO field {field.name!r} requires sampled values.")
    values = samples.values
    sample_shape = samples.sample_shape
    count = prod(sample_shape)
    scalar_shape = case_shape + sample_shape
    if field.channels == "scalar" and tuple(values.shape) == scalar_shape:
        normalized = field.nondimensionalize(values)
        return normalized.reshape(case_shape + (count, 1))
    expected = scalar_shape + (field.channel_count,)
    if tuple(int(size) for size in values.shape) != expected:
        raise ValueError(
            f"CoDA-NO field {field.name!r} values must have shape {expected}; "
            f"got {values.shape}."
        )
    normalized = field.nondimensionalize(values)
    return normalized.reshape(case_shape + (count, field.channel_count))


class CoDAOperatorState(StrictModule):
    """Processed common-grid field tokens retained for independent decoding."""

    values: Array
    field_mask: Array
    coordinates: Array
    quadrature_weights: Array
    case_shape: tuple[int, ...]
    layer_values: tuple[Array, ...]

    def __init__(
        self,
        *,
        values: Array,
        field_mask: Array,
        coordinates: Array,
        quadrature_weights: Array,
        case_shape: Sequence[int],
        layer_values: Sequence[Array] = (),
    ):
        values_ = jnp.asarray(values)
        cases = tuple(int(size) for size in case_shape)
        if values_.ndim < len(cases) + 3:
            raise ValueError(
                "CoDA state values require case, spatial, field, and channel axes."
            )
        field_mask_ = jnp.asarray(field_mask, dtype=bool)
        if field_mask_.shape != cases + (int(values_.shape[-2]),):
            raise ValueError("CoDA state field_mask must have case_shape + (fields,).")
        coordinates_ = jnp.asarray(coordinates)
        if coordinates_.shape[: len(cases)] != cases:
            raise ValueError("CoDA state coordinates must begin with case_shape.")
        weights_ = jnp.asarray(quadrature_weights)
        if weights_.shape != coordinates_.shape[:-1]:
            raise ValueError("CoDA state quadrature weights must match coordinates.")
        self.values = values_
        self.field_mask = field_mask_
        self.coordinates = coordinates_
        self.quadrature_weights = weights_
        self.case_shape = cases
        layers = tuple(jnp.asarray(layer) for layer in layer_values)
        if any(layer.shape != values_.shape for layer in layers):
            raise ValueError("Every CoDA state layer must match final values.")
        self.layer_values = layers if layers else (values_,)


class CoDABlock(StrictModule):
    """Alternating within-field domain mixing and cross-field codomain attention."""

    domain_mixer: SpectralConvND
    pointwise: Linear
    codomain_attention: CodomainAttention
    domain_norm: eqx.nn.RMSNorm
    codomain_norm: eqx.nn.RMSNorm
    feed_forward_norm: eqx.nn.RMSNorm
    gate: Linear
    value: Linear
    output: Linear
    width: int
    spatial_ndim: int

    def __init__(
        self,
        *,
        width: int,
        spatial_ndim: int,
        n_modes: int | Sequence[int],
        num_heads: int,
        head_dim: int,
        feed_forward_multiplier: float,
        factorization: Factorization,
        rank: int | float,
        key: Key[Array, ""],
    ):
        self.width = int(width)
        self.spatial_ndim = int(spatial_ndim)
        hidden = round(float(feed_forward_multiplier) * self.width)
        if min(self.width, self.spatial_ndim, hidden) <= 0:
            raise ValueError("CoDA block dimensions must be positive.")
        keys = jr.split(key, 6)
        self.domain_mixer = SpectralConvND(
            in_channels=self.width,
            out_channels=self.width,
            n_modes=n_modes,
            factorization=factorization,
            rank=rank,
            key=keys[0],
        )
        self.pointwise = Linear(
            in_size=self.width,
            out_size=self.width,
            activation=None,
            key=keys[1],
        )
        self.codomain_attention = CodomainAttention(
            channels=self.width,
            num_heads=num_heads,
            head_dim=head_dim,
            key=keys[2],
        )
        self.domain_norm = eqx.nn.RMSNorm(self.width, eps=1e-6, use_bias=False)
        self.codomain_norm = eqx.nn.RMSNorm(self.width, eps=1e-6, use_bias=False)
        self.feed_forward_norm = eqx.nn.RMSNorm(self.width, eps=1e-6, use_bias=False)
        self.gate = Linear(
            in_size=self.width,
            out_size=hidden,
            activation=None,
            use_bias=False,
            key=keys[3],
        )
        self.value = Linear(
            in_size=self.width,
            out_size=hidden,
            activation=None,
            use_bias=False,
            key=keys[4],
        )
        self.output = Linear(
            in_size=hidden,
            out_size=self.width,
            activation=None,
            use_bias=False,
            key=keys[5],
        )

    def __call__(
        self,
        values: Array,
        field_mask: Array,
        /,
    ) -> Array:
        hidden = jnp.asarray(values)
        normalized = _feature_norm(self.domain_norm, hidden)
        field_axis = normalized.ndim - 2
        case_ndim = normalized.ndim - self.spatial_ndim - 2
        field_first = jnp.moveaxis(normalized, field_axis, case_ndim)
        mixed = self.domain_mixer(field_first) + self.pointwise(field_first)
        mixed = jnp.moveaxis(mixed, case_ndim, field_axis)
        sample_mask = jnp.broadcast_to(
            field_mask,
            hidden.shape[:case_ndim] + (int(hidden.shape[-2]),),
        )
        for _ in range(self.spatial_ndim):
            sample_mask = jnp.expand_dims(sample_mask, axis=case_ndim)
        sample_mask = jnp.broadcast_to(sample_mask, hidden.shape[:-1])
        hidden = (hidden + mixed) * sample_mask[..., None]
        attended = self.codomain_attention(
            _feature_norm(self.codomain_norm, hidden), sample_mask
        )
        hidden = (hidden + attended) * sample_mask[..., None]
        normalized = _feature_norm(self.feed_forward_norm, hidden)
        feed_forward = self.output(
            jnn.silu(self.gate(normalized)) * self.value(normalized)
        )
        return (hidden + feed_forward) * sample_mask[..., None]


def _predict_codano(
    model: Any,
    batch: OperatorBatch,
    key: EvalKey,
    /,
) -> OperatorPrediction:
    state = model.encode_inputs(batch, key=key)
    queries = {
        name: batch.query(model._field(name).query_name) for name in model.target_names
    }
    values = model.decode_fields(state, queries)
    fields = {}
    for name in model.target_names:
        field = model._field(name)
        assert field.query_name is not None
        assert field.output_spec is not None
        fields[name] = OperatorFieldBatch(
            values[name],
            query_name=field.query_name,
            spec=field.output_spec,
        )
    return OperatorPrediction(
        fields,
        batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


class CoDANO(AbstractEncodedOperatorModel):
    """Codomain-attention neural operator for variable multiphysics field sets."""

    operator_architecture = "CoDANO"

    _operator_prediction_builder: ClassVar = staticmethod(_predict_codano)

    fields: tuple[OperatorFieldSpec, ...]
    latent_geometry: TensorGridLatentGeometry
    source_lifts: tuple[Linear, ...]
    source_transfer: tuple[MeasureAwareAttention, ...]
    latent_query_lift: Linear
    field_embeddings: Array
    blocks: tuple[CoDABlock, ...]
    decode_query_lifts: tuple[Linear, ...]
    decoders: tuple[MeasureAwareAttention, ...]
    projections: tuple[Linear, ...]
    source_names: tuple[str, ...]
    target_names: tuple[str, ...]
    default_target: str
    width: int
    depth: int
    coord_dim: int
    in_size: tuple[int, ...]
    out_size: int | Literal["scalar"]

    def __init__(
        self,
        fields: Sequence[OperatorFieldSpec],
        latent_shape: Sequence[int],
        /,
        *,
        n_modes: int | Sequence[int],
        width: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        head_dim: int = 16,
        feed_forward_multiplier: float = 2.0,
        factorization: Factorization = "dense",
        rank: int | float = 0.5,
        bounds: Array | None = None,
        bounds_policy: Literal["global", "case_bbox"] = "global",
        default_target: str | None = None,
        attention_kernel: AttentionKernel = "softmax",
        attention_execution: AttentionExecution = "auto",
        attention_block_size: int = 256,
        accumulation_dtype: str = "input",
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.fields = tuple(fields)
        if not self.fields or len({field.name for field in self.fields}) != len(
            self.fields
        ):
            raise ValueError("CoDA-NO fields must be non-empty and uniquely named.")
        self.source_names = tuple(field.name for field in self.fields if field.is_source)
        self.target_names = tuple(field.name for field in self.fields if field.is_target)
        if not self.source_names or not self.target_names:
            raise ValueError("CoDA-NO requires source and target fields.")
        for field in self.fields:
            if field.is_target:
                assert field.output_spec is not None
                if _get_size(field.output_spec.channels) != field.channel_count:
                    raise ValueError(
                        "CoDA-NO target output channels must match field channels."
                    )
        chosen_target = (
            self.target_names[0] if default_target is None else str(default_target)
        )
        if chosen_target not in self.target_names:
            raise ValueError("default_target must name a target field.")
        self.default_target = chosen_target
        self.width = int(width)
        self.depth = int(depth)
        self.latent_geometry = TensorGridLatentGeometry(
            latent_shape,
            bounds=bounds,
            bounds_policy=bounds_policy,
        )
        self.coord_dim = self.latent_geometry.coord_dim
        if min(self.width, self.depth, int(num_heads), int(head_dim)) <= 0:
            raise ValueError("CoDA-NO dimensions must be positive.")
        self.in_size = tuple(
            next(field.channel_count for field in self.fields if field.name == name)
            for name in self.source_names
        )
        default_spec = next(
            field.output_spec for field in self.fields if field.name == chosen_target
        )
        assert default_spec is not None
        self.out_size = default_spec.channels
        self.source_lifts = tuple(
            Linear(
                in_size=self._field(name).channel_count + self.coord_dim,
                out_size=self.width,
                activation=jnn.gelu,
                key=_named_key(key, f"source_lift:{name}"),
            )
            for name in self.source_names
        )
        attention_kwargs = dict(
            source_channels=self.width,
            query_channels=self.width,
            out_channels=self.width,
            num_heads=int(num_heads),
            head_dim=int(head_dim),
            kernel=attention_kernel,
            execution=attention_execution,
            block_size=attention_block_size,
            accumulation_dtype=accumulation_dtype,
        )
        self.source_transfer = tuple(
            MeasureAwareAttention(
                key=_named_key(key, f"source_transfer:{name}"),
                **attention_kwargs,
            )
            for name in self.source_names
        )
        self.latent_query_lift = Linear(
            in_size=self.coord_dim,
            out_size=self.width,
            activation=jnn.gelu,
            key=_named_key(key, "latent_query"),
        )
        self.field_embeddings = jnp.stack(
            tuple(
                jr.normal(
                    _named_key(key, f"field_embedding:{field.name}"),
                    (self.width,),
                )
                for field in self.fields
            )
        ) / jnp.sqrt(float(self.width))
        self.blocks = tuple(
            CoDABlock(
                width=self.width,
                spatial_ndim=self.coord_dim,
                n_modes=n_modes,
                num_heads=int(num_heads),
                head_dim=int(head_dim),
                feed_forward_multiplier=feed_forward_multiplier,
                factorization=factorization,
                rank=rank,
                key=_named_key(key, f"block:{index}"),
            )
            for index in range(self.depth)
        )
        self.decode_query_lifts = tuple(
            Linear(
                in_size=self.coord_dim,
                out_size=self.width,
                activation=jnn.gelu,
                key=_named_key(key, f"decode_query:{name}"),
            )
            for name in self.target_names
        )
        self.decoders = tuple(
            MeasureAwareAttention(
                key=_named_key(key, f"decoder:{name}"),
                **attention_kwargs,
            )
            for name in self.target_names
        )
        self.projections = tuple(
            Linear(
                in_size=self.width,
                out_size=self._field(name).channel_count,
                activation=None,
                key=_named_key(key, f"projection:{name}"),
            )
            for name in self.target_names
        )

    def _field(self, name: str, /) -> OperatorFieldSpec:
        return next(field for field in self.fields if field.name == name)

    @property
    def operator_output_specs(self) -> dict[str, OperatorOutputSpec]:
        specs: dict[str, OperatorOutputSpec] = {}
        for name in self.target_names:
            output_spec = self._field(name).output_spec
            if output_spec is not None:
                specs[name] = output_spec
        return specs

    def encode_inputs(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> CoDAOperatorState:
        del key
        bounds_coordinates: Array | None = None
        bounds_mask: Array | None = None
        if self.latent_geometry.bounds_policy == "case_bbox":
            for field in self.fields:
                if (
                    field.is_source
                    and field.source_name is not None
                    and field.source_name in batch.inputs
                ):
                    bounds_source = batch.input(field.source_name)
                    bounds_coordinates = bounds_source.coordinates_array(
                        case_shape=batch.case_shape, flatten=True
                    )
                    bounds_mask = bounds_source.mask_array(
                        case_shape=batch.case_shape
                    ).reshape(bounds_coordinates.shape[:-1])
                    break
            if bounds_coordinates is None:
                raise ValueError(
                    "case_bbox CoDA-NO requires one available source geometry."
                )
        latent_coordinates = self.latent_geometry.coordinates(
            batch.case_shape,
            source_coordinates=bounds_coordinates,
            source_mask=bounds_mask,
            flatten=True,
        )
        latent_weights = self.latent_geometry.quadrature(
            batch.case_shape,
            source_coordinates=bounds_coordinates,
            source_mask=bounds_mask,
            flatten=True,
        )
        cases = prod(batch.case_shape) if batch.case_shape else 1
        latent_count = self.latent_geometry.point_count
        flat_latent_coordinates = latent_coordinates.reshape(
            (cases, latent_count, self.coord_dim)
        )
        base_query = self.latent_query_lift(flat_latent_coordinates)
        source_index = {name: index for index, name in enumerate(self.source_names)}
        encoded_fields: list[Array] = []
        presence: list[Array] = []
        for field_index, field in enumerate(self.fields):
            query_features = base_query + self.field_embeddings[field_index]
            if not field.is_source:
                encoded_fields.append(query_features)
                presence.append(jnp.ones((cases,), dtype=bool))
                continue
            assert field.source_name is not None
            if field.source_name not in batch.inputs:
                if field.required:
                    raise KeyError(
                        f"Missing required CoDA-NO field source {field.source_name!r}."
                    )
                target_presence = jnp.ones((cases,), dtype=bool)
                encoded_fields.append(
                    query_features if field.is_target else jnp.zeros_like(query_features)
                )
                presence.append(
                    target_presence
                    if field.is_target
                    else jnp.zeros((cases,), dtype=bool)
                )
                continue
            source = batch.input(field.source_name)
            values = _field_values(field, source, batch.case_shape)
            coordinates = source.coordinates_array(
                case_shape=batch.case_shape, flatten=True
            )
            if int(coordinates.shape[-1]) != self.coord_dim:
                raise ValueError(
                    f"CoDA-NO field {field.name!r} expected coordinate dimension "
                    f"{self.coord_dim}; got {coordinates.shape[-1]}."
                )
            source_count = int(values.shape[-2])
            lifted = self.source_lifts[source_index[field.name]](
                jnp.concatenate((values, coordinates), axis=-1)
            ).reshape((cases, source_count, self.width))
            source_mask = source.mask_array(case_shape=batch.case_shape).reshape(
                (cases, source_count)
            )
            source_weights = source.quadrature(case_shape=batch.case_shape).reshape(
                (cases, source_count)
            )
            field_presence = jnp.any(source_mask, axis=-1)
            transferred = self.source_transfer[source_index[field.name]](
                lifted,
                query_features,
                source_weights,
                source_mask=source_mask,
            )
            model_presence = (
                jnp.ones_like(field_presence) if field.is_target else field_presence
            )
            encoded_fields.append(
                (query_features + transferred) * model_presence[:, None, None]
            )
            presence.append(model_presence)
        stacked = jnp.stack(encoded_fields, axis=-2)
        field_mask = jnp.stack(presence, axis=-1)
        spatial_shape = self.latent_geometry.shape
        hidden = stacked.reshape(
            batch.case_shape + spatial_shape + (len(self.fields), self.width)
        )
        shaped_mask = field_mask.reshape(batch.case_shape + (len(self.fields),))
        layers: list[Array] = []
        for block in self.blocks:
            hidden = block(hidden, shaped_mask)
            layers.append(hidden)
        return CoDAOperatorState(
            values=hidden,
            field_mask=shaped_mask,
            coordinates=latent_coordinates,
            quadrature_weights=latent_weights,
            case_shape=batch.case_shape,
            layer_values=layers,
        )

    def decode_field(
        self,
        state: CoDAOperatorState,
        field_name: str,
        query: FunctionSamples,
        /,
    ) -> Array:
        if field_name not in self.target_names:
            raise KeyError(
                f"Unknown CoDA-NO target field {field_name!r}; "
                f"expected {self.target_names}."
            )
        target_index = self.target_names.index(field_name)
        field_index = tuple(field.name for field in self.fields).index(field_name)
        field = self._field(field_name)
        query_coordinates = query.coordinates_array(
            case_shape=state.case_shape, flatten=True
        )
        if int(query_coordinates.shape[-1]) != self.coord_dim:
            raise ValueError(
                f"CoDA-NO query expected coordinate dimension {self.coord_dim}; "
                f"got {query_coordinates.shape[-1]}."
            )
        cases = prod(state.case_shape) if state.case_shape else 1
        query_count = prod(query.sample_shape)
        query_features = (
            self.decode_query_lifts[target_index](
                query_coordinates.reshape((cases, query_count, self.coord_dim))
            )
            + self.field_embeddings[field_index]
        )
        source_values = state.values[..., field_index, :].reshape(
            (cases, self.latent_geometry.point_count, self.width)
        )
        query_mask = query.mask_array(case_shape=state.case_shape).reshape(
            (cases, query_count)
        )
        decoded = query_features + self.decoders[target_index](
            source_values,
            query_features,
            state.quadrature_weights.reshape((cases, self.latent_geometry.point_count)),
            source_mask=jnp.broadcast_to(
                state.field_mask[..., field_index].reshape((cases, 1)),
                (cases, self.latent_geometry.point_count),
            ),
            query_mask=query_mask,
        )
        projected = self.projections[target_index](decoded) * query_mask[..., None]
        shaped = projected.reshape(
            state.case_shape + query.sample_shape + (field.channel_count,)
        )
        if field.channels == "scalar":
            return field.dimensionalize(shaped[..., 0])
        return field.dimensionalize(shaped)

    def decode_fields(
        self,
        state: CoDAOperatorState,
        queries: Mapping[str, FunctionSamples],
        /,
    ) -> frozendict[str, Array]:
        if set(queries) != set(self.target_names):
            raise ValueError("queries must cover every CoDA-NO target field exactly.")
        return frozendict(
            {
                name: self.decode_field(state, name, queries[name])
                for name in self.target_names
            }
        )

    def decode_query(
        self,
        state: CoDAOperatorState,
        query: FunctionSamples,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        del key
        return self.decode_field(state, self.default_target, query)

    def __call__(
        self,
        x: OperatorBatch,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("CoDANO requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


__all__ = ["CoDABlock", "CoDANO", "CoDAOperatorState"]
