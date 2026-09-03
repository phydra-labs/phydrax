#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import hashlib
from collections.abc import Sequence
from math import prod
from typing import Literal

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jax import core as jax_core
from jaxtyping import Array, Key

import phydrax.ein as ein

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from .data import FunctionSamples


ContextKind = Literal["learned", "pooled_geometry", "sampled_anchor"]


def operator_context_fingerprint(
    samples: FunctionSamples,
    /,
    *,
    case_shape: Sequence[int],
    normalization_id: str = "",
) -> str:
    """Return a stable schema fingerprint without hashing runtime field values."""
    axes = tuple(
        (
            axis.name,
            axis.size,
            axis.basis,
            axis.periodic,
            axis.quadrature_weights is not None,
        )
        for axis in samples.axes
    )
    payload = repr(
        (
            tuple(int(size) for size in case_shape),
            samples.sample_shape,
            axes,
            None if samples.coordinates is None else int(samples.coordinates.shape[-1]),
            samples.mask is not None,
            samples.quadrature_weights is not None,
            str(normalization_id),
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class EncodedOperatorState(StrictModule):
    """Reusable context state for encode-once/decode-many operators."""

    kind: ContextKind
    values: Array
    coordinates: Array | None
    weights: Array
    mask: Array
    layer_values: tuple[Array, ...]
    projected_keys: tuple[Array, ...]
    projected_values: tuple[Array, ...]
    case_shape: tuple[int, ...]
    schema_fingerprint: str

    def __init__(
        self,
        *,
        kind: ContextKind,
        values: Array,
        coordinates: Array | None,
        weights: Array,
        mask: Array,
        case_shape: Sequence[int],
        schema_fingerprint: str,
        layer_values: Sequence[Array] = (),
        projected_keys: Sequence[Array] = (),
        projected_values: Sequence[Array] = (),
    ):
        values_ = jnp.asarray(values)
        cases = tuple(int(size) for size in case_shape)
        if values_.ndim != len(cases) + 2:
            raise ValueError(
                "Encoded context values must have shape case_shape + (tokens, channels)."
            )
        if tuple(int(size) for size in values_.shape[: len(cases)]) != cases:
            raise ValueError("Encoded context values do not begin with case_shape.")
        tokens = int(values_.shape[-2])
        weights_ = jnp.asarray(weights)
        mask_ = jnp.asarray(mask, dtype=bool)
        expected_geometry = cases + (tokens,)
        if tuple(int(size) for size in weights_.shape) != expected_geometry:
            raise ValueError(
                f"Encoded context weights must have shape {expected_geometry}."
            )
        if tuple(int(size) for size in mask_.shape) != expected_geometry:
            raise ValueError(f"Encoded context mask must have shape {expected_geometry}.")
        coordinates_: Array | None
        if coordinates is None:
            coordinates_ = None
        else:
            coordinates_ = jnp.asarray(coordinates)
            if coordinates_.ndim != len(cases) + 2:
                raise ValueError(
                    "Context coordinates must have shape case_shape + "
                    "(tokens, coordinate_dimension)."
                )
            if tuple(int(size) for size in coordinates_.shape[:-1]) != expected_geometry:
                raise ValueError("Context coordinates do not align with context tokens.")
        layers = tuple(jnp.asarray(layer) for layer in layer_values)
        if not layers:
            layers = (values_,)
        if any(layer.shape != values_.shape for layer in layers):
            raise ValueError(
                "Every encoded layer value must match the context value shape."
            )
        keys = tuple(jnp.asarray(value) for value in projected_keys)
        projected = tuple(jnp.asarray(value) for value in projected_values)
        if len(keys) != len(projected):
            raise ValueError("Projected key/value cache lengths must match.")
        if keys and len(keys) != len(layers):
            raise ValueError("Projected key/value caches must cover every context layer.")
        self.kind = kind
        self.values = values_
        self.coordinates = coordinates_
        self.weights = weights_
        self.mask = mask_
        self.layer_values = layers
        self.projected_keys = keys
        self.projected_values = projected
        self.case_shape = cases
        self.schema_fingerprint = str(schema_fingerprint)

    @property
    def num_tokens(self) -> int:
        return int(self.values.shape[-2])

    @property
    def channels(self) -> int:
        return int(self.values.shape[-1])

    def at_layer(self, index: int, /) -> Array:
        return self.layer_values[int(index)]

    def replace_layers(
        self,
        layer_values: Sequence[Array],
        /,
        *,
        projected_keys: Sequence[Array] = (),
        projected_values: Sequence[Array] = (),
    ) -> "EncodedOperatorState":
        layers = tuple(layer_values)
        return EncodedOperatorState(
            kind=self.kind,
            values=layers[-1],
            coordinates=self.coordinates,
            weights=self.weights,
            mask=self.mask,
            case_shape=self.case_shape,
            schema_fingerprint=self.schema_fingerprint,
            layer_values=layers,
            projected_keys=projected_keys,
            projected_values=projected_values,
        )


def _flatten_values(
    values: Array,
    samples: FunctionSamples,
    channels: int,
    /,
) -> tuple[Array, tuple[int, ...]]:
    array = jnp.asarray(values)
    sample_shape = samples.sample_shape
    sample_ndim = len(sample_shape)
    if not sample_shape:
        raise ValueError("Operator contexts require a non-empty sample geometry.")
    has_channels = (
        array.ndim > sample_ndim
        and tuple(array.shape[-sample_ndim - 1 : -1]) == sample_shape
        and int(array.shape[-1]) == int(channels)
    )
    if not has_channels:
        if int(channels) != 1 or tuple(array.shape[-sample_ndim:]) != sample_shape:
            raise ValueError(
                f"Context values do not contain sample shape {sample_shape} with "
                f"{channels} channels; got {array.shape}."
            )
        array = array[..., None]
    case_shape = tuple(int(size) for size in array.shape[: -sample_ndim - 1])
    cases = prod(case_shape) if case_shape else 1
    return array.reshape((cases, prod(sample_shape), int(channels))), case_shape


def _flatten_geometry(
    samples: FunctionSamples,
    case_shape: tuple[int, ...],
    /,
) -> tuple[Array, Array, Array]:
    cases = prod(case_shape) if case_shape else 1
    count = prod(samples.sample_shape)
    coordinates = samples.coordinates_array(case_shape=case_shape, flatten=True)
    coordinates = coordinates.reshape((cases, count, int(coordinates.shape[-1])))
    weights = samples.quadrature(case_shape=case_shape).reshape((cases, count))
    mask = samples.mask_array(case_shape=case_shape).reshape((cases, count))
    return coordinates, weights, mask


class AbstractOperatorContextStrategy(StrictModule):
    """Base contract for constructing an operator context token bank."""

    channels: int

    @abc.abstractmethod
    def __call__(
        self,
        values: Array,
        samples: FunctionSamples,
        /,
        *,
        indices: Array | None = None,
        normalization_id: str = "",
    ) -> EncodedOperatorState:
        raise NotImplementedError


class LearnedTokenContext(AbstractOperatorContextStrategy):
    """A fixed learned abstract token bank for UPT-style operators."""

    tokens: Array
    channels: int
    num_tokens: int

    def __init__(
        self,
        *,
        channels: int,
        num_tokens: int,
        initial_scale: float = 1.0,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.channels = int(channels)
        self.num_tokens = int(num_tokens)
        if self.channels <= 0 or self.num_tokens <= 0:
            raise ValueError("channels and num_tokens must be positive.")
        self.tokens = float(initial_scale) * jr.normal(
            key, (self.num_tokens, self.channels)
        )

    def __call__(
        self,
        values: Array,
        samples: FunctionSamples,
        /,
        *,
        indices: Array | None = None,
        normalization_id: str = "",
    ) -> EncodedOperatorState:
        del indices
        _, case_shape = _flatten_values(values, samples, self.channels)
        context = jnp.broadcast_to(
            self.tokens, case_shape + (self.num_tokens, self.channels)
        )
        geometry_shape = case_shape + (self.num_tokens,)
        return EncodedOperatorState(
            kind="learned",
            values=context,
            coordinates=None,
            weights=jnp.ones(geometry_shape, dtype=context.dtype),
            mask=jnp.ones(geometry_shape, dtype=bool),
            case_shape=case_shape,
            schema_fingerprint=operator_context_fingerprint(
                samples,
                case_shape=case_shape,
                normalization_id=normalization_id,
            ),
        )


class PooledGeometryContext(AbstractOperatorContextStrategy):
    """Measure-weighted deterministic pooling into geometry-bearing tokens."""

    channels: int
    num_tokens: int

    def __init__(self, *, channels: int, num_tokens: int):
        self.channels = int(channels)
        self.num_tokens = int(num_tokens)
        if self.channels <= 0 or self.num_tokens <= 0:
            raise ValueError("channels and num_tokens must be positive.")

    def __call__(
        self,
        values: Array,
        samples: FunctionSamples,
        /,
        *,
        indices: Array | None = None,
        normalization_id: str = "",
    ) -> EncodedOperatorState:
        del indices
        flattened, case_shape = _flatten_values(values, samples, self.channels)
        coordinates, weights, mask = _flatten_geometry(samples, case_shape)
        count = int(flattened.shape[1])
        segment = jnp.minimum(
            (jnp.arange(count) * self.num_tokens) // count,
            self.num_tokens - 1,
        )
        assignment = jnn.one_hot(segment, self.num_tokens, dtype=flattened.dtype)
        effective = weights * mask.astype(weights.dtype)
        mass = ein.contract("bn,nm->bm", effective, assignment)
        denominator = jnp.maximum(mass, jnp.finfo(flattened.dtype).tiny)
        pooled_values = (
            ein.contract("bn,bnc,nm->bmc", effective, flattened, assignment)
            / denominator[..., None]
        )
        pooled_coordinates = (
            ein.contract("bn,bnd,nm->bmd", effective, coordinates, assignment)
            / denominator[..., None]
        )
        pooled_mask = mass > 0.0
        pooled_values = jnp.where(pooled_mask[..., None], pooled_values, 0.0)
        pooled_coordinates = jnp.where(pooled_mask[..., None], pooled_coordinates, 0.0)
        shape = case_shape + (self.num_tokens,)
        return EncodedOperatorState(
            kind="pooled_geometry",
            values=pooled_values.reshape(shape + (self.channels,)),
            coordinates=pooled_coordinates.reshape(shape + (int(coordinates.shape[-1]),)),
            weights=mass.reshape(shape),
            mask=pooled_mask.reshape(shape),
            case_shape=case_shape,
            schema_fingerprint=operator_context_fingerprint(
                samples,
                case_shape=case_shape,
                normalization_id=normalization_id,
            ),
        )


class SampledAnchorContext(AbstractOperatorContextStrategy):
    """A context formed from sampled physical source or geometry locations."""

    channels: int
    num_anchors: int

    def __init__(self, *, channels: int, num_anchors: int):
        self.channels = int(channels)
        self.num_anchors = int(num_anchors)
        if self.channels <= 0 or self.num_anchors <= 0:
            raise ValueError("channels and num_anchors must be positive.")

    def __call__(
        self,
        values: Array,
        samples: FunctionSamples,
        /,
        *,
        indices: Array | None = None,
        normalization_id: str = "",
    ) -> EncodedOperatorState:
        flattened, case_shape = _flatten_values(values, samples, self.channels)
        coordinates, weights, mask = _flatten_geometry(samples, case_shape)
        cases, count, _ = flattened.shape
        if self.num_anchors > count:
            raise ValueError(
                f"Cannot select {self.num_anchors} anchors from {count} samples."
            )
        if indices is None:
            shared = jnp.rint(jnp.linspace(0, count - 1, self.num_anchors)).astype(
                jnp.int32
            )
            selected = jnp.broadcast_to(shared, (cases, self.num_anchors))
        else:
            selected_ = jnp.asarray(indices, dtype=jnp.int32)
            if selected_.shape == (self.num_anchors,):
                selected = jnp.broadcast_to(selected_, (cases, self.num_anchors))
            elif selected_.shape == case_shape + (self.num_anchors,):
                selected = selected_.reshape((cases, self.num_anchors))
            else:
                raise ValueError(
                    "Anchor indices must have shape (num_anchors,) or "
                    "case_shape + (num_anchors,)."
                )
        if not isinstance(selected, jax_core.Tracer):
            if bool(jnp.any((selected < 0) | (selected >= count))):
                raise ValueError("Anchor indices are out of bounds.")
        channel_index = jnp.broadcast_to(
            selected[..., None], selected.shape + (self.channels,)
        )
        coordinate_index = jnp.broadcast_to(
            selected[..., None], selected.shape + (int(coordinates.shape[-1]),)
        )
        anchor_values = jnp.take_along_axis(flattened, channel_index, axis=1)
        anchor_coordinates = jnp.take_along_axis(coordinates, coordinate_index, axis=1)
        anchor_weights = jnp.take_along_axis(weights, selected, axis=1)
        anchor_mask = jnp.take_along_axis(mask, selected, axis=1)
        shape = case_shape + (self.num_anchors,)
        return EncodedOperatorState(
            kind="sampled_anchor",
            values=anchor_values.reshape(shape + (self.channels,)),
            coordinates=anchor_coordinates.reshape(shape + (int(coordinates.shape[-1]),)),
            weights=anchor_weights.reshape(shape),
            mask=anchor_mask.reshape(shape),
            case_shape=case_shape,
            schema_fingerprint=operator_context_fingerprint(
                samples,
                case_shape=case_shape,
                normalization_id=normalization_id,
            ),
        )


OperatorContextStrategy = AbstractOperatorContextStrategy


__all__ = [
    "ContextKind",
    "EncodedOperatorState",
    "LearnedTokenContext",
    "OperatorContextStrategy",
    "PooledGeometryContext",
    "SampledAnchorContext",
    "operator_context_fingerprint",
]
