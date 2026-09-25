#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import abc
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ..._strict import StrictModule
from ..._trainable import ParameterOwner


class AbstractFlowDistribution(StrictModule, ParameterOwner, abc.ABC):
    """Trainable exact-density distribution with optional conditioning."""

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def cond_shape(self) -> tuple[int, ...] | None:
        raise NotImplementedError

    @abc.abstractmethod
    def sample(
        self,
        key: Key,
        *,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(
        self,
        value: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> Array:
        raise NotImplementedError

    def sample_and_log_prob(
        self,
        key: Key,
        *,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        sample = self.sample(key, sample_shape=sample_shape, condition=condition)
        return sample, self.log_prob(sample, condition=condition)


class NormalFlowDistribution(AbstractFlowDistribution):
    location: Array
    _raw_scale: Array
    _shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, location: ArrayLike, scale: ArrayLike, /):
        location_ = jnp.asarray(location)
        scale_ = jnp.asarray(scale, dtype=location_.dtype)
        if location_.ndim != 1 or location_.shape == (0,):
            raise ValueError("Normal flow locations must be a nonempty vector.")
        if scale_.shape != location_.shape:
            raise ValueError("Normal flow scale must match the location shape.")
        scale_ = eqx.error_if(
            scale_,
            jnp.any(~jnp.isfinite(scale_) | (scale_ <= 0.0)),
            "Normal flow scales must be finite and positive.",
        )
        self.location = location_
        self._raw_scale = scale_ + jnp.log(-jnp.expm1(-scale_))
        self._shape = tuple(location_.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def scale(self) -> Array:
        return jax.nn.softplus(self._raw_scale) + jnp.finfo(self._raw_scale.dtype).tiny

    @property
    def cond_shape(self) -> None:
        return None

    def sample(
        self,
        key: Key,
        *,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ) -> Array:
        if condition is not None:
            raise ValueError("An unconditional normal flow does not accept condition.")
        return self.location + self.scale * jax.random.normal(
            key,
            tuple(sample_shape) + self.shape,
            dtype=self.location.dtype,
        )

    def log_prob(
        self,
        value: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> Array:
        if condition is not None:
            raise ValueError("An unconditional normal flow does not accept condition.")
        value_ = jnp.asarray(value, dtype=self.location.dtype)
        if value_.shape[-1:] != self.shape:
            raise ValueError("Normal flow values have an incompatible event shape.")
        standardized = (value_ - self.location) / self.scale
        return -0.5 * jnp.sum(
            standardized**2
            + 2.0 * jnp.log(self.scale)
            + jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=value_.dtype)),
            axis=-1,
        )


class AffineCouplingLayer(StrictModule, ParameterOwner):
    network: eqx.nn.MLP
    mask: Array
    event_size: int = eqx.field(static=True)
    condition_size: int = eqx.field(static=True)

    def __init__(
        self,
        event_size: int,
        condition_size: int,
        mask: ArrayLike,
        /,
        *,
        width: int,
        depth: int,
        key: Key,
    ):
        event = int(event_size)
        condition = int(condition_size)
        mask_value = jnp.asarray(mask)
        if mask_value.shape != (event,):
            raise ValueError("Coupling masks must match the event size.")
        if not (
            jnp.issubdtype(mask_value.dtype, jnp.number)
            or jnp.issubdtype(mask_value.dtype, jnp.bool_)
        ):
            raise TypeError("Coupling masks must contain boolean or numeric values.")
        if not bool(
            jnp.all(jnp.isfinite(mask_value) & ((mask_value == 0) | (mask_value == 1)))
        ):
            raise ValueError("Coupling masks must contain only binary values.")
        mask_ = mask_value.astype(jnp.bool_)
        self.network = eqx.nn.MLP(
            event + condition,
            2 * event,
            int(width),
            int(depth),
            activation=jax.nn.tanh,
            final_activation=lambda value: 0.1 * value,
            key=key,
        )
        self.mask = mask_
        self.event_size = event
        self.condition_size = condition

    def _parameters(self, value: Array, condition: Array, /) -> tuple[Array, Array]:
        inputs = jnp.concatenate((self.mask * value, condition), axis=0)
        raw = self.network(inputs)
        inverse_mask = 1.0 - self.mask
        log_scale = 2.0 * jnp.tanh(raw[: self.event_size]) * inverse_mask
        shift = raw[self.event_size :] * inverse_mask
        return log_scale, shift

    def forward(self, value: Array, condition: Array, /) -> tuple[Array, Array]:
        log_scale, shift = self._parameters(value, condition)
        inverse_mask = 1.0 - self.mask
        transformed = self.mask * value + inverse_mask * (
            value * jnp.exp(log_scale) + shift
        )
        return transformed, jnp.sum(log_scale)

    def inverse(self, value: Array, condition: Array, /) -> tuple[Array, Array]:
        log_scale, shift = self._parameters(value, condition)
        inverse_mask = 1.0 - self.mask
        transformed = self.mask * value + inverse_mask * (
            (value - shift) * jnp.exp(-log_scale)
        )
        return transformed, -jnp.sum(log_scale)


class CouplingFlowDistribution(AbstractFlowDistribution):
    base: NormalFlowDistribution
    layers: tuple[AffineCouplingLayer, ...]
    condition_size: int = eqx.field(static=True)

    def __init__(
        self,
        base: NormalFlowDistribution,
        layers: tuple[AffineCouplingLayer, ...],
        /,
        *,
        condition_size: int,
    ):
        if not isinstance(base, NormalFlowDistribution):
            raise TypeError("base must be a NormalFlowDistribution.")
        if not layers:
            raise ValueError("A coupling flow requires at least one layer.")
        if any(layer.event_size != base.shape[0] for layer in layers):
            raise ValueError("Every coupling layer must match the base event size.")
        if any(layer.condition_size != int(condition_size) for layer in layers):
            raise ValueError("Every coupling layer must share the condition size.")
        self.base = base
        self.layers = tuple(layers)
        self.condition_size = int(condition_size)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.base.shape

    @property
    def cond_shape(self) -> tuple[int, ...] | None:
        return None if self.condition_size == 0 else (self.condition_size,)

    def _condition(
        self,
        condition: ArrayLike | None,
        leading_shape: tuple[int, ...],
        /,
        *,
        dtype: Any,
    ) -> Array:
        if self.condition_size == 0:
            if condition is not None:
                raise ValueError("An unconditional flow does not accept condition.")
            return jnp.zeros(leading_shape + (0,), dtype=dtype)
        if condition is None:
            raise ValueError("A conditional flow requires condition.")
        condition_ = jnp.asarray(condition, dtype=dtype)
        if condition_.shape[-1:] != (self.condition_size,):
            raise ValueError("Flow condition has an incompatible trailing shape.")
        return jnp.broadcast_to(condition_, leading_shape + (self.condition_size,))

    def _forward_one(self, value: Array, condition: Array, /) -> tuple[Array, Array]:
        log_det = jnp.zeros((), dtype=value.dtype)
        current = value
        for layer in self.layers:
            current, increment = layer.forward(current, condition)
            log_det = log_det + increment
        return current, log_det

    def _inverse_one(self, value: Array, condition: Array, /) -> tuple[Array, Array]:
        log_det = jnp.zeros((), dtype=value.dtype)
        current = value
        for layer in reversed(self.layers):
            current, increment = layer.inverse(current, condition)
            log_det = log_det + increment
        return current, log_det

    def sample(
        self,
        key: Key,
        *,
        sample_shape: tuple[int, ...] = (),
        condition: ArrayLike | None = None,
    ) -> Array:
        samples = tuple(sample_shape)
        if any(size <= 0 for size in samples):
            raise ValueError("Flow sample dimensions must be positive.")
        batch_shape = () if condition is None else jnp.asarray(condition).shape[:-1]
        leading = samples + tuple(batch_shape)
        base = self.base.sample(key, sample_shape=leading)
        conditions = self._condition(condition, leading, dtype=base.dtype)
        count = prod(leading) if leading else 1
        values, _ = jax.vmap(self._forward_one)(
            base.reshape((count, self.shape[0])),
            conditions.reshape((count, self.condition_size)),
        )
        return values.reshape(leading + self.shape)

    def log_prob(
        self,
        value: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> Array:
        value_ = jnp.asarray(value, dtype=self.base.location.dtype)
        if value_.shape[-1:] != self.shape:
            raise ValueError("Flow values have an incompatible event shape.")
        leading = tuple(value_.shape[:-1])
        conditions = self._condition(condition, leading, dtype=value_.dtype)
        count = prod(leading) if leading else 1
        base_values, inverse_log_det = jax.vmap(self._inverse_one)(
            value_.reshape((count, self.shape[0])),
            conditions.reshape((count, self.condition_size)),
        )
        result = self.base.log_prob(base_values) + inverse_log_det
        return result.reshape(leading)


def coupling_flow(
    key: Key,
    /,
    *,
    base_dist: NormalFlowDistribution,
    cond_dim: int | None = None,
    flow_layers: int = 8,
    nn_width: int = 50,
    nn_depth: int = 1,
    invert: bool = True,
    **_: Any,
) -> CouplingFlowDistribution:
    """Build an alternating affine coupling flow."""

    del invert
    layers = int(flow_layers)
    event_size = base_dist.shape[0]
    condition_size = 0 if cond_dim is None else int(cond_dim)
    if layers <= 0 or int(nn_width) <= 0 or int(nn_depth) <= 0:
        raise ValueError("Flow layers, width, and depth must be positive.")
    keys = jax.random.split(key, layers)
    result = []
    for index, layer_key in enumerate(keys):
        if event_size == 1:
            mask = jnp.zeros((1,), dtype=base_dist.location.dtype)
        else:
            mask = (jnp.arange(event_size) + index) % 2
        result.append(
            AffineCouplingLayer(
                event_size,
                condition_size,
                mask,
                width=int(nn_width),
                depth=int(nn_depth),
                key=layer_key,
            )
        )
    return CouplingFlowDistribution(
        base_dist,
        tuple(result),
        condition_size=condition_size,
    )


def triangular_flow(
    key: Key,
    /,
    *,
    base_dist: NormalFlowDistribution,
    flow_layers: int,
    nn_width: int = 16,
    nn_depth: int = 1,
    **kwargs: Any,
) -> CouplingFlowDistribution:
    return coupling_flow(
        key,
        base_dist=base_dist,
        flow_layers=flow_layers,
        nn_width=nn_width,
        nn_depth=nn_depth,
        **kwargs,
    )


__all__ = [
    "AbstractFlowDistribution",
    "AffineCouplingLayer",
    "CouplingFlowDistribution",
    "NormalFlowDistribution",
    "coupling_flow",
    "triangular_flow",
]
