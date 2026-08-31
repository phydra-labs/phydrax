#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._term import AbstractSamplingTerm, AbstractScalarTerm
from ..domain import DomainFunction
from ..nn.models.wrappers._implicit_modal import ImplicitModalField


ModalTimeProvider: TypeAlias = Callable[[Key[Array, ""]], Array]


def _weight(value: ArrayLike, /, *, name: str) -> Array:
    weight = jnp.asarray(value, dtype=float).reshape(())
    if bool(~jnp.isfinite(weight)) or float(weight) < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return weight


def _time_batch(value: ArrayLike, /) -> Array:
    times = jnp.asarray(value)
    if times.ndim != 1 or int(times.size) == 0 or jnp.iscomplexobj(times):
        raise ValueError("times must be a non-empty rank-one real array.")
    return eqx.error_if(times, jnp.any(~jnp.isfinite(times)), "times must be finite.")


def _modal_field(
    functions: Mapping[str, DomainFunction],
    name: str,
    /,
) -> ImplicitModalField:
    if name not in functions:
        raise KeyError(f"Missing implicit modal field {name!r}.")
    function = functions[name]
    if not isinstance(function.func, ImplicitModalField):
        raise TypeError(
            f"Function {name!r} must be created by ImplicitModalField.as_domain_function()."
        )
    return function.func


def _squared_magnitude(value: ArrayLike, /) -> Array:
    residual = jnp.asarray(value)
    return jnp.real(jnp.conj(residual) * residual)


class CompiledModalResidualTerm(AbstractSamplingTerm):
    """Train a time-conditioned modal field against compiled spectral dynamics."""

    compiled: Any
    fixed_times: Array | None
    time_provider: ModalTimeProvider | None
    args: Any
    scalar_weight: Array
    function_name: str = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        compiled: Any,
        /,
        *,
        function_name: str,
        times: ArrayLike | ModalTimeProvider,
        args: Any = None,
        scalar_weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        from ..equations._spectral_compile import CompiledSpectralDynamics

        if not isinstance(compiled, CompiledSpectralDynamics):
            raise TypeError("compiled must be CompiledSpectralDynamics.")
        name = str(function_name)
        if not name:
            raise ValueError("function_name must be non-empty.")
        if callable(times):
            fixed_times = None
            provider = times
        else:
            fixed_times = _time_batch(times)
            provider = None
        self.compiled = compiled
        self.fixed_times = fixed_times
        self.time_provider = provider
        self.args = args
        self.scalar_weight = _weight(scalar_weight, name="scalar_weight")
        self.function_name = name
        self.label = label

    def sample(self, *, key: Key[Array, ""] = DOC_KEY0) -> Array:
        if self.fixed_times is not None:
            return self.fixed_times
        if self.time_provider is None:
            raise RuntimeError("Modal time provider is unavailable.")
        return _time_batch(self.time_provider(key))

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        batch: ArrayLike | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        field = _modal_field(functions, self.function_name)
        if field.discretization.prepared_id != self.compiled.discretization.prepared_id:
            raise ValueError(
                "Implicit modal field and compiled dynamics use different discretizations."
            )
        if field.state_shape != self.compiled.state_shape:
            raise ValueError(
                f"Implicit modal state shape {field.state_shape} does not match "
                f"compiled shape {self.compiled.state_shape}."
            )
        times = self.sample(key=key) if batch is None else _time_batch(batch)
        sites = jnp.arange(times.size, dtype=jnp.uint32)

        def residual_loss(time: Array, site: Array) -> Array:
            site_key = jr.fold_in(key, site)
            state, tangent = field.time_tangent(time, key=site_key)
            residual = tangent - self.compiled(time, state, self.args)
            return jnp.mean(_squared_magnitude(residual))

        values = jax.vmap(residual_loss)(times, sites)
        return self.scalar_weight * jnp.mean(values)


class ModalObservationTerm(AbstractScalarTerm):
    """Supervise complete or masked modal states at fixed times."""

    times: Array
    targets: Array
    mask: Array
    weights: Array
    scalar_weight: Array
    normalization: Array
    function_name: str = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        targets: ArrayLike,
        /,
        *,
        function_name: str,
        mask: ArrayLike | None = None,
        weights: ArrayLike = 1.0,
        scalar_weight: ArrayLike = 1.0,
        label: str | None = None,
    ):
        times_ = np.asarray(_time_batch(times))
        targets_ = np.asarray(targets)
        name = str(function_name)
        if not name:
            raise ValueError("function_name must be non-empty.")
        if targets_.ndim < 1 or targets_.shape[0] != times_.size:
            raise ValueError("targets must have one leading state per time.")
        mask_ = np.ones(targets_.shape, dtype=bool) if mask is None else np.asarray(mask)
        if mask_.shape != targets_.shape or mask_.dtype != np.dtype(bool):
            raise ValueError("mask must be boolean with the same shape as targets.")
        if np.any(~np.isfinite(targets_[mask_])):
            raise ValueError("Observed target values must be finite.")
        weights_ = np.broadcast_to(np.asarray(weights, dtype=float), targets_.shape).copy()
        if np.any(~np.isfinite(weights_)) or np.any(weights_ < 0.0):
            raise ValueError("weights must be finite and nonnegative.")
        normalization = float(np.sum(np.where(mask_, weights_, 0.0)))
        if normalization <= 0.0:
            raise ValueError("At least one observed coefficient must have positive weight.")
        self.times = jnp.asarray(times_)
        self.targets = jnp.asarray(targets_)
        self.mask = jnp.asarray(mask_)
        self.weights = jnp.asarray(weights_)
        self.scalar_weight = _weight(scalar_weight, name="scalar_weight")
        self.normalization = jnp.asarray(normalization, dtype=float)
        self.function_name = name
        self.label = label

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> Array:
        del iter_, kwargs
        field = _modal_field(functions, self.function_name)
        expected = (self.times.size,) + field.state_shape
        if self.targets.shape != expected:
            raise ValueError(
                f"targets must have time/state shape {expected}; got {self.targets.shape}."
            )
        sites = jnp.arange(self.times.size, dtype=jnp.uint32)
        predictions = jax.vmap(
            lambda time, site: field(time, key=jr.fold_in(key, site))
        )(self.times, sites)
        difference = jnp.where(self.mask, predictions - self.targets, 0.0)
        weighted = self.weights * _squared_magnitude(difference)
        return self.scalar_weight * jnp.sum(weighted) / self.normalization


__all__ = [
    "CompiledModalResidualTerm",
    "ModalObservationTerm",
    "ModalTimeProvider",
]
