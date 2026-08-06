#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ..._strict import StrictModule


ProbeDistribution = Literal["rademacher", "normal"]


def _state_axes(value: Array, state_ndim: int, /) -> tuple[int, ...]:
    return tuple(range(value.ndim - state_ndim, value.ndim))


def _contract_state(value: Array, vector: Array, state_ndim: int, /) -> Array:
    leading = value.ndim - state_ndim
    expanded = vector.reshape((1,) * leading + vector.shape)
    return jnp.sum(value * expanded, axis=_state_axes(value, state_ndim))


def _directional_second_derivative(
    function: Callable[[Array], Array],
    state: Array,
    direction: Array,
    left_vector: Array,
    /,
) -> Array:
    gradient = jax.jacrev(function)
    _, hessian_direction = jax.jvp(gradient, (state,), (direction,))
    return _contract_state(hessian_direction, left_vector, state.ndim)


def factor_hvp_contraction(
    function: Callable[[Array], Array],
    state: ArrayLike,
    factor: ArrayLike,
    /,
) -> Array:
    r"""Compute ``sum_k factor_k^T Hessian(function) factor_k`` by HVPs.

    ``factor`` must have shape ``state.shape + (noise_rank,)``. The Hessian is never
    materialized. Array-valued functions are contracted componentwise and retain their
    output shape.
    """
    state_array = jnp.asarray(state)
    factor_array = jnp.asarray(factor)
    if state_array.ndim < 1:
        raise ValueError("state must have at least one axis.")
    if factor_array.shape[:-1] != state_array.shape:
        raise ValueError(
            "factor must have shape state.shape + (noise_rank,); got "
            f"state {state_array.shape} and factor {factor_array.shape}."
        )
    directions = jnp.moveaxis(factor_array, -1, 0)
    terms = jax.vmap(
        lambda direction: _directional_second_derivative(
            function,
            state_array,
            direction,
            direction,
        )
    )(directions)
    return jnp.sum(terms, axis=0)


def directional_stratonovich_correction(
    diffusion: Callable[[Array], Array],
    state: ArrayLike,
    /,
) -> Array:
    r"""Compute ``0.5 sum_k D sigma_k(state)[sigma_k(state)]`` by JVPs.

    The diffusion output must have shape ``state.shape + (noise_rank,)``. This avoids
    constructing the full derivative of the diffusion factor.
    """
    state_array = jnp.asarray(state)
    sigma = jnp.asarray(diffusion(state_array))
    if state_array.ndim < 1:
        raise ValueError("state must have at least one axis.")
    if sigma.shape[:-1] != state_array.shape:
        raise ValueError(
            "diffusion must return state.shape + (noise_rank,); got "
            f"state {state_array.shape} and diffusion {sigma.shape}."
        )
    indices = jnp.arange(sigma.shape[-1])
    directions = jnp.moveaxis(sigma, -1, 0)

    def one(index: Array, direction: Array, /) -> Array:
        column = lambda value: jnp.asarray(diffusion(value))[..., index]
        _, derivative = jax.jvp(column, (state_array,), (direction,))
        return derivative

    return 0.5 * jnp.sum(jax.vmap(one)(indices, directions), axis=0)


class StochasticTracePolicy(StrictModule):
    """Probe policy for an explicit stochastic trace approximation."""

    num_probes: int = eqx.field(static=True)
    distribution: ProbeDistribution = eqx.field(static=True)

    def __init__(
        self,
        num_probes: int = 16,
        /,
        *,
        distribution: ProbeDistribution = "rademacher",
    ):
        count = int(num_probes)
        if count < 2:
            raise ValueError("num_probes must be at least two to estimate uncertainty.")
        if distribution not in ("rademacher", "normal"):
            raise ValueError("distribution must be 'rademacher' or 'normal'.")
        self.num_probes = count
        self.distribution = distribution


class StochasticOperatorEstimate(StrictModule):
    """Value and Monte Carlo standard error of one stochastic operator estimate."""

    value: Array
    standard_error: Array
    num_probes: int = eqx.field(static=True)
    distribution: ProbeDistribution = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike,
        standard_error: ArrayLike,
        num_probes: int,
        distribution: ProbeDistribution,
        /,
    ):
        value_array = jnp.asarray(value)
        error_array = jnp.asarray(standard_error)
        if value_array.shape != error_array.shape:
            raise ValueError("value and standard_error must have the same shape.")
        if int(num_probes) < 2:
            raise ValueError("num_probes must be at least two.")
        if distribution not in ("rademacher", "normal"):
            raise ValueError("distribution must be 'rademacher' or 'normal'.")
        self.value = value_array
        self.standard_error = error_array
        self.num_probes = int(num_probes)
        self.distribution = distribution

    @property
    def relative_standard_error(self) -> Array:
        scale = jnp.maximum(jnp.abs(self.value), jnp.finfo(self.value.dtype).eps)
        return self.standard_error / scale


class StochasticOperatorSamples(StrictModule):
    """Raw probe realizations with their mean and sampling uncertainty."""

    values: Array
    mean: Array
    sample_variance: Array
    standard_error: Array
    dependence_ids: Array
    num_probes: int = eqx.field(static=True)
    distribution: ProbeDistribution = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        /,
        *,
        distribution: ProbeDistribution,
        dependence_ids: ArrayLike | None = None,
    ):
        samples = jnp.asarray(values)
        if samples.ndim < 1 or int(samples.shape[0]) < 2:
            raise ValueError("values must contain at least two probe realizations.")
        if distribution not in ("rademacher", "normal"):
            raise ValueError("distribution must be 'rademacher' or 'normal'.")
        count = int(samples.shape[0])
        mean = jnp.mean(samples, axis=0)
        centered = samples - mean
        sample_variance = jnp.sum(jnp.abs(centered) ** 2, axis=0) / float(
            count - 1
        )
        standard_error = jnp.sqrt(sample_variance / float(count))
        ids = (
            jnp.arange(count, dtype=jnp.int32)
            if dependence_ids is None
            else jnp.asarray(dependence_ids, dtype=jnp.int32)
        )
        if ids.shape != (count,):
            raise ValueError("dependence_ids must have shape (num_probes,).")
        self.values = samples
        self.mean = mean
        self.sample_variance = sample_variance
        self.standard_error = standard_error
        self.dependence_ids = ids
        self.num_probes = count
        self.distribution = distribution

    def estimate(self) -> StochasticOperatorEstimate:
        return StochasticOperatorEstimate(
            self.mean,
            self.standard_error,
            self.num_probes,
            self.distribution,
        )

def _probes(
    key: Key[Array, ""],
    shape: tuple[int, ...],
    dtype,
    policy: StochasticTracePolicy,
    /,
) -> Array:
    full_shape = (policy.num_probes, *shape)
    if policy.distribution == "rademacher":
        return jr.rademacher(key, full_shape, dtype=dtype)
    return jr.normal(key, full_shape, dtype=dtype)


def stochastic_trace_samples(
    function: Callable[[Array], Array],
    state: ArrayLike,
    covariance_action: Callable[[Array, Array], Array],
    key: Key[Array, ""],
    /,
    *,
    policy: StochasticTracePolicy | None = None,
) -> StochasticOperatorSamples:
    """Return individual Hutchinson trace realizations without reducing probes."""
    resolved = StochasticTracePolicy() if policy is None else policy
    if not isinstance(resolved, StochasticTracePolicy):
        raise TypeError("policy must be a StochasticTracePolicy or None.")
    state_array = jnp.asarray(state)
    if state_array.ndim < 1:
        raise ValueError("state must have at least one axis.")
    probes = _probes(key, state_array.shape, state_array.dtype, resolved)

    def one(probe: Array, /) -> Array:
        action = jnp.asarray(covariance_action(state_array, probe))
        if action.shape != state_array.shape:
            raise ValueError(
                "covariance_action must preserve state shape; got "
                f"{action.shape}, expected {state_array.shape}."
            )
        return _directional_second_derivative(
            function,
            state_array,
            action,
            probe,
        )

    return StochasticOperatorSamples(
        jax.vmap(one)(probes),
        distribution=resolved.distribution,
    )


def stochastic_divergence_samples(
    vector_field: Callable[[Array], Array],
    state: ArrayLike,
    key: Key[Array, ""],
    /,
    *,
    policy: StochasticTracePolicy | None = None,
) -> StochasticOperatorSamples:
    """Return probe samples of ``vᵀ J(vector_field) v`` using only JVPs."""
    resolved = StochasticTracePolicy() if policy is None else policy
    if not isinstance(resolved, StochasticTracePolicy):
        raise TypeError("policy must be a StochasticTracePolicy or None.")
    state_array = jnp.asarray(state)
    if state_array.ndim < 1:
        raise ValueError("state must have at least one axis.")
    field_value = jnp.asarray(vector_field(state_array))
    if field_value.shape != state_array.shape:
        raise ValueError("vector_field must preserve the complete state shape.")
    probes = _probes(key, state_array.shape, state_array.dtype, resolved)

    def one(probe: Array, /) -> Array:
        _, derivative = jax.jvp(vector_field, (state_array,), (probe,))
        derivative_array = jnp.asarray(derivative)
        if derivative_array.shape != state_array.shape:
            raise ValueError("vector_field JVP must preserve state shape.")
        return jnp.sum(probe * derivative_array)

    return StochasticOperatorSamples(
        jax.vmap(one)(probes),
        distribution=resolved.distribution,
    )


def estimate_stochastic_trace(
    function: Callable[[Array], Array],
    state: ArrayLike,
    covariance_action: Callable[[Array, Array], Array],
    key: Key[Array, ""],
    /,
    *,
    policy: StochasticTracePolicy | None = None,
) -> StochasticOperatorEstimate:
    r"""Estimate ``trace(a(state) Hessian(function)(state))`` with visible error.

    ``covariance_action(state, probe)`` must return ``a(state) @ probe`` with the
    same shape as ``state``. The estimator never constructs either the covariance or
    Hessian. The caller owns the PRNG key and therefore the probe realization.
    """
    return stochastic_trace_samples(
        function,
        state,
        covariance_action,
        key,
        policy=policy,
    ).estimate()


def estimate_kolmogorov_generator(
    observable: Callable[[Array], Array],
    drift: Callable[[Array], Array],
    state: ArrayLike,
    covariance_action: Callable[[Array, Array], Array],
    key: Key[Array, ""],
    /,
    *,
    policy: StochasticTracePolicy | None = None,
) -> StochasticOperatorEstimate:
    """Estimate a matrix-free backward generator and report probe uncertainty."""
    state_array = jnp.asarray(state)
    drift_array = jnp.asarray(drift(state_array))
    if drift_array.shape != state_array.shape:
        raise ValueError(
            f"drift must preserve state shape {state_array.shape}; got {drift_array.shape}."
        )
    observable_gradient = jax.jacrev(observable)(state_array)
    first_order = _contract_state(
        jnp.asarray(observable_gradient),
        drift_array,
        state_array.ndim,
    )
    trace = estimate_stochastic_trace(
        observable,
        state_array,
        covariance_action,
        key,
        policy=policy,
    )
    return StochasticOperatorEstimate(
        first_order + 0.5 * trace.value,
        0.5 * trace.standard_error,
        trace.num_probes,
        trace.distribution,
    )


__all__ = [
    "ProbeDistribution",
    "stochastic_divergence_samples",
    "StochasticOperatorEstimate",
    "StochasticOperatorSamples",
    "StochasticTracePolicy",
    "directional_stratonovich_correction",
    "estimate_kolmogorov_generator",
    "estimate_stochastic_trace",
    "factor_hvp_contraction",
    "stochastic_trace_samples",
]
