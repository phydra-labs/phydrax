"""Adam's Optax boundary with precise moments and explicit dtype conversion."""

from __future__ import annotations

from math import isfinite, log

import jax
import jax.numpy as jnp
import optax
from jax import Array
from jax.typing import DTypeLike


def adam(
    learning_rate: float, /, *, b1: float = 0.9, b2: float = 0.999, eps: float = 1.0e-8
) -> optax.GradientTransformation:
    """Adam with at least float32 moments and updates in the input leaf dtype."""
    if not isfinite(learning_rate) or learning_rate < 0.0:
        raise ValueError("Adam learning_rate must be finite and nonnegative.")
    if not all(isfinite(value) and 0.0 <= value < 1.0 for value in (b1, b2)):
        raise ValueError("Adam moment decays must be finite and lie in [0, 1).")
    if not isfinite(eps) or eps <= 0.0:
        raise ValueError("Adam epsilon must be finite and positive.")

    def first_zeros(parameter: Array) -> Array:
        dtype = (
            parameter.dtype
            if parameter.dtype in (jnp.float64, jnp.complex64, jnp.complex128)
            else jnp.float32
        )
        return jnp.zeros(parameter.shape, dtype=dtype)

    def second_zeros(parameter: Array) -> Array:
        dtype = (
            jnp.float64
            if parameter.dtype in (jnp.float64, jnp.complex128)
            else jnp.float32
        )
        return jnp.zeros(parameter.shape, dtype=dtype)

    def initialize(params: optax.Params) -> optax.OptState:
        return optax.ScaleByAdamState(
            count=jnp.zeros((), dtype=jnp.int32),
            mu=jax.tree.map(first_zeros, params),
            nu=jax.tree.map(second_zeros, params),
        )

    def promote(gradient: Array, moment: Array) -> Array:
        return jnp.asarray(gradient, dtype=moment.dtype)

    def first_moment(gradient: Array, moment: Array) -> Array:
        decay = jnp.asarray(b1, dtype=moment.dtype)
        weight = jnp.asarray(1.0 - b1, dtype=moment.dtype)
        return decay * moment + weight * gradient

    def second_moment(gradient: Array, moment: Array) -> Array:
        real = jnp.real(gradient)
        norm_squared = real * real
        if jnp.issubdtype(gradient.dtype, jnp.complexfloating):
            imaginary = jnp.imag(gradient)
            norm_squared = norm_squared + imaginary * imaginary
        decay = jnp.asarray(b2, dtype=moment.dtype)
        weight = jnp.asarray(1.0 - b2, dtype=moment.dtype)
        return decay * moment + weight * norm_squared

    def update(
        updates: optax.Updates,
        state: optax.OptState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, optax.OptState]:
        del params
        if not isinstance(state, optax.ScaleByAdamState):
            raise TypeError("Adam updates require the owning ScaleByAdamState.")
        gradients = jax.tree.map(promote, updates, state.mu)
        first = jax.tree.map(first_moment, gradients, state.mu)
        second = jax.tree.map(second_moment, gradients, state.nu)
        count = optax.safe_increment(state.count)

        def correction(decay: float, dtype: DTypeLike) -> Array:
            if decay == 0.0:
                return jnp.ones((), dtype=dtype)
            numeric_count = jnp.asarray(count, dtype=dtype)
            return -jnp.expm1(jnp.asarray(log(decay), dtype=dtype) * numeric_count)

        def rescale(moment: Array, squared_moment: Array, gradient: Array) -> Array:
            real_dtype = squared_moment.dtype
            first_scale = correction(b1, real_dtype)
            second_scale = correction(b2, real_dtype)
            corrected_first = moment / jnp.asarray(first_scale, dtype=moment.dtype)
            corrected_second = squared_moment / second_scale
            denominator = jnp.sqrt(corrected_second) + jnp.asarray(eps, dtype=real_dtype)
            rate = jnp.asarray(-learning_rate, dtype=moment.dtype)
            result = rate * corrected_first / jnp.asarray(denominator, dtype=moment.dtype)
            return jnp.asarray(result, dtype=gradient.dtype)

        result = jax.tree.map(rescale, first, second, updates)
        return result, optax.ScaleByAdamState(count=count, mu=first, nu=second)

    # Retain the canonical (ScaleByAdamState, EmptyState) chain layout. Scaling
    # precedes the final dtype conversion rather than narrowing its arithmetic.
    return optax.chain(optax.GradientTransformation(initialize, update), optax.identity())
