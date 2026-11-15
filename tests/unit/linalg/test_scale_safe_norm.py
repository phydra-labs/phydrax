# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Numerical and differentiation contracts for the scale-safe norm."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.linalg._norm import scaled_norm


pytestmark = pytest.mark.strict_jax


@pytest.mark.parametrize("scale", [1e200, 1e-200], ids=["overflow", "underflow"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_frobenius_norm_at_extreme_scales(scale: float, compiled: bool) -> None:
    vector = jnp.asarray([[3.0, 0.0], [0.0, 4.0]]) * scale
    norm = jax.jit(scaled_norm) if compiled else scaled_norm
    np.testing.assert_allclose(norm(vector) / scale, 5.0, rtol=1e-14)


@pytest.mark.parametrize("scale", [1e200, 1e-200], ids=["overflow", "underflow"])
def test_complex_and_declared_metric_at_extreme_scales(scale: float) -> None:
    vector = jnp.asarray([3.0 + 4.0j, 0.0 + 12.0j]) * scale

    # |3 + 4i| = 5; weighting the first component by four gives sqrt(244).
    def inner(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.vdot(x, jnp.asarray([4.0, 1.0], dtype=y.dtype) * y)

    np.testing.assert_allclose(scaled_norm(vector) / scale, 13.0, rtol=1e-14)
    np.testing.assert_allclose(
        scaled_norm(vector, inner) / scale, np.sqrt(244.0), rtol=1e-14
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("metric", [False, True], ids=["euclidean", "weighted"])
@pytest.mark.parametrize("complex_input", [False, True], ids=["real", "complex"])
def test_zero_has_finite_zero_jvp_and_vjp(
    compiled: bool, metric: bool, complex_input: bool
) -> None:
    def inner(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.vdot(x, jnp.asarray([4.0, 1.0], dtype=y.dtype) * y)

    def norm(x: jax.Array) -> jax.Array:
        return scaled_norm(x, inner if metric else None)

    selected = jax.jit(norm) if compiled else norm
    zero = jnp.zeros(2, dtype=jnp.complex128 if complex_input else jnp.float64)
    value, tangent = jax.jvp(selected, (zero,), (jnp.ones_like(zero),))
    _, pullback = jax.vjp(selected, zero)
    (gradient,) = pullback(jnp.asarray(1.0))
    np.testing.assert_array_equal(value, 0.0)
    np.testing.assert_array_equal(tangent, 0.0)
    np.testing.assert_array_equal(gradient, np.zeros(2))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("scale", [1.0, 1e200, 1e-200], ids=["unit", "large", "small"])
def test_nonzero_complex_jvp_and_vjp(compiled: bool, scale: float) -> None:
    norm = jax.jit(scaled_norm) if compiled else scaled_norm
    vector = jnp.asarray([3.0 + 4.0j, 12.0j], dtype=jnp.complex128) * scale
    direction = jnp.asarray([1.0 + 2.0j, 3.0 - 1.0j])
    value, tangent = jax.jvp(norm, (vector,), (direction,))
    _, pullback = jax.vjp(norm, vector)
    (gradient,) = pullback(jnp.asarray(1.0))
    np.testing.assert_allclose(value / scale, 13.0, rtol=1e-14)
    np.testing.assert_allclose(tangent, -1.0 / 13.0, rtol=1e-14)
    np.testing.assert_allclose(
        gradient, np.conjugate(np.asarray(vector) / scale) / 13.0, rtol=1e-14
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_derivative_with_respect_to_metric(compiled: bool) -> None:
    vector = jnp.asarray([3.0, 4.0])

    def norm(weights: jax.Array) -> jax.Array:
        return scaled_norm(vector, lambda x, y: jnp.vdot(x, weights * y))

    selected = jax.jit(norm) if compiled else norm
    weights = jnp.asarray([4.0, 1.0])
    expected = np.asarray([9.0, 16.0]) / (2.0 * np.sqrt(52.0))
    _, tangent = jax.jvp(selected, (weights,), (jnp.asarray([1.0, 0.0]),))
    _, pullback = jax.vjp(selected, weights)
    (gradient,) = pullback(jnp.asarray(1.0))
    np.testing.assert_allclose(tangent, expected[0], rtol=1e-14)
    np.testing.assert_allclose(gradient, expected, rtol=1e-14)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("metric", [False, True], ids=["euclidean", "weighted"])
def test_nonfinite_input_is_not_reported_as_finite(
    bad: float, compiled: bool, metric: bool
) -> None:
    def inner(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.vdot(x, jnp.asarray([4.0, 1.0]) * y)

    def norm(x: jax.Array) -> jax.Array:
        return scaled_norm(x, inner if metric else None)

    selected = jax.jit(norm) if compiled else norm
    assert not bool(jnp.isfinite(selected(jnp.asarray([bad, 1.0]))))


def test_complex_components_near_float_limit_keep_representable_norm() -> None:
    vector = jnp.asarray([1e308 + 1e308j], dtype=jnp.complex128)
    np.testing.assert_allclose(
        jax.jit(scaled_norm)(vector) / 1e308, np.sqrt(2.0), rtol=1e-14
    )
