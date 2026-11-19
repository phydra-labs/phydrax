"""Point and jet execution of native interval programs against JAX's own evaluation."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry._interval_enclosure import prepare_interval_function
from phydrax.geometry._interval_native import NativeIntervalProgram


_CENTER = jnp.asarray((0.1, -0.2, 0.05), dtype=jnp.float64)


def _torus(x: jax.Array) -> jax.Array:
    shifted = x - _CENTER
    ring = jnp.sqrt(shifted[0] * shifted[0] + shifted[1] * shifted[1]) - 0.742
    return jnp.stack((ring * ring + shifted[2] * shifted[2] - 0.371 * 0.371,))


def _trigonometric(x: jax.Array) -> jax.Array:
    return jnp.stack((jnp.sin(x[0]) * jnp.cos(x[1]) + jnp.tan(0.3 * x[2]),))


def _transcendental(x: jax.Array) -> jax.Array:
    return jnp.stack(
        (
            jnp.exp(0.5 * x[0]) * jnp.log(2.0 + x[1] * x[1])
            + jnp.arctan(x[2]) / (3.0 + x[0] * x[0]),
        )
    )


def _piecewise(x: jax.Array) -> jax.Array:
    return jnp.stack(
        (
            jnp.maximum(x[0], x[1]) - jnp.minimum(x[1], x[2]) + jnp.abs(x[0] - 0.25),
            jnp.where(x[0] > x[2], x[1] * x[1], 1.0 / (2.0 + x[2])),
        )
    )


def _power(x: jax.Array) -> jax.Array:
    base = 1.5 + x[0] * x[0]
    return jnp.stack((base**2.5 + (2.0 + x[1]) ** 3 - jnp.sqrt(4.0 + x[2]),))


_FUNCTIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "torus": _torus,
    "trigonometric": _trigonometric,
    "transcendental": _transcendental,
    "piecewise": _piecewise,
    "power": _power,
}


def _program(function: Callable[[jax.Array], jax.Array]) -> NativeIntervalProgram:
    prepared = prepare_interval_function(
        function, 3, batch_capacity=1, execution="batched"
    )
    program = prepared.native_program
    assert program is not None
    return program


def _points(count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(count, 3))


@pytest.mark.parametrize("name", sorted(_FUNCTIONS))
def test_point_values_match_jax_evaluation(name: str) -> None:
    function = _FUNCTIONS[name]
    points = _points(4096, 20261119)
    expected = np.asarray(jax.vmap(function)(jnp.asarray(points)))
    actual = _program(function).evaluate_points(points)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("name", sorted(_FUNCTIONS))
def test_jets_match_jax_derivatives(name: str) -> None:
    function = _FUNCTIONS[name]
    points = _points(512, 847)
    value, gradient, hessian = _program(function).evaluate_jets(points)
    expected_value = np.asarray(jax.vmap(function)(jnp.asarray(points)))
    expected_gradient = np.asarray(jax.vmap(jax.jacfwd(function))(jnp.asarray(points)))
    expected_hessian = np.asarray(jax.vmap(jax.hessian(function))(jnp.asarray(points)))
    np.testing.assert_allclose(value, expected_value, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-11, atol=1e-11)


def test_point_values_lie_in_the_certified_point_enclosure() -> None:
    prepared = prepare_interval_function(_torus, 3, batch_capacity=1, execution="batched")
    program = prepared.native_program
    assert program is not None
    points = _points(2048, 5)
    lower, upper = program.evaluate_batch(points, points)
    values = program.evaluate_points(points)
    width = np.spacing(np.abs(values)) * 8.0
    assert np.all(lower - width <= values)
    assert np.all(values <= upper + width)


def test_point_execution_refuses_misaligned_points() -> None:
    program = _program(_torus)
    with pytest.raises(ValueError):
        program.evaluate_points(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        program.evaluate_jets(np.zeros((4,)))
