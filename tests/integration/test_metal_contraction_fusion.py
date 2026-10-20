"""Dynamic weighted uvu contractions through ordinary JAX lowering and AD.

NumPy is an independent host oracle, not an execution fallback. Device cases
require the real explicitly selected MPS wheel and keep every operand FP32.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._jax_config import default_device
from phydrax.ein import contract
from tests._support.metal_qualification import metal_device, record_metal_execution


pytestmark = pytest.mark.metal_native

type RecordProperty = Callable[[str, object], None]


@pytest.fixture
def device() -> Iterator[jax.Device]:
    with default_device(selected := metal_device()):
        yield selected


def _uvu(
    left: jax.Array, coefficients: jax.Array, right: jax.Array, weights: jax.Array
) -> jax.Array:
    coupled = contract("...ui,oij->...uoj", left, coefficients)
    paired = contract("...uoj,...vj->...uvo", coupled, right)
    return contract("...uvo,...uv->...uo", paired, weights)


def _reference(
    left: np.ndarray, coefficients: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    coupled = np.einsum("...ui,oij->...uoj", left, coefficients)
    paired = np.einsum("...uoj,...vj->...uvo", coupled, right)
    return np.einsum("...uvo,...uv->...uo", paired, weights)


def _energy_gradients(
    left: np.ndarray, coefficients: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, ...]:
    coupled = np.einsum("...ui,oij->...uoj", left, coefficients)
    paired = np.einsum("...uoj,...vj->...uvo", coupled, right)
    result = np.einsum("...uvo,...uv->...uo", paired, weights)
    paired_bar = np.einsum("...uo,...uv->...uvo", result, weights)
    coupled_bar = np.einsum("...uvo,...vj->...uoj", paired_bar, right)
    # Explicitly flatten only the host batch axes when summing shared-table AD.
    flat_left = left.reshape((-1,) + left.shape[-2:])
    flat_bar = coupled_bar.reshape((-1,) + coupled_bar.shape[-3:])
    return (
        np.einsum("...uoj,oij->...ui", coupled_bar, coefficients),
        np.einsum("buoj,bui->oij", flat_bar, flat_left),
        np.einsum("...uvo,...uoj->...vj", paired_bar, coupled),
        np.einsum("...uvo,...uo->...uv", paired, result),
    )


def _close(actual: jax.Array, expected: np.ndarray) -> None:
    assert actual.dtype == np.float32
    # Tighter than the existing MACE energy/force first-order 1e-4 gate.
    scale = max(1.0, float(np.max(np.abs(expected))))
    np.testing.assert_allclose(
        np.asarray(actual), expected, rtol=0.0, atol=2**-18 * scale
    )


@pytest.mark.parametrize(
    ("batch", "channels", "dimensions", "right_channels"),
    [
        ((8,), 16, (3, 3, 3), 1),
        ((8,), 8, (1, 3, 3), 1),
        ((8,), 8, (3, 5, 3), 1),
        ((32,), 8, (1, 1, 1), 1),
        ((32,), 8, (3, 3, 1), 1),
        ((32,), 8, (3, 1, 3), 1),
        ((2, 4), 8, (3, 3, 5), 1),
        ((), 8, (3, 3, 3), 1),
        ((4,), 8, (3, 3, 3), 2),
        ((4,), 8, (9, 3, 3), 1),
    ],
)
def test_weighted_uvu_dynamic_forward_jvp_and_reverse(
    device: jax.Device,
    record_property: RecordProperty,
    batch: tuple[int, ...],
    channels: int,
    dimensions: tuple[int, int, int],
    right_channels: int,
) -> None:
    left_dim, right_dim, output_dim = dimensions
    shapes = (
        batch + (channels, left_dim),
        (output_dim, left_dim, right_dim),
        batch + (right_channels, right_dim),
        batch + (channels, right_channels),
    )
    rng = np.random.default_rng(407)
    host = tuple(
        (rng.integers(-8, 9, size=shape) / 16).astype(np.float32) for shape in shapes
    )
    directions = tuple(
        (rng.integers(-4, 5, size=shape) / 32).astype(np.float32) for shape in shapes
    )
    arguments = tuple(jax.device_put(value, device) for value in host)
    tangents = tuple(jax.device_put(value, device) for value in directions)

    @jax.jit
    def program(*args: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        return jax.value_and_grad(
            lambda *xs: 0.5 * jnp.sum(_uvu(*xs) ** 2), argnums=(0, 1, 2, 3)
        )(*args)

    compiled = jax.jit(_uvu)
    forward = compiled(*arguments)
    _, tangent = jax.jit(lambda xs, ts: jax.jvp(_uvu, xs, ts))(arguments, tangents)
    energy, gradients = program(*arguments)
    energy_gradient = jax.grad(
        lambda *xs: 0.5 * jnp.sum(_uvu(*xs) ** 2), argnums=(0, 1, 2, 3)
    )
    _, gradient_tangent = jax.jit(lambda xs, ts: jax.jvp(energy_gradient, xs, ts))(
        arguments, tangents
    )
    expected = _reference(*host)
    tangent_reference = sum(
        (
            _reference(*(directions[j] if j == index else host[j] for j in range(4)))
            for index in range(4)
        ),
        start=np.zeros_like(expected),
    )
    _close(forward, expected)
    _close(tangent, tangent_reference)
    _close(energy, np.asarray(0.5 * np.sum(expected.astype(np.float64) ** 2)))
    for actual, reference in zip(gradients, _energy_gradients(*host), strict=True):
        _close(actual, reference)
    # Complex-step the independent polynomial host gradient, not a device
    # execution or a finite-difference approximation to a JAX result.
    complex_arguments = tuple(
        value.astype(np.complex128) + 1e-30j * direction
        for value, direction in zip(host, directions, strict=True)
    )
    for actual, reference in zip(
        gradient_tangent, _energy_gradients(*complex_arguments), strict=True
    ):
        _close(actual, np.imag(reference) / 1e-30)

    # Repeat one executable with new runtime tables and weights: neither may
    # have been captured as a compile-time coupling table or path constant.
    changed = list(arguments)
    changed[1] = jax.device_put(-host[1], device)
    changed[3] = jax.device_put(2 * host[3], device)
    changed_result = compiled(*changed)
    _close(changed_result, -2 * expected)
    record_metal_execution(
        record_property,
        forward,
        tangent,
        energy,
        *gradients,
        *gradient_tangent,
        changed_result,
    )


def test_weighted_uvu_preserves_staged_reduction(
    device: jax.Device, record_property: RecordProperty
) -> None:
    # Every first-stage product overflows FP32. Multiplication by zero in the
    # second stage must then produce NaN, not be reassociated to a zero product.
    left = np.full((8, 8, 3), 2.0**100, np.float32)
    coefficients = np.full((3, 3, 3), 2.0**30, np.float32)
    right = np.zeros((8, 1, 3), np.float32)
    weights = np.full((8, 8, 1), 0.5, np.float32)
    result = jax.jit(_uvu)(
        *(jax.device_put(x, device) for x in (left, coefficients, right, weights))
    )
    np.testing.assert_array_equal(np.isnan(np.asarray(result)), True)
    record_metal_execution(record_property, result)


def test_weighted_uvu_singleton_signed_zero(
    device: jax.Device, record_property: RecordProperty
) -> None:
    left = np.full((8, 8, 1), -0.0, np.float32)
    coefficients = np.ones((1, 1, 1), np.float32)
    right = np.ones((8, 1, 1), np.float32)
    weights = np.ones((8, 8, 1), np.float32)
    result = jax.jit(_uvu)(
        *(jax.device_put(x, device) for x in (left, coefficients, right, weights))
    )
    expected = np.full((8, 8, 1), -0.0, np.float32)
    np.testing.assert_array_equal(
        np.asarray(result).view(np.uint32), expected.view(np.uint32)
    )
    record_metal_execution(record_property, result)
