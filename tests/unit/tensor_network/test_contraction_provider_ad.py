"""Native prepared replay AD; optional CUDA primitive AD needs GPU qualification.

Preparation remains a host operation. These tests replace only dynamic operands
of an already prepared native executor, never emulate a vendor route.
"""

from collections.abc import Iterator, Mapping, Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax import config
from phydrax.tensor_network import PreparedContraction


tn = phx.tensor_network


@pytest.fixture(autouse=True)
def _cpu_execution() -> Iterator[None]:
    with (
        config.context(config.NumericsConfig(enable_x64=True)),
        jax.default_device(jax.devices("cpu")[0]),
    ):
        yield


def _prepared(
    inputs: Sequence[str],
    outputs: str,
    dimensions: Mapping[str, int],
    arrays: Sequence[ArrayLike],
    *,
    dtype: str = "float64",
) -> PreparedContraction:
    structure = tn.ContractionStructure(
        tuple(
            tn.ContractionOperand(
                f"operand-{index}",
                tuple(tn.ContractionLeg(label, dimensions[label]) for label in labels),
            )
            for index, labels in enumerate(inputs)
        ),
        tuple(outputs),
    )
    path = tuple((0, 1) for _ in range(len(inputs) - 1))
    plan = tn.plan_contraction(structure, path=path, dtype=dtype)
    prepared = tn.prepare_contraction(plan, arrays)
    assert prepared.executor.policy.provider == "native"
    return prepared


def _replay(prepared: PreparedContraction, *operands: Array) -> Array:
    dynamic = eqx.tree_at(lambda item: item.operands, prepared, tuple(operands))
    return tn.execute_contraction(dynamic).value


@pytest.mark.parametrize(
    "inputs,outputs,dimensions,shapes",
    [
        (("ii",), "", {"i": 3}, ((3, 3),)),
        (("iij",), "ji", {"i": 2, "j": 3}, ((2, 2, 3),)),
        (("", "i"), "i", {"i": 3}, ((), (3,))),
        (("",), "", {}, ((),)),
        (("i", "j"), "ji", {"i": 2, "j": 3}, ((2,), (3,))),
        (("ijk",), "kij", {"i": 2, "j": 3, "k": 4}, ((2, 3, 4),)),
        (("ij", "jk"), "ki", {"i": 2, "j": 3, "k": 4}, ((2, 3), (3, 4))),
        (("ij", "j"), "", {"i": 2, "j": 3}, ((2, 3), (3,))),
        (("i", "i", "i"), "", {"i": 3}, ((3,), (3,), (3,))),
        (
            ("ij", "jk", "jl"),
            "kil",
            {"i": 2, "j": 3, "k": 2, "l": 4},
            ((2, 3), (3, 2), (3, 4)),
        ),
    ],
)
def test_native_layout_scalar_diagonal_and_hyperedge_values_and_jvp(
    inputs: tuple[str, ...],
    outputs: str,
    dimensions: dict[str, int],
    shapes: tuple[tuple[int, ...], ...],
) -> None:
    arrays = tuple(
        (np.arange(prod(shape), dtype=np.float64).reshape(shape) + index + 1) / 7
        for index, shape in enumerate(shapes)
    )
    tangents = tuple(
        np.full_like(value, (index + 1) / 11) for index, value in enumerate(arrays)
    )
    prepared = _prepared(inputs, outputs, dimensions, arrays)
    equation = ",".join(inputs) + "->" + outputs
    expected = np.einsum(equation, *arrays)
    # Multilinearity gives an independent product-rule oracle, including all
    # retained hyperedges, repeated labels, and scalar factors.
    expected_tangent = np.zeros_like(expected)
    for index, tangent in enumerate(tangents):
        replaced = list(arrays)
        replaced[index] = tangent
        expected_tangent = expected_tangent + np.einsum(equation, *replaced)
    function = lambda *values: _replay(prepared, *values)
    values = tuple(jnp.asarray(value) for value in arrays)
    directions = tuple(jnp.asarray(value) for value in tangents)
    actual, tangent = jax.jvp(function, values, directions)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        jax.jit(function)(*values), expected, rtol=1e-12, atol=1e-12
    )


def test_repeated_labels_transpose_scatter_only_onto_diagonal() -> None:
    value = np.arange(12, dtype=np.float64).reshape(2, 2, 3) / 5
    prepared = _prepared(("iij",), "ji", {"i": 2, "j": 3}, (value,))
    weights = np.asarray([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    _, pullback = jax.vjp(lambda x: _replay(prepared, x), jnp.asarray(value))
    (gradient,) = pullback(jnp.asarray(weights))
    expected = np.zeros_like(value)
    for index in range(2):
        expected[index, index, :] = weights[:, index]
    np.testing.assert_array_equal(gradient, expected)
    # A constant incoming cotangent makes the second derivative exactly zero.
    hessian = jax.jacfwd(jax.grad(lambda x: jnp.sum(_replay(prepared, x) * weights)))(
        jnp.asarray(value)
    )
    np.testing.assert_array_equal(hessian, np.zeros(value.shape + value.shape))


def test_exclusive_reduced_mode_transpose_broadcasts_over_missing_axis() -> None:
    matrix = np.arange(6, dtype=np.float64).reshape(2, 3) / 3
    vector = np.asarray([2.0, -3.0, 5.0])
    prepared = _prepared(("ij", "j"), "", {"i": 2, "j": 3}, (matrix, vector))
    function = lambda x, y: _replay(prepared, x, y)
    gx, gy = jax.grad(function, argnums=(0, 1))(jnp.asarray(matrix), jnp.asarray(vector))
    np.testing.assert_array_equal(gx, np.broadcast_to(vector, matrix.shape))
    np.testing.assert_allclose(gy, np.sum(matrix, axis=0))
    mixed = jax.jacfwd(jax.grad(function, argnums=0), argnums=1)(
        jnp.asarray(matrix), jnp.asarray(vector)
    )
    np.testing.assert_array_equal(mixed, np.broadcast_to(np.eye(3), (2, 3, 3)))


def test_retained_hyperedge_first_and_mixed_second_derivatives() -> None:
    x = np.asarray([1.0, 2.0, -3.0])
    y = np.asarray([4.0, -5.0, 6.0])
    z = np.asarray([-2.0, 3.0, 7.0])
    prepared = _prepared(("i", "i", "i"), "", {"i": 3}, (x, y, z))
    function = lambda a, b, c: _replay(prepared, a, b, c)
    gradients = jax.grad(function, argnums=(0, 1, 2))(*map(jnp.asarray, (x, y, z)))
    for actual, expected in zip(gradients, (y * z, x * z, x * y), strict=True):
        np.testing.assert_array_equal(actual, expected)
    mixed = jax.jacfwd(jax.grad(function, argnums=0), argnums=1)(
        *map(jnp.asarray, (x, y, z))
    )
    np.testing.assert_array_equal(mixed, np.diag(z))
    same = jax.jacfwd(jax.grad(function, argnums=0), argnums=0)(
        *map(jnp.asarray, (x, y, z))
    )
    np.testing.assert_array_equal(same, np.zeros((3, 3)))


def test_repeated_operand_accumulates_both_contributions_through_second_order() -> None:
    x = np.asarray([1.0, -2.0, 4.0])
    prepared = _prepared(("i", "i"), "", {"i": 3}, (x, x))
    function = lambda value: _replay(prepared, value, value)
    value = jnp.asarray(x)
    np.testing.assert_array_equal(function(value), np.sum(x * x))
    np.testing.assert_array_equal(jax.grad(function)(value), 2 * x)
    np.testing.assert_array_equal(jax.hessian(function)(value), 2 * np.eye(3))
    direction = jnp.asarray([2.0, 3.0, -1.0])
    _, hvp = jax.jvp(jax.grad(function), (value,), (direction,))
    np.testing.assert_array_equal(hvp, 2 * np.asarray(direction))


def test_zero_tangents_and_closed_operands_do_not_create_extra_terms() -> None:
    x = np.asarray([1.0, -2.0, 4.0])
    y = np.asarray([3.0, 5.0, -7.0])
    prepared = _prepared(("i", "i"), "", {"i": 3}, (x, y))
    values = (jnp.asarray(x), jnp.asarray(y))
    function = lambda a, b: _replay(prepared, a, b)
    _, zero = jax.jvp(function, values, tuple(jnp.zeros_like(value) for value in values))
    np.testing.assert_array_equal(zero, 0.0)
    # The closed second operand has no tangent in this trace.
    closed = lambda a: _replay(prepared, a, values[1])
    np.testing.assert_array_equal(jax.grad(closed)(values[0]), y)
    np.testing.assert_array_equal(jax.hessian(closed)(values[0]), np.zeros((3, 3)))
    _, tangent = jax.jvp(
        function, values, (jnp.ones_like(values[0]), jnp.zeros_like(values[1]))
    )
    np.testing.assert_array_equal(tangent, np.sum(y))


def test_complex_bilinear_jvp_and_vjp_do_not_unconditionally_conjugate() -> None:
    x = np.asarray([1 + 2j, -3 + 4j], dtype=np.complex128)
    y = np.asarray([2 - 5j, 7 + 3j], dtype=np.complex128)
    dx = np.asarray([3 + 1j, -2 + 6j], dtype=np.complex128)
    dy = np.asarray([-1 + 4j, 5 - 2j], dtype=np.complex128)
    prepared = _prepared(("i", "i"), "", {"i": 2}, (x, y), dtype="complex128")
    function = lambda a, b: _replay(prepared, a, b)
    actual, tangent = jax.jvp(
        function, (jnp.asarray(x), jnp.asarray(y)), (jnp.asarray(dx), jnp.asarray(dy))
    )
    np.testing.assert_allclose(actual, np.sum(x * y))
    np.testing.assert_allclose(tangent, np.sum(dx * y + x * dy))
    _, pullback = jax.vjp(function, jnp.asarray(x), jnp.asarray(y))
    cotangent = 2 - 3j
    gx, gy = pullback(jnp.asarray(cotangent, dtype=jnp.complex128))
    np.testing.assert_allclose(gx, cotangent * y)
    np.testing.assert_allclose(gy, cotangent * x)


def test_native_batching_rebinds_each_lanes_values() -> None:
    x = np.asarray([1.0, 2.0, 3.0])
    y = np.asarray([4.0, 5.0, 6.0])
    prepared = _prepared(("i", "i"), "", {"i": 3}, (x, y))
    batch = np.stack((x, -x, x + 3))
    function = lambda a: _replay(prepared, a, jnp.asarray(y))
    np.testing.assert_allclose(jax.jit(jax.vmap(function))(jnp.asarray(batch)), batch @ y)
    gradients = jax.vmap(jax.grad(function))(jnp.asarray(batch))
    np.testing.assert_array_equal(gradients, np.broadcast_to(y, batch.shape))
