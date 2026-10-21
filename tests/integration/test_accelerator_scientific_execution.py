# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Independent installed-wheel consumer contracts on one actual selected device."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._support.accelerator_qualification import accelerator_device, record_execution


@pytest.fixture(autouse=True)
def execution_device() -> Iterator[jax.Device]:
    if "PHYDRAX_ACCELERATOR" not in os.environ:
        pytest.skip("Explicit accelerator qualification process required")
    device = accelerator_device()
    with jax.default_device(device):
        yield device


def test_mlp_pde_derivatives_and_update(
    record_property: Callable[[str, object], None],
) -> None:
    from examples.accelerator_scientific_execution import poisson_steps

    values, arrays = poisson_steps(steps=4)
    record_execution(record_property, *arrays)
    assert all(np.isfinite(value) for value in values.values())
    assert values["gradient_norm"] > 0 and values["update_norm"] > 0
    assert values["final_loss"] < values["initial_loss"]
    np.testing.assert_allclose(
        values["directional_derivative"],
        values["finite_difference_derivative"],
        rtol=0.03,
        atol=0.02,
    )
    assert all(array.dtype == jnp.float32 for array in arrays)


def test_polynomial_coordinate_gradient_and_hessian(
    record_property: Callable[[str, object], None],
) -> None:
    def polynomial(point: jax.Array) -> jax.Array:
        return point[0] ** 3 * point[1] ** 2

    point = jnp.asarray([0.5, 0.25], dtype=jnp.float32)
    gradient = jax.jit(jax.grad(polynomial))(point)
    hessian = jax.jit(jax.hessian(polynomial))(point)
    record_execution(record_property, gradient, hessian)
    np.testing.assert_array_equal(gradient, np.asarray([0.046875, 0.0625], np.float32))
    np.testing.assert_array_equal(
        hessian, np.asarray([[0.1875, 0.375], [0.375, 0.25]], np.float32)
    )


@pytest.mark.parametrize("kind", ("int64", "uint64"))
def test_wide_integer_sort_and_arithmetic(
    record_property: Callable[[str, object], None], kind: str
) -> None:
    values = (
        np.asarray([2**60 + 3, 2**60 + 1, 2**60 + 3, 2**60 + 2], np.int64)
        if kind == "int64"
        else np.asarray([2**63 + 7, 2**63 + 2, 2**63 + 9, 2**63 + 1], np.uint64)
    )

    @jax.jit
    def execute(keys: jax.Array) -> tuple[jax.Array, ...]:
        order = jnp.argsort(keys, stable=True)
        return order, keys[order], keys + jnp.asarray(11, dtype=keys.dtype)

    outputs = execute(jnp.asarray(values))
    record_execution(record_property, *outputs)
    np.testing.assert_array_equal(outputs[0], np.argsort(values, kind="stable"))
    np.testing.assert_array_equal(outputs[1], np.sort(values, kind="stable"))
    np.testing.assert_array_equal(outputs[2], values + np.asarray(11, dtype=values.dtype))
    if kind == "uint64":
        flipped = jax.jit(lambda value: value ^ jnp.uint64(2**62))(jnp.asarray(values))
        record_execution(record_property, flipped)
        np.testing.assert_array_equal(flipped, values ^ np.uint64(2**62))


def test_point_gather_scatter_full_width_addresses(
    record_property: Callable[[str, object], None],
) -> None:
    host = np.asarray([2**60 + 3, 2**60 + 1, 2**60 + 3, 2**60 + 2], np.int64)

    @jax.jit
    def execute(keys: jax.Array) -> tuple[jax.Array, jax.Array]:
        addresses = jnp.asarray([0, 2**32 + 1, 2, -(2**32) + 1], dtype=jnp.int64)
        gathered = jnp.take(keys, addresses, mode="fill", fill_value=-17)
        scattered = jnp.zeros((4,), dtype=jnp.int64).at[addresses].set(keys, mode="drop")
        return gathered, scattered

    gathered, scattered = execute(jnp.asarray(host))
    record_execution(record_property, gathered, scattered)
    np.testing.assert_array_equal(
        gathered, np.asarray([host[0], -17, host[2], -17], np.int64)
    )
    np.testing.assert_array_equal(
        scattered, np.asarray([host[0], 0, host[2], 0], np.int64)
    )


def _starts(kind: str) -> tuple[int, ...]:
    return (
        (0, 1, 3, 4, 2**32 + 1, -(2**32) + 1, -(2**63), 2**63 - 1)
        if kind == "int64"
        else (0, 1, 3, 4, 2**32 + 1, 2**63, 2**64 - 1)
    )


@pytest.mark.parametrize("kind", ("int64", "uint64"))
def test_full_width_window_gather(
    record_property: Callable[[str, object], None], kind: str
) -> None:
    starts = _starts(kind)
    indices = np.asarray(starts, dtype=np.dtype(kind))
    operand = np.asarray([2**60 + 1, 2**60 + 3, 7, 9, 11], np.int64)
    dimensions = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,), collapsed_slice_dims=(), start_index_map=(0,)
    )

    @jax.jit
    def windows(values: jax.Array, addresses: jax.Array) -> jax.Array:
        return jax.lax.gather(
            values,
            addresses[:, None],
            dimensions,
            slice_sizes=(2,),
            mode="fill",
            fill_value=-17,
        )

    gathered = windows(jnp.asarray(operand), jnp.asarray(indices))
    expected = np.stack(
        [
            operand[start : start + 2] if 0 <= start <= 3 else np.full(2, -17, np.int64)
            for start in starts
        ]
    )
    record_execution(record_property, gathered)
    np.testing.assert_array_equal(gathered, expected)


@pytest.mark.parametrize("kind", ("int64", "uint64"))
def test_scatter_update_cotangent_drops_out_of_bounds(
    record_property: Callable[[str, object], None], kind: str
) -> None:
    dimensions = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(0,),
        inserted_window_dims=(),
        scatter_dims_to_operand_dims=(0,),
    )

    @jax.jit
    def transpose(address: jax.Array, updates: jax.Array) -> jax.Array:
        def loss(value: jax.Array) -> jax.Array:
            return jnp.sum(
                jax.lax.scatter_add(
                    jnp.zeros(5, dtype=jnp.float32),
                    address,
                    value,
                    dimensions,
                    mode="drop",
                    unique_indices=True,
                )
            )

        return jax.grad(loss)(updates)

    for start in _starts(kind):
        cotangent = transpose(
            jnp.asarray([start], dtype=np.dtype(kind)), jnp.ones(2, dtype=jnp.float32)
        )
        record_execution(record_property, cotangent)
        np.testing.assert_array_equal(
            cotangent,
            np.ones(2, np.float32) if 0 <= start <= 3 else np.zeros(2, np.float32),
        )


def test_compiled_loops_and_ordered_callbacks(
    record_property: Callable[[str, object], None],
) -> None:
    events: list[int] = []

    def observe(value: np.ndarray) -> None:
        events.append(int(value))

    @jax.jit
    def execute(start: jax.Array) -> jax.Array:
        def body(index: jax.Array, value: jax.Array) -> jax.Array:
            jax.debug.callback(observe, index, ordered=True)
            return jax.lax.cond(index % 2 == 0, lambda x: x + 2, lambda x: x - 1, value)

        return jax.lax.fori_loop(0, 6, body, start)

    result = execute(jnp.asarray(0, dtype=jnp.int64))
    result.block_until_ready()
    jax.effects_barrier()
    assert int(result) == 3 and events == list(range(6))
    record_execution(record_property, result)

    @jax.jit
    def other_loops(start: jax.Array) -> tuple[jax.Array, jax.Array]:
        def condition(value: jax.Array) -> jax.Array:
            return value < 6

        def advance(value: jax.Array) -> jax.Array:
            return value + 1

        def accumulate(value: jax.Array, index: jax.Array) -> tuple[jax.Array, None]:
            return value + index, None

        counted = jax.lax.while_loop(condition, advance, start)
        summed, _ = jax.lax.scan(accumulate, start, jnp.arange(6, dtype=jnp.int64))
        return counted, summed

    counted, summed = other_loops(jnp.asarray(0, dtype=jnp.int64))
    record_execution(record_property, counted, summed)
    assert int(counted) == 6 and int(summed) == 15


def test_compiled_callback_failure(
    record_property: Callable[[str, object], None],
) -> None:
    @eqx.filter_jit
    def guarded(value: jax.Array) -> jax.Array:
        return eqx.error_if(value, value > 0, "accelerator callback failure sentinel")

    accepted = guarded(jnp.asarray(0, dtype=jnp.int64))
    record_execution(record_property, accepted)
    with pytest.raises(Exception, match="accelerator callback failure sentinel"):
        jax.block_until_ready(guarded(jnp.asarray(1, dtype=jnp.int64)))


@pytest.fixture
def linear_problem() -> tuple[Any, jax.Array]:
    import phydrax as phx

    la = phx.linalg
    space = la.ArraySpace((8,), dtype=jnp.float32)

    def action(value: jax.Array) -> jax.Array:
        return (
            3 * value
            - jnp.concatenate((jnp.zeros((1,), value.dtype), value[:-1]))
            - jnp.concatenate((value[1:], jnp.zeros((1,), value.dtype)))
        )

    properties = la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )
    problem = la.LinearSystem(
        la.FunctionLinearOperator(
            action,
            source=space,
            target=space,
            transpose_action=action,
            properties=properties,
        )
    )
    return problem, jnp.asarray(np.arange(1, 9, dtype=np.float32))


def test_bounded_linear_solve(
    record_property: Callable[[str, object], None], linear_problem: tuple[Any, jax.Array]
) -> None:
    import phydrax as phx

    la = phx.linalg
    problem, rhs = linear_problem
    policy = la.LinearSolvePolicy(
        la.PCG(),
        tolerance=la.TolerancePolicy(relative=1e-4, absolute=0.0, max_steps=32),
        differentiation=la.DifferentiationPolicy("none"),
        failure=la.FailurePolicy("status"),
    )

    def solve(target: jax.Array) -> Any:
        return la.solve(problem, target, policy=policy)

    result = eqx.filter_jit(solve)(rhs)
    record_execution(
        record_property, result.value, result.status, result.diagnostics.residual_norm
    )
    assert int(result.status) == int(la.LinearSolveStatus.SUCCESS)
    matrix = 3 * np.eye(8) - np.eye(8, k=-1) - np.eye(8, k=1)
    host_rhs = np.arange(1, 9, dtype=np.float64)
    true_residual = np.linalg.norm(matrix @ np.asarray(result.value) - host_rhs)
    assert true_residual <= 1.1e-4 * np.linalg.norm(host_rhs)
    np.testing.assert_allclose(
        result.value, np.linalg.solve(matrix, host_rhs), rtol=2e-4, atol=2e-5
    )


def test_linear_iteration_exhaustion(
    record_property: Callable[[str, object], None], linear_problem: tuple[Any, jax.Array]
) -> None:
    import phydrax as phx

    la = phx.linalg
    problem, rhs = linear_problem
    policy = la.LinearSolvePolicy(
        la.PCG(),
        tolerance=la.TolerancePolicy(relative=1e-8, absolute=0.0, max_steps=1),
        differentiation=la.DifferentiationPolicy("none"),
        failure=la.FailurePolicy("status"),
    )
    failed = la.solve(problem, rhs, policy=policy)
    record_execution(record_property, failed.value, failed.status)
    assert int(failed.status) == int(la.LinearSolveStatus.MAXIMUM_STEPS_REACHED)


def test_real_fourier_resampling(record_property: Callable[[str, object], None]) -> None:
    from phydrax.signal import fourier_resample

    source = np.cos(2 * np.pi * np.arange(8) / 8).astype(np.float32)

    @jax.jit
    def resample(values: jax.Array) -> jax.Array:
        return fourier_resample(values, (13,))

    output = resample(jnp.asarray(source))
    derivative = jax.jit(jax.grad(lambda values: jnp.sum(resample(values))))(
        jnp.asarray(source)
    )
    record_execution(record_property, output, derivative)
    np.testing.assert_allclose(
        output, np.cos(2 * np.pi * np.arange(13) / 13), atol=3e-5, rtol=3e-5
    )
    np.testing.assert_allclose(derivative, np.full(8, 13 / 8), atol=3e-5, rtol=3e-5)


@pytest.fixture
def rocm_precision_problem(
    kind: str,
) -> tuple[Any, jax.Array, np.ndarray, np.ndarray]:
    """A small HPD system with a non-real operator in the complex case."""
    if os.environ.get("PHYDRAX_ACCELERATOR") != "rocm10":
        pytest.skip("Bounded ROCm precision contract")
    import phydrax as phx

    dtype = np.dtype(kind)
    matrix = (4 * np.eye(8) - np.eye(8, k=1) - np.eye(8, k=-1)).astype(dtype)
    rhs = np.arange(1, 9, dtype=np.float64).astype(dtype)
    if kind == "complex128":
        phases = np.exp(0.3j * np.arange(8))
        matrix = phases[:, None] * matrix * phases.conj()[None, :]
        rhs = rhs + 1j * rhs[::-1]
    operator = phx.linalg.DenseLinearOperator(
        jnp.asarray(matrix),
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
    )
    return phx.linalg.LinearSystem(operator), jnp.asarray(rhs), matrix, rhs


@pytest.mark.parametrize("kind", ("float64", "complex128"))
def test_rocm_native_linear_precision(
    record_property: Callable[[str, object], None],
    kind: str,
    rocm_precision_problem: tuple[Any, jax.Array, np.ndarray, np.ndarray],
) -> None:
    import phydrax as phx

    la = phx.linalg
    problem, rhs, matrix, host_rhs = rocm_precision_problem
    policy = la.LinearSolvePolicy(
        la.PCG(),
        tolerance=la.TolerancePolicy(relative=1e-12, absolute=1e-14, max_steps=32),
        differentiation=la.DifferentiationPolicy("mathematical"),
        derivative_solve=la.LinearDerivativeSolvePolicy(
            relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_steps=32
        ),
        failure=la.FailurePolicy("status"),
    )

    def solve(target: jax.Array) -> Any:
        return la.solve(problem, target, policy=policy)

    direction_host = np.linspace(-0.7, 0.9, 8).astype(np.dtype(kind))
    if kind == "complex128":
        direction_host = direction_host + 0.4j * direction_host[::-1]
    direction = jnp.asarray(direction_host)
    result = eqx.filter_jit(solve)(rhs)
    _, tangent = eqx.filter_jit(
        lambda target, vector: jax.jvp(
            lambda value: solve(value).value, (target,), (vector,)
        )
    )(rhs, direction)
    record_execution(
        record_property,
        rhs,
        result.value,
        result.status,
        result.diagnostics.residual_norm,
        result.derivative_valid,
        tangent,
    )
    assert rhs.dtype == result.value.dtype == tangent.dtype == np.dtype(kind)
    assert result.diagnostics.residual_norm.dtype == jnp.float64
    assert int(result.status) == int(la.LinearSolveStatus.SUCCESS)
    assert bool(result.derivative_valid)
    np.testing.assert_allclose(
        result.value, np.linalg.solve(matrix, host_rhs), rtol=2e-11, atol=2e-12
    )
    np.testing.assert_allclose(
        tangent, np.linalg.solve(matrix, direction_host), rtol=2e-11, atol=2e-12
    )
    assert np.linalg.norm(matrix @ np.asarray(result.value) - host_rhs) <= (
        2e-12 * np.linalg.norm(host_rhs)
    )


@pytest.mark.parametrize("kind", ("float64", "complex128"))
def test_rocm_linear_precision_exhaustion(
    record_property: Callable[[str, object], None],
    kind: str,
    rocm_precision_problem: tuple[Any, jax.Array, np.ndarray, np.ndarray],
) -> None:
    import phydrax as phx

    la = phx.linalg
    problem, rhs, matrix, host_rhs = rocm_precision_problem
    policy = la.LinearSolvePolicy(
        la.PCG(),
        tolerance=la.TolerancePolicy(relative=1e-12, absolute=0.0, max_steps=1),
        differentiation=la.DifferentiationPolicy("none"),
        failure=la.FailurePolicy("status"),
    )
    failed = eqx.filter_jit(lambda target: la.solve(problem, target, policy=policy))(rhs)
    record_execution(
        record_property, failed.value, failed.status, failed.diagnostics.residual_norm
    )
    assert failed.value.dtype == np.dtype(kind)
    assert int(failed.status) == int(la.LinearSolveStatus.MAXIMUM_STEPS_REACHED)
    assert np.linalg.norm(matrix @ np.asarray(failed.value) - host_rhs) > (
        1e-12 * np.linalg.norm(host_rhs)
    )


@pytest.mark.parametrize("kind", ("float64", "complex128"))
def test_rocm_fourier_precision(
    record_property: Callable[[str, object], None], kind: str
) -> None:
    if os.environ.get("PHYDRAX_ACCELERATOR") != "rocm10":
        pytest.skip("Bounded ROCm precision contract")
    from phydrax.signal import fourier_resample

    def samples(size: int) -> np.ndarray:
        angle = 2 * np.pi * np.arange(size) / size
        values = np.cos(angle) + 0.25 * np.sin(2 * angle)
        if kind == "complex128":
            values = values + 0.6j * np.sin(3 * angle)
        return values.astype(np.dtype(kind))

    # Analytic band-limited modes give an oracle independent of FFT code.
    source = jnp.asarray(samples(16))
    direction = jnp.asarray(2 * samples(16)[::-1])

    @jax.jit
    def resample(values: jax.Array) -> jax.Array:
        return fourier_resample(values, (23,))

    output, tangent = jax.jit(
        lambda values, vector: jax.jvp(resample, (values,), (vector,))
    )(source, direction)
    angle = 2 * np.pi * (15 / 16 - np.arange(23) / 23)
    expected_tangent = 2 * (np.cos(angle) + 0.25 * np.sin(2 * angle))
    if kind == "complex128":
        expected_tangent = expected_tangent + 1.2j * np.sin(3 * angle)
    record_execution(record_property, source, output, tangent)
    assert source.dtype == output.dtype == tangent.dtype == np.dtype(kind)
    np.testing.assert_allclose(output, samples(23), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=2e-12, atol=2e-12)
