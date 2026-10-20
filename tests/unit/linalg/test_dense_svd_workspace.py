# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Economy-factor reconstruction, differentiation, and resource contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.linalg as la
from phydrax._execution_device import device_platform
from phydrax.linalg._dense_svd import economy_svd


def _matrix(rows: int, columns: int, complex_values: bool) -> np.ndarray:
    rng = np.random.default_rng(52)
    value = rng.standard_normal((2, rows, columns))
    if complex_values:
        value = value + 1j * rng.standard_normal(value.shape)
    return value


@pytest.mark.parametrize(
    "shape", ((12, 3), (3, 12), (5, 3)), ids=("tall", "wide", "near-square")
)
@pytest.mark.parametrize("complex_values", (False, True), ids=("real", "complex"))
@pytest.mark.parametrize(
    "driver",
    (
        None,
        jax.lax.linalg.SvdAlgorithm.QR,
        jax.lax.linalg.SvdAlgorithm.DIVIDE_AND_CONQUER,
    ),
    ids=("default", "qr", "divide-and-conquer"),
)
def test_batched_economy_svd_factors_preserve_contract(
    shape: tuple[int, int],
    complex_values: bool,
    driver: jax.lax.linalg.SvdAlgorithm | None,
) -> None:
    if (
        driver == jax.lax.linalg.SvdAlgorithm.DIVIDE_AND_CONQUER
        and device_platform(jax.devices()[0]) == "cuda"
    ):
        pytest.skip("Divide-and-conquer is a CPU driver.")
    matrix = _matrix(*shape, complex_values)
    left, values, right = jax.jit(lambda value: economy_svd(value, algorithm=driver))(
        jnp.asarray(matrix)
    )
    np.testing.assert_allclose(
        values, np.linalg.svd(matrix, compute_uv=False), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        (left * values[..., None, :]) @ right, matrix, rtol=1e-12, atol=1e-12
    )
    identity = np.broadcast_to(np.eye(min(shape)), (2, min(shape), min(shape)))
    np.testing.assert_allclose(left.conj().swapaxes(-1, -2) @ left, identity, atol=1e-12)
    np.testing.assert_allclose(
        right @ right.conj().swapaxes(-1, -2), identity, atol=1e-12
    )


@pytest.mark.parametrize("shape", ((12, 3), (3, 12)), ids=("tall", "wide"))
@pytest.mark.parametrize("complex_values", (False, True), ids=("real", "complex"))
@pytest.mark.parametrize("mode", ("jvp", "vjp"))
def test_economy_svd_derivative_matches_independent_difference(
    shape: tuple[int, int], complex_values: bool, mode: str
) -> None:
    matrix = _matrix(*shape, complex_values)
    direction = _matrix(*shape, complex_values)[:, ::-1, ::-1] * 0.1
    probe = np.random.default_rng(91).standard_normal((2, shape[1], shape[0]))

    def objective(parameter: Array) -> Array:
        left, values, right = economy_svd(
            jnp.asarray(matrix) + parameter * jnp.asarray(direction)
        )
        inverse = (
            right.conj().swapaxes(-1, -2) / values[..., None, :]
        ) @ left.conj().swapaxes(-1, -2)
        return jnp.real(jnp.sum(inverse * jnp.asarray(probe))) + jnp.sum(values)

    def reference(parameter: float) -> float:
        value = matrix + parameter * direction
        return float(
            np.real(np.sum(np.linalg.pinv(value) * probe))
            + np.sum(np.linalg.svd(value, compute_uv=False))
        )

    step = 1e-5
    expected = (reference(step) - reference(-step)) / (2 * step)
    parameter = jnp.asarray(0.0, dtype=jnp.float64)
    actual = (
        jax.jvp(objective, (parameter,), (jnp.ones_like(parameter),))[1]
        if mode == "jvp"
        else jax.grad(objective)(parameter)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)


def test_hermitian_economy_svd_of_a_singular_matrix_keeps_unitary_factors() -> None:
    # Indefinite with an exactly zero eigenvalue: the zero singular value has
    # no eigenvalue sign to transfer onto its right vector.
    rotation = np.asarray([[1.0, 1j, 0.0], [1j, 1.0, 0.0], [0.0, 0.0, np.sqrt(2.0)]])
    rotation = rotation / np.sqrt(2.0)
    matrix = (rotation * np.asarray([2.0, -1.0, 0.0])) @ rotation.conj().T
    left, values, right = economy_svd(jnp.asarray(matrix), hermitian=True)

    np.testing.assert_allclose(values, [2.0, 1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose((left * values) @ right, matrix, atol=1e-15)
    identity = np.eye(3)
    np.testing.assert_allclose(left.conj().T @ left, identity, atol=1e-15)
    np.testing.assert_allclose(right @ right.conj().T, identity, atol=1e-15)


@pytest.mark.parametrize(
    "shape",
    ((1, 1000), (1000, 1), (3, 12), (12, 3), (3, 5)),
    ids=("extreme-wide", "extreme-tall", "wide", "tall", "near-square"),
)
def test_dense_svd_factor_workspace_is_bounded_by_plan(shape: tuple[int, int]) -> None:
    matrix = jnp.asarray(_matrix(*shape, False))
    problem = la.LeastSquaresProblem(la.DenseLinearOperator(matrix))
    estimate = la.plan(problem, la.LinearSolvePolicy(la.DenseSVD())).candidates[-1]
    executable = jax.jit(economy_svd).lower(matrix).compile()
    memory = executable.memory_analysis()
    assert memory is not None
    assert memory.temp_size_in_bytes <= estimate.preparation_workspace_bytes
    factors = executable(matrix)
    assert sum(factor.nbytes for factor in factors) <= estimate.factorization_bytes
    # Compiled output bytes also include the result tuple's pointer table.
    assert memory.temp_size_in_bytes + memory.output_size_in_bytes <= (
        estimate.preparation_workspace_bytes + estimate.factorization_bytes
    )
    with pytest.raises(ValueError, match="workspace"):
        la.plan(
            problem,
            la.LinearSolvePolicy(
                la.DenseSVD(),
                resources=la.SolveResourcePolicy(
                    workspace_bytes=estimate.preparation_workspace_bytes - 1
                ),
            ),
        )


@pytest.mark.parametrize("shape", ((12, 3), (3, 12)), ids=("tall", "wide"))
@pytest.mark.parametrize("complex_values", (False, True), ids=("real", "complex"))
def test_rectangular_dense_svd_preserves_certified_spectrum(
    shape: tuple[int, int], complex_values: bool
) -> None:
    matrix = jnp.asarray(_matrix(*shape, complex_values)[0])
    result = la.svd.svd(
        la.svd.SVDProblem(la.DenseLinearOperator(matrix)),
        policy=la.svd.SVDSolvePolicy(la.svd.DenseSVD(algorithm="qr"), count=3),
    )
    assert bool(result.successful)
    assert int(la.svd.require_exact_svd_rank(result)) == 3
    np.testing.assert_allclose(
        result.singular_values,
        np.linalg.svd(np.asarray(matrix), compute_uv=False),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        (result.left_coordinates * result.singular_values)
        @ result.right_coordinates.conj().T,
        matrix,
        rtol=1e-12,
        atol=1e-12,
    )
