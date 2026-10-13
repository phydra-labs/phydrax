from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.conditions import ArrayCodomain, FieldSpec, ProductFieldSpec
from phydrax.kernels import (
    kernel_functional_diagonal,
    KernelFunctional,
    KernelFunctionalTerm,
    ProductFieldKernelMetric,
    SquaredExponentialKernel,
)


def _fixture() -> tuple[ProductFieldKernelMetric, KernelFunctional, KernelFunctional]:
    spec = ProductFieldSpec(
        (FieldSpec("u", ArrayCodomain.from_shape((), dtype="complex128")),)
    )
    metric = ProductFieldKernelMetric.independent(
        spec, {"u": SquaredExponentialKernel(length_scale=0.7)}
    )
    rng = np.random.default_rng(120)

    def functional(rows: int, points: int, name: str) -> KernelFunctional:
        coefficients = rng.normal(size=(rows, points, 2, 1)) + 1j * rng.normal(
            size=(rows, points, 2, 1)
        )
        return KernelFunctional(
            (
                KernelFunctionalTerm(
                    "u",
                    jnp.linspace(-0.8, 0.9, points, dtype=jnp.float64)[:, None],
                    ((0,), (1,)),
                    jnp.asarray(coefficients),
                ),
            ),
            functional_id=name,
        )

    return metric, functional(5, 5, "left"), functional(7, 4, "right")


def _reference(left: KernelFunctional, right: KernelFunctional) -> np.ndarray:
    # Independent analytic derivatives of exp(-(x-y)^2/(2 ell^2)).
    a, b = left.terms[0], right.terms[0]
    delta = np.asarray(a.points)[:, 0, None] - np.asarray(b.points)[None, :, 0]
    covariance = np.exp(-(delta**2) / (2.0 * 0.7**2))
    values = np.empty((a.points.shape[0], 2, b.points.shape[0], 2), dtype=np.float64)
    values[:, 0, :, 0] = covariance
    values[:, 1, :, 0] = -delta * covariance / 0.7**2
    values[:, 0, :, 1] = delta * covariance / 0.7**2
    values[:, 1, :, 1] = (1.0 / 0.7**2 - delta**2 / 0.7**4) * covariance
    return np.einsum(
        "ipt,ptqs,jqs->ij",
        np.conj(np.asarray(a.coefficients)[..., 0]),
        values,
        np.asarray(b.coefficients)[..., 0],
    )


@pytest.mark.parametrize("block", [1, 3, 8], ids=["singleton", "partial", "oversized"])
def test_blocked_complex_derivative_gram_action(block: int) -> None:
    metric, left, right = _fixture()
    rhs = jnp.arange(14, dtype=jnp.float64).reshape(7, 2) * (0.3 + 0.2j)
    result = metric.functional_gram_action(
        left, rhs, right, row_block_size=block, column_block_size=block
    )
    expected = _reference(left, right) @ np.asarray(rhs)
    np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        result, metric.functional_gram(left, right).matrix @ rhs, rtol=2e-12, atol=2e-12
    )


@pytest.mark.parametrize("block", [1, 3, 8], ids=["singleton", "partial", "oversized"])
def test_direct_complex_derivative_diagonal(block: int) -> None:
    metric, functional, _ = _fixture()
    result = kernel_functional_diagonal(
        metric, functional, row_block_size=block, column_block_size=block
    )
    expected = np.real(np.diag(_reference(functional, functional)))
    np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        result,
        jnp.real(jnp.diag(metric.functional_gram(functional).matrix)),
        rtol=2e-12,
        atol=2e-12,
    )


def test_blocked_gram_vector_and_jvp() -> None:
    metric, left, right = _fixture()
    vector = jnp.linspace(0.2, 1.4, right.row_count, dtype=jnp.float64)

    def action(value: Array) -> Array:
        return metric.functional_gram_action(
            left, value, right, row_block_size=3, column_block_size=2
        )

    result, tangent = jax.jvp(action, (vector,), (jnp.ones_like(vector),))
    reference = _reference(left, right)
    np.testing.assert_allclose(
        result, reference @ np.asarray(vector), rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        tangent, reference @ np.ones(right.row_count), rtol=2e-12, atol=2e-12
    )


@pytest.mark.parametrize("block", [0, -1], ids=["zero", "negative"])
def test_gram_workspace_requires_positive_blocks(block: int) -> None:
    metric, left, right = _fixture()
    with pytest.raises(ValueError, match="positive"):
        metric.functional_gram_action(
            left, jnp.ones(right.row_count), right, row_block_size=block
        )
    with pytest.raises(ValueError, match="positive"):
        kernel_functional_diagonal(metric, left, column_block_size=block)


def test_blocked_gram_combines_all_functional_terms() -> None:
    metric, first, other = _fixture()
    term = other.terms[0]
    second = KernelFunctional(
        (
            KernelFunctionalTerm(
                term.field_name,
                term.points,
                term.derivative_orders,
                term.coefficients[: first.row_count],
            ),
        ),
        functional_id="second",
    )
    combined = KernelFunctional((*first.terms, *second.terms), functional_id="combined")
    reference = (
        _reference(first, first)
        + _reference(first, second)
        + _reference(second, first)
        + _reference(second, second)
    )
    vector = jnp.linspace(0.2, 1.4, first.row_count, dtype=jnp.float64)
    np.testing.assert_allclose(
        metric.functional_gram_action(
            combined, vector, row_block_size=3, column_block_size=2
        ),
        reference @ np.asarray(vector),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        kernel_functional_diagonal(
            metric, combined, row_block_size=3, column_block_size=2
        ),
        np.real(np.diag(reference)),
        rtol=2e-12,
        atol=2e-12,
    )
