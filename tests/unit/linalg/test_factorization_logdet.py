# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Prepared determinant values and mathematical derivatives against NumPy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.linalg as la


@pytest.mark.parametrize("complex_values", (False, True))
@pytest.mark.parametrize("kind", ("lu", "cholesky"))
def test_batched_prepared_logdet_and_derivatives(
    kind: la.FactorizationKind, complex_values: bool
) -> None:
    rng = np.random.default_rng(706)
    seed = rng.normal(size=(2, 4, 4))
    direction = rng.normal(size=(2, 4, 4))
    if complex_values:
        seed = seed + 1j * rng.normal(size=seed.shape)
        direction = direction + 1j * rng.normal(size=direction.shape)
    matrix = seed @ seed.conj().swapaxes(-1, -2) + np.eye(4)
    direction = direction + direction.conj().swapaxes(-1, -2)
    properties = la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "asserted",
            "positive_definite": "asserted",
            "positive_semidefinite": "asserted",
        },
    )

    def objective(parameter: Array) -> Array:
        operator = la.DenseLinearOperator(
            jnp.asarray(matrix) + parameter * jnp.asarray(direction),
            properties=properties,
        )
        factorization = la.factorize(operator, la.FactorizationPolicy(kind))
        return factorization.log_abs_determinant()

    factorization = la.factorize(
        la.DenseLinearOperator(jnp.asarray(matrix), properties=properties),
        la.FactorizationPolicy(kind),
    )
    sign, logdet = np.linalg.slogdet(matrix)
    np.testing.assert_allclose(factorization.determinant_sign(), sign, atol=1e-12)
    np.testing.assert_allclose(factorization.log_abs_determinant(), logdet, rtol=1e-12)
    step = 1e-6
    reference = (
        np.linalg.slogdet(matrix + step * direction)[1]
        - np.linalg.slogdet(matrix - step * direction)[1]
    ) / (2 * step)
    np.testing.assert_allclose(
        jax.jvp(objective, (jnp.asarray(0.0),), (jnp.asarray(1.0),))[1],
        reference,
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        jax.jacrev(objective)(jnp.asarray(0.0)), reference, rtol=1e-7
    )


def test_non_positive_definite_cholesky_does_not_report_a_valid_logdet() -> None:
    operator = la.DenseLinearOperator(
        jnp.asarray([[1.0, 0.0], [0.0, -2.0]]),
        properties=la.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "asserted",
                "positive_definite": "asserted",
                "positive_semidefinite": "asserted",
            },
        ),
    )
    factorization = la.factorize(operator, la.FactorizationPolicy("cholesky"))
    result = factorization.solve(jnp.ones(2, dtype=jnp.float64))
    assert not bool(result.successful)
    assert int(result.status) != int(la.LinearSolveStatus.SUCCESS)
    assert np.isnan(float(factorization.log_abs_determinant()))
    assert np.isnan(float(factorization.determinant_sign()))
