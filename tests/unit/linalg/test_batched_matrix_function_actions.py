from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import expm, expm_frechet, solve, sqrtm

from phydrax import linalg as la


@pytest.mark.parametrize("self_adjoint", [True, False])
def test_batched_exponential_actions_match_scipy(self_adjoint: bool) -> None:
    matrix: NDArray[np.float64] = np.diag(np.linspace(1.0, 2.0, 8)).astype(np.float64)
    matrix[0, 1] = 0.2
    if self_adjoint:
        matrix[1, 0] = 0.2
    matrices: NDArray[np.float64] = np.stack([matrix, matrix * 1.2])
    properties = (
        la.OperatorProperties(self_adjoint=True, evidence={"self_adjoint": "asserted"})
        if self_adjoint
        else la.OperatorProperties()
    )
    operator = la.DenseLinearOperator(jnp.asarray(matrices), properties=properties)
    rhs = np.arange(16, dtype=np.float64).reshape(8, 2) / 16
    result = la.matrix_function_action(
        operator,
        jnp.asarray(rhs),
        0.2,
        kind="exp",
        rhs_layout=la.RHSLayout((2,)),
        policy=la.MatrixFunctionPolicy(max_dimension=8, error_tolerance=1e-8),
    )
    np.testing.assert_allclose(
        result.value,
        np.stack([expm(0.2 * matrix) @ rhs, expm(0.24 * matrix) @ rhs]),
        rtol=1e-9,
        atol=1e-9,
    )
    assert np.all(result.successful)
    assert np.all(result.diagnostics.finite)
    assert result.provenance.method != "batched-dense-exact"


def test_batched_complex_resolvent_solves_rhs_block() -> None:
    matrices = np.stack([np.diag(np.arange(1.0, 9.0)), np.diag(np.arange(2.0, 10.0))])
    rhs = np.arange(16, dtype=np.float64).reshape(8, 2)
    operator = la.DenseLinearOperator(jnp.asarray(matrices))
    shift = 0.3 + 0.7j
    result = la.matrix_function_action(
        operator,
        jnp.asarray(rhs),
        0.4,
        kind="resolvent",
        shift=shift,
        rhs_layout=la.RHSLayout((2,)),
    )
    reference = np.stack(
        [solve(shift * np.eye(8) - 0.4 * item, rhs) for item in matrices]
    )
    np.testing.assert_allclose(result.value, reference, rtol=1e-12, atol=1e-12)
    assert np.all(result.successful)


def test_dense_matrix_function_refuses_workspace_before_execution() -> None:
    operator = la.DenseLinearOperator(jnp.ones((2, 8, 8), dtype=jnp.float64))
    with pytest.raises(ValueError, match="workspace"):
        la.matrix_function_action(
            operator,
            jnp.ones(8, dtype=jnp.float64),
            kind="exp",
            policy=la.MatrixFunctionPolicy("dense", max_workspace_bytes=1),
        )


def test_singular_batched_resolvent_does_not_claim_success() -> None:
    operator = la.DenseLinearOperator(jnp.ones((2, 1, 1), dtype=jnp.float64))
    result = la.matrix_function_action(
        operator, jnp.ones(1, dtype=jnp.float64), kind="resolvent", shift=1.0
    )
    assert not np.any(result.successful)
    assert not np.any(result.diagnostics.finite)


def test_eigenspace_action_jvp_at_repeated_eigenvalues() -> None:
    matrix = np.eye(4, dtype=np.float64)
    tangent = np.asarray(
        [
            [0.2, 0.1, 0.0, 0.0],
            [0.1, -0.3, 0.2, 0.0],
            [0.0, 0.2, 0.4, 0.1],
            [0.0, 0.0, 0.1, -0.2],
        ],
        dtype=np.float64,
    )
    rhs = jnp.arange(8, dtype=jnp.float64).reshape((4, 2))
    properties = la.OperatorProperties(
        self_adjoint=True, evidence={"self_adjoint": "asserted"}
    )

    def action(value: jax.Array) -> jax.Array:
        operator = la.DenseLinearOperator(
            value[None], properties=properties, operator_id="repeated-spectrum"
        )
        result = la.matrix_function_action(
            operator, rhs, 0.4, kind="exp", rhs_layout=la.RHSLayout((2,))
        ).value
        if not isinstance(result, jax.Array):
            raise TypeError("Array-space action must return an array.")
        return result[0]

    value, derivative = jax.jvp(action, (jnp.asarray(matrix),), (jnp.asarray(tangent),))
    reference, reference_derivative = expm_frechet(0.4 * matrix, 0.4 * tangent)
    np.testing.assert_allclose(value, reference @ np.asarray(rhs), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        derivative, reference_derivative @ np.asarray(rhs), rtol=1e-12, atol=1e-12
    )


def test_nonselfadjoint_sqrt_action_preserves_domain_evidence() -> None:
    matrix: NDArray[np.float64] = np.asarray(
        [[2.0, 0.3, 0.0], [0.0, 3.0, 0.2], [0.0, 0.0, 4.0]], dtype=np.float64
    )
    rhs = np.asarray([1.0, -0.3, 0.7], dtype=np.float64)
    operator = la.DenseLinearOperator(jnp.asarray(matrix)[None])
    result = la.matrix_function_action(
        operator,
        jnp.asarray(rhs),
        kind="sqrt",
        spectral_bounds=(1.0, 5.0),
        policy=la.MatrixFunctionPolicy("arnoldi", max_dimension=3, error_tolerance=1e-8),
    )
    np.testing.assert_allclose(result.value[0], sqrtm(matrix) @ rhs, rtol=1e-8, atol=1e-8)
    assert np.all(result.successful)
    assert result.provenance.method == "batched-arnoldi-action"


def test_weighted_selfadjoint_batch_uses_pairing_aware_action() -> None:
    matrix: NDArray[np.float64] = np.asarray([[2.0, 0.6], [0.3, 3.0]], dtype=np.float64)
    space = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray([1.0, 2.0], dtype=jnp.float64)),
    )
    operator = la.DenseLinearOperator(
        jnp.asarray(matrix)[None],
        source=space,
        target=space,
        properties=la.OperatorProperties(
            self_adjoint=True, evidence={"self_adjoint": "asserted"}
        ),
    )
    rhs = jnp.asarray([1.0, -0.2], dtype=jnp.float64)
    result = la.matrix_function_action(
        operator,
        rhs,
        0.3,
        kind="exp",
        policy=la.MatrixFunctionPolicy(max_dimension=2, error_tolerance=1e-8),
    )
    np.testing.assert_allclose(
        result.value[0], expm(0.3 * matrix) @ np.asarray(rhs), rtol=1e-10, atol=1e-10
    )
    assert np.all(result.successful)
    assert result.provenance.method == "batched-lanczos-action"
