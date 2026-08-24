#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


pytestmark = pytest.mark.skipif(
    not bool(jax.config.read("jax_enable_x64")),
    reason="Mixed float64/float32 certification requires JAX x64.",
)


def _mixed_policy(*, refinement_steps: int = 3):
    return la.LinearSolvePolicy(
        la.DenseLU(),
        tolerance=la.TolerancePolicy(relative=0.0, absolute=0.0),
        precision=la.MixedPrecisionPolicy(
            operator_dtype=jnp.float64,
            factorization_dtype=jnp.float32,
            residual_dtype=jnp.float64,
            accumulation_dtype=jnp.float64,
            maximum_refinement_steps=refinement_steps,
        ),
    )


def test_dense_mixed_precision_refinement_improves_certified_solution():
    matrix = jnp.asarray(
        [
            [1.234567890123, 0.345678901234],
            [0.456789012345, 1.876543210987],
        ],
        dtype=jnp.float64,
    )
    expected = jnp.asarray([0.1234567890123, -0.9876543210987])
    rhs = matrix @ expected
    problem = la.LinearSystem(la.DenseLinearOperator(matrix))

    unrefined = la.solve(problem, rhs, policy=_mixed_policy(refinement_steps=0))
    refined_policy = _mixed_policy(refinement_steps=3)
    refined = la.solve(problem, rhs, policy=refined_policy)

    unrefined_error = jnp.linalg.norm(unrefined.value - expected)
    refined_error = jnp.linalg.norm(refined.value - expected)
    assert refined_error < unrefined_error
    assert refined.diagnostics.residual_norm < unrefined.diagnostics.residual_norm
    assert refined.diagnostics.refinement_steps >= 1
    assert refined.status == int(la.LinearSolveStatus.SUCCESS)

    requested = refined.provenance.requested_precision
    effective = refined.provenance.effective_precision
    assert effective is not None
    assert requested is refined_policy.precision
    assert requested.factorization_dtype == "float32"
    assert effective.operator_dtype == "float64"
    assert effective.factorization_dtype == "float32"
    assert effective.residual_dtype == "float64"
    assert effective.accumulation_dtype == "float64"
    assert effective.krylov_dtype is None
    assert effective.preconditioner_dtype is None
    assert effective.maximum_refinement_steps == 3
    assert effective.condition_limit is not None
    assert effective.condition_limit <= 0.1 / jnp.finfo(jnp.float32).eps

    selected = la.plan(problem, refined_policy).candidates[-1]
    assert selected.factorization_bytes == matrix.size * jnp.dtype(jnp.float32).itemsize


def test_mixed_precision_rejects_unsafe_condition_before_factorization():
    matrix = jnp.diag(jnp.asarray([1.0, 1.0e-8], dtype=jnp.float64))
    problem = la.LinearSystem(la.DenseLinearOperator(matrix))

    with pytest.raises(
        ValueError,
        match="capability rejected before low-precision factorization",
    ):
        la.prepare(problem, _mixed_policy())


def test_mixed_precision_rejects_unsupported_factor_and_accumulation_dtypes():
    matrix = jnp.asarray([[2.0, 0.25], [0.5, 1.5]], dtype=jnp.float64)
    problem = la.LinearSystem(la.DenseLinearOperator(matrix))
    unsupported_factor = la.LinearSolvePolicy(
        la.DenseLU(),
        precision=la.MixedPrecisionPolicy(
            operator_dtype=jnp.float64,
            factorization_dtype=jnp.float16,
            residual_dtype=jnp.float64,
        ),
    )
    unsupported_accumulation = la.LinearSolvePolicy(
        la.DenseLU(),
        precision=la.MixedPrecisionPolicy(
            operator_dtype=jnp.float64,
            factorization_dtype=jnp.float32,
            residual_dtype=jnp.float64,
            accumulation_dtype=jnp.float32,
        ),
    )

    with pytest.raises(ValueError, match="jax-dense LU does not support"):
        la.plan(problem, unsupported_factor)
    with pytest.raises(ValueError, match="accumulation_dtype"):
        la.plan(problem, unsupported_accumulation)


def test_default_dense_precision_behavior_and_evidence_remain_unchanged():
    matrix = jnp.asarray([[2.25, -0.5], [0.75, 1.5]], dtype=jnp.float64)
    rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    result = la.solve(la.LinearSystem(la.DenseLinearOperator(matrix)), rhs)

    assert result.value.dtype == rhs.dtype
    assert jnp.allclose(result.value, jnp.linalg.solve(matrix, rhs))
    assert result.provenance.requested_precision is None
    assert result.provenance.effective_precision is None
    assert result.diagnostics.refinement_steps == 0


def test_dense_mixed_precision_is_compatible_with_eager_and_jit_execution():
    matrix = jnp.asarray(
        [[1.234567890123, 0.345678901234], [0.456789012345, 1.876543210987]],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([0.25, -1.5], dtype=jnp.float64)
    policy = _mixed_policy()
    eager = la.solve(
        la.LinearSystem(la.DenseLinearOperator(matrix)),
        rhs,
        policy=policy,
    )

    compiled = jax.jit(
        lambda coefficients, value: (
            la.solve(
                la.LinearSystem(la.DenseLinearOperator(coefficients)),
                value,
                policy=policy,
            ).value
        )
    )
    prepared = la.prepare(
        la.LinearSystem(la.DenseLinearOperator(matrix)),
        policy,
    )

    def solve_prepared(value):
        result = la.solve(prepared, value)
        return result.value, result.diagnostics.refinement_steps

    compiled_prepared = jax.jit(solve_prepared)
    prepared_value, refinement_steps = compiled_prepared(rhs)

    assert jnp.allclose(compiled(matrix, rhs), eager.value)
    assert jnp.allclose(prepared_value, eager.value)
    assert refinement_steps == eager.diagnostics.refinement_steps
