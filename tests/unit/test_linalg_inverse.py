#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


def _positive_definite_properties():
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "asserted",
        },
    )


def test_inverse_materializes_batched_lu_with_matrix_level_evidence():
    matrices = jnp.asarray(
        (
            ((3.0, 1.0), (0.5, 2.0)),
            ((2.0, -0.25), (0.75, 4.0)),
        )
    )

    result = la.inverse(matrices)

    identity = jnp.broadcast_to(jnp.eye(2), matrices.shape)
    assert result.operation == "inverse"
    assert result.value.shape == matrices.shape
    assert result.status.shape == (2,)
    assert jnp.all(result.successful)
    assert jnp.allclose(matrices @ result.value, identity, rtol=1e-10, atol=1e-10)
    assert result.diagnostics.rank.shape == (2,)
    assert jnp.all(result.diagnostics.rank == 2)
    assert jnp.all(jnp.isfinite(result.diagnostics.condition_estimate))


def test_inverse_cholesky_is_jittable_and_has_mathematical_jvp():
    matrix = jnp.asarray(((4.0, 1.0), (1.0, 3.0)))
    tangent = jnp.asarray(((0.2, -0.1), (-0.1, 0.3)))
    policy = la.FactorizationPolicy("cholesky")
    properties = _positive_definite_properties()

    evaluate = jax.jit(
        lambda value: la.inverse(value, policy, properties=properties).value
    )
    inverse, derivative = jax.jvp(evaluate, (matrix,), (tangent,))

    assert jnp.allclose(matrix @ inverse, jnp.eye(2), rtol=1e-10, atol=1e-10)
    assert jnp.allclose(
        derivative,
        -inverse @ tangent @ inverse,
        rtol=1e-9,
        atol=1e-9,
    )


def test_inverse_reports_singular_without_pseudoinverse_fallback():
    matrix = jnp.asarray(((1.0, 2.0), (2.0, 4.0)))

    result = la.inverse(matrix)

    assert not bool(result.successful)
    assert int(result.status) == int(la.LinearSolveStatus.SINGULAR)


def test_pseudoinverse_handles_tall_wide_and_rank_deficient_batches():
    matrices = (
        jnp.asarray(((1.0, 0.0), (0.0, 2.0), (1.0, 1.0))),
        jnp.asarray(((1.0, 0.0, 1.0), (0.0, 2.0, 1.0))),
        jnp.asarray(((1.0, 0.0), (0.0, 0.0), (0.0, 0.0))),
    )

    for matrix in matrices:
        result = la.pseudoinverse(matrix)
        pseudoinverse = result.value
        assert bool(result.successful)
        assert pseudoinverse.shape == (matrix.shape[1], matrix.shape[0])
        assert jnp.allclose(
            matrix @ pseudoinverse @ matrix,
            matrix,
            rtol=1e-9,
            atol=1e-9,
        )
        assert jnp.allclose(
            pseudoinverse @ matrix @ pseudoinverse,
            pseudoinverse,
            rtol=1e-9,
            atol=1e-9,
        )
        assert jnp.allclose(
            matrix @ pseudoinverse,
            jnp.conj(jnp.swapaxes(matrix @ pseudoinverse, -1, -2)),
            rtol=1e-9,
            atol=1e-9,
        )
        assert jnp.allclose(
            pseudoinverse @ matrix,
            jnp.conj(jnp.swapaxes(pseudoinverse @ matrix, -1, -2)),
            rtol=1e-9,
            atol=1e-9,
        )


def test_pseudoinverse_combines_absolute_and_relative_rank_cutoffs():
    matrix = jnp.diag(jnp.asarray((10.0, 1.0e-3, 1.0e-7)))
    policy = la.FactorizationPolicy(
        "svd",
        rank=la.RankPolicy(relative_cutoff=1.0e-5, absolute_cutoff=5.0e-4),
    )

    result = la.pseudoinverse(matrix, policy)

    assert bool(result.successful)
    assert int(result.diagnostics.rank) == 2
    assert jnp.allclose(result.diagnostics.rank_cutoff, 6.0e-4)
    assert jnp.allclose(jnp.diag(result.value), jnp.asarray((0.1, 1.0e3, 0.0)))


def test_pseudoinverse_full_rank_requirement_changes_status_only():
    matrix = jnp.asarray(((1.0, 0.0), (0.0, 0.0)))
    permissive = la.pseudoinverse(matrix)
    strict = la.pseudoinverse(
        matrix,
        la.FactorizationPolicy(
            "svd",
            rank=la.RankPolicy(require_full_rank=True),
        ),
    )

    assert bool(permissive.successful)
    assert int(strict.status) == int(la.LinearSolveStatus.RANK_DEFICIENT)
    assert jnp.allclose(strict.value, permissive.value)


def test_operator_pseudoinverse_respects_both_diagonal_pairings():
    matrix = jnp.asarray(((1.0, 2.0, 0.0), (0.0, 1.0, 1.0)))
    source_weights = jnp.asarray((2.0, 3.0, 5.0))
    target_weights = jnp.asarray((7.0, 11.0))
    source = la.ArraySpace(
        (3,),
        dtype=matrix.dtype,
        pairing=la.DiagonalPairing(source_weights),
    )
    target = la.ArraySpace(
        (2,),
        dtype=matrix.dtype,
        pairing=la.DiagonalPairing(target_weights),
    )
    operator = la.DenseLinearOperator(matrix, source=source, target=target)

    result = la.pseudoinverse(operator)

    source_inverse_root = jax.lax.rsqrt(source_weights)
    target_root = jnp.sqrt(target_weights)
    reduced = target_root[:, None] * matrix * source_inverse_root
    expected = (
        source_inverse_root[:, None] * jnp.linalg.pinv(reduced) * target_root[None, :]
    )
    right_hand_side = jnp.asarray((0.5, -1.0))
    solved = la.factorize(
        operator,
        la.FactorizationPolicy("svd"),
    ).solve(right_hand_side)
    assert bool(result.successful)
    assert jnp.allclose(result.value, expected, rtol=1e-9, atol=1e-9)
    assert bool(solved.successful)
    assert jnp.allclose(solved.value, expected @ right_hand_side, rtol=1e-9, atol=1e-9)


def test_hermitian_pseudoinverse_and_fixed_rank_jvp_are_finite():
    matrix = jnp.asarray(((2.0, 1.0j), (-1.0j, 0.5)), dtype=jnp.complex128)
    tangent = jnp.asarray(((0.2, 0.1j), (-0.1j, -0.3)), dtype=jnp.complex128)
    properties = la.OperatorProperties(
        self_adjoint=True,
        evidence={"self_adjoint": "construction"},
    )

    value, derivative = jax.jvp(
        lambda argument: la.pseudoinverse(argument, properties=properties).value,
        (matrix,),
        (tangent,),
    )

    assert jnp.all(jnp.isfinite(value))
    assert jnp.all(jnp.isfinite(derivative))
    assert jnp.allclose(value, jnp.conj(value.T), rtol=1e-9, atol=1e-9)


def test_small_inverse_scales_extreme_complex_matrices():
    matrix = jnp.asarray(
        (
            (2.0e100 + 1.0e99j, 1.0e100),
            (1.0e100, 3.0e100 - 2.0e99j),
        ),
        dtype=jnp.complex128,
    )

    result = la.inverse_small_linear(la.SmallLinearSolvePlan(2), matrix)

    assert bool(result.successful)
    assert jnp.all(jnp.isfinite(result.value))
    assert jnp.allclose(matrix @ result.value, jnp.eye(2), rtol=1e-10, atol=1e-10)


def test_factorization_refresh_and_batched_capabilities_remain_truthful():
    operator = la.DenseLinearOperator(jnp.stack((jnp.eye(2), 2.0 * jnp.eye(2))))
    prepared = la.factorize(operator)

    first = prepared.materialize_inverse()
    refreshed = la.refresh_factorization(
        prepared,
        la.DenseLinearOperator(jnp.stack((3.0 * jnp.eye(2), 4.0 * jnp.eye(2)))),
    )
    second = refreshed.materialize_inverse()

    assert jnp.all(first.successful)
    assert jnp.all(second.successful)
    assert jnp.allclose(second.value[0], jnp.eye(2) / 3.0)
    assert refreshed.factorization_id != prepared.factorization_id
    assert int(refreshed.prepared_solve.numeric_version) == 1
    assert not refreshed.capabilities.nullspaces


def test_inverse_and_pseudoinverse_reject_incompatible_methods():
    matrix = jnp.eye(2)

    with pytest.raises(ValueError, match="inverse requires"):
        la.inverse(matrix, la.FactorizationPolicy("svd"))
    with pytest.raises(ValueError, match="pseudoinverse requires"):
        la.pseudoinverse(matrix, la.FactorizationPolicy("lu"))
