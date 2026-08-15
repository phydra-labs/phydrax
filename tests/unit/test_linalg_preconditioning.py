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
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )


def _coordinate_term(global_space, local_space, row):
    restriction = la.DenseLinearOperator(
        jnp.asarray(row, dtype=jnp.float64)[None, :],
        source=global_space,
        target=local_space,
    )
    prolongation = la.DenseLinearOperator(
        jnp.asarray(row, dtype=jnp.float64)[:, None],
        source=local_space,
        target=global_space,
    )
    return la.SubspaceCorrectionTerm(
        restriction,
        prolongation,
        la.DenseInversePreconditionerBuilder(),
    )


def test_additive_and_multiplicative_subspace_corrections_match_dense_references():
    space = la.ArraySpace((2,), dtype=jnp.float64)
    local = la.ArraySpace((1,), dtype=jnp.float64)
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
    operator = la.DenseLinearOperator(
        matrix,
        source=space,
        target=space,
        properties=_positive_definite_properties(),
    )
    terms = (
        _coordinate_term(space, local, [1.0, 0.0]),
        _coordinate_term(space, local, [0.0, 1.0]),
    )
    materialization = la.MaterializationPolicy(max_entries=16, max_bytes=1024)
    additive = la.AdditiveSubspaceCorrectionBuilder(terms).prepare(
        operator,
        materialization=materialization,
    )
    forward = la.MultiplicativeSubspaceCorrectionBuilder(
        terms,
        sweep="forward",
    ).prepare(operator, materialization=materialization)
    backward = la.MultiplicativeSubspaceCorrectionBuilder(
        terms,
        sweep="backward",
    ).prepare(operator, materialization=materialization)
    residual = jnp.asarray([2.0, -1.0])

    additive_expected = residual / jnp.diag(matrix)
    forward_expected = jnp.asarray(
        [
            residual[0] / matrix[0, 0],
            (residual[1] - matrix[1, 0] * residual[0] / matrix[0, 0]) / matrix[1, 1],
        ]
    )
    backward_expected = jnp.asarray(
        [
            (residual[0] - matrix[0, 1] * residual[1] / matrix[1, 1]) / matrix[0, 0],
            residual[1] / matrix[1, 1],
        ]
    )

    assert jnp.allclose(additive.apply(residual), additive_expected)
    assert jnp.allclose(forward.apply(residual), forward_expected)
    assert jnp.allclose(backward.apply(residual), backward_expected)
    assert jnp.allclose(
        jax.jit(lambda value: additive.apply(value))(residual),
        additive_expected,
    )
    assert jnp.allclose(
        jax.jit(lambda value: forward.apply(value))(residual),
        forward_expected,
    )
    cost = additive.cost_for(operator)
    assert cost.storage_bytes == 112
    assert cost.setup_matvec_count == 0
    accepted_plan = la.plan(
        la.LinearSystem(operator),
        la.LinearSolvePolicy(
            la.GMRES(restart=2),
            preconditioning=la.PreconditioningPolicy(additive),
            differentiation=la.DifferentiationPolicy("none"),
            resources=la.SolveResourcePolicy(preconditioner_bytes=112),
        ),
    )
    assert accepted_plan.preconditioner_plan is not None
    assert accepted_plan.preconditioner_plan.cost.storage_bytes == 112
    with pytest.raises(ValueError, match="requires 112 preconditioner state bytes"):
        la.plan(
            la.LinearSystem(operator),
            la.LinearSolvePolicy(
                la.GMRES(restart=2),
                preconditioning=la.PreconditioningPolicy(additive),
                differentiation=la.DifferentiationPolicy("none"),
                resources=la.SolveResourcePolicy(preconditioner_bytes=111),
            ),
        )
    with pytest.raises(ValueError, match="at least one"):
        la.AdditiveSubspaceCorrectionBuilder(())


def test_chebyshev_preconditioner_is_fixed_degree_matrix_free_and_jittable():
    diagonal = jnp.asarray([2.0, 3.0, 5.0])
    operator = la.DiagonalLinearOperator(
        diagonal,
        properties=_positive_definite_properties(),
    )
    builder = la.ChebyshevPreconditionerBuilder(
        8,
        interval=(2.0, 5.0),
    )
    action = builder.prepare(operator, materialization=la.MaterializationPolicy())
    residual = jnp.asarray([1.0, -2.0, 0.5])
    value = action.apply(residual)
    compiled = jax.jit(lambda value: action.apply(value))(residual)

    assert jnp.allclose(compiled, value)
    assert jnp.linalg.norm(operator.mv(value) - residual) < 1e-4
    assert action.degree == 8
    assert builder.cost_for(operator).setup_matvec_count == 0
    with pytest.raises(ValueError, match="degree"):
        la.ChebyshevPreconditionerBuilder(0, interval=(1.0, 2.0))
    with pytest.raises(ValueError, match="interval"):
        la.ChebyshevPreconditionerBuilder(2, interval=(2.0, 1.0))


def _two_by_two_block_operator():
    first = la.ArraySpace((1,), dtype=jnp.float64)
    second = la.ArraySpace((1,), dtype=jnp.float64)
    block_space = la.BlockSpace((first, second))
    blocks = (
        (
            la.DenseLinearOperator(jnp.asarray([[4.0]]), source=first, target=first),
            la.DenseLinearOperator(jnp.asarray([[1.0]]), source=second, target=first),
        ),
        (
            la.DenseLinearOperator(jnp.asarray([[2.0]]), source=first, target=second),
            la.DenseLinearOperator(jnp.asarray([[3.0]]), source=second, target=second),
        ),
    )
    return la.BlockLinearOperator(blocks, source=block_space, target=block_space)


def test_block_factorization_forms_match_block_algebra_and_require_typed_schur_action():
    operator = _two_by_two_block_operator()
    rhs = (jnp.asarray([2.0]), jnp.asarray([-1.0]))
    matrix = la.materialize(operator, la.MaterializationPolicy(max_entries=4))
    pivot = la.DenseInversePreconditionerBuilder()
    schur = la.DenseInversePreconditionerBuilder()
    actions = {
        form: la.BlockFactorizationPreconditionerBuilder(
            pivot,
            schur,
            form=form,
        ).prepare(operator, materialization=la.MaterializationPolicy(max_entries=16))
        for form in ("diagonal", "lower", "upper", "ldu")
    }
    pivot_value = 1.0 / matrix[0, 0]
    schur_value = matrix[1, 1] - matrix[1, 0] * pivot_value * matrix[0, 1]
    diagonal_expected = jnp.asarray([pivot_value * rhs[0][0], rhs[1][0] / schur_value])
    lower_expected = jnp.asarray(
        [
            pivot_value * rhs[0][0],
            (rhs[1][0] - matrix[1, 0] * pivot_value * rhs[0][0]) / schur_value,
        ]
    )
    upper_expected = jnp.asarray(
        [
            pivot_value * (rhs[0][0] - matrix[0, 1] * rhs[1][0] / schur_value),
            rhs[1][0] / schur_value,
        ]
    )
    exact = jnp.linalg.solve(matrix, jnp.asarray([rhs[0][0], rhs[1][0]]))

    assert jnp.allclose(
        operator.source.flatten(actions["diagonal"].apply(rhs)), diagonal_expected
    )
    assert jnp.allclose(
        operator.source.flatten(actions["lower"].apply(rhs)), lower_expected
    )
    assert jnp.allclose(
        operator.source.flatten(actions["upper"].apply(rhs)), upper_expected
    )
    assert jnp.allclose(operator.source.flatten(actions["ldu"].apply(rhs)), exact)
    assert jnp.allclose(
        operator.source.flatten(jax.jit(lambda value: actions["ldu"].apply(value))(rhs)),
        exact,
    )
    with pytest.raises(TypeError, match="AbstractPreconditioner"):
        la.SchurComplementLinearOperator(
            operator.blocks[1][1],
            operator.blocks[1][0],
            operator.blocks[0][1],
            lambda value: value,
        )
