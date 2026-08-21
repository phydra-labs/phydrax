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


def _poisson_matrix(size):
    diagonal = 2.0 * jnp.eye(size)
    off_diagonal = jnp.eye(size, k=1) + jnp.eye(size, k=-1)
    return diagonal - off_diagonal


def _sparse_map(matrix, *, properties=None):
    rows, columns = jnp.nonzero(matrix)
    relation = phx.sparse.EdgeRelation(
        columns,
        rows,
        source_size=matrix.shape[1],
        target_size=matrix.shape[0],
    )
    return phx.sparse.SparseLinearMap(
        relation,
        matrix[rows, columns],
        properties=properties,
    )


def test_three_level_cycle_policies_execute_distinct_recursive_schedules():
    fine = la.ArraySpace((3,), dtype=jnp.float64)
    middle = la.ArraySpace((2,), dtype=jnp.float64)
    coarse = la.ArraySpace((1,), dtype=jnp.float64)
    fine_operator = la.DenseLinearOperator(
        jnp.asarray([[3.0, -0.2, 0.1], [-0.2, 2.5, -0.3], [0.1, -0.3, 2.0]]),
        source=fine,
        target=fine,
        properties=_positive_definite_properties(),
    )
    middle_operator = la.DenseLinearOperator(
        jnp.asarray([[2.0, 0.25], [0.25, 1.5]]),
        source=middle,
        target=middle,
        properties=_positive_definite_properties(),
    )
    coarse_operator = la.DenseLinearOperator(
        jnp.asarray([[1.25]]),
        source=coarse,
        target=coarse,
        properties=_positive_definite_properties(),
    )
    fine_restriction = la.DenseLinearOperator(
        jnp.asarray([[1.0, 0.1, 0.0], [0.0, 0.7, 0.3]]),
        source=fine,
        target=middle,
    )
    fine_prolongation = la.DenseLinearOperator(
        jnp.asarray([[0.8, 0.0], [0.2, 0.6], [0.0, 0.9]]),
        source=middle,
        target=fine,
    )
    middle_restriction = la.DenseLinearOperator(
        jnp.asarray([[0.4, 0.8]]),
        source=middle,
        target=coarse,
    )
    middle_prolongation = la.DenseLinearOperator(
        jnp.asarray([[0.6], [0.5]]),
        source=coarse,
        target=middle,
    )
    builder = la.MultigridHierarchyBuilder(
        (
            la.MultigridLevelBuilder(
                fine_operator,
                la.JacobiPreconditionerBuilder(relaxation=0.21),
                restriction=fine_restriction,
                prolongation=fine_prolongation,
            ),
            la.MultigridLevelBuilder(
                middle_operator,
                la.JacobiPreconditionerBuilder(relaxation=0.37),
                restriction=middle_restriction,
                prolongation=middle_prolongation,
            ),
            la.MultigridLevelBuilder(
                coarse_operator,
                la.JacobiPreconditionerBuilder(relaxation=0.43),
            ),
        )
    )
    hierarchy = builder.prepare_hierarchy(
        fine_operator,
        materialization=la.MaterializationPolicy(
            max_entries=1_000,
            max_bytes=1_000_000,
        ),
    )
    residual = jnp.asarray([1.0, -0.75, 0.4])
    values = tuple(
        la.MultigridPreconditioner(
            hierarchy,
            cycle_policy=la.MultigridCyclePolicy(kind),
        ).apply(residual)
        for kind in ("v", "w", "f", "full")
    )

    assert all(jnp.all(jnp.isfinite(value)) for value in values)
    assert all(
        not jnp.allclose(values[left], values[right], rtol=1e-8, atol=1e-10)
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )


@pytest.mark.parametrize("direction", ("forward", "backward", "symmetric"))
def test_gauss_seidel_is_jittable_and_reuses_triangular_analysis(direction):
    space = la.ArraySpace((3,), dtype=jnp.float64)
    matrix = jnp.asarray([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]])
    operator = la.DenseLinearOperator(
        matrix,
        source=space,
        target=space,
        operator_id=f"gauss-seidel-{direction}",
    )
    builder = la.GaussSeidelPreconditionerBuilder(direction=direction)
    materialization = la.MaterializationPolicy(
        max_entries=1_000,
        max_bytes=1_000_000,
    )
    action = builder.prepare(operator, materialization=materialization)
    residual = jnp.asarray([1.0, -2.0, 0.5])
    if direction == "forward":
        expected = jnp.linalg.solve(jnp.tril(matrix), residual)
    elif direction == "backward":
        expected = jnp.linalg.solve(jnp.triu(matrix), residual)
    else:
        first = jnp.linalg.solve(jnp.tril(matrix), residual)
        expected = first + jnp.linalg.solve(jnp.triu(matrix), residual - matrix @ first)

    assert jnp.allclose(jax.jit(lambda value: action.apply(value))(residual), expected)

    updated_operator = la.DenseLinearOperator(
        matrix + jnp.diag(jnp.asarray([0.5, 0.25, 0.75])),
        source=space,
        target=space,
        operator_id=operator.operator_id,
    )
    refreshed = builder.refresh(
        action,
        updated_operator,
        materialization=materialization,
    )
    original_factors = tuple(
        factor
        for factor in (action.forward_factor, action.backward_factor)
        if factor is not None
    )
    refreshed_factors = tuple(
        factor
        for factor in (refreshed.forward_factor, refreshed.backward_factor)
        if factor is not None
    )

    assert len(original_factors) == len(refreshed_factors)
    assert all(
        new.analysis is old.analysis
        for old, new in zip(original_factors, refreshed_factors, strict=True)
    )
    assert all(
        not jnp.allclose(old.values, new.values)
        for old, new in zip(original_factors, refreshed_factors, strict=True)
    )

    changed_pattern = (
        matrix.at[0, 1].set(0.0) if direction == "backward" else matrix.at[1, 0].set(0.0)
    )
    with pytest.raises(ValueError, match="unchanged triangular pattern"):
        builder.refresh(
            action,
            la.DenseLinearOperator(
                changed_pattern,
                source=space,
                target=space,
                operator_id=operator.operator_id,
            ),
            materialization=materialization,
        )


def test_sparse_coarse_factor_and_smoother_refresh_share_symbolic_state():
    size = 6
    matrix = _poisson_matrix(size)
    operator = _sparse_map(
        matrix,
        properties=_positive_definite_properties(),
    )
    prolongation_matrix = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    prolongation = _sparse_map(prolongation_matrix)
    restriction = _sparse_map(prolongation_matrix.T)
    coarse_builder = la.SparseFactorizationPreconditionerBuilder(
        la.SparseFactorizationPolicy("lu")
    )
    builder = la.GalerkinHierarchyBuilder(
        ((restriction, prolongation),),
        (la.GaussSeidelPreconditionerBuilder(direction="forward"),),
        coarse_builder,
        refresh_mode="reuse-symbolic-sparse-products",
    )
    materialization = la.MaterializationPolicy(
        max_entries=10_000,
        max_bytes=2_000_000,
    )
    action = builder.prepare(operator, materialization=materialization)
    original_transfer = action.hierarchy.levels[0].prolongation
    original_smoother = action.hierarchy.levels[0].smoother
    original_coarse = action.hierarchy.levels[-1].smoother
    assert isinstance(original_smoother, la.GaussSeidelPreconditioner)
    assert isinstance(original_coarse, la.SparseFactorizationPreconditioner)
    assert original_smoother.forward_factor is not None

    updated = _sparse_map(
        matrix + 0.25 * jnp.eye(size),
        properties=_positive_definite_properties(),
    )
    refreshed = builder.refresh(
        action,
        updated,
        materialization=materialization,
    )
    refreshed_smoother = refreshed.hierarchy.levels[0].smoother
    refreshed_coarse = refreshed.hierarchy.levels[-1].smoother

    assert isinstance(refreshed_smoother, la.GaussSeidelPreconditioner)
    assert isinstance(refreshed_coarse, la.SparseFactorizationPreconditioner)
    assert refreshed_smoother.forward_factor is not None
    assert refreshed.hierarchy.levels[0].prolongation is original_transfer
    assert (
        refreshed_smoother.forward_factor.analysis
        is original_smoother.forward_factor.analysis
    )
    assert refreshed_coarse.factorization.plan is original_coarse.factorization.plan
    assert not jnp.allclose(
        refreshed_coarse.factorization.factor_values,
        original_coarse.factorization.factor_values,
    )
    assert all(
        f"level-{index}:builder-action-refreshed"
        in refreshed.hierarchy.diagnostics.reuse_decisions
        for index in range(2)
    )

    changed_pattern = matrix.at[0, 1].set(0.0).at[1, 0].set(0.0)
    rebuilt = builder.refresh(
        action,
        _sparse_map(
            changed_pattern,
            properties=_positive_definite_properties(),
        ),
        materialization=materialization,
    )
    rebuilt_coarse = rebuilt.hierarchy.levels[-1].smoother
    assert isinstance(rebuilt_coarse, la.SparseFactorizationPreconditioner)
    assert rebuilt_coarse.factorization.plan is not original_coarse.factorization.plan
    assert any(
        "reuse-invalidated-pattern-change" in decision
        for decision in rebuilt.hierarchy.diagnostics.reuse_decisions
    )

    changed_builder = la.GalerkinHierarchyBuilder(
        ((restriction, prolongation),),
        (la.JacobiPreconditionerBuilder(),),
        coarse_builder,
        refresh_mode="reuse-symbolic-sparse-products",
    )
    reprepared = changed_builder.refresh(
        action,
        operator,
        materialization=materialization,
    )
    assert isinstance(
        reprepared.hierarchy.levels[0].smoother,
        la.DiagonalPreconditioner,
    )
    assert all(
        f"level-{index}:builder-action-prepared"
        in reprepared.hierarchy.diagnostics.reuse_decisions
        for index in range(2)
    )


def test_smoothed_aggregation_propagates_explicit_near_nullspace_candidates():
    size = 12
    space = la.ArraySpace((size,), dtype=jnp.float64)
    operator = _sparse_map(
        _poisson_matrix(size),
        properties=_positive_definite_properties(),
    )
    candidate = la.LinearSubspace(
        space,
        jnp.ones((size, 1), dtype=jnp.float64),
        orthonormal=False,
        subspace_id="constant-near-nullspace",
    )
    builder = la.SmoothedAggregationHierarchyBuilder(
        la.SmoothedAggregationPolicy(
            max_levels=4,
            minimum_coarse_size=2,
            prolongation_smoothing_steps=0,
        ),
        la.JacobiPreconditionerBuilder(),
        la.DenseInversePreconditionerBuilder(),
        near_nullspaces=(candidate,),
    )
    hierarchy = builder.prepare_hierarchy(
        operator,
        materialization=la.MaterializationPolicy(
            max_entries=10_000,
            max_bytes=2_000_000,
        ),
    )
    ranks = hierarchy.diagnostics.aggregate_candidate_ranks
    dimensions = hierarchy.diagnostics.level_dimensions

    assert len(ranks) == len(dimensions) - 1
    assert all(
        level_ranks and all(rank == 1 for rank in level_ranks) for level_ranks in ranks
    )
    assert all(
        coarse_dimension == sum(level_ranks)
        for coarse_dimension, level_ranks in zip(dimensions[1:], ranks, strict=True)
    )


def test_smoothed_aggregation_rejects_fine_level_storage_before_setup():
    size = 8
    space = la.ArraySpace((size,), dtype=jnp.float64)
    operator = la.DenseLinearOperator(
        _poisson_matrix(size),
        source=space,
        target=space,
        properties=_positive_definite_properties(),
    )
    builder = la.SmoothedAggregationHierarchyBuilder(
        la.SmoothedAggregationPolicy(
            minimum_coarse_size=2,
            maximum_level_storage_bytes=64,
        ),
        la.JacobiPreconditionerBuilder(),
        la.DenseInversePreconditionerBuilder(),
    )
    materialization = la.MaterializationPolicy(
        max_entries=10_000,
        max_bytes=2_000_000,
    )
    estimate = builder.cost_for(operator, materialization=materialization)

    assert not estimate.accepted
    assert "Level 0 storage" in estimate.reason
    with pytest.raises(la.LinearCapabilityError, match="Level 0 storage"):
        builder.prepare_hierarchy(operator, materialization=materialization)


@pytest.mark.parametrize(
    ("triangle", "matrix", "right_hand_side", "expected"),
    (
        (
            "upper",
            jnp.asarray([[0.0, 2.0], [0.0, 0.0]]),
            jnp.asarray([1.0, 3.0]),
            jnp.asarray([-5.0, 3.0]),
        ),
        (
            "lower",
            jnp.asarray([[0.0, 0.0], [3.0, 0.0]]),
            jnp.asarray([1.0, 2.0]),
            jnp.asarray([1.0, -1.0]),
        ),
    ),
)
def test_implicit_unit_triangular_solve_retains_first_stored_offdiagonal(
    triangle,
    matrix,
    right_hand_side,
    expected,
):
    operator = _sparse_map(matrix)
    storage = operator.sparse_storage()
    analysis = la.analyze_sparse_triangular(
        storage,
        triangle=triangle,
        unit_diagonal=True,
    )
    result = la.solve_sparse_triangular(
        analysis,
        storage.values,
        right_hand_side,
    )

    assert result.status == int(la.SparseTriangularStatus.SUCCESS)
    assert jnp.allclose(result.value, expected)


def test_cholesky_factor_action_preserves_builder_property_evidence():
    properties = la.OperatorProperties(
        self_adjoint=True,
        evidence={"self_adjoint": "construction"},
    )
    operator = _sparse_map(
        jnp.diag(jnp.asarray([-1.0, 2.0])),
        properties=properties,
    )
    builder = la.SparseFactorizationPreconditionerBuilder(
        la.SparseFactorizationPolicy(
            "cholesky",
            allow_pivot_replacement=True,
            replacement_value=1.0,
        )
    )
    expected = builder.properties_for(operator)
    action = builder.prepare(
        operator,
        materialization=la.MaterializationPolicy(
            max_entries=100,
            max_bytes=10_000,
        ),
    )

    assert expected.certifies("self_adjoint")
    assert not expected.certifies("positive_definite")
    assert not action.properties.certifies("positive_definite")
    assert (
        action.properties.linear,
        action.properties.stationary,
        action.properties.self_adjoint,
        action.properties.positive_definite,
        action.properties.evidence,
    ) == (
        expected.linear,
        expected.stationary,
        expected.self_adjoint,
        expected.positive_definite,
        expected.evidence,
    )


def test_exact_sparse_cholesky_tolerates_complex_hermitian_roundoff():
    matrix = jnp.asarray(
        [
            [5.0 + 2.0e-16j, 1.0 + 2.0j, -0.5 + 0.25j],
            [1.0 - 2.0j, 6.0 - 1.0e-16j, 0.75 + 0.5j],
            [-0.5 - 0.25j, 0.75 - 0.5j, 4.0 + 3.0e-16j],
        ],
        dtype=jnp.complex128,
    )
    operator = _sparse_map(
        matrix,
        properties=_positive_definite_properties(),
    )
    builder = la.SparseFactorizationPreconditionerBuilder(
        la.SparseFactorizationPolicy("cholesky")
    )
    action = builder.prepare(
        operator,
        materialization=la.MaterializationPolicy(
            max_entries=1_000,
            max_bytes=1_000_000,
        ),
    )
    right_hand_side = jnp.asarray(
        [1.0 + 0.5j, -2.0 + 0.25j, 0.75 - 1.0j],
        dtype=jnp.complex128,
    )
    value = action.apply(right_hand_side)

    assert action.factorization.status == int(la.SparseFactorizationStatus.SUCCESS)
    assert jnp.linalg.norm(matrix @ value - right_hand_side) < 1.0e-12
