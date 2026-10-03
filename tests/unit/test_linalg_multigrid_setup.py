#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


def _positive_definite_properties() -> Any:
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )


def _poisson_matrix(size: Any) -> Any:
    diagonal = 2.0 * jnp.eye(size)
    off_diagonal = jnp.eye(size, k=1) + jnp.eye(size, k=-1)
    return diagonal - off_diagonal


def _sparse_map(matrix: Any, *, properties: Any = None) -> Any:
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


def test_exact_galerkin_builder_constructs_rap_and_reuses_transfers_on_refresh() -> None:
    fine = la.ArraySpace((4,), dtype=jnp.float64)
    coarse = la.ArraySpace((2,), dtype=jnp.float64)
    fine_matrix = _poisson_matrix(4)
    prolongation_matrix = jnp.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    operator = la.DenseLinearOperator(
        fine_matrix,
        source=fine,
        target=fine,
        properties=_positive_definite_properties(),
        operator_id="galerkin-fine-operator",
    )
    prolongation = la.DenseLinearOperator(
        prolongation_matrix,
        source=coarse,
        target=fine,
        operator_id="galerkin-prolongation",
    )
    restriction = la.adjoint(prolongation)
    builder = la.GalerkinHierarchyBuilder(
        ((restriction, prolongation),),
        (la.JacobiPreconditionerBuilder(),),
        la.DenseInversePreconditionerBuilder(),
        refresh_mode="reuse-transfers",
    )
    materialization = la.MaterializationPolicy(
        max_entries=1_000,
        max_bytes=1_000_000,
    )
    hierarchy = builder.prepare_hierarchy(
        operator,
        materialization=materialization,
    )
    action = la.MultigridPreconditioner(hierarchy)
    coarse_matrix = la.materialize(hierarchy.levels[1].operator, materialization)
    residual = jnp.asarray([1.0, -1.0, 2.0, 0.5])

    assert jnp.allclose(
        coarse_matrix,
        prolongation_matrix.T @ fine_matrix @ prolongation_matrix,
    )
    assert hierarchy.diagnostics.level_dimensions == (4, 2)
    assert hierarchy.diagnostics.coarse_construction_modes == ("bounded-dense-product",)
    assert jnp.allclose(
        jax.jit(lambda value: action.apply(value))(residual),
        action.apply(residual),
    )

    updated_matrix = fine_matrix + 0.25 * jnp.eye(4)
    updated_operator = la.DenseLinearOperator(
        updated_matrix,
        source=fine,
        target=fine,
        properties=_positive_definite_properties(),
        operator_id=operator.operator_id,
    )
    refreshed = builder.refresh(
        action,
        updated_operator,
        materialization=materialization,
    )
    refreshed_coarse = la.materialize(
        # ty: ignore[unresolved-attribute]
        refreshed.hierarchy.levels[1].operator,
        materialization,
    )

    assert any(
        "transfers-reused" in decision
        # ty: ignore[unresolved-attribute]
        for decision in refreshed.hierarchy.diagnostics.reuse_decisions
    )
    assert jnp.allclose(
        refreshed_coarse,
        prolongation_matrix.T @ updated_matrix @ prolongation_matrix,
    )


def test_explicit_hierarchy_reports_peak_level_setup_workspace() -> None:
    fine = la.ArraySpace((4,), dtype=jnp.float64)
    coarse = la.ArraySpace((2,), dtype=jnp.float64)
    fine_matrix = _poisson_matrix(4)
    prolongation_matrix = jnp.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    fine_operator = la.DenseLinearOperator(
        fine_matrix,
        source=fine,
        target=fine,
        properties=_positive_definite_properties(),
        operator_id="explicit-multigrid-fine",
    )
    prolongation = la.DenseLinearOperator(
        prolongation_matrix,
        source=coarse,
        target=fine,
    )
    restriction = la.adjoint(prolongation)
    coarse_operator = la.DenseLinearOperator(
        prolongation_matrix.T @ fine_matrix @ prolongation_matrix,
        source=coarse,
        target=coarse,
        properties=_positive_definite_properties(),
    )
    builder = la.MultigridHierarchyBuilder(
        (
            la.MultigridLevelBuilder(
                fine_operator,
                la.JacobiPreconditionerBuilder(),
                restriction=restriction,
                prolongation=prolongation,
            ),
            la.MultigridLevelBuilder(
                coarse_operator,
                la.DenseInversePreconditionerBuilder(),
            ),
        )
    )
    materialization = la.MaterializationPolicy(
        max_entries=1_000,
        max_bytes=1_000_000,
    )
    estimate = builder.cost_for(
        fine_operator,
        materialization=materialization,
    )
    hierarchy = builder.prepare_hierarchy(
        fine_operator,
        materialization=materialization,
    )

    assert estimate.accepted
    assert estimate.preparation_workspace_bytes > 0
    assert (
        hierarchy.diagnostics.setup_workspace_bytes
        == estimate.preparation_workspace_bytes
    )


def test_smoothed_aggregation_builds_decreasing_deterministic_hierarchy() -> None:
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
            strength_threshold=0.25,
            max_levels=4,
            minimum_coarse_size=2,
            prolongation_smoothing_steps=1,
        ),
        la.JacobiPreconditionerBuilder(),
        la.DenseInversePreconditionerBuilder(),
        refresh_mode="reuse-aggregates",
    )
    materialization = la.MaterializationPolicy(
        max_entries=10_000,
        max_bytes=2_000_000,
    )
    estimate = builder.cost_for(operator, materialization=materialization)
    hierarchy = builder.prepare_hierarchy(
        operator,
        materialization=materialization,
    )
    dimensions = hierarchy.diagnostics.level_dimensions
    action = la.MultigridPreconditioner(hierarchy)
    residual = jnp.linspace(-1.0, 1.0, size)
    value = action.apply(residual)

    assert estimate.accepted
    assert (
        estimate.storage_bytes + operator.matrix.nbytes
        == hierarchy.diagnostics.prepared_state_bytes
    )
    assert len(dimensions) >= 2
    assert all(fine > coarse for fine, coarse in zip(dimensions, dimensions[1:]))
    assert len(hierarchy.diagnostics.aggregate_assignments) == len(dimensions) - 1
    assert jnp.all(jnp.isfinite(value))
    assert jnp.allclose(jax.jit(lambda rhs: action.apply(rhs))(residual), value)
    changed_builder = la.SmoothedAggregationHierarchyBuilder(
        la.SmoothedAggregationPolicy(
            strength_threshold=0.5,
            max_levels=4,
            minimum_coarse_size=2,
            prolongation_smoothing_steps=1,
        ),
        la.JacobiPreconditionerBuilder(),
        la.DenseInversePreconditionerBuilder(),
        refresh_mode="reuse-aggregates",
    )
    invalidated = changed_builder.refresh(
        action,
        operator,
        materialization=materialization,
    )
    assert any(
        "reuse-invalidated-builder-dependency-change" in decision
        # ty: ignore[unresolved-attribute]
        for decision in invalidated.hierarchy.diagnostics.reuse_decisions
    )
    assert not any(
        "aggregates-reused" in decision
        # ty: ignore[unresolved-attribute]
        for decision in invalidated.hierarchy.diagnostics.reuse_decisions
    )
    assert (
        # ty: ignore[unresolved-attribute]
        invalidated.hierarchy.diagnostics.reuse_dependency_fingerprint
        == changed_builder.builder_id
    )

    matrix_free = la.FunctionLinearOperator(
        lambda value: operator.mv(value),
        source=space,
        target=space,
        properties=_positive_definite_properties(),
    )
    rejected = builder.cost_for(matrix_free, materialization=materialization)
    assert not rejected.accepted
    assert "explicit dense/canonical-CSR" in rejected.reason


def test_galerkin_route_planning_preserves_sparse_and_matrix_free_paths() -> None:
    fine_matrix = _poisson_matrix(4)
    prolongation_matrix = jnp.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    sparse_operator = _sparse_map(
        fine_matrix,
        properties=_positive_definite_properties(),
    )
    sparse_prolongation = _sparse_map(prolongation_matrix)
    sparse_restriction = _sparse_map(prolongation_matrix.T)
    sparse_builder = la.GalerkinHierarchyBuilder(
        ((sparse_restriction, sparse_prolongation),),
        (la.JacobiPreconditionerBuilder(),),
        la.JacobiPreconditionerBuilder(),
    )
    tight = la.MaterializationPolicy(max_entries=1, max_bytes=64)
    sparse_estimate = sparse_builder.cost_for(
        sparse_operator,
        materialization=tight,
    )
    sparse_hierarchy = sparse_builder.prepare_hierarchy(
        sparse_operator,
        materialization=tight,
    )

    assert sparse_estimate.accepted
    assert sparse_hierarchy.diagnostics.coarse_construction_modes == (
        "planned-sparse-assembly",
    )
    # ty: ignore[unresolved-attribute]
    assert sparse_hierarchy.levels[1].operator.sparse_storage().canonical
    # Rebuild-all refresh retains no symbolic product recipe as state.
    assert sparse_hierarchy.sparse_assemblies == (None,)
    assert jnp.allclose(
        la.materialize(
            sparse_hierarchy.levels[1].operator,
            la.MaterializationPolicy(max_entries=4, max_bytes=1024),
        ),
        prolongation_matrix.T @ fine_matrix @ prolongation_matrix,
    )
    fine = la.ArraySpace((4,), dtype=jnp.float64)
    coarse = la.ArraySpace((2,), dtype=jnp.float64)
    dense_operator = la.DenseLinearOperator(
        fine_matrix,
        source=fine,
        target=fine,
        properties=_positive_definite_properties(),
    )
    dense_prolongation = la.DenseLinearOperator(
        prolongation_matrix,
        source=coarse,
        target=fine,
    )
    dense_restriction = la.adjoint(dense_prolongation)
    matrix_free_builder = la.GalerkinHierarchyBuilder(
        ((dense_restriction, dense_prolongation),),
        (la.IdentityPreconditioner(fine),),
        la.IdentityPreconditioner(coarse),
    )
    matrix_free_estimate = matrix_free_builder.cost_for(
        dense_operator,
        materialization=tight,
    )
    matrix_free_hierarchy = matrix_free_builder.prepare_hierarchy(
        dense_operator,
        materialization=tight,
    )

    assert matrix_free_estimate.accepted
    assert matrix_free_hierarchy.diagnostics.coarse_construction_modes == (
        "matrix-free-composition",
    )
    assert not matrix_free_hierarchy.levels[1].operator.capabilities.materialize
    assert jnp.allclose(
        matrix_free_hierarchy.levels[1].operator.mv(jnp.asarray([1.0, -1.0])),
        prolongation_matrix.T
        @ fine_matrix
        @ prolongation_matrix
        @ jnp.asarray([1.0, -1.0]),
    )

    unsupported = la.GalerkinHierarchyBuilder(
        ((dense_restriction, dense_prolongation),),
        (la.IdentityPreconditioner(fine),),
        la.JacobiPreconditionerBuilder(),
    ).cost_for(dense_operator, materialization=tight)
    assert not unsupported.accepted
    assert "matrix-free" in unsupported.reason
    output_only_fits = la.MaterializationPolicy(max_entries=4, max_bytes=1_000_000)
    guarded = la.GalerkinHierarchyBuilder(
        ((dense_restriction, dense_prolongation),),
        (la.IdentityPreconditioner(fine),),
        la.JacobiPreconditionerBuilder(),
    ).cost_for(dense_operator, materialization=output_only_fits)
    assert not guarded.accepted
    assert "matrix-free" in guarded.reason


def test_symbolic_sparse_galerkin_refresh_reuses_routes_and_invalidates_patterns() -> (
    None
):
    fine_matrix = _poisson_matrix(4)
    prolongation_matrix = jnp.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    operator = _sparse_map(
        fine_matrix,
        properties=_positive_definite_properties(),
    )
    prolongation = _sparse_map(prolongation_matrix)
    restriction = _sparse_map(prolongation_matrix.T)
    builder = la.GalerkinHierarchyBuilder(
        ((restriction, prolongation),),
        (la.JacobiPreconditionerBuilder(),),
        la.JacobiPreconditionerBuilder(),
        refresh_mode="reuse-symbolic-sparse-products",
    )
    materialization = la.MaterializationPolicy(max_entries=1, max_bytes=64)
    action = builder.prepare(operator, materialization=materialization)
    # ty: ignore[unresolved-attribute]
    original_assembly = action.hierarchy.sparse_assemblies[0]
    assert original_assembly is not None

    changed_matrix = fine_matrix + 0.5 * jnp.eye(4)
    changed_operator = _sparse_map(
        changed_matrix,
        properties=_positive_definite_properties(),
    )
    refreshed = builder.refresh(
        action,
        changed_operator,
        materialization=materialization,
    )
    # ty: ignore[unresolved-attribute]
    refreshed_assembly = refreshed.hierarchy.sparse_assemblies[0]

    assert refreshed_assembly is not None
    assert refreshed_assembly.plan.plan_id == original_assembly.plan.plan_id
    assert refreshed_assembly.numeric_version == 1
    assert any(
        "sparse-route-reused;coarse-values-refreshed" in decision
        # ty: ignore[unresolved-attribute]
        for decision in refreshed.hierarchy.diagnostics.reuse_decisions
    )
    assert jnp.allclose(
        la.materialize(
            # ty: ignore[unresolved-attribute]
            refreshed.hierarchy.levels[1].operator,
            la.MaterializationPolicy(max_entries=4, max_bytes=1024),
        ),
        prolongation_matrix.T @ changed_matrix @ prolongation_matrix,
    )

    changed_pattern = changed_matrix.at[0, 1].set(0.0).at[1, 0].set(0.0)
    rebuilt = builder.refresh(
        action,
        _sparse_map(
            changed_pattern,
            properties=_positive_definite_properties(),
        ),
        materialization=materialization,
    )
    # ty: ignore[unresolved-attribute]
    rebuilt_assembly = rebuilt.hierarchy.sparse_assemblies[0]

    assert rebuilt_assembly is not None
    assert rebuilt_assembly.numeric_version == 0
    assert any(
        "reuse-invalidated-pattern-change" in decision
        # ty: ignore[unresolved-attribute]
        for decision in rebuilt.hierarchy.diagnostics.reuse_decisions
    )


def test_smoothed_aggregation_reuses_symbolic_sparse_products() -> None:
    size = 8
    matrix = _poisson_matrix(size)
    operator = _sparse_map(
        matrix,
        properties=_positive_definite_properties(),
    )
    builder = la.SmoothedAggregationHierarchyBuilder(
        la.SmoothedAggregationPolicy(
            max_levels=3,
            minimum_coarse_size=2,
        ),
        la.JacobiPreconditionerBuilder(),
        la.DenseInversePreconditionerBuilder(),
        refresh_mode="reuse-symbolic-sparse-products",
    )
    materialization = la.MaterializationPolicy(
        max_entries=1_000,
        max_bytes=1_000_000,
    )
    action = builder.prepare(operator, materialization=materialization)
    changed = _sparse_map(
        matrix + 0.25 * jnp.eye(size),
        properties=_positive_definite_properties(),
    )
    refreshed = builder.refresh(
        action,
        changed,
        materialization=materialization,
    )

    # ty: ignore[unresolved-attribute]
    assert all(value is not None for value in action.hierarchy.sparse_assemblies)
    assert all(
        value is not None and value.numeric_version == 1
        # ty: ignore[unresolved-attribute]
        for value in refreshed.hierarchy.sparse_assemblies
    )
    assert all(
        "sparse-route-reused;coarse-values-refreshed" in decision
        # ty: ignore[unresolved-attribute]
        for decision in refreshed.hierarchy.diagnostics.reuse_decisions[
            # ty: ignore[unresolved-attribute]
            : len(refreshed.hierarchy.sparse_assemblies)
        ]
    )


def test_smoothed_aggregation_costs_terminal_coarse_solver_not_fine_operator() -> None:
    size = 8
    operator = _sparse_map(
        _poisson_matrix(size),
        properties=_positive_definite_properties(),
    )
    # Vaněk aggregation of the 8-node chain gives the aggregates {0,1},
    # {2,3,4}, {5,6,7}: a 3-dimensional terminal coarse level (9 dense
    # entries) below the 64-entry fine operator.
    builder = la.SmoothedAggregationHierarchyBuilder(
        la.SmoothedAggregationPolicy(
            strength_threshold=0.25,
            max_levels=2,
            minimum_coarse_size=2,
            prolongation_smoothing_steps=1,
        ),
        la.JacobiPreconditionerBuilder(),
        la.DenseInversePreconditionerBuilder(),
    )
    fitting = la.MaterializationPolicy(max_entries=9, max_bytes=1_000_000)
    accepted = builder.cost_for(operator, materialization=fitting)
    hierarchy = builder.prepare_hierarchy(operator, materialization=fitting)
    rejected = builder.cost_for(
        operator,
        materialization=la.MaterializationPolicy(
            max_entries=8,
            max_bytes=1_000_000,
        ),
    )

    assert accepted.accepted
    assert hierarchy.diagnostics.level_dimensions == (8, 3)
    assert hierarchy.diagnostics.aggregate_assignments == ((0, 0, 1, 1, 1, 2, 2, 2),)
    assert not rejected.accepted
    assert "dense" in rejected.reason


def test_optional_pyamg_conversion_produces_jittable_phydrax_hierarchy() -> None:
    pyamg = pytest.importorskip("pyamg")
    scipy_sparse = pytest.importorskip("scipy.sparse")
    matrix = scipy_sparse.diags(
        (-jnp.ones(7), 2.0 * jnp.ones(8), -jnp.ones(7)),
        offsets=(-1, 0, 1),
        format="csr",
    )
    solver = pyamg.smoothed_aggregation_solver(matrix, max_levels=3, max_coarse=2)
    hierarchy = la.multigrid_hierarchy_from_pyamg(
        solver,
        materialization=la.MaterializationPolicy(
            max_entries=10_000,
            max_bytes=1_000_000,
        ),
    )
    action = la.MultigridPreconditioner(hierarchy)
    rhs = jnp.linspace(-1.0, 1.0, 8)

    assert len(hierarchy.levels) >= 2
    assert jnp.all(jnp.isfinite(action.apply(rhs)))
    assert jnp.allclose(
        jax.jit(lambda value: action.apply(value))(rhs),
        action.apply(rhs),
    )


def _grid_laplacian(dimension: int, side: int) -> Any:
    """Dirichlet five-point (2-D) or seven-point (3-D) graph Laplacian."""
    chain = _poisson_matrix(side)
    identity = jnp.eye(side)
    matrix = jnp.zeros((side**dimension, side**dimension))
    for axis in range(dimension):
        term = chain if axis == 0 else identity
        for index in range(1, dimension):
            term = jnp.kron(term, chain if index == axis else identity)
        matrix = matrix + term
    return matrix


def _jacobi_smoothed_aggregation(
    *, minimum_coarse_size: int, refresh_mode: la.MultigridRefreshMode = "rebuild-all"
) -> Any:
    return la.SmoothedAggregationHierarchyBuilder(
        la.SmoothedAggregationPolicy(minimum_coarse_size=minimum_coarse_size),
        la.JacobiPreconditionerBuilder(relaxation=2.0 / 3.0),
        la.DenseInversePreconditionerBuilder(),
        refresh_mode=refresh_mode,
        properties=la.PreconditionerProperties(
            linear=True,
            stationary=True,
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "linear": "construction",
                "stationary": "construction",
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
    )


_SMALL_DENSE = la.MaterializationPolicy(max_entries=1_000_000, max_bytes=100_000_000)


@pytest.mark.parametrize(
    ("dimension", "side", "complexity_limit"),
    [(2, 32, 1.4), (3, 12, 1.65)],
    ids=["2d-five-point", "3d-seven-point"],
)
def test_vanek_aggregation_bounds_operator_complexity_on_grid_laplacians(
    dimension: int, side: int, complexity_limit: float
) -> None:
    operator = _sparse_map(
        _grid_laplacian(dimension, side),
        properties=_positive_definite_properties(),
    )
    hierarchy = _jacobi_smoothed_aggregation(minimum_coarse_size=16).prepare_hierarchy(
        operator, materialization=_SMALL_DENSE
    )
    diagnostics = hierarchy.diagnostics

    # Measured operator complexity 1.33 (2-D) and 1.56 (3-D).
    assert diagnostics.operator_complexity is not None
    assert diagnostics.operator_complexity < complexity_limit
    assert diagnostics.grid_complexity < 1.25
    assert len(diagnostics.prolongation_spectral_radii) == (
        len(diagnostics.level_dimensions) - 1
    )
    # Gershgorin bounds rho(D^-1 A) by 2 on the fine grid; Ritz moduli of the
    # similar symmetric matrix approach that radius from inside.
    assert 1.8 < diagnostics.prolongation_spectral_radii[0] <= 2.0 + 1e-12


def test_smoothed_aggregation_pcg_iterations_stay_flat_under_refinement() -> None:
    def iterations(side: int) -> int:
        operator = _sparse_map(
            _grid_laplacian(2, side), properties=_positive_definite_properties()
        )
        action = _jacobi_smoothed_aggregation(minimum_coarse_size=16).prepare(
            operator, materialization=_SMALL_DENSE
        )
        result = la.solve(
            la.LinearSystem(operator),
            jnp.sin(jnp.arange(side * side, dtype=jnp.float64)),
            policy=la.LinearSolvePolicy(
                la.PCG(),
                preconditioning=la.PreconditioningPolicy(action),
                tolerance=la.TolerancePolicy(relative=1e-8, absolute=0.0, max_steps=200),
                differentiation=la.DifferentiationPolicy("none"),
            ),
        )
        assert bool(result.successful)
        return int(result.diagnostics.iterations)

    coarse_iterations = iterations(16)
    fine_iterations = iterations(32)

    # Four times the unknowns: unpreconditioned CG would roughly double.
    assert fine_iterations <= coarse_iterations + 2


@pytest.mark.parametrize(
    ("kind", "refresh_mode"),
    [
        ("smoothed-aggregation", "reuse-symbolic-sparse-products"),
        ("smoothed-aggregation", "reuse-transfers"),
        ("galerkin", "reuse-symbolic-sparse-products"),
        ("galerkin-gauss-seidel", "reuse-symbolic-sparse-products"),
    ],
    ids=["sa-symbolic", "sa-transfers", "galerkin-symbolic", "galerkin-gauss-seidel"],
)
def test_numeric_hierarchy_refresh_runs_inside_jit(
    kind: str, refresh_mode: la.MultigridRefreshMode
) -> None:
    matrix = _grid_laplacian(2, 12)
    operator = _sparse_map(matrix, properties=_positive_definite_properties())
    rows, columns = jnp.nonzero(matrix)
    shift = jnp.where(rows == columns, 0.3, 0.0)
    aggregation = _jacobi_smoothed_aggregation(
        minimum_coarse_size=8, refresh_mode=refresh_mode
    )
    builder: Any = aggregation
    if kind != "smoothed-aggregation":
        reference = aggregation.prepare_hierarchy(operator, materialization=_SMALL_DENSE)
        transfers = tuple(
            (level.restriction, level.prolongation) for level in reference.levels[:-1]
        )
        smoother = (
            la.GaussSeidelPreconditionerBuilder(direction="symmetric")
            if kind == "galerkin-gauss-seidel"
            else la.JacobiPreconditionerBuilder(relaxation=2.0 / 3.0)
        )
        builder = la.GalerkinHierarchyBuilder(
            transfers,
            (smoother,) * len(transfers),
            la.DenseInversePreconditionerBuilder(),
            refresh_mode=refresh_mode,
        )
    action = builder.prepare(operator, materialization=_SMALL_DENSE)
    residual = jnp.cos(jnp.arange(matrix.shape[0], dtype=jnp.float64))

    def refreshed(scale: Any, offset: Any, value: Any) -> Any:
        changed = eqx.tree_at(
            lambda current: current.coefficients,
            operator,
            operator.coefficients * scale + offset,
        )
        return builder.refresh(action, changed, materialization=_SMALL_DENSE).apply(value)

    eager = refreshed(1.5, shift, residual)
    traced = jax.jit(refreshed)(1.5, shift, residual)
    scaled = jax.jit(refreshed)(2.0, jnp.zeros_like(shift), residual)

    assert jnp.allclose(traced, eager, rtol=0.0, atol=1e-13)
    assert not jnp.allclose(eager, action.apply(residual))
    # Frozen transfers make the cycle homogeneous of degree -1 in the operator.
    assert jnp.allclose(2.0 * scaled, action.apply(residual), rtol=0.0, atol=1e-12)
