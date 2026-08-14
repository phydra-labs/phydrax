#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
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


def test_block_arnoldi_deflates_and_preserves_the_block_relation():
    matrix = jnp.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
    )
    initial = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    decomposition = la.krylov.block_arnoldi(
        lambda block: matrix @ block,
        initial,
        max_blocks=2,
    )
    jitted_projected = jax.jit(
        lambda block: (
            la.krylov.block_arnoldi(
                lambda value: matrix @ value,
                block,
                max_blocks=2,
            ).projected
        )
    )(initial)

    assert jnp.array_equal(decomposition.block_ranks, jnp.asarray([2, 1, 1]))
    assert int(decomposition.effective_dimension) == 3
    assert int(decomposition.matvec_count) == 3
    assert jnp.allclose(
        matrix @ decomposition.basis[:, :4],
        decomposition.basis @ decomposition.projected,
    )
    assert float(decomposition.orthogonality_error) < 1e-12
    assert jnp.allclose(jitted_projected, decomposition.projected)


def test_recycling_subspace_coarse_correction_is_jittable_and_metric_orthogonal():
    space = la.ArraySpace((3,), dtype=jnp.float64)
    matrix = jnp.diag(jnp.asarray([1.0, 10.0, 100.0]))
    operator = la.DenseLinearOperator(matrix, source=space, target=space)
    recycling = la.prepare_recycling_subspace(operator, jnp.eye(3)[:, :2])
    right_hand_side = jnp.asarray([2.0, 30.0, 5.0])

    correction = jax.jit(lambda artifact, residual: artifact.correction(residual))(
        recycling, right_hand_side
    )
    residual = right_hand_side - operator.mv(correction)

    assert recycling.operator_id == operator.operator_id
    assert recycling.dimension == 2
    assert jnp.allclose(correction, jnp.asarray([2.0, 3.0, 0.0]))
    assert jnp.allclose(residual, recycling.project_residual(right_hand_side))
    assert jnp.allclose(
        jax.vmap(
            lambda left: jax.vmap(
                lambda right: space.inner(
                    space.unflatten(left),
                    space.unflatten(right),
                ),
                in_axes=1,
            )(recycling.image_basis),
            in_axes=1,
        )(recycling.image_basis),
        jnp.eye(2),
    )
    with pytest.raises(ValueError, match="linearly independent"):
        la.prepare_recycling_subspace(operator, jnp.ones((3, 2)))


def test_saddle_point_system_and_schur_complement_match_dense_algebra():
    primal_space = la.ArraySpace((2,), dtype=jnp.float64)
    dual_space = la.ArraySpace((1,), dtype=jnp.float64)
    primal_matrix = jnp.asarray([[2.0, 0.0], [0.0, 3.0]])
    constraint_matrix = jnp.asarray([[1.0, 1.0]])
    primal = la.DenseLinearOperator(
        primal_matrix,
        source=primal_space,
        target=primal_space,
        properties=_positive_definite_properties(),
    )
    constraint = la.DenseLinearOperator(
        constraint_matrix,
        source=primal_space,
        target=dual_space,
    )
    system = la.saddle_point_system(primal, constraint)
    right_hand_side = (jnp.asarray([1.0, 2.0]), jnp.asarray([0.5]))
    prepared = la.prepare(
        system,
        la.LinearSolvePolicy(
            la.MINRES(),
            tolerance=la.TolerancePolicy(
                relative=1e-10,
                absolute=1e-12,
                max_steps=20,
            ),
        ),
    )
    result = jax.jit(lambda value: la.solve(prepared, value))(right_hand_side)
    dense = la.materialize(system.operator, la.MaterializationPolicy())
    expected = system.operator.source.unflatten(
        jnp.linalg.solve(dense, system.operator.target.flatten(right_hand_side))
    )
    schur = la.saddle_point_schur_complement(
        primal,
        constraint,
        lambda value: jnp.linalg.solve(primal_matrix, value),
    )

    assert system.operator.properties.self_adjoint
    assert system.operator.properties.evidence_for("self_adjoint") == "transformed"
    assert bool(result.successful)
    assert jax.tree.all(jax.tree.map(jnp.allclose, result.value, expected))
    assert jnp.allclose(
        schur.mv(jnp.ones((1,))),
        -constraint_matrix @ jnp.linalg.solve(primal_matrix, constraint_matrix.T),
    )


def test_tensor_and_low_rank_operators_retain_structure_and_exact_solves():
    left_matrix = jnp.asarray([[2.0, 1.0], [0.0, 3.0]])
    right_matrix = jnp.asarray([[4.0, 0.0], [1.0, 5.0]])
    left = la.DenseLinearOperator(left_matrix)
    right = la.DenseLinearOperator(right_matrix)
    operator = la.KroneckerLinearOperator((left, right))
    dense = jnp.kron(left_matrix, right_matrix)
    expected = jnp.arange(4.0)
    right_hand_side = operator.target.unflatten(dense @ expected)
    result = la.solve(la.LinearSystem(operator), right_hand_side)

    factor = jnp.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    diagonal_low_rank = la.DiagonalPlusLowRankLinearOperator(
        jnp.asarray([3.0, 4.0, 5.0]),
        factor,
        factor,
    )
    low_rank_dense = jnp.diag(jnp.asarray([3.0, 4.0, 5.0])) + factor @ factor.T
    low_rank_expected = jnp.asarray([0.5, -1.0, 2.0])
    low_rank_result = la.solve(
        la.LinearSystem(diagonal_low_rank),
        low_rank_dense @ low_rank_expected,
    )

    assert result.provenance.backend == "jax-structured"
    assert jnp.allclose(operator.source.flatten(result.value), expected)
    assert jnp.allclose(
        la.materialize(operator, la.MaterializationPolicy()),
        dense,
    )
    assert low_rank_result.provenance.backend == "jax-structured"
    assert jnp.allclose(low_rank_result.value, low_rank_expected)

    certified_low_rank = la.DiagonalPlusLowRankLinearOperator(
        jnp.asarray([3.0, 4.0, 5.0]),
        factor,
        factor,
        nonsingular_diagonal=True,
    )
    no_dense_fallback = la.LinearSolvePolicy(
        materialization=la.MaterializationPolicy(max_entries=1, max_bytes=8),
    )
    certified_plan = la.plan(
        la.LinearSystem(certified_low_rank),
        no_dense_fallback,
    )
    certified_result = la.solve(
        la.LinearSystem(certified_low_rank),
        low_rank_dense @ low_rank_expected,
        policy=no_dense_fallback,
    )
    assert certified_plan.backend == "jax-structured"
    assert certified_plan.candidates[-1].additional_matrix_bytes == 0
    assert jnp.allclose(certified_result.value, low_rank_expected)


def test_factorization_capabilities_nullspaces_and_numeric_refresh_are_truthful():
    matrix = jnp.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    operator = la.DenseLinearOperator(matrix, operator_id="rank-one-design")
    factorization = la.factorize(
        operator,
        la.FactorizationPolicy(
            "svd",
            rank=la.RankPolicy(relative_cutoff=1e-12),
        ),
    )
    right_hand_side = jnp.asarray([1.0, 2.0])
    result = factorization.solve(right_hand_side)
    expected = jnp.linalg.lstsq(matrix, right_hand_side, rcond=1e-12)[0]
    changed = la.DenseLinearOperator(2.0 * matrix, operator_id="rank-one-design")
    refreshed = la.refresh_factorization(factorization, changed)

    assert factorization.capabilities.singular_values
    assert factorization.capabilities.pseudodeterminant
    assert factorization.capabilities.nullspaces
    assert int(factorization.rank()) == 1
    assert int(factorization.right_nullspace().dimension) == 2
    assert int(factorization.left_nullspace().dimension) == 1
    assert jnp.allclose(result.value, expected)
    assert refreshed.prepared_solve.numeric_version == 1
    assert refreshed.factorization_id != factorization.factorization_id
    assert jnp.allclose(
        refreshed.solve(right_hand_side).value,
        jnp.linalg.lstsq(2.0 * matrix, right_hand_side, rcond=1e-12)[0],
    )


def test_nullspace_projection_returns_compatible_gauge_fixed_solution():
    space = la.ArraySpace((2,), dtype=jnp.float64)
    matrix = jnp.asarray([[1.0, -1.0], [-1.0, 1.0]])
    nullspace = la.LinearSubspace(
        space,
        jnp.asarray([[1.0], [1.0]]) / jnp.sqrt(2.0),
        orthonormal=True,
    )
    problem = la.LinearSystem(
        la.FunctionLinearOperator(
            lambda value: matrix @ value,
            source=space,
            target=space,
            operator_id="singular-laplacian",
        ),
        nullspace_policy=la.NullspacePolicy(
            right=nullspace,
            left=nullspace,
            compatibility="project",
            gauge="project",
        ),
    )
    result = la.solve(
        problem,
        jnp.asarray([1.0, 0.0]),
        policy=la.LinearSolvePolicy(
            la.FGMRES(restart=2),
            tolerance=la.TolerancePolicy(
                relative=1e-10,
                absolute=1e-12,
                max_steps=4,
            ),
        ),
    )

    assert bool(result.successful)
    assert jnp.allclose(result.value, jnp.asarray([0.25, -0.25]))
    assert float(result.diagnostics.compatibility_residual) > 0.0
    assert float(result.diagnostics.gauge_residual) < 1e-12


def test_spectral_matrix_functions_and_stochastic_estimators_are_replayable():
    matrix = jnp.diag(jnp.asarray([1.5, 2.0, 4.0]))
    operator = la.DenseLinearOperator(
        matrix,
        properties=_positive_definite_properties(),
        operator_id="diagonal-spectrum",
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    spectral = la.SpectralMatrixRepresentation(
        operator,
        eigenvalues,
        eigenvectors.T,
        eigenvectors,
    )
    vector = jnp.asarray([1.0, -0.5, 2.0])
    logarithm = la.matrix_function_action(
        operator,
        vector,
        kind="log",
        policy=la.MatrixFunctionPolicy("spectral"),
        spectral=spectral,
    )
    first_trace = la.stochastic_trace(
        operator,
        key=jr.key(4),
        num_probes=4,
        max_dimension=3,
    )
    replay_trace = la.stochastic_trace(
        operator,
        key=jr.key(4),
        num_probes=4,
        max_dimension=3,
    )
    log_determinant = la.stochastic_log_determinant(
        operator,
        key=jr.key(8),
        num_probes=4,
        max_dimension=3,
    )

    assert logarithm.provenance == "explicit spectral representation"
    assert jnp.allclose(logarithm.value, jnp.log(eigenvalues) * vector)
    assert jnp.array_equal(first_trace.samples, replay_trace.samples)
    assert jnp.allclose(first_trace.estimate, jnp.trace(matrix), rtol=1e-10, atol=1e-10)
    assert jnp.allclose(
        log_determinant.estimate,
        jnp.linalg.slogdet(matrix)[1],
        rtol=1e-10,
        atol=1e-10,
    )


def test_host_sparse_direct_and_incomplete_factorization_are_explicit():
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1, 0, 1, 2], dtype=jnp.int32),
        jnp.asarray([0, 0, 1, 1, 2], dtype=jnp.int32),
        source_size=3,
        target_size=3,
    )
    operator = phx.sparse.SparseLinearMap(
        relation,
        jnp.asarray([4.0, 1.0, 1.0, 3.0, 2.0]),
    )
    matrix = operator.as_dense()
    right_hand_side = jnp.asarray([1.0, 2.0, -1.0])
    result = la.solve(
        la.LinearSystem(operator),
        right_hand_side,
        policy=la.LinearSolvePolicy(
            la.HostSparseLU(),
            differentiation=la.DifferentiationPolicy("none"),
        ),
    )
    lower = jnp.asarray([[1.0, 0.0], [0.25, 1.0]])
    upper = jnp.asarray([[4.0, 1.0], [0.0, 2.75]])
    preconditioner = la.IncompleteFactorizationPreconditioner(
        lower,
        upper,
        unit_lower=True,
    )
    residual = jnp.asarray([2.0, -1.0])

    assert result.provenance.backend == "host-sparse"
    assert bool(result.successful)
    assert jnp.allclose(result.value, jnp.linalg.solve(matrix, right_hand_side))
    assert jnp.allclose(
        preconditioner.apply(residual),
        jnp.linalg.solve(lower @ upper, residual),
    )


def test_structured_exact_edge_cases_and_matrix_free_tensor_actions():
    tridiagonal = la.TridiagonalLinearOperator(
        jnp.asarray([1.0]),
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([1.0]),
    )
    right_hand_side = jnp.asarray([1.0, 2.0])
    result = la.solve(la.LinearSystem(tridiagonal), right_hand_side)
    compiled = jax.jit(lambda rhs: la.solve(la.LinearSystem(tridiagonal), rhs).value)(
        right_hand_side
    )
    assert bool(result.successful)
    assert jnp.allclose(result.value, jnp.asarray([1.0, 1.0]))
    assert jnp.allclose(compiled, result.value)

    diagonal_low_rank = la.DiagonalPlusLowRankLinearOperator(
        jnp.asarray([0.0, 2.0]),
        jnp.asarray([[1.0], [0.0]]),
        jnp.asarray([[1.0], [0.0]]),
    )
    expected = jnp.asarray([3.0, -1.0])
    low_rank_result = la.solve(
        la.LinearSystem(diagonal_low_rank),
        jnp.asarray([3.0, -2.0]),
    )
    assert bool(low_rank_result.successful)
    assert jnp.allclose(low_rank_result.value, expected)

    singular_low_rank = la.DiagonalPlusLowRankLinearOperator(
        jnp.asarray([0.0, 2.0]),
        jnp.zeros((2, 1)),
        jnp.zeros((2, 1)),
    )
    singular = la.solve(
        la.LinearSystem(singular_low_rank),
        jnp.asarray([1.0, 2.0]),
    )
    assert singular.status == int(la.LinearSolveStatus.SINGULAR)
    assert not bool(singular.successful)

    space = la.ArraySpace((2,), dtype=jnp.float64)
    left_matrix = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    right_matrix = jnp.asarray([[2.0, -1.0], [0.5, 3.0]])
    left = la.FunctionLinearOperator(
        lambda vector: left_matrix @ vector,
        source=space,
        target=space,
        transpose_action=lambda vector: left_matrix.T @ vector,
    )
    right = la.FunctionLinearOperator(
        lambda vector: right_matrix @ vector,
        source=space,
        target=space,
        transpose_action=lambda vector: right_matrix.T @ vector,
    )
    left = eqx.tree_at(
        lambda operator: operator.capabilities,
        left,
        la.OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
        ),
    )
    right = eqx.tree_at(
        lambda operator: operator.capabilities,
        right,
        la.OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
        ),
    )
    kronecker = la.KroneckerLinearOperator((left, right))
    vector = jnp.arange(4.0).reshape((2, 2))
    expected_image = (
        jnp.kron(left_matrix, right_matrix) @ vector.reshape((-1,))
    ).reshape((2, 2))
    assert not kronecker.capabilities.materialize
    assert jnp.allclose(kronecker.mv(vector), expected_image)


def test_krylov_validation_scale_invariance_and_stagnation_status():
    matrix = jnp.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
    )
    initial = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    reference = la.krylov.block_arnoldi(
        lambda block: matrix @ block,
        initial,
        max_blocks=2,
    )
    for scale in (1e-12, 1e12):
        scaled = la.krylov.block_arnoldi(
            lambda block: matrix @ block,
            scale * initial,
            max_blocks=2,
        )
        assert jnp.array_equal(scaled.block_ranks, reference.block_ranks)
        assert scaled.effective_dimension == reference.effective_dimension

    for invalid in (-1.0, jnp.inf, jnp.nan):
        with pytest.raises(ValueError, match="finite and non-negative"):
            la.krylov.arnoldi(
                lambda vector: vector,
                jnp.ones((2,)),
                max_dimension=2,
                breakdown_tolerance=invalid,
            )
    nonfinite = la.krylov.arnoldi(
        lambda vector: vector,
        jnp.asarray([jnp.nan, 1.0]),
        max_dimension=2,
    )
    assert nonfinite.breakdown_status == int(
        la.krylov.KrylovBreakdownStatus.NONFINITE_ACTION
    )

    rotation = jnp.asarray([[0.0, 1.0], [-1.0, 0.0]])
    space = la.ArraySpace((2,), dtype=jnp.float64)
    operator = la.FunctionLinearOperator(
        lambda vector: rotation @ vector,
        source=space,
        target=space,
        operator_id="stagnating-rotation",
    )
    stagnated = la.solve(
        la.LinearSystem(operator),
        jnp.asarray([1.0, 0.0]),
        policy=la.LinearSolvePolicy(
            la.FGMRES(restart=1, stagnation_iterations=2),
            tolerance=la.TolerancePolicy(
                relative=1e-12,
                absolute=1e-14,
                max_steps=10,
            ),
        ),
    )
    assert stagnated.status == int(la.LinearSolveStatus.STAGNATION)
    assert stagnated.diagnostics.iterations == 2
    assert not bool(stagnated.successful)


def test_matrix_function_and_stochastic_evidence_rejects_false_success():
    jordan = jnp.asarray([[2.0, 1.0], [0.0, 2.0]])
    operator = la.DenseLinearOperator(jordan)
    vector = jnp.asarray([1.0, -0.5])
    zero_scale = la.matrix_exponential_action(
        operator,
        vector,
        jnp.asarray(0.0),
        policy=la.MatrixFunctionPolicy("arnoldi", max_dimension=1),
    )
    logarithm = la.matrix_function_action(
        operator,
        vector,
        kind="log",
        policy=la.MatrixFunctionPolicy(
            "arnoldi",
            max_dimension=2,
            error_tolerance=1e-10,
        ),
        spectral_bounds=(1.0, 3.0),
    )
    expected_logarithm = (
        jnp.log(2.0) * jnp.eye(2) + jnp.asarray([[0.0, 0.5], [0.0, 0.0]])
    ) @ vector
    assert bool(zero_scale.converged)
    assert jnp.array_equal(zero_scale.value, vector)
    assert zero_scale.error_estimate == 0.0
    assert zero_scale.residual_estimate == 0.0
    assert bool(logarithm.converged)
    assert jnp.allclose(logarithm.value, expected_logarithm)

    properties = _positive_definite_properties()
    diagonal = la.DenseLinearOperator(
        jnp.diag(jnp.asarray([1.0, 3.0])),
        properties=properties,
        operator_id="matrix-function-state",
    )
    chebyshev = la.matrix_exponential_action(
        diagonal,
        vector,
        policy=la.MatrixFunctionPolicy("chebyshev", max_dimension=8),
        spectral_bounds=(1.0, 3.0),
    )
    assert not bool(chebyshev.converged)
    assert jnp.isnan(chebyshev.error_estimate)

    spectral = la.SpectralMatrixRepresentation(
        diagonal,
        jnp.asarray([1.0, 3.0]),
        jnp.eye(2),
        jnp.eye(2),
    )
    changed = la.DenseLinearOperator(
        jnp.diag(jnp.asarray([2.0, 4.0])),
        properties=properties,
        operator_id="matrix-function-state",
    )
    with pytest.raises(ValueError, match="numerical operator state"):
        la.matrix_function_action(
            changed,
            vector,
            kind="exp",
            policy=la.MatrixFunctionPolicy("spectral"),
            spectral=spectral,
        )

    rectangular_matrix = jnp.asarray([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
    rectangular = la.DenseLinearOperator(rectangular_matrix)
    norm_estimate = la.estimate_operator_norm(
        rectangular,
        max_dimension=2,
        initial=jnp.asarray([1.0, 0.2, -0.3]),
    )
    condition_estimate = la.estimate_condition_number(
        rectangular,
        max_dimension=2,
        initial=jnp.asarray([1.0, 0.2, -0.3]),
    )
    singular_values = jnp.linalg.svd(rectangular_matrix, compute_uv=False)
    assert bool(norm_estimate.converged)
    assert bool(condition_estimate.converged)
    assert jnp.allclose(norm_estimate.value, singular_values[0])
    assert jnp.allclose(
        condition_estimate.value,
        singular_values[0] / singular_values[-1],
    )
    assert norm_estimate.matvec_count == 2
    assert norm_estimate.adjoint_matvec_count == 2

    truncated_trace = la.stochastic_trace(
        diagonal,
        key=jr.key(0),
        num_probes=2,
        max_dimension=1,
    )
    assert not bool(truncated_trace.converged)
    assert not jnp.any(truncated_trace.probe_converged)

    singular = la.DenseLinearOperator(jnp.asarray([[1.0, 0.0], [0.0, 0.0]]))
    inverse_diagonal = la.estimate_inverse_diagonal(
        singular,
        key=jr.key(1),
        num_probes=3,
        solve_policy=la.LinearSolvePolicy(la.DenseLU()),
    )
    assert not bool(inverse_diagonal.converged)
    assert jnp.all(
        inverse_diagonal.diagnostics.source_statuses == int(la.LinearSolveStatus.SINGULAR)
    )
    assert jnp.all(jnp.isnan(inverse_diagonal.estimate))


def test_sparse_preparation_validates_csr_numerical_content():
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray([0, 1], dtype=jnp.int32),
        source_size=2,
        target_size=2,
    )
    operator = phx.sparse.SparseLinearMap(
        relation,
        jnp.asarray([jnp.nan, 1.0]),
    )
    with pytest.raises(
        eqx.EquinoxRuntimeError,
        match="CSR storage must be finite",
    ):
        la.prepare(
            la.LinearSystem(operator),
            la.LinearSolvePolicy(
                la.HostSparseLU(),
                differentiation=la.DifferentiationPolicy("none"),
            ),
        )


def test_unknown_property_claims_never_authorize_certified_algorithms():
    matrix = jnp.asarray([[2.0, 1.0], [1.0, 3.0]])
    unknown = la.DenseLinearOperator(
        matrix,
        properties=la.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            rank=2,
        ),
    )
    vector = jnp.asarray([1.0, -1.0])

    assert not unknown.properties.certifies("self_adjoint")
    assert not unknown.properties.certifies("positive_definite")
    assert not unknown.properties.certifies("rank")
    with pytest.raises(ValueError, match="certified self-adjoint"):
        la.estimate_spectral_bounds(unknown, max_dimension=2, initial=vector)
    with pytest.raises(ValueError, match="Lanczos requires certified"):
        la.matrix_exponential_action(
            unknown,
            vector,
            policy=la.MatrixFunctionPolicy("lanczos", max_dimension=2),
        )

    automatic_factorization = la.factorize(unknown)
    assert automatic_factorization.prepared_solve.plan.method == "dense-lu"
    iterative = la.solve(
        la.LinearSystem(unknown),
        vector,
        policy=la.LinearSolvePolicy(
            la.FGMRES(restart=2),
            tolerance=la.TolerancePolicy(max_steps=4),
        ),
    )
    assert iterative.diagnostics.rank == -1

    certified_properties = la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={"positive_definite": "asserted"},
    )
    assert certified_properties.certifies("self_adjoint")
    assert certified_properties.certifies("positive_semidefinite")
    certified = la.DenseLinearOperator(matrix, properties=certified_properties)
    assert la.factorize(certified).prepared_solve.plan.method == "dense-cholesky"
    certified_iterative = la.solve(
        la.LinearSystem(certified),
        vector,
        policy=la.LinearSolvePolicy(la.PCG()),
    )
    assert certified_iterative.diagnostics.rank == 2


def test_square_krylov_requires_an_explicit_endomorphism():
    source = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray([2.0, 3.0])),
    )
    target = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(jnp.asarray([5.0, 7.0])),
    )
    matrix = jnp.asarray([[2.0, 1.0], [1.0, 3.0]])
    problem = la.LinearSystem(
        la.DenseLinearOperator(matrix, source=source, target=target)
    )
    rhs = jnp.asarray([1.0, -1.0])

    with pytest.raises(ValueError, match="compatible source and target spaces"):
        la.plan(problem, la.LinearSolvePolicy(la.FGMRES(restart=2)))

    direct = la.solve(problem, rhs, policy=la.LinearSolvePolicy(la.DenseLU()))
    assert bool(direct.successful)
    assert jnp.allclose(direct.value, jnp.linalg.solve(matrix, rhs))
