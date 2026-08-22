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
    inverse_action = la.DenseInversePreconditionerBuilder().prepare(
        primal,
        materialization=la.MaterializationPolicy(),
    )
    schur = la.saddle_point_schur_complement(
        primal,
        constraint,
        inverse_action,
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


def test_kernel_certificate_selects_projected_pcg_and_rebinds_coefficients():
    space = la.ArraySpace((3,), dtype=jnp.float64)
    matrix = jnp.asarray([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    properties = la.OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_semidefinite": "construction",
        },
    )
    operator = la.DenseLinearOperator(
        matrix,
        source=space,
        target=space,
        properties=properties,
        operator_id="certified-path-laplacian",
    )
    kernel = la.LinearSubspace(space, jnp.ones((3, 1)))
    certificate = la.KernelCertificate(
        operator,
        kernel,
        complete=True,
        evidence="verified",
    )
    problem = la.LinearSystem(
        operator,
        nullspace_policy=la.NullspacePolicy(certificate=certificate),
    )
    right_hand_side = jnp.asarray([1.0, 0.0, -1.0])

    template = la.prepare_template(problem)
    result = la.solve(la.bind_numeric(template, problem), right_hand_side)
    compiled = jax.jit(lambda rhs: la.solve(problem, rhs).value)(right_hand_side)

    assert template.plan.method == "projected-pcg"
    assert bool(certificate.valid)
    assert bool(result.successful)
    assert int(result.diagnostics.rank) == 2
    assert int(result.diagnostics.nullity) == 1
    assert jnp.allclose(operator.matrix @ result.value, right_hand_side)
    assert jnp.allclose(jnp.sum(result.value), 0.0, atol=1e-12)
    assert jnp.allclose(compiled, result.value)

    changed_operator = la.DenseLinearOperator(
        2.0 * matrix,
        source=space,
        target=space,
        properties=properties,
        operator_id=operator.operator_id,
    )
    changed_certificate = la.KernelCertificate(
        changed_operator,
        kernel,
        complete=True,
        evidence="verified",
    )
    changed_problem = la.LinearSystem(
        changed_operator,
        nullspace_policy=la.NullspacePolicy(certificate=changed_certificate),
        problem_id=problem.problem_id,
    )
    rebound = la.bind_numeric(template, changed_problem, numeric_version=1)
    changed_result = la.solve(rebound, right_hand_side)

    assert int(rebound.numeric_version) == 1
    assert jnp.allclose(changed_result.value, 0.5 * result.value)
    with pytest.raises(ValueError, match="stale numerical kernel certificate"):
        la.plan(
            la.LinearSystem(
                changed_operator,
                nullspace_policy=la.NullspacePolicy(certificate=certificate),
                problem_id=problem.problem_id,
            ),
            la.LinearSolvePolicy(la.ProjectedPCG()),
        )


def test_projected_pcg_requires_complete_valid_kernel_evidence():
    space = la.ArraySpace((2,), dtype=jnp.float64)
    matrix = jnp.asarray([[1.0, -1.0], [-1.0, 1.0]])
    properties = la.OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_semidefinite": "construction",
        },
    )
    operator = la.DenseLinearOperator(
        matrix,
        source=space,
        target=space,
        properties=properties,
    )
    kernel = la.LinearSubspace(space, jnp.ones((2, 1)))
    incomplete = la.KernelCertificate(operator, kernel, complete=False)
    problem = la.LinearSystem(
        operator,
        nullspace_policy=la.NullspacePolicy(certificate=incomplete),
    )

    with pytest.raises(ValueError, match="complete kernel/nullity certificate"):
        la.plan(problem, la.LinearSolvePolicy(la.ProjectedPCG()))

    invalid_kernel = la.LinearSubspace(space, jnp.asarray([[1.0], [0.0]]))
    invalid = la.KernelCertificate(
        operator,
        invalid_kernel,
        complete=True,
        tolerance=1e-12,
    )
    invalid_problem = la.LinearSystem(
        operator,
        nullspace_policy=la.NullspacePolicy(
            certificate=invalid,
            compatibility="project",
        ),
    )
    with pytest.raises(Exception, match="valid nonempty kernel certificate"):
        la.solve(
            invalid_problem,
            jnp.asarray([1.0, -1.0]),
            policy=la.LinearSolvePolicy(la.ProjectedPCG()),
        )


def test_spectral_matrix_functions_and_stochastic_estimators_are_replayable():
    matrix = jnp.diag(jnp.asarray([1.5, 2.0, 4.0]))
    operator = la.DenseLinearOperator(
        matrix,
        properties=_positive_definite_properties(),
        operator_id="diagonal-spectrum",
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    spectral = la.TransformDiagonalRepresentation(
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
            la.SparseLU(),
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

    spectral = la.TransformDiagonalRepresentation(
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
                la.SparseLU(),
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


class _AlternatingPreconditioner(la.AbstractPreconditioner):
    def __init__(self, space):
        self.space = space
        self.properties = la.PreconditionerProperties(
            linear=True,
            evidence={"linear": "construction"},
        )
        self.preconditioner_id = "alternating-preconditioner"

    def apply(self, residual, /, *, iteration=None):
        index = jnp.asarray(0 if iteration is None else iteration)
        scale = jnp.where(index % 2 == 0, 1.0, 0.5)
        return scale * self.space.validate(residual)

    def cost_for(self, setup_operator, /, *, materialization=None):
        return la.PreconditionerCostEstimate(
            component=self.preconditioner_id,
            apply_workspace_bytes_per_rhs=(
                setup_operator.source.size * jnp.dtype(jnp.float64).itemsize
            ),
            reason="supplied alternating test action",
        )


def test_preconditioner_builder_owns_distinct_setup_and_refresh_provenance():
    properties = _positive_definite_properties()
    space = la.ArraySpace((2,), dtype=jnp.float64)
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
    operator = la.FunctionLinearOperator(
        lambda vector: matrix @ vector,
        source=space,
        target=space,
        transpose_action=lambda vector: matrix.T @ vector,
        properties=properties,
        operator_id="preconditioning-system",
    )
    setup = la.DiagonalLinearOperator(
        jnp.asarray([8.0, 6.0]),
        space=space,
        properties=properties,
        operator_id="preconditioning-setup",
    )
    policy = la.LinearSolvePolicy(
        la.PCG(),
        tolerance=la.TolerancePolicy(relative=1e-12, max_steps=10),
        preconditioning=la.PreconditioningPolicy(
            la.JacobiPreconditionerBuilder(),
            setup_operator=setup,
        ),
    )
    prepared = la.prepare(la.LinearSystem(operator), policy)
    preconditioning_state = prepared.preconditioning_state
    assert preconditioning_state is not None
    assert isinstance(preconditioning_state.action, la.DiagonalPreconditioner)

    assert prepared.plan.preconditioner_plan is not None
    assert prepared.plan.preconditioner_plan.setup_operator_id == setup.operator_id
    assert prepared.plan.preconditioner_plan.side == "left"
    assert jnp.allclose(
        preconditioning_state.action.inverse_diagonal,
        jnp.asarray([1.0 / 8.0, 1.0 / 6.0]),
    )

    updated_matrix = jnp.asarray([[5.0, 1.0], [1.0, 4.0]])
    updated_operator = la.FunctionLinearOperator(
        lambda vector: updated_matrix @ vector,
        source=space,
        target=space,
        transpose_action=lambda vector: updated_matrix.T @ vector,
        properties=properties,
        operator_id="preconditioning-system",
    )
    updated_setup = la.DiagonalLinearOperator(
        jnp.asarray([10.0, 8.0]),
        space=space,
        properties=properties,
        operator_id="preconditioning-setup",
    )
    with pytest.raises(ValueError, match="setup_operator"):
        la.refresh(prepared, la.LinearSystem(updated_operator))

    refreshed = la.refresh(
        prepared,
        la.LinearSystem(updated_operator),
        setup_operator=updated_setup,
    )
    refreshed_preconditioning_state = refreshed.preconditioning_state
    assert refreshed_preconditioning_state is not None
    assert isinstance(refreshed_preconditioning_state.action, la.DiagonalPreconditioner)
    result = la.solve(refreshed, jnp.asarray([1.0, 2.0]))

    assert refreshed.numeric_version == 1
    assert refreshed_preconditioning_state.refresh_kind == "refreshed"
    assert refreshed_preconditioning_state.built_numeric_version == 1
    assert jnp.allclose(
        refreshed_preconditioning_state.action.inverse_diagonal,
        jnp.asarray([0.1, 0.125]),
    )
    assert jnp.allclose(
        result.value,
        jnp.linalg.solve(updated_matrix, jnp.asarray([1.0, 2.0])),
    )
    assert result.provenance.preconditioner_plan_id is not None
    assert result.provenance.preconditioning_side == "left"
    assert result.provenance.preconditioner_refresh == "refreshed"
    assert result.provenance.preconditioner_numeric_version == 1
    assert result.provenance.preconditioner_built_numeric_version == 1


def test_fixed_krylov_rejects_variable_preconditioning_but_fgmres_accepts_it():
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
    space = la.ArraySpace((2,), dtype=jnp.float64)
    operator = la.DenseLinearOperator(
        matrix,
        source=space,
        target=space,
        properties=_positive_definite_properties(),
    )
    preconditioner = _AlternatingPreconditioner(space)
    preconditioning = la.PreconditioningPolicy(preconditioner)

    with pytest.raises(ValueError, match="fixed linear preconditioning"):
        la.plan(
            la.LinearSystem(operator),
            la.LinearSolvePolicy(
                la.GMRES(restart=2),
                preconditioning=preconditioning,
            ),
        )
    with pytest.raises(ValueError, match="requires right preconditioning"):
        la.plan(
            la.LinearSystem(operator),
            la.LinearSolvePolicy(
                la.FGMRES(restart=2),
                preconditioning=la.PreconditioningPolicy(
                    preconditioner,
                    side="left",
                ),
            ),
        )

    result = la.solve(
        la.LinearSystem(operator),
        jnp.asarray([1.0, 2.0]),
        policy=la.LinearSolvePolicy(
            la.FGMRES(restart=2),
            tolerance=la.TolerancePolicy(relative=1e-12, max_steps=10),
            preconditioning=preconditioning,
        ),
    )

    assert bool(result.successful)
    assert jnp.allclose(
        result.value,
        jnp.linalg.solve(matrix, jnp.asarray([1.0, 2.0])),
    )
    assert result.provenance.preconditioning_side == "right"


def test_bicgstab_preconditioning_is_explicitly_right_sided():
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
    operator = la.DenseLinearOperator(matrix)
    preconditioner = la.DiagonalPreconditioner(jnp.diag(matrix))
    policy = la.LinearSolvePolicy(
        la.BiCGStab(),
        tolerance=la.TolerancePolicy(relative=1e-12, max_steps=10),
        preconditioning=la.PreconditioningPolicy(preconditioner),
    )

    result = la.solve(la.LinearSystem(operator), jnp.asarray([1.0, 2.0]), policy=policy)

    assert bool(result.successful)
    assert result.provenance.preconditioning_side == "right"
    with pytest.raises(ValueError, match="requires right preconditioning"):
        la.plan(
            la.LinearSystem(operator),
            la.LinearSolvePolicy(
                la.BiCGStab(),
                preconditioning=la.PreconditioningPolicy(
                    preconditioner,
                    side="left",
                ),
            ),
        )


def test_supplied_preconditioner_is_reused_across_refresh():
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]])
    operator = la.DenseLinearOperator(matrix, operator_id="frozen-action-system")
    action = la.DiagonalPreconditioner(
        jnp.diag(matrix),
        preconditioner_id="frozen-action",
    )
    policy = la.LinearSolvePolicy(
        la.FGMRES(restart=2),
        preconditioning=la.PreconditioningPolicy(action),
    )
    prepared = la.prepare(la.LinearSystem(operator), policy)
    updated_operator = la.DenseLinearOperator(
        matrix + jnp.eye(2),
        operator_id="frozen-action-system",
    )
    refreshed = la.refresh(prepared, la.LinearSystem(updated_operator))
    refreshed_preconditioning_state = refreshed.preconditioning_state
    assert refreshed_preconditioning_state is not None
    assert isinstance(refreshed_preconditioning_state.action, la.DiagonalPreconditioner)

    assert refreshed_preconditioning_state.refresh_kind == "reused"
    assert refreshed_preconditioning_state.numeric_version == 1
    assert refreshed_preconditioning_state.built_numeric_version == 0
    assert jnp.allclose(
        refreshed_preconditioning_state.action.inverse_diagonal,
        jnp.reciprocal(jnp.diag(matrix)),
    )


def test_sparse_jacobi_builder_avoids_dense_materialization():
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1, 0, 1, 2, 1, 2]),
        jnp.asarray([0, 0, 1, 1, 1, 2, 2]),
        source_size=3,
        target_size=3,
    )
    operator = phx.sparse.SparseLinearMap(
        relation,
        jnp.asarray([2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0]),
        properties=_positive_definite_properties(),
    )
    prepared = la.prepare(
        la.LinearSystem(operator),
        la.LinearSolvePolicy(
            la.PCG(),
            materialization=la.MaterializationPolicy(max_entries=1, max_bytes=8),
            preconditioning=la.PreconditioningPolicy(la.JacobiPreconditionerBuilder()),
        ),
    )
    result = la.solve(prepared, jnp.asarray([1.0, 0.0, 1.0]))
    preconditioning_state = prepared.preconditioning_state
    assert preconditioning_state is not None
    assert isinstance(preconditioning_state.action, la.DiagonalPreconditioner)

    assert jnp.allclose(
        preconditioning_state.action.inverse_diagonal,
        jnp.full((3,), 0.5),
    )
    assert bool(result.successful)
    assert jnp.allclose(operator.mv(result.value), jnp.asarray([1.0, 0.0, 1.0]))


def test_multigrid_hierarchy_is_explicit_jittable_and_pyamg_convertible():
    from types import SimpleNamespace

    import scipy.sparse as sp

    operator_properties = _positive_definite_properties()
    preconditioner_properties = la.PreconditionerProperties(
        linear=True,
        stationary=True,
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "linear": "asserted",
            "stationary": "asserted",
            "self_adjoint": "asserted",
            "positive_definite": "asserted",
        },
    )
    fine_matrix = jnp.asarray(
        [
            [2.0, -1.0, 0.0, 0.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [0.0, 0.0, -1.0, 2.0],
        ]
    )
    coarse_matrix = jnp.asarray([[1.5, -0.5], [-0.5, 1.5]])
    restriction_matrix = jnp.asarray([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
    prolongation_matrix = jnp.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    fine = la.DenseLinearOperator(
        fine_matrix,
        properties=operator_properties,
        operator_id="multigrid-fine",
    )
    coarse = la.DenseLinearOperator(
        coarse_matrix,
        properties=operator_properties,
        operator_id="multigrid-coarse",
    )
    restriction = la.DenseLinearOperator(restriction_matrix)
    prolongation = la.DenseLinearOperator(prolongation_matrix)
    builder = la.MultigridHierarchyBuilder(
        (
            la.MultigridLevelBuilder(
                fine,
                la.JacobiPreconditionerBuilder(relaxation=2.0 / 3.0),
                restriction=restriction,
                prolongation=prolongation,
            ),
            la.MultigridLevelBuilder(
                coarse,
                la.DenseInversePreconditionerBuilder(),
                pre_smoothing=0,
                post_smoothing=0,
            ),
        ),
        properties=preconditioner_properties,
    )
    variable_levels = (
        la.MultigridLevelBuilder(
            fine,
            _AlternatingPreconditioner(fine.source),
            restriction=restriction,
            prolongation=prolongation,
        ),
        la.MultigridLevelBuilder(
            coarse,
            _AlternatingPreconditioner(coarse.source),
            pre_smoothing=0,
            post_smoothing=0,
        ),
    )
    variable_builder = la.MultigridHierarchyBuilder(variable_levels)
    assert variable_builder.properties.certifies("linear")
    assert not variable_builder.properties.certifies("stationary")
    with pytest.raises(ValueError, match="every level source"):
        la.MultigridHierarchyBuilder(
            variable_levels,
            properties=preconditioner_properties,
        )
    prepared = la.prepare(
        la.LinearSystem(fine),
        la.LinearSolvePolicy(
            la.PCG(),
            tolerance=la.TolerancePolicy(relative=1e-12, max_steps=20),
            preconditioning=la.PreconditioningPolicy(builder),
        ),
    )
    preconditioning_state = prepared.preconditioning_state
    assert preconditioning_state is not None
    action = preconditioning_state.action
    residual = jnp.ones(4)
    correction = jax.jit(lambda selected, value: selected.apply(value))(action, residual)
    result = la.solve(prepared, residual)

    assert isinstance(action, la.MultigridPreconditioner)
    assert jnp.linalg.norm(residual - fine.mv(correction)) < jnp.linalg.norm(residual)
    assert bool(result.successful)
    assert jnp.allclose(result.value, jnp.linalg.solve(fine_matrix, residual))
    updated_fine = la.DenseLinearOperator(
        fine_matrix + 0.25 * jnp.eye(4),
        properties=operator_properties,
        operator_id="multigrid-fine",
    )
    refreshed = la.refresh(prepared, la.LinearSystem(updated_fine))
    refreshed_preconditioning_state = refreshed.preconditioning_state
    assert refreshed_preconditioning_state is not None
    assert refreshed_preconditioning_state.refresh_kind == "reused"
    assert refreshed_preconditioning_state.numeric_version == 1
    assert refreshed_preconditioning_state.built_numeric_version == 0
    assert (
        refreshed_preconditioning_state.action.preconditioner_id
        == action.preconditioner_id
    )

    fake_solver = SimpleNamespace(
        levels=(
            SimpleNamespace(
                A=sp.csr_matrix(fine_matrix),
                R=sp.csr_matrix(restriction_matrix),
                P=sp.csr_matrix(prolongation_matrix),
            ),
            SimpleNamespace(A=sp.csr_matrix(coarse_matrix)),
        )
    )
    converted = la.multigrid_hierarchy_from_pyamg(
        fake_solver,
        properties=preconditioner_properties,
    )
    converted_action = la.MultigridPreconditioner(converted)
    converted_correction = jax.jit(lambda selected, value: selected.apply(value))(
        converted_action, residual
    )

    assert len(converted.levels) == 2
    assert jnp.all(jnp.isfinite(converted_correction))
    assert jnp.linalg.norm(
        residual - converted.levels[0].operator.mv(converted_correction)
    ) < jnp.linalg.norm(residual)


def test_preconditioner_setup_is_not_part_of_mathematical_solve_derivative():
    rhs = jnp.asarray([2.0, -3.0])
    properties = _positive_definite_properties()

    def objective(system_diagonal, setup_diagonal):
        operator = la.DiagonalLinearOperator(
            system_diagonal,
            properties=properties,
            operator_id="differentiable-system",
        )
        setup = la.DiagonalLinearOperator(
            setup_diagonal,
            properties=properties,
            operator_id="differentiable-setup",
        )
        result = la.solve(
            la.LinearSystem(operator),
            rhs,
            policy=la.LinearSolvePolicy(
                la.PCG(),
                tolerance=la.TolerancePolicy(relative=1e-12, max_steps=4),
                preconditioning=la.PreconditioningPolicy(
                    la.JacobiPreconditionerBuilder(),
                    setup_operator=setup,
                ),
                differentiation=la.DifferentiationPolicy("mathematical"),
            ),
        )
        return jnp.sum(result.value)

    system_diagonal = jnp.asarray([2.0, 4.0])
    setup_diagonal = jnp.asarray([3.0, 5.0])
    system_gradient, setup_gradient = jax.grad(objective, argnums=(0, 1))(
        system_diagonal,
        setup_diagonal,
    )

    assert jnp.allclose(system_gradient, -rhs / system_diagonal**2)
    assert jnp.allclose(setup_gradient, jnp.zeros_like(setup_diagonal))
