# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.linalg import (
    ArraySpace,
    DifferentiationPolicy,
    FailurePolicy,
    GMRES,
    LinearSolvePolicy,
    OperatorProperties,
    SolveResourcePolicy,
    SparseFactorizationPolicy,
    TolerancePolicy,
)
from phydrax.optim import (
    cone_barrier_oracle,
    ConicProgram,
    ConvexProgramStatus,
    ConvexSolvePolicy,
    ConvexTermination,
    NativeConicNewtonRoute,
    NativeHomogeneousConic,
    NonnegativeCone,
    prepare_convex_program,
    ProductCone,
    SecondOrderCone,
    solve_conic_program,
    ZeroCone,
)
from phydrax.optim._programming._native_conic import prepare_native_conic
from phydrax.optim._programming._native_hsd import _direction, _embedding_residual
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator


def _positive_metric_program(blocks: int = 1) -> ConicProgram:
    size = 2 * blocks
    variables = ArraySpace(
        (size,), dtype=jnp.float64, space_id=f"sparse-hsd-regression:metric:{blocks}"
    )
    constraints = ArraySpace(
        (2 * size,), dtype=jnp.float64, space_id=f"sparse-hsd-regression:moments:{blocks}"
    )
    base = jnp.arange(blocks, dtype=jnp.int32) * 2
    columns = jnp.stack((base, base + 1, base, base + 1, base, base + 1), axis=1).reshape(
        -1
    )
    rows = jnp.stack(
        (base, base, base + 1, base + 1, size + base, size + base + 1), axis=1
    ).reshape(-1)
    relation = EdgeRelation(columns, rows, source_size=size, target_size=2 * size)
    matrix = SparseCoordinateOperator(
        relation,
        jnp.tile(jnp.asarray([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0]), blocks),
        source=variables,
        target=constraints,
    )
    diagonal = EdgeRelation(
        jnp.arange(size, dtype=jnp.int32),
        jnp.arange(size, dtype=jnp.int32),
        source_size=size,
        target_size=size,
    )
    quadratic = SparseCoordinateOperator(
        diagonal,
        jnp.full((size,), np.exp((1 / 1.1) ** 2), dtype=jnp.float64),
        source=variables,
        target=variables,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    return ConicProgram(
        quadratic,
        jnp.zeros((size,), dtype=jnp.float64),
        matrix,
        jnp.concatenate((jnp.tile(jnp.asarray([0.0, 2.0]), blocks), jnp.zeros((size,)))),
        ProductCone((ZeroCone(size), NonnegativeCone(size))),
        problem_id=f"positive-metric-hsd-regression:{blocks}",
        convexity_evidence="construction",
    )


def _initial_vector(program: ConicProgram) -> Array:
    barrier = cone_barrier_oracle(program.cone)
    slack = barrier.interior_reference(program.linear.dtype)
    return jnp.concatenate(
        (
            jnp.zeros_like(program.linear),
            -barrier.gradient(slack),
            slack,
            jnp.ones((2,), dtype=slack.dtype),
        )
    )


@pytest.mark.parametrize("route", ("factorized", "matrix-free"))
@pytest.mark.parametrize("blocks", (1, 32))
def test_sparse_feasible_conic_stops_only_after_original_kkt_complementarity(
    blocks: int, route: NativeConicNewtonRoute
) -> None:
    program = _positive_metric_program(blocks)
    tolerance = 1e-7
    result = solve_conic_program(
        program,
        policy=ConvexSolvePolicy(
            NativeHomogeneousConic(newton=route),
            termination=ConvexTermination(absolute=tolerance, maximum_steps=64),
            failure=FailurePolicy("status"),
        ),
    )
    assert bool(result.successful)
    assert bool(jnp.all(result.primal > 0))
    np.testing.assert_allclose(
        result.primal, jnp.ones_like(program.linear), atol=tolerance
    )
    left, right = result.primal[::2], result.primal[1::2]
    equations = jnp.stack((right - left, left + right), axis=1).reshape(-1)
    constraint_values = jnp.concatenate((equations, -result.primal))
    np.testing.assert_allclose(
        constraint_values + result.cone_slack, program.constraint_rhs, atol=tolerance
    )
    assert float(result.kkt_residual_norm) <= tolerance
    assert abs(float(result.complementarity_gap)) <= tolerance


def test_factorized_newton_refuses_reduced_pattern_over_budget() -> None:
    program = _positive_metric_program(32)
    starved = NativeHomogeneousConic(
        factorization=SparseFactorizationPolicy(
            "lu", ordering="approximate-minimum-degree", max_factor_nnz=8
        )
    )
    with pytest.raises(ValueError):
        solve_conic_program(
            program,
            policy=ConvexSolvePolicy(starved, failure=FailurePolicy("status")),
        )
    with pytest.raises(ValueError, match="symmetric indefinite"):
        NativeHomogeneousConic(factorization=SparseFactorizationPolicy("cholesky"))


def test_repeated_factorized_preparation_reuses_one_compiled_solver_key() -> None:
    # Each preparation re-plans the reduced-Newton correction; identical programs
    # must still present one JIT static key so the solver is compiled once.
    program = _positive_metric_program(4)
    policy = ConvexSolvePolicy(
        NativeHomogeneousConic(newton="factorized"), failure=FailurePolicy("status")
    )
    first = prepare_native_conic(program, policy)
    second = prepare_native_conic(program, policy)
    assert first.newton is not None
    first_static = eqx.partition(first, eqx.is_array)[1]
    second_static = eqx.partition(second, eqx.is_array)[1]
    assert first_static == second_static
    assert hash(first_static) == hash(second_static)


def _dense_metric_program(
    blocks: int, padding: tuple[int, int, int] = (0, 0, 0)
) -> ConicProgram:
    # The positive-metric program as dense data: per block right - left = 0 and
    # left + right = 2 with x >= 0, zero-padded by (variables, equalities, rows).
    size = 2 * blocks
    pad_variables, pad_equalities, pad_rows = padding
    variables = size + pad_variables
    equalities = size + pad_equalities
    matrix = np.zeros((equalities + size + pad_rows, variables), dtype=np.float64)
    rhs = np.zeros(matrix.shape[0], dtype=np.float64)
    for block in range(blocks):
        left, right = 2 * block, 2 * block + 1
        matrix[left, [left, right]] = (-1.0, 1.0)
        matrix[right, [left, right]] = (1.0, 1.0)
        rhs[right] = 2.0
    matrix[equalities + np.arange(size), np.arange(size)] = -1.0
    quadratic = np.eye(variables, dtype=np.float64)
    quadratic[:size, :size] *= np.exp((1 / 1.1) ** 2)
    return ConicProgram(
        jnp.asarray(quadratic),
        jnp.zeros((variables,), dtype=jnp.float64),
        jnp.asarray(matrix),
        jnp.asarray(rhs),
        ProductCone((ZeroCone(equalities), NonnegativeCone(size + pad_rows))),
        problem_id=f"dense-metric:{variables}:{equalities}:{size + pad_rows}",
        convexity_evidence="construction",
    )


def _reduced_policy(steps: int = 64) -> ConvexSolvePolicy:
    return ConvexSolvePolicy(
        NativeHomogeneousConic(newton="reduced"),
        termination=ConvexTermination(absolute=1e-7, maximum_steps=steps),
        failure=FailurePolicy("status"),
    )


@pytest.mark.parametrize("blocks", (1, 4))
@pytest.mark.parametrize(
    "padding", ((0, 0, 0), (3, 2, 5)), ids=("exact-size", "capacity-padded")
)
def test_reduced_dense_route_solves_the_program_and_ignores_vacuous_padding(
    blocks: int, padding: tuple[int, int, int]
) -> None:
    size = 2 * blocks
    result = solve_conic_program(
        _dense_metric_program(blocks, padding), policy=_reduced_policy()
    )
    assert bool(result.successful)
    np.testing.assert_allclose(result.primal[:size], np.ones(size), atol=1e-7)
    np.testing.assert_array_equal(result.primal[size:], np.zeros(padding[0]))
    equalities = size + padding[1]
    slack = np.asarray(result.cone_slack)
    dual = np.asarray(result.cone_dual)
    np.testing.assert_array_equal(slack[equalities + size :], np.zeros(padding[2]))
    np.testing.assert_array_equal(dual[equalities + size :], np.zeros(padding[2]))
    np.testing.assert_array_equal(dual[size:equalities], np.zeros(padding[1]))
    assert float(result.kkt_residual_norm) <= 1e-7


def _active_box_program(pad_rows: int) -> ConicProgram:
    # min 1/2 |x|^2 - c.x with coupled upper limits that bind at the optimum,
    # so the centering path decides every interrupted iterate.
    matrix = np.asarray(
        [
            [1.0, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.5, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    rows = matrix.shape[0]
    padded = np.zeros((rows + pad_rows, 4), dtype=np.float64)
    padded[:rows] = matrix
    rhs = np.zeros(rows + pad_rows, dtype=np.float64)
    rhs[:rows] = (0.5, 0.75, 1.0, 0.6, 0.0, 0.0)
    return ConicProgram(
        jnp.eye(4, dtype=jnp.float64),
        -jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
        jnp.asarray(padded),
        jnp.asarray(rhs),
        NonnegativeCone(rows + pad_rows),
        problem_id=f"dense-active-box:{rows + pad_rows}",
        convexity_evidence="construction",
    )


@pytest.mark.parametrize("steps", (1, 2, 64), ids=("one-step", "two-steps", "solved"))
def test_reduced_dense_route_follows_the_unpadded_path_under_padding(steps: int) -> None:
    # Vacuous rows must stay out of the barrier and its degree: an interrupted
    # solve stops at the iterate the unpadded program reaches.
    padded = solve_conic_program(_active_box_program(10), policy=_reduced_policy(steps))
    reference = solve_conic_program(_active_box_program(0), policy=_reduced_policy(steps))
    assert int(padded.status) == int(reference.status)
    assert int(padded.iterations) == int(reference.iterations)
    np.testing.assert_allclose(padded.primal, reference.primal, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        padded.cone_dual[:6], reference.cone_dual, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        padded.cone_slack[:6], reference.cone_slack, rtol=0.0, atol=1e-12
    )


def test_reduced_dense_route_keeps_an_infeasible_zero_row() -> None:
    # 0 x + s = -1 with s >= 0 has zero coefficients but is not vacuous; treating
    # it as padding would report the infeasible program optimal.
    program = ConicProgram(
        jnp.eye(2, dtype=jnp.float64),
        jnp.zeros((2,), dtype=jnp.float64),
        jnp.asarray([[-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=jnp.float64),
        jnp.asarray([-1.0, -1.0, 0.0], dtype=jnp.float64),
        NonnegativeCone(3),
        problem_id="dense-infeasible-zero-row",
        convexity_evidence="construction",
    )
    result = solve_conic_program(program, policy=_reduced_policy())
    assert int(result.status) == int(ConvexProgramStatus.PRIMAL_INFEASIBLE)
    assert bool(result.certificate.dual_ray_valid)


def test_reduced_route_refuses_sparse_programs_and_curved_cones() -> None:
    with pytest.raises(ValueError, match="requires dense program data"):
        solve_conic_program(_positive_metric_program(), policy=_reduced_policy())
    curved = ConicProgram(
        jnp.eye(3, dtype=jnp.float64),
        jnp.zeros((3,), dtype=jnp.float64),
        -jnp.eye(3, dtype=jnp.float64),
        jnp.zeros((3,), dtype=jnp.float64),
        SecondOrderCone(3),
        problem_id="dense-second-order",
        convexity_evidence="construction",
    )
    with pytest.raises(ValueError, match="zero and nonnegative cone blocks"):
        solve_conic_program(curved, policy=_reduced_policy())


def test_sparse_newton_direction_satisfies_consumer_equation_and_work_refusal() -> None:
    program = _positive_metric_program()
    barrier = cone_barrier_oracle(program.cone)
    vector = _initial_vector(program)
    mu = jnp.asarray(1.0, dtype=vector.dtype)
    result = _direction(program, barrier, vector, mu)
    assert bool(result.successful)
    epsilon = 1e-6
    action = (
        _embedding_residual(program, barrier, vector + epsilon * result.value, mu)
        - _embedding_residual(program, barrier, vector - epsilon * result.value, mu)
    ) / (2 * epsilon)
    residual = _embedding_residual(program, barrier, vector, mu)
    np.testing.assert_allclose(action + residual, jnp.zeros_like(vector), atol=1e-7)
    insufficient = LinearSolvePolicy(
        GMRES(restart=1),
        tolerance=TolerancePolicy(relative=1e-12, absolute=1e-12, max_steps=1),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
    )
    failed = _direction(program, barrier, vector, mu, linear=insufficient)
    assert not bool(failed.successful)
    assert int(failed.diagnostics.iterations) == 1
    denied = LinearSolvePolicy(
        GMRES(restart=12),
        tolerance=TolerancePolicy(max_steps=64),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
        resources=SolveResourcePolicy(krylov_basis_bytes=0),
    )
    with pytest.raises(ValueError, match="Krylov|krylov|basis|budget"):
        _direction(program, barrier, vector, mu, linear=denied)


def _orthant_ray_program(blocks: int, kind: str) -> tuple[ConicProgram, np.ndarray]:
    """Sparse LP pairs with x >= 0 that are primal infeasible or unbounded.

    ``primal``: x0 + x1 = -1 per block. ``dual``: min -(x0 + x1) with x0 = x1.
    Returns the program and its dense constraint matrix for independent audits.
    """
    size = 2 * blocks
    variables = ArraySpace(
        (size,), dtype=jnp.float64, space_id=f"sparse-hsd-ray:{kind}:x:{blocks}"
    )
    constraints = ArraySpace(
        (blocks + size,),
        dtype=jnp.float64,
        space_id=f"sparse-hsd-ray:{kind}:c:{blocks}",
    )
    base = np.arange(blocks) * 2
    columns = np.concatenate((np.stack((base, base + 1), 1).reshape(-1), np.arange(size)))
    rows = np.concatenate((np.repeat(np.arange(blocks), 2), blocks + np.arange(size)))
    second = 1.0 if kind == "primal" else -1.0
    values = np.concatenate((np.tile([1.0, second], blocks), -np.ones(size)))
    dense = np.zeros((blocks + size, size))
    dense[rows, columns] = values
    matrix = SparseCoordinateOperator(
        EdgeRelation(
            jnp.asarray(columns, dtype=jnp.int32),
            jnp.asarray(rows, dtype=jnp.int32),
            source_size=size,
            target_size=blocks + size,
        ),
        jnp.asarray(values),
        source=variables,
        target=constraints,
    )
    rhs = np.concatenate(
        (np.full(blocks, -1.0 if kind == "primal" else 0.0), np.zeros(size))
    )
    program = ConicProgram(
        None,
        jnp.full((size,), 1.0 if kind == "primal" else -1.0, dtype=jnp.float64),
        matrix,
        jnp.asarray(rhs),
        ProductCone((ZeroCone(blocks), NonnegativeCone(size))),
        problem_id=f"sparse-hsd-ray-{kind}:{blocks}",
        convexity_evidence="construction",
    )
    return program, dense


def _ray_policy(maximum_steps: int) -> ConvexSolvePolicy:
    return ConvexSolvePolicy(
        NativeHomogeneousConic(),
        termination=ConvexTermination(absolute=1e-7, maximum_steps=maximum_steps),
        failure=FailurePolicy("status"),
    )


def test_sparse_primal_infeasible_program_stops_on_audited_dual_ray() -> None:
    blocks = 32
    program, dense = _orthant_ray_program(blocks, "primal")
    result = solve_conic_program(program, policy=_ray_policy(64))
    assert int(result.status) == int(ConvexProgramStatus.PRIMAL_INFEASIBLE)
    assert not bool(result.successful)
    assert int(result.iterations) < 64
    certificate = result.certificate
    assert bool(certificate.dual_ray_valid)
    # Farkas audit independent of the solver: A^T y = 0, y in K*, b^T y < 0.
    ray = np.asarray(certificate.inequality_dual_ray)
    assert np.max(np.abs(dense.T @ ray)) <= 1e-8
    assert np.min(ray[blocks:]) >= -1e-12
    assert float(np.asarray(program.constraint_rhs) @ ray) < -1e-8


def test_sparse_unbounded_program_stops_on_audited_primal_ray() -> None:
    blocks = 32
    program, dense = _orthant_ray_program(blocks, "dual")
    result = solve_conic_program(program, policy=_ray_policy(64))
    assert int(result.status) == int(ConvexProgramStatus.DUAL_INFEASIBLE)
    assert int(result.iterations) < 64
    certificate = result.certificate
    assert bool(certificate.primal_ray_valid)
    # Recession audit: -A d in K (zero rows vanish, orthant rows >= 0), q^T d < 0.
    ray = np.asarray(certificate.primal_ray)
    recession = -dense @ ray
    assert np.max(np.abs(recession[:blocks])) <= 1e-8
    assert np.min(recession[blocks:]) >= -1e-8
    assert float(np.asarray(program.linear) @ ray) < -1e-8


def test_sparse_infeasibility_without_certificate_stays_unresolved() -> None:
    # Same infeasible program, but the requested certificates demand a ray
    # objective below -1e3, which no normalized ray of this program can reach.
    # Without an admissible certificate the bounded run stays unresolved.
    program, _ = _orthant_ray_program(32, "primal")
    policy = ConvexSolvePolicy(
        NativeHomogeneousConic(),
        termination=ConvexTermination(
            absolute=1e-7,
            primal_infeasible=1e3,
            dual_infeasible=1e3,
            maximum_steps=2,
        ),
        failure=FailurePolicy("status"),
    )
    result = solve_conic_program(program, policy=policy)
    assert int(result.status) == int(ConvexProgramStatus.ITERATION_LIMIT)
    assert int(result.iterations) == 2
    assert not bool(result.successful)
    assert not bool(result.certificate.dual_ray_valid)
    assert not bool(result.certificate.primal_ray_valid)
    assert float(result.certificate.dual_ray_objective) < 0.0


def test_source_bound_sparse_qp_returns_an_original_coordinate_infeasibility_ray() -> (
    None
):
    # Captured implicit-sphere distance model at generation 6690, with three
    # source-bound cuts; packet SHA256 92290058fd39d4fc5412adad2e9cc061bc9880a698c57a9f4b673ef3b903f763.
    fixture_name = "sparse_hsd_distance_constraints.npz"
    with np.load(Path(__file__).with_name(fixture_name)) as data:
        rows = data["rows"]
        columns = data["columns"]
        coefficients = data["coefficients"]
        rhs = data["rhs"]
        size = 3 * len(data["vertex_ids"])
    variables = ArraySpace(
        (size,),
        dtype=jnp.float64,
        space_id="distance-constraint-regression:displacements",
    )
    constraints = ArraySpace(
        rhs.shape, dtype=jnp.float64, space_id="distance-constraint-regression:rows"
    )
    matrix = SparseCoordinateOperator(
        EdgeRelation(columns, rows, source_size=size, target_size=rhs.size),
        coefficients,
        source=variables,
        target=constraints,
    )
    diagonal = jnp.arange(size, dtype=jnp.int32)
    quadratic = SparseCoordinateOperator(
        EdgeRelation(diagonal, diagonal, source_size=size, target_size=size),
        jnp.ones((size,), dtype=jnp.float64),
        source=variables,
        target=variables,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    program = ConicProgram(
        quadratic,
        jnp.zeros((size,), dtype=jnp.float64),
        matrix,
        rhs,
        NonnegativeCone(rhs.size),
        problem_id="source-bound-distance-infeasibility",
        convexity_evidence="Identity Hessian is positive definite.",
    )
    result = solve_conic_program(program, policy=_ray_policy(100))
    assert int(result.status) == int(ConvexProgramStatus.PRIMAL_INFEASIBLE)
    assert bool(result.certificate.dual_ray_valid)
    assert not bool(result.successful)
    reference = np.zeros((rhs.size, size), dtype=np.float64)
    reference[rows, columns] = coefficients
    ray = np.asarray(result.certificate.inequality_dual_ray)
    assert np.max(np.abs(reference.T @ ray)) <= 1e-8
    assert np.min(ray) >= -1e-12
    assert float(rhs @ ray) < -1e-8


def test_sparse_dependent_equalities_preserve_the_primal_and_kkt_contract() -> None:
    variables = ArraySpace(
        (2,), dtype=jnp.float64, space_id="dependent-equality-regression:variables"
    )
    constraints = ArraySpace(
        (4,), dtype=jnp.float64, space_id="dependent-equality-regression:constraints"
    )
    matrix = SparseCoordinateOperator(
        EdgeRelation(
            jnp.asarray([0, 1, 0, 1, 0, 1], dtype=jnp.int32),
            jnp.asarray([0, 0, 1, 1, 2, 3], dtype=jnp.int32),
            source_size=2,
            target_size=4,
        ),
        jnp.asarray([1.0, 1.0, 1.0, 1.0, -1.0, -1.0], dtype=jnp.float64),
        source=variables,
        target=constraints,
    )
    program = ConicProgram(
        jnp.eye(2, dtype=jnp.float64),
        jnp.asarray([-2.0, 0.0], dtype=jnp.float64),
        matrix,
        jnp.asarray([1.0, 1.0, 0.0, 0.0], dtype=jnp.float64),
        ProductCone((ZeroCone(2), NonnegativeCone(2))),
        problem_id="dependent-equality-regression",
        convexity_evidence="Identity Hessian is positive definite.",
    )
    result = solve_conic_program(program, policy=_ray_policy(100))
    assert bool(result.successful)
    np.testing.assert_allclose(result.primal, [1.0, 0.0], atol=1e-6)
    assert float(result.kkt_residual_norm) <= 1e-7
    assert not bool(result.certificate.dual_ray_valid)


@pytest.mark.parametrize(
    "resources",
    (
        pytest.param(SolveResourcePolicy(krylov_basis_bytes=0), id="krylov-basis"),
        pytest.param(SolveResourcePolicy(preconditioner_bytes=0), id="preconditioner"),
        pytest.param(SolveResourcePolicy(factorization_bytes=0), id="factorization"),
        pytest.param(SolveResourcePolicy(workspace_bytes=0), id="workspace"),
    ),
)
def test_factorized_correction_refuses_unadmitted_resources_at_preparation(
    resources: SolveResourcePolicy,
) -> None:
    program = _positive_metric_program()
    with pytest.raises(ValueError):
        prepare_convex_program(
            program,
            ConvexSolvePolicy(
                NativeHomogeneousConic(),
                resources=resources,
                failure=FailurePolicy("status"),
            ),
        )
