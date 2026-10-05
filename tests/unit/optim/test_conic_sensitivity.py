#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.optim._programming._clarabel import _audit_result


def _policy(*, regularization: Any = 0.0) -> Any:
    return phx.optim.ConvexSolvePolicy(
        phx.optim.ClarabelInteriorPoint(presolve=False),
        termination=phx.optim.ConvexTermination(
            absolute=1e-9,
            relative=1e-9,
            maximum_steps=500,
        ),
        regularization=regularization,
    )


def _prepare(
    program: Any,
    *,
    solution: Any = None,
    regularization: Any = 0.0,
    linear: Any = None,
    regularity_tolerance: Any = 1e-7,
    representation: Any = "dense",
    stability: Any = None,
    generalized: Any = None,
) -> Any:
    policy = _policy(regularization=regularization)
    if solution is None:
        pytest.importorskip("clarabel")
        prepared = phx.optim.prepare_convex_program(program, policy)
        execution = phx.optim.solve_convex_program(prepared)
    else:
        plan = phx.optim.plan_convex_program(program, policy)
        prepared = phx.optim.bind_convex_numeric(
            phx.optim.ConvexProgramTemplate(plan),
            program,
        )
        primal, slack, dual, lower_dual, upper_dual = solution
        result = _audit_result(
            program,
            jnp.asarray(primal),
            jnp.asarray(slack),
            jnp.asarray(dual),
            jnp.asarray(lower_dual),
            jnp.asarray(upper_dual),
            jnp.ones(program.batch_shape, dtype="bool"),
            jnp.zeros(program.batch_shape, dtype=jnp.int32),
            policy,
            "analytic-test",
        )
        provenance = phx.optim.ConvexProgramProvenance(
            numeric_version=prepared.numeric_version,
            problem_id=program.problem_id,
            structure_id=program.structure_id,
            policy_id=policy.policy_id,
            method_id=policy.method.method_id,
            backend=result.provenance.backend,
            backend_version=result.provenance.backend_version,
            convexity_evidence=program.convexity_evidence,
            regularization=policy.regularization,
            numeric_binding_id=prepared.numeric_binding_id,
        )
        result = eqx.tree_at(lambda value: value.provenance, result, provenance)
        execution = phx.optim.ConvexProgramExecution(
            result,
            numeric_version=prepared.numeric_version,
            plan_id=prepared.plan.plan_id,
            numeric_binding_id=prepared.numeric_binding_id,
        )
    sensitivity = phx.optim.prepare_conic_sensitivity(
        prepared,
        execution,
        linear=linear,
        representation=representation,
        stability=stability,
        generalized=generalized,
        regularity_tolerance=regularity_tolerance,
    )
    return prepared, execution, sensitivity


def _tangent(
    program: Any,
    *,
    quadratic: Any = None,
    linear: Any = None,
    matrix: Any = None,
    rhs: Any = None,
    lower: Any = None,
    upper: Any = None,
) -> Any:
    zero = phx.optim.ConicProgramData.zeros_like(program)
    return phx.optim.ConicProgramData(
        zero.quadratic if quadratic is None else quadratic,
        zero.linear if linear is None else linear,
        zero.constraint_matrix if matrix is None else matrix,
        zero.constraint_rhs if rhs is None else rhs,
        zero.lower_bounds if lower is None else lower,
        zero.upper_bounds if upper is None else upper,
    )


def _data_pairing(left: Any, right: Any) -> Any:
    value = jnp.vdot(left.linear, right.linear)
    value += jnp.vdot(left.constraint_matrix, right.constraint_matrix)
    value += jnp.vdot(left.constraint_rhs, right.constraint_rhs)
    value += jnp.vdot(left.lower_bounds, right.lower_bounds)
    value += jnp.vdot(left.upper_bounds, right.upper_bounds)
    if left.quadratic is not None and right.quadratic is not None:
        value += jnp.vdot(left.quadratic, right.quadratic)
    return value


def test_conic_sensitivity_scenario_1() -> None:
    zero = phx.optim.ZeroCone(2)
    nonnegative = phx.optim.NonnegativeCone(3)
    soc = phx.optim.SecondOrderCone(3)
    rotated = phx.optim.RotatedSecondOrderCone(3)
    product = phx.optim.ProductCone((nonnegative, soc))

    assert jnp.isinf(zero.dual_projection_smoothness_margin(jnp.ones(2)))
    np.testing.assert_allclose(
        nonnegative.dual_projection_smoothness_margin(jnp.asarray([2.0, -3.0, 0.5])),
        0.5,
    )
    assert (
        nonnegative.dual_projection_smoothness_margin(jnp.asarray([1.0, 0.0, -1.0])) == 0
    )
    assert soc.dual_projection_smoothness_margin(jnp.asarray([2.0, 0.5, 0.0])) > 0
    assert soc.dual_projection_smoothness_margin(jnp.asarray([1.0, 1.0, 0.0])) == 0
    assert soc.dual_projection_smoothness_margin(jnp.zeros(3)) == 0
    rotated_point = jnp.asarray([2.0, 1.0, 0.25])
    np.testing.assert_allclose(
        rotated.dual_projection_smoothness_margin(rotated_point),
        rotated._soc.dual_projection_smoothness_margin(rotated._to_soc(rotated_point)),
    )
    blocks = jnp.asarray([2.0, -3.0, 0.5, 2.0, 0.5, 0.0])
    np.testing.assert_allclose(product.dual_projection_smoothness_margin(blocks), 0.5)
    empty = phx.optim.ProductCone(())
    assert jnp.isinf(empty.dual_projection_smoothness_margin(jnp.empty((0,))))
    problem = phx.optim.ConicProgram(
        jnp.ones((1, 1)),
        jnp.asarray([-2.0]),
        jnp.ones((1, 1)),
        jnp.asarray([1.0]),
        phx.optim.NonnegativeCone(1),
        problem_id="dense-sensitivity-sparse-tangent",
    )
    _, _, sensitivity = _prepare(
        problem,
        solution=(
            jnp.ones(1),
            jnp.zeros(1),
            jnp.ones(1),
            jnp.zeros(1),
            jnp.zeros(1),
        ),
    )
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([0], dtype=jnp.int32),
        source_size=1,
        target_size=1,
    )
    sparse_matrix = phx.sparse.SparseLinearMap(
        relation, jnp.zeros(1, dtype=problem.linear.dtype)
    )
    tangent = _tangent(problem, matrix=sparse_matrix)

    with pytest.raises(TypeError, match="dense tangent data"):
        phx.optim.conic_primal_jvp(sensitivity, tangent)
    problem = phx.optim.ConicProgram(
        jnp.ones((1, 1)),
        jnp.asarray([-2.0]),
        jnp.ones((1, 1)),
        jnp.asarray([1.0]),
        phx.optim.NonnegativeCone(1),
        problem_id="active-orthant-sensitivity",
    )
    _, execution, sensitivity = _prepare(
        problem,
        solution=(
            jnp.ones(1),
            jnp.zeros(1),
            jnp.ones(1),
            jnp.zeros(1),
            jnp.zeros(1),
        ),
    )
    assert execution.result.successful
    tangent = _tangent(problem, rhs=jnp.ones(1))

    derivative = phx.optim.conic_primal_jvp(sensitivity, tangent)
    np.testing.assert_allclose(derivative.value, jnp.ones(1), atol=2e-6)
    assert derivative.available
    assert derivative.status == int(phx.optim.ConicSensitivityStatus.REGULAR_FIXED_ACTIVE)
    assert derivative.active_set.projection_residual_norm < 1e-7

    alpha = jnp.asarray(1e-9, dtype=problem.linear.dtype)
    scaled = _tangent(problem, rhs=alpha * jnp.ones(1))
    scaled_derivative = phx.optim.conic_primal_jvp(sensitivity, scaled)
    np.testing.assert_allclose(
        scaled_derivative.value,
        alpha * derivative.value,
        rtol=2e-4,
        atol=1e-12,
    )


def test_conic_sensitivity_scenario_2() -> None:
    cone = phx.optim.SecondOrderCone(2)
    center = jnp.asarray([0.0, 2.0])
    direction = jnp.asarray([0.3, -0.2])
    problem = phx.optim.ConicProgram(
        jnp.eye(2),
        -center,
        -jnp.eye(2),
        jnp.zeros(2),
        cone,
        problem_id="soc-projection-sensitivity",
    )
    primal = cone.project(center)
    _, execution, sensitivity = _prepare(
        problem,
        solution=(
            primal,
            primal,
            primal - center,
            jnp.zeros(2),
            jnp.zeros(2),
        ),
    )
    np.testing.assert_allclose(execution.result.primal, cone.project(center), atol=2e-6)
    tangent = _tangent(problem, linear=-direction)

    derivative = phx.optim.conic_primal_jvp(sensitivity, tangent)
    expected = jax.jvp(cone.project, (center,), (direction,))[1]
    np.testing.assert_allclose(derivative.value, expected, atol=2e-6, rtol=2e-6)
    assert derivative.available

    cotangent = jnp.asarray([0.7, -0.4])
    adjoint = phx.optim.conic_primal_vjp(sensitivity, cotangent)
    np.testing.assert_allclose(
        jnp.vdot(cotangent, derivative.value),
        _data_pairing(adjoint.value, tangent),
        atol=2e-7,
        rtol=2e-7,
    )
    assert adjoint.available
    cone = phx.optim.SecondOrderCone(2)
    centers = jnp.asarray([[0.0, 2.0], [0.0, 3.0]])
    directions = jnp.asarray([[0.3, -0.2], [-0.4, 0.1]])
    problem = phx.optim.ConicProgram(
        jnp.eye(2),
        -centers,
        -jnp.eye(2),
        jnp.zeros(2),
        cone,
        problem_id="batched-soc-projection-sensitivity",
    )
    primal = jax.vmap(cone.project)(centers)
    _, _, sensitivity = _prepare(
        problem,
        solution=(
            primal,
            primal,
            primal - centers,
            jnp.zeros_like(centers),
            jnp.zeros_like(centers),
        ),
    )
    tangent = _tangent(problem, linear=-directions)

    derivative = phx.optim.conic_primal_jvp(sensitivity, tangent)
    expected = jax.vmap(
        lambda point, direction: jax.jvp(cone.project, (point,), (direction,))[1]
    )(centers, directions)
    np.testing.assert_allclose(derivative.value, expected, atol=3e-6, rtol=3e-6)
    np.testing.assert_array_equal(derivative.available, jnp.asarray([True, True]))
    assert derivative.value.shape == centers.shape
    assert derivative.linear_status.shape == (2,)
    cotangent = jnp.asarray([[0.7, -0.4], [-0.2, 0.5]])
    adjoint = phx.optim.conic_primal_vjp(sensitivity, cotangent)
    np.testing.assert_allclose(
        jnp.vdot(cotangent, derivative.value),
        _data_pairing(adjoint.value, tangent),
        atol=3e-7,
        rtol=3e-7,
    )
    np.testing.assert_array_equal(adjoint.available, jnp.asarray([True, True]))
    problem = phx.optim.ConicProgram(
        jnp.ones((1, 1)),
        jnp.zeros(1),
        jnp.empty((0, 1)),
        jnp.empty((0,)),
        phx.optim.ProductCone(()),
        bounds=phx.optim.Bounds(2.0, 2.0),
        problem_id="fixed-bound-sensitivity",
    )
    _, execution, sensitivity = _prepare(
        problem,
        solution=(
            jnp.asarray([2.0]),
            jnp.empty((0,)),
            jnp.empty((0,)),
            jnp.asarray([2.0]),
            jnp.zeros(1),
        ),
    )
    np.testing.assert_allclose(execution.result.primal, jnp.asarray([2.0]), atol=2e-6)
    tangent = _tangent(problem, lower=jnp.ones(1), upper=jnp.ones(1))

    derivative = phx.optim.conic_primal_jvp(sensitivity, tangent)
    np.testing.assert_allclose(derivative.value, jnp.ones(1), atol=2e-6)
    adjoint = phx.optim.conic_primal_vjp(sensitivity, jnp.ones(1))
    np.testing.assert_allclose(adjoint.value.lower_bounds, jnp.asarray([0.5]), atol=2e-6)
    np.testing.assert_allclose(adjoint.value.upper_bounds, jnp.asarray([0.5]), atol=2e-6)
    np.testing.assert_allclose(_data_pairing(adjoint.value, tangent), 1.0, atol=2e-6)

    invalid = _tangent(problem, lower=jnp.ones(1), upper=jnp.zeros(1))
    with pytest.raises((ValueError, RuntimeError), match="preserve fixed"):
        jax.block_until_ready(phx.optim.conic_primal_jvp(sensitivity, invalid).value)


def test_conic_sensitivity_scenario_3() -> None:
    problem = phx.optim.ConicProgram(
        jnp.ones((1, 1)),
        jnp.zeros(1),
        jnp.ones((1, 1)),
        jnp.zeros(1),
        phx.optim.NonnegativeCone(1),
        problem_id="weak-complementarity-sensitivity",
    )
    _, execution, sensitivity = _prepare(
        problem,
        solution=(
            jnp.zeros(1),
            jnp.zeros(1),
            jnp.zeros(1),
            jnp.zeros(1),
            jnp.zeros(1),
        ),
        regularity_tolerance=1e-5,
    )
    assert execution.result.successful
    derivative = phx.optim.conic_primal_jvp(
        sensitivity,
        _tangent(problem, linear=jnp.ones(1)),
    )

    assert derivative.status == int(phx.optim.ConicSensitivityStatus.AMBIGUOUS_ACTIVE_SET)
    assert derivative.active_set.roles[0] == int(phx.optim.ConicConstraintRole.AMBIGUOUS)
    assert not derivative.available
    assert jnp.isnan(derivative.value[0])
    problem = phx.optim.ConicProgram(
        None,
        jnp.asarray([-1.0]),
        jnp.empty((0, 1)),
        jnp.empty((0,)),
        phx.optim.ProductCone(()),
        problem_id="regularized-linear-conic-sensitivity",
    )
    _, execution, sensitivity = _prepare(
        problem,
        solution=(
            jnp.asarray([0.5]),
            jnp.empty((0,)),
            jnp.empty((0,)),
            jnp.zeros(1),
            jnp.zeros(1),
        ),
        regularization=2.0,
    )
    np.testing.assert_allclose(execution.result.primal, jnp.asarray([0.5]), atol=2e-7)

    derivative = phx.optim.conic_primal_jvp(
        sensitivity,
        _tangent(problem, linear=jnp.ones(1)),
    )
    np.testing.assert_allclose(derivative.value, jnp.asarray([-0.5]), atol=2e-7)
    adjoint = phx.optim.conic_primal_vjp(sensitivity, jnp.ones(1))
    assert adjoint.value.quadratic is None
    np.testing.assert_allclose(adjoint.value.linear, jnp.asarray([-0.5]), atol=2e-7)
    solution = (
        jnp.ones(1),
        jnp.zeros(1),
        jnp.ones(1),
        jnp.zeros(1),
        jnp.zeros(1),
    )
    problem = phx.optim.ConicProgram(
        jnp.ones((1, 1)),
        jnp.asarray([-2.0]),
        jnp.ones((1, 1)),
        jnp.ones(1),
        phx.optim.NonnegativeCone(1),
        problem_id="independent-conic-binding",
    )
    first, execution, _ = _prepare(problem, solution=solution)
    second, _, _ = _prepare(problem, solution=solution)
    assert first.numeric_binding_id != second.numeric_binding_id

    with pytest.raises(ValueError, match="numeric binding"):
        phx.optim.prepare_conic_sensitivity(second, execution)


def test_prepared_sensitivity_rejects_stale_numeric_execution() -> None:
    def problem(rhs: Any) -> Any:
        return phx.optim.ConicProgram(
            jnp.ones((1, 1)),
            jnp.asarray([-2.0]),
            jnp.ones((1, 1)),
            jnp.asarray([rhs]),
            phx.optim.NonnegativeCone(1),
            problem_id="stale-conic-sensitivity",
        )

    prepared, execution, _ = _prepare(
        problem(1.0),
        solution=(
            jnp.ones(1),
            jnp.zeros(1),
            jnp.ones(1),
            jnp.zeros(1),
            jnp.zeros(1),
        ),
    )
    refreshed = phx.optim.refresh_convex_program(prepared, problem(1.5))

    with pytest.raises(ValueError, match="numeric version"):
        phx.optim.prepare_conic_sensitivity(refreshed, execution)


def test_conic_sensitivity_scenario_4() -> None:
    problem = phx.optim.ConicProgram(
        jnp.ones((1, 1)),
        jnp.asarray([-2.0]),
        jnp.ones((1, 1)),
        jnp.ones(1),
        phx.optim.NonnegativeCone(1),
        problem_id="projection-point-consistency",
    )
    prepared, execution, _ = _prepare(
        problem,
        solution=(
            jnp.ones(1),
            jnp.zeros(1),
            jnp.ones(1),
            jnp.zeros(1),
            jnp.zeros(1),
        ),
    )
    inconsistent = eqx.tree_at(
        lambda value: value.result.cone_slack,
        execution,
        jnp.ones(1),
    )

    sensitivity = phx.optim.prepare_conic_sensitivity(prepared, inconsistent)
    # The audited slack no longer satisfies Ax + s = b, so the witness is not a
    # KKT point in original coordinates and no derivative may be published.
    assert sensitivity.active_set.status == int(
        phx.optim.ConicSensitivityStatus.FORWARD_FAILED
    )
    np.testing.assert_allclose(sensitivity.active_set.primal_residual_norm, 1.0)
    problem = phx.optim.ConicProgram(
        jnp.ones((1, 1)),
        jnp.asarray([-2.0]),
        jnp.ones((1, 1)),
        jnp.ones(1),
        phx.optim.NonnegativeCone(1),
        problem_id="damped-derivative-rejection",
    )
    damped = phx.linalg.LinearSolvePolicy(phx.linalg.DenseSVD(damping=1e-3))

    with pytest.raises(ValueError, match="zero derivative-solver damping"):
        _prepare(
            problem,
            solution=(
                jnp.ones(1),
                jnp.zeros(1),
                jnp.ones(1),
                jnp.zeros(1),
                jnp.zeros(1),
            ),
            linear=damped,
        )
    psd = phx.optim.PositiveSemidefiniteCone(2)
    cases = (
        (
            psd,
            psd.pack(jnp.asarray([[1.0, 2.0], [2.0, -1.0]])),
            psd.pack(jnp.asarray([[0.2, -0.1], [-0.1, 0.3]])),
        ),
        (
            phx.optim.ExponentialCone(),
            jnp.asarray([1.0, 2.0, 3.0]),
            jnp.asarray([0.2, -0.1, 0.3]),
        ),
        (
            phx.optim.PowerCone(0.4),
            jnp.asarray([-1.0, 2.0, 1.0]),
            jnp.asarray([0.2, -0.1, 0.3]),
        ),
    )
    cotangent = jnp.asarray([0.7, -0.4, 0.2])
    for index, (cone, value, direction) in enumerate(cases):
        problem = phx.optim.ConicProgram(
            jnp.eye(cone.dimension),
            -value,
            -jnp.eye(cone.dimension),
            jnp.zeros(cone.dimension),
            cone,
            problem_id=f"advanced-cone-sensitivity-{index}",
        )
        primal = cone.project(value)
        _, _, sensitivity = _prepare(
            problem,
            solution=(
                primal,
                primal,
                primal - value,
                jnp.zeros(cone.dimension),
                jnp.zeros(cone.dimension),
            ),
        )
        tangent = _tangent(problem, linear=-direction)

        derivative = phx.optim.conic_primal_jvp(sensitivity, tangent)
        expected = jax.jvp(cone.project, (value,), (direction,))[1]
        np.testing.assert_allclose(
            derivative.value,
            expected,
            atol=5e-6,
            rtol=5e-6,
        )
        assert derivative.available
        adjoint = phx.optim.conic_primal_vjp(
            sensitivity,
            cotangent[: cone.dimension],
        )
        np.testing.assert_allclose(
            jnp.vdot(cotangent[: cone.dimension], derivative.value),
            _data_pairing(adjoint.value, tangent),
            atol=5e-6,
            rtol=5e-6,
        )
    psd = phx.optim.PositiveSemidefiniteCone(2)
    cases = (
        (psd, psd.pack(jnp.diag(jnp.asarray([1.0, 0.0])))),
        (phx.optim.ExponentialCone(), jnp.zeros(3)),
        (phx.optim.PowerCone(0.4), jnp.zeros(3)),
    )
    for index, (cone, value) in enumerate(cases):
        problem = phx.optim.ConicProgram(
            jnp.eye(cone.dimension),
            -value,
            -jnp.eye(cone.dimension),
            jnp.zeros(cone.dimension),
            cone,
            problem_id=f"advanced-cone-boundary-{index}",
        )
        primal = cone.project(value)
        _, _, sensitivity = _prepare(
            problem,
            solution=(
                primal,
                primal,
                primal - value,
                jnp.zeros(cone.dimension),
                jnp.zeros(cone.dimension),
            ),
            regularity_tolerance=1e-7,
        )
        derivative = phx.optim.conic_primal_jvp(
            sensitivity,
            _tangent(problem, linear=jnp.ones(cone.dimension)),
        )

        assert derivative.status == int(
            phx.optim.ConicSensitivityStatus.AMBIGUOUS_ACTIVE_SET
        )
        assert not derivative.available
        assert jnp.all(jnp.isnan(derivative.value))


def test_matrix_free_sensitivity_requires_matching_verified_stability() -> None:
    program = phx.optim.ConicProgram(
        jnp.asarray([[1.0]]),
        jnp.asarray([-1.0]),
        jnp.asarray([[1.0]]),
        jnp.asarray([1.0]),
        phx.optim.ZeroCone(1),
    )
    solution = (
        jnp.asarray([1.0]),
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
    )
    linear = phx.linalg.LinearSolvePolicy(phx.linalg.LSMR())

    def stability(operator: Any) -> Any:
        return phx.linalg.StabilityLowerBound(operator, 0.25, evidence="verified")

    _, _, prepared = _prepare(
        program,
        solution=solution,
        linear=linear,
        representation="matrix-free",
        stability=stability,
    )
    tangent = _tangent(program, rhs=jnp.asarray([1.0]))
    result = phx.optim.conic_primal_jvp(prepared, tangent)
    assert result.representation == "matrix-free"
    assert result.available
    assert jnp.allclose(result.value, jnp.asarray([1.0]), atol=1e-5)


_STATUS = phx.optim.ConicSensitivityStatus
_ROLE = phx.optim.ConicConstraintRole


def _native_policy(*, absolute: float = 1e-10, maximum_steps: int = 100) -> Any:
    return phx.optim.ConvexSolvePolicy(
        phx.optim.NativeHomogeneousConic(),
        termination=phx.optim.ConvexTermination(
            absolute=absolute, maximum_steps=maximum_steps
        ),
        failure=phx.linalg.FailurePolicy("status"),
    )


def _tight_lsmr() -> Any:
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.LSMR(),
        tolerance=phx.linalg.TolerancePolicy(
            relative=1e-13, absolute=1e-14, max_steps=4000
        ),
    )


def _verified_stability(jacobian: np.ndarray) -> Any:
    # Independent oracle: the smallest singular value of the analytically
    # assembled projection-KKT Jacobian, discounted for floating-point safety.
    sigma = float(np.linalg.svd(jacobian, compute_uv=False).min())

    def stability(operator: Any) -> Any:
        return phx.linalg.StabilityLowerBound(operator, 0.5 * sigma, evidence="verified")

    return stability


# Sparse simplex QP per block: min 1/2 |x|^2 + q.x with x1 + x2 + x3 = 1, x >= 0.
# For q = (0, 0.2, 2) the solution is x = (0.6, 0.4, 0) with equality multiplier
# -0.6 and a strictly positive multiplier 1.4 on x3 >= 0.
_SIMPLEX_LINEAR = np.asarray([0.0, 0.2, 2.0])


def _simplex_triplets(blocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = 3 * blocks
    columns = np.concatenate((np.arange(size), np.arange(size)))
    rows = np.concatenate((np.repeat(np.arange(blocks), 3), blocks + np.arange(size)))
    values = np.concatenate((np.ones(size), -np.ones(size)))
    return columns, rows, values


def _simplex_program(
    blocks: int, linear: Any, rhs: Any, coefficients: Any, diagonal: Any
) -> Any:
    size = 3 * blocks
    variables = phx.linalg.ArraySpace(
        (size,), dtype=jnp.float64, space_id=f"simplex-qp:x:{blocks}"
    )
    constraints = phx.linalg.ArraySpace(
        (blocks + size,), dtype=jnp.float64, space_id=f"simplex-qp:c:{blocks}"
    )
    columns, rows, _ = _simplex_triplets(blocks)
    matrix = phx.sparse.SparseCoordinateOperator(
        phx.sparse.EdgeRelation(
            jnp.asarray(columns, dtype=jnp.int32),
            jnp.asarray(rows, dtype=jnp.int32),
            source_size=size,
            target_size=blocks + size,
        ),
        jnp.asarray(coefficients, dtype=jnp.float64),
        source=variables,
        target=constraints,
    )
    quadratic = phx.sparse.SparseCoordinateOperator(
        phx.sparse.EdgeRelation(
            jnp.arange(size, dtype=jnp.int32),
            jnp.arange(size, dtype=jnp.int32),
            source_size=size,
            target_size=size,
        ),
        jnp.asarray(diagonal, dtype=jnp.float64),
        source=variables,
        target=variables,
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    return phx.optim.ConicProgram(
        quadratic,
        jnp.asarray(linear, dtype=jnp.float64),
        matrix,
        jnp.asarray(rhs, dtype=jnp.float64),
        phx.optim.ProductCone(
            (phx.optim.ZeroCone(blocks), phx.optim.NonnegativeCone(size))
        ),
        problem_id=f"sparse-simplex-qp:{blocks}",
        convexity_evidence="construction",
    )


def _simplex_kkt_jacobian(blocks: int) -> np.ndarray:
    size = 3 * blocks
    columns, rows, values = _simplex_triplets(blocks)
    matrix = np.zeros((blocks + size, size))
    matrix[rows, columns] = values
    # Zero-cone rows and the strictly active x3 >= 0 rows have unit
    # dual-projection derivative; strictly inactive orthant rows have zero.
    selected = np.concatenate((np.ones(blocks), np.tile([0.0, 0.0, 1.0], blocks)))
    return np.block(
        [
            [np.eye(size), matrix.T],
            [-selected[:, None] * matrix, np.diag(1.0 - selected)],
        ]
    )


def test_sparse_nonnegative_qp_fixed_active_jvp_vjp_match_central_differences() -> None:
    blocks = 16
    size = 3 * blocks
    linear = np.tile(_SIMPLEX_LINEAR, blocks)
    rhs = np.concatenate((np.ones(blocks), np.zeros(size)))
    _, _, coefficients = _simplex_triplets(blocks)
    diagonal = np.ones(size)
    rng = np.random.default_rng(13)
    direction_linear = rng.normal(size=size)
    direction_rhs = np.concatenate((rng.normal(size=blocks), np.zeros(size)))
    direction_coefficients = np.concatenate((rng.normal(size=size), np.zeros(size)))
    policy = _native_policy()
    program = _simplex_program(blocks, linear, rhs, coefficients, diagonal)
    prepared = phx.optim.prepare_convex_program(program, policy)
    execution = phx.optim.solve_convex_program(prepared)
    assert bool(execution.result.successful)
    np.testing.assert_allclose(
        execution.result.primal, np.tile([0.6, 0.4, 0.0], blocks), atol=1e-8
    )
    sensitivity = phx.optim.prepare_conic_sensitivity(
        prepared,
        execution,
        linear=_tight_lsmr(),
        representation="matrix-free",
        stability=_verified_stability(_simplex_kkt_jacobian(blocks)),
    )
    evidence = sensitivity.active_set
    assert int(evidence.status) == int(_STATUS.REGULAR_FIXED_ACTIVE)
    expected_roles = np.concatenate(
        (
            np.full(blocks, int(_ROLE.EQUALITY)),
            np.tile(
                [int(_ROLE.INACTIVE), int(_ROLE.INACTIVE), int(_ROLE.ACTIVE)], blocks
            ),
        )
    )
    np.testing.assert_array_equal(evidence.roles, expected_roles)
    assert float(evidence.strict_complementarity_margin) > 0.3
    assert float(evidence.dual_residual_norm) <= float(evidence.kkt_tolerance)

    tangent = phx.optim.ConicProgramData.zeros_like(program)
    tangent = eqx.tree_at(
        lambda data: data.linear, tangent, jnp.asarray(direction_linear)
    )
    tangent = eqx.tree_at(
        lambda data: data.constraint_rhs, tangent, jnp.asarray(direction_rhs)
    )
    tangent = eqx.tree_at(
        lambda data: data.constraint_matrix.coefficients,
        tangent,
        jnp.asarray(direction_coefficients),
    )
    derivative = phx.optim.conic_primal_jvp(sensitivity, tangent)
    assert bool(derivative.available)
    assert int(derivative.status) == int(_STATUS.REGULAR_FIXED_ACTIVE)

    step = 1e-3

    def solved(sign: float) -> np.ndarray:
        perturbed = _simplex_program(
            blocks,
            linear + sign * step * direction_linear,
            rhs + sign * step * direction_rhs,
            coefficients + sign * step * direction_coefficients,
            diagonal,
        )
        result = phx.optim.solve_conic_program(perturbed, policy=policy)
        assert bool(result.successful)
        return np.asarray(result.primal)

    central = (solved(1.0) - solved(-1.0)) / (2.0 * step)
    np.testing.assert_allclose(derivative.value, central, atol=2e-5)
    # Inactive-multiplier variables stay on the simplex face; x3 stays at zero.
    np.testing.assert_allclose(np.asarray(derivative.value)[2::3], 0.0, atol=1e-9)

    cotangent = rng.normal(size=size)
    adjoint = phx.optim.conic_primal_vjp(sensitivity, jnp.asarray(cotangent))
    assert bool(adjoint.available)
    pairing = (
        np.vdot(adjoint.value.linear, direction_linear)
        + np.vdot(adjoint.value.constraint_rhs, direction_rhs)
        + np.vdot(adjoint.value.constraint_matrix.coefficients, direction_coefficients)
    )
    np.testing.assert_allclose(
        pairing, np.vdot(cotangent, derivative.value), rtol=1e-7, atol=1e-8
    )


def test_weak_complementarity_has_no_default_derivative_but_selected_policy_does() -> (
    None
):
    # min 1/2 x^2 subject to x >= 0 at q = 0: x = 0 and the multiplier is zero.
    program = phx.optim.ConicProgram(
        jnp.asarray([[1.0]]),
        jnp.asarray([0.0]),
        jnp.asarray([[-1.0]]),
        jnp.asarray([0.0]),
        phx.optim.NonnegativeCone(1),
        problem_id="weak-orthant-selection",
    )
    solution = (jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), jnp.zeros(1))
    selection = 0.5
    # Selected dual-projection derivative theta gives J = [[1, -1], [theta, 1 - theta]].
    stability = _verified_stability(
        np.asarray([[1.0, -1.0], [selection, 1.0 - selection]])
    )
    tangent = _tangent(program, linear=jnp.ones(1))
    _, _, default = _prepare(
        program,
        solution=solution,
        linear=_tight_lsmr(),
        representation="matrix-free",
        stability=_verified_stability(np.asarray([[1.0, -1.0], [0.0, 1.0]])),
    )
    refused = phx.optim.conic_primal_jvp(default, tangent)
    assert int(refused.status) == int(_STATUS.AMBIGUOUS_ACTIVE_SET)
    assert not bool(refused.available)
    assert jnp.isnan(refused.value[0])

    _, _, selected = _prepare(
        program,
        solution=solution,
        linear=_tight_lsmr(),
        representation="matrix-free",
        stability=stability,
        generalized=phx.optim.ConicGeneralizedDerivativePolicy(
            orthant_zero_value=selection
        ),
    )
    published = phx.optim.conic_primal_jvp(selected, tangent)
    # One-sided derivatives are -1 (q < 0) and 0 (q > 0); theta selects
    # the convex combination -(1 - theta).
    assert int(published.status) == int(_STATUS.AMBIGUOUS_ACTIVE_SET)
    assert bool(published.available)
    assert published.generalized_selection != "smooth"
    np.testing.assert_allclose(published.value, [-(1.0 - selection)], atol=1e-9)


def _orthant_program(center: float, problem_id: str = "orthant-crossing") -> Any:
    # min 1/2 x^2 - center x subject to x >= 0, solution max(center, 0).
    return phx.optim.ConicProgram(
        jnp.asarray([[1.0]]),
        jnp.asarray([-center]),
        jnp.asarray([[-1.0]]),
        jnp.asarray([0.0]),
        phx.optim.NonnegativeCone(1),
        problem_id=problem_id,
    )


def test_fixed_active_set_crossing_reports_change_without_derivative() -> None:
    policy = _native_policy(absolute=1e-10)
    prepared = phx.optim.prepare_convex_program(_orthant_program(1.0), policy)
    reference = phx.optim.prepare_conic_sensitivity(
        prepared, phx.optim.solve_convex_program(prepared)
    )
    assert int(reference.active_set.status) == int(_STATUS.REGULAR_FIXED_ACTIVE)
    assert int(reference.active_set.roles[0]) == int(_ROLE.INACTIVE)
    tangent = _tangent(_orthant_program(1.0), linear=jnp.ones(1))

    nearby = phx.optim.refresh_convex_program(prepared, _orthant_program(1.25))
    same = phx.optim.prepare_conic_sensitivity(
        nearby,
        phx.optim.solve_convex_program(nearby),
        fixed_active_set=reference.active_set,
    )
    derivative = phx.optim.conic_primal_jvp(same, tangent)
    assert int(derivative.status) == int(_STATUS.REGULAR_FIXED_ACTIVE)
    np.testing.assert_allclose(derivative.value, [-1.0], atol=1e-7)

    crossed = phx.optim.refresh_convex_program(prepared, _orthant_program(-1.0))
    changed = phx.optim.prepare_conic_sensitivity(
        crossed,
        phx.optim.solve_convex_program(crossed),
        fixed_active_set=reference.active_set,
    )
    assert int(changed.active_set.roles[0]) == int(_ROLE.ACTIVE)
    derivative = phx.optim.conic_primal_jvp(changed, tangent)
    assert int(derivative.status) == int(_STATUS.ACTIVE_SET_CHANGED)
    assert not bool(derivative.available)
    assert jnp.isnan(derivative.value[0])
    adjoint = phx.optim.conic_primal_vjp(changed, jnp.ones(1))
    assert int(adjoint.status) == int(_STATUS.ACTIVE_SET_CHANGED)
    assert jnp.isnan(adjoint.value.linear[0])

    other = phx.optim.prepare_convex_program(
        _orthant_program(1.0, "other-orthant-structure"), policy
    )
    with pytest.raises(ValueError, match="different conic program structure"):
        phx.optim.prepare_conic_sensitivity(
            other,
            phx.optim.solve_convex_program(other),
            fixed_active_set=reference.active_set,
        )


def test_dense_degenerate_active_constraints_report_singular_kkt() -> None:
    # min 1/2 (x - 2)^2 subject to x <= 1 twice: multipliers split 0.5/0.5 are
    # strictly complementary but violate LICQ, so the KKT Jacobian is singular.
    program = phx.optim.ConicProgram(
        jnp.asarray([[1.0]]),
        jnp.asarray([-2.0]),
        jnp.asarray([[1.0], [1.0]]),
        jnp.asarray([1.0, 1.0]),
        phx.optim.NonnegativeCone(2),
        problem_id="duplicate-active-constraint",
    )
    _, _, sensitivity = _prepare(
        program,
        solution=(
            jnp.ones(1),
            jnp.zeros(2),
            jnp.asarray([0.5, 0.5]),
            jnp.zeros(1),
            jnp.zeros(1),
        ),
    )
    assert int(sensitivity.active_set.status) == int(_STATUS.REGULAR_FIXED_ACTIVE)
    # Moving one duplicate side is a genuine kink of x = min(2, b1, b2).
    derivative = phx.optim.conic_primal_jvp(
        sensitivity, _tangent(program, rhs=jnp.asarray([1.0, 0.0]))
    )
    assert int(derivative.status) == int(_STATUS.SINGULAR_KKT)
    assert not bool(derivative.available)
    assert int(derivative.linear_status) != int(phx.linalg.LinearSolveStatus.SUCCESS)
    assert jnp.isnan(derivative.value[0])
    adjoint = phx.optim.conic_primal_vjp(sensitivity, jnp.ones(1))
    assert int(adjoint.status) == int(_STATUS.SINGULAR_KKT)
    assert jnp.isnan(adjoint.value.constraint_rhs[0])


def test_matrix_free_negligible_stability_constant_reports_singular_kkt() -> None:
    program = phx.optim.ConicProgram(
        jnp.asarray([[1.0]]),
        jnp.asarray([-2.0]),
        jnp.asarray([[1.0], [1.0]]),
        jnp.asarray([1.0, 1.0]),
        phx.optim.NonnegativeCone(2),
        problem_id="duplicate-active-constraint-matrix-free",
    )
    # [[P, A^T], [-A, 0]] with two identical active rows is exactly singular; its
    # numerically computed smallest singular value is a negligible positive number.
    jacobian = np.asarray([[1.0, 1.0, 1.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    _, _, sensitivity = _prepare(
        program,
        solution=(
            jnp.ones(1),
            jnp.zeros(2),
            jnp.asarray([0.5, 0.5]),
            jnp.zeros(1),
            jnp.zeros(1),
        ),
        linear=_tight_lsmr(),
        representation="matrix-free",
        stability=_verified_stability(jacobian),
    )
    derivative = phx.optim.conic_primal_jvp(
        sensitivity, _tangent(program, rhs=jnp.asarray([1.0, 0.0]))
    )
    assert int(derivative.status) == int(_STATUS.SINGULAR_KKT)
    assert not bool(derivative.available)
    assert jnp.isnan(derivative.value[0])


def test_failed_forward_execution_reports_forward_failed() -> None:
    policy = _native_policy(absolute=1e-12, maximum_steps=1)
    prepared = phx.optim.prepare_convex_program(_orthant_program(1.0), policy)
    execution = phx.optim.solve_convex_program(prepared)
    assert not bool(execution.result.successful)
    sensitivity = phx.optim.prepare_conic_sensitivity(prepared, execution)
    assert int(sensitivity.active_set.status) == int(_STATUS.FORWARD_FAILED)
    derivative = phx.optim.conic_primal_jvp(
        sensitivity, _tangent(_orthant_program(1.0), linear=jnp.ones(1))
    )
    assert int(derivative.status) == int(_STATUS.FORWARD_FAILED)
    assert not bool(derivative.available)
    assert jnp.isnan(derivative.value[0])


def test_interior_point_witness_with_small_active_multiplier_is_regular() -> None:
    # min 1/2 x^2 + 5e-3 x subject to x >= 0: x* = 0 with multiplier 5e-3. An
    # interior-point witness stops on s z <= tau, leaving s = x = 1e-8, so the
    # projection-KKT residual min(s, z) = 1e-8 exceeds tau = 1e-9 while the audited
    # complementarity 5e-11 does not; the strict margin ~5e-3 fixes the roles.
    program = phx.optim.ConicProgram(
        jnp.asarray([[1.0]]),
        jnp.asarray([5e-3]),
        jnp.asarray([[-1.0]]),
        jnp.asarray([0.0]),
        phx.optim.NonnegativeCone(1),
        problem_id="small-active-multiplier",
    )
    slack = 1e-8
    _, execution, sensitivity = _prepare(
        program,
        solution=(
            jnp.asarray([slack]),
            jnp.asarray([slack]),
            jnp.asarray([5e-3 + slack]),
            jnp.zeros(1),
            jnp.zeros(1),
        ),
    )
    assert bool(execution.result.successful)
    evidence = sensitivity.active_set
    assert float(evidence.projection_residual_norm) > 1e-9
    assert int(evidence.status) == int(_STATUS.REGULAR_FIXED_ACTIVE)
    assert int(evidence.roles[0]) == int(_ROLE.ACTIVE)
    # On the active face x stays at zero for every linear-cost perturbation.
    derivative = phx.optim.conic_primal_jvp(
        sensitivity, _tangent(program, linear=jnp.ones(1))
    )
    assert bool(derivative.available)
    np.testing.assert_allclose(derivative.value, [0.0], atol=1e-5)
