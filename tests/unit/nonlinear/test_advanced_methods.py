#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg
nl = phx.nonlinear


class _CoordinateState(NamedTuple):
    value: jax.Array


class _CoordinateResidual(NamedTuple):
    value: jax.Array


class _NoDenseDistinctJacobian(la.AbstractLinearOperator):
    def __init__(self, source: la.PyTreeSpace, target: la.PyTreeSpace):
        self.source = source
        self.target = target
        self.properties = la.OperatorProperties()
        self.capabilities = la.OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
        )
        self.batch_shape = ()
        self.operator_id = "no-dense-distinct-jacobian"

    def mv(self, vector):
        value = self.source.validate(vector)
        return self.target.validate(_CoordinateResidual(2.0 * value.value))

    def transpose_mv(self, vector):
        value = self.target.validate(vector)
        return self.source.validate(_CoordinateState(2.0 * value.value))

    def adjoint_mv(self, vector):
        return self.transpose_mv(vector)

    def _materialize(self):
        raise AssertionError("The distinct-space Jacobian must remain matrix-free.")

    def to_dense(self):
        raise AssertionError("The distinct-space Jacobian must remain matrix-free.")


def test_advanced_nonlinear_public_exports_are_available():
    names = {
        "AbstractNonlinearSystemTransformation",
        "Bounds",
        "FASCyclePolicy",
        "FASHierarchy",
        "FASLevel",
        "FunctionLeftNonlinearPreconditioner",
        "FunctionRightNonlinearPreconditioner",
        "NonlinearGMRES",
        "NonlinearTransformationEvidence",
        "SemismoothNewton",
        "VariationalInequalityProblem",
        "complementarity_certificate",
        "fas_cycle",
        "left_precondition",
        "implicit_root",
        "right_precondition",
    }

    assert names <= set(nl.__all__)
    assert all(hasattr(nl, name) for name in names)


@pytest.mark.parametrize("linear_solver", ("matrix-free", "dense-lu"))
def test_implicit_root_matches_analytic_primal_forward_and_reverse_sensitivities(
    linear_solver,
):
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state * state - target,
        problem_id="implicit-positive-square-root",
    )
    initial = jnp.ones((3,), dtype=jnp.float64)
    target = jnp.asarray([0.5, 1.0, 2.0], dtype=jnp.float64)
    direction = jnp.asarray([0.2, -0.3, 0.4], dtype=jnp.float64)
    termination = nl.NonlinearTermination(
        absolute_residual=1e-11,
        relative_residual=1e-11,
        maximum_steps=20,
        maximum_evaluations=40,
    )
    method = (
        nl.NewtonKrylov()
        if linear_solver == "matrix-free"
        else nl.NewtonKrylov(linear_policy=la.LinearSolvePolicy(la.DenseLU()))
    )

    def solution(expected):
        return nl.implicit_root(
            problem,
            initial,
            method=method,
            termination=termination,
            args=expected,
        )

    root, tangent = jax.jvp(solution, (target,), (direction,))
    gradient = jax.grad(lambda expected: jnp.sum(solution(expected)))(target)
    expected_root = jnp.sqrt(target)
    expected_diagonal = 0.5 / expected_root

    assert jnp.allclose(root, expected_root, rtol=1e-9, atol=1e-10)
    assert jnp.allclose(
        tangent,
        expected_diagonal * direction,
        rtol=1e-8,
        atol=1e-10,
    )
    assert jnp.allclose(gradient, expected_diagonal, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("globalization", ("line-search", "trust-region"))
def test_newton_rebases_distinct_pytree_spaces_without_dense_fallback(
    globalization,
):
    problem = nl.NonlinearSystemProblem(
        lambda state, args: _CoordinateResidual(2.0 * state.value - 4.0),
        problem_id=f"distinct-space-{globalization}",
    )

    def jacobian(state, args):
        del args
        source = la.PyTreeSpace(state)
        target = la.PyTreeSpace(_CoordinateResidual(jnp.zeros_like(state.value)))
        return _NoDenseDistinctJacobian(source, target)

    jacobian_policy = nl.JacobianPolicy("explicit", operator=jacobian)
    if globalization == "line-search":
        method = nl.NewtonKrylov(jacobian_policy=jacobian_policy)
    else:
        method = nl.NewtonTrustRegion(jacobian_policy=jacobian_policy)
    termination = nl.NonlinearTermination(
        absolute_residual=1e-10,
        relative_residual=1e-10,
        maximum_steps=8,
    )
    initial = jnp.asarray([0.0])

    result = method.solve(
        problem,
        _CoordinateState(initial),
        termination=termination,
    )
    jitted_state = jax.jit(
        lambda value: (
            method.solve(
                problem,
                _CoordinateState(value),
                termination=termination,
            ).state.value
        )
    )(initial)

    assert bool(result.successful)
    assert result.transformation_evidence is None
    assert isinstance(result.state, _CoordinateState)
    assert isinstance(result.residual, _CoordinateResidual)
    assert jnp.allclose(result.state.value, jnp.asarray([2.0]))
    assert jnp.allclose(jitted_state, jnp.asarray([2.0]))
    assert method.linear_policy.method.name == "gmres"
    assert result.provenance.derivative_id == "explicit"
    assert result.provenance.linear_plan_id
    assert result.provenance.notes == ("linear-method=gmres;linear-backend=native-krylov")


def test_newton_still_rejects_unequal_coordinate_dimensions():
    problem = nl.NonlinearSystemProblem(
        lambda state, args: _CoordinateResidual(
            jnp.concatenate((state.value, state.value))
        )
    )

    with pytest.raises(ValueError, match="square Jacobian coordinate map"):
        nl.NewtonKrylov().solve(
            problem,
            _CoordinateState(jnp.asarray([1.0])),
            termination=nl.NonlinearTermination(maximum_steps=2),
        )


def test_nonlinear_system_explicit_spaces_validate_and_missing_spaces_bind():
    state_space = la.ArraySpace((1,), dtype=jnp.float64)
    residual_space = la.ArraySpace((2,), dtype=jnp.float64)
    problem = nl.NonlinearSystemProblem(
        lambda state, args: jnp.concatenate((state, state)),
        state_space=state_space,
        residual_space=residual_space,
    )

    assert jnp.allclose(problem.residual(jnp.ones((1,))), jnp.ones((2,)))
    with pytest.raises(ValueError, match="shape"):
        problem.evaluate(jnp.ones((2,)))
    with pytest.raises(ValueError, match="shape"):
        nl.NonlinearSystemProblem(
            lambda state, args: state,
            state_space=state_space,
            residual_space=residual_space,
        ).evaluate(jnp.ones((1,)))

    calls = 0

    def residual(state, args):
        nonlocal calls
        calls += 1
        return jnp.concatenate((state, -state))

    inferred = nl.NonlinearSystemProblem(residual)
    bound = inferred.bind_spaces(
        jnp.ones((1,), dtype=jnp.float64),
        jnp.ones((2,), dtype=jnp.float64),
    )
    assert calls == 0

    assert isinstance(bound.state_space, la.PyTreeSpace)
    assert isinstance(bound.residual_space, la.PyTreeSpace)
    assert jnp.allclose(bound.residual(jnp.ones((1,))), jnp.asarray([1.0, -1.0]))
    assert calls == 1


def test_left_and_right_preconditioned_roots_expose_physical_results():
    physical_space = la.ArraySpace((1,), dtype=jnp.float64)
    residual_space = la.ArraySpace((1,), dtype=jnp.float64)
    latent_space = la.ArraySpace((1,), dtype=jnp.float64)
    problem = nl.NonlinearSystemProblem(
        lambda state, args: (state - args, state + 10.0),
        state_space=physical_space,
        residual_space=residual_space,
        has_aux=True,
        problem_id="shifted-root",
    )
    left = nl.left_precondition(
        problem,
        nl.FunctionLeftNonlinearPreconditioner(
            lambda state, residual, args: residual / (1.0 + state**2),
            state_space=physical_space,
            source=residual_space,
            target=residual_space,
            preconditioner_id="diagonal-left",
        ),
    )
    right = nl.right_precondition(
        problem,
        nl.FunctionRightNonlinearPreconditioner(
            lambda latent, args: 2.0 * latent,
            source=latent_space,
            target=physical_space,
            preconditioner_id="doubling-right",
        ),
    )

    termination = nl.NonlinearTermination(
        absolute_residual=1e-10,
        relative_residual=1e-10,
        maximum_steps=10,
    )
    left_result = nl.root(
        left,
        jnp.asarray([1.0]),
        termination=termination,
        args=jnp.asarray([2.0]),
    )
    right_result = nl.root(
        right,
        jnp.asarray([0.0]),
        termination=termination,
        args=jnp.asarray([2.0]),
    )

    assert bool(left_result.successful)
    assert jnp.allclose(left_result.state, jnp.asarray([2.0]))
    assert jnp.allclose(left_result.residual, 0.0)
    assert jnp.allclose(left_result.auxiliary, jnp.asarray([12.0]))
    assert left_result.provenance.problem_id == problem.problem_id
    assert isinstance(
        left_result.transformation_evidence, nl.NonlinearTransformationEvidence
    )
    assert jnp.allclose(left_result.transformation_evidence.state, jnp.asarray([2.0]))
    assert jnp.allclose(left_result.transformation_evidence.residual, 0.0)
    assert jnp.allclose(
        left_result.transformation_evidence.auxiliary, jnp.asarray([12.0])
    )

    assert bool(right_result.successful)
    assert jnp.allclose(right_result.state, jnp.asarray([2.0]))
    assert jnp.allclose(right_result.residual, 0.0)
    assert jnp.allclose(right_result.auxiliary, jnp.asarray([12.0]))
    assert right_result.provenance.problem_id == problem.problem_id
    assert isinstance(
        right_result.transformation_evidence, nl.NonlinearTransformationEvidence
    )
    assert jnp.allclose(right_result.transformation_evidence.state, jnp.asarray([1.0]))
    assert jnp.allclose(right_result.transformation_evidence.residual, 0.0)
    assert jnp.allclose(
        right_result.transformation_evidence.auxiliary, jnp.asarray([12.0])
    )

    with pytest.raises(ValueError, match="shape"):
        right.reconstruct(jnp.ones((2,), dtype=jnp.float64))
    invalid_target = nl.FunctionRightNonlinearPreconditioner(
        lambda latent, args: jnp.concatenate((latent, latent)),
        source=latent_space,
        target=physical_space,
    )
    with pytest.raises(ValueError, match="shape"):
        invalid_target.reconstruct(jnp.ones((1,), dtype=jnp.float64))


def test_zeroing_left_preconditioner_cannot_certify_a_false_physical_root():
    space = la.ArraySpace((1,), dtype=jnp.float64)
    problem = nl.NonlinearSystemProblem(
        lambda state, args: state - 2.0,
        state_space=space,
        residual_space=space,
        problem_id="nonzero-physical-residual",
    )
    transformed = nl.left_precondition(
        problem,
        nl.FunctionLeftNonlinearPreconditioner(
            lambda state, residual, args: jnp.zeros_like(residual),
            state_space=space,
            source=space,
            target=space,
            preconditioner_id="zero-residual",
        ),
    )

    result = nl.root(
        transformed,
        jnp.asarray([0.0]),
        termination=nl.NonlinearTermination(maximum_steps=2),
    )

    assert not bool(result.successful)
    assert int(result.status) == int(
        nl.NonlinearStatus.TRANSFORMATION_CERTIFICATION_FAILED
    )
    assert jnp.allclose(result.state, jnp.asarray([0.0]))
    assert jnp.allclose(result.residual, jnp.asarray([-2.0]))
    assert result.transformation_evidence is not None
    assert jnp.allclose(result.transformation_evidence.residual, 0.0)


def test_left_preconditioner_validates_state_source_and_target_spaces():
    scalar = la.ArraySpace((1,), dtype=jnp.float64)
    pair = la.ArraySpace((2,), dtype=jnp.float64)
    preconditioner = nl.FunctionLeftNonlinearPreconditioner(
        lambda state, residual, args: jnp.concatenate((residual, residual)),
        state_space=scalar,
        source=scalar,
        target=pair,
    )
    bad_target = nl.FunctionLeftNonlinearPreconditioner(
        lambda state, residual, args: jnp.concatenate((residual, residual)),
        state_space=scalar,
        source=scalar,
        target=scalar,
    )

    assert jnp.allclose(
        preconditioner.apply(jnp.ones((1,)), jnp.ones((1,))),
        jnp.ones((2,)),
    )
    with pytest.raises(ValueError, match="shape"):
        preconditioner.apply(jnp.ones((2,)), jnp.ones((1,)))
    with pytest.raises(ValueError, match="shape"):
        preconditioner.apply(jnp.ones((1,)), jnp.ones((2,)))
    with pytest.raises(ValueError, match="shape"):
        bad_target.apply(jnp.ones((1,)), jnp.ones((1,)))


def test_nonlinear_gmres_rejects_a_harmful_affine_combination_and_restarts():
    def residual(state, args):
        del args
        return 1.0 - 0.1 * (state - 1.0) + 10.0 * jnp.maximum(state - 2.0, 0.0) ** 2

    method = nl.NonlinearGMRES(
        lambda state, args: state + 1.0,
        history=2,
        safeguard_factor=1.0,
    )
    result = method.solve(
        nl.NonlinearSystemProblem(residual, problem_id="harmful-secant"),
        jnp.asarray([0.0]),
        termination=nl.NonlinearTermination(
            absolute_residual=0.0,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=2,
        ),
    )

    assert jnp.allclose(result.state, jnp.asarray([2.0]))
    assert jnp.allclose(result.residual, jnp.asarray([0.9]))
    assert int(result.diagnostics.accepted_steps) == 2
    assert int(result.diagnostics.rejected_steps) == 1
    assert int(result.diagnostics.acceleration_restarts) == 1
    assert method.capabilities.nonlinear_preconditioning


def _nonlinear_fas_hierarchy():
    fine_space = la.ArraySpace((4,), dtype=jnp.float64)
    middle_space = la.ArraySpace((2,), dtype=jnp.float64)
    coarse_space = la.ArraySpace((1,), dtype=jnp.float64)

    def operator(state, args):
        del args
        return state**2

    def smoother(state, right_hand_side, args):
        del args
        return 0.5 * (state + right_hand_side / state)

    def restrict(value):
        return jnp.mean(value.reshape((-1, 2)), axis=1)

    def prolong(value):
        return jnp.repeat(value, 2)

    fine = nl.FASLevel(
        operator,
        smoother,
        state_space=fine_space,
        residual_space=fine_space,
        restrict_state=restrict,
        restrict_residual=restrict,
        prolong_correction=prolong,
        level_id="fine-square",
    )
    middle = nl.FASLevel(
        operator,
        smoother,
        state_space=middle_space,
        residual_space=middle_space,
        restrict_state=restrict,
        restrict_residual=restrict,
        prolong_correction=prolong,
        level_id="middle-square",
    )
    coarse = nl.FASLevel(
        operator,
        smoother,
        state_space=coarse_space,
        residual_space=coarse_space,
        level_id="coarse-square",
    )
    return nl.FASHierarchy(
        (fine, middle, coarse),
        lambda state, right_hand_side, args: jnp.sqrt(right_hand_side),
        hierarchy_id="square-fas",
        numeric_refreshes=3,
    )


@pytest.mark.parametrize(
    ("kind", "coarse_solves", "level_visits"),
    (
        ("v", 1, (1, 1, 1)),
        ("w", 4, (1, 2, 4)),
        ("f", 3, (1, 2, 3)),
    ),
)
def test_fas_cycles_reduce_a_nonlinear_residual_across_levels(
    kind, coarse_solves, level_visits
):
    hierarchy = _nonlinear_fas_hierarchy()
    result = nl.fas_cycle(
        hierarchy,
        jnp.full((4,), 0.5),
        right_hand_side=jnp.ones((4,)),
        policy=nl.FASCyclePolicy(kind),
    )

    assert bool(result.successful)
    assert float(result.diagnostics.final_residual_norm) < float(
        result.diagnostics.initial_residual_norm
    )
    assert jnp.allclose(result.state, jnp.ones((4,)))
    assert int(result.diagnostics.coarse_solves) == coarse_solves
    assert jnp.array_equal(result.diagnostics.level_visits, jnp.asarray(level_visits))
    assert result.provenance.problem_id == "square-fas"
    assert int(hierarchy.numeric_refreshes) == 3


def test_vi_certificate_separates_feasibility_from_complementarity():
    problem = nl.VariationalInequalityProblem(
        lambda state, args: -jnp.ones_like(state),
        nl.Bounds(0.0, 1.0),
        problem_id="upper-bound-vi",
    )
    feasible_but_wrong = nl.complementarity_certificate(problem, jnp.asarray([0.5]))
    complementary = nl.complementarity_certificate(problem, jnp.asarray([1.0]))
    solved = nl.SemismoothNewton(
        formulation="fischer-burmeister",
        certification_tolerance=1e-7,
    ).solve(
        problem,
        jnp.asarray([0.5]),
        termination=nl.NonlinearTermination(
            absolute_residual=1e-9,
            relative_residual=1e-9,
            maximum_steps=20,
        ),
    )

    assert bool(feasible_but_wrong.feasible)
    assert not bool(feasible_but_wrong.complementary)
    assert not bool(feasible_but_wrong.certified)
    assert float(feasible_but_wrong.natural_residual_norm) > 0.0
    assert float(feasible_but_wrong.fischer_burmeister_norm) > 0.0
    assert int(feasible_but_wrong.free) == 1
    assert bool(complementary.certified)
    assert int(complementary.upper_active) == 1
    assert jnp.allclose(complementary.fischer_burmeister_norm, 0.0)
    assert bool(solved.successful)
    assert bool(solved.certificate.certified)
    assert int(solved.certificate.upper_active) == 1
    assert jnp.allclose(solved.state, jnp.asarray([1.0]))
    assert int(solved.diagnostics.iterations) > 0


def test_fischer_burmeister_residual_is_finite_for_unbounded_variables():
    problem = nl.VariationalInequalityProblem(
        lambda state, args: state - 1.0,
        nl.Bounds(),
    )
    state = jnp.asarray([0.25])
    residual, tangent = jax.jvp(
        problem.fischer_burmeister_residual,
        (state,),
        (jnp.ones_like(state),),
    )

    assert jnp.allclose(residual, state - 1.0)
    assert jnp.allclose(tangent, jnp.ones_like(state))
    assert jnp.all(jnp.isfinite(tangent))


def test_semismooth_solver_cannot_report_false_success_from_a_loose_tolerance():
    problem = nl.VariationalInequalityProblem(
        lambda state, args: -jnp.ones_like(state),
        nl.Bounds(0.0, 1.0),
    )
    result = nl.SemismoothNewton(
        formulation="fischer-burmeister",
        certification_tolerance=1e-8,
    ).solve(
        problem,
        jnp.asarray([0.5]),
        termination=nl.NonlinearTermination(
            absolute_residual=10.0,
            relative_residual=0.0,
            maximum_steps=1,
            maximum_evaluations=1,
        ),
    )

    assert bool(result.certificate.feasible)
    assert not bool(result.certificate.complementary)
    assert not bool(result.successful)
    assert int(result.status) == int(nl.NonlinearStatus.RESIDUAL_STAGNATION)
    assert result.provenance.problem_id.endswith("/fischer-burmeister")


def test_generalized_derivative_policy_requires_a_clarke_unit_vector():
    with pytest.raises(ValueError, match="Clarke unit ball"):
        nl.GeneralizedDerivativePolicy(origin_coefficient=0.8)


@pytest.mark.parametrize("globalization", ("line-search", "trust-region"))
def test_newton_globalization_reports_finite_domain_rejections(globalization):
    problem = nl.NonlinearSystemProblem(
        lambda state, args: state - 2.0,
        validity=lambda state, residual, auxiliary, args: jnp.all(state <= 0.0),
        problem_id=f"{globalization}-domain",
    )
    if globalization == "line-search":
        method = nl.NewtonKrylov(line_search=nl.RootLineSearch(maximum_steps=3))
    else:
        method = nl.NewtonTrustRegion(trust_region=nl.RootTrustRegion(maximum_attempts=3))

    result = method.solve(
        problem,
        jnp.asarray([0.0]),
        termination=nl.NonlinearTermination(maximum_steps=2),
    )

    assert int(result.status) == int(nl.NonlinearStatus.RECOVERABLE_DOMAIN_FAILURE)
    assert int(result.diagnostics.domain_failures) == 3
    assert int(result.diagnostics.nonfinite_trials) == 0
    assert int(result.diagnostics.rejected_steps) == 1
    assert jnp.allclose(result.state, jnp.asarray([0.0]))
    assert jnp.all(jnp.isfinite(result.residual))


def test_newton_rejects_a_finite_but_invalid_initial_state_using_auxiliary_data():
    problem = nl.NonlinearSystemProblem(
        lambda state, args: (jnp.zeros_like(state), state < 0.0),
        has_aux=True,
        validity=lambda state, residual, auxiliary, args: jnp.all(auxiliary),
        problem_id="auxiliary-domain",
    )
    result = nl.NewtonKrylov().solve(
        problem,
        jnp.asarray([0.5]),
        termination=nl.NonlinearTermination(maximum_steps=2),
    )

    assert int(result.status) == int(nl.NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE)
    assert int(result.diagnostics.iterations) == 0
    assert int(result.diagnostics.domain_failures) == 1
    assert jnp.allclose(result.state, jnp.asarray([0.5]))


def test_newton_keeps_nonfinite_trials_distinct_from_domain_rejections():
    problem = nl.NonlinearSystemProblem(
        lambda state, args: jnp.where(state > 0.0, jnp.asarray(jnp.nan), state - 2.0),
        validity=lambda state, residual, auxiliary, args: jnp.asarray(True),
        problem_id="nonfinite-trials",
    )
    result = nl.NewtonKrylov(line_search=nl.RootLineSearch(maximum_steps=3)).solve(
        problem,
        jnp.asarray([0.0]),
        termination=nl.NonlinearTermination(maximum_steps=2),
    )

    assert int(result.status) == int(nl.NonlinearStatus.NONFINITE_EVALUATION)
    assert int(result.diagnostics.domain_failures) == 0
    assert int(result.diagnostics.nonfinite_trials) == 3
