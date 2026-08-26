import jax
import jax.numpy as jnp
import pytest

import phydrax.linalg as la
import phydrax.nonlinear as nl


jax.config.update("jax_enable_x64", True)


def _gmres_policy(*, max_steps=64, restart=8):
    return la.LinearSolvePolicy(
        la.GMRES(restart=restart),
        tolerance=la.TolerancePolicy(
            relative=1e-10,
            absolute=1e-12,
            max_steps=max_steps,
        ),
    )


def _termination(*, maximum_steps=50):
    return nl.NonlinearTermination(
        absolute_residual=1e-10,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=maximum_steps,
        maximum_evaluations=300,
        maximum_linear_iterations=5000,
    )


def _scaled_operator(space, scale, *, operator_id="scaled-lagged"):
    return la.FunctionLinearOperator(
        lambda direction: scale * direction,
        source=space,
        target=space,
        operator_id=operator_id,
    )


def test_lagged_linear_update_applies_prepared_physical_correction():
    space = la.ArraySpace((3,), dtype=jnp.float64)
    target = jnp.asarray([2.0, -1.0, 4.0])
    problem = nl.NonlinearSystemProblem(
        lambda state, rhs: 2.0 * state - rhs,
        state_space=space,
        residual_space=space,
        problem_id="lagged-linear-update",
    )
    update = nl.LaggedLinearSolveUpdate(
        lambda state, args: _scaled_operator(space, 2.0),
        linear_policy=_gmres_policy(),
    )
    prepared = nl.prepare_nonlinear_update(
        problem,
        jnp.zeros_like(target),
        update,
        args=target,
    )
    template_id = prepared.internal_state.template.template_id
    result, refreshed = nl.apply_prepared_nonlinear_update(
        prepared,
        jnp.zeros_like(target),
        args=target,
    )

    assert result.applied
    assert result.inner_status == int(la.LinearSolveStatus.SUCCESS)
    assert jnp.allclose(result.state, target / 2.0, rtol=1e-10, atol=1e-12)
    assert jnp.linalg.norm(result.residual) < 1e-10
    assert result.diagnostics.residual_evaluations == 2
    assert result.diagnostics.validity_evaluations == 1
    assert result.diagnostics.linear_refreshes == 1
    assert result.diagnostics.linear_solves == 1
    assert result.diagnostics.jvp_evaluations > 0
    assert refreshed.internal_state.template.template_id == template_id
    assert refreshed.internal_state.numeric_version == 1


def test_lagged_linear_update_refresh_preserves_symbolic_identity():
    space = la.ArraySpace((2,), dtype=jnp.float64)
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state**2 - target,
        state_space=space,
        residual_space=space,
        problem_id="lagged-refresh",
    )
    update = nl.LaggedLinearSolveUpdate(
        lambda state, args: _scaled_operator(space, state),
        linear_policy=_gmres_policy(),
    )
    prepared = nl.prepare_nonlinear_update(
        problem,
        jnp.asarray([1.0, 1.5]),
        update,
        args=jnp.asarray([2.0, 3.0]),
    )
    refreshed = nl.refresh_nonlinear_update(
        prepared,
        problem,
        jnp.asarray([1.2, 1.7]),
        args=jnp.asarray([2.0, 3.0]),
    )

    assert refreshed.internal_state.template.template_id == (
        prepared.internal_state.template.template_id
    )
    assert refreshed.internal_state.plan.plan_id == prepared.internal_state.plan.plan_id
    assert refreshed.internal_state.numeric_version == 1


def test_lagged_linear_update_rejects_changed_operator_identity():
    space = la.ArraySpace((2,), dtype=jnp.float64)
    problem = nl.NonlinearSystemProblem(
        lambda state, args: state - 1.0,
        state_space=space,
        residual_space=space,
        problem_id="lagged-identity-change",
    )
    update = nl.LaggedLinearSolveUpdate(
        lambda state, operator_id: _scaled_operator(
            space,
            1.0,
            operator_id=operator_id,
        ),
        linear_policy=_gmres_policy(),
    )
    prepared = nl.prepare_nonlinear_update(
        problem,
        jnp.zeros((2,)),
        update,
        args="first-operator",
    )

    with pytest.raises(ValueError, match="problem_id"):
        nl.refresh_nonlinear_update(
            prepared,
            problem,
            jnp.zeros((2,)),
            args="changed-operator",
        )


def test_lagged_linear_update_fails_closed_on_budget_and_domain_rejection():
    space = la.ArraySpace((), dtype=jnp.float64)
    problem = nl.NonlinearSystemProblem(
        lambda state, args: state + 1.0,
        state_space=space,
        residual_space=space,
        validity=lambda state, residual, auxiliary, args: state > 0.0,
        problem_id="lagged-domain-rejection",
    )
    update = nl.LaggedLinearSolveUpdate(
        lambda state, args: _scaled_operator(space, jnp.asarray(1.0)),
        linear_policy=_gmres_policy(max_steps=8),
    )
    prepared = nl.prepare_nonlinear_update(
        problem,
        jnp.asarray(1.0),
        update,
    )

    skipped, _ = nl.apply_prepared_nonlinear_update(
        prepared,
        jnp.asarray(1.0),
        control=nl.NonlinearUpdateControl(maximum_linear_iterations=1),
    )
    rejected, _ = nl.apply_prepared_nonlinear_update(
        prepared,
        jnp.asarray(1.0),
    )

    assert skipped.status == int(nl.NonlinearUpdateStatus.BUDGET_EXHAUSTED)
    assert jnp.array_equal(skipped.state, jnp.asarray(1.0))
    assert rejected.status == int(nl.NonlinearUpdateStatus.DOMAIN_REJECTED)
    assert jnp.array_equal(rejected.state, jnp.asarray(1.0))
    assert jnp.array_equal(rejected.residual, jnp.asarray(2.0))


def test_lagged_linear_update_validates_configuration_and_spaces():
    with pytest.raises(TypeError, match="operator_function"):
        nl.LaggedLinearSolveUpdate(object())
    with pytest.raises(ValueError, match="damping"):
        nl.LaggedLinearSolveUpdate(lambda state, args: None, damping=0.0)
    with pytest.raises(ValueError, match="update_id"):
        nl.LaggedLinearSolveUpdate(lambda state, args: None, update_id="")

    state_space = la.ArraySpace((2,), dtype=jnp.float64)
    wrong_space = la.ArraySpace((3,), dtype=jnp.float64)
    problem = nl.NonlinearSystemProblem(
        lambda state, args: state - 1.0,
        state_space=state_space,
        residual_space=state_space,
        problem_id="lagged-space-validation",
    )
    wrong_update = nl.LaggedLinearSolveUpdate(
        lambda state, args: _scaled_operator(wrong_space, 1.0),
        linear_policy=_gmres_policy(),
    )
    nonoperator_update = nl.LaggedLinearSolveUpdate(
        lambda state, args: state,
        linear_policy=_gmres_policy(),
    )

    with pytest.raises(ValueError, match="source"):
        nl.prepare_nonlinear_update(
            problem,
            jnp.zeros((2,)),
            wrong_update,
        )
    with pytest.raises(TypeError, match="AbstractLinearOperator"):
        nl.prepare_nonlinear_update(
            problem,
            jnp.zeros((2,)),
            nonoperator_update,
        )


def test_lagged_linear_root_matches_newton_and_has_exact_implicit_derivatives():
    space = la.ArraySpace((), dtype=jnp.float64)
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state**2 - target,
        state_space=space,
        residual_space=space,
        problem_id="lagged-square-root",
    )
    update = nl.LaggedLinearSolveUpdate(
        lambda state, args: _scaled_operator(space, state),
        linear_policy=_gmres_policy(max_steps=16),
        damping=0.5,
    )
    method = nl.NonlinearRichardson(update)
    derivative_policy = nl.ImplicitRootDerivativePolicy(
        tangent_linear_policy=la.LinearSolvePolicy(la.DenseLU())
    )
    target = jnp.asarray(2.0)

    lagged_result = method.solve(
        problem,
        jnp.asarray(1.0),
        termination=_termination(),
        args=target,
    )
    newton_result = nl.NewtonKrylov().solve(
        problem,
        jnp.asarray(1.0),
        termination=_termination(),
        args=target,
    )

    def root(argument):
        return nl.implicit_root(
            problem,
            jnp.asarray(1.0),
            method=method,
            termination=_termination(),
            derivative_policy=derivative_policy,
            args=argument,
        )

    def primal(argument):
        return method.solve(
            problem,
            jnp.asarray(1.0),
            termination=_termination(),
            args=argument,
        ).state

    value, tangent = jax.jvp(root, (target,), (jnp.asarray(1.0),))
    gradient = jax.grad(root)(target)
    expected_derivative = 1.0 / (2.0 * jnp.sqrt(target))
    compiled = jax.jit(primal)(target)
    batched_targets = jnp.asarray([1.5, 2.0, 3.0])
    batched = jax.vmap(primal)(batched_targets)

    assert lagged_result.successful
    assert newton_result.successful
    assert jnp.allclose(lagged_result.state, newton_result.state, atol=1e-9)
    assert jnp.allclose(value, jnp.sqrt(target), atol=1e-9)
    assert jnp.allclose(tangent, expected_derivative, rtol=1e-8, atol=1e-10)
    assert jnp.allclose(gradient, expected_derivative, rtol=1e-8, atol=1e-10)
    assert jnp.allclose(compiled, jnp.sqrt(target), atol=1e-9)
    assert jnp.allclose(batched, jnp.sqrt(batched_targets), atol=1e-9)
