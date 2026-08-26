import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax.linalg as la
import phydrax.nonlinear as nl


def _termination(*, maximum_steps=20):
    return nl.NonlinearTermination(
        absolute_residual=1e-11,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=maximum_steps,
    )


def test_implicit_root_result_retains_native_evidence_and_analytic_derivatives():
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state**2 - target,
        problem_id="implicit-square",
    )

    def root(target):
        result = nl.implicit_root_result(
            problem,
            jnp.asarray([1.0, 1.0]),
            termination=_termination(),
            args=target,
        )
        return result.state

    target = jnp.asarray([1.0, 4.0])
    value, tangent = jax.jvp(root, (target,), (jnp.ones_like(target),))
    gradient = jax.grad(lambda argument: jnp.sum(root(argument)))(target)
    result = nl.implicit_root_result(
        problem,
        jnp.asarray([1.0, 1.0]),
        termination=_termination(),
        args=target,
    )

    assert result.successful
    assert result.provenance.problem_id == "implicit-square"
    assert result.diagnostics.iterations > 0
    assert jnp.allclose(value, jnp.asarray([1.0, 2.0]), rtol=1e-9, atol=1e-11)
    assert jnp.allclose(tangent, jnp.asarray([0.5, 0.25]), rtol=1e-8, atol=1e-10)
    assert jnp.allclose(gradient, tangent, rtol=1e-8, atol=1e-10)


def test_implicit_root_result_recomputes_differentiable_auxiliary_at_root():
    problem = nl.NonlinearSystemProblem(
        lambda state, target: (state**2 - target, target * state),
        has_aux=True,
        problem_id="implicit-auxiliary",
    )

    def observable(target):
        result = nl.implicit_root_result(
            problem,
            jnp.asarray(1.0),
            termination=_termination(),
            args=target,
        )
        return result.auxiliary

    value, gradient = jax.value_and_grad(observable)(jnp.asarray(4.0))

    assert jnp.allclose(value, 8.0, rtol=1e-9, atol=1e-11)
    assert jnp.allclose(gradient, 3.0, rtol=1e-8, atol=1e-10)


def test_prepared_implicit_root_refresh_preserves_symbolic_linear_identity():
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state**2 - target,
        problem_id="prepared-implicit-square",
    )
    prepared = nl.prepare_nonlinear(
        problem,
        jnp.asarray([1.0, 1.0]),
        termination=_termination(),
        args=jnp.asarray([1.0, 1.0]),
    )
    refreshed = nl.refresh_nonlinear(
        prepared,
        problem,
        jnp.asarray([2.0, 3.0]),
        args=jnp.asarray([4.0, 9.0]),
    )
    result = nl.implicit_root_result(refreshed)

    assert result.successful
    assert jnp.allclose(result.state, jnp.asarray([2.0, 3.0]), atol=1e-11)
    assert refreshed.linear_template_id == prepared.linear_template_id
    assert refreshed.linear_plan_id == prepared.linear_plan_id
    assert refreshed.numeric_version == prepared.numeric_version + 1
    assert result.provenance.linear_plan_id == prepared.linear_plan_id


def test_failed_implicit_root_remains_inspectable_and_checked_root_raises():
    problem = nl.NonlinearSystemProblem(
        lambda state, _: state**2 - 2.0,
        problem_id="failed-implicit-root",
    )
    termination = _termination(maximum_steps=1)
    result = nl.implicit_root_result(
        problem,
        jnp.asarray(10.0),
        termination=termination,
    )

    assert not result.successful
    assert result.status == int(nl.NonlinearStatus.MAXIMUM_STEPS_REACHED)
    assert jnp.isfinite(result.state)
    with pytest.raises(
        eqx.EquinoxRuntimeError, match="Implicit nonlinear root solve failed"
    ):
        nl.implicit_root(problem, jnp.asarray(10.0), termination=termination)


def _nonnormal_root():
    matrix = jnp.asarray(
        [
            [1.0, 4.0, 0.0, 0.0],
            [0.0, 2.0, 4.0, 0.0],
            [0.0, 0.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
    )
    problem = nl.NonlinearSystemProblem(
        lambda state, target: matrix @ state - target,
        problem_id="implicit-nonnormal-linear",
    )
    method = nl.NewtonKrylov(linear_policy=la.LinearSolvePolicy(la.DenseLU()))
    return matrix, problem, method


def _underresolved_gmres():
    return la.LinearSolvePolicy(
        la.GMRES(restart=1),
        tolerance=la.TolerancePolicy(
            relative=1e-14,
            absolute=1e-14,
            max_steps=1,
        ),
    )


def test_implicit_root_uses_distinct_tangent_and_adjoint_policies():
    matrix, problem, method = _nonnormal_root()
    dense = la.LinearSolvePolicy(la.DenseLU())
    target = jnp.asarray([1.0, -0.5, 0.25, 2.0])
    direction = jnp.asarray([0.2, -0.1, 0.4, 0.3])

    def root(argument, policy):
        return nl.implicit_root(
            problem,
            jnp.zeros_like(argument),
            method=method,
            termination=_termination(),
            derivative_policy=policy,
            args=argument,
        )

    tangent_policy = nl.ImplicitRootDerivativePolicy(
        tangent_linear_policy=dense,
        adjoint_linear_policy=_underresolved_gmres(),
    )
    value, tangent = jax.jvp(
        lambda argument: root(argument, tangent_policy),
        (target,),
        (direction,),
    )

    assert jnp.allclose(matrix @ value, target, rtol=1e-10, atol=1e-11)
    assert jnp.allclose(matrix @ tangent, direction, rtol=1e-10, atol=1e-11)
    failed_adjoint = eqx.filter_jit(
        jax.grad(lambda argument: jnp.sum(root(argument, tangent_policy)))
    )
    with pytest.raises(
        eqx.EquinoxRuntimeError, match="root derivative solve failed"
    ):
        failed_adjoint(target)

    adjoint_policy = nl.ImplicitRootDerivativePolicy(
        tangent_linear_policy=_underresolved_gmres(),
        adjoint_linear_policy=dense,
    )
    gradient = jax.grad(
        lambda argument: jnp.sum(root(argument, adjoint_policy))
    )(target)
    expected_gradient = jnp.linalg.solve(matrix.T, jnp.ones_like(target))
    assert jnp.allclose(gradient, expected_gradient, rtol=1e-10, atol=1e-11)
    failed_tangent = eqx.filter_jit(
        lambda argument: jax.jvp(
            lambda value: root(value, adjoint_policy),
            (argument,),
            (direction,),
        )[1]
    )
    with pytest.raises(
        eqx.EquinoxRuntimeError, match="root derivative solve failed"
    ):
        failed_tangent(target)


def test_implicit_root_derivative_policy_defaults_adjoint_to_tangent():
    matrix, problem, method = _nonnormal_root()
    policy = nl.ImplicitRootDerivativePolicy(
        tangent_linear_policy=la.LinearSolvePolicy(la.DenseLU())
    )
    target = jnp.asarray([1.0, -0.5, 0.25, 2.0])

    gradient = jax.grad(
        lambda argument: jnp.sum(
            nl.implicit_root(
                problem,
                jnp.zeros_like(argument),
                method=method,
                termination=_termination(),
                derivative_policy=policy,
                args=argument,
            )
        )
    )(target)

    assert jnp.allclose(
        gradient,
        jnp.linalg.solve(matrix.T, jnp.ones_like(target)),
        rtol=1e-10,
        atol=1e-11,
    )


def test_implicit_root_requires_tangent_policy_for_non_newton_method():
    problem = nl.NonlinearSystemProblem(
        lambda state, target: state**2 - target,
        problem_id="implicit-fixed-point-policy",
    )
    method = nl.NonlinearRichardson(
        nl.FunctionNonlinearUpdate(lambda state, target: target / state)
    )

    with pytest.raises(ValueError, match="tangent linear policy is required"):
        nl.implicit_root_result(
            problem,
            jnp.asarray(1.0),
            method=method,
            termination=_termination(),
            args=jnp.asarray(2.0),
        )


def test_implicit_root_derivative_policy_validates_linear_policies():
    with pytest.raises(TypeError, match="tangent_linear_policy"):
        nl.ImplicitRootDerivativePolicy(tangent_linear_policy=object())
    with pytest.raises(TypeError, match="adjoint_linear_policy"):
        nl.ImplicitRootDerivativePolicy(adjoint_linear_policy=object())
