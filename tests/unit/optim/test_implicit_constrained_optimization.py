#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _termination(*, maximum_steps=100):
    return phx.optim.OptimizationTermination(
        absolute_optimality=1e-6,
        relative_optimality=0.0,
        maximum_steps=maximum_steps,
    )


def _bound_problem():
    return phx.optim.MinimizationProblem(
        lambda state, target: 0.5 * jnp.sum((state - target) ** 2),
        bounds=phx.optim.Bounds(-jnp.inf, 1.0),
    )


@pytest.mark.parametrize(
    ("method", "uses_nonlinear_constraint"),
    [
        (phx.optim.ProjectedGradient(), False),
        (phx.optim.ProjectedLBFGS(), False),
        (phx.optim.ActiveSetNewton(), False),
        (phx.optim.AugmentedLagrangian(), True),
        (phx.optim.SQP(), False),
        (
            phx.optim.PrimalDualInteriorPoint(
                mode="matrix-free-centered",
            ),
            False,
        ),
    ],
)
def test_compatible_constrained_methods_supply_implicit_kkt_derivatives(
    method,
    uses_nonlinear_constraint,
):
    if uses_nonlinear_constraint:
        constraint = phx.optim.NonlinearConstraint(lambda state, _: state, upper=1.0)
        problem = phx.optim.MinimizationProblem(
            lambda state, target: 0.5 * jnp.sum((state - target) ** 2),
            constraints=(constraint,),
        )
    else:
        problem = _bound_problem()

    result = phx.optim.minimize(
        problem,
        jnp.array([0.0]),
        method=method,
        args=jnp.array(2.0),
        termination=_termination(),
    )

    assert result.successful
    assert result.certificate is not None
    assert method.capabilities.implicit_differentiation
    assert result.provenance.implicit_differentiation

    def solution(target):
        return phx.optim.implicit_constrained_minimize(
            problem,
            jnp.array([0.0]),
            method=method,
            args=target,
            termination=_termination(),
        )[0]

    np.testing.assert_allclose(solution(jnp.array(2.0)), 1.0, atol=2e-5)
    np.testing.assert_allclose(jax.grad(solution)(jnp.array(2.0)), 0.0, atol=2e-6)


def test_active_and_inactive_bounds_compose_with_jit_jvp_and_vmap():
    problem = _bound_problem()

    def solution(target):
        return phx.optim.implicit_constrained_minimize(
            problem,
            jnp.array([0.0]),
            args=target,
            termination=_termination(),
        )[0]

    active_value, active_tangent = jax.jvp(
        eqx.filter_jit(solution),
        (jnp.array(2.0),),
        (jnp.array(1.0),),
    )
    inactive_gradient = eqx.filter_jit(jax.grad(solution))(jnp.array(0.25))
    batched = eqx.filter_jit(jax.vmap(solution))(jnp.array([0.25, 2.0]))

    np.testing.assert_allclose(active_value, 1.0, atol=2e-5)
    np.testing.assert_allclose(active_tangent, 0.0, atol=2e-6)
    np.testing.assert_allclose(inactive_gradient, 1.0, atol=2e-6)
    np.testing.assert_allclose(batched, jnp.array([0.25, 1.0]), atol=2e-5)


def test_equality_kkt_derivative_supports_nested_parameters_and_dynamic_args():
    initial = {"fixed": jnp.array([0.0]), "free": jnp.array([0.0])}
    constraint = phx.optim.NonlinearConstraint(
        lambda state, target: state["fixed"] - target["fixed"],
        lower=0.0,
        upper=0.0,
    )
    problem = phx.optim.MinimizationProblem(
        lambda state, target: (
            jnp.sum((state["fixed"] - 5.0) ** 2)
            + jnp.sum((state["free"] - target["free"]) ** 2)
        ),
        constraints=(constraint,),
    )

    def summed_solution(target):
        solution = phx.optim.implicit_constrained_minimize(
            problem,
            initial,
            args=target,
            termination=_termination(),
        )
        return solution["fixed"][0] + solution["free"][0]

    target = {"fixed": jnp.array([1.5]), "free": jnp.array([-0.5])}
    tangent = {"fixed": jnp.array([0.25]), "free": jnp.array([0.75])}
    value, derivative = jax.jvp(summed_solution, (target,), (tangent,))

    np.testing.assert_allclose(value, 1.0, atol=2e-5)
    np.testing.assert_allclose(derivative, 1.0, atol=2e-6)


def test_nonlinear_active_constraint_derivative_matches_finite_difference():
    constraint = phx.optim.NonlinearConstraint(
        lambda state, parameter: state**2 - parameter,
        upper=0.0,
    )
    problem = phx.optim.MinimizationProblem(
        lambda state, _: 0.5 * jnp.sum((state - 2.0) ** 2),
        constraints=(constraint,),
    )

    def solution(parameter):
        return phx.optim.implicit_constrained_minimize(
            problem,
            jnp.array([0.5]),
            args=parameter,
            termination=_termination(),
        )[0]

    parameter = jnp.array(1.0)
    step = 1e-4
    finite_difference = (solution(parameter + step) - solution(parameter - step)) / (
        2.0 * step
    )

    np.testing.assert_allclose(solution(parameter), 1.0, atol=2e-5)
    np.testing.assert_allclose(jax.grad(solution)(parameter), 0.5, atol=2e-5)
    np.testing.assert_allclose(
        jax.grad(solution)(parameter),
        finite_difference,
        rtol=2e-4,
        atol=2e-5,
    )


def test_constrained_initial_guess_has_zero_implicit_sensitivity():
    problem = _bound_problem()

    def solution(initial):
        return phx.optim.implicit_constrained_minimize(
            problem,
            initial,
            args=jnp.array(2.0),
            termination=_termination(),
        )[0]

    np.testing.assert_allclose(
        jax.grad(solution)(jnp.array([-2.0])),
        jnp.array([0.0]),
        atol=1e-12,
    )


def test_ambiguous_active_set_fails_instead_of_selecting_a_subgradient():
    problem = _bound_problem()
    solve = eqx.filter_jit(
        lambda target: phx.optim.implicit_constrained_minimize(
            problem,
            jnp.array([0.0]),
            args=target,
            termination=_termination(),
        )
    )

    with pytest.raises(Exception, match="strictly complementary active set"):
        solve(jnp.array(1.0))


def test_rank_deficient_active_kkt_system_fails_explicitly():
    constraint = phx.optim.NonlinearConstraint(
        lambda state, target: jnp.repeat(state - target, 2),
        lower=0.0,
        upper=0.0,
    )
    problem = phx.optim.MinimizationProblem(
        lambda state, _: 0.5 * jnp.sum((state - 2.0) ** 2),
        constraints=(constraint,),
    )
    solve = eqx.filter_jit(
        lambda target: phx.optim.implicit_constrained_minimize(
            problem,
            jnp.array([0.0]),
            args=target,
            termination=_termination(),
        )
    )

    with pytest.raises(Exception, match="singular or did not converge"):
        solve(jnp.array([1.0]))


def test_unsuccessful_constrained_primal_solve_fails_explicitly():
    problem = _bound_problem()
    solve = eqx.filter_jit(
        lambda target: phx.optim.implicit_constrained_minimize(
            problem,
            jnp.array([0.0]),
            args=target,
            termination=_termination(maximum_steps=1),
        )
    )

    with pytest.raises(Exception, match="successful KKT point"):
        solve(jnp.array(2.0))
