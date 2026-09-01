#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import optax
import pytest

import phydrax as phx


def _scalar_solver(initial: float, *, target: float = 0.0, integrand=None):
    domain = phx.domain.Interval1d(0.0, 1.0)
    field = domain.Parameter(initial)
    density = (
        (lambda functions: (functions["u"] - target) ** 2)
        if integrand is None
        else integrand
    )
    objective = phx.terms.IntegralFunctional(
        source=phx.integration.per_step(
            phx.integration.over(domain.component()),
            phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8)),
        ),
        integrand=density,
    )
    return phx.solver.FunctionalSolver(
        functions={"u": field},
        terms=(objective,),
    )


def _parameter_value(solver):
    return jnp.asarray(solver["u"].func()).reshape(())


def test_public_solve_rejects_negative_iterations_before_optimizer_dispatch():
    with pytest.raises(ValueError, match="num_iter must be non-negative"):
        _scalar_solver(1.0).solve(num_iter=-1, optim=object())


def test_public_zero_iteration_solve_returns_original_solver():
    solver = _scalar_solver(1.0)

    assert solver.solve(num_iter=0, optim=object()) is solver


def test_standard_optimizer_best_score_keeps_its_preupdate_parameters():
    base_optimizer = optax.sgd(1.0)
    optimizer = optax.GradientTransformation(
        base_optimizer.init,
        base_optimizer.update,
    )
    trained = _scalar_solver(1.0).solve(
        num_iter=1,
        optim=optimizer,
        keep_best=True,
        jit=False,
        log_every=0,
    )

    assert jnp.allclose(_parameter_value(trained), 1.0)


def test_schedule_free_returns_the_optimizer_evaluation_parameters():
    optimizer = optax.contrib.schedule_free(optax.sgd(0.1), 0.1)
    initial = jnp.asarray(2.0)
    state = optimizer.init(initial)
    updates, state = optimizer.update(jnp.asarray(2.0), state, initial)
    raw_parameters = optax.apply_updates(initial, updates)
    expected = optax.contrib.schedule_free_eval_params(state, raw_parameters)

    trained = _scalar_solver(2.0, target=1.0).solve(
        num_iter=1,
        optim=optimizer,
        evaluation_parameters=optax.contrib.schedule_free_eval_params,
        keep_best=False,
        jit=True,
        log_every=0,
    )

    assert jnp.allclose(_parameter_value(trained), expected)


def test_evaluation_transform_controls_selection_and_returned_functions():
    def shifted(_state, parameters):
        return jax.tree.map(lambda value: value + 5.0, parameters)

    trained = _scalar_solver(2.0, target=1.0).solve(
        num_iter=1,
        optim=optax.sgd(0.1),
        evaluation_parameters=shifted,
        keep_best=True,
        jit=False,
        log_every=0,
    )

    assert jnp.allclose(_parameter_value(trained), 6.8)
    assert jnp.allclose(trained.loss(), 5.8**2)


def test_identity_lifecycle_adds_no_objective_evaluations():
    calls = 0

    def counting_integrand(functions):
        nonlocal calls
        calls += 1
        return functions["u"] ** 2

    base_optimizer = optax.sgd(0.1)
    optimizer = optax.GradientTransformation(
        base_optimizer.init,
        base_optimizer.update,
    )

    _scalar_solver(1.0, integrand=counting_integrand).solve(
        num_iter=3,
        optim=optimizer,
        keep_best=True,
        jit=False,
        log_every=0,
    )

    assert calls == 3


def test_evaluation_transform_must_preserve_parameter_structure():
    with pytest.raises(ValueError, match="PyTree structure"):
        _scalar_solver(1.0).solve(
            num_iter=1,
            optim=optax.sgd(0.1),
            evaluation_parameters=lambda _state, parameters: (parameters,),
            jit=False,
            log_every=0,
        )
