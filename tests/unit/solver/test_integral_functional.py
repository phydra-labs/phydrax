#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx


def _fixed_problem(integrand, *, label=None):
    domain = integrand.domain
    return phx.terms.IntegralFunctional(
        target=phx.integration.over(domain.component()),
        plan=phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(12)),
        integrand=integrand,
        label=label,
        materialization_policy="fixed",
    )


def test_integral_functional_returns_raw_signed_value_through_solver():
    domain = phx.domain.Interval1d(0.0, 1.0)
    density = domain.Function()(-2.0)
    objective = _fixed_problem(density, label="negative_energy")
    solver = phx.solver.FunctionalSolver(functions={"density": density}, terms=(objective,))

    assert jnp.allclose(solver.loss(key=jr.key(0)), -2.0, atol=1e-12)


def test_adaptive_plan_uses_the_same_integral_functional_and_trains_parameter():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="t")
    parameter = domain.Parameter(2.0)
    objective = phx.terms.IntegralFunctional.from_operator(
        target=phx.integration.over(domain.component()),
        plan=phx.integration.AdaptiveQuadraturePlan(),
        operator=lambda value: (value - 1.0) ** 2,
        objective_vars="u",
    )
    solver = phx.solver.FunctionalSolver(functions={"u": parameter}, terms=(objective,))

    initial = solver.loss()
    trained = solver.solve(
        num_iter=2,
        optim=optax.sgd(0.1),
        keep_best=False,
        jit=True,
        log_every=0,
    )

    assert jnp.allclose(initial, 1.0, atol=1e-12)
    assert jnp.allclose(trained.loss(), 0.4096, rtol=1e-8, atol=1e-10)


def test_integral_functional_gradient_matches_analytic_value():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="t")
    target = phx.integration.over(domain.component())
    plan = phx.integration.AdaptiveQuadraturePlan()

    def loss(scale):
        density = domain.Function("t")(lambda time: scale * time**2)
        objective = phx.terms.IntegralFunctional(
            target=target, plan=plan, integrand=density
        )
        return objective.loss({"density": density})

    assert jnp.allclose(jax.grad(loss)(2.0), 1.0 / 3.0, atol=1e-11)
    scales = jnp.asarray((1.0, 2.0, 3.0))
    assert jnp.allclose(jax.jit(jax.vmap(loss))(scales), scales / 3.0, atol=1e-11)


def test_integral_functional_materialization_policies_are_explicit():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    function = domain.Function("x")(lambda x: x)
    target = phx.integration.over(domain.component())
    plan = phx.integration.MonteCarloPlan(128)

    with pytest.raises(ValueError, match="requires fixed_key"):
        phx.terms.IntegralFunctional(
            target=target,
            plan=plan,
            integrand=function,
            materialization_policy="fixed",
        )

    fixed = phx.terms.IntegralFunctional(
        target=target,
        plan=plan,
        integrand=function,
        materialization_policy="fixed",
        fixed_key=jr.key(1),
    )
    assert fixed.sample(key=jr.key(2)) is fixed.fixed_realization

    caller = phx.terms.IntegralFunctional(
        target=target,
        plan=plan,
        integrand=function,
        materialization_policy="caller",
    )
    with pytest.raises(ValueError, match="requires batch"):
        caller.loss({"u": function})
    realization = phx.integration.materialize(target, plan, key=jr.key(3))
    assert jnp.isfinite(caller.loss({"u": function}, batch=realization))


def test_integral_functional_accepts_planless_external_measures():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    density = domain.Function("x")(lambda x: x)
    component_target = phx.integration.over(domain.component())
    realization = phx.integration.materialize(
        component_target,
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8)),
    )
    external_target = phx.integration.discrete(
        realization.batch.points,
        realization.batch.weights,
        axes=realization.batch.axes,
    )
    objective = phx.terms.IntegralFunctional(
        target=external_target,
        integrand=density,
    )

    assert objective.plan is None
    assert jnp.allclose(objective.loss({"density": density}), 0.5, atol=1e-12)


def test_integral_functional_rejects_complex_and_failed_estimates():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    complex_density = domain.Function()(1.0 + 2.0j)
    complex_objective = _fixed_problem(complex_density)
    with pytest.raises(TypeError, match="requires a real scalar integrand"):
        complex_objective.loss({"density": complex_density})

    discontinuity = domain.Function("x")(lambda x: jnp.where(x < 0.123, 1.0, 0.0))
    failed_objective = phx.terms.IntegralFunctional(
        target=phx.integration.over(domain.component()),
        plan=phx.integration.AdaptiveQuadraturePlan(
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
            max_intervals=1,
            throw=False,
        ),
        integrand=discontinuity,
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="did not converge"):
        jax.block_until_ready(failed_objective.loss({"u": discontinuity}))


def test_deep_ritz_energy_optimizes_with_fixed_realization():
    domain = phx.domain.Interval1d(0.0, 1.0)
    coordinate = domain.Function("x")(lambda x: x[0])
    field = domain.Parameter(0.0) * coordinate * (1.0 - coordinate)

    def density(functions):
        value = functions["u"]
        gradient = phx.operators.grad(value, var="x")
        gradient_sq = phx.operators.einsum("...i,...i->...", gradient, gradient)
        return 0.5 * gradient_sq - value

    objective = phx.terms.IntegralFunctional(
        target=phx.integration.over(domain.component()),
        plan=phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(24)),
        integrand=density,
        materialization_policy="fixed",
        label="deep_ritz_energy",
    )
    solver = phx.solver.FunctionalSolver(functions={"u": field}, terms=(objective,))
    trained = solver.solve(
        num_iter=60,
        optim=optax.adam(5e-2),
        seed=2,
        jit=True,
        keep_best=True,
        log_every=0,
    )

    assert jnp.allclose(trained.loss(key=jr.key(3)), -1.0 / 24.0, atol=3e-4)
    assert jnp.allclose(trained["u"].func(jnp.asarray([0.5])), 0.125, atol=3e-3)
