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
    target = phx.integration.over(domain.component())
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(12))
    realization = phx.integration.materialize(target, plan)
    return phx.terms.IntegralFunctional(
        source=phx.integration.fixed(realization),
        integrand=integrand,
        label=label,
    )


def test_integral_functional_returns_raw_signed_value_through_solver():
    domain = phx.domain.Interval1d(0.0, 1.0)
    density = domain.Function()(-2.0)
    objective = _fixed_problem(density, label="negative_energy")
    solver = phx.solver.FunctionalSolver(
        functions={"density": density}, terms=(objective,)
    )

    assert jnp.allclose(solver.loss(key=jr.key(0)), -2.0, atol=1e-12)


def test_adaptive_plan_uses_the_same_integral_functional_and_trains_parameter():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="t")
    parameter = domain.Parameter(2.0)
    objective = phx.terms.IntegralFunctional.from_operator(
        source=phx.integration.per_step(
            phx.integration.over(domain.component()),
            phx.integration.AdaptiveQuadraturePlan(),
        ),
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
            source=phx.integration.per_step(target, plan), integrand=density
        )
        return objective.loss({"density": density})

    assert jnp.allclose(jax.grad(loss)(2.0), 1.0 / 3.0, atol=1e-11)
    scales = jnp.asarray((1.0, 2.0, 3.0))
    assert jnp.allclose(jax.jit(jax.vmap(loss))(scales), scales / 3.0, atol=1e-11)


def test_integral_functional_integration_sources_are_explicit():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    function = domain.Function("x")(lambda x: x)
    target = phx.integration.over(domain.component())
    plan = phx.integration.MonteCarloPlan(128)

    realization = phx.integration.materialize(target, plan, key=jr.key(1))
    fixed = phx.terms.IntegralFunctional(
        source=phx.integration.fixed(realization),
        integrand=function,
    )
    assert fixed.sample(key=jr.key(2)) is realization

    caller = phx.terms.IntegralFunctional(
        source=phx.integration.caller(target),
        integrand=function,
    )
    with pytest.raises(ValueError, match="requires batch"):
        caller.loss({"u": function})
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
        source=phx.integration.per_step(external_target),
        integrand=density,
    )

    assert jnp.allclose(objective.loss({"density": density}), 0.5, atol=1e-12)


def test_integral_functional_rejects_complex_and_failed_estimates():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    complex_density = domain.Function()(1.0 + 2.0j)
    complex_objective = _fixed_problem(complex_density)
    with pytest.raises(TypeError, match="requires a real scalar integrand"):
        complex_objective.loss({"density": complex_density})

    discontinuity = domain.Function("x")(lambda x: jnp.where(x < 0.123, 1.0, 0.0))
    failed_objective = phx.terms.IntegralFunctional(
        source=phx.integration.per_step(
            phx.integration.over(domain.component()),
            phx.integration.AdaptiveQuadraturePlan(
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
                max_intervals=1,
                throw=False,
            ),
        ),
        integrand=discontinuity,
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="did not converge"):
        jax.block_until_ready(failed_objective.loss({"u": discontinuity}))


def test_integral_functional_nonfinite_integrand_policy_is_narrow():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    nonfinite = domain.Function()(jnp.nan)
    target = phx.integration.over(domain.component())
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(4))

    strict = phx.terms.IntegralFunctional(
        source=phx.integration.per_step(target, plan),
        integrand=nonfinite,
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="did not converge"):
        jax.block_until_ready(strict.loss({"u": nonfinite}))

    propagated = phx.terms.IntegralFunctional(
        source=phx.integration.per_step(target, plan),
        integrand=nonfinite,
        nonfinite_integrand="propagate",
    )
    assert not bool(jnp.isfinite(propagated.loss({"u": nonfinite})))

    discontinuity = domain.Function("x")(lambda x: jnp.where(x < 0.123, 1.0, 0.0))
    failed_for_budget = phx.terms.IntegralFunctional(
        source=phx.integration.per_step(
            target,
            phx.integration.AdaptiveQuadraturePlan(
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
                max_intervals=1,
                throw=False,
            ),
        ),
        integrand=discontinuity,
        nonfinite_integrand="propagate",
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="did not converge"):
        jax.block_until_ready(failed_for_budget.loss({"u": discontinuity}))

    with pytest.raises(ValueError, match="nonfinite_integrand"):
        phx.terms.IntegralFunctional(
            source=phx.integration.per_step(target, plan),
            integrand=nonfinite,
            nonfinite_integrand="ignore",
        )


def test_integral_functional_from_operator_forwards_nonfinite_policy():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    parameter = domain.Parameter(1.0)
    objective = phx.terms.IntegralFunctional.from_operator(
        source=phx.integration.per_step(
            phx.integration.over(domain.component()),
            phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(4)),
        ),
        operator=lambda value: value,
        objective_vars="u",
        nonfinite_integrand="propagate",
    )
    assert objective.nonfinite_integrand == "propagate"


def test_lbfgs_rejects_nonfinite_neo_hookean_trial():
    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    coordinate = domain.Function("x")(lambda x: x)
    displacement = domain.Parameter(0.0) * coordinate

    def density(functions):
        current = functions["u"]
        stored_energy = phx.operators.neo_hookean_reference_energy(
            current,
            mu=1.0,
            lambda_=2.0,
        )
        compression = 20.0 * phx.operators.einsum("...i,...i->...", current, coordinate)
        return stored_energy + compression

    target = phx.integration.over(domain.component())
    plan = phx.integration.MonteCarloPlan(128)
    realization = phx.integration.materialize(target, plan, key=jr.key(91))
    objective = phx.terms.IntegralFunctional(
        source=phx.integration.fixed(realization),
        integrand=density,
        nonfinite_integrand="propagate",
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": displacement},
        terms=(objective,),
    )
    trained = solver.solve(
        num_iter=12,
        optim=optax.lbfgs(learning_rate=1.0),
        seed=92,
        jit=True,
        keep_best=False,
        log_every=0,
    )
    probe = trained["u"].func(jnp.asarray((1.0, 0.0)))
    stretch = 1.0 + probe[0]
    assert jnp.isfinite(trained.loss())
    assert 0.0 < stretch < 1.0


def test_deep_ritz_energy_optimizes_with_fixed_realization():
    domain = phx.domain.Interval1d(0.0, 1.0)
    coordinate = domain.Function("x")(lambda x: x[0])
    field = domain.Parameter(0.0) * coordinate * (1.0 - coordinate)

    def density(functions):
        value = functions["u"]
        gradient = phx.operators.grad(value, var="x")
        gradient_sq = phx.operators.einsum("...i,...i->...", gradient, gradient)
        return 0.5 * gradient_sq - value

    target = phx.integration.over(domain.component())
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(24))
    objective = phx.terms.IntegralFunctional(
        source=phx.integration.fixed(phx.integration.materialize(target, plan)),
        integrand=density,
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


def test_field_stationarity_reuses_one_prepared_scalar_term_realization():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    parameter = domain.Parameter(2.0)
    functions = {"u": parameter}
    term = phx.terms.IntegralFunctional.from_operator(
        source=phx.integration.per_step(
            phx.integration.over(domain.component()),
            phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(6)),
        ),
        operator=lambda value: (value - 1.0) ** 2,
        objective_vars="u",
    )
    subspace = phx.nn.parameters.ParameterSubspace(functions, eqx.is_inexact_array)
    prepared = phx.solver.prepare_functional_stationarity(
        functions,
        term,
        subspace,
        realization_id="fixed-quadrature-realization",
        provenance_id="scalar-term-stationarity",
        key=jr.key(17),
    )

    first = prepared.problem.residual(prepared.initial_state)
    second = prepared.problem.residual(prepared.initial_state)

    assert jnp.array_equal(first, second)
    assert jnp.allclose(first, jnp.asarray((0.0, 0.0, 2.0)), atol=1e-12)
