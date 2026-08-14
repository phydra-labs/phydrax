import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx


def _spacetime():
    return phx.domain.Interval1d(-2.0, 2.0) @ phx.domain.TimeInterval(0.0, 1.0)


def _fixed_residual(condition, num_samples, key):
    target = phx.integration.mean_over(condition.on)
    plan = phx.integration.MonteCarloPlan(num_samples)
    realization = phx.integration.materialize(target, plan, key=key)
    return phx.terms.ResidualPenalty(
        condition,
        phx.integration.fixed(realization),
    )


def test_manufactured_brownian_backward_constraint_is_zero_and_perturbation_is_not():
    domain = _spacetime()
    truth = 0.7
    observable = domain.Function("x", "t")(lambda x, t: x[0] ** 2 - truth**2 * t)
    drift = domain.Function("x", "t")(lambda x, t: jnp.asarray([0.0]))
    exact_diffusion = domain.Function("x", "t")(lambda x, t: jnp.asarray([[truth]]))
    wrong_diffusion = domain.Function("x", "t")(lambda x, t: jnp.asarray([[0.5]]))

    exact_condition = phx.conditions.stochastic.Kolmogorov(
        "u",
        domain.component(),
        drift=drift,
        evolution_var="t",
        diffusion=exact_diffusion,
    )
    perturbed_condition = phx.conditions.stochastic.Kolmogorov(
        "u",
        domain.component(),
        drift=drift,
        evolution_var="t",
        diffusion=wrong_diffusion,
    )
    exact = _fixed_residual(exact_condition, 32, jr.key(0))
    perturbed = _fixed_residual(perturbed_condition, 32, jr.key(0))

    assert exact.loss({"u": observable}, key=jr.key(1)) < 1e-12
    assert perturbed.loss({"u": observable}, key=jr.key(1)) > 1e-3


def test_stationary_ornstein_uhlenbeck_fokker_planck_constraint_is_zero():
    domain = phx.domain.Interval1d(-3.0, 3.0)
    theta, sigma = 0.8, 0.6
    density = domain.Function("x")(lambda x: jnp.exp(-theta * x[0] ** 2 / sigma**2))
    drift = domain.Function("x")(lambda x: jnp.asarray([-theta * x[0]]))
    diffusion = domain.Function("x")(lambda x: jnp.asarray([[sigma]]))
    condition = phx.conditions.stochastic.FokkerPlanck(
        "p",
        domain.component(),
        drift=drift,
        evolution_var=None,
        diffusion=diffusion,
    )
    constraint = _fixed_residual(condition, 48, jr.key(2))

    assert constraint.loss({"p": density}, key=jr.key(3)) < 1e-12


def test_fokker_planck_stationary_and_evolution_modes_use_opposite_time_contracts():
    domain = _spacetime()
    density = domain.Function("x", "t")(lambda x, t: t)
    drift = domain.Function("x", "t")(lambda x, t: jnp.asarray([0.0]))
    stationary_condition = phx.conditions.stochastic.FokkerPlanck(
        "p",
        domain.component(),
        drift=drift,
        evolution_var=None,
    )
    evolving_condition = phx.conditions.stochastic.FokkerPlanck(
        "p",
        domain.component(),
        drift=drift,
        evolution_var="t",
    )
    stationary = _fixed_residual(stationary_condition, 16, jr.key(4))
    evolving = _fixed_residual(evolving_condition, 16, jr.key(4))

    assert stationary.loss({"p": density}, key=jr.key(5)) < 1e-12
    assert jnp.allclose(evolving.loss({"p": density}, key=jr.key(5)), 1.0)


def test_named_diffusion_field_is_jointly_optimized_by_functional_solver():
    domain = _spacetime()
    truth = 0.7
    observable = domain.Function("x", "t")(lambda x, t: x[0] ** 2 - truth**2 * t)
    drift = domain.Function("x", "t")(lambda x, t: jnp.asarray([0.0]))
    scale = domain.Parameter(0.2)
    diffusion = scale * jnp.ones((1, 1))
    condition = phx.conditions.stochastic.Kolmogorov(
        "u",
        domain.component(),
        drift=drift,
        evolution_var="t",
        diffusion="sigma",
    )
    constraint = _fixed_residual(condition, 24, jr.key(6))
    solver = phx.solver.FunctionalSolver(
        functions={"u": observable, "sigma": diffusion},
        terms=[constraint],
    )
    initial_loss = solver.loss(key=jr.key(7))

    trained = solver.solve(
        num_iter=80,
        optim=optax.adam(5e-2),
        seed=8,
        jit=True,
        keep_best=True,
        log_every=0,
    )
    learned = trained["sigma"].func()

    assert trained.loss(key=jr.key(7)) < initial_loss * 1e-4
    assert jnp.allclose(learned, jnp.asarray([[truth]]), atol=2e-3)


def test_stochastic_constraints_preserve_fixed_and_resampled_collocation_semantics():
    domain = _spacetime()
    observable = domain.Function("x", "t")(lambda x, t: x[0] ** 2 - t)
    drift = domain.Function("x", "t")(lambda x, t: jnp.asarray([0.0]))
    diffusion = domain.Function("x", "t")(lambda x, t: jnp.asarray([[1.0]]))
    condition = phx.conditions.stochastic.Kolmogorov(
        "u",
        domain.component(),
        drift=drift,
        evolution_var="t",
        diffusion=diffusion,
    )
    target = phx.integration.mean_over(condition.on)
    plan = phx.integration.MonteCarloPlan(12)
    fixed_realization = phx.integration.materialize(target, plan, key=jr.key(9))
    fixed = phx.terms.ResidualPenalty(
        condition,
        phx.integration.fixed(fixed_realization),
    )
    resampled = phx.terms.ResidualPenalty(
        condition,
        phx.integration.per_step(target, plan),
    )
    random_a = phx.integration.materialize(
        resampled.source.target,
        resampled.source.plan,
        key=jr.key(10),
    )
    random_b = phx.integration.materialize(
        resampled.source.target,
        resampled.source.plan,
        key=jr.key(11),
    )

    assert eqx.tree_equal(fixed.source.realization.batch, fixed_realization.batch)
    assert not eqx.tree_equal(random_a.batch, random_b.batch)
    assert fixed.loss({"u": observable}, key=jr.key(12)) < 1e-12


def test_fokker_planck_constraint_composes_with_explicit_density_normalization():
    domain = phx.domain.Interval1d(-1.0, 1.0)
    density = domain.Function("x")(lambda x: jnp.asarray(0.5))
    drift = domain.Function("x")(lambda x: jnp.asarray([0.0]))
    dynamics_condition = phx.conditions.stochastic.FokkerPlanck(
        "p",
        domain.component(),
        drift=drift,
        evolution_var=None,
    )
    rule = phx.integration.GaussLegendreRule(12)
    plan = phx.integration.FixedQuadraturePlan(rule)
    dynamics_target = phx.integration.mean_over(dynamics_condition.on)
    dynamics = phx.terms.ResidualPenalty(
        dynamics_condition,
        phx.integration.fixed(phx.integration.materialize(dynamics_target, plan)),
    )
    normalization_condition = phx.conditions.Moment(
        "p",
        domain.component(),
        lambda p: p,
        target=1.0,
    )
    normalization_target = phx.integration.over(normalization_condition.on)
    normalization = phx.terms.MomentPenalty(
        normalization_condition,
        phx.integration.fixed(phx.integration.materialize(normalization_target, plan)),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"p": density},
        terms=[dynamics, normalization],
    )

    assert solver.loss(key=jr.key(13)) < 1e-12


def test_probability_flux_boundary_constraint_enforces_reflecting_current():
    domain = phx.domain.Interval1d(-2.0, 2.0)
    component = domain.component({"x": phx.domain.Boundary()})
    theta, sigma = 0.8, 0.6
    density = domain.Function("x")(lambda x: jnp.exp(-theta * x[0] ** 2 / sigma**2))
    drift = domain.Function("x")(lambda x: jnp.asarray([-theta * x[0]]))
    diffusion = domain.Function("x")(lambda x: jnp.asarray([[sigma]]))
    condition = phx.conditions.stochastic.ProbabilityFlux(
        "p",
        component,
        drift=drift,
        diffusion=diffusion,
    )
    constraint = _fixed_residual(condition, 16, jr.key(14))

    assert constraint.loss({"p": density}, key=jr.key(15)) < 1e-12


def test_pointwise_spde_residual_rejects_rough_or_non_strong_solution_concepts():
    domain = phx.domain.Interval1d(0.0, 1.0)
    rough = phx.stochastic.SPDESolutionSpec(
        "mild",
        noise_regularization="space_time_white",
        cutoff_id="fourier:64",
    )

    with pytest.raises(ValueError, match="pointwise strong residual"):
        phx.stochastic.validate_spde_formulation(rough, "pointwise_strong")

    regularized = phx.stochastic.SPDESolutionSpec(
        "strong",
        noise_regularization="finite_rank",
        cutoff_id="fourier:64",
    )
    phx.stochastic.validate_spde_formulation(regularized, "pointwise_strong")
    condition = phx.conditions.Residual(
        "u",
        domain.component(),
        lambda u: u,
    )
    constraint = phx.terms.ResidualPenalty(
        condition,
        phx.integration.per_step(
            phx.integration.mean_over(condition.on),
            phx.integration.MonteCarloPlan(8),
        ),
    )
    assert isinstance(constraint, phx.terms.ResidualPenalty)
