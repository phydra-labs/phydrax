import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx


def _spacetime():
    domain = phx.domain.Interval1d(-2.0, 2.0) @ phx.domain.TimeInterval(0.0, 1.0)
    return domain, phx.domain.SampleLayout((("x", "t"),))


def test_manufactured_brownian_backward_constraint_is_zero_and_perturbation_is_not():
    domain, structure = _spacetime()
    truth = 0.7
    observable = domain.Function("x", "t")(lambda x, t: x[0] ** 2 - truth**2 * t)
    drift = domain.Function("x", "t")(lambda x, t: jnp.asarray([0.0]))
    exact_diffusion = domain.Function("x", "t")(lambda x, t: jnp.asarray([[truth]]))
    wrong_diffusion = domain.Function("x", "t")(lambda x, t: jnp.asarray([[0.5]]))

    exact = phx.constraints.ContinuousKolmogorovConstraint(
        "u",
        domain.component(),
        drift=drift,
        evolution_var="t",
        diffusion=exact_diffusion,
        sampling=phx.domain.PointSampling(32, layout=structure),
        sampling_mode="fixed",
        fixed_batch_key=jr.key(0),
    )
    perturbed = phx.constraints.ContinuousKolmogorovConstraint(
        "u",
        domain.component(),
        drift=drift,
        evolution_var="t",
        diffusion=wrong_diffusion,
        sampling=phx.domain.PointSampling(32, layout=structure),
        sampling_mode="fixed",
        fixed_batch_key=jr.key(0),
    )

    assert exact.loss({"u": observable}, key=jr.key(1)) < 1e-12
    assert perturbed.loss({"u": observable}, key=jr.key(1)) > 1e-3


def test_stationary_ornstein_uhlenbeck_fokker_planck_constraint_is_zero():
    domain = phx.domain.Interval1d(-3.0, 3.0)
    structure = phx.domain.SampleLayout((("x",),))
    theta, sigma = 0.8, 0.6
    density = domain.Function("x")(lambda x: jnp.exp(-theta * x[0] ** 2 / sigma**2))
    drift = domain.Function("x")(lambda x: jnp.asarray([-theta * x[0]]))
    diffusion = domain.Function("x")(lambda x: jnp.asarray([[sigma]]))
    constraint = phx.constraints.ContinuousFokkerPlanckConstraint(
        "p",
        domain.component(),
        drift=drift,
        evolution_var=None,
        diffusion=diffusion,
        sampling=phx.domain.PointSampling(48, layout=structure),
        sampling_mode="fixed",
        fixed_batch_key=jr.key(2),
    )

    assert constraint.loss({"p": density}, key=jr.key(3)) < 1e-12


def test_fokker_planck_stationary_and_evolution_modes_use_opposite_time_contracts():
    domain, structure = _spacetime()
    density = domain.Function("x", "t")(lambda x, t: t)
    drift = domain.Function("x", "t")(lambda x, t: jnp.asarray([0.0]))
    stationary = phx.constraints.ContinuousFokkerPlanckConstraint(
        "p",
        domain.component(),
        drift=drift,
        evolution_var=None,
        sampling=phx.domain.PointSampling(16, layout=structure),
        sampling_mode="fixed",
        fixed_batch_key=jr.key(4),
    )
    evolving = phx.constraints.ContinuousFokkerPlanckConstraint(
        "p",
        domain.component(),
        drift=drift,
        evolution_var="t",
        sampling=phx.domain.PointSampling(16, layout=structure),
        sampling_mode="fixed",
        fixed_batch_key=jr.key(4),
    )

    assert stationary.loss({"p": density}, key=jr.key(5)) < 1e-12
    assert jnp.allclose(evolving.loss({"p": density}, key=jr.key(5)), 1.0)


def test_named_diffusion_field_is_jointly_optimized_by_functional_solver():
    domain, structure = _spacetime()
    truth = 0.7
    observable = domain.Function("x", "t")(lambda x, t: x[0] ** 2 - truth**2 * t)
    drift = domain.Function("x", "t")(lambda x, t: jnp.asarray([0.0]))
    scale = domain.Parameter(0.2)
    diffusion = scale * jnp.ones((1, 1))
    constraint = phx.constraints.ContinuousKolmogorovConstraint(
        "u",
        domain.component(),
        drift=drift,
        evolution_var="t",
        diffusion="sigma",
        sampling=phx.domain.PointSampling(24, layout=structure),
        sampling_mode="fixed",
        fixed_batch_key=jr.key(6),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": observable, "sigma": diffusion},
        constraints=[constraint],
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
    domain, structure = _spacetime()
    observable = domain.Function("x", "t")(lambda x, t: x[0] ** 2 - t)
    drift = domain.Function("x", "t")(lambda x, t: jnp.asarray([0.0]))
    diffusion = domain.Function("x", "t")(lambda x, t: jnp.asarray([[1.0]]))

    def make(mode):
        return phx.constraints.ContinuousKolmogorovConstraint(
            "u",
            domain.component(),
            drift=drift,
            evolution_var="t",
            diffusion=diffusion,
            sampling=phx.domain.PointSampling(12, layout=structure),
            sampling_mode=mode,
            fixed_batch_key=jr.key(9),
        )

    fixed = make("fixed")
    resampled = make("resample")
    fixed_a, fixed_b = fixed.sample(key=jr.key(10)), fixed.sample(key=jr.key(11))
    random_a = resampled.sample(key=jr.key(10))
    random_b = resampled.sample(key=jr.key(11))

    assert jnp.array_equal(fixed_a.points["x"].data, fixed_b.points["x"].data)
    assert not jnp.array_equal(random_a.points["x"].data, random_b.points["x"].data)
    assert fixed.loss({"u": observable}, key=jr.key(12)) < 1e-12


def test_fokker_planck_constraint_composes_with_explicit_density_normalization():
    domain = phx.domain.Interval1d(-1.0, 1.0)
    structure = phx.domain.SampleLayout((("x",),))
    density = domain.Function("x")(lambda x: jnp.asarray(0.5))
    drift = domain.Function("x")(lambda x: jnp.asarray([0.0]))
    dynamics = phx.constraints.ContinuousFokkerPlanckConstraint(
        "p",
        domain.component(),
        drift=drift,
        evolution_var=None,
        sampling=phx.domain.GridSampling({"x": phx.domain.LegendreAxisSpec(12)}),
        sampling_mode="fixed",
    )
    normalization = phx.constraints.ContinuousIntegralInteriorConstraint(
        "p",
        domain,
        lambda p: p,
        sampling=phx.domain.GridSampling({"x": phx.domain.LegendreAxisSpec(12)}),
        equal_to=1.0,
    )
    solver = phx.solver.FunctionalSolver(
        functions={"p": density},
        constraints=[dynamics, normalization],
    )

    assert solver.loss(key=jr.key(13)) < 1e-12


def test_probability_flux_boundary_constraint_enforces_reflecting_current():
    domain = phx.domain.Interval1d(-2.0, 2.0)
    component = domain.component({"x": phx.domain.Boundary()})
    structure = phx.domain.SampleLayout((("x",),))
    theta, sigma = 0.8, 0.6
    density = domain.Function("x")(lambda x: jnp.exp(-theta * x[0] ** 2 / sigma**2))
    drift = domain.Function("x")(lambda x: jnp.asarray([-theta * x[0]]))
    diffusion = domain.Function("x")(lambda x: jnp.asarray([[sigma]]))
    constraint = phx.constraints.ContinuousProbabilityFluxBoundaryConstraint(
        "p",
        component,
        drift=drift,
        diffusion=diffusion,
        sampling=phx.domain.PointSampling(16, layout=structure),
        sampling_mode="fixed",
        fixed_batch_key=jr.key(14),
    )

    assert constraint.loss({"p": density}, key=jr.key(15)) < 1e-12


def test_pointwise_spde_residual_rejects_rough_or_non_strong_solution_concepts():
    domain = phx.domain.Interval1d(0.0, 1.0)
    structure = phx.domain.SampleLayout((("x",),))
    rough = phx.stochastic.SPDESolutionSpec(
        "mild",
        noise_regularization="space_time_white",
        cutoff_id="fourier:64",
    )

    with pytest.raises(ValueError, match="pointwise strong residual"):
        phx.constraints.ContinuousPointwiseInteriorConstraint(
            "u",
            domain,
            lambda u: u,
            sampling=phx.domain.PointSampling(8, layout=structure),
            solution_spec=rough,
        )

    regularized = phx.stochastic.SPDESolutionSpec(
        "strong",
        noise_regularization="finite_rank",
        cutoff_id="fourier:64",
    )
    constraint = phx.constraints.ContinuousPointwiseInteriorConstraint(
        "u",
        domain,
        lambda u: u,
        sampling=phx.domain.PointSampling(8, layout=structure),
        solution_spec=regularized,
    )
    assert isinstance(constraint, phx.constraints.FunctionalConstraint)
