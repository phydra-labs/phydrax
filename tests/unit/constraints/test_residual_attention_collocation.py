#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx


def _interval_term(policy, *, scale=1.0, zero=False, trainable=False):
    domain = phx.domain.Interval1d(0.0, 1.0)
    component = domain.component()
    if zero:
        residual = domain.Function()(0.0)
    else:

        @domain.Function("x")
        def residual(x):
            return scale * x[0] ** 2

    condition = phx.conditions.Residual(
        "u",
        component,
        lambda field: field if trainable else residual,
    )
    source = phx.integration.adaptive(
        phx.integration.mean_over(component),
        phx.domain.PointSampling(
            16,
            layout=phx.domain.SampleLayout((("x",),)),
            design="uniform",
        ),
        policy,
    )
    term = phx.terms.ResidualPenalty(condition, source)
    if trainable:
        model = phx.nn.models.MLP(
            in_size=1,
            out_size="scalar",
            width_size=4,
            depth=1,
            key=jr.key(0),
        )
        functions = {"u": domain.Model("x")(model)}
    else:
        functions = {"u": domain.Function()(0.0)}
    return domain, term, functions


def _coordinates(population):
    return jnp.asarray(population.batch.points["x"].data).reshape((-1,))


def test_residual_attention_preserves_points_and_global_weight_mass():
    policy = phx.sampling.collocation.ResidualAttentionCollocation(
        refresh_every=1,
        decay=0.0,
        uniform_fraction=0.0,
        minimum_ess_fraction=0.1,
    )
    _domain, term, functions = _interval_term(policy)
    initial = policy.initialize(term, key=jr.key(1))
    refreshed = policy.refresh(term, functions, initial, key=jr.key(2), iter_=1)
    weights = jnp.asarray(refreshed.weight.data)
    coordinates = _coordinates(refreshed)

    assert jnp.array_equal(_coordinates(initial), coordinates)
    assert jnp.allclose(jnp.mean(weights), 1.0)
    assert jnp.argmax(weights) == jnp.argmax(coordinates)
    assert int(refreshed.refresh_count) == 1
    assert int(refreshed.last_refresh) == 1
    assert refreshed.effective_sample_size <= coordinates.size


def test_zero_residual_attention_remains_uniform_and_finite():
    policy = phx.sampling.collocation.ResidualAttentionCollocation(
        decay=0.0,
        minimum_ess_fraction=0.5,
    )
    _domain, term, functions = _interval_term(policy, zero=True)
    initial = policy.initialize(term, key=jr.key(3))
    refreshed = policy.refresh(term, functions, initial, key=jr.key(4), iter_=1)

    assert jnp.allclose(refreshed.weight.data, 1.0)
    assert jnp.all(jnp.isfinite(refreshed.probability.data))
    assert jnp.isclose(refreshed.effective_sample_size, 16.0)


def test_attention_is_invariant_to_residual_units_and_enforces_ess_guard():
    policy = phx.sampling.collocation.ResidualAttentionCollocation(
        decay=0.0,
        uniform_fraction=0.0,
        minimum_ess_fraction=0.75,
    )
    _domain, base_term, base_functions = _interval_term(policy, scale=1.0)
    _domain, scaled_term, scaled_functions = _interval_term(policy, scale=1_000.0)
    base = policy.initialize(base_term, key=jr.key(5))
    scaled = policy.initialize(scaled_term, key=jr.key(5))
    base = policy.refresh(base_term, base_functions, base, key=jr.key(6), iter_=1)
    scaled = policy.refresh(
        scaled_term,
        scaled_functions,
        scaled,
        key=jr.key(6),
        iter_=1,
    )

    assert jnp.allclose(base.weight.data, scaled.weight.data, rtol=1e-6, atol=1e-7)
    assert base.effective_sample_size / 16.0 >= 0.75 - 1e-12
    assert bool(base.ess_guard_triggered)


def test_attention_support_is_conditional_and_fixed_support_rejects_anchors():
    policy = phx.sampling.collocation.ResidualAttentionCollocation()
    support = phx.sampling.collocation.collocation_policy_support(policy)

    assert support.name == "residual_attention"
    assert support.tier == "conditional"
    with pytest.raises(ValueError, match="does not accept coverage anchors"):
        phx.sampling.collocation.controlled_collocation(
            policy,
            anchors=phx.sampling.collocation.CoverageAnchors(0.25),
        )


def test_functional_solver_persists_attention_population_and_diagnostics():
    policy = phx.sampling.collocation.ResidualAttentionCollocation(
        refresh_every=1,
        decay=0.5,
    )
    _domain, term, functions = _interval_term(policy, trainable=True)
    solver = phx.solver.FunctionalSolver(functions=functions, terms=(term,))
    trained = solver.solve(
        num_iter=2,
        optim=optax.adam(1e-3),
        seed=8,
        jit=True,
        keep_best=False,
        log_every=0,
    )
    population = trained.collocation[0]
    metrics = policy.data_metrics(population)

    assert isinstance(
        population,
        phx.sampling.collocation.ResidualAttentionPopulation,
    )
    assert int(population.refresh_count) == 2
    assert jnp.isfinite(trained.loss(key=jr.key(9), step=3))
    assert jnp.isclose(metrics["attention_weight_mean"], 1.0)
    assert metrics["attention_effective_sample_size"] > 0.0


def test_residual_attention_validates_configuration():
    with pytest.raises(ValueError, match="decay"):
        phx.sampling.collocation.ResidualAttentionCollocation(decay=1.0)
    with pytest.raises(ValueError, match="minimum_ess_fraction"):
        phx.sampling.collocation.ResidualAttentionCollocation(minimum_ess_fraction=0.0)
