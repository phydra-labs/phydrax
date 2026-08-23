import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _problem():
    domain = phx.domain.Interval1d(0.0, 1.0)
    component = domain.component()
    condition = phx.conditions.Moment(
        "u",
        component,
        lambda function: function,
        target=0.0,
    )
    target = phx.integration.mean_over(component)
    source = phx.integration.per_step(
        target,
        phx.integration.MonteCarloPlan(2),
    )
    function = domain.Function("x")(lambda point: point[0])
    return component, condition, target, source, {"u": function}


def _realization(component, target, value):
    points = component.points({"x": jnp.full((2, 1), value)})
    return phx.integration.from_samples(target, points)


def _batch(component, target, left, right=None):
    left_realizations = tuple(_realization(component, target, value) for value in left)
    right_realizations = (
        None
        if right is None
        else tuple(_realization(component, target, value) for value in right)
    )
    return phx.terms.RandomizedMomentBatch(
        left_realizations,
        right_realizations,
    )


def test_randomized_moment_u_statistic_and_plugin_have_declared_values():
    component, condition, target, source, functions = _problem()
    batch = _batch(component, target, (0.0, 1.0, 0.0, 1.0))
    unbiased = phx.terms.RandomizedMomentPenalty(
        condition,
        source,
        num_realizations=4,
        loss_mode="u_statistic",
    )
    plug_in = phx.terms.RandomizedMomentPenalty(
        condition,
        source,
        num_realizations=4,
        loss_mode="plug_in",
    )

    unbiased_value = eqx.filter_jit(
        lambda supplied: unbiased.loss(functions, batch=supplied)
    )(batch)
    plug_in_value = plug_in.loss(functions, batch=batch)
    diagnostics = unbiased.diagnostics(functions, batch=batch)

    assert jnp.allclose(unbiased_value, 1.0 / 6.0, atol=1e-12)
    assert jnp.allclose(plug_in_value, 0.25, atol=1e-12)
    assert jnp.allclose(diagnostics.plug_in_moment_norm, 0.5, atol=1e-12)
    assert diagnostics.passed
    assert len(diagnostics.integration_diagnostics) == 4


def test_randomized_moment_independent_product_uses_independent_group_mean():
    component, condition, target, source, functions = _problem()
    batch = _batch(
        component,
        target,
        (0.0, 1.0),
        right=(0.25, 0.25),
    )
    objective = phx.terms.RandomizedMomentPenalty(
        condition,
        source,
        num_realizations=2,
        loss_mode="independent_product",
    )

    assert jnp.allclose(objective.loss(functions, batch=batch), 0.125, atol=1e-12)


def test_moment_penalty_rejects_resampled_stochastic_integration():
    _component, condition, _target, source, _functions = _problem()

    with pytest.raises(ValueError, match="RandomizedMomentPenalty"):
        phx.terms.MomentPenalty(condition, source)


def test_fixed_random_realization_remains_an_explicit_moment_objective():
    component, condition, target, source, functions = _problem()
    realization = phx.integration.materialize(
        target,
        source.plan,
        key=jr.key(1),
    )
    objective = phx.terms.MomentPenalty(
        condition,
        phx.integration.fixed(realization),
    )

    assert jnp.isfinite(objective.loss(functions))


def test_randomized_moment_requires_random_per_step_source():
    component, condition, _target, _source, _functions = _problem()
    deterministic = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(4)),
    )

    with pytest.raises(ValueError, match="randomized integration plan"):
        phx.terms.RandomizedMomentPenalty(condition, deterministic)


def test_randomized_moment_batch_requires_two_equal_groups():
    component, _condition, target, _source, _functions = _problem()
    realization = _realization(component, target, 0.5)

    with pytest.raises(ValueError, match="at least two"):
        phx.terms.RandomizedMomentBatch((realization,))
    with pytest.raises(ValueError, match="equal sizes"):
        phx.terms.RandomizedMomentBatch(
            (realization, realization),
            (realization, realization, realization),
        )
