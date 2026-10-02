from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _problem() -> Any:
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


def _realization(component: Any, target: Any, value: Any, *, key: Any = jr.key(0)) -> Any:
    points = component.points({"x": jnp.full((2, 1), value)})
    return phx.integration.from_samples(target, points, key=key)


def _batch(component: Any, target: Any, left: Any, right: Any = None) -> Any:
    right_count = 0 if right is None else len(right)
    keys = tuple(jr.split(jr.key(5), len(left) + right_count))
    left_realizations = tuple(
        _realization(component, target, value, key=sample_key)
        for value, sample_key in zip(left, keys[: len(left)], strict=True)
    )
    right_realizations = (
        None
        if right is None
        else tuple(
            _realization(component, target, value, key=sample_key)
            for value, sample_key in zip(right, keys[len(left) :], strict=True)
        )
    )
    return phx.terms.RandomizedMomentBatch(
        left_realizations,
        right_realizations,
        sampling_design="iid",
        right_sampling_design="iid",
    )


def test_randomized_moment_objective_scenario_1() -> None:
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
    component, condition, _target, _source, _functions = _problem()
    deterministic = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(4)),
    )

    with pytest.raises(ValueError, match="randomized integration plan"):
        phx.terms.RandomizedMomentPenalty(condition, deterministic)
    _component, condition, _target, source, functions = _problem()
    precision = phx.integration.IntegrationPrecisionPolicy(
        evaluation_dtype="float32",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float64",
    )
    objective = phx.terms.RandomizedMomentPenalty(
        condition,
        source,
        num_realizations=4,
        precision=precision,
    )
    diagnostics = objective.diagnostics(functions, key=jr.key(11))

    assert diagnostics.objective.dtype == jnp.float64
    assert dict(diagnostics.precision_evidence.observed)["accumulation"] == "float64"
    assert len(diagnostics.precision_evidence.children) == 4
    _component, condition, _target, source, _functions = _problem()

    with pytest.raises(ValueError, match="RandomizedMomentPenalty"):
        phx.terms.MomentPenalty(condition, source)
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


def test_independent_moments_allow_singleton_and_unequal_group_counts() -> None:
    component, condition, target, source, functions = _problem()
    batch = _batch(component, target, (0.5,), right=(0.25, 0.75))
    objective = phx.terms.RandomizedMomentPenalty(
        condition, source, num_realizations=1, loss_mode="independent_product"
    )
    diagnostics = objective.diagnostics(functions, batch=batch)
    assert jnp.allclose(objective.loss(functions, batch=batch), 0.25)
    assert not diagnostics.uncertainty_available
    assert jnp.isnan(diagnostics.mean_standard_error)
    assert diagnostics.passed


def test_raw_moment_law_is_unknown_and_exact_singleton_is_admitted() -> None:
    component, condition, target, source, functions = _problem()
    realization = _realization(component, target, 0.5)
    raw = phx.terms.RandomizedMomentBatch((realization, realization))
    unbiased = phx.terms.RandomizedMomentPenalty(condition, source)
    with pytest.raises(ValueError, match="iid"):
        unbiased.loss(functions, batch=raw)
    plug_in = phx.terms.RandomizedMomentPenalty(condition, source, loss_mode="plug_in")
    diagnostics = plug_in.diagnostics(functions, batch=raw)
    assert jnp.allclose(diagnostics.objective, 0.25)
    assert not diagnostics.uncertainty_available
    assert jnp.isnan(diagnostics.mean_standard_error)
    assert diagnostics.passed
    exact = phx.terms.RandomizedMomentBatch((realization,), sampling_design="exact")
    exact_diagnostics = unbiased.diagnostics(functions, batch=exact)
    assert jnp.allclose(exact_diagnostics.objective, 0.25)
    assert exact_diagnostics.uncertainty_available
    assert jnp.allclose(exact_diagnostics.mean_standard_error, 0.0)


def test_moment_finite_population_error_is_corrected_and_u_statistic_refuses_it() -> None:
    component, condition, target, source, functions = _problem()
    realizations = _batch(component, target, (0.0, 1.0)).left
    batch = phx.terms.RandomizedMomentBatch(
        realizations, sampling_design="finite_population", population_size=4
    )
    plug_in = phx.terms.RandomizedMomentPenalty(condition, source, loss_mode="plug_in")
    diagnostics = plug_in.diagnostics(functions, batch=batch)
    assert jnp.allclose(diagnostics.mean_standard_error, jnp.sqrt(0.125))
    unbiased = phx.terms.RandomizedMomentPenalty(condition, source)
    with pytest.raises(ValueError, match="independent_product"):
        unbiased.loss(functions, batch=batch)


def test_independent_moment_groups_cannot_reuse_random_keys() -> None:
    component, condition, target, source, functions = _problem()
    left = _realization(component, target, 0.5, key=jr.key(7))
    right = _realization(component, target, 0.75, key=jr.key(7))
    batch = phx.terms.RandomizedMomentBatch(
        (left,),
        (right,),
        sampling_design="iid",
        right_sampling_design="iid",
    )
    objective = phx.terms.RandomizedMomentPenalty(
        condition, source, num_realizations=1, loss_mode="independent_product"
    )
    with pytest.raises(Exception, match="distinct owner-generated"):
        objective.loss(functions, batch=batch)


@pytest.mark.parametrize("loss_mode", ("u_statistic", "independent_product"))
def test_same_sample_fitted_control_variates_refuse_unbiased_moment_losses(
    loss_mode: Any,
) -> None:
    domain = phx.domain.Interval1d(0.0, 1.0)
    component = domain.component()
    control = domain.Function("x")(lambda point: point[0])
    function = domain.Function("x")(lambda point: point[0] ** 2)
    condition = phx.conditions.Moment(
        "u", component, lambda field: field, target=1.0 / 3.0
    )
    estimator = phx.integration.ControlVariateEstimator(
        (control,), (0.5,), same_sample_asymptotic=True, regularization=0.0
    )
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.integration.MonteCarloPlan(2, control_variate=estimator),
    )
    term = phx.terms.RandomizedMomentPenalty(condition, source, loss_mode=loss_mode)
    with pytest.raises(ValueError, match="iid|known-unbiased"):
        term.loss({"u": function}, key=jr.key(17))
    plug_in = phx.terms.RandomizedMomentPenalty(condition, source, loss_mode="plug_in")
    diagnostics = plug_in.diagnostics({"u": function}, key=jr.key(17))
    assert diagnostics.passed
    assert not diagnostics.uncertainty_available
    assert jnp.isnan(diagnostics.mean_standard_error)


def test_same_sample_quadratic_control_bias_has_exact_independent_reference() -> None:
    domain = phx.domain.Interval1d(0.0, 1.0)
    component = domain.component()
    target = phx.integration.mean_over(component)
    control = domain.Function("x")(lambda point: point[0])
    function = domain.Function("x")(lambda point: point[0] ** 2)
    estimator = phx.integration.ControlVariateEstimator(
        (control,), (0.5,), same_sample_asymptotic=True, regularization=0.0
    )
    plan = phx.integration.MonteCarloPlan(2, control_variate=estimator)
    # Disjoint 2- and 3-point Gaussian rules integrate the two independent
    # uniform draws exactly without the equal-draw, rank-zero diagonal.
    left_nodes = (0.5 - 0.5 / jnp.sqrt(3.0), 0.5 + 0.5 / jnp.sqrt(3.0))
    right_nodes = (
        0.5 - 0.5 * jnp.sqrt(3.0 / 5.0),
        jnp.asarray(0.5, dtype=jnp.float64),
        0.5 + 0.5 * jnp.sqrt(3.0 / 5.0),
    )
    right_weights = (5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0)
    expectation = jnp.asarray(0.0, dtype=jnp.float64)
    for left in left_nodes:
        for right, weight in zip(right_nodes, right_weights, strict=True):
            points = component.points(
                {"x": jnp.asarray([[left], [right]], dtype=jnp.float64)}
            )
            supplied = phx.integration.from_samples(target, points)
            realization = phx.integration.IntegrationRealization(
                target, plan, supplied.batch, supplied.key
            )
            actual = phx.integration.reduce(function, realization).value.data
            # The fitted secant slope is left+right, giving this biased mean.
            reference = 0.5 * (left + right) - left * right
            assert jnp.allclose(actual, reference, atol=1e-12)
            expectation = expectation + 0.5 * weight * actual
    assert jnp.allclose(expectation, 0.25, atol=1e-12)
    assert not jnp.allclose(expectation, 1.0 / 3.0)


@pytest.mark.parametrize("fixed_coefficients", (False, True))
def test_fixed_or_independent_pilot_controls_preserve_unbiased_moment_losses(
    fixed_coefficients: bool,
) -> None:
    domain = phx.domain.Interval1d(0.0, 1.0)
    component = domain.component()
    control = domain.Function("x")(lambda point: point[0])
    function = domain.Function("x")(lambda point: 2.0 * point[0] + 2.0)
    condition = phx.conditions.Moment("u", component, lambda field: field, target=1.0)
    estimator = phx.integration.ControlVariateEstimator(
        (control,),
        (0.5,),
        coefficients=jnp.asarray([2.0]) if fixed_coefficients else None,
        pilot_samples=2,
        same_sample_asymptotic=fixed_coefficients,
        regularization=0.0,
    )
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.integration.MonteCarloPlan(
            2 if fixed_coefficients else 4, control_variate=estimator
        ),
    )
    term = phx.terms.RandomizedMomentPenalty(condition, source)
    assert jnp.allclose(term.loss({"u": function}, key=jr.key(23)), 4.0, atol=1e-10)


def test_deterministic_qmc_keys_do_not_establish_unbiased_moment_replicates() -> None:
    domain = phx.domain.Interval1d(0.0, 1.0)
    component = domain.component()
    function = domain.Function("x")(lambda point: point[0] ** 2)
    target = phx.integration.mean_over(component)
    plan = phx.integration.QuasiMonteCarloPlan(
        2, sequence="sobol", scrambled=False, num_replicates=1
    )
    estimate = phx.integration.integrate(function, target, plan)
    assert jnp.allclose(estimate.value.data, 1.0 / 8.0, atol=1e-12)
    with pytest.raises(ValueError, match="deterministic.*does not consume"):
        phx.integration.integrate(function, target, plan, key=jr.key(5))
    condition = phx.conditions.Moment(
        "u", component, lambda field: field, target=1.0 / 3.0
    )
    source = phx.integration.per_step(target, plan)
    with pytest.raises(ValueError, match="randomized integration plan"):
        phx.terms.RandomizedMomentPenalty(condition, source)


@pytest.mark.parametrize("sequence", ("sobol", "halton"))
@pytest.mark.parametrize("same_sample_fit", (False, True))
def test_randomized_qmc_admits_fixed_controls_but_refuses_self_fitting(
    sequence: Any,
    same_sample_fit: bool,
) -> None:
    domain = phx.domain.Interval1d(0.0, 1.0)
    component = domain.component()
    function = domain.Function("x")(lambda point: point[0] ** 2)
    condition = phx.conditions.Moment("u", component, lambda field: field, target=0.0)
    estimator = phx.integration.ControlVariateEstimator(
        (function,),
        (1.0 / 3.0,),
        coefficients=None if same_sample_fit else jnp.asarray([1.0]),
        same_sample_asymptotic=same_sample_fit,
        regularization=0.0,
    )
    plan = phx.integration.QuasiMonteCarloPlan(
        2,
        sequence=sequence,
        scrambled=True,
        num_replicates=1,
        control_variate=estimator,
    )
    term = phx.terms.RandomizedMomentPenalty(
        condition,
        phx.integration.per_step(phx.integration.mean_over(component), plan),
    )
    if same_sample_fit:
        with pytest.raises(ValueError, match="iid"):
            term.loss({"u": function}, key=jr.key(7))
    else:
        assert jnp.allclose(term.loss({"u": function}, key=jr.key(7)), 1.0 / 9.0)
