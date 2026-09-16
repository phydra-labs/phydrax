#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_marginal_structural_model_recovers_weighted_linear_effect() -> None:
    treatment = jnp.asarray([[False], [False], [True], [True]])
    design = jnp.stack((jnp.ones(4), treatment[:, 0]), axis=1)
    outcome = jnp.asarray((1.0, 1.0, 3.0, 3.0))
    probability = jnp.full((4, 1), 0.5)

    result = phx.causal.fit_marginal_structural_model(
        design,
        outcome,
        treatment,
        probability,
        probability,
    )

    assert bool(result.successful)
    assert jnp.allclose(result.coefficients, jnp.asarray((1.0, 2.0)), atol=1.0e-12)
    assert jnp.isclose(result.effective_sample_size, 4.0)


def test_tmle_targets_outcome_regression_and_returns_influence_curve() -> None:
    treatment = jnp.asarray((False, False, True, True))
    outcome = jnp.asarray((0.0, 0.0, 2.0, 2.0))

    result = phx.causal.tmle_ate(
        outcome,
        treatment,
        jnp.full((4,), 1.5),
        jnp.full((4,), 0.5),
        jnp.full((4,), 0.5),
    )

    assert bool(result.successful)
    assert jnp.isclose(result.treatment_effect, 2.0)
    assert jnp.isclose(jnp.mean(result.influence_curve), 0.0, atol=1.0e-12)


def test_binary_instrument_late_and_weak_instrument_status() -> None:
    strong = phx.causal.binary_instrument_late(
        jnp.asarray((0.0, 0.0, 4.0, 4.0)),
        jnp.asarray((0.0, 0.0, 1.0, 1.0)),
        jnp.asarray((False, False, True, True)),
    )

    assert bool(strong.successful)
    assert jnp.isclose(strong.late, 4.0)


def test_aalen_johansen_conserves_survival_and_incidence() -> None:
    result = phx.causal.aalen_johansen(
        jnp.asarray((1.0, 2.0, 3.0, 4.0)),
        jnp.asarray((1, 2, 0, 1)),
        cause_count=2,
    )

    total = result.survival + jnp.sum(result.cumulative_incidence, axis=1)
    assert jnp.allclose(total, 1.0, atol=1.0e-12)
    assert jnp.all(jnp.diff(result.survival) <= 0.0)


def test_transport_sensitivity_and_dynamic_regime_are_explicit() -> None:
    weights = phx.causal.transportability_weights(
        jnp.asarray((0.5, 0.75)), jnp.asarray((0.5, 0.25))
    )
    sensitivity = phx.causal.e_value(jnp.asarray(2.0))
    value = phx.causal.dynamic_regime_value(
        jnp.asarray((1.0, 3.0)),
        jnp.asarray(((False, True), (True, False))),
        jnp.asarray(((False, True), (False, False))),
        jnp.full((2, 2), 0.5),
    )

    assert jnp.all(weights > 0.0)
    assert sensitivity > 2.0
    assert jnp.isclose(value, 1.0)
