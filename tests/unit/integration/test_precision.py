#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_fixed_integration_records_output_and_decision_precision():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    function = domain.Function("x")(lambda x: x**2)
    precision = phx.integration.IntegrationPrecisionPolicy(
        evaluation_dtype="float32",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float32",
    )
    estimate = phx.integration.integrate(
        function,
        phx.integration.over(domain.component()),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(12)),
        precision=precision,
    )

    assert estimate.value.data.dtype == jnp.float32
    assert estimate.precision_evidence is not None
    assert dict(estimate.precision_evidence.observed) == {
        "accumulation": "float64",
        "certification": "float64",
        "compute": "float32",
        "output": "float32",
    }
    assert jnp.allclose(estimate.value.data, 1.0 / 3.0, atol=2e-7)


def test_weighted_integration_uses_explicit_accumulation_dtype():
    samples = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    target = phx.integration.weighted(
        samples,
        jnp.asarray([0.0, -100.0, -200.0], dtype=jnp.float32),
        normalized=True,
    )
    precision = phx.integration.IntegrationPrecisionPolicy(
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float64",
    )
    estimate = phx.integration.integrate(lambda value: value, target, precision=precision)

    assert estimate.value.data.dtype == jnp.float64
    assert estimate.diagnostics.weights.weight_ess.dtype == jnp.float64


def test_cubature_rule_identity_is_independent_of_execution_precision():
    rule = phx.integration.CubatureRule("triangle", 5)
    first = rule.rule_id
    target = phx.integration.mapped(
        rule,
        lambda reference: reference,
        lambda reference: jnp.ones((reference.shape[0],)),
    )
    precision = phx.integration.IntegrationPrecisionPolicy(
        evaluation_dtype="float32",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float64",
    )
    estimate = phx.integration.integrate(
        lambda points: jnp.ones(points.shape[0], dtype=jnp.float32),
        target,
        phx.integration.CellQuadraturePlan(rule),
        precision=precision,
    )

    assert rule.rule_id == first
    assert estimate.value.data.dtype == jnp.float64


def test_adaptive_interval_keeps_error_and_partition_decisions_high_precision():
    domain = phx.domain.ScalarInterval(-1.0, 2.0, label="x")
    function = domain.Function("x")(lambda x: x**4 - 2.0 * x + 1.0)
    precision = phx.integration.IntegrationPrecisionPolicy(
        evaluation_dtype="float32",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float32",
    )
    estimate = phx.integration.integrate(
        function,
        phx.integration.over(domain.component()),
        phx.integration.AdaptiveQuadraturePlan(
            absolute_tolerance=1e-7,
            relative_tolerance=1e-7,
            max_intervals=32,
            collect_partition=True,
        ),
        precision=precision,
    )

    assert estimate.value.data.dtype == jnp.float32
    assert estimate.error_estimate.dtype == jnp.float64
    assert estimate.diagnostics.estimated_error.dtype == jnp.float64
    assert estimate.diagnostics.partition is not None
    assert estimate.diagnostics.partition.integral_estimates.dtype == jnp.float64
    assert estimate.diagnostics.partition.estimated_errors.dtype == jnp.float64


def test_monte_carlo_statistics_follow_accumulation_and_decision_precision():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    function = domain.Function("x")(lambda x: x**2)
    precision = phx.integration.IntegrationPrecisionPolicy(
        evaluation_dtype="float32",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float32",
    )
    estimate = phx.integration.integrate(
        function,
        phx.integration.over(domain.component()),
        phx.integration.MonteCarloPlan(256),
        key=jr.key(7),
        precision=precision,
    )

    assert estimate.value.data.dtype == jnp.float32
    assert estimate.error_estimate.dtype == jnp.float64
    assert estimate.diagnostics.standard_error.dtype == jnp.float64
