import jax.numpy as jnp

import phydrax as phx


def test_operator_flow_matching_metric_uses_query_quadrature_and_mask():
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=jnp.asarray([[0.0], [0.5], [1.0]]),
        quadrature_weights=jnp.asarray([0.2, 0.3, 0.5]),
        mask=jnp.asarray([True, False, True]),
    )
    metric = phx.nn.operator.training.OperatorFlowMatchingMetric(
        query,
        phx.nn.operator.OperatorOutputSpec("scalar"),
    )
    state = jnp.zeros((3,))
    prediction = jnp.asarray([1.0, 100.0, 2.0])
    target = jnp.zeros((3,))

    assert jnp.allclose(metric(state, prediction, target), 2.2)


def test_operator_flow_matching_metric_applies_channel_geometry():
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=jnp.asarray([[0.0], [1.0]]),
        quadrature_weights=jnp.asarray([0.25, 0.75]),
    )
    channel_metric = jnp.diag(jnp.asarray([2.0, 0.5]))
    metric = phx.nn.operator.training.OperatorFlowMatchingMetric(
        query,
        phx.nn.operator.OperatorOutputSpec(2, component_names=("u", "v")),
        channel_metric=channel_metric,
    )
    residual = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])

    expected = 0.25 * (2.0 * 1.0**2 + 0.5 * 2.0**2) + 0.75 * (2.0 * 3.0**2 + 0.5 * 4.0**2)
    assert jnp.allclose(
        metric(jnp.zeros_like(residual), residual, jnp.zeros_like(residual)),
        expected,
    )


def test_operator_flow_matching_metric_integrates_with_generic_term():
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=jnp.linspace(0.0, 1.0, 4)[:, None],
        quadrature_weights=jnp.full((4,), 0.25),
    )
    metric = phx.nn.operator.training.OperatorFlowMatchingMetric(
        query,
        phx.nn.operator.OperatorOutputSpec("scalar"),
    )
    source = jnp.zeros((8, 4))
    target = jnp.ones((8, 4))
    endpoints = phx.transport.EndpointCouplingSample(
        source=source,
        target=target,
        source_indices=jnp.arange(8),
        target_indices=jnp.arange(8),
        valid=jnp.ones((8,), dtype=bool),
        log_weights=jnp.zeros((8,)),
        coupling_id="operator-paired",
        provenance="unit-test",
    )
    state_domain = phx.domain.HyperRectangle(
        jnp.full((4,), -2.0), jnp.full((4,), 2.0), label="x"
    )
    domain = state_domain @ phx.domain.TimeInterval(0.0, 1.0)
    velocity = domain.Function("x", "t")(lambda state, time: jnp.ones_like(state))
    term = phx.terms.FlowMatchingTerm(
        "velocity",
        endpoints,
        phx.transport.LinearEndpointInterpolant((4,)),
        metric=metric,
    )

    assert jnp.allclose(term.loss({"velocity": velocity}), 0.0)
