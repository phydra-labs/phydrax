import coordax as cx
import jax.numpy as jnp

import phydrax as phx
from phydrax._frozendict import frozendict


def _point(x, time):
    return frozendict(
        {
            "x": cx.Field(jnp.asarray([x]), dims=(None,)),
            "t": cx.Field(jnp.asarray(time), dims=()),
        }
    )


def _unit_kernel(lag):
    return jnp.ones_like(lag)


def test_integral_heat_residual_includes_nonzero_initial_field():
    domain = phx.domain.Interval1d(0.0, 1.0) @ phx.domain.TimeInterval(0.0, 1.0)
    solution = domain.Function("x", "t")(
        lambda x, time: jnp.exp(-(jnp.pi**2) * time) * jnp.sin(jnp.pi * x[0])
    )
    initial = domain.Function("x")(lambda x: jnp.sin(jnp.pi * x[0]))
    right_hand_side = phx.operators.laplacian(solution, var="x")
    history = phx.operators.time_convolution(
        _unit_kernel,
        right_hand_side,
        rule=phx.integration.GaussLegendreRule(64),
    )
    residual = solution - initial - history
    omitted_initial = solution - history
    points = _point(0.5, 0.4)

    assert jnp.allclose(residual(points).data, 0.0, atol=2e-11)
    assert jnp.allclose(omitted_initial(points).data, 1.0, atol=2e-11)


def test_integral_residual_equals_integrated_strong_residual():
    domain = phx.domain.Interval1d(0.0, 1.0) @ phx.domain.TimeInterval(0.0, 1.0)
    function = domain.Function("x", "t")(lambda x, time: time**2 * jnp.sin(jnp.pi * x[0]))
    initial = domain.Function("x")(lambda x: jnp.zeros_like(x[0]))
    right_hand_side = phx.operators.laplacian(function, var="x")
    rule = phx.integration.GaussLegendreRule(64)
    integral_residual = (
        function
        - initial
        - phx.operators.time_convolution(
            _unit_kernel,
            right_hand_side,
            rule=rule,
        )
    )
    strong_residual = phx.operators.dt(function, var="t") - right_hand_side
    integrated_strong = phx.operators.time_convolution(
        _unit_kernel,
        strong_residual,
        rule=rule,
    )
    points = _point(0.37, 0.6)

    assert jnp.allclose(
        integral_residual(points).data,
        integrated_strong(points).data,
        atol=2e-11,
    )
