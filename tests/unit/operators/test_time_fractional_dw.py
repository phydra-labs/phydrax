#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp
import pytest

from phydrax._frozendict import frozendict
from phydrax.domain import Interval1d, TimeInterval
from phydrax.operators.differential import (
    caputo_time_fractional,
    caputo_time_fractional_dw,
)


def test_caputo_time_fractional_dw_time_only_smoke():
    dom = TimeInterval(0.0, 2.0)

    @dom.Function("t")
    def u(t):
        return jnp.sin(t)

    D = caputo_time_fractional_dw(u, alpha=1.5)
    y = jnp.asarray(
        D(frozendict({"t": cx.Field(jnp.array(1.0), dims=())}), key=jr.key(0)).data
    )
    assert jnp.ndim(y) == 0
    assert jnp.isfinite(y)


def test_caputo_time_fractional_dw_broadcasts_over_space(sample_batch):
    dom = Interval1d(-1.0, 1.0) @ TimeInterval(0.0, 2.0)

    @dom.Function("t")
    def u(t):
        return jnp.cos(t)

    D = caputo_time_fractional_dw(u, alpha=1.25, M=64)
    component = dom.component()
    batch = sample_batch(component, blocks=(("x",), ("t",)), num_points=(2, 5), key=1)
    Y = jnp.asarray(D(batch, key=jr.key(1)).data)
    assert Y.shape == (2, 5)
    assert jnp.all(jnp.isfinite(Y))


@pytest.mark.parametrize(
    ("alpha", "power", "mode", "atol"),
    [
        (0.25, 2.0, "gj", 2e-11),
        (0.75, 2.0, "gj", 2e-11),
        (1.25, 3.0, "gj", 2e-10),
        (1.75, 3.0, "gj", 2e-10),
    ],
)
def test_caputo_time_fractional_matches_power_law(alpha, power, mode, atol):
    start = 0.3
    endpoint = 1.1
    domain = TimeInterval(start, 1.5)
    function = domain.Function("t")(lambda time: (time - start) ** power)
    derivative = caputo_time_fractional(
        function,
        alpha=alpha,
        mode=mode,
        order=128,
    )
    points = frozendict({"t": cx.Field(jnp.array(endpoint), dims=())})
    expected = (
        jsp.gamma(power + 1.0)
        / jsp.gamma(power + 1.0 - alpha)
        * (endpoint - start) ** (power - alpha)
    )

    assert jnp.allclose(derivative(points).data, expected, atol=atol, rtol=0.0)


def test_caputo_gauss_legendre_converges_under_rule_refinement():
    alpha = 0.75
    power = 2.0
    endpoint = 0.8
    domain = TimeInterval(0.0, 1.0)
    function = domain.Function("t")(lambda time: time**power)
    point = frozendict({"t": cx.Field(jnp.array(endpoint), dims=())})
    expected = (
        jsp.gamma(power + 1.0)
        / jsp.gamma(power + 1.0 - alpha)
        * endpoint ** (power - alpha)
    )
    coarse = caputo_time_fractional(
        function,
        alpha=alpha,
        mode="gl",
        order=16,
    )(point).data
    fine = caputo_time_fractional(
        function,
        alpha=alpha,
        mode="gl",
        order=128,
    )(point).data

    assert jnp.abs(fine - expected) < jnp.abs(coarse - expected)


@pytest.mark.parametrize("alpha", [0.4, 1.4])
def test_caputo_time_fractional_is_exact_zero_at_initial_time(alpha):
    domain = TimeInterval(0.2, 1.0)
    function = domain.Function("t")(lambda time: jnp.sin(time))
    derivative = caputo_time_fractional(function, alpha=alpha, order=64)
    start = frozendict({"t": cx.Field(jnp.array(0.2), dims=())})

    assert jnp.array_equal(derivative(start).data, jnp.array(0.0))
    assert derivative.metadata["fractional_randomized"] is False


def test_caputo_time_fractional_rejects_hidden_randomized_mode():
    domain = TimeInterval(0.0, 1.0)
    function = domain.Function("t")(lambda time: time**2)

    with pytest.raises(ValueError, match="mode"):
        caputo_time_fractional(function, alpha=1.5, mode="qmc")
