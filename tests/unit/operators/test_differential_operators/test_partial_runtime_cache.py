#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.domain import TimeInterval
from phydrax.operators.differential import dt_n
from phydrax.operators.differential._runtime import derivative_runtime_context


def test_partial_n_runtime_cache_reuses_eval_within_context():
    dom = TimeInterval(0.0, 1.0)
    calls = {"count": 0}

    @dom.Function("t")
    def f(t):
        calls["count"] += 1
        return t**3

    d2 = dt_n(f, var="t", order=2, backend="jet")
    t0 = jnp.asarray(0.25)

    with derivative_runtime_context():
        _ = jnp.asarray(d2.func(t0))
        first = int(calls["count"])
        _ = jnp.asarray(d2.func(t0))
        second = int(calls["count"])

    assert second == first


def test_partial_n_runtime_cache_is_not_global():
    dom = TimeInterval(0.0, 1.0)
    calls = {"count": 0}

    @dom.Function("t")
    def f(t):
        calls["count"] += 1
        return t**3

    d2 = dt_n(f, var="t", order=2, backend="jet")
    t0 = jnp.asarray(0.25)

    _ = jnp.asarray(d2.func(t0))
    first = int(calls["count"])
    _ = jnp.asarray(d2.func(t0))
    second = int(calls["count"])

    assert second > first
