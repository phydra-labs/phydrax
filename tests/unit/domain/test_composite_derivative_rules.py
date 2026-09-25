#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._strict import StrictModule
from phydrax.domain import DerivativeRule, Interval1d
from phydrax.operators.differential import partial_n
from phydrax.operators.differential._hooks import blend_with_gate


class _ScaledSquare(StrictModule, phx.ParameterOwner):
    scale: jax.Array

    def __call__(self, x, /):
        return self.scale * x[0] ** 2


_DOMAIN = Interval1d(0.0, 1.0)


def _trainable(scale: float, /):
    return _DOMAIN.Function("x")(_ScaledSquare(jnp.asarray(scale)))


def _sum(u, /):
    return u + _DOMAIN.Function("x")(lambda x: jnp.sin(x[0]))


def _gated_blend(u, /):
    # Its derivative rule differentiates the operands even when they are trainable.
    return blend_with_gate(
        u,
        _DOMAIN.Function("x")(lambda x: jnp.sin(x[0])),
        _DOMAIN.Function("x")(lambda x: x[0]),
    )


_COMPOSITES = pytest.mark.parametrize("build", [_sum, _gated_blend], ids=["sum", "blend"])


@_COMPOSITES
def test_composite_derivatives_follow_updated_parameters(build):
    parameters, state, fixed = phx.partition_parameters(build(_trainable(1.0)))
    trained = phx.combine_parameters(
        jax.tree_util.tree_map(lambda p: 3.0 * p, parameters), state, fixed
    )
    fresh = build(_trainable(3.0))

    assert trained.derivative_rule is not None
    x = jnp.asarray([0.4])
    for order in (1, 2):
        got = partial_n(trained, var="x", order=order).func(x)
        want = partial_n(fresh, var="x", order=order).func(x)
        assert jnp.allclose(got, want)
    assert not jnp.allclose(
        partial_n(trained, var="x", order=1).func(x),
        partial_n(build(_trainable(1.0)), var="x", order=1).func(x),
    )


@_COMPOSITES
def test_composite_holds_each_parameter_once_and_no_rule_copies(build):
    u = _trainable(2.0)
    (scale,) = jax.tree_util.tree_leaves(phx.partition_parameters(u)[0])
    leaves = jax.tree_util.tree_leaves(build(u))

    assert sum(leaf is scale for leaf in leaves) == 1
    assert not any(isinstance(leaf, DerivativeRule) for leaf in leaves)


@_COMPOSITES
def test_trained_composite_passes_training_preflight(build):
    parameters, state, fixed = phx.partition_parameters(build(_trainable(1.0)))
    trained = phx.combine_parameters(
        jax.tree_util.tree_map(lambda p: p + 0.5, parameters), state, fixed
    )

    resolution = phx.require_parameter_roles(trained, context="composite field")
    assert len(resolution.selectable_paths) == 1
