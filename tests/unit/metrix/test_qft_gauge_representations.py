#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.metrix import SpecialUnitaryGroup, UnitaryGroup
from phydrax.metrix._gauge_representation import (
    AdjointGaugeRepresentation,
    FundamentalGaugeRepresentation,
    U1ChargeRepresentation,
)


def test_u1_charge_and_fundamental_actions_are_exact_representations():
    u1 = UnitaryGroup(1)
    charged = U1ChargeRepresentation(u1, -2)
    phase = jnp.asarray([[jnp.exp(0.3j)]])
    value = jnp.asarray([2.0 - 0.5j])
    assert jnp.allclose(charged.apply(phase, value), phase[0, 0] ** -2 * value)

    su2 = SpecialUnitaryGroup(2)
    fundamental = FundamentalGaugeRepresentation(su2)
    left = su2.exp(su2.hat(jnp.asarray([0.2, -0.1, 0.05])))
    right = su2.exp(su2.hat(jnp.asarray([-0.03, 0.07, 0.11])))
    vector = jnp.asarray([0.4 + 0.2j, -0.1 + 0.3j])
    assert jnp.allclose(
        fundamental.apply(su2.compose(left, right), vector),
        fundamental.apply(left, fundamental.apply(right, vector)),
    )


def test_adjoint_action_preserves_lie_bracket_and_generator_dimension():
    group = SpecialUnitaryGroup(3)
    adjoint = AdjointGaugeRepresentation(group)
    element = group.exp(
        group.hat(jnp.asarray([0.1, -0.05, 0.03, 0.02, -0.04, 0.01, 0.06, -0.02]))
    )
    left = jnp.arange(8.0) / 20.0
    right = jnp.flip(left) / 7.0
    bracket = group.vee(group.lie_bracket(group.hat(left), group.hat(right)))
    transformed_bracket = adjoint.apply(element, bracket)
    transformed_inputs = group.vee(
        group.lie_bracket(
            group.hat(adjoint.apply(element, left)),
            group.hat(adjoint.apply(element, right)),
        )
    )

    assert adjoint.matrix(element).shape == (8, 8)
    assert adjoint.generators().shape == (8, 8, 8)
    assert jnp.allclose(transformed_bracket, transformed_inputs, atol=1e-5)
