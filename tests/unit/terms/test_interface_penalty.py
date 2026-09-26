#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _interface_problem():
    domain = phx.domain.Interval1d(0.0, 1.0)
    component = domain.component()
    residual_field = domain.Function("x")(lambda x: 2.0 + 0.0 * x[0])
    level_set = domain.Function("x")(lambda x: x[0] - 0.5)
    condition = phx.conditions.Residual("u", component, lambda field: field)
    target = phx.integration.over(component)
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(64))
    source = phx.integration.fixed(phx.integration.materialize(target, plan))
    return condition, source, {"u": residual_field, "phi": level_set}


def test_implicit_interface_penalty_integrates_squared_residual_over_interface():
    condition, source, functions = _interface_problem()
    term = phx.terms.implicit_interface_penalty(
        condition, source, level_set_field="phi", width=0.25
    )

    # One unit-gradient crossing: the coarea density integrates to one.
    assert jnp.allclose(term.loss(functions, key=jr.key(0)), 4.0, rtol=1.0e-3)


def test_implicit_phase_penalty_integrates_squared_residual_over_inside_phase():
    condition, source, functions = _interface_problem()
    term = phx.terms.implicit_phase_penalty(
        condition, source, level_set_field="phi", width=0.25, phase="inside"
    )

    # The symmetric regularized indicator covers exactly half of [0, 1].
    assert jnp.allclose(term.loss(functions, key=jr.key(0)), 2.0, rtol=1.0e-10)
