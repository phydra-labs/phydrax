#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _operators():
    return phx.applications.incompressible_flow.IncompressibleFlowOperators(
        lambda velocity, time, args: jnp.zeros_like(velocity),
        lambda rhs, gamma, time, args: rhs / gamma,
        lambda velocity, time, args: velocity,
        lambda rhs, time, args: -rhs,
        lambda pressure, time, args: pressure,
    )


def test_pressure_correction_eliminates_identity_divergence():
    flow = phx.applications.incompressible_flow
    state = flow.IncompressibleFlowState(jnp.asarray([1.0, 2.0]), jnp.zeros((2,)))
    updated, diagnostics = flow.pressure_correction_step(
        state,
        1.0,
        _operators(),
        flow.IncompressibleFlowPolicy(pressure_increment=False),
        0.0,
    )

    assert jnp.allclose(updated.velocity, 0.0)
    assert diagnostics.divergence_before > 0.0
    assert diagnostics.divergence_after == 0.0
    assert diagnostics.successful


def test_oifs_history_combination_preserves_normalized_partial_sum():
    flow = phx.applications.incompressible_flow
    history = (jnp.asarray([1.0]), jnp.asarray([3.0]), jnp.asarray([5.0]))
    coefficients = jnp.asarray([1.0, 2.0, 1.0])

    combined = flow.oifs_history_combination(history, coefficients)

    assert jnp.allclose(combined, jnp.asarray([3.0]))


def test_multirate_trace_history_predicts_linear_trace():
    history = phx.solver.DGTraceHistory.empty(3, (1,), jnp.float64)
    history = history.update(jnp.asarray([1.0]), 0.0)
    history = history.update(jnp.asarray([3.0]), 1.0)

    assert jnp.allclose(history.predict(2.0), jnp.asarray([5.0]), atol=1.0e-12)


def test_multirate_interface_flux_is_exactly_conservative():
    result = phx.solver.conservative_multirate_flux(
        jnp.asarray([2.0]),
        jnp.asarray([1.0]),
        jnp.asarray([[1.0, 0.0]]),
        lambda plus, minus, normal: plus - minus,
    )

    assert jnp.allclose(result.plus, -result.minus)
    assert result.conservation_defect == 0.0


def test_power_of_two_multirate_tick_schedule():
    plan = phx.solver.DGMultirateTracePlan(jnp.asarray([[0, 2], [1, 2]], dtype=jnp.int32))

    assert plan.ticks_per_macro_step == 4
    assert tuple(plan.active_level(tick) for tick in range(4)) == (2, 0, 1, 0)
