#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _grid(points=33, *, lower=0.0, upper=1.0):
    return phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(points),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[lower], [upper]]))


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_diagonal_norm_sbp_families_certify_identity_and_closure_exactness(order):
    grid = _grid(41)
    sbp = phx.discretization.SBPDerivativePlan(
        grid,
        "x",
        interior_order=order,
    ).prepare()
    x = grid.axes[0].nodes
    degree = order // 2
    values = x**degree
    exact = degree * x ** (degree - 1) if degree > 0 else jnp.zeros_like(x)

    derivative = sbp.operator.mv(values)

    assert sbp.stability_report.passed
    assert sbp.operator.consistency_report.passed
    assert sbp.operator.stencil_set.interior_accuracy_order == order
    assert sbp.operator.stencil_set.closure_accuracy_order == order // 2
    assert jnp.all(sbp.norm_weights > 0.0)
    assert jnp.max(jnp.abs(sbp.identity_residual())) < 5e-11
    np.testing.assert_allclose(derivative, exact, rtol=2e-8, atol=2e-8)


def test_compatible_second_derivative_annihilates_constants_and_is_dissipative_on_zero_trace():
    grid = _grid(65)
    sbp = phx.discretization.SBPDerivativePlan(
        grid,
        "x",
        interior_order=6,
    ).prepare()
    x = grid.axes[0].nodes
    state = jnp.sin(jnp.pi * x)
    second = sbp.second_derivative

    constant = second.mv(jnp.ones_like(state))
    action = second.mv(state)
    energy_rate = jnp.sum(sbp.norm_weights * state * action)
    left = jnp.vdot(state, second.mv(state))
    right = jnp.vdot(second.transpose_mv(state), state)

    np.testing.assert_allclose(constant, 0.0, rtol=0.0, atol=2e-10)
    assert energy_rate < 0.0
    np.testing.assert_allclose(left, right, rtol=2e-11, atol=2e-11)


def test_advection_inflow_sat_makes_discrete_energy_nonincreasing():
    grid = _grid(49)
    sbp = phx.discretization.SBPDerivativePlan(
        grid,
        "x",
        interior_order=4,
    ).prepare()
    sat = phx.discretization.SATBoundaryPlan.advection_inflow(sbp, 2.0)
    x = grid.axes[0].nodes
    state = 0.7 + jnp.sin(3.0 * jnp.pi * x)

    rhs = -2.0 * sbp.operator.mv(state) + sat.correction(state, 0.0, 0.0)
    energy_rate = 2.0 * jnp.sum(sbp.norm_weights * state * rhs)

    assert sat.stability_report.passed
    assert energy_rate <= 2e-10


def test_conforming_central_and_upwind_sat_interfaces_have_expected_energy():
    left = phx.discretization.SBPDerivativePlan(
        _grid(33, lower=0.0, upper=0.5),
        "x",
        interior_order=4,
    ).prepare()
    right = phx.discretization.SBPDerivativePlan(
        _grid(33, lower=0.5, upper=1.0),
        "x",
        interior_order=4,
    ).prepare()
    left_state = jnp.sin(jnp.pi * left.grid.axes[0].nodes)
    right_state = 0.3 * jnp.sin(jnp.pi * (1.0 - right.grid.axes[0].nodes))

    rates = []
    for flux in ("central", "upwind"):
        interface = phx.discretization.SATInterfacePlan(
            left,
            right,
            1.0,
            flux=flux,
        )
        left_sat, right_sat = interface.corrections(left_state, right_state)
        left_rhs = -left.operator.mv(left_state) + left_sat
        right_rhs = -right.operator.mv(right_state) + right_sat
        rates.append(
            2.0
            * (
                jnp.sum(left.norm_weights * left_state * left_rhs)
                + jnp.sum(right.norm_weights * right_state * right_rhs)
            )
        )
        assert interface.stability_report.passed

    np.testing.assert_allclose(rates[0], 0.0, rtol=0.0, atol=2e-10)
    assert rates[1] <= rates[0] + 2e-10


def test_sbp_tensor_axis_preserves_other_entity_axes_and_norm_measure():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(33),
            phx.discretization.UniformCellAxisSpec(5),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, -1.0], [1.0, 2.0]]))
    sbp = phx.discretization.SBPDerivativePlan(
        grid,
        "x",
        interior_order=4,
    ).prepare()
    x = grid.axes[0].nodes[:, None]
    values = jnp.broadcast_to(x**2, grid.shape)

    derivative = sbp.operator.mv(values)

    assert derivative.shape == grid.shape
    assert sbp.norm_weights.shape == grid.shape
    np.testing.assert_allclose(
        derivative,
        jnp.broadcast_to(2.0 * x, grid.shape),
        rtol=2e-10,
        atol=2e-10,
    )


def test_sbp_family_rejects_grid_too_short_for_boundary_closures():
    with pytest.raises(ValueError, match="requires at least"):
        phx.discretization.SBPDerivativePlan(
            _grid(15),
            "x",
            interior_order=8,
        ).prepare()
