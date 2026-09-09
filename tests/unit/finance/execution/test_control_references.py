#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.control.stochastic._hjb import BoundedUniformGrid1D
from phydrax.control.stochastic._impulse_qvi import (
    BoundedImpulseQVIProblem,
    ImpulseQVIPlan,
    ImpulseQVIStatus,
    refine_impulse_qvi_reference,
    solve_impulse_qvi_reference,
)
from phydrax.dynamics import TimeGrid
from phydrax.finance.execution._market_making import (
    AvellanedaStoikovPlan,
    solve_avellaneda_stoikov_reference,
)


def _zero(time, state, action, args):
    del time, state, action, args
    return 0.0


def test_impulse_qvi_reports_complementarity_and_nested_refinement():
    grid = BoundedUniformGrid1D(-1.0, 1.0, 9)
    time_grid = TimeGrid(jnp.linspace(0.0, 0.2, 5), time_id="qvi-grid")
    terminal = np.asarray(grid.points) ** 2
    boundary = np.ones((time_grid.num_times, 2))
    problem = BoundedImpulseQVIProblem(
        grid,
        time_grid,
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
        terminal,
        boundary,
        _zero,
        _zero,
        _zero,
        lambda time, state, impulse, args: 0.0,
        lambda time, state, impulse, args: 0.1,
        problem_id="impulse-liquidation",
    )
    plan = ImpulseQVIPlan(
        residual_tolerance=1.0e-10,
        refinement_absolute_tolerance=1.0e-10,
        refinement_relative_tolerance=1.0e-10,
        plan_id="qvi-plan",
    )

    result = solve_impulse_qvi_reference(problem, plan)
    assert bool(result.successful)
    assert int(result.status) == int(ImpulseQVIStatus.SUCCESS_DISCRETE_QVI_REFERENCE)
    assert result.evidence.maximum_complementarity_residual < 1.0e-12
    assert bool(result.evidence.refinement_passed)
    assert bool(result.intervention_region[-1, 1])
    assert not bool(result.intervention_region[-1, grid.num_points // 2])
    np.testing.assert_allclose(
        jnp.minimum(result.continuation_gap, result.intervention_gap), 0.0
    )

    refinement = refine_impulse_qvi_reference(problem, plan)
    assert bool(refinement.passed)
    assert refinement.refined_values.shape == (
        4 * (time_grid.num_times - 1) + 1,
        2 * (grid.num_points - 1) + 1,
    )


def test_avellaneda_stoikov_inventory_skews_reservation_price_downward():
    time_grid = TimeGrid(jnp.asarray([0.0, 0.5, 1.0]), time_id="as-grid")
    plan = AvellanedaStoikovPlan(
        risk_aversion=0.1,
        volatility=0.2,
        arrival_scale=5.0,
        arrival_decay=1.5,
        inventory_bound=2.0,
        plan_id="as-plan",
    )
    result = solve_avellaneda_stoikov_reference(
        plan, time_grid, jnp.asarray([-1.0, 0.0, 1.0])
    )

    assert bool(result.finite)
    assert bool(result.inventory_skew_passed)
    assert result.reservation_price_adjustment[0, 0] > 0.0
    assert result.reservation_price_adjustment[0, 2] < 0.0
    assert result.bid_offsets[0, 2] > result.bid_offsets[0, 0]
    assert result.ask_offsets[0, 2] < result.ask_offsets[0, 0]
