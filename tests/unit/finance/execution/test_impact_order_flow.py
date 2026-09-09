#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.dynamics import TimeGrid
from phydrax.finance.execution._impact import (
    AlmgrenChrissModel,
    diagnose_transient_manipulation,
    solve_almgren_chriss_schedule,
    transient_impact_path,
    TransientPropagatorModel,
)
from phydrax.finance.execution._order_flow import (
    diagnose_hawkes_stability,
    diagnose_queue_intensities,
    HawkesOrderFlowModel,
    QueueReactiveModel,
)


def test_almgren_chriss_zero_risk_schedule_is_analytic_linear_liquidation():
    grid = TimeGrid(jnp.linspace(0.0, 1.0, 5), time_id="ac-grid")
    model = AlmgrenChrissModel(
        volatility=0.2,
        risk_aversion=0.0,
        temporary_impact=2.0,
        permanent_impact=0.1,
        model_id="ac",
    )
    result = solve_almgren_chriss_schedule(model, grid, 12.0)

    np.testing.assert_allclose(result.inventory, jnp.asarray([12.0, 9.0, 6.0, 3.0, 0.0]))
    np.testing.assert_allclose(result.trading_rates, 12.0)
    assert result.conservation_residual < 1.0e-12
    assert result.temporary_cost == 288.0
    zero = solve_almgren_chriss_schedule(model, grid, 0.0)
    np.testing.assert_array_equal(zero.inventory, jnp.zeros((5,)))
    assert zero.objective == 0.0


def test_transient_exponential_state_uses_only_past_flow_and_decays():
    grid = TimeGrid(jnp.asarray([0.0, 1.0, 2.0]), time_id="transient-grid")
    model = TransientPropagatorModel(
        jnp.asarray([2.0]),
        jnp.asarray([jnp.log(2.0)]),
        model_id="transient",
    )
    result = transient_impact_path(model, grid, jnp.asarray([1.0, 0.0]))

    np.testing.assert_allclose(result.impact, jnp.asarray([0.0, 1.0, 0.5]), rtol=1e-6)
    evidence = diagnose_transient_manipulation(model, grid)
    assert bool(evidence.passed)
    adversarial = TransientPropagatorModel(
        jnp.asarray([-1.0]), jnp.asarray([1.0]), model_id="negative-kernel"
    )
    assert not bool(diagnose_transient_manipulation(adversarial, grid).passed)


def test_queue_intensity_probes_and_hawkes_spectral_stability_are_explicit():
    queue = QueueReactiveModel(
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([[0.5, -0.5], [-0.25, 0.25]]),
        jnp.zeros((2, 1)),
        jnp.asarray([[-1.0, 0.0], [0.0, -1.0]]),
        model_id="queue",
    )
    evidence = diagnose_queue_intensities(
        queue,
        jnp.asarray([[2.0, 1.0], [0.0, 0.0]]),
        jnp.zeros((2, 1)),
    )
    assert bool(evidence.passed)
    np.testing.assert_array_equal(
        queue.apply_channel(jnp.asarray([0.0, 2.0]), jnp.asarray(0)),
        jnp.asarray([0.0, 2.0]),
    )

    stable = HawkesOrderFlowModel(
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([[0.2, 0.1], [0.1, 0.2]]),
        jnp.asarray([1.0, 1.0]),
        model_id="stable",
    )
    unstable = HawkesOrderFlowModel(
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([[1.1, 0.0], [0.0, 1.1]]),
        jnp.asarray([1.0, 1.0]),
        model_id="unstable",
    )
    assert bool(diagnose_hawkes_stability(stable).stable)
    assert not bool(diagnose_hawkes_stability(unstable).stable)
