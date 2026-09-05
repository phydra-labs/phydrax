# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Consumer contracts at the RC/planning boundary, using an actual native plan."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.energy.building_dispatch import make_dispatch_system, replay_dispatch_heat
from phydrax.applications import energy_planning as ep


@pytest.fixture(scope="module")
def dispatch():
    with jax.enable_x64(True):
        heat_W = np.array([1500.0, 1800.0])
        spec = make_dispatch_system(
            heat_W, np.array([300.0, 300.0]), np.array([3600.0, 7200.0])
        )
        solution = ep.solve_energy_system(ep.compile_energy_system(spec))
        assert solution.successful, solution.replay.failures
        yield spec, solution.plan, heat_W


def test_amount_per_hour_must_not_be_consumed_as_watts(dispatch):
    spec, plan, heat_W = dispatch
    replay, delivered_W = replay_dispatch_heat(spec, plan, heat_W)
    assert replay.successful
    np.testing.assert_allclose(delivered_W, [1500.0, 1800.0], rtol=0, atol=0.01)
    # A perfectly feasible kWh/h plan can still be coupled incorrectly as W.
    # The physical boundary must reject the 1000x error rather than trusting LP success.
    with pytest.raises(RuntimeError, match="boundary mismatch"):
        replay_dispatch_heat(spec, plan, heat_W / 1000.0)


def test_corrupted_inventory_replay_is_rejected_before_building_integration(dispatch):
    spec, plan, heat_W = dispatch
    name = "inventory/thermal-store/state/day"
    corrupted = ep.EnergyPlan(
        tuple(
            ep.EnergyDispatch(
                item.name,
                item.values.at[1].add(0.25) if item.name == name else item.values,
            )
            for item in plan.dispatch
        ),
        jnp.asarray(plan.objective),
    )
    replay = ep.replay_energy_system(spec, corrupted)
    assert not replay.successful
    assert "inventory-dynamics/thermal-store/day" in replay.failures
    with pytest.raises(RuntimeError, match="Independent dispatch replay failed"):
        replay_dispatch_heat(spec, corrupted, heat_W)
