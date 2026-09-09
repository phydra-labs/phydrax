#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._trainable import partition_trainable
from phydrax.applications.battery._protocol import (
    BatteryProtocolPlan,
    BatteryProtocolValues,
    CurrentStepPlan,
    PASSIVE_TERMINAL_CONVENTION,
    RestStepPlan,
    StoichiometryStopGuard,
    TemperatureStopGuard,
    VoltageStopGuard,
)


def test_passive_terminal_convention_is_one_immutable_sign_source():
    convention = PASSIVE_TERMINAL_CONVENTION
    np.testing.assert_allclose(convention.terminal_voltage(4.2, 0.1), 4.1)
    np.testing.assert_allclose(convention.absorbed_power(4.0, 2.0), 8.0)
    np.testing.assert_allclose(convention.absorbed_power(4.0, -2.0), -8.0)
    np.testing.assert_allclose(convention.discharge_areal_current(-2.0, 0.5), 4.0)
    with pytest.raises(AttributeError):
        convention.current_definition = "discharge positive"


def test_protocol_topology_is_nontrainable_while_values_are_dynamic_leaves():
    plan = BatteryProtocolPlan(
        (
            CurrentStepPlan(
                2.0,
                stop_guards=(VoltageStopGuard("below"), TemperatureStopGuard("above")),
            ),
            RestStepPlan(1.0),
            CurrentStepPlan(
                3.0,
                stop_guards=(StoichiometryStopGuard("positive", "above"),),
            ),
        )
    )
    values = BatteryProtocolValues(
        plan,
        jnp.asarray((-2.0, 0.5)),
        jnp.asarray((2.7, 323.15, 0.95)),
    )

    plan_trainable, _ = partition_trainable(plan)
    values_trainable, _ = partition_trainable(values)
    assert not jax.tree.leaves(plan_trainable)
    leaves = [leaf for leaf in jax.tree.leaves(values_trainable) if eqx.is_array(leaf)]
    assert [leaf.shape for leaf in leaves] == [(2,), (3,)]
    np.testing.assert_array_equal(plan.interval_current_indices, np.asarray((0, -1, 1)))
    np.testing.assert_allclose(
        plan.interval_currents(values), np.asarray((-2.0, 0.0, 0.5))
    )
    np.testing.assert_allclose(plan.transition_times_s, np.asarray((0.0, 2.0, 3.0, 6.0)))


def test_held_current_endpoint_side_is_declared_and_jittable():
    steps = (CurrentStepPlan(1.0), CurrentStepPlan(1.0), RestStepPlan(1.0))
    left = BatteryProtocolPlan(steps, node_side="left")
    right = BatteryProtocolPlan(steps, node_side="right")
    left_values = BatteryProtocolValues(left, jnp.asarray((1.0, -2.0)))
    right_values = BatteryProtocolValues(right, jnp.asarray((1.0, -2.0)))

    evaluate_left = jax.jit(lambda time: left.current(time, left_values))
    evaluate_right = jax.jit(lambda time: right.current(time, right_values))
    np.testing.assert_allclose(evaluate_left(jnp.asarray(1.0)), 1.0)
    np.testing.assert_allclose(evaluate_right(jnp.asarray(1.0)), -2.0)
    np.testing.assert_allclose(evaluate_left(jnp.asarray(2.0)), -2.0)
    np.testing.assert_allclose(evaluate_right(jnp.asarray(2.0)), 0.0)


def test_protocol_refuses_unavailable_controls_and_untyped_guards():
    class VoltageControl:
        duration_s = 1.0

    with pytest.raises(TypeError, match="current/rest"):
        BatteryProtocolPlan((VoltageControl(),))
    with pytest.raises(TypeError, match="typed battery stop guards"):
        CurrentStepPlan(1.0, stop_guards=(object(),))
    with pytest.raises(ValueError, match="direction"):
        VoltageStopGuard("inside")
