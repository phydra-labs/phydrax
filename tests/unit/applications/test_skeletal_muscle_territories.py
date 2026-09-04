#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.skeletal_muscle.fibers import (
    MotorUnitTerritoryPlan,
    SkeletalFiberBundlePlan,
)


def _territory():
    return MotorUnitTerritoryPlan(
        ("unit-0", "unit-1"),
        ("fiber-0", "fiber-1", "fiber-2"),
        jnp.asarray((0, 1, 0)),
        jnp.asarray((1, 2, 3)),
        5,
        jnp.asarray((100.0, 150.0)),
        jnp.asarray((0.2, 0.1)),
        stimulus_source_id="declared-endplate-pulse-protocol",
    )


def test_sparse_territory_counts_coverage_and_endplate_routing():
    plan = _territory()
    stimulus = plan.bind_events(
        jnp.asarray(((1.0, jnp.inf), (1.05, jnp.inf))),
        jnp.asarray(((True, False), (True, False))),
        event_source_id="fixed-test-events",
    )

    assert bool(plan.evidence.valid)
    np.testing.assert_array_equal(plan.evidence.fiber_count_per_unit, (2, 1))
    current = stimulus.current(1.08)
    expected = np.zeros((3, 5))
    expected[0, 1] = 100.0
    expected[1, 2] = 150.0
    expected[2, 3] = 100.0
    np.testing.assert_array_equal(current, expected)
    assert stimulus.fiber_motor_unit_index.shape == (3,)
    assert stimulus.event_times_ms.shape == (2, 2)


def test_bound_events_are_accepted_by_fiber_bundle_without_dense_territory_tensor():
    stimulus = _territory().bind_events(
        jnp.asarray(((0.0,), (0.0,))),
        jnp.asarray(((True,), (False,))),
        event_source_id="fixed-test-events",
    )
    plan = SkeletalFiberBundlePlan(
        ("fiber-0", "fiber-1", "fiber-2"),
        5,
        jnp.asarray((10.0, 10.0, 10.0)),
        jnp.asarray((0.1, 0.1, 0.1)),
        stimulus,
        maximum_step_ms=0.1,
    )
    assert plan.stimulus is stimulus
    assert plan.stimulus.current(0.05)[0, 1] == 100.0
    assert plan.stimulus.current(0.25)[0, 1] == 0.0


def test_invalid_territory_and_active_event_data_fail_at_host_boundary():
    with pytest.raises(ValueError, match="out-of-range unit"):
        MotorUnitTerritoryPlan(
            ("unit-0",),
            ("fiber-0",),
            jnp.asarray((1,)),
            jnp.asarray((0,)),
            3,
            jnp.asarray((100.0,)),
            jnp.asarray((0.1,)),
            stimulus_source_id="test",
        )
    with pytest.raises(ValueError, match="Active motor-unit event times"):
        _territory().bind_events(
            jnp.asarray(((jnp.nan,), (1.0,))),
            jnp.asarray(((True,), (True,))),
            event_source_id="invalid-events",
        )
