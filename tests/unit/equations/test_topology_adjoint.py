#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_reverse_time_tape_applies_step_and_remap_transposes_in_order():
    cross = jnp.asarray(((0.8, 0.2), (0.3, 0.7)))
    remap = phx.equations.fem.ConservativeRemapPlan(jnp.eye(2), jnp.eye(2), cross)
    event = phx.equations.fem.conservative_remap_adjoint_event(
        remap, (2,), (2,), event_id="hp-remap"
    )
    tape = phx.equations.fem.ReverseTimeTopologyTape()
    tape = tape.append_step(
        phx.equations.fem.AcceptedStepAdjointRecord(
            lambda cotangent: 2.0 * cotangent, 0, "step-0"
        )
    )
    tape = tape.append_event(event)
    tape = tape.append_step(
        phx.equations.fem.AcceptedStepAdjointRecord(
            lambda cotangent: 3.0 * cotangent, 1, "step-1"
        )
    )
    final = jnp.asarray((1.0, -0.5))
    result = tape.reverse(final)
    expected = 2.0 * (cross.T @ (3.0 * final))
    np.testing.assert_allclose(result.initial_cotangent, expected, atol=3.0e-12)
    assert result.valid
    assert len(result.traversed_record_ids) == 3


def test_reverse_checkpoint_schedule_and_unsupported_event_are_explicit():
    schedule = phx.equations.fem.ReverseCheckpointSchedule(10, 3)
    assert schedule.should_checkpoint(0)
    assert schedule.should_checkpoint(10)
    assert len(schedule.checkpoint_indices) <= 4
    event = phx.equations.fem.TopologyAdjointEvent(
        lambda cotangent: cotangent,
        (2,),
        (2,),
        policy="unsupported",
        event_id="nonsmooth-remesh",
    )
    result = phx.equations.fem.ReverseTimeTopologyTape((event,)).reverse(jnp.ones((2,)))
    assert not result.valid
    np.testing.assert_allclose(result.initial_cotangent, 0.0)
