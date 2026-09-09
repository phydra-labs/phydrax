#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

from phydrax.control.stochastic._controlled_jump import (
    ControlledJumpPlan,
    ControlledJumpProblem,
    rollout_controlled_jumps_reference,
)
from phydrax.dynamics import TimeGrid
from phydrax.stochastic._jump import JUMP_MAX_EVENTS, JumpProcess, PoissonClockRealization


def test_controlled_jump_capacity_exhaustion_invalidates_truncated_path():
    process = JumpProcess(
        lambda time, state, controlled_args: jnp.asarray([1.0e12]),
        lambda state, channel, mark, controlled_args: state + 1.0,
        state_shape=(1,),
        num_channels=1,
        process_id="controlled-jump",
    )
    problem = ControlledJumpProblem(
        process,
        jnp.asarray([0.0]),
        action_shape=(1,),
        problem_id="controlled-problem",
    )
    time_grid = TimeGrid(jnp.asarray([0.0, 1.0]), time_id="controlled-time")
    plan = ControlledJumpPlan(time_grid, plan_id="controlled-plan")
    realization = PoissonClockRealization(
        jr.key(0),
        1,
        support=(0.0, 1.0),
        max_events_per_channel=1,
        process_id=process.process_id,
    )

    result = rollout_controlled_jumps_reference(
        problem,
        plan,
        realization,
        lambda time, state, args: jnp.asarray([0.0]),
        policy_id="zero-control",
    )

    assert int(result.status) == JUMP_MAX_EVENTS
    assert not bool(result.valid)
    assert int(result.events.counts) == 1
    assert result.events.pre_states[0, 0] == 0.0
    assert result.events.post_states[0, 0] == 1.0
    assert result.same_time_order == (
        "event-time-then-channel-then-channel-event-index;"
        "boundary-events-before-next-control"
    )
