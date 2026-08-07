#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _events(times, marks, *, capacity=3):
    count = len(times)
    padded_times = jnp.full((capacity,), jnp.nan).at[:count].set(jnp.asarray(times))
    channels = jnp.full((capacity,), -1, dtype=jnp.int32).at[:count].set(0)
    padded_marks = jnp.zeros((capacity, 1)).at[:count, 0].set(jnp.asarray(marks))
    valid = jnp.zeros((capacity,), dtype=bool).at[:count].set(True)
    return phx.stochastic.JumpEventBatch(
        padded_times,
        channels,
        padded_marks,
        valid,
        jnp.asarray(0, dtype=jnp.int32),
        mark_shape=(1,),
    )


def test_jump_delay_uses_right_continuous_history_at_exact_event_times():
    base = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: jnp.zeros_like(state),
        lambda time, args: jnp.ones((1,)),
        (phx.solver.ConstantDelay("lag", 0.2),),
        t0=0.0,
        t1=1.0,
    )
    problem = phx.solver.JumpDelayProblem(
        base,
        lambda time, state, memory, channel, mark, args: (
            state + mark + memory["lag"]
        ),
        mark_shape=(1,),
    )
    solution = phx.solver.solve_jump_delay(
        problem,
        _events([0.3, 0.6], [2.0, -1.0]),
        save_times=jnp.asarray([0.0, 0.3, 0.4, 0.6, 1.0]),
        solver=dfx.Euler(),
        dt0=0.05,
        dense=True,
        max_steps=32,
    )

    assert jnp.array_equal(solution.states[:, 0], jnp.asarray([1.0, 4.0, 4.0, 7.0, 7.0]))
    assert jnp.array_equal(
        solution.evaluate(jnp.asarray([0.3, 0.6]), left=True)[:, 0],
        jnp.asarray([1.0, 4.0]),
    )
    assert jnp.array_equal(
        solution.evaluate(jnp.asarray([0.3, 0.6]), left=False)[:, 0],
        jnp.asarray([4.0, 7.0]),
    )
    resolved = solution.backend_result.events
    assert jnp.array_equal(resolved.pre_states[:, 0], jnp.asarray([1.0, 4.0, 0.0]))
    assert jnp.array_equal(resolved.post_states[:, 0], jnp.asarray([4.0, 7.0, 0.0]))
    assert solution.stats["num_jumps"] == 2
    assert solution.metadata["jump_side_convention"] == "right-continuous"


def test_jump_delay_replays_one_global_wiener_path_across_events():
    noise = phx.solver.DelayWienerTerm(
        "driver",
        lambda time, state, memory, args: 0.2 * jnp.ones(state.shape + (1,)),
        (1,),
        structure="additive",
        basis_id="jump-delay-wiener",
    )
    base = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: jnp.zeros_like(state),
        lambda time, args: jnp.ones((1,)),
        (phx.solver.ConstantDelay("lag", 0.2),),
        t0=0.0,
        t1=1.0,
        wiener_terms=(noise,),
    )
    problem = phx.solver.JumpDelayProblem(
        base,
        lambda time, state, memory, channel, mark, args: state + mark,
        mark_shape=(1,),
    )
    realization = phx.stochastic.WienerRealization(
        jr.key(43),
        (1,),
        support=(0.0, 1.0),
        tolerance=1e-5,
        noise_id=base.noise_id,
    )
    times = jnp.asarray([0.0, 0.3, 0.6, 1.0])
    solution = phx.solver.solve_jump_delay(
        problem,
        _events([0.3, 0.6], [0.5, -0.2]),
        save_times=times,
        realization=realization,
        solver=dfx.Euler(),
        dt0=0.05,
        max_steps=32,
    )
    increments = realization.increments(
        jnp.zeros(times.shape), times, dtype=solution.states.dtype
    )[..., 0]
    cumulative_jumps = jnp.asarray([0.0, 0.5, 0.3, 0.3])
    expected = 1.0 + 0.2 * increments + cumulative_jumps

    assert jnp.allclose(solution.states[:, 0], expected, rtol=0.0, atol=5e-9)
    assert solution.realization is realization
    assert solution.metadata["driver_family"] == (
        "wiener-plus-finite-activity-jump"
    )


def test_jump_delay_accepts_an_empty_successful_schedule():
    base = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: memory["lag"],
        lambda time, args: jnp.ones((1,)),
        (phx.solver.ConstantDelay("lag", 0.5),),
        t0=0.0,
        t1=0.2,
    )
    problem = phx.solver.JumpDelayProblem(
        base,
        lambda time, state, memory, channel, mark, args: state + mark,
        mark_shape=(1,),
    )
    solution = phx.solver.solve_jump_delay(
        problem,
        _events([], []),
        save_times=jnp.asarray([0.0, 0.2]),
        solver=dfx.Euler(),
        dt0=0.05,
        max_steps=16,
    )

    assert jnp.allclose(solution.states[:, 0], jnp.asarray([1.0, 1.2]))
    assert solution.stats["num_jumps"] == 0


def test_jump_delay_rejects_endpoint_and_unsorted_events():
    base = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: jnp.zeros_like(state),
        lambda time, args: jnp.ones((1,)),
        (phx.solver.ConstantDelay("lag", 0.2),),
        t0=0.0,
        t1=1.0,
    )
    problem = phx.solver.JumpDelayProblem(
        base,
        lambda time, state, memory, channel, mark, args: state,
        mark_shape=(1,),
    )

    for schedule in (_events([0.0], [1.0]), _events([0.7, 0.4], [1.0, 1.0])):
        with pytest.raises(ValueError, match="strictly increasing"):
            phx.solver.solve_jump_delay(
                problem,
                schedule,
                save_times=jnp.asarray([1.0]),
                solver=dfx.Euler(),
                dt0=0.05,
            )
