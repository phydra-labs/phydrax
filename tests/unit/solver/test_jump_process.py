import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _counting_process(*, rate=2.0, process_id="counting"):
    return phx.stochastic.JumpProcess(
        lambda time, state, args: jnp.asarray([rate]),
        lambda state, channel, mark, args: state + jnp.asarray([1.0]),
        state_shape=(1,),
        num_channels=1,
        process_id=process_id,
    )


def test_poisson_clock_growth_preserves_every_existing_path_event_prefix():
    small = phx.stochastic.PoissonClockRealization(
        jr.key(0),
        2,
        support=(0.0, 2.0),
        max_events_per_channel=4,
        sample_shape=(2,),
        process_id="prefix-process",
    )
    extended = small.extend(7)
    wider = phx.stochastic.PoissonClockRealization(
        jr.key(0),
        2,
        support=(0.0, 2.0),
        max_events_per_channel=7,
        sample_shape=(3,),
        process_id="prefix-process",
    )

    assert small.coupling_id == extended.coupling_id == wider.coupling_id
    assert small.realization_id != extended.realization_id
    assert jnp.array_equal(small.thresholds, extended.thresholds[..., :4])
    assert jnp.array_equal(small.mark_keys, extended.mark_keys[..., :4])
    assert jnp.array_equal(extended.thresholds, wider.thresholds[:2])
    assert jnp.array_equal(extended.mark_keys, wider.mark_keys[:2])


def test_jump_event_batch_has_explicit_status_and_left_right_state_semantics():
    events = phx.stochastic.JumpEventBatch(
        jnp.asarray([[0.5, 0.0]]),
        jnp.asarray([[0, 0]]),
        jnp.zeros((1, 2)),
        jnp.asarray([[True, False]]),
        jnp.asarray([phx.stochastic.JUMP_SUCCESS]),
        state_shape=(1,),
        pre_states=jnp.asarray([[[0.0], [0.0]]]),
        post_states=jnp.asarray([[[1.0], [0.0]]]),
    )

    left = events.states_at(jnp.asarray([0.5]), jnp.asarray([0.0]), side="left")
    right = events.states_at(jnp.asarray([0.5]), jnp.asarray([0.0]), side="right")

    assert events.counts.tolist() == [1]
    assert events.successful.tolist() == [True]
    assert left.shape == right.shape == (1, 1, 1)
    assert left[0, 0, 0] == 0.0
    assert right[0, 0, 0] == 1.0
    assert phx.stochastic.jump_status_name(phx.stochastic.JUMP_MAX_EVENTS) == "max_events"


def test_mass_action_propensities_and_conservation_are_combinatorial():
    process = phx.stochastic.MassActionJumpProcess(
        jnp.asarray([[2, 0], [0, 1]]),
        jnp.asarray([[1, 1], [1, 0]]),
        jnp.asarray([0.5, 2.0]),
        process_id="mass-action",
    )
    reversible = phx.stochastic.MassActionJumpProcess(
        jnp.asarray([[1, 0], [0, 1]]),
        jnp.asarray([[0, 1], [1, 0]]),
        jnp.asarray([1.0, 3.0]),
    )

    assert jnp.allclose(
        process.intensities(0.0, jnp.asarray([3.0, 4.0])),
        jnp.asarray([1.5, 8.0]),
    )
    assert jnp.array_equal(
        process.jump(jnp.asarray([3, 4]), 0, jnp.asarray(0)),
        jnp.asarray([2, 5]),
    )
    assert jnp.array_equal(
        reversible.conservation_residual(jnp.asarray([1.0, 1.0])),
        jnp.zeros((2,)),
    )


def test_exact_jump_solvers_replay_and_recover_poisson_moments():
    process = _counting_process()
    realization = phx.stochastic.PoissonClockRealization(
        jr.key(1),
        1,
        support=(0.0, 1.0),
        max_events_per_channel=16,
        sample_shape=(4096,),
        process_id=process.process_id,
    )
    times = jnp.asarray([0.0, 0.25, 0.5, 1.0])

    next_reaction = phx.solver.solve_next_reaction(
        process,
        realization,
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
        save_times=times,
    )
    next_replay = phx.solver.solve_next_reaction(
        process,
        realization,
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
        save_times=jnp.asarray([0.0, 1.0]),
    )
    direct = phx.solver.solve_direct_ssa(
        process,
        realization,
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
        save_times=times,
    )

    assert jnp.all(next_reaction.successful)
    assert jnp.all(direct.successful)
    assert jnp.array_equal(
        next_reaction.events.times,
        next_replay.events.times,
        equal_nan=True,
    )
    assert jnp.array_equal(next_reaction.events.valid, next_replay.events.valid)
    assert jnp.abs(jnp.mean(next_reaction.events.counts) - 2.0) < 0.08
    assert jnp.abs(jnp.var(next_reaction.events.counts) - 2.0) < 0.12
    assert jnp.abs(jnp.mean(direct.events.counts) - 2.0) < 0.08
    assert jnp.abs(jnp.var(direct.events.counts) - 2.0) < 0.12
    assert jnp.array_equal(next_reaction.states[..., -1, 0], next_reaction.events.counts)
    assert jnp.array_equal(direct.states[..., -1, 0], direct.events.counts)
    trajectory = next_reaction.to_stochastic_trajectory(
        realization_axes=("path",),
        state_axes=("count",),
    )
    assert jnp.array_equal(trajectory.states, next_reaction.states)
    assert trajectory.realizations == (realization,)
    assert trajectory.metadata["process_id"] == process.process_id
    assert trajectory.metadata["jump_algorithm"] == "next_reaction"


def test_marked_compound_poisson_records_marks_and_post_states():
    process = phx.stochastic.JumpProcess(
        lambda time, state, args: jnp.asarray([3.0]),
        lambda state, channel, mark, args: state + mark[None],
        state_shape=(1,),
        num_channels=1,
        process_id="compound-poisson",
        mark_fn=lambda key, time, state, channel, args: jr.exponential(key),
    )
    realization = phx.stochastic.PoissonClockRealization(
        jr.key(2),
        1,
        support=(0.0, 1.0),
        max_events_per_channel=20,
        sample_shape=(1024,),
        process_id=process.process_id,
    )
    solution = phx.solver.solve_next_reaction(
        process,
        realization,
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
        save_times=jnp.asarray([0.0, 1.0]),
    )

    mark_sums = jnp.sum(
        jnp.where(solution.events.valid, solution.events.marks, 0.0), axis=-1
    )
    assert jnp.all(solution.successful)
    assert jnp.all(solution.events.marks[solution.events.valid] > 0.0)
    assert jnp.allclose(solution.states[:, -1, 0], mark_sums)
    assert jnp.allclose(
        solution.events.post_states[..., 0] - solution.events.pre_states[..., 0],
        jnp.where(solution.events.valid, solution.events.marks, 0.0),
    )


def test_finite_state_generator_matches_two_state_chain_and_boundary_policies():
    forward, backward = 2.0, 3.0
    process = phx.stochastic.JumpProcess(
        lambda time, state, args: jnp.asarray(
            [jnp.where(state[0] == 0, forward, backward)]
        ),
        lambda state, channel, mark, args: 1 - state,
        state_shape=(1,),
        num_channels=1,
        process_id="two-state",
    )
    generator = phx.solver.finite_state_generator(
        process,
        jnp.asarray([[0], [1]]),
    )

    expected = jnp.asarray([[-forward, forward], [backward, -backward]])
    assert jnp.allclose(generator.matrix, expected)
    assert jnp.allclose(generator.transition_matrix(0.7).sum(axis=-1), 1.0)
    assert jnp.allclose(
        generator.stationary_distribution(),
        jnp.asarray([backward, forward]) / (forward + backward),
    )

    birth = _counting_process(rate=1.0, process_id="finite-birth")
    states = jnp.asarray([[0.0], [1.0]])
    with pytest.raises(ValueError, match="omits reachable states"):
        phx.solver.finite_state_generator(birth, states)
    suppressed = phx.solver.finite_state_generator(
        birth, states, boundary_policy="suppress"
    )
    leaked = phx.solver.finite_state_generator(birth, states, boundary_policy="leak")
    assert jnp.allclose(suppressed.matrix.sum(axis=-1), 0.0)
    assert jnp.allclose(leaked.matrix.sum(axis=-1), -leaked.escaped_rates)


def test_event_capacity_exhaustion_is_explicit_not_an_infinite_sentinel():
    process = _counting_process(rate=100.0, process_id="overflow")
    realization = phx.stochastic.PoissonClockRealization(
        jr.key(3),
        1,
        support=(0.0, 1.0),
        max_events_per_channel=4,
        sample_shape=(64,),
        process_id=process.process_id,
    )
    solution = phx.solver.solve_next_reaction(
        process,
        realization,
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
        save_times=jnp.asarray([0.0, 1.0]),
        max_events=1,
    )

    assert jnp.all(solution.events.status == phx.stochastic.JUMP_MAX_EVENTS)
    assert jnp.all(solution.events.counts == 1)
    assert not jnp.any(solution.valid)
