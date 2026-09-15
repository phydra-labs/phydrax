import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


class _MixedDriverOperator(eqx.Module):
    def __call__(self, batch, /, *, key=None):
        del key
        return (
            batch.input("state").values
            + batch.input("wiener_increment").values
            + batch.input("jump_counts").values
        )


def _counting_process(*, rate, process_id):
    return phx.stochastic.JumpProcess(
        lambda time, state, args: jnp.asarray([rate]),
        lambda state, channel, mark, args: state + jnp.asarray([1.0]),
        state_shape=(1,),
        num_channels=1,
        process_id=process_id,
    )


def _one_shot_process(*, rate, process_id):
    return phx.stochastic.JumpProcess(
        lambda time, state, args: jnp.asarray([rate * state[1]]),
        lambda state, channel, mark, args: state.at[1].set(0.0),
        state_shape=(2,),
        num_channels=1,
        process_id=process_id,
    )


def _guard_schedule(
    root,
    reset_shift,
    schedule_id,
    /,
    *,
    priority=0,
    terminal=False,
):
    guard = phx.solver.HybridGuardPlan(
        lambda time, state, args: state[0] - root,
        direction=1,
        priority=priority,
        terminal=terminal,
        guard_id=f"{schedule_id}:guard",
    )
    event = phx.solver.HybridEventPlan(
        guard,
        lambda time, state, args: state + jnp.asarray([reset_shift, 0.0]),
        lambda time, state, args: jnp.asarray([1.0, 0.0]),
        lambda time, state, args: jnp.asarray([1.0, 0.0]),
        event_tolerance=1e-8,
        plan_id=f"{schedule_id}:event",
    )
    return phx.solver.ScheduledHybridGuard(guard, event=event)


def _timed_one_shot_problem(*, jump_time, process_id):
    realization = phx.stochastic.PoissonClockRealization(
        jr.key(901),
        1,
        support=(0.0, 1.0),
        max_events_per_channel=2,
        sample_shape=(1,),
        process_id=process_id,
    )
    rate = realization.thresholds[0, 0, 0] / jump_time
    process = _one_shot_process(rate=rate, process_id=process_id)
    differential = phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.asarray([1.0, 0.0]),
        jnp.asarray([0.0, 1.0]),
        t0=0.0,
        t1=1.0,
    )
    return (
        phx.solver.JumpDifferentialProblem(differential, process),
        realization,
    )


def test_hybrid_jump_rejects_nontrivial_state_geometry():
    geometry = phx.metrix.SpecialOrthogonalStateGeometry(2)
    differential = phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.zeros_like(state),
        jnp.eye(2),
        t0=0.0,
        t1=1.0,
        state_geometry=geometry,
    )
    process = phx.stochastic.JumpProcess(
        lambda time, state, args: jnp.asarray([0.0]),
        lambda state, channel, mark, args: state,
        state_shape=(2, 2),
        num_channels=1,
        process_id="hybrid-geometry-rejection",
    )
    realization = phx.stochastic.PoissonClockRealization(
        jr.key(101),
        1,
        support=(0.0, 1.0),
        max_events_per_channel=1,
        sample_shape=(1,),
        process_id=process.process_id,
    )
    with pytest.raises(ValueError, match="nontrivial state_geometry"):
        phx.solver.solve_jump_differential(
            phx.solver.JumpDifferentialProblem(differential, process),
            realization,
            save_times=jnp.asarray([0.0, 1.0]),
        )


def test_hybrid_ode_jump_solution_replays_and_is_independent_of_save_partition():
    process = _counting_process(rate=2.0, process_id="hybrid-counting")
    differential = phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.full_like(state, 0.5),
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
    )
    problem = phx.solver.JumpDifferentialProblem(differential, process)
    realization = phx.stochastic.PoissonClockRealization(
        jr.key(0),
        1,
        support=(0.0, 1.0),
        max_events_per_channel=16,
        sample_shape=(32,),
        process_id=process.process_id,
    )
    coarse_times = jnp.asarray([0.0, 0.4, 1.0])
    fine_times = jnp.asarray([0.0, 0.2, 0.4, 0.7, 1.0])

    coarse = phx.solver.solve_jump_differential(
        problem,
        realization,
        save_times=coarse_times,
    )
    fine = phx.solver.solve_jump_differential(
        problem,
        realization,
        save_times=fine_times,
    )
    counts = jnp.sum(
        coarse.events.valid[..., None] & (coarse.events.times[..., None] <= coarse_times),
        axis=-2,
    )
    expected = 0.5 * coarse_times + counts

    assert jnp.all(coarse.successful)
    assert jnp.all(fine.successful)
    assert jnp.allclose(coarse.states[..., 0], expected, atol=1e-10)
    assert jnp.allclose(coarse.states[:, -1], fine.states[:, -1], atol=1e-10)
    assert jnp.array_equal(coarse.events.valid, fine.events.valid)
    assert jnp.allclose(
        coarse.events.times[coarse.events.valid],
        fine.events.times[fine.events.valid],
        atol=1e-12,
    )
    trajectory = coarse.to_stochastic_trajectory(
        realization_axes=("path",),
        state_axes=("state",),
    )
    assert jnp.array_equal(trajectory.states, coarse.states)
    assert trajectory.realizations == (realization,)
    assert trajectory.metadata["solver_name"] == coarse.solver_name
    assert jnp.array_equal(trajectory.metadata["terminal"], coarse.terminal)
    assert jnp.array_equal(
        trajectory.metadata["numerical_successful"],
        coarse.numerical_successful,
    )
    assert jnp.array_equal(trajectory.metadata["completed"], coarse.completed)
    assert coarse.deterministic_events is None
    assert jnp.all(~coarse.terminal)
    assert jnp.array_equal(coarse.numerical_successful, coarse.events.successful)


def test_state_dependent_hazard_localizes_known_integrated_intensity_events():
    process = phx.stochastic.JumpProcess(
        lambda time, state, args: jnp.asarray([jnp.maximum(state[0], 0.0)]),
        lambda state, channel, mark, args: state,
        state_shape=(1,),
        num_channels=1,
        process_id="linear-integrated-hazard",
    )
    differential = phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.ones_like(state),
        jnp.asarray([0.0]),
        t0=0.0,
        t1=2.0,
    )
    realization = phx.stochastic.PoissonClockRealization(
        jr.key(1),
        1,
        support=(0.0, 2.0),
        max_events_per_channel=8,
        sample_shape=(16,),
        process_id=process.process_id,
    )
    solution = phx.solver.solve_jump_differential(
        phx.solver.JumpDifferentialProblem(differential, process),
        realization,
        save_times=jnp.asarray([0.0, 2.0]),
        event_rtol=1e-9,
        event_atol=1e-11,
    )
    expected_first = jnp.sqrt(2.0 * realization.thresholds[:, 0, 0])
    observed = expected_first <= 2.0

    assert jnp.all(solution.successful)
    assert jnp.any(observed)
    assert jnp.allclose(
        solution.events.times[observed, 0],
        expected_first[observed],
        rtol=1e-8,
        atol=1e-10,
    )
    assert jnp.allclose(solution.states[:, -1, 0], 2.0, atol=1e-10)


def test_jump_diffusion_uses_one_global_wiener_path_across_event_restarts():
    process = _counting_process(rate=1.5, process_id="coupled-jump-diffusion")
    differential = phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.zeros_like(state),
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
        wiener_terms=(
            phx.solver.WienerTerm(
                "additive-noise",
                lambda time, state, args: jnp.ones((1, 1)),
                (1,),
                structure="additive",
            ),
        ),
    )
    problem = phx.solver.JumpDifferentialProblem(differential, process)
    poisson = phx.stochastic.PoissonClockRealization(
        jr.key(2),
        1,
        support=(0.0, 1.0),
        max_events_per_channel=16,
        sample_shape=(16,),
        process_id=process.process_id,
    )
    wiener = phx.stochastic.WienerRealization(
        jr.key(3),
        (1,),
        support=(0.0, 1.0),
        sample_shape=(16,),
        tolerance=1e-3,
    )
    times = jnp.asarray([0.0, 0.5, 1.0])
    initial_states = (
        jnp.arange(16, dtype=differential.initial_state.dtype)[:, None] / 16.0
    )

    solution = phx.solver.solve_jump_differential(
        problem,
        poisson,
        save_times=times,
        initial_states=initial_states,
        wiener_realization=wiener,
        dt0=0.01,
    )
    replay = phx.solver.solve_jump_differential(
        problem,
        poisson,
        save_times=times,
        initial_states=initial_states,
        wiener_realization=wiener,
        dt0=0.01,
    )
    expected = (
        initial_states[:, 0]
        + wiener.increments(jnp.asarray(0.0), jnp.asarray(1.0))[..., 0]
        + solution.events.counts
    )

    assert jnp.all(solution.successful)
    assert isinstance(
        solution.realization,
        phx.stochastic.CompositeStochasticRealization,
    )
    assert set(solution.realization.components) == {"wiener", "jump"}
    assert jnp.array_equal(solution.states[:, 0], initial_states)
    assert jnp.allclose(solution.states[:, -1, 0], expected, atol=1e-10)
    assert jnp.array_equal(solution.states, replay.states)
    assert jnp.array_equal(
        solution.events.times,
        replay.events.times,
        equal_nan=True,
    )


def test_hybrid_paths_accept_distinct_initial_states_and_validate_sample_shape():
    process = _counting_process(rate=0.0, process_id="path-initial-states")
    differential = phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.full_like(state, 0.5),
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
    )
    problem = phx.solver.JumpDifferentialProblem(differential, process)
    realization = phx.stochastic.PoissonClockRealization(
        jr.key(902),
        1,
        support=(0.0, 1.0),
        max_events_per_channel=2,
        sample_shape=(2, 2),
        process_id=process.process_id,
    )
    initial_states = jnp.asarray([[[0.0], [1.0]], [[2.0], [3.0]]])

    solution = phx.solver.solve_jump_differential(
        problem,
        realization,
        save_times=jnp.asarray([0.0, 1.0]),
        initial_states=initial_states,
    )

    assert jnp.array_equal(solution.states[..., 0, :], initial_states)
    assert jnp.allclose(solution.states[..., -1, :], initial_states + 0.5)
    with pytest.raises(ValueError, match="initial_states must have shape"):
        phx.solver.solve_jump_differential(
            problem,
            realization,
            save_times=jnp.asarray([0.0, 1.0]),
            initial_states=jnp.zeros((4, 1)),
        )


def test_jump_event_precedes_and_survives_later_terminal_guard():
    problem, realization = _timed_one_shot_problem(
        jump_time=0.2,
        process_id="jump-before-guard",
    )
    schedule = phx.solver.HybridSchedulePlan(
        (
            _guard_schedule(
                0.7,
                3.0,
                "jump-before-guard",
                terminal=True,
            ),
        ),
        maximum_events=2,
    )

    solution = phx.solver.solve_jump_differential(
        problem,
        realization,
        save_times=jnp.asarray([0.0, 0.5, 0.7, 1.0]),
        hybrid_schedule=schedule,
        event_rtol=1e-9,
        event_atol=1e-11,
    )
    deterministic = solution.deterministic_events
    assert deterministic is not None

    assert solution.events.valid[0, 0]
    assert deterministic.valid[0, 0]
    assert solution.events.times[0, 0] < deterministic.event_times[0, 0]
    assert jnp.isclose(solution.events.times[0, 0], 0.2, atol=1e-8)
    assert jnp.isclose(deterministic.event_times[0, 0], 0.7, atol=1e-8)
    assert solution.terminal[0]
    assert solution.successful[0]


def test_guard_reset_precedes_jump_and_continuation_uses_reset_state():
    problem, realization = _timed_one_shot_problem(
        jump_time=0.7,
        process_id="guard-before-jump",
    )
    schedule = phx.solver.HybridSchedulePlan(
        (_guard_schedule(0.2, 10.0, "guard-before-jump"),),
        maximum_events=2,
    )

    solution = phx.solver.solve_jump_differential(
        problem,
        realization,
        save_times=jnp.asarray([0.0, 0.5, 1.0]),
        hybrid_schedule=schedule,
        event_rtol=1e-9,
        event_atol=1e-11,
    )
    deterministic = solution.deterministic_events
    assert deterministic is not None

    assert deterministic.event_times[0, 0] < solution.events.times[0, 0]
    assert jnp.isclose(deterministic.event_times[0, 0], 0.2, atol=1e-8)
    assert jnp.isclose(solution.events.times[0, 0], 0.7, atol=1e-8)
    assert jnp.allclose(solution.states[0, -1], jnp.asarray([11.0, 0.0]))
    assert solution.completed[0]


def test_simultaneous_deterministic_guards_use_declared_priority():
    problem, realization = _timed_one_shot_problem(
        jump_time=2.0,
        process_id="deterministic-guard-priority",
    )
    low = _guard_schedule(
        0.3,
        10.0,
        "simultaneous-low",
        priority=0,
        terminal=True,
    )
    high = _guard_schedule(
        0.3,
        20.0,
        "simultaneous-high",
        priority=5,
        terminal=True,
    )
    schedule = phx.solver.HybridSchedulePlan((low, high), maximum_events=2)

    solution = phx.solver.solve_jump_differential(
        problem,
        realization,
        save_times=jnp.asarray([0.0, 0.3, 1.0]),
        hybrid_schedule=schedule,
        event_rtol=1e-9,
        event_atol=1e-11,
    )
    deterministic = solution.deterministic_events
    assert deterministic is not None

    assert deterministic.event_indices[0, 0] == 1
    assert jnp.allclose(
        deterministic.event_states_after[0, 0],
        jnp.asarray([20.3, 1.0]),
        atol=1e-8,
    )
    assert deterministic.terminal[0, 0]


def test_guard_reset_tape_replays_and_is_save_partition_independent():
    problem, realization = _timed_one_shot_problem(
        jump_time=2.0,
        process_id="guard-reset-replay",
    )
    schedule = phx.solver.HybridSchedulePlan(
        (_guard_schedule(0.25, 1.0, "guard-reset-replay"),),
        maximum_events=2,
    )
    coarse_times = jnp.asarray([0.0, 0.5, 1.0])
    fine_times = jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0])

    coarse = phx.solver.solve_jump_differential(
        problem,
        realization,
        save_times=coarse_times,
        hybrid_schedule=schedule,
        event_rtol=1e-9,
        event_atol=1e-11,
    )
    replayed = phx.solver.solve_jump_differential(
        problem,
        realization,
        save_times=coarse_times,
        hybrid_schedule=schedule,
        event_rtol=1e-9,
        event_atol=1e-11,
    )
    fine = phx.solver.solve_jump_differential(
        problem,
        realization,
        save_times=fine_times,
        hybrid_schedule=schedule,
        event_rtol=1e-9,
        event_atol=1e-11,
    )
    coarse_events = coarse.deterministic_events
    fine_events = fine.deterministic_events
    assert coarse_events is not None
    assert fine_events is not None
    path_tape = jax.tree.map(lambda value: value[0], coarse_events.tape)
    prepared = phx.solver.prepare_hybrid_schedule(
        schedule,
        jnp.asarray([0.0, 1.0]),
    )
    tape_replay = phx.solver.replay_hybrid_schedule(
        prepared,
        path_tape,
        path_tape.states_before[0],
    )

    assert jnp.array_equal(coarse.states, replayed.states)
    assert jnp.array_equal(
        coarse_events.event_times,
        replayed.deterministic_events.event_times,
    )
    assert jnp.allclose(coarse.states[:, 0], fine.states[:, 0])
    assert jnp.allclose(coarse.states[:, 1], fine.states[:, 2])
    assert jnp.allclose(coarse.states[:, 2], fine.states[:, 4])
    assert jnp.allclose(
        coarse_events.event_times,
        fine_events.event_times,
        atol=1e-10,
    )
    assert jnp.allclose(coarse.states[0, -1], jnp.asarray([2.0, 1.0]))
    assert tape_replay.valid
    assert jnp.allclose(tape_replay.state, path_tape.states_after[0])


def test_terminal_guard_freezes_state_and_invalidates_only_later_save_nodes():
    problem, realization = _timed_one_shot_problem(
        jump_time=2.0,
        process_id="terminal-guard-validity",
    )
    schedule = phx.solver.HybridSchedulePlan(
        (
            _guard_schedule(
                0.5,
                2.0,
                "terminal-guard-validity",
                terminal=True,
            ),
        ),
        maximum_events=1,
    )

    solution = phx.solver.solve_jump_differential(
        problem,
        realization,
        save_times=jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
        hybrid_schedule=schedule,
        event_rtol=1e-9,
        event_atol=1e-11,
    )

    assert solution.terminal[0]
    assert solution.numerical_successful[0]
    assert solution.successful[0]
    assert not solution.completed[0]
    assert jnp.array_equal(
        solution.valid[0],
        jnp.asarray([True, True, True, False, False]),
    )
    assert jnp.allclose(
        solution.states[0, 2:],
        jnp.broadcast_to(solution.states[0, 2], (3, 2)),
    )


def test_deterministic_event_capacity_failure_is_not_physical_termination():
    problem, realization = _timed_one_shot_problem(
        jump_time=2.0,
        process_id="guard-capacity-failure",
    )
    schedule = phx.solver.HybridSchedulePlan(
        (_guard_schedule(0.2, -0.1, "guard-capacity-failure"),),
        maximum_events=1,
    )

    solution = phx.solver.solve_jump_differential(
        problem,
        realization,
        save_times=jnp.asarray([0.0, 0.5, 1.0]),
        hybrid_schedule=schedule,
        event_rtol=1e-9,
        event_atol=1e-11,
    )
    deterministic = solution.deterministic_events
    assert deterministic is not None

    assert deterministic.capacity_exceeded[0]
    assert not solution.terminal[0]
    assert not solution.numerical_successful[0]
    assert not solution.successful[0]


def test_composite_process_operator_rollout_combines_wiener_and_jump_drivers():
    x_axis = phx.nn.operator.OperatorAxis("x", jnp.linspace(0.0, 1.0, 3, endpoint=False))
    channel_axis = phx.nn.operator.OperatorAxis("channel", jnp.arange(1.0))
    template = phx.nn.operator.OperatorBatch(
        inputs={
            "state": phx.nn.operator.FunctionSamples(
                values=jnp.zeros((2, 3)),
                axes=(x_axis,),
            ),
            "duration": phx.nn.operator.FunctionSamples(
                values=jnp.ones((2, 3)),
                axes=(x_axis,),
            ),
            "wiener_increment": phx.nn.operator.FunctionSamples(
                values=jnp.zeros((2, 3)),
                axes=(x_axis,),
            ),
            "jump_counts": phx.nn.operator.FunctionSamples(
                values=jnp.zeros((2, 1)),
                axes=(channel_axis,),
            ),
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(values=None, axes=(x_axis,)),
        },
        case_axes=("case",),
    )
    spec = phx.nn.operator.training.OperatorTransitionSpec(
        phx.nn.operator.OperatorOutputSpec("scalar"),
        driver_bindings=(
            phx.nn.operator.training.OperatorDriverBinding(
                "wiener_increment",
                "wiener",
                kind="wiener",
                quantity="increment",
            ),
            phx.nn.operator.training.OperatorDriverBinding(
                "jump_counts",
                "jump",
                kind="jump",
                quantity="channel_counts",
            ),
        ),
    )
    law = phx.nn.operator.training.OperatorProcessTransition(
        _MixedDriverOperator(),
        template,
        spec,
        process_id="mixed-operator-transition",
    )
    jump_process = _counting_process(rate=2.0, process_id="operator-jumps")
    poisson = phx.stochastic.PoissonClockRealization(
        jr.key(4),
        1,
        support=(0.0, 1.0),
        max_events_per_channel=16,
        sample_shape=(8,),
        process_id=jump_process.process_id,
    )
    jump_solution = phx.solver.solve_next_reaction(
        jump_process,
        poisson,
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
        save_times=jnp.asarray([0.0, 1.0]),
    )
    wiener = phx.stochastic.WienerRealization(
        jr.key(5),
        (3,),
        support=(0.0, 1.0),
        sample_shape=(8,),
    )
    realization = phx.stochastic.CompositeStochasticRealization(
        {"wiener": wiener, "jump": poisson}
    )
    coarse = phx.nn.operator.training.process_operator_rollout(
        law,
        realization,
        jnp.asarray([0.0, 0.4, 1.0]),
        jump_events={"jump": jump_solution.events},
    )
    fine = phx.nn.operator.training.process_operator_rollout(
        law,
        realization,
        jnp.asarray([0.0, 0.2, 0.4, 0.7, 1.0]),
        jump_events={"jump": jump_solution.events},
    )
    expected = (
        wiener.increments(jnp.asarray(0.0), jnp.asarray(1.0))
        + jump_solution.events.counts[:, None]
    )

    assert coarse.kind == "process"
    assert coarse.states.shape == (2, 8, 3, 3)
    assert coarse.trajectory.realization_shape == (8,)
    assert jnp.allclose(coarse.states[:, :, -1], expected[None], atol=1e-12)
    assert jnp.allclose(coarse.states[:, :, -1], fine.states[:, :, -1], atol=1e-12)
    assert all(
        item is not None and item.realization_id == realization.realization_id
        for item in coarse.trajectory.realizations
    )
