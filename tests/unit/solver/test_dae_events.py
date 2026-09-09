import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax.solver._dae_adaptive import _replay_solution


def _problem(*, problem_id="dae-events"):
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, args: state_rate - args[0],
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id=problem_id,
    )
    return phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.asarray((0.0,)),
        initial_state_rate=jnp.asarray((1.0,)),
        args=jnp.asarray((1.0, 0.35, 2.0)),
        problem_id=problem_id,
    )


def _termination():
    return phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-10,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=24,
    )


def _policy(*, adaptive=False):
    controller = (
        phx.solver.DAEAdaptivePolicy(
            relative_tolerance=1e-7,
            absolute_tolerance=1e-9,
            initial_step=0.2,
            maximum_accepted_steps=64,
            maximum_attempts=128,
        )
        if adaptive
        else None
    )
    return phx.solver.DAESolvePolicy(
        method=phx.solver.BDFMethod(2),
        adaptive=controller,
        nonlinear_termination=_termination(),
        initialization_termination=_termination(),
        failure="status",
    )


def _event_plan(
    *,
    terminal=False,
    maximum_events=4,
    reset=None,
    consistency=None,
    guard=None,
    name="threshold",
):
    guard_function = (
        (lambda time, state, args: state[0] - args[1]) if guard is None else guard
    )
    guard_plan = phx.solver.HybridGuardPlan(
        guard_function,
        direction=1,
        terminal=terminal,
        guard_id=f"{name}-guard",
    )
    schedule = phx.solver.HybridSchedulePlan(
        (phx.solver.ScheduledHybridGuard(guard_plan),),
        maximum_events=maximum_events,
    )
    reset_function = (
        (lambda time, state, state_rate, args: (args[2] * state, state_rate))
        if reset is None
        else reset
    )
    reset_map = phx.solver.DAEResetMap(
        reset_function,
        phx.solver.DAEInitializationSpec.index_one(),
        reset_id=f"{name}-reset",
    )
    consistency_policy = (
        phx.solver.DAEConsistencyPolicy(1e-8, 1e-8, 1e-8)
        if consistency is None
        else consistency
    )
    return phx.solver.DAEEventPlan(
        schedule,
        (reset_map,),
        consistency_policy,
    )


def _grid(name):
    return phx.dynamics.TimeGrid(jnp.asarray((0.0, 0.5, 1.0)), time_id=name)


def test_fixed_terminal_and_nonterminal_events_restart_and_terminate():
    terminal = phx.solver.solve_dae(
        _problem(problem_id="fixed-terminal-event"),
        _grid("fixed-terminal-event"),
        policy=_policy(),
        event_plan=_event_plan(terminal=True, name="fixed-terminal"),
    )
    continued = phx.solver.solve_dae(
        _problem(problem_id="fixed-nonterminal-event"),
        _grid("fixed-nonterminal-event"),
        policy=_policy(),
        event_plan=_event_plan(name="fixed-nonterminal"),
    )

    assert terminal.successful
    assert terminal.termination_status == int(
        phx.solver.DAETerminationStatus.EVENT_TERMINATED
    )
    assert terminal.events.event_count == 1
    assert jnp.isclose(terminal.events.event_times[0], 0.35, atol=1e-8)
    assert terminal.events.derivative_valid[0]
    assert jnp.array_equal(terminal.valid, jnp.asarray((True, False, False)))
    assert continued.successful
    assert continued.events.event_count == 1
    assert jnp.all(continued.valid)
    assert jnp.isclose(continued.states[-1, 0], 1.35, atol=1e-8)
    event_step = continued.events.replay.bracket_step_indices[0]
    assert continued.step_history.orders[event_step + 1] == 1
    count = int(continued.attempt_history.count)
    assert jnp.all(continued.attempt_history.residual_evaluations[:count] > 0)


def test_adaptive_terminal_and_nonterminal_events_use_accepted_brackets():
    terminal = phx.solver.solve_dae(
        _problem(problem_id="adaptive-terminal-event"),
        _grid("adaptive-terminal-event"),
        policy=_policy(adaptive=True),
        event_plan=_event_plan(terminal=True, name="adaptive-terminal"),
    )
    continued = phx.solver.solve_dae(
        _problem(problem_id="adaptive-nonterminal-event"),
        _grid("adaptive-nonterminal-event"),
        policy=_policy(adaptive=True),
        event_plan=_event_plan(name="adaptive-nonterminal"),
    )

    assert terminal.successful
    assert terminal.events.terminal
    assert terminal.termination_status == int(
        phx.solver.DAETerminationStatus.EVENT_TERMINATED
    )
    assert continued.successful
    assert continued.events.event_count == 1
    assert jnp.isclose(continued.states[-1, 0], 1.35, atol=2e-7)


def test_priority_resolves_simultaneous_guards_with_invalid_derivative():
    low = phx.solver.HybridGuardPlan(
        lambda time, state, args: state[0] - args[1],
        direction=1,
        priority=0,
        guard_id="simultaneous-low",
    )
    high = phx.solver.HybridGuardPlan(
        lambda time, state, args: state[0] - args[1],
        direction=1,
        priority=10,
        guard_id="simultaneous-high",
    )
    schedule = phx.solver.HybridSchedulePlan(
        (
            phx.solver.ScheduledHybridGuard(low),
            phx.solver.ScheduledHybridGuard(high),
        ),
        maximum_events=2,
    )
    initialization = phx.solver.DAEInitializationSpec.index_one()
    event_plan = phx.solver.DAEEventPlan(
        schedule,
        (
            phx.solver.DAEResetMap(
                lambda time, state, state_rate, args: (state + 1.0, state_rate),
                initialization,
                reset_id="simultaneous-low-reset",
            ),
            phx.solver.DAEResetMap(
                lambda time, state, state_rate, args: (state + 2.0, state_rate),
                initialization,
                reset_id="simultaneous-high-reset",
            ),
        ),
        phx.solver.DAEConsistencyPolicy(1e-8, 1e-8, 1e-8),
    )
    problem = _problem(problem_id="simultaneous-events")
    prepared = phx.solver.prepare_dae(
        problem,
        _grid("simultaneous-events"),
        policy=_policy(),
        event_plan=event_plan,
    )
    solution = phx.solver.solve_dae(prepared)

    assert solution.events.event_indices[0] == 1
    assert solution.events.simultaneous[0]
    assert not solution.events.derivative_valid[0]
    assert jnp.isclose(solution.states[-1, 0], 3.0, atol=1e-8)
    _, invalid_tangent = jax.jvp(
        lambda args: phx.solver.solve_dae(prepared, args=args).states[-1, 0],
        (problem.args,),
        (jnp.ones_like(problem.args),),
    )
    assert jnp.isnan(invalid_tangent)


def test_simultaneity_uses_crossing_times_not_shallow_guard_magnitude():
    early = phx.solver.HybridGuardPlan(
        lambda time, state, args: state[0] - 0.35,
        direction=1,
        priority=1,
        guard_id="actual-time-early",
    )
    shallow_late = phx.solver.HybridGuardPlan(
        lambda time, state, args: 1e-12 * (state[0] - 0.8),
        direction=1,
        guard_id="actual-time-shallow-late",
    )
    schedule = phx.solver.HybridSchedulePlan(
        (
            phx.solver.ScheduledHybridGuard(early),
            phx.solver.ScheduledHybridGuard(shallow_late),
        ),
        maximum_events=2,
    )
    initialization = phx.solver.DAEInitializationSpec.index_one()
    resets = tuple(
        phx.solver.DAEResetMap(
            lambda time, state, state_rate, args: (state + 2.0, state_rate),
            initialization,
            reset_id=f"actual-time-reset-{index}",
        )
        for index in range(2)
    )
    event_plan = phx.solver.DAEEventPlan(
        schedule,
        resets,
        phx.solver.DAEConsistencyPolicy(1e-8, 1e-8, 1e-8),
        grazing_tolerance=1e-14,
    )
    solution = phx.solver.solve_dae(
        _problem(problem_id="actual-time-simultaneity"),
        _grid("actual-time-simultaneity"),
        policy=_policy(),
        event_plan=event_plan,
    )

    assert solution.events.event_indices[0] == 0
    assert not solution.events.simultaneous[0]
    assert solution.events.derivative_valid[0]


def test_large_absolute_time_narrow_bracket_has_representable_lower_bound():
    start = jnp.asarray(1.0e12)
    grid = phx.dynamics.TimeGrid(
        jnp.asarray((start, start + 1.0e-3, start + 2.0e-3)),
        time_id="large-time-narrow-event-bracket",
    )
    first_representable_offset = jnp.nextafter(start, grid.times[1]) - start
    solution = phx.solver.solve_dae(
        _problem(problem_id="large-time-narrow-event-bracket"),
        grid,
        policy=_policy(),
        event_plan=_event_plan(
            terminal=True,
            guard=lambda time, state, args: state[0] - first_representable_offset,
            name="large-time-narrow",
        ),
    )

    assert solution.successful
    assert solution.events.event_times[0] > start
    assert solution.events.event_times[0] < grid.times[1]


def test_fixed_event_path_runs_periodic_regularity_and_promotes_failure():
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, args: state_rate - 10.0 * state,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="fixed-event-regularity",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.zeros(1),
        initial_state_rate=jnp.zeros(1),
        problem_id="fixed-event-regularity",
    )
    grid = phx.dynamics.TimeGrid(
        jnp.asarray((0.0, 0.1)),
        time_id="fixed-event-regularity",
    )
    guard = phx.solver.HybridGuardPlan(
        lambda time, state, args: state[0] - 10.0,
        direction=1,
        guard_id="fixed-event-regularity-guard",
    )
    event_plan = phx.solver.DAEEventPlan(
        phx.solver.HybridSchedulePlan(
            (phx.solver.ScheduledHybridGuard(guard),),
            maximum_events=1,
        ),
        (
            phx.solver.DAEResetMap(
                lambda time, state, state_rate, args: (state, state_rate),
                phx.solver.DAEInitializationSpec.index_one(),
                reset_id="fixed-event-regularity-reset",
            ),
        ),
        phx.solver.DAEConsistencyPolicy(1e-8, 1e-8, 1e-8),
    )
    solution = phx.solver.solve_dae(
        problem,
        grid,
        policy=phx.solver.DAESolvePolicy(
            method=phx.solver.BDFMethod(1),
            nonlinear_termination=_termination(),
            regularity=phx.solver.DAERegularityPolicy(
                "periodic",
                interval=1,
                failure="status",
            ),
            failure="status",
        ),
        event_plan=event_plan,
    )

    assert solution.regularity.stage_valid[0]
    assert solution.regularity.stage_status[0] == int(
        phx.solver.DAERegularityStatus.NUMERICALLY_SINGULAR
    )
    assert solution.termination_status == int(
        phx.solver.DAETerminationStatus.REGULARITY_FAILED
    )


def test_terminal_reset_singular_consistency_root_fails_regularity_status():
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, args: state * state_rate,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="terminal-singular-reset-consistency",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.ones(1),
        initial_state_rate=jnp.zeros(1),
        problem_id="terminal-singular-reset-consistency",
    )
    guard = phx.solver.HybridGuardPlan(
        lambda time, state, args: time - 0.05,
        direction=1,
        terminal=True,
        guard_id="terminal-singular-reset-guard",
    )
    event_plan = phx.solver.DAEEventPlan(
        phx.solver.HybridSchedulePlan(
            (phx.solver.ScheduledHybridGuard(guard),),
            maximum_events=1,
        ),
        (
            phx.solver.DAEResetMap(
                lambda time, state, state_rate, args: (
                    jnp.zeros_like(state),
                    jnp.zeros_like(state_rate),
                ),
                phx.solver.DAEInitializationSpec.index_one(),
                reset_id="terminal-singular-reset",
            ),
        ),
        phx.solver.DAEConsistencyPolicy(1e-8, 1e-8, 1e-8),
    )
    solution = phx.solver.solve_dae(
        problem,
        phx.dynamics.TimeGrid(
            jnp.asarray((0.0, 0.1)),
            time_id="terminal-singular-reset-consistency",
        ),
        policy=phx.solver.DAESolvePolicy(
            method=phx.solver.BDFMethod(1),
            nonlinear_termination=_termination(),
            initialization_termination=_termination(),
            regularity=phx.solver.DAERegularityPolicy(
                "periodic",
                failure="status",
            ),
            failure="status",
        ),
        event_plan=event_plan,
    )

    assert solution.events.consistency_regularity_valid[0]
    assert solution.events.consistency_regularity_status[0] == int(
        phx.solver.DAERegularityStatus.NUMERICALLY_SINGULAR
    )
    assert solution.termination_status == int(
        phx.solver.DAETerminationStatus.REGULARITY_FAILED
    )
    assert not solution.successful


def test_adaptive_terminal_reset_singular_consistency_root_fails_regularity_status():
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, args: state * state_rate,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="adaptive-terminal-singular-reset-consistency",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system,
        jnp.ones(1),
        initial_state_rate=jnp.zeros(1),
        problem_id="adaptive-terminal-singular-reset-consistency",
    )
    guard = phx.solver.HybridGuardPlan(
        lambda time, state, args: time - 0.05,
        direction=1,
        terminal=True,
        guard_id="adaptive-terminal-singular-reset-guard",
    )
    event_plan = phx.solver.DAEEventPlan(
        phx.solver.HybridSchedulePlan(
            (phx.solver.ScheduledHybridGuard(guard),),
            maximum_events=1,
        ),
        (
            phx.solver.DAEResetMap(
                lambda time, state, state_rate, args: (
                    jnp.zeros_like(state),
                    jnp.zeros_like(state_rate),
                ),
                phx.solver.DAEInitializationSpec.index_one(),
                reset_id="adaptive-terminal-singular-reset",
            ),
        ),
        phx.solver.DAEConsistencyPolicy(1e-8, 1e-8, 1e-8),
    )
    solution = phx.solver.solve_dae(
        problem,
        phx.dynamics.TimeGrid(
            jnp.asarray((0.0, 0.1)),
            time_id="adaptive-terminal-singular-reset-consistency",
        ),
        policy=phx.solver.DAESolvePolicy(
            method=phx.solver.BDFMethod(1),
            adaptive=phx.solver.DAEAdaptivePolicy(
                relative_tolerance=1e-6,
                absolute_tolerance=1e-9,
                initial_step=0.1,
                maximum_accepted_steps=16,
                maximum_attempts=32,
            ),
            nonlinear_termination=_termination(),
            initialization_termination=_termination(),
            regularity=phx.solver.DAERegularityPolicy(
                "periodic",
                failure="status",
            ),
            failure="status",
        ),
        event_plan=event_plan,
    )

    assert solution.events.consistency_regularity_valid[0]
    assert solution.events.consistency_regularity_status[0] == int(
        phx.solver.DAERegularityStatus.NUMERICALLY_SINGULAR
    )
    assert solution.termination_status == int(
        phx.solver.DAETerminationStatus.REGULARITY_FAILED
    )
    assert not solution.successful


def test_grazing_nonfinite_capacity_and_consistency_fail_closed():
    grazing = phx.solver.solve_dae(
        _problem(problem_id="grazing-event"),
        _grid("grazing-event"),
        policy=_policy(),
        event_plan=_event_plan(
            guard=lambda time, state, args: (state[0] - args[1]) ** 3,
            name="grazing",
        ),
    )
    assert not grazing.successful
    assert not grazing.events.derivative_valid[0]

    problem = _problem(problem_id="nonfinite-event")
    prepared = phx.solver.prepare_dae(
        problem,
        _grid("nonfinite-event"),
        policy=_policy(),
        event_plan=_event_plan(name="nonfinite"),
    )
    nonfinite = phx.solver.solve_dae(prepared, args=jnp.asarray((1.0, jnp.nan, 2.0)))
    assert nonfinite.termination_status == int(
        phx.solver.DAETerminationStatus.EVENT_FAILED
    )
    assert nonfinite.events.status[0] == int(phx.solver.DAEEventStatus.NONFINITE)

    first = phx.solver.HybridGuardPlan(
        lambda time, state, args: state[0] - 0.25,
        direction=1,
        guard_id="capacity-first",
    )
    second = phx.solver.HybridGuardPlan(
        lambda time, state, args: state[0] - 0.4,
        direction=1,
        guard_id="capacity-second",
    )
    schedule = phx.solver.HybridSchedulePlan(
        (
            phx.solver.ScheduledHybridGuard(first),
            phx.solver.ScheduledHybridGuard(second),
        ),
        maximum_events=1,
    )
    reset_maps = tuple(
        phx.solver.DAEResetMap(
            lambda time, state, state_rate, args: (state, state_rate),
            phx.solver.DAEInitializationSpec.index_one(),
            reset_id=f"capacity-reset-{index}",
        )
        for index in range(2)
    )
    capacity_plan = phx.solver.DAEEventPlan(
        schedule,
        reset_maps,
        phx.solver.DAEConsistencyPolicy(1e-8, 1e-8, 1e-8),
    )
    capacity = phx.solver.solve_dae(
        _problem(problem_id="capacity-event"),
        _grid("capacity-event"),
        policy=_policy(),
        event_plan=capacity_plan,
    )
    assert capacity.events.capacity_exceeded
    assert capacity.termination_status == int(
        phx.solver.DAETerminationStatus.EVENT_CAPACITY_EXCEEDED
    )

    failed_consistency = phx.solver.solve_dae(
        _problem(problem_id="consistency-event"),
        _grid("consistency-event"),
        policy=_policy(),
        event_plan=_event_plan(
            reset=lambda time, state, state_rate, args: (
                state,
                state_rate + 10.0,
            ),
            consistency=phx.solver.DAEConsistencyPolicy(1e-8, 0.1, 0.1),
            name="consistency-failure",
        ),
    )
    assert failed_consistency.termination_status == int(
        phx.solver.DAETerminationStatus.EVENT_FAILED
    )
    assert failed_consistency.events.status[0] == int(
        phx.solver.DAEEventStatus.CONSISTENCY_FAILED
    )


def test_adaptive_replay_recomputes_dynamic_root_reset_jvp_and_vjp():
    problem = _problem(problem_id="adaptive-event-derivative")
    prepared = phx.solver.prepare_dae(
        problem,
        _grid("adaptive-event-derivative"),
        policy=_policy(adaptive=True),
        event_plan=_event_plan(name="adaptive-derivative"),
    )

    def terminal(args):
        return phx.solver.solve_dae(prepared, args=args).states[-1, 0]

    args = jnp.asarray((1.0, 0.35, 2.0))
    tangent = jnp.asarray((0.2, -0.3, 0.4))
    cotangent = jnp.asarray(-0.7)
    value, jvp = jax.jvp(terminal, (args,), (tangent,))
    _, pullback = jax.vjp(terminal, args)
    vjp = pullback(cotangent)[0]
    epsilon = 2e-4
    finite_difference = (
        terminal(args + epsilon * tangent) - terminal(args - epsilon * tangent)
    ) / (2.0 * epsilon)

    assert jnp.isclose(value, 1.35, atol=2e-7)
    assert jnp.isclose(jvp, finite_difference, rtol=2e-3, atol=2e-4)
    assert jnp.isclose(cotangent * jvp, jnp.vdot(vjp, tangent), atol=2e-6)


def test_recorded_audit_values_are_not_replayed_and_continuation_survives_reset():
    problem = _problem(problem_id="event-audit-replay")
    prepared = phx.solver.prepare_dae(
        problem,
        _grid("event-audit-replay-first"),
        policy=_policy(adaptive=True),
        event_plan=_event_plan(name="event-audit"),
    )
    frozen = phx.solver.solve_dae(prepared)
    tampered = eqx.tree_at(
        lambda value: (
            value.events.replay.recorded_event_times,
            value.events.replay.recorded_states_before,
            value.events.replay.recorded_states_after,
        ),
        frozen,
        (
            frozen.events.replay.recorded_event_times + 100.0,
            frozen.events.replay.recorded_states_before + 100.0,
            frozen.events.replay.recorded_states_after - 100.0,
        ),
    )
    replayed = _replay_solution(prepared, problem.args, None, None, None, tampered)
    assert jnp.allclose(replayed.states, frozen.states, atol=2e-7)

    second_grid = phx.dynamics.TimeGrid(
        jnp.asarray((1.0, 1.5, 2.0)),
        time_id="event-audit-replay-second",
    )
    continued = phx.solver.solve_dae(
        problem,
        second_grid,
        policy=_policy(adaptive=True),
        event_plan=_event_plan(name="event-audit-continuation"),
        continuation=frozen.continuation,
    )
    assert continued.successful
    assert jnp.isclose(continued.states[-1, 0], 2.35, atol=2e-7)


def test_no_event_plan_preserves_existing_fixed_and_adaptive_paths():
    problem = _problem(problem_id="no-event-parity")
    grid = _grid("no-event-parity")
    fixed = phx.solver.solve_dae(problem, grid, policy=_policy())
    adaptive = phx.solver.solve_dae(problem, grid, policy=_policy(adaptive=True))

    assert fixed.events is None
    assert adaptive.events is None
    assert fixed.successful & adaptive.successful
    assert jnp.allclose(fixed.states[:, 0], grid.times, atol=1e-9)
    assert jnp.allclose(adaptive.states[:, 0], grid.times, atol=2e-7)
