import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

import phydrax as phx


def _problem(
    lag,
    /,
    *,
    drift=lambda time, state, memory, args: memory[0],
    history=lambda time, args: jnp.ones((1,)),
    t1=2.0,
    minimum_delay=0.4,
    maximum_delay=1.1,
    monotone_argument=True,
    root_isolation_step=None,
    args=None,
    second_lag=None,
):
    delays = [
        phx.solver.StateDependentDelay(
            "state_delay",
            lag,
            minimum_delay=minimum_delay,
            maximum_delay=maximum_delay,
            monotone_argument=monotone_argument,
            root_isolation_step=root_isolation_step,
        )
    ]
    if second_lag is not None:
        delays.append(
            phx.solver.StateDependentDelay(
                "second_state_delay",
                second_lag,
                minimum_delay=minimum_delay,
                maximum_delay=maximum_delay,
                monotone_argument=monotone_argument,
                root_isolation_step=root_isolation_step,
            )
        )
    return phx.solver.DelayDifferentialProblem(
        drift,
        history,
        tuple(delays),
        t0=0.0,
        t1=t1,
        args=args,
    )


def _piecewise_exact(time):
    return 1.0 + time + 0.5 * jnp.maximum(time - 1.0, 0.0) ** 2


def test_state_dependent_tracker_finds_manufactured_known_root_and_restarts():
    known_root = 0.5 / 0.9
    problem = _problem(
        lambda time, state, args: 0.5 + 0.1 * state[0],
        drift=lambda time, state, memory, args: jnp.ones_like(state),
        history=lambda time, args: jnp.asarray([time]),
        t1=0.9,
        minimum_delay=0.5,
        maximum_delay=0.6,
    )
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.9]),
        dense=True,
        max_steps=512,
    )
    interpolation = solution.interpolation
    assert interpolation is not None
    history = interpolation.computed_history
    accepted_ends = history.ends[: int(history.size)]

    assert bool(solution.successful)
    assert solution.stats["state_dependent_tracking"] == "high-order-dynamic-roots"
    assert int(solution.stats["num_dynamic_discontinuity_roots"]) >= 1
    assert int(solution.stats["num_internal_discontinuity_restarts"]) >= 1
    assert jnp.min(jnp.abs(accepted_ends - known_root)) < 2e-9
    root_count = int(solution.stats["num_dynamic_discontinuity_roots"])
    roots = solution.stats["dynamic_discontinuity_root_times"][:root_count]
    assert jnp.all(jnp.diff(roots) > 0.0)
    assert jnp.min(jnp.abs(roots - known_root)) < 2e-9


def test_multiple_state_delays_deduplicate_a_simultaneous_root():
    lag = lambda time, state, args: 0.5 + 0.0 * state[0]
    problem = _problem(
        lag,
        second_lag=lag,
        drift=lambda time, state, memory, args: jnp.ones_like(state),
        t1=0.8,
        minimum_delay=0.5,
        maximum_delay=0.5,
    )
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.8]),
        max_steps=512,
    )

    assert bool(solution.successful)
    assert int(solution.stats["num_dynamic_discontinuity_roots"]) == 1
    assert int(solution.stats["num_internal_discontinuity_restarts"]) == 1
    roots = solution.stats["dynamic_discontinuity_root_times"]
    assert jnp.allclose(roots[0], 0.5, atol=2e-9)
    assert jnp.all(jnp.isinf(roots[1:]))


def test_state_dependent_root_capacity_is_enforced_at_runtime():
    problem = _problem(
        lambda time, state, args: 0.5 + 0.0 * state[0],
        drift=lambda time, state, memory, args: jnp.ones_like(state),
        t1=0.8,
        minimum_delay=0.5,
        maximum_delay=0.5,
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="exceed|max_discontinuities|capacity",
    ):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.8]),
            max_discontinuities=1,
            max_steps=512,
        )


def test_state_dependent_lag_bound_violation_fails_explicitly():
    problem = _problem(
        lambda time, state, args: 0.3 - 0.4 * time,
        drift=lambda time, state, memory, args: jnp.ones_like(state),
        t1=0.5,
        minimum_delay=0.2,
        maximum_delay=0.3,
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="violated its declared bounds",
    ):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.5]),
            max_steps=256,
        )


def test_user_events_before_and_after_an_internal_root_are_distinguished():
    problem = _problem(
        lambda time, state, args: 0.5 + 0.0 * state[0],
        drift=lambda time, state, memory, args: jnp.ones_like(state),
        history=lambda time, args: jnp.asarray([0.0]),
        t1=1.0,
        minimum_delay=0.5,
        maximum_delay=0.5,
    )

    def solve_until(level):
        event = dfx.Event(
            lambda t, y, args, **kwargs: y[0] - level,
            root_finder=optx.Newton(rtol=1e-10, atol=1e-10),
        )
        return phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.2, 0.4, 0.6, 0.9]),
            event=event,
            dt0=0.05,
            max_steps=512,
        )

    before = solve_until(0.3)
    after = solve_until(0.8)

    assert bool(before.event_mask)
    assert bool(after.event_mask)
    assert int(before.stats["num_internal_discontinuity_restarts"]) == 0
    assert int(after.stats["num_internal_discontinuity_restarts"]) == 1
    assert jnp.array_equal(before.valid, jnp.asarray([True, False, False, False]))
    assert jnp.array_equal(after.valid, jnp.asarray([True, True, True, False]))


def test_fixed_and_adaptive_state_dependent_solves_converge():
    rate = 0.7
    problem = _problem(
        lambda time, state, args: 1.0 + 0.0 * state[0],
        drift=lambda time, state, memory, args: rate * jnp.exp(rate) * memory[0],
        history=lambda time, args: jnp.asarray([jnp.exp(rate * time)]),
        minimum_delay=1.0,
        maximum_delay=1.0,
    )
    exact = jnp.exp(2.0 * rate)
    fixed_errors = []
    for step in (0.2, 0.1, 0.05):
        solution = phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([2.0]),
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=step,
            max_steps=1024,
        )
        fixed_errors.append(jnp.abs(solution.states[0, 0] - exact))
    fixed_errors = jnp.stack(fixed_errors)

    adaptive_errors = []
    for tolerance in (1e-4, 1e-6, 1e-8):
        solution = phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([2.0]),
            rtol=tolerance,
            atol=tolerance * 1e-2,
            max_steps=2048,
        )
        adaptive_errors.append(jnp.abs(solution.states[0, 0] - exact))
    adaptive_errors = jnp.stack(adaptive_errors)

    assert jnp.all(fixed_errors[1:] < fixed_errors[:-1])
    assert fixed_errors[-1] < 0.6 * fixed_errors[0]
    assert jnp.all(adaptive_errors[1:] < adaptive_errors[:-1])
    assert adaptive_errors[-1] < 0.05 * adaptive_errors[0]


def test_dynamic_tracking_recovers_high_order_across_a_delayed_discontinuity():
    problem = _problem(
        lambda time, state, args: 1.0 + 0.0 * state[0],
        minimum_delay=1.0,
        maximum_delay=1.0,
    )
    tracked = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([2.0]),
        rtol=1e-8,
        atol=1e-10,
        max_steps=2048,
    )
    untracked = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([2.0]),
        discontinuity_depth=0,
        rtol=1e-8,
        atol=1e-10,
        max_steps=2048,
    )
    exact = _piecewise_exact(2.0)
    tracked_error = jnp.abs(tracked.states[0, 0] - exact)
    untracked_error = jnp.abs(untracked.states[0, 0] - exact)

    assert tracked_error < 1e-7
    assert tracked_error < untracked_error
    assert int(tracked.stats["num_dynamic_discontinuity_roots"]) >= 1
    assert int(untracked.stats["num_dynamic_discontinuity_roots"]) == 0


def test_state_dependent_solution_agrees_with_a_highly_refined_reference():
    lag = lambda time, state, args: 0.45 + 0.04 * jnp.tanh(state[0])
    problem = _problem(
        lag,
        drift=lambda time, state, memory, args: -0.2 * state + 0.7 * memory[0],
        t1=1.5,
        minimum_delay=0.4,
        maximum_delay=0.5,
    )
    reference = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([1.5]),
        rtol=2e-11,
        atol=2e-13,
        max_steps=4096,
    )
    candidate = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([1.5]),
        rtol=2e-7,
        atol=2e-9,
        max_steps=2048,
    )

    assert bool(reference.successful)
    assert bool(candidate.successful)
    assert jnp.allclose(candidate.states, reference.states, rtol=3e-6, atol=3e-7)


def test_lag_parameter_gradient_away_from_a_topology_change():
    terminal_time = 0.25

    def terminal(delay):
        problem = _problem(
            lambda time, state, args: args + 0.0 * state[0],
            history=lambda time, args: jnp.asarray([time]),
            t1=terminal_time,
            minimum_delay=0.4,
            maximum_delay=0.6,
            args=delay,
        )
        return phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([terminal_time]),
            dt0=0.02,
            max_steps=256,
        ).states[0, 0]

    delay = jnp.asarray(0.5)
    expected = 0.5 * terminal_time**2 - delay * terminal_time
    assert jnp.allclose(terminal(delay), expected, atol=2e-8)
    assert jnp.allclose(jax.grad(terminal)(delay), -terminal_time, atol=2e-7)


def test_nonmonotone_declaration_requires_a_root_isolation_contract():
    with pytest.raises(ValueError, match="root_isolation_step"):
        _problem(
            lambda time, state, args: 0.5 + 0.05 * jnp.sin(20.0 * time),
            t1=0.8,
            minimum_delay=0.4,
            maximum_delay=0.6,
            monotone_argument=False,
        )


def test_nonmonotone_tracking_isolates_forward_and_reverse_crossings():
    frequency = 4.0 * jnp.pi
    problem = _problem(
        lambda time, state, args: 0.5 + 0.2 * jnp.sin(frequency * time),
        drift=lambda time, state, memory, args: jnp.ones_like(state),
        history=lambda time, args: jnp.asarray([time]),
        t1=1.5,
        minimum_delay=0.3,
        maximum_delay=0.7,
        monotone_argument=False,
        root_isolation_step=0.025,
    )
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([1.5]),
        dt0=0.1,
        max_steps=4096,
        max_discontinuities=64,
    )
    segmented = phx.solver.solve_diffrax_delay_segmented(
        problem,
        save_times=jnp.asarray([1.5]),
        dt0=0.1,
        history_capacity=1024,
        max_steps_per_segment=8,
        max_discontinuities=64,
    )
    root_count = int(solution.stats["num_dynamic_discontinuity_roots"])
    roots = solution.stats["dynamic_discontinuity_root_times"][:root_count]

    assert bool(solution.successful)
    assert solution.stats["state_dependent_tracking"] == (
        "sign-isolated-nonmonotone-roots"
    )
    assert root_count >= 2
    assert jnp.all(jnp.diff(roots) > 0.0)
    expected_first_generation = jnp.asarray([0.33048475, 0.5, 0.66951525])
    assert jnp.allclose(roots[:3], expected_first_generation, atol=2e-8)
    assert jnp.array_equal(segmented.states, solution.states)
    assert segmented.stats["state_dependent_tracking"] == (
        "sign-isolated-nonmonotone-roots"
    )
    segmented_count = int(segmented.stats["num_dynamic_discontinuity_roots"])
    assert jnp.allclose(
        segmented.stats["dynamic_discontinuity_root_times"][:segmented_count],
        roots,
        atol=2e-8,
    )
