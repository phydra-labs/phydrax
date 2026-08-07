from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

import phydrax as phx
from phydrax.solver._diffrax_delay_backend import _delay_discontinuity_times


def _constant_delay_problem(
    drift,
    history,
    delays,
    /,
    *,
    diffusion=None,
    noise_shape=None,
    noise_id=None,
    interpretation="ito",
    **kwargs,
):
    delay_values = jnp.asarray(delays).reshape((-1,))
    delay_terms = tuple(
        phx.solver.ConstantDelay(f"delay_{index}", delay_values[index])
        for index in range(int(delay_values.size))
    )
    if diffusion is None:
        wiener_terms = ()
    else:
        if noise_shape is None:
            raise ValueError("noise_shape is required with diffusion.")
        wiener_terms = (
            phx.solver.DelayWienerTerm(
                "noise",
                diffusion,
                noise_shape,
                basis_id=noise_id,
            ),
        )
    return phx.solver.DelayDifferentialProblem(
        drift,
        history,
        delay_terms,
        wiener_terms=wiener_terms,
        interpretation=interpretation,
        **kwargs,
    )


def _piecewise_problem(*, t1=2.0):
    return _constant_delay_problem(
        lambda time, state, delayed, args: delayed[0],
        lambda time, args: jnp.ones((1,)),
        jnp.asarray([1.0]),
        t0=0.0,
        t1=t1,
    )


def _piecewise_exact(times):
    return 1.0 + times + 0.5 * jnp.maximum(times - 1.0, 0.0) ** 2


def test_diffrax_delay_recovers_piecewise_method_of_steps_and_dense_output():
    times = jnp.linspace(0.0, 2.0, 21)
    solution = phx.solver.solve_diffrax_delay(
        _piecewise_problem(),
        save_times=times,
        dense=True,
    )

    assert solution.states.shape == (21, 1)
    assert bool(solution.successful)
    assert solution.has_dense_interpolation
    assert solution.solver_name == "Tsit5"
    assert solution.solver_id == "solver:diffrax-delay:Tsit5:retarded-v1"
    assert solution.resolved_method == "Tsit5:causal-retarded-method-of-steps"
    assert solution.metadata["backend"] == "diffrax"
    assert solution.stats["num_delays"] == 1
    assert solution.stats["num_tracked_discontinuities"] == 6
    assert jnp.allclose(solution.states[:, 0], _piecewise_exact(times), atol=2e-7)

    query = jnp.asarray([[0.25, 0.75], [1.25, 1.75]])
    dense = solution.evaluate(query)
    assert dense.shape == (2, 2, 1)
    assert jnp.allclose(dense[..., 0], _piecewise_exact(query), atol=2e-7)


def test_diffrax_delay_preserves_matrix_state_and_multiple_delay_ordering():
    rate = 0.3
    delays = jnp.asarray([0.35, 0.6])
    weights = jnp.asarray([0.4, 0.6])
    base = jnp.asarray([[1.0, 0.5], [-0.25, 2.0]])

    def history(time, args):
        return jnp.exp(rate * time) * base

    def drift(time, state, delayed, args):
        del time, state, args
        coefficients = rate * weights * jnp.exp(rate * delays)
        return jnp.tensordot(coefficients, delayed.stacked, axes=((0,), (0,)))

    problem = _constant_delay_problem(
        drift,
        history,
        delays,
        t0=0.0,
        t1=1.2,
    )
    times = jnp.linspace(0.0, 1.2, 13)
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=times,
        dense=True,
    )
    expected = jnp.exp(rate * times)[:, None, None] * base

    assert solution.states.shape == (13, 2, 2)
    assert jnp.allclose(solution.states, expected, rtol=2e-6, atol=2e-7)
    dense = solution.evaluate(jnp.asarray([0.13, 0.71, 1.07]))
    dense_expected = jnp.exp(rate * jnp.asarray([0.13, 0.71, 1.07]))[:, None, None] * base
    assert jnp.allclose(dense, dense_expected, rtol=2e-6, atol=2e-7)


def test_diffrax_delay_supports_stiff_implicit_solver_and_stage_time_bound():
    delay = 0.2

    def history(time, args):
        return jnp.exp(-time) * jnp.ones((1,))

    def drift(time, state, delayed, args):
        del time, args
        return -1000.0 * state + 999.0 * jnp.exp(-delay) * delayed[0]

    problem = _constant_delay_problem(
        drift,
        history,
        jnp.asarray([delay]),
        t0=0.0,
        t1=0.5,
    )
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.linspace(0.0, 0.5, 6),
        solver=dfx.Kvaerno5(),
        dense=True,
        max_steps=1024,
    )

    assert bool(solution.successful)
    assert jnp.allclose(solution.states[-1, 0], jnp.exp(-0.5), atol=3e-6)
    assert solution.stats["stage_time_extent"] > 1.0
    interpolation = solution.interpolation
    assert interpolation is not None
    history_buffer = interpolation.computed_history
    used = int(history_buffer.size)
    lengths = history_buffer.ends[:used] - history_buffer.starts[:used]
    assert used == int(solution.stats["num_accepted_steps"])
    assert jnp.max(lengths) <= solution.stats["maximum_causal_step"] + 1e-14


def test_rejected_steps_never_enter_accepted_delay_history():
    solution = phx.solver.solve_diffrax_delay(
        _piecewise_problem(),
        save_times=jnp.asarray([2.0]),
        solver=dfx.Kvaerno5(),
        dense=True,
    )
    interpolation = solution.interpolation
    assert interpolation is not None
    history_buffer = interpolation.computed_history
    used = int(history_buffer.size)

    assert int(solution.stats["num_rejected_steps"]) > 0
    assert used == int(solution.stats["num_accepted_steps"])
    assert jnp.all(jnp.diff(history_buffer.starts[:used]) > 0.0)
    assert jnp.all(history_buffer.ends[:used] > history_buffer.starts[:used])
    assert jnp.allclose(solution.states[0, 0], 3.5, atol=2e-5)


def test_delay_discontinuity_schedule_generates_additive_descendants():
    delays = jnp.asarray([1.0, jnp.sqrt(2.0)])
    sources = jnp.asarray([-0.25, 0.0])
    schedule = _delay_discontinuity_times(
        delays,
        sources,
        depth=2,
        max_discontinuities=32,
    )
    offsets = jnp.asarray(
        [0.0, 1.0, jnp.sqrt(2.0), 2.0, 1.0 + jnp.sqrt(2.0), 2.0 * jnp.sqrt(2.0)]
    )
    expected = jnp.sort((sources[:, None] + offsets[None, :]).reshape((-1,)))

    assert schedule.shape == (12,)
    assert jnp.allclose(schedule, expected)
    empty = _delay_discontinuity_times(
        delays,
        jnp.asarray([]),
        depth=2,
        max_discontinuities=1,
    )
    assert empty.shape == (0,)
    with pytest.raises(ValueError, match="exceeds max_discontinuities"):
        _delay_discontinuity_times(
            delays,
            sources,
            depth=2,
            max_discontinuities=11,
        )


def test_diffrax_delay_is_jittable_vectorizable_and_differentiable():
    def terminal(rate):
        problem = _constant_delay_problem(
            lambda time, state, delayed, args: rate * state,
            lambda time, args: jnp.exp(rate * time) * jnp.ones((1,)),
            jnp.asarray([0.8]),
            t0=0.0,
            t1=0.5,
        )
        return phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.5]),
            max_steps=128,
        ).states[0, 0]

    rate = jnp.asarray(0.4)
    expected = jnp.exp(0.5 * rate)
    assert jnp.allclose(jax.jit(terminal)(rate), expected, atol=2e-7)
    assert jnp.allclose(jax.grad(terminal)(rate), 0.5 * expected, atol=2e-7)

    rates = jnp.asarray([0.1, 0.3])
    assert jnp.allclose(jax.vmap(terminal)(rates), jnp.exp(0.5 * rates), atol=2e-7)

    def dense_value(value):
        problem = _constant_delay_problem(
            lambda time, state, delayed, args: value * state,
            lambda time, args: jnp.exp(value * time) * jnp.ones((1,)),
            jnp.asarray([0.8]),
            t0=0.0,
            t1=0.5,
        )
        solution = phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([0.5]),
            dense=True,
            max_steps=128,
        )
        return solution.evaluate(jnp.asarray(0.37))[0]

    dense_expected = jnp.exp(0.37 * rate)
    assert jnp.allclose(dense_value(rate), dense_expected, atol=2e-7)
    assert jnp.allclose(
        jax.grad(dense_value)(rate),
        0.37 * dense_expected,
        atol=2e-7,
    )


def test_diffrax_delay_differentiates_constant_delay_away_from_schedule_changes():
    terminal_time = 0.25

    def terminal(delay):
        problem = _constant_delay_problem(
            lambda time, state, delayed, args: delayed[0],
            lambda time, args: jnp.asarray([time]),
            jnp.reshape(delay, (1,)),
            t0=0.0,
            t1=terminal_time,
        )
        return phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([terminal_time]),
            dt0=0.02,
            max_steps=128,
        ).states[0, 0]

    delay = jnp.asarray(0.5)
    expected = 0.5 * terminal_time**2 - delay * terminal_time
    assert jnp.allclose(terminal(delay), expected, atol=2e-9)
    assert jnp.allclose(jax.grad(terminal)(delay), -terminal_time, atol=2e-9)


def test_diffrax_delay_event_bounds_saved_and_dense_values():
    problem = _constant_delay_problem(
        lambda time, state, delayed, args: jnp.ones_like(state),
        lambda time, args: jnp.asarray([0.0]),
        jnp.asarray([0.5]),
        t0=0.0,
        t1=1.0,
    )
    event = dfx.Event(
        lambda t, y, args, **kwargs: y[0] - 0.3,
        root_finder=optx.Newton(rtol=1e-9, atol=1e-9),
    )
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.0, 0.2, 0.4, 0.8]),
        event=event,
        dense=True,
    )

    assert jnp.array_equal(solution.valid, jnp.asarray([True, True, False, False]))
    assert bool(solution.event_mask)
    assert jnp.allclose(
        solution.evaluate(jnp.asarray([0.1, 0.3]))[:, 0],
        jnp.asarray([0.1, 0.3]),
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="within the solved interval",
    ):
        solution.evaluate(jnp.asarray(0.31))


def test_diffrax_delay_validates_unsupported_configurations():
    problem = _piecewise_problem(t1=0.5)
    times = jnp.asarray([0.5])

    invalid_problem: Any = object()
    with pytest.raises(TypeError, match="DelayDifferentialProblem"):
        phx.solver.solve_diffrax_delay(invalid_problem, save_times=times)
    stochastic = _constant_delay_problem(
        lambda time, state, delayed, args: delayed[0],
        lambda time, args: jnp.ones((1,)),
        jnp.asarray([0.2]),
        t0=0.0,
        t1=0.5,
        diffusion=lambda time, state, delayed, args: jnp.ones((1, 1)),
        noise_shape=(1,),
    )
    with pytest.raises(ValueError, match="WienerRealization"):
        phx.solver.solve_diffrax_delay(stochastic, save_times=times)
    with pytest.raises(ValueError, match="dt0"):
        phx.solver.solve_diffrax_delay(problem, save_times=times, solver=dfx.Euler())
    with pytest.raises(ValueError, match="state_geometry"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=times,
            solver=phx.solver.GeometricEuler(phx.metrix.EuclideanStateGeometry()),
        )
    with pytest.raises(ValueError, match="BacksolveAdjoint"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=times,
            adjoint=dfx.BacksolveAdjoint(),
        )
    with pytest.raises(ValueError, match="finite max_steps"):
        phx.solver.solve_diffrax_delay(problem, save_times=times, max_steps=None)
    with pytest.raises(ValueError, match="positive integer"):
        phx.solver.solve_diffrax_delay(problem, save_times=times, max_steps=0)
    with pytest.raises(ValueError, match="rank-1"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=times,
            initial_discontinuities=jnp.zeros((1, 1)),
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=times,
            discontinuity_depth=-1,
        )
    with pytest.raises(ValueError, match="exceeds max_discontinuities"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=times,
            max_discontinuities=1,
        )

    no_schedule = phx.solver.solve_diffrax_delay(
        problem,
        save_times=times,
        initial_discontinuities=(),
        max_steps=128,
    )
    assert no_schedule.stats["num_tracked_discontinuities"] == 0
    assert not no_schedule.has_dense_interpolation
    with pytest.raises(ValueError, match="no dense interpolation"):
        no_schedule.evaluate(jnp.asarray(0.25))


def test_diffrax_delay_accepts_direct_adjoint():
    problem = _constant_delay_problem(
        lambda time, state, delayed, args: 0.2 * state,
        lambda time, args: jnp.ones((1,)),
        jnp.asarray([0.4]),
        t0=0.0,
        t1=0.3,
    )
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.3]),
        adjoint=dfx.DirectAdjoint(),
        max_steps=64,
    )
    assert jnp.allclose(solution.states[0, 0], jnp.exp(0.06), atol=2e-7)
