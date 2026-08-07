#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

import phydrax as phx


def _piecewise_problem(*, scale=1.0, t1=2.0):
    return phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: args * memory["lag"],
        lambda time, args: jnp.ones((1,)),
        (phx.solver.ConstantDelay("lag", 1.0),),
        t0=0.0,
        t1=t1,
        args=scale,
    )


def _piecewise_exact(times, scale=1.0):
    first = 1.0 + scale * times
    shifted = times - 1.0
    second = 1.0 + scale * times + 0.5 * scale**2 * shifted**2
    return jnp.where(times <= 1.0, first, second)


@pytest.mark.parametrize("solver", [dfx.Euler(), dfx.Heun(), dfx.Tsit5()])
def test_fixed_diffrax_delay_solvers_execute_through_unified_api(solver):
    times = jnp.linspace(0.0, 2.0, 41)
    solution = phx.solver.solve_diffrax_delay(
        _piecewise_problem(),
        save_times=times,
        solver=solver,
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.025,
        dense=True,
    )

    tolerance = 2e-2 if isinstance(solver, dfx.Euler) else 2e-6
    assert jnp.allclose(solution.states[:, 0], _piecewise_exact(times), atol=tolerance)
    assert solution.stats["num_rejected_steps"] == 0
    assert solution.stats["maximum_causal_step"] == 1.0
    assert jnp.allclose(
        solution.evaluate(jnp.asarray([0.35, 1.25, 1.9]))[:, 0],
        _piecewise_exact(jnp.asarray([0.35, 1.25, 1.9])),
        atol=tolerance,
    )


def test_fixed_controller_shortens_a_step_at_propagated_discontinuity():
    solution = phx.solver.solve_diffrax_delay(
        _piecewise_problem(),
        save_times=jnp.asarray([0.0, 1.0, 2.0]),
        solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.3,
    )

    assert solution.stats["num_tracked_discontinuities"] >= 2
    assert solution.stats["num_accepted_steps"] == 8
    assert jnp.allclose(solution.states[:, 0], jnp.asarray([1.0, 2.0, 3.36]))


def test_fixed_implicit_delay_solver_uses_explicit_nonlinear_tolerances():
    delay = 0.2
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: (
            -50.0 * state + 49.0 * jnp.exp(-delay) * memory["lag"]
        ),
        lambda time, args: jnp.exp(-time).reshape((1,)),
        (phx.solver.ConstantDelay("lag", delay),),
        t0=0.0,
        t1=0.5,
    )
    solver = dfx.ImplicitEuler(
        root_finder=optx.Chord(rtol=1e-10, atol=1e-10, norm=optx.rms_norm)
    )
    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([0.5]),
        solver=solver,
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.005,
        max_steps=128,
    )
    assert jnp.isclose(solution.states[0, 0], jnp.exp(-0.5), atol=2e-3)
    assert solution.stats["controller_mode"] == "fixed"


def test_fixed_delay_solver_converges_at_euler_order():
    def terminal(step):
        solution = phx.solver.solve_diffrax_delay(
            _piecewise_problem(),
            save_times=jnp.asarray([2.0]),
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=step,
        )
        return solution.states[0, 0]

    errors = jnp.asarray(
        [abs(float(terminal(step)) - 3.5) for step in (0.2, 0.1, 0.05, 0.025)]
    )
    orders = jnp.log2(errors[:-1] / errors[1:])
    assert jnp.all(orders > 0.85)


def test_fixed_delay_solver_is_jittable_vectorizable_and_differentiable():
    def terminal(scale):
        solution = phx.solver.solve_diffrax_delay(
            _piecewise_problem(scale=scale),
            save_times=jnp.asarray([2.0]),
            solver=dfx.Heun(),
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=0.05,
        )
        return solution.states[0, 0]

    scales = jnp.asarray([0.5, 1.0, 1.5])
    observed = jax.jit(jax.vmap(terminal))(scales)
    expected = _piecewise_exact(jnp.full_like(scales, 2.0), scales)
    assert jnp.allclose(observed, expected, atol=1e-6)
    assert jnp.isclose(jax.grad(terminal)(1.0), 3.0, atol=1e-5)

def test_rolling_whole_solve_matches_full_history_with_bounded_storage():
    problem = _piecewise_problem(t1=4.0)
    times = jnp.linspace(0.0, 4.0, 41)
    common = {
        "save_times": times,
        "solver": dfx.Tsit5(),
        "stepsize_controller": dfx.ConstantStepSize(),
        "dt0": 0.05,
        "dense": True,
    }
    full = phx.solver.solve_diffrax_delay(problem, max_steps=128, **common)
    rolling = phx.solver.solve_diffrax_delay(
        problem,
        history_mode="rolling",
        max_steps=None,
        **common,
    )

    assert jnp.allclose(rolling.states, full.states, rtol=1e-10, atol=1e-10)
    assert rolling.stats["history_mode"] == "rolling"
    assert rolling.stats["history_capacity"] == 22
    assert rolling.stats["history_max_occupancy"] <= 22
    assert rolling.stats["num_history_evictions"] > 0
    assert not rolling.stats["history_capacity_exhausted"]
    assert rolling.stats["retained_history_interval"][0] > 0.0
    assert jnp.allclose(
        rolling.evaluate(jnp.asarray([3.25, 3.75])),
        full.evaluate(jnp.asarray([3.25, 3.75])),
        rtol=1e-10,
        atol=1e-10,
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="available solved interval"):
        rolling.evaluate(jnp.asarray([0.5]))


def test_fixed_history_capacity_counts_only_step_splitting_breakpoints():
    aligned = phx.solver.fixed_delay_history_capacity(
        1.0,
        0.05,
        breakpoints=jnp.asarray([1.0, 2.0, 3.0]),
        initial_time=0.0,
    )
    split = phx.solver.fixed_delay_history_capacity(
        1.0,
        0.2,
        breakpoints=jnp.asarray([0.35, 0.7]),
        initial_time=0.0,
    )

    assert aligned == 22
    assert split == 9


def test_rolling_history_capacity_is_explicit_for_adaptive_execution():
    problem = _piecewise_problem(t1=2.0)
    with pytest.raises(ValueError, match="explicit history_capacity"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([2.0]),
            history_mode="rolling",
            max_steps=None,
        )

    solution = phx.solver.solve_diffrax_delay(
        problem,
        save_times=jnp.asarray([2.0]),
        history_mode="rolling",
        history_capacity=1,
        max_steps=None,
    )
    assert solution.stats["history_capacity_exhausted"]
    assert solution.backend_result == dfx.RESULTS.max_steps_reached
    with pytest.raises(eqx.EquinoxRuntimeError, match="exhausted history_capacity"):
        phx.solver.solve_diffrax_delay(
            problem,
            save_times=jnp.asarray([2.0]),
            history_mode="rolling",
            history_capacity=1,
            max_steps=None,
            throw=True,
        )


def test_fixed_delay_solver_rejects_missing_or_noncausal_step_size():
    common = {
        "save_times": jnp.asarray([1.0]),
        "solver": dfx.Euler(),
        "stepsize_controller": dfx.ConstantStepSize(),
    }
    boundary = phx.solver.solve_diffrax_delay(
        _piecewise_problem(t1=1.0),
        dt0=1.0,
        **common,
    )
    assert jnp.isclose(boundary.states[0, 0], 2.0)
    with pytest.raises(ValueError, match="require dt0"):
        phx.solver.solve_diffrax_delay(_piecewise_problem(), **common)
    with pytest.raises(Exception, match="causal delay step bound"):
        phx.solver.solve_diffrax_delay(_piecewise_problem(), dt0=1.1, **common)


def test_fixed_delay_solver_rejects_unconstrained_fixed_controller():
    with pytest.raises(ValueError, match="ConstantStepSize"):
        phx.solver.solve_diffrax_delay(
            _piecewise_problem(),
            save_times=jnp.asarray([2.0]),
            solver=dfx.Euler(),
            stepsize_controller=dfx.StepTo(ts=jnp.asarray([0.0, 1.0, 2.0])),
        )
