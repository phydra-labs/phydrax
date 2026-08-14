#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from math import factorial

import diffrax as dfx
import jax.numpy as jnp
import pytest

import phydrax as phx


def _smooth_control(times):
    return phx.stochastic.GeometricRoughPath.from_values(times, times[:, None])


def test_rough_delay_euler_reduces_to_fixed_step_dde_on_smooth_driver():
    times = jnp.linspace(0.0, 1.0, 21)
    delay = phx.solver.ConstantDelay("past", 0.2)
    rate = 0.7
    rough_problem = phx.solver.RoughDelayDifferentialProblem(
        lambda time, state, memory, args: (rate * memory["past"])[..., None],
        lambda time, args: jnp.ones((1,)),
        (delay,),
        t0=0.0,
        driver_dimension=1,
        problem_id="rough-delay:smooth-reduction",
    )
    rough = phx.solver.solve_rough_delay(
        rough_problem,
        _smooth_control(times),
        solver=phx.solver.RoughEuler(),
    )
    dde_problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: rate * memory["past"],
        lambda time, args: jnp.ones((1,)),
        (delay,),
        t0=0.0,
        t1=1.0,
    )
    dde = phx.solver.solve_diffrax_delay(
        dde_problem,
        save_times=times,
        solver=dfx.Euler(),
        dt0=0.05,
        max_steps=64,
    )

    assert jnp.allclose(rough.states, dde.states, rtol=0.0, atol=5e-15)
    assert bool(rough.successful)
    assert rough.metadata["equation_kind"] == "rough-retarded"
    assert rough.metadata["history_interpolation"] == "retraction-linear"
    assert rough.metadata["delayed_second_level"] == "not-required-young"


def test_delayed_davie_cross_level_improves_constant_delay_accuracy():
    times = jnp.linspace(0.0, 1.0, 21)
    delay_value = 0.2
    rate = 0.8
    delay = phx.solver.ConstantDelay("past", delay_value)
    problem = phx.solver.RoughDelayDifferentialProblem(
        lambda time, state, memory, args: (rate * memory["past"])[..., None],
        lambda time, args: jnp.ones((1,)),
        (delay,),
        t0=0.0,
        driver_dimension=1,
    )
    control = _smooth_control(times)
    euler = phx.solver.solve_rough_delay(
        problem,
        control,
        solver=phx.solver.RoughEuler(),
        save_times=jnp.asarray([1.0]),
    )
    davie = phx.solver.solve_rough_delay(
        problem,
        control,
        solver=phx.solver.Davie(),
        save_times=jnp.asarray([1.0]),
    )
    exact = 1.0 + sum(
        rate**order * (1.0 - (order - 1) * delay_value) ** order / factorial(order)
        for order in range(1, 6)
    )
    euler_error = jnp.abs(euler.states[0, 0] - exact)
    davie_error = jnp.abs(davie.states[0, 0] - exact)

    assert davie_error < 0.02 * euler_error
    assert davie.metadata["delayed_second_level"] == (
        "grid-aligned-piecewise-linear-cross-integrals"
    )


def test_young_rough_delay_supports_bounded_history_functionals():
    times = jnp.linspace(0.0, 0.8, 17)
    lags = jnp.asarray([0.2, 0.3, 0.4])
    functional = phx.solver.FunctionalDelay(
        "window",
        lambda time, state, history, args: jnp.mean(history.values(lags), axis=0),
        (0.2, 0.4),
    )
    problem = phx.solver.RoughDelayDifferentialProblem(
        lambda time, state, memory, args: (0.4 * memory["window"])[..., None],
        lambda time, args: jnp.ones((1,)),
        (functional,),
        t0=0.0,
        driver_dimension=1,
    )
    solution = phx.solver.solve_rough_delay(
        problem,
        _smooth_control(times),
        solver=phx.solver.RoughEuler(),
    )

    assert solution.states.shape == (17, 1)
    assert jnp.all(jnp.isfinite(solution.states))
    assert jnp.all(jnp.diff(solution.states[:, 0]) > 0.0)
    with pytest.raises(ValueError, match="constant point delays"):
        phx.solver.solve_rough_delay(
            problem,
            _smooth_control(times),
            solver=phx.solver.Davie(),
        )
