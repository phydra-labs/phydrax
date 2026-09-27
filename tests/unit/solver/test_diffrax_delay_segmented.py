from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optimistix as optx
import pytest

import phydrax as phx
from phydrax.solver._delay_history import RollingDelayHistory


class _LinearInterpolation(dfx.AbstractLocalInterpolation):
    t0: jax.Array  # ty: ignore[invalid-attribute-override]
    t1: jax.Array  # ty: ignore[invalid-attribute-override]
    y0: jax.Array
    y1: jax.Array

    def evaluate(self, t0: Any, t1: Any = None, left: Any = True) -> Any:
        del left
        start = jnp.asarray(t0)

        def value(time: Any) -> Any:
            fraction = (time - self.t0) / (self.t1 - self.t0)
            return self.y0 + fraction * (self.y1 - self.y0)

        if t1 is None:
            return value(start)
        return value(jnp.asarray(t1)) - value(start)


def _problem(
    *,
    t1: Any = 2.0,
    delay: Any = 0.5,
    rate: Any = 0.2,
    problem_id: Any = "segmented-test",
) -> Any:
    def history(time: Any, args: Any) -> Any:
        del args
        return jnp.exp(rate * time) * jnp.ones((1,))

    def drift(time: Any, state: Any, memory: Any, args: Any) -> Any:
        del time, state, args
        return rate * jnp.exp(rate * delay) * memory["past"]

    return phx.solver.DelayDifferentialProblem(
        drift,
        history,
        (phx.solver.ConstantDelay("past", delay),),
        t0=0.0,
        t1=t1,
        problem_id=problem_id,
    )


def _fixed_segmented(problem: Any, times: Any, **kwargs: Any) -> Any:
    return phx.solver.solve_diffrax_delay_segmented(
        problem,
        save_times=times,
        solver=dfx.Tsit5(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.05,
        max_steps_per_segment=7,
        **kwargs,
    )


def test_segmented_contracts() -> None:
    problem = _problem(t1=3.0)
    times = jnp.linspace(0.0, 3.0, 31)
    one_shot = phx.solver.solve_diffrax_delay(
        problem,
        save_times=times,
        solver=dfx.Tsit5(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.05,
        max_steps=512,
    )
    segmented = _fixed_segmented(problem, times)

    assert segmented.backend_result is phx.solver.SegmentedDelayResult.successful
    assert jnp.array_equal(segmented.valid, one_shot.valid)
    assert jnp.allclose(segmented.states, one_shot.states, rtol=2e-12, atol=2e-12)
    assert int(segmented.stats["num_segments"]) > 1
    assert segmented.stats["controller_mode"] == "fixed"
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: jnp.ones_like(state),
        lambda time, args: jnp.zeros((1,)),
        (phx.solver.ConstantDelay("past", 0.5),),
        t0=0.0,
        t1=1.0,
    )
    event = dfx.Event(
        lambda t, y, args, **kwargs: y[0] - 0.3,
        root_finder=optx.Newton(rtol=1e-10, atol=1e-10),
    )
    solution = _fixed_segmented(
        problem,
        jnp.asarray([0.0, 0.2, 0.4, 0.8]),
        event=event,
        dense=True,
    )

    assert solution.backend_result is phx.solver.SegmentedDelayResult.event_occurred
    assert jnp.array_equal(solution.valid, jnp.asarray([True, True, False, False]))
    assert jnp.allclose(solution.continuation.time, 0.3, atol=2e-10)
    active = solution.continuation.active_history
    visible_ends = active.logical_ends[jnp.isfinite(active.logical_ends)]
    assert jnp.max(visible_ends) <= solution.continuation.time
    assert jnp.max(active.ends) <= solution.continuation.time
    assert not solution.continuation.resumable
    archive = solution.interpolation
    assert archive is not None
    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(archive))
    assert jnp.allclose(
        solution.evaluate(jnp.asarray([0.1, 0.3]))[:, 0], jnp.asarray([0.1, 0.3])
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="outside the archived"
    ):
        solution.evaluate(jnp.asarray(0.31))
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="outside retained"):
        active.evaluate(jnp.asarray(0.31))
    with pytest.raises(ValueError, match="terminal and cannot be resumed"):
        _fixed_segmented(
            problem,
            jnp.asarray([solution.continuation.time, problem.t1]),
            continuation=solution.continuation,
        )
    state_dependent = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: memory["past"],
        lambda time, args: jnp.ones((1,)),
        (
            phx.solver.StateDependentDelay(
                "past",
                lambda time, state, args: jnp.asarray(0.2),
                minimum_delay=0.1,
            ),
        ),
        t0=0.0,
        t1=0.5,
    )
    with pytest.raises(ValueError, match="finite maximum lag"):
        phx.solver.solve_diffrax_delay_segmented(
            state_dependent,
            save_times=jnp.asarray([0.5]),
            history_capacity=16,
        )

    with pytest.raises(ValueError, match="explicit history_capacity"):
        phx.solver.solve_diffrax_delay_segmented(
            _problem(t1=0.5),
            save_times=jnp.asarray([0.5]),
            solver=dfx.Tsit5(),
        )
    bounded_state_dependent = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: memory["past"],
        lambda time, args: jnp.ones((1,)),
        (
            phx.solver.StateDependentDelay(
                "past",
                lambda time, state, args: jnp.asarray(0.2),
                minimum_delay=0.1,
                maximum_delay=0.3,
            ),
        ),
        t0=0.0,
        t1=0.8,
    )
    times = jnp.linspace(0.0, 0.8, 9)
    whole = phx.solver.solve_diffrax_delay(
        bounded_state_dependent,
        save_times=times,
        max_steps=4096,
    )
    segmented = phx.solver.solve_diffrax_delay_segmented(
        bounded_state_dependent,
        save_times=times,
        history_capacity=64,
        max_steps_per_segment=16,
    )
    assert jnp.allclose(segmented.states, whole.states, rtol=1e-7, atol=1e-8)
    assert segmented.stats["state_dependent_tracking"] == "high-order-dynamic-roots"
    assert segmented.stats["num_dynamic_discontinuity_roots"] > 0
    assert segmented.stats["num_segments"] > 1

    point = phx.solver.ConstantDelay("point", 0.2)
    neutral = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: memory["derivative"],
        lambda time, args: jnp.ones((1,)),
        (phx.solver.DerivativeDelay("derivative", point),),
        t0=0.0,
        t1=0.5,
        history_derivative=lambda time, args: jnp.zeros((1,)),
    )
    neutral_segmented = phx.solver.solve_diffrax_delay_segmented(
        neutral,
        save_times=jnp.linspace(0.0, 0.5, 6),
        solver=dfx.Tsit5(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.05,
        max_steps_per_segment=3,
    )
    assert jnp.allclose(neutral_segmented.states, 1.0)
    assert neutral_segmented.stats["num_segments"] > 1
    term = phx.solver.DistributedDelay(
        "spread",
        lambda time, lag, state, args: jnp.asarray(5.0),
        (0.2, 0.4),
        quadrature=phx.integration.GaussLegendreRule(4),
    )
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: memory["spread"],
        lambda time, args: jnp.ones((1,)),
        (term,),
        t0=0.0,
        t1=0.8,
    )
    times = jnp.linspace(0.0, 0.8, 9)
    whole = phx.solver.solve_diffrax_delay(
        problem,
        save_times=times,
        rtol=1e-9,
        atol=1e-11,
        max_steps=2048,
    )
    segmented = phx.solver.solve_diffrax_delay_segmented(
        problem,
        save_times=times,
        rtol=1e-9,
        atol=1e-11,
        history_capacity=64,
        max_steps_per_segment=8,
    )

    assert jnp.allclose(segmented.states, whole.states, rtol=1e-8, atol=1e-9)
    assert segmented.stats["num_segments"] > 1


def test_diffrax_delay_segmented_scenario_1() -> None:
    short = _fixed_segmented(_problem(t1=2.0), jnp.asarray([2.0]))
    long = _fixed_segmented(_problem(t1=8.0), jnp.asarray([8.0]))

    assert short.stats["history_capacity"] == long.stats["history_capacity"]
    assert short.stats["active_history_bytes"] == long.stats["active_history_bytes"]
    assert long.continuation.active_history.size <= long.stats["history_capacity"]
    assert int(long.stats["num_segments"]) > int(short.stats["num_segments"])
    structure = {
        "y0": jax.ShapeDtypeStruct((1,), jnp.float64),
        "y1": jax.ShapeDtypeStruct((1,), jnp.float64),
    }
    history = RollingDelayHistory.allocate(
        time=jnp.asarray(0.0),
        dense_info_structure=structure,
        capacity=3,
        interpolation_cls=_LinearInterpolation,
        maximum_lag=jnp.asarray(1.0),
    )
    for start in (0.0, 0.4, 0.8, 1.2):
        history = history.append(
            jnp.asarray(start),
            jnp.asarray(start + 0.4),
            {"y0": jnp.asarray([start]), "y1": jnp.asarray([start + 0.4])},
        )

    assert int(history.start) != 0
    assert jnp.allclose(history.logical_starts, jnp.asarray([0.4, 0.8, 1.2]))
    assert jnp.allclose(
        history.values(jnp.asarray([0.5, 1.0, 1.5]))[:, 0],
        jnp.asarray([0.5, 1.0, 1.5]),
    )
    structure = {
        "y0": jax.ShapeDtypeStruct((1,), jnp.float64),
        "y1": jax.ShapeDtypeStruct((1,), jnp.float64),
    }
    accepted = RollingDelayHistory.allocate(
        time=jnp.asarray(0.0),
        dense_info_structure=structure,
        capacity=4,
        interpolation_cls=_LinearInterpolation,
        maximum_lag=jnp.asarray(1.0),
    ).append(
        jnp.asarray(0.0),
        jnp.asarray(0.25),
        {"y0": jnp.asarray([0.0]), "y1": jnp.asarray([0.25])},
    )
    candidate = accepted.append(
        jnp.asarray(0.25),
        jnp.asarray(0.5),
        {"y0": jnp.asarray([0.25]), "y1": jnp.asarray([10.0])},
    )

    assert accepted.size == 1
    assert jnp.allclose(accepted.evaluate(jnp.asarray(0.25)), jnp.asarray([0.25]))
    assert candidate.size == 2
    assert jnp.allclose(candidate.evaluate(jnp.asarray(0.5)), jnp.asarray([10.0]))
    solution = phx.solver.solve_diffrax_delay_segmented(
        _problem(t1=0.5, delay=1.0),
        save_times=jnp.asarray([0.5]),
        solver=dfx.Kvaerno5(),
        dt0=0.5,
        rtol=1e-10,
        atol=1e-12,
        history_capacity=64,
        max_steps_per_segment=128,
    )
    active = solution.continuation.active_history

    assert int(solution.stats["num_rejected_steps"]) > 0
    assert active.size == int(solution.stats["num_accepted_steps"])
    starts = active.logical_starts[: active.size]
    assert jnp.all(jnp.diff(starts) > 0.0)


@pytest.mark.filterwarnings("error:invalid value encountered in cast:RuntimeWarning")
def test_scalar_stochastic_segments_replay_one_realization() -> None:
    problem = phx.solver.DelayDifferentialProblem(
        lambda time, state, memory, args: 0.3 * memory["past"],
        lambda time, args: jnp.ones((1,)),
        (phx.solver.ConstantDelay("past", 0.2),),
        t0=0.0,
        t1=0.8,
        wiener_terms=(
            phx.solver.DelayWienerTerm(
                "noise",
                lambda time, state, memory, args: 0.4 * jnp.ones(state.shape + (1,)),
                (1,),
                structure="additive",
                basis_id="segmented-path",
            ),
        ),
    )
    realization = phx.stochastic.WienerRealization(
        jr.key(41),
        problem.noise_shape,
        support=(0.0, 0.8),
        tolerance=1e-4,
        noise_id=problem.noise_id,
    )
    times = jnp.linspace(0.0, 0.8, 9)
    one_shot = phx.solver.solve_diffrax_delay(
        problem,
        save_times=times,
        realization=realization,
        solver=dfx.Euler(),
        dt0=0.05,
        max_steps=128,
    )
    uninterrupted_rolling = phx.solver.solve_diffrax_delay_segmented(
        problem,
        save_times=times,
        realization=realization,
        solver=dfx.Euler(),
        dt0=0.05,
        max_steps_per_segment=128,
    )
    segmented = phx.solver.solve_diffrax_delay_segmented(
        problem,
        save_times=times,
        realization=realization,
        solver=dfx.Euler(),
        dt0=0.05,
        max_steps_per_segment=3,
    )

    assert segmented.realization is realization
    assert segmented.continuation.realization is realization
    assert int(segmented.stats["num_segments"]) > 1
    assert jnp.array_equal(segmented.states, uninterrupted_rolling.states)
    assert jnp.allclose(segmented.states, one_shot.states, rtol=1e-7, atol=1e-9)


def test_diffrax_delay_segmented_scenario_2() -> None:
    problem = _problem(t1=2.0, problem_id="restartable")
    full = _fixed_segmented(problem, jnp.asarray([2.0]))
    partial = _fixed_segmented(
        problem,
        jnp.asarray([0.0]),
        max_segments=1,
    )
    restart_times = jnp.asarray([partial.continuation.time, problem.t1])
    restarted = _fixed_segmented(
        problem,
        restart_times,
        continuation=partial.continuation,
    )

    assert partial.backend_result is phx.solver.SegmentedDelayResult.segment_limit_reached
    assert jnp.allclose(restarted.states[-1], full.states[-1], rtol=2e-12, atol=2e-12)
    assert int(restarted.stats["num_segments"]) == int(full.stats["num_segments"])
    problem = _problem(t1=1.0)
    solution = phx.solver.solve_diffrax_delay_segmented(
        problem,
        save_times=jnp.asarray([1.0]),
        solver=dfx.Tsit5(),
        history_capacity=1,
        max_steps_per_segment=16,
    )

    assert (
        solution.backend_result
        is phx.solver.SegmentedDelayResult.history_capacity_exhausted
    )
    assert bool(solution.continuation.active_history.overflowed)
    with pytest.raises(RuntimeError, match="exhausted history_capacity"):
        phx.solver.solve_diffrax_delay_segmented(
            problem,
            save_times=jnp.asarray([1.0]),
            solver=dfx.Tsit5(),
            history_capacity=1,
            max_steps_per_segment=16,
            throw=True,
        )


def test_whole_solve_jit_is_rejected_as_host_dynamic() -> None:
    problem = _problem(t1=0.5)

    @jax.jit
    def run(t1: Any) -> Any:
        traced = phx.solver.DelayDifferentialProblem(
            problem.drift,
            problem.history,
            problem.delay_terms,
            t0=0.0,
            t1=t1,
        )
        return phx.solver.solve_diffrax_delay_segmented(
            traced,
            save_times=jnp.asarray([0.5]),
            solver=dfx.Tsit5(),
            history_capacity=16,
        ).states

    with pytest.raises(TypeError, match="host driver"):
        run(jnp.asarray(0.5))
