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

    def evaluate(self, t0, t1=None, left=True):
        del left
        start = jnp.asarray(t0)

        def value(time):
            fraction = (time - self.t0) / (self.t1 - self.t0)
            return self.y0 + fraction * (self.y1 - self.y0)

        if t1 is None:
            return value(start)
        return value(jnp.asarray(t1)) - value(start)


def _problem(*, t1=2.0, delay=0.5, rate=0.2, problem_id="segmented-test"):
    def history(time, args):
        del args
        return jnp.exp(rate * time) * jnp.ones((1,))

    def drift(time, state, memory, args):
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


def _fixed_segmented(problem, times, **kwargs):
    return phx.solver.solve_diffrax_delay_segmented(
        problem,
        save_times=times,
        solver=dfx.Tsit5(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.05,
        max_steps_per_segment=7,
        **kwargs,
    )


def test_segmented_and_one_shot_fixed_solves_are_equivalent():
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


def test_active_history_bytes_plateau_with_horizon():
    short = _fixed_segmented(_problem(t1=2.0), jnp.asarray([2.0]))
    long = _fixed_segmented(_problem(t1=8.0), jnp.asarray([8.0]))

    assert short.stats["history_capacity"] == long.stats["history_capacity"]
    assert short.stats["active_history_bytes"] == long.stats["active_history_bytes"]
    assert int(long.continuation.active_history.size) <= long.stats["history_capacity"]
    assert int(long.stats["num_segments"]) > int(short.stats["num_segments"])


def test_rolling_history_wrap_preserves_logical_lookup_order():
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


def test_rejected_candidate_history_is_functionally_isolated():
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

    assert int(accepted.size) == 1
    assert jnp.allclose(accepted.evaluate(jnp.asarray(0.25)), jnp.asarray([0.25]))
    assert int(candidate.size) == 2
    assert jnp.allclose(candidate.evaluate(jnp.asarray(0.5)), jnp.asarray([10.0]))


def test_rejected_diffrax_candidates_never_enter_rolling_history():
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
    assert int(active.size) == int(solution.stats["num_accepted_steps"])
    starts = active.logical_starts[: int(active.size)]
    assert jnp.all(jnp.diff(starts) > 0.0)


def test_segmented_event_stops_archive_at_root_boundary():
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


@pytest.mark.filterwarnings("error:invalid value encountered in cast:RuntimeWarning")
def test_scalar_stochastic_segments_replay_one_realization():
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


def test_continuation_restart_matches_uninterrupted_segments():
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


def test_adaptive_history_overflow_is_an_explicit_result():
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


def test_segmented_requires_maximum_lag_and_adaptive_capacity():
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


def test_segmented_state_dependent_and_neutral_delays_match_whole_solve():
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


def test_segmented_distributed_delay_matches_whole_solve():
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


def test_whole_solve_jit_is_rejected_as_host_dynamic():
    problem = _problem(t1=0.5)

    @jax.jit
    def run(t1):
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
