#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


# Relaxation y' = -RATE (y - 1), y(0) = 0: a fast initial transient followed by an
# increasingly quiet approach to equilibrium, advanced by backward Euler.
RATE = 20.0
INCREMENT_LIMIT = 0.05
MINIMUM_STEP = 1.0e-3
MAXIMUM_STEP = 0.25


def _backward_euler(
    step_index: Array, time: Array, state: Array, step_size: Array, args: Any
) -> Any:
    del step_index, time, args
    accepted = (state + step_size * RATE) / (1.0 + step_size * RATE)
    return phx.solver.FixedStepResult(
        candidate_state=accepted,
        accepted_state=accepted,
        successful=jnp.asarray(True),
        residual=jnp.zeros((), dtype=accepted.dtype),
        iterations=jnp.asarray(1, dtype=jnp.int32),
        work=jnp.asarray(1, dtype=jnp.int32),
        transform_applied=jnp.asarray(False),
        transform_correction_norm=jnp.zeros((), dtype=accepted.dtype),
    )


def _increments(previous: Array, accepted: Array, step: Array, args: Any) -> Array:
    del step, args
    return jnp.abs(accepted - previous)


METHOD = phx.solver.CallableFixedStepMethod(
    _backward_euler, "method:relaxation-backward-euler"
)


def _controller(
    *,
    limit: float = INCREMENT_LIMIT,
    minimum_step: float = MINIMUM_STEP,
    maximum_step: float = MAXIMUM_STEP,
) -> phx.solver.IncrementLimitedStepController:
    return phx.solver.IncrementLimitedStepController(
        _increments,
        jnp.asarray((limit,)),
        minimum_step=minimum_step,
        maximum_step=maximum_step,
        controller_id="increment:state-change",
    )


def _manifest() -> phx.solver.ProductionCaseManifest:
    return phx.solver.ProductionCaseManifest(
        problem_id="relaxation",
        method_id=METHOD.method_id,
        precision_id="float64",
        topology_id="one-state",
        geometry_layout_id="static",
        dtype="float64",
    )


def _store(root: Path) -> phx.solver.DurableCheckpointStore:
    return phx.solver.DurableCheckpointStore(
        root, _manifest(), phx.solver.CheckpointGenerationPolicy(4)
    )


def _committed_steps(
    root: Path,
    plan: phx.solver.ProductionRunPlan,
    **options: Any,
) -> tuple[phx.solver.ProductionRunResult, list[Any]]:
    committed: list[Any] = []
    session = phx.execution.IterationSession(
        "adaptive-production",
        sinks=(
            phx.execution.CallableIterationSink(
                lambda event: (
                    committed.append(event.record.metrics)
                    if int(event.record.coordinates.phase)
                    == int(phx.execution.IterationPhase.COMMIT)
                    else None
                ),
                "commit-collector",
            ),
        ),
    )
    prepared = phx.solver.PreparedProductionRun(
        _manifest(), plan, _store(root), session=session, **options
    )
    return prepared.run(prepared.initial_state(jnp.zeros((1,)))), committed


def test_adaptive_steps_land_on_every_schedule_target(tmp_path: Path) -> None:
    targets = (0.013, 0.37, 0.8125, 1.5, 2.0)
    schedule = phx.solver.ExactTimeSchedule(jnp.asarray(targets))
    plan = phx.solver.ProductionRunPlan(
        METHOD,
        phx.solver.RobustRetryPolicy(maximum_retries=0),
        step_size=0.05,
        end_time=2.0,
        maximum_steps=400,
        checkpoint_interval=1000,
        output_schedule=schedule,
        step_controller=_controller(),
    )
    published: list[Any] = []
    publisher = phx.solver.ByteBoundedAsyncPublisher(
        lambda event_id, snapshot: published.append(snapshot),
        maximum_pending=8,
        maximum_pending_bytes=4096,
    )
    result, committed = _committed_steps(tmp_path / "schedule", plan, publisher=publisher)

    assert result.successful
    times = np.asarray([float(metrics.time) for metrics in committed])
    output_times = np.asarray(
        [float(metrics.time) for metrics in committed if bool(metrics.output_due)]
    )
    np.testing.assert_allclose(output_times, targets, rtol=0.0, atol=schedule.tolerance)
    # No accepted step crosses a target without landing on it.
    for target in targets:
        assert np.min(np.abs(times - target)) <= schedule.tolerance
    assert len(published) == len(targets)
    assert float(result.state.time) == pytest.approx(2.0, abs=schedule.tolerance)


def test_adaptive_steps_shrink_on_large_increments_and_grow_to_maximum(
    tmp_path: Path,
) -> None:
    initial_step = 0.1
    plan = phx.solver.ProductionRunPlan(
        METHOD,
        phx.solver.RobustRetryPolicy(maximum_retries=0),
        step_size=initial_step,
        end_time=4.0,
        maximum_steps=400,
        checkpoint_interval=1000,
        step_controller=_controller(),
    )
    result, committed = _committed_steps(tmp_path / "growth", plan)

    assert result.successful
    steps = np.asarray([float(metrics.accepted_step_size) for metrics in committed])
    # First increment 2/3 of the gap exceeds the limit: the next step shrinks.
    assert steps[0] == initial_step
    assert steps[1] < steps[0]
    assert np.all(steps <= MAXIMUM_STEP)
    # The quiet tail runs at the ceiling, which the transient never reached.
    assert np.max(steps[:5]) < MAXIMUM_STEP
    assert np.count_nonzero(steps == MAXIMUM_STEP) >= 5


def test_adaptive_restart_reproduces_continued_run_bitwise(tmp_path: Path) -> None:
    plan = phx.solver.ProductionRunPlan(
        METHOD,
        phx.solver.RobustRetryPolicy(maximum_retries=0),
        step_size=0.05,
        end_time=3.0,
        maximum_steps=400,
        checkpoint_interval=5,
        segment_steps=8,
        step_controller=_controller(),
    )
    initial_value = jnp.zeros((1,))
    reference_runtime = phx.solver.PreparedProductionRun(
        _manifest(), plan, _store(tmp_path / "reference")
    )
    reference = reference_runtime.run(reference_runtime.initial_state(initial_value))
    assert reference.successful

    store = _store(tmp_path / "interrupted")
    interrupted = phx.solver.PreparedProductionRun(_manifest(), plan, store)
    state = interrupted.initial_state(initial_value)
    for _ in range(5):
        state, _transition = interrupted.step(state)
    assert state.last_checkpoint_id

    restarted = phx.solver.PreparedProductionRun(_manifest(), plan, store)
    resumed = restarted.resume(restarted.initial_state(initial_value))
    assert int(resumed.step_index) == 5
    np.testing.assert_array_equal(resumed.controller_state, state.controller_state)
    np.testing.assert_array_equal(resumed.time, state.time)
    np.testing.assert_array_equal(resumed.accepted_state, state.accepted_state)

    continued = restarted.run(resumed)
    assert continued.successful
    assert int(continued.state.step_index) == int(reference.state.step_index)
    np.testing.assert_array_equal(continued.state.time, reference.state.time)
    np.testing.assert_array_equal(
        continued.state.accepted_state, reference.state.accepted_state
    )
    np.testing.assert_array_equal(
        continued.state.controller_state, reference.state.controller_state
    )


def test_minimum_step_exhausting_capacity_fails_without_success(tmp_path: Path) -> None:
    # A tiny increment limit pins every step at minimum_step, and
    # 10 * minimum_step cannot reach end_time although 10 * maximum_step could.
    plan = phx.solver.ProductionRunPlan(
        METHOD,
        phx.solver.RobustRetryPolicy(maximum_retries=0),
        step_size=0.05,
        end_time=1.0,
        maximum_steps=10,
        checkpoint_interval=1000,
        step_controller=_controller(limit=1.0e-9, minimum_step=0.01, maximum_step=0.2),
    )
    result, committed = _committed_steps(tmp_path / "exhausted", plan)

    assert not result.successful
    assert result.state.status == "failed"
    assert result.failure is not None
    assert result.failure.category == "step-capacity-exhausted"
    assert result.failure.error_code == "PRODUCTION_STEP_CAPACITY_EXHAUSTED"
    assert int(result.state.step_index) == 10
    assert float(result.state.time) < 1.0
    steps = [float(metrics.accepted_step_size) for metrics in committed]
    assert min(steps) == 0.01
    assert steps[-1] == 0.01


def test_adaptive_plan_refuses_out_of_bounds_initial_step() -> None:
    with pytest.raises(ValueError, match="step-controller bounds"):
        phx.solver.ProductionRunPlan(
            METHOD,
            phx.solver.RobustRetryPolicy(maximum_retries=0),
            step_size=0.5,
            end_time=1.0,
            maximum_steps=10,
            checkpoint_interval=10,
            step_controller=_controller(maximum_step=0.2),
        )


def test_adaptive_capacity_below_maximum_step_bound_is_refused(tmp_path: Path) -> None:
    plan = phx.solver.ProductionRunPlan(
        METHOD,
        phx.solver.RobustRetryPolicy(maximum_retries=0),
        step_size=0.05,
        end_time=1.0,
        maximum_steps=4,
        checkpoint_interval=10,
        step_controller=_controller(maximum_step=0.2),
    )
    prepared = phx.solver.PreparedProductionRun(
        _manifest(), plan, _store(tmp_path / "capacity")
    )
    with pytest.raises(ValueError, match="cannot reach end_time"):
        prepared.initial_state(jnp.zeros((1,)))


def test_adaptive_run_refuses_caller_controller_state(tmp_path: Path) -> None:
    plan = phx.solver.ProductionRunPlan(
        METHOD,
        phx.solver.RobustRetryPolicy(maximum_retries=0),
        step_size=0.05,
        end_time=1.0,
        maximum_steps=100,
        checkpoint_interval=10,
        step_controller=_controller(),
    )
    prepared = phx.solver.PreparedProductionRun(
        _manifest(), plan, _store(tmp_path / "owned")
    )
    with pytest.raises(ValueError, match="owns the production controller state"):
        prepared.initial_state(jnp.zeros((1,)), controller_state=jnp.asarray(0.1))


@pytest.mark.parametrize(
    ("limits", "options"),
    [
        pytest.param((0.1, -1.0), {}, id="nonpositive-limit"),
        pytest.param(((0.1,),), {}, id="two-dimensional-limits"),
        pytest.param((np.inf,), {}, id="nonfinite-limit"),
        pytest.param((0.1,), {"minimum_step": 0.5}, id="minimum-above-maximum"),
        pytest.param((0.1,), {"growth": 0.5}, id="growth-below-one"),
        pytest.param((0.1,), {"shrink": 0.0}, id="nonpositive-shrink"),
        pytest.param((0.1,), {"safety": 1.5}, id="safety-above-one"),
    ],
)
def test_increment_limited_controller_refuses_invalid_configuration(
    limits: Any, options: dict[str, float]
) -> None:
    settings = {"minimum_step": 1.0e-3, "maximum_step": 0.2, **options}
    with pytest.raises(ValueError):
        phx.solver.IncrementLimitedStepController(
            _increments,
            np.asarray(limits, dtype=np.float64),
            controller_id="increment:state-change",
            **settings,
        )
