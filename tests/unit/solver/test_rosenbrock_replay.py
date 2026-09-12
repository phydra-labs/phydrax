#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _scaled_decay_problem(scale: float, rate: float = 2.0):
    initial = jnp.asarray([1.0, scale])

    def drift(time, state, runtime_rate):
        del time
        return -runtime_rate * state

    return phx.solver.DifferentialProblem(
        drift,
        initial,
        t0=0.0,
        t1=1.0,
        args=jnp.asarray(rate),
        problem_id=f"scaled-decay:{scale}",
    )


def _controller():
    return phx.solver.RosenbrockAdaptivePolicy(
        relative_tolerance=1.0e-4,
        absolute_tolerance=1.0e-10,
        initial_step=0.1,
        maximum_step=0.25,
        maximum_accepted_steps=512,
        maximum_attempts=1024,
    )


def _grid(identifier: str = "replay-grid"):
    return phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.5, 1.0]),
        time_id=identifier,
    )


def test_adaptive_rosenbrock_wrms_is_covariant_to_component_scale():
    unit = phx.solver.solve_rosenbrock(
        _scaled_decay_problem(1.0),
        _grid("wrms-unit"),
        adaptive=_controller(),
    )
    scaled = phx.solver.solve_rosenbrock(
        _scaled_decay_problem(1.0e6),
        _grid("wrms-scaled"),
        adaptive=_controller(),
    )

    assert unit.successful
    assert scaled.successful
    assert int(unit.stats["accepted_steps"]) == int(scaled.stats["accepted_steps"])
    assert jnp.allclose(
        unit.states,
        scaled.states / jnp.asarray([1.0, 1.0e6]),
        rtol=2e-6,
        atol=2e-8,
    )
    unit_adequacy = unit.stats["replay_adequacy"]
    scaled_adequacy = scaled.stats["replay_adequacy"]
    assert unit_adequacy.maximum_error_ratio <= 1.0
    assert scaled_adequacy.maximum_error_ratio <= 1.0


def test_scheduled_rosenbrock_replays_differentiates_and_refreshes_explicitly():
    problem = _scaled_decay_problem(10.0)
    prepared = phx.solver.prepare_rosenbrock(
        problem,
        _grid(),
        adaptive=_controller(),
    )
    source = phx.solver.solve_rosenbrock(prepared)
    scheduled = phx.solver.schedule_rosenbrock(
        prepared,
        source,
        replay=phx.solver.FixedStepReplayPolicy("block", block_size=8),
    )
    replayed = phx.solver.solve_scheduled_rosenbrock(scheduled)

    assert source.successful
    assert replayed.successful
    assert source.temporal_mesh is not None
    assert source.temporal_mesh.adaptive
    assert replayed.temporal_mesh is not None
    assert not replayed.temporal_mesh.adaptive
    assert scheduled.temporal_mesh.interval_count == int(source.stats["accepted_steps"])
    assert jnp.allclose(replayed.states, source.states, rtol=2e-6, atol=2e-8)
    assert replayed.temporal_evidence.differentiation.checkpointing == "chunked-replay"

    def terminal(rate):
        return phx.solver.solve_scheduled_rosenbrock(
            scheduled,
            args=rate,
        ).states[-1, 0]

    rate = jnp.asarray(2.0)
    value, gradient = jax.jit(jax.value_and_grad(terminal))(rate)
    tangent = jax.jvp(terminal, (rate,), (jnp.asarray(1.0),))[1]
    _, pullback = jax.vjp(terminal, rate)
    transpose = pullback(jnp.asarray(1.0))[0]
    assert jnp.allclose(value, jnp.exp(-2.0), rtol=2e-3)
    assert jnp.allclose(gradient, -jnp.exp(-2.0), rtol=5e-3)
    assert jnp.allclose(tangent, transpose, rtol=2e-5, atol=2e-7)

    changed_rate = jnp.asarray(20.0)
    stale = phx.solver.solve_scheduled_rosenbrock(scheduled, args=changed_rate)
    assert not stale.successful
    assert stale.stats["replay_adequacy"].refresh_required
    assert int(stale.stats["replay_adequacy"].status) == int(
        phx.solver.RosenbrockReplayStatus.ERROR_RATIO_EXCEEDED
    )
    assert jnp.any(~stale.valid)
    assert jnp.any(~jnp.isfinite(stale.states[1:]))

    refreshed_source = phx.solver.solve_rosenbrock(prepared, args=changed_rate)
    refreshed = phx.solver.refresh_rosenbrock_schedule(
        scheduled,
        refreshed_source,
        reference_args=changed_rate,
    )
    refreshed_result = phx.solver.solve_scheduled_rosenbrock(refreshed)
    assert refreshed_result.successful
    assert refreshed.numeric_version == scheduled.numeric_version + 1
    assert refreshed.record_point_id != scheduled.record_point_id


def test_scheduled_rosenbrock_rejects_incompatible_sources_and_inputs():
    prepared = phx.solver.prepare_rosenbrock(
        _scaled_decay_problem(1.0),
        _grid("compatible"),
        adaptive=_controller(),
    )
    source = phx.solver.solve_rosenbrock(prepared)
    scheduled = phx.solver.schedule_rosenbrock(prepared, source)
    incompatible = phx.solver.prepare_rosenbrock(
        _scaled_decay_problem(1.0),
        _grid("incompatible"),
        adaptive=_controller(),
    )

    with pytest.raises(ValueError, match="configuration do not match"):
        phx.solver.schedule_rosenbrock(incompatible, source)
    with pytest.raises(ValueError, match="intrinsic shape"):
        phx.solver.solve_scheduled_rosenbrock(
            scheduled,
            initial_state=jnp.ones((3,)),
        )
    with pytest.raises(ValueError, match="configuration overrides"):
        phx.solver.solve_rosenbrock(prepared, adaptive=_controller())
    step_count = int(source.stats["accepted_steps"])
    wrong_bytes = prepared.replay_state_bytes + 8
    replay_schedule = phx.solver.prepare_replay_schedule(
        step_count,
        wrong_bytes,
        phx.solver.AdaptiveReplayPreparationPolicy(
            wrong_bytes * 4,
            step_count * 4,
        ),
    )
    incompatible_replay = phx.solver.schedule_rosenbrock(
        prepared,
        source,
        replay=phx.solver.FixedStepReplayPolicy(
            "scheduled",
            schedule=replay_schedule,
        ),
    )
    with pytest.raises(ValueError, match="replay boundary bytes"):
        phx.solver.solve_scheduled_rosenbrock(incompatible_replay)
