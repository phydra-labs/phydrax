import jax
import jax.numpy as jnp

from phydrax.solver._hybrid_event import (
    hybrid_event_jvp,
    hybrid_event_vjp,
    HybridEventPlan,
    HybridGuardPlan,
    HybridReplayPolicy,
)
from phydrax.solver._hybrid_schedule import (
    execute_hybrid_schedule,
    HybridSchedulePlan,
    prepare_hybrid_schedule,
    replay_hybrid_schedule,
    ScheduledHybridGuard,
)


def _event():
    guard = HybridGuardPlan(
        lambda time, state, args: state[0] - 0.5,
        guard_id="scale-guard",
    )
    return HybridEventPlan(
        guard,
        lambda time, state, args: 2.0 * state,
        lambda time, state, args: jnp.ones_like(state),
        lambda time, state, args: 3.0 * jnp.ones_like(state),
        dense_diagnostics=True,
        plan_id="scale-event",
    )


def test_prepared_schedule_emits_replayable_log_jacobian_tape():
    schedule = HybridSchedulePlan(
        (ScheduledHybridGuard(_event().guard_plan, event=_event()),),
        maximum_events=2,
    )
    policy = HybridReplayPolicy(2, simultaneous_tolerance=1.0e-9)
    prepared = prepare_hybrid_schedule(schedule, jnp.asarray([0.0]), replay_policy=policy)
    result = execute_hybrid_schedule(
        prepared,
        lambda time, args: jnp.asarray([time]),
        jnp.asarray([[0.0, 1.0]]),
    )
    replay = replay_hybrid_schedule(
        prepared,
        result.tape,
        result.tape.states_before[0],
    )
    assert result.event_count == 1
    assert replay.valid
    assert jnp.allclose(replay.state, result.tape.states_after[0])
    assert result.tape.log_jacobian_valid[0]
    assert jnp.isclose(result.tape.total_log_abs_determinant, jnp.log(3.0))


def test_matrix_free_hybrid_jvp_vjp_are_transposes():
    plan = _event()
    state = jnp.asarray([0.5])
    tangent = jnp.asarray([0.3])
    cotangent = jnp.asarray([-0.7])
    forward = hybrid_event_jvp(plan, 0.5, state, tangent)
    _, reverse, _, evidence = hybrid_event_vjp(plan, 0.5, state, cotangent)
    assert forward.successful & evidence.successful
    assert jnp.allclose(jnp.vdot(cotangent, forward.action), jnp.vdot(reverse, tangent))
    assert (
        jax.jit(lambda value: hybrid_event_jvp(plan, 0.5, state, value).action)(
            tangent
        ).shape
        == state.shape
    )


def _timed_event(root, reset_shift, plan_id, *, priority=0):
    guard = HybridGuardPlan(
        lambda time, state, args: state[0] - root,
        priority=priority,
        guard_id=f"{plan_id}:guard",
    )
    return HybridEventPlan(
        guard,
        lambda time, state, args: state + reset_shift,
        lambda time, state, args: jnp.ones_like(state),
        lambda time, state, args: jnp.ones_like(state),
        plan_id=plan_id,
    )


def test_schedule_selects_earliest_root_before_priority_and_applies_its_reset():
    late_event = _timed_event(0.75, 20.0, "late-high-priority", priority=100)
    late = ScheduledHybridGuard(late_event.guard_plan, event=late_event)
    early_event = _timed_event(0.25, 10.0, "early-low-priority", priority=-100)
    early = ScheduledHybridGuard(early_event.guard_plan, event=early_event)
    schedule = HybridSchedulePlan((late, early), maximum_events=2)
    prepared = prepare_hybrid_schedule(schedule, jnp.asarray([0.0]))

    result = execute_hybrid_schedule(
        prepared,
        lambda time, args: jnp.asarray([time]),
        jnp.asarray([[0.0, 1.0]]),
    )

    assert result.event_count == 1
    assert result.event_indices[0] == 1
    assert jnp.isclose(result.event_times[0], 0.25)
    assert jnp.allclose(result.event_states_after[0], jnp.asarray([10.25]))


def test_schedule_uses_priority_to_resolve_simultaneous_roots():
    low_event = _timed_event(0.5, 10.0, "simultaneous-low-priority", priority=0)
    low = ScheduledHybridGuard(low_event.guard_plan, event=low_event)
    high_event = _timed_event(0.5, 20.0, "simultaneous-high-priority", priority=1)
    high = ScheduledHybridGuard(high_event.guard_plan, event=high_event)
    schedule = HybridSchedulePlan((low, high), maximum_events=2)
    prepared = prepare_hybrid_schedule(schedule, jnp.asarray([0.0]))

    result = execute_hybrid_schedule(
        prepared,
        lambda time, args: jnp.asarray([time]),
        jnp.asarray([[0.0, 1.0]]),
    )

    assert result.event_count == 1
    assert result.event_indices[0] == 1
    assert jnp.allclose(result.event_states_after[0], jnp.asarray([20.5]))
