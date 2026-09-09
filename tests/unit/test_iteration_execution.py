#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from phydrax._iteration import (
    bind_iteration_scope,
    CallableIterationHostControl,
    CallableIterationSink,
    CallableIterationStopRule,
    finalize_iteration,
    initialize_iteration,
    IterationCapabilities,
    IterationChildPlan,
    IterationCoordinates,
    IterationCountObserver,
    IterationMomentObserver,
    IterationPhase,
    IterationPlan,
    IterationRecord,
    IterationSession,
    IterationTraceObserver,
    update_iteration,
)


def _record(
    ordinal,
    value,
    /,
    *,
    phase=IterationPhase.COMMIT,
    active=True,
    committed=True,
    terminal=False,
):
    return IterationRecord(
        IterationCoordinates(
            phase,
            ordinal,
            attempt=ordinal,
            accepted=ordinal if committed else ordinal - 1,
            rejected=0 if committed else 1,
            active=active,
            committed=committed,
            terminal=terminal,
        ),
        0,
        {"value": jnp.asarray(value)},
    )


def test_trace_is_bounded_without_losing_terminal_evidence() -> None:
    observer = IterationTraceObserver(2, cadence=2)
    plan = IterationPlan(observers=(observer,))
    capabilities = IterationCapabilities(("terminal", "step"), device_stop=True)
    scope = bind_iteration_scope(plan, capabilities, "unit-driver")
    state = initialize_iteration(plan, _record(0, -1.0, phase=IterationPhase.START))
    for ordinal in range(1, 7):
        state = update_iteration(plan, state, _record(ordinal, float(ordinal)))
    evidence = finalize_iteration(
        plan,
        scope,
        capabilities,
        state,
        _record(6, 99.0, phase=IterationPhase.TERMINAL, terminal=True),
    )
    trace = evidence.observer_outputs[0]
    assert int(trace.stored_count) == 2
    assert int(trace.seen_count) == 3
    assert int(trace.dropped_count) == 1
    assert jnp.array_equal(trace.records.metrics["value"], jnp.asarray([2.0, 4.0]))
    assert float(trace.initial.metrics["value"]) == -1.0
    assert float(trace.terminal.metrics["value"]) == 99.0


def test_stop_rule_only_changes_execution_at_active_safe_boundaries() -> None:
    stop_rule = CallableIterationStopRule(
        lambda initial: jnp.asarray(0, dtype=jnp.int32),
        lambda count, record: (count + 1, count + 1 >= 2),
        "two-commits",
    )
    plan = IterationPlan(observers=(IterationCountObserver(),), stop_rule=stop_rule)
    capabilities = IterationCapabilities(("terminal", "step"), device_stop=True)
    scope = bind_iteration_scope(plan, capabilities, "unit-driver")

    @jax.jit
    def run(value):
        state = initialize_iteration(plan, _record(0, value, phase=IterationPhase.START))

        def body(ordinal, carry):
            active = ~carry.stop_requested
            record = _record(ordinal + 1, value + ordinal + 1, active=active)
            return update_iteration(plan, carry, record)

        state = jax.lax.fori_loop(0, 5, body, state)
        terminal = _record(
            state.last.coordinates.ordinal,
            state.last.metrics["value"],
            phase=IterationPhase.TERMINAL,
            terminal=True,
        )
        return finalize_iteration(plan, scope, capabilities, state, terminal)

    evidence = run(jnp.asarray(3.0))
    counts = evidence.observer_outputs[0]
    assert bool(evidence.stop_requested)
    assert int(counts.committed) == 2
    assert int(evidence.terminal.coordinates.ordinal) == 2
    assert float(evidence.terminal.metrics["value"]) == 5.0


def test_observation_is_batched_and_does_not_change_gradients() -> None:
    observer = IterationMomentObserver(
        lambda record: record.metrics["value"], "value-moments"
    )
    plan = IterationPlan(observers=(observer,))
    capabilities = IterationCapabilities(("terminal", "step"))
    scope = bind_iteration_scope(plan, capabilities, "mapped-driver")

    def objective(values):
        initial = _record(0, values, phase=IterationPhase.START)
        state = initialize_iteration(plan, initial)
        state = update_iteration(plan, state, _record(1, values * 2.0))
        evidence = finalize_iteration(
            plan,
            scope,
            capabilities,
            state,
            _record(1, values * 2.0, phase=IterationPhase.TERMINAL, terminal=True),
        )
        return jnp.sum(values**2), evidence

    values = jnp.asarray([1.0, 2.0, 3.0])
    (result, evidence), gradient = jax.value_and_grad(objective, has_aux=True)(values)
    moments = evidence.observer_outputs[0]
    assert float(result) == 14.0
    assert jnp.array_equal(gradient, 2.0 * values)
    assert jnp.array_equal(moments.mean, 2.0 * values)
    assert evidence.terminal.metrics["value"].shape == (3,)


def test_capabilities_reject_unsupported_granularity_control_and_children() -> None:
    terminal_only = IterationCapabilities.terminal_only()
    with pytest.raises(ValueError, match="granularity"):
        bind_iteration_scope(IterationPlan(), terminal_only, "direct")

    stop_plan = IterationPlan(
        granularity="terminal",
        stop_rule=CallableIterationStopRule(
            lambda initial: jnp.asarray(0),
            lambda state, record: (state, False),
            "never",
        ),
    )
    with pytest.raises(ValueError, match="device-side stopping"):
        bind_iteration_scope(stop_plan, terminal_only, "direct")

    child = IterationChildPlan("linear", IterationPlan(granularity="terminal"))
    with pytest.raises(ValueError, match="child roles"):
        bind_iteration_scope(
            IterationPlan(granularity="terminal", children=(child,)),
            terminal_only,
            "direct",
        )


def test_host_sinks_and_control_have_separate_ordered_semantics() -> None:
    received = []
    sink = CallableIterationSink(
        lambda event: received.append((event.sequence, event.event_id)), "collector"
    )
    control = CallableIterationHostControl(
        lambda event: event.sequence == 1, "stop-after-two"
    )
    session = IterationSession("unit-session", sinks=(sink,), control=control)
    capabilities = IterationCapabilities(("terminal", "step"), host_stop=True)
    scope = bind_iteration_scope(IterationPlan(), capabilities, "host-driver")
    assert not session.emit(scope, _record(0, 0.0, phase=IterationPhase.START))
    assert session.emit(scope, _record(1, 1.0))
    state = session.snapshot()
    resumed = IterationSession(
        "unit-session", sinks=(sink,), control=control, state=state
    )
    resumed.emit(scope, _record(2, 2.0))
    assert [sequence for sequence, _ in received] == [0, 1, 2]
    assert len({event_id for _, event_id in received}) == 3
    assert resumed.stop_requested


def test_host_sink_return_value_cannot_control_execution() -> None:
    sink = CallableIterationSink(lambda event: True, "invalid-sink")
    session = IterationSession("unit-session", sinks=(sink,))
    scope = bind_iteration_scope(
        IterationPlan(),
        IterationCapabilities(("terminal", "step")),
        "host-driver",
    )
    with pytest.raises(TypeError, match="must not return"):
        session.emit(scope, _record(1, 1.0))
