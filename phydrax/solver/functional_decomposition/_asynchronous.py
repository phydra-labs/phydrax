#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from ..._frozendict import frozendict
from ..._iteration import (
    bind_iteration_scope,
    IterationCapabilities,
    IterationPhase,
    IterationPlan,
    IterationSession,
    IterationSessionState,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...domain import LocalFieldFamily
from .._functional_solver import FunctionalSolver
from ._prepare import PreparedFunctionalDecomposition
from ._schwarz import capture_schwarz_trace_state, SchwarzTraceState
from ._solve import (
    _decomposition_iteration_record,
    _result_family,
    _solve_local,
    FunctionalDecompositionIterationMetrics,
    Optimizer,
)


class AsynchronousSchwarzPlan(StrictModule, NonTrainableState):
    """Deterministic bounded-staleness local update schedule."""

    updates: int = eqx.field(static=True)
    inner_iterations: int = eqx.field(static=True)
    maximum_staleness: int = eqx.field(static=True)
    patch_order: tuple[str, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        updates: int,
        inner_iterations: int,
        /,
        *,
        maximum_staleness: int = 1,
        patch_order: Sequence[str] | None = None,
    ):
        updates_ = int(updates)
        inner_ = int(inner_iterations)
        staleness = int(maximum_staleness)
        if updates_ < 0 or inner_ <= 0 or staleness < 0:
            raise ValueError("Asynchronous work and staleness bounds are invalid.")
        order = (
            None if patch_order is None else tuple(str(value) for value in patch_order)
        )
        if order is not None and (not order or len(set(order)) != len(order)):
            raise ValueError("patch_order must contain distinct patch IDs.")
        self.updates = updates_
        self.inner_iterations = inner_
        self.maximum_staleness = staleness
        self.patch_order = order


class AsynchronousSchwarzState(StrictModule):
    functions: Any
    optimizer_states: tuple[Any | None, ...]
    local_steps: tuple[int, ...] = eqx.field(static=True)
    patch_revisions: tuple[int, ...] = eqx.field(static=True)
    trace_history: tuple[SchwarzTraceState, ...]
    completed_updates: int = eqx.field(static=True)
    maximum_observed_staleness: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        functions: Any,
        optimizer_states: tuple[Any | None, ...],
        local_steps: tuple[int, ...],
        patch_revisions: tuple[int, ...],
        trace_history: tuple[SchwarzTraceState, ...],
        completed_updates: int,
        maximum_observed_staleness: int,
    ):
        self.functions = frozendict(functions)
        self.optimizer_states = tuple(optimizer_states)
        self.local_steps = tuple(int(value) for value in local_steps)
        self.patch_revisions = tuple(int(value) for value in patch_revisions)
        self.trace_history = tuple(trace_history)
        self.completed_updates = int(completed_updates)
        self.maximum_observed_staleness = int(maximum_observed_staleness)


class AsynchronousSchwarzResult(StrictModule):
    solver: FunctionalSolver
    family: LocalFieldFamily
    state: AsynchronousSchwarzState
    iteration_session_state: IterationSessionState | None

    def __init__(
        self,
        solver: FunctionalSolver,
        family: LocalFieldFamily,
        state: AsynchronousSchwarzState,
        iteration_session_state: IterationSessionState | None = None,
        /,
    ):
        self.solver = solver
        self.family = family
        self.state = state
        self.iteration_session_state = iteration_session_state


def solve_asynchronous_schwarz(
    prepared: PreparedFunctionalDecomposition,
    plan: AsynchronousSchwarzPlan,
    optimizer: Optimizer,
    /,
    *,
    state: AsynchronousSchwarzState | None = None,
    seed: int = 0,
    jit: bool = True,
    session: IterationSession | None = None,
) -> AsynchronousSchwarzResult:
    """Execute deterministic local updates against explicitly stale trace snapshots."""
    if prepared.problem.assembly != "broken" or not prepared.problem.cover.pairings:
        raise ValueError("Asynchronous Schwarz requires a paired broken-field problem.")
    if not isinstance(plan, AsynchronousSchwarzPlan):
        raise TypeError("plan must be an AsynchronousSchwarzPlan.")
    if session is not None and not isinstance(session, IterationSession):
        raise TypeError("session must be IterationSession or None.")
    patch_ids = prepared.problem.cover.patch_ids
    order = patch_ids if plan.patch_order is None else plan.patch_order
    if set(order) != set(patch_ids):
        raise ValueError("Asynchronous patch_order must contain every cover patch.")
    patch_indices = {patch_id: index for index, patch_id in enumerate(patch_ids)}
    patch_count = len(patch_ids)
    if state is None:
        functions = frozendict(prepared.solver.functions)
        optimizer_states: list[Any | None] = [None] * patch_count
        local_steps = [0] * patch_count
        revisions = [0] * patch_count
        trace_history = [
            capture_schwarz_trace_state(
                prepared.problem,
                functions,
                prepared.trace_batches,
                sweep=0,
            )
        ]
        completed = 0
        maximum_observed = 0
    else:
        functions = frozendict(state.functions)
        optimizer_states = list(state.optimizer_states)
        local_steps = list(state.local_steps)
        revisions = list(state.patch_revisions)
        trace_history = list(state.trace_history)
        completed = state.completed_updates
        maximum_observed = state.maximum_observed_staleness
    iteration_scope = None
    stopped = False
    if session is not None:
        capabilities = IterationCapabilities(
            ("terminal", "segment"),
            host_stop=True,
            host_streaming=True,
        )
        iteration_scope = bind_iteration_scope(
            IterationPlan(granularity="segment"),
            capabilities,
            "asynchronous-schwarz",
        )
        stopped = session.emit(
            iteration_scope,
            _decomposition_iteration_record(
                IterationPhase.START,
                FunctionalDecompositionIterationMetrics(
                    completed,
                    tuple(local_steps),
                    trace_history[-1].maximum_defect,
                    jnp.asarray(jnp.nan),
                ),
            ),
        )

    for update in range(completed, plan.updates):
        if stopped:
            break
        patch_id = order[update % len(order)]
        patch_index = patch_indices[patch_id]
        requested_staleness = update % (plan.maximum_staleness + 1)
        available_staleness = min(requested_staleness, len(trace_history) - 1)
        trace_state = trace_history[-1 - available_staleness]
        maximum_observed = max(maximum_observed, available_staleness)
        functions, optimizer_state, local_step = _solve_local(
            prepared,
            functions,
            optimizer_states[patch_index],
            patch_index,
            inner_iterations=plan.inner_iterations,
            optim=optimizer,
            seed=seed,
            start_step=local_steps[patch_index],
            jit=jit,
            trace_state=trace_state,
        )
        optimizer_states[patch_index] = optimizer_state
        local_steps[patch_index] = local_step
        revisions[patch_index] += 1
        trace_history.append(
            capture_schwarz_trace_state(
                prepared.problem,
                functions,
                prepared.trace_batches,
                sweep=update + 1,
                previous=trace_history[-1],
            )
        )
        trace_history = trace_history[-(plan.maximum_staleness + 1) :]
        completed = update + 1
        if session is not None:
            assert iteration_scope is not None
            stopped = session.emit(
                iteration_scope,
                _decomposition_iteration_record(
                    IterationPhase.COMMIT,
                    FunctionalDecompositionIterationMetrics(
                        completed,
                        tuple(local_steps),
                        trace_history[-1].maximum_defect,
                        jnp.asarray(jnp.nan),
                    ),
                ),
            )

    solver = eqx.tree_at(
        lambda value: value.functions,
        prepared.solver,
        functions,
    )
    family, _, _ = _result_family(prepared, solver)
    result_state = AsynchronousSchwarzState(
        functions=functions,
        optimizer_states=tuple(optimizer_states),
        local_steps=tuple(local_steps),
        patch_revisions=tuple(revisions),
        trace_history=tuple(trace_history),
        completed_updates=completed,
        maximum_observed_staleness=maximum_observed,
    )
    if session is not None:
        assert iteration_scope is not None
        session.emit(
            iteration_scope,
            _decomposition_iteration_record(
                IterationPhase.TERMINAL,
                FunctionalDecompositionIterationMetrics(
                    completed,
                    tuple(local_steps),
                    trace_history[-1].maximum_defect,
                    jnp.asarray(jnp.nan),
                ),
                terminal=True,
                status=0 if completed == plan.updates else 1,
            ),
        )
    return AsynchronousSchwarzResult(
        solver,
        family,
        result_state,
        None if session is None else session.snapshot(),
    )


__all__ = [
    "AsynchronousSchwarzPlan",
    "AsynchronousSchwarzResult",
    "AsynchronousSchwarzState",
    "solve_asynchronous_schwarz",
]
