#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._frozendict import frozendict
from ..._iteration import (
    bind_iteration_scope,
    IterationCapabilities,
    IterationCoordinates,
    IterationPhase,
    IterationPlan,
    IterationRecord,
    IterationScope,
    IterationSession,
    IterationSessionState,
)
from ..._strict import StrictModule
from ..._trainable import partition_trainable
from ...conditions import SubdomainValueJump
from ...domain import (
    broken_field,
    BrokenField,
    DomainFunction,
    LocalFieldFamily,
    partition_of_unity_family,
)
from ...nn.parameters import ParameterSubspace
from ...terms import ResidualPenalty
from .._functional_solver import FunctionalSolver
from .._functional_training import FunctionalTrainingPlan
from .._functional_update_kernel import FunctionalUpdateKernel, FunctionalUpdateState
from ._prepare import PreparedFunctionalDecomposition
from ._problem import GlobalScope, PairScope, PatchScope
from ._schwarz import (
    capture_schwarz_trace_state,
    DiscreteTracePenalty,
    SchwarzTraceState,
)
from ._strategy import (
    BlockDecompositionTraining,
    JointDecompositionTraining,
    SchwarzDecompositionTraining,
)


class FunctionalDecompositionEvidence(StrictModule):
    """Independent final objective and interface evidence."""

    training_loss: Array
    evaluation_loss: Array
    maximum_pair_loss: Array
    patch_losses: tuple[tuple[str, Array], ...]
    pair_losses: tuple[tuple[str, Array], ...]
    cover_verified: bool = eqx.field(static=True)
    certified: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        training_loss: Array,
        evaluation_loss: Array,
        maximum_pair_loss: Array,
        patch_losses: tuple[tuple[str, Array], ...],
        pair_losses: tuple[tuple[str, Array], ...],
        cover_verified: bool,
        certified: bool,
    ):
        self.training_loss = jnp.asarray(training_loss).reshape(())
        self.evaluation_loss = jnp.asarray(evaluation_loss).reshape(())
        self.maximum_pair_loss = jnp.asarray(maximum_pair_loss).reshape(())
        self.patch_losses = tuple(
            (str(name), jnp.asarray(value).reshape(())) for name, value in patch_losses
        )
        self.pair_losses = tuple(
            (str(name), jnp.asarray(value).reshape(())) for name, value in pair_losses
        )
        self.cover_verified = bool(cover_verified)
        self.certified = bool(certified)


class FunctionalDecompositionState(StrictModule):
    """Accepted decomposition state at a completed joint run or local sweep."""

    functions: frozendict[str, DomainFunction]
    optimizer_states: tuple[Any | None, ...]
    trace_state: SchwarzTraceState | None
    local_steps: tuple[int, ...] = eqx.field(static=True)
    completed_sweeps: int = eqx.field(static=True)
    strategy: str = eqx.field(static=True)

    def __init__(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        optimizer_states: tuple[Any | None, ...] = (),
        local_steps: tuple[int, ...] = (),
        completed_sweeps: int = 0,
        trace_state: SchwarzTraceState | None = None,
        strategy: str,
    ):
        states = tuple(optimizer_states)
        steps = tuple(int(value) for value in local_steps)
        if states and len(states) != len(steps):
            raise ValueError("optimizer_states and local_steps must have equal length.")
        if any(value < 0 for value in steps):
            raise ValueError("local_steps must be non-negative.")
        self.functions = frozendict(functions)
        self.optimizer_states = states
        self.local_steps = steps
        self.trace_state = trace_state
        self.completed_sweeps = int(completed_sweeps)
        self.strategy = str(strategy)


class FunctionalDecompositionIterationMetrics(StrictModule):
    completed_sweeps: Array
    local_steps: Array
    maximum_interface_defect: Array
    training_loss: Array

    def __init__(
        self,
        completed_sweeps,
        local_steps,
        maximum_interface_defect,
        training_loss,
        /,
    ):
        self.completed_sweeps = jnp.asarray(completed_sweeps, dtype=jnp.int32)
        self.local_steps = jnp.asarray(local_steps, dtype=jnp.int32)
        self.maximum_interface_defect = jnp.asarray(maximum_interface_defect)
        self.training_loss = jnp.asarray(training_loss)


def _decomposition_iteration_record(
    phase,
    metrics: FunctionalDecompositionIterationMetrics,
    /,
    *,
    terminal=False,
    status=0,
) -> IterationRecord:
    return IterationRecord(
        IterationCoordinates(
            phase,
            metrics.completed_sweeps,
            attempt=metrics.completed_sweeps,
            accepted=metrics.completed_sweeps,
            active=True,
            committed=int(phase) == int(IterationPhase.COMMIT),
            terminal=terminal,
        ),
        status,
        metrics,
    )


class FunctionalDecompositionResult(StrictModule):
    """Trained native local fields, assembly, state, and certification evidence."""

    solver: FunctionalSolver
    family: LocalFieldFamily
    global_field: DomainFunction | None
    broken: BrokenField | None
    state: FunctionalDecompositionState
    evidence: FunctionalDecompositionEvidence
    status: str = eqx.field(static=True)
    iteration_session_state: IterationSessionState | None

    def __init__(
        self,
        *,
        solver: FunctionalSolver,
        family: LocalFieldFamily,
        global_field: DomainFunction | None,
        broken: BrokenField | None,
        state: FunctionalDecompositionState,
        evidence: FunctionalDecompositionEvidence,
        status: str,
        iteration_session_state: IterationSessionState | None = None,
    ):
        self.solver = solver
        self.family = family
        self.global_field = global_field
        self.broken = broken
        self.state = state
        self.evidence = evidence
        self.status = str(status)
        self.iteration_session_state = iteration_session_state

    @property
    def completed(self) -> bool:
        return self.status == "completed"


Optimizer = optax.GradientTransformation | optax.GradientTransformationExtraArgs


def _term_values(
    prepared: PreparedFunctionalDecomposition,
    functions: Mapping[str, DomainFunction],
    /,
    *,
    key: Key[Array, ""],
    evaluation: bool,
) -> tuple[Array, tuple[tuple[str, Array], ...], tuple[tuple[str, Array], ...]]:
    scoped_terms = (
        prepared.problem.evaluation_terms if evaluation else prepared.problem.terms
    )
    total = jnp.asarray(0.0)
    patch_totals = {
        patch_id: jnp.asarray(0.0) for patch_id in prepared.problem.cover.patch_ids
    }
    pair_totals = {
        pairing_id: jnp.asarray(0.0) for pairing_id in prepared.problem.cover.pairing_ids
    }
    for index, scoped in enumerate(scoped_terms):
        value = scoped.term.evaluate(
            functions,
            key=jr.fold_in(key, index),
        ).value
        total = total + value
        if isinstance(scoped.scope, PatchScope):
            patch_totals[scoped.scope.patch_id] = (
                patch_totals[scoped.scope.patch_id] + value
            )
        elif isinstance(scoped.scope, PairScope):
            pair_totals[scoped.scope.pairing_id] = (
                pair_totals[scoped.scope.pairing_id] + value
            )
        elif not isinstance(scoped.scope, GlobalScope):
            raise TypeError("Unknown decomposition term scope.")
    return total, tuple(patch_totals.items()), tuple(pair_totals.items())


def _evidence(
    prepared: PreparedFunctionalDecomposition,
    functions: Mapping[str, DomainFunction],
    /,
    *,
    key: Key[Array, ""],
) -> FunctionalDecompositionEvidence:
    training, training_patches, training_pairs = _term_values(
        prepared,
        functions,
        key=jr.fold_in(key, 100),
        evaluation=False,
    )
    evaluation, evaluation_patches, evaluation_pairs = _term_values(
        prepared,
        functions,
        key=jr.fold_in(key, 200),
        evaluation=True,
    )
    evaluation_patch_ids = {
        scoped.scope.patch_id
        for scoped in prepared.problem.evaluation_terms
        if isinstance(scoped.scope, PatchScope)
    }
    evaluation_pair_ids = {
        scoped.scope.pairing_id
        for scoped in prepared.problem.evaluation_terms
        if isinstance(scoped.scope, PairScope)
    }
    training_patch_map = dict(training_patches)
    evaluation_patch_map = dict(evaluation_patches)
    patch_losses = tuple(
        (
            patch_id,
            evaluation_patch_map[patch_id]
            if patch_id in evaluation_patch_ids
            else training_patch_map[patch_id],
        )
        for patch_id in prepared.problem.cover.patch_ids
    )
    training_pair_map = dict(training_pairs)
    evaluation_pair_map = dict(evaluation_pairs)
    pair_losses = tuple(
        (
            pairing_id,
            evaluation_pair_map[pairing_id]
            if pairing_id in evaluation_pair_ids
            else training_pair_map[pairing_id],
        )
        for pairing_id in prepared.problem.cover.pairing_ids
    )
    pair_values = tuple(value for _, value in pair_losses)
    maximum_pair = jnp.max(jnp.stack(pair_values)) if pair_values else jnp.asarray(0.0)
    tolerance = prepared.plan.certification_tolerance
    finite = bool(
        jax.device_get(
            jnp.isfinite(training) & jnp.isfinite(evaluation) & jnp.isfinite(maximum_pair)
        )
    )
    pair_coverage = set(prepared.problem.cover.pairing_ids).issubset(evaluation_pair_ids)
    certified = (
        tolerance is not None
        and bool(prepared.problem.evaluation_terms)
        and pair_coverage
        and finite
        and prepared.cover_evidence.verified
        and float(evaluation) <= tolerance
        and float(maximum_pair) <= tolerance
    )
    return FunctionalDecompositionEvidence(
        training_loss=training,
        evaluation_loss=evaluation,
        maximum_pair_loss=maximum_pair,
        patch_losses=patch_losses,
        pair_losses=pair_losses,
        cover_verified=prepared.cover_evidence.verified,
        certified=certified,
    )


def _result_family(
    prepared: PreparedFunctionalDecomposition,
    solver: FunctionalSolver,
    /,
) -> tuple[LocalFieldFamily, DomainFunction | None, BrokenField | None]:
    problem = prepared.problem
    if problem.assembly == "partition-of-unity":
        global_field = solver.functions[problem.field_name]
        family = partition_of_unity_family(global_field)
        return family, global_field, None
    fields = {
        patch_id: solver.functions[problem.family.field_name(patch_id)]
        for patch_id in problem.cover.patch_ids
    }
    family = LocalFieldFamily(problem.family.field_id, problem.cover, fields)
    return family, None, broken_field(family)


def _terms_for_patch(
    prepared: PreparedFunctionalDecomposition,
    patch_id: str,
    /,
    *,
    trace_state: SchwarzTraceState | None = None,
) -> tuple[Any, ...]:
    terms = []
    cover = prepared.problem.cover
    for scoped in prepared.problem.terms:
        if isinstance(scoped.scope, GlobalScope):
            terms.append(scoped.term)
        elif isinstance(scoped.scope, PatchScope):
            if scoped.scope.patch_id == patch_id:
                terms.append(scoped.term)
        elif isinstance(scoped.scope, PairScope):
            pairing = cover.pairing(scoped.scope.pairing_id)
            if patch_id not in (pairing.left_patch_id, pairing.right_patch_id):
                continue
            term = scoped.term
            if (
                trace_state is not None
                and isinstance(term, ResidualPenalty)
                and isinstance(term.condition, SubdomainValueJump)
            ):
                exchange = trace_state.exchange(pairing.pairing_id)
                side = "left" if patch_id == pairing.left_patch_id else "right"
                jump = term.condition.target(exchange.points).data
                target = (
                    exchange.left_target - jump
                    if side == "left"
                    else exchange.right_target + jump
                )
                terms.append(
                    DiscreteTracePenalty(
                        prepared.problem.family.field_name(patch_id),
                        pairing,
                        exchange.points,
                        target,
                        side=side,
                        scale=float(term.scale),
                        label=term.label,
                    )
                )
            else:
                terms.append(term)
    if not terms:
        raise ValueError(f"Patch {patch_id!r} has no incident training terms.")
    return tuple(terms)


def _trainable_paths(
    functions: Mapping[str, DomainFunction],
    /,
) -> tuple[str, ...]:
    trainable, _ = partition_trainable(functions)
    return tuple(
        jax.tree_util.keystr(path)
        for path, leaf in jax.tree_util.tree_flatten_with_path(trainable)[0]
        if eqx.is_inexact_array(leaf)
    )


def _patch_parameter_subspace(
    prepared: PreparedFunctionalDecomposition,
    functions: Mapping[str, DomainFunction],
    patch_index: int,
    /,
) -> ParameterSubspace:
    problem = prepared.problem
    if problem.assembly == "broken":
        patch_id = problem.cover.patch_ids[patch_index]
        token = f"['{problem.family.field_name(patch_id)}']"
    else:
        token = f"['{problem.field_name}'].func.fields[{patch_index}].func.source"
    selected = tuple(path for path in _trainable_paths(functions) if token in path)
    if not selected:
        raise ValueError(
            f"Patch {problem.cover.patch_ids[patch_index]!r} has no trainable "
            "inexact-array parameters."
        )
    return ParameterSubspace.from_leaf_paths(functions, selected)


def _merge_patch_candidate(
    prepared: PreparedFunctionalDecomposition,
    current: Mapping[str, DomainFunction],
    candidate: Mapping[str, DomainFunction],
    patch_index: int,
    /,
) -> frozendict[str, DomainFunction]:
    current_subspace = _patch_parameter_subspace(prepared, current, patch_index)
    candidate_subspace = ParameterSubspace.from_leaf_paths(
        candidate,
        current_subspace.leaf_paths,
    )
    return frozendict(current_subspace.reconstruct(candidate_subspace.initial))


def _solve_local(
    prepared: PreparedFunctionalDecomposition,
    functions: Mapping[str, DomainFunction],
    optimizer_state: Any | None,
    patch_index: int,
    *,
    inner_iterations: int,
    optim: Optimizer,
    seed: int,
    start_step: int,
    jit: bool,
    trace_state: SchwarzTraceState | None = None,
) -> tuple[frozendict[str, DomainFunction], Any, int]:
    patch_id = prepared.problem.cover.patch_ids[patch_index]
    local_solver = FunctionalSolver(
        functions=functions,
        terms=_terms_for_patch(prepared, patch_id, trace_state=trace_state),
        collocation_key=jr.key(seed + patch_index),
    )
    subspace = _patch_parameter_subspace(prepared, functions, patch_index)
    kernel = FunctionalUpdateKernel.from_subspace(
        local_solver,
        optim,
        subspace,
        jit=jit,
    )
    state = (
        kernel.initialize(functions)
        if optimizer_state is None
        else FunctionalUpdateState(functions, optimizer_state, start_step)
    )
    for _ in range(inner_iterations):
        update_key = jr.fold_in(jr.key(seed + patch_index), state.step)
        state, evidence = kernel.advance(state, key=update_key)
        if not bool(jax.device_get(evidence.accepted)):
            raise FloatingPointError(
                f"Local update for patch {patch_id!r} was rejected as non-finite."
            )
    return state.functions, state.optimizer_state, state.step


def _solve_blocks(
    prepared: PreparedFunctionalDecomposition,
    strategy: BlockDecompositionTraining | SchwarzDecompositionTraining,
    optim: Optimizer,
    /,
    *,
    state: FunctionalDecompositionState | None = None,
    max_sweeps: int | None = None,
    seed: int,
    jit: bool,
    session: IterationSession | None = None,
    iteration_scope: IterationScope | None = None,
) -> tuple[FunctionalSolver, FunctionalDecompositionState]:
    patch_count = len(prepared.problem.cover.patches)
    if state is None:
        functions = frozendict(prepared.solver.functions)
        optimizer_states: list[Any | None] = [None for _ in range(patch_count)]
        local_steps = [0 for _ in range(patch_count)]
        completed = 0
    else:
        expected = (
            "schwarz" if isinstance(strategy, SchwarzDecompositionTraining) else "block"
        )
        if state.strategy != expected:
            raise ValueError("Resume state strategy does not match the prepared plan.")
        if len(state.optimizer_states) != patch_count:
            raise ValueError("Resume state optimizer count does not match the cover.")
        if state.completed_sweeps > strategy.sweeps:
            raise ValueError("Resume state exceeds the planned number of sweeps.")
        functions = frozendict(state.functions)
        optimizer_states = list(state.optimizer_states)
        local_steps = list(state.local_steps)
        completed = state.completed_sweeps
    if isinstance(strategy, SchwarzDecompositionTraining):
        if state is None:
            trace_state = capture_schwarz_trace_state(
                prepared.problem,
                functions,
                prepared.trace_batches,
                sweep=completed,
            )
        else:
            trace_state = state.trace_state
            if trace_state is None:
                raise ValueError("Schwarz resume state has no trace exchange state.")
    else:
        trace_state = None
    if isinstance(strategy, BlockDecompositionTraining):
        requested = (
            set(prepared.problem.cover.patch_ids)
            if strategy.active_patch_ids is None
            else set(strategy.active_patch_ids)
        )
        active_ids = requested - set(strategy.fixed_patch_ids)
    else:
        active_ids = set(prepared.problem.cover.patch_ids)
    active_indices = tuple(
        index
        for index, patch_id in enumerate(prepared.problem.cover.patch_ids)
        if patch_id in active_ids
    )
    if max_sweeps is None:
        stop_sweep = strategy.sweeps
    else:
        count = int(max_sweeps)
        if count < 0:
            raise ValueError("max_sweeps must be non-negative or None.")
        stop_sweep = min(strategy.sweeps, completed + count)
    for sweep_index in range(completed, stop_sweep):
        if session is not None and session.stop_requested:
            break
        snapshot = frozendict(functions)
        if strategy.sweep == "jacobi":
            candidates = []
            for patch_index in active_indices:
                candidate, optimizer_state, local_step = _solve_local(
                    prepared,
                    snapshot,
                    optimizer_states[patch_index],
                    patch_index,
                    inner_iterations=strategy.inner_iterations,
                    optim=optim,
                    seed=seed,
                    start_step=local_steps[patch_index],
                    jit=jit,
                    trace_state=trace_state,
                )
                candidates.append((patch_index, candidate))
                optimizer_states[patch_index] = optimizer_state
                local_steps[patch_index] = local_step
            merged = snapshot
            for patch_index, candidate in candidates:
                merged = _merge_patch_candidate(
                    prepared,
                    merged,
                    candidate,
                    patch_index,
                )
            functions = merged
            if isinstance(strategy, SchwarzDecompositionTraining):
                trace_state = capture_schwarz_trace_state(
                    prepared.problem,
                    functions,
                    prepared.trace_batches,
                    sweep=sweep_index + 1,
                    relaxation=strategy.relaxation,
                    previous=trace_state,
                )
        elif strategy.sweep == "gauss-seidel":
            current = snapshot
            for patch_index in active_indices:
                current, optimizer_state, local_step = _solve_local(
                    prepared,
                    current,
                    optimizer_states[patch_index],
                    patch_index,
                    inner_iterations=strategy.inner_iterations,
                    optim=optim,
                    seed=seed,
                    start_step=local_steps[patch_index],
                    jit=jit,
                    trace_state=trace_state,
                )
                optimizer_states[patch_index] = optimizer_state
                local_steps[patch_index] = local_step
                if isinstance(strategy, SchwarzDecompositionTraining):
                    trace_state = capture_schwarz_trace_state(
                        prepared.problem,
                        current,
                        prepared.trace_batches,
                        sweep=sweep_index + 1,
                        relaxation=strategy.relaxation,
                        previous=trace_state,
                    )
            functions = current
        else:
            current = snapshot
            patch_indices = {
                patch_id: index
                for index, patch_id in enumerate(prepared.problem.cover.patch_ids)
            }
            for color in prepared.routing.colors:
                color_snapshot = current
                candidates = []
                indices = tuple(
                    patch_indices[patch_id]
                    for patch_id in color
                    if patch_id in active_ids
                )
                for patch_index in indices:
                    candidate, optimizer_state, local_step = _solve_local(
                        prepared,
                        color_snapshot,
                        optimizer_states[patch_index],
                        patch_index,
                        inner_iterations=strategy.inner_iterations,
                        optim=optim,
                        seed=seed,
                        start_step=local_steps[patch_index],
                        jit=jit,
                        trace_state=trace_state,
                    )
                    candidates.append(candidate)
                    optimizer_states[patch_index] = optimizer_state
                    local_steps[patch_index] = local_step
                for patch_index, candidate in zip(indices, candidates, strict=True):
                    current = _merge_patch_candidate(
                        prepared,
                        current,
                        candidate,
                        patch_index,
                    )
                if isinstance(strategy, SchwarzDecompositionTraining):
                    trace_state = capture_schwarz_trace_state(
                        prepared.problem,
                        current,
                        prepared.trace_batches,
                        sweep=sweep_index + 1,
                        relaxation=strategy.relaxation,
                        previous=trace_state,
                    )
            functions = current
        completed = sweep_index + 1
        if session is not None:
            assert iteration_scope is not None
            defect = (
                jnp.asarray(jnp.nan)
                if trace_state is None
                else trace_state.maximum_defect
            )
            session.emit(
                iteration_scope,
                _decomposition_iteration_record(
                    IterationPhase.COMMIT,
                    FunctionalDecompositionIterationMetrics(
                        completed,
                        tuple(local_steps),
                        defect,
                        jnp.asarray(jnp.nan, dtype=defect.dtype),
                    ),
                ),
            )
        if (
            isinstance(strategy, SchwarzDecompositionTraining)
            and strategy.interface_tolerance is not None
            and trace_state is not None
            and float(trace_state.maximum_defect) <= strategy.interface_tolerance
        ):
            break

    solver = eqx.tree_at(
        lambda value: value.functions,
        prepared.solver,
        functions,
    )
    name = "schwarz" if isinstance(strategy, SchwarzDecompositionTraining) else "block"
    state = FunctionalDecompositionState(
        functions,
        optimizer_states=tuple(optimizer_states),
        local_steps=tuple(local_steps),
        completed_sweeps=completed,
        trace_state=trace_state,
        strategy=name,
    )
    return solver, state


def solve_functional_decomposition(
    prepared: PreparedFunctionalDecomposition,
    optim: Optimizer,
    /,
    *,
    seed: int = 0,
    jit: bool = True,
    keep_best: bool = True,
    log_every: int = 1,
    training: FunctionalTrainingPlan | None = None,
    state: FunctionalDecompositionState | None = None,
    max_sweeps: int | None = None,
    key: Key[Array, ""] = DOC_KEY0,
    session: IterationSession | None = None,
) -> FunctionalDecompositionResult:
    """Execute one prepared native functional domain-decomposition problem."""
    if not isinstance(prepared, PreparedFunctionalDecomposition):
        raise TypeError("prepared must be a PreparedFunctionalDecomposition.")
    if session is not None and not isinstance(session, IterationSession):
        raise TypeError("session must be IterationSession or None.")
    strategy = prepared.plan.training
    iteration_scope = None
    if session is not None:
        if (
            isinstance(strategy, JointDecompositionTraining)
            and session.control_id is not None
        ):
            raise ValueError(
                "Joint decomposition does not expose host-safe stopping boundaries."
            )
        capabilities = IterationCapabilities(
            ("terminal", "segment"),
            host_stop=not isinstance(strategy, JointDecompositionTraining),
            host_streaming=True,
            checkpointable=True,
        )
        iteration_scope = bind_iteration_scope(
            IterationPlan(granularity="segment"),
            capabilities,
            f"functional-decomposition:{prepared.plan.plan_id}",
        )
        initial_steps = () if state is None else state.local_steps
        initial_sweeps = 0 if state is None else state.completed_sweeps
        session.emit(
            iteration_scope,
            _decomposition_iteration_record(
                IterationPhase.START,
                FunctionalDecompositionIterationMetrics(
                    initial_sweeps,
                    initial_steps,
                    jnp.asarray(jnp.nan),
                    jnp.asarray(jnp.nan),
                ),
            ),
        )
    if max_sweeps is not None and isinstance(strategy, JointDecompositionTraining):
        raise ValueError("max_sweeps is only valid for block and Schwarz strategies.")
    if state is not None and isinstance(strategy, JointDecompositionTraining):
        raise ValueError(
            "Joint decomposition resumes through FunctionalTrainingCheckpoint, "
            "not FunctionalDecompositionState."
        )
    if isinstance(strategy, JointDecompositionTraining):
        solver = prepared.solver.solve(
            num_iter=strategy.num_iterations,
            optim=optim,
            seed=seed,
            jit=jit,
            keep_best=keep_best,
            log_every=log_every,
            training=training,
        )
        state = FunctionalDecompositionState(
            solver.functions,
            completed_sweeps=0,
            strategy="joint",
        )
    elif isinstance(strategy, (BlockDecompositionTraining, SchwarzDecompositionTraining)):
        if training is not None:
            raise ValueError(
                "Block and Schwarz execution own their local training states; "
                "an external FunctionalTrainingPlan is unsupported."
            )
        solver, state = _solve_blocks(
            prepared,
            strategy,
            optim,
            state=state,
            max_sweeps=max_sweeps,
            seed=seed,
            jit=jit,
            session=session,
            iteration_scope=iteration_scope,
        )
    else:
        raise TypeError("Unknown decomposition training strategy.")

    family, global_field, broken = _result_family(prepared, solver)
    evidence = _evidence(
        prepared,
        solver.functions,
        key=key,
    )
    finite = bool(jax.device_get(jnp.isfinite(evidence.training_loss)))
    status: Literal["completed", "nonfinite", "stopped"] = (
        "stopped"
        if session is not None
        and session.stop_requested
        and isinstance(
            strategy, (BlockDecompositionTraining, SchwarzDecompositionTraining)
        )
        and state.completed_sweeps < strategy.sweeps
        else "completed"
        if finite
        else "nonfinite"
    )
    if session is not None:
        assert iteration_scope is not None
        defect = (
            jnp.asarray(jnp.nan)
            if state.trace_state is None
            else state.trace_state.maximum_defect
        )
        session.emit(
            iteration_scope,
            _decomposition_iteration_record(
                IterationPhase.TERMINAL,
                FunctionalDecompositionIterationMetrics(
                    state.completed_sweeps,
                    state.local_steps,
                    defect,
                    evidence.training_loss,
                ),
                terminal=True,
                status=0 if status == "completed" else 1,
            ),
        )
    return FunctionalDecompositionResult(
        solver=solver,
        family=family,
        global_field=global_field,
        broken=broken,
        state=state,
        evidence=evidence,
        status=status,
        iteration_session_state=None if session is None else session.snapshot(),
    )


__all__ = [
    "FunctionalDecompositionEvidence",
    "FunctionalDecompositionResult",
    "FunctionalDecompositionIterationMetrics",
    "FunctionalDecompositionState",
    "solve_functional_decomposition",
]
