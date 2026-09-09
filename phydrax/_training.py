#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ._iteration import (
    bind_iteration_scope,
    IterationCapabilities,
    IterationCoordinates,
    IterationPhase,
    IterationPlan,
    IterationRecord,
    IterationSession,
    IterationSessionState,
)
from ._strict import StrictModule
from ._trainable import NonTrainableState
from .logging import emit


SelectionMode = Literal["min", "max"]
EvaluationParametersFn = Callable[[Any, Any], Any]
TargetParameterSource = Literal["raw", "evaluation"]


@dataclass(frozen=True, slots=True)
class DelayedTargetPolicy:
    """Exact accepted-update lag with one static ring capacity."""

    delay: int

    def __post_init__(self):
        if isinstance(self.delay, bool) or int(self.delay) < 0:
            raise ValueError("Delayed target delay must be a nonnegative integer.")
        object.__setattr__(self, "delay", int(self.delay))

    @property
    def capacity(self) -> int:
        return self.delay + 1


@dataclass(frozen=True, slots=True)
class ExponentialMovingAverageTargetPolicy:
    """Stopped EMA target recurrence after accepted optimizer updates."""

    decay: float = 0.999
    start_step: int = 0
    update_every: int = 1
    source: TargetParameterSource = "raw"

    def __post_init__(self):
        if not 0.0 <= float(self.decay) < 1.0:
            raise ValueError("EMA decay must lie in [0, 1).")
        if int(self.start_step) < 0 or int(self.update_every) <= 0:
            raise ValueError("EMA start/update cadence is invalid.")
        if self.source not in ("raw", "evaluation"):
            raise ValueError("EMA target source must be raw or evaluation.")
        object.__setattr__(self, "decay", float(self.decay))
        object.__setattr__(self, "start_step", int(self.start_step))
        object.__setattr__(self, "update_every", int(self.update_every))


class TargetParameterState(eqx.Module):
    """Checkpointable stopped target tree and exact update cursor."""

    target: Any
    history: Any
    update_count: Array
    write_index: Array
    policy: DelayedTargetPolicy | ExponentialMovingAverageTargetPolicy = eqx.field(
        static=True
    )

    @classmethod
    def initialize(
        cls,
        parameters: Any,
        policy: DelayedTargetPolicy | ExponentialMovingAverageTargetPolicy,
        /,
    ) -> "TargetParameterState":
        if not isinstance(
            policy,
            (DelayedTargetPolicy, ExponentialMovingAverageTargetPolicy),
        ):
            raise TypeError("Unsupported target parameter policy.")
        stopped = jax.tree.map(jax.lax.stop_gradient, parameters)
        history = (
            jax.tree.map(
                lambda value: jnp.broadcast_to(
                    value,
                    (policy.capacity,) + value.shape,
                ),
                stopped,
            )
            if isinstance(policy, DelayedTargetPolicy)
            else None
        )
        return cls(
            stopped,
            history,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            policy,
        )

    def update(
        self,
        raw_parameters: Any,
        /,
        *,
        accepted: ArrayLike = True,
        evaluation_parameters: Any | None = None,
    ) -> "TargetParameterState":
        accepted_value = jnp.asarray(accepted, dtype=bool)
        if accepted_value.shape != ():
            raise ValueError("accepted must be scalar.")
        next_count = self.update_count + accepted_value.astype(jnp.int32)
        policy = self.policy
        if isinstance(policy, DelayedTargetPolicy):
            stopped = jax.tree.map(jax.lax.stop_gradient, raw_parameters)
            history = jax.tree.map(
                lambda old, value: old.at[self.write_index].set(
                    jnp.where(accepted_value, value, old[self.write_index])
                ),
                self.history,
                stopped,
            )
            next_index = jnp.where(
                accepted_value,
                (self.write_index + 1) % policy.capacity,
                self.write_index,
            )
            target_index = (next_index - policy.delay - 1) % policy.capacity
            target = jax.tree.map(
                lambda value: jax.lax.stop_gradient(value[target_index]),
                history,
            )
            return TargetParameterState(
                target,
                history,
                next_count,
                next_index,
                policy,
            )
        source = raw_parameters if policy.source == "raw" else evaluation_parameters
        if source is None:
            raise ValueError("EMA evaluation source requires evaluation_parameters.")
        apply = (
            accepted_value
            & (next_count >= policy.start_step)
            & ((next_count - policy.start_step) % policy.update_every == 0)
        )
        target = jax.tree.map(
            lambda old, value: jax.lax.stop_gradient(
                jnp.where(
                    apply,
                    policy.decay * old + (1.0 - policy.decay) * value,
                    old,
                )
            ),
            self.target,
            source,
        )
        return TargetParameterState(
            target,
            None,
            next_count,
            self.write_index,
            policy,
        )


def resolve_evaluation_parameters(
    transform: EvaluationParametersFn | None,
    optimizer_state: Any,
    training_parameters: Any,
    /,
) -> Any:
    """Return the optimizer-prescribed evaluation view of training parameters."""

    if transform is None:
        return training_parameters
    evaluation_parameters = transform(optimizer_state, training_parameters)
    expected_structure = jax.tree_util.tree_structure(training_parameters)
    actual_structure = jax.tree_util.tree_structure(evaluation_parameters)
    if actual_structure != expected_structure:
        raise ValueError(
            "evaluation_parameters must preserve the training-parameter PyTree structure."
        )
    expected_leaves = jax.tree_util.tree_leaves(training_parameters)
    actual_leaves = jax.tree_util.tree_leaves(evaluation_parameters)
    for expected, actual in zip(expected_leaves, actual_leaves, strict=True):
        if eqx.is_array(expected) != eqx.is_array(actual) or (
            eqx.is_array(expected)
            and (expected.shape != actual.shape or expected.dtype != actual.dtype)
        ):
            raise ValueError(
                "evaluation_parameters must preserve every training-parameter "
                "leaf shape and dtype."
            )
    return evaluation_parameters


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    """Serializable logical cursor and model-selection state for one training run."""

    epoch: int = 0
    next_batch_index: int = 0
    microstep: int = 0
    update_step: int = 0
    best_value: float | None = None
    best_step: int = 0
    stale_validations: int = 0
    stopped_early: bool = False
    iteration_session_id: str | None = None
    iteration_control_id: str | None = None
    iteration_session_cursor: int = 0
    iteration_stop_requested: bool = False

    def __post_init__(self):
        indices = (
            ("epoch", self.epoch),
            ("next_batch_index", self.next_batch_index),
            ("microstep", self.microstep),
            ("update_step", self.update_step),
            ("best_step", self.best_step),
            ("iteration_session_cursor", self.iteration_session_cursor),
        )
        for name, value in indices:
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if int(self.stale_validations) < 0:
            raise ValueError("stale_validations must be non-negative.")
        if self.iteration_session_id is None and (
            self.iteration_control_id is not None
            or self.iteration_session_cursor != 0
            or self.iteration_stop_requested
        ):
            raise ValueError("Iteration session progress requires a session identity.")

    @property
    def iteration_session_state(self) -> IterationSessionState | None:
        if self.iteration_session_id is None:
            return None
        return IterationSessionState(
            self.iteration_session_id,
            self.iteration_session_cursor,
            self.iteration_stop_requested,
        )


class TrainingIterationKind(IntEnum):
    """Closed lifecycle vocabulary shared by training frontends."""

    RUN_START = 0
    EPOCH_START = 1
    UPDATE = 2
    VALIDATION = 3
    CHECKPOINT = 4
    SKIP = 5
    FAILURE = 6
    RUN_TERMINAL = 7


_TRAINING_LOG_EVENTS = {
    TrainingIterationKind.RUN_START: "training.started",
    TrainingIterationKind.EPOCH_START: "training.epoch.started",
    TrainingIterationKind.UPDATE: "training.step.completed",
    TrainingIterationKind.VALIDATION: "training.validation.completed",
    TrainingIterationKind.CHECKPOINT: "training.checkpoint.committed",
    TrainingIterationKind.SKIP: "training.support.empty",
    TrainingIterationKind.FAILURE: "training.failed",
    TrainingIterationKind.RUN_TERMINAL: "training.completed",
}
_TRAINING_WARNING_EVENTS = frozenset(
    {TrainingIterationKind.SKIP, TrainingIterationKind.FAILURE}
)
_TRAINING_INFO_EVENTS = frozenset(
    {
        TrainingIterationKind.RUN_START,
        TrainingIterationKind.CHECKPOINT,
        TrainingIterationKind.RUN_TERMINAL,
    }
)


def _emit_training_event(
    kind: TrainingIterationKind,
    progress: TrainingProgress,
    metrics: Mapping[str, Any] | None,
    /,
) -> None:
    level = (
        "WARNING"
        if kind in _TRAINING_WARNING_EVENTS
        else "INFO"
        if kind in _TRAINING_INFO_EVENTS
        else "DEBUG"
    )
    metric_records = (
        ()
        if metrics is None
        else tuple(
            {
                "name": str(name),
                "value": float(jax.device_get(jnp.asarray(value).reshape(()))),
            }
            for name, value in metrics.items()
        )
    )
    emit(
        level,
        _TRAINING_LOG_EVENTS[kind],
        "Training lifecycle event",
        iteration_kind=kind.name.lower(),
        progress={
            "best_step": progress.best_step,
            "best_value": progress.best_value,
            "epoch": progress.epoch,
            "microstep": progress.microstep,
            "next_batch_index": progress.next_batch_index,
            "stale_validations": progress.stale_validations,
            "stopped_early": progress.stopped_early,
            "update_step": progress.update_step,
        },
        metrics=metric_records,
    )


class TrainingIterationMetrics(StrictModule, NonTrainableState):
    """Typed progress and named scalar metrics for one training event."""

    kind: Array
    epoch: Array
    next_batch_index: Array
    microstep: Array
    update_step: Array
    metric_names: tuple[str, ...] = eqx.field(static=True)
    metric_values: tuple[Array, ...]

    def __init__(
        self,
        kind: TrainingIterationKind,
        progress: TrainingProgress,
        metrics: Mapping[str, Any] | None,
        /,
    ):
        if not isinstance(kind, TrainingIterationKind):
            raise TypeError("kind must be TrainingIterationKind.")
        names = () if metrics is None else tuple(str(name) for name in metrics)
        values = (
            ()
            if metrics is None
            else tuple(jnp.asarray(value) for value in metrics.values())
        )
        if any(value.shape != () for value in values):
            raise ValueError("Training iteration metrics must be scalar.")
        self.kind = jnp.asarray(int(kind), dtype=jnp.int32)
        self.epoch = jnp.asarray(progress.epoch, dtype=jnp.int32)
        self.next_batch_index = jnp.asarray(progress.next_batch_index, dtype=jnp.int32)
        self.microstep = jnp.asarray(progress.microstep, dtype=jnp.int32)
        self.update_step = jnp.asarray(progress.update_step, dtype=jnp.int32)
        self.metric_names = names
        self.metric_values = values

    def metric(self, name: str, /) -> Array:
        for metric_name, metric_value in zip(
            self.metric_names, self.metric_values, strict=True
        ):
            if metric_name == name:
                return metric_value
        raise KeyError(name)


def training_iteration_record(
    kind: TrainingIterationKind,
    progress: TrainingProgress,
    /,
    *,
    metrics: Mapping[str, Any] | None = None,
) -> IterationRecord:
    """Build one host-deliverable typed training iteration record."""
    phases = {
        TrainingIterationKind.RUN_START: IterationPhase.START,
        TrainingIterationKind.EPOCH_START: IterationPhase.START,
        TrainingIterationKind.UPDATE: IterationPhase.COMMIT,
        TrainingIterationKind.VALIDATION: IterationPhase.VALIDATE,
        TrainingIterationKind.CHECKPOINT: IterationPhase.COMMIT,
        TrainingIterationKind.SKIP: IterationPhase.ATTEMPT,
        TrainingIterationKind.FAILURE: IterationPhase.ATTEMPT,
        TrainingIterationKind.RUN_TERMINAL: IterationPhase.TERMINAL,
    }
    committed = kind in (
        TrainingIterationKind.UPDATE,
        TrainingIterationKind.CHECKPOINT,
    )
    terminal = kind is TrainingIterationKind.RUN_TERMINAL
    return IterationRecord(
        IterationCoordinates(
            phases[kind],
            progress.update_step,
            invocation=progress.epoch,
            attempt=progress.microstep,
            accepted=progress.update_step,
            active=True,
            committed=committed,
            terminal=terminal,
        ),
        int(progress.stopped_early or progress.iteration_stop_requested),
        TrainingIterationMetrics(kind, progress, metrics),
    )


def training_key(
    master_key: Key[Array, ""],
    index: int,
    /,
    *,
    site: int = 0,
) -> Key[Array, ""]:
    """Derive a deterministic key from a persisted logical index and named site."""

    if int(index) < 0 or int(site) < 0:
        raise ValueError("Training key indices must be non-negative.")
    return jr.fold_in(jr.fold_in(master_key, int(index)), int(site))


def update_training_selection(
    progress: TrainingProgress,
    value: float,
    /,
    *,
    step: int,
    mode: SelectionMode = "min",
    min_delta: float = 0.0,
    patience: int | None = None,
) -> tuple[TrainingProgress, bool]:
    """Update strict best-state and early-stopping counters deterministically."""

    if mode not in ("min", "max"):
        raise ValueError("mode must be 'min' or 'max'.")
    if float(min_delta) < 0.0:
        raise ValueError("min_delta must be non-negative.")
    if patience is not None and int(patience) <= 0:
        raise ValueError("patience must be positive when provided.")
    current = float(value)
    best = progress.best_value
    improved = best is None
    if best is not None:
        improved = (
            current < best - float(min_delta)
            if mode == "min"
            else current > best + float(min_delta)
        )
    if improved:
        return (
            replace(
                progress,
                best_value=current,
                best_step=int(step),
                stale_validations=0,
            ),
            True,
        )
    stale = progress.stale_validations + 1
    return (
        replace(
            progress,
            stale_validations=stale,
            stopped_early=patience is not None and stale >= int(patience),
        ),
        False,
    )


class TrainingController:
    """Shared host lifecycle for PRNG, progress, selection, and typed events."""

    def __init__(
        self,
        *,
        total_steps: int,
        key: Key[Array, ""],
        algorithm_id: str,
        progress: TrainingProgress | None = None,
        session: IterationSession | None = None,
    ):
        if int(total_steps) < 0:
            raise ValueError("total_steps must be non-negative.")
        algorithm_id_ = str(algorithm_id)
        if not algorithm_id_:
            raise ValueError("algorithm_id must be non-empty.")
        if session is not None and not isinstance(session, IterationSession):
            raise TypeError("session must be IterationSession or None.")
        progress_ = TrainingProgress() if progress is None else progress
        if progress_.iteration_session_id is not None:
            if session is None:
                raise ValueError(
                    "Continuation progress requires its persisted iteration session."
                )
            if (
                session.session_id != progress_.iteration_session_id
                or session.control_id != progress_.iteration_control_id
            ):
                raise ValueError(
                    "Iteration session or control identity changed across continuation."
                )
            session_state = progress_.iteration_session_state
            assert session_state is not None
            session.restore(session_state)
        elif session is not None:
            if session.cursor != 0 or session.stop_requested:
                raise ValueError("A new training run requires a fresh iteration session.")
            progress_ = replace(
                progress_,
                iteration_session_id=session.session_id,
                iteration_control_id=session.control_id,
            )
        self.total_steps = int(total_steps)
        self.key = key
        self.progress = progress_
        self.session = session
        self.best_payload: Any | None = None
        self.stop_requested = bool(
            self.progress.stopped_early or self.progress.iteration_stop_requested
        )
        plan = IterationPlan(granularity="step")
        capabilities = IterationCapabilities(
            ("terminal", "step"),
            host_stop=True,
            host_streaming=True,
            checkpointable=True,
        )
        self.iteration_scope = bind_iteration_scope(plan, capabilities, algorithm_id_)

    def split_key(self) -> Key[Array, ""]:
        """Advance a sequential key stream for compatibility-sensitive loops."""

        self.key, step_key = jr.split(self.key)
        return step_key

    def key_for(self, index: int, /, *, site: int = 0) -> Key[Array, ""]:
        return training_key(self.key, index, site=site)

    def emit(
        self,
        kind: TrainingIterationKind,
        /,
        *,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        _emit_training_event(kind, self.progress, metrics)
        if self.session is None:
            return
        record = training_iteration_record(kind, self.progress, metrics=metrics)
        session_stop = self.session.emit(self.iteration_scope, record)
        self.stop_requested = self.stop_requested or session_stop
        self.progress = replace(
            self.progress,
            iteration_session_cursor=self.session.cursor,
            iteration_stop_requested=self.session.stop_requested,
        )

    def complete_update(self, step: int, /) -> None:
        self.progress = replace(self.progress, update_step=int(step))

    def select(
        self,
        value: float,
        payload: Any,
        /,
        *,
        step: int,
        mode: SelectionMode = "min",
        min_delta: float = 0.0,
        patience: int | None = None,
    ) -> bool:
        self.progress, improved = update_training_selection(
            self.progress,
            value,
            step=step,
            mode=mode,
            min_delta=min_delta,
            patience=patience,
        )
        if improved:
            self.best_payload = payload
        if self.progress.stopped_early:
            self.stop_requested = True
        return improved

    def selected(self, current: Any, /) -> Any:
        return current if self.best_payload is None else self.best_payload


class TrainingSignalGuard:
    """Convert process interrupts into a graceful training-loop stop request."""

    def __init__(self):
        self._previous_handlers: dict[int, Any] = {}
        self._signum: int | None = None
        self._reason: str | None = None
        self._installed = False

    def __enter__(self) -> "TrainingSignalGuard":
        if threading.current_thread() is not threading.main_thread():
            return self
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous_handlers[int(sig)] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
        self._installed = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._installed:
            return
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)

    @property
    def stop_requested(self) -> bool:
        return self._signum is not None or self._reason is not None

    @property
    def signal_name(self) -> str:
        if self._signum is None:
            return self._reason or "signal"
        return signal.Signals(self._signum).name

    def _handle_signal(self, signum: int, frame: Any) -> None:
        del frame
        if self._signum is None:
            self._signum = int(signum)

    def request_stop_from_exception(self, exc: BaseException, /) -> None:
        if self.stop_requested:
            return
        self._reason = (
            "SIGINT" if isinstance(exc, KeyboardInterrupt) else type(exc).__name__
        )


class TensorBoardLogger:
    """Small context-managed scalar writer shared by training frontends."""

    def __init__(self, log_dir: str | Path):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._event_cls = Event
        self._summary_cls = Summary
        self._writer = EventFileWriter(str(path))

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.flush()
        self._writer.close()

    def scalar(self, tag: str, value: Any, step: int) -> None:
        summary = self._summary_cls(
            value=[
                {
                    "tag": str(tag),
                    "simple_value": float(jnp.asarray(value, dtype=float).reshape(())),
                }
            ]
        )
        event = self._event_cls(
            wall_time=time.time(),
            step=int(step),
            summary=summary,
        )
        self._writer.add_event(event)

    def flush(self) -> None:
        self._writer.flush()


def tensorboard_every(
    *,
    tensorboard_log_dir: str | Path | None,
    tensorboard_every: int | None,
    log_every: int,
) -> int | None:
    if tensorboard_log_dir is None:
        return None
    if tensorboard_every is None:
        return log_every if log_every > 0 else 1
    every = int(tensorboard_every)
    if every <= 0:
        raise ValueError(
            "tensorboard_every must be positive when TensorBoard is enabled."
        )
    return every


def emit_training_signal_stop(
    backend: str,
    guard: TrainingSignalGuard,
    /,
    *,
    completed: int,
    total: int,
) -> None:
    emit(
        "WARNING",
        "training.stopped",
        "Training stopped by process signal",
        backend=backend,
        completed_steps=int(completed),
        signal_name=guard.signal_name,
        total_steps=int(total),
    )


__all__ = [
    "DelayedTargetPolicy",
    "ExponentialMovingAverageTargetPolicy",
    "EvaluationParametersFn",
    "SelectionMode",
    "TensorBoardLogger",
    "TrainingController",
    "TrainingIterationKind",
    "TrainingIterationMetrics",
    "TrainingProgress",
    "TargetParameterSource",
    "TargetParameterState",
    "TrainingSignalGuard",
    "resolve_evaluation_parameters",
    "emit_training_signal_stop",
    "tensorboard_every",
    "training_key",
    "update_training_selection",
    "training_iteration_record",
]
