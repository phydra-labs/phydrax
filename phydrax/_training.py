#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key


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

    def __post_init__(self):
        for name in (
            "epoch",
            "next_batch_index",
            "microstep",
            "update_step",
            "best_step",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if int(self.stale_validations) < 0:
            raise ValueError("stale_validations must be non-negative.")


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    """One ordered host-side lifecycle event emitted by a training controller."""

    name: str
    progress: TrainingProgress
    metrics: tuple[tuple[str, float], ...] = ()


class TrainingCallback(Protocol):
    def __call__(self, event: TrainingEvent, /) -> bool | None: ...


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
    """Shared host lifecycle for PRNG, progress, selection, callbacks, and stopping."""

    def __init__(
        self,
        *,
        total_steps: int,
        key: Key[Array, ""],
        progress: TrainingProgress | None = None,
        callbacks: Sequence[TrainingCallback] = (),
    ):
        if int(total_steps) < 0:
            raise ValueError("total_steps must be non-negative.")
        if any(not callable(callback) for callback in callbacks):
            raise TypeError("training callbacks must be callable.")
        self.total_steps = int(total_steps)
        self.key = key
        self.progress = TrainingProgress() if progress is None else progress
        self.callbacks = tuple(callbacks)
        self.best_payload: Any | None = None
        self.stop_requested = bool(self.progress.stopped_early)

    def split_key(self) -> Key[Array, ""]:
        """Advance a sequential key stream for compatibility-sensitive loops."""

        self.key, step_key = jr.split(self.key)
        return step_key

    def key_for(self, index: int, /, *, site: int = 0) -> Key[Array, ""]:
        return training_key(self.key, index, site=site)

    def emit(
        self,
        name: str,
        /,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        pairs = (
            ()
            if metrics is None
            else tuple(
                (str(key), float(jnp.asarray(value, dtype=float).reshape(())))
                for key, value in metrics.items()
            )
        )
        event = TrainingEvent(str(name), self.progress, pairs)
        for callback in self.callbacks:
            if callback(event):
                self.stop_requested = True

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


def log_training_signal_stop(
    backend: str,
    guard: TrainingSignalGuard,
    /,
    *,
    completed: int,
    total: int,
    file: Any,
) -> None:
    print(
        f"[phydrax][{backend}] received {guard.signal_name}; "
        f"exiting training loop after {completed}/{total} iteration(s).",
        file=file,
        flush=True,
    )


__all__ = [
    "DelayedTargetPolicy",
    "ExponentialMovingAverageTargetPolicy",
    "EvaluationParametersFn",
    "SelectionMode",
    "TensorBoardLogger",
    "TrainingCallback",
    "TrainingController",
    "TrainingEvent",
    "TrainingProgress",
    "TargetParameterSource",
    "TargetParameterState",
    "TrainingSignalGuard",
    "resolve_evaluation_parameters",
    "log_training_signal_stop",
    "tensorboard_every",
    "training_key",
    "update_training_selection",
]
