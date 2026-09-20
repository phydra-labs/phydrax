#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Optional outbound Weights & Biases training observation."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .._iteration import HostIterationEvent
from .._privacy import REDACTED, SecretRedactor
from .._training import TrainingIterationKind, TrainingIterationMetrics
from ..logging import emit as emit_log


@runtime_checkable
class _WandbRun(Protocol):
    def define_metric(self, name: str, /, **kwargs: Any) -> Any: ...

    def log(self, data: dict[str, object], /, **kwargs: Any) -> Any: ...


class WandbTrainingSink:
    """Project typed training events into an application-owned W&B run.

    The caller owns W&B initialization, authentication, distributed policy, and
    finalization. Provider failures after construction disable this observational
    sink without changing scientific training status.
    """

    __slots__ = (
        "_disabled",
        "_failure_count",
        "_last_failure_type",
        "_maximum_metrics",
        "_redactor",
        "_run",
        "_update_every",
        "sink_id",
    )

    def __init__(
        self,
        run: _WandbRun,
        /,
        *,
        update_every: int = 1,
        maximum_metrics: int = 512,
        sink_id: str = "wandb-training",
    ):
        if not isinstance(run, _WandbRun):
            raise TypeError("run must provide callable define_metric() and log().")
        update_every_ = int(update_every)
        maximum_metrics_ = int(maximum_metrics)
        sink_id_ = str(sink_id).strip()
        if update_every_ <= 0:
            raise ValueError("update_every must be positive.")
        if maximum_metrics_ <= 0:
            raise ValueError("maximum_metrics must be positive.")
        if not sink_id_:
            raise ValueError("sink_id must be non-empty.")

        self._run = run
        self._update_every = update_every_
        self._maximum_metrics = maximum_metrics_
        self.sink_id = sink_id_
        self._redactor = SecretRedactor()
        self._disabled = False
        self._failure_count = 0
        self._last_failure_type: str | None = None

        run.define_metric("phydrax/update_step", hidden=True)
        for pattern in (
            "train/*",
            "eval/*",
            "optimizer/*",
            "phydrax/lifecycle/*",
        ):
            run.define_metric(pattern, step_metric="phydrax/update_step")

    @property
    def disabled(self) -> bool:
        """Whether a provider failure has disabled further delivery."""

        return self._disabled

    @property
    def failure_count(self) -> int:
        """Number of provider failures observed before disabling delivery."""

        return self._failure_count

    @property
    def last_failure_type(self) -> str | None:
        """Provider exception class name, without exception text."""

        return self._last_failure_type

    def emit(self, event: HostIterationEvent, /) -> None:
        if not isinstance(event.record.metrics, TrainingIterationMetrics):
            raise TypeError("WandbTrainingSink accepts typed training events only.")
        if self._disabled:
            return

        metrics = event.record.metrics
        kind = TrainingIterationKind(int(np.asarray(metrics.kind)))
        update_step = int(np.asarray(metrics.update_step))
        if kind is TrainingIterationKind.UPDATE and (
            update_step % self._update_every != 0
        ):
            return

        payload: dict[str, object] = {
            "phydrax/algorithm_id": event.scope.algorithm_id,
            "phydrax/epoch": int(np.asarray(metrics.epoch)),
            "phydrax/event_id": event.event_id,
            "phydrax/event_kind": kind.name.lower(),
            "phydrax/event_sequence": int(event.sequence),
            "phydrax/microstep": int(np.asarray(metrics.microstep)),
            "phydrax/process_count": int(event.process_count),
            "phydrax/process_index": int(event.process_index),
            "phydrax/scope_id": event.scope.scope_id,
            "phydrax/status": int(np.asarray(event.record.status)),
            "phydrax/update_step": update_step,
        }
        omitted = max(0, len(metrics.metric_names) - self._maximum_metrics)
        nonfinite = 0
        for name, value in zip(
            metrics.metric_names[: self._maximum_metrics],
            metrics.metric_values[: self._maximum_metrics],
            strict=True,
        ):
            if self._redactor.redact(0, field_name=name) == REDACTED:
                omitted += 1
                continue
            scalar = np.asarray(value)
            if scalar.shape != ():
                raise ValueError("Training iteration metrics must be scalar.")
            item = scalar.item()
            key = self._metric_key(kind, name)
            if isinstance(item, (bool, np.bool_)):
                payload[key] = bool(item)
            elif isinstance(item, Integral):
                payload[key] = int(item)
            elif isinstance(item, Real):
                numeric = float(item)
                if np.isfinite(numeric):
                    payload[key] = numeric
                else:
                    payload[key] = None
                    nonfinite += 1
            else:
                omitted += 1

        if omitted:
            payload["phydrax/omitted_metric_count"] = omitted
        if nonfinite:
            payload["phydrax/nonfinite_metric_count"] = nonfinite

        try:
            self._run.log(payload)
        except Exception as error:
            self._failure_count += 1
            self._last_failure_type = type(error).__name__
            self._disabled = True
            emit_log(
                "WARNING",
                "tracking.wandb.failed",
                "W&B training observation was disabled after a provider failure",
                exception_type=self._last_failure_type,
                operation="log",
                sink_id=self.sink_id,
            )

    @staticmethod
    def _metric_key(kind: TrainingIterationKind, name: str, /) -> str:
        name_ = str(name)
        if "/" in name_:
            return name_
        if kind is TrainingIterationKind.UPDATE:
            return f"train/{name_}"
        if kind is TrainingIterationKind.VALIDATION:
            return f"eval/{name_}"
        return f"phydrax/lifecycle/{kind.name.lower()}/{name_}"


__all__ = ["WandbTrainingSink"]
