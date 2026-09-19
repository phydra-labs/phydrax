#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from phydrax._iteration import (
    bind_iteration_scope,
    HostIterationEvent,
    IterationCapabilities,
    IterationCoordinates,
    IterationPhase,
    IterationPlan,
    IterationRecord,
)
from phydrax._training import (
    training_iteration_record,
    TrainingIterationKind,
    TrainingProgress,
)
from phydrax.service import WandbTrainingSink


class _FakeRun:
    def __init__(self, *, fail: bool = False):
        self.defined: list[tuple[str, dict[str, Any]]] = []
        self.logged: list[dict[str, object]] = []
        self.fail = bool(fail)

    def define_metric(self, name: str, /, **kwargs: Any) -> None:
        self.defined.append((name, kwargs))

    def log(self, data: dict[str, object], /, **kwargs: Any) -> None:
        assert kwargs == {}
        if self.fail:
            raise RuntimeError("provider credential secret")
        self.logged.append(dict(data))


def _scope():
    return bind_iteration_scope(
        IterationPlan(granularity="step"),
        IterationCapabilities(
            ("terminal", "step"),
            host_stop=True,
            host_streaming=True,
            checkpointable=True,
        ),
        "wandb-test-training",
    )


def _event(
    kind: TrainingIterationKind,
    /,
    *,
    update_step: int,
    metrics: dict[str, object] | None = None,
    sequence: int = 0,
) -> HostIterationEvent:
    progress = TrainingProgress(epoch=update_step, update_step=update_step)
    return HostIterationEvent(
        "private-session-id",
        f"event-{sequence}",
        sequence,
        _scope(),
        training_iteration_record(kind, progress, metrics=metrics),
        process_index=0,
        process_count=2,
    )


def test_wandb_sink_defines_axes_and_maps_typed_training_metrics() -> None:
    run = _FakeRun()
    sink = WandbTrainingSink(run)

    sink.emit(
        _event(
            TrainingIterationKind.UPDATE,
            update_step=2,
            metrics={
                "loss": 1.25,
                "optimizer/step_size": 0.5,
                "train/terms/000_data/value": 0.75,
            },
        )
    )

    assert run.defined == [
        ("phydrax/update_step", {"hidden": True}),
        ("train/*", {"step_metric": "phydrax/update_step"}),
        ("eval/*", {"step_metric": "phydrax/update_step"}),
        ("optimizer/*", {"step_metric": "phydrax/update_step"}),
        ("phydrax/lifecycle/*", {"step_metric": "phydrax/update_step"}),
    ]
    assert run.logged == [
        {
            "phydrax/algorithm_id": "wandb-test-training",
            "phydrax/epoch": 2,
            "phydrax/event_id": "event-0",
            "phydrax/event_kind": "update",
            "phydrax/event_sequence": 0,
            "phydrax/microstep": 0,
            "phydrax/process_count": 2,
            "phydrax/process_index": 0,
            "phydrax/scope_id": _scope().scope_id,
            "phydrax/status": 0,
            "phydrax/update_step": 2,
            "train/loss": 1.25,
            "optimizer/step_size": 0.5,
            "train/terms/000_data/value": 0.75,
        }
    ]
    assert "phydrax/session_id" not in run.logged[0]


def test_wandb_sink_filters_updates_but_keeps_validation_and_terminal() -> None:
    run = _FakeRun()
    sink = WandbTrainingSink(run, update_every=2)

    sink.emit(
        _event(
            TrainingIterationKind.UPDATE,
            update_step=1,
            metrics={"loss": 3.0},
        )
    )
    sink.emit(
        _event(
            TrainingIterationKind.VALIDATION,
            update_step=1,
            metrics={"loss": 2.0},
            sequence=1,
        )
    )
    sink.emit(
        _event(
            TrainingIterationKind.RUN_TERMINAL,
            update_step=1,
            metrics={"completed_steps": 1},
            sequence=2,
        )
    )

    assert [row["phydrax/event_kind"] for row in run.logged] == [
        "validation",
        "run_terminal",
    ]
    assert run.logged[0]["eval/loss"] == 2.0
    assert run.logged[1]["phydrax/lifecycle/run_terminal/completed_steps"] == 1


def test_wandb_sink_bounds_redacts_and_normalizes_metrics() -> None:
    run = _FakeRun()
    sink = WandbTrainingSink(run, maximum_metrics=5)

    sink.emit(
        _event(
            TrainingIterationKind.UPDATE,
            update_step=1,
            metrics={
                "loss": 1.0,
                "token": 2.0,
                "nonfinite": float("nan"),
                "complex": 1.0 + 2.0j,
                "retained": 3,
                "overflow": 4.0,
            },
        )
    )

    payload = run.logged[0]
    assert payload["train/loss"] == 1.0
    assert payload["train/nonfinite"] is None
    assert payload["train/retained"] == 3
    assert "train/token" not in payload
    assert "train/complex" not in payload
    assert "train/overflow" not in payload
    assert payload["phydrax/nonfinite_metric_count"] == 1
    assert payload["phydrax/omitted_metric_count"] == 3


def test_wandb_sink_disables_after_provider_failure(phydrax_events) -> None:
    run = _FakeRun(fail=True)
    sink = WandbTrainingSink(run)
    event = _event(
        TrainingIterationKind.UPDATE,
        update_step=1,
        metrics={"loss": 1.0},
    )

    sink.emit(event)
    sink.emit(event)

    assert sink.disabled
    assert sink.failure_count == 1
    assert sink.last_failure_type == "RuntimeError"
    records = phydrax_events.records("tracking.wandb.failed")
    assert len(records) == 1
    assert records[0]["fields"] == {
        "exception_type": "RuntimeError",
        "operation": "log",
        "sink_id": "wandb-training",
    }
    assert "provider credential secret" not in str(records[0])


def test_wandb_sink_rejects_nontraining_records() -> None:
    run = _FakeRun()
    sink = WandbTrainingSink(run)
    record = IterationRecord(
        IterationCoordinates(IterationPhase.COMMIT, 1, committed=True),
        0,
        jnp.asarray(1.0),
    )
    event = HostIterationEvent("session", "event", 0, _scope(), record)

    with pytest.raises(TypeError, match="typed training events"):
        sink.emit(event)


def test_wandb_sink_validates_constructor_contract() -> None:
    with pytest.raises(TypeError, match="define_metric"):
        WandbTrainingSink(object())
    with pytest.raises(ValueError, match="update_every"):
        WandbTrainingSink(_FakeRun(), update_every=0)
    with pytest.raises(ValueError, match="maximum_metrics"):
        WandbTrainingSink(_FakeRun(), maximum_metrics=0)
    with pytest.raises(ValueError, match="sink_id"):
        WandbTrainingSink(_FakeRun(), sink_id=" ")
