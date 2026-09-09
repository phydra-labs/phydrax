#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import asyncio
import io
import json
import os
import stat
from collections.abc import Iterator
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import pytest
from loguru import logger

from phydrax import logging as pxlogging
from phydrax._training import TrainingController, TrainingIterationKind


class _ExplosiveRepresentation:
    def __repr__(self) -> str:
        raise AssertionError("Logging attempted to represent an unsupported object.")


@pytest.fixture(autouse=True)
def _reset_phydrax_logging() -> Iterator[None]:
    pxlogging.disable()
    yield
    pxlogging.disable()


def _read_events(buffer: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in buffer.getvalue().splitlines()]


def test_logging_is_disabled_by_default() -> None:
    buffer = io.StringIO()
    handler_id = pxlogging.add_json_sink(buffer)
    try:
        pxlogging.emit("INFO", "runtime.operation.completed", "Completed")
    finally:
        pxlogging.remove_sink(handler_id)
    assert buffer.getvalue() == ""


def test_json_sink_emits_canonical_privacy_bounded_event() -> None:
    buffer = io.StringIO()
    handler_id = pxlogging.add_json_sink(buffer)
    pxlogging.enable()
    try:
        with pxlogging.context(run_id="run-1", authorization="Bearer secret"):
            pxlogging.emit(
                "INFO",
                "solver.training.completed",
                "Training\ncompleted",
                loss=float("nan"),
                payload=_ExplosiveRepresentation(),
                token="secret-value",
            )
    finally:
        pxlogging.remove_sink(handler_id)

    events = _read_events(buffer)
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "solver.training.completed"
    assert event["message"] == "Training completed"
    assert event["context"] == {"authorization": "<redacted>", "run_id": "run-1"}
    assert event["fields"] == {
        "loss": None,
        "payload": None,
        "token": "<redacted>",
    }
    assert event["nonfinite_fields"] == ["fields.loss"]
    assert event["omitted_fields"] == ["fields.payload"]
    assert set(event["source"]) == {"function", "line", "module"}
    assert "path" not in event["source"]
    assert buffer.getvalue().count("\n") == 1


def test_context_nesting_restores_previous_values() -> None:
    buffer = io.StringIO()
    handler_id = pxlogging.add_json_sink(buffer)
    pxlogging.enable()
    try:
        with pxlogging.context(run_id="outer"):
            pxlogging.emit("INFO", "runtime.scope.entered", "Outer")
            with pxlogging.context(run_id="inner", backend="cpu"):
                pxlogging.emit("INFO", "runtime.scope.entered", "Inner")
            pxlogging.emit("INFO", "runtime.scope.completed", "Outer again")
    finally:
        pxlogging.remove_sink(handler_id)

    contexts = [event["context"] for event in _read_events(buffer)]
    assert contexts == [
        {"run_id": "outer"},
        {"backend": "cpu", "run_id": "inner"},
        {"run_id": "outer"},
    ]


def test_async_tasks_keep_independent_context() -> None:
    buffer = io.StringIO()
    handler_id = pxlogging.add_json_sink(buffer)
    pxlogging.enable()

    async def emit_for(run_id: str) -> None:
        with pxlogging.context(run_id=run_id):
            await asyncio.sleep(0)
            pxlogging.emit("INFO", "runtime.task.completed", "Task completed")

    async def scenario() -> None:
        await asyncio.gather(emit_for("run-a"), emit_for("run-b"))

    try:
        asyncio.run(scenario())
    finally:
        pxlogging.remove_sink(handler_id)

    contexts = {event["context"]["run_id"] for event in _read_events(buffer)}
    assert contexts == {"run-a", "run-b"}


def test_sink_removal_does_not_remove_application_handler() -> None:
    phydrax_buffer = io.StringIO()
    application_buffer = io.StringIO()
    application_handler = logger.add(application_buffer, format="{message}")
    phydrax_handler = pxlogging.add_json_sink(phydrax_buffer)
    pxlogging.enable()
    try:
        pxlogging.remove_sink(phydrax_handler)
        logger.info("application remains")
    finally:
        logger.remove(application_handler)
    assert application_buffer.getvalue() == "application remains\n"
    assert phydrax_buffer.getvalue() == ""


def test_json_file_sink_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    handler_id = pxlogging.add_json_sink(path)
    pxlogging.enable()
    try:
        pxlogging.emit("INFO", "runtime.operation.completed", "Completed")
    finally:
        pxlogging.remove_sink(handler_id)

    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert _read_events(io.StringIO(path.read_text()))[0]["event"] == (
        "runtime.operation.completed"
    )


def test_text_sink_is_single_line_and_structured() -> None:
    buffer = io.StringIO()
    handler_id = pxlogging.add_text_sink(buffer)
    pxlogging.enable()
    try:
        with pxlogging.context(run_id="run-1"):
            pxlogging.emit("INFO", "runtime.operation.completed", "One\nline", value=2)
    finally:
        pxlogging.remove_sink(handler_id)

    output = buffer.getvalue()
    assert output.count("\n") == 1
    assert "runtime.operation.completed" in output
    assert 'context={"run_id":"run-1"}' in output
    assert 'fields={"value":2}' in output


def test_event_collections_are_bounded() -> None:
    buffer = io.StringIO()
    handler_id = pxlogging.add_json_sink(buffer)
    pxlogging.enable()
    try:
        pxlogging.emit(
            "INFO",
            "runtime.operation.completed",
            "Completed",
            values=list(range(1_000)),
        )
    finally:
        pxlogging.remove_sink(handler_id)

    event = _read_events(buffer)[0]
    assert len(event["fields"]["values"]) < 1_000
    assert "fields.values.*" in event["omitted_fields"]


def test_training_iteration_kind_emits_canonical_logging_event() -> None:
    buffer = io.StringIO()
    handler_id = pxlogging.add_json_sink(buffer)
    pxlogging.enable()
    try:
        controller = TrainingController(
            total_steps=1,
            key=jr.key(0),
            algorithm_id="logging-test-training",
        )
        controller.emit(
            TrainingIterationKind.RUN_START,
            metrics={"loss": jnp.asarray(1.25)},
        )
    finally:
        pxlogging.remove_sink(handler_id)

    event = _read_events(buffer)[0]
    assert event["event"] == "training.started"
    assert event["fields"]["iteration_kind"] == "run_start"
    assert event["fields"]["metrics"] == [{"name": "loss", "value": 1.25}]
