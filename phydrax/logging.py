#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Structured, privacy-bounded event logging for PhydraX."""

from __future__ import annotations

import heapq
import json
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC
from types import MappingProxyType
from typing import Any, cast, IO, TYPE_CHECKING, TypeAlias

from loguru import logger as _loguru_logger


if TYPE_CHECKING:
    from loguru import FormatFunction, Record

from ._privacy import JSONValue, REDACTED, SecretRedactor


LogValue: TypeAlias = JSONValue
LogSink: TypeAlias = str | os.PathLike[str] | IO[str] | Callable[[str], None]
LogLevel: TypeAlias = str | int

_EVENT_NAME = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_MAX_DEPTH = 6
_MAX_ITEMS = 512
_MAX_TEXT_LENGTH = 4096
_MAX_EVENT_BYTES = 131_072
_CONTEXT: ContextVar[Mapping[str, object]] = ContextVar(
    "phydrax_logging_context", default=MappingProxyType({})
)
_REDACTOR = SecretRedactor()
_ENABLED = False


def enable() -> None:
    """Enable PhydraX event emission without changing application handlers."""

    global _ENABLED
    _loguru_logger.enable("phydrax")
    _ENABLED = True


def disable() -> None:
    """Disable PhydraX event emission without changing application handlers."""

    global _ENABLED
    _ENABLED = False
    _loguru_logger.disable("phydrax")


def is_enabled() -> bool:
    """Return whether PhydraX event emission is enabled."""

    return _ENABLED


@contextmanager
def context(**values: object) -> Iterator[None]:
    """Bind structured correlation values for the dynamic execution scope."""

    for name in values:
        if not _FIELD_NAME.fullmatch(name):
            raise ValueError(f"Invalid logging context field name: {name!r}.")
    merged = dict(_CONTEXT.get())
    merged.update(values)
    token = _CONTEXT.set(MappingProxyType(merged))
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def emit(level: LogLevel, event: str, message: str, /, **fields: object) -> None:
    """Emit one canonical PhydraX event when logging is enabled."""

    if not _ENABLED:
        return
    if not _EVENT_NAME.fullmatch(event):
        raise ValueError(f"Invalid logging event name: {event!r}.")
    for name in fields:
        if not _FIELD_NAME.fullmatch(name):
            raise ValueError(f"Invalid logging field name: {name!r}.")
    if not isinstance(message, str) or not message:
        raise ValueError("Logging event messages must be nonempty strings.")
    _LOGGER.bind(_phydrax_event=(event, fields)).opt(depth=1, capture=False).log(
        level, message
    )


def add_text_sink(
    sink: LogSink,
    /,
    *,
    level: LogLevel = "INFO",
    enqueue: bool = False,
    rotation: str | int | None = None,
    retention: str | int | None = None,
    compression: str | None = None,
) -> int:
    """Add a PhydraX-only human-readable sink and return its Loguru handler ID."""

    return _add_sink(
        sink,
        level=level,
        enqueue=enqueue,
        formatter=_text_format,
        rotation=rotation,
        retention=retention,
        compression=compression,
    )


def add_json_sink(
    sink: LogSink,
    /,
    *,
    level: LogLevel = "DEBUG",
    enqueue: bool = False,
    rotation: str | int | None = None,
    retention: str | int | None = None,
    compression: str | None = None,
) -> int:
    """Add a PhydraX-only canonical JSON-lines sink and return its handler ID."""

    return _add_sink(
        sink,
        level=level,
        enqueue=enqueue,
        formatter=_json_format,
        rotation=rotation,
        retention=retention,
        compression=compression,
    )


def remove_sink(handler_id: int, /) -> None:
    """Remove exactly one previously returned sink handler."""

    _loguru_logger.remove(handler_id)


def _add_sink(
    sink: LogSink,
    /,
    *,
    level: LogLevel,
    enqueue: bool,
    formatter: FormatFunction,
    rotation: str | int | None,
    retention: str | int | None,
    compression: str | None,
) -> int:
    path_sink = isinstance(sink, (str, os.PathLike))
    if not path_sink and any(
        value is not None for value in (rotation, retention, compression)
    ):
        raise ValueError(
            "Rotation, retention, and compression require a filesystem sink."
        )
    if path_sink:
        return _loguru_logger.add(
            cast(str | os.PathLike[str], sink),
            backtrace=False,
            catch=True,
            colorize=False,
            compression=compression,
            delay=True,
            diagnose=False,
            encoding="utf8",
            enqueue=enqueue,
            filter=_event_filter,
            format=formatter,
            level=level,
            mode="a",
            opener=_owner_only_opener,
            retention=retention,
            rotation=rotation,
        )
    return _loguru_logger.add(
        cast(Any, sink),
        backtrace=False,
        catch=True,
        colorize=False,
        diagnose=False,
        enqueue=enqueue,
        filter=_event_filter,
        format=formatter,
        level=level,
    )


def _owner_only_opener(path: str, flags: int) -> int:
    if sys.platform != "win32":
        flags |= os.O_NOFOLLOW
    return os.open(path, flags, 0o600)


def _event_filter(record: Record) -> bool:
    return "phydrax" in record["extra"]


def _patch_record(record: Record) -> None:
    raw_event = record["extra"].pop("_phydrax_event", None)
    if raw_event is None:
        return
    event, raw_fields = raw_event
    omitted: list[str] = []
    nonfinite: list[str] = []
    budget = [_MAX_ITEMS]
    raw_context = _CONTEXT.get()
    normalized_context = _normalize_mapping(
        raw_context,
        path="context",
        depth=0,
        budget=budget,
        omitted=omitted,
        nonfinite=nonfinite,
    )
    normalized_fields = _normalize_mapping(
        raw_fields,
        path="fields",
        depth=0,
        budget=budget,
        omitted=omitted,
        nonfinite=nonfinite,
    )
    message = _sanitize_text(record["message"], path="message", omitted=omitted)
    component = record["name"] or "phydrax"
    payload: dict[str, JSONValue] = {
        "component": component,
        "context": normalized_context,
        "event": event,
        "fields": normalized_fields,
        "nonfinite_fields": nonfinite,
        "omitted_fields": omitted,
    }
    if len(_canonical_json(payload).encode("utf8")) > _MAX_EVENT_BYTES:
        payload["fields"] = {"event_too_large": True}
        payload["omitted_fields"] = [*omitted, "fields"]
    record["message"] = message
    record["extra"]["phydrax"] = payload


def _normalize_mapping(
    value: Mapping[str, object],
    /,
    *,
    path: str,
    depth: int,
    budget: list[int],
    omitted: list[str],
    nonfinite: list[str],
) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    keys = heapq.nsmallest(
        max(budget[0], 0),
        (
            key
            for key in value
            if isinstance(key, str) and _FIELD_NAME.fullmatch(key)
        ),
    )
    if len(keys) < len(value):
        omitted.append(f"{path}.*")
    for key in keys:
        if budget[0] <= 0:
            omitted.append(f"{path}.*")
            break
        child_path = f"{path}.{key}"
        result[key] = _normalize_value(
            value[key],
            field_name=key,
            path=child_path,
            depth=depth + 1,
            budget=budget,
            omitted=omitted,
            nonfinite=nonfinite,
        )
    return result


def _normalize_value(
    value: object,
    /,
    *,
    field_name: str,
    path: str,
    depth: int,
    budget: list[int],
    omitted: list[str],
    nonfinite: list[str],
) -> JSONValue:
    if budget[0] <= 0 or depth > _MAX_DEPTH:
        omitted.append(path)
        return None
    budget[0] -= 1
    if value is None or isinstance(value, (bool, int)):
        normalized: JSONValue = value
    elif isinstance(value, float):
        if float("-inf") < value < float("inf"):
            normalized = value
        else:
            nonfinite.append(path)
            normalized = None
    elif isinstance(value, str):
        normalized = _sanitize_text(value, path=path, omitted=omitted)
    elif isinstance(value, Mapping):
        normalized = _normalize_mapping(
            value,
            path=path,
            depth=depth,
            budget=budget,
            omitted=omitted,
            nonfinite=nonfinite,
        )
    elif isinstance(value, (tuple, list)):
        items: list[JSONValue] = []
        for index, item in enumerate(value):
            if budget[0] <= 0:
                omitted.append(f"{path}.*")
                break
            items.append(
                _normalize_value(
                    item,
                    field_name=field_name,
                    path=f"{path}.{index}",
                    depth=depth + 1,
                    budget=budget,
                    omitted=omitted,
                    nonfinite=nonfinite,
                )
            )
        normalized = items
    else:
        omitted.append(path)
        normalized = None
    return _REDACTOR.redact(normalized, field_name=field_name)


def _sanitize_text(value: str, /, *, path: str, omitted: list[str]) -> str:
    redacted = _REDACTOR.redact(value)
    if redacted == REDACTED:
        return REDACTED
    text = cast(str, redacted).replace("\r", " ").replace("\n", " ")
    if len(text) <= _MAX_TEXT_LENGTH:
        return text
    omitted.append(path)
    return f"{text[:_MAX_TEXT_LENGTH]}<truncated>"


def _json_format(record: Record) -> str:
    event = cast(dict[str, JSONValue], record["extra"]["phydrax"])
    timestamp = record["time"].astimezone(UTC).isoformat().replace(
        "+00:00", "Z"
    )
    payload: dict[str, object] = {
        "component": event["component"],
        "context": event["context"],
        "event": event["event"],
        "fields": event["fields"],
        "level": record["level"].name,
        "message": record["message"],
        "nonfinite_fields": event["nonfinite_fields"],
        "omitted_fields": event["omitted_fields"],
        "process": {
            "id": record["process"].id,
            "name": record["process"].name,
        },
        "source": {
            "function": record["function"],
            "line": record["line"],
            "module": record["module"],
        },
        "thread": {
            "id": record["thread"].id,
            "name": record["thread"].name,
        },
        "time": timestamp,
    }
    record["extra"]["_phydrax_serialized"] = _canonical_json(payload)
    return "{extra[_phydrax_serialized]}\n"


def _text_format(record: Record) -> str:
    event = cast(dict[str, JSONValue], record["extra"]["phydrax"])
    timestamp = record["time"].astimezone(UTC).isoformat().replace(
        "+00:00", "Z"
    )
    context_text = _canonical_json(event["context"])
    fields_text = _canonical_json(event["fields"])
    text = (
        f"{timestamp} | {record['level'].name:<8} | {event['event']} | "
        f"{record['message']} | context={context_text} | fields={fields_text}"
    )
    record["extra"]["_phydrax_text"] = text
    return "{extra[_phydrax_text]}\n"


def _canonical_json(value: object, /) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


_LOGGER = _loguru_logger.patch(_patch_record)
disable()


__all__ = [
    "LogValue",
    "add_json_sink",
    "add_text_sink",
    "context",
    "disable",
    "emit",
    "enable",
    "is_enabled",
    "remove_sink",
]
