#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Real optional HELICS value federates with explicit, noniterative time grants."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ..._fingerprint import canonical_fingerprint
from ...artifacts import ScientificArtifactEnvelope
from ..energy_runtime import _artifact, _host_only, _HostWorker, _require_optional


_TYPES = ("double", "integer", "boolean", "string", "complex", "vector")


@dataclass(frozen=True, slots=True)
class HelicsChannel:
    """An exact typed/unit endpoint; subscription targets name global publications."""

    name: str
    type: str
    unit: str = ""
    target: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or "\x00" in self.name or self.type not in _TYPES:
            raise ValueError(
                "HELICS channels require a nonempty name and a supported exact type."
            )
        if "\x00" in self.unit or (
            self.target is not None and (not self.target.strip() or "\x00" in self.target)
        ):
            raise ValueError("Invalid HELICS channel unit/target.")


@dataclass(frozen=True, slots=True)
class HelicsSample:
    channel: str
    value: float | int | bool | str | complex | tuple[float, ...] | None
    granted_time: float
    last_update_time: float | None
    updated: bool
    has_value: bool


@dataclass(frozen=True, slots=True)
class HelicsTimeGrant:
    requested_time: float
    granted_time: float
    interrupted: bool
    terminated: bool
    artifact: ScientificArtifactEnvelope


def _value(channel: HelicsChannel, value: Any) -> Any:
    kind = channel.type
    if kind == "double":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("HELICS double values must be finite numbers.")
        return float(value)
    if kind == "integer":
        if type(value) is not int or not -(2**63) <= value < 2**63:
            raise ValueError("HELICS integer values must be int64 integers.")
    elif kind == "boolean" and type(value) is not bool:
        raise TypeError("HELICS Boolean values must be bool.")
    elif kind == "string" and (not isinstance(value, str) or "\x00" in value):
        raise TypeError("HELICS string values must be NUL-free strings.")
    elif kind == "complex":
        if (
            not isinstance(value, complex)
            or not math.isfinite(value.real)
            or not math.isfinite(value.imag)
        ):
            raise ValueError(
                "HELICS complex values must have finite real and imaginary parts."
            )
        return [value.real, value.imag]
    elif kind == "vector":
        if not isinstance(value, (tuple, list)) or len(value) > 65536:
            raise ValueError(
                "HELICS vectors must be bounded lists/tuples of finite real numbers."
            )
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in value
        ):
            raise ValueError("HELICS vectors require finite real entries.")
        return list(value)
    return value


class HelicsValueSession:
    """A resource-owned HELICS3 value federate in an isolated native process.

    With ``broker=None``, owns a broker for ``federate_count`` participants; its
    address is available before execution entry. With an external broker, never
    disconnects or frees that broker. Async start/complete pairs allow a single
    host thread to coordinate several sessions without pretending time requests
    are independent. No rollback, convergence, iteration or derivatives are claimed.
    """

    def __init__(
        self,
        name: str,
        *,
        publications: Sequence[HelicsChannel] = (),
        subscriptions: Sequence[HelicsChannel] = (),
        license_id: str,
        broker: str | None = None,
        federate_count: int = 1,
        core_type: str = "zmq",
        time_delta: float = 1e-9,
        timeout: float = 30,
        expected_version: str | None = None,
        source_url: str = "",
    ):
        _host_only(time_delta, publications, subscriptions)
        _require_optional(
            "helics", "install helics>=3 with a host-compatible native HELICS library"
        )
        if not name.strip() or "\x00" in name or not license_id.strip():
            raise ValueError(
                "A nonempty federate name and explicit license_id are required."
            )
        if core_type not in ("zmq", "tcp", "ipc"):
            raise ValueError("Supported cross-process HELICS cores are zmq, tcp and ipc.")
        if type(federate_count) is not int or federate_count < 1:
            raise ValueError("federate_count must be a positive integer.")
        if not math.isfinite(time_delta) or time_delta <= 0:
            raise ValueError("time_delta must be finite and positive.")
        publications, subscriptions = tuple(publications), tuple(subscriptions)
        if len(publications) + len(subscriptions) > 4096:
            raise ValueError("HELICS channel count exceeds the configured subset.")
        for channels, is_input in ((publications, False), (subscriptions, True)):
            if any(not isinstance(channel, HelicsChannel) for channel in channels):
                raise TypeError("Channels must be HelicsChannel records.")
            if len({channel.name for channel in channels}) != len(channels):
                raise ValueError(
                    "HELICS channel names must be unique within each direction."
                )
            if any((channel.target is not None) != is_input for channel in channels):
                raise ValueError(
                    "Subscriptions require targets; publications cannot specify targets."
                )
        self._publications = {channel.name: channel for channel in publications}
        self._subscriptions = {channel.name: channel for channel in subscriptions}
        self._license = license_id
        self.mode = "created"
        self.time = 0.0
        self._requested_time = None
        config = {
            "name": name,
            "publications": [asdict(c) for c in publications],
            "subscriptions": [asdict(c) for c in subscriptions],
            "broker": broker,
            "federate_count": federate_count,
            "core_type": core_type,
            "time_delta": float(time_delta),
            "expected_version": expected_version,
        }
        self._resource_id = canonical_fingerprint(
            {"config": config, "source_url": source_url}
        )
        self._worker = _HostWorker("helics", config, inputs={}, timeout=timeout)
        self.broker_address = self._worker.info["broker_address"]

    @property
    def closed(self) -> bool:
        return self._worker.closed

    @property
    def artifact(self) -> ScientificArtifactEnvelope:
        failed = next(
            (
                call["error"]
                for call in reversed(self._worker.calls)
                if call["status"] == "failed"
            ),
            "",
        )
        return _artifact(
            "helics-value-session",
            {"calls": self._worker.calls, "runtime": self._worker.info},
            producer="HELICS",
            version=self._worker.info["version"],
            build_id=self._worker.info["build_id"],
            license_id=self._license,
            resource_id=self._resource_id,
            error=failed,
        )

    def _require_mode(self, mode: str) -> None:
        if self.closed or self.mode != mode:
            raise RuntimeError(
                f"HELICS operation requires {mode!r}; current mode is {self.mode!r}."
            )

    def enter_execution_async(self) -> None:
        self._require_mode("created")
        self._worker.call("enter_async")
        self.mode = "entering"

    def complete_execution(self) -> None:
        self._require_mode("entering")
        self.time = self._worker.call("enter_complete")
        self.mode = "executing"

    def enter_execution(self) -> None:
        self.enter_execution_async()
        self.complete_execution()

    def publish(self, values: Mapping[str, Any]) -> None:
        self._require_mode("executing")
        _host_only(values)
        normalized = {
            name: _value(self._publications[name], value)
            for name, value in values.items()
        }
        self._worker.call("publish", {"values": normalized})

    def request_time_async(self, target_time: float) -> None:
        self._require_mode("executing")
        _host_only(target_time)
        target_time = float(target_time)
        if (
            not math.isfinite(target_time)
            or target_time <= self.time
            or target_time >= 9223372036.854774
        ):
            raise ValueError(
                "HELICS requests must advance to a finite time below HELICS_TIME_MAXTIME."
            )
        self._worker.call("time_async", {"target_time": target_time})
        self._requested_time = target_time
        self.mode = "advancing"

    def complete_time(self) -> HelicsTimeGrant:
        self._require_mode("advancing")
        result = self._worker.call("time_complete")
        self.time = result["time"]
        self.mode = "terminated" if result["terminated"] else "executing"
        return HelicsTimeGrant(
            self._requested_time,
            self.time,
            not result["terminated"] and self.time < self._requested_time,
            result["terminated"],
            self.artifact,
        )

    def advance(self, target_time: float) -> HelicsTimeGrant:
        self.request_time_async(target_time)
        return self.complete_time()

    def read_values(self, names: Sequence[str] | None = None) -> tuple[HelicsSample, ...]:
        self._require_mode("executing")
        selected = tuple(self._subscriptions) if names is None else tuple(names)
        for name in selected:
            self._subscriptions[name]
        data = self._worker.call("read", {"names": selected})
        result = []
        for item in data:
            value = item["value"]
            if item["has_value"]:
                if self._subscriptions[item["channel"]].type == "complex":
                    value = complex(*value)
                elif self._subscriptions[item["channel"]].type == "vector":
                    value = tuple(value)
            result.append(
                HelicsSample(
                    item["channel"],
                    value,
                    self.time,
                    item["last_update_time"],
                    item["updated"],
                    item["has_value"],
                )
            )
        return tuple(result)

    def close(self) -> None:
        try:
            self._worker.close()
        finally:
            self.mode = "closed"

    def __enter__(self) -> HelicsValueSession:
        if self.closed:
            raise RuntimeError("HELICS session is closed.")
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> bool:
        try:
            self.close()
        except Exception as cleanup_error:
            if exception is None:
                raise
            exception.add_note(f"HELICS cleanup also failed: {cleanup_error}")
        return False


__all__ = ["HelicsChannel", "HelicsSample", "HelicsTimeGrant", "HelicsValueSession"]
