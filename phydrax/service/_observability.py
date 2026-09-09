#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Privacy-classified local telemetry and allowlisted support bundles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import jax

from .._privacy import (
    JSONValue,
    PrivacyClassification,
    REDACTED,
    SecretRedactor,
)
from ..logging import emit
from ._auth import Clock, SystemClock


@dataclass(frozen=True, slots=True)
class TelemetryDatum:
    name: str
    value: JSONValue
    unit: str
    classification: PrivacyClassification
    observed_at: int

    def __post_init__(self) -> None:
        if not self.name or not self.unit or self.observed_at < 0:
            raise ValueError("Telemetry name, unit, and timestamp must be valid.")
        # Reject NaN/infinity and non-JSON values at the collection boundary.
        json.dumps(self.value, allow_nan=False, separators=(",", ":"))


def _telemetry_record(value: TelemetryDatum, /) -> dict[str, JSONValue]:
    return {
        "classification": value.classification.name.lower(),
        "name": value.name,
        "observed_at": value.observed_at,
        "unit": value.unit,
        "value": value.value,
    }

@dataclass(frozen=True, slots=True)
class HostTelemetrySnapshot:
    observations: tuple[TelemetryDatum, ...]
    snapshot_id: str

    @classmethod
    def create(
        cls, observations: tuple[TelemetryDatum, ...], /
    ) -> "HostTelemetrySnapshot":
        values = tuple(sorted(observations, key=lambda value: value.name))
        if len({value.name for value in values}) != len(values):
            raise ValueError("Telemetry observation names must be unique.")
        payload = [_telemetry_record(value) for value in values]
        digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        return cls(values, digest)

    def export(
        self, maximum_classification: PrivacyClassification, /
    ) -> dict[str, JSONValue]:
        return {
            value.name: value.value
            for value in self.observations
            if value.classification <= maximum_classification
        }

    def to_record(
        self,
        maximum_classification: PrivacyClassification = PrivacyClassification.INTERNAL,
        /,
    ) -> dict[str, JSONValue]:
        observations = [
            _telemetry_record(value)
            for value in self.observations
            if value.classification <= maximum_classification
        ]
        return {
            "observations": observations,
            "snapshot_id": self.snapshot_id,
        }

    def to_json(
        self,
        maximum_classification: PrivacyClassification = PrivacyClassification.INTERNAL,
        /,
    ) -> str:
        return _canonical_bytes(self.to_record(maximum_classification)).decode("utf-8")


@dataclass(frozen=True, slots=True)
class HostTelemetryPolicy:
    """Explicit privacy and runtime-initialization controls for host collection."""

    include_host_identity: bool = False
    include_jax_runtime: bool = True
    include_jax_devices: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("include_host_identity", self.include_host_identity),
            ("include_jax_runtime", self.include_jax_runtime),
            ("include_jax_devices", self.include_jax_devices),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean.")
        if self.include_jax_devices and not self.include_jax_runtime:
            raise ValueError("JAX device collection requires JAX runtime collection.")


class HostTelemetryCollector:
    """Explicit pull-only collector. Construction and import perform no collection."""

    def __init__(
        self,
        /,
        *,
        clock: Clock | None = None,
        policy: HostTelemetryPolicy | None = None,
    ):
        self._clock = SystemClock() if clock is None else clock
        self._policy = HostTelemetryPolicy() if policy is None else policy
        if not isinstance(self._policy, HostTelemetryPolicy):
            raise TypeError("policy must be HostTelemetryPolicy or None.")

    def collect(self) -> HostTelemetrySnapshot:
        now = self._clock.now()
        observations = [
            TelemetryDatum(
                "host.architecture",
                platform.machine() or "unknown",
                "string",
                PrivacyClassification.INTERNAL,
                now,
            ),
            TelemetryDatum(
                "host.cpu.logical_count",
                os.cpu_count() or 0,
                "count",
                PrivacyClassification.INTERNAL,
                now,
            ),
            TelemetryDatum(
                "host.os",
                platform.system() or "unknown",
                "string",
                PrivacyClassification.INTERNAL,
                now,
            ),
            TelemetryDatum(
                "host.os.release",
                platform.release() or "unknown",
                "string",
                PrivacyClassification.INTERNAL,
                now,
            ),
            TelemetryDatum(
                "host.os.kernel",
                platform.version() or "unknown",
                "string",
                PrivacyClassification.INTERNAL,
                now,
            ),
            TelemetryDatum(
                "host.python.implementation",
                platform.python_implementation(),
                "string",
                PrivacyClassification.INTERNAL,
                now,
            ),
            TelemetryDatum(
                "host.python.version",
                platform.python_version(),
                "string",
                PrivacyClassification.INTERNAL,
                now,
            ),
        ]
        for distribution, name in (
            ("phydrax", "runtime.phydrax.version"),
            ("jax", "runtime.jax.version"),
            ("jaxlib", "runtime.jaxlib.version"),
        ):
            version = _distribution_version(distribution)
            if version is not None:
                observations.append(
                    TelemetryDatum(
                        name,
                        version,
                        "string",
                        PrivacyClassification.INTERNAL,
                        now,
                    )
                )
        if self._policy.include_jax_runtime:
            observations.extend(
                (
                    TelemetryDatum(
                        "runtime.jax.process_count",
                        int(jax.process_count()),
                        "count",
                        PrivacyClassification.INTERNAL,
                        now,
                    ),
                    TelemetryDatum(
                        "runtime.jax.process_index",
                        int(jax.process_index()),
                        "index",
                        PrivacyClassification.INTERNAL,
                        now,
                    ),
                    TelemetryDatum(
                        "runtime.jax.x64_enabled",
                        bool(jax.config.read("jax_enable_x64")),
                        "boolean",
                        PrivacyClassification.INTERNAL,
                        now,
                    ),
                )
            )
        if self._policy.include_jax_devices:
            devices = tuple(jax.devices())
            observations.extend(
                (
                    TelemetryDatum(
                        "runtime.jax.device_count",
                        len(devices),
                        "count",
                        PrivacyClassification.INTERNAL,
                        now,
                    ),
                    TelemetryDatum(
                        "runtime.jax.device_kinds",
                        sorted({device.device_kind for device in devices}),
                        "string-list",
                        PrivacyClassification.INTERNAL,
                        now,
                    ),
                    TelemetryDatum(
                        "runtime.jax.device_platforms",
                        sorted({device.platform for device in devices}),
                        "string-list",
                        PrivacyClassification.INTERNAL,
                        now,
                    ),
                )
            )
        memory = _physical_memory_bytes()
        if memory is not None:
            observations.append(
                TelemetryDatum(
                    "host.memory.physical",
                    memory,
                    "bytes",
                    PrivacyClassification.INTERNAL,
                    now,
                )
            )
        if self._policy.include_host_identity:
            observations.append(
                TelemetryDatum(
                    "host.name",
                    socket.gethostname(),
                    "string",
                    PrivacyClassification.SENSITIVE,
                    now,
                )
            )
        snapshot = HostTelemetrySnapshot.create(tuple(observations))
        emit(
            "INFO",
            "runtime.environment.captured",
            "Runtime environment captured",
            include_host_identity=self._policy.include_host_identity,
            include_jax_devices=self._policy.include_jax_devices,
            observation_count=len(snapshot.observations),
            snapshot_id=snapshot.snapshot_id,
        )
        return snapshot


def _distribution_version(distribution: str, /) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _physical_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if (
        not isinstance(pages, int)
        or not isinstance(page_size, int)
        or pages <= 0
        or page_size <= 0
    ):
        return None
    return pages * page_size




@dataclass(frozen=True, slots=True)
class SupportBundlePolicy:
    allowed_fields: Mapping[str, frozenset[str]]
    maximum_telemetry_classification: PrivacyClassification = (
        PrivacyClassification.INTERNAL
    )

    def __post_init__(self) -> None:
        normalized: dict[str, frozenset[str]] = {}
        for section, fields in self.allowed_fields.items():
            if not section or not fields or any(not field for field in fields):
                raise ValueError(
                    "Support bundle allowlist sections and fields must be nonempty."
                )
            normalized[section] = frozenset(fields)
        if not normalized:
            raise ValueError(
                "Support bundle policy requires an explicit nonempty allowlist."
            )
        object.__setattr__(self, "allowed_fields", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class SupportBundle:
    bundle_id: str
    created_at: int
    sections: Mapping[str, Mapping[str, JSONValue]]
    redaction_marker: str = REDACTED

    def __post_init__(self) -> None:
        if not self.bundle_id or self.created_at < 0:
            raise ValueError("Support bundle identity and timestamp must be valid.")
        frozen = {
            section: MappingProxyType(dict(values))
            for section, values in self.sections.items()
        }
        object.__setattr__(self, "sections", MappingProxyType(frozen))

    def to_json(self) -> str:
        return json.dumps(
            {
                "bundle_id": self.bundle_id,
                "created_at": self.created_at,
                "redaction_marker": self.redaction_marker,
                "sections": {
                    name: dict(values) for name, values in self.sections.items()
                },
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


def create_support_bundle(
    sources: Mapping[str, Mapping[str, object]],
    policy: SupportBundlePolicy,
    /,
    *,
    clock: Clock | None = None,
    telemetry: HostTelemetrySnapshot | None = None,
    redactor: SecretRedactor | None = None,
) -> SupportBundle:
    """Build a bundle from explicitly supplied data; it performs no collection or I/O."""

    redact = SecretRedactor() if redactor is None else redactor
    selected: dict[str, dict[str, JSONValue]] = {}
    for section, allowed in sorted(policy.allowed_fields.items()):
        source = sources.get(section, {})
        values = {
            field: redact.redact(source[field], field_name=field)
            for field in sorted(allowed)
            if field in source
        }
        selected[section] = values
    if telemetry is not None and "telemetry" in policy.allowed_fields:
        allowed = policy.allowed_fields["telemetry"]
        exported = telemetry.export(policy.maximum_telemetry_classification)
        selected["telemetry"] = {
            field: redact.redact(exported[field], field_name=field)
            for field in sorted(allowed)
            if field in exported
        }
    now = (SystemClock() if clock is None else clock).now()
    content = {"created_at": now, "sections": selected}
    bundle_id = hashlib.sha256(_canonical_bytes(content)).hexdigest()
    return SupportBundle(
        bundle_id,
        now,
        {section: MappingProxyType(values) for section, values in selected.items()},
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


__all__ = [
    "HostTelemetryCollector",
    "HostTelemetryPolicy",
    "HostTelemetrySnapshot",
    "PrivacyClassification",
    "SecretRedactor",
    "SupportBundle",
    "SupportBundlePolicy",
    "TelemetryDatum",
    "create_support_bundle",
]
