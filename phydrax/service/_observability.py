#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Privacy-classified local telemetry and allowlisted support bundles."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Mapping, TypeAlias

from ._auth import Clock, SystemClock


JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
_REDACTED = "<redacted>"
_SENSITIVE_NAMES = re.compile(
    r"(?:^|[_\-.])(authorization|cookie|credential|password|passwd|secret|token|private[_-]?key|api[_-]?key|session)(?:$|[_\-.])",
    re.IGNORECASE,
)
_SECRET_VALUES = (
    re.compile(r"(?i)^bearer\s+\S+$"),
    re.compile(r"^eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$"),
    re.compile(r"^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"^(?:AKIA|ASIA)[A-Z0-9]{16}$"),
    re.compile(
        r"(?i)(?:authorization|credential|password|passwd|secret|token|"
        r"api[_-]?key)\s*[:=]\s*\S+"
    ),
)


class PrivacyClassification(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    SENSITIVE = 2
    RESTRICTED = 3


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
        payload = [
            {
                "classification": value.classification.name.lower(),
                "name": value.name,
                "observed_at": value.observed_at,
                "unit": value.unit,
                "value": value.value,
            }
            for value in values
        ]
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


class HostTelemetryCollector:
    """Explicit pull-only collector. Construction and import perform no collection."""

    def __init__(
        self, /, *, clock: Clock | None = None, include_host_identity: bool = False
    ):
        self._clock = SystemClock() if clock is None else clock
        self._include_identity = include_host_identity

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
        if self._include_identity:
            observations.append(
                TelemetryDatum(
                    "host.name",
                    socket.gethostname(),
                    "string",
                    PrivacyClassification.SENSITIVE,
                    now,
                )
            )
        return HostTelemetrySnapshot.create(tuple(observations))


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


class SecretRedactor:
    """Structural redactor used only after support-bundle field allowlisting."""

    def redact(self, value: object, /, *, field_name: str = "") -> JSONValue:
        if field_name and _SENSITIVE_NAMES.search(field_name):
            return _REDACTED
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            if not (float("-inf") < value < float("inf")):
                return _REDACTED
            return value
        if isinstance(value, bytes):
            return _REDACTED
        if isinstance(value, str):
            if any(pattern.search(value) for pattern in _SECRET_VALUES):
                return _REDACTED
            return value
        if isinstance(value, Mapping):
            return {
                str(key): self.redact(item, field_name=str(key))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (tuple, list)):
            return [self.redact(item, field_name=field_name) for item in value]
        return _REDACTED


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
    redaction_marker: str = _REDACTED

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
    "HostTelemetrySnapshot",
    "PrivacyClassification",
    "SecretRedactor",
    "SupportBundle",
    "SupportBundlePolicy",
    "TelemetryDatum",
    "create_support_bundle",
]
