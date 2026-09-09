#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared privacy classification and structural secret redaction."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import IntEnum
from typing import TypeAlias


JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
REDACTED = "<redacted>"
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
    """Maximum permitted disclosure of an observation or field."""

    PUBLIC = 0
    INTERNAL = 1
    SENSITIVE = 2
    RESTRICTED = 3


class SecretRedactor:
    """Recursively redact secret-shaped names and values without using repr()."""

    def redact(self, value: object, /, *, field_name: str = "") -> JSONValue:
        if field_name and _SENSITIVE_NAMES.search(field_name):
            return REDACTED
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            if not (float("-inf") < value < float("inf")):
                return REDACTED
            return value
        if isinstance(value, bytes):
            return REDACTED
        if isinstance(value, str):
            if any(pattern.search(value) for pattern in _SECRET_VALUES):
                return REDACTED
            return value
        if isinstance(value, Mapping):
            return {
                str(key): self.redact(item, field_name=str(key))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (tuple, list)):
            return [self.redact(item, field_name=field_name) for item in value]
        return REDACTED


__all__ = ["JSONValue", "PrivacyClassification", "SecretRedactor", "REDACTED"]
