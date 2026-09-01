#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np


CcsdsMessageKind: TypeAlias = Literal["OEM", "OPM", "AEM", "APM", "TDM", "CDM"]


@dataclass(frozen=True)
class CcsdsHeader:
    version: str
    creation_date: str
    originator: str
    comments: tuple[str, ...]


@dataclass(frozen=True)
class CcsdsMessage:
    kind: CcsdsMessageKind
    header: CcsdsHeader
    metadata: tuple[tuple[str, str], ...]
    data: tuple[tuple[str, ...], ...]
    raw_extensions: tuple[tuple[str, str], ...]

    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)


def parse_ccsds_kvn(text: str, kind: CcsdsMessageKind, /) -> CcsdsMessage:
    """Parse a strict CCSDS KVN product while preserving unknown key/value extensions."""

    if kind not in ("OEM", "OPM", "AEM", "APM", "TDM", "CDM"):
        raise ValueError("Unknown CCSDS message kind.")
    version = ""
    creation = ""
    originator = ""
    comments: list[str] = []
    metadata: list[tuple[str, str]] = []
    data: list[tuple[str, ...]] = []
    extensions: list[tuple[str, str]] = []
    in_metadata = False
    in_data = False
    known_metadata = {
        "OBJECT_NAME",
        "OBJECT_ID",
        "CENTER_NAME",
        "REF_FRAME",
        "TIME_SYSTEM",
        "START_TIME",
        "STOP_TIME",
        "INTERPOLATION",
        "INTERPOLATION_DEGREE",
    }
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "META_START":
            in_metadata = True
            continue
        if line == "META_STOP":
            in_metadata = False
            in_data = True
            continue
        if line.startswith("COMMENT"):
            comments.append(line.removeprefix("COMMENT").strip())
            continue
        if "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            if key.endswith("_VERS"):
                version = value
            elif key == "CREATION_DATE":
                creation = value
            elif key == "ORIGINATOR":
                originator = value
            elif in_metadata and key in known_metadata:
                metadata.append((key, value))
            else:
                extensions.append((key, value))
            continue
        if in_data:
            data.append(tuple(line.split()))
    if not version or not creation or not originator:
        raise ValueError("CCSDS KVN header is incomplete.")
    if kind in ("OEM", "AEM", "TDM") and not data:
        raise ValueError("CCSDS message has no data records.")
    return CcsdsMessage(
        kind,
        CcsdsHeader(version, creation, originator, tuple(comments)),
        tuple(metadata),
        tuple(data),
        tuple(extensions),
    )


def ccsds_numeric_records(message: CcsdsMessage, /) -> np.ndarray:
    rows = []
    for record in message.data:
        if len(record) < 2:
            raise ValueError("CCSDS numeric record is incomplete.")
        rows.append(tuple(float(value) for value in record[1:]))
    if not rows:
        return np.empty((0, 0), dtype=float)
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("CCSDS numeric records have inconsistent widths.")
    return np.asarray(rows, dtype=float)


__all__ = [
    "CcsdsHeader",
    "CcsdsMessage",
    "CcsdsMessageKind",
    "ccsds_numeric_records",
    "parse_ccsds_kvn",
]
