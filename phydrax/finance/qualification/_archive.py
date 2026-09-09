#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._array_archive import (
    array_collection_digest,
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    read_array_archive,
    write_array_archive,
)
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...lifecycle._archive import payload_digest
from ...lifecycle._models import ResultManifest
from ...qualification._registry import SupportTuple


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _identifiers(
    values: Sequence[str], name: str, /, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of strings.")
    result = tuple(sorted(_text(value, name) for value in values))
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique.")
    return result


def _named_arrays(values: Mapping[str, Any], /) -> tuple[tuple[str, np.ndarray], ...]:
    if not isinstance(values, Mapping) or not values:
        raise TypeError("arrays must be a non-empty mapping of named physical arrays.")
    result: list[tuple[str, np.ndarray]] = []
    for name, value in values.items():
        name_ = _text(name, "array name")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"Finance result array {name_!r} cannot have object dtype.")
        result.append((name_, array))
    result.sort(key=lambda item: item[0])
    if len({name for name, _ in result}) != len(result):
        raise ValueError("Finance result array names must be unique.")
    return tuple(result)


def _support_tuples(values: Sequence[SupportTuple], /) -> tuple[SupportTuple, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise TypeError("support_tuples must be a sequence.")
    result = tuple(values)
    if not result or any(not isinstance(item, SupportTuple) for item in result):
        raise TypeError("support_tuples must contain at least one SupportTuple.")
    result = tuple(sorted(result, key=lambda item: item.support_tuple_id))
    if len({item.support_tuple_id for item in result}) != len(result):
        raise ValueError("support_tuples must be unique.")
    return result


def _manifest_record(manifest: ResultManifest, /) -> dict[str, object]:
    return {
        "kind": "result-manifest",
        "result_id": manifest.result_id,
        "run_id": manifest.run_id,
        "fields": [list(item) for item in manifest.fields],
        "payloads": [list(item) for item in manifest.payloads],
        "evidence_ids": list(manifest.evidence_ids),
        "diagnostic_ids": list(manifest.diagnostic_ids),
        "manifest_id": manifest.manifest_id,
    }


def _manifest_from_record(record: Mapping[str, object], /) -> ResultManifest:
    expected = {
        "kind",
        "result_id",
        "run_id",
        "fields",
        "payloads",
        "evidence_ids",
        "diagnostic_ids",
        "manifest_id",
    }
    if set(record) != expected or record["kind"] != "result-manifest":
        raise ValueError("Archived finance result manifest fields are not canonical.")
    fields_record = record["fields"]
    payloads_record = record["payloads"]
    evidence_record = record["evidence_ids"]
    diagnostics_record = record["diagnostic_ids"]
    sequences = (fields_record, payloads_record, evidence_record, diagnostics_record)
    if any(
        not isinstance(value, Sequence) or isinstance(value, str) for value in sequences
    ):
        raise TypeError("Archived result manifest collections must be sequences.")
    fields: list[tuple[str, str, str]] = []
    for item in fields_record:
        if not isinstance(item, Sequence) or isinstance(item, str) or len(item) != 3:
            raise TypeError("Archived result fields must be three-string records.")
        fields.append((str(item[0]), str(item[1]), str(item[2])))
    payloads: list[tuple[str, str]] = []
    for item in payloads_record:
        if not isinstance(item, Sequence) or isinstance(item, str) or len(item) != 2:
            raise TypeError("Archived result payloads must be two-string records.")
        payloads.append((str(item[0]), str(item[1])))
    value = ResultManifest(
        str(record["result_id"]),
        str(record["run_id"]),
        fields,
        payloads,
        evidence_ids=tuple(str(item) for item in evidence_record),
        diagnostic_ids=tuple(str(item) for item in diagnostics_record),
    )
    if value.manifest_id != record["manifest_id"]:
        raise ValueError("Archived finance result manifest identity is invalid.")
    return value


def _verify_arrays(manifest: ResultManifest, arrays: Mapping[str, Any], /) -> None:
    values = _named_arrays(arrays)
    names = {name for name, _ in values}
    payloads = dict(manifest.payloads)
    fields = {field for field, _, _ in manifest.fields}
    field_payloads = {payload for _, payload, _ in manifest.fields}
    if names != set(payloads) or names != fields or names != field_payloads:
        raise ValueError(
            "Finance result fields, payloads, physical arrays, and units "
            "must align exactly."
        )
    for name, value in values:
        if payload_digest(value) != payloads[name]:
            raise ValueError(
                f"Finance result array {name!r} does not match its manifest."
            )


class FinanceArchiveRecord(StrictModule, NonTrainableState):
    """Reopened, checksum-bound finance result with named physical arrays."""

    path: str = eqx.field(static=True)
    result_manifest: ResultManifest
    array_names: tuple[str, ...] = eqx.field(static=True)
    array_values: tuple[Array, ...]
    support_tuples: tuple[SupportTuple, ...]
    replay_id: str = eqx.field(static=True)
    law_ids: tuple[str, ...] = eqx.field(static=True)
    archive_id: str = eqx.field(static=True)

    def __init__(
        self,
        path: str | Path,
        result_manifest: ResultManifest,
        arrays: Mapping[str, Any],
        support_tuples: Sequence[SupportTuple],
        replay_id: str,
        law_ids: Sequence[str],
        archive_id: str,
        /,
    ):
        if not isinstance(result_manifest, ResultManifest):
            raise TypeError("result_manifest must be a ResultManifest.")
        values = _named_arrays(arrays)
        _verify_arrays(result_manifest, dict(values))
        self.path = str(Path(path))
        self.result_manifest = result_manifest
        self.array_names = tuple(name for name, _ in values)
        self.array_values = tuple(jnp.asarray(value) for _, value in values)
        self.support_tuples = _support_tuples(support_tuples)
        self.replay_id = _text(replay_id, "replay_id")
        self.law_ids = _identifiers(law_ids, "law_ids", allow_empty=True)
        self.archive_id = _text(archive_id, "archive_id")

    @property
    def arrays(self) -> Mapping[str, Array]:
        """Return a read-only name-to-array view in canonical field order."""
        return MappingProxyType(
            dict(zip(self.array_names, self.array_values, strict=True))
        )


def finance_result_manifest(
    result_id: str,
    run_id: str,
    arrays: Mapping[str, Any],
    field_units: Mapping[str, str],
    /,
    *,
    evidence_ids: Sequence[str] = (),
    diagnostic_ids: Sequence[str] = (),
) -> ResultManifest:
    """Build a lifecycle result manifest for named arrays with explicit physical units."""
    values = _named_arrays(arrays)
    if not isinstance(field_units, Mapping):
        raise TypeError("field_units must be a mapping.")
    if set(field_units) != {name for name, _ in values}:
        raise ValueError("field_units must cover every physical result array exactly.")
    units = {name: _text(field_units[name], f"unit for {name}") for name, _ in values}
    payloads = {name: payload_digest(value) for name, value in values}
    evidence = _identifiers(evidence_ids, "evidence_ids", allow_empty=True)
    diagnostics = _identifiers(diagnostic_ids, "diagnostic_ids", allow_empty=True)
    return ResultManifest(
        _text(result_id, "result_id"),
        _text(run_id, "run_id"),
        units,
        payloads,
        evidence_ids=evidence,
        diagnostic_ids=diagnostics,
    )


def _archive_content(
    result_manifest: ResultManifest,
    arrays: Mapping[str, Any],
    support_tuples: Sequence[SupportTuple],
    replay_id: str,
    law_ids: Sequence[str],
    /,
) -> dict[str, object]:
    supports = _support_tuples(support_tuples)
    laws = _identifiers(law_ids, "law_ids", allow_empty=True)
    replay = _text(replay_id, "replay_id")
    _verify_arrays(result_manifest, arrays)
    return {
        "kind": "finance-result-archive",
        "result_manifest": _manifest_record(result_manifest),
        "array_collection_id": array_collection_digest(arrays),
        "replay_id": replay,
        "law_ids": list(laws),
        "support_tuples": [item.to_record() for item in supports],
    }


def archive_finance_result(
    path: str | Path,
    /,
    *,
    result_manifest: ResultManifest,
    arrays: Mapping[str, Any],
    replay_id: str,
    support_tuples: Sequence[SupportTuple],
    law_ids: Sequence[str] = (),
) -> Path:
    """Atomically archive one result without pickle, callbacks, or executable oracles."""
    if not isinstance(result_manifest, ResultManifest):
        raise TypeError("result_manifest must be a ResultManifest.")
    named = dict(_named_arrays(arrays))
    content = _archive_content(result_manifest, named, support_tuples, replay_id, law_ids)
    manifest = {**content, "archive_id": canonical_fingerprint(content)}
    return write_array_archive(path, manifest=manifest, arrays=named)


def reopen_finance_result(
    path: str | Path,
    /,
    *,
    limits: ArrayArchiveLimits | None = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> FinanceArchiveRecord:
    """Boundedly reopen and independently identity-check a finance result archive."""
    manifest, arrays = read_array_archive(path, limits=limits)
    inventory = manifest.pop("arrays", None)
    expected = {
        "kind",
        "result_manifest",
        "array_collection_id",
        "replay_id",
        "law_ids",
        "support_tuples",
        "archive_id",
    }
    if inventory is None or set(manifest) != expected:
        raise ValueError("Finance archive manifest fields are not canonical.")
    if manifest["kind"] != "finance-result-archive":
        raise ValueError("File is not a finance result archive.")
    result_record = manifest["result_manifest"]
    support_records = manifest["support_tuples"]
    laws_record = manifest["law_ids"]
    if not isinstance(result_record, Mapping):
        raise TypeError("Archived result_manifest must be a mapping.")
    if not isinstance(support_records, Sequence) or isinstance(support_records, str):
        raise TypeError("Archived support_tuples must be a sequence.")
    if not isinstance(laws_record, Sequence) or isinstance(laws_record, str):
        raise TypeError("Archived law_ids must be a sequence.")
    supports = []
    for item in support_records:
        if not isinstance(item, Mapping):
            raise TypeError("Archived support tuples must be mappings.")
        supports.append(SupportTuple.from_record(item))
    result_manifest = _manifest_from_record(result_record)
    content = {
        "kind": manifest["kind"],
        "result_manifest": manifest["result_manifest"],
        "array_collection_id": manifest["array_collection_id"],
        "replay_id": manifest["replay_id"],
        "law_ids": manifest["law_ids"],
        "support_tuples": manifest["support_tuples"],
    }
    if manifest["array_collection_id"] != array_collection_digest(arrays):
        raise ValueError("Finance archive array collection identity is invalid.")
    archive_id = canonical_fingerprint(content)
    if manifest["archive_id"] != archive_id:
        raise ValueError("Finance archive content address is invalid.")
    return FinanceArchiveRecord(
        path,
        result_manifest,
        arrays,
        supports,
        str(manifest["replay_id"]),
        tuple(str(item) for item in laws_record),
        archive_id,
    )


__all__ = [
    "FinanceArchiveRecord",
    "archive_finance_result",
    "finance_result_manifest",
    "reopen_finance_result",
]
