#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias

import equinox as eqx

from .._fingerprint import canonical_fingerprint, canonical_json
from .._strict import StrictModule
from .._trainable import NonTrainableState


JsonValue: TypeAlias = (
    bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None
)
MigrationTransform: TypeAlias = Callable[[Mapping[str, JsonValue]], Mapping[str, object]]

_HEX_DIGITS = frozenset("0123456789abcdef")
_MIGRATION_REQUEST_FIELDS = frozenset({"format_id", "record", "lineage"})
_MIGRATION_REPORT_FIELDS = frozenset(
    {
        "kind",
        "input_format_id",
        "output_format_id",
        "input_digest",
        "output_digest",
        "migration_ids",
        "lineage",
        "lossy",
        "input_record",
        "output_record",
        "report_id",
    }
)


class MigrationError(ValueError):
    """Base class for fail-closed compatibility migration refusal."""


class UnsupportedMigrationError(MigrationError):
    """Raised when no path reaches the current writer format."""


class AmbiguousMigrationError(MigrationError):
    """Raised when more than one shortest migration path is available."""


class CyclicMigrationError(MigrationError):
    """Raised when a migration graph contains a directed cycle."""


class LossyMigrationError(MigrationError):
    """Raised when a path is lossy and loss was not explicitly authorized."""


class MigrationPurityError(MigrationError):
    """Raised when a migration transform mutates its input record."""


def _identifier(value: object, name: str, /) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _digest(value: object, name: str, /) -> str:
    digest = _identifier(value, name)
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return digest


def _exact_fields(
    record: Mapping[str, object], expected: frozenset[str], label: str, /
) -> None:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    keys = set(record)
    if any(type(key) is not str for key in keys):
        raise TypeError(f"{label} field names must be strings.")
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError(f"{label} has {'; '.join(details)}.")


def _normalize_json(value: object, path: str = "$", /) -> JsonValue:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite JSON number.")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} contains a non-string object key.")
            normalized[key] = _normalize_json(item, f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains a value that is not canonical JSON data.")


def _object(value: object, name: str, /) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object mapping.")
    normalized = _normalize_json(value)
    if not isinstance(normalized, dict):
        raise TypeError(f"{name} must normalize to a JSON object.")
    return normalized


def _reject_nonfinite(value: str, /) -> None:
    raise ValueError(f"Non-finite JSON value {value!r} is not permitted.")


def _load_json_object(payload: str, label: str, /) -> Mapping[str, object]:
    if type(payload) is not str:
        raise TypeError(f"{label} JSON payload must be a string.")
    value = json.loads(payload, parse_constant=_reject_nonfinite)
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} JSON root must be an object.")
    return value


def _artifact_digest(format_id: str, record: Mapping[str, JsonValue], /) -> str:
    return canonical_fingerprint(
        {
            "kind": "migration-artifact",
            "format_id": format_id,
            "record": record,
        }
    )


def _lineage(values: Sequence[str], name: str = "lineage", /) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of artifact digests.")
    return tuple(_digest(value, f"{name} digest") for value in values)


class MigrationEdge(StrictModule, NonTrainableState):
    """One explicit, forward-only pure record transformation."""

    source_format_id: str = eqx.field(static=True)
    target_format_id: str = eqx.field(static=True)
    migration_id: str = eqx.field(static=True)
    transform: MigrationTransform = eqx.field(static=True, repr=False)
    lossy: bool = eqx.field(static=True)
    edge_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_format_id: str,
        target_format_id: str,
        transform: MigrationTransform,
        /,
        *,
        migration_id: str,
        lossy: bool = False,
    ):
        source = _identifier(source_format_id, "source-format ID")
        target = _identifier(target_format_id, "target-format ID")
        if source == target:
            raise ValueError("A migration edge must change the format identity.")
        if not callable(transform):
            raise TypeError("Migration transform must be callable.")
        if type(lossy) is not bool:
            raise TypeError("lossy must be a boolean.")
        self.source_format_id = source
        self.target_format_id = target
        self.migration_id = _identifier(migration_id, "migration ID")
        self.transform = transform
        self.lossy = lossy
        self.edge_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "migration-edge",
            "source_format_id": self.source_format_id,
            "target_format_id": self.target_format_id,
            "migration_id": self.migration_id,
            "lossy": self.lossy,
        }

    def to_record(self) -> dict[str, object]:
        """Return the stable edge identity, excluding executable code."""
        return {**self._content_record(), "edge_id": self.edge_id}

    def apply(self, record: Mapping[str, object], /) -> dict[str, JsonValue]:
        """Apply the transform without exposing the caller's record to mutation."""
        transform_input = _object(record, "Migration input record")
        before = canonical_json(transform_input)
        output = self.transform(transform_input)
        if canonical_json(transform_input) != before:
            raise MigrationPurityError(
                f"Migration {self.migration_id!r} mutated its input record."
            )
        if output is transform_input:
            raise MigrationPurityError(
                f"Migration {self.migration_id!r} returned its input in place."
            )
        return _object(output, "Migration output record")


class MigrationReport(StrictModule, NonTrainableState):
    """Content-verified migration result with complete artifact lineage."""

    input_format_id: str = eqx.field(static=True)
    output_format_id: str = eqx.field(static=True)
    input_digest: str = eqx.field(static=True)
    output_digest: str = eqx.field(static=True)
    migration_ids: tuple[str, ...] = eqx.field(static=True)
    lineage: tuple[str, ...] = eqx.field(static=True)
    lossy: bool = eqx.field(static=True)
    _input_json: str = eqx.field(static=True, repr=False)
    _output_json: str = eqx.field(static=True, repr=False)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        input_format_id: str,
        output_format_id: str,
        input_record: Mapping[str, object],
        output_record: Mapping[str, object],
        /,
        *,
        migration_ids: Sequence[str],
        lineage: Sequence[str],
        lossy: bool,
    ):
        input_format = _identifier(input_format_id, "input-format ID")
        output_format = _identifier(output_format_id, "output-format ID")
        input_value = _object(input_record, "Migration input record")
        output_value = _object(output_record, "Migration output record")
        migrations = tuple(_identifier(value, "migration ID") for value in migration_ids)
        if len(set(migrations)) != len(migrations):
            raise ValueError("Migration report cannot repeat migration IDs.")
        lineage_ = _lineage(lineage)
        if type(lossy) is not bool:
            raise TypeError("lossy must be a boolean.")
        input_digest = _artifact_digest(input_format, input_value)
        output_digest = _artifact_digest(output_format, output_value)
        if len(lineage_) < len(migrations) + 1:
            raise ValueError("Migration lineage is too short for the applied path.")
        if lineage_[-len(migrations) - 1] != input_digest:
            raise ValueError("Migration lineage does not select the input artifact.")
        if lineage_[-1] != output_digest:
            raise ValueError(
                "Migration lineage does not terminate at the output artifact."
            )
        if not migrations and (
            input_format != output_format or input_digest != output_digest
        ):
            raise ValueError("A changed artifact must cite at least one migration.")
        self.input_format_id = input_format
        self.output_format_id = output_format
        self.input_digest = input_digest
        self.output_digest = output_digest
        self.migration_ids = migrations
        self.lineage = lineage_
        self.lossy = lossy
        self._input_json = canonical_json(input_value)
        self._output_json = canonical_json(output_value)
        self.report_id = canonical_fingerprint(self._content_record())

    @property
    def input_record(self) -> dict[str, JsonValue]:
        """Return an independent copy of the selected parent record."""
        value = json.loads(self._input_json)
        if not isinstance(value, dict):
            raise RuntimeError("Stored migration input is not an object.")
        return value

    @property
    def output_record(self) -> dict[str, JsonValue]:
        """Return an independent copy of the migrated record."""
        value = json.loads(self._output_json)
        if not isinstance(value, dict):
            raise RuntimeError("Stored migration output is not an object.")
        return value

    @property
    def rollback_artifact_id(self) -> str:
        """Select the immutable parent artifact; never attempt reverse mutation."""
        return self.input_digest

    def select_parent(self) -> dict[str, object]:
        """Return the explicit parent artifact selected for rollback."""
        parent_lineage = (
            self.lineage
            if not self.migration_ids
            else self.lineage[: -len(self.migration_ids)]
        )
        return {
            "format_id": self.input_format_id,
            "record": self.input_record,
            "artifact_id": self.input_digest,
            "lineage": list(parent_lineage),
        }

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "migration-report",
            "input_format_id": self.input_format_id,
            "output_format_id": self.output_format_id,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "migration_ids": list(self.migration_ids),
            "lineage": list(self.lineage),
            "lossy": self.lossy,
        }

    def to_record(self) -> dict[str, object]:
        """Return the complete deterministic and reconstructable report."""
        return {
            **self._content_record(),
            "input_record": self.input_record,
            "output_record": self.output_record,
            "report_id": self.report_id,
        }

    def to_json(self) -> str:
        """Serialize this migration report to canonical JSON."""
        return canonical_json(self.to_record())

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> MigrationReport:
        """Strictly reconstruct and content-verify a migration report."""
        _exact_fields(record, _MIGRATION_REPORT_FIELDS, "Migration-report record")
        if record["kind"] != "migration-report":
            raise ValueError("Migration-report record has an unsupported kind.")
        input_record = record["input_record"]
        output_record = record["output_record"]
        migration_ids = record["migration_ids"]
        lineage = record["lineage"]
        if not isinstance(input_record, Mapping) or not isinstance(
            output_record, Mapping
        ):
            raise TypeError("Serialized migration records must be object mappings.")
        for name, values in (("migration_ids", migration_ids), ("lineage", lineage)):
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise TypeError(f"Serialized {name} must be a sequence.")
        if type(record["lossy"]) is not bool:
            raise TypeError("Serialized lossy must be a boolean.")
        value = cls(
            _identifier(record["input_format_id"], "input-format ID"),
            _identifier(record["output_format_id"], "output-format ID"),
            input_record,
            output_record,
            migration_ids=tuple(
                _identifier(item, "migration ID") for item in migration_ids
            ),
            lineage=_lineage(tuple(lineage)),
            lossy=record["lossy"],
        )
        if _digest(record["input_digest"], "input digest") != value.input_digest:
            raise ValueError("Migration report has an invalid input digest.")
        if _digest(record["output_digest"], "output digest") != value.output_digest:
            raise ValueError("Migration report has an invalid output digest.")
        if _digest(record["report_id"], "report ID") != value.report_id:
            raise ValueError("Migration report has an invalid content address.")
        return value

    @classmethod
    def from_json(cls, payload: str, /) -> MigrationReport:
        """Load strict canonical migration-report JSON."""
        return cls.from_record(_load_json_object(payload, "Migration-report"))


class CompatibilityRegistry(StrictModule, NonTrainableState):
    """Acyclic compatibility graph whose sole writer is the current format."""

    current_writer_id: str = eqx.field(static=True)
    edges: tuple[MigrationEdge, ...]
    registry_id: str = eqx.field(static=True)

    def __init__(
        self,
        current_writer_id: str,
        edges: Sequence[MigrationEdge],
        /,
    ):
        current = _identifier(current_writer_id, "current-writer ID")
        if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
            raise TypeError("edges must be a sequence of MigrationEdge values.")
        edges_ = tuple(edges)
        if any(not isinstance(edge, MigrationEdge) for edge in edges_):
            raise TypeError("edges must contain only MigrationEdge values.")
        edge_ids = tuple(edge.edge_id for edge in edges_)
        migration_ids = tuple(edge.migration_id for edge in edges_)
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("Compatibility registry contains duplicate edges.")
        if len(set(migration_ids)) != len(migration_ids):
            raise ValueError("Migration IDs must be globally unique within a registry.")
        if any(edge.source_format_id == current for edge in edges_):
            raise ValueError("The current writer format cannot have outgoing migrations.")
        normalized = tuple(sorted(edges_, key=lambda edge: edge.edge_id))
        self._require_acyclic(normalized)
        self.current_writer_id = current
        self.edges = normalized
        self.registry_id = canonical_fingerprint(self._content_record())

    @staticmethod
    def _require_acyclic(edges: Sequence[MigrationEdge], /) -> None:
        nodes = {
            format_id
            for edge in edges
            for format_id in (edge.source_format_id, edge.target_format_id)
        }
        indegree = {node: 0 for node in nodes}
        outgoing: dict[str, list[str]] = {node: [] for node in nodes}
        for edge in edges:
            outgoing[edge.source_format_id].append(edge.target_format_id)
            indegree[edge.target_format_id] += 1
        ready = deque(sorted(node for node, count in indegree.items() if count == 0))
        visited = 0
        while ready:
            node = ready.popleft()
            visited += 1
            for target in sorted(outgoing[node]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited != len(nodes):
            raise CyclicMigrationError("Compatibility migration graph must be acyclic.")

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "compatibility-registry",
            "current_writer_id": self.current_writer_id,
            "edges": [edge.to_record() for edge in self.edges],
        }

    def to_record(self) -> dict[str, object]:
        """Return the deterministic registry identity and edge manifest."""
        return {**self._content_record(), "registry_id": self.registry_id}

    def migration_path(self, source_format_id: str, /) -> tuple[MigrationEdge, ...]:
        """Select the unique shortest path from a source to the current writer."""
        source = _identifier(source_format_id, "source-format ID")
        if source == self.current_writer_id:
            return ()
        outgoing: dict[str, list[MigrationEdge]] = {}
        for edge in self.edges:
            outgoing.setdefault(edge.source_format_id, []).append(edge)
        queue: deque[tuple[str, tuple[MigrationEdge, ...]]] = deque([(source, ())])
        shortest_length: int | None = None
        candidates: list[tuple[MigrationEdge, ...]] = []
        while queue:
            node, path = queue.popleft()
            if shortest_length is not None and len(path) >= shortest_length:
                continue
            for edge in sorted(outgoing.get(node, ()), key=lambda value: value.edge_id):
                candidate = path + (edge,)
                if edge.target_format_id == self.current_writer_id:
                    shortest_length = len(candidate)
                    candidates.append(candidate)
                elif shortest_length is None or len(candidate) < shortest_length:
                    queue.append((edge.target_format_id, candidate))
        if not candidates:
            raise UnsupportedMigrationError(
                f"Format {source!r} cannot migrate to current writer "
                f"{self.current_writer_id!r}."
            )
        shortest = tuple(
            candidate for candidate in candidates if len(candidate) == shortest_length
        )
        if len(shortest) != 1:
            raise AmbiguousMigrationError(
                f"Format {source!r} has multiple shortest paths to current writer "
                f"{self.current_writer_id!r}."
            )
        return shortest[0]

    def resolve(
        self,
        record: Mapping[str, object],
        /,
        *,
        source_format_id: str,
        lineage: Sequence[str] = (),
        allow_lossy: bool = False,
    ) -> MigrationReport:
        """Resolve one immutable record to the current writer format."""
        if type(allow_lossy) is not bool:
            raise TypeError("allow_lossy must be a boolean.")
        source = _identifier(source_format_id, "source-format ID")
        input_record = _object(record, "Migration input record")
        input_digest = _artifact_digest(source, input_record)
        lineage_ = _lineage(lineage)
        if lineage_:
            if lineage_[-1] != input_digest:
                raise ValueError(
                    "Input lineage does not terminate at the input artifact."
                )
            resolved_lineage = list(lineage_)
        else:
            resolved_lineage = [input_digest]
        path = self.migration_path(source)
        if not allow_lossy and any(edge.lossy for edge in path):
            raise LossyMigrationError(
                "Selected migration path is lossy; explicit authorization is required."
            )
        current_record = input_record
        current_format = source
        for edge in path:
            if edge.source_format_id != current_format:
                raise RuntimeError("Registry returned a discontinuous migration path.")
            current_record = edge.apply(current_record)
            current_format = edge.target_format_id
            resolved_lineage.append(_artifact_digest(current_format, current_record))
        return MigrationReport(
            source,
            current_format,
            input_record,
            current_record,
            migration_ids=tuple(edge.migration_id for edge in path),
            lineage=tuple(resolved_lineage),
            lossy=any(edge.lossy for edge in path),
        )

    def load(self, payload: str, /, *, allow_lossy: bool = False) -> MigrationReport:
        """Load an explicit canonical request and resolve it to the current writer."""
        request = _load_json_object(payload, "Migration request")
        _exact_fields(request, _MIGRATION_REQUEST_FIELDS, "Migration request")
        record = request["record"]
        lineage = request["lineage"]
        if not isinstance(record, Mapping):
            raise TypeError("Migration request record must be an object mapping.")
        if not isinstance(lineage, Sequence) or isinstance(lineage, (str, bytes)):
            raise TypeError("Migration request lineage must be a sequence.")
        return self.resolve(
            record,
            source_format_id=_identifier(request["format_id"], "source-format ID"),
            lineage=_lineage(tuple(lineage)),
            allow_lossy=allow_lossy,
        )

    def rollback(self, report: MigrationReport, /) -> dict[str, object]:
        """Select a report's parent artifact without constructing a reverse edge."""
        if not isinstance(report, MigrationReport):
            raise TypeError("report must be a MigrationReport.")
        if report.output_format_id != self.current_writer_id:
            raise ValueError("Migration report was not resolved to this current writer.")
        path = self.migration_path(report.input_format_id)
        expected_ids = tuple(edge.migration_id for edge in path)
        if report.migration_ids != expected_ids:
            raise ValueError("Migration report path does not match this registry.")
        if report.lossy != any(edge.lossy for edge in path):
            raise ValueError("Migration report lossiness does not match its path.")
        return report.select_parent()


def resolve_migration(
    registry: CompatibilityRegistry,
    record: Mapping[str, object],
    /,
    *,
    source_format_id: str,
    lineage: Sequence[str] = (),
    allow_lossy: bool = False,
) -> MigrationReport:
    """Resolve a record through an explicit compatibility registry."""
    if not isinstance(registry, CompatibilityRegistry):
        raise TypeError("registry must be a CompatibilityRegistry.")
    return registry.resolve(
        record,
        source_format_id=source_format_id,
        lineage=lineage,
        allow_lossy=allow_lossy,
    )


def load_and_resolve_migration(
    registry: CompatibilityRegistry,
    payload: str,
    /,
    *,
    allow_lossy: bool = False,
) -> MigrationReport:
    """Load a strict canonical migration request and resolve it."""
    if not isinstance(registry, CompatibilityRegistry):
        raise TypeError("registry must be a CompatibilityRegistry.")
    return registry.load(payload, allow_lossy=allow_lossy)


__all__ = [
    "AmbiguousMigrationError",
    "CompatibilityRegistry",
    "CyclicMigrationError",
    "LossyMigrationError",
    "MigrationEdge",
    "MigrationError",
    "MigrationPurityError",
    "MigrationReport",
    "UnsupportedMigrationError",
    "load_and_resolve_migration",
    "resolve_migration",
]
