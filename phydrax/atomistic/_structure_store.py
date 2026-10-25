#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable atomistic structure sources and the canonical Zarr structure store.

A schema declares cardinality levels and typed fields. Records keep their
record identity, independent-unit lineage, per-structure stable particle
identities, conditions and exact label dtypes. The canonical in-memory and
stored representation is one columnar layout: rows of every level are
contiguous per structure, counted per parent row, and integer indices are
stored structure-local. Reads preserve requested order and duplicates and
rebase indices only when ``index_basis="batch-local"`` is requested.

A store directory holds immutable content-addressed Zarr v3 shards under
``shards/<shard_id>`` and one exclusively published canonical JSON commit per
revision under ``commits/<sequence>.json``. Shards are sealed before their
commit is linked, so a reader always resolves a complete prior or new
revision; a failed append leaves the previous head authoritative. Every
revision is admitted from its commit metadata alone (retained records, exact
shard entries, decoded chunk and record-index bytes, int64 cardinalities)
before any member is enumerated, decoded or published, and every read charges
the decoded chunks it touches separately from its output. Retraction and
compaction publish new lineage revisions. Storage is host-only NumPy and
the optional ``zarr`` dependency is imported only by store operations.
"""

from __future__ import annotations

import abc
import functools
import hashlib
import importlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, assert_never, final, Literal, NoReturn, Protocol

import equinox as eqx
import numpy as np
import numpy.typing as npt

from .._external_resource import read_bounded_resource, ResourceLimits, ResourceReadError
from .._fingerprint import canonical_fingerprint, canonical_json, iter_array_bytes
from .._host_io import open_directory_beneath, remote_location
from .._identity import NumericRevision, SemanticProvenance
from .._publication import publish_bytes, publish_resource_set
from .._resource_set import (
    read_bounded_resource_set,
    ResourceSetLimits,
    ResourceSetReadError,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..axes import Axis, AxisKey
from ..lifecycle import RevisionLineage
from ..typing import ConvertibleToArray, parse
from ..units import UnitDefinition


type AtomisticStoreLevelKind = Literal[
    "structure", "atom", "edge", "group", "segment", "product"
]
type AtomisticStoreFieldRole = Literal[
    "geometry", "identity", "condition", "label", "derived"
]
type AtomisticStoreValueKind = Literal[
    "continuous", "integer", "categorical", "mask", "index"
]
type AtomisticStoreExtensivity = Literal["intensive", "extensive", "not-applicable"]
type AtomisticStoreAvailability = Literal["required", "optional"]
type AtomisticStoreDType = Literal[
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
type AtomisticStoreIndexBasis = Literal["structure-local", "batch-local"]
type AtomisticStoreCompression = Literal["none", "zstd"]
type AtomisticStoreOperation = Literal["create", "append", "retract", "compact"]
type AtomisticStoreFailure = Literal[
    "malformed", "limit", "inconsistent", "conflict", "identity"
]
type _Extents = dict[str, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]
type _Runs = tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]

_NAME = re.compile(r"[a-z][a-z0-9_]{0,62}")
_RECORD_TEXT = re.compile(r"[A-Za-z0-9][\x21-\x7e]{0,255}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_COMMIT_NAME = re.compile(r"([0-9]{20})\.json")
_COMMIT_TEMPORARY = re.compile(r"\.[0-9]{20}\.json\.[0-9a-f]{32}\.tmp")
_MAXIMUM_TRAILING_RANK = 6
_MAXIMUM_ROW_ELEMENTS = 1 << 20
# Shard member paths are ``<array>/c/<row-chunk>/<trailing chunk>...``.
_MEMBER_DEPTH = 3 + _MAXIMUM_TRAILING_RANK
_STORE_PATH_DEPTH = 2 + _MEMBER_DEPTH
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_INT64_MAX = int(np.iinfo(np.int64).max)
# A float64 total of nonnegative int64 values below 2**62 proves the exact total
# fits int64: pairwise summation error is far below the remaining factor two.
_EXACT_SUM_GUARD = float(1 << 62)
_ZARR_ARRAY_KEYS = frozenset(
    {
        "zarr_format",
        "node_type",
        "shape",
        "data_type",
        "chunk_grid",
        "chunk_key_encoding",
        "fill_value",
        "codecs",
        "attributes",
        "storage_transformers",
        "dimension_names",
    }
)
_INDEX_ARRAY_PREFIXES = ("record_id.", "lineage_id.", "record_digest", "count.")


class AtomisticStoreError(ValueError):
    """Fail-closed structure-store error with a stable failure category.

    ``conflict`` reports a lost optimistic publication race or a stale parent;
    ``identity`` reports schema, record or revision identity refusal; ``limit``
    reports a configured resource bound; ``malformed`` and ``inconsistent``
    report invalid or changed persisted content.
    """

    reason: AtomisticStoreFailure

    def __init__(self, reason: AtomisticStoreFailure, message: str, /) -> None:
        self.reason = parse(reason, AtomisticStoreFailure, "reason")
        super().__init__(str(message))


def _name(value: str, label: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if _NAME.fullmatch(value) is None:
        raise ValueError(
            f"{label} must match [a-z][a-z0-9_]* with at most 63 characters."
        )
    return value


def _record_text(value: str, label: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if _RECORD_TEXT.fullmatch(value) is None:
        raise ValueError(
            f"{label} must be 1-256 printable ASCII characters without spaces, "
            "starting with a letter or digit."
        )
    return value


def _text(value: str, label: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if (
        not value
        or value.strip() != value
        or len(value) > 256
        or not value.isascii()
        or not value.isprintable()
    ):
        raise ValueError(f"{label} must be stripped printable ASCII of 1-256 characters.")
    return value


def _digest_text(value: str, label: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest.")
    return value


def _positive(value: int, label: str, /) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer.")
    if value <= 0:
        raise ValueError(f"{label} must be positive.")
    return value


def _nonnegative(value: int, label: str, /) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer.")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative.")
    return value


def _bound(value: int, label: str, /) -> int:
    bound = _positive(value, label)
    if bound > _INT64_MAX:
        raise ValueError(f"{label} must be representable as int64.")
    return bound


def _frozen[T: np.generic](array: npt.NDArray[T], /) -> npt.NDArray[T]:
    array.setflags(write=False)
    return array


def _compression(
    compression: AtomisticStoreCompression, level: int | None, /
) -> tuple[AtomisticStoreCompression, int | None]:
    selected = parse(compression, AtomisticStoreCompression, "compression")
    match selected:
        case "none":
            if level is not None:
                raise ValueError("Uncompressed shards do not take a compression level.")
            return selected, None
        case "zstd":
            if type(level) is not int or not 1 <= level <= 22:
                raise ValueError("Zstandard compression requires an integer level 1-22.")
            return selected, level
        case _:
            assert_never(selected)


# --------------------------------------------------------------------- schema


def _level_content(
    name: str,
    kind: AtomisticStoreLevelKind,
    parent: str | None,
    factors: tuple[str, ...],
    maximum_rows_per_structure: int | None,
    derivation_id: str | None,
    /,
) -> dict[str, Any]:
    return {
        "name": name,
        "level_kind": kind,
        "parent": parent,
        "factors": list(factors),
        "maximum_rows_per_structure": maximum_rows_per_structure,
        "derivation_id": derivation_id,
    }


@final
class AtomisticStoreLevel(StrictModule, NonTrainableState):
    """One cardinality level of an atomistic structure store.

    The ``structure`` level has one row per record. ``atom``, ``edge`` and
    ``group`` levels are children of the structure level; a ``segment`` level
    names any earlier non-product parent level. Rows of a child level are stored
    contiguously by parent row and counted per parent row, so every structure
    owns one contiguous row range of every level. A ``product`` level has the
    row-major Cartesian product of its factor levels' per-structure rows (for
    example atom x atom Hessian blocks) under a required per-structure bound.

    ``derivation_id`` names the exact geometry, cutoff, image and selection
    derivation of a cached ``edge`` level; ``None`` declares source-authoritative
    edges such as bonds. A cached level is valid only for a consumer whose
    derivation matches exactly, never because its extents happen to fit.
    """

    name: str = eqx.field(static=True)
    kind: AtomisticStoreLevelKind = eqx.field(static=True)
    parent: str | None = eqx.field(static=True)
    factors: tuple[str, ...] = eqx.field(static=True)
    maximum_rows_per_structure: int | None = eqx.field(static=True)
    derivation_id: str | None = eqx.field(static=True)
    level_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        kind: AtomisticStoreLevelKind,
        /,
        *,
        parent: str | None = None,
        factors: Sequence[str] = (),
        maximum_rows_per_structure: int | None = None,
        derivation_id: str | None = None,
    ) -> None:
        name_ = _name(name, "level name")
        kind_ = parse(kind, AtomisticStoreLevelKind, "kind")
        if isinstance(factors, str):
            raise TypeError("factors must be a sequence of level names.")
        factors_ = tuple(_name(factor, "product factor") for factor in factors)
        derivation = (
            None if derivation_id is None else _text(derivation_id, "derivation_id")
        )
        parent_: str | None
        bound: int | None
        match kind_:
            case "structure":
                if (
                    parent is not None
                    or factors_
                    or maximum_rows_per_structure is not None
                    or derivation is not None
                ):
                    raise ValueError(
                        "The structure level has no parent, factors, row bound or derivation."
                    )
                parent_, bound = None, None
            case "product":
                if parent is not None or derivation is not None:
                    raise ValueError(
                        "Product levels belong to structures and have no derivation."
                    )
                if len(factors_) < 2:
                    raise ValueError("Product levels require at least two factor levels.")
                if maximum_rows_per_structure is None:
                    raise ValueError("Product levels require maximum_rows_per_structure.")
                parent_ = None
                bound = _bound(maximum_rows_per_structure, "maximum_rows_per_structure")
            case "atom" | "edge" | "group" | "segment":
                if parent is None:
                    raise ValueError(f"{kind_!r} levels require a parent level.")
                if factors_ or maximum_rows_per_structure is not None:
                    raise ValueError("Only product levels take factors and a row bound.")
                if derivation is not None and kind_ != "edge":
                    raise ValueError("Only edge levels carry a derivation identity.")
                parent_, bound = _name(parent, "parent level"), None
            case _:
                assert_never(kind_)
        self.name = name_
        self.kind = kind_
        self.parent = parent_
        self.factors = factors_
        self.maximum_rows_per_structure = bound
        self.derivation_id = derivation
        self.level_id = canonical_fingerprint(
            {
                "kind": "atomistic-store-level",
                **_level_content(name_, kind_, parent_, factors_, bound, derivation),
            }
        )


@final
class AtomisticStoreIndex(StrictModule, NonTrainableState):
    """Structure-local row-index semantics of an integer field.

    Stored values address rows of ``target`` within the same structure. Reads
    keep that canonical basis or rebase explicitly to batch-local rows. A
    negative ``sentinel`` marks an absent reference and is never rebased.
    """

    target: str = eqx.field(static=True)
    sentinel: int | None = eqx.field(static=True)

    def __init__(self, target: str, /, *, sentinel: int | None = None) -> None:
        target_ = _name(target, "index target level")
        if sentinel is not None:
            if type(sentinel) is not int:
                raise TypeError("Index sentinel must be an integer or None.")
            if sentinel >= 0:
                raise ValueError(
                    "Index sentinel must be negative so it never names a row."
                )
        self.target = target_
        self.sentinel = sentinel


def _axis_payload(axis: Axis, /) -> dict[str, Any]:
    return {
        "scope": axis.key.scope,
        "name": axis.key.name,
        "size": axis.size,
        "labels": list(axis.labels),
        "support_id": axis.support_id,
    }


def _field_content(
    name: str,
    level: str,
    dtype: AtomisticStoreDType,
    axes: tuple[Axis, ...],
    unit: UnitDefinition | None,
    convention: str,
    role: AtomisticStoreFieldRole,
    kind: AtomisticStoreValueKind,
    extensivity: AtomisticStoreExtensivity,
    availability: AtomisticStoreAvailability,
    index: AtomisticStoreIndex | None,
    /,
) -> dict[str, Any]:
    return {
        "name": name,
        "level": level,
        "dtype": dtype,
        "axes": [_axis_payload(axis) for axis in axes],
        "unit": None if unit is None else unit.to_dict(),
        "convention": convention,
        "role": role,
        "value_kind": kind,
        "extensivity": extensivity,
        "availability": availability,
        "index": (
            None
            if index is None
            else {"target": index.target, "sentinel": index.sentinel}
        ),
    }


@final
class AtomisticStoreField(StrictModule, NonTrainableState):
    """Typed semantic declaration of one stored per-row quantity.

    ``kind`` states the value meaning: ``continuous`` floating/complex physical
    values carry a unit and an extensivity; ``integer`` exact integer
    quantities such as image shifts; ``categorical`` codes and ``mask`` flags
    have no unit; ``index`` fields carry `AtomisticStoreIndex` semantics. A
    field is never an index merely because it is integral. ``convention`` names
    the component, sign or coding convention. ``optional`` fields store a
    per-row availability mask and canonical zero values in unavailable rows.
    Values are stored with the exact declared dtype; no cast is applied.
    """

    name: str = eqx.field(static=True)
    level: str = eqx.field(static=True)
    dtype: AtomisticStoreDType = eqx.field(static=True)
    axes: tuple[Axis, ...] = eqx.field(static=True)
    unit: UnitDefinition | None
    convention: str = eqx.field(static=True)
    role: AtomisticStoreFieldRole = eqx.field(static=True)
    kind: AtomisticStoreValueKind = eqx.field(static=True)
    extensivity: AtomisticStoreExtensivity = eqx.field(static=True)
    availability: AtomisticStoreAvailability = eqx.field(static=True)
    index: AtomisticStoreIndex | None
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        level: str,
        dtype: AtomisticStoreDType,
        /,
        *,
        kind: AtomisticStoreValueKind,
        role: AtomisticStoreFieldRole,
        convention: str,
        axes: Sequence[Axis] = (),
        unit: UnitDefinition | None = None,
        extensivity: AtomisticStoreExtensivity = "not-applicable",
        availability: AtomisticStoreAvailability = "required",
        index: AtomisticStoreIndex | None = None,
    ) -> None:
        name_ = _name(name, "field name")
        level_ = _name(level, "field level")
        dtype_ = parse(dtype, AtomisticStoreDType, "dtype")
        kind_ = parse(kind, AtomisticStoreValueKind, "kind")
        role_ = parse(role, AtomisticStoreFieldRole, "role")
        convention_ = _text(convention, "convention")
        extensivity_ = parse(extensivity, AtomisticStoreExtensivity, "extensivity")
        availability_ = parse(availability, AtomisticStoreAvailability, "availability")
        if isinstance(axes, Axis):
            raise TypeError("axes must be a sequence of Axis values.")
        axes_ = tuple(axes)
        if any(not isinstance(axis, Axis) for axis in axes_):
            raise TypeError("Field axes must be phydrax.axes.Axis values.")
        if len(axes_) > _MAXIMUM_TRAILING_RANK:
            raise ValueError(
                f"Fields admit at most {_MAXIMUM_TRAILING_RANK} trailing axes."
            )
        if any(type(axis.size) is not int or axis.size < 1 for axis in axes_):
            raise ValueError("Field axes must have positive integer sizes.")
        for axis in axes_:
            _text(axis.key.scope, "axis scope")
            _text(axis.key.name, "axis name")
            for label in axis.labels:
                _text(label, "axis label")
            if axis.support_id is not None:
                _text(axis.support_id, "axis support")
        if math.prod(axis.size for axis in axes_) > _MAXIMUM_ROW_ELEMENTS:
            raise ValueError("A field row exceeds the admitted element count.")
        if unit is not None and not isinstance(unit, UnitDefinition):
            raise TypeError("unit must be a UnitDefinition or None.")
        if index is not None and not isinstance(index, AtomisticStoreIndex):
            raise TypeError("index must be an AtomisticStoreIndex or None.")
        numeric = np.dtype(dtype_)
        match kind_:
            case "continuous":
                if numeric.kind not in "fc":
                    raise ValueError(
                        "Continuous fields require a floating or complex dtype."
                    )
                if unit is None:
                    raise ValueError("Continuous fields require an explicit unit.")
                if extensivity_ == "not-applicable":
                    raise ValueError("Continuous fields are intensive or extensive.")
            case "integer":
                if numeric.kind not in "iu":
                    raise ValueError("Integer fields require an integer dtype.")
            case "categorical" | "mask" | "index":
                if unit is not None or extensivity_ != "not-applicable":
                    raise ValueError(f"{kind_!r} fields have no unit and no extensivity.")
                if kind_ == "categorical" and numeric.kind not in "iub":
                    raise ValueError(
                        "Categorical fields require an integer or bool dtype."
                    )
                if kind_ == "mask" and numeric.kind != "b":
                    raise ValueError("Mask fields require the bool dtype.")
                if kind_ == "index" and numeric not in (
                    np.dtype(np.int32),
                    np.dtype(np.int64),
                ):
                    raise ValueError("Index fields require int32 or int64.")
            case _:
                assert_never(kind_)
        if (kind_ == "index") != (index is not None):
            raise ValueError("Exactly the index-kind fields carry index semantics.")
        if (
            index is not None
            and index.sentinel is not None
            and index.sentinel < np.iinfo(numeric).min
        ):
            raise ValueError("The index sentinel is outside the field dtype range.")
        self.name = name_
        self.level = level_
        self.dtype = dtype_
        self.axes = axes_
        self.unit = unit
        self.convention = convention_
        self.role = role_
        self.kind = kind_
        self.extensivity = extensivity_
        self.availability = availability_
        self.index = index
        self.field_id = canonical_fingerprint(
            {
                "kind": "atomistic-store-field",
                **_field_content(
                    name_,
                    level_,
                    dtype_,
                    axes_,
                    unit,
                    convention_,
                    role_,
                    kind_,
                    extensivity_,
                    availability_,
                    index,
                ),
            }
        )

    @property
    def numpy_dtype(self) -> np.dtype[Any]:
        return np.dtype(self.dtype)

    @property
    def trailing_shape(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.axes)

    @property
    def row_bytes(self) -> int:
        return self.numpy_dtype.itemsize * math.prod(self.trailing_shape)


@final
class AtomisticStructureStoreSchema(StrictModule, NonTrainableState):
    """Complete level and field declaration of a structure source or store.

    Levels are declared parents-first, starting with the single ``structure``
    level; exactly one ``atom`` level carries the per-structure stable particle
    identities. Field order is the canonical field order of reads and digests.
    """

    levels: tuple[AtomisticStoreLevel, ...]
    fields: tuple[AtomisticStoreField, ...]
    level_names: tuple[str, ...] = eqx.field(static=True)
    field_names: tuple[str, ...] = eqx.field(static=True)
    counted_levels: tuple[str, ...] = eqx.field(static=True)
    structure_level: str = eqx.field(static=True)
    atom_level: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        levels: Sequence[AtomisticStoreLevel],
        fields: Sequence[AtomisticStoreField],
        /,
    ) -> None:
        levels_ = tuple(levels)
        fields_ = tuple(fields)
        if any(not isinstance(level, AtomisticStoreLevel) for level in levels_):
            raise TypeError("Schema levels must be AtomisticStoreLevel values.")
        if any(not isinstance(field, AtomisticStoreField) for field in fields_):
            raise TypeError("Schema fields must be AtomisticStoreField values.")
        if not levels_ or levels_[0].kind != "structure":
            raise ValueError("The first schema level must be the structure level.")
        kinds = [level.kind for level in levels_]
        if kinds.count("structure") != 1 or kinds.count("atom") != 1:
            raise ValueError("A schema has exactly one structure and one atom level.")
        declared: dict[str, AtomisticStoreLevel] = {}
        structure = levels_[0].name
        for level in levels_:
            if level.name in declared:
                raise ValueError(f"Level {level.name!r} is declared twice.")
            references = level.factors if level.parent is None else (level.parent,)
            for reference in references:
                earlier = declared.get(reference)
                if earlier is None:
                    raise ValueError(
                        f"Level {level.name!r} references undeclared level {reference!r}."
                    )
                if earlier.kind == "product":
                    raise ValueError(
                        "Product levels cannot parent or factor other levels."
                    )
            if level.kind in ("atom", "edge", "group") and level.parent != structure:
                raise ValueError(
                    f"{level.kind!r} levels are direct children of the structure level."
                )
            declared[level.name] = level
        field_names: set[str] = set()
        for field in fields_:
            if field.name in field_names:
                raise ValueError(f"Field {field.name!r} is declared twice.")
            field_names.add(field.name)
            if field.level not in declared:
                raise ValueError(f"Field {field.name!r} names an undeclared level.")
            if field.index is not None and field.index.target not in declared:
                raise ValueError(
                    f"Index field {field.name!r} targets an undeclared level."
                )
        self.levels = levels_
        self.fields = fields_
        self.level_names = tuple(level.name for level in levels_)
        self.field_names = tuple(field.name for field in fields_)
        self.counted_levels = tuple(
            level.name for level in levels_[1:] if level.kind != "product"
        )
        self.structure_level = structure
        self.atom_level = next(level.name for level in levels_ if level.kind == "atom")
        self.schema_id = canonical_fingerprint(
            {
                "kind": "atomistic-structure-store-schema",
                "levels": [level.level_id for level in levels_],
                "fields": [field.field_id for field in fields_],
            }
        )

    def level(self, name: str, /) -> AtomisticStoreLevel:
        if name not in self.level_names:
            raise ValueError(f"Unknown store level {name!r}.")
        return self.levels[self.level_names.index(name)]

    def field(self, name: str, /) -> AtomisticStoreField:
        if name not in self.field_names:
            raise ValueError(f"Unknown store field {name!r}.")
        return self.fields[self.field_names.index(name)]


def _parent(schema: AtomisticStructureStoreSchema, level: str, /) -> str:
    parent = schema.level(level).parent
    if parent is None:
        raise RuntimeError(f"Counted level {level!r} has no parent level.")
    return parent


def _schema_payload(schema: AtomisticStructureStoreSchema, /) -> dict[str, Any]:
    return {
        "levels": [
            {
                **_level_content(
                    level.name,
                    level.kind,
                    level.parent,
                    level.factors,
                    level.maximum_rows_per_structure,
                    level.derivation_id,
                ),
                "level_id": level.level_id,
            }
            for level in schema.levels
        ],
        "fields": [
            {
                **_field_content(
                    field.name,
                    field.level,
                    field.dtype,
                    field.axes,
                    field.unit,
                    field.convention,
                    field.role,
                    field.kind,
                    field.extensivity,
                    field.availability,
                    field.index,
                ),
                "field_id": field.field_id,
            }
            for field in schema.fields
        ],
        "schema_id": schema.schema_id,
    }


def _require_keys(payload: object, keys: Iterable[str], label: str, /) -> dict[str, Any]:
    expected = set(keys)
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}.")
    return payload


def _list(value: object, label: str, /) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return value


def _schema_from_payload(payload: object, /) -> AtomisticStructureStoreSchema:
    record = _require_keys(payload, ("levels", "fields", "schema_id"), "schema")
    levels = []
    for item in _list(record["levels"], "schema levels"):
        level = _require_keys(
            item,
            (*_level_content("a", "structure", None, (), None, None), "level_id"),
            "level",
        )
        restored = AtomisticStoreLevel(
            level["name"],
            level["level_kind"],
            parent=level["parent"],
            factors=tuple(_list(level["factors"], "level factors")),
            maximum_rows_per_structure=level["maximum_rows_per_structure"],
            derivation_id=level["derivation_id"],
        )
        if restored.level_id != level["level_id"]:
            raise ValueError("A persisted level identity does not match its content.")
        levels.append(restored)
    fields = []
    for item in _list(record["fields"], "schema fields"):
        field = _require_keys(
            item,
            (
                "name",
                "level",
                "dtype",
                "axes",
                "unit",
                "convention",
                "role",
                "value_kind",
                "extensivity",
                "availability",
                "index",
                "field_id",
            ),
            "field",
        )
        axes = []
        for axis in _list(field["axes"], "field axes"):
            axis_ = _require_keys(
                axis, ("scope", "name", "size", "labels", "support_id"), "axis"
            )
            axes.append(
                Axis(
                    AxisKey(
                        _text(axis_["scope"], "axis scope"),
                        _text(axis_["name"], "axis name"),
                    ),
                    _positive(axis_["size"], "axis size"),
                    tuple(
                        _text(label, "axis label")
                        for label in _list(axis_["labels"], "axis labels")
                    ),
                    None
                    if axis_["support_id"] is None
                    else _text(axis_["support_id"], "axis support"),
                )
            )
        index_payload = field["index"]
        index = None
        if index_payload is not None:
            index_ = _require_keys(index_payload, ("target", "sentinel"), "field index")
            index = AtomisticStoreIndex(index_["target"], sentinel=index_["sentinel"])
        restored_field = AtomisticStoreField(
            field["name"],
            field["level"],
            field["dtype"],
            kind=field["value_kind"],
            role=field["role"],
            convention=field["convention"],
            axes=tuple(axes),
            unit=None
            if field["unit"] is None
            else UnitDefinition.from_dict(field["unit"]),
            extensivity=field["extensivity"],
            availability=field["availability"],
            index=index,
        )
        if restored_field.field_id != field["field_id"]:
            raise ValueError("A persisted field identity does not match its content.")
        fields.append(restored_field)
    schema = AtomisticStructureStoreSchema(tuple(levels), tuple(fields))
    if schema.schema_id != record["schema_id"]:
        raise ValueError("The persisted schema identity does not match its content.")
    return schema


# ------------------------------------------------------------ columnar records


@dataclass(frozen=True, slots=True)
class _Columns:
    """Canonical concatenated records; every array is read-only."""

    record_ids: tuple[str, ...]
    lineage_ids: tuple[str, ...]
    digests: tuple[str, ...]
    counts: Mapping[str, npt.NDArray[np.int64]]
    stable_ids: npt.NDArray[np.int64]
    values: Mapping[str, npt.NDArray[Any]]
    available: Mapping[str, npt.NDArray[np.bool_]]


def _host_integers(value: ConvertibleToArray, label: str, /) -> npt.NDArray[np.int64]:
    array = np.asarray(value)
    if array.size == 0 and array.dtype.kind in "iuf":
        return np.zeros(array.shape, dtype=np.int64)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{label} must hold integers.")
    if (
        array.dtype.kind == "u"
        and array.dtype.itemsize == 8
        and np.any(array > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{label} exceeds the int64 range.")
    return array.astype(np.int64)


def _exact_array(
    value: ConvertibleToArray, dtype: np.dtype[Any], shape: tuple[int, ...], label: str, /
) -> npt.NDArray[Any]:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise TypeError(
            f"{label} must have dtype {dtype.name} exactly, not {array.dtype}."
        )
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, not {array.shape}.")
    return _frozen(np.array(array, copy=True, order="C"))


def _ranges(
    starts: npt.NDArray[np.int64], counts: npt.NDArray[np.int64], /
) -> npt.NDArray[np.int64]:
    """Concatenate ``arange(start, start + count)`` for every range, in order."""

    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    ends = np.cumsum(counts)
    return np.repeat(starts - (ends - counts), counts) + np.arange(total, dtype=np.int64)


def _offsets(counts: npt.NDArray[np.int64], level: str, /) -> npt.NDArray[np.int64]:
    """Return ``[0, *cumsum(counts)]`` of nonnegative counts, refusing int64 wrap."""

    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )
    # The first wrapped prefix sum of nonnegative int64 values is negative.
    if np.any(cumulative < 0):
        raise ValueError(f"Row counts of level {level!r} overflow int64 row offsets.")
    return cumulative


def _extents(
    schema: AtomisticStructureStoreSchema,
    record_count: int,
    counts: Mapping[str, npt.NDArray[np.int64]],
    /,
) -> _Extents:
    """Return per-structure ``(row start, row count)`` of every level."""

    extents: _Extents = {
        schema.structure_level: (
            np.arange(record_count, dtype=np.int64),
            np.ones(record_count, dtype=np.int64),
        )
    }
    for level in schema.levels[1:]:
        if level.kind == "product":
            bound = level.maximum_rows_per_structure
            if bound is None:
                raise RuntimeError("Product levels always carry a row bound.")
            count = np.ones(record_count, dtype=np.int64)
            for factor in level.factors:
                factor_count = extents[factor][1]
                if np.any(
                    (factor_count > 0) & (count > bound // np.maximum(factor_count, 1))
                ):
                    raise ValueError(
                        f"Product level {level.name!r} exceeds its {bound}-row bound "
                        "per structure."
                    )
                count = count * factor_count
            start = _offsets(count, level.name)[:-1]
        else:
            parent_start, parent_count = extents[_parent(schema, level.name)]
            level_counts = counts[level.name]
            parent_rows = int(parent_count.sum())
            if level_counts.shape != (parent_rows,):
                raise ValueError(
                    f"Counts of level {level.name!r} need one entry per parent row "
                    f"({parent_rows}), not shape {level_counts.shape}."
                )
            if np.any(level_counts < 0):
                raise ValueError(f"Counts of level {level.name!r} must be nonnegative.")
            cumulative = _offsets(level_counts, level.name)
            start = cumulative[parent_start]
            count = cumulative[parent_start + parent_count] - start
        extents[level.name] = (start, count)
    return extents


def _broadcast_rows(
    values: npt.NDArray[Any], rows: npt.NDArray[Any], /
) -> npt.NDArray[Any]:
    return rows.reshape(rows.shape + (1,) * (values.ndim - 1))


def _validate_index_values(
    field: AtomisticStoreField,
    values: npt.NDArray[Any],
    available: npt.NDArray[np.bool_] | None,
    extents: _Extents,
    basis: AtomisticStoreIndexBasis,
    /,
) -> None:
    index = field.index
    if index is None:
        raise RuntimeError("Index validation requires index semantics.")
    target_start, target_count = extents[index.target]
    level_count = extents[field.level][1]
    owner = np.repeat(np.arange(level_count.size, dtype=np.int64), level_count)
    match basis:
        case "structure-local":
            lower = np.zeros(owner.size, dtype=np.int64)
        case "batch-local":
            lower = target_start[owner]
        case _:
            assert_never(basis)
    upper = lower + target_count[owner]
    valid = (values >= _broadcast_rows(values, lower)) & (
        values < _broadcast_rows(values, upper)
    )
    if index.sentinel is not None:
        valid |= values == index.sentinel
    if available is not None:
        valid |= ~_broadcast_rows(values, available)
    if not np.all(valid):
        raise ValueError(
            f"Index field {field.name!r} addresses rows outside its {basis} "
            f"{index.target!r} range."
        )


class _Digest(Protocol):
    def update(self, data: bytes | memoryview, /) -> None: ...


def _frame(digest: _Digest, data: bytes, /) -> None:
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def _frame_array(digest: _Digest, path: str, value: npt.NDArray[Any], /) -> None:
    _frame(
        digest,
        canonical_json(
            {"path": path, "dtype": value.dtype.str, "shape": list(value.shape)}
        ).encode("ascii"),
    )
    digest.update(value.nbytes.to_bytes(8, "big"))
    for chunk in iter_array_bytes(value):
        digest.update(chunk)


def _record_digests(
    schema: AtomisticStructureStoreSchema, columns: _Columns, extents: _Extents, /
) -> tuple[str, ...]:
    """Content digest of every complete record, independent of storage layout."""

    if tuple(columns.values) != schema.field_names:
        raise RuntimeError("Record digests require every schema field.")
    digests: list[str] = []
    for index, (record_id, lineage_id) in enumerate(
        zip(columns.record_ids, columns.lineage_ids, strict=True)
    ):
        digest = hashlib.sha256()
        _frame(
            digest,
            canonical_json(
                {
                    "kind": "atomistic-structure-record",
                    "schema_id": schema.schema_id,
                    "record_id": record_id,
                    "lineage_id": lineage_id,
                }
            ).encode("ascii"),
        )
        for level in schema.counted_levels:
            start, count = extents[_parent(schema, level)]
            _frame_array(
                digest,
                f"count.{level}",
                columns.counts[level][start[index] : start[index] + count[index]],
            )
        start, count = extents[schema.atom_level]
        _frame_array(
            digest,
            "stable_id",
            columns.stable_ids[start[index] : start[index] + count[index]],
        )
        for field in schema.fields:
            start, count = extents[field.level]
            rows = slice(start[index], start[index] + count[index])
            _frame_array(digest, f"value.{field.name}", columns.values[field.name][rows])
            if field.availability == "optional":
                _frame_array(
                    digest, f"available.{field.name}", columns.available[field.name][rows]
                )
        digests.append(digest.hexdigest())
    return tuple(digests)


def _content_id(
    schema_id: str, record_ids: Iterable[str], digests: Iterable[str], /
) -> str:
    """Identity of one ordered record sequence, independent of storage layout."""

    digest = hashlib.sha256()
    _frame(
        digest,
        canonical_json(
            {"kind": "atomistic-structure-content", "schema_id": schema_id}
        ).encode("ascii"),
    )
    for record_id, record_digest in zip(record_ids, digests, strict=True):
        _frame(digest, record_id.encode("ascii"))
        digest.update(bytes.fromhex(record_digest))
    return digest.hexdigest()


def _projection(
    schema: AtomisticStructureStoreSchema, fields: Sequence[str] | None, /
) -> tuple[str, ...]:
    if fields is None:
        return schema.field_names
    if isinstance(fields, str):
        raise TypeError("fields must be a sequence of field names or None.")
    names = tuple(fields)
    for name in names:
        schema.field(name)
    if len(set(names)) != len(names):
        raise ValueError("A field projection cannot repeat a field.")
    return names


def _validated_columns(
    schema: AtomisticStructureStoreSchema,
    record_ids: Sequence[str],
    lineage_ids: Sequence[str],
    record_digests: Sequence[str] | None,
    counts: Mapping[str, ConvertibleToArray],
    stable_ids: ConvertibleToArray,
    values: Mapping[str, ConvertibleToArray],
    available: Mapping[str, ConvertibleToArray] | None,
    fields: tuple[str, ...],
    basis: AtomisticStoreIndexBasis,
    /,
    *,
    absent_optional: bool,
) -> tuple[_Columns, _Extents]:
    """Validate and freeze one columnar record set before it is exposed."""

    if isinstance(record_ids, str) or isinstance(lineage_ids, str):
        raise TypeError("Record and lineage identifiers must be sequences of strings.")
    ids = tuple(_record_text(value, "record_id") for value in record_ids)
    lineages = tuple(_record_text(value, "lineage_id") for value in lineage_ids)
    if not ids or len(lineages) != len(ids):
        raise ValueError("Records require one lineage identifier per record identifier.")
    if not isinstance(counts, Mapping) or set(counts) != set(schema.counted_levels):
        raise ValueError(
            f"Counts must name exactly the counted levels {list(schema.counted_levels)}."
        )
    counts_ = {
        level: _frozen(_host_integers(counts[level], f"counts[{level!r}]"))
        for level in schema.counted_levels
    }
    if any(value.ndim != 1 for value in counts_.values()):
        raise ValueError("Level counts are one-dimensional per parent row.")
    extents = _extents(schema, len(ids), counts_)
    atom_start, atom_count = extents[schema.atom_level]
    atom_rows = int(atom_count.sum())
    stable = _frozen(_host_integers(stable_ids, "stable_ids"))
    if stable.shape != (atom_rows,):
        raise ValueError(f"stable_ids must have shape ({atom_rows},).")
    owner = np.repeat(np.arange(len(ids), dtype=np.int64), atom_count)
    order = np.lexsort((stable, owner))
    if np.any(
        (stable[order][1:] == stable[order][:-1])
        & (owner[order][1:] == owner[order][:-1])
    ):
        raise ValueError("Stable particle identities must be unique within a structure.")
    if not isinstance(values, Mapping):
        raise TypeError("values must map field names to arrays.")
    projected = set(fields)
    if not set(values) <= projected:
        raise ValueError(f"Unexpected fields {sorted(set(values) - projected)}.")
    available_ = {} if available is None else available
    if not isinstance(available_, Mapping):
        raise TypeError("available must map optional field names to masks.")
    optional = {name for name in fields if schema.field(name).availability == "optional"}
    if not set(available_) <= optional:
        raise ValueError("Availability masks are declared only for optional fields.")
    values_: dict[str, npt.NDArray[Any]] = {}
    masks: dict[str, npt.NDArray[np.bool_]] = {}
    for name in fields:
        field = schema.field(name)
        rows = int(extents[field.level][1].sum())
        shape = (rows, *field.trailing_shape)
        if name not in values:
            if field.availability == "required" or not absent_optional:
                raise ValueError(f"Field {name!r} is missing.")
            if name in available_:
                raise ValueError(f"Absent field {name!r} cannot declare availability.")
            values_[name] = _frozen(np.zeros(shape, dtype=field.numpy_dtype))
            masks[name] = _frozen(np.zeros(rows, dtype=np.bool_))
            continue
        array = _exact_array(values[name], field.numpy_dtype, shape, f"values[{name!r}]")
        values_[name] = array
        if field.availability == "required":
            continue
        if name in available_:
            mask = _exact_array(
                available_[name], np.dtype(np.bool_), (rows,), f"available[{name!r}]"
            )
        elif absent_optional:
            mask = _frozen(np.ones(rows, dtype=np.bool_))
        else:
            raise ValueError(f"Optional field {name!r} requires its availability mask.")
        if np.any(array[~mask] != 0):
            raise ValueError(
                f"Unavailable rows of {name!r} must hold canonical zero values."
            )
        masks[name] = mask
    for name in fields:
        field = schema.field(name)
        if field.kind == "index":
            _validate_index_values(field, values_[name], masks.get(name), extents, basis)
    columns = _Columns(
        ids,
        lineages,
        (),
        MappingProxyType(counts_),
        stable,
        MappingProxyType(values_),
        MappingProxyType(masks),
    )
    if record_digests is None:
        digests = _record_digests(schema, columns, extents)
    else:
        if isinstance(record_digests, str):
            raise TypeError("record_digests must be a sequence of digests.")
        digests = tuple(_digest_text(value, "record digest") for value in record_digests)
        if len(digests) != len(ids):
            raise ValueError("Records require one digest per record identifier.")
    return (
        _Columns(
            ids,
            lineages,
            digests,
            columns.counts,
            stable,
            columns.values,
            columns.available,
        ),
        extents,
    )


def _take(
    schema: AtomisticStructureStoreSchema,
    columns: _Columns,
    positions: npt.NDArray[np.int64],
    fields: tuple[str, ...],
    /,
) -> _Columns:
    """Gather records in request order, preserving duplicates."""

    extents = _extents(schema, len(columns.record_ids), columns.counts)
    counts = {}
    for level in schema.counted_levels:
        start, count = extents[_parent(schema, level)]
        counts[level] = columns.counts[level][_ranges(start[positions], count[positions])]
    start, count = extents[schema.atom_level]
    stable = columns.stable_ids[_ranges(start[positions], count[positions])]
    values = {}
    available = {}
    for name in fields:
        field = schema.field(name)
        start, count = extents[field.level]
        rows = _ranges(start[positions], count[positions])
        values[name] = columns.values[name][rows]
        if field.availability == "optional":
            available[name] = columns.available[name][rows]
    selected = positions.tolist()
    return _Columns(
        tuple(columns.record_ids[index] for index in selected),
        tuple(columns.lineage_ids[index] for index in selected),
        tuple(columns.digests[index] for index in selected),
        MappingProxyType(counts),
        stable,
        MappingProxyType(values),
        MappingProxyType(available),
    )


def _concatenate_columns(
    schema: AtomisticStructureStoreSchema,
    parts: Sequence[_Columns],
    fields: tuple[str, ...],
    /,
) -> _Columns:
    if len(parts) == 1:
        return parts[0]
    return _Columns(
        tuple(value for part in parts for value in part.record_ids),
        tuple(value for part in parts for value in part.lineage_ids),
        tuple(value for part in parts for value in part.digests),
        MappingProxyType(
            {
                level: np.concatenate([part.counts[level] for part in parts])
                for level in schema.counted_levels
            }
        ),
        np.concatenate([part.stable_ids for part in parts]),
        MappingProxyType(
            {
                name: np.concatenate([part.values[name] for part in parts])
                for name in fields
            }
        ),
        MappingProxyType(
            {
                name: np.concatenate([part.available[name] for part in parts])
                for name in fields
                if schema.field(name).availability == "optional"
            }
        ),
    )


def _rebased(
    field: AtomisticStoreField,
    values: npt.NDArray[Any],
    available: npt.NDArray[np.bool_] | None,
    extents: _Extents,
    /,
) -> npt.NDArray[Any]:
    index = field.index
    if index is None:
        raise RuntimeError("Only index fields are rebased.")
    target_start, target_count = extents[index.target]
    if int(target_count.sum()) - 1 > np.iinfo(values.dtype).max:
        raise ValueError(
            f"Batch-local rows of {index.target!r} exceed the {values.dtype} range of "
            f"index field {field.name!r}."
        )
    level_count = extents[field.level][1]
    owner = np.repeat(np.arange(level_count.size, dtype=np.int64), level_count)
    offset = _broadcast_rows(values, target_start[owner])
    shift = np.ones(values.shape, dtype=np.bool_)
    if index.sentinel is not None:
        shift &= values != index.sentinel
    if available is not None:
        shift &= _broadcast_rows(values, available)
    return np.where(shift, values + offset, values).astype(values.dtype)


def _read_batch(
    schema: AtomisticStructureStoreSchema,
    source_id: str,
    ordinals: npt.NDArray[np.int64],
    columns: _Columns,
    positions: npt.NDArray[np.int64],
    fields: tuple[str, ...],
    basis: AtomisticStoreIndexBasis,
    /,
) -> AtomisticStructureBatch:
    taken = _take(schema, columns, positions, fields)
    values = dict(taken.values)
    if basis == "batch-local":
        extents = _extents(schema, len(taken.record_ids), taken.counts)
        for name in fields:
            field = schema.field(name)
            if field.kind == "index":
                values[name] = _rebased(
                    field, values[name], taken.available.get(name), extents
                )
    return AtomisticStructureBatch(
        schema,
        source_id=source_id,
        ordinals=ordinals,
        record_ids=taken.record_ids,
        lineage_ids=taken.lineage_ids,
        record_digests=taken.digests,
        fields=fields,
        index_basis=basis,
        parent_counts=taken.counts,
        stable_ids=taken.stable_ids,
        values=values,
        available=taken.available,
    )


def _ordinals(
    ordinals: ConvertibleToArray, record_count: int, /
) -> npt.NDArray[np.int64]:
    array = np.asarray(ordinals)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Reads require a non-empty one-dimensional ordinal sequence.")
    if array.dtype.kind not in "iu":
        raise TypeError("Read ordinals must be integers.")
    if np.any(array < 0) or np.any(array >= record_count):
        raise ValueError(f"Read ordinals must lie in [0, {record_count}).")
    return _frozen(array.astype(np.int64))


@final
class AtomisticStructureRecord(StrictModule, NonTrainableState):
    """One validated structure with exact field dtypes and stable identities.

    ``counts`` gives, for every counted (non-structure, non-product) level, the
    row count per parent row; levels parented by the structure take one entry.
    ``stable_ids`` are the int64 stable particle identities of the atom rows and
    are unique within the record. ``values`` holds each field with shape
    ``(rows, *axes)`` and its exact declared dtype. An omitted optional field is
    unavailable in every row; ``available`` masks partially available optional
    fields, whose unavailable rows must hold zero.
    """

    schema_id: str = eqx.field(static=True)
    record_id: str = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)
    counts: Mapping[str, npt.NDArray[np.int64]]
    stable_ids: npt.NDArray[np.int64]
    values: Mapping[str, npt.NDArray[Any]]
    available: Mapping[str, npt.NDArray[np.bool_]]
    record_digest: str = eqx.field(static=True)

    def __init__(
        self,
        schema: AtomisticStructureStoreSchema,
        /,
        *,
        record_id: str,
        lineage_id: str,
        counts: Mapping[str, ConvertibleToArray],
        stable_ids: ConvertibleToArray,
        values: Mapping[str, ConvertibleToArray],
        available: Mapping[str, ConvertibleToArray] | None = None,
    ) -> None:
        if not isinstance(schema, AtomisticStructureStoreSchema):
            raise TypeError("schema must be an AtomisticStructureStoreSchema.")
        columns, _ = _validated_columns(
            schema,
            (record_id,),
            (lineage_id,),
            None,
            counts,
            stable_ids,
            values,
            available,
            schema.field_names,
            "structure-local",
            absent_optional=True,
        )
        self.schema_id = schema.schema_id
        self.record_id = columns.record_ids[0]
        self.lineage_id = columns.lineage_ids[0]
        self.counts = columns.counts
        self.stable_ids = columns.stable_ids
        self.values = columns.values
        self.available = columns.available
        self.record_digest = columns.digests[0]


@final
class AtomisticStructureBatch(StrictModule, NonTrainableState):
    """Coalesced host rows of requested records in request order.

    ``ordinals`` are the requested source positions including duplicates.
    ``parent_counts`` holds the canonical per-parent-row counts and
    ``structure_counts`` the per-record row count of every non-structure level.
    ``stable_ids``, ``values`` and ``available`` concatenate the records' rows.
    Index fields are structure-local or batch-local as ``index_basis`` states.
    """

    schema: AtomisticStructureStoreSchema
    source_id: str = eqx.field(static=True)
    ordinals: npt.NDArray[np.int64]
    record_ids: tuple[str, ...] = eqx.field(static=True)
    lineage_ids: tuple[str, ...] = eqx.field(static=True)
    record_digests: tuple[str, ...] = eqx.field(static=True)
    fields: tuple[str, ...] = eqx.field(static=True)
    index_basis: AtomisticStoreIndexBasis = eqx.field(static=True)
    parent_counts: Mapping[str, npt.NDArray[np.int64]]
    structure_counts: Mapping[str, npt.NDArray[np.int64]]
    stable_ids: npt.NDArray[np.int64]
    values: Mapping[str, npt.NDArray[Any]]
    available: Mapping[str, npt.NDArray[np.bool_]]

    def __init__(
        self,
        schema: AtomisticStructureStoreSchema,
        /,
        *,
        source_id: str,
        ordinals: ConvertibleToArray,
        record_ids: Sequence[str],
        lineage_ids: Sequence[str],
        record_digests: Sequence[str],
        fields: Sequence[str],
        index_basis: AtomisticStoreIndexBasis,
        parent_counts: Mapping[str, ConvertibleToArray],
        stable_ids: ConvertibleToArray,
        values: Mapping[str, ConvertibleToArray],
        available: Mapping[str, ConvertibleToArray],
    ) -> None:
        if not isinstance(schema, AtomisticStructureStoreSchema):
            raise TypeError("schema must be an AtomisticStructureStoreSchema.")
        names = _projection(schema, fields)
        basis = parse(index_basis, AtomisticStoreIndexBasis, "index_basis")
        columns, extents = _validated_columns(
            schema,
            record_ids,
            lineage_ids,
            record_digests,
            parent_counts,
            stable_ids,
            values,
            available,
            names,
            basis,
            absent_optional=False,
        )
        positions = _frozen(_host_integers(ordinals, "ordinals"))
        if positions.shape != (len(columns.record_ids),) or np.any(positions < 0):
            raise ValueError("Batches carry one nonnegative ordinal per record.")
        self.schema = schema
        self.source_id = _digest_text(source_id, "source_id")
        self.ordinals = positions
        self.record_ids = columns.record_ids
        self.lineage_ids = columns.lineage_ids
        self.record_digests = columns.digests
        self.fields = names
        self.index_basis = basis
        self.parent_counts = columns.counts
        self.structure_counts = MappingProxyType(
            {level.name: _frozen(extents[level.name][1]) for level in schema.levels[1:]}
        )
        self.stable_ids = columns.stable_ids
        self.values = columns.values
        self.available = columns.available

    def offsets(self, level: str, /) -> npt.NDArray[np.int64]:
        """Batch row offsets ``(records + 1,)`` of one non-structure level."""

        if level not in self.structure_counts:
            raise ValueError(f"Level {level!r} has no per-record row counts.")
        counts = self.structure_counts[level]
        return _frozen(np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts))))


class AbstractAtomisticStructureSource(StrictModule, NonTrainableState):
    """Immutable ordered structure records under one schema.

    ``source_id`` identifies the ordered record content (record identities and
    layout-independent record digests); it is equal for an in-memory source and
    any store revision holding the same ordered records.
    """

    schema: eqx.AbstractVar[AtomisticStructureStoreSchema]
    source_id: eqx.AbstractVar[str]
    record_ids: eqx.AbstractVar[tuple[str, ...]]
    lineage_ids: eqx.AbstractVar[tuple[str, ...]]
    record_digests: eqx.AbstractVar[tuple[str, ...]]

    @property
    @abc.abstractmethod
    def background_read_safe(self) -> bool:
        """Whether independent reads may execute in a host prefetch thread."""
        raise NotImplementedError

    @property
    def record_count(self) -> int:
        return len(self.record_ids)

    @abc.abstractmethod
    def level_counts(self, level: str, /) -> npt.NDArray[np.int64]:
        """Per-record row counts of one level, without reading field data."""
        raise NotImplementedError

    @abc.abstractmethod
    def read(
        self,
        ordinals: ConvertibleToArray,
        /,
        *,
        fields: Sequence[str] | None = None,
        index_basis: AtomisticStoreIndexBasis = "structure-local",
    ) -> AtomisticStructureBatch:
        """Read records in request order, preserving duplicates and projection."""
        raise NotImplementedError


def _level_counts_of(
    schema: AtomisticStructureStoreSchema,
    counts: Mapping[str, npt.NDArray[np.int64]],
    level: str,
    /,
) -> npt.NDArray[np.int64]:
    schema.level(level)
    return counts[level]


@final
class AtomisticStructureSource(AbstractAtomisticStructureSource):
    """Immutable in-memory structure source in canonical columnar form."""

    schema: AtomisticStructureStoreSchema
    source_id: str = eqx.field(static=True)
    record_ids: tuple[str, ...] = eqx.field(static=True)
    lineage_ids: tuple[str, ...] = eqx.field(static=True)
    record_digests: tuple[str, ...] = eqx.field(static=True)
    _columns: _Columns
    _structure_counts: Mapping[str, npt.NDArray[np.int64]]

    def __init__(
        self,
        schema: AtomisticStructureStoreSchema,
        records: Sequence[AtomisticStructureRecord],
        /,
    ) -> None:
        if not isinstance(schema, AtomisticStructureStoreSchema):
            raise TypeError("schema must be an AtomisticStructureStoreSchema.")
        records_ = tuple(records)
        if not records_ or any(
            not isinstance(record, AtomisticStructureRecord) for record in records_
        ):
            raise TypeError(
                "A structure source requires AtomisticStructureRecord values."
            )
        if any(record.schema_id != schema.schema_id for record in records_):
            raise ValueError("Every record must belong to the source schema identity.")
        ids = tuple(record.record_id for record in records_)
        if len(set(ids)) != len(ids):
            raise ValueError("Record identifiers must be unique within a source.")
        columns = _Columns(
            ids,
            tuple(record.lineage_id for record in records_),
            tuple(record.record_digest for record in records_),
            MappingProxyType(
                {
                    level: _frozen(
                        np.concatenate([record.counts[level] for record in records_])
                    )
                    for level in schema.counted_levels
                }
            ),
            _frozen(np.concatenate([record.stable_ids for record in records_])),
            MappingProxyType(
                {
                    name: _frozen(
                        np.concatenate([record.values[name] for record in records_])
                    )
                    for name in schema.field_names
                }
            ),
            MappingProxyType(
                {
                    name: _frozen(
                        np.concatenate([record.available[name] for record in records_])
                    )
                    for name in schema.field_names
                    if schema.field(name).availability == "optional"
                }
            ),
        )
        extents = _extents(schema, len(ids), columns.counts)
        self.schema = schema
        self.source_id = _content_id(schema.schema_id, ids, columns.digests)
        self.record_ids = ids
        self.lineage_ids = columns.lineage_ids
        self.record_digests = columns.digests
        self._columns = columns
        self._structure_counts = MappingProxyType(
            {level.name: _frozen(extents[level.name][1]) for level in schema.levels}
        )

    @property
    def background_read_safe(self) -> bool:
        return True

    def level_counts(self, level: str, /) -> npt.NDArray[np.int64]:
        return _level_counts_of(self.schema, self._structure_counts, level)

    def read(
        self,
        ordinals: ConvertibleToArray,
        /,
        *,
        fields: Sequence[str] | None = None,
        index_basis: AtomisticStoreIndexBasis = "structure-local",
    ) -> AtomisticStructureBatch:
        positions = _ordinals(ordinals, self.record_count)
        names = _projection(self.schema, fields)
        basis = parse(index_basis, AtomisticStoreIndexBasis, "index_basis")
        return _read_batch(
            self.schema, self.source_id, positions, self._columns, positions, names, basis
        )


# ------------------------------------------------------------- store metadata


@final
class AtomisticStoreLayout(StrictModule, NonTrainableState):
    """Execution-only partition, chunking and compression of new shards.

    The layout never enters record digests or source identity: the same
    ordered records in another layout are the same scientific content.
    """

    records_per_shard: int = eqx.field(static=True)
    target_chunk_bytes: int = eqx.field(static=True)
    compression: AtomisticStoreCompression = eqx.field(static=True)
    compression_level: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        records_per_shard: int,
        target_chunk_bytes: int,
        compression: AtomisticStoreCompression = "none",
        compression_level: int | None = None,
    ) -> None:
        self.records_per_shard = _positive(records_per_shard, "records_per_shard")
        self.target_chunk_bytes = _positive(target_chunk_bytes, "target_chunk_bytes")
        self.compression, self.compression_level = _compression(
            compression, compression_level
        )


@final
class AtomisticStoreLimits(StrictModule, NonTrainableState):
    """Host resource bounds checked before reading, allocating or decoding.

    ``max_shard_members`` bounds the entries of one shard directory including
    its subdirectories. ``max_chunk_bytes`` bounds the decoded bytes of any one
    Zarr chunk that is written or read. ``max_index_bytes`` bounds the decoded
    record-index chunks of a revision that is opened or published;
    ``max_decoded_bytes`` bounds the decoded chunks one read touches and
    ``max_read_bytes`` its output rows. ``max_records`` bounds the retained
    records of a revision that is opened or published. Every bound is a
    positive int64-representable integer and all admission arithmetic is exact.
    """

    max_commit_bytes: int = eqx.field(static=True)
    max_member_bytes: int = eqx.field(static=True)
    max_shard_bytes: int = eqx.field(static=True)
    max_shard_members: int = eqx.field(static=True)
    max_chunk_bytes: int = eqx.field(static=True)
    max_index_bytes: int = eqx.field(static=True)
    max_decoded_bytes: int = eqx.field(static=True)
    max_read_bytes: int = eqx.field(static=True)
    max_records: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_commit_bytes: int,
        max_member_bytes: int,
        max_shard_bytes: int,
        max_shard_members: int,
        max_chunk_bytes: int,
        max_index_bytes: int,
        max_decoded_bytes: int,
        max_read_bytes: int,
        max_records: int,
    ) -> None:
        self.max_commit_bytes = _bound(max_commit_bytes, "max_commit_bytes")
        self.max_member_bytes = _bound(max_member_bytes, "max_member_bytes")
        self.max_shard_bytes = _bound(max_shard_bytes, "max_shard_bytes")
        self.max_shard_members = _bound(max_shard_members, "max_shard_members")
        self.max_chunk_bytes = _bound(max_chunk_bytes, "max_chunk_bytes")
        self.max_index_bytes = _bound(max_index_bytes, "max_index_bytes")
        self.max_decoded_bytes = _bound(max_decoded_bytes, "max_decoded_bytes")
        self.max_read_bytes = _bound(max_read_bytes, "max_read_bytes")
        self.max_records = _bound(max_records, "max_records")
        if self.max_member_bytes > self.max_shard_bytes:
            raise ValueError("max_member_bytes cannot exceed max_shard_bytes.")
        if self.max_chunk_bytes > self.max_decoded_bytes:
            raise ValueError("max_chunk_bytes cannot exceed max_decoded_bytes.")


def _aggregate_digest(members: Sequence[tuple[str, int, str]], /) -> str:
    """Aggregate member identity, framed exactly as the resource-set owner does."""

    aggregate = hashlib.sha256()
    for path, size, digest in members:
        for part in (
            path.encode("utf-8"),
            size.to_bytes(8, "big"),
            bytes.fromhex(digest),
        ):
            aggregate.update(len(part).to_bytes(8, "big"))
            aggregate.update(part)
    return aggregate.hexdigest()


@final
class AtomisticStoreShard(StrictModule, NonTrainableState):
    """One sealed content-addressed Zarr shard as referenced by a commit.

    ``shard_id`` is the aggregate SHA-256 of the exact member inventory.
    ``level_rows`` are the total rows of every level, which fix every array
    shape before any member is read. ``retracted`` lists local records that the
    referencing revision excludes; the sealed shard itself never changes.
    """

    shard_id: str = eqx.field(static=True)
    record_count: int = eqx.field(static=True)
    level_rows: tuple[tuple[str, int], ...] = eqx.field(static=True)
    record_id_bytes: int = eqx.field(static=True)
    lineage_id_bytes: int = eqx.field(static=True)
    target_chunk_bytes: int = eqx.field(static=True)
    compression: AtomisticStoreCompression = eqx.field(static=True)
    compression_level: int | None = eqx.field(static=True)
    members: tuple[tuple[str, int, str], ...] = eqx.field(static=True)
    retracted: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        shard_id: str,
        /,
        *,
        record_count: int,
        level_rows: Mapping[str, int],
        record_id_bytes: int,
        lineage_id_bytes: int,
        target_chunk_bytes: int,
        compression: AtomisticStoreCompression,
        compression_level: int | None,
        members: Sequence[tuple[str, int, str]],
        retracted: Sequence[int] = (),
    ) -> None:
        identity = _digest_text(shard_id, "shard_id")
        count = _positive(record_count, "record_count")
        if not isinstance(level_rows, Mapping):
            raise TypeError("level_rows must map level names to row counts.")
        rows = tuple(
            (_name(name, "level name"), _nonnegative(level_rows[name], "level rows"))
            for name in sorted(level_rows)
        )
        record_bytes = _positive(record_id_bytes, "record_id_bytes")
        lineage_bytes = _positive(lineage_id_bytes, "lineage_id_bytes")
        if record_bytes < count or lineage_bytes < count:
            raise ValueError("Every record and lineage identifier has at least one byte.")
        if max(count, record_bytes, lineage_bytes, *(value for _, value in rows)) > (
            _INT64_MAX
        ):
            raise ValueError("Shard cardinalities must be representable as int64.")
        inventory: list[tuple[str, int, str]] = []
        for member in members:
            if not isinstance(member, (tuple, list)) or len(member) != 3:
                raise TypeError("Shard members are (path, size, sha256) records.")
            path, size, digest = member
            if not isinstance(path, str) or not path or path.startswith("/"):
                raise ValueError("Shard member paths are non-empty relative paths.")
            inventory.append(
                (
                    path,
                    _positive(size, "member size"),
                    _digest_text(digest, "member digest"),
                )
            )
        paths = [path for path, _, _ in inventory]
        if not inventory or paths != sorted(set(paths)):
            raise ValueError("Shard members must be non-empty, unique and path-sorted.")
        if _aggregate_digest(inventory) != identity:
            raise ValueError("The shard identity does not address its member inventory.")
        retracted_ = tuple(_nonnegative(value, "retracted record") for value in retracted)
        if list(retracted_) != sorted(set(retracted_)) or any(
            value >= count for value in retracted_
        ):
            raise ValueError(
                "Retracted local records must be sorted, unique and in range."
            )
        self.shard_id = identity
        self.record_count = count
        self.level_rows = rows
        self.record_id_bytes = record_bytes
        self.lineage_id_bytes = lineage_bytes
        self.target_chunk_bytes = _positive(target_chunk_bytes, "target_chunk_bytes")
        self.compression, self.compression_level = _compression(
            compression, compression_level
        )
        self.members = tuple(inventory)
        self.retracted = retracted_

    @property
    def retained_count(self) -> int:
        return self.record_count - len(self.retracted)


def _shard_payload(shard: AtomisticStoreShard, /) -> dict[str, Any]:
    return {
        "shard_id": shard.shard_id,
        "record_count": shard.record_count,
        "level_rows": dict(shard.level_rows),
        "record_id_bytes": shard.record_id_bytes,
        "lineage_id_bytes": shard.lineage_id_bytes,
        "target_chunk_bytes": shard.target_chunk_bytes,
        "compression": shard.compression,
        "compression_level": shard.compression_level,
        "members": [list(member) for member in shard.members],
        "retracted": list(shard.retracted),
    }


def _shard_from_payload(payload: object, /) -> AtomisticStoreShard:
    record = _require_keys(
        payload,
        (
            "shard_id",
            "record_count",
            "level_rows",
            "record_id_bytes",
            "lineage_id_bytes",
            "target_chunk_bytes",
            "compression",
            "compression_level",
            "members",
            "retracted",
        ),
        "shard",
    )
    level_rows = record["level_rows"]
    if not isinstance(level_rows, dict):
        raise ValueError("Shard level rows must be an object.")
    return AtomisticStoreShard(
        record["shard_id"],
        record_count=record["record_count"],
        level_rows=level_rows,
        record_id_bytes=record["record_id_bytes"],
        lineage_id_bytes=record["lineage_id_bytes"],
        target_chunk_bytes=record["target_chunk_bytes"],
        compression=record["compression"],
        compression_level=record["compression_level"],
        members=[
            tuple(_list(member, "shard member"))
            for member in _list(record["members"], "shard members")
        ],
        retracted=tuple(_list(record["retracted"], "retracted records")),
    )


def _with_retracted(
    shard: AtomisticStoreShard, retracted: Iterable[int], /
) -> AtomisticStoreShard:
    return AtomisticStoreShard(
        shard.shard_id,
        record_count=shard.record_count,
        level_rows=dict(shard.level_rows),
        record_id_bytes=shard.record_id_bytes,
        lineage_id_bytes=shard.lineage_id_bytes,
        target_chunk_bytes=shard.target_chunk_bytes,
        compression=shard.compression,
        compression_level=shard.compression_level,
        members=shard.members,
        retracted=tuple(sorted(set(retracted))),
    )


@final
class AtomisticStoreRetraction(StrictModule, NonTrainableState):
    """Lineage evidence that a revision withdrew records, with its reason."""

    sequence: int = eqx.field(static=True)
    record_ids: tuple[str, ...] = eqx.field(static=True)
    reason: str = eqx.field(static=True)

    def __init__(
        self, sequence: int, record_ids: Sequence[str], /, *, reason: str
    ) -> None:
        if isinstance(record_ids, str):
            raise TypeError("record_ids must be a sequence of record identifiers.")
        ids = tuple(_record_text(value, "retracted record_id") for value in record_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("A retraction names unique records.")
        self.sequence = _positive(sequence, "retraction sequence")
        self.record_ids = ids
        self.reason = _text(reason, "retraction reason")


def _retraction_payload(retraction: AtomisticStoreRetraction, /) -> dict[str, Any]:
    return {
        "sequence": retraction.sequence,
        "record_ids": list(retraction.record_ids),
        "reason": retraction.reason,
    }


@final
class AtomisticStoreCommit(StrictModule, NonTrainableState):
    """One immutable store revision, published atomically as a canonical commit.

    ``content_id`` is the `AbstractAtomisticStructureSource.source_id` of the
    revision's ordered retained records. ``revision`` and ``lineage`` are the
    canonical lifecycle identity and ancestry; ``commit_id`` addresses the
    complete canonical commit record. Sequence zero creates an empty store.
    """

    schema: AtomisticStructureStoreSchema
    revision: NumericRevision
    lineage: RevisionLineage
    store_name: str = eqx.field(static=True)
    store_id: str = eqx.field(static=True)
    sequence: int = eqx.field(static=True)
    operation: AtomisticStoreOperation = eqx.field(static=True)
    shards: tuple[AtomisticStoreShard, ...]
    retractions: tuple[AtomisticStoreRetraction, ...]
    record_count: int = eqx.field(static=True)
    content_id: str = eqx.field(static=True)
    parent_commit_id: str | None = eqx.field(static=True)
    commit_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema: AtomisticStructureStoreSchema,
        /,
        *,
        store_name: str,
        sequence: int,
        operation: AtomisticStoreOperation,
        shards: Sequence[AtomisticStoreShard],
        retractions: Sequence[AtomisticStoreRetraction],
        content_id: str,
        parent_commit_id: str | None = None,
        parent_revision_id: str | None = None,
        parent_lineage_id: str | None = None,
    ) -> None:
        if not isinstance(schema, AtomisticStructureStoreSchema):
            raise TypeError("schema must be an AtomisticStructureStoreSchema.")
        name = _record_text(store_name, "store_name")
        sequence_ = _nonnegative(sequence, "sequence")
        operation_ = parse(operation, AtomisticStoreOperation, "operation")
        shards_ = tuple(shards)
        retractions_ = tuple(retractions)
        if any(not isinstance(shard, AtomisticStoreShard) for shard in shards_):
            raise TypeError("Commit shards must be AtomisticStoreShard values.")
        if any(not isinstance(item, AtomisticStoreRetraction) for item in retractions_):
            raise TypeError("Commit retractions must be AtomisticStoreRetraction values.")
        content = _digest_text(content_id, "content_id")
        parents = (parent_commit_id, parent_revision_id, parent_lineage_id)
        if (sequence_ == 0) != (operation_ == "create"):
            raise ValueError("Exactly sequence zero creates a store.")
        if operation_ == "create":
            if any(parent is not None for parent in parents) or shards_ or retractions_:
                raise ValueError("Store creation has no parent, shards or retractions.")
        else:
            for parent in parents:
                if parent is None:
                    raise ValueError("Non-create revisions name their complete parent.")
                _digest_text(parent, "parent identity")
        if len({shard.shard_id for shard in shards_}) != len(shards_):
            raise ValueError("A revision references each shard once.")
        level_names = set(schema.level_names)
        for shard in shards_:
            rows = dict(shard.level_rows)
            if (
                set(rows) != level_names
                or rows[schema.structure_level] != shard.record_count
            ):
                raise ValueError("Shard level rows do not match the store schema.")
        sequences = [item.sequence for item in retractions_]
        if sequences != sorted(set(sequences)) or any(
            value > sequence_ for value in sequences
        ):
            raise ValueError("Retractions are ordered by their unique past sequences.")
        if (operation_ == "retract") != bool(sequences and sequences[-1] == sequence_):
            raise ValueError("Exactly a retract revision adds a retraction.")
        retracted_ids = [value for item in retractions_ for value in item.record_ids]
        if len(set(retracted_ids)) != len(retracted_ids):
            raise ValueError("A record is retracted at most once in a lineage.")
        store_id = canonical_fingerprint(
            {
                "kind": "atomistic-structure-store",
                "name": name,
                "schema_id": schema.schema_id,
            }
        )
        revision = NumericRevision(
            SemanticProvenance(
                {
                    "kind": "atomistic-structure-store",
                    "store_id": store_id,
                    "schema_id": schema.schema_id,
                }
            ),
            {
                "sequence": sequence_,
                "operation": operation_,
                "content_id": content,
                "shards": [[shard.shard_id, list(shard.retracted)] for shard in shards_],
                "retractions": [_retraction_payload(item) for item in retractions_],
            },
        )
        self.schema = schema
        self.revision = revision
        self.lineage = RevisionLineage(
            revision,
            label=f"atomistic-structure-store-{operation_}",
            parent_revision_id=parent_revision_id,
            parent_lineage_id=parent_lineage_id,
        )
        self.store_name = name
        self.store_id = store_id
        self.sequence = sequence_
        self.operation = operation_
        self.shards = shards_
        self.retractions = retractions_
        self.record_count = sum(shard.retained_count for shard in shards_)
        self.content_id = content
        self.parent_commit_id = parent_commit_id
        self.commit_id = canonical_fingerprint(_commit_content(self))


def _commit_content(commit: AtomisticStoreCommit, /) -> dict[str, Any]:
    return {
        "kind": "atomistic-structure-store-commit",
        "store_name": commit.store_name,
        "store_id": commit.store_id,
        "sequence": commit.sequence,
        "operation": commit.operation,
        "schema": _schema_payload(commit.schema),
        "shards": [_shard_payload(shard) for shard in commit.shards],
        "retractions": [_retraction_payload(item) for item in commit.retractions],
        "record_count": commit.record_count,
        "content_id": commit.content_id,
        "parent": (
            None
            if commit.parent_commit_id is None
            else {
                "commit_id": commit.parent_commit_id,
                "revision_id": commit.lineage.parent_revision_id,
                "lineage_id": commit.lineage.parent_lineage_id,
            }
        ),
        "revision_id": commit.revision.revision_id,
        "lineage_id": commit.lineage.lineage_id,
    }


def _commit_payload(commit: AtomisticStoreCommit, /) -> dict[str, Any]:
    return {**_commit_content(commit), "commit_id": commit.commit_id}


def _commit_from_payload(payload: object, /) -> AtomisticStoreCommit:
    record = _require_keys(
        payload,
        (
            "kind",
            "store_name",
            "store_id",
            "sequence",
            "operation",
            "schema",
            "shards",
            "retractions",
            "record_count",
            "content_id",
            "parent",
            "revision_id",
            "lineage_id",
            "commit_id",
        ),
        "commit",
    )
    parent = record["parent"]
    parent_ids: tuple[str | None, str | None, str | None] = (None, None, None)
    if parent is not None:
        parent_ = _require_keys(
            parent, ("commit_id", "revision_id", "lineage_id"), "parent"
        )
        parent_ids = (parent_["commit_id"], parent_["revision_id"], parent_["lineage_id"])
    retractions = []
    for item in _list(record["retractions"], "retractions"):
        retraction = _require_keys(
            item, ("sequence", "record_ids", "reason"), "retraction"
        )
        retractions.append(
            AtomisticStoreRetraction(
                retraction["sequence"],
                tuple(_list(retraction["record_ids"], "retracted records")),
                reason=retraction["reason"],
            )
        )
    commit = AtomisticStoreCommit(
        _schema_from_payload(record["schema"]),
        store_name=record["store_name"],
        sequence=record["sequence"],
        operation=record["operation"],
        shards=tuple(
            _shard_from_payload(item) for item in _list(record["shards"], "shards")
        ),
        retractions=tuple(retractions),
        content_id=record["content_id"],
        parent_commit_id=parent_ids[0],
        parent_revision_id=parent_ids[1],
        parent_lineage_id=parent_ids[2],
    )
    if _commit_payload(commit) != record:
        raise ValueError("Persisted commit identities do not match the commit content.")
    return commit


# ------------------------------------------------------------- Zarr encoding


@dataclass(frozen=True, slots=True)
class _ZarrRuntime:
    zarr: ModuleType
    storage: ModuleType
    codecs: ModuleType
    buffer: ModuleType


@functools.cache
def _zarr_runtime() -> _ZarrRuntime:
    try:
        return _ZarrRuntime(
            importlib.import_module("zarr"),
            importlib.import_module("zarr.storage"),
            importlib.import_module("zarr.codecs"),
            importlib.import_module("zarr.buffer.cpu"),
        )
    except ModuleNotFoundError as error:
        if error.name is None or error.name.split(".")[0] != "zarr":
            raise
        raise ModuleNotFoundError(
            "Atomistic structure stores require the optional 'zarr' dependency.",
            name="zarr",
        ) from error


@dataclass(frozen=True, slots=True)
class _ArraySpec:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    rows_per_chunk: int

    @property
    def trailing(self) -> tuple[int, ...]:
        return self.shape[1:]

    @property
    def row_bytes(self) -> int:
        return self.dtype.itemsize * math.prod(self.trailing)

    @property
    def chunk_bytes(self) -> int:
        return self.rows_per_chunk * self.row_bytes

    @property
    def chunk_count(self) -> int:
        return -(-self.shape[0] // self.rows_per_chunk)

    @property
    def nbytes(self) -> int:
        return self.shape[0] * self.row_bytes


def _array_specs(
    schema: AtomisticStructureStoreSchema,
    *,
    record_count: int,
    level_rows: Mapping[str, int],
    record_id_bytes: int,
    lineage_id_bytes: int,
    target_chunk_bytes: int,
) -> dict[str, _ArraySpec]:
    """Canonical shard arrays and their exact shapes, dtypes and row chunking."""

    if set(level_rows) != set(schema.level_names):
        raise ValueError("Shard level rows do not name the schema levels.")
    if level_rows[schema.structure_level] != record_count:
        raise ValueError("Shard structure rows differ from its record count.")
    int64, uint8 = np.dtype(np.int64), np.dtype(np.uint8)
    shapes: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
        "record_id.utf8": ((record_id_bytes,), uint8),
        "record_id.offsets": ((record_count + 1,), int64),
        "lineage_id.utf8": ((lineage_id_bytes,), uint8),
        "lineage_id.offsets": ((record_count + 1,), int64),
        "record_digest": ((record_count, 32), uint8),
    }
    for level in schema.counted_levels:
        shapes[f"count.{level}"] = ((level_rows[_parent(schema, level)],), int64)
    shapes["stable_id"] = ((level_rows[schema.atom_level],), int64)
    for field in schema.fields:
        rows = level_rows[field.level]
        shapes[f"value.{field.name}"] = ((rows, *field.trailing_shape), field.numpy_dtype)
        if field.availability == "optional":
            shapes[f"available.{field.name}"] = ((rows,), np.dtype(np.bool_))
    specs = {}
    for name, (shape, dtype) in shapes.items():
        row_bytes = dtype.itemsize * math.prod(shape[1:])
        specs[name] = _ArraySpec(shape, dtype, max(1, target_chunk_bytes // row_bytes))
    return specs


def _chunk_key(row_chunk: int, trailing_rank: int, /) -> str:
    return "/".join(("c", str(row_chunk), *("0",) * trailing_rank))


def _member_paths(specs: Mapping[str, _ArraySpec], /) -> dict[str, str | None]:
    """Every shard member path, mapped to its array for chunk members."""

    paths: dict[str, str | None] = {"zarr.json": None}
    for name, spec in specs.items():
        paths[f"{name}/zarr.json"] = None
        for chunk in range(spec.chunk_count):
            paths[f"{name}/{_chunk_key(chunk, len(spec.trailing))}"] = name
    return paths


def _shard_entries(specs: Mapping[str, _ArraySpec], /) -> tuple[int, int]:
    """Count the canonical member files and all directory entries of one shard."""

    files = entries = 1
    for spec in specs.values():
        chunks = spec.chunk_count
        files += 1 + chunks
        # The array directory and metadata; when chunked, ``c/`` plus per chunk
        # its file and one directory per trailing axis (``c/<row>/0/.../0``).
        entries += 2 + (1 + chunks * (1 + len(spec.trailing)) if chunks else 0)
    return files, entries


def _admit_specs(
    specs: Mapping[str, _ArraySpec], limits: AtomisticStoreLimits, /
) -> tuple[int, int]:
    """Admit decoded chunk sizes and shard entries with exact integer arithmetic.

    Runs before any member path is enumerated or any chunk is encoded or
    decoded. Returns the canonical member-file count and the decoded bytes of
    the record-index chunks.
    """

    for name, spec in specs.items():
        if spec.chunk_bytes > limits.max_chunk_bytes:
            raise AtomisticStoreError(
                "limit",
                f"Shard array {name!r} has {spec.chunk_bytes}-byte decoded chunks, "
                f"above the {limits.max_chunk_bytes}-byte chunk limit.",
            )
    files, entries = _shard_entries(specs)
    if entries > limits.max_shard_members:
        raise AtomisticStoreError(
            "limit",
            f"A shard has {entries} directory entries, above the "
            f"{limits.max_shard_members}-entry limit.",
        )
    index_bytes = sum(
        spec.chunk_count * spec.chunk_bytes
        for name, spec in specs.items()
        if name.startswith(_INDEX_ARRAY_PREFIXES)
    )
    return files, index_bytes


def _shard_arrays(
    schema: AtomisticStructureStoreSchema, columns: _Columns, /
) -> dict[str, npt.NDArray[Any]]:
    arrays: dict[str, npt.NDArray[Any]] = {}
    for prefix, identifiers in (
        ("record_id", columns.record_ids),
        ("lineage_id", columns.lineage_ids),
    ):
        encoded = [value.encode("ascii") for value in identifiers]
        arrays[f"{prefix}.utf8"] = np.frombuffer(b"".join(encoded), dtype=np.uint8)
        arrays[f"{prefix}.offsets"] = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum([len(value) for value in encoded], dtype=np.int64),
            )
        )
    arrays["record_digest"] = np.frombuffer(
        b"".join(bytes.fromhex(value) for value in columns.digests), dtype=np.uint8
    ).reshape(len(columns.digests), 32)
    for level in schema.counted_levels:
        arrays[f"count.{level}"] = columns.counts[level]
    arrays["stable_id"] = columns.stable_ids
    for field in schema.fields:
        arrays[f"value.{field.name}"] = columns.values[field.name]
        if field.availability == "optional":
            arrays[f"available.{field.name}"] = columns.available[field.name]
    return arrays


def _encode_shard(
    schema_id: str,
    arrays: Mapping[str, npt.NDArray[Any]],
    specs: Mapping[str, _ArraySpec],
    compression: AtomisticStoreCompression,
    compression_level: int | None,
    /,
) -> dict[str, bytes]:
    """Encode one shard as an in-memory Zarr v3 hierarchy and return its members."""

    runtime = _zarr_runtime()
    store_members: dict[str, Any] = {}
    group = runtime.zarr.create_group(
        store=runtime.storage.MemoryStore(store_members),
        zarr_format=3,
        attributes={"phydrax_atomistic_structure_shard": {"schema_id": schema_id}},
    )
    match compression:
        case "none":
            compressors = None
        case "zstd":
            compressors = (
                runtime.codecs.ZstdCodec(level=compression_level, checksum=False),
            )
        case _:
            assert_never(compression)
    for name, values in arrays.items():
        spec = specs[name]
        if values.shape != spec.shape or values.dtype != spec.dtype:
            raise RuntimeError(f"Shard array {name!r} differs from its canonical spec.")
        array = group.create_array(
            name=name,
            shape=spec.shape,
            dtype=spec.dtype,
            chunks=(spec.rows_per_chunk, *spec.trailing),
            filters=None,
            compressors=compressors,
            fill_value=spec.dtype.type(0),
            config={"write_empty_chunks": True},
        )
        if values.size:
            array[...] = values
    return {key: buffer.to_bytes() for key, buffer in store_members.items()}


def _json_object(data: bytes, label: str, /) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"), parse_constant=_refuse_constant)
    except (UnicodeDecodeError, ValueError) as error:
        raise AtomisticStoreError("malformed", f"{label} is not valid JSON.") from error
    if not isinstance(payload, dict):
        raise AtomisticStoreError("malformed", f"{label} must be a JSON object.")
    return payload


def _refuse_constant(value: str, /) -> NoReturn:
    raise ValueError(f"Non-finite JSON constant {value} is not admitted.")


def _bytes_codec(codec: object, itemsize: int, /) -> bool:
    if not isinstance(codec, dict) or codec.get("name") != "bytes":
        return False
    configuration = codec.get("configuration", {})
    if not isinstance(configuration, dict) or set(configuration) - {"endian"}:
        return False
    endian = configuration.get("endian")
    return endian == "little" or (itemsize == 1 and endian is None)


def _validate_array_metadata(
    data: bytes, spec: _ArraySpec, shard: AtomisticStoreShard, name: str, /
) -> None:
    """Refuse Zarr metadata that differs from the committed cardinality and codec."""

    payload = _json_object(data, f"Zarr metadata of {name!r}")
    codecs = payload.get("codecs")
    expected_codecs = 1 if shard.compression == "none" else 2
    valid = (
        not set(payload) - _ZARR_ARRAY_KEYS
        and payload.get("zarr_format") == 3
        and payload.get("node_type") == "array"
        and payload.get("shape") == list(spec.shape)
        and payload.get("data_type") == spec.dtype.name
        and payload.get("chunk_grid")
        == {
            "name": "regular",
            "configuration": {"chunk_shape": [spec.rows_per_chunk, *spec.trailing]},
        }
        and payload.get("chunk_key_encoding")
        == {"name": "default", "configuration": {"separator": "/"}}
        and payload.get("storage_transformers", []) == []
        and isinstance(codecs, list)
        and len(codecs) == expected_codecs
        and _bytes_codec(codecs[0], spec.dtype.itemsize)
    )
    if valid and isinstance(codecs, list) and shard.compression == "zstd":
        codec = codecs[1]
        configuration = codec.get("configuration") if isinstance(codec, dict) else None
        valid = (
            isinstance(codec, dict)
            and codec.get("name") == "zstd"
            and isinstance(configuration, dict)
            and configuration.get("level") == shard.compression_level
        )
    if not valid:
        raise AtomisticStoreError(
            "malformed",
            f"Zarr metadata of {name!r} differs from its committed shape, dtype, "
            "chunking or codec.",
        )


def _zstd_content_size(data: bytes, /) -> int:
    """Return the declared decompressed size of one Zstandard frame."""

    if len(data) < 6 or data[:4] != _ZSTD_MAGIC or data[4] & 0x08:
        raise AtomisticStoreError("malformed", "A shard chunk is not a Zstandard frame.")
    descriptor = data[4]
    single_segment = bool(descriptor & 0x20)
    position = 5 + (0 if single_segment else 1) + (0, 1, 2, 4)[descriptor & 0x03]
    width = (1 if single_segment else 0, 2, 4, 8)[descriptor >> 6]
    if width == 0 or len(data) < position + width:
        raise AtomisticStoreError(
            "malformed", "A Zstandard shard chunk does not declare its content size."
        )
    size = int.from_bytes(data[position : position + width], "little")
    return size + 256 if width == 2 else size


def _validate_chunk(data: bytes, spec: _ArraySpec, shard: AtomisticStoreShard, /) -> None:
    match shard.compression:
        case "none":
            declared = len(data)
        case "zstd":
            declared = _zstd_content_size(data)
        case _:
            assert_never(shard.compression)
    if declared != spec.chunk_bytes:
        raise AtomisticStoreError(
            "malformed",
            f"A shard chunk declares {declared} decoded bytes, not {spec.chunk_bytes}.",
        )


@dataclass(frozen=True, slots=True)
class _ShardFiles:
    shard: AtomisticStoreShard
    specs: Mapping[str, _ArraySpec]
    inventory: Mapping[str, tuple[int, str]]


@dataclass(frozen=True, slots=True)
class _ShardIndex:
    files: _ShardFiles
    counts: Mapping[str, npt.NDArray[np.int64]]
    extents: _Extents
    record_ids: tuple[str, ...]
    lineage_ids: tuple[str, ...]
    digests: tuple[str, ...]


def _validate_shard_inventory(
    shard: AtomisticStoreShard,
    specs: Mapping[str, _ArraySpec],
    limits: AtomisticStoreLimits,
    /,
) -> int:
    """Admit one shard inventory and return its decoded record-index bytes."""

    files, index_bytes = _admit_specs(specs, limits)
    if len(shard.members) != files:
        raise AtomisticStoreError(
            "malformed",
            f"The shard inventory lists {len(shard.members)} members, not its "
            f"{files} canonical Zarr members.",
        )
    paths = _member_paths(specs)
    if {path for path, _, _ in shard.members} != set(paths):
        raise AtomisticStoreError(
            "malformed", "The shard inventory differs from its canonical Zarr members."
        )
    total = 0
    for path, size, _ in shard.members:
        if size > limits.max_member_bytes:
            raise AtomisticStoreError(
                "limit", f"Shard member {path!r} exceeds its limit."
            )
        total += size
        array = paths[path]
        if (
            array is not None
            and shard.compression == "none"
            and size != specs[array].chunk_bytes
        ):
            raise AtomisticStoreError(
                "malformed", f"Uncompressed chunk {path!r} has an inconsistent size."
            )
    if total > limits.max_shard_bytes:
        raise AtomisticStoreError("limit", "A shard exceeds its byte limit.")
    return index_bytes


def _read_member(root: str, files: _ShardFiles, path: str, /) -> bytes:
    size, digest = files.inventory[path]
    try:
        resource = read_bounded_resource(
            f"shards/{files.shard.shard_id}/{path}",
            trusted_root=root,
            limits=ResourceLimits(
                max_bytes=size,
                max_depth=_STORE_PATH_DEPTH,
                max_nodes=1,
                max_attributes=0,
                max_losses=0,
            ),
        )
    except ResourceReadError as error:
        raise AtomisticStoreError(
            "inconsistent", f"Sealed shard member {path!r} is missing or changed."
        ) from error
    if resource.manifest.size_bytes != size or resource.manifest.content_sha256 != digest:
        raise AtomisticStoreError(
            "inconsistent", f"Sealed shard member {path!r} differs from its commit."
        )
    return resource.data


def _row_runs(starts: npt.NDArray[np.int64], counts: npt.NDArray[np.int64], /) -> _Runs:
    """Merge ascending, non-overlapping row ranges into maximal contiguous runs."""

    keep = counts > 0
    begin = starts[keep]
    end = begin + counts[keep]
    if begin.size == 0:
        return begin, end
    opens = np.ones(begin.size, dtype=np.bool_)
    opens[1:] = begin[1:] != end[:-1]
    closes = np.ones(begin.size, dtype=np.bool_)
    closes[:-1] = opens[1:]
    return begin[opens], end[closes]


def _chunk_bounds(runs: _Runs, rows_per_chunk: int, /) -> _Runs:
    """First and last row chunk covered by every run."""

    begin, end = runs
    return begin // rows_per_chunk, (end - 1) // rows_per_chunk


def _touched_chunk_count(runs: _Runs, rows_per_chunk: int, /) -> int:
    """Count the distinct row chunks covered by maximal ascending runs."""

    first, last = _chunk_bounds(runs, rows_per_chunk)
    # Separated ascending runs share at most their boundary chunk.
    shared = int(np.count_nonzero(first[1:] == last[:-1]))
    return int(np.sum(last - first + 1)) - shared


def _read_rows(
    root: str, files: _ShardFiles, name: str, runs: _Runs, /
) -> npt.NDArray[Any]:
    """Read verified, metadata-checked row runs of one admitted shard array."""

    spec = files.specs[name]
    begin, end = runs
    if begin.size == 0:
        return _frozen(np.zeros((0, *spec.trailing), dtype=spec.dtype))
    runtime = _zarr_runtime()
    metadata = _read_member(root, files, f"{name}/zarr.json")
    _validate_array_metadata(metadata, spec, files.shard, name)
    first, last = _chunk_bounds(runs, spec.rows_per_chunk)
    members = {"zarr.json": runtime.buffer.Buffer.from_bytes(metadata)}
    for chunk in np.unique(_ranges(first, last - first + 1)).tolist():
        key = _chunk_key(chunk, len(spec.trailing))
        data = _read_member(root, files, f"{name}/{key}")
        _validate_chunk(data, spec, files.shard)
        members[key] = runtime.buffer.Buffer.from_bytes(data)
    array = runtime.zarr.open_array(
        runtime.storage.MemoryStore(members, read_only=True), mode="r", zarr_format=3
    )
    pieces = []
    for start, stop in zip(begin.tolist(), end.tolist(), strict=True):
        piece = np.asarray(array[start:stop])
        if piece.dtype != spec.dtype or piece.shape != (stop - start, *spec.trailing):
            raise AtomisticStoreError(
                "inconsistent", f"Decoded rows of {name!r} differ from their spec."
            )
        pieces.append(piece)
    return _frozen(pieces[0] if len(pieces) == 1 else np.concatenate(pieces))


def _whole(spec: _ArraySpec, /) -> _Runs:
    runs = 1 if spec.shape[0] else 0
    return (
        np.zeros(runs, dtype=np.int64),
        np.full(runs, spec.shape[0], dtype=np.int64),
    )


def _decode_identifiers(
    data: npt.NDArray[Any], offsets: npt.NDArray[Any], count: int, label: str, /
) -> tuple[str, ...]:
    if (
        offsets.shape != (count + 1,)
        or offsets[0] != 0
        or offsets[-1] != data.size
        or np.any(np.diff(offsets) <= 0)
    ):
        raise AtomisticStoreError("malformed", f"Shard {label} offsets are invalid.")
    raw = data.tobytes()
    identifiers = []
    for begin, end in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True):
        try:
            identifiers.append(_record_text(raw[begin:end].decode("ascii"), label))
        except (UnicodeDecodeError, ValueError) as error:
            raise AtomisticStoreError(
                "malformed", f"A shard {label} is not a valid identifier."
            ) from error
    return tuple(identifiers)


def _load_shard_index(
    root: str,
    schema: AtomisticStructureStoreSchema,
    shard: AtomisticStoreShard,
    specs: Mapping[str, _ArraySpec],
    /,
) -> _ShardIndex:
    files = _ShardFiles(
        shard,
        MappingProxyType(dict(specs)),
        MappingProxyType({path: (size, digest) for path, size, digest in shard.members}),
    )

    def whole(name: str) -> npt.NDArray[Any]:
        return _read_rows(root, files, name, _whole(specs[name]))

    record_ids = _decode_identifiers(
        whole("record_id.utf8"),
        whole("record_id.offsets"),
        shard.record_count,
        "record identifier",
    )
    lineage_ids = _decode_identifiers(
        whole("lineage_id.utf8"),
        whole("lineage_id.offsets"),
        shard.record_count,
        "lineage identifier",
    )
    digests = tuple(row.tobytes().hex() for row in whole("record_digest"))
    counts = {level: whole(f"count.{level}") for level in schema.counted_levels}
    try:
        extents = _extents(schema, shard.record_count, counts)
    except ValueError as error:
        raise AtomisticStoreError("malformed", str(error)) from error
    rows = dict(shard.level_rows)
    if any(int(extents[level][1].sum()) != rows[level] for level in schema.level_names):
        raise AtomisticStoreError(
            "malformed", "Shard level counts differ from the committed level rows."
        )
    return _ShardIndex(
        files,
        MappingProxyType(counts),
        extents,
        record_ids,
        lineage_ids,
        digests,
    )


@dataclass(frozen=True, slots=True)
class _StoreIndex:
    shards: tuple[_ShardIndex, ...]
    shard_of: npt.NDArray[np.int64]
    local_of: npt.NDArray[np.int64]
    record_ids: tuple[str, ...]
    lineage_ids: tuple[str, ...]
    digests: tuple[str, ...]
    level_counts: Mapping[str, npt.NDArray[np.int64]]


def _shard_specs(
    schema: AtomisticStructureStoreSchema, shard: AtomisticStoreShard, /
) -> dict[str, _ArraySpec]:
    try:
        return _array_specs(
            schema,
            record_count=shard.record_count,
            level_rows=dict(shard.level_rows),
            record_id_bytes=shard.record_id_bytes,
            lineage_id_bytes=shard.lineage_id_bytes,
            target_chunk_bytes=shard.target_chunk_bytes,
        )
    except ValueError as error:
        raise AtomisticStoreError("malformed", str(error)) from error


def _admit_revision(
    commit: AtomisticStoreCommit, limits: AtomisticStoreLimits, /
) -> tuple[Mapping[str, _ArraySpec], ...]:
    """Admit a revision's records, shard layouts and decoded index from metadata.

    Opening and publishing share this admission, so a writer never links a head
    that its own limits refuse to reopen, and no member is read beforehand.
    """

    if commit.record_count > limits.max_records:
        raise AtomisticStoreError(
            "limit",
            f"The revision retains {commit.record_count} records, above the "
            f"{limits.max_records}-record limit.",
        )
    admitted: list[Mapping[str, _ArraySpec]] = []
    index_bytes = 0
    for shard in commit.shards:
        specs = _shard_specs(commit.schema, shard)
        index_bytes += _validate_shard_inventory(shard, specs, limits)
        if index_bytes > limits.max_index_bytes:
            raise AtomisticStoreError(
                "limit",
                f"The revision index decodes more than the "
                f"{limits.max_index_bytes}-byte index limit.",
            )
        admitted.append(MappingProxyType(specs))
    return tuple(admitted)


def _load_index(
    root: str, commit: AtomisticStoreCommit, limits: AtomisticStoreLimits, /
) -> _StoreIndex:
    """Load and verify the complete record index of one committed revision."""

    schema = commit.schema
    all_specs = _admit_revision(commit, limits)
    shards = tuple(
        _load_shard_index(root, schema, shard, specs)
        for shard, specs in zip(commit.shards, all_specs, strict=True)
    )
    shard_of = []
    local_of = []
    retracted_present: set[str] = set()
    for position, shard_index in enumerate(shards):
        keep = np.ones(shard_index.files.shard.record_count, dtype=np.bool_)
        retracted = np.asarray(shard_index.files.shard.retracted, dtype=np.int64)
        keep[retracted] = False
        local = np.flatnonzero(keep).astype(np.int64)
        shard_of.append(np.full(local.size, position, dtype=np.int64))
        local_of.append(local)
        retracted_present.update(
            shard_index.record_ids[index] for index in retracted.tolist()
        )
    shard_of_ = (
        _frozen(np.concatenate(shard_of))
        if shard_of
        else _frozen(np.zeros(0, dtype=np.int64))
    )
    local_of_ = (
        _frozen(np.concatenate(local_of))
        if local_of
        else _frozen(np.zeros(0, dtype=np.int64))
    )
    pairs = list(zip(shard_of_.tolist(), local_of_.tolist(), strict=True))
    record_ids = tuple(shards[shard].record_ids[local] for shard, local in pairs)
    lineage_ids = tuple(shards[shard].lineage_ids[local] for shard, local in pairs)
    digests = tuple(shards[shard].digests[local] for shard, local in pairs)
    if len(set(record_ids)) != len(record_ids):
        raise AtomisticStoreError("inconsistent", "A revision retains a record twice.")
    lineage_retracted = {
        value for item in commit.retractions for value in item.record_ids
    }
    if not retracted_present <= lineage_retracted or lineage_retracted & set(record_ids):
        raise AtomisticStoreError(
            "inconsistent", "Retracted shard records differ from the retraction lineage."
        )
    if _content_id(schema.schema_id, record_ids, digests) != commit.content_id:
        raise AtomisticStoreError(
            "inconsistent", "The revision records differ from the committed content."
        )
    level_counts = {
        level: _frozen(
            np.concatenate(
                [
                    shard_index.extents[level][1][local]
                    for shard_index, local in zip(shards, local_of, strict=True)
                ]
            )
            if shards
            else np.zeros(0, dtype=np.int64)
        )
        for level in schema.level_names
    }
    return _StoreIndex(
        shards,
        shard_of_,
        local_of_,
        record_ids,
        lineage_ids,
        digests,
        MappingProxyType(level_counts),
    )


def _touched_arrays(
    schema: AtomisticStructureStoreSchema, fields: tuple[str, ...], /
) -> tuple[tuple[str, str], ...]:
    """The ``(array, level)`` pairs whose chunks a projected read decodes."""

    arrays = [("stable_id", schema.atom_level)]
    for name in fields:
        field = schema.field(name)
        arrays.append((f"value.{name}", field.level))
        if field.availability == "optional":
            arrays.append((f"available.{name}", field.level))
    return tuple(arrays)


def _level_runs(
    shard_index: _ShardIndex,
    local: npt.NDArray[np.int64],
    arrays: tuple[tuple[str, str], ...],
    /,
) -> dict[str, _Runs]:
    """Merged row runs of the sorted unique local records at every read level."""

    runs: dict[str, _Runs] = {}
    for level in sorted({level for _, level in arrays}):
        start, count = shard_index.extents[level]
        runs[level] = _row_runs(start[local], count[local])
    return runs


def _decoded_bytes(
    shard_index: _ShardIndex,
    runs: Mapping[str, _Runs],
    arrays: tuple[tuple[str, str], ...],
    /,
) -> int:
    """Exact decoded chunk bytes that reading ``runs`` of ``arrays`` touches."""

    specs = shard_index.files.specs
    return sum(
        _touched_chunk_count(runs[level], specs[array].rows_per_chunk)
        * specs[array].chunk_bytes
        for array, level in arrays
    )


def _read_shard_columns(
    root: str,
    schema: AtomisticStructureStoreSchema,
    shard_index: _ShardIndex,
    local: npt.NDArray[np.int64],
    fields: tuple[str, ...],
    runs: Mapping[str, _Runs],
    /,
) -> _Columns:
    """Read the sorted unique local records of one shard, projected."""

    extents = shard_index.extents
    counts = {}
    for level in schema.counted_levels:
        start, count = extents[_parent(schema, level)]
        counts[level] = shard_index.counts[level][_ranges(start[local], count[local])]
    stable = _read_rows(root, shard_index.files, "stable_id", runs[schema.atom_level])
    values = {}
    available = {}
    for name in fields:
        field = schema.field(name)
        values[name] = _read_rows(
            root, shard_index.files, f"value.{name}", runs[field.level]
        )
        if field.availability == "optional":
            available[name] = _read_rows(
                root, shard_index.files, f"available.{name}", runs[field.level]
            )
    selected = local.tolist()
    return _Columns(
        tuple(shard_index.record_ids[index] for index in selected),
        tuple(shard_index.lineage_ids[index] for index in selected),
        tuple(shard_index.digests[index] for index in selected),
        MappingProxyType(counts),
        stable,
        MappingProxyType(values),
        MappingProxyType(available),
    )


def _read_bytes(
    schema: AtomisticStructureStoreSchema,
    level_counts: Mapping[str, npt.NDArray[np.int64]],
    positions: npt.NDArray[np.int64],
    fields: tuple[str, ...],
    /,
) -> int:
    rows = {
        level: _exact_sum(level_counts[level][positions], level)
        for level in schema.level_names
    }
    total = 8 * rows[schema.atom_level]
    total += sum(8 * rows[_parent(schema, level)] for level in schema.counted_levels)
    for name in fields:
        field = schema.field(name)
        total += rows[field.level] * field.row_bytes
        if field.availability == "optional":
            total += rows[field.level]
    return total


def _exact_sum(values: npt.NDArray[np.int64], level: str, /) -> int:
    """Exact total of nonnegative int64 row counts, refusing int64 overflow."""

    if float(np.sum(values, dtype=np.float64)) >= _EXACT_SUM_GUARD:
        raise AtomisticStoreError(
            "limit",
            f"The requested rows of level {level!r} exceed exact int64 accounting.",
        )
    return int(np.sum(values, dtype=np.int64))


@final
class AtomisticStoredStructureSource(AbstractAtomisticStructureSource):
    """One complete committed store revision with lazy, verified, bounded reads.

    Constructed by `open_atomistic_structure_store`. Every read verifies the
    exact committed bytes and Zarr metadata of the members it touches before
    decoding, and only decodes chunks covering the requested rows.
    """

    schema: AtomisticStructureStoreSchema
    commit: AtomisticStoreCommit
    limits: AtomisticStoreLimits
    root: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    record_ids: tuple[str, ...] = eqx.field(static=True)
    lineage_ids: tuple[str, ...] = eqx.field(static=True)
    record_digests: tuple[str, ...] = eqx.field(static=True)
    _index: _StoreIndex

    def __init__(
        self,
        commit: AtomisticStoreCommit,
        index: _StoreIndex,
        /,
        *,
        root: str,
        limits: AtomisticStoreLimits,
    ) -> None:
        if not isinstance(commit, AtomisticStoreCommit):
            raise TypeError("commit must be an AtomisticStoreCommit.")
        if not isinstance(index, _StoreIndex) or len(index.shards) != len(commit.shards):
            raise TypeError("index must be the loaded index of the commit.")
        if not isinstance(limits, AtomisticStoreLimits):
            raise TypeError("limits must be AtomisticStoreLimits.")
        self.schema = commit.schema
        self.commit = commit
        self.limits = limits
        self.root = root
        self.source_id = commit.content_id
        self.record_ids = index.record_ids
        self.lineage_ids = index.lineage_ids
        self.record_digests = index.digests
        self._index = index

    @property
    def background_read_safe(self) -> bool:
        # Every read owns its verified member map and read-only Zarr handle.
        return True

    def level_counts(self, level: str, /) -> npt.NDArray[np.int64]:
        return _level_counts_of(self.schema, self._index.level_counts, level)

    def read(
        self,
        ordinals: ConvertibleToArray,
        /,
        *,
        fields: Sequence[str] | None = None,
        index_basis: AtomisticStoreIndexBasis = "structure-local",
    ) -> AtomisticStructureBatch:
        positions = _ordinals(ordinals, self.record_count)
        names = _projection(self.schema, fields)
        basis = parse(index_basis, AtomisticStoreIndexBasis, "index_basis")
        requested = _read_bytes(self.schema, self._index.level_counts, positions, names)
        if requested > self.limits.max_read_bytes:
            raise AtomisticStoreError(
                "limit",
                f"The read needs {requested} bytes, above the "
                f"{self.limits.max_read_bytes}-byte limit.",
            )
        shard_of = self._index.shard_of[positions]
        local_of = self._index.local_of[positions]
        arrays = _touched_arrays(self.schema, names)
        plan: list[tuple[int, npt.NDArray[np.int64], dict[str, _Runs]]] = []
        decoded = 0
        for shard_position in np.unique(shard_of).tolist():
            shard_index = self._index.shards[shard_position]
            local = np.unique(local_of[shard_of == shard_position])
            runs = _level_runs(shard_index, local, arrays)
            decoded += _decoded_bytes(shard_index, runs, arrays)
            plan.append((shard_position, local, runs))
        if decoded > self.limits.max_decoded_bytes:
            raise AtomisticStoreError(
                "limit",
                f"The read decodes {decoded} chunk bytes, above the "
                f"{self.limits.max_decoded_bytes}-byte decode limit.",
            )
        parts = []
        mapped = np.empty(positions.size, dtype=np.int64)
        offset = 0
        for shard_position, local, runs in plan:
            parts.append(
                _read_shard_columns(
                    self.root,
                    self.schema,
                    self._index.shards[shard_position],
                    local,
                    names,
                    runs,
                )
            )
            selected = shard_of == shard_position
            mapped[selected] = offset + np.searchsorted(local, local_of[selected])
            offset += local.size
        columns = _concatenate_columns(self.schema, parts, names)
        return _read_batch(
            self.schema, self.source_id, positions, columns, mapped, names, basis
        )


# ------------------------------------------------------------ store lifecycle


def _store_root(root: str | os.PathLike[str], /) -> str:
    text = os.fspath(root)
    if not isinstance(text, str):
        raise TypeError("Store roots must be text paths.")
    if not text or "\x00" in text or remote_location(text):
        raise ValueError("Store roots must be local filesystem paths.")
    return os.path.abspath(os.path.expanduser(text))


def _commit_name(sequence: int, /) -> str:
    return f"{sequence:020d}.json"


def _commit_sequences(root: str, /) -> tuple[int, ...]:
    try:
        with open_directory_beneath(
            "commits", trusted_root=root, maximum_depth=1
        ) as opened:
            names = os.listdir(opened.descriptor)
    except (OSError, ValueError, RuntimeError) as error:
        raise AtomisticStoreError(
            "malformed", "The path is not a readable atomistic structure store."
        ) from error
    sequences = []
    for name in names:
        match = _COMMIT_NAME.fullmatch(name)
        if match is not None:
            sequences.append(int(match[1]))
        elif _COMMIT_TEMPORARY.fullmatch(name) is None:
            raise AtomisticStoreError(
                "malformed", f"Unexpected entry {name!r} in the commit directory."
            )
    sequences.sort()
    if not sequences or sequences != list(range(len(sequences))):
        raise AtomisticStoreError("malformed", "The store commit sequence is incomplete.")
    return tuple(sequences)


def _read_commit(
    root: str, sequence: int, limits: AtomisticStoreLimits, /
) -> AtomisticStoreCommit:
    try:
        resource = read_bounded_resource(
            f"commits/{_commit_name(sequence)}",
            trusted_root=root,
            limits=ResourceLimits(
                max_bytes=limits.max_commit_bytes,
                max_depth=2,
                max_nodes=1,
                max_attributes=0,
                max_losses=0,
            ),
        )
    except ResourceReadError as error:
        reason: AtomisticStoreFailure = (
            "limit" if error.reason == "limit" else "malformed"
        )
        raise AtomisticStoreError(reason, f"Commit {sequence} cannot be read.") from error
    payload = _json_object(resource.data, f"Commit {sequence}")
    if canonical_json(payload).encode("ascii") != resource.data:
        raise AtomisticStoreError(
            "malformed", f"Commit {sequence} is not canonical JSON."
        )
    try:
        commit = _commit_from_payload(payload)
    except AtomisticStoreError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise AtomisticStoreError(
            "malformed", f"Commit {sequence} is invalid: {error}"
        ) from error
    if commit.sequence != sequence:
        raise AtomisticStoreError(
            "malformed", f"Commit {sequence} names another sequence."
        )
    return commit


def open_atomistic_structure_store(
    root: str | os.PathLike[str],
    /,
    *,
    limits: AtomisticStoreLimits,
    sequence: int | None = None,
) -> AtomisticStoredStructureSource:
    """Open the head revision, or the explicit ``sequence``, of a structure store.

    The commit, its link to the preceding commit, every shard inventory and the
    complete record index are verified before the source is returned. Earlier
    revisions stay readable after later appends, retractions and compactions.
    """

    if not isinstance(limits, AtomisticStoreLimits):
        raise TypeError("limits must be AtomisticStoreLimits.")
    root_ = _store_root(root)
    head = _commit_sequences(root_)[-1]
    if sequence is None:
        target = head
    else:
        target = _nonnegative(sequence, "sequence")
        if target > head:
            raise AtomisticStoreError(
                "identity", f"Revision {target} is not committed; the head is {head}."
            )
    commit = _read_commit(root_, target, limits)
    if target:
        previous = _read_commit(root_, target - 1, limits)
        if (
            commit.parent_commit_id != previous.commit_id
            or commit.lineage.parent_revision_id != previous.revision.revision_id
            or commit.lineage.parent_lineage_id != previous.lineage.lineage_id
            or commit.store_id != previous.store_id
        ):
            raise AtomisticStoreError(
                "identity", f"Commit {target} does not continue its predecessor."
            )
    return AtomisticStoredStructureSource(
        commit, _load_index(root_, commit, limits), root=root_, limits=limits
    )


def _columns_from_batch(
    schema: AtomisticStructureStoreSchema, batch: AtomisticStructureBatch, /
) -> _Columns:
    """Admit complete structure-local batch data after recomputing its digests."""

    if batch.schema.schema_id != schema.schema_id:
        raise AtomisticStoreError("identity", "The batch belongs to another schema.")
    if batch.fields != schema.field_names or batch.index_basis != "structure-local":
        raise ValueError("Stored records require every field in structure-local basis.")
    columns = _Columns(
        batch.record_ids,
        batch.lineage_ids,
        batch.record_digests,
        batch.parent_counts,
        batch.stable_ids,
        batch.values,
        batch.available,
    )
    extents = _extents(schema, len(columns.record_ids), columns.counts)
    if _record_digests(schema, columns, extents) != batch.record_digests:
        raise AtomisticStoreError(
            "inconsistent", "Source records differ from their record digests."
        )
    return columns


@final
class AtomisticStructureStoreWriter(StrictModule, NonTrainableState):
    """Host writer publishing new revisions of one store directory.

    Every operation names its expected ``parent`` head commit. Shards are sealed
    and published first; the commit is then linked exclusively at the next
    sequence, so concurrent or stale writers fail with a ``conflict`` and an
    interrupted operation leaves the parent head authoritative. Rerunning an
    interrupted operation reuses identical sealed shards.
    """

    root: str = eqx.field(static=True)
    layout: AtomisticStoreLayout
    limits: AtomisticStoreLimits

    def __init__(
        self,
        root: str | os.PathLike[str],
        /,
        *,
        layout: AtomisticStoreLayout,
        limits: AtomisticStoreLimits,
    ) -> None:
        if not isinstance(layout, AtomisticStoreLayout):
            raise TypeError("layout must be an AtomisticStoreLayout.")
        if not isinstance(limits, AtomisticStoreLimits):
            raise TypeError("limits must be AtomisticStoreLimits.")
        self.root = _store_root(root)
        self.layout = layout
        self.limits = limits

    def create(
        self, schema: AtomisticStructureStoreSchema, /, *, name: str
    ) -> AtomisticStoreCommit:
        """Publish sequence zero of a new, empty store."""

        commit = AtomisticStoreCommit(
            schema,
            store_name=name,
            sequence=0,
            operation="create",
            shards=(),
            retractions=(),
            content_id=_content_id(schema.schema_id, (), ()),
        )
        self._publish(commit)
        return commit

    def append(
        self,
        parent: AtomisticStoreCommit,
        source: AbstractAtomisticStructureSource,
        /,
    ) -> AtomisticStoreCommit:
        """Append every record of ``source`` after the parent's records."""

        head = self._head(parent)
        if not isinstance(source, AbstractAtomisticStructureSource):
            raise TypeError("source must be an atomistic structure source.")
        schema = parent.schema
        if source.schema.schema_id != schema.schema_id:
            raise AtomisticStoreError(
                "identity", "The appended source has another schema identity."
            )
        known = set(head.record_ids)
        known.update(value for item in parent.retractions for value in item.record_ids)
        for record_id in source.record_ids:
            if record_id in known:
                raise AtomisticStoreError(
                    "identity",
                    f"Record {record_id!r} already exists in the store lineage.",
                )
            known.add(record_id)
        retained = head.record_count + source.record_count
        if retained > self.limits.max_records:
            raise AtomisticStoreError(
                "limit",
                f"The appended revision would retain {retained} records, above the "
                f"{self.limits.max_records}-record limit.",
            )
        shards = []
        for start in range(0, source.record_count, self.layout.records_per_shard):
            stop = min(start + self.layout.records_per_shard, source.record_count)
            batch = source.read(np.arange(start, stop, dtype=np.int64))
            if batch.record_digests != source.record_digests[start:stop]:
                raise AtomisticStoreError(
                    "inconsistent", "The source read differs from its record digests."
                )
            shards.append(self._seal(schema, _columns_from_batch(schema, batch)))
        content = _content_id(
            schema.schema_id,
            (*head.record_ids, *source.record_ids),
            (*head.record_digests, *source.record_digests),
        )
        return self._commit(
            parent,
            "append",
            parent.shards + tuple(shards),
            parent.retractions,
            content,
        )

    def retract(
        self,
        parent: AtomisticStoreCommit,
        record_ids: Sequence[str],
        /,
        *,
        reason: str,
    ) -> AtomisticStoreCommit:
        """Publish a revision withdrawing retained records, keeping their lineage."""

        head = self._head(parent)
        retraction = AtomisticStoreRetraction(
            parent.sequence + 1, record_ids, reason=reason
        )
        position = {record_id: index for index, record_id in enumerate(head.record_ids)}
        missing = [value for value in retraction.record_ids if value not in position]
        if missing:
            raise AtomisticStoreError(
                "identity", f"Records {missing} are not retained by the parent revision."
            )
        ordinals = np.asarray(
            [position[value] for value in retraction.record_ids], dtype=np.int64
        )
        shard_of = head._index.shard_of[ordinals]
        local_of = head._index.local_of[ordinals]
        shards = tuple(
            _with_retracted(
                shard,
                (*shard.retracted, *local_of[shard_of == index].tolist()),
            )
            if np.any(shard_of == index)
            else shard
            for index, shard in enumerate(parent.shards)
        )
        keep = np.ones(head.record_count, dtype=np.bool_)
        keep[ordinals] = False
        retained = np.flatnonzero(keep).tolist()
        content = _content_id(
            parent.schema.schema_id,
            (head.record_ids[index] for index in retained),
            (head.record_digests[index] for index in retained),
        )
        return self._commit(
            parent, "retract", shards, parent.retractions + (retraction,), content
        )

    def compact(self, parent: AtomisticStoreCommit, /) -> AtomisticStoreCommit:
        """Rewrite the retained records into new shards under this writer's layout."""

        head = self._head(parent)
        schema = parent.schema
        shards = []
        for start in range(0, head.record_count, self.layout.records_per_shard):
            stop = min(start + self.layout.records_per_shard, head.record_count)
            batch = head.read(np.arange(start, stop, dtype=np.int64))
            shards.append(self._seal(schema, _columns_from_batch(schema, batch)))
        return self._commit(
            parent, "compact", tuple(shards), parent.retractions, parent.content_id
        )

    def _head(self, parent: AtomisticStoreCommit, /) -> AtomisticStoredStructureSource:
        if not isinstance(parent, AtomisticStoreCommit):
            raise TypeError("parent must be an AtomisticStoreCommit.")
        head = open_atomistic_structure_store(self.root, limits=self.limits)
        if head.commit.commit_id != parent.commit_id:
            raise AtomisticStoreError(
                "conflict", "The parent commit is not the current store head."
            )
        return head

    def _commit(
        self,
        parent: AtomisticStoreCommit,
        operation: AtomisticStoreOperation,
        shards: tuple[AtomisticStoreShard, ...],
        retractions: tuple[AtomisticStoreRetraction, ...],
        content_id: str,
        /,
    ) -> AtomisticStoreCommit:
        commit = AtomisticStoreCommit(
            parent.schema,
            store_name=parent.store_name,
            sequence=parent.sequence + 1,
            operation=operation,
            shards=shards,
            retractions=retractions,
            content_id=content_id,
            parent_commit_id=parent.commit_id,
            parent_revision_id=parent.revision.revision_id,
            parent_lineage_id=parent.lineage.lineage_id,
        )
        self._publish(commit)
        return commit

    def _publish(self, commit: AtomisticStoreCommit, /) -> None:
        """Admit the complete revision under this writer's limits, then link it."""

        _admit_revision(commit, self.limits)
        data = canonical_json(_commit_payload(commit)).encode("ascii")
        if len(data) > self.limits.max_commit_bytes:
            raise AtomisticStoreError("limit", "The commit exceeds its byte limit.")
        try:
            publish_bytes(
                Path(self.root) / "commits" / _commit_name(commit.sequence),
                data,
                maximum_bytes=self.limits.max_commit_bytes,
                mode="exclusive",
            )
        except FileExistsError as error:
            raise AtomisticStoreError(
                "conflict", f"Revision {commit.sequence} was already published."
            ) from error

    def _resource_set_limits(self) -> ResourceSetLimits:
        return ResourceSetLimits(
            max_total_bytes=self.limits.max_shard_bytes,
            max_member_bytes=self.limits.max_member_bytes,
            max_members=self.limits.max_shard_members,
            max_depth=_MEMBER_DEPTH,
        )

    def _seal(
        self, schema: AtomisticStructureStoreSchema, columns: _Columns, /
    ) -> AtomisticStoreShard:
        """Encode, content-address and exclusively publish one immutable shard."""

        record_count = len(columns.record_ids)
        extents = _extents(schema, record_count, columns.counts)
        level_rows = {level: int(extents[level][1].sum()) for level in schema.level_names}
        arrays = _shard_arrays(schema, columns)
        specs = _array_specs(
            schema,
            record_count=record_count,
            level_rows=level_rows,
            record_id_bytes=arrays["record_id.utf8"].size,
            lineage_id_bytes=arrays["lineage_id.utf8"].size,
            target_chunk_bytes=self.layout.target_chunk_bytes,
        )
        _admit_specs(specs, self.limits)
        if sum(spec.nbytes for spec in specs.values()) > self.limits.max_shard_bytes:
            raise AtomisticStoreError("limit", "A shard exceeds its byte limit.")
        members = _encode_shard(
            schema.schema_id,
            arrays,
            specs,
            self.layout.compression,
            self.layout.compression_level,
        )
        if set(members) != set(_member_paths(specs)):
            raise RuntimeError(
                "The Zarr encoder did not produce the canonical shard members."
            )
        inventory = tuple(
            sorted(
                (path, len(data), hashlib.sha256(data).hexdigest())
                for path, data in members.items()
            )
        )
        shard = AtomisticStoreShard(
            _aggregate_digest(inventory),
            record_count=record_count,
            level_rows=level_rows,
            record_id_bytes=arrays["record_id.utf8"].size,
            lineage_id_bytes=arrays["lineage_id.utf8"].size,
            target_chunk_bytes=self.layout.target_chunk_bytes,
            compression=self.layout.compression,
            compression_level=self.layout.compression_level,
            members=inventory,
        )
        _validate_shard_inventory(shard, specs, self.limits)
        destination = Path(self.root) / "shards" / shard.shard_id
        try:
            receipt = publish_resource_set(
                destination, members, limits=self._resource_set_limits(), mode="exclusive"
            )
        except FileExistsError:
            self._verify_sealed(shard)
        else:
            if receipt.aggregate_sha256 != shard.shard_id:
                raise RuntimeError(
                    "The published shard differs from its content address."
                )
        return shard

    def _verify_sealed(self, shard: AtomisticStoreShard, /) -> None:
        """Admit an already sealed shard only when its exact inventory matches."""

        try:
            manifest = read_bounded_resource_set(
                f"shards/{shard.shard_id}",
                trusted_root=self.root,
                limits=self._resource_set_limits(),
            ).manifest
        except ResourceSetReadError as error:
            raise AtomisticStoreError(
                "inconsistent", "An existing sealed shard cannot be verified."
            ) from error
        existing = tuple(
            (member.relative_path, member.size_bytes, member.content_sha256)
            for member in manifest.members
        )
        if manifest.aggregate_sha256 != shard.shard_id or existing != shard.members:
            raise AtomisticStoreError(
                "inconsistent", "An existing sealed shard differs from its address."
            )


__all__ = [
    "AbstractAtomisticStructureSource",
    "AtomisticStoreAvailability",
    "AtomisticStoreCommit",
    "AtomisticStoreCompression",
    "AtomisticStoreDType",
    "AtomisticStoreError",
    "AtomisticStoreExtensivity",
    "AtomisticStoreFailure",
    "AtomisticStoreField",
    "AtomisticStoreFieldRole",
    "AtomisticStoreIndex",
    "AtomisticStoreIndexBasis",
    "AtomisticStoreLayout",
    "AtomisticStoreLevel",
    "AtomisticStoreLevelKind",
    "AtomisticStoreLimits",
    "AtomisticStoreOperation",
    "AtomisticStoreRetraction",
    "AtomisticStoreShard",
    "AtomisticStoreValueKind",
    "AtomisticStoredStructureSource",
    "AtomisticStructureBatch",
    "AtomisticStructureRecord",
    "AtomisticStructureSource",
    "AtomisticStructureStoreSchema",
    "AtomisticStructureStoreWriter",
    "open_atomistic_structure_store",
]
