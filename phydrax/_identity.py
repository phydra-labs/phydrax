#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import base64
import dataclasses
import enum
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import numpy as np

from ._fingerprint import canonical_fingerprint, canonical_json
from ._strict import StrictModule


RecordInput = Mapping[str, Any] | Sequence[tuple[str, Any]]
_ARRAY_TYPES = (jax.Array, jax.ShapeDtypeStruct, np.ndarray)


def _type_id(value: Any, /) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _named_records(value: RecordInput, name: str, /) -> tuple[tuple[str, Any], ...]:
    raw = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    records: list[tuple[str, Any]] = []
    for record in raw:
        if not isinstance(record, (tuple, list)) or len(record) != 2:
            raise TypeError(f"{name} must be a mapping or a sequence of pairs.")
        key = _identifier(record[0], f"{name} key")
        records.append((key, record[1]))
    keys = tuple(key for key, _ in records)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} keys must be unique.")
    return tuple(sorted(records, key=lambda record: record[0]))


def _array_payload(value: Any, path: str, /) -> dict[str, Any]:
    if isinstance(value, jax.ShapeDtypeStruct):
        raise TypeError(f"Numeric identity cannot realize abstract array {path}.")
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError(f"Numeric identity cannot encode object-dtype array {path}.")
    return {
        "kind": "array",
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "storage_bytes": array.nbytes,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _static_payload(value: Any, path: str, /) -> Any:
    if isinstance(value, np.generic):
        return _static_payload(value.item(), path)
    if isinstance(value, _ARRAY_TYPES):
        raise TypeError(
            f"Static identity field {path} cannot contain a numeric array; "
            "array values belong to a numeric revision."
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"Identity field {path} must be finite.")
        return value
    if isinstance(value, complex):
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError(f"Identity field {path} must be finite.")
        return {"kind": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, np.dtype):
        return {"kind": "dtype", "value": value.str}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, jax.tree_util.PyTreeDef):
        return {
            "kind": "pytree-def",
            "value": str(value),
            "leaves": value.num_leaves,
        }
    if isinstance(value, enum.Enum):
        return {
            "kind": "enum",
            "type": _type_id(value),
            "name": value.name,
        }
    if isinstance(value, slice):
        return {
            "kind": "slice",
            "start": _static_payload(value.start, f"{path}.start"),
            "stop": _static_payload(value.stop, f"{path}.stop"),
            "step": _static_payload(value.step, f"{path}.step"),
        }
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if isinstance(value, StrictModule):
        return _static_module_payload(value, path)
    if isinstance(value, Mapping):
        records = _named_records(value, path)
        return {
            "kind": "mapping",
            "items": [
                [key, _static_payload(item, f"{path}.{key}")] for key, item in records
            ],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _static_payload(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _static_payload(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if callable(value):
        raise TypeError(
            f"Opaque callable {path} requires explicit semantic and numeric IDs."
        )
    raise TypeError(f"Identity field {path} has unsupported type {type(value).__name__}.")


def _static_module_payload(module: StrictModule, path: str, /) -> dict[str, Any]:
    return {
        "kind": "strict-module",
        "type": _type_id(module),
        "fields": [
            [
                field.name,
                _static_payload(
                    object.__getattribute__(module, field.name),
                    f"{path}.{field.name}",
                ),
            ]
            for field in dataclasses.fields(module)
        ],
    }


def _dynamic_payload(value: Any, path: str, /) -> tuple[Any, Any]:
    if isinstance(value, np.generic):
        return _dynamic_payload(value.item(), path)
    if isinstance(value, _ARRAY_TYPES):
        array = _array_payload(value, path)
        return (
            {
                "kind": "array",
                "shape": array["shape"],
                "dtype": array["dtype"],
            },
            array,
        )
    if value is None or isinstance(value, str):
        return value, None
    if isinstance(value, (bool, int)):
        return {"kind": type(value).__name__}, value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"Identity field {path} must be finite.")
        return {"kind": "float"}, value
    if isinstance(value, complex):
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError(f"Identity field {path} must be finite.")
        return (
            {"kind": "complex"},
            {"real": value.real, "imag": value.imag},
        )
    if isinstance(value, np.dtype):
        return {"kind": "dtype", "value": value.str}, None
    if isinstance(value, bytes):
        return _static_payload(value, path), None
    if isinstance(value, enum.Enum):
        return _static_payload(value, path), None
    if isinstance(value, slice) or value is Ellipsis:
        return _static_payload(value, path), None
    if isinstance(value, StrictModule):
        semantic_fields: list[list[Any]] = []
        numeric_fields: list[list[Any]] = []
        for field in dataclasses.fields(value):
            field_value = object.__getattribute__(value, field.name)
            field_path = f"{path}.{field.name}"
            if bool(field.metadata.get("static", False)):
                semantic_fields.append(
                    [field.name, _static_payload(field_value, field_path)]
                )
                continue
            semantic, numeric = _dynamic_payload(field_value, field_path)
            semantic_fields.append([field.name, semantic])
            numeric_fields.append([field.name, numeric])
        return (
            {
                "kind": "strict-module",
                "type": _type_id(value),
                "fields": semantic_fields,
            },
            {
                "kind": "strict-module",
                "fields": numeric_fields,
            },
        )
    if isinstance(value, Mapping):
        records = _named_records(value, path)
        semantic_items: list[list[Any]] = []
        numeric_items: list[list[Any]] = []
        for key, item in records:
            semantic, numeric = _dynamic_payload(item, f"{path}.{key}")
            semantic_items.append([key, semantic])
            numeric_items.append([key, numeric])
        return (
            {"kind": "mapping", "items": semantic_items},
            {"kind": "mapping", "items": numeric_items},
        )
    if isinstance(value, (tuple, list)):
        semantic_items = []
        numeric_items = []
        for index, item in enumerate(value):
            semantic, numeric = _dynamic_payload(item, f"{path}[{index}]")
            semantic_items.append(semantic)
            numeric_items.append(numeric)
        kind = "tuple" if isinstance(value, tuple) else "list"
        return (
            {"kind": kind, "items": semantic_items},
            {"kind": kind, "items": numeric_items},
        )
    if callable(value):
        raise TypeError(
            f"Opaque callable {path} requires explicit semantic and numeric IDs."
        )
    raise TypeError(f"Identity field {path} has unsupported type {type(value).__name__}.")


def strict_module_payload(module: StrictModule, /) -> dict[str, Any]:
    """Return content-addressed semantic and numeric payloads for a StrictModule.

    Static fields contribute only to semantic content. Dynamic array/scalar values
    contribute to numeric content, while their structure contributes to semantics.
    Static numeric arrays and opaque callable fields are rejected.
    """
    if not isinstance(module, StrictModule):
        raise TypeError("strict_module_payload requires a StrictModule.")
    semantic, numeric = _dynamic_payload(module, "module")
    semantic_id = canonical_fingerprint(
        {"kind": "strict-module-semantic-content", "payload": semantic}
    )
    numeric_id = canonical_fingerprint(
        {
            "kind": "strict-module-numeric-content",
            "semantic_content_id": semantic_id,
            "payload": numeric,
        }
    )
    return {
        "semantic_payload": semantic,
        "numeric_payload": numeric,
        "semantic_content_id": semantic_id,
        "numeric_content_id": numeric_id,
    }


def callable_payload(
    value: Any,
    /,
    *,
    semantic_id: str | None = None,
    numeric_id: str | None = None,
) -> dict[str, Any]:
    """Identify a callable without falling back to its Python class or name.

    Callable StrictModules are content-addressed from their fields. Other
    callables are opaque and therefore require both identities explicitly.
    """
    if not callable(value):
        raise TypeError("callable_payload requires a callable value.")
    if isinstance(value, StrictModule):
        if semantic_id is not None or numeric_id is not None:
            raise ValueError(
                "Content-addressed StrictModule callables do not accept ID overrides."
            )
        return strict_module_payload(value)
    if semantic_id is None or numeric_id is None:
        raise TypeError(
            "Opaque callables require explicit semantic_id and numeric_id values."
        )
    semantic_id_ = _identifier(semantic_id, "semantic_id")
    numeric_id_ = _identifier(numeric_id, "numeric_id")
    return {
        "semantic_payload": {
            "kind": "opaque-callable",
            "semantic_id": semantic_id_,
        },
        "numeric_payload": {
            "kind": "opaque-callable",
            "numeric_id": numeric_id_,
        },
        "semantic_content_id": semantic_id_,
        "numeric_content_id": numeric_id_,
    }


class SemanticProvenance(StrictModule):
    """Content-addressed semantics with separately named external resources."""

    content_json: str = eqx.field(static=True)
    content_id: str = eqx.field(static=True)
    resource_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    semantic_id: str = eqx.field(static=True)

    def __init__(
        self,
        content: Any,
        /,
        *,
        resource_ids: Mapping[str, str] | Sequence[tuple[str, str]] = (),
    ):
        content_ = _static_payload(content, "semantic_content")
        resources = tuple(
            (name, _identifier(identifier, f"resource_ids[{name!r}]"))
            for name, identifier in _named_records(resource_ids, "resource_ids")
        )
        content_json = canonical_json(content_)
        content_id = canonical_fingerprint(
            {"kind": "semantic-content", "payload": content_}
        )
        self.content_json = content_json
        self.content_id = content_id
        self.resource_ids = resources
        self.semantic_id = canonical_fingerprint(
            {
                "kind": "semantic-provenance",
                "content_id": content_id,
                "resource_ids": [list(record) for record in resources],
            }
        )


class NumericRevision(StrictModule):
    """Dynamic numeric realization bound to one semantic provenance identity."""

    semantic_id: str = eqx.field(static=True)
    content_json: str = eqx.field(static=True)
    content_id: str = eqx.field(static=True)
    revision_id: str = eqx.field(static=True)

    def __init__(
        self,
        semantic: SemanticProvenance | str,
        content: Any,
        /,
    ):
        semantic_id = (
            semantic.semantic_id
            if isinstance(semantic, SemanticProvenance)
            else _identifier(semantic, "semantic_id")
        )
        structure, realization = _dynamic_payload(content, "numeric_content")
        payload = {"structure": structure, "realization": realization}
        content_json = canonical_json(payload)
        content_id = canonical_fingerprint(
            {"kind": "numeric-content", "payload": payload}
        )
        self.semantic_id = semantic_id
        self.content_json = content_json
        self.content_id = content_id
        self.revision_id = canonical_fingerprint(
            {
                "kind": "numeric-revision",
                "semantic_id": semantic_id,
                "content_id": content_id,
            }
        )


def _shape_records(value: RecordInput, /) -> tuple[tuple[str, tuple[int, ...]], ...]:
    records: list[tuple[str, tuple[int, ...]]] = []
    for name, raw_shape in _named_records(value, "shapes"):
        if isinstance(raw_shape, _ARRAY_TYPES) or isinstance(raw_shape, (str, bytes)):
            raise TypeError("Executable shapes must be explicit integer sequences.")
        shape_values = tuple(raw_shape)
        if any(
            isinstance(size, bool) or not isinstance(size, (int, np.integer))
            for size in shape_values
        ):
            raise TypeError("Executable shape dimensions must be integers.")
        shape = tuple(int(size) for size in shape_values)
        if any(size < 0 for size in shape):
            raise ValueError("Executable shape dimensions must be nonnegative.")
        records.append((name, shape))
    return tuple(records)


def _dtype_records(value: RecordInput, /) -> tuple[tuple[str, str], ...]:
    records: list[tuple[str, str]] = []
    for name, raw_dtype in _named_records(value, "dtypes"):
        if isinstance(raw_dtype, _ARRAY_TYPES):
            raise TypeError("Executable dtypes must be dtype specifications, not arrays.")
        records.append((name, np.dtype(raw_dtype).str))
    return tuple(records)


def _identifier_records(value: RecordInput, name: str, /) -> tuple[tuple[str, str], ...]:
    return tuple(
        (key, _identifier(identifier, f"{name}[{key!r}]"))
        for key, identifier in _named_records(value, name)
    )


def _capacity_records(value: RecordInput, /) -> tuple[tuple[str, int], ...]:
    records: list[tuple[str, int]] = []
    for name, raw_capacity in _named_records(value, "capacities"):
        if isinstance(raw_capacity, bool) or not isinstance(
            raw_capacity, (int, np.integer)
        ):
            raise TypeError("Executable capacities must be integers.")
        capacity = int(raw_capacity)
        if capacity < 0:
            raise ValueError("Executable capacities must be nonnegative.")
        records.append((name, capacity))
    return tuple(records)


def _fact_records(value: RecordInput, name: str, /) -> tuple[tuple[str, str], ...]:
    return tuple(
        (key, canonical_json(_static_payload(fact, f"{name}.{key}")))
        for key, fact in _named_records(value, name)
    )


class ExecutableSignature(StrictModule):
    """Static compilation identity with no dynamic numeric realization values."""

    shapes: tuple[tuple[str, tuple[int, ...]], ...] = eqx.field(static=True)
    dtypes: tuple[tuple[str, str], ...] = eqx.field(static=True)
    space_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    topology_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    capacities: tuple[tuple[str, int], ...] = eqx.field(static=True)
    algorithm_facts: tuple[tuple[str, str], ...] = eqx.field(static=True)
    backend_facts: tuple[tuple[str, str], ...] = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        shapes: RecordInput = (),
        dtypes: RecordInput = (),
        space_ids: RecordInput = (),
        topology_ids: RecordInput = (),
        capacities: RecordInput = (),
        algorithm_facts: RecordInput = (),
        backend_facts: RecordInput = (),
    ):
        shapes_ = _shape_records(shapes)
        dtypes_ = _dtype_records(dtypes)
        spaces_ = _identifier_records(space_ids, "space_ids")
        topology_ = _identifier_records(topology_ids, "topology_ids")
        capacities_ = _capacity_records(capacities)
        algorithms_ = _fact_records(algorithm_facts, "algorithm_facts")
        backend_ = _fact_records(backend_facts, "backend_facts")
        payload = {
            "kind": "executable-signature",
            "shapes": [[name, list(shape)] for name, shape in shapes_],
            "dtypes": [list(record) for record in dtypes_],
            "space_ids": [list(record) for record in spaces_],
            "topology_ids": [list(record) for record in topology_],
            "capacities": [list(record) for record in capacities_],
            "algorithm_facts": [list(record) for record in algorithms_],
            "backend_facts": [list(record) for record in backend_],
        }
        self.shapes = shapes_
        self.dtypes = dtypes_
        self.space_ids = spaces_
        self.topology_ids = topology_
        self.capacities = capacities_
        self.algorithm_facts = algorithms_
        self.backend_facts = backend_
        self.signature_id = canonical_fingerprint(payload)


__all__ = [
    "callable_payload",
    "ExecutableSignature",
    "NumericRevision",
    "SemanticProvenance",
    "strict_module_payload",
]
