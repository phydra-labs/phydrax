#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import base64
import dataclasses
import enum
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from fractions import Fraction
from math import gcd
from pathlib import Path
from types import MappingProxyType, MethodType
from typing import Any, BinaryIO, Literal, NoReturn, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .._array_archive import (
    _read_npy_metadata,
    _reject_json_constant,
    _unique_json_object,
    _validate_json_nesting,
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
)
from .._frozendict import frozendict
from .._strict import mark_strict_initialized, Strict
from .._typing_plan import validate_tree
from ..typing import parse
from ._artifacts import (
    artifact_value,
    artifact_value_codec,
    artifact_value_codec_for,
    artifact_value_id,
)


if TYPE_CHECKING:
    from _typeshed import DataclassInstance


type _NonfiniteFloatToken = Literal["nan", "inf", "-inf"]
# Python identity of each completed shared-capable object (array, PRNG key or
# dataclass instance) -> (canonical ordinal, object). Retaining the object keeps
# its identity from being reused while the recipe is encoded.
type _SharedObjects = dict[int, tuple[int, Any]]
type _RecipeNode = dict[str, Any]
# Restores the array or PRNG key declared by node ``index`` of an admitted table.
type _ArrayFactory = Callable[[int, Mapping[str, Any]], Any]

# Logical recipes are nested records whose later reaches of a shared array or
# dataclass are ordinal ``reference`` records. Admission lowers them to a
# post-order table whose composite nodes name children by earlier index; the
# same table, with references resolved to indices, is the shallow wire form.
_COMPOSITE_KINDS = frozenset(
    {
        "slice",
        "value",
        "namedtuple",
        "dataclass",
        "tuple",
        "list",
        "set",
        "frozenset",
        "mapping",
        "frozendict",
        "mappingproxy",
    }
)
_ARRAY_KINDS = frozenset({"array", "prng_key"})


def _shared_record(
    shared: _SharedObjects | None,
    value: Any,
    record: dict[str, Any],
    /,
) -> dict[str, Any]:
    # Ordinals number completed nodes, so a dataclass follows its fields; that
    # is the order in which restoration completes each object.
    if shared is not None:
        shared[id(value)] = (len(shared), value)
    return record


def _is_prng_key(value: Any, /) -> bool:
    return isinstance(value, jax.Array) and jax.dtypes.issubdtype(
        value.dtype, jax.dtypes.prng_key
    )


def _preflight_serialized_leaf(
    file: Any,
    shape: tuple[int, ...],
    dtype: Any,
    /,
) -> int:
    start = file.tell()
    try:
        payload_dtype, payload_shape, elements = _read_npy_metadata(
            file,
            None,
            DEFAULT_ARRAY_ARCHIVE_LIMITS,
        )
    except ArrayArchiveCorruptionError as error:
        raise ValueError("Serialized model leaf metadata is invalid.") from error
    expected_dtype = np.dtype(dtype)
    if payload_shape != shape or payload_dtype != expected_dtype:
        raise ValueError(
            "Serialized model leaf shape or dtype does not match its recipe."
        )
    payload_end = file.tell() + elements * payload_dtype.itemsize
    file.seek(start)
    return payload_end


def _canonical_recipe_payload(recipe: Mapping[str, Any], /) -> bytes:
    try:
        return json.dumps(
            recipe,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (RecursionError, TypeError, ValueError) as error:
        raise ValueError("Model artifact recipe is not canonical finite JSON.") from error


def _recipe_contains_array(recipe: Mapping[str, Any], /) -> bool:
    pending = [recipe]
    while pending:
        node = pending.pop()
        kind = node["kind"]
        if kind in ("array", "prng_key", "reference"):
            return True
        if kind == "slice":
            pending.extend((node["start"], node["stop"], node["step"]))
        elif kind == "dataclass":
            pending.extend(node["items"])
        elif kind == "value":
            pending.append(node["payload"])
        elif kind in ("tuple", "list", "set", "frozenset", "namedtuple"):
            pending.extend(node["items"])
        elif kind in ("mapping", "frozendict", "mappingproxy"):
            pending.extend(item for pair in node["items"] for item in pair)
    return False


def serialize_model_leaf(file: BinaryIO, value: Any, /) -> None:
    """Serialize one model leaf with canonical, pickle-free NPY encoding."""
    if _is_prng_key(value):
        np.save(file, np.asarray(jr.key_data(value)), allow_pickle=False)
        return
    if eqx.is_array_like(value):
        array = np.asarray(value)
        _recipe_array_spec(
            list(array.shape),
            str(array.dtype),
            DEFAULT_ARRAY_ARCHIVE_LIMITS,
            "serialized model leaf",
        )
        payload = np.ascontiguousarray(array) if array.ndim else array
        np.save(file, payload, allow_pickle=False)
        return
    eqx.default_serialise_filter_spec(file, value)


def deserialize_model_leaf(file: BinaryIO, value: Any, /) -> Any:
    """Deserialize one model leaf after bounded NPY metadata admission."""
    if isinstance(value, _RecipeArrayDescriptor):
        _preflight_serialized_leaf(file, value.shape, value.dtype)
        payload = np.load(
            file,
            allow_pickle=False,
            max_header_size=DEFAULT_ARRAY_ARCHIVE_LIMITS.max_npy_header_bytes,
        )
        if value.backend == "prng_key":
            return jr.wrap_key_data(
                jnp.asarray(payload, dtype=jnp.uint32),
                impl=value.implementation,
            )
        if value.backend == "jax":
            return jnp.asarray(payload)
        host = np.array(payload, copy=True, order="C")
        host.setflags(write=value.writable)
        return host
    if _is_prng_key(value):
        key_data = jr.key_data(value)
        _preflight_serialized_leaf(
            file,
            tuple(key_data.shape),
            np.dtype(np.uint32),
        )
        data = jnp.asarray(
            np.load(
                file,
                allow_pickle=False,
                max_header_size=DEFAULT_ARRAY_ARCHIVE_LIMITS.max_npy_header_bytes,
            ),
            dtype=jnp.uint32,
        )
        return jr.wrap_key_data(data, impl=str(jr.key_impl(value)))
    if eqx.is_array_like(value):
        if isinstance(value, (jax.Array, np.ndarray)):
            shape = tuple(value.shape)
            dtype = value.dtype
        else:
            scalar = np.asarray(value)
            shape = tuple(scalar.shape)
            dtype = scalar.dtype
        _preflight_serialized_leaf(file, shape, dtype)
    restored = eqx.default_deserialise_filter_spec(file, value)
    if isinstance(value, np.ndarray):
        if not isinstance(restored, np.ndarray):
            raise TypeError("A NumPy model leaf must restore its declared host backend.")
        restored.setflags(write=value.flags.writeable)
    return restored


def model_array_bytes(value: object, /) -> int:
    """Count distinct declared scientific buffers without copying or gathering.

    Dataclass fields include immutable host preparation hidden from ordinary
    PyTree leaves. Bound methods retain their numerical receiver, and custom
    PyTrees contribute their declared dynamic leaves. Sharing is object identity,
    not equal contents or a guessed underlying buffer allocation.
    """
    pending: list[object] = [value]
    seen: dict[int, object] = {}
    total = 0
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen[identity] = current
        if isinstance(current, (jax.Array, np.ndarray)):
            if isinstance(current, jax.Array) and _is_prng_key(current):
                total += jr.key_data(current).nbytes
            else:
                if current.dtype.hasobject:
                    raise TypeError(
                        "Scientific buffer accounting refuses object-dtype arrays."
                    )
                total += current.nbytes
        else:
            pending.extend(_model_owner_children(current))
    return total


def _model_owner_children(value: object, /) -> Iterable[object]:
    """Declared full-field owners shared by model and live source accounting."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield object.__getattribute__(value, field.name)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield key
            yield child
    elif isinstance(value, (tuple, list, set, frozenset)):
        yield from value
    elif isinstance(value, MethodType):
        yield value.__self__
    else:
        leaves = jax.tree_util.tree_leaves(value)
        if len(leaves) != 1 or leaves[0] is not value:
            yield from leaves


def model_structure_recipe(
    value: Any,
    /,
    *,
    path: str = "model",
) -> dict[str, Any]:
    """Encode static model structure without embedding Python module paths.

    The recipe is a DAG over object identity. An array, PRNG key or dataclass
    instance reachable through several declared paths is encoded once, at its
    first canonical occurrence; every later occurrence is
    ``{"kind": "reference", "target": n}``, where ``n`` numbers those objects in
    the order restoration completes them. Values without shared objects keep
    their tree-shaped recipe unchanged.
    """
    return _structure_recipe(value, path, {})


def _structure_recipe(
    value: Any,
    path: str,
    shared: _SharedObjects | None,
    /,
) -> dict[str, Any]:
    # Set items, mapping keys and immutable-value payloads are unshared: they
    # encode without references, and validation refuses any array they hold.
    if shared is not None and id(value) in shared:
        return {"kind": "reference", "target": shared[id(value)][0]}
    if _is_prng_key(value):
        data = jr.key_data(value)
        return _shared_record(
            shared,
            value,
            {
                "kind": "prng_key",
                "shape": list(data.shape),
                "implementation": str(jr.key_impl(value)),
            },
        )
    if isinstance(value, (jax.Array, np.ndarray)):
        array = value
        if array.dtype.hasobject:
            raise TypeError(f"{path} contains an object-dtype array.")
        record: dict[str, Any] = {
            "kind": "array",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "backend": "jax" if isinstance(value, jax.Array) else "numpy",
        }
        if isinstance(value, np.ndarray):
            record["writable"] = value.flags.writeable
        return _shared_record(shared, value, record)
    if isinstance(value, enum.Enum):
        return {
            "kind": "enum",
            "type": artifact_value_id(type(value)),
            "name": value.name,
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            token = "nan" if np.isnan(value) else "inf" if value > 0.0 else "-inf"
            return {"kind": "nonfinite_float", "value": token}
        return {"kind": "literal", "value": value}
    if type(value) is Fraction:
        return {
            "kind": "fraction",
            "numerator": value.numerator,
            "denominator": value.denominator,
        }
    if (
        isinstance(value, tuple)
        and "_fields" not in vars(type(value))
        and all(
            item is None
            or type(item) in (str, bool, int, float)
            and (not isinstance(item, float) or np.isfinite(item))
            for item in value
        )
    ):
        return {"kind": "literal_tuple", "values": list(value)}
    if (
        isinstance(value, tuple)
        and "_fields" not in vars(type(value))
        and value
        and all(type(item) is Fraction for item in value)
    ):
        return {
            "kind": "fraction_tuple",
            "values": [[item.numerator, item.denominator] for item in value],
        }
    if isinstance(value, complex):
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError(f"{path} contains a non-finite static complex value.")
        return {"kind": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return _structure_recipe(value.item(), path, shared)
    if isinstance(value, np.dtype):
        return {"kind": "dtype", "value": str(value)}
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, Path):
        return {"kind": "path", "value": str(value)}
    if isinstance(value, slice):
        return {
            "kind": "slice",
            "start": _structure_recipe(value.start, f"{path}.start", shared),
            "stop": _structure_recipe(value.stop, f"{path}.stop", shared),
            "step": _structure_recipe(value.step, f"{path}.step", shared),
        }
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if isinstance(value, type):
        return {"kind": "type", "value": artifact_value_id(value)}
    codec = artifact_value_codec_for(value)
    if codec is not None:
        payload = _structure_recipe(codec.encode_value(value), f"{path}.payload", None)
        if _recipe_contains_array(payload):
            raise TypeError(f"{path} immutable value payload cannot contain arrays.")
        return {"kind": "value", "type": codec.value_id, "payload": payload}
    if isinstance(value, tuple) and "_fields" in vars(type(value)):
        return {
            "kind": "namedtuple",
            "type": artifact_value_id(type(value)),
            "items": [
                _structure_recipe(item, f"{path}[{index}]", shared)
                for index, item in enumerate(value)
            ],
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, frozendict):
        cls = type(value)
        return _shared_record(
            shared,
            value,
            {
                "kind": "dataclass",
                "type": artifact_value_id(cls),
                "items": [
                    _structure_recipe(
                        object.__getattribute__(value, field.name),
                        f"{path}.{field.name}",
                        shared,
                    )
                    for field in dataclasses.fields(value)
                ],
            },
        )
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _structure_recipe(item, f"{path}[{index}]", shared)
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _structure_recipe(item, f"{path}[{index}]", shared)
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, (set, frozenset)):
        encoded_items = [_structure_recipe(item, f"{path}.item", None) for item in value]
        encoded_items.sort(key=_canonical_recipe_payload)
        return {
            "kind": "frozenset" if isinstance(value, frozenset) else "set",
            "items": encoded_items,
        }
    if isinstance(value, Mapping):
        mapping_kind = (
            "frozendict"
            if isinstance(value, frozendict)
            else "mappingproxy"
            if isinstance(value, MappingProxyType)
            else "mapping"
        )
        # Values encode in canonical key order, which is their restoration order.
        encoded_keys = sorted(
            (
                (_structure_recipe(key, f"{path}.key", None), item)
                for key, item in value.items()
            ),
            key=lambda pair: _canonical_recipe_payload(pair[0]),
        )
        key_payloads = [_canonical_recipe_payload(key) for key, _ in encoded_keys]
        if len(set(key_payloads)) != len(key_payloads):
            raise TypeError(f"{path} contains keys with the same canonical encoding.")
        return {
            "kind": mapping_kind,
            "items": [
                [key, _structure_recipe(item, f"{path}.value", shared)]
                for key, item in encoded_keys
            ],
        }
    if callable(value):
        return {"kind": "callable", "value": artifact_value_id(value)}
    raise TypeError(
        f"Portable model artifact cannot represent {path} of type {type(value).__name__}."
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ModelRecipeArray:
    """One distinct declared scientific array.

    ``tree_index`` is the array's first runtime PyTree leaf; non-runtime fields
    have ``tree_index=-1``. An array shared by several paths has one entry.
    """

    name: str
    tree_index: int
    path: str
    shape: tuple[int, ...]
    dtype: str
    backend: str
    implementation: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class _RecipeArrayDescriptor:
    shape: tuple[int, ...]
    dtype: str
    backend: str
    implementation: str | None = None
    writable: bool = True


_PRNG_IMPLEMENTATION_WORDS = {
    "threefry2x32": 2,
    "rbg": 4,
    "unsafe_rbg": 4,
}


def _recipe_fields(
    recipe: Mapping[str, Any],
    expected: set[str],
    path: str,
    /,
) -> None:
    if set(recipe) != expected:
        raise ValueError(
            f"{path} must use exactly {sorted(expected)}; found {sorted(recipe)}."
        )


def _recipe_array_spec(
    shape_value: Any,
    dtype_value: Any,
    limits: ArrayArchiveLimits,
    path: str,
    /,
) -> tuple[tuple[int, ...], np.dtype[Any], int, int]:
    if not isinstance(shape_value, list) or any(
        type(extent) is not int or extent < 0 for extent in shape_value
    ):
        raise ValueError(f"{path} shape must contain exact nonnegative integers.")
    shape = tuple(shape_value)
    if len(shape) > limits.max_array_rank:
        raise ValueError(f"{path} exceeds the recipe array-rank limit.")
    try:
        dtype = np.dtype(dtype_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} dtype is invalid.") from error
    if (
        dtype.hasobject
        or (
            (dtype.fields is not None or dtype.subdtype is not None)
            and not limits.allow_structured_dtypes
        )
        or dtype.metadata is not None
        or dtype.kind not in limits.allowed_dtype_kinds
        or dtype.itemsize > limits.max_dtype_itemsize
    ):
        raise ValueError(f"{path} dtype is not admitted by recipe policy.")
    elements = 1
    for extent in shape:
        if extent > limits.max_axis_length:
            raise ValueError(f"{path} exceeds the recipe axis-length limit.")
        if extent and elements > limits.max_array_elements // extent:
            raise ValueError(f"{path} exceeds the recipe element limit.")
        elements *= extent
    if elements > limits.max_array_elements:
        raise ValueError(f"{path} exceeds the recipe element limit.")
    byte_count = elements * dtype.itemsize
    if byte_count > limits.max_member_bytes:
        raise ValueError(f"{path} exceeds the recipe array-byte limit.")
    return shape, dtype, elements, byte_count


def _admit_array_node(
    node: Mapping[str, Any],
    kind: str,
    limits: ArrayArchiveLimits,
    path: str,
    /,
) -> tuple[int, int]:
    if kind == "array":
        expected = {"kind", "shape", "dtype", "backend"}
        if node.get("backend") == "numpy":
            expected.add("writable")
        _recipe_fields(node, expected, path)
        if node["backend"] == "numpy" and type(node["writable"]) is not bool:
            raise ValueError(f"{path} NumPy mutability metadata must be boolean.")
        if node["backend"] not in ("jax", "numpy"):
            raise ValueError(f"{path} has an invalid array backend.")
        _, _, elements, byte_count = _recipe_array_spec(
            node["shape"], node["dtype"], limits, path
        )
        return elements, byte_count
    _recipe_fields(node, {"kind", "shape", "implementation"}, path)
    implementation = node["implementation"]
    if (
        not isinstance(implementation, str)
        or implementation not in _PRNG_IMPLEMENTATION_WORDS
    ):
        raise ValueError(f"{path} has an unsupported PRNG implementation.")
    shape, _, elements, byte_count = _recipe_array_spec(
        node["shape"], np.dtype(np.uint32).str, limits, path
    )
    if not shape or shape[-1] != _PRNG_IMPLEMENTATION_WORDS[implementation]:
        raise ValueError(f"{path} PRNG key-data shape is incompatible.")
    return elements, byte_count


def _admit_static_node(
    node: Mapping[str, Any],
    kind: str,
    limits: ArrayArchiveLimits,
    path: str,
    /,
) -> None:
    match kind:
        case "literal":
            _recipe_fields(node, {"kind", "value"}, path)
            value = node["value"]
            if value is not None and type(value) not in (str, bool, int, float):
                raise TypeError(f"{path} literal has an invalid type.")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"{path} literal must be finite.")
        case "nonfinite_float":
            _recipe_fields(node, {"kind", "value"}, path)
            parse(node["value"], _NonfiniteFloatToken, f"{path}.value")
        case "fraction":
            _recipe_fields(node, {"kind", "numerator", "denominator"}, path)
            numerator, denominator = node["numerator"], node["denominator"]
            if type(numerator) is not int or type(denominator) is not int:
                raise TypeError(f"{path} fraction components must be exact integers.")
            if denominator <= 0:
                raise ValueError(f"{path} fraction denominator must be positive.")
            # A decimal JSON integer fitting this limit cannot exceed four bits
            # per byte; refuse oversized integers before exact reduction work.
            if max(numerator.bit_length(), denominator.bit_length()) > (
                4 * limits.max_manifest_bytes
            ):
                raise ValueError(f"{path} fraction exceeds the manifest byte limit.")
            if gcd(numerator, denominator) != 1:
                raise ValueError(
                    f"{path} fraction must have reduced canonical components."
                )
        case "fraction_tuple":
            _recipe_fields(node, {"kind", "values"}, path)
            values = node["values"]
            if (
                not isinstance(values, list)
                or not values
                or any(
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or any(type(component) is not int for component in pair)
                    for pair in values
                )
            ):
                raise TypeError(
                    f"{path} fraction tuple components must be exact integers."
                )
            for numerator, denominator in values:
                if denominator <= 0:
                    raise ValueError(
                        f"{path} fraction tuple denominator must be positive."
                    )
                if max(numerator.bit_length(), denominator.bit_length()) > (
                    4 * limits.max_manifest_bytes
                ):
                    raise ValueError(
                        f"{path} fraction tuple exceeds the manifest byte limit."
                    )
                if gcd(numerator, denominator) != 1:
                    raise ValueError(
                        f"{path} fraction tuple must have reduced canonical components."
                    )
        case "literal_tuple":
            _recipe_fields(node, {"kind", "values"}, path)
            values = node["values"]
            if not isinstance(values, list) or any(
                value is not None and type(value) not in (str, bool, int, float)
                for value in values
            ):
                raise TypeError(f"{path} literal tuple has an invalid component.")
            if any(
                isinstance(value, float) and not np.isfinite(value) for value in values
            ):
                raise ValueError(f"{path} literal tuple components must be finite.")
        case "complex":
            _recipe_fields(node, {"kind", "real", "imag"}, path)
            if any(
                type(component) not in (int, float) or not np.isfinite(component)
                for component in (node["real"], node["imag"])
            ):
                raise ValueError(f"{path} complex components must be finite numbers.")
        case "dtype":
            _recipe_fields(node, {"kind", "value"}, path)
            if not isinstance(node["value"], str):
                raise TypeError(f"{path} dtype value must be text.")
            _recipe_array_spec([], node["value"], limits, path)
        case "bytes":
            _recipe_fields(node, {"kind", "value"}, path)
            if not isinstance(node["value"], str):
                raise TypeError(f"{path} bytes value must be base64 text.")
            try:
                decoded = base64.b64decode(node["value"].encode("ascii"), validate=True)
            except (UnicodeEncodeError, ValueError) as error:
                raise ValueError(f"{path} bytes value is invalid base64.") from error
            if len(decoded) > limits.max_member_bytes:
                raise ValueError(f"{path} bytes value exceeds the static-byte limit.")
        case "path":
            _recipe_fields(node, {"kind", "value"}, path)
            if not isinstance(node["value"], str):
                raise TypeError(f"{path} path value must be text.")
        case "ellipsis":
            _recipe_fields(node, {"kind"}, path)
        case "enum":
            _recipe_fields(node, {"kind", "type", "name"}, path)
            if not isinstance(node["type"], str) or not isinstance(node["name"], str):
                raise TypeError(f"{path} enum identity must contain text.")
            enum_type = artifact_value(node["type"])
            if not isinstance(enum_type, type) or not issubclass(enum_type, enum.Enum):
                raise TypeError(f"{path} enum identity does not resolve to an enum.")
            if node["name"] not in enum_type.__members__:
                raise ValueError(f"{path} enum member is unknown.")
        case "type" | "callable":
            _recipe_fields(node, {"kind", "value"}, path)
            if not isinstance(node["value"], str):
                raise TypeError(f"{path} artifact identity must be text.")
            resolved = artifact_value(node["value"])
            if kind == "type" and not isinstance(resolved, type):
                raise TypeError(f"{path} does not resolve to a type.")
            if kind == "callable" and not callable(resolved):
                raise TypeError(f"{path} does not resolve to a callable.")
        case _:
            raise ValueError(f"Unknown model artifact recipe kind {kind!r}.")


def _admit_composite_node(node: Mapping[str, Any], kind: str, path: str, /) -> list[Any]:
    """Validate one composite record; return its raw children in traversal order.

    Dataclass fields follow their registered declaration order, which is the
    order in which encoding and restoration complete shared objects.
    """
    match kind:
        case "slice":
            _recipe_fields(node, {"kind", "start", "stop", "step"}, path)
            return [node["start"], node["stop"], node["step"]]
        case "value":
            _recipe_fields(node, {"kind", "type", "payload"}, path)
            if not isinstance(node["type"], str):
                raise TypeError(f"{path} immutable value identity must be text.")
            artifact_value_codec(node["type"])
            return [node["payload"]]
        case "namedtuple":
            _recipe_fields(node, {"kind", "type", "items"}, path)
            cls = artifact_value(node["type"]) if isinstance(node["type"], str) else None
            if not isinstance(cls, type) or "_fields" not in vars(cls):
                raise TypeError(f"{path} identity does not resolve to a named tuple.")
            items = node["items"]
            if not isinstance(items, list) or len(items) != len(cls._fields):
                raise ValueError(f"{path} named-tuple fields are invalid.")
            return list(items)
        case "dataclass":
            _recipe_fields(node, {"kind", "type", "items"}, path)
            cls = artifact_value(node["type"]) if isinstance(node["type"], str) else None
            items = node["items"]
            if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
                raise TypeError(f"{path} identity does not resolve to a dataclass.")
            if not isinstance(items, list) or len(items) != len(dataclasses.fields(cls)):
                raise ValueError(
                    f"{path} dataclass field count does not match its registered type."
                )
            return list(items)
        case "tuple" | "list" | "set" | "frozenset":
            _recipe_fields(node, {"kind", "items"}, path)
            if not isinstance(node["items"], list):
                raise TypeError(f"{path} items must be a list.")
            return list(node["items"])
        case "mapping" | "frozendict" | "mappingproxy":
            _recipe_fields(node, {"kind", "items"}, path)
            items = node["items"]
            if not isinstance(items, list) or any(
                not isinstance(pair, list) or len(pair) != 2 for pair in items
            ):
                raise TypeError(f"{path} mapping items must be key/value pairs.")
            return [entry for pair in items for entry in pair]
        case _:
            raise ValueError(f"Unknown model artifact recipe kind {kind!r}.")


def _child_labels(node: Mapping[str, Any], /) -> list[tuple[str, bool]]:
    """Path label and shared-context flag of each child, in traversal order.

    Set items, mapping keys and immutable-value payloads are unshared contexts:
    they hold no references and contribute no shared-object ordinals.
    """
    match node["kind"]:
        case "slice":
            return [(f".{name}", True) for name in ("start", "stop", "step")]
        case "value":
            return [(".payload", False)]
        case "dataclass":
            return [
                (f".items[{position}:{field.name!r}]", True)
                for position, field in enumerate(
                    dataclasses.fields(artifact_value(node["type"]))
                )
            ]
        case "namedtuple" | "tuple" | "list":
            return [
                (f".items[{position}]", True) for position in range(len(node["items"]))
            ]
        case "set" | "frozenset":
            return [
                (f".items[{position}]", False) for position in range(len(node["items"]))
            ]
        case "mapping" | "frozendict" | "mappingproxy":
            return [
                (f".items[{position}].{role}", role == "value")
                for position in range(len(node["items"]))
                for role in ("key", "value")
            ]
        case _:
            return []


def _table_children(node: Mapping[str, Any], /) -> list[int]:
    """Table indices of a lowered composite node, in traversal order."""
    match node["kind"]:
        case "slice":
            return [node["start"], node["stop"], node["step"]]
        case "value":
            return [node["payload"]]
        case "dataclass":
            return list(node["items"])
        case "namedtuple" | "tuple" | "list" | "set" | "frozenset":
            return list(node["items"])
        case "mapping" | "frozendict" | "mappingproxy":
            return [entry for pair in node["items"] for entry in pair]
        case _:
            return []


def _with_children(node: Mapping[str, Any], children: list[Any], /) -> _RecipeNode:
    """Rebuild one composite record with replacement children in traversal order."""
    match node["kind"]:
        case "slice":
            start, stop, step = children
            return {"kind": "slice", "start": start, "stop": stop, "step": step}
        case "value":
            return {"kind": "value", "type": node["type"], "payload": children[0]}
        case "dataclass":
            return {
                "kind": "dataclass",
                "type": node["type"],
                "items": list(children),
            }
        case "namedtuple":
            return {"kind": "namedtuple", "type": node["type"], "items": children}
        case "tuple" | "list" | "set" | "frozenset" as kind:
            return {"kind": kind, "items": children}
        case "mapping" | "frozendict" | "mappingproxy" as kind:
            return {
                "kind": kind,
                "items": [
                    [key, item]
                    for key, item in zip(children[::2], children[1::2], strict=True)
                ],
            }
        case kind:
            raise ValueError(f"Unknown model artifact recipe kind {kind!r}.")


@dataclasses.dataclass(frozen=True, slots=True)
class _RecipeTable:
    """Admitted logical recipe lowered to a post-order node table.

    Children precede parents and each shared object, including every reference
    to it, is one node, so restoration is iterative and never expands sharing.
    ``ordinals`` maps each logical shared-object ordinal to its node.
    """

    nodes: list[_RecipeNode]
    ordinals: list[int]


def _lower_recipe(recipe: Any, limits: ArrayArchiveLimits, /) -> _RecipeTable:
    """Admit a complete logical recipe iteratively before any reconstruction.

    Admission bounds visited records, canonical bytes and every distinct array
    before allocation; it refuses unknown kinds, unregistered types, inexact
    field sets, noncanonical set or mapping keys, arrays or references in
    unshared contexts and every reference that does not name an earlier
    completed object.
    """
    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be an ArrayArchiveLimits instance.")
    recipe_bytes = len(_canonical_recipe_payload(recipe))
    if recipe_bytes > limits.max_manifest_bytes:
        raise ValueError(
            "Model artifact recipe exceeds its manifest byte limit "
            f"({recipe_bytes} > {limits.max_manifest_bytes})."
        )
    nodes: list[_RecipeNode] = []
    ordinals: list[int] = []
    statics: dict[bytes, int] = {}
    values: list[tuple[int, Mapping[str, Any], str]] = []
    completed: list[int] = []
    total_elements = total_bytes = visits = 0
    # Frames: (record, path, shared context, completed-stack base or -1).
    pending: list[tuple[Any, str, bool, int]] = [(recipe, "recipe", True, -1)]
    while pending:
        node, path, shared, base = pending.pop()
        if base >= 0:
            children = completed[base:]
            del completed[base:]
            nodes.append(_with_children(node, children))
            index = len(nodes) - 1
            if node["kind"] == "dataclass" and shared:
                ordinals.append(index)
            if node["kind"] == "value":
                values.append((index, node, path))
            completed.append(index)
            continue
        visits += 1
        if visits > limits.max_manifest_bytes:
            raise ValueError("Model artifact recipe exceeds its node limit.")
        if not isinstance(node, dict):
            raise TypeError(f"{path} must be a recipe object.")
        kind = node.get("kind")
        if not isinstance(kind, str):
            raise ValueError(f"{path} has no valid recipe kind.")
        if kind == "reference":
            _recipe_fields(node, {"kind", "target"}, path)
            target = node["target"]
            if not shared:
                raise ValueError(
                    f"{path} unshared recipe positions cannot contain references."
                )
            if type(target) is not int or not 0 <= target < len(ordinals):
                # Targets must be completed objects: no forward or ancestor cycles.
                raise ValueError(
                    "Model artifact recipe references must name an earlier completed object."
                )
            completed.append(ordinals[target])
        elif kind in _ARRAY_KINDS:
            elements, byte_count = _admit_array_node(node, kind, limits, path)
            if not shared:
                raise ValueError(
                    f"{path} unshared recipe positions cannot contain arrays."
                )
            # References carry no bytes: limits apply to distinct stored objects.
            if total_elements > limits.max_total_array_elements - elements:
                raise ValueError("Model artifact recipe exceeds its total element limit.")
            if total_bytes > limits.max_aggregate_bytes - byte_count:
                raise ValueError(
                    "Model artifact recipe exceeds its total array-byte limit."
                )
            total_elements += elements
            total_bytes += byte_count
            nodes.append(dict(node))
            ordinals.append(len(nodes) - 1)
            completed.append(len(nodes) - 1)
        elif kind in _COMPOSITE_KINDS:
            children = _admit_composite_node(node, kind, path)
            if kind in ("set", "frozenset", "mapping", "frozendict", "mappingproxy"):
                keys = children if kind in ("set", "frozenset") else children[::2]
                payloads = [_canonical_recipe_payload(key) for key in keys]
                if payloads != sorted(payloads) or len(payloads) != len(set(payloads)):
                    raise ValueError(
                        f"{path} set items or mapping keys are not uniquely canonical."
                    )
            pending.append((node, path, shared, len(completed)))
            pending.extend(
                (child, f"{path}{label}", shared and child_shared, -1)
                for child, (label, child_shared) in reversed(
                    list(
                        zip(
                            children,
                            _child_labels(node),
                            strict=True,
                        )
                    )
                )
            )
        else:
            _admit_static_node(node, kind, limits, path)
            payload = _canonical_recipe_payload(node)
            if payload not in statics:
                nodes.append(dict(node))
                statics[payload] = len(nodes) - 1
            completed.append(statics[payload])
    if completed != [len(nodes) - 1]:
        raise ValueError("Model artifact recipe root must be one complete object.")
    table = _RecipeTable(nodes, ordinals)
    for index, node, path in values:
        _require_canonical_value(table, index, node, path)
    return table


def _require_canonical_value(
    table: _RecipeTable,
    index: int,
    node: Mapping[str, Any],
    path: str,
    /,
) -> None:
    reached = {index}
    for position in range(index, -1, -1):
        if position in reached:
            reached.update(_table_children(table.nodes[position]))

    def refuse_array(_index: int, _node: Mapping[str, Any], /) -> NoReturn:
        raise ValueError(f"{path} immutable value payload cannot contain arrays.")

    value = _restore_nodes(table.nodes, refuse_array, sorted(reached))[index]
    if _canonical_recipe_payload(
        model_structure_recipe(value)
    ) != _canonical_recipe_payload(node):
        raise ValueError(f"{path} immutable value payload is not canonical.")


def validate_model_structure_recipe(
    recipe: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> None:
    """Validate a complete recipe and every allocation bound without allocating."""
    _lower_recipe(recipe, limits)


def _admit_wire_table(
    text: Any, limits: ArrayArchiveLimits, /
) -> list[Mapping[str, Any]]:
    """Admit wire JSON shape, child order and reachability before lifting."""
    if not isinstance(text, str):
        raise TypeError("Model recipe wire encoding must be text.")
    if len(text.encode("utf-8")) > limits.max_manifest_bytes:
        raise ValueError("Model recipe wire encoding exceeds its manifest byte limit.")
    try:
        _validate_json_nesting(text, limits.max_manifest_nesting)
        table = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (
        ArrayArchiveCorruptionError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as error:
        raise ValueError(
            "Model recipe wire encoding is not bounded duplicate-free JSON."
        ) from error
    if not isinstance(table, dict):
        raise TypeError("Model recipe wire encoding must be a node-table object.")
    _recipe_fields(table, {"nodes", "root"}, "recipe")
    nodes, root = table["nodes"], table["root"]
    if not isinstance(nodes, list) or not nodes:
        raise TypeError("Model recipe wire nodes must be a nonempty list.")
    if type(root) is not int or root != len(nodes) - 1:
        raise ValueError("Model recipe wire root must be its final node.")
    for index, node in enumerate(nodes):
        path = f"recipe.nodes[{index}]"
        if not isinstance(node, dict) or not isinstance(node.get("kind"), str):
            raise TypeError(f"{path} must be a recipe object with a kind.")
        if node["kind"] in _COMPOSITE_KINDS:
            for child in _admit_composite_node(node, node["kind"], path):
                # Children strictly precede parents: no cycle or forward reference.
                if type(child) is not int or not 0 <= child < index:
                    raise ValueError(f"{path} must reference an earlier recipe node.")
        elif node["kind"] == "reference":
            raise ValueError(f"{path} wire tables express sharing by node index only.")
    reachable = [False] * len(nodes)
    reachable[root] = True
    for index in range(root, -1, -1):
        if reachable[index]:
            for child in _table_children(nodes[index]):
                reachable[child] = True
    if not all(reachable):
        raise ValueError(
            "Model recipe wire table contains a node unreachable from its root."
        )
    return nodes


def _lift_wire_table(
    nodes: list[Mapping[str, Any]],
    limits: ArrayArchiveLimits,
    /,
) -> dict[str, Any]:
    """Rebuild the logical recipe from an admitted table in bounded space.

    Only array and dataclass nodes completed in a shared context may be reached
    again, and every later reach becomes a logical reference. Childless static
    nodes may be reached repeatedly, but each reach contributes its full record
    to the logical JSON byte limit. Refuse that expansion before serializing it.
    Any other repeated reach is noncanonical sharing.
    """
    ordinals: dict[int, int] = {}
    visited: set[int] = set()
    completed: list[Any] = []
    pending: list[tuple[int, bool, int]] = [(len(nodes) - 1, True, -1)]
    static_sizes: dict[int, int] = {}
    remaining_static_bytes = limits.max_manifest_bytes
    while pending:
        index, shared, base = pending.pop()
        node = nodes[index]
        kind = node["kind"]
        if base >= 0:
            children = completed[base:]
            del completed[base:]
            completed.append(_with_children(node, children))
            if kind == "dataclass" and shared:
                ordinals[index] = len(ordinals)
            continue
        if shared and index in ordinals:
            completed.append({"kind": "reference", "target": ordinals[index]})
            continue
        if kind not in _COMPOSITE_KINDS and kind not in _ARRAY_KINDS:
            if index not in static_sizes:
                static_sizes[index] = len(_canonical_recipe_payload(node))
            remaining_static_bytes -= static_sizes[index]
            if remaining_static_bytes < 0:
                raise ValueError("Model artifact recipe exceeds its manifest byte limit.")
            completed.append(dict(node))
            continue
        if index in visited:
            raise ValueError(
                "Model recipe wire table shares a node outside canonical object identity."
            )
        visited.add(index)
        if kind in _ARRAY_KINDS:
            if not shared:
                raise ValueError(
                    "Model recipe wire table holds an array in an unshared context."
                )
            ordinals[index] = len(ordinals)
            completed.append(dict(node))
            continue
        pending.append((index, shared, len(completed)))
        pending.extend(
            (child, shared and child_shared, -1)
            for child, (_, child_shared) in reversed(
                list(
                    zip(
                        _table_children(node),
                        _child_labels(node),
                        strict=True,
                    )
                )
            )
        )
    (recipe,) = completed
    return recipe


def model_recipe_wire_json(
    recipe: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> str:
    """Encode an admitted logical recipe as its shallow canonical wire table.

    The wire table has constant JSON nesting however deep the logical recipe
    is: shared objects appear once and references become node indices. It is
    serialization only; the logical recipe and every identity derived from it
    are unchanged.
    """
    table = _lower_recipe(recipe, limits)
    return _canonical_recipe_payload(
        {"nodes": table.nodes, "root": len(table.nodes) - 1}
    ).decode("ascii")


def model_recipe_from_wire_json(
    text: str,
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> dict[str, Any]:
    """Admit wire text under byte and nesting limits; return its exact logical recipe."""
    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be an ArrayArchiveLimits instance.")
    recipe = _lift_wire_table(_admit_wire_table(text, limits), limits)
    if model_recipe_wire_json(recipe, limits=limits) != text:
        raise ValueError("Model recipe wire encoding is not canonical.")
    return recipe


def _restore_dataclass(
    cls: type[DataclassInstance], items: list[int], restored: list[Any], /
) -> Any:
    if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
        raise TypeError(
            "Artifact reconstruction requires an exactly registered dataclass."
        )
    if artifact_value(artifact_value_id(cls)) is not cls:
        raise TypeError("Artifact reconstruction requires its exact registered type.")
    declared = dataclasses.fields(cls)
    if len(items) != len(declared):
        raise ValueError("Artifact field count does not match its registered type.")
    instance = object.__new__(cls)
    for field, item in zip(declared, items, strict=True):
        object.__setattr__(instance, field.name, restored[item])
    if issubclass(cls, Strict):
        mark_strict_initialized(instance)
    return instance


def _restore_nodes(
    nodes: list[_RecipeNode],
    array_factory: _ArrayFactory,
    indices: Iterable[int] | None = None,
    /,
) -> list[Any]:
    """Restore admitted table nodes in order, building each node exactly once.

    Children precede parents, so every path to a shared node receives the
    identical restored array or object. ``array_factory`` is called once per
    array node, in logical restoration order.
    """
    restored: list[Any] = [None] * len(nodes)
    for index in range(len(nodes)) if indices is None else indices:
        restored[index] = _restore_node(nodes[index], index, restored, array_factory)
    return restored


def _restore_node(
    node: Mapping[str, Any],
    index: int,
    restored: list[Any],
    array_factory: _ArrayFactory,
    /,
) -> Any:
    match node["kind"]:
        case "array" | "prng_key":
            return array_factory(index, node)
        case "literal":
            return node["value"]
        case "nonfinite_float":
            return float(node["value"])
        case "fraction":
            return Fraction(node["numerator"], node["denominator"])
        case "fraction_tuple":
            return tuple(
                Fraction(numerator, denominator)
                for numerator, denominator in node["values"]
            )
        case "literal_tuple":
            return tuple(node["values"])
        case "complex":
            return complex(node["real"], node["imag"])
        case "dtype":
            return np.dtype(node["value"])
        case "bytes":
            return base64.b64decode(node["value"].encode("ascii"), validate=True)
        case "path":
            return Path(node["value"])
        case "ellipsis":
            return Ellipsis
        case "enum":
            return artifact_value(node["type"])[node["name"]]
        case "type" | "callable":
            return artifact_value(node["value"])
        case "slice":
            return slice(
                restored[node["start"]], restored[node["stop"]], restored[node["step"]]
            )
        case "value":
            return artifact_value_codec(node["type"]).decode_value(
                restored[node["payload"]]
            )
        case "namedtuple":
            return artifact_value(node["type"])(
                *(restored[item] for item in node["items"])
            )
        case "dataclass":
            return _restore_dataclass(
                artifact_value(node["type"]), node["items"], restored
            )
        case "tuple":
            return tuple(restored[item] for item in node["items"])
        case "list":
            return [restored[item] for item in node["items"]]
        case "set":
            return {restored[item] for item in node["items"]}
        case "frozenset":
            return frozenset(restored[item] for item in node["items"])
        case "mapping" | "frozendict" | "mappingproxy" as kind:
            value = {restored[key]: restored[item] for key, item in node["items"]}
            if kind == "frozendict":
                return frozendict(value)
            if kind == "mappingproxy":
                return MappingProxyType(value)
            return value
        case kind:
            raise ValueError(f"Unknown model artifact recipe kind {kind!r}.")


def _array_descriptor(node: Mapping[str, Any], /) -> _RecipeArrayDescriptor:
    if node["kind"] == "prng_key":
        return _RecipeArrayDescriptor(
            tuple(node["shape"]),
            np.dtype(np.uint32).str,
            "prng_key",
            node["implementation"],
        )
    return _RecipeArrayDescriptor(
        tuple(node["shape"]),
        np.dtype(node["dtype"]).str,
        node["backend"],
        writable=node["writable"] if node["backend"] == "numpy" else True,
    )


def model_recipe_template(
    recipe: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> Any:
    """Construct a nonallocating structural template for serialized leaves."""

    nodes = _lower_recipe(recipe, limits).nodes
    return _restore_nodes(nodes, lambda _index, node: _array_descriptor(node))[-1]


def _serialized_leaf_specification(
    value: Any,
    /,
) -> tuple[tuple[int, ...], np.dtype[Any]] | None:
    if isinstance(value, _RecipeArrayDescriptor):
        return value.shape, np.dtype(value.dtype)
    if _is_prng_key(value):
        data = jr.key_data(value)
        return tuple(data.shape), np.dtype(np.uint32)
    if eqx.is_array_like(value):
        if isinstance(value, (jax.Array, np.ndarray)):
            return tuple(value.shape), np.dtype(value.dtype)
        scalar = np.asarray(value)
        return tuple(scalar.shape), scalar.dtype
    return None


def preflight_model_tree_serialization(file: Any, template: Any, /) -> None:
    """Bind every serialized NPY leaf to a template before device allocation."""

    start = file.tell()
    file.seek(0, 2)
    payload_end = file.tell()
    file.seek(start)
    for leaf in jax.tree_util.tree_leaves(template):
        specification = _serialized_leaf_specification(leaf)
        if specification is None:
            continue
        shape, dtype = specification
        leaf_end = _preflight_serialized_leaf(file, shape, dtype)
        if leaf_end > payload_end:
            raise ValueError("Serialized model leaf payload is truncated.")
        file.seek(leaf_end)
    if file.tell() != payload_end:
        raise ValueError(
            "Serialized model leaf inventory has trailing or missing payloads."
        )
    file.seek(start)


def _host_leaf_payload(value: jax.Array | np.ndarray, /) -> np.ndarray:
    return np.asarray(jr.key_data(value) if _is_prng_key(value) else value)


def deserialize_model_tree(
    file: BinaryIO,
    recipe: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> Any:
    """Restore a recipe from its per-leaf NPY stream, binding shared objects once.

    The stream keeps one payload per PyTree leaf, so an array shared by several
    leaf paths repeats and every repetition must be bitwise equal. Arrays outside
    the runtime PyTree keep their zero template values, and serialized scalar
    leaves must equal their recipe literals.
    """
    nodes = _lower_recipe(recipe, limits).nodes
    zeros: dict[int, jax.Array | np.ndarray] = {}

    def zero_array(index: int, node: Mapping[str, Any], /) -> jax.Array | np.ndarray:
        zeros[index] = _zero_array(node)
        return zeros[index]

    template = _restore_nodes(nodes, zero_array)[-1]
    stored: dict[int, Any] = {}
    for leaf in jax.tree_util.tree_leaves(template):
        value = deserialize_model_leaf(file, leaf)
        if not isinstance(leaf, (jax.Array, np.ndarray)):
            if value is not leaf and _canonical_recipe_payload(
                model_structure_recipe(value)
            ) != _canonical_recipe_payload(model_structure_recipe(leaf)):
                raise ValueError(
                    "A serialized static model leaf differs from its recipe."
                )
            continue
        first = stored.setdefault(id(leaf), value)
        if first is not value:
            expected, actual = _host_leaf_payload(first), _host_leaf_payload(value)
            if (
                expected.shape != actual.shape
                or expected.dtype != actual.dtype
                or expected.tobytes() != actual.tobytes()
            ):
                raise ValueError(
                    "A shared model array has inconsistent serialized payloads."
                )
    restored = _restore_nodes(
        nodes,
        lambda index, _node: stored.get(id(zeros[index]), zeros[index]),
    )[-1]
    validate_tree(restored)
    return restored


def _first_paths(nodes: list[_RecipeNode], /) -> dict[int, str]:
    """Declared path of every node at its first canonical pre-order occurrence."""
    paths: dict[int, str] = {}
    pending = [(len(nodes) - 1, "model")]
    while pending:
        index, path = pending.pop()
        if index in paths:
            continue
        paths[index] = path
        node = nodes[index]
        pending.extend(
            (child, f"{path}{label}")
            for child, (label, _) in reversed(
                list(
                    zip(
                        _table_children(node),
                        _child_labels(node),
                        strict=True,
                    )
                )
            )
        )
    return paths


def _model_array_catalog(
    recipe: Mapping[str, Any],
    /,
    *,
    prefix: str,
    limits: ArrayArchiveLimits,
) -> tuple[_RecipeTable, tuple[tuple[ModelRecipeArray, int], ...]]:
    """Name every distinct array node: runtime leaves first, then static fields."""
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("Recipe array prefix must be non-empty.")
    table = _lower_recipe(recipe, limits)
    descriptors: dict[int, int] = {}

    def descriptor(index: int, node: Mapping[str, Any], /) -> _RecipeArrayDescriptor:
        value = _array_descriptor(node)
        descriptors[id(value)] = index
        return value

    restored = _restore_nodes(table.nodes, descriptor)
    ordered: list[tuple[int, int, str]] = []
    visited: set[int] = set()
    for tree_index, (path, leaf) in enumerate(
        jax.tree_util.tree_flatten_with_path(restored[-1])[0]
    ):
        if not isinstance(leaf, _RecipeArrayDescriptor):
            continue
        index = descriptors[id(leaf)]
        if index not in visited:
            ordered.append((index, tree_index, jax.tree_util.keystr(path) or "<root>"))
            visited.add(index)
    paths = _first_paths(table.nodes)
    ordered.extend(
        (index, -1, paths[index])
        for index in table.ordinals
        if table.nodes[index]["kind"] in _ARRAY_KINDS and index not in visited
    )
    catalog: list[tuple[ModelRecipeArray, int]] = []
    for position, (index, tree_index, display_path) in enumerate(ordered):
        declared = _array_descriptor(table.nodes[index])
        catalog.append(
            (
                ModelRecipeArray(
                    name=f"{prefix}/{position:06d}",
                    tree_index=tree_index,
                    path=display_path,
                    shape=declared.shape,
                    dtype=declared.dtype,
                    backend=declared.backend,
                    implementation=declared.implementation,
                ),
                index,
            )
        )
    return table, tuple(catalog)


def model_recipe_array_inventory(
    recipe: Mapping[str, Any],
    /,
    *,
    prefix: str,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[ModelRecipeArray, ...]:
    """Inventory every scientific array without changing runtime PyTree leaves."""
    _, catalog = _model_array_catalog(recipe, prefix=prefix, limits=limits)
    return tuple(entry for entry, _ in catalog)


def model_recipe_array_values(
    value: Any,
    recipe: Mapping[str, Any],
    /,
    *,
    prefix: str,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> dict[str, jax.Array | np.ndarray]:
    """Return exact numerical fields, retaining global shards and host backends."""
    table, catalog = _model_array_catalog(recipe, prefix=prefix, limits=limits)
    return {
        entry.name: payload
        for (entry, _), payload in zip(
            catalog, _cataloged_array_values(value, recipe, table, catalog), strict=True
        )
    }


def _table_objects(
    value: Any, recipe: Mapping[str, Any], table: _RecipeTable, /
) -> dict[int, Any]:
    """Bind every shared-capable node of an admitted recipe to the live object."""
    shared: _SharedObjects = {}
    encoded = _structure_recipe(value, "model", shared)
    if _canonical_recipe_payload(encoded) != _canonical_recipe_payload(recipe):
        raise TypeError(
            "Value metadata and registered source structure differ from its recipe."
        )
    return {table.ordinals[ordinal]: item for ordinal, item in shared.values()}


def _cataloged_array_values(
    value: Any,
    recipe: Mapping[str, Any],
    table: _RecipeTable,
    catalog: tuple[tuple[ModelRecipeArray, int], ...],
    /,
) -> list[jax.Array | np.ndarray]:
    leaves = _table_objects(value, recipe, table)
    arrays: list[jax.Array | np.ndarray] = []
    for entry, index in catalog:
        leaf = leaves[index]
        if entry.backend == "prng_key":
            if not _is_prng_key(leaf) or str(jr.key_impl(leaf)) != entry.implementation:
                raise TypeError(
                    "Scientific PRNG data differs from its declared implementation."
                )
            payload = jr.key_data(leaf)
        elif entry.backend == "jax":
            if not isinstance(leaf, jax.Array):
                raise TypeError(
                    "A JAX scientific source field changed its numerical backend."
                )
            payload = leaf
        else:
            if not isinstance(leaf, np.ndarray):
                raise TypeError(
                    "A NumPy scientific source field changed its numerical backend."
                )
            payload = leaf
        if payload.shape != entry.shape or payload.dtype.str != entry.dtype:
            raise TypeError(
                "Scientific source array shape or dtype differs from its recipe."
            )
        arrays.append(payload)
    return arrays


def _content_forms(nodes: list[_RecipeNode], /) -> list[bytes]:
    """Canonical content of every table node, independent of sharing.

    A childless node embeds its record; a composite child contributes its
    content digest, so all forms together stay linear in the table size.
    """
    forms: list[bytes] = []
    tokens: list[Any] = []
    for node in nodes:
        children = [tokens[index] for index in _table_children(node)]
        form = _canonical_recipe_payload(
            _with_children(node, children) if node["kind"] in _COMPOSITE_KINDS else node
        )
        forms.append(form)
        tokens.append(
            hashlib.sha256(form).hexdigest() if node["kind"] in _COMPOSITE_KINDS else node
        )
    return forms


def _paired_children(node: Mapping[str, Any], /) -> list[int]:
    """Children compared positionally; set items and payloads compare by content."""
    match node["kind"]:
        case "slice" | "dataclass" | "tuple" | "list" | "namedtuple":
            return _table_children(node)
        case "mapping" | "frozendict" | "mappingproxy":
            return [item for _, item in node["items"]]
        case _:
            return []


def _paired_metadata(
    nodes: list[_RecipeNode], forms: list[bytes], index: int, /
) -> bytes:
    # Object sharing is not compared; NumPy mutability is storage metadata.
    node = nodes[index]
    match node["kind"]:
        case "array" | "prng_key":
            return _canonical_recipe_payload(
                {name: item for name, item in node.items() if name != "writable"}
            )
        case "slice" | "dataclass" | "tuple" | "list" | "namedtuple":
            return _canonical_recipe_payload(
                {
                    "kind": node["kind"],
                    "type": node.get("type"),
                    "count": len(_paired_children(node)),
                }
            )
        case "mapping" | "frozendict" | "mappingproxy":
            return _canonical_recipe_payload(
                {
                    "kind": node["kind"],
                    "keys": [
                        hashlib.sha256(forms[key]).hexdigest() for key, _ in node["items"]
                    ],
                }
            )
        case _:
            return forms[index]


def model_recipe_array_pairs(
    first: Any,
    second: Any,
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[tuple[jax.Array | np.ndarray, jax.Array | np.ndarray], ...] | None:
    """Pair the arrays of two values whose unshared scientific structure agrees.

    Agreement is over the trees the recipes unfold to: object sharing and NumPy
    mutability are storage metadata and do not participate. Returns ``None`` when
    the structures differ; otherwise each distinct pair of corresponding arrays
    (PRNG keys as key data) once, in canonical first-correspondence order. Both
    recipes are bounded by ``limits`` over their distinct arrays.
    """
    sides: list[tuple[list[_RecipeNode], list[bytes], dict[int, Any]]] = []
    for value in (first, second):
        recipe = model_structure_recipe(value)
        table = _lower_recipe(recipe, limits)
        sides.append(
            (
                table.nodes,
                _content_forms(table.nodes),
                _table_objects(value, recipe, table),
            )
        )
    (left_nodes, left_forms, left_objects), (right_nodes, right_forms, right_objects) = (
        sides
    )
    pairs: dict[
        tuple[int, int], tuple[jax.Array | np.ndarray, jax.Array | np.ndarray]
    ] = {}
    visited: set[tuple[int, int]] = set()
    pending = [(len(left_nodes) - 1, len(right_nodes) - 1)]
    while pending:
        key = pending.pop()
        if key in visited:
            continue
        visited.add(key)
        left, right = key
        if _paired_metadata(left_nodes, left_forms, left) != _paired_metadata(
            right_nodes,
            right_forms,
            right,
        ):
            return None
        if left_nodes[left]["kind"] in _ARRAY_KINDS:
            pairs[key] = (
                _paired_payload(left_objects[left]),
                _paired_payload(right_objects[right]),
            )
            continue
        pending.extend(
            reversed(
                list(
                    zip(
                        _paired_children(left_nodes[left]),
                        _paired_children(right_nodes[right]),
                        strict=True,
                    )
                )
            )
        )
    return tuple(pairs.values())


def _paired_payload(value: jax.Array | np.ndarray, /) -> jax.Array | np.ndarray:
    return jr.key_data(value) if _is_prng_key(value) else value


def pack_model_array_tree(
    value: Any,
    recipe: Mapping[str, Any],
    /,
    *,
    prefix: str,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> dict[str, np.ndarray]:
    """Extract the exact dynamic arrays declared by a canonical recipe."""

    values = model_recipe_array_values(value, recipe, prefix=prefix, limits=limits)
    if any(
        isinstance(value, jax.Array) and not value.is_fully_addressable
        for value in values.values()
    ):
        raise ValueError(
            "Host array packing requires explicit materialization; use logical source array values for shards."
        )
    return {name: np.asarray(value) for name, value in values.items()}


def _preflight_recipe_arrays(
    catalog: tuple[tuple[ModelRecipeArray, int], ...],
    arrays: Mapping[str, Any],
    /,
    *,
    host_materialization: bool = False,
) -> None:
    if set(arrays) != {entry.name for entry, _ in catalog}:
        raise ValueError(
            "Source recipe numerical data does not cover its exact inventory."
        )
    for entry, _ in catalog:
        payload = arrays[entry.name]
        if not isinstance(payload, (jax.Array, np.ndarray)) or (
            payload.shape != entry.shape or payload.dtype.str != entry.dtype
        ):
            raise ValueError(
                "Source array shape/dtype is inconsistent with its exact recipe."
            )
        if (
            isinstance(payload, jax.Array)
            and not payload.is_fully_addressable
            and (host_materialization or entry.backend == "numpy")
        ):
            raise ValueError(
                "Host recipe restoration requires explicit bounded materialization."
            )


def model_from_array_recipe(
    recipe: Mapping[str, Any],
    arrays: Mapping[str, Any],
    /,
    *,
    prefix: str,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> Any:
    """Restore a model recipe from an exact, preflighted host-array inventory."""

    _, catalog = _model_array_catalog(recipe, prefix=prefix, limits=limits)
    _preflight_recipe_arrays(catalog, arrays, host_materialization=True)
    host = {name: np.asarray(value) for name, value in arrays.items()}
    return model_from_logical_array_recipe(recipe, host, prefix=prefix, limits=limits)


def model_from_logical_array_recipe(
    recipe: Mapping[str, Any],
    arrays: Mapping[str, jax.Array | np.ndarray],
    /,
    *,
    prefix: str,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> Any:
    """Restore every exact registered field with deliberate host/device placement."""
    table, catalog = _model_array_catalog(recipe, prefix=prefix, limits=limits)
    _preflight_recipe_arrays(catalog, arrays)
    entries = {index: entry for entry, index in catalog}

    def restore_array(index: int, node: Mapping[str, Any], /) -> jax.Array | np.ndarray:
        entry = entries[index]
        payload = arrays[entry.name]
        if entry.backend == "prng_key":
            return jr.wrap_key_data(
                jnp.asarray(payload, dtype=jnp.uint32), impl=entry.implementation
            )
        if entry.backend == "jax":
            return payload if isinstance(payload, jax.Array) else jnp.asarray(payload)
        host = np.array(payload, copy=True, order="C")
        host.setflags(write=node["writable"])
        return host

    restored = _restore_nodes(table.nodes, restore_array)[-1]
    validate_tree(restored)
    return restored


def _zero_array(node: Mapping[str, Any], /) -> jax.Array | np.ndarray:
    if node["kind"] == "prng_key":
        return jr.wrap_key_data(
            jnp.zeros(tuple(node["shape"]), dtype=jnp.uint32),
            impl=node["implementation"],
        )
    dtype = np.dtype(node["dtype"])
    if node["backend"] == "jax":
        return jnp.zeros(tuple(node["shape"]), dtype=dtype)
    host = np.zeros(tuple(node["shape"]), dtype=dtype)
    host.setflags(write=node["writable"])
    return host


def model_from_structure_recipe(
    recipe: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> Any:
    """Construct a bounded array-zeroed template from a validated recipe."""

    nodes = _lower_recipe(recipe, limits).nodes
    restored = _restore_nodes(nodes, lambda _index, node: _zero_array(node))[-1]
    validate_tree(restored)
    return restored


__all__ = [
    "deserialize_model_leaf",
    "deserialize_model_tree",
    "model_from_array_recipe",
    "model_from_logical_array_recipe",
    "model_from_structure_recipe",
    "model_recipe_from_wire_json",
    "model_recipe_template",
    "model_recipe_array_values",
    "model_recipe_array_inventory",
    "model_recipe_array_pairs",
    "model_recipe_wire_json",
    "ModelRecipeArray",
    "model_structure_recipe",
    "pack_model_array_tree",
    "serialize_model_leaf",
    "preflight_model_tree_serialization",
    "validate_model_structure_recipe",
]
