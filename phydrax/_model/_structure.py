#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import base64
import dataclasses
import enum
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .._array_archive import (
    _read_npy_metadata,
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
)
from .._frozendict import frozendict
from .._strict import Strict
from ._artifacts import artifact_value, artifact_value_id


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
        if kind in ("array", "prng_key"):
            return True
        if kind == "slice":
            pending.extend((node["start"], node["stop"], node["step"]))
        elif kind == "dataclass":
            pending.extend(node["fields"].values())
        elif kind in ("tuple", "list", "set", "frozenset", "namedtuple"):
            pending.extend(node["items"])
        elif kind in ("mapping", "frozendict", "mappingproxy"):
            pending.extend(item for pair in node["items"] for item in pair)
    return False


def serialize_model_leaf(file, value: Any, /) -> None:
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


def deserialize_model_leaf(file, value: Any, /) -> Any:
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
        return np.array(payload, copy=True, order="C")
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
    return eqx.default_deserialise_filter_spec(file, value)


def model_structure_recipe(
    value: Any,
    /,
    *,
    path: str = "model",
) -> dict[str, Any]:
    """Encode static model structure without embedding Python module paths."""
    if _is_prng_key(value):
        data = np.asarray(jr.key_data(value))
        return {
            "kind": "prng_key",
            "shape": list(data.shape),
            "implementation": str(jr.key_impl(value)),
        }
    if isinstance(value, (jax.Array, np.ndarray)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"{path} contains an object-dtype array.")
        return {
            "kind": "array",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "backend": "jax" if isinstance(value, jax.Array) else "numpy",
        }
    if isinstance(value, enum.Enum):
        return {
            "kind": "enum",
            "type": artifact_value_id(type(value)),
            "name": value.name,
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{path} contains a non-finite static float.")
        return {"kind": "literal", "value": value}
    if isinstance(value, complex):
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError(f"{path} contains a non-finite static complex value.")
        return {"kind": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return model_structure_recipe(value.item(), path=path)
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
            "start": model_structure_recipe(value.start, path=f"{path}.start"),
            "stop": model_structure_recipe(value.stop, path=f"{path}.stop"),
            "step": model_structure_recipe(value.step, path=f"{path}.step"),
        }
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if isinstance(value, type):
        return {"kind": "type", "value": artifact_value_id(value)}
    if isinstance(value, tuple) and "_fields" in vars(type(value)):
        return {
            "kind": "namedtuple",
            "type": artifact_value_id(type(value)),
            "items": [
                model_structure_recipe(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if dataclasses.is_dataclass(value):
        return {
            "kind": "dataclass",
            "type": artifact_value_id(type(value)),
            "fields": {
                field.name: model_structure_recipe(
                    object.__getattribute__(value, field.name),
                    path=f"{path}.{field.name}",
                )
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                model_structure_recipe(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                model_structure_recipe(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, (set, frozenset)):
        encoded_items = [
            model_structure_recipe(item, path=f"{path}.item") for item in value
        ]
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
        encoded_items = [
            (
                model_structure_recipe(key, path=f"{path}.key"),
                model_structure_recipe(item, path=f"{path}.value"),
            )
            for key, item in value.items()
        ]
        encoded_items.sort(key=lambda pair: _canonical_recipe_payload(pair[0]))
        key_payloads = [_canonical_recipe_payload(key) for key, _ in encoded_items]
        if len(set(key_payloads)) != len(key_payloads):
            raise TypeError(f"{path} contains keys with the same canonical encoding.")
        return {
            "kind": mapping_kind,
            "items": [[key, item] for key, item in encoded_items],
        }
    if callable(value):
        return {"kind": "callable", "value": artifact_value_id(value)}
    raise TypeError(
        f"Portable model artifact cannot represent {path} of type {type(value).__name__}."
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ModelRecipeArray:
    """One exact dynamic array leaf declared by a model structure recipe."""

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


def validate_model_structure_recipe(
    recipe: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> None:
    """Validate a complete recipe and every allocation bound without allocating."""

    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be an ArrayArchiveLimits instance.")
    pending: list[tuple[Any, int, str]] = [(recipe, 1, "recipe")]
    canonical_sets: list[tuple[str, list[Mapping[str, Any]]]] = []
    canonical_mappings: list[tuple[str, list[Mapping[str, Any]]]] = []
    total_elements = 0
    total_bytes = 0
    node_count = 0
    while pending:
        node, depth, path = pending.pop()
        node_count += 1
        if node_count > limits.max_manifest_bytes:
            raise ValueError("Model artifact recipe exceeds its node limit.")
        if depth > limits.max_manifest_nesting:
            raise ValueError("Model artifact recipe exceeds its nesting limit.")
        if not isinstance(node, dict):
            raise TypeError(f"{path} must be a recipe object.")
        kind = node.get("kind")
        if not isinstance(kind, str):
            raise ValueError(f"{path} has no valid recipe kind.")

        if kind == "array":
            _recipe_fields(node, {"kind", "shape", "dtype", "backend"}, path)
            if node["backend"] not in ("jax", "numpy"):
                raise ValueError(f"{path} has an invalid array backend.")
            _, _, elements, byte_count = _recipe_array_spec(
                node["shape"], node["dtype"], limits, path
            )
        elif kind == "prng_key":
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
        else:
            elements = 0
            byte_count = 0
        if elements:
            if total_elements > limits.max_total_array_elements - elements:
                raise ValueError("Model artifact recipe exceeds its total element limit.")
            if total_bytes > limits.max_aggregate_bytes - byte_count:
                raise ValueError(
                    "Model artifact recipe exceeds its total array-byte limit."
                )
            total_elements += elements
            total_bytes += byte_count

        if kind in ("array", "prng_key"):
            continue
        if kind == "literal":
            _recipe_fields(node, {"kind", "value"}, path)
            value = node["value"]
            if value is not None and type(value) not in (str, bool, int, float):
                raise TypeError(f"{path} literal has an invalid type.")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError(f"{path} literal must be finite.")
            continue
        if kind == "complex":
            _recipe_fields(node, {"kind", "real", "imag"}, path)
            components = (node["real"], node["imag"])
            if any(
                type(component) not in (int, float) or not np.isfinite(component)
                for component in components
            ):
                raise ValueError(f"{path} complex components must be finite numbers.")
            continue
        if kind == "dtype":
            _recipe_fields(node, {"kind", "value"}, path)
            if not isinstance(node["value"], str):
                raise TypeError(f"{path} dtype value must be text.")
            _recipe_array_spec([], node["value"], limits, path)
            continue
        if kind == "bytes":
            _recipe_fields(node, {"kind", "value"}, path)
            if not isinstance(node["value"], str):
                raise TypeError(f"{path} bytes value must be base64 text.")
            try:
                decoded = base64.b64decode(node["value"].encode("ascii"), validate=True)
            except (UnicodeEncodeError, ValueError) as error:
                raise ValueError(f"{path} bytes value is invalid base64.") from error
            if len(decoded) > limits.max_member_bytes:
                raise ValueError(f"{path} bytes value exceeds the static-byte limit.")
            continue
        if kind == "path":
            _recipe_fields(node, {"kind", "value"}, path)
            if not isinstance(node["value"], str):
                raise TypeError(f"{path} path value must be text.")
            continue
        if kind == "slice":
            _recipe_fields(node, {"kind", "start", "stop", "step"}, path)
            pending.extend(
                (
                    (node["start"], depth + 1, f"{path}.start"),
                    (node["stop"], depth + 1, f"{path}.stop"),
                    (node["step"], depth + 1, f"{path}.step"),
                )
            )
            continue
        if kind == "ellipsis":
            _recipe_fields(node, {"kind"}, path)
            continue
        if kind == "enum":
            _recipe_fields(node, {"kind", "type", "name"}, path)
            if not isinstance(node["type"], str) or not isinstance(node["name"], str):
                raise TypeError(f"{path} enum identity must contain text.")
            enum_type = artifact_value(node["type"])
            if not isinstance(enum_type, type) or not issubclass(enum_type, enum.Enum):
                raise TypeError(f"{path} enum identity does not resolve to an enum.")
            if node["name"] not in enum_type.__members__:
                raise ValueError(f"{path} enum member is unknown.")
            continue
        if kind in ("type", "callable"):
            _recipe_fields(node, {"kind", "value"}, path)
            if not isinstance(node["value"], str):
                raise TypeError(f"{path} artifact identity must be text.")
            resolved = artifact_value(node["value"])
            if kind == "type" and not isinstance(resolved, type):
                raise TypeError(f"{path} does not resolve to a type.")
            if kind == "callable" and not callable(resolved):
                raise TypeError(f"{path} does not resolve to a callable.")
            continue
        if kind == "namedtuple":
            _recipe_fields(node, {"kind", "type", "items"}, path)
            cls = artifact_value(node["type"]) if isinstance(node["type"], str) else None
            if not isinstance(cls, type) or "_fields" not in vars(cls):
                raise TypeError(f"{path} identity does not resolve to a named tuple.")
            items = node["items"]
            if not isinstance(items, list) or len(items) != len(cls._fields):
                raise ValueError(f"{path} named-tuple fields are invalid.")
            pending.extend(
                (item, depth + 1, f"{path}.items[{index}]")
                for index, item in enumerate(items)
            )
            continue
        if kind == "dataclass":
            _recipe_fields(node, {"kind", "type", "fields"}, path)
            cls = artifact_value(node["type"]) if isinstance(node["type"], str) else None
            fields = node["fields"]
            if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
                raise TypeError(f"{path} identity does not resolve to a dataclass.")
            if not isinstance(fields, dict):
                raise TypeError(f"{path} dataclass fields must be an object.")
            expected = tuple(field.name for field in dataclasses.fields(cls))
            if set(fields) != set(expected):
                raise ValueError(f"{path} dataclass fields do not match their type.")
            pending.extend(
                (fields[name], depth + 1, f"{path}.fields.{name}") for name in expected
            )
            continue
        if kind in ("tuple", "list", "set", "frozenset"):
            _recipe_fields(node, {"kind", "items"}, path)
            items = node["items"]
            if not isinstance(items, list):
                raise TypeError(f"{path} items must be a list.")
            if kind in ("set", "frozenset"):
                canonical_sets.append((path, items))
            pending.extend(
                (item, depth + 1, f"{path}.items[{index}]")
                for index, item in enumerate(items)
            )
            continue
        if kind in ("mapping", "frozendict", "mappingproxy"):
            _recipe_fields(node, {"kind", "items"}, path)
            items = node["items"]
            if not isinstance(items, list) or any(
                not isinstance(pair, list) or len(pair) != 2 for pair in items
            ):
                raise TypeError(f"{path} mapping items must be key/value pairs.")
            keys = [pair[0] for pair in items]
            canonical_mappings.append((path, keys))
            for index, (key, value) in enumerate(items):
                pending.append((key, depth + 1, f"{path}.items[{index}].key"))
                pending.append((value, depth + 1, f"{path}.items[{index}].value"))
            continue
        raise ValueError(f"Unknown model artifact recipe kind {kind!r}.")

    for path, items in canonical_sets:
        payloads = [_canonical_recipe_payload(item) for item in items]
        if payloads != sorted(payloads) or len(payloads) != len(set(payloads)):
            raise ValueError(f"{path} set items are not uniquely canonical.")
        if any(_recipe_contains_array(item) for item in items):
            raise ValueError(f"{path} set items cannot contain array recipes.")
    for path, keys in canonical_mappings:
        payloads = [_canonical_recipe_payload(key) for key in keys]
        if payloads != sorted(payloads) or len(payloads) != len(set(payloads)):
            raise ValueError(f"{path} mapping keys are not uniquely canonical.")
        if any(_recipe_contains_array(key) for key in keys):
            raise ValueError(f"{path} mapping keys cannot contain array recipes.")


def _restore_dataclass(
    cls: type,
    fields: Mapping[str, Any],
    array_factory: Any,
    /,
) -> Any:
    instance = object.__new__(cls)
    for field in dataclasses.fields(cls):
        object.__setattr__(
            instance,
            field.name,
            _restore_recipe(fields[field.name], array_factory),
        )
    if issubclass(cls, Strict):
        object.__setattr__(instance, "_strict_initialized", True)
    return instance


def _restore_recipe(recipe: Mapping[str, Any], array_factory: Any, /) -> Any:
    kind = recipe["kind"]
    if kind in ("array", "prng_key"):
        return array_factory(recipe)
    if kind == "literal":
        return recipe["value"]
    if kind == "complex":
        return complex(recipe["real"], recipe["imag"])
    if kind == "dtype":
        return np.dtype(recipe["value"])
    if kind == "bytes":
        return base64.b64decode(recipe["value"].encode("ascii"), validate=True)
    if kind == "path":
        return Path(recipe["value"])
    if kind == "slice":
        return slice(
            _restore_recipe(recipe["start"], array_factory),
            _restore_recipe(recipe["stop"], array_factory),
            _restore_recipe(recipe["step"], array_factory),
        )
    if kind == "ellipsis":
        return Ellipsis
    if kind == "enum":
        return artifact_value(recipe["type"])[recipe["name"]]
    if kind in ("type", "callable"):
        return artifact_value(recipe["value"])
    if kind == "namedtuple":
        cls = artifact_value(recipe["type"])
        return cls(*(_restore_recipe(item, array_factory) for item in recipe["items"]))
    if kind == "dataclass":
        return _restore_dataclass(
            artifact_value(recipe["type"]), recipe["fields"], array_factory
        )
    if kind in ("tuple", "list", "set", "frozenset"):
        items = [_restore_recipe(item, array_factory) for item in recipe["items"]]
        if kind == "tuple":
            return tuple(items)
        if kind == "list":
            return items
        if kind == "set":
            return set(items)
        return frozenset(items)
    value = {
        _restore_recipe(key, array_factory): _restore_recipe(item, array_factory)
        for key, item in recipe["items"]
    }
    if kind == "frozendict":
        return frozendict(value)
    if kind == "mappingproxy":
        return MappingProxyType(value)
    return value


def _array_descriptor(recipe: Mapping[str, Any], /) -> _RecipeArrayDescriptor:
    if recipe["kind"] == "prng_key":
        return _RecipeArrayDescriptor(
            tuple(recipe["shape"]),
            np.dtype(np.uint32).str,
            "prng_key",
            recipe["implementation"],
        )
    return _RecipeArrayDescriptor(
        tuple(recipe["shape"]),
        np.dtype(recipe["dtype"]).str,
        recipe["backend"],
    )


def _descriptor_tree(
    recipe: Mapping[str, Any],
    limits: ArrayArchiveLimits,
    /,
) -> Any:
    validate_model_structure_recipe(recipe, limits=limits)
    return _restore_recipe(recipe, _array_descriptor)


def model_recipe_template(
    recipe: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> Any:
    """Construct a nonallocating structural template for serialized leaves."""

    return _descriptor_tree(recipe, limits)


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


def model_recipe_array_inventory(
    recipe: Mapping[str, Any],
    /,
    *,
    prefix: str,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[ModelRecipeArray, ...]:
    """Return the exact dynamic array inventory without allocating arrays."""

    if not isinstance(prefix, str) or not prefix:
        raise ValueError("Recipe array prefix must be non-empty.")
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(
        _descriptor_tree(recipe, limits)
    )
    inventory: list[ModelRecipeArray] = []
    for tree_index, (path, leaf) in enumerate(path_leaves):
        if not isinstance(leaf, _RecipeArrayDescriptor):
            continue
        inventory.append(
            ModelRecipeArray(
                name=f"{prefix}/{len(inventory):06d}",
                tree_index=tree_index,
                path=jax.tree_util.keystr(path) or "<root>",
                shape=leaf.shape,
                dtype=leaf.dtype,
                backend=leaf.backend,
                implementation=leaf.implementation,
            )
        )
    return tuple(inventory)


def pack_model_array_tree(
    value: Any,
    recipe: Mapping[str, Any],
    /,
    *,
    prefix: str,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> dict[str, np.ndarray]:
    """Extract the exact dynamic arrays declared by a canonical recipe."""

    descriptor_tree = _descriptor_tree(recipe, limits)
    descriptor_leaves, descriptor_definition = jax.tree_util.tree_flatten_with_path(
        descriptor_tree
    )
    value_leaves, value_definition = jax.tree_util.tree_flatten_with_path(value)
    if descriptor_definition != value_definition or len(descriptor_leaves) != len(
        value_leaves
    ):
        raise TypeError("Value PyTree does not match its model structure recipe.")
    inventory = model_recipe_array_inventory(recipe, prefix=prefix, limits=limits)
    arrays: dict[str, np.ndarray] = {}
    for entry in inventory:
        expected_path, descriptor = descriptor_leaves[entry.tree_index]
        actual_path, leaf = value_leaves[entry.tree_index]
        if expected_path != actual_path or not isinstance(
            descriptor, _RecipeArrayDescriptor
        ):
            raise TypeError("Value array paths do not match their model recipe.")
        if descriptor.backend == "prng_key":
            if not _is_prng_key(leaf):
                raise TypeError("Value PRNG leaf does not match its model recipe.")
            payload = np.asarray(jr.key_data(leaf))
            if str(jr.key_impl(leaf)) != descriptor.implementation:
                raise TypeError("Value PRNG implementation changed from its recipe.")
        else:
            if not isinstance(leaf, (jax.Array, np.ndarray)):
                raise TypeError("Value array leaf does not match its model recipe.")
            if descriptor.backend == "jax" and not isinstance(leaf, jax.Array):
                raise TypeError("Value array backend changed from its model recipe.")
            if descriptor.backend == "numpy" and not isinstance(leaf, np.ndarray):
                raise TypeError("Value array backend changed from its model recipe.")
            payload = np.asarray(leaf)
        if payload.shape != descriptor.shape or payload.dtype.str != descriptor.dtype:
            raise TypeError("Value array shape or dtype changed from its model recipe.")
        arrays[entry.name] = payload
    return arrays


def model_from_array_recipe(
    recipe: Mapping[str, Any],
    arrays: Mapping[str, Any],
    /,
    *,
    prefix: str,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> Any:
    """Restore a model recipe from an exact, preflighted host-array inventory."""

    descriptor_tree = _descriptor_tree(recipe, limits)
    path_leaves, definition = jax.tree_util.tree_flatten_with_path(descriptor_tree)
    inventory = model_recipe_array_inventory(recipe, prefix=prefix, limits=limits)
    if set(arrays) != {entry.name for entry in inventory}:
        raise ValueError("Model recipe arrays do not match its exact leaf inventory.")
    admitted: dict[str, np.ndarray] = {}
    for entry in inventory:
        payload = np.asarray(arrays[entry.name])
        if payload.shape != entry.shape or payload.dtype.str != entry.dtype:
            raise ValueError("Model recipe array shape or dtype is inconsistent.")
        admitted[entry.name] = payload

    restored = [leaf for _, leaf in path_leaves]
    for entry in inventory:
        descriptor = restored[entry.tree_index]
        if not isinstance(descriptor, _RecipeArrayDescriptor):
            raise RuntimeError("Model recipe inventory lost its array descriptor.")
        payload = admitted[entry.name]
        if entry.backend == "prng_key":
            restored[entry.tree_index] = jr.wrap_key_data(
                jnp.asarray(payload, dtype=jnp.uint32),
                impl=entry.implementation,
            )
        elif entry.backend == "jax":
            restored[entry.tree_index] = jnp.asarray(payload)
        else:
            restored[entry.tree_index] = np.array(payload, copy=True, order="C")
    return jax.tree_util.tree_unflatten(definition, restored)


def model_from_structure_recipe(
    recipe: Mapping[str, Any],
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> Any:
    """Construct a bounded array-zeroed template from a validated recipe."""

    validate_model_structure_recipe(recipe, limits=limits)

    def zeros(node: Mapping[str, Any], /) -> Any:
        if node["kind"] == "prng_key":
            return jr.wrap_key_data(
                jnp.zeros(tuple(node["shape"]), dtype=jnp.uint32),
                impl=node["implementation"],
            )
        dtype = np.dtype(node["dtype"])
        if node["backend"] == "jax":
            return jnp.zeros(tuple(node["shape"]), dtype=dtype)
        return np.zeros(tuple(node["shape"]), dtype=dtype)

    return _restore_recipe(recipe, zeros)


__all__ = [
    "deserialize_model_leaf",
    "model_from_array_recipe",
    "model_from_structure_recipe",
    "model_recipe_template",
    "model_recipe_array_inventory",
    "ModelRecipeArray",
    "model_structure_recipe",
    "pack_model_array_tree",
    "serialize_model_leaf",
    "preflight_model_tree_serialization",
    "validate_model_structure_recipe",
]
