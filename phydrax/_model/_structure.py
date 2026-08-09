#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import base64
import dataclasses
import enum
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .._frozendict import frozendict
from ._artifacts import artifact_value, artifact_value_id


def _is_prng_key(value: Any, /) -> bool:
    return isinstance(value, jax.Array) and jax.dtypes.issubdtype(
        value.dtype, jax.dtypes.prng_key
    )


def serialise_model_leaf(file, value: Any, /) -> None:
    """Serialise one model leaf, preserving typed JAX PRNG keys."""
    if _is_prng_key(value):
        np.save(file, np.asarray(jr.key_data(value)))
        return
    eqx.default_serialise_filter_spec(file, value)


def deserialise_model_leaf(file, value: Any, /) -> Any:
    """Deserialise one model leaf against its structural template."""
    if _is_prng_key(value):
        data = jnp.asarray(np.load(file), dtype=jnp.uint32)
        return jr.wrap_key_data(data, impl=str(jr.key_impl(value)))
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
    if isinstance(value, enum.Enum):
        return {
            "kind": "enum",
            "type": artifact_value_id(type(value)),
            "name": value.name,
        }
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
        items = sorted(value, key=repr)
        return {
            "kind": "frozenset" if isinstance(value, frozenset) else "set",
            "items": [
                model_structure_recipe(item, path=f"{path}[{index}]")
                for index, item in enumerate(items)
            ],
        }
    if isinstance(value, Mapping):
        mapping_kind = (
            "frozendict"
            if isinstance(value, frozendict)
            else "mappingproxy"
            if isinstance(value, MappingProxyType)
            else "mapping"
        )
        return {
            "kind": mapping_kind,
            "items": [
                [
                    model_structure_recipe(key, path=f"{path}.key"),
                    model_structure_recipe(item, path=f"{path}[{key!r}]"),
                ]
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            ],
        }
    if callable(value):
        return {"kind": "callable", "value": artifact_value_id(value)}
    raise TypeError(
        f"Portable model artifact cannot represent {path} of type {type(value).__name__}."
    )


def model_from_structure_recipe(recipe: Mapping[str, Any], /) -> Any:
    """Construct an array-zeroed model template from a structure recipe."""
    kind = recipe["kind"]
    if kind == "array":
        return jnp.zeros(tuple(recipe["shape"]), dtype=np.dtype(recipe["dtype"]))
    if kind == "prng_key":
        return jr.wrap_key_data(
            jnp.zeros(tuple(recipe["shape"]), dtype=jnp.uint32),
            impl=recipe["implementation"],
        )
    if kind == "literal":
        return recipe["value"]
    if kind == "complex":
        return complex(recipe["real"], recipe["imag"])
    if kind == "dtype":
        return np.dtype(recipe["value"])
    if kind == "bytes":
        return base64.b64decode(recipe["value"].encode("ascii"))
    if kind == "path":
        return Path(recipe["value"])
    if kind == "slice":
        return slice(
            model_from_structure_recipe(recipe["start"]),
            model_from_structure_recipe(recipe["stop"]),
            model_from_structure_recipe(recipe["step"]),
        )
    if kind == "ellipsis":
        return Ellipsis
    if kind == "enum":
        return artifact_value(recipe["type"])[recipe["name"]]
    if kind in ("type", "callable"):
        return artifact_value(recipe["value"])
    if kind == "namedtuple":
        cls = artifact_value(recipe["type"])
        return cls(*(model_from_structure_recipe(item) for item in recipe["items"]))
    if kind == "dataclass":
        cls = artifact_value(recipe["type"])
        instance = object.__new__(cls)
        for name, value in recipe["fields"].items():
            object.__setattr__(
                instance,
                name,
                model_from_structure_recipe(value),
            )
        return instance
    if kind in ("tuple", "list", "set", "frozenset"):
        items = [model_from_structure_recipe(item) for item in recipe["items"]]
        if kind == "tuple":
            return tuple(items)
        if kind == "list":
            return items
        if kind == "set":
            return set(items)
        return frozenset(items)
    if kind in ("mapping", "frozendict", "mappingproxy"):
        value = {
            model_from_structure_recipe(key): model_from_structure_recipe(item)
            for key, item in recipe["items"]
        }
        if kind == "frozendict":
            return frozendict(value)
        if kind == "mappingproxy":
            return MappingProxyType(value)
        return value
    raise ValueError(f"Unknown model artifact recipe kind {kind!r}.")


__all__ = [
    "deserialise_model_leaf",
    "model_from_structure_recipe",
    "model_structure_recipe",
    "serialise_model_leaf",
]
