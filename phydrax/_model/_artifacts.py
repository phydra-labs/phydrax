#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


ArchitectureEncoder = Callable[[Any], Mapping[str, Any]]
ArchitectureDecoder = Callable[[Mapping[str, Any]], Any]

_TRUSTED_ARTIFACT_ROOTS = frozenset(
    {
        "builtins",
        "collections",
        "equinox",
        "functools",
        "jax",
        "numpy",
        "optax",
        "phydrax",
    }
)
_ARTIFACT_VALUES_BY_ID: dict[str, Any] = {}
_ARTIFACT_IDS_BY_VALUE: dict[int, tuple[Any, str]] = {}


def _register_artifact_identity(
    value_id: str,
    value: Any,
    /,
    *,
    require_version: bool,
) -> Any:
    identity = str(value_id).strip()
    if not identity or (require_version and "@" not in identity):
        qualifier = " and explicitly versioned" if require_version else ""
        raise ValueError(f"Artifact value IDs must be non-empty{qualifier}.")
    if not callable(value):
        raise TypeError("Registered artifact values must be types or callables.")
    existing_value = _ARTIFACT_VALUES_BY_ID.get(identity)
    if existing_value is not None and existing_value is not value:
        raise ValueError(f"Artifact value ID {identity!r} is already registered.")
    existing_identity = _ARTIFACT_IDS_BY_VALUE.get(id(value))
    if (
        existing_identity is not None
        and existing_identity[0] is value
        and existing_identity[1] != identity
    ):
        raise ValueError(
            f"Artifact value {value.__qualname__} already has identity "
            f"{existing_identity[1]!r}."
        )
    _ARTIFACT_VALUES_BY_ID[identity] = value
    _ARTIFACT_IDS_BY_VALUE[id(value)] = (value, identity)
    return value


def register_artifact_value(value_id: str, value: Any, /) -> Any:
    """Register one path-independent, versioned artifact value identity."""
    return _register_artifact_identity(
        value_id,
        value,
        require_version=True,
    )


def artifact_value_id(value: Any, /) -> str:
    """Return a stable value identity without recording its defining module path."""
    existing = _ARTIFACT_IDS_BY_VALUE.get(id(value))
    if existing is not None and existing[0] is value:
        return existing[1]
    module = value.__module__
    qualname = value.__qualname__
    if "<locals>" in qualname or "<lambda>" in qualname:
        raise TypeError("Portable artifacts cannot encode local callables.")
    root = module.split(".", 1)[0]
    if root not in _TRUSTED_ARTIFACT_ROOTS:
        raise TypeError(f"Portable artifacts do not trust package root {root!r}.")
    identity = f"{root}.artifact:{qualname}@1"
    register_artifact_value(identity, value)
    return identity


def artifact_value(value_id: str, /) -> Any:
    """Resolve a registered value by stable identity, independent of module layout."""
    identity = str(value_id)
    registered = _ARTIFACT_VALUES_BY_ID.get(identity)
    if registered is not None:
        return registered
    root, separator, versioned_qualname = identity.partition(".artifact:")
    qualname, version_separator, version = versioned_qualname.rpartition("@")
    if (
        not separator
        or not version_separator
        or version != "1"
        or root not in _TRUSTED_ARTIFACT_ROOTS
    ):
        raise ValueError(f"Unknown artifact value ID {identity!r}.")
    importlib.import_module(root)
    matches: dict[int, Any] = {}
    for module_name, module in tuple(sys.modules.items()):
        if module is None or (
            module_name != root and not module_name.startswith(f"{root}.")
        ):
            continue
        resolved: Any = module
        for component in qualname.split("."):
            namespace = vars(resolved)
            if component not in namespace:
                break
            resolved = namespace[component]
        else:
            matches[id(resolved)] = resolved
    if not matches:
        raise ValueError(
            f"Artifact value ID {identity!r} is not registered by the installed package."
        )
    if len(matches) != 1:
        raise ValueError(
            f"Artifact value ID {identity!r} is ambiguous; register an explicit identity."
        )
    resolved = next(iter(matches.values()))
    register_artifact_value(identity, resolved)
    return resolved


@dataclass(frozen=True, slots=True)
class OperatorArchitectureCodec:
    """Stable portable-artifact identity and optional model configuration codec."""

    architecture_id: str
    model_type: type
    encode: ArchitectureEncoder | None = None
    decode: ArchitectureDecoder | None = None

    def __post_init__(self) -> None:
        architecture_id = self.architecture_id.strip()
        if not architecture_id:
            raise ValueError("Operator architecture IDs must be non-empty.")
        if not isinstance(self.model_type, type):
            raise TypeError("Operator architecture codec model_type must be a type.")
        if (self.encode is None) != (self.decode is None):
            raise ValueError(
                "Operator architecture codecs define both encode and decode."
            )
        object.__setattr__(self, "architecture_id", architecture_id)


_CODECS_BY_ID: dict[str, OperatorArchitectureCodec] = {}
_CODECS_BY_TYPE: dict[type, OperatorArchitectureCodec] = {}


def register_operator_architecture_codec(
    codec: OperatorArchitectureCodec, /
) -> OperatorArchitectureCodec:
    """Register one stable architecture codec, rejecting ambiguous identities."""
    if not isinstance(codec, OperatorArchitectureCodec):
        raise TypeError("codec must be an OperatorArchitectureCodec.")
    existing_id = _CODECS_BY_ID.get(codec.architecture_id)
    if existing_id is not None and existing_id != codec:
        raise ValueError(
            f"Operator architecture ID {codec.architecture_id!r} is already registered."
        )
    existing_type = _CODECS_BY_TYPE.get(codec.model_type)
    if existing_type is not None and existing_type != codec:
        raise ValueError(
            f"Operator model type {codec.model_type.__name__} already has architecture ID "
            f"{existing_type.architecture_id!r}."
        )
    _register_artifact_identity(
        codec.architecture_id,
        codec.model_type,
        require_version=False,
    )
    _CODECS_BY_ID[codec.architecture_id] = codec
    _CODECS_BY_TYPE[codec.model_type] = codec
    return codec


def operator_architecture_codec(architecture_id: str, /) -> OperatorArchitectureCodec:
    """Resolve one explicitly registered architecture ID."""
    codec = _CODECS_BY_ID.get(str(architecture_id))
    if codec is None:
        raise ValueError(
            f"Unknown operator architecture ID {architecture_id!r}; "
            "register its codec before loading the artifact."
        )
    return codec


def operator_architecture_codec_for(model: Any, /) -> OperatorArchitectureCodec:
    """Resolve the codec registered for an execution model's exact type."""
    codec = _CODECS_BY_TYPE.get(type(model))
    if codec is None:
        raise TypeError(
            f"Portable artifacts do not have a registered architecture codec for "
            f"{type(model).__name__}."
        )
    return codec


__all__ = [
    "artifact_value",
    "artifact_value_id",
    "ArchitectureDecoder",
    "ArchitectureEncoder",
    "OperatorArchitectureCodec",
    "operator_architecture_codec",
    "operator_architecture_codec_for",
    "register_artifact_value",
    "register_operator_architecture_codec",
]
