#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax

from .._frozendict import frozendict
from ..typing import checked
from ._binding import ModelBinding


ArchitectureEncoder = Callable[[Any], Mapping[str, Any]]
ArchitectureDecoder = Callable[[Mapping[str, Any]], Any]

_ARTIFACT_VALUES_BY_ID: dict[str, Any] = {}
_ARTIFACT_IDS_BY_VALUE: dict[int, tuple[Any, str]] = {}
_ARTIFACT_RESTORE_VALIDATORS_BY_ID: dict[str, Callable[[Any], None]] = {}


def _load_builtin_registrations() -> None:
    """Import available built-ins without requiring absent optional providers.

    The package loads lazily, so a cold restore cannot rely on a prior import of
    the owner of a serialized identity. The fixed public module list and the
    facades' declared owners and optional dependencies, never the serialized
    identity, select what loads.
    """
    from .._deferred_exports import load_all, load_module
    from .._package_modules import PUBLIC_MODULES

    for name in PUBLIC_MODULES:
        load_module(f"phydrax.{name}")
    load_all()


def _register_artifact_identity(
    value_id: str,
    value: Any,
    /,
) -> Any:
    identity = str(value_id).strip()
    if not identity:
        raise ValueError("Artifact value IDs must be non-empty.")
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
            f"Artifact value {value.__qualname__} already has identity {existing_identity[1]!r}."
        )
    _ARTIFACT_VALUES_BY_ID[identity] = value
    _ARTIFACT_IDS_BY_VALUE[id(value)] = (value, identity)
    return value


def register_artifact_value(
    value_id: str,
    value: Any,
    /,
    *,
    validate_restored: Callable[[Any], None] | None = None,
) -> Any:
    """Register one artifact identity and its restored-value validator."""
    if validate_restored is not None and not callable(validate_restored):
        raise TypeError(
            f"artifact.validate_restored must be callable; got {type(validate_restored).__name__}"
        )
    identity = str(value_id).strip()
    existing = _ARTIFACT_RESTORE_VALIDATORS_BY_ID.get(identity)
    if (
        existing is not None
        and validate_restored is not None
        and existing is not validate_restored
    ):
        raise ValueError(
            f"artifact.value_id must retain its registered restored-value validator; got {identity!r} with a different validator"
        )
    registered = _register_artifact_identity(value_id, value)
    if validate_restored is not None:
        _ARTIFACT_RESTORE_VALIDATORS_BY_ID[identity] = validate_restored
    return registered


def registered_artifact_value_id(value: Any, /) -> str | None:
    """Inspect registration without admitting an unregistered value."""
    existing = _ARTIFACT_IDS_BY_VALUE.get(id(value))
    return existing[1] if existing is not None and existing[0] is value else None


def artifact_value_id(value: Any, /) -> str:
    """Return one explicitly registered path-independent artifact identity."""
    identity = registered_artifact_value_id(value)
    if identity is not None:
        return identity
    owner = value if isinstance(value, type) else type(value)
    raise TypeError(
        "artifact.value must have an explicit registration; got "
        f"{owner.__module__}.{owner.__qualname__}."
    )


def artifact_value(value_id: str, /) -> Any:
    """Resolve one explicitly registered path-independent artifact identity."""
    identity = str(value_id)
    registered = _ARTIFACT_VALUES_BY_ID.get(identity)
    if registered is None:
        _load_builtin_registrations()
        registered = _ARTIFACT_VALUES_BY_ID.get(identity)
    if registered is None:
        raise ValueError(f"Unknown artifact value ID {identity!r}.")
    return registered


def validate_restored_artifact_value(value: Any, /) -> None:
    """Validate a restored value through its registered owner."""
    identity = artifact_value_id(type(value))
    validator = _ARTIFACT_RESTORE_VALIDATORS_BY_ID.get(identity)
    if validator is not None:
        validator(value)


@dataclass(frozen=True, slots=True)
class ArtifactValueCodec:
    """Constructor codec for a registered, array-free immutable value.

    Encoders declare a closed mapping, rather than capturing instance
    attributes. Decoders must validate that mapping through the value's public
    constructor; recipe loading additionally checks the returned type and
    deterministic re-encoding.
    """

    value_id: str
    value_type: type
    encode: Callable[[Any], Mapping[str, Any]]
    decode: Callable[[Mapping[str, Any]], Any]

    def __post_init__(self) -> None:
        identity = self.value_id.strip()
        if not identity:
            raise ValueError(
                f"ArtifactValueCodec.value_id must be nonempty; got {self.value_id!r}"
            )
        if not isinstance(self.value_type, type):
            raise TypeError(
                f"ArtifactValueCodec.value_type must be a type; got {type(self.value_type).__name__}"
            )
        if not callable(self.encode) or not callable(self.decode):
            raise TypeError(
                f"ArtifactValueCodec.encode_decode must be callable; got encode={type(self.encode).__name__}, decode={type(self.decode).__name__}"
            )
        object.__setattr__(self, "value_id", identity)

    def encode_value(self, value: Any, /) -> Mapping[str, Any]:
        if type(value) is not self.value_type:
            raise TypeError(
                f"ArtifactValueCodec.value must have registered type {self.value_type.__qualname__}; got {type(value).__qualname__}"
            )
        payload = self.encode(value)
        if not isinstance(payload, Mapping) or any(
            not isinstance(key, str) for key in payload
        ):
            raise TypeError(
                f"ArtifactValueCodec.payload must be a text-keyed mapping; got {payload!r}"
            )
        return payload

    def decode_value(self, payload: Mapping[str, Any], /) -> Any:
        if not isinstance(payload, Mapping) or any(
            not isinstance(key, str) for key in payload
        ):
            raise TypeError(
                f"ArtifactValueCodec.payload must be a text-keyed mapping; got {payload!r}"
            )
        value = self.decode(payload)
        if type(value) is not self.value_type:
            raise TypeError(
                f"ArtifactValueCodec.decoded_value must have registered type {self.value_type.__qualname__}; got {type(value).__qualname__}"
            )
        return value


_VALUE_CODECS_BY_ID: dict[str, ArtifactValueCodec] = {}
_VALUE_CODECS_BY_TYPE: dict[type, ArtifactValueCodec] = {}


@checked
def register_artifact_value_codec(codec: ArtifactValueCodec, /) -> ArtifactValueCodec:
    """Register a constructor codec under the value's artifact identity."""
    existing_id = _VALUE_CODECS_BY_ID.get(codec.value_id)
    if existing_id is not None and existing_id != codec:
        raise ValueError(
            f"artifact.codec.value_id must retain its registered codec; got {codec.value_id!r} with a different codec"
        )
    existing_type = _VALUE_CODECS_BY_TYPE.get(codec.value_type)
    if existing_type is not None and existing_type != codec:
        raise ValueError(
            f"artifact.codec.value_type must retain codec ID {existing_type.value_id!r}; got {codec.value_type.__name__} with "
            f"{existing_type.value_id!r}."
        )
    _register_artifact_identity(codec.value_id, codec.value_type)
    _VALUE_CODECS_BY_ID[codec.value_id] = codec
    _VALUE_CODECS_BY_TYPE[codec.value_type] = codec
    return codec


def artifact_value_codec(value_id: str, /) -> ArtifactValueCodec:
    """Resolve only an explicitly registered immutable-value constructor codec."""
    codec = _VALUE_CODECS_BY_ID.get(value_id)
    if codec is None:
        raise ValueError(
            f"artifact.codec.value_id must identify a registered codec; got {value_id!r}"
        )
    return codec


def artifact_value_codec_for(value: Any, /) -> ArtifactValueCodec | None:
    """Inspect the constructor codec registered for a value's type."""
    return _VALUE_CODECS_BY_TYPE.get(type(value))


register_artifact_value("jax.artifact:tanh", jax.nn.tanh)
register_artifact_value("jax.artifact:gelu", jax.nn.gelu)
register_artifact_value("jax.artifact:relu", jax.nn.relu)
register_artifact_value("jax.artifact:sigmoid", jax.nn.sigmoid)
register_artifact_value("jax.artifact:silu", jax.nn.silu)
register_artifact_value("jax.artifact:softplus", jax.nn.softplus)
register_artifact_value("equinox.nn:Linear", eqx.nn.Linear)
register_artifact_value("phydrax.core:frozendict", frozendict)
register_artifact_value("phydrax.core:ModelBinding", ModelBinding)


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
    _register_artifact_identity(codec.architecture_id, codec.model_type)
    _CODECS_BY_ID[codec.architecture_id] = codec
    _CODECS_BY_TYPE[codec.model_type] = codec
    return codec


def operator_architecture_codec(architecture_id: str, /) -> OperatorArchitectureCodec:
    """Resolve one explicitly registered architecture ID."""
    codec = _CODECS_BY_ID.get(str(architecture_id))
    if codec is None:
        _load_builtin_registrations()
        codec = _CODECS_BY_ID.get(str(architecture_id))
    if codec is None:
        raise ValueError(
            f"Unknown operator architecture ID {architecture_id!r}; register its codec before loading the artifact."
        )
    return codec


def operator_architecture_codec_for(model: Any, /) -> OperatorArchitectureCodec:
    """Resolve the codec registered for an execution model's exact type."""
    codec = _CODECS_BY_TYPE.get(type(model))
    if codec is None:
        raise TypeError(
            f"Portable artifacts do not have a registered architecture codec for {type(model).__name__}."
        )
    return codec


__all__ = [
    "artifact_value",
    "artifact_value_id",
    "ArtifactValueCodec",
    "artifact_value_codec",
    "artifact_value_codec_for",
    "ArchitectureDecoder",
    "ArchitectureEncoder",
    "OperatorArchitectureCodec",
    "operator_architecture_codec",
    "operator_architecture_codec_for",
    "register_artifact_value",
    "register_artifact_value_codec",
    "registered_artifact_value_id",
    "register_operator_architecture_codec",
    "validate_restored_artifact_value",
]
