#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import dataclasses
import importlib.metadata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..._array_archive import (
    ArrayArchiveCorruptionError,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    read_array_archive,
    write_array_archive,
)
from ..._differentiation import (
    derivative_contract_from_payload,
    derivative_contract_payload,
    DerivativeContract,
)
from ..._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from ..._model import (
    AbstractArrayModel,
    artifact_value_id,
    FrozenModel,
    model_structure_recipe,
    ModelPorts,
    PortProvider,
)
from ..._model._structure import (
    model_from_array_recipe,
    model_recipe_array_inventory,
    model_recipe_template,
    pack_model_array_tree,
)
from .._contracts import FitResult
from .._schema import AbstractFittedModel, FeatureSchema, TargetSchema
from ._registry import register_native_ml_artifacts


_ML_ARTIFACT_FORMAT = "phydrax-ml-artifact"
_ML_ARTIFACT_LIMITS = dataclasses.replace(
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    # Registered dataclass recipes legitimately nest through composed numerical
    # modules. Keep every byte/member/rank limit while admitting that bounded tree.
    max_manifest_nesting=32,
)
_LEAF_PREFIX = "model/leaves"
_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "model_type",
        "model_recipe",
        "feature_schema",
        "target_schema",
        "ports",
        "derivative_contract",
        "fit",
        "identity",
        "provenance",
        "licenses",
        "versions",
        "arrays",
    }
)


@dataclass(frozen=True, slots=True)
class MLArtifactManifest:
    """Validated metadata for one portable native Phydrax ML model.

    `feature_schema`, `target_schema`, and `ports` are those of the restored
    executable (verified against the recorded ones); `derivative_contract` and
    `fit` describe the archived fit when one was supplied. The identity triplet
    is recomputed from the restored executable and verified on load.
    """

    model_type: str
    model_recipe: Mapping[str, Any]
    feature_schema: FeatureSchema | None
    target_schema: TargetSchema | None
    ports: ModelPorts | None
    derivative_contract: DerivativeContract | None
    fit: Mapping[str, Any] | None
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    executable_signature: ExecutableSignature
    provenance: Mapping[str, Any]
    licenses: tuple[str, ...]
    versions: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class MLArtifact:
    """A restored immutable model together with verified artifact metadata."""

    model: Any
    manifest: MLArtifactManifest


def _runtime_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name) for name in ("phydrax", "equinox", "jax")
    }


def _json_value(value: Any, /, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path} must contain only finite values.")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item(), path=path)
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_value(
                object.__getattribute__(value, field.name),
                path=f"{path}.{field.name}",
            )
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, path=f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [
            _json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        ]
    array = np.asarray(value)
    if array.ndim == 0:
        return _json_value(array.item(), path=path)
    raise TypeError(f"{path} contains a non-scalar value that is not JSON serializable.")


def _identity_triplet(
    recipe: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    resource_ids: Mapping[str, str] | Sequence[tuple[str, str]],
    /,
) -> tuple[SemanticProvenance, NumericRevision, ExecutableSignature]:
    semantic = SemanticProvenance(
        {"kind": "native-ml-executable", "structure": recipe},
        resource_ids=resource_ids,
    )
    numeric = NumericRevision(semantic, dict(arrays))
    signature = ExecutableSignature(
        shapes={name: array.shape for name, array in arrays.items()},
        dtypes={name: array.dtype for name, array in arrays.items()},
        algorithm_facts={"model_type": recipe["type"]},
    )
    return semantic, numeric, signature


def executable_identity(
    model: AbstractArrayModel,
    /,
    *,
    resource_ids: Mapping[str, str] | Sequence[tuple[str, str]] = (),
) -> tuple[SemanticProvenance, NumericRevision, ExecutableSignature]:
    """Return the canonical identity triplet of one native ML executable.

    The semantic provenance content-addresses the executable's portable structure
    recipe (types, static fields, schemas, and array specifications) together
    with any named external `resource_ids`; the numeric revision binds the exact
    array values to it; the executable signature records the array shapes and
    dtypes and the model type.
    """
    if not isinstance(model, AbstractArrayModel):
        raise TypeError("Executable identity requires an AbstractArrayModel.")
    register_native_ml_artifacts()
    recipe = model_structure_recipe(model)
    arrays = pack_model_array_tree(
        model, recipe, prefix=_LEAF_PREFIX, limits=_ML_ARTIFACT_LIMITS
    )
    return _identity_triplet(recipe, arrays, resource_ids)


def _identity_record(
    triplet: tuple[SemanticProvenance, NumericRevision, ExecutableSignature], /
) -> dict[str, str]:
    semantic, numeric, signature = triplet
    return {
        "semantic_id": semantic.semantic_id,
        "numeric_revision_id": numeric.revision_id,
        "executable_signature_id": signature.signature_id,
    }


def _executable(model: AbstractArrayModel, /) -> AbstractArrayModel:
    return model.as_trainable() if isinstance(model, FrozenModel) else model


def _schemas(
    model: AbstractArrayModel, /
) -> tuple[FeatureSchema | None, TargetSchema | None]:
    executable = _executable(model)
    if not isinstance(executable, AbstractFittedModel):
        return None, None
    return executable.feature_schema, executable.target_schema


def _ports(model: AbstractArrayModel, /) -> ModelPorts | None:
    executable = _executable(model)
    if isinstance(executable, AbstractFittedModel):
        if executable.feature_schema is None:
            return None
        return executable.model_ports()
    if isinstance(executable, PortProvider):
        return executable.model_ports()
    return None


def _dimensions_record(dimensions: Any, /) -> Any:
    if dimensions is None:
        return None
    return [[list(term) for term in dimension.terms] for dimension in dimensions]


def _feature_schema_record(schema: FeatureSchema | None, /) -> Any:
    if schema is None:
        return None
    return {
        "names": list(schema.names),
        "kinds": list(schema.kinds),
        "layout_id": schema.layout_id,
        "dimensions": _dimensions_record(schema.dimensions),
    }


def _target_schema_record(schema: TargetSchema | None, /) -> Any:
    if schema is None:
        return None
    return {
        "kind": schema.kind,
        "names": list(schema.names),
        "class_labels": _json_value(
            schema.class_labels, path="target_schema.class_labels"
        ),
        "dimensions": _dimensions_record(schema.dimensions),
    }


def _ports_record(ports: ModelPorts | None, /) -> Any:
    if ports is None:
        return None
    return {
        "ports_id": ports.ports_id,
        "inputs": [port.port_id for port in ports.inputs],
        "outputs": [port.port_id for port in ports.outputs],
    }


def _fit_metadata(result: FitResult | None, /) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "valid": np.asarray(result.valid).tolist(),
        "status": np.asarray(result.status).tolist(),
        "method": result.method,
    }


def save_ml_artifact(
    path: str | Path,
    model: Any,
    /,
    *,
    fit_result: FitResult | None = None,
    provenance: Mapping[str, Any] | None = None,
    licenses: Sequence[str] = (),
) -> Path:
    """Write a checksum-validated, pickle-free native ML model artifact.

    Schemas and ports are those the executable carries (`phydrax.ml.fit` binds
    them); a supplied `fit_result` contributes its derivative contract, validity,
    status, and method. The canonical identity triplet of the executable is
    recorded and verified again on load.
    """
    if not isinstance(model, AbstractArrayModel):
        raise TypeError("Native ML artifacts require an AbstractArrayModel.")
    if (
        fit_result is not None
        and fit_result.model is not model
        and fit_result.as_trainable() is not model
    ):
        raise ValueError("fit_result must describe the model being archived.")
    register_native_ml_artifacts()
    recipe = model_structure_recipe(model)
    if recipe.get("kind") != "dataclass" or not isinstance(recipe.get("type"), str):
        raise TypeError("Native ML artifacts require a registered dataclass model.")
    arrays = pack_model_array_tree(
        model,
        recipe,
        prefix=_LEAF_PREFIX,
        limits=_ML_ARTIFACT_LIMITS,
    )
    feature_schema, target_schema = _schemas(model)
    manifest = {
        "format": _ML_ARTIFACT_FORMAT,
        "model_type": recipe["type"],
        "model_recipe": recipe,
        "feature_schema": _feature_schema_record(feature_schema),
        "target_schema": _target_schema_record(target_schema),
        "ports": _ports_record(_ports(model)),
        "derivative_contract": (
            None
            if fit_result is None
            else derivative_contract_payload(fit_result.derivative_contract)
        ),
        "fit": _fit_metadata(fit_result),
        "identity": _identity_record(_identity_triplet(recipe, arrays, ())),
        "provenance": _json_value(dict(provenance or {}), path="provenance"),
        "licenses": [str(item) for item in licenses],
        "versions": _runtime_versions(),
    }
    return write_array_archive(
        path,
        manifest=manifest,
        limits=_ML_ARTIFACT_LIMITS,
        arrays=arrays,
    )


def _validated_metadata(manifest: Mapping[str, Any], /) -> None:
    if set(manifest) != _MANIFEST_FIELDS:
        raise ArrayArchiveCorruptionError("ML artifact manifest fields are invalid.")
    if manifest["format"] != _ML_ARTIFACT_FORMAT:
        raise ArrayArchiveCorruptionError(
            "Archive is not a supported Phydrax ML artifact."
        )
    licenses = manifest["licenses"]
    versions = manifest["versions"]
    optional_mappings = (
        manifest["feature_schema"],
        manifest["target_schema"],
        manifest["ports"],
        manifest["derivative_contract"],
        manifest["fit"],
    )
    if (
        not isinstance(manifest["model_type"], str)
        or not manifest["model_type"]
        or not isinstance(manifest["model_recipe"], dict)
        or not isinstance(manifest["identity"], dict)
        or not isinstance(licenses, list)
        or any(not isinstance(item, str) or not item for item in licenses)
        or not isinstance(manifest["provenance"], dict)
        or not isinstance(versions, dict)
        or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in versions.items()
        )
        or any(
            value is not None and not isinstance(value, dict)
            for value in optional_mappings
        )
    ):
        raise ArrayArchiveCorruptionError("ML artifact metadata is invalid.")


def _restored_model(
    recipe: Mapping[str, Any], arrays: Mapping[str, Any], model_type: str, /
) -> AbstractArrayModel:
    try:
        template = model_recipe_template(recipe, limits=_ML_ARTIFACT_LIMITS)
        inventory = model_recipe_array_inventory(
            recipe,
            prefix=_LEAF_PREFIX,
            limits=_ML_ARTIFACT_LIMITS,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArrayArchiveCorruptionError(
            "ML artifact model recipe is invalid."
        ) from error
    if not isinstance(template, AbstractArrayModel):
        raise ArrayArchiveCorruptionError(
            "ML artifact recipe did not declare an AbstractArrayModel."
        )
    expected_type = artifact_value_id(type(template))
    if model_type != expected_type or recipe["type"] != expected_type:
        raise ArrayArchiveCorruptionError("ML artifact model type is inconsistent.")
    if set(arrays) != {entry.name for entry in inventory}:
        raise ArrayArchiveCorruptionError(
            "ML artifact model payload does not match its recipe."
        )
    try:
        model = model_from_array_recipe(
            recipe,
            arrays,
            prefix=_LEAF_PREFIX,
            limits=_ML_ARTIFACT_LIMITS,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArrayArchiveCorruptionError(
            "ML artifact model payload is incompatible with its recipe."
        ) from error
    if not isinstance(model, AbstractArrayModel):
        raise ArrayArchiveCorruptionError(
            "ML artifact payload did not restore an AbstractArrayModel."
        )
    if model_structure_recipe(model) != recipe:
        raise ArrayArchiveCorruptionError(
            "ML artifact model structure changed during restoration."
        )
    return model


def _recorded_contract(payload: Any, /) -> DerivativeContract | None:
    if payload is None:
        return None
    try:
        return derivative_contract_from_payload(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ArrayArchiveCorruptionError(
            "ML artifact derivative contract is invalid."
        ) from error


def read_ml_artifact(path: str | Path, /) -> MLArtifact:
    """Restore and verify one portable native ML model artifact."""
    register_native_ml_artifacts()
    manifest, arrays = read_array_archive(path, limits=_ML_ARTIFACT_LIMITS)
    _validated_metadata(manifest)
    recipe = manifest["model_recipe"]
    model = _restored_model(recipe, arrays, manifest["model_type"])
    triplet = _identity_triplet(
        recipe,
        pack_model_array_tree(
            model, recipe, prefix=_LEAF_PREFIX, limits=_ML_ARTIFACT_LIMITS
        ),
        (),
    )
    if manifest["identity"] != _identity_record(triplet):
        raise ArrayArchiveCorruptionError(
            "ML artifact identity does not match the restored executable."
        )
    feature_schema, target_schema = _schemas(model)
    ports = _ports(model)
    if (
        manifest["feature_schema"] != _feature_schema_record(feature_schema)
        or manifest["target_schema"] != _target_schema_record(target_schema)
        or manifest["ports"] != _ports_record(ports)
    ):
        raise ArrayArchiveCorruptionError(
            "ML artifact schemas or ports do not match the restored executable."
        )
    fit = manifest["fit"]
    if fit is not None and set(fit) != {"valid", "status", "method"}:
        raise ArrayArchiveCorruptionError("ML artifact fit metadata is invalid.")
    semantic, numeric, signature = triplet
    parsed = MLArtifactManifest(
        model_type=manifest["model_type"],
        model_recipe=MappingProxyType(recipe),
        feature_schema=feature_schema,
        target_schema=target_schema,
        ports=ports,
        derivative_contract=_recorded_contract(manifest["derivative_contract"]),
        fit=None if fit is None else MappingProxyType(fit),
        semantic_provenance=semantic,
        numeric_revision=numeric,
        executable_signature=signature,
        provenance=MappingProxyType(manifest["provenance"]),
        licenses=tuple(manifest["licenses"]),
        versions=MappingProxyType(dict(manifest["versions"])),
    )
    return MLArtifact(model=model, manifest=parsed)


def load_ml_model(path: str | Path, /) -> Any:
    """Restore the verified schema- and port-bound executable of an ML artifact."""
    return read_ml_artifact(path).model


__all__ = [
    "MLArtifact",
    "MLArtifactManifest",
    "executable_identity",
    "load_ml_model",
    "read_ml_artifact",
    "save_ml_artifact",
]
