#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import dataclasses
import importlib.metadata
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import equinox as eqx
import numpy as np

from ..._array_archive import (
    ArrayArchiveCorruptionError,
    read_array_archive,
    write_array_archive,
)
from ..._model import (
    artifact_value_id,
    deserialise_model_leaf,
    model_from_structure_recipe,
    model_structure_recipe,
    serialise_model_leaf,
)
from .._contracts import FitResult
from ._registry import register_native_ml_artifacts


_ML_ARTIFACT_FORMAT = "phydrax-ml-artifact"
_ML_ARTIFACT_VERSION = 1


@dataclass(frozen=True, slots=True)
class MLArtifactManifest:
    """Validated metadata for one portable native Phydrax ML model."""

    model_type: str
    model_recipe: Mapping[str, Any]
    feature_schema: Mapping[str, Any] | None
    target_schema: Mapping[str, Any] | None
    fit: Mapping[str, Any] | None
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


def _fit_metadata(result: FitResult | None, /) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "valid": np.asarray(result.valid).tolist(),
        "status": np.asarray(result.status).tolist(),
        "method": result.method,
        "gradient_contract": {
            "prediction_inputs": result.gradient_contract.prediction_inputs,
            "prediction_parameters": result.gradient_contract.prediction_parameters,
            "fit_features": result.gradient_contract.fit_features,
            "fit_targets": result.gradient_contract.fit_targets,
            "fit_weights": result.gradient_contract.fit_weights,
            "fit_hyperparameters": result.gradient_contract.fit_hyperparameters,
            "fit_mode": result.gradient_contract.fit_mode,
            "nondifferentiable_outputs": list(
                result.gradient_contract.nondifferentiable_outputs
            ),
            "conditions": list(result.gradient_contract.conditions),
        },
    }


def save_ml_artifact(
    path: str | Path,
    model: Any,
    /,
    *,
    fit_result: FitResult | None = None,
    feature_schema: Any = None,
    target_schema: Any = None,
    provenance: Mapping[str, Any] | None = None,
    licenses: Sequence[str] = (),
) -> Path:
    """Write a checksum-validated, pickle-free native ML model artifact."""
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
    stream = io.BytesIO()
    eqx.tree_serialise_leaves(stream, model, filter_spec=serialise_model_leaf)
    leaves = np.frombuffer(stream.getvalue(), dtype=np.uint8).copy()
    fit_metadata = _fit_metadata(fit_result)
    encoded_feature_schema = (
        None
        if feature_schema is None
        else _json_value(feature_schema, path="feature_schema")
    )
    encoded_target_schema = (
        None
        if target_schema is None
        else _json_value(target_schema, path="target_schema")
    )
    manifest = {
        "format": _ML_ARTIFACT_FORMAT,
        "version": _ML_ARTIFACT_VERSION,
        "model_type": recipe["type"],
        "model_recipe": recipe,
        "feature_schema": encoded_feature_schema,
        "target_schema": encoded_target_schema,
        "fit": fit_metadata,
        "provenance": _json_value(dict(provenance or {}), path="provenance"),
        "licenses": [str(item) for item in licenses],
        "versions": _runtime_versions(),
    }
    return write_array_archive(
        path,
        manifest=manifest,
        arrays={"model/leaves": leaves},
    )


def read_ml_artifact(path: str | Path, /) -> MLArtifact:
    """Restore and verify one portable native ML model artifact."""
    register_native_ml_artifacts()
    manifest, arrays = read_array_archive(path)
    expected = {
        "format",
        "version",
        "model_type",
        "model_recipe",
        "feature_schema",
        "target_schema",
        "fit",
        "provenance",
        "licenses",
        "versions",
        "arrays",
    }
    if set(manifest) != expected:
        raise ArrayArchiveCorruptionError("ML artifact manifest fields are invalid.")
    if (
        manifest["format"] != _ML_ARTIFACT_FORMAT
        or manifest["version"] != _ML_ARTIFACT_VERSION
    ):
        raise ArrayArchiveCorruptionError(
            "Archive is not a supported Phydrax ML artifact."
        )
    if set(arrays) != {"model/leaves"}:
        raise ArrayArchiveCorruptionError("ML artifact model payload is invalid.")
    recipe = manifest["model_recipe"]
    if not isinstance(recipe, dict):
        raise ArrayArchiveCorruptionError("ML artifact model recipe is invalid.")
    template = model_from_structure_recipe(recipe)
    payload = np.asarray(arrays["model/leaves"], dtype=np.uint8).tobytes()
    model = eqx.tree_deserialise_leaves(
        io.BytesIO(payload),
        template,
        filter_spec=deserialise_model_leaf,
    )
    expected_type = artifact_value_id(type(model))
    if manifest["model_type"] != expected_type or recipe["type"] != expected_type:
        raise ArrayArchiveCorruptionError("ML artifact model type is inconsistent.")
    licenses = manifest["licenses"]
    provenance = manifest["provenance"]
    versions = manifest["versions"]
    if (
        not isinstance(licenses, list)
        or not isinstance(provenance, dict)
        or not isinstance(versions, dict)
    ):
        raise ArrayArchiveCorruptionError("ML artifact metadata is invalid.")
    parsed = MLArtifactManifest(
        model_type=expected_type,
        model_recipe=MappingProxyType(recipe),
        feature_schema=(
            None
            if manifest["feature_schema"] is None
            else MappingProxyType(manifest["feature_schema"])
        ),
        target_schema=(
            None
            if manifest["target_schema"] is None
            else MappingProxyType(manifest["target_schema"])
        ),
        fit=(None if manifest["fit"] is None else MappingProxyType(manifest["fit"])),
        provenance=MappingProxyType(provenance),
        licenses=tuple(str(item) for item in licenses),
        versions=MappingProxyType(
            {str(key): str(value) for key, value in versions.items()}
        ),
    )
    return MLArtifact(model=model, manifest=parsed)


def load_ml_model(path: str | Path, /) -> Any:
    """Restore only the model payload from a verified native ML artifact."""
    return read_ml_artifact(path).model


__all__ = [
    "MLArtifact",
    "MLArtifactManifest",
    "load_ml_model",
    "read_ml_artifact",
    "save_ml_artifact",
]
