#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np

from .._array_archive import (
    array_collection_digest,
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from .._identity import NumericRevision, SemanticProvenance
from ..lifecycle import ModelManifest
from ._affine import PreparedAffineLinearROM
from ._basis import ReducedBasisArtifact
from ._empirical_interpolation import EmpiricalInterpolationArtifact


_STRICT_KIND = "phydrax-rom-strict-model"
_EIM_KIND = "phydrax-rom-empirical-interpolation"


def _manifest_record(manifest: ModelManifest, /) -> dict[str, object]:
    return {
        "model_id": manifest.model_id,
        "analysis_plan_id": manifest.analysis_plan_id,
        "numeric_revision_id": manifest.numeric_revision_id,
        "payloads": [list(record) for record in manifest.payloads],
        "unit_contract_id": manifest.unit_contract_id,
        "association_ids": list(manifest.association_ids),
        "manifest_id": manifest.manifest_id,
    }


def _model_manifest(record: object, /) -> ModelManifest:
    if not isinstance(record, dict):
        raise TypeError("ROM archive model manifest is invalid.")
    required = {
        "model_id",
        "analysis_plan_id",
        "numeric_revision_id",
        "payloads",
        "unit_contract_id",
        "association_ids",
        "manifest_id",
    }
    if set(record) != required:
        raise ValueError("ROM archive model manifest keys are invalid.")
    manifest = ModelManifest(
        record["model_id"],
        record["analysis_plan_id"],
        record["numeric_revision_id"],
        tuple(tuple(item) for item in record["payloads"]),
        unit_contract_id=record["unit_contract_id"],
        association_ids=tuple(record["association_ids"]),
    )
    if manifest.manifest_id != record["manifest_id"]:
        raise ValueError("ROM archive model manifest identity mismatch.")
    return manifest


def _empirical_interpolation_revision(
    record: dict[str, Any], arrays: dict[str, Any], /
) -> NumericRevision:
    return NumericRevision(
        SemanticProvenance(
            {
                "kind": "empirical-interpolation",
                "source_artifact_id": record["source_artifact_id"],
                "basis_role": record["basis_role"],
                "support_id": record["support_id"],
                "measure_id": record["measure_id"],
                "geometry_id": record["geometry_id"],
            }
        ),
        arrays,
    )


def _write_strict_model(
    path: str | os.PathLike[str],
    model: Any,
    /,
    *,
    model_id: str,
    numeric_revision_id: str,
    analysis_plan_id: str,
    unit_contract_id: str | None,
    association_ids: tuple[str, ...],
    model_kind: str,
) -> Path:
    arrays: dict[str, Any] = {}
    dynamic, _ = eqx.partition(model, eqx.is_array)
    specification = pack_array_tree("model", dynamic, arrays)
    digest = array_collection_digest(arrays)
    manifest = ModelManifest(
        model_id,
        analysis_plan_id,
        numeric_revision_id,
        {"model-arrays": digest},
        unit_contract_id=unit_contract_id,
        association_ids=association_ids,
    )
    return write_array_archive(
        path,
        manifest={
            "kind": _STRICT_KIND,
            "model_kind": model_kind,
            "model": specification,
            "model_manifest": _manifest_record(manifest),
        },
        arrays=arrays,
    )


def _read_strict_model(
    path: str | os.PathLike[str],
    template: Any,
    /,
    *,
    model_id: str,
    numeric_revision_id: str,
    analysis_plan_id: str,
    unit_contract_id: str | None,
    association_ids: tuple[str, ...],
    model_kind: str,
):
    manifest, arrays = read_array_archive(path)
    expected = {"kind", "model_kind", "model", "model_manifest", "arrays"}
    if (
        set(manifest) != expected
        or manifest.get("kind") != _STRICT_KIND
        or manifest.get("model_kind") != model_kind
    ):
        raise ValueError("Archive is not the requested native ROM model kind.")
    lifecycle = _model_manifest(manifest["model_manifest"])
    expected_lifecycle = (
        lifecycle.model_id == model_id
        and lifecycle.numeric_revision_id == numeric_revision_id
        and lifecycle.analysis_plan_id == analysis_plan_id
        and lifecycle.unit_contract_id == unit_contract_id
        and lifecycle.association_ids == association_ids
    )
    if not expected_lifecycle:
        raise ValueError(
            "ROM archive lifecycle identity does not match the runtime template."
        )
    digest = array_collection_digest(arrays)
    if lifecycle.payloads != (("model-arrays", digest),):
        raise ValueError("ROM archive array collection identity mismatch.")
    dynamic, static = eqx.partition(template, eqx.is_array)
    restored = unpack_array_tree(manifest["model"], arrays, dynamic)
    return eqx.combine(restored, static)


def write_reduced_basis_artifact(
    path: str | os.PathLike[str],
    artifact: ReducedBasisArtifact,
    /,
    *,
    analysis_plan_id: str,
) -> Path:
    if not isinstance(artifact, ReducedBasisArtifact):
        raise TypeError("artifact must be a ReducedBasisArtifact.")
    return _write_strict_model(
        path,
        artifact,
        model_id=artifact.artifact_id,
        numeric_revision_id=artifact.numeric_revision.revision_id,
        analysis_plan_id=analysis_plan_id,
        unit_contract_id=None,
        association_ids=artifact.source_artifact_ids,
        model_kind="reduced-basis",
    )


def read_reduced_basis_artifact(
    path: str | os.PathLike[str],
    template: ReducedBasisArtifact,
    /,
    *,
    analysis_plan_id: str,
) -> ReducedBasisArtifact:
    if not isinstance(template, ReducedBasisArtifact):
        raise TypeError("template must be a ReducedBasisArtifact.")
    restored = _read_strict_model(
        path,
        template,
        model_id=template.artifact_id,
        model_kind="reduced-basis",
        numeric_revision_id=template.numeric_revision.revision_id,
        analysis_plan_id=analysis_plan_id,
        unit_contract_id=None,
        association_ids=template.source_artifact_ids,
    )
    if restored.artifact_id != template.artifact_id:
        raise ValueError("Restored basis artifact identity mismatch.")
    return restored


def write_affine_linear_rom(
    path: str | os.PathLike[str],
    model: PreparedAffineLinearROM,
    /,
    *,
    analysis_plan_id: str,
) -> Path:
    if not isinstance(model, PreparedAffineLinearROM):
        raise TypeError("model must be a PreparedAffineLinearROM.")
    return _write_strict_model(
        path,
        model,
        model_id=model.model_id,
        numeric_revision_id=model.numeric_revision.revision_id,
        analysis_plan_id=analysis_plan_id,
        unit_contract_id=model.coefficient_map.unit_contract_id,
        association_ids=(
            model.family_id,
            model.reduction.reduction_id,
            model.coefficient_map.coefficient_map_id,
        ),
        model_kind="affine-linear",
    )


def read_affine_linear_rom(
    path: str | os.PathLike[str],
    template: PreparedAffineLinearROM,
    /,
    *,
    analysis_plan_id: str,
) -> PreparedAffineLinearROM:
    if not isinstance(template, PreparedAffineLinearROM):
        raise TypeError("template must be a PreparedAffineLinearROM.")
    restored = _read_strict_model(
        path,
        template,
        model_id=template.model_id,
        model_kind="affine-linear",
        numeric_revision_id=template.numeric_revision.revision_id,
        analysis_plan_id=analysis_plan_id,
        unit_contract_id=template.coefficient_map.unit_contract_id,
        association_ids=(
            template.family_id,
            template.reduction.reduction_id,
            template.coefficient_map.coefficient_map_id,
        ),
    )
    if (
        restored.model_id != template.model_id
        or restored.coefficient_map.coefficient_map_id
        != template.coefficient_map.coefficient_map_id
    ):
        raise ValueError("Restored affine ROM dependency identity mismatch.")
    return restored


def write_empirical_interpolation_artifact(
    path: str | os.PathLike[str],
    artifact: EmpiricalInterpolationArtifact,
    /,
    *,
    analysis_plan_id: str,
) -> Path:
    if not isinstance(artifact, EmpiricalInterpolationArtifact):
        raise TypeError("artifact must be an EmpiricalInterpolationArtifact.")
    arrays = {
        "node_indices": artifact.node_indices,
        "interpolation_matrix": artifact.interpolation_matrix,
        "reconstruction_matrix": artifact.reconstruction_matrix,
    }
    digest = array_collection_digest(arrays)
    record = {
        "source_artifact_id": artifact.source_artifact_id,
        "basis_role": artifact.basis_role,
        "support_id": artifact.support_id,
        "measure_id": artifact.measure_id,
        "geometry_id": artifact.geometry_id,
    }
    revision = _empirical_interpolation_revision(record, arrays)
    lifecycle = ModelManifest(
        artifact.artifact_id,
        analysis_plan_id,
        revision.revision_id,
        {"interpolation-arrays": digest},
        association_ids=(artifact.source_artifact_id,),
    )
    return write_array_archive(
        path,
        manifest={
            "kind": _EIM_KIND,
            **record,
            "condition_number": artifact.condition_number,
            "maximum_reproduction_error": artifact.maximum_reproduction_error,
            "plan_id": artifact.plan_id,
            "artifact_id": artifact.artifact_id,
            "numeric_revision_id": revision.revision_id,
            "model_manifest": _manifest_record(lifecycle),
        },
        arrays=arrays,
    )


def read_empirical_interpolation_artifact(
    path: str | os.PathLike[str],
    /,
) -> EmpiricalInterpolationArtifact:
    manifest, arrays = read_array_archive(path)
    expected = {
        "kind",
        "source_artifact_id",
        "basis_role",
        "support_id",
        "measure_id",
        "geometry_id",
        "condition_number",
        "maximum_reproduction_error",
        "plan_id",
        "artifact_id",
        "numeric_revision_id",
        "model_manifest",
        "arrays",
    }
    if set(manifest) != expected or manifest.get("kind") != _EIM_KIND:
        raise ValueError("Archive is not a native empirical-interpolation artifact.")
    digest = array_collection_digest(arrays)
    revision = _empirical_interpolation_revision(manifest, arrays)
    if revision.revision_id != manifest["numeric_revision_id"]:
        raise ValueError("Empirical-interpolation numeric revision mismatch.")
    lifecycle = _model_manifest(manifest["model_manifest"])
    if lifecycle.payloads != (("interpolation-arrays", digest),):
        raise ValueError("Empirical-interpolation payload identity mismatch.")
    artifact = EmpiricalInterpolationArtifact(
        manifest["source_artifact_id"],
        manifest["basis_role"],
        manifest["support_id"],
        manifest["measure_id"],
        manifest["geometry_id"],
        np.asarray(arrays["node_indices"]),
        np.asarray(arrays["interpolation_matrix"]),
        np.asarray(arrays["reconstruction_matrix"]),
        manifest["condition_number"],
        manifest["maximum_reproduction_error"],
        manifest["plan_id"],
    )
    if (
        artifact.artifact_id != manifest["artifact_id"]
        or lifecycle.model_id != artifact.artifact_id
    ):
        raise ValueError("Empirical-interpolation artifact identity mismatch.")
    return artifact


__all__ = [
    "read_affine_linear_rom",
    "read_empirical_interpolation_artifact",
    "read_reduced_basis_artifact",
    "write_affine_linear_rom",
    "write_empirical_interpolation_artifact",
    "write_reduced_basis_artifact",
]
