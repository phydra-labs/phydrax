#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned, artifact-import-only OpenXRayMC/XRayMClib HDF5 score profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from io import BytesIO

import h5py
import numpy as np

from ...._fingerprint import canonical_fingerprint
from ....interchange import (
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
    AdapterWaiver,
    BoundedResource,
)
from ....qualification import ReferenceArtifactManifest
from .._scores import (
    _identifier,
    _positive_integer,
    _UNCERTAINTY_KINDS,
    ExternalRadiationRunIdentity,
    ExternalRadiationScoreResult,
    radiation_score_content_id,
    RadiationEstimatorEvidence,
    RadiationScoreDefinition,
    require_import_rights,
    require_profile_semantics,
)


_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_BASE_SEMANTICS = (
    "build-identity",
    "calibration-identity",
    "configuration-identity",
    "grid-affine",
    "score-dtype",
    "score-normalization",
    "score-quantity",
    "score-shape",
    "seed-lineage",
    "table-identity",
    "units",
)


def _dataset_path(value: str, name: str, /) -> str:
    path = _identifier(value, name)
    if not path.startswith("/") or path == "/" or "//" in path:
        raise ValueError(f"{name} must be one absolute HDF5 dataset path.")
    return path


def _attribute_text(value, name: str, /) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"HDF5 attribute {name!r} must be scalar.")
    scalar = array.reshape(()).item()
    if isinstance(scalar, bytes):
        text = scalar.decode("utf-8", errors="strict")
    else:
        text = str(scalar)
    return _identifier(text, f"HDF5 attribute {name!r}")


def _fixed_numeric_dataset(
    handle,
    path: str,
    /,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: str | None,
    max_logical_bytes: int,
) -> np.ndarray:
    if path not in handle or not isinstance(handle[path], h5py.Dataset):
        raise ValueError(f"Pinned HDF5 score dataset {path!r} is absent.")
    dataset = handle[path]
    if dataset.is_virtual or dataset.external:
        raise ValueError("External and virtual HDF5 score storage is forbidden.")
    dtype = dataset.dtype
    if (
        dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or not np.issubdtype(dtype, np.number)
    ):
        raise TypeError("HDF5 score datasets require fixed-width scalar numeric storage.")
    shape = tuple(int(size) for size in dataset.shape)
    if shape != expected_shape:
        raise ValueError("HDF5 dataset shape differs from the exact profile.")
    if expected_dtype is not None and dtype.str != np.dtype(expected_dtype).str:
        raise ValueError("HDF5 dataset dtype differs from the exact profile.")
    logical_bytes = math.prod(shape) * int(dtype.itemsize)
    if logical_bytes > max_logical_bytes:
        raise ValueError("HDF5 logical dataset exceeds the admitted resource byte limit.")
    return np.asarray(dataset[()])


@dataclass(frozen=True, slots=True)
class OpenXRayMCHDF5Profile:
    """One exact HDF5 dataset profile and its explicit upstream semantics."""

    provider: str
    run_identity: ExternalRadiationRunIdentity
    score: RadiationScoreDefinition
    score_dataset: str
    uncertainty_dataset: str | None
    uncertainty_kind: str
    estimator: str
    history_count: int
    batch_count: int
    correlation_model: str
    producer_attribute: str = "producer"
    revision_attribute: str = "producer_revision"
    required_attributes: tuple[tuple[str, str], ...] = ()
    declared_losses: tuple[AdapterLoss, ...] = ()
    profile_id: str = field(init=False)
    available_semantics: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        provider = _identifier(self.provider, "provider")
        if provider not in ("OpenXRayMC", "XRayMClib"):
            raise ValueError("provider must be OpenXRayMC or XRayMClib.")
        if not isinstance(self.run_identity, ExternalRadiationRunIdentity):
            raise TypeError("run_identity must be ExternalRadiationRunIdentity.")
        if self.run_identity.engine != provider:
            raise ValueError("HDF5 provider and external run engine identity disagree.")
        if not isinstance(self.score, RadiationScoreDefinition):
            raise TypeError("score must be RadiationScoreDefinition.")
        score_dataset = _dataset_path(self.score_dataset, "score_dataset")
        uncertainty_dataset = (
            None
            if self.uncertainty_dataset is None
            else _dataset_path(self.uncertainty_dataset, "uncertainty_dataset")
        )
        if uncertainty_dataset == score_dataset:
            raise ValueError("Score and uncertainty datasets must be distinct.")
        uncertainty_kind = _identifier(self.uncertainty_kind, "uncertainty_kind")
        if uncertainty_kind not in _UNCERTAINTY_KINDS:
            raise ValueError("Unknown HDF5 estimator uncertainty kind.")
        correlation = _identifier(self.correlation_model, "correlation_model")
        if (uncertainty_dataset is None) != (uncertainty_kind == "unreported"):
            raise ValueError(
                "HDF5 uncertainty dataset and reported uncertainty kind must agree."
            )
        if uncertainty_dataset is None and correlation != "unreported":
            raise ValueError("Absent HDF5 uncertainty cannot claim a correlation model.")
        if uncertainty_dataset is not None and correlation in (
            "unreported",
            "assumed-independent",
        ):
            raise ValueError(
                "Reported HDF5 uncertainty requires upstream correlation evidence."
            )
        histories = _positive_integer(self.history_count, "history_count")
        batches = _positive_integer(self.batch_count, "batch_count")
        producer_attribute = _identifier(self.producer_attribute, "producer_attribute")
        revision_attribute = _identifier(self.revision_attribute, "revision_attribute")
        if producer_attribute == revision_attribute:
            raise ValueError("Producer and revision HDF5 attributes must be distinct.")
        attributes = tuple(
            (
                _identifier(name, "HDF5 attribute name"),
                _identifier(value, "HDF5 attribute value"),
            )
            for name, value in self.required_attributes
        )
        names = tuple(name for name, _ in attributes)
        if (
            len(set(names)) != len(names)
            or producer_attribute in names
            or revision_attribute in names
        ):
            raise ValueError("Pinned HDF5 attribute names must be unique.")
        losses = tuple(self.declared_losses)
        if any(not isinstance(loss, AdapterLoss) for loss in losses):
            raise TypeError("declared_losses must contain AdapterLoss values.")
        available = _BASE_SEMANTICS + (
            ("estimator",)
            + (() if uncertainty_dataset is None else ("uncertainty", "correlation"))
        )
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "score_dataset", score_dataset)
        object.__setattr__(self, "uncertainty_dataset", uncertainty_dataset)
        object.__setattr__(self, "uncertainty_kind", uncertainty_kind)
        object.__setattr__(self, "estimator", _identifier(self.estimator, "estimator"))
        object.__setattr__(self, "history_count", histories)
        object.__setattr__(self, "batch_count", batches)
        object.__setattr__(self, "correlation_model", correlation)
        object.__setattr__(self, "producer_attribute", producer_attribute)
        object.__setattr__(self, "revision_attribute", revision_attribute)
        object.__setattr__(self, "required_attributes", attributes)
        object.__setattr__(self, "declared_losses", losses)
        object.__setattr__(
            self,
            "available_semantics",
            tuple(sorted(set(available))),
        )
        object.__setattr__(
            self,
            "profile_id",
            canonical_fingerprint(
                {
                    "kind": "openxraymc-hdf5-score-profile",
                    "provider": provider,
                    "run": self.run_identity.identity_id,
                    "score": self.score.definition_id,
                    "score_dataset": score_dataset,
                    "uncertainty_dataset": uncertainty_dataset,
                    "uncertainty_kind": uncertainty_kind,
                    "estimator": self.estimator,
                    "history_count": histories,
                    "batch_count": batches,
                    "correlation_model": correlation,
                    "producer_attribute": producer_attribute,
                    "revision_attribute": revision_attribute,
                    "required_attributes": [list(item) for item in attributes],
                    "declared_losses": [loss.loss_id for loss in losses],
                }
            ),
        )


def import_openxraymc_hdf5(
    resource: BoundedResource,
    reference: ReferenceArtifactManifest,
    profile: OpenXRayMCHDF5Profile,
    /,
    *,
    required_semantics: tuple[str, ...] = (),
    commercial_use: bool = False,
    redistribution: bool = False,
    training_use: bool = False,
    export: bool = False,
) -> ExternalRadiationScoreResult:
    """Import only the dataset pinned by ``profile``; never invoke the provider."""

    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be BoundedResource.")
    if not isinstance(reference, ReferenceArtifactManifest):
        raise TypeError("reference must be ReferenceArtifactManifest.")
    if not isinstance(profile, OpenXRayMCHDF5Profile):
        raise TypeError("profile must be OpenXRayMCHDF5Profile.")
    require_profile_semantics(required_semantics, profile.available_semantics)
    reference.verify_bytes(resource.data)
    require_import_rights(
        (reference,),
        profile.run_identity,
        commercial_use=commercial_use,
        redistribution=redistribution,
        training_use=training_use,
        export=export,
    )
    if not resource.data.startswith(_HDF5_SIGNATURE):
        raise ValueError("OpenXRayMC/XRayMClib score artifact must be HDF5.")
    expected_uncertainty_shape = (
        None
        if profile.uncertainty_dataset is None
        else (
            (math.prod(profile.score.shape),) * 2
            if profile.uncertainty_kind == "covariance"
            else profile.score.shape
        )
    )
    max_logical_bytes = resource.manifest.limits.max_bytes
    with h5py.File(BytesIO(resource.data), "r") as handle:
        for name, expected in (
            (profile.producer_attribute, profile.provider),
            (profile.revision_attribute, profile.run_identity.engine_revision),
            *profile.required_attributes,
        ):
            if name not in handle.attrs:
                raise ValueError(f"Pinned HDF5 attribute {name!r} is absent.")
            if _attribute_text(handle.attrs[name], name) != expected:
                raise ValueError(
                    f"Pinned HDF5 attribute {name!r} does not match the profile."
                )
        values = _fixed_numeric_dataset(
            handle,
            profile.score_dataset,
            expected_shape=profile.score.shape,
            expected_dtype=profile.score.dtype,
            max_logical_bytes=max_logical_bytes,
        )
        uncertainty = (
            None
            if profile.uncertainty_dataset is None
            else _fixed_numeric_dataset(
                handle,
                profile.uncertainty_dataset,
                expected_shape=expected_uncertainty_shape,
                expected_dtype=None,
                max_logical_bytes=max_logical_bytes,
            )
        )
    evidence = RadiationEstimatorEvidence(
        profile.estimator,
        profile.history_count,
        profile.batch_count,
        profile.uncertainty_kind,
        uncertainty,
        profile.correlation_model,
        (reference.manifest_id,),
    )
    score_id = radiation_score_content_id(
        values,
        profile.score,
        profile.run_identity,
        evidence,
        (reference,),
        profile.profile_id,
    )
    status = (
        AdapterStatus.DECLARED_LOSS if profile.declared_losses else AdapterStatus.LOSSLESS
    )
    report = AdapterReport(
        status,
        f"{profile.provider}-HDF5",
        "ExternalRadiationScoreResult",
        source_id=reference.manifest_id,
        target_id=score_id,
        source_profile=AdapterFormatProfile(
            f"{profile.provider}-HDF5",
            qualifiers={
                "external_profile_id": profile.profile_id,
                "provider_revision": profile.run_identity.engine_revision,
                "score_quantity": profile.score.quantity_kind.value,
            },
        ),
        coordinate_mapping=(
            "stored array axes are used exactly in the profile-declared order",
            f"grid affine {profile.score.grid_affine.affine_id}",
        ),
        preserved_fields=(
            "score values",
            "score dtype",
            "score normalization",
            "quantity and unit",
            "grid affine",
            "build/configuration/table/calibration/seed identities",
            "native estimator and correlation declaration",
        ),
        assumptions=(
            "caller-supplied profile metadata is authoritative for semantics absent from HDF5",
            "artifact import does not qualify or execute the provider",
            "result scope is research-only",
        ),
        losses=profile.declared_losses,
        waivers=tuple(
            AdapterWaiver(
                loss,
                "The exact research-only profile explicitly retains this declared source loss.",
            )
            for loss in profile.declared_losses
            if loss.changes_interpretation
        ),
    )
    return ExternalRadiationScoreResult(
        values,
        profile.score,
        profile.run_identity,
        evidence,
        (reference,),
        profile.profile_id,
        report,
    )


__all__ = ["OpenXRayMCHDF5Profile", "import_openxraymc_hdf5"]
