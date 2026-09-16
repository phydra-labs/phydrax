#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Governed, research-only contracts for imported external radiation scores."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from jaxtyping import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...imaging import ImageIndexAffine
from ...interchange import AdapterLoss, AdapterReport, AdapterStatus
from ...measurement import (
    QuantitySpec,
    RadiationQuantityKind,
    resolve_radiation_quantity,
)
from ...qualification import ReferenceArtifactManifest
from ...units import UnitDefinition


_RESEARCH_ONLY = "research-only"
_DOSE_KINDS = frozenset(
    {
        RadiationQuantityKind.ABSORBED_DOSE,
        RadiationQuantityKind.DOSE_TO_WATER,
        RadiationQuantityKind.DOSE_TO_MEDIUM,
    }
)
_SUPPORTED_SCORE_KINDS = _DOSE_KINDS | frozenset(
    {
        RadiationQuantityKind.KERMA,
        RadiationQuantityKind.RELATIVE_DOSE,
        RadiationQuantityKind.LET,
    }
)
_UNCERTAINTY_KINDS = frozenset(
    {"unreported", "standard-error", "relative-standard-error", "covariance"}
)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    result = value.strip()
    if not result or result != value:
        raise ValueError(f"{name} must be nonempty canonical text.")
    return result


def _identifiers(
    values: Sequence[str], name: str, /, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence, not one string.")
    result = tuple(_identifier(value, name) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique.")
    return result


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _manifests(
    values: Sequence[ReferenceArtifactManifest], name: str, /
) -> tuple[ReferenceArtifactManifest, ...]:
    result = tuple(values)
    if not result or any(
        not isinstance(value, ReferenceArtifactManifest) for value in result
    ):
        raise TypeError(f"{name} must contain ReferenceArtifactManifest values.")
    identifiers = tuple(value.manifest_id for value in result)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must have unique manifest identities.")
    return result


def _readonly(value: ArrayLike, name: str, /) -> np.ndarray:
    array = np.array(value, copy=True)
    if array.dtype.hasobject:
        raise TypeError(f"{name} must not use an object dtype.")
    array.setflags(write=False)
    return array


def _normal_dtype(value: str | np.dtype, /) -> np.dtype:
    dtype = np.dtype(value)
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise TypeError("Radiation scores require one fixed-width scalar dtype.")
    if not np.issubdtype(dtype, np.floating):
        raise TypeError("Radiation scores require floating-point storage.")
    return dtype


def _manifest_ids(values: Sequence[ReferenceArtifactManifest], /) -> tuple[str, ...]:
    return tuple(value.manifest_id for value in values)


def _deduplicate_manifests(
    values: Sequence[ReferenceArtifactManifest], /
) -> tuple[ReferenceArtifactManifest, ...]:
    by_id: dict[str, ReferenceArtifactManifest] = {}
    for value in values:
        if not isinstance(value, ReferenceArtifactManifest):
            raise TypeError(
                "Governing references must be ReferenceArtifactManifest values."
            )
        previous = by_id.get(value.manifest_id)
        if previous is not None and previous.to_record() != value.to_record():
            raise ValueError("One manifest identity has conflicting content.")
        by_id[value.manifest_id] = value
    return tuple(by_id[key] for key in sorted(by_id))


@dataclass(frozen=True, slots=True)
class ExternalRadiationRunIdentity:
    """Exact retained run lineage; it does not execute or discover an engine."""

    run_id: str
    engine: str
    engine_revision: str
    build: ReferenceArtifactManifest
    configuration: ReferenceArtifactManifest
    tables: tuple[ReferenceArtifactManifest, ...]
    calibrations: tuple[ReferenceArtifactManifest, ...]
    seeds: tuple[ReferenceArtifactManifest, ...]
    identity_id: str = field(init=False)

    def __post_init__(self) -> None:
        run = _identifier(self.run_id, "run_id")
        engine = _identifier(self.engine, "engine")
        revision = _identifier(self.engine_revision, "engine_revision")
        if not isinstance(self.build, ReferenceArtifactManifest):
            raise TypeError("build must be a ReferenceArtifactManifest.")
        if not isinstance(self.configuration, ReferenceArtifactManifest):
            raise TypeError("configuration must be a ReferenceArtifactManifest.")
        tables = _manifests(self.tables, "tables")
        calibrations = _manifests(self.calibrations, "calibrations")
        seeds = _manifests(self.seeds, "seeds")
        references = (self.build, self.configuration, *tables, *calibrations, *seeds)
        if len(set(_manifest_ids(references))) != len(references):
            raise ValueError(
                "Build, configuration, table, calibration, and seed identities must be distinct."
            )
        object.__setattr__(self, "run_id", run)
        object.__setattr__(self, "engine", engine)
        object.__setattr__(self, "engine_revision", revision)
        object.__setattr__(self, "tables", tables)
        object.__setattr__(self, "calibrations", calibrations)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(
            self,
            "identity_id",
            canonical_fingerprint(
                {
                    "kind": "external-radiation-run-identity",
                    "run": run,
                    "engine": engine,
                    "engine_revision": revision,
                    "build": self.build.manifest_id,
                    "configuration": self.configuration.manifest_id,
                    "tables": list(_manifest_ids(tables)),
                    "calibrations": list(_manifest_ids(calibrations)),
                    "seeds": list(_manifest_ids(seeds)),
                }
            ),
        )

    @property
    def references(self) -> tuple[ReferenceArtifactManifest, ...]:
        return (
            self.build,
            self.configuration,
            *self.tables,
            *self.calibrations,
            *self.seeds,
        )

    def require_rights(
        self,
        /,
        *,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ) -> None:
        for reference in self.references:
            reference.require_rights(
                commercial_use=commercial_use,
                redistribution=redistribution,
                training_use=training_use,
                export=export,
            )


@dataclass(frozen=True, slots=True)
class RadiationScoreDefinition:
    """Pinned quantity, normalization, storage, and grid meaning for one score."""

    name: str
    quantity_kind: RadiationQuantityKind
    unit: UnitDefinition
    normalization: str
    shape: tuple[int, ...]
    dtype: str | np.dtype
    grid_affine: ImageIndexAffine
    axes: tuple[str, ...] = ("x", "y", "z")
    support_association: str = "voxel-cell-average"
    reference_configuration: str = ""
    representation: str = "voxel-grid"
    quantity: QuantitySpec = field(init=False)
    definition_id: str = field(init=False)

    def __post_init__(self) -> None:
        name = _identifier(self.name, "score name")
        if not isinstance(self.quantity_kind, RadiationQuantityKind):
            raise TypeError("quantity_kind must be RadiationQuantityKind.")
        if self.quantity_kind not in _SUPPORTED_SCORE_KINDS:
            raise ValueError(
                "External score quantity must be dose, kerma, relative dose, or LET."
            )
        if not isinstance(self.unit, UnitDefinition):
            raise TypeError("unit must be UnitDefinition.")
        normalization = _identifier(self.normalization, "score normalization")
        representation = _identifier(self.representation, "score representation")
        if representation not in ("voxel-grid", "dij"):
            raise ValueError("score representation must be voxel-grid or dij.")
        shape = tuple(
            _positive_integer(value, "score shape entry") for value in self.shape
        )
        expected_rank = 4 if representation == "dij" else 3
        if len(shape) != expected_rank:
            raise ValueError(
                f"{representation} scores require rank {expected_rank} storage."
            )
        axes = _identifiers(self.axes, "score axes")
        if len(axes) != len(shape):
            raise ValueError("Score axes must label every stored dimension.")
        if representation == "dij":
            if self.quantity_kind not in _DOSE_KINDS:
                raise ValueError(
                    "Dij is a dose influence representation, not another quantity."
                )
            if axes[-1] not in ("beamlet", "source-element"):
                raise ValueError(
                    "Dij requires an explicit trailing beamlet/source-element axis."
                )
        elif axes[-1] in ("beamlet", "source-element"):
            raise ValueError("Beamlet axes require the explicit dij representation.")
        dtype = _normal_dtype(self.dtype)
        if not isinstance(self.grid_affine, ImageIndexAffine):
            raise TypeError("grid_affine must be ImageIndexAffine.")
        support = _identifier(self.support_association, "support_association")
        reference = _identifier(self.reference_configuration, "reference_configuration")
        if support in ("unspecified", "generic") or reference in (
            "absolute",
            "unspecified",
            "generic",
        ):
            raise ValueError(
                "Radiation score support and reference configuration must be profile-specific."
            )
        quantity = resolve_radiation_quantity(
            name,
            self.quantity_kind,
            self.unit,
            axes=axes,
            sign_convention="nonnegative",
            support_association=support,
            reference_configuration=reference,
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype.str)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "support_association", support)
        object.__setattr__(self, "reference_configuration", reference)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(
            self,
            "definition_id",
            canonical_fingerprint(
                {
                    "kind": "external-radiation-score-definition",
                    "name": name,
                    "quantity": quantity.quantity_id,
                    "normalization": normalization,
                    "shape": list(shape),
                    "dtype": dtype.str,
                    "grid_affine": self.grid_affine.affine_id,
                    "representation": representation,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class RadiationEstimatorEvidence:
    """Native estimator uncertainty and correlation declaration, without repair."""

    estimator: str
    history_count: int
    batch_count: int
    uncertainty_kind: str
    uncertainty_values: np.ndarray | None
    correlation_model: str
    source_manifest_ids: tuple[str, ...]
    evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        estimator = _identifier(self.estimator, "estimator")
        histories = _positive_integer(self.history_count, "history_count")
        batches = _positive_integer(self.batch_count, "batch_count")
        kind = _identifier(self.uncertainty_kind, "uncertainty_kind")
        if kind not in _UNCERTAINTY_KINDS:
            raise ValueError("Unknown radiation estimator uncertainty kind.")
        correlation = _identifier(self.correlation_model, "correlation_model")
        sources = _identifiers(self.source_manifest_ids, "estimator source manifests")
        uncertainty = (
            None
            if self.uncertainty_values is None
            else _readonly(self.uncertainty_values, "uncertainty_values")
        )
        if kind == "unreported":
            if uncertainty is not None or correlation != "unreported":
                raise ValueError(
                    "Unreported uncertainty cannot carry values or a correlation model."
                )
        else:
            if uncertainty is None:
                raise ValueError("Reported estimator uncertainty requires source values.")
            if not np.issubdtype(uncertainty.dtype, np.floating):
                raise TypeError("Estimator uncertainty requires floating-point storage.")
            if np.any(~np.isfinite(uncertainty)):
                raise ValueError("Estimator uncertainty values must be finite.")
            if correlation in ("unreported", "assumed-independent"):
                raise ValueError(
                    "Reported uncertainty requires upstream correlation evidence; independence is never fabricated."
                )
            if kind == "covariance":
                if (
                    uncertainty.ndim != 2
                    or uncertainty.shape[0] != uncertainty.shape[1]
                    or not np.allclose(
                        uncertainty,
                        uncertainty.T,
                        rtol=0.0,
                        atol=64.0 * np.finfo(uncertainty.dtype).eps,
                    )
                ):
                    raise ValueError(
                        "Covariance evidence must be a finite symmetric matrix."
                    )
                eigenvalues = np.linalg.eigvalsh(uncertainty)
                tolerance = (
                    256.0
                    * np.finfo(uncertainty.dtype).eps
                    * max(1.0, float(np.max(np.abs(uncertainty))))
                )
                if float(np.min(eigenvalues)) < -tolerance:
                    raise ValueError("Covariance evidence must be positive semidefinite.")
            elif np.any(uncertainty < 0.0):
                raise ValueError("Pointwise estimator uncertainty must be nonnegative.")
        object.__setattr__(self, "estimator", estimator)
        object.__setattr__(self, "history_count", histories)
        object.__setattr__(self, "batch_count", batches)
        object.__setattr__(self, "uncertainty_kind", kind)
        object.__setattr__(self, "uncertainty_values", uncertainty)
        object.__setattr__(self, "correlation_model", correlation)
        object.__setattr__(self, "source_manifest_ids", sources)
        object.__setattr__(
            self,
            "evidence_id",
            canonical_fingerprint(
                {
                    "kind": "radiation-estimator-evidence",
                    "estimator": estimator,
                    "history_count": histories,
                    "batch_count": batches,
                    "uncertainty_kind": kind,
                    "uncertainty": None
                    if uncertainty is None
                    else array_tree_fingerprint(uncertainty),
                    "correlation_model": correlation,
                    "sources": list(sources),
                }
            ),
        )

    @property
    def correlated(self) -> bool:
        return self.uncertainty_kind != "unreported" and self.correlation_model not in (
            "independent",
            "not-applicable",
        )


def radiation_score_content_id(
    values: ArrayLike,
    definition: RadiationScoreDefinition,
    run_identity: ExternalRadiationRunIdentity,
    estimator_evidence: RadiationEstimatorEvidence,
    artifact_references: Sequence[ReferenceArtifactManifest],
    profile_id: str,
    /,
) -> str:
    """Content identity shared by profile reports and governed results."""

    return canonical_fingerprint(
        {
            "kind": "external-radiation-score",
            "values": array_tree_fingerprint(np.asarray(values)),
            "definition": definition.definition_id,
            "run": run_identity.identity_id,
            "estimator": estimator_evidence.evidence_id,
            "artifacts": list(_manifest_ids(tuple(artifact_references))),
            "profile": _identifier(profile_id, "profile_id"),
            "scope": _RESEARCH_ONLY,
        }
    )


@dataclass(frozen=True, slots=True)
class ExternalRadiationScoreResult:
    """One checksum-governed imported score; never a clinical dose product."""

    values: np.ndarray
    definition: RadiationScoreDefinition
    run_identity: ExternalRadiationRunIdentity
    estimator_evidence: RadiationEstimatorEvidence
    artifact_references: tuple[ReferenceArtifactManifest, ...]
    profile_id: str
    report: AdapterReport
    intended_use: str = field(init=False, default=_RESEARCH_ONLY)
    references: tuple[ReferenceArtifactManifest, ...] = field(init=False)
    score_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.definition, RadiationScoreDefinition):
            raise TypeError("definition must be RadiationScoreDefinition.")
        if not isinstance(self.run_identity, ExternalRadiationRunIdentity):
            raise TypeError("run_identity must be ExternalRadiationRunIdentity.")
        if not isinstance(self.estimator_evidence, RadiationEstimatorEvidence):
            raise TypeError("estimator_evidence must be RadiationEstimatorEvidence.")
        artifacts = _manifests(self.artifact_references, "artifact_references")
        profile = _identifier(self.profile_id, "profile_id")
        if not isinstance(self.report, AdapterReport) or not self.report.valid:
            raise ValueError("A governed score requires one valid adapter report.")
        values = _readonly(self.values, "score values")
        if values.shape != self.definition.shape:
            raise ValueError("Imported score shape differs from its pinned definition.")
        if values.dtype.str != self.definition.dtype:
            raise TypeError("Imported score dtype differs from its pinned definition.")
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                "External radiation score values must be finite and nonnegative."
            )
        uncertainty = self.estimator_evidence.uncertainty_values
        if uncertainty is not None:
            if self.estimator_evidence.uncertainty_kind == "covariance":
                expected = math.prod(self.definition.shape)
                if uncertainty.shape != (expected, expected):
                    raise ValueError(
                        "Estimator covariance does not cover every score value."
                    )
            elif uncertainty.shape != self.definition.shape:
                raise ValueError(
                    "Pointwise estimator uncertainty must match score shape."
                )
        references = _deduplicate_manifests((*artifacts, *self.run_identity.references))
        if not set(self.estimator_evidence.source_manifest_ids).issubset(
            set(_manifest_ids(references))
        ):
            raise ValueError("Estimator evidence cites an ungoverned source manifest.")
        score_id = radiation_score_content_id(
            values,
            self.definition,
            self.run_identity,
            self.estimator_evidence,
            artifacts,
            profile,
        )
        if self.report.target_id != score_id:
            raise ValueError(
                "Adapter report target does not identify the imported score."
            )
        qualifiers = dict(self.report.source_profile.qualifiers)
        if qualifiers.get("external_profile_id") != profile:
            raise ValueError(
                "Adapter report does not pin the exact external score profile."
            )
        if self.report.status not in (
            AdapterStatus.LOSSLESS,
            AdapterStatus.DECLARED_LOSS,
        ):
            raise ValueError("External score adapter report is not admissible.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "artifact_references", artifacts)
        object.__setattr__(self, "profile_id", profile)
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "score_id", score_id)

    @property
    def declared_losses(self) -> tuple[AdapterLoss, ...]:
        return self.report.losses

    @property
    def research_only(self) -> bool:
        return True

    def require_rights(
        self,
        /,
        *,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ) -> None:
        for reference in self.references:
            reference.require_rights(
                commercial_use=commercial_use,
                redistribution=redistribution,
                training_use=training_use,
                export=export,
            )


def require_profile_semantics(
    required: Sequence[str], available: Sequence[str], /
) -> tuple[str, ...]:
    """Reject unknown or absent upstream semantics rather than synthesizing them."""

    required_ = _identifiers(required, "required semantics", allow_empty=True)
    available_ = _identifiers(available, "available semantics")
    missing = tuple(sorted(set(required_) - set(available_)))
    if missing:
        raise ValueError(
            f"External score profile omits required semantics: {list(missing)}"
        )
    return required_


def require_import_rights(
    artifacts: Sequence[ReferenceArtifactManifest],
    run_identity: ExternalRadiationRunIdentity,
    /,
    *,
    commercial_use: bool,
    redistribution: bool,
    training_use: bool,
    export: bool,
) -> None:
    for reference in artifacts:
        reference.require_rights(
            commercial_use=commercial_use,
            redistribution=redistribution,
            training_use=training_use,
            export=export,
        )
    run_identity.require_rights(
        commercial_use=commercial_use,
        redistribution=redistribution,
        training_use=training_use,
        export=export,
    )


__all__ = [
    "ExternalRadiationRunIdentity",
    "ExternalRadiationScoreResult",
    "RadiationEstimatorEvidence",
    "RadiationScoreDefinition",
    "radiation_score_content_id",
]
