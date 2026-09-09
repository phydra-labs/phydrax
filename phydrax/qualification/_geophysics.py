#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast, Literal

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._evidence import QualificationEvidence
from ._reference import ReferenceArtifactManifest


GeophysicalReferenceKind = Literal["external-oracle", "field"]


def _identifier(value: str, name: str, /) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be nonempty.")
    return normalized


def _record_float(record: Mapping[str, object], name: str, /) -> float:
    value = record[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Reference recipe {name} must be a real number.")
    return float(value)


def _record_int(record: Mapping[str, object], name: str, /) -> int:
    value = record[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Reference recipe {name} must be an integer.")
    return value


class GeophysicalReferenceComparison(StrictModule, NonTrainableState):
    """Deterministic comparison against one exact governed reference artifact."""

    recipe_id: str = eqx.field(static=True)
    reference_kind: GeophysicalReferenceKind = eqx.field(static=True)
    reference_manifest_id: str = eqx.field(static=True)
    valid_sample_count: int = eqx.field(static=True)
    maximum_absolute_error: float = eqx.field(static=True)
    relative_l2_error: float = eqx.field(static=True)
    maximum_normalized_error: float = eqx.field(static=True)
    standardized_rms: float | None = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    comparison_id: str = eqx.field(static=True)

    def __init__(
        self,
        recipe_id: str,
        reference_kind: GeophysicalReferenceKind,
        reference_manifest_id: str,
        valid_sample_count: int,
        maximum_absolute_error: float,
        relative_l2_error: float,
        maximum_normalized_error: float,
        standardized_rms: float | None,
        passed: bool,
        /,
    ):
        if reference_kind not in ("external-oracle", "field"):
            raise ValueError("Reference comparison kind is invalid.")
        self.reference_kind = reference_kind
        self.recipe_id = _identifier(recipe_id, "reference recipe ID")
        self.reference_manifest_id = _identifier(
            reference_manifest_id, "reference manifest ID"
        )
        self.valid_sample_count = int(valid_sample_count)
        self.maximum_absolute_error = float(maximum_absolute_error)
        self.relative_l2_error = float(relative_l2_error)
        self.maximum_normalized_error = float(maximum_normalized_error)
        self.standardized_rms = (
            None if standardized_rms is None else float(standardized_rms)
        )
        self.passed = bool(passed)
        metrics = (
            self.maximum_absolute_error,
            self.relative_l2_error,
            self.maximum_normalized_error,
        )
        if self.valid_sample_count <= 0 or any(
            not math.isfinite(value) for value in metrics
        ):
            raise ValueError("Reference comparison metrics must be finite and nonempty.")
        if self.standardized_rms is not None and not math.isfinite(self.standardized_rms):
            raise ValueError("Reference standardized RMS must be finite when present.")
        self.comparison_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "geophysical-reference-comparison",
            "reference_kind": self.reference_kind,
            "recipe_id": self.recipe_id,
            "reference_manifest_id": self.reference_manifest_id,
            "valid_sample_count": self.valid_sample_count,
            "maximum_absolute_error": self.maximum_absolute_error,
            "relative_l2_error": self.relative_l2_error,
            "maximum_normalized_error": self.maximum_normalized_error,
            "standardized_rms": self.standardized_rms,
            "passed": self.passed,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._content_record(), "comparison_id": self.comparison_id}

    def evidence(
        self,
        subject_ids: Sequence[str],
        /,
        *,
        build_id: str,
        environment_id: str,
        backend: str,
        topology: str,
        precision: str,
        reduction: str,
        replay_id: str,
        campaign_start_record_ids: Sequence[str],
        campaign_observation_record_ids: Sequence[str],
        reviewer_id: str,
        issued_at: int,
        expires_at: int,
        requalification_triggers: Sequence[str] = (),
    ) -> QualificationEvidence:
        """Bind the comparison outcome to the repository qualification ledger."""
        if self.reference_kind == "field" and (
            not campaign_start_record_ids or not campaign_observation_record_ids
        ):
            raise ValueError(
                "Field qualification evidence requires campaign start and observation records."
            )
        return QualificationEvidence(
            "reference",
            "passed" if self.passed else "failed",
            subject_ids,
            build_id=build_id,
            environment_id=environment_id,
            backend=backend,
            topology=topology,
            precision=precision,
            reduction=reduction,
            replay_id=replay_id,
            criteria_ids=(self.recipe_id,),
            raw_artifact_ids=(self.reference_manifest_id, self.comparison_id),
            campaign_start_record_ids=campaign_start_record_ids,
            campaign_observation_record_ids=campaign_observation_record_ids,
            reviewer_id=reviewer_id,
            issued_at=issued_at,
            expires_at=expires_at,
            reason=(
                "governed-reference-criteria-satisfied"
                if self.passed
                else "governed-reference-criteria-failed"
            ),
            requalification_triggers=requalification_triggers,
        )


class GeophysicalReferenceRecipe(StrictModule, NonTrainableState):
    """Pinned external-oracle or field-data comparison semantics."""

    reference_kind: GeophysicalReferenceKind = eqx.field(static=True)
    modality: str = eqx.field(static=True)
    case_name: str = eqx.field(static=True)
    observable: str = eqx.field(static=True)
    source_locator: str = eqx.field(static=True)
    artifact: ReferenceArtifactManifest
    coordinate_contract_id: str = eqx.field(static=True)
    time_contract_id: str | None = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    maximum_standardized_rms: float | None = eqx.field(static=True)
    minimum_valid_samples: int = eqx.field(static=True)
    maximum_samples: int = eqx.field(static=True)
    recipe_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_kind: GeophysicalReferenceKind,
        modality: str,
        case_name: str,
        observable: str,
        source_locator: str,
        artifact: ReferenceArtifactManifest,
        coordinate_contract_id: str,
        /,
        *,
        time_contract_id: str | None = None,
        absolute_tolerance: float,
        relative_tolerance: float,
        maximum_standardized_rms: float | None = None,
        maximum_samples: int,
        minimum_valid_samples: int = 1,
    ):
        if reference_kind not in ("external-oracle", "field"):
            raise ValueError("Reference kind must be external-oracle or field.")
        if not isinstance(artifact, ReferenceArtifactManifest):
            raise TypeError("Geophysical reference recipe requires an artifact manifest.")
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        standardized = (
            None if maximum_standardized_rms is None else float(maximum_standardized_rms)
        )
        minimum, maximum = int(minimum_valid_samples), int(maximum_samples)
        if (
            not math.isfinite(absolute)
            or absolute < 0
            or not math.isfinite(relative)
            or relative < 0
            or absolute == relative == 0
            or (
                standardized is not None
                and (not math.isfinite(standardized) or standardized <= 0)
            )
            or minimum <= 0
            or maximum < minimum
        ):
            raise ValueError("Reference tolerances and sample threshold are invalid.")
        if reference_kind == "field" and standardized is None:
            raise ValueError("Field qualification requires a standardized RMS criterion.")
        self.reference_kind = reference_kind
        self.modality = _identifier(modality, "reference modality")
        self.case_name = _identifier(case_name, "reference case")
        self.observable = _identifier(observable, "reference observable")
        self.source_locator = _identifier(source_locator, "reference source locator")
        self.artifact = artifact
        self.coordinate_contract_id = _identifier(
            coordinate_contract_id, "reference coordinate contract ID"
        )
        self.time_contract_id = (
            None
            if time_contract_id is None
            else _identifier(time_contract_id, "reference time contract ID")
        )
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.maximum_standardized_rms = standardized
        self.minimum_valid_samples, self.maximum_samples = minimum, maximum
        self.recipe_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "geophysical-reference-recipe",
            "reference_kind": self.reference_kind,
            "modality": self.modality,
            "case_name": self.case_name,
            "observable": self.observable,
            "source_locator": self.source_locator,
            "artifact": self.artifact.manifest_id,
            "coordinate_contract_id": self.coordinate_contract_id,
            "time_contract_id": self.time_contract_id,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "maximum_standardized_rms": self.maximum_standardized_rms,
            "minimum_valid_samples": self.minimum_valid_samples,
            "maximum_samples": self.maximum_samples,
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self._content_record(),
            "artifact_record": self.artifact.to_record(),
            "recipe_id": self.recipe_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> GeophysicalReferenceRecipe:
        if not isinstance(record, Mapping):
            raise TypeError("Geophysical reference recipe record must be a mapping.")
        artifact_record = record["artifact_record"]
        if not isinstance(artifact_record, Mapping):
            raise TypeError("Reference recipe artifact record must be a mapping.")
        time_contract = record["time_contract_id"]
        standardized = record["maximum_standardized_rms"]
        absolute = _record_float(record, "absolute_tolerance")
        relative = _record_float(record, "relative_tolerance")
        standardized_value = (
            None
            if standardized is None
            else _record_float(record, "maximum_standardized_rms")
        )
        value = cls(
            cast(GeophysicalReferenceKind, str(record["reference_kind"])),
            str(record["modality"]),
            str(record["case_name"]),
            str(record["observable"]),
            str(record["source_locator"]),
            ReferenceArtifactManifest.from_record(artifact_record),
            str(record["coordinate_contract_id"]),
            time_contract_id=None if time_contract is None else str(time_contract),
            absolute_tolerance=absolute,
            relative_tolerance=relative,
            maximum_standardized_rms=standardized_value,
            minimum_valid_samples=_record_int(record, "minimum_valid_samples"),
            maximum_samples=_record_int(record, "maximum_samples"),
        )
        if record.get("artifact") != value.artifact.manifest_id:
            raise ValueError("Reference recipe artifact identity is invalid.")
        recorded_id = record.get("recipe_id")
        if recorded_id is not None and str(recorded_id) != value.recipe_id:
            raise ValueError("Reference recipe content address is invalid.")
        return value

    def compare(
        self,
        prediction: ArrayLike,
        reference: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        standard_deviation: ArrayLike | None = None,
    ) -> GeophysicalReferenceComparison:
        predicted = np.asarray(prediction)
        expected = np.asarray(reference)
        if (
            predicted.shape != expected.shape
            or predicted.size == 0
            or predicted.size > self.maximum_samples
        ):
            raise ValueError(
                "Prediction/reference shape or declared sample bound is invalid."
            )
        mask = (
            np.ones(predicted.shape, dtype=bool) if valid is None else np.asarray(valid)
        )
        if mask.dtype != np.bool_ or mask.shape != predicted.shape:
            raise ValueError(
                "Reference validity must be a Boolean array of matching shape."
            )
        count = int(np.count_nonzero(mask))
        if count == 0:
            raise ValueError("Reference comparison contains no valid samples.")
        if np.any(~np.isfinite(predicted[mask])) or np.any(~np.isfinite(expected[mask])):
            raise ValueError("Prediction and reference must be finite where valid.")
        error = np.abs(predicted[mask] - expected[mask]).astype(float)
        expected_magnitude = np.abs(expected[mask]).astype(float)
        envelope = self.absolute_tolerance + self.relative_tolerance * expected_magnitude
        tiny = np.finfo(float).tiny
        maximum_absolute = float(np.max(error))
        maximum_normalized = float(np.max(error / np.maximum(envelope, tiny)))
        denominator = max(
            float(np.linalg.norm(expected_magnitude)),
            self.absolute_tolerance * math.sqrt(count),
            tiny,
        )
        relative_l2 = float(np.linalg.norm(error) / denominator)
        standardized_rms: float | None = None
        if standard_deviation is not None:
            uncertainty = np.asarray(standard_deviation)
            if uncertainty.shape != predicted.shape:
                raise ValueError("Reference uncertainty must match the observable shape.")
            selected = uncertainty[mask]
            if np.any(~np.isfinite(selected)) or np.any(selected <= 0):
                raise ValueError(
                    "Reference uncertainty must be positive finite where valid."
                )
            standardized_rms = float(np.sqrt(np.mean((error / selected) ** 2)))
        if self.maximum_standardized_rms is not None and standardized_rms is None:
            raise ValueError("This reference recipe requires per-sample uncertainty.")
        passed = count >= self.minimum_valid_samples and bool(np.all(error <= envelope))
        if self.maximum_standardized_rms is not None:
            passed = passed and (
                cast(float, standardized_rms) <= self.maximum_standardized_rms
            )
        return GeophysicalReferenceComparison(
            self.recipe_id,
            self.reference_kind,
            self.artifact.manifest_id,
            count,
            maximum_absolute,
            relative_l2,
            maximum_normalized,
            standardized_rms,
            passed,
        )


__all__ = [
    "GeophysicalReferenceComparison",
    "GeophysicalReferenceKind",
    "GeophysicalReferenceRecipe",
]
