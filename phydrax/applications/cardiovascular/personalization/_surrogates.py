#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....uq import GaussianScaleCalibrator, SplitConformal
from .._quantities import CardiovascularQuantitySpec
from ._cohorts import PreparedLearningCohort, TrainOnlyFeaturePreprocessor


def _identifier(value: str, name: str, /) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{name} must be non-empty.")
    return resolved


def _frozen_finite_array(
    value: ArrayLike, name: str, /, *, ndim: int | None = None
) -> Array:
    array = jnp.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}.")
    if array.size == 0 or bool(jnp.any(~jnp.isfinite(array))):
        raise ValueError(f"{name} must be non-empty and finite.")
    return array


@dataclass(frozen=True, slots=True)
class FixedTopologyReferenceGeometry:
    """Reference coordinates and immutable simplex connectivity for qualification."""

    coordinates_mm: Array
    cells: Array
    topology_id: str
    geometry_id: str = field(init=False)

    def __post_init__(self) -> None:
        points = _frozen_finite_array(
            self.coordinates_mm, "reference coordinates", ndim=2
        )
        cells = jnp.asarray(self.cells, dtype=jnp.int32)
        if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] < 2:
            raise ValueError(
                "Reference cells must be a non-empty rank-2 connectivity array."
            )
        if bool(jnp.any(cells < 0)) or bool(jnp.any(cells >= points.shape[0])):
            raise ValueError("Reference cell connectivity is out of bounds.")
        if any(
            len(set(int(value) for value in row)) != len(row) for row in np.asarray(cells)
        ):
            raise ValueError("Reference cells cannot repeat vertices.")
        dimension = int(points.shape[1])
        width = int(cells.shape[1])
        if not (width == 2 or width == dimension + 1 or (dimension == 3 and width == 3)):
            raise ValueError(
                "Cells must be line, full-simplex, or embedded 3-D triangle cells."
            )
        topology = _identifier(self.topology_id, "topology_id")
        _, reference_positive = _relative_cell_measures(points, points, cells)
        if not bool(jnp.all(reference_positive)):
            raise ValueError("Reference geometry contains degenerate or inverted cells.")
        geometry_id = canonical_fingerprint(
            {
                "kind": "cardiac-fixed-topology-reference-geometry",
                "topology": topology,
                "coordinates_mm": array_tree_fingerprint(points),
                "cells": array_tree_fingerprint(cells),
            }
        )
        object.__setattr__(self, "coordinates_mm", points)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "topology_id", topology)
        object.__setattr__(self, "geometry_id", geometry_id)


@dataclass(frozen=True, slots=True)
class GenerativeGeometryCandidate:
    """An anatomy/motion proposal on a declared existing topology."""

    coordinates_mm: Array
    topology_id: str
    generator_artifact_id: str
    motion_coordinates_mm: Array | None = None
    candidate_id: str = field(init=False)

    def __post_init__(self) -> None:
        points = _frozen_finite_array(
            self.coordinates_mm, "candidate coordinates", ndim=2
        )
        motion = (
            None
            if self.motion_coordinates_mm is None
            else _frozen_finite_array(
                self.motion_coordinates_mm, "motion coordinates", ndim=3
            )
        )
        if motion is not None and motion.shape[1:] != points.shape:
            raise ValueError("Motion frames must use the candidate point layout.")
        topology = _identifier(self.topology_id, "topology_id")
        artifact = _identifier(self.generator_artifact_id, "generator_artifact_id")
        candidate_id = canonical_fingerprint(
            {
                "kind": "generative-fixed-topology-cardiac-candidate",
                "topology": topology,
                "generator_artifact": artifact,
                "coordinates_mm": array_tree_fingerprint(points),
                "motion_coordinates_mm": (
                    None if motion is None else array_tree_fingerprint(motion)
                ),
            }
        )
        object.__setattr__(self, "coordinates_mm", points)
        object.__setattr__(self, "motion_coordinates_mm", motion)
        object.__setattr__(self, "topology_id", topology)
        object.__setattr__(self, "generator_artifact_id", artifact)
        object.__setattr__(self, "candidate_id", candidate_id)


@dataclass(frozen=True, slots=True)
class GeometryQualificationPolicy:
    """Physical admissibility limits for generated anatomy and motion."""

    minimum_cell_measure_ratio: float = 0.1
    maximum_displacement_mm: float = 30.0
    maximum_motion_increment_mm: float = 10.0

    def __post_init__(self) -> None:
        ratio = float(self.minimum_cell_measure_ratio)
        displacement = float(self.maximum_displacement_mm)
        increment = float(self.maximum_motion_increment_mm)
        if not 0.0 < ratio <= 1.0:
            raise ValueError("minimum_cell_measure_ratio must lie in (0, 1].")
        if (
            not isfinite(displacement)
            or displacement <= 0.0
            or not isfinite(increment)
            or increment <= 0.0
        ):
            raise ValueError("Geometry displacement limits must be finite and positive.")
        object.__setattr__(self, "minimum_cell_measure_ratio", ratio)
        object.__setattr__(self, "maximum_displacement_mm", displacement)
        object.__setattr__(self, "maximum_motion_increment_mm", increment)


class GeometryCandidateStatus(StrEnum):
    QUALIFIED = "qualified"
    TOPOLOGY_MISMATCH = "topology_mismatch"
    NONFINITE_OR_LAYOUT_INVALID = "nonfinite_or_layout_invalid"
    INVERTED_OR_DEGENERATE = "inverted_or_degenerate"
    DISPLACEMENT_EXCEEDED = "displacement_exceeded"
    MOTION_INCREMENT_EXCEEDED = "motion_increment_exceeded"


@dataclass(frozen=True, slots=True)
class GeometryQualificationEvidence:
    status: GeometryCandidateStatus
    topology_preserved: bool
    finite: bool
    minimum_cell_measure_ratio: float
    maximum_displacement_mm: float
    maximum_motion_increment_mm: float
    reference_geometry_id: str
    candidate_id: str
    qualification_id: str

    @property
    def qualified(self) -> bool:
        return self.status is GeometryCandidateStatus.QUALIFIED


def qualify_generative_geometry(
    reference: FixedTopologyReferenceGeometry,
    candidate: GenerativeGeometryCandidate,
    policy: GeometryQualificationPolicy,
    /,
) -> GeometryQualificationEvidence:
    """Qualify fixed-topology anatomy/motion without accepting generated geometry."""

    if not isinstance(reference, FixedTopologyReferenceGeometry):
        raise TypeError("reference must be FixedTopologyReferenceGeometry.")
    if not isinstance(candidate, GenerativeGeometryCandidate):
        raise TypeError("candidate must be GenerativeGeometryCandidate.")
    if not isinstance(policy, GeometryQualificationPolicy):
        raise TypeError("policy must be GeometryQualificationPolicy.")
    topology_preserved = (
        candidate.topology_id == reference.topology_id
        and candidate.coordinates_mm.shape == reference.coordinates_mm.shape
    )
    finite = bool(jnp.all(jnp.isfinite(candidate.coordinates_mm)))
    if candidate.motion_coordinates_mm is not None:
        finite = finite and bool(jnp.all(jnp.isfinite(candidate.motion_coordinates_mm)))
    minimum_ratio = 0.0
    maximum_displacement = 0.0
    maximum_increment = 0.0
    if topology_preserved and finite:
        ratio, orientation = _relative_cell_measures(
            reference.coordinates_mm,
            candidate.coordinates_mm,
            reference.cells,
        )
        minimum_ratio = float(jnp.min(ratio))
        topology_preserved = topology_preserved and bool(jnp.all(orientation))
        displacement = candidate.coordinates_mm - reference.coordinates_mm
        maximum_displacement = float(
            jnp.sqrt(jnp.max(contract("...d,...d->...", displacement, displacement)))
        )
        if candidate.motion_coordinates_mm is not None:
            previous = candidate.coordinates_mm
            frame_ratios = [minimum_ratio]
            frame_orientations = [topology_preserved]
            increments = []
            for frame in candidate.motion_coordinates_mm:
                ratio, orientation = _relative_cell_measures(
                    reference.coordinates_mm, frame, reference.cells
                )
                frame_ratios.append(float(jnp.min(ratio)))
                frame_orientations.append(bool(jnp.all(orientation)))
                frame_displacement = frame - reference.coordinates_mm
                maximum_displacement = max(
                    maximum_displacement,
                    float(
                        jnp.sqrt(
                            jnp.max(
                                contract(
                                    "...d,...d->...",
                                    frame_displacement,
                                    frame_displacement,
                                )
                            )
                        )
                    ),
                )
                step = frame - previous
                increments.append(
                    float(jnp.sqrt(jnp.max(contract("...d,...d->...", step, step))))
                )
                previous = frame
            minimum_ratio = min(frame_ratios)
            topology_preserved = topology_preserved and all(frame_orientations)
            maximum_increment = max(increments, default=0.0)
    if candidate.topology_id != reference.topology_id:
        status = GeometryCandidateStatus.TOPOLOGY_MISMATCH
    elif not finite or candidate.coordinates_mm.shape != reference.coordinates_mm.shape:
        status = GeometryCandidateStatus.NONFINITE_OR_LAYOUT_INVALID
    elif not topology_preserved or minimum_ratio < policy.minimum_cell_measure_ratio:
        status = GeometryCandidateStatus.INVERTED_OR_DEGENERATE
    elif maximum_displacement > policy.maximum_displacement_mm:
        status = GeometryCandidateStatus.DISPLACEMENT_EXCEEDED
    elif maximum_increment > policy.maximum_motion_increment_mm:
        status = GeometryCandidateStatus.MOTION_INCREMENT_EXCEEDED
    else:
        status = GeometryCandidateStatus.QUALIFIED
    qualification_id = canonical_fingerprint(
        {
            "kind": "cardiac-generative-geometry-qualification",
            "reference": reference.geometry_id,
            "candidate": candidate.candidate_id,
            "status": status.value,
            "minimum_ratio": minimum_ratio,
            "maximum_displacement_mm": maximum_displacement,
            "maximum_motion_increment_mm": maximum_increment,
        }
    )
    return GeometryQualificationEvidence(
        status,
        topology_preserved,
        finite,
        minimum_ratio,
        maximum_displacement,
        maximum_increment,
        reference.geometry_id,
        candidate.candidate_id,
        qualification_id,
    )


def _relative_cell_measures(
    reference: Array,
    candidate: Array,
    cells: Array,
    /,
) -> tuple[Array, Array]:
    reference_vertices = reference[cells]
    candidate_vertices = candidate[cells]
    width = int(cells.shape[1])
    dimension = int(reference.shape[1])
    if width == 2:
        reference_edges = reference_vertices[:, 1] - reference_vertices[:, 0]
        candidate_edges = candidate_vertices[:, 1] - candidate_vertices[:, 0]
        reference_measure = jnp.sqrt(
            contract("...d,...d->...", reference_edges, reference_edges)
        )
        candidate_measure = jnp.sqrt(
            contract("...d,...d->...", candidate_edges, candidate_edges)
        )
        orientation = candidate_measure > 0.0
    elif dimension == 3 and width == 3:
        reference_cross = jnp.cross(
            reference_vertices[:, 1] - reference_vertices[:, 0],
            reference_vertices[:, 2] - reference_vertices[:, 0],
        )
        candidate_cross = jnp.cross(
            candidate_vertices[:, 1] - candidate_vertices[:, 0],
            candidate_vertices[:, 2] - candidate_vertices[:, 0],
        )
        reference_measure = jnp.sqrt(
            contract("...d,...d->...", reference_cross, reference_cross)
        )
        candidate_measure = jnp.sqrt(
            contract("...d,...d->...", candidate_cross, candidate_cross)
        )
        orientation = contract("...d,...d->...", reference_cross, candidate_cross) > 0.0
    else:
        reference_edges = reference_vertices[:, 1:] - reference_vertices[:, :1]
        candidate_edges = candidate_vertices[:, 1:] - candidate_vertices[:, :1]
        reference_determinant = _determinant(reference_edges)
        candidate_determinant = _determinant(candidate_edges)
        reference_measure = jnp.abs(reference_determinant)
        candidate_measure = jnp.abs(candidate_determinant)
        orientation = reference_determinant * candidate_determinant > 0.0
    tolerance = (
        64.0 * jnp.finfo(reference.dtype).eps * jnp.maximum(reference_measure, 1.0)
    )
    orientation = (
        orientation & (reference_measure > tolerance) & (candidate_measure > tolerance)
    )
    ratio = candidate_measure / jnp.maximum(reference_measure, tolerance)
    return ratio, orientation


def _determinant(matrices: Array, /) -> Array:
    dimension = int(matrices.shape[-1])
    if dimension == 1:
        return matrices[..., 0, 0]
    if dimension == 2:
        return (
            matrices[..., 0, 0] * matrices[..., 1, 1]
            - matrices[..., 0, 1] * matrices[..., 1, 0]
        )
    if dimension == 3:
        return (
            matrices[..., 0, 0]
            * (
                matrices[..., 1, 1] * matrices[..., 2, 2]
                - matrices[..., 1, 2] * matrices[..., 2, 1]
            )
            - matrices[..., 0, 1]
            * (
                matrices[..., 1, 0] * matrices[..., 2, 2]
                - matrices[..., 1, 2] * matrices[..., 2, 0]
            )
            + matrices[..., 0, 2]
            * (
                matrices[..., 1, 0] * matrices[..., 2, 1]
                - matrices[..., 1, 1] * matrices[..., 2, 0]
            )
        )
    raise ValueError(
        "Geometry qualification supports simplex dimensions one through three."
    )


@dataclass(frozen=True, slots=True)
class CardiacSurrogateCalibration:
    """Native held-out calibration bound exactly to one prepared cohort partition."""

    scale_calibrator: GaussianScaleCalibrator
    conformal: SplitConformal
    calibration_case_ids: tuple[str, ...]
    split_id: str
    preparation_id: str
    preprocessing_id: str
    empirical_simultaneous_coverage: float
    calibration_id: str

    @classmethod
    def fit(
        cls,
        location: ArrayLike,
        scale: ArrayLike,
        target: ArrayLike,
        prepared: PreparedLearningCohort,
        /,
        *,
        alpha: float = 0.1,
    ) -> CardiacSurrogateCalibration:
        if not isinstance(prepared, PreparedLearningCohort):
            raise TypeError("prepared must be a PreparedLearningCohort.")
        center = jnp.asarray(location, dtype=float)
        raw_scale = jnp.asarray(scale, dtype=float)
        truth = jnp.asarray(target, dtype=float)
        if (
            center.shape != raw_scale.shape
            or center.shape != truth.shape
            or center.ndim < 1
        ):
            raise ValueError("Calibration location, scale, and target shapes must match.")
        case_ids = prepared.split.calibration_ids
        if center.shape[0] != len(case_ids):
            raise ValueError(
                "Calibration arrays must exactly match the prepared calibration partition."
            )
        if (
            bool(jnp.any(~jnp.isfinite(center)))
            or bool(jnp.any(~jnp.isfinite(truth)))
            or bool(jnp.any(~jnp.isfinite(raw_scale)))
            or bool(jnp.any(raw_scale <= 0.0))
        ):
            raise ValueError("Calibration arrays must be finite with positive scales.")
        scale_calibrator = GaussianScaleCalibrator.fit(center, raw_scale, truth)
        calibrated_scale = scale_calibrator(raw_scale)
        standardized = jnp.abs(truth - center) / calibrated_scale
        scores = jnp.max(standardized.reshape((center.shape[0], -1)), axis=1)
        conformal = SplitConformal.calibrate(
            jnp.zeros_like(scores), scores, alpha=alpha, case_dim=0
        )
        coverage = float(jnp.mean(scores <= conformal.radius))
        calibration_id = canonical_fingerprint(
            {
                "kind": "cardiac-held-out-surrogate-calibration",
                "split": prepared.split.split_id,
                "preparation": prepared.preparation_id,
                "preprocessing": prepared.features.preprocessing_id,
                "cases": case_ids,
                "scale_multiplier": float(scale_calibrator.scale_multiplier),
                "conformal_radius": float(conformal.radius),
                "alpha": conformal.alpha,
                "coverage": coverage,
                "target_signature": array_tree_fingerprint(truth),
            }
        )
        return cls(
            scale_calibrator,
            conformal,
            case_ids,
            prepared.split.split_id,
            prepared.preparation_id,
            prepared.features.preprocessing_id,
            coverage,
            calibration_id,
        )

    def interval_half_width(self, raw_scale: ArrayLike, /) -> Array:
        scale = jnp.asarray(raw_scale, dtype=float)
        if bool(jnp.any(~jnp.isfinite(scale))) or bool(jnp.any(scale <= 0.0)):
            raise ValueError("Predictive scale must be finite and strictly positive.")
        return self.conformal.radius * self.scale_calibrator(scale)


@dataclass(frozen=True, slots=True)
class CardiacSurrogateProposalManifest:
    """Immutable provenance binding for a learned proposal, never an authority record."""

    surrogate_artifact_id: str
    operator_contract_id: str
    training_corpus_id: str
    split_id: str
    preprocessing_id: str
    preparation_id: str
    calibration_id: str
    topology_id: str
    output_quantities: tuple[CardiovascularQuantitySpec, ...]
    manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        identities = {
            name: _identifier(value, name)
            for name, value in (
                ("surrogate_artifact_id", self.surrogate_artifact_id),
                ("operator_contract_id", self.operator_contract_id),
                ("training_corpus_id", self.training_corpus_id),
                ("split_id", self.split_id),
                ("preprocessing_id", self.preprocessing_id),
                ("preparation_id", self.preparation_id),
                ("calibration_id", self.calibration_id),
                ("topology_id", self.topology_id),
            )
        }
        quantities = tuple(self.output_quantities)
        if not quantities or any(
            not isinstance(quantity, CardiovascularQuantitySpec)
            for quantity in quantities
        ):
            raise TypeError(
                "output_quantities must contain cardiovascular quantity specs."
            )
        quantity_ids = tuple(quantity.quantity_id for quantity in quantities)
        if len(set(quantity_ids)) != len(quantity_ids):
            raise ValueError("Surrogate output quantities must be unique.")
        for name, value in identities.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "output_quantities", quantities)
        object.__setattr__(
            self,
            "manifest_id",
            canonical_fingerprint(
                {
                    "kind": "cardiac-surrogate-proposal-manifest",
                    **identities,
                    "output_quantities": quantity_ids,
                    "authority": "proposal-only",
                }
            ),
        )


class SurrogateInputStatus(StrEnum):
    SUPPORTED = "supported"
    OOD_REFUSAL = "ood_refusal"
    CONTRACT_REFUSAL = "contract_refusal"


@dataclass(frozen=True, slots=True)
class SurrogateInputEvidence:
    """Pre-inference support decision so an operator need not run on OOD input."""

    status: SurrogateInputStatus
    mahalanobis_squared: float | None
    reason: str
    preprocessing_id: str
    topology_id: str
    evidence_id: str

    @property
    def supported(self) -> bool:
        return self.status is SurrogateInputStatus.SUPPORTED


def assess_surrogate_input(
    manifest: CardiacSurrogateProposalManifest,
    parameters: Mapping[str, float] | Sequence[tuple[str, float]],
    preprocessing: TrainOnlyFeaturePreprocessor,
    /,
    *,
    topology_id: str,
) -> SurrogateInputEvidence:
    """Refuse unsupported input before invoking a trained operator."""

    if not isinstance(manifest, CardiacSurrogateProposalManifest):
        raise TypeError("manifest must be CardiacSurrogateProposalManifest.")
    if not isinstance(preprocessing, TrainOnlyFeaturePreprocessor):
        raise TypeError("preprocessing must be TrainOnlyFeaturePreprocessor.")
    topology = _identifier(topology_id, "topology_id")
    if (
        manifest.preprocessing_id != preprocessing.preprocessing_id
        or manifest.topology_id != topology
    ):
        status = SurrogateInputStatus.CONTRACT_REFUSAL
        distance = None
        reason = "surrogate manifest does not match preprocessing or topology"
    else:
        distance = preprocessing.mahalanobis_squared(parameters)
        if distance > preprocessing.support_mahalanobis_squared:
            status = SurrogateInputStatus.OOD_REFUSAL
            reason = "parameters lie outside the train-fitted covariance support"
        else:
            status = SurrogateInputStatus.SUPPORTED
            reason = "parameters and topology lie inside train-fitted support"
    evidence_id = canonical_fingerprint(
        {
            "kind": "cardiac-surrogate-pre-inference-support",
            "manifest": manifest.manifest_id,
            "preprocessing": preprocessing.preprocessing_id,
            "topology": topology,
            "status": status.value,
            "mahalanobis_squared": distance,
        }
    )
    return SurrogateInputEvidence(
        status,
        distance,
        reason,
        preprocessing.preprocessing_id,
        topology,
        evidence_id,
    )


@dataclass(frozen=True, slots=True)
class SurrogateRefusalPolicy:
    """Fail-closed OOD, calibrated-width, and geometry proposal gates."""

    maximum_interval_half_width: float
    require_geometry_evidence: bool = True

    def __post_init__(self) -> None:
        width = float(self.maximum_interval_half_width)
        if not isfinite(width) or width <= 0.0:
            raise ValueError("maximum_interval_half_width must be finite and positive.")
        object.__setattr__(self, "maximum_interval_half_width", width)


class SurrogateProposalStatus(StrEnum):
    QUALIFIED_FOR_REANALYSIS = "qualified_for_reanalysis"
    OOD_REFUSAL = "ood_refusal"
    UNCERTAINTY_REFUSAL = "uncertainty_refusal"
    GEOMETRY_REFUSAL = "geometry_refusal"
    INVALID_OUTPUT_REFUSAL = "invalid_output_refusal"
    CONTRACT_REFUSAL = "contract_refusal"


@dataclass(frozen=True, slots=True)
class CardiacSurrogateProposal:
    """Calibrated learned output that can only seed a full native reanalysis."""

    predicted_state: Array | None
    interval_half_width: Array | None
    status: SurrogateProposalStatus
    reason: str
    parameter_mahalanobis_squared: float | None
    manifest_id: str
    preprocessing_id: str
    calibration_id: str
    topology_id: str
    geometry_evidence: GeometryQualificationEvidence | None
    proposal_id: str

    @property
    def qualified_for_reanalysis(self) -> bool:
        return self.status is SurrogateProposalStatus.QUALIFIED_FOR_REANALYSIS

    @property
    def accepted(self) -> bool:
        """Learned output is never an accepted cardiovascular result."""

        return False


def propose_cardiac_surrogate(
    manifest: CardiacSurrogateProposalManifest,
    parameters: Mapping[str, float] | Sequence[tuple[str, float]],
    predicted_state: ArrayLike,
    raw_predictive_scale: ArrayLike,
    preprocessing: TrainOnlyFeaturePreprocessor,
    calibration: CardiacSurrogateCalibration,
    policy: SurrogateRefusalPolicy,
    /,
    *,
    topology_id: str,
    geometry_evidence: GeometryQualificationEvidence | None = None,
) -> CardiacSurrogateProposal:
    """Apply train-only support, held-out calibration, and geometry refusal gates."""

    if not isinstance(manifest, CardiacSurrogateProposalManifest):
        raise TypeError("manifest must be CardiacSurrogateProposalManifest.")
    if not isinstance(preprocessing, TrainOnlyFeaturePreprocessor):
        raise TypeError("preprocessing must be TrainOnlyFeaturePreprocessor.")
    if not isinstance(calibration, CardiacSurrogateCalibration):
        raise TypeError("calibration must be CardiacSurrogateCalibration.")
    if not isinstance(policy, SurrogateRefusalPolicy):
        raise TypeError("policy must be SurrogateRefusalPolicy.")
    topology = _identifier(topology_id, "topology_id")
    prediction = jnp.asarray(predicted_state, dtype=float)
    scale = jnp.asarray(raw_predictive_scale, dtype=float)
    support = assess_surrogate_input(
        manifest,
        parameters,
        preprocessing,
        topology_id=topology,
    )
    contract_valid = (
        support.status is not SurrogateInputStatus.CONTRACT_REFUSAL
        and manifest.calibration_id == calibration.calibration_id
        and manifest.split_id == calibration.split_id
        and manifest.preparation_id == calibration.preparation_id
        and manifest.preprocessing_id == calibration.preprocessing_id
    )
    output_valid = (
        prediction.shape == scale.shape
        and prediction.size > 0
        and bool(jnp.all(jnp.isfinite(prediction)))
        and bool(jnp.all(jnp.isfinite(scale)))
        and bool(jnp.all(scale > 0.0))
    )
    distance = support.mahalanobis_squared
    width = calibration.interval_half_width(scale) if output_valid else None
    if not contract_valid:
        status = SurrogateProposalStatus.CONTRACT_REFUSAL
        reason = (
            "surrogate manifest does not match preprocessing, calibration, or topology"
        )
    elif support.status is SurrogateInputStatus.OOD_REFUSAL:
        status = SurrogateProposalStatus.OOD_REFUSAL
        reason = support.reason
    elif not output_valid or width is None:
        status = SurrogateProposalStatus.INVALID_OUTPUT_REFUSAL
        reason = (
            "surrogate output or predictive scale is nonfinite, empty, or shape-invalid"
        )
    elif float(jnp.max(width)) > policy.maximum_interval_half_width:
        status = SurrogateProposalStatus.UNCERTAINTY_REFUSAL
        reason = "held-out calibrated predictive width exceeds the declared limit"
    elif policy.require_geometry_evidence and (
        geometry_evidence is None or not geometry_evidence.qualified
    ):
        status = SurrogateProposalStatus.GEOMETRY_REFUSAL
        reason = "generated anatomy or motion lacks qualifying fixed-topology evidence"
    else:
        status = SurrogateProposalStatus.QUALIFIED_FOR_REANALYSIS
        reason = (
            "proposal may initialize full native reanalysis; it is not accepted output"
        )
    retained_prediction = prediction if output_valid else None
    retained_width = width if output_valid else None
    proposal_id = canonical_fingerprint(
        {
            "kind": "cardiac-surrogate-proposal",
            "manifest": manifest.manifest_id,
            "status": status.value,
            "parameters": tuple(
                sorted(
                    (str(name), float(value)) for name, value in dict(parameters).items()
                )
            )
            if isinstance(parameters, Mapping)
            else tuple(sorted((str(name), float(value)) for name, value in parameters)),
            "prediction": (
                None
                if retained_prediction is None
                else array_tree_fingerprint(retained_prediction)
            ),
            "calibrated_half_width": (
                None if retained_width is None else array_tree_fingerprint(retained_width)
            ),
            "geometry_qualification": (
                None if geometry_evidence is None else geometry_evidence.qualification_id
            ),
        }
    )
    return CardiacSurrogateProposal(
        retained_prediction,
        retained_width,
        status,
        reason,
        distance,
        manifest.manifest_id,
        preprocessing.preprocessing_id,
        calibration.calibration_id,
        topology,
        geometry_evidence,
        proposal_id,
    )


__all__ = [
    "SurrogateInputEvidence",
    "SurrogateInputStatus",
    "assess_surrogate_input",
    "CardiacSurrogateCalibration",
    "CardiacSurrogateProposal",
    "CardiacSurrogateProposalManifest",
    "FixedTopologyReferenceGeometry",
    "GenerativeGeometryCandidate",
    "GeometryCandidateStatus",
    "GeometryQualificationEvidence",
    "GeometryQualificationPolicy",
    "SurrogateProposalStatus",
    "SurrogateRefusalPolicy",
    "propose_cardiac_surrogate",
    "qualify_generative_geometry",
]
