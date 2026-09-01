#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._closure import (
    CoordinateLayout,
    CorrelatedGaussianPlan,
    CorrelatedGaussianResult,
    LinearObservationPlan,
    PrecisionCovarianceAction,
    ScientificArtifactEnvelope,
    TheoryVector,
)


class SurveyCoordinate(StrictModule, NonTrainableState):
    domain: str = eqx.field(static=True)
    statistic: str = eqx.field(static=True)
    observable: str = eqx.field(static=True)
    fields: tuple[str, ...] = eqx.field(static=True)
    tracer_ids: tuple[str, ...] = eqx.field(static=True)
    selection_ids: tuple[str, ...] = eqx.field(static=True)
    tomographic_bins: tuple[int, ...] = eqx.field(static=True)
    component: str = eqx.field(static=True)
    coordinate_kind: str = eqx.field(static=True)
    coordinate_value: float = eqx.field(static=True)
    unit: str = eqx.field(static=True)
    frame: str = eqx.field(static=True)
    h_convention: str = eqx.field(static=True)
    coordinate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        domain: str,
        statistic: str,
        observable: str,
        fields: tuple[str, ...],
        tracer_ids: tuple[str, ...],
        selection_ids: tuple[str, ...],
        tomographic_bins: tuple[int, ...],
        component: str,
        coordinate_kind: str,
        coordinate_value: float,
        unit: str,
        frame: str,
        h_convention: str,
    ):
        scalars = tuple(
            str(value).strip()
            for value in (
                domain,
                statistic,
                observable,
                component,
                coordinate_kind,
                unit,
                frame,
                h_convention,
            )
        )
        groups = tuple(
            tuple(str(value).strip() for value in group)
            for group in (fields, tracer_ids, selection_ids)
        )
        bins = tuple(int(value) for value in tomographic_bins)
        value = float(coordinate_value)
        if (
            any(not item for item in scalars)
            or any(not group or any(not item for item in group) for group in groups)
            or not bins
            or not jnp.isfinite(value)
        ):
            raise ValueError("Survey coordinate is incomplete or non-finite.")
        (
            self.domain,
            self.statistic,
            self.observable,
            self.component,
            self.coordinate_kind,
            self.unit,
            self.frame,
            self.h_convention,
        ) = scalars
        self.fields, self.tracer_ids, self.selection_ids = groups
        self.tomographic_bins = bins
        self.coordinate_value = value
        self.coordinate_id = canonical_fingerprint(
            {
                "kind": "survey-coordinate",
                "domain": scalars[0],
                "statistic": scalars[1],
                "observable": scalars[2],
                "fields": list(groups[0]),
                "tracers": list(groups[1]),
                "selections": list(groups[2]),
                "bins": list(bins),
                "component": scalars[3],
                "coordinate_kind": scalars[4],
                "coordinate_value": value,
                "unit": scalars[5],
                "frame": scalars[6],
                "h_convention": scalars[7],
            }
        )

    def label(self, /) -> str:
        return self.coordinate_id


class SurveyTheoryProduct(StrictModule):
    values: Array
    coordinates: tuple[SurveyCoordinate, ...] = eqx.field(static=True)
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        coordinates: tuple[SurveyCoordinate, ...],
        product_id: str,
        /,
    ):
        values_ = jnp.asarray(values)
        if values_.shape != (len(coordinates),) or not product_id:
            raise ValueError("Survey theory values and coordinates disagree.")
        self.values = values_
        self.coordinates = coordinates
        self.product_id = str(product_id)

    def as_theory_vector(self, /) -> TheoryVector:
        return TheoryVector(
            self.values,
            CoordinateLayout(
                tuple(coordinate.label() for coordinate in self.coordinates)
            ),
            self.product_id,
        )


class SurveyVerticalSliceManifest(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    release: str = eqx.field(static=True)
    capabilities: tuple[str, ...] = eqx.field(static=True)
    negative_boundaries: tuple[str, ...] = eqx.field(static=True)
    artifact: ScientificArtifactEnvelope
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        release: str,
        capabilities: tuple[str, ...],
        negative_boundaries: tuple[str, ...],
        artifact: ScientificArtifactEnvelope,
        /,
    ):
        name_ = str(name).strip()
        release_ = str(release).strip()
        capability = tuple(str(value).strip() for value in capabilities)
        negative = tuple(str(value).strip() for value in negative_boundaries)
        if (
            not name_
            or not release_
            or not capability
            or not negative
            or any(not value for value in (*capability, *negative))
        ):
            raise ValueError("Survey vertical-slice manifest is incomplete.")
        self.name = name_
        self.release = release_
        self.capabilities = capability
        self.negative_boundaries = negative
        self.artifact = artifact
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "survey-vertical-slice",
                "name": name_,
                "release": release_,
                "capabilities": list(capability),
                "negative_boundaries": list(negative),
                "artifact": artifact.artifact_id,
            }
        )


class SurveyFrameworkPlan(StrictModule, NonTrainableState):
    source_layout: CoordinateLayout
    observation: LinearObservationPlan
    covariance: PrecisionCovarianceAction
    likelihood: CorrelatedGaussianPlan
    manifest: SurveyVerticalSliceManifest
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_layout: CoordinateLayout,
        observed_layout: CoordinateLayout,
        response_matrix: ArrayLike,
        data: ArrayLike,
        precision: ArrayLike,
        logdet_covariance: ArrayLike,
        manifest: SurveyVerticalSliceManifest,
        /,
    ):
        observation = LinearObservationPlan(
            response_matrix, source_layout, observed_layout
        )
        covariance = PrecisionCovarianceAction(
            precision, logdet_covariance, observed_layout
        )
        likelihood = CorrelatedGaussianPlan(data, observation, covariance)
        self.source_layout = source_layout
        self.observation = observation
        self.covariance = covariance
        self.likelihood = likelihood
        self.manifest = manifest
        self.plan_id = canonical_fingerprint(
            {
                "kind": "survey-framework-plan",
                "observation": observation.plan_id,
                "covariance": covariance.action_id,
                "manifest": manifest.manifest_id,
            }
        )

    def evaluate(self, theory: SurveyTheoryProduct, /) -> CorrelatedGaussianResult:
        vector = theory.as_theory_vector()
        if vector.layout.layout_id != self.source_layout.layout_id:
            raise ValueError("Survey theory and framework source layouts disagree.")
        return self.likelihood.evaluate(vector)


def desi_full_shape_slice(
    artifact: ScientificArtifactEnvelope,
) -> SurveyVerticalSliceManifest:
    return SurveyVerticalSliceManifest(
        "DESI-DR1-LRG-full-shape",
        "DESI-DR1-v1.5",
        ("P0-P2-P4", "window", "fixed-covariance", "Gaussian-likelihood"),
        ("catalog-estimator", "native-EFT", "cross-survey-covariance"),
        artifact,
    )


def spin2_pseudocl_slice(
    artifact: ScientificArtifactEnvelope,
) -> SurveyVerticalSliceManifest:
    return SurveyVerticalSliceManifest(
        "public-spin2-pseudo-Cl",
        "adapter-selected-release",
        ("shear-shear", "galaxy-shear", "workspace-response", "fixed-covariance"),
        ("map-estimation", "workspace-generation", "generic-mask-builder"),
        artifact,
    )


def joint_survey_slice(
    artifact: ScientificArtifactEnvelope,
) -> SurveyVerticalSliceManifest:
    return SurveyVerticalSliceManifest(
        "joint-full-shape-angular",
        "composed-qualified-releases",
        ("direct-sum-layout", "shared-cosmology", "release-nuisance-namespaces"),
        ("implicit-cross-covariance", "automatic-blinding", "generic-release-discovery"),
        artifact,
    )


__all__ = [
    "SurveyCoordinate",
    "SurveyFrameworkPlan",
    "SurveyTheoryProduct",
    "SurveyVerticalSliceManifest",
    "desi_full_shape_slice",
    "joint_survey_slice",
    "spin2_pseudocl_slice",
]
