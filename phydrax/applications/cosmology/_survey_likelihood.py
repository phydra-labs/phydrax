#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
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


class SurveyReleaseManifest(StrictModule, NonTrainableState):
    release: str = eqx.field(static=True)
    tracer: str = eqx.field(static=True)
    redshift_bin: str = eqx.field(static=True)
    statistic: str = eqx.field(static=True)
    fiducial_id: str = eqx.field(static=True)
    scale_cut_id: str = eqx.field(static=True)
    covariance_corrections: str = eqx.field(static=True)
    artifact: ScientificArtifactEnvelope
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        release: str,
        tracer: str,
        redshift_bin: str,
        statistic: str,
        fiducial_id: str,
        scale_cut_id: str,
        covariance_corrections: str,
        artifact: ScientificArtifactEnvelope,
    ):
        values = tuple(
            str(value).strip()
            for value in (
                release,
                tracer,
                redshift_bin,
                statistic,
                fiducial_id,
                scale_cut_id,
                covariance_corrections,
            )
        )
        if any(not value for value in values):
            raise ValueError("Survey release-manifest fields must be non-empty.")
        (
            self.release,
            self.tracer,
            self.redshift_bin,
            self.statistic,
            self.fiducial_id,
            self.scale_cut_id,
            self.covariance_corrections,
        ) = values
        self.artifact = artifact
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "survey-release-manifest",
                "values": list(values),
                "artifact": artifact.artifact_id,
            }
        )


class SurveyReleaseProduct(StrictModule, NonTrainableState):
    source_layout: CoordinateLayout
    observed_layout: CoordinateLayout
    data: Array
    window: Array
    precision: Array
    logdet_covariance: Array
    manifest: SurveyReleaseManifest
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_layout: CoordinateLayout,
        observed_layout: CoordinateLayout,
        data: ArrayLike,
        window: ArrayLike,
        precision: ArrayLike,
        logdet_covariance: ArrayLike,
        manifest: SurveyReleaseManifest,
        /,
    ):
        data_ = jax.lax.stop_gradient(jnp.asarray(data))
        window_ = jax.lax.stop_gradient(jnp.asarray(window, dtype=data_.dtype))
        precision_ = jax.lax.stop_gradient(jnp.asarray(precision, dtype=data_.dtype))
        logdet_ = jax.lax.stop_gradient(jnp.asarray(logdet_covariance, dtype=data_.dtype))
        if (
            data_.shape != (observed_layout.size,)
            or window_.shape != (observed_layout.size, source_layout.size)
            or precision_.shape != (observed_layout.size, observed_layout.size)
            or logdet_.shape != ()
        ):
            raise ValueError("Survey release arrays do not match coordinate layouts.")
        stacked_finite = (
            jnp.all(jnp.isfinite(data_))
            & jnp.all(jnp.isfinite(window_))
            & jnp.all(jnp.isfinite(precision_))
            & jnp.isfinite(logdet_)
        )
        data_ = eqx.error_if(
            data_,
            ~stacked_finite,
            "Survey release arrays must be finite.",
        )
        self.source_layout = source_layout
        self.observed_layout = observed_layout
        self.data = data_
        self.window = window_
        self.precision = precision_
        self.logdet_covariance = logdet_
        self.manifest = manifest
        self.product_id = canonical_fingerprint(
            {
                "kind": "survey-release-product",
                "manifest": manifest.manifest_id,
                "source_layout": source_layout.layout_id,
                "observed_layout": observed_layout.layout_id,
                "arrays": array_tree_fingerprint((data_, window_, precision_, logdet_)),
            }
        )

    @classmethod
    def from_hdf5(
        cls,
        path: str,
        source_layout: CoordinateLayout,
        observed_layout: CoordinateLayout,
        manifest: SurveyReleaseManifest,
        /,
        *,
        data_dataset: str,
        window_dataset: str,
        precision_dataset: str,
        logdet_dataset: str,
    ) -> SurveyReleaseProduct:
        with h5py.File(Path(path), "r") as handle:
            data = np.asarray(handle[data_dataset])
            window = np.asarray(handle[window_dataset])
            precision = np.asarray(handle[precision_dataset])
            logdet = np.asarray(handle[logdet_dataset])
        return cls(
            source_layout,
            observed_layout,
            data,
            window,
            precision,
            logdet,
            manifest,
        )


class DesiFullShapeLikelihoodPlan(StrictModule, NonTrainableState):
    release: SurveyReleaseProduct
    gaussian: CorrelatedGaussianPlan
    plan_id: str = eqx.field(static=True)

    def __init__(self, release: SurveyReleaseProduct, /):
        if not isinstance(release, SurveyReleaseProduct):
            raise TypeError("release must be SurveyReleaseProduct.")
        if (
            release.manifest.release != "DESI-DR1-v1.5"
            or release.manifest.tracer != "LRG-GCcomb"
            or release.manifest.redshift_bin != "0.4-0.6"
            or release.manifest.statistic != "P0-P2-P4"
        ):
            raise ValueError(
                "First DESI likelihood supports only DR1 v1.5 LRG GCcomb 0.4-0.6 P0/P2/P4."
            )
        observation = LinearObservationPlan(
            release.window, release.source_layout, release.observed_layout
        )
        covariance = PrecisionCovarianceAction(
            release.precision,
            release.logdet_covariance,
            release.observed_layout,
        )
        self.release = release
        self.gaussian = CorrelatedGaussianPlan(release.data, observation, covariance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "desi-dr1-full-shape-likelihood",
                "release": release.product_id,
                "gaussian": self.gaussian.plan_id,
            }
        )

    def evaluate(self, prewindow_theory: TheoryVector, /) -> CorrelatedGaussianResult:
        return self.gaussian.evaluate(prewindow_theory)


__all__ = [
    "DesiFullShapeLikelihoodPlan",
    "SurveyReleaseManifest",
    "SurveyReleaseProduct",
]
