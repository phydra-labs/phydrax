#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....observation import CovarianceAction, LinearNuisancePlan
from ..potential_fields import RegionalTrendPlan


class DeformationPrediction(StrictModule):
    gnss_displacement_m: Array
    insar_los_displacement_m: Array
    tilt_radians: Array
    strain: Array
    finite: Array


class GeodeticDeformationObservationPlan(StrictModule, NonTrainableState):
    """Linear GNSS, InSAR LOS, tilt, and strain observation actions."""

    gnss_matrix: Array
    insar_matrix: Array
    tilt_matrix: Array
    strain_matrix: Array
    model_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gnss_matrix: ArrayLike,
        insar_matrix: ArrayLike,
        tilt_matrix: ArrayLike,
        strain_matrix: ArrayLike,
        model_size: int,
        /,
    ):
        size = int(model_size)
        matrices = tuple(
            jnp.asarray(value)
            for value in (gnss_matrix, insar_matrix, tilt_matrix, strain_matrix)
        )
        if size <= 0 or any(
            value.ndim != 2
            or value.shape[1] != size
            or bool(jnp.any(~jnp.isfinite(value)))
            for value in matrices
        ):
            raise ValueError(
                "Deformation observation matrices must be finite and model-sized."
            )
        if matrices[0].shape[0] % 3:
            raise ValueError("GNSS observation rows must contain XYZ triples.")
        self.gnss_matrix, self.insar_matrix, self.tilt_matrix, self.strain_matrix = (
            matrices
        )
        self.model_size = size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "geodetic-deformation-observation",
                "model_size": size,
                "shapes": [value.shape for value in matrices],
            }
        )

    def predict(self, displacement_parameters: ArrayLike, /) -> DeformationPrediction:
        values = jnp.asarray(displacement_parameters)
        if values.shape != (self.model_size,):
            raise ValueError("Deformation parameter vector has wrong shape.")
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Deformation parameters must be finite.",
        )
        gnss = (self.gnss_matrix @ values).reshape((-1, 3))
        insar = self.insar_matrix @ values
        tilt = self.tilt_matrix @ values
        strain = self.strain_matrix @ values
        finite = all(
            jnp.all(jnp.isfinite(value)) for value in (gnss, insar, tilt, strain)
        )
        return DeformationPrediction(gnss, insar, tilt, strain, jnp.asarray(finite))


class InSARObservationPlan(StrictModule, NonTrainableState):
    forward: GeodeticDeformationObservationPlan
    observed_los_m: Array
    covariance: CovarianceAction
    nuisance: LinearNuisancePlan | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        forward: GeodeticDeformationObservationPlan,
        observed_los_m: ArrayLike,
        covariance: CovarianceAction,
        /,
        *,
        coordinates_xy_m: ArrayLike | None = None,
        ramp_degree: int | None = None,
    ):
        if not isinstance(forward, GeodeticDeformationObservationPlan):
            raise TypeError("InSAR observation requires geodetic forward plan.")
        observed = jnp.asarray(observed_los_m)
        count = forward.insar_matrix.shape[0]
        if observed.shape != (count,) or covariance.layout.size != count:
            raise ValueError("InSAR observations/covariance must match LOS rows.")
        observed = eqx.error_if(
            observed,
            jnp.any(~jnp.isfinite(observed)),
            "InSAR observations must be finite.",
        )
        nuisance = None
        if ramp_degree is not None:
            coordinates = np.asarray(coordinates_xy_m, dtype=float)
            if coordinates.shape != (count, 2):
                raise ValueError(
                    "InSAR ramp coordinates must have shape (observations,2)."
                )
            trend = RegionalTrendPlan(
                coordinates[:, 0], coordinates[:, 1], total_degree=int(ramp_degree)
            )
            names = tuple(f"insar-ramp-{term[0]}-{term[1]}" for term in trend.terms)
            nuisance = LinearNuisancePlan(
                trend.design, covariance, covariance.layout, names
            )
        self.forward, self.observed_los_m, self.covariance = forward, observed, covariance
        self.nuisance = nuisance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "insar-observation-plan",
                "forward": forward.plan_id,
                "covariance": covariance.action_id,
                "ramp": None if nuisance is None else nuisance.plan_id,
            }
        )

    def evaluate(self, displacement_parameters: ArrayLike, /):
        prediction = self.forward.predict(
            displacement_parameters
        ).insar_los_displacement_m
        if self.nuisance is None:
            residual = self.observed_los_m - prediction
            return prediction, residual, self.covariance.quadratic(residual), None
        projected = self.nuisance.evaluate(self.observed_los_m, prediction)
        return (
            prediction + projected.nuisance_prediction,
            projected.residual,
            projected.quadratic,
            projected,
        )


__all__ = [
    "DeformationPrediction",
    "GeodeticDeformationObservationPlan",
    "InSARObservationPlan",
]
