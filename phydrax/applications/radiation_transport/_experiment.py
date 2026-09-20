#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..astrophysics._radiative_transfer import (
    PolarizedRadiativeTransferPlan,
    RayTransferPlan,
)


class CorrelatedKDistributionPlan(StrictModule, NonTrainableState):
    quadrature_weights: Array
    band_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature_weights: ArrayLike,
        band_ids: tuple[str, ...],
        /,
    ):
        weights = np.asarray(quadrature_weights, dtype=np.float64)
        bands = tuple(str(value).strip() for value in band_ids)
        if (
            weights.ndim != 2
            or weights.shape[0] < 1
            or weights.shape[1] < 1
            or len(bands) != weights.shape[0]
            or len(set(bands)) != len(bands)
            or any(not value for value in bands)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.allclose(np.sum(weights, axis=-1), 1.0)
        ):
            raise ValueError("Correlated-k bands or quadrature weights are invalid.")
        self.quadrature_weights = jnp.asarray(weights)
        self.band_ids = bands
        self.plan_id = canonical_fingerprint(
            {
                "kind": "correlated-k-distribution",
                "weights": array_tree_fingerprint(weights),
                "bands": bands,
            }
        )

    @property
    def band_count(self) -> int:
        return len(self.band_ids)

    @property
    def ordinate_count(self) -> int:
        return self.quadrature_weights.shape[1]


class RadiativeSensorPlan(StrictModule, NonTrainableState):
    response_weights: Array
    sensor_id: str = eqx.field(static=True)

    def __init__(self, response_weights: ArrayLike, /, *, sensor_id: str):
        weights = np.asarray(response_weights, dtype=np.float64)
        identifier = str(sensor_id).strip()
        if (
            weights.ndim != 1
            or weights.size < 1
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
            or np.sum(weights) <= 0.0
            or not identifier
        ):
            raise ValueError("Radiative sensor response or identity is invalid.")
        normalized = weights / np.sum(weights)
        self.response_weights = jnp.asarray(normalized)
        self.sensor_id = canonical_fingerprint(
            {
                "kind": "radiative-spectral-sensor",
                "declared_id": identifier,
                "weights": array_tree_fingerprint(normalized),
            }
        )


class ScalarRadiativeExperimentResult(StrictModule):
    ordinate_radiance: Array
    band_radiance: Array
    measured_radiance: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ScalarRadiativeExperimentPlan(StrictModule, NonTrainableState):
    transfer: RayTransferPlan
    spectral: CorrelatedKDistributionPlan
    sensor: RadiativeSensorPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: RayTransferPlan,
        spectral: CorrelatedKDistributionPlan,
        sensor: RadiativeSensorPlan,
        /,
    ):
        if not isinstance(transfer, RayTransferPlan):
            raise TypeError("transfer must be RayTransferPlan.")
        if not isinstance(spectral, CorrelatedKDistributionPlan):
            raise TypeError("spectral must be CorrelatedKDistributionPlan.")
        if not isinstance(sensor, RadiativeSensorPlan):
            raise TypeError("sensor must be RadiativeSensorPlan.")
        if sensor.response_weights.shape != (spectral.band_count,):
            raise ValueError("Sensor response must contain one weight per spectral band.")
        self.transfer = transfer
        self.spectral = spectral
        self.sensor = sensor
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scalar-radiative-experiment",
                "transfer": transfer.plan_id,
                "spectral": spectral.plan_id,
                "sensor": sensor.sensor_id,
            }
        )

    def evaluate(
        self,
        emission: ArrayLike,
        extinction: ArrayLike,
        incident: ArrayLike = 0.0,
        /,
    ) -> ScalarRadiativeExperimentResult:
        emission_ = jnp.asarray(emission)
        extinction_ = jnp.asarray(extinction, dtype=emission_.dtype)
        expected = (
            self.spectral.band_count,
            self.spectral.ordinate_count,
            self.transfer.ray_count,
            self.transfer.sample_count,
        )
        if emission_.shape != expected or extinction_.shape != expected:
            raise ValueError(
                "Scalar experiment arrays must be band-by-k-by-ray-by-segment."
            )
        incident_ = jnp.broadcast_to(
            jnp.asarray(incident, dtype=emission_.dtype), expected[:-1]
        )
        values = []
        valid = []
        for band in range(self.spectral.band_count):
            band_values = []
            band_valid = []
            for ordinate in range(self.spectral.ordinate_count):
                result = self.transfer.evaluate(
                    emission_[band, ordinate],
                    extinction_[band, ordinate],
                    incident_[band, ordinate],
                )
                band_values.append(result.intensity)
                band_valid.append(result.valid)
            values.append(jnp.stack(tuple(band_values)))
            valid.append(jnp.stack(tuple(band_valid)))
        ordinate_radiance = jnp.stack(tuple(values))
        validity = jnp.stack(tuple(valid))
        band_radiance = jnp.sum(
            self.spectral.quadrature_weights[..., None] * ordinate_radiance,
            axis=1,
        )
        measured = jnp.sum(self.sensor.response_weights[:, None] * band_radiance, axis=0)
        finite = jnp.all(jnp.isfinite(ordinate_radiance)) & jnp.all(
            jnp.isfinite(measured)
        )
        successful = finite & jnp.all(validity)
        return ScalarRadiativeExperimentResult(
            ordinate_radiance,
            band_radiance,
            measured,
            finite,
            successful,
            self.plan_id,
        )


class PolarizedRadiativeExperimentResult(StrictModule):
    stokes_spectrum: Array
    measured_stokes: Array
    stokes_cone_margin: Array
    finite: Array
    physically_valid: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class PolarizedRadiativeExperimentPlan(StrictModule, NonTrainableState):
    transfers: tuple[PolarizedRadiativeTransferPlan, ...]
    sensor: RadiativeSensorPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        segment_lengths: ArrayLike,
        sensor: RadiativeSensorPlan,
        /,
    ):
        lengths = np.asarray(segment_lengths, dtype=np.float64)
        if (
            lengths.ndim != 2
            or lengths.shape[0] < 1
            or lengths.shape[1] < 1
            or np.any(~np.isfinite(lengths))
            or np.any(lengths < 0.0)
            or not isinstance(sensor, RadiativeSensorPlan)
            or sensor.response_weights.shape != (lengths.shape[0],)
        ):
            raise ValueError("Polarized spectral paths or sensor response are invalid.")
        transfers = tuple(
            PolarizedRadiativeTransferPlan(
                lengths[index], plan_id=f"polarized-spectrum:{index}"
            )
            for index in range(lengths.shape[0])
        )
        self.transfers = transfers
        self.sensor = sensor
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polarized-radiative-experiment",
                "transfers": [value.plan_id for value in transfers],
                "sensor": sensor.sensor_id,
            }
        )

    def evaluate(
        self,
        emission: ArrayLike,
        propagation_matrix: ArrayLike,
        incident: ArrayLike,
        /,
    ) -> PolarizedRadiativeExperimentResult:
        emission_ = jnp.asarray(emission)
        matrix = jnp.asarray(propagation_matrix, dtype=emission_.dtype)
        incident_ = jnp.asarray(incident, dtype=emission_.dtype)
        frequency_count = len(self.transfers)
        segment_count = self.transfers[0].segment_lengths.size
        if (
            emission_.shape != (frequency_count, segment_count, 4)
            or matrix.shape != (frequency_count, segment_count, 4, 4)
            or incident_.shape != (frequency_count, 4)
        ):
            raise ValueError("Polarized experiment arrays have incompatible shapes.")
        results = tuple(
            transfer.evaluate(emission_[index], matrix[index], incident_[index])
            for index, transfer in enumerate(self.transfers)
        )
        spectrum = jnp.stack(tuple(value.emergent for value in results))
        measured = jnp.sum(self.sensor.response_weights[:, None] * spectrum, axis=0)
        polarized = jnp.linalg.norm(measured[1:])
        margin = measured[0] - polarized
        finite = jnp.all(jnp.isfinite(spectrum)) & jnp.all(jnp.isfinite(measured))
        physical = (measured[0] >= 0.0) & (
            margin >= -64.0 * jnp.finfo(measured.dtype).eps
        )
        successful = (
            finite
            & physical
            & jnp.all(jnp.stack(tuple(value.valid for value in results)))
        )
        return PolarizedRadiativeExperimentResult(
            spectrum,
            measured,
            margin,
            finite,
            physical,
            successful,
            self.plan_id,
        )


__all__ = [
    "CorrelatedKDistributionPlan",
    "PolarizedRadiativeExperimentPlan",
    "PolarizedRadiativeExperimentResult",
    "RadiativeSensorPlan",
    "ScalarRadiativeExperimentPlan",
    "ScalarRadiativeExperimentResult",
]
