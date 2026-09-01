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
from ._observation_status import AstrophysicsObservationStatus
from ._operators import ImageResponsePlan
from ._photometry import ObservationDataProvenance


class ImagingCalibration(StrictModule, NonTrainableState):
    bias: Array
    dark_rate: Array
    flat: Array
    gain: Array
    bad_pixel_mask: Array
    saturation: Array
    provenance: ObservationDataProvenance
    calibration_id: str = eqx.field(static=True)

    def __init__(
        self, bias, dark_rate, flat, gain, bad_pixel_mask, saturation, provenance, /
    ):
        arrays = tuple(
            jnp.asarray(value)
            for value in (bias, dark_rate, flat, gain, bad_pixel_mask, saturation)
        )
        shape = arrays[0].shape
        if any(value.shape != shape for value in arrays[1:]):
            raise ValueError("Imaging calibration arrays must share one shape.")
        self.bias, self.dark_rate, self.flat, self.gain, mask, self.saturation = arrays
        self.bad_pixel_mask = mask.astype(bool)
        self.provenance = provenance
        self.calibration_id = canonical_fingerprint(
            {
                "kind": "imaging-calibration",
                "shape": list(shape),
                "provenance": provenance.provenance_id,
            }
        )


class CalibratedImageResult(StrictModule):
    expected_electrons: Array
    expected_adu: Array
    saturated: Array
    valid_mask: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class CalibratedImagingPlan(StrictModule, NonTrainableState):
    response: ImageResponsePlan
    calibration: ImagingCalibration
    plan_id: str = eqx.field(static=True)

    def __init__(self, response, calibration, /):
        self.response = response
        self.calibration = calibration
        self.plan_id = canonical_fingerprint(
            {
                "kind": "calibrated-imaging-plan",
                "response": response.plan_id,
                "calibration": calibration.calibration_id,
            }
        )

    def evaluate(
        self, incident_electrons_per_second: ArrayLike, exposure_seconds: ArrayLike, /
    ) -> CalibratedImageResult:
        exposure = jnp.asarray(exposure_seconds).reshape(())
        image = self.response.evaluate(incident_electrons_per_second)
        electrons = (image.image + self.calibration.dark_rate) * exposure
        adu = (
            electrons / self.calibration.gain * self.calibration.flat
            + self.calibration.bias
        )
        saturated = adu >= self.calibration.saturation
        valid_mask = ~self.calibration.bad_pixel_mask & ~saturated & jnp.isfinite(adu)
        valid = image.valid & (exposure >= 0.0) & jnp.all(jnp.isfinite(electrons))
        status = jnp.where(
            valid,
            int(AstrophysicsObservationStatus.SUCCESS),
            int(AstrophysicsObservationStatus.NONPHYSICAL_MODEL),
        ).astype(jnp.int32)
        return CalibratedImageResult(
            electrons, adu, saturated, valid_mask, valid, status, self.plan_id
        )


__all__ = ["CalibratedImageResult", "CalibratedImagingPlan", "ImagingCalibration"]
