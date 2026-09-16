#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...measurement import OperationalInterval
from ._core import DigitBank


class CalibrationAuthority(StrEnum):
    CANDIDATE = "candidate"
    EXTERNAL_OFFICIAL = "external-official"


class DetectorCalibrationPayload(StrictModule, NonTrainableState):
    channel_ids: Array
    gains: Array
    offsets: Array
    covariance: Array
    interval: OperationalInterval
    conditions_snapshot_id: str = eqx.field(static=True)
    authority: CalibrationAuthority = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)

    def __init__(
        self,
        channel_ids: ArrayLike,
        gains: ArrayLike,
        offsets: ArrayLike,
        covariance: ArrayLike,
        interval: OperationalInterval,
        /,
        *,
        conditions_snapshot_id: str,
        authority: CalibrationAuthority,
        source_id: str,
    ):
        channels = np.asarray(channel_ids)
        gains_ = np.asarray(gains, dtype=float)
        offsets_ = np.asarray(offsets, dtype=float)
        covariance_ = np.asarray(covariance, dtype=float)
        if (
            channels.ndim != 1
            or channels.size < 1
            or not np.issubdtype(channels.dtype, np.integer)
        ):
            raise ValueError("channel_ids must be a non-empty integer vector.")
        count = int(channels.size)
        if (
            gains_.shape != (count,)
            or offsets_.shape != (count,)
            or covariance_.shape != (2 * count, 2 * count)
        ):
            raise ValueError("Calibration arrays do not align with channel support.")
        if (
            len(set(channels.tolist())) != count
            or np.any(~np.isfinite(gains_))
            or np.any(gains_ <= 0.0)
            or np.any(~np.isfinite(offsets_))
            or np.any(~np.isfinite(covariance_))
            or not np.allclose(covariance_, covariance_.T)
        ):
            raise ValueError(
                "Calibration channels, gains, offsets, or covariance are invalid."
            )
        if not isinstance(interval, OperationalInterval) or not isinstance(
            authority, CalibrationAuthority
        ):
            raise TypeError(
                "Calibration interval and authority must be explicit typed values."
            )
        conditions = str(conditions_snapshot_id).strip()
        source = str(source_id).strip()
        if not conditions or not source:
            raise ValueError("Calibration conditions snapshot and source are required.")
        self.channel_ids = jnp.asarray(channels, dtype=jnp.int32)
        self.gains = jnp.asarray(gains_)
        self.offsets = jnp.asarray(offsets_)
        self.covariance = jnp.asarray(covariance_)
        self.interval = interval
        self.conditions_snapshot_id = conditions
        self.authority = authority
        self.source_id = source
        self.payload_id = canonical_fingerprint(
            {
                "kind": "detector-calibration-payload",
                "arrays": array_tree_fingerprint(
                    (channels, gains_, offsets_, covariance_)
                ),
                "interval": interval.interval_id,
                "conditions": conditions,
                "authority": authority.value,
                "source": source,
            }
        )


class CalibratedDigitResult(StrictModule, NonTrainableState):
    digits: DigitBank
    calibration_valid: Array
    derivative_valid: Array
    payload_id: str = eqx.field(static=True)


def apply_detector_calibration(
    payload: DetectorCalibrationPayload,
    digits: DigitBank,
    /,
) -> CalibratedDigitResult:
    if not isinstance(payload, DetectorCalibrationPayload) or not isinstance(
        digits, DigitBank
    ):
        raise TypeError("payload and digits must use detector calibration types.")
    matches = digits.channel_ids[..., None] == payload.channel_ids
    match_count = jnp.sum(matches, axis=-1)
    index = jnp.argmax(matches, axis=-1)
    calibration_valid = (~digits.active) | (match_count == 1)
    gains = payload.gains[index]
    offsets = payload.offsets[index]
    signals = jnp.where(
        digits.active & calibration_valid, gains * digits.signals + offsets, 0.0
    )
    calibrated = DigitBank(
        event_ids=digits.event_ids,
        digit_ids=digits.digit_ids,
        channel_ids=digits.channel_ids,
        signals=signals,
        times=digits.times,
        active=digits.active & calibration_valid,
        saturated=digits.saturated,
        conditions_id=digits.conditions_id,
    )
    derivative_valid = calibrated.active & ~calibrated.saturated & calibration_valid
    return CalibratedDigitResult(
        calibrated, calibration_valid, derivative_valid, payload.payload_id
    )


__all__ = [
    "CalibratedDigitResult",
    "CalibrationAuthority",
    "DetectorCalibrationPayload",
    "apply_detector_calibration",
]
