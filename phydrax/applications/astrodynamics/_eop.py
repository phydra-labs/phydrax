#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._data import AstrodynamicsDataProvenance
from ._status import AstrodynamicsStatus


_EARTH_ANGULAR_RATE = 7.292115146706979e-5


def _r1(angle: Array, /) -> Array:
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.asarray(((1.0, 0.0, 0.0), (0.0, c, s), (0.0, -s, c)))


def _r2(angle: Array, /) -> Array:
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.asarray(((c, 0.0, -s), (0.0, 1.0, 0.0), (s, 0.0, c)))


def _r3(angle: Array, /) -> Array:
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.asarray(((c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)))


class EarthOrientationEvaluation(StrictModule):
    rotation_gcrs_to_itrs: Array
    rotation_rate: Array
    dut1_seconds: Array
    lod_seconds: Array
    predicted: Array
    valid: Array
    status: Array
    product_id: str = eqx.field(static=True)


class EarthOrientationRecordSet(StrictModule, NonTrainableState):
    """Prepared IERS Earth-orientation and CIO coordinates on one UTC grid."""

    relative_utc_seconds: Array
    xp_radians: Array
    yp_radians: Array
    dut1_seconds: Array
    lod_seconds: Array
    cip_x_radians: Array
    cip_y_radians: Array
    cio_s_radians: Array
    predicted: Array
    provenance: AstrodynamicsDataProvenance
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        relative_utc_seconds: ArrayLike,
        xp_radians: ArrayLike,
        yp_radians: ArrayLike,
        dut1_seconds: ArrayLike,
        lod_seconds: ArrayLike,
        cip_x_radians: ArrayLike,
        cip_y_radians: ArrayLike,
        cio_s_radians: ArrayLike,
        predicted: ArrayLike,
        provenance: AstrodynamicsDataProvenance,
        /,
    ):
        values = tuple(
            np.asarray(value, dtype=float)
            for value in (
                relative_utc_seconds,
                xp_radians,
                yp_radians,
                dut1_seconds,
                lod_seconds,
                cip_x_radians,
                cip_y_radians,
                cio_s_radians,
            )
        )
        times = values[0]
        if (
            times.ndim != 1
            or times.size < 2
            or any(value.shape != times.shape for value in values[1:])
            or any(np.any(~np.isfinite(value)) for value in values)
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("Earth-orientation arrays must be finite matching vectors.")
        predicted_host = np.asarray(predicted, dtype=bool)
        if predicted_host.shape != times.shape:
            raise ValueError("EOP prediction mask must match time nodes.")
        (
            self.relative_utc_seconds,
            self.xp_radians,
            self.yp_radians,
            self.dut1_seconds,
            self.lod_seconds,
            self.cip_x_radians,
            self.cip_y_radians,
            self.cio_s_radians,
        ) = tuple(jnp.asarray(value) for value in values)
        self.predicted = jnp.asarray(predicted_host)
        self.provenance = provenance
        self.product_id = canonical_fingerprint(
            {
                "kind": "earth-orientation-record-set",
                "nodes": times.tolist(),
                "provenance": provenance.provenance_id,
            }
        )

    def interpolate(self, relative_utc_seconds: ArrayLike, /) -> tuple[Array, ...]:
        query = jnp.asarray(relative_utc_seconds).reshape(())
        support = (query >= self.relative_utc_seconds[0]) & (
            query <= self.relative_utc_seconds[-1]
        )
        values = tuple(
            jnp.interp(query, self.relative_utc_seconds, array)
            for array in (
                self.xp_radians,
                self.yp_radians,
                self.dut1_seconds,
                self.lod_seconds,
                self.cip_x_radians,
                self.cip_y_radians,
                self.cio_s_radians,
            )
        )
        index = jnp.clip(
            jnp.searchsorted(self.relative_utc_seconds, query, side="right") - 1,
            0,
            int(self.relative_utc_seconds.size) - 1,
        )
        return (*values, self.predicted[index], support & jnp.isfinite(query))


class PreparedEarthOrientation(StrictModule, NonTrainableState):
    records: EarthOrientationRecordSet
    reference_jd_utc: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, records: EarthOrientationRecordSet, reference_jd_utc: float, /):
        if not isinstance(records, EarthOrientationRecordSet):
            raise TypeError("records must be an EarthOrientationRecordSet.")
        reference = float(reference_jd_utc)
        if not np.isfinite(reference):
            raise ValueError("reference_jd_utc must be finite.")
        self.records = records
        self.reference_jd_utc = reference
        self.plan_id = canonical_fingerprint(
            {
                "kind": "prepared-earth-orientation",
                "records": records.product_id,
                "reference_jd_utc": reference,
            }
        )

    def evaluate(self, relative_utc_seconds: ArrayLike, /) -> EarthOrientationEvaluation:
        query = jnp.asarray(relative_utc_seconds).reshape(())
        xp, yp, dut1, lod, x, y, s, predicted, support = self.records.interpolate(query)
        z = jnp.sqrt(jnp.maximum(1.0 - x * x - y * y, 0.0))
        a = 1.0 / (1.0 + z)
        celestial_to_intermediate = jnp.asarray(
            (
                (1.0 - a * x * x, -a * x * y, x),
                (-a * x * y, 1.0 - a * y * y, y),
                (-x, -y, 1.0 - a * (x * x + y * y)),
            )
        )
        celestial_to_intermediate = _r3(s) @ celestial_to_intermediate
        jd_ut1 = self.reference_jd_utc + (query + dut1) / 86400.0
        era = (
            2.0
            * jnp.pi
            * jnp.mod(
                0.7790572732640 + 1.00273781191135448 * (jd_ut1 - 2451545.0),
                1.0,
            )
        )
        polar_motion = _r2(-xp) @ _r1(-yp)
        rotation = polar_motion @ _r3(era) @ celestial_to_intermediate
        omega = _EARTH_ANGULAR_RATE * (1.0 - lod / 86400.0)
        generator = jnp.asarray(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        rotation_rate = (
            polar_motion @ (omega * generator @ _r3(era)) @ celestial_to_intermediate
        )
        finite = jnp.all(jnp.isfinite(rotation)) & jnp.all(jnp.isfinite(rotation_rate))
        valid = support & finite
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            jnp.where(
                support,
                int(AstrodynamicsStatus.NONFINITE_INPUT),
                int(AstrodynamicsStatus.INVALID_DOMAIN),
            ),
        ).astype(jnp.int32)
        return EarthOrientationEvaluation(
            rotation,
            rotation_rate,
            dut1,
            lod,
            predicted,
            valid,
            status,
            self.plan_id,
        )


__all__ = [
    "EarthOrientationEvaluation",
    "EarthOrientationRecordSet",
    "PreparedEarthOrientation",
]
