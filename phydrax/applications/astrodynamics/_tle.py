#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._context import AstrodynamicsContext, JulianDate, TimeInstant
from ._elements import classical_to_cartesian, ClassicalOrbitalElements
from ._state import CartesianOrbitState
from ._status import AstrodynamicsStatus


def _checksum(line: str, /) -> int:
    return (
        sum(
            int(character) if character.isdigit() else 1 if character == "-" else 0
            for character in line[:68]
        )
        % 10
    )


def _implied_decimal(value: str, /) -> float:
    text = value.strip()
    if not text:
        return 0.0
    sign = -1.0 if text.startswith("-") else 1.0
    digits = text.lstrip("+-")
    mantissa = float(f"0.{digits[:-2]}")
    exponent = int(digits[-2:])
    return sign * mantissa * 10.0**exponent


def _january_first_jd(year: int, /) -> float:
    y = year - 1
    return 1721425.5 + 365 * y + y // 4 - y // 100 + y // 400


@dataclass(frozen=True)
class TleRecord:
    line1: str
    line2: str
    satellite_number: int
    classification: str
    international_designator: str
    epoch: TimeInstant
    mean_motion_derivative: float
    mean_motion_second_derivative: float
    bstar: float
    inclination: float
    raan: float
    eccentricity: float
    argument_of_perigee: float
    mean_anomaly: float
    mean_motion_revolutions_per_day: float
    revolution_number: int


def parse_tle(line1: str, line2: str, /) -> TleRecord:
    first = line1.rstrip("\n")
    second = line2.rstrip("\n")
    if len(first) < 69 or len(second) < 69 or first[0] != "1" or second[0] != "2":
        raise ValueError("TLE lines are malformed.")
    if _checksum(first) != int(first[68]) or _checksum(second) != int(second[68]):
        raise ValueError("TLE checksum failed.")
    satellite = int(first[2:7])
    if satellite != int(second[2:7]):
        raise ValueError("TLE satellite numbers disagree.")
    short_year = int(first[18:20])
    year = 1900 + short_year if short_year >= 57 else 2000 + short_year
    day = float(first[20:32])
    epoch_jd = _january_first_jd(year) + day - 1.0
    return TleRecord(
        first,
        second,
        satellite,
        first[7],
        first[9:17].strip(),
        TimeInstant(JulianDate(epoch_jd), "UTC"),
        float(first[33:43]),
        _implied_decimal(first[44:52]),
        _implied_decimal(first[53:61]),
        np.deg2rad(float(second[8:16])),
        np.deg2rad(float(second[17:25])),
        float(f"0.{second[26:33].strip()}"),
        np.deg2rad(float(second[34:42])),
        np.deg2rad(float(second[43:51])),
        float(second[52:63]),
        int(second[63:68]),
    )


class Sgp4Result(StrictModule):
    state: CartesianOrbitState
    deep_space: Array
    valid: Array
    status: Array
    residual_indicator: Array
    plan_id: str = eqx.field(static=True)


class Sgp4Plan(StrictModule, NonTrainableState):
    """Native bounded near-Earth Brouwer/SGP4 secular state propagation."""

    record: TleRecord = eqx.field(static=True)
    context: AstrodynamicsContext
    mu: Array
    equatorial_radius: Array
    j2: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        record: TleRecord,
        context: AstrodynamicsContext,
        /,
        *,
        mu: ArrayLike = 398600.8,
        equatorial_radius: ArrayLike = 6378.135,
        j2: ArrayLike = 1.082616e-3,
    ):
        if not isinstance(record, TleRecord):
            raise TypeError("record must be a TleRecord.")
        self.record = record
        self.context = context
        self.mu = jnp.asarray(mu).reshape(())
        self.equatorial_radius = jnp.asarray(equatorial_radius).reshape(())
        self.j2 = jnp.asarray(j2).reshape(())
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-sgp4-near-earth",
                "satellite": record.satellite_number,
                "context": context.context_id,
            }
        )

    def propagate(self, minutes_since_epoch: ArrayLike, /) -> Sgp4Result:
        minutes = jnp.asarray(minutes_since_epoch).reshape(())
        n0 = self.record.mean_motion_revolutions_per_day * 2.0 * jnp.pi / 86400.0
        semi_major = (self.mu / n0**2) ** (1.0 / 3.0)
        eccentricity = jnp.asarray(self.record.eccentricity)
        inclination = jnp.asarray(self.record.inclination)
        p = semi_major * (1.0 - eccentricity * eccentricity)
        factor = 1.5 * self.j2 * n0 * (self.equatorial_radius / p) ** 2
        cosine = jnp.cos(inclination)
        elapsed = minutes * 60.0
        raan = self.record.raan - factor * cosine * elapsed
        argument = (
            self.record.argument_of_perigee
            + 0.5 * factor * (5.0 * cosine * cosine - 1.0) * elapsed
        )
        mean_motion = n0 + 0.5 * factor * jnp.sqrt(1.0 - eccentricity**2) * (
            3.0 * cosine * cosine - 1.0
        )
        mean_anomaly = self.record.mean_anomaly + mean_motion * elapsed

        def newton(_, anomaly):
            residual = anomaly - eccentricity * jnp.sin(anomaly) - mean_anomaly
            derivative = 1.0 - eccentricity * jnp.cos(anomaly)
            return anomaly - residual / derivative

        eccentric_anomaly = jax.lax.fori_loop(0, 12, newton, mean_anomaly)
        true_anomaly = 2.0 * jnp.arctan2(
            jnp.sqrt(1.0 + eccentricity) * jnp.sin(0.5 * eccentric_anomaly),
            jnp.sqrt(1.0 - eccentricity) * jnp.cos(0.5 * eccentric_anomaly),
        )
        elements = ClassicalOrbitalElements(
            jnp.asarray((p, eccentricity, inclination, raan, argument, true_anomaly)),
            self.context,
        )
        state, state_valid, _ = classical_to_cartesian(elements, self.mu)
        period_minutes = 2.0 * jnp.pi / n0 / 60.0
        deep_space = period_minutes >= 225.0
        residual = jnp.abs(
            eccentric_anomaly - eccentricity * jnp.sin(eccentric_anomaly) - mean_anomaly
        )
        valid = state_valid & ~deep_space & jnp.isfinite(minutes) & (residual <= 1.0e-10)
        status = jnp.where(
            deep_space,
            int(AstrodynamicsStatus.UNSUPPORTED_REGIME),
            jnp.where(
                valid,
                int(AstrodynamicsStatus.SUCCESS),
                int(AstrodynamicsStatus.NONCONVERGED),
            ),
        ).astype(jnp.int32)
        return Sgp4Result(state, deep_space, valid, status, residual, self.plan_id)


__all__ = ["Sgp4Plan", "Sgp4Result", "TleRecord", "parse_tle"]
