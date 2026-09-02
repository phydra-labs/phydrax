#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._context import (
    AstrodynamicsContext,
    AstrodynamicsScaleContract,
    FrameDefinition,
    JulianDate,
    ReferenceEpoch,
    TimeInstant,
)
from ._sgp4 import initialize_sgp4, propagate_sgp4, SGP4Coefficients
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


def _record_content_id(record: TleRecord, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "tle-record",
            "line1": record.line1,
            "line2": record.line2,
            "satellite_number": record.satellite_number,
            "classification": record.classification,
            "international_designator": record.international_designator,
            "epoch": record.epoch.instant_id,
            "mean_motion_derivative": record.mean_motion_derivative,
            "mean_motion_second_derivative": record.mean_motion_second_derivative,
            "bstar": record.bstar,
            "inclination": record.inclination,
            "raan": record.raan,
            "eccentricity": record.eccentricity,
            "argument_of_perigee": record.argument_of_perigee,
            "mean_anomaly": record.mean_anomaly,
            "mean_motion_revolutions_per_day": (record.mean_motion_revolutions_per_day),
            "revolution_number": record.revolution_number,
        }
    )


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


TLEPropagationRegime: TypeAlias = Literal["near-earth", "deep-space"]
TLEDeepSpaceResonance: TypeAlias = Literal["none", "synchronous", "twelve-hour"]


class TLEPropagationEpoch(StrictModule):
    """UTC-like SGP4 epoch represented as an offset from the source TLE epoch."""

    source: TimeInstant
    offset_seconds: Array
    epoch_model_id: str = eqx.field(static=True)

    def __init__(self, source: TimeInstant, offset_seconds: ArrayLike, /):
        if not isinstance(source, TimeInstant):
            raise TypeError("source must be a TimeInstant.")
        if source.scale != "UTC":
            raise ValueError("A TLE propagation epoch must use the UTC scale.")
        offset = jnp.asarray(offset_seconds).reshape(())
        if not jnp.issubdtype(offset.dtype, jnp.inexact):
            offset = offset.astype(float)
        self.source = source
        self.offset_seconds = offset
        self.epoch_model_id = canonical_fingerprint(
            {
                "kind": "tle-propagation-epoch",
                "source": source.instant_id,
                "offset_model": "uniform-sgp4-seconds",
            }
        )

    @property
    def scale(self) -> str:
        return self.source.scale

    @property
    def julian_date_high(self) -> float:
        return self.source.julian_date.high

    @property
    def julian_date_low(self) -> Array:
        return jnp.asarray(self.source.julian_date.low) + self.offset_seconds / 86400.0

    @property
    def julian_date(self) -> Array:
        return jnp.asarray(self.julian_date_high) + self.julian_date_low


class TLEPropagationResult(StrictModule):
    state: CartesianOrbitState
    minutes_since_epoch: Array
    epoch: TLEPropagationEpoch
    deep_space: Array
    resonance_steps: Array
    radius_check: Array
    decayed: Array
    range_valid: Array
    precision_reference: Array
    valid: Array
    status: Array
    residual_indicator: Array
    regime: TLEPropagationRegime = eqx.field(static=True)
    resonance_kind: TLEDeepSpaceResonance = eqx.field(static=True)
    frame: str = eqx.field(static=True)
    position_unit: str = eqx.field(static=True)
    velocity_unit: str = eqx.field(static=True)
    constant_set: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class TLEPropagationPlan(StrictModule, NonTrainableState):
    """Bounded SGP4/SDP4 emitting native Earth-TEME kilometre/second states."""

    record: TleRecord = eqx.field(static=True)
    context: AstrodynamicsContext
    mu: float = eqx.field(static=True)
    equatorial_radius: float = eqx.field(static=True)
    j2: float = eqx.field(static=True)
    j3: float = eqx.field(static=True)
    j4: float = eqx.field(static=True)
    coefficients: SGP4Coefficients = eqx.field(static=True)
    maximum_minutes: float = eqx.field(static=True)
    resonance_step_minutes: float = eqx.field(static=True)
    resonance_capacity: int = eqx.field(static=True)
    regime: TLEPropagationRegime = eqx.field(static=True)
    resonance_kind: TLEDeepSpaceResonance = eqx.field(static=True)
    constant_set: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @staticmethod
    def native_context(record: TleRecord, /) -> AstrodynamicsContext:
        """Construct the unconverted kilometre/second Earth-TEME output contract."""

        if not isinstance(record, TleRecord):
            raise TypeError("record must be a TleRecord.")
        if record.epoch.scale != "UTC":
            raise ValueError("A TLE record epoch must use the UTC scale.")
        return AstrodynamicsContext(
            AstrodynamicsScaleContract(
                "km",
                "kg",
                "s",
                length_coordinate_kind="physical",
                length_to_reference=1000.0,
            ),
            ReferenceEpoch(record.epoch, continuous=False),
            FrameDefinition("earth", "TEME", pseudo_inertial=True),
        )

    def __init__(
        self,
        record: TleRecord,
        context: AstrodynamicsContext | None = None,
        /,
        *,
        maximum_minutes: float = 43_200.0,
        resonance_step_minutes: float = 720.0,
        mu: ArrayLike = 398600.8,
        equatorial_radius: ArrayLike = 6378.135,
        j2: ArrayLike = 1.082616e-3,
        j3: ArrayLike = -2.53881e-6,
        j4: ArrayLike = -1.65597e-6,
    ):
        if not isinstance(record, TleRecord):
            raise TypeError("record must be a TleRecord.")
        native_context = self.native_context(record)
        if context is not None:
            if not isinstance(context, AstrodynamicsContext):
                raise TypeError("context must be an AstrodynamicsContext.")
            native_context.require_compatible(context)
        window = float(maximum_minutes)
        step = float(resonance_step_minutes)
        constants = tuple(float(value) for value in (mu, equatorial_radius, j2, j3, j4))
        record_values = (
            record.mean_motion_derivative,
            record.mean_motion_second_derivative,
            record.bstar,
            record.inclination,
            record.raan,
            record.eccentricity,
            record.argument_of_perigee,
            record.mean_anomaly,
            record.mean_motion_revolutions_per_day,
        )
        if (
            not np.isfinite(window)
            or window <= 0.0
            or not np.isfinite(step)
            or step <= 0.0
            or step > 720.0
            or any(not np.isfinite(value) for value in constants)
            or any(not np.isfinite(value) for value in record_values)
            or constants[0] <= 0.0
            or constants[1] <= 0.0
            or constants[2] == 0.0
            or record.epoch.scale != "UTC"
            or not 0.0 <= record.inclination <= np.pi
            or not 0.0 <= record.eccentricity < 1.0
            or record.mean_motion_revolutions_per_day <= 0.0
        ):
            raise ValueError("TLE propagation window or physical constants are invalid.")
        coefficients = initialize_sgp4(
            satellite_number=record.satellite_number,
            epoch_julian_day=(
                record.epoch.julian_date.high + record.epoch.julian_date.low
            ),
            mean_motion_derivative=record.mean_motion_derivative,
            mean_motion_second_derivative=record.mean_motion_second_derivative,
            bstar=record.bstar,
            eccentricity=record.eccentricity,
            argument_of_perigee=record.argument_of_perigee,
            inclination=record.inclination,
            mean_anomaly=record.mean_anomaly,
            mean_motion_revolutions_per_day=(record.mean_motion_revolutions_per_day),
            raan=record.raan,
            mu=constants[0],
            equatorial_radius=constants[1],
            j2=constants[2],
            j3=constants[3],
            j4=constants[4],
        )
        regime: TLEPropagationRegime = (
            "deep-space" if coefficients.method == "d" else "near-earth"
        )
        resonance: TLEDeepSpaceResonance = (
            "synchronous"
            if coefficients.irez == 1
            else "twelve-hour"
            if coefficients.irez == 2
            else "none"
        )
        self.record = record
        self.context = native_context
        self.mu = constants[0]
        self.equatorial_radius = constants[1]
        self.j2 = constants[2]
        self.j3 = constants[3]
        self.j4 = constants[4]
        self.coefficients = coefficients
        self.maximum_minutes = window
        self.resonance_step_minutes = step
        self.resonance_capacity = int(np.ceil(window / step))
        self.regime = regime
        self.resonance_kind = resonance
        self.constant_set = (
            "WGS-72"
            if constants == (398600.8, 6378.135, 1.082616e-3, -2.53881e-6, -1.65597e-6)
            else "custom"
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-tle-sgp4-sdp4",
                "record": _record_content_id(record),
                "context": native_context.context_id,
                "regime": regime,
                "resonance": resonance,
                "maximum_minutes": window,
                "resonance_step_minutes": step,
                "constants": {
                    "mu": constants[0],
                    "equatorial_radius": constants[1],
                    "j2": constants[2],
                    "j3": constants[3],
                    "j4": constants[4],
                },
                "constant_set": self.constant_set,
                "output_contract": {
                    "origin": native_context.frame.origin_id,
                    "orientation": native_context.frame.orientation_id,
                    "position_unit": native_context.scale.length_unit,
                    "velocity_unit": native_context.scale.velocity_unit,
                    "reference_epoch": native_context.epoch.epoch_id,
                    "propagation_epoch_model": "uniform-sgp4-seconds",
                },
            }
        )

    def propagate(self, minutes_since_epoch: ArrayLike, /) -> TLEPropagationResult:
        minutes = jnp.asarray(minutes_since_epoch).reshape(())
        if not jnp.issubdtype(minutes.dtype, jnp.inexact):
            minutes = minutes.astype(float)
        finite_time = jnp.isfinite(minutes)
        within_capacity = jnp.abs(minutes) <= self.maximum_minutes
        range_valid = finite_time & within_capacity
        safe_minutes = jnp.where(range_valid, minutes, 0.0)
        (
            position,
            velocity,
            model_valid,
            model_error,
            residual,
            radius_check,
            resonance_steps,
        ) = propagate_sgp4(
            self.coefficients,
            safe_minutes,
            resonance_step_minutes=self.resonance_step_minutes,
            resonance_capacity=self.resonance_capacity,
        )
        state = CartesianOrbitState(position, velocity, self.context)
        decayed = jnp.isfinite(radius_check) & (radius_check < 0.0)
        valid = range_valid & model_valid
        status = jnp.where(
            ~finite_time,
            int(AstrodynamicsStatus.NONFINITE_INPUT),
            jnp.where(
                ~within_capacity,
                int(AstrodynamicsStatus.CAPACITY_EXCEEDED),
                jnp.where(
                    model_error != 0,
                    int(AstrodynamicsStatus.INVALID_DOMAIN),
                    jnp.where(
                        valid,
                        int(AstrodynamicsStatus.SUCCESS),
                        int(AstrodynamicsStatus.NONCONVERGED),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return TLEPropagationResult(
            state=state,
            minutes_since_epoch=minutes,
            epoch=TLEPropagationEpoch(self.record.epoch, minutes * 60.0),
            deep_space=jnp.asarray(self.regime == "deep-space"),
            resonance_steps=resonance_steps,
            radius_check=radius_check,
            decayed=decayed,
            range_valid=range_valid,
            precision_reference=jnp.asarray(minutes.dtype == jnp.dtype(jnp.float64)),
            valid=valid,
            status=status,
            residual_indicator=residual,
            regime=self.regime,
            resonance_kind=self.resonance_kind,
            frame=self.context.frame.orientation_id,
            position_unit=self.context.scale.length_unit,
            velocity_unit=self.context.scale.velocity_unit,
            constant_set=self.constant_set,
            plan_id=self.plan_id,
        )


__all__ = [
    "TLEPropagationEpoch",
    "TLEDeepSpaceResonance",
    "TLEPropagationPlan",
    "TLEPropagationRegime",
    "TLEPropagationResult",
    "TleRecord",
    "parse_tle",
]
