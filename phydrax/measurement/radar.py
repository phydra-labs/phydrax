#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Weather-radar polar volumes and FMCW signal transforms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module, util
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import open_reference_artifact, ReferenceArtifactManifest
from ..units import (
    ANGLE,
    AREA,
    conversion_factor,
    DEGREE,
    derived_unit,
    LENGTH,
    METER,
    ONE,
    RADIAN,
    TIME,
    UnitDefinition,
)
from ._asset import (
    AcquisitionIdentity,
    DataOrigin,
    DataStage,
    DerivationRecord,
    MeasurementAsset,
)
from ._collection import MeasurementCollection, MeasurementRole, MeasurementRoleAssignment
from ._field import QuantityField, SamplingSemantics, SpatialSamplingKind
from ._quantity import QuantitySpec, ValueLayout


@dataclass(frozen=True, slots=True)
class PolarVolumeSupport:
    azimuth: np.ndarray
    elevation: np.ndarray
    range_gates: np.ndarray
    frame_id: str
    angle_unit: UnitDefinition
    range_unit: UnitDefinition
    sample_shape: tuple[int, int] = field(init=False)
    support_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.angle_unit, UnitDefinition)
            or self.angle_unit.dimension != ANGLE
            or not isinstance(self.range_unit, UnitDefinition)
            or self.range_unit.dimension != LENGTH
        ):
            raise ValueError(
                "Polar support requires explicit angle and length UnitDefinition values."
            )
        if (
            not isinstance(self.frame_id, str)
            or not self.frame_id
            or self.frame_id != self.frame_id.strip()
        ):
            raise ValueError("frame_id must be canonical nonempty text.")
        azimuth = np.array(self.azimuth, dtype=np.float64, copy=True)
        elevation = np.array(self.elevation, dtype=np.float64, copy=True)
        ranges = np.array(self.range_gates, dtype=np.float64, copy=True)
        if (
            azimuth.ndim != 1
            or azimuth.size < 1
            or ranges.size < 1
            or elevation.shape != azimuth.shape
            or ranges.ndim != 1
            or np.any(ranges < 0.0)
            or not np.all(np.diff(ranges) > 0.0)
        ):
            raise ValueError(
                "Polar support requires ray azimuth/elevation and increasing ranges."
            )
        for value in (azimuth, elevation, ranges):
            if not np.all(np.isfinite(value)):
                raise ValueError("Polar support coordinates must be finite.")
            value.setflags(write=False)
        object.__setattr__(self, "azimuth", azimuth)
        object.__setattr__(self, "elevation", elevation)
        object.__setattr__(self, "range_gates", ranges)
        object.__setattr__(self, "sample_shape", (azimuth.size, ranges.size))
        object.__setattr__(
            self,
            "support_id",
            canonical_fingerprint(
                {
                    "kind": "radar-polar-volume-support",
                    "azimuth": array_tree_fingerprint(azimuth),
                    "elevation": array_tree_fingerprint(elevation),
                    "ranges": array_tree_fingerprint(ranges),
                    "frame": self.frame_id,
                    "angle_unit": self.angle_unit.unit_id,
                    "range_unit": self.range_unit.unit_id,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class FMCWAcquisition:
    carrier_frequency: float
    chirp_slope: float
    sample_interval: float
    chirp_interval: float
    receiver_positions: np.ndarray
    acquisition_id: str
    coordinate_contract: SpatialCoordinateContract
    time_unit: UnitDefinition
    acquisition_contract_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if (
            not isinstance(self.time_unit, UnitDefinition)
            or self.time_unit.dimension != TIME
        ):
            raise ValueError("time_unit must be a time UnitDefinition.")
        if (
            not isinstance(self.acquisition_id, str)
            or not self.acquisition_id
            or self.acquisition_id != self.acquisition_id.strip()
        ):
            raise ValueError("acquisition_id must be canonical nonempty text.")
        positions = np.array(self.receiver_positions, dtype=np.float64, copy=True)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or positions.shape[0] < 1
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("receiver_positions must have shape (receivers, 3).")
        if any(
            value <= 0.0 or not np.isfinite(value)
            for value in (
                self.carrier_frequency,
                self.chirp_slope,
                self.sample_interval,
                self.chirp_interval,
            )
        ):
            raise ValueError("FMCW timing and carrier parameters must be positive.")
        positions.setflags(write=False)
        object.__setattr__(self, "receiver_positions", positions)
        object.__setattr__(
            self,
            "acquisition_contract_id",
            canonical_fingerprint(
                {
                    "kind": "fmcw-acquisition",
                    "carrier_frequency": self.carrier_frequency,
                    "chirp_slope": self.chirp_slope,
                    "sample_interval": self.sample_interval,
                    "chirp_interval": self.chirp_interval,
                    "receiver_positions": array_tree_fingerprint(positions),
                    "coordinates": self.coordinate_contract.spatial_id,
                    "time_unit": self.time_unit.unit_id,
                    "acquisition": self.acquisition_id,
                }
            ),
        )


class FMCWTransformResult(StrictModule, NonTrainableState):
    range_doppler_channels: Array
    range_doppler_power: Array
    range_axis: Array
    doppler_axis: Array
    finite: Array
    successful: Array
    range_unit_id: str = eqx.field(static=True)
    doppler_unit_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)


class FMCWTransformPlan(StrictModule, NonTrainableState):
    acquisition: FMCWAcquisition = eqx.field(static=True)
    window_fast: Array
    window_slow: Array
    propagation_speed: float = eqx.field(static=True)

    def __init__(
        self,
        acquisition: FMCWAcquisition,
        fast_samples: int,
        chirps: int,
        /,
        *,
        propagation_speed: float,
        propagation_speed_unit: UnitDefinition,
    ):
        if not isinstance(acquisition, FMCWAcquisition):
            raise TypeError("acquisition must be FMCWAcquisition.")
        for name, value in (("fast_samples", fast_samples), ("chirps", chirps)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if not isinstance(propagation_speed_unit, UnitDefinition):
            raise TypeError("propagation_speed_unit must be UnitDefinition.")
        target_speed_unit = derived_unit(
            "fmcw-length-per-time",
            (
                (acquisition.coordinate_contract.length_unit, 1),
                (acquisition.time_unit, -1),
            ),
        )
        speed = float(propagation_speed) * float(
            conversion_factor(propagation_speed_unit, target_speed_unit)
        )
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError("propagation_speed must be finite and positive.")
        self.acquisition = acquisition
        self.window_fast = jnp.hanning(fast_samples)
        self.window_slow = jnp.hanning(chirps)
        self.propagation_speed = speed

    def evaluate(self, adc: ArrayLike, /) -> FMCWTransformResult:
        values = jnp.asarray(adc)
        expected = (
            self.window_slow.size,
            self.window_fast.size,
            self.acquisition.receiver_positions.shape[0],
        )
        if values.shape != expected or not jnp.issubdtype(
            values.dtype, jnp.complexfloating
        ):
            raise ValueError(f"adc must be complex with shape {expected}.")
        windowed = (
            values * self.window_slow[:, None, None] * self.window_fast[None, :, None]
        )
        range_spectrum = jnp.fft.fft(windowed, axis=1, norm="ortho")[
            :, : self.window_fast.size // 2
        ]
        range_doppler = jnp.fft.fftshift(
            jnp.fft.fft(range_spectrum, axis=0, norm="ortho"), axes=0
        )
        beat = jnp.fft.fftfreq(self.window_fast.size, d=self.acquisition.sample_interval)[
            : self.window_fast.size // 2
        ]
        ranges = self.propagation_speed * beat / (2.0 * self.acquisition.chirp_slope)
        doppler = jnp.fft.fftshift(
            jnp.fft.fftfreq(self.window_slow.size, d=self.acquisition.chirp_interval)
        )
        power = jnp.sum(jnp.abs(range_doppler) ** 2, axis=-1)
        finite = jnp.all(jnp.isfinite(power))
        return FMCWTransformResult(
            range_doppler,
            power,
            ranges,
            doppler,
            finite,
            finite,
            self.acquisition.coordinate_contract.length_unit.unit_id,
            derived_unit(
                "fmcw-doppler-frequency",
                ((self.acquisition.time_unit, -1),),
            ).unit_id,
            self.acquisition.coordinate_contract.reference_frame,
        )


@dataclass(frozen=True, slots=True)
class AutomotiveRadarProfile:
    frame_id: str
    range_unit: UnitDefinition
    velocity_unit: UnitDefinition
    radar_cross_section_unit: UnitDefinition

    def __post_init__(self) -> None:
        if (
            not isinstance(self.frame_id, str)
            or not self.frame_id
            or self.frame_id != self.frame_id.strip()
        ):
            raise ValueError("frame_id must be canonical nonempty text.")
        if not all(
            isinstance(unit, UnitDefinition)
            for unit in (
                self.range_unit,
                self.velocity_unit,
                self.radar_cross_section_unit,
            )
        ):
            raise TypeError("Radar profile units must be UnitDefinition values.")
        if (
            self.range_unit.dimension != LENGTH
            or self.velocity_unit.dimension != LENGTH / TIME
            or self.radar_cross_section_unit.dimension != AREA
        ):
            raise ValueError(
                "Radar profile units must have length, velocity, and area dimensions."
            )

    def lower(
        self,
        detections: ArrayLike,
        reference: ReferenceArtifactManifest,
        /,
        *,
        asset_id: str,
    ) -> MeasurementCollection:
        values = np.asarray(detections, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError(
                "detections must contain range, azimuth, radial velocity, and radar cross section."
            )
        from ._support import IndexSampleSupport

        support = IndexSampleSupport(
            (values.shape[0],), ("detection",), frame_id=self.frame_id
        )
        acquisition = AcquisitionIdentity(
            asset_id, asset_id, "automotive-radar", "detection-profile"
        )
        derivation = DerivationRecord(
            DataOrigin.EXTERNAL,
            DataStage.DERIVED,
            transformation_id=f"{asset_id}:radar-profile",
        )
        specifications = (
            ("range", self.range_unit, "physical.range"),
            ("azimuth", RADIAN, "sensor.azimuth"),
            ("radial-velocity", self.velocity_unit, "physical.radial-velocity"),
            ("radar-cross-section", self.radar_cross_section_unit, "radar.cross-section"),
        )
        assets = []
        for column, (name, unit, compatibility) in enumerate(specifications):
            field = QuantityField(
                f"{asset_id}.{name}.field",
                QuantitySpec("radar", name, name, unit, compatibility),
                ValueLayout.scalar(),
                support,
                SamplingSemantics(SpatialSamplingKind.EVENT),
                values[:, column],
            )
            assets.append(
                MeasurementAsset.from_single_reference(
                    f"{asset_id}.{name}",
                    field,
                    reference,
                    derivation,
                    acquisition=acquisition,
                )
            )
        return MeasurementCollection(
            asset_id,
            asset_id,
            tuple(assets),
            tuple(
                MeasurementRoleAssignment(value.asset_id, MeasurementRole.OBSERVATION)
                for value in assets
            ),
        )


_CF_UNIT_ALIASES = {
    "1": ONE,
    "m": METER,
    "meter": METER,
    "meters": METER,
    "rad": RADIAN,
    "radian": RADIAN,
    "radians": RADIAN,
    "deg": DEGREE,
    "degree": DEGREE,
    "degrees": DEGREE,
}


def _require_cf_unit(variable, expected: UnitDefinition, name: str, /) -> None:
    raw = variable.attrs.get("units")
    if not isinstance(raw, str):
        raise ValueError(f"CF/Radial variable {name!r} lacks a units attribute.")
    normalized = raw.strip().lower()
    alias = _CF_UNIT_ALIASES.get(normalized)
    if raw.strip() != expected.symbol and (
        alias is None or alias.unit_id != expected.unit_id
    ):
        raise ValueError(
            f"CF/Radial source unit for {name!r} disagrees with the declared unit."
        )


class CfRadialProvider:
    def read(
        self,
        path: str | Path,
        reference: ReferenceArtifactManifest,
        /,
        *,
        campaign_id: str,
        quantity_units: Mapping[str, UnitDefinition],
        angle_unit: UnitDefinition,
        range_unit: UnitDefinition,
        frame_id: str,
        maximum_gates: int = 100_000_000,
    ) -> MeasurementCollection:
        if util.find_spec("xarray") is None:
            raise ImportError(
                "CF/Radial admission requires the optional radar-cfradial extra."
            )
        if (
            not isinstance(angle_unit, UnitDefinition)
            or angle_unit.dimension != ANGLE
            or not isinstance(range_unit, UnitDefinition)
            or range_unit.dimension != LENGTH
        ):
            raise ValueError(
                "CF/Radial angle_unit and range_unit must have angle and length dimensions."
            )
        if not isinstance(frame_id, str) or not frame_id or frame_id != frame_id.strip():
            raise ValueError("frame_id must be canonical nonempty text.")
        if isinstance(maximum_gates, bool) or not isinstance(maximum_gates, Integral):
            raise TypeError("maximum_gates must be an integer.")
        if maximum_gates < 1:
            raise ValueError("maximum_gates must be positive.")
        reference.require_rights()
        source = Path(path).expanduser().absolute()
        backend = import_module("xarray")
        with (
            TemporaryDirectory(prefix="phydrax-cfradial-read-") as temporary,
            open_reference_artifact(source, reference) as resource,
        ):
            staged = Path(temporary) / source.name
            with staged.open("wb") as output:
                while chunk := resource.stream.read(1024 * 1024):
                    output.write(chunk)
            with backend.open_dataset(staged) as dataset:
                azimuth = np.asarray(dataset["azimuth"].values)
                elevation = np.asarray(dataset["elevation"].values)
                ranges = np.asarray(dataset["range"].values)
                if dataset.attrs.get("coordinate_frame") != frame_id:
                    raise ValueError(
                        "CF/Radial coordinate_frame must exactly match the declared frame_id."
                    )
                _require_cf_unit(dataset["azimuth"], angle_unit, "azimuth")
                _require_cf_unit(dataset["elevation"], angle_unit, "elevation")
                _require_cf_unit(dataset["range"], range_unit, "range")
                support = PolarVolumeSupport(
                    azimuth,
                    elevation,
                    ranges,
                    frame_id,
                    angle_unit,
                    range_unit,
                )
                if int(np.prod(support.sample_shape)) > maximum_gates:
                    raise MemoryError("CF/Radial volume exceeds maximum_gates.")
                known = (
                    "reflectivity",
                    "velocity",
                    "spectrum_width",
                    "differential_reflectivity",
                    "cross_correlation_ratio",
                    "differential_phase",
                )
                assets = []
                for name in known:
                    if name not in dataset:
                        continue
                    if name not in quantity_units:
                        raise ValueError(
                            f"Missing physical unit for CF/Radial field {name!r}."
                        )
                    unit = quantity_units[name]
                    if not isinstance(unit, UnitDefinition):
                        raise TypeError(
                            f"Physical unit for CF/Radial field {name!r} must be UnitDefinition."
                        )
                    _require_cf_unit(dataset[name], unit, name)
                    values = np.asarray(dataset[name].values).reshape(
                        support.sample_shape
                    )
                    field = QuantityField(
                        f"{campaign_id}.{name}",
                        QuantitySpec(
                            "radar",
                            name.replace("_", "-"),
                            name.replace("_", "-"),
                            unit,
                            f"radar.{name}",
                        ),
                        ValueLayout.scalar(),
                        support,
                        SamplingSemantics(SpatialSamplingKind.DETECTOR_BIN),
                        values,
                        np.isfinite(values),
                    )
                    assets.append(
                        MeasurementAsset.from_single_reference(
                            f"{campaign_id}.{name}",
                            field,
                            reference,
                            DerivationRecord(
                                DataOrigin.EXTERNAL,
                                DataStage.CALIBRATED,
                                transformation_id=f"{campaign_id}:cfradial",
                            ),
                        )
                    )
        if not assets:
            raise ValueError("CF/Radial file contains no supported fields.")
        return MeasurementCollection(
            campaign_id,
            campaign_id,
            tuple(assets),
            tuple(
                MeasurementRoleAssignment(value.asset_id, MeasurementRole.OBSERVATION)
                for value in assets
            ),
        )


__all__ = [
    "AutomotiveRadarProfile",
    "CfRadialProvider",
    "FMCWAcquisition",
    "FMCWTransformPlan",
    "FMCWTransformResult",
    "PolarVolumeSupport",
]
