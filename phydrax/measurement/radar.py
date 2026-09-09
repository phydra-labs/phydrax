#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Weather-radar polar volumes and FMCW signal transforms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import new as new_digest
from importlib import import_module, util
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import ReferenceArtifactManifest
from ..units import ONE, RADIAN, UnitDefinition
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
    sample_shape: tuple[int, int] = field(init=False)
    support_id: str = field(init=False)

    def __post_init__(self) -> None:
        azimuth = np.array(self.azimuth, dtype=float, copy=True)
        elevation = np.array(self.elevation, dtype=float, copy=True)
        ranges = np.array(self.range_gates, dtype=float, copy=True)
        if (
            azimuth.ndim != 1
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

    def __post_init__(self) -> None:
        positions = np.array(self.receiver_positions, dtype=float, copy=True)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
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


class FMCWTransformResult(StrictModule, NonTrainableState):
    range_doppler_channels: Array
    range_doppler_power: Array
    range_axis: Array
    doppler_axis: Array
    finite: Array
    successful: Array


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
    ):
        self.acquisition = acquisition
        self.window_fast = jnp.hanning(int(fast_samples))
        self.window_slow = jnp.hanning(int(chirps))
        self.propagation_speed = float(propagation_speed)

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
        return FMCWTransformResult(range_doppler, power, ranges, doppler, finite, finite)


@dataclass(frozen=True, slots=True)
class AutomotiveRadarProfile:
    frame_id: str
    range_unit: UnitDefinition
    velocity_unit: UnitDefinition

    def lower(
        self,
        detections: ArrayLike,
        reference: ReferenceArtifactManifest,
        /,
        *,
        asset_id: str,
    ) -> MeasurementCollection:
        values = np.asarray(detections, dtype=float)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError(
                "detections must contain range, azimuth, radial velocity, and radar cross section."
            )
        if not isinstance(self.range_unit, UnitDefinition) or not isinstance(
            self.velocity_unit, UnitDefinition
        ):
            raise TypeError("range_unit and velocity_unit must be UnitDefinition values.")
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
            ("radar-cross-section", ONE, "radar.cross-section"),
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


class CfRadialProvider:
    def read(
        self,
        path: str | Path,
        reference: ReferenceArtifactManifest,
        /,
        *,
        campaign_id: str,
        quantity_units: Mapping[str, UnitDefinition],
        maximum_gates: int = 100_000_000,
    ) -> MeasurementCollection:
        if util.find_spec("xarray") is None:
            raise ImportError(
                "CF/Radial admission requires the optional radar-cfradial extra."
            )
        reference.require_rights()
        source = Path(path).resolve()
        _verify_source(source, reference)
        backend = import_module("xarray")
        dataset = backend.open_dataset(source)
        azimuth = np.asarray(dataset["azimuth"].values)
        elevation = np.asarray(dataset["elevation"].values)
        ranges = np.asarray(dataset["range"].values)
        support = PolarVolumeSupport(
            azimuth, elevation, ranges, str(dataset.attrs.get("instrument_name", "radar"))
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
                raise ValueError(f"Missing physical unit for CF/Radial field {name!r}.")
            values = np.asarray(dataset[name].values).reshape(support.sample_shape)
            field = QuantityField(
                f"{campaign_id}.{name}",
                QuantitySpec(
                    "radar",
                    name.replace("_", "-"),
                    name.replace("_", "-"),
                    quantity_units[name],
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
        dataset.close()
        return MeasurementCollection(
            campaign_id,
            campaign_id,
            tuple(assets),
            tuple(
                MeasurementRoleAssignment(value.asset_id, MeasurementRole.OBSERVATION)
                for value in assets
            ),
        )


def _verify_source(path: Path, reference: ReferenceArtifactManifest) -> None:
    if not path.is_file() or path.stat().st_size != reference.size_bytes:
        raise ValueError("CF/Radial source size disagrees with its manifest.")
    digest = new_digest(reference.checksum_algorithm)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != reference.checksum:
        raise ValueError("CF/Radial source checksum disagrees with its manifest.")


__all__ = [
    "AutomotiveRadarProfile",
    "CfRadialProvider",
    "FMCWAcquisition",
    "FMCWTransformPlan",
    "FMCWTransformResult",
    "PolarVolumeSupport",
]
