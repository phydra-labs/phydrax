#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Calibrated sonar waveforms, beamforming, and bounded side-scan products."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module, util
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._interpolation import linear_interpolate

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import open_reference_artifact, ReferenceArtifactManifest
from ..units import (
    conversion_factor,
    derived_unit,
    LENGTH,
    PASCAL,
    TIME,
    UnitDefinition,
)
from ._asset import DataOrigin, DataStage, DerivationRecord, MeasurementAsset
from ._field import QuantityField, SamplingSemantics, SpatialSamplingKind
from ._quantity import QuantitySpec, ValueLayout
from ._support import IndexSampleSupport
from ._time import SampleTimeAxis


@dataclass(frozen=True, slots=True)
class SonarAcquisition:
    source_position: np.ndarray
    receiver_positions: np.ndarray
    sound_speed: float
    sound_speed_unit: UnitDefinition
    sample_axis: SampleTimeAxis
    coordinate_contract: SpatialCoordinateContract
    acquisition_id: str
    acquisition_contract_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if (
            not isinstance(self.sound_speed_unit, UnitDefinition)
            or self.sound_speed_unit.dimension != LENGTH / TIME
        ):
            raise ValueError("sound_speed_unit must have velocity dimension.")
        if not isinstance(self.sample_axis, SampleTimeAxis):
            raise TypeError("sample_axis must be SampleTimeAxis.")
        target_speed_unit = derived_unit(
            "sonar-length-per-time",
            (
                (self.coordinate_contract.length_unit, 1),
                (self.sample_axis.time_unit, -1),
            ),
        )
        speed = float(self.sound_speed) * float(
            conversion_factor(self.sound_speed_unit, target_speed_unit)
        )
        if (
            not isinstance(self.acquisition_id, str)
            or not self.acquisition_id
            or self.acquisition_id != self.acquisition_id.strip()
        ):
            raise ValueError("acquisition_id must be canonical nonempty text.")
        source = np.array(self.source_position, dtype=np.float64, copy=True)
        receivers = np.array(self.receiver_positions, dtype=np.float64, copy=True)
        if (
            source.shape != (3,)
            or receivers.ndim != 2
            or receivers.shape[1] != 3
            or receivers.shape[0] < 1
            or not np.all(np.isfinite(source))
            or not np.all(np.isfinite(receivers))
            or not np.isfinite(speed)
            or speed <= 0.0
        ):
            raise ValueError("Sonar geometry and sound speed are invalid.")
        if self.sample_axis.time_unit.dimension != TIME:
            raise ValueError("Sonar sample_axis must use a time unit.")
        source.setflags(write=False)
        receivers.setflags(write=False)
        object.__setattr__(self, "source_position", source)
        object.__setattr__(self, "sound_speed_unit", target_speed_unit)
        object.__setattr__(self, "sound_speed", speed)
        object.__setattr__(self, "receiver_positions", receivers)
        object.__setattr__(
            self,
            "acquisition_contract_id",
            canonical_fingerprint(
                {
                    "kind": "sonar-acquisition",
                    "source": array_tree_fingerprint(source),
                    "receivers": array_tree_fingerprint(receivers),
                    "sound_speed": speed,
                    "sound_speed_unit": target_speed_unit.unit_id,
                    "time": self.sample_axis.time_axis_id,
                    "coordinates": self.coordinate_contract.spatial_id,
                    "acquisition": self.acquisition_id,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class SonarWaveformAsset:
    measurement: MeasurementAsset
    acquisition: SonarAcquisition

    def __post_init__(self) -> None:
        expected = (
            self.acquisition.receiver_positions.shape[0],
            self.acquisition.sample_axis.sample_count,
        )
        if self.measurement.field.values.shape != expected:
            raise ValueError(f"Sonar waveform values must have shape {expected}.")
        if self.measurement.field.quantity.unit.dimension != PASCAL.dimension:
            raise ValueError("Sonar waveforms require pressure units.")


class BeamformingResult(StrictModule, NonTrainableState):
    image: Array
    delays: Array
    finite: Array
    successful: Array


class DelayAndSumBeamformingPlan(StrictModule, NonTrainableState):
    acquisition: SonarAcquisition = eqx.field(static=True)
    image_points: Array
    delays: Array

    def __init__(self, acquisition: SonarAcquisition, image_points: ArrayLike, /):
        if not isinstance(acquisition, SonarAcquisition):
            raise TypeError("acquisition must be SonarAcquisition.")
        points = np.asarray(image_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("image_points must be finite with shape (point_count, 3).")
        source_distance = np.sqrt(
            np.sum((points - acquisition.source_position) ** 2, axis=-1)
        )
        receiver_distance = np.sqrt(
            np.sum(
                (points[:, None, :] - acquisition.receiver_positions[None, :, :]) ** 2,
                axis=-1,
            )
        )
        self.acquisition = acquisition
        self.image_points = jnp.asarray(points)
        self.delays = jnp.asarray(
            (source_distance[:, None] + receiver_distance) / acquisition.sound_speed
        )

    def evaluate(self, waveforms: ArrayLike, /) -> BeamformingResult:
        values = jnp.asarray(waveforms)
        expected = (
            self.acquisition.receiver_positions.shape[0],
            self.acquisition.sample_axis.sample_count,
        )
        if values.shape != expected:
            raise ValueError(f"waveforms must have shape {expected}.")
        times = jnp.asarray(self.acquisition.sample_axis.sample_times)
        samples = jax.vmap(
            lambda delays: jax.vmap(
                lambda waveform, delay: (
                    linear_interpolate(
                        times,
                        waveform,
                        delay,
                        bounds="fill",
                        fill_value=0.0,
                    ).values
                )
            )(values, delays)
        )(self.delays)
        image = jnp.sum(samples, axis=-1)
        finite = jnp.all(jnp.isfinite(image))
        return BeamformingResult(image, self.delays, finite, finite)


class XtfSideScanProvider:
    def read(
        self,
        path: str | Path,
        reference: ReferenceArtifactManifest,
        acquisition: SonarAcquisition,
        /,
        *,
        asset_id: str,
        pressure_scale_pascals_per_count: float,
        maximum_samples: int = 100_000_000,
        maximum_source_bytes: int = 4_000_000_000,
        maximum_channels: int = 4096,
    ) -> SonarWaveformAsset:
        for name, value in (
            ("maximum_samples", maximum_samples),
            ("maximum_source_bytes", maximum_source_bytes),
            ("maximum_channels", maximum_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if util.find_spec("pyxtf") is None:
            raise ImportError(
                "XTF side-scan admission requires the optional sonar-xtf extra."
            )
        scale = float(pressure_scale_pascals_per_count)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "pressure_scale_pascals_per_count must be finite and positive."
            )
        reference.require_rights()
        if reference.size_bytes > maximum_source_bytes:
            raise MemoryError("XTF source exceeds maximum_source_bytes.")
        source = Path(path).expanduser().absolute()
        backend = import_module("pyxtf")
        with (
            open_reference_artifact(source, reference) as resource,
            TemporaryDirectory(prefix="phydrax-xtf-read-") as temporary,
        ):
            staged = Path(temporary) / source.name
            with staged.open("wb") as output:
                while chunk := resource.stream.read(1024 * 1024):
                    output.write(chunk)
            _, packets = backend.xtf_read(str(staged))
        channels = []
        for packet_group in packets.values():
            for packet in packet_group:
                if packet.__class__.__name__ == "XTFPingHeader":
                    channels.extend(
                        np.asarray(channel, dtype=np.float64) for channel in packet.data
                    )
        if not channels:
            raise ValueError("XTF source contains no supported sonar channels.")
        if any(channel.ndim != 1 or channel.size < 1 for channel in channels):
            raise ValueError("XTF sonar channels must be nonempty rank-one arrays.")
        if len(channels) > maximum_channels:
            raise MemoryError("XTF source exceeds maximum_channels.")
        width = max(value.size for value in channels)
        padded_sample_count = len(channels) * width
        if padded_sample_count > maximum_samples:
            raise MemoryError("XTF padded output exceeds maximum_samples.")
        values = np.zeros((len(channels), width), dtype=np.float64)
        valid = np.zeros_like(values, dtype=np.bool_)
        for index, channel in enumerate(channels):
            values[index, : channel.size] = scale * channel
            valid[index, : channel.size] = True
        support = IndexSampleSupport(
            values.shape,
            ("channel", "sample"),
            acquisition.sample_axis,
            1,
            acquisition.coordinate_contract.reference_frame,
        )
        field = QuantityField(
            f"{asset_id}.pressure",
            QuantitySpec("sonar", "pressure", "pressure", PASCAL, "physical.pressure"),
            ValueLayout.scalar(),
            support,
            SamplingSemantics(SpatialSamplingKind.POINT),
            values,
            valid,
        )
        asset = MeasurementAsset.from_single_reference(
            asset_id,
            field,
            reference,
            DerivationRecord(
                DataOrigin.EXTERNAL,
                DataStage.CALIBRATED,
                transformation_id=canonical_fingerprint(
                    {
                        "kind": "xtf-pressure-calibration",
                        "asset": asset_id,
                        "pascals_per_count": scale,
                    }
                ),
            ),
        )
        return SonarWaveformAsset(asset, acquisition)


__all__ = [
    "BeamformingResult",
    "DelayAndSumBeamformingPlan",
    "SonarAcquisition",
    "SonarWaveformAsset",
    "XtfSideScanProvider",
]
