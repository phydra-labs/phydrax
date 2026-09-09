#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Calibrated sonar waveforms, beamforming, and bounded side-scan products."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import new as new_digest
from importlib import import_module, util
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import ReferenceArtifactManifest
from ..units import PASCAL, SECOND
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
    sample_axis: SampleTimeAxis
    frame_id: str
    acquisition_id: str
    acquisition_contract_id: str = field(init=False)

    def __post_init__(self) -> None:
        source = np.array(self.source_position, dtype=float, copy=True)
        receivers = np.array(self.receiver_positions, dtype=float, copy=True)
        if (
            source.shape != (3,)
            or receivers.ndim != 2
            or receivers.shape[1] != 3
            or not np.all(np.isfinite(source))
            or not np.all(np.isfinite(receivers))
            or self.sound_speed <= 0.0
        ):
            raise ValueError("Sonar geometry and sound speed are invalid.")
        if (
            not isinstance(self.sample_axis, SampleTimeAxis)
            or self.sample_axis.time_unit != SECOND
        ):
            raise ValueError("Sonar sample_axis must use seconds.")
        source.setflags(write=False)
        receivers.setflags(write=False)
        object.__setattr__(self, "source_position", source)
        object.__setattr__(self, "receiver_positions", receivers)
        object.__setattr__(
            self,
            "acquisition_contract_id",
            canonical_fingerprint(
                {
                    "kind": "sonar-acquisition",
                    "source": array_tree_fingerprint(source),
                    "receivers": array_tree_fingerprint(receivers),
                    "sound_speed": self.sound_speed,
                    "time": self.sample_axis.time_axis_id,
                    "frame": self.frame_id,
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
        points = np.asarray(image_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("image_points must have shape (point_count, 3).")
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
                lambda waveform, delay: jnp.interp(
                    delay, times, waveform, left=0.0, right=0.0
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
    ) -> SonarWaveformAsset:
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
        source = Path(path).resolve()
        _verify_source(source, reference)
        backend = import_module("pyxtf")
        _, packets = backend.xtf_read(str(source))
        channels = []
        for packet_group in packets.values():
            for packet in packet_group:
                if packet.__class__.__name__ == "XTFPingHeader":
                    channels.extend(
                        np.asarray(channel, dtype=float) for channel in packet.data
                    )
        if not channels:
            raise ValueError("XTF source contains no supported sonar channels.")
        sample_count = sum(value.size for value in channels)
        if sample_count > maximum_samples:
            raise MemoryError("XTF source exceeds maximum_samples.")
        width = max(value.size for value in channels)
        values = np.zeros((len(channels), width))
        valid = np.zeros_like(values, dtype=bool)
        for index, channel in enumerate(channels):
            values[index, : channel.size] = scale * channel
            valid[index, : channel.size] = True
        support = IndexSampleSupport(
            values.shape,
            ("channel", "sample"),
            acquisition.sample_axis,
            1,
            acquisition.frame_id,
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


def _verify_source(path: Path, reference: ReferenceArtifactManifest) -> None:
    if not path.is_file() or path.stat().st_size != reference.size_bytes:
        raise ValueError("XTF source size disagrees with its manifest.")
    digest = new_digest(reference.checksum_algorithm)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != reference.checksum:
        raise ValueError("XTF source checksum disagrees with its manifest.")


__all__ = [
    "BeamformingResult",
    "DelayAndSumBeamformingPlan",
    "SonarAcquisition",
    "SonarWaveformAsset",
    "XtfSideScanProvider",
]
