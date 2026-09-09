#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import struct
from pathlib import Path
from typing import Literal, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..series import SampledSeries, SeriesSupport
from ..units import METER, PASCAL
from ._report import (
    AdapterCapability,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
)
from ._resource import (
    account_bounded_resource,
    bounded_resource_from_bytes,
    BoundedResource,
    read_bounded_resource,
    ResourceLimits,
)
from ._segy import (
    _pressure_factor,
    _scalar,
    DecodedSEGY,
    SEGYDecodeError,
)


if TYPE_CHECKING:
    from ._geospatial import GeospatialContract


class SEGYRev2IEEEProfile(StrictModule, NonTrainableState):
    """SEG-Y 2.0 IEEE profile with fixed or variable 16-bit trace counts.

    Extended textual headers are supported when their exact count is declared.
    Extended trace headers and 32-bit extended sample-count fields are rejected;
    callers needing those distinct rev2 features must not use this profile.
    """

    byte_order: Literal["big", "little"] = eqx.field(static=True)
    text_encoding: Literal["ascii", "cp500"] = eqx.field(static=True)
    coordinate_mapping: str = eqx.field(static=True)
    pressure_calibration: float | None = eqx.field(static=True)
    pressure_polarity: Literal["positive", "negative"] | None = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    trace_length: Literal["fixed", "variable"] = eqx.field(static=True)
    extended_textual_header_count: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        byte_order: Literal["big", "little"],
        text_encoding: Literal["ascii", "cp500"],
        trace_length: Literal["fixed", "variable"],
        extended_textual_header_count: int = 0,
        coordinate_mapping: str = "source-group-xyz",
        pressure_calibration: float | None = None,
        pressure_polarity: Literal["positive", "negative"] | None = None,
    ):
        if byte_order not in ("big", "little") or text_encoding not in (
            "ascii",
            "cp500",
        ):
            raise ValueError("SEG-Y rev2 byte order and text encoding are unsupported.")
        if coordinate_mapping != "source-group-xyz":
            raise ValueError("SEG-Y rev2 requires explicit source/group XYZ mapping.")
        if pressure_calibration is not None and (
            not np.isfinite(pressure_calibration) or pressure_calibration <= 0
        ):
            raise ValueError("SEG-Y rev2 pressure calibration must be positive finite.")
        if pressure_polarity not in (None, "positive", "negative"):
            raise ValueError("SEG-Y rev2 pressure polarity is invalid.")
        self.byte_order, self.text_encoding = byte_order, text_encoding
        self.coordinate_mapping = coordinate_mapping
        self.pressure_calibration, self.pressure_polarity = (
            pressure_calibration,
            pressure_polarity,
        )
        count = int(extended_textual_header_count)
        if (
            trace_length not in ("fixed", "variable")
            or count != extended_textual_header_count
            or count < 0
        ):
            raise ValueError(
                "SEG-Y rev2 trace length and extended header count must be explicit."
            )
        self.trace_length = trace_length
        self.extended_textual_header_count = count
        self.profile_id = canonical_fingerprint(
            {
                "kind": "segy-rev2-ieee-profile",
                "byte_order": byte_order,
                "text_encoding": text_encoding,
                "trace_length": trace_length,
                "extended_textual_header_count": count,
                "mapping": coordinate_mapping,
                "pressure_calibration": pressure_calibration,
                "pressure_polarity": pressure_polarity,
                "trace_sample_count": "unsigned-16-bit",
                "trace_header_extensions": "none",
            }
        )


def _decode_text(data: bytes, profile: SEGYRev2IEEEProfile) -> str:
    count = 1 + profile.extended_textual_header_count
    try:
        text = data[: 3200 * count].decode(profile.text_encoding, errors="strict")
    except UnicodeDecodeError as error:
        raise SEGYDecodeError(
            "SEG-Y textual headers do not match the declared encoding."
        ) from error
    if any(ord(character) < 32 or ord(character) > 126 for character in text):
        raise SEGYDecodeError(
            "SEG-Y textual headers must contain printable card-image characters."
        )
    return text


def decode_segy_rev2_resource(
    resource: BoundedResource,
    /,
    *,
    profile: SEGYRev2IEEEProfile,
    coordinate_metadata: GeospatialContract | None = None,
) -> DecodedSEGY:
    if not isinstance(profile, SEGYRev2IEEEProfile):
        raise TypeError("SEG-Y revision 2 decoder requires SEGYRev2IEEEProfile.")
    data = resource.data
    data_start = 3600 + 3200 * profile.extended_textual_header_count
    if len(data) < data_start + 240:
        raise SEGYDecodeError(
            "SEG-Y revision 2 resource is shorter than its declared headers."
        )
    endian = ">" if profile.byte_order == "big" else "<"
    binary = memoryview(data)[3200:3600]

    def u16(offset):
        return struct.unpack_from(endian + "H", binary, offset)[0]

    def i16(offset):
        return struct.unpack_from(endian + "h", binary, offset)[0]

    if u16(300) != 0x0200 or u16(24) != 5:
        raise SEGYDecodeError(
            "SEG-Y rev2 profile requires revision 2.0 and IEEE binary32 samples."
        )
    fixed = u16(302)
    expected_fixed = 1 if profile.trace_length == "fixed" else 0
    if fixed != expected_fixed or i16(304) != profile.extended_textual_header_count:
        raise SEGYDecodeError(
            "SEG-Y rev2 binary header disagrees with the declared trace/header profile."
        )
    if u16(52) != 1:
        raise SEGYDecodeError(
            "SEG-Y rev2 profile requires explicitly no amplitude recovery."
        )
    measurement_system = u16(54)
    if measurement_system not in (1, 2):
        raise SEGYDecodeError("SEG-Y measurement system must identify metres or feet.")
    length_factor = 1.0 if measurement_system == 1 else 0.3048
    binary_count, binary_interval = u16(20), u16(16)
    if profile.trace_length == "fixed" and (binary_count == 0 or binary_interval == 0):
        raise SEGYDecodeError("Fixed-length SEG-Y requires nonzero binary trace clock.")

    starts: list[int] = []
    counts: list[int] = []
    intervals: list[int] = []
    cursor = data_start
    while cursor < len(data):
        if len(data) - cursor < 240:
            raise SEGYDecodeError("SEG-Y revision 2 has a truncated trace header.")
        header = memoryview(data)[cursor : cursor + 240]
        trace_count, trace_interval = struct.unpack_from(endian + "HH", header, 114)
        if trace_count == 0 or trace_interval == 0:
            raise SEGYDecodeError(
                "Every supported SEG-Y rev2 trace needs a 16-bit sample clock."
            )
        if profile.trace_length == "fixed" and (
            trace_count != binary_count or trace_interval != binary_interval
        ):
            raise SEGYDecodeError(
                "Fixed SEG-Y rev2 trace clock disagrees with binary header."
            )
        end = cursor + 240 + 4 * trace_count
        if end > len(data):
            raise SEGYDecodeError("SEG-Y revision 2 has a truncated trace payload.")
        starts.append(cursor)
        counts.append(trace_count)
        intervals.append(trace_interval)
        cursor = end
    if not starts or cursor != len(data):
        raise SEGYDecodeError(
            "SEG-Y revision 2 trace stream is empty or has trailing bytes."
        )
    trace_count, maximum_count = len(starts), max(counts)
    resource = account_bounded_resource(
        resource,
        depth=2,
        nodes=3 + trace_count * (maximum_count + 1),
        attributes=12 + 34 * trace_count,
        losses=4,
    )
    text = _decode_text(data, profile)
    samples = np.zeros((trace_count, maximum_count), dtype=np.float64)
    valid = np.zeros((trace_count, maximum_count), dtype=bool)
    times = np.zeros((trace_count, maximum_count), dtype=np.float64)
    sources, receivers = np.empty((trace_count, 3)), np.empty((trace_count, 3))
    sequence = np.empty(trace_count, dtype=np.int32)
    identification = np.empty(trace_count, dtype=np.int32)
    delays, sample_intervals = (
        np.empty(trace_count),
        np.asarray(intervals, dtype=float) * 1e-6,
    )
    qualified: bool | None = None
    previous_sequence = 0
    for trace, (start, count, interval_us) in enumerate(
        zip(starts, counts, intervals, strict=True)
    ):
        header = memoryview(data)[start : start + 240]

        def h(offset, current=header):
            return struct.unpack_from(endian + "h", current, offset)[0]

        def i(offset, current=header):
            return struct.unpack_from(endian + "i", current, offset)[0]

        sequence[trace], identification[trace] = i(4), h(28)
        if sequence[trace] <= previous_sequence:
            raise SEGYDecodeError(
                "SEG-Y file trace sequences must be positive and strictly increasing."
            )
        previous_sequence = int(sequence[trace])
        if identification[trace] not in (1, 2, 11) or h(88) != 1:
            raise SEGYDecodeError(
                "SEG-Y rev2 profile accepts linear seismic, pressure, or dead traces only."
            )
        if h(110) != 0 or h(112) != 0:
            raise SEGYDecodeError(
                "SEG-Y nonzero mutes need an unsupported validity interpretation."
            )
        coordinate_scale = _scalar(h(70), "coordinate") * length_factor
        elevation_scale = _scalar(h(68), "elevation") * length_factor
        time_scale = _scalar(h(214), "time")
        if i(48) < 0:
            raise SEGYDecodeError(
                "SEG-Y source depth must be nonnegative below source surface."
            )
        sources[trace] = (
            i(72) * coordinate_scale,
            i(76) * coordinate_scale,
            (i(44) - i(48)) * elevation_scale,
        )
        receivers[trace] = (
            i(80) * coordinate_scale,
            i(84) * coordinate_scale,
            i(40) * elevation_scale,
        )
        delays[trace] = h(108) * time_scale * 1e-3
        factor, pressure = _pressure_factor(header, endian, profile, u16(56))
        if qualified is not None and qualified != pressure:
            raise SEGYDecodeError(
                "SEG-Y file mixes pressure-calibrated and unqualified traces."
            )
        qualified = pressure
        raw = np.frombuffer(data, dtype=endian + "f4", count=count, offset=start + 240)
        samples[trace, :count] = raw.astype(np.float64) * factor
        valid[trace, :count] = identification[trace] != 2
        if np.any(valid[trace, :count] & ~np.isfinite(samples[trace, :count])):
            raise SEGYDecodeError("SEG-Y live trace contains nonfinite samples.")
        times[trace] = delays[trace] + np.arange(maximum_count) * interval_us * 1e-6
    if coordinate_metadata is not None:
        spatial = coordinate_metadata.require_cartesian(dimensions=3)
        if spatial.length_unit != METER:
            raise SEGYDecodeError(
                "SEG-Y coordinate metadata must describe normalized metres."
            )
    identity = canonical_fingerprint(
        {
            "kind": "decoded-segy-rev2",
            "resource": resource.manifest.content_sha256,
            "profile": profile.profile_id,
            "coordinates": None
            if coordinate_metadata is None
            else coordinate_metadata.geospatial_id,
        }
    )
    support = SeriesSupport(
        times,
        node_valid=valid,
        series_shape=(trace_count,),
        series_axes=("trace",),
        coordinate_name="time_s",
        coordinate_id=identity,
    )
    series = SampledSeries(
        support, jnp.asarray(samples), value_valid=jnp.asarray(valid), series_id=identity
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "SEG-Y-2.0",
        "SampledSeries",
        source_id=resource.manifest.content_sha256,
        target_id=identity,
        source_profile=AdapterFormatProfile(
            "SEG-Y-2.0",
            qualifiers={
                "revision": "2.0",
                "sample_format": "5-ieee-binary32",
                "trace_length": profile.trace_length,
                "sample_count": "unsigned-16-bit-trace-header",
                "extended_textual_headers": profile.extended_textual_header_count,
                "trace_header_extensions": "none",
            },
        ),
        coordinate_mapping=(
            "source/group XYZ normalized to positive-up metres",
            "CRS/datum supplied externally or remains unqualified",
        ),
        preserved_fields=(
            "exact resource bytes",
            "variable trace clocks and validity",
            "sequence gaps",
            "source and receiver coordinates",
            "textual headers",
        ),
        assumptions=(
            "Trace-header 16-bit sample counts are authoritative.",
            "No extended trace headers, response removal, statics, or mute interpretation.",
        ),
        losses=(
            AdapterLoss(
                "variable-trace-shape",
                "import",
                "transformed",
                "Short traces are padded to the longest trace with explicit invalid samples.",
                changes_interpretation=False,
            ),
        ),
        capabilities=(
            AdapterCapability("seismic-samples"),
            AdapterCapability("variable-trace-clock"),
            AdapterCapability("sample-validity"),
            AdapterCapability("source-group-geometry"),
            AdapterCapability("exact-source-bytes"),
        ),
    )
    return DecodedSEGY(
        series,
        jnp.asarray(sources),
        jnp.asarray(receivers),
        jnp.asarray(sequence),
        jnp.asarray(identification),
        jnp.asarray(sample_intervals),
        jnp.asarray(delays),
        resource,
        report,
        profile,
        text,
        coordinate_metadata,
        PASCAL if qualified else None,
    )


def decode_segy_rev2_bytes(
    data: bytes,
    /,
    *,
    profile: SEGYRev2IEEEProfile,
    limits: ResourceLimits,
    coordinate_metadata: GeospatialContract | None = None,
) -> DecodedSEGY:
    return decode_segy_rev2_resource(
        bounded_resource_from_bytes(data, limits=limits),
        profile=profile,
        coordinate_metadata=coordinate_metadata,
    )


def read_segy_rev2(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    profile: SEGYRev2IEEEProfile,
    limits: ResourceLimits,
    coordinate_metadata: GeospatialContract | None = None,
) -> DecodedSEGY:
    return decode_segy_rev2_resource(
        read_bounded_resource(path, trusted_root=trusted_root, limits=limits),
        profile=profile,
        coordinate_metadata=coordinate_metadata,
    )


__all__ = [
    "SEGYRev2IEEEProfile",
    "decode_segy_rev2_bytes",
    "decode_segy_rev2_resource",
    "read_segy_rev2",
]
