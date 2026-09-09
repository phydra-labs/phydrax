#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded host decoder for an explicit, narrow SEG-Y revision-1 profile.

Profile: byte-stream files without tape labels, revision 1.0, fixed-length
trace records, and format-5 IEEE samples. ASCII/CP500 textual encoding is
caller-declared, never guessed. Big endian is standard rev1; explicitly
requested little endian is a named nonstandard profile.
Only length coordinates (metres/feet), seismic/pressure/dead trace IDs, and
unmuted source-relative delay clocks are interpreted. Other modalities, variable
lengths, angular coordinates, mutes, ambiguous calibration, and unsupported
header variants fail closed. Header bytes are retained exactly in the resource.

Reference: SEG Technical Standards Committee, SEG Y rev 1 (May 2002), Tables
2–3; https://www.iris.edu/hq/es_course/content/2009/session1/seg_y_rev1.pdf .
This is not a general SEG-Y reader, coordinate reprojection, or instrument-
response removal. Raw unqualified samples are never labelled acoustic pressure.
"""

from __future__ import annotations

import os
import struct
from math import isfinite
from typing import Literal, Protocol, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..series import SampledSeries, SeriesSupport
from ..units import METER, PASCAL, UnitDefinition
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
    ResourceManifest,
)


if TYPE_CHECKING:
    from ._geospatial import GeospatialContract


class SEGYDecodeError(ValueError):
    """Fail-closed unsupported, malformed, or inconsistent SEG-Y semantics."""


class _SEGYProfile(Protocol):
    pressure_calibration: float | None
    pressure_polarity: Literal["positive", "negative"] | None
    profile_id: str


class SEGYRev1IEEEProfile(StrictModule, NonTrainableState):
    """Explicit decoding choices; no inference from plausible header values.

    ``pressure_calibration`` is positive Pa per stored sample, permitted only
    when neither direct pressure units nor a transduction constant is present.
    ``pressure_polarity`` resolves unknown polarity only; contradictions with a
    populated binary header are rejected. Nonzero mutes are rejected rather than
    guessed to mean missing, tapered, or zero-valued data.
    """

    byte_order: Literal["big", "little"] = eqx.field(static=True)
    text_encoding: Literal["ascii", "cp500"] = eqx.field(static=True)
    coordinate_mapping: str = eqx.field(static=True)
    pressure_calibration: float | None = eqx.field(static=True)
    pressure_polarity: Literal["positive", "negative"] | None = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        byte_order: Literal["big", "little"],
        text_encoding: Literal["ascii", "cp500"],
        coordinate_mapping: str = "source-group-xyz",
        pressure_calibration: float | None = None,
        pressure_polarity: Literal["positive", "negative"] | None = None,
    ):
        if byte_order not in ("big", "little") or text_encoding not in ("ascii", "cp500"):
            raise ValueError(
                "SEG-Y byte order and text encoding must be explicitly supported choices."
            )
        if coordinate_mapping != "source-group-xyz":
            raise ValueError(
                "SEG-Y supports only explicit source/group XY and elevation-minus-depth Z mapping."
            )
        if pressure_calibration is not None and (
            not isfinite(pressure_calibration) or pressure_calibration <= 0
        ):
            raise ValueError(
                "External SEG-Y pressure calibration must be finite and positive."
            )
        if pressure_polarity not in (None, "positive", "negative"):
            raise ValueError(
                "SEG-Y pressure polarity must be positive, negative, or unspecified."
            )
        self.byte_order, self.text_encoding, self.coordinate_mapping = (
            byte_order,
            text_encoding,
            coordinate_mapping,
        )
        self.pressure_calibration, self.pressure_polarity = (
            pressure_calibration,
            pressure_polarity,
        )
        self.profile_id = canonical_fingerprint(
            {
                "kind": "segy-rev1-ieee-fixed-profile",
                "byte_order": byte_order,
                "text_encoding": text_encoding,
                "mapping": coordinate_mapping,
                "pressure_calibration": pressure_calibration,
                "pressure_polarity": pressure_polarity,
            }
        )


class DecodedSEGY(StrictModule, NonTrainableState):
    """Native trace-major samples, explicit clock/validity, geometry and audit.

    Coordinates are metres and Z is positive upwards: source Z equals surface
    elevation minus positive source depth; receiver Z is receiver group elevation.
    CRS/datum remain unqualified unless explicit compatible metadata was supplied.
    Sequence gaps remain explicit IDs; absent traces are never fabricated.
    Source clocks are relative to each trace's own source initiation, not UTC.
    """

    series: SampledSeries
    source_positions: Array
    receiver_positions: Array
    trace_sequence: Array
    trace_identification: Array
    sample_intervals: Array
    recording_delays: Array
    resource: BoundedResource = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)
    profile: _SEGYProfile = eqx.field(static=True)
    textual_header: str = eqx.field(static=True)
    coordinate_metadata: GeospatialContract | None = eqx.field(static=True)
    amplitude_unit: UnitDefinition | None = eqx.field(static=True)
    time_unit: str = eqx.field(static=True, default="s")
    time_reference: str = eqx.field(static=True, default="source-initiation")

    @property
    def manifest(self) -> ResourceManifest:
        return self.resource.manifest

    @property
    def pressure_qualified(self) -> bool:
        return self.amplitude_unit is not None


def _scalar(value: int, role: str) -> float:
    if value not in (0, 1, -1, 10, -10, 100, -100, 1000, -1000, 10000, -10000):
        raise SEGYDecodeError(f"Unsupported SEG-Y {role} scalar {value}.")
    return 1.0 if value == 0 else float(value) if value > 0 else 1.0 / abs(value)


def _pressure_factor(
    header: memoryview, endian: str, profile: _SEGYProfile, polarity: int
) -> tuple[float, bool]:
    unit = struct.unpack_from(endian + "h", header, 202)[0]
    mantissa = struct.unpack_from(endian + "i", header, 204)[0]
    exponent, transduction_unit = struct.unpack_from(endian + "hh", header, 208)
    if unit not in (0, 1):
        raise SEGYDecodeError(
            "SEG-Y profile accepts unknown or Pascal trace-value units only."
        )
    if mantissa == 0 and (exponent != 0 or transduction_unit != 0):
        raise SEGYDecodeError(
            "SEG-Y transduction metadata is incomplete or has zero gain."
        )
    if unit == 1:
        if profile.pressure_calibration is not None:
            raise SEGYDecodeError(
                "External calibration conflicts with SEG-Y Pascal sample units."
            )
        if mantissa and (
            transduction_unit != 1
            or not -12 <= exponent <= 12
            or mantissa * 10.0**exponent != 1.0
        ):
            raise SEGYDecodeError(
                "SEG-Y direct Pascal samples have a conflicting transduction constant."
            )
        factor, qualified = 1.0, True
    elif mantissa:
        if (
            profile.pressure_calibration is not None
            or transduction_unit != 1
            or not -12 <= exponent <= 12
        ):
            raise SEGYDecodeError(
                "SEG-Y requires an unambiguous bounded transduction constant to Pascal."
            )
        factor, qualified = mantissa * 10.0**exponent, True
        if not isfinite(factor) or factor <= 0:
            raise SEGYDecodeError(
                "SEG-Y pressure transduction gain must be finite and positive; polarity is separate."
            )
    elif profile.pressure_calibration is not None:
        factor, qualified = profile.pressure_calibration, True
    else:
        return 1.0, False
    specified = {None: 0, "negative": 1, "positive": 2}[profile.pressure_polarity]
    if polarity not in (0, 1, 2) or (specified and polarity and specified != polarity):
        raise SEGYDecodeError(
            "SEG-Y pressure polarity is invalid or contradicts the explicit profile."
        )
    resolved = polarity or specified
    if resolved == 0:
        raise SEGYDecodeError(
            "SEG-Y pressure requires a declared positive/negative impulse polarity."
        )
    return factor * (-1.0 if resolved == 1 else 1.0), qualified


def decode_segy_resource(
    resource: BoundedResource,
    /,
    *,
    profile: SEGYRev1IEEEProfile,
    coordinate_metadata: GeospatialContract | None = None,
) -> DecodedSEGY:
    """Decode one already-bounded resource, accounting before sample allocation."""
    data = resource.data
    if len(data) < 3600:
        raise SEGYDecodeError(
            "SEG-Y resource is shorter than its textual and binary headers."
        )
    endian = ">" if profile.byte_order == "big" else "<"
    binary = memoryview(data)[3200:3600]

    def u16(offset):
        return struct.unpack_from(endian + "H", binary, offset)[0]

    def i16(offset):
        return struct.unpack_from(endian + "h", binary, offset)[0]

    if u16(300) != 0x0100:
        raise SEGYDecodeError(
            "SEG-Y profile requires revision 1.0 in the declared byte order."
        )
    if u16(24) != 5:
        raise SEGYDecodeError(
            "SEG-Y profile supports only format 5 IEEE binary32 samples."
        )
    if u16(302) != 1 or i16(304) != 0:
        raise SEGYDecodeError(
            "SEG-Y profile requires fixed-length traces and zero extended textual headers."
        )
    count, interval = u16(20), u16(16)
    if count == 0 or interval == 0:
        raise SEGYDecodeError(
            "SEG-Y binary sample count and microsecond interval must be nonzero."
        )
    measurement_system = u16(54)
    if measurement_system not in (1, 2):
        raise SEGYDecodeError(
            "SEG-Y measurement system must explicitly identify metres or feet."
        )
    if u16(52) != 1:
        raise SEGYDecodeError(
            "SEG-Y profile requires explicitly no amplitude recovery (binary code 1)."
        )
    length_factor = 1.0 if measurement_system == 1 else 0.3048
    record_size = 240 + 4 * count
    trace_count, trailing = divmod(len(data) - 3600, record_size)
    if trace_count == 0 or trailing:
        raise SEGYDecodeError(
            "SEG-Y trace records are empty, truncated, or have trailing bytes."
        )
    if coordinate_metadata is not None:
        spatial = coordinate_metadata.require_cartesian(dimensions=3)
        if spatial.length_unit != METER:
            raise SEGYDecodeError(
                "SEG-Y coordinate metadata must describe the normalized metre coordinates."
            )
    # Each sample and trace is counted before any decoded array is allocated.
    resource = account_bounded_resource(
        resource,
        depth=2,
        nodes=3 + trace_count * (count + 1),
        attributes=10 + 32 * trace_count,
        losses=3,
    )
    try:
        text = data[:3200].decode(profile.text_encoding, errors="strict")
    except UnicodeDecodeError as error:
        raise SEGYDecodeError(
            "SEG-Y textual header does not match the declared encoding."
        ) from error
    if any(ord(character) < 32 or ord(character) > 126 for character in text):
        raise SEGYDecodeError(
            "SEG-Y textual header must contain printable characters in forty 80-column cards."
        )
    samples = np.empty((trace_count, count), dtype=np.float64)
    sources, receivers = np.empty((trace_count, 3)), np.empty((trace_count, 3))
    sequence, identification = (
        np.empty(trace_count, dtype=np.int32),
        np.empty(trace_count, dtype=np.int32),
    )
    delays = np.empty(trace_count)
    valid = np.empty((trace_count, count), dtype=bool)
    qualified: bool | None = None
    for trace in range(trace_count):
        start = 3600 + trace * record_size
        header = memoryview(data)[start : start + 240]

        def h(offset, current_header=header):
            return struct.unpack_from(endian + "h", current_header, offset)[0]

        def i(offset, current_header=header):
            return struct.unpack_from(endian + "i", current_header, offset)[0]

        trace_samples, trace_interval = struct.unpack_from(endian + "HH", header, 114)
        if trace_samples != count or trace_interval != interval:
            raise SEGYDecodeError(
                "SEG-Y trace and binary fixed-length sample clocks disagree."
            )
        sequence[trace], identification[trace] = i(4), h(28)
        if sequence[trace] <= 0 or (trace and sequence[trace] <= sequence[trace - 1]):
            raise SEGYDecodeError(
                "SEG-Y file trace sequences must be positive and strictly increasing; gaps are retained."
            )
        if identification[trace] not in (1, 2, 11):
            raise SEGYDecodeError(
                "SEG-Y profile accepts seismic, pressure-sensor, and dead traces only."
            )
        if h(88) != 1:
            raise SEGYDecodeError(
                "SEG-Y angular, unknown, or proprietary coordinate units are unsupported."
            )
        if h(110) != 0 or h(112) != 0:
            raise SEGYDecodeError(
                "SEG-Y nonzero mute headers require unsupported sample-validity interpretation."
            )
        coordinate_scale = _scalar(h(70), "coordinate") * length_factor
        elevation_scale = _scalar(h(68), "elevation") * length_factor
        time_scale = _scalar(h(214), "time")
        if i(48) < 0:
            raise SEGYDecodeError(
                "SEG-Y source depth must be nonnegative below the source surface."
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
                "SEG-Y file mixes calibrated pressure and unqualified sample units."
            )
        qualified = pressure
        raw = np.frombuffer(data, dtype=endian + "f4", count=count, offset=start + 240)
        samples[trace] = raw.astype(np.float64) * factor
        valid[trace] = identification[trace] != 2
        if np.any(valid[trace] & ~np.isfinite(samples[trace])):
            raise SEGYDecodeError("SEG-Y live trace contains nonfinite samples.")
    times = delays[:, None] + np.arange(count)[None, :] * interval * 1e-6
    identity = canonical_fingerprint(
        {
            "kind": "decoded-segy",
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
    losses = (
        AdapterLoss(
            "coordinates",
            "import",
            "transformed",
            "Header scalars and metres/feet are normalized to metre source/group "
            "XYZ; native floating precision may round. Original bytes are retained.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "sample-clock",
            "import",
            "transformed",
            "Microsecond intervals and time-scaled millisecond source delays become "
            "seconds; native floating precision may round. Dead traces remain invalid "
            "and sequence gaps are not filled.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "sample-values",
            "import",
            "transformed",
            "Explicit Pascal calibration/polarity is applied when qualified; otherwise "
            "samples remain unqualified. Original bytes and all uninterpreted headers "
            "are retained.",
            changes_interpretation=False,
        ),
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "SEG-Y",
        "SampledSeries",
        source_id=resource.manifest.content_sha256,
        target_id=identity,
        source_profile=AdapterFormatProfile(
            "SEG-Y",
            qualifiers={
                "revision": "1.0",
                "sample_format": "5-ieee-binary32",
                "byte_order": profile.byte_order,
                "text_encoding": profile.text_encoding,
                "trace_length": "fixed",
                "extended_headers": "none",
            },
        ),
        coordinate_mapping=(
            "source=(scaled sx,scaled sy,scaled(surface_elevation-source_depth))",
            "receiver=(scaled gx,scaled gy,scaled group_elevation)",
            "positive-up metre coordinates; CRS not inferred",
        ),
        preserved_fields=(
            "exact resource bytes",
            "trace order and sequence gaps",
            "dead trace validity",
            "per-trace source-relative sample clocks",
            "source and receiver coordinates",
        ),
        assumptions=(
            "Little-endian rev1 is an explicitly selected nonstandard variant, not byte-order autodetection.",
            "No instrument-response removal, statics reapplication, or processing-history interpretation.",
            "Coordinates without explicit GeospatialContract are not CRS/datum-qualified.",
            "Per-trace source clocks are not global UTC or simultaneous-shot clocks.",
            "2D line-source acquisition requires explicit geometric reduction by the caller.",
        ),
        losses=losses,
        capabilities=(
            AdapterCapability("seismic-samples"),
            AdapterCapability("sample-validity"),
            AdapterCapability("source-relative-time"),
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
        jnp.full((trace_count,), interval * 1e-6),
        jnp.asarray(delays),
        resource,
        report,
        profile,
        text,
        coordinate_metadata,
        PASCAL if qualified else None,
    )


def decode_segy_bytes(
    data: bytes,
    /,
    *,
    profile: SEGYRev1IEEEProfile,
    limits: ResourceLimits,
    coordinate_metadata: GeospatialContract | None = None,
) -> DecodedSEGY:
    return decode_segy_resource(
        bounded_resource_from_bytes(data, limits=limits),
        profile=profile,
        coordinate_metadata=coordinate_metadata,
    )


def read_segy(
    path: str | os.PathLike[str],
    /,
    *,
    trusted_root: str | os.PathLike[str],
    profile: SEGYRev1IEEEProfile,
    limits: ResourceLimits,
    coordinate_metadata: GeospatialContract | None = None,
) -> DecodedSEGY:
    """Read beneath a trusted root without symlink/remote/unbounded file access."""
    return decode_segy_resource(
        read_bounded_resource(path, trusted_root=trusted_root, limits=limits),
        profile=profile,
        coordinate_metadata=coordinate_metadata,
    )


__all__ = [
    "DecodedSEGY",
    "SEGYDecodeError",
    "SEGYRev1IEEEProfile",
    "decode_segy_bytes",
    "decode_segy_resource",
    "read_segy",
]
