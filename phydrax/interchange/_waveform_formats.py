#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import Any, cast, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..series import SampledSeries, SeriesSupport
from ..units import UnitDefinition
from ._geospatial import GeospatialContract
from ._report import (
    AdapterCapability,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
)
from ._resource import BoundedResource, read_bounded_resource, ResourceLimits
from ._time_reference import TimeReferenceContract


class SeismicFormatDependencyError(RuntimeError):
    """An optional standards-compliant waveform runtime is unavailable."""


class QualifiedWaveformTrace(StrictModule, NonTrainableState):
    series: SampledSeries
    network: str = eqx.field(static=True)
    station: str = eqx.field(static=True)
    location: str = eqx.field(static=True)
    channel: str = eqx.field(static=True)
    start_tai_seconds: float = eqx.field(static=True)
    sample_unit: UnitDefinition = eqx.field(static=True)
    source_resource_id: str = eqx.field(static=True)
    trace_id: str = eqx.field(static=True)


class QualifiedWaveformCollection(StrictModule, NonTrainableState):
    format: Literal["miniseed3", "sac"] = eqx.field(static=True)
    traces: tuple[QualifiedWaveformTrace, ...]
    resource: BoundedResource = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)
    collection_id: str = eqx.field(static=True)


class StationChannelMetadata(StrictModule, NonTrainableState):
    network: str = eqx.field(static=True)
    station: str = eqx.field(static=True)
    location: str = eqx.field(static=True)
    channel: str = eqx.field(static=True)
    position: tuple[float, float, float] = eqx.field(static=True)
    orientation: tuple[float, float, float] = eqx.field(static=True)
    sample_rate_hz: float = eqx.field(static=True)
    sensitivity: float | None = eqx.field(static=True)
    input_units: str | None = eqx.field(static=True)
    output_units: str | None = eqx.field(static=True)
    channel_id: str = eqx.field(static=True)


class StationXMLMetadata(StrictModule, NonTrainableState):
    channels: tuple[StationChannelMetadata, ...]
    coordinates: GeospatialContract
    resource: BoundedResource = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)
    metadata_id: str = eqx.field(static=True)


def _obspy():
    try:
        return cast(Any, import_module("obspy"))
    except ImportError as error:
        raise SeismicFormatDependencyError(
            "SAC and StationXML require the optional ObsPy runtime."
        ) from error


def _pymseed():
    try:
        return cast(Any, import_module("pymseed"))
    except ImportError as error:
        raise SeismicFormatDependencyError(
            "miniSEED3 requires the optional pymseed runtime."
        ) from error


def _utc_start_tai(timestamp: float, contract: TimeReferenceContract) -> float:
    if contract.scale != "utc" or contract.epoch_nominal_seconds is None:
        raise ValueError("Absolute waveform timestamps require a UTC time contract.")
    recorded = timestamp - contract.epoch_nominal_seconds
    return float(contract.to_tai(recorded))


def _qualified_trace(
    samples: np.ndarray,
    valid: np.ndarray,
    codes: tuple[str, str, str, str],
    start_timestamp: float,
    delta: float,
    contract: TimeReferenceContract,
    sample_unit: UnitDefinition,
    resource_id: str,
    row: int,
) -> QualifiedWaveformTrace:
    if samples.ndim != 1 or samples.size == 0 or np.any(valid & ~np.isfinite(samples)):
        raise ValueError(
            "Waveform trace samples must be one-dimensional and finite where valid."
        )
    if valid.shape != samples.shape:
        raise ValueError("Waveform sample validity must match the trace shape.")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("Waveform sample interval must be positive and finite.")
    start = _utc_start_tai(start_timestamp, contract)
    identity = canonical_fingerprint(
        {
            "kind": "qualified-waveform-trace",
            "resource": resource_id,
            "row": row,
            "codes": codes,
            "start_tai_seconds": start,
            "delta_seconds": delta,
            "count": samples.size,
            "sample_unit": sample_unit.unit_id,
        }
    )
    support = SeriesSupport(
        delta * np.arange(samples.size),
        node_valid=valid,
        coordinate_name="seconds_since_trace_start",
        coordinate_id=identity,
    )
    return QualifiedWaveformTrace(
        SampledSeries(
            support,
            jnp.asarray(samples),
            value_valid=jnp.asarray(valid),
            series_id=identity,
        ),
        *codes,
        start,
        sample_unit,
        resource_id,
        identity,
    )


def _trace_record(trace, contract, sample_unit, resource_id, row):
    values = np.asanyarray(trace.data)
    if np.ma.isMaskedArray(values):
        valid = ~np.ma.getmaskarray(values)
        samples = np.asarray(values.filled(0), dtype=float)
    else:
        samples = np.asarray(values, dtype=float)
        valid = np.ones(samples.shape, dtype=bool)
    return _qualified_trace(
        samples,
        valid,
        (
            str(trace.stats.network),
            str(trace.stats.station),
            str(trace.stats.location),
            str(trace.stats.channel),
        ),
        float(trace.stats.starttime.timestamp),
        float(trace.stats.delta),
        contract,
        sample_unit,
        resource_id,
        row,
    )


def _waveform_collection(
    profile: Literal["miniseed3", "sac"],
    traces: tuple[QualifiedWaveformTrace, ...],
    resource: BoundedResource,
) -> QualifiedWaveformCollection:
    identity = canonical_fingerprint(
        {
            "kind": "qualified-waveform-collection",
            "profile": profile,
            "resource": resource.manifest.content_sha256,
            "traces": [trace.trace_id for trace in traces],
        }
    )
    source_format = "miniSEED3" if profile == "miniseed3" else "SAC"
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        source_format,
        "QualifiedWaveformCollection",
        source_id=resource.manifest.content_sha256,
        target_id=identity,
        source_profile=AdapterFormatProfile(
            source_format, qualifiers={"profile": profile}
        ),
        preserved_fields=(
            "trace samples and explicit invalid mask",
            "network station location channel codes",
            "sample interval and UTC start",
            "exact source bytes",
        ),
        assumptions=(
            "Sample physical unit is caller supplied, not inferred from waveform bytes.",
            "Instrument response removal is a separate StationXML-qualified operation.",
        ),
        losses=(
            AdapterLoss(
                "format-specific-headers",
                "import",
                "dropped",
                "Unmodeled headers remain available only through the exact source resource.",
                changes_interpretation=False,
            ),
        ),
        capabilities=(
            AdapterCapability("seismic-waveform"),
            AdapterCapability("absolute-time"),
            AdapterCapability("sample-validity"),
        ),
    )
    return QualifiedWaveformCollection(profile, traces, resource, report, identity)


def _read_waveforms(
    resource: BoundedResource,
    *,
    format: Literal["MSEED", "SAC"],
    profile: Literal["miniseed3", "sac"],
    utc_time: TimeReferenceContract,
    sample_unit: UnitDefinition,
    maximum_samples: int,
) -> QualifiedWaveformCollection:
    obspy = _obspy()
    headers = obspy.read(BytesIO(resource.data), format=format, headonly=True)
    declared_samples = sum(int(trace.stats.npts) for trace in headers)
    if not headers or declared_samples <= 0 or declared_samples > maximum_samples:
        raise ValueError("Waveform trace count exceeds its decoded-sample limit.")
    stream = obspy.read(BytesIO(resource.data), format=format)
    traces = tuple(
        _trace_record(trace, utc_time, sample_unit, resource.manifest.content_sha256, row)
        for row, trace in enumerate(stream)
    )
    return _waveform_collection(profile, traces, resource)


def read_miniseed3(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    limits: ResourceLimits,
    utc_time: TimeReferenceContract,
    sample_unit: UnitDefinition,
) -> QualifiedWaveformCollection:
    resource = read_bounded_resource(path, trusted_root=trusted_root, limits=limits)
    pymseed = _pymseed()
    record_count = 0
    declared_samples = 0
    for record in pymseed.MS3Record.from_buffer(resource.data, validate_crc=True):
        if record.formatversion != 3:
            raise ValueError("read_miniseed3 accepts only miniSEED version 3 records.")
        samples = int(record.samplecnt)
        if samples <= 0:
            raise ValueError("miniSEED3 profile accepts only nonempty data records.")
        declared_samples += samples
        if declared_samples > limits.max_nodes:
            raise ValueError("miniSEED3 data exceed the decoded-sample limit.")
        record_count += 1
    if record_count == 0:
        raise ValueError("miniSEED3 resource contains no data records.")
    qualified: list[QualifiedWaveformTrace] = []
    with pymseed.MS3TraceList.from_buffer(
        resource.data, unpack_data=True, validate_crc=True, split_version=True
    ) as trace_list:
        for trace_identifier in trace_list:
            codes = tuple(pymseed.sourceid2nslc(trace_identifier.sourceid))
            for segment in trace_identifier:
                sample_rate = float(segment.samprate)
                if not np.isfinite(sample_rate) or sample_rate == 0:
                    raise ValueError("miniSEED3 sample rate must be finite and nonzero.")
                delta = 1.0 / sample_rate if sample_rate > 0 else -sample_rate
                samples = np.array(segment.np_datasamples, dtype=float, copy=True)
                valid = np.ones(samples.shape, dtype=bool)
                qualified.append(
                    _qualified_trace(
                        samples,
                        valid,
                        codes,
                        float(segment.starttime_seconds),
                        delta,
                        utc_time,
                        sample_unit,
                        resource.manifest.content_sha256,
                        len(qualified),
                    )
                )
    if not qualified:
        raise ValueError("miniSEED3 resource contains no decodable traces.")
    return _waveform_collection("miniseed3", tuple(qualified), resource)


def read_sac(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    limits: ResourceLimits,
    utc_time: TimeReferenceContract,
    sample_unit: UnitDefinition,
) -> QualifiedWaveformCollection:
    resource = read_bounded_resource(path, trusted_root=trusted_root, limits=limits)
    return _read_waveforms(
        resource,
        format="SAC",
        profile="sac",
        utc_time=utc_time,
        sample_unit=sample_unit,
        maximum_samples=limits.max_nodes,
    )


def read_stationxml(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    limits: ResourceLimits,
    coordinates: GeospatialContract,
) -> StationXMLMetadata:
    if (
        not isinstance(coordinates, GeospatialContract)
        or coordinates.horizontal_kind != "geographic"
    ):
        raise ValueError("StationXML positions require an explicit geographic contract.")
    resource = read_bounded_resource(path, trusted_root=trusted_root, limits=limits)
    inventory = _obspy().read_inventory(BytesIO(resource.data), format="STATIONXML")
    channels: list[StationChannelMetadata] = []
    for network in inventory:
        for station in network:
            for channel in station:
                azimuth = np.deg2rad(float(channel.azimuth))
                dip = np.deg2rad(float(channel.dip))
                orientation = (
                    float(np.cos(dip) * np.sin(azimuth)),
                    float(np.cos(dip) * np.cos(azimuth)),
                    float(-np.sin(dip)),
                )
                response = channel.response
                sensitivity = None
                input_units = output_units = None
                if response is not None and response.instrument_sensitivity is not None:
                    sensitivity = float(response.instrument_sensitivity.value)
                    input_units = str(response.instrument_sensitivity.input_units)
                    output_units = str(response.instrument_sensitivity.output_units)
                values = (
                    str(network.code),
                    str(station.code),
                    str(channel.location_code),
                    str(channel.code),
                )
                position = (
                    float(channel.longitude),
                    float(channel.latitude),
                    float(channel.elevation) - float(channel.depth),
                )
                sample_rate = float(channel.sample_rate)
                if (
                    any(not np.isfinite(value) for value in (*position, *orientation))
                    or not -180 <= position[0] <= 180
                    or not -90 <= position[1] <= 90
                    or not np.isfinite(sample_rate)
                    or sample_rate <= 0
                    or (
                        sensitivity is not None
                        and (not np.isfinite(sensitivity) or sensitivity == 0)
                    )
                ):
                    raise ValueError(
                        "StationXML channel geometry, orientation, sample rate, or sensitivity is invalid."
                    )
                channel_count = len(channels) + 1
                if (
                    channel_count > limits.max_nodes
                    or 10 * channel_count > limits.max_attributes
                ):
                    raise ValueError("StationXML channels exceed resource limits.")
                channel_id = canonical_fingerprint(
                    {
                        "kind": "stationxml-channel",
                        "codes": values,
                        "position": position,
                        "orientation": orientation,
                        "sample_rate_hz": sample_rate,
                        "sensitivity": sensitivity,
                        "input_units": input_units,
                        "output_units": output_units,
                        "resource": resource.manifest.content_sha256,
                    }
                )
                channels.append(
                    StationChannelMetadata(
                        *values,
                        position,
                        orientation,
                        sample_rate,
                        sensitivity,
                        input_units,
                        output_units,
                        channel_id,
                    )
                )
    if not channels:
        raise ValueError("StationXML contains no channels.")
    identity = canonical_fingerprint(
        {
            "kind": "stationxml-metadata",
            "resource": resource.manifest.content_sha256,
            "coordinate_id": coordinates.coordinate_id,
            "channels": [item.channel_id for item in channels],
        }
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "StationXML",
        "StationXMLMetadata",
        source_id=resource.manifest.content_sha256,
        target_id=identity,
        source_profile=AdapterFormatProfile("StationXML"),
        preserved_fields=(
            "channel codes and positions",
            "component orientations",
            "sample rates",
            "instrument sensitivity summary",
            "exact response XML bytes",
        ),
        losses=(
            AdapterLoss(
                "network-station-response-detail",
                "import",
                "dropped",
                "The normalized view summarizes channels and sensitivity; exact response XML remains retained.",
                changes_interpretation=False,
            ),
        ),
        capabilities=(
            AdapterCapability("station-geometry"),
            AdapterCapability("component-orientation"),
            AdapterCapability("instrument-response-provenance"),
        ),
    )
    return StationXMLMetadata(tuple(channels), coordinates, resource, report, identity)


__all__ = [
    "QualifiedWaveformCollection",
    "QualifiedWaveformTrace",
    "SeismicFormatDependencyError",
    "StationChannelMetadata",
    "StationXMLMetadata",
    "read_miniseed3",
    "read_sac",
    "read_stationxml",
]
