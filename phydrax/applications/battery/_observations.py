#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable, provider-neutral battery observation records.

The records in this module contain normalized arrays only.  They deliberately do
not fetch, parse, or otherwise execute a data source.  Ingestion code must attach
an existing local :class:`~phydrax.artifacts.ArtifactManifest` before the arrays
can cross this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import ArtifactManifest


CANONICAL_BATTERY_CHANNELS = ("current_a", "voltage_v", "temperature_k")
CANONICAL_BATTERY_UNITS = ("A", "V", "K")
CANONICAL_MASKED_FILL = 0.0


class BatteryCurrentSign(str, Enum):
    """Source-current convention prior to canonical passive-sign conversion."""

    PASSIVE = "positive-enters-positive-terminal"
    DISCHARGE_POSITIVE = "positive-during-discharge"


class BatteryRecordRole(str, Enum):
    """Top-level distinction between protocol traces and diagnostics."""

    PROTOCOL = "protocol"
    RPT = "rpt"
    EIS = "eis"


class BatteryDiagnosticRole(str, Enum):
    """Supported diagnostic record roles kept outside protocol time series."""

    RPT = "rpt"
    EIS = "eis"


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _identifiers(values: tuple[str, ...], name: str, /) -> tuple[str, ...]:
    resolved = tuple(_identifier(value, name) for value in values)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError(f"{name}s must be non-empty and unique.")
    return resolved


def _source_sign(value: BatteryCurrentSign | str, /) -> BatteryCurrentSign:
    if isinstance(value, BatteryCurrentSign):
        return value
    if not isinstance(value, str):
        raise TypeError("Battery source-current sign convention must be a string.")
    aliases = {
        "passive": BatteryCurrentSign.PASSIVE,
        "charge-positive": BatteryCurrentSign.PASSIVE,
        "discharge-positive": BatteryCurrentSign.DISCHARGE_POSITIVE,
        BatteryCurrentSign.PASSIVE.value: BatteryCurrentSign.PASSIVE,
        BatteryCurrentSign.DISCHARGE_POSITIVE.value: BatteryCurrentSign.DISCHARGE_POSITIVE,
    }
    if value not in aliases:
        raise ValueError("Unknown battery source-current sign convention.")
    return aliases[value]


def _local_manifest(value: ArtifactManifest, /) -> ArtifactManifest:
    if not isinstance(value, ArtifactManifest):
        raise TypeError("artifact_manifest must be an ArtifactManifest.")
    if not value.source_uri.startswith(("file:", "local:", "package:")):
        raise ValueError(
            "Battery observations require an already-local file:, local:, or package: manifest."
        )
    return value


def _host_vector(
    value: ArrayLike,
    name: str,
    /,
    *,
    dtype: np.dtype[Any] | type[Any] = np.float64,
    finite: bool = True,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a rank-one array.")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _host_matrix(
    value: ArrayLike,
    name: str,
    /,
    *,
    dtype: np.dtype[Any] | type[Any] = np.float64,
    finite: bool = True,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a rank-two array.")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


@dataclass(frozen=True, slots=True)
class BatterySourceUnits:
    """Units and current sign exactly as declared by a source resource."""

    time: str
    current: str
    voltage: str
    temperature: str
    current_sign: BatteryCurrentSign | str
    units_id: str = field(init=False)

    def __post_init__(self) -> None:
        time = _identifier(self.time, "source time unit")
        current = _identifier(self.current, "source current unit")
        voltage = _identifier(self.voltage, "source voltage unit")
        temperature = _identifier(self.temperature, "source temperature unit")
        sign = _source_sign(self.current_sign)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "voltage", voltage)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "current_sign", sign)
        object.__setattr__(
            self,
            "units_id",
            canonical_fingerprint(
                {
                    "kind": "battery-source-units",
                    "time": time,
                    "current": current,
                    "voltage": voltage,
                    "temperature": temperature,
                    "current_sign": sign.value,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class BatteryGapSegment:
    """One half-open contiguous sample range separated by declared source gaps."""

    start_index: int
    stop_index: int
    start_time_s: float
    stop_time_s: float
    preceding_gap_s: float | None
    segment_id: str = field(init=False)

    def __post_init__(self) -> None:
        start = int(self.start_index)
        stop = int(self.stop_index)
        start_time = float(self.start_time_s)
        stop_time = float(self.stop_time_s)
        preceding = None if self.preceding_gap_s is None else float(self.preceding_gap_s)
        if start < 0 or stop <= start:
            raise ValueError("A battery gap segment must be a non-empty half-open range.")
        if not np.isfinite(start_time) or not np.isfinite(stop_time):
            raise ValueError("Battery segment times must be finite.")
        if stop_time < start_time:
            raise ValueError("Battery segment stop_time_s must not precede start_time_s.")
        if preceding is not None and (not np.isfinite(preceding) or preceding <= 0.0):
            raise ValueError("preceding_gap_s must be finite and positive or None.")
        object.__setattr__(self, "start_index", start)
        object.__setattr__(self, "stop_index", stop)
        object.__setattr__(self, "start_time_s", start_time)
        object.__setattr__(self, "stop_time_s", stop_time)
        object.__setattr__(self, "preceding_gap_s", preceding)
        object.__setattr__(
            self,
            "segment_id",
            canonical_fingerprint(
                {
                    "kind": "battery-gap-segment",
                    "range": [start, stop],
                    "time_s": [start_time, stop_time],
                    "preceding_gap_s": preceding,
                }
            ),
        )


class BatteryTimeSeriesRecord(StrictModule, NonTrainableState):
    """Canonical SI protocol trace with immutable masks and explicit gaps."""

    time_s: Array
    current_a: Array
    voltage_v: Array
    temperature_k: Array
    current_mask: Array
    voltage_mask: Array
    temperature_mask: Array
    segment_indices: Array
    artifact_manifest: ArtifactManifest
    record_id: str = eqx.field(static=True)
    experiment_id: str = eqx.field(static=True)
    cell_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    resource_id: str = eqx.field(static=True)
    row_id_groups: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    raw_digest: str = eqx.field(static=True)
    rights_id: str = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    chemistry: str = eqx.field(static=True)
    form_factor: str = eqx.field(static=True)
    source_units: BatterySourceUnits = eqx.field(static=True)
    protocol_role: str = eqx.field(static=True)
    role: BatteryRecordRole = eqx.field(static=True)
    segments: tuple[BatteryGapSegment, ...] = eqx.field(static=True)
    preprocessing_id: str = eqx.field(static=True)
    content_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        record_id: str,
        experiment_id: str,
        cell_id: str,
        source_id: str,
        resource_id: str,
        row_id_groups: tuple[tuple[str, ...], ...],
        rights_id: str,
        chemistry: str,
        form_factor: str,
        source_units: BatterySourceUnits,
        protocol_role: str,
        artifact_manifest: ArtifactManifest,
        time_s: ArrayLike,
        current_a: ArrayLike,
        voltage_v: ArrayLike,
        temperature_k: ArrayLike,
        current_mask: ArrayLike,
        voltage_mask: ArrayLike,
        temperature_mask: ArrayLike,
        segment_indices: ArrayLike,
        segments: tuple[BatteryGapSegment, ...],
        preprocessing_id: str,
    ):
        identifiers = tuple(
            _identifier(value, name)
            for value, name in (
                (record_id, "record_id"),
                (experiment_id, "experiment_id"),
                (cell_id, "cell_id"),
                (source_id, "source_id"),
                (resource_id, "resource_id"),
                (rights_id, "rights_id"),
                (chemistry, "chemistry"),
                (form_factor, "form_factor"),
                (protocol_role, "protocol_role"),
                (preprocessing_id, "preprocessing_id"),
            )
        )
        if not isinstance(source_units, BatterySourceUnits):
            raise TypeError("source_units must be BatterySourceUnits.")
        manifest = _local_manifest(artifact_manifest)

        time = _host_vector(time_s, "time_s")
        current = _host_vector(current_a, "current_a")
        voltage = _host_vector(voltage_v, "voltage_v")
        temperature = _host_vector(temperature_k, "temperature_k")
        n_samples = time.size
        if n_samples == 0:
            raise ValueError("Battery time-series records require at least one sample.")
        if any(array.size != n_samples for array in (current, voltage, temperature)):
            raise ValueError(
                "Canonical battery channels must have the same length as time_s."
            )
        if n_samples > 1 and not np.all(np.diff(time) > 0.0):
            raise ValueError("Canonical battery time_s must be strictly increasing.")

        masks = tuple(
            _host_vector(value, name, dtype=np.bool_, finite=False)
            for value, name in (
                (current_mask, "current_mask"),
                (voltage_mask, "voltage_mask"),
                (temperature_mask, "temperature_mask"),
            )
        )
        if any(mask.size != n_samples for mask in masks):
            raise ValueError("Battery channel masks must have the same length as time_s.")
        for values, mask, name in zip(
            (current, voltage, temperature),
            masks,
            CANONICAL_BATTERY_CHANNELS,
            strict=True,
        ):
            if np.any(values[~mask] != CANONICAL_MASKED_FILL):
                raise ValueError(
                    f"Masked {name} entries must use the canonical zero fill."
                )
        if np.any(temperature[masks[2]] <= 0.0):
            raise ValueError("Retained canonical temperature_k values must be positive.")

        segment_array = _host_vector(
            segment_indices, "segment_indices", dtype=np.int64, finite=False
        )
        if segment_array.size != n_samples or np.any(segment_array < 0):
            raise ValueError(
                "segment_indices must provide one non-negative index per sample."
            )
        resolved_segments = tuple(segments)
        if not resolved_segments or any(
            not isinstance(segment, BatteryGapSegment) for segment in resolved_segments
        ):
            raise TypeError("segments must contain BatteryGapSegment values.")
        expected_start = 0
        for index, segment in enumerate(resolved_segments):
            if segment.start_index != expected_start or segment.stop_index > n_samples:
                raise ValueError(
                    "Battery gap segments must exhaustively and contiguously cover samples."
                )
            if not np.all(
                segment_array[segment.start_index : segment.stop_index] == index
            ):
                raise ValueError(
                    "segment_indices do not agree with the declared gap segments."
                )
            if (
                time[segment.start_index] != segment.start_time_s
                or time[segment.stop_index - 1] != segment.stop_time_s
            ):
                raise ValueError("Battery segment time bounds do not agree with time_s.")
            if index == 0 and segment.preceding_gap_s is not None:
                raise ValueError("The first battery segment cannot have a preceding gap.")
            if index > 0:
                actual_gap = float(
                    time[segment.start_index] - time[segment.start_index - 1]
                )
                if segment.preceding_gap_s != actual_gap:
                    raise ValueError(
                        "Battery segment preceding_gap_s does not match time_s."
                    )
            expected_start = segment.stop_index
        if expected_start != n_samples:
            raise ValueError("Battery gap segments must cover every sample exactly once.")

        groups = tuple(tuple(group) for group in row_id_groups)
        if len(groups) != n_samples:
            raise ValueError(
                "row_id_groups must provide one provenance group per sample."
            )
        flattened: list[str] = []
        for group in groups:
            resolved_group = _identifiers(group, "row_id")
            flattened.extend(resolved_group)
        if len(flattened) != len(set(flattened)):
            raise ValueError(
                "A source row ID cannot occur in more than one collapsed group."
            )

        payload = {
            "kind": "battery-si-time-series",
            "record_id": identifiers[0],
            "experiment_id": identifiers[1],
            "cell_id": identifiers[2],
            "source_id": identifiers[3],
            "resource_id": identifiers[4],
            "row_id_groups": groups,
            "raw_digest": manifest.sha256,
            "rights_id": identifiers[5],
            "license_id": manifest.license_id,
            "chemistry": identifiers[6],
            "form_factor": identifiers[7],
            "source_units_id": source_units.units_id,
            "protocol_role": identifiers[8],
            "preprocessing_id": identifiers[9],
            "artifact_manifest_id": manifest.manifest_id,
            "arrays": array_tree_fingerprint(
                {
                    "time_s": time,
                    "current_a": current,
                    "voltage_v": voltage,
                    "temperature_k": temperature,
                    "current_mask": masks[0],
                    "voltage_mask": masks[1],
                    "temperature_mask": masks[2],
                    "segment_indices": segment_array,
                }
            ),
            "segments": [segment.segment_id for segment in resolved_segments],
        }
        self.time_s = jnp.asarray(time)
        self.current_a = jnp.asarray(current)
        self.voltage_v = jnp.asarray(voltage)
        self.temperature_k = jnp.asarray(temperature)
        self.current_mask = jnp.asarray(masks[0])
        self.voltage_mask = jnp.asarray(masks[1])
        self.temperature_mask = jnp.asarray(masks[2])
        self.segment_indices = jnp.asarray(segment_array)
        self.artifact_manifest = manifest
        self.record_id = identifiers[0]
        self.experiment_id = identifiers[1]
        self.cell_id = identifiers[2]
        self.source_id = identifiers[3]
        self.resource_id = identifiers[4]
        self.row_id_groups = groups
        self.raw_digest = manifest.sha256
        self.rights_id = identifiers[5]
        self.license_id = manifest.license_id
        self.chemistry = identifiers[6]
        self.form_factor = identifiers[7]
        self.source_units = source_units
        self.protocol_role = identifiers[8]
        self.role = BatteryRecordRole.PROTOCOL
        self.segments = resolved_segments
        self.preprocessing_id = identifiers[9]
        self.content_fingerprint = canonical_fingerprint(payload)

    @property
    def row_ids(self) -> tuple[str, ...]:
        """All contributing source rows in canonical sample order."""

        return tuple(row_id for group in self.row_id_groups for row_id in group)

    @property
    def channel_masks(self) -> tuple[Array, Array, Array]:
        return self.current_mask, self.voltage_mask, self.temperature_mask

    @property
    def source_current_sign(self) -> BatteryCurrentSign:
        return self.source_units.current_sign

    @property
    def canonical_current_sign(self) -> BatteryCurrentSign:
        return BatteryCurrentSign.PASSIVE

    @property
    def channel_names(self) -> tuple[str, str, str]:
        return CANONICAL_BATTERY_CHANNELS

    @property
    def channel_units(self) -> tuple[str, str, str]:
        return CANONICAL_BATTERY_UNITS


class BatteryDiagnosticRecord(StrictModule, NonTrainableState):
    """Separate fixed-shape RPT or EIS diagnostic observation in declared SI units."""

    coordinate: Array
    values: Array
    valid_mask: Array
    artifact_manifest: ArtifactManifest
    record_id: str = eqx.field(static=True)
    experiment_id: str = eqx.field(static=True)
    cell_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    resource_id: str = eqx.field(static=True)
    row_ids: tuple[str, ...] = eqx.field(static=True)
    raw_digest: str = eqx.field(static=True)
    rights_id: str = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    chemistry: str = eqx.field(static=True)
    form_factor: str = eqx.field(static=True)
    diagnostic_role: BatteryDiagnosticRole = eqx.field(static=True)
    role: BatteryRecordRole = eqx.field(static=True)
    coordinate_name: str = eqx.field(static=True)
    coordinate_unit: str = eqx.field(static=True)
    channel_names: tuple[str, ...] = eqx.field(static=True)
    channel_units: tuple[str, ...] = eqx.field(static=True)
    source_coordinate_unit: str = eqx.field(static=True)
    source_channel_units: tuple[str, ...] = eqx.field(static=True)
    content_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        record_id: str,
        experiment_id: str,
        cell_id: str,
        source_id: str,
        resource_id: str,
        row_ids: tuple[str, ...],
        rights_id: str,
        chemistry: str,
        form_factor: str,
        diagnostic_role: BatteryDiagnosticRole | str,
        coordinate_name: str,
        coordinate_unit: str,
        channel_names: tuple[str, ...],
        channel_units: tuple[str, ...],
        source_coordinate_unit: str,
        source_channel_units: tuple[str, ...],
        artifact_manifest: ArtifactManifest,
        coordinate: ArrayLike,
        values: ArrayLike,
        valid_mask: ArrayLike,
    ):
        identifiers = tuple(
            _identifier(value, name)
            for value, name in (
                (record_id, "record_id"),
                (experiment_id, "experiment_id"),
                (cell_id, "cell_id"),
                (source_id, "source_id"),
                (resource_id, "resource_id"),
                (rights_id, "rights_id"),
                (chemistry, "chemistry"),
                (form_factor, "form_factor"),
                (coordinate_name, "coordinate_name"),
                (coordinate_unit, "coordinate_unit"),
                (source_coordinate_unit, "source_coordinate_unit"),
            )
        )
        if diagnostic_role not in (
            BatteryDiagnosticRole.RPT,
            BatteryDiagnosticRole.EIS,
            BatteryDiagnosticRole.RPT.value,
            BatteryDiagnosticRole.EIS.value,
        ):
            raise ValueError("diagnostic_role must be 'rpt' or 'eis'.")
        diagnostic = BatteryDiagnosticRole(diagnostic_role)
        channels = _identifiers(tuple(channel_names), "channel_name")
        units = tuple(_identifier(value, "channel_unit") for value in channel_units)
        source_units = tuple(
            _identifier(value, "source_channel_unit") for value in source_channel_units
        )
        if len(channels) != len(units) or len(channels) != len(source_units):
            raise ValueError(
                "Diagnostic channel names and source/canonical units must align."
            )
        rows = _identifiers(tuple(row_ids), "row_id")
        manifest = _local_manifest(artifact_manifest)
        coordinate_array = _host_vector(coordinate, "coordinate")
        values_array = _host_matrix(values, "values")
        mask_array = _host_matrix(valid_mask, "valid_mask", dtype=np.bool_, finite=False)
        if coordinate_array.size == 0 or values_array.shape != mask_array.shape:
            raise ValueError(
                "Diagnostic values and masks must have one non-empty common shape."
            )
        if values_array.shape != (coordinate_array.size, len(channels)):
            raise ValueError(
                "Diagnostic values must have shape (coordinate rows, channels)."
            )
        if len(rows) != coordinate_array.size:
            raise ValueError("Diagnostic row_ids must align one-to-one with coordinates.")
        if coordinate_array.size > 1 and not np.all(np.diff(coordinate_array) > 0.0):
            raise ValueError("Diagnostic coordinates must be strictly increasing.")
        if np.any(values_array[~mask_array] != CANONICAL_MASKED_FILL):
            raise ValueError("Masked diagnostic values must use the canonical zero fill.")

        payload = {
            "kind": "battery-diagnostic-record",
            "record_id": identifiers[0],
            "experiment_id": identifiers[1],
            "cell_id": identifiers[2],
            "source_id": identifiers[3],
            "resource_id": identifiers[4],
            "row_ids": rows,
            "raw_digest": manifest.sha256,
            "rights_id": identifiers[5],
            "license_id": manifest.license_id,
            "chemistry": identifiers[6],
            "form_factor": identifiers[7],
            "diagnostic_role": diagnostic.value,
            "coordinate": [identifiers[8], identifiers[9]],
            "channels": list(zip(channels, units, strict=True)),
            "source_units": [identifiers[10], *source_units],
            "artifact_manifest_id": manifest.manifest_id,
            "arrays": array_tree_fingerprint(
                {
                    "coordinate": coordinate_array,
                    "values": values_array,
                    "valid_mask": mask_array,
                }
            ),
        }
        self.coordinate = jnp.asarray(coordinate_array)
        self.values = jnp.asarray(values_array)
        self.valid_mask = jnp.asarray(mask_array)
        self.artifact_manifest = manifest
        self.record_id = identifiers[0]
        self.experiment_id = identifiers[1]
        self.cell_id = identifiers[2]
        self.source_id = identifiers[3]
        self.resource_id = identifiers[4]
        self.row_ids = rows
        self.raw_digest = manifest.sha256
        self.rights_id = identifiers[5]
        self.license_id = manifest.license_id
        self.chemistry = identifiers[6]
        self.form_factor = identifiers[7]
        self.diagnostic_role = diagnostic
        self.role = BatteryRecordRole(diagnostic.value)
        self.coordinate_name = identifiers[8]
        self.coordinate_unit = identifiers[9]
        self.channel_names = channels
        self.channel_units = units
        self.source_coordinate_unit = identifiers[10]
        self.source_channel_units = source_units
        self.content_fingerprint = canonical_fingerprint(payload)


class BatteryInterpolatedChannels(StrictModule):
    """JAX-ready masked evaluation of a canonical protocol trace."""

    time_s: Array
    current_a: Array
    voltage_v: Array
    temperature_k: Array
    in_support: Array
    current_mask: Array
    voltage_mask: Array
    temperature_mask: Array

    def __init__(
        self,
        time_s: Array,
        current_a: Array,
        voltage_v: Array,
        temperature_k: Array,
        in_support: Array,
        current_mask: Array,
        voltage_mask: Array,
        temperature_mask: Array,
        /,
    ):
        self.time_s = time_s
        self.current_a = current_a
        self.voltage_v = voltage_v
        self.temperature_k = temperature_k
        self.in_support = in_support
        self.current_mask = current_mask
        self.voltage_mask = voltage_mask
        self.temperature_mask = temperature_mask


def interpolate_battery_time_series(
    record: BatteryTimeSeriesRecord, query_time_s: ArrayLike, /
) -> BatteryInterpolatedChannels:
    """Linearly sample only within a declared segment, never across gaps or bounds."""

    if not isinstance(record, BatteryTimeSeriesRecord):
        raise TypeError("record must be a BatteryTimeSeriesRecord.")
    query = jnp.asarray(query_time_s, dtype=record.time_s.dtype)
    if query.ndim != 1:
        raise ValueError("query_time_s must be rank one.")
    count = record.time_s.shape[0]
    right_unclipped = jnp.searchsorted(record.time_s, query, side="right")
    left = jnp.clip(right_unclipped - 1, 0, count - 1)
    exact = query == record.time_s[left]
    right = jnp.where(exact, left, jnp.clip(right_unclipped, 0, count - 1))
    finite = jnp.isfinite(query)
    bounded = finite & (query >= record.time_s[0]) & (query <= record.time_s[-1])
    same_segment = record.segment_indices[left] == record.segment_indices[right]
    support = bounded & same_segment
    denominator = record.time_s[right] - record.time_s[left]
    safe_denominator = jnp.where(right == left, 1.0, denominator)
    fraction = jnp.where(
        right == left, 0.0, (query - record.time_s[left]) / safe_denominator
    )

    def interpolate_channel(values: Array, mask: Array) -> tuple[Array, Array]:
        valid = support & mask[left] & mask[right]
        interpolated = values[left] + fraction * (values[right] - values[left])
        return jnp.where(valid, interpolated, CANONICAL_MASKED_FILL), valid

    current, current_mask = interpolate_channel(record.current_a, record.current_mask)
    voltage, voltage_mask = interpolate_channel(record.voltage_v, record.voltage_mask)
    temperature, temperature_mask = interpolate_channel(
        record.temperature_k, record.temperature_mask
    )
    return BatteryInterpolatedChannels(
        query,
        current,
        voltage,
        temperature,
        support,
        current_mask,
        voltage_mask,
        temperature_mask,
    )


__all__ = [
    "BatteryCurrentSign",
    "BatteryDiagnosticRecord",
    "BatteryDiagnosticRole",
    "BatteryGapSegment",
    "BatteryInterpolatedChannels",
    "BatteryRecordRole",
    "BatterySourceUnits",
    "BatteryTimeSeriesRecord",
    "CANONICAL_BATTERY_CHANNELS",
    "CANONICAL_BATTERY_UNITS",
    "CANONICAL_MASKED_FILL",
    "interpolate_battery_time_series",
]
