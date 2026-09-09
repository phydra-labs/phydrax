#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic, provider-neutral preprocessing and leakage-safe battery splits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import ArtifactManifest
from ._observations import (
    _host_vector,
    _identifier,
    _identifiers,
    _local_manifest,
    BatteryCurrentSign,
    BatteryDiagnosticRecord,
    BatteryGapSegment,
    BatterySourceUnits,
    BatteryTimeSeriesRecord,
    CANONICAL_MASKED_FILL,
)


_TIME_FACTORS = {"s": 1.0, "min": 60.0, "h": 3600.0}
_CURRENT_FACTORS = {"A": 1.0, "mA": 1.0e-3, "uA": 1.0e-6, "µA": 1.0e-6}
_VOLTAGE_FACTORS = {"V": 1.0, "mV": 1.0e-3}
_TEMPERATURE_UNITS = frozenset(("K", "degC", "C", "°C", "degF", "F", "°F"))


def _readonly(array: np.ndarray, /) -> np.ndarray:
    array.setflags(write=False)
    return array


def _source_mask(
    value: ArrayLike | None, channel: np.ndarray, name: str, /
) -> np.ndarray:
    mask = (
        np.isfinite(channel)
        if value is None
        else _host_vector(value, name, dtype=np.bool_, finite=False)
    )
    if mask.shape != channel.shape:
        raise ValueError(f"{name} must have the same shape as its source channel.")
    if not np.all(np.isfinite(channel[mask])):
        raise ValueError(f"{name} marks non-finite source values as retained.")
    return _readonly(np.array(mask, dtype=np.bool_, copy=True))


@dataclass(frozen=True, slots=True)
class BatteryRawTimeSeries:
    """Immutable host-side rows plus explicit source lineage and conventions."""

    record_id: str
    experiment_id: str
    cell_id: str
    source_id: str
    resource_id: str
    row_ids: tuple[str, ...]
    rights_id: str
    chemistry: str
    form_factor: str
    protocol_role: str
    source_units: BatterySourceUnits
    artifact_manifest: ArtifactManifest
    time: np.ndarray
    current: np.ndarray
    voltage: np.ndarray
    temperature: np.ndarray
    current_mask: np.ndarray | None = None
    voltage_mask: np.ndarray | None = None
    temperature_mask: np.ndarray | None = None
    raw_digest: str = field(init=False)
    license_id: str = field(init=False)
    raw_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        names = (
            "record_id",
            "experiment_id",
            "cell_id",
            "source_id",
            "resource_id",
            "rights_id",
            "chemistry",
            "form_factor",
            "protocol_role",
        )
        resolved = tuple(
            _identifier(value, name)
            for value, name in zip(
                (
                    self.record_id,
                    self.experiment_id,
                    self.cell_id,
                    self.source_id,
                    self.resource_id,
                    self.rights_id,
                    self.chemistry,
                    self.form_factor,
                    self.protocol_role,
                ),
                names,
                strict=True,
            )
        )
        if not isinstance(self.source_units, BatterySourceUnits):
            raise TypeError("source_units must be BatterySourceUnits.")
        manifest = _local_manifest(self.artifact_manifest)
        time = _host_vector(self.time, "source time")
        current = _host_vector(self.current, "source current", finite=False)
        voltage = _host_vector(self.voltage, "source voltage", finite=False)
        temperature = _host_vector(self.temperature, "source temperature", finite=False)
        count = time.size
        if count == 0:
            raise ValueError("Raw battery time series require at least one row.")
        if any(array.size != count for array in (current, voltage, temperature)):
            raise ValueError("Raw battery source channels must have a common row count.")
        rows = _identifiers(tuple(self.row_ids), "row_id")
        if len(rows) != count:
            raise ValueError("row_ids must align one-to-one with raw source rows.")
        masks = (
            _source_mask(self.current_mask, current, "current_mask"),
            _source_mask(self.voltage_mask, voltage, "voltage_mask"),
            _source_mask(self.temperature_mask, temperature, "temperature_mask"),
        )
        source_arrays = tuple(
            _readonly(np.array(array, copy=True))
            for array in (time, current, voltage, temperature)
        )
        fingerprint = canonical_fingerprint(
            {
                "kind": "battery-raw-time-series",
                "identities": list(resolved),
                "row_ids": rows,
                "rights_id": resolved[5],
                "raw_digest": manifest.sha256,
                "license_id": manifest.license_id,
                "source_units_id": self.source_units.units_id,
                "artifact_manifest_id": manifest.manifest_id,
                "arrays": array_tree_fingerprint(
                    {
                        "time": source_arrays[0],
                        "current": source_arrays[1],
                        "voltage": source_arrays[2],
                        "temperature": source_arrays[3],
                        "current_mask": masks[0],
                        "voltage_mask": masks[1],
                        "temperature_mask": masks[2],
                    }
                ),
            }
        )
        for name, value in zip(names, resolved, strict=True):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "artifact_manifest", manifest)
        object.__setattr__(self, "time", source_arrays[0])
        object.__setattr__(self, "current", source_arrays[1])
        object.__setattr__(self, "voltage", source_arrays[2])
        object.__setattr__(self, "temperature", source_arrays[3])
        object.__setattr__(self, "current_mask", masks[0])
        object.__setattr__(self, "voltage_mask", masks[1])
        object.__setattr__(self, "temperature_mask", masks[2])
        object.__setattr__(self, "raw_digest", manifest.sha256)
        object.__setattr__(self, "license_id", manifest.license_id)
        object.__setattr__(self, "raw_fingerprint", fingerprint)

    @property
    def source_current_sign(self) -> BatteryCurrentSign:
        return self.source_units.current_sign


@dataclass(frozen=True, slots=True)
class BatteryDuplicateCriterion:
    """Per-channel SI tolerances governing whether duplicate rows may collapse."""

    current_absolute_a: float = 0.0
    voltage_absolute_v: float = 0.0
    temperature_absolute_k: float = 0.0
    relative: float = 0.0
    criterion_id: str = field(init=False)

    def __post_init__(self) -> None:
        values = tuple(
            float(value)
            for value in (
                self.current_absolute_a,
                self.voltage_absolute_v,
                self.temperature_absolute_k,
                self.relative,
            )
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Duplicate-collapse tolerances must be finite and non-negative."
            )
        object.__setattr__(self, "current_absolute_a", values[0])
        object.__setattr__(self, "voltage_absolute_v", values[1])
        object.__setattr__(self, "temperature_absolute_k", values[2])
        object.__setattr__(self, "relative", values[3])
        object.__setattr__(
            self,
            "criterion_id",
            canonical_fingerprint(
                {
                    "kind": "battery-duplicate-collapse-criterion",
                    "current_absolute_a": values[0],
                    "voltage_absolute_v": values[1],
                    "temperature_absolute_k": values[2],
                    "relative": values[3],
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class BatteryPreprocessingPolicy:
    """Complete deterministic policy for duplicate collapse and gap segmentation."""

    duplicate: BatteryDuplicateCriterion
    maximum_gap_s: float
    preprocessing_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.duplicate, BatteryDuplicateCriterion):
            raise TypeError("duplicate must be a BatteryDuplicateCriterion.")
        gap = float(self.maximum_gap_s)
        if not np.isfinite(gap) or gap <= 0.0:
            raise ValueError("maximum_gap_s must be finite and positive.")
        object.__setattr__(self, "maximum_gap_s", gap)
        object.__setattr__(
            self,
            "preprocessing_id",
            canonical_fingerprint(
                {
                    "kind": "battery-preprocessing-policy",
                    "duplicate_criterion_id": self.duplicate.criterion_id,
                    "maximum_gap_s": gap,
                    "canonical_units": ["s", "A", "V", "K"],
                    "canonical_current_sign": BatteryCurrentSign.PASSIVE.value,
                    "masked_fill": CANONICAL_MASKED_FILL,
                    "ordering": "stable-time-then-row-id",
                }
            ),
        )


def _factor(unit: str, factors: Mapping[str, float], quantity: str, /) -> float:
    if unit not in factors:
        allowed = ", ".join(sorted(factors))
        raise ValueError(
            f"Unsupported source {quantity} unit {unit!r}; expected one of {allowed}."
        )
    return factors[unit]


def _temperature_to_k(values: np.ndarray, unit: str, /) -> np.ndarray:
    if unit not in _TEMPERATURE_UNITS:
        allowed = ", ".join(sorted(_TEMPERATURE_UNITS))
        raise ValueError(
            f"Unsupported source temperature unit {unit!r}; expected one of {allowed}."
        )
    if unit == "K":
        return values
    if unit in ("degC", "C", "°C"):
        return values + 273.15
    return (values - 32.0) * (5.0 / 9.0) + 273.15


def _normalize_source(raw: BatteryRawTimeSeries, /) -> tuple[np.ndarray, ...]:
    units = raw.source_units
    time = raw.time * _factor(units.time, _TIME_FACTORS, "time")
    current = raw.current * _factor(units.current, _CURRENT_FACTORS, "current")
    if units.current_sign is BatteryCurrentSign.DISCHARGE_POSITIVE:
        current = -current
    voltage = raw.voltage * _factor(units.voltage, _VOLTAGE_FACTORS, "voltage")
    temperature = _temperature_to_k(raw.temperature, units.temperature)
    return time, current, voltage, temperature


def _collapse_channel(
    values: np.ndarray,
    mask: np.ndarray,
    indices: tuple[int, ...],
    *,
    absolute: float,
    relative: float,
    channel: str,
    timestamp_s: float,
) -> tuple[float, bool]:
    retained = np.asarray([values[index] for index in indices if mask[index]])
    if retained.size == 0:
        return CANONICAL_MASKED_FILL, False
    reference = retained[0]
    if not np.allclose(retained, reference, atol=absolute, rtol=relative):
        raise ValueError(
            f"Conflicting duplicate {channel} values at canonical time {timestamp_s:.17g} s."
        )
    return float(reference), True


def _gap_segments(
    time_s: np.ndarray, maximum_gap_s: float, /
) -> tuple[np.ndarray, tuple[BatteryGapSegment, ...]]:
    count = time_s.size
    starts = [0]
    for index in range(1, count):
        if time_s[index] - time_s[index - 1] > maximum_gap_s:
            starts.append(index)
    stops = [*starts[1:], count]
    segment_indices = np.empty(count, dtype=np.int64)
    segments: list[BatteryGapSegment] = []
    for segment_index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        segment_indices[start:stop] = segment_index
        preceding = None if start == 0 else float(time_s[start] - time_s[start - 1])
        segments.append(
            BatteryGapSegment(
                start,
                stop,
                float(time_s[start]),
                float(time_s[stop - 1]),
                preceding,
            )
        )
    return segment_indices, tuple(segments)


def preprocess_battery_time_series(
    raw: BatteryRawTimeSeries,
    policy: BatteryPreprocessingPolicy,
    /,
) -> BatteryTimeSeriesRecord:
    """Normalize, stably order, criterion-check duplicates, and expose gaps."""

    if not isinstance(raw, BatteryRawTimeSeries):
        raise TypeError("raw must be a BatteryRawTimeSeries.")
    if not isinstance(policy, BatteryPreprocessingPolicy):
        raise TypeError("policy must be a BatteryPreprocessingPolicy.")
    time, current, voltage, temperature = _normalize_source(raw)
    order = tuple(
        sorted(range(time.size), key=lambda index: (time[index], raw.row_ids[index]))
    )
    order_indices = np.asarray(order, dtype=np.int64)
    time = time[order_indices]
    current = current[order_indices]
    voltage = voltage[order_indices]
    temperature = temperature[order_indices]
    masks = tuple(
        np.asarray(mask)[order_indices]
        for mask in (raw.current_mask, raw.voltage_mask, raw.temperature_mask)
    )
    ordered_rows = tuple(raw.row_ids[index] for index in order)

    groups: list[tuple[int, ...]] = []
    start = 0
    while start < time.size:
        stop = start + 1
        while stop < time.size and time[stop] == time[start]:
            stop += 1
        groups.append(tuple(range(start, stop)))
        start = stop

    collapsed_time: list[float] = []
    collapsed_current: list[float] = []
    collapsed_voltage: list[float] = []
    collapsed_temperature: list[float] = []
    collapsed_masks: tuple[list[bool], list[bool], list[bool]] = ([], [], [])
    row_id_groups: list[tuple[str, ...]] = []
    absolutes = (
        policy.duplicate.current_absolute_a,
        policy.duplicate.voltage_absolute_v,
        policy.duplicate.temperature_absolute_k,
    )
    channels = (current, voltage, temperature)
    channel_names = ("current", "voltage", "temperature")
    destinations = (collapsed_current, collapsed_voltage, collapsed_temperature)
    for group in groups:
        timestamp = float(time[group[0]])
        collapsed_time.append(timestamp)
        row_id_groups.append(tuple(ordered_rows[index] for index in group))
        for values, mask, absolute, name, destination, destination_mask in zip(
            channels,
            masks,
            absolutes,
            channel_names,
            destinations,
            collapsed_masks,
            strict=True,
        ):
            value, valid = _collapse_channel(
                values,
                mask,
                group,
                absolute=absolute,
                relative=policy.duplicate.relative,
                channel=name,
                timestamp_s=timestamp,
            )
            destination.append(value)
            destination_mask.append(valid)

    canonical_time = np.asarray(collapsed_time, dtype=np.float64)
    canonical_channels = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (collapsed_current, collapsed_voltage, collapsed_temperature)
    )
    canonical_masks = tuple(np.asarray(mask, dtype=np.bool_) for mask in collapsed_masks)
    if canonical_time.size > 1 and not np.all(np.diff(canonical_time) > 0.0):
        raise ValueError("Duplicate collapse did not produce strictly increasing time.")
    segment_indices, segments = _gap_segments(canonical_time, policy.maximum_gap_s)
    return BatteryTimeSeriesRecord(
        record_id=raw.record_id,
        experiment_id=raw.experiment_id,
        cell_id=raw.cell_id,
        source_id=raw.source_id,
        resource_id=raw.resource_id,
        row_id_groups=tuple(row_id_groups),
        rights_id=raw.rights_id,
        chemistry=raw.chemistry,
        form_factor=raw.form_factor,
        source_units=raw.source_units,
        protocol_role=raw.protocol_role,
        artifact_manifest=raw.artifact_manifest,
        time_s=canonical_time,
        current_a=canonical_channels[0],
        voltage_v=canonical_channels[1],
        temperature_k=canonical_channels[2],
        current_mask=canonical_masks[0],
        voltage_mask=canonical_masks[1],
        temperature_mask=canonical_masks[2],
        segment_indices=segment_indices,
        segments=segments,
        preprocessing_id=policy.preprocessing_id,
    )


def _digest(value: str, name: str, /) -> str:
    digest = _identifier(value, name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 identity.")
    return digest


@dataclass(frozen=True, slots=True)
class BatteryPipelineIDs:
    """Hashed identities for every statistically consequential pipeline stage."""

    preprocessing_id: str
    normalization_id: str
    noise_model_id: str
    model_selection_id: str
    pipeline_id: str = field(init=False)

    def __post_init__(self) -> None:
        values = tuple(
            _digest(value, name)
            for value, name in (
                (self.preprocessing_id, "preprocessing_id"),
                (self.normalization_id, "normalization_id"),
                (self.noise_model_id, "noise_model_id"),
                (self.model_selection_id, "model_selection_id"),
            )
        )
        for name, value in zip(
            (
                "preprocessing_id",
                "normalization_id",
                "noise_model_id",
                "model_selection_id",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "pipeline_id",
            canonical_fingerprint(
                {
                    "kind": "battery-learning-pipeline-identities",
                    "preprocessing_id": values[0],
                    "normalization_id": values[1],
                    "noise_model_id": values[2],
                    "model_selection_id": values[3],
                }
            ),
        )

    @classmethod
    def from_specifications(
        cls,
        *,
        preprocessing_id: str,
        normalization: Any,
        noise_model: Any,
        model_selection: Any,
    ) -> BatteryPipelineIDs:
        """Hash JSON-compatible stage specifications without retaining mutable inputs."""

        return cls(
            preprocessing_id,
            canonical_fingerprint(
                {"kind": "battery-normalization-spec", "value": normalization}
            ),
            canonical_fingerprint(
                {"kind": "battery-noise-model-spec", "value": noise_model}
            ),
            canonical_fingerprint(
                {"kind": "battery-model-selection-spec", "value": model_selection}
            ),
        )


BatteryObservationRecord = BatteryTimeSeriesRecord | BatteryDiagnosticRecord


@dataclass(frozen=True, slots=True)
class BatteryRecordBinding:
    """Immutable content binding for one typed observation admitted to a split."""

    record_id: str
    cell_id: str
    content_fingerprint: str
    raw_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _identifier(self.record_id, "record_id"))
        object.__setattr__(self, "cell_id", _identifier(self.cell_id, "cell_id"))
        object.__setattr__(
            self,
            "content_fingerprint",
            _digest(self.content_fingerprint, "content_fingerprint"),
        )
        object.__setattr__(self, "raw_digest", _digest(self.raw_digest, "raw_digest"))

    @classmethod
    def from_record(cls, record: BatteryObservationRecord, /) -> BatteryRecordBinding:
        if not isinstance(record, (BatteryTimeSeriesRecord, BatteryDiagnosticRecord)):
            raise TypeError("Battery record bindings require a typed observation record.")
        return cls(
            record.record_id,
            record.cell_id,
            record.content_fingerprint,
            record.raw_digest,
        )


@dataclass(frozen=True, slots=True, init=False)
class BatteryGroupSplit:
    """Exhaustive disjoint record membership induced only by whole-cell groups."""

    train_cell_ids: tuple[str, ...]
    calibration_cell_ids: tuple[str, ...]
    test_cell_ids: tuple[str, ...]
    train_record_ids: tuple[str, ...]
    calibration_record_ids: tuple[str, ...]
    test_record_ids: tuple[str, ...]
    all_cell_ids: tuple[str, ...]
    all_record_ids: tuple[str, ...]
    time_series_record_ids: tuple[str, ...]
    diagnostic_record_ids: tuple[str, ...]
    record_bindings: tuple[BatteryRecordBinding, ...]
    preprocessing_id: str
    normalization_id: str
    noise_model_id: str
    model_selection_id: str
    split_id: str

    def __init__(
        self,
        records: Sequence[BatteryObservationRecord],
        /,
        *,
        train_cell_ids: Sequence[str],
        calibration_cell_ids: Sequence[str],
        test_cell_ids: Sequence[str],
        pipeline_ids: BatteryPipelineIDs,
    ):
        observations = tuple(records)
        if not observations:
            raise ValueError("Battery group splits require observation records.")
        if any(
            not isinstance(record, (BatteryTimeSeriesRecord, BatteryDiagnosticRecord))
            for record in observations
        ):
            raise TypeError(
                "Battery group splits accept battery observation records only."
            )
        if not isinstance(pipeline_ids, BatteryPipelineIDs):
            raise TypeError("pipeline_ids must be BatteryPipelineIDs.")
        record_ids = tuple(record.record_id for record in observations)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError(
                "Battery record_id values must be unique within a split corpus."
            )
        bindings = tuple(
            sorted(
                (BatteryRecordBinding.from_record(record) for record in observations),
                key=lambda binding: binding.record_id,
            )
        )
        cells = tuple(sorted({record.cell_id for record in observations}))
        partitions = tuple(
            tuple(sorted(_identifiers(tuple(values), "cell_id")))
            for values in (train_cell_ids, calibration_cell_ids, test_cell_ids)
        )
        flattened = tuple(cell for partition in partitions for cell in partition)
        if len(set(flattened)) != len(flattened):
            raise ValueError("Train, calibration, and test cell groups must be disjoint.")
        if set(flattened) != set(cells):
            raise ValueError(
                "Battery cell partitions must exhaustively cover the record corpus."
            )
        for record in observations:
            if (
                isinstance(record, BatteryTimeSeriesRecord)
                and record.preprocessing_id != pipeline_ids.preprocessing_id
            ):
                raise ValueError(
                    "A time-series preprocessing ID does not match the split pipeline."
                )

        cell_sets = tuple(set(partition) for partition in partitions)
        record_partitions = tuple(
            tuple(
                sorted(
                    record.record_id
                    for record in observations
                    if record.cell_id in cell_set
                )
            )
            for cell_set in cell_sets
        )
        all_records = tuple(sorted(record_ids))
        time_series_records = tuple(
            sorted(
                record.record_id
                for record in observations
                if isinstance(record, BatteryTimeSeriesRecord)
            )
        )
        diagnostic_records = tuple(
            sorted(
                record.record_id
                for record in observations
                if isinstance(record, BatteryDiagnosticRecord)
            )
        )
        split_id = canonical_fingerprint(
            {
                "kind": "battery-whole-cell-group-split",
                "train_cell_ids": partitions[0],
                "calibration_cell_ids": partitions[1],
                "test_cell_ids": partitions[2],
                "train_record_ids": record_partitions[0],
                "calibration_record_ids": record_partitions[1],
                "test_record_ids": record_partitions[2],
                "all_cell_ids": cells,
                "all_record_ids": all_records,
                "time_series_record_ids": time_series_records,
                "diagnostic_record_ids": diagnostic_records,
                "record_bindings": [
                    {
                        "record_id": binding.record_id,
                        "cell_id": binding.cell_id,
                        "content_fingerprint": binding.content_fingerprint,
                        "raw_digest": binding.raw_digest,
                    }
                    for binding in bindings
                ],
                "pipeline_id": pipeline_ids.pipeline_id,
            }
        )
        object.__setattr__(self, "train_cell_ids", partitions[0])
        object.__setattr__(self, "calibration_cell_ids", partitions[1])
        object.__setattr__(self, "test_cell_ids", partitions[2])
        object.__setattr__(self, "train_record_ids", record_partitions[0])
        object.__setattr__(self, "calibration_record_ids", record_partitions[1])
        object.__setattr__(self, "test_record_ids", record_partitions[2])
        object.__setattr__(self, "all_cell_ids", cells)
        object.__setattr__(self, "all_record_ids", all_records)
        object.__setattr__(self, "time_series_record_ids", time_series_records)
        object.__setattr__(self, "diagnostic_record_ids", diagnostic_records)
        object.__setattr__(self, "record_bindings", bindings)
        object.__setattr__(self, "preprocessing_id", pipeline_ids.preprocessing_id)
        object.__setattr__(self, "normalization_id", pipeline_ids.normalization_id)
        object.__setattr__(self, "noise_model_id", pipeline_ids.noise_model_id)
        object.__setattr__(self, "model_selection_id", pipeline_ids.model_selection_id)
        object.__setattr__(self, "split_id", split_id)

    @property
    def train_ids(self) -> tuple[str, ...]:
        return self.train_record_ids

    @property
    def calibration_ids(self) -> tuple[str, ...]:
        return self.calibration_record_ids

    @property
    def test_ids(self) -> tuple[str, ...]:
        return self.test_record_ids

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.all_record_ids

    def require_time_series_records(
        self, records: Sequence[BatteryTimeSeriesRecord], /
    ) -> None:
        """Refuse missing, additional, relabeled, or content-mutated fit records."""

        observations = tuple(records)
        if any(
            not isinstance(record, BatteryTimeSeriesRecord) for record in observations
        ):
            raise TypeError("Split verification requires battery time-series records.")
        actual = tuple(
            sorted(
                (BatteryRecordBinding.from_record(record) for record in observations),
                key=lambda binding: binding.record_id,
            )
        )
        expected_ids = set(self.time_series_record_ids)
        expected = tuple(
            binding
            for binding in self.record_bindings
            if binding.record_id in expected_ids
        )
        if actual != expected:
            raise ValueError(
                "Transformation records do not match the split's immutable content bindings."
            )


class TransformedBatteryTimeSeries(StrictModule):
    """JAX-ready masked channel matrix produced by one train-fitted transform."""

    time_s: Array
    values: Array
    valid_mask: Array
    segment_indices: Array
    record_id: str = eqx.field(static=True)
    transformation_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_s: Array,
        values: Array,
        valid_mask: Array,
        segment_indices: Array,
        /,
        *,
        record_id: str,
        transformation_id: str,
    ):
        self.time_s = time_s
        self.values = values
        self.valid_mask = valid_mask
        self.segment_indices = segment_indices
        self.record_id = record_id
        self.transformation_id = transformation_id


class BatteryChannelTransformation(StrictModule, NonTrainableState):
    """Masked channel standardization whose statistics are fit on train records only."""

    location: Array
    scale: Array
    split_id: str = eqx.field(static=True)
    preprocessing_id: str = eqx.field(static=True)
    normalization_id: str = eqx.field(static=True)
    training_record_ids: tuple[str, ...] = eqx.field(static=True)
    transformation_id: str = eqx.field(static=True)

    def __init__(
        self,
        location: ArrayLike,
        scale: ArrayLike,
        /,
        *,
        split_id: str,
        preprocessing_id: str,
        normalization_id: str,
        training_record_ids: tuple[str, ...],
    ):
        location_array = _host_vector(location, "location")
        scale_array = _host_vector(scale, "scale")
        if location_array.shape != (3,) or scale_array.shape != (3,):
            raise ValueError(
                "Battery channel transforms require three channel statistics."
            )
        if np.any(scale_array <= 0.0):
            raise ValueError("Battery channel transformation scales must be positive.")
        split_identity = _digest(split_id, "split_id")
        preprocessing_identity = _digest(preprocessing_id, "preprocessing_id")
        normalization_identity = _digest(normalization_id, "normalization_id")
        training = tuple(sorted(_identifiers(training_record_ids, "training_record_id")))
        transformation_id = canonical_fingerprint(
            {
                "kind": "battery-train-only-channel-transformation",
                "split_id": split_identity,
                "preprocessing_id": preprocessing_identity,
                "normalization_id": normalization_identity,
                "training_record_ids": training,
                "location": array_tree_fingerprint(location_array),
                "scale": array_tree_fingerprint(scale_array),
            }
        )
        self.location = jnp.asarray(location_array)
        self.scale = jnp.asarray(scale_array)
        self.split_id = split_identity
        self.preprocessing_id = preprocessing_identity
        self.normalization_id = normalization_identity
        self.training_record_ids = training
        self.transformation_id = transformation_id

    @classmethod
    def fit(
        cls,
        records: Sequence[BatteryTimeSeriesRecord],
        split: BatteryGroupSplit | None = None,
        /,
    ) -> BatteryChannelTransformation:
        """Fit masked moments after, and only after, an exhaustive group split exists."""

        if not isinstance(split, BatteryGroupSplit):
            raise TypeError(
                "A BatteryGroupSplit is required before fitting transformations."
            )
        observations = tuple(records)
        if any(
            not isinstance(record, BatteryTimeSeriesRecord) for record in observations
        ):
            raise TypeError(
                "Battery channel transformations accept time-series records only."
            )
        by_id = {record.record_id: record for record in observations}
        if len(by_id) != len(observations):
            raise ValueError(
                "Battery transformation records must have unique record IDs."
            )
        split.require_time_series_records(observations)
        training = tuple(
            by_id[record_id] for record_id in split.train_record_ids if record_id in by_id
        )
        if not training:
            raise ValueError("The group split contains no training time-series records.")
        if any(record.cell_id not in split.train_cell_ids for record in training):
            raise ValueError(
                "Training records must belong only to declared training cells."
            )
        if any(
            record.preprocessing_id != split.preprocessing_id for record in observations
        ):
            raise ValueError(
                "Transformation records must share the split preprocessing ID."
            )

        locations: list[float] = []
        scales: list[float] = []
        channel_groups = (
            tuple((record.current_a, record.current_mask) for record in training),
            tuple((record.voltage_v, record.voltage_mask) for record in training),
            tuple((record.temperature_k, record.temperature_mask) for record in training),
        )
        channel_names = ("current_a", "voltage_v", "temperature_k")
        for channel, value_name in zip(channel_groups, channel_names, strict=True):
            retained = np.concatenate(
                [np.asarray(values)[np.asarray(mask)] for values, mask in channel]
            )
            if retained.size == 0:
                raise ValueError(
                    f"Training records contain no retained {value_name} values."
                )
            location = float(np.mean(retained))
            scale = float(np.std(retained))
            locations.append(location)
            scales.append(scale if scale > 0.0 else 1.0)
        return cls(
            np.asarray(locations),
            np.asarray(scales),
            split_id=split.split_id,
            preprocessing_id=split.preprocessing_id,
            normalization_id=split.normalization_id,
            training_record_ids=tuple(record.record_id for record in training),
        )

    def apply(self, record: BatteryTimeSeriesRecord, /) -> TransformedBatteryTimeSeries:
        """Apply fixed train statistics while preserving masks and canonical fills."""

        if not isinstance(record, BatteryTimeSeriesRecord):
            raise TypeError("record must be a BatteryTimeSeriesRecord.")
        if record.preprocessing_id != self.preprocessing_id:
            raise ValueError(
                "Record preprocessing does not match the fitted transformation."
            )
        values = jnp.stack(
            (record.current_a, record.voltage_v, record.temperature_k), axis=-1
        )
        mask = jnp.stack(
            (record.current_mask, record.voltage_mask, record.temperature_mask), axis=-1
        )
        normalized = (values - self.location) / self.scale
        normalized = jnp.where(mask, normalized, CANONICAL_MASKED_FILL)
        return TransformedBatteryTimeSeries(
            record.time_s,
            normalized,
            mask,
            record.segment_indices,
            record_id=record.record_id,
            transformation_id=self.transformation_id,
        )


def fit_battery_transformation(
    records: Sequence[BatteryTimeSeriesRecord],
    split: BatteryGroupSplit | None = None,
    /,
) -> BatteryChannelTransformation:
    """Functional train-only fit API; ``split=None`` is an explicit refusal."""

    return BatteryChannelTransformation.fit(records, split)


__all__ = [
    "BatteryChannelTransformation",
    "BatteryDuplicateCriterion",
    "BatteryGroupSplit",
    "BatteryObservationRecord",
    "BatteryPipelineIDs",
    "BatteryPreprocessingPolicy",
    "BatteryRecordBinding",
    "BatteryRawTimeSeries",
    "TransformedBatteryTimeSeries",
    "fit_battery_transformation",
    "preprocess_battery_time_series",
]
