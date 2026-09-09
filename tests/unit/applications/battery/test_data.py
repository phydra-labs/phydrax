#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.battery._data import (
    BatteryChannelTransformation,
    BatteryDuplicateCriterion,
    BatteryGroupSplit,
    BatteryPipelineIDs,
    BatteryPreprocessingPolicy,
    BatteryRawTimeSeries,
    BatteryRecordBinding,
    fit_battery_transformation,
    preprocess_battery_time_series,
)
from phydrax.applications.battery._observations import (
    BatteryDiagnosticRecord,
    BatteryDiagnosticRole,
    BatteryRecordRole,
    BatterySourceUnits,
    BatteryTimeSeriesRecord,
    interpolate_battery_time_series,
)
from phydrax.artifacts import ArtifactManifest


def _manifest(digit: str = "a") -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id=f"local-battery-{digit}",
        producer="unit-test-local-fixture",
        version="1",
        sha256=digit * 64,
        byte_size=0,
        source_uri=f"file:///fixtures/battery-{digit}.csv",
        license_id="CC-BY-4.0",
        model="battery-tabular-observation",
        coverage="synthetic-unit-test",
    )


def _policy(maximum_gap_s: float = 5.0) -> BatteryPreprocessingPolicy:
    return BatteryPreprocessingPolicy(BatteryDuplicateCriterion(), maximum_gap_s)


def _raw(
    record_id: str,
    cell_id: str,
    *,
    time=(0.0, 1.0, 2.0),
    current=(1.0, 2.0, 3.0),
    voltage=(3.0, 3.1, 3.2),
    temperature=(25.0, 26.0, 27.0),
    row_ids=None,
    source_units=None,
    current_mask=None,
    voltage_mask=None,
    temperature_mask=None,
    manifest=None,
) -> BatteryRawTimeSeries:
    rows = (
        tuple(f"{record_id}-row-{index}" for index in range(len(time)))
        if row_ids is None
        else tuple(row_ids)
    )
    return BatteryRawTimeSeries(
        record_id=record_id,
        experiment_id=f"experiment-{record_id}",
        cell_id=cell_id,
        source_id="local-test-source",
        resource_id=f"resource-{record_id}",
        row_ids=rows,
        rights_id="rights-research-local",
        chemistry="NMC811-graphite",
        form_factor="cylindrical-21700",
        protocol_role="cycling",
        source_units=(
            BatterySourceUnits("s", "A", "V", "degC", "passive")
            if source_units is None
            else source_units
        ),
        artifact_manifest=_manifest() if manifest is None else manifest,
        time=np.asarray(time),
        current=np.asarray(current),
        voltage=np.asarray(voltage),
        temperature=np.asarray(temperature),
        current_mask=current_mask,
        voltage_mask=voltage_mask,
        temperature_mask=temperature_mask,
    )


def _series(record_id: str, cell_id: str, *, offset: float = 0.0):
    return preprocess_battery_time_series(
        _raw(
            record_id,
            cell_id,
            current=(1.0 + offset, 2.0 + offset, 3.0 + offset),
            voltage=(3.0 + offset, 3.1 + offset, 3.2 + offset),
            temperature=(25.0 + offset, 26.0 + offset, 27.0 + offset),
        ),
        _policy(),
    )


def _pipelines(preprocessing_id: str) -> BatteryPipelineIDs:
    return BatteryPipelineIDs.from_specifications(
        preprocessing_id=preprocessing_id,
        normalization={"method": "masked-standardization", "version": 1},
        noise_model={"family": "independent-Gaussian", "version": 1},
        model_selection={"criterion": "cell-group-calibration", "version": 1},
    )


def test_preprocessing_stably_orders_collapses_and_normalizes_to_si_passive_sign():
    raw = _raw(
        "trace",
        "cell-a",
        time=(2.0, 1.0, 0.0, 1.0, 10.0),
        current=(3000.0, 2000.0, 1000.0, 2000.0, 4000.0),
        voltage=(3200.0, 3100.0, 3000.0, 3100.0, 3300.0),
        temperature=(80.6, 78.8, 77.0, 78.8, 82.4),
        row_ids=("row-e", "row-b", "row-a", "row-c", "row-f"),
        source_units=BatterySourceUnits("min", "mA", "mV", "degF", "discharge-positive"),
    )
    record = preprocess_battery_time_series(raw, _policy(maximum_gap_s=300.0))

    assert isinstance(record, BatteryTimeSeriesRecord)
    np.testing.assert_allclose(record.time_s, (0.0, 60.0, 120.0, 600.0))
    assert np.all(np.diff(np.asarray(record.time_s)) > 0.0)
    np.testing.assert_allclose(record.current_a, (-1.0, -2.0, -3.0, -4.0))
    np.testing.assert_allclose(record.voltage_v, (3.0, 3.1, 3.2, 3.3))
    np.testing.assert_allclose(record.temperature_k, (298.15, 299.15, 300.15, 301.15))
    assert record.row_id_groups == (
        ("row-a",),
        ("row-b", "row-c"),
        ("row-e",),
        ("row-f",),
    )
    assert record.role is BatteryRecordRole.PROTOCOL
    assert len(record.segments) == 2
    assert record.segments[1].preceding_gap_s == 480.0
    np.testing.assert_array_equal(record.segment_indices, (0, 0, 0, 1))


def test_duplicate_collapse_requires_every_retained_channel_to_agree():
    equal = _raw(
        "equal",
        "cell-a",
        time=(0.0, 0.0, 1.0),
        current=(1.0, 1.0, 2.0),
        voltage=(3.0, 3.0, 3.1),
        temperature=(25.0, 25.0, 26.0),
        row_ids=("row-b", "row-a", "row-c"),
    )
    record = preprocess_battery_time_series(equal, _policy())
    assert record.row_id_groups[0] == ("row-a", "row-b")
    np.testing.assert_allclose(record.time_s, (0.0, 1.0))

    conflict = _raw(
        "conflict",
        "cell-a",
        time=(0.0, 0.0, 1.0),
        current=(1.0, 1.0, 2.0),
        voltage=(3.0, 3.01, 3.1),
        temperature=(25.0, 25.0, 26.0),
        row_ids=("row-a", "row-b", "row-c"),
    )
    with pytest.raises(ValueError, match="Conflicting duplicate voltage"):
        preprocess_battery_time_series(conflict, _policy())

    tolerant = BatteryPreprocessingPolicy(
        BatteryDuplicateCriterion(voltage_absolute_v=0.02), 5.0
    )
    collapsed = preprocess_battery_time_series(conflict, tolerant)
    np.testing.assert_allclose(collapsed.voltage_v, (3.0, 3.1))


def test_masks_use_canonical_fills_and_interpolation_never_crosses_gaps_or_bounds():
    raw = _raw(
        "masked",
        "cell-a",
        time=(0.0, 1.0, 10.0),
        current=(1.0, np.nan, 3.0),
        voltage=(3.0, 3.2, 4.0),
        temperature=(25.0, 26.0, np.nan),
        current_mask=(True, False, True),
        voltage_mask=(True, True, True),
        temperature_mask=(True, True, False),
    )
    record = preprocess_battery_time_series(raw, _policy(maximum_gap_s=2.0))
    np.testing.assert_allclose(record.current_a, (1.0, 0.0, 3.0))
    np.testing.assert_allclose(record.temperature_k, (298.15, 299.15, 0.0))
    np.testing.assert_array_equal(record.current_mask, (True, False, True))
    np.testing.assert_array_equal(record.temperature_mask, (True, True, False))

    sampled = interpolate_battery_time_series(
        record, jnp.asarray((-1.0, 0.5, 5.0, 10.0, 11.0))
    )
    np.testing.assert_array_equal(sampled.in_support, (False, True, False, True, False))
    np.testing.assert_array_equal(
        sampled.current_mask, (False, False, False, True, False)
    )
    np.testing.assert_array_equal(sampled.voltage_mask, (False, True, False, True, False))
    np.testing.assert_allclose(sampled.voltage_v, (0.0, 3.1, 0.0, 4.0, 0.0))
    np.testing.assert_allclose(sampled.current_a, (0.0, 0.0, 0.0, 3.0, 0.0))


def test_rpt_and_eis_are_separate_diagnostic_records_with_local_rights_lineage():
    manifest = _manifest("b")
    common = dict(
        experiment_id="diagnostic-experiment",
        cell_id="cell-a",
        source_id="local-test-source",
        rights_id="rights-diagnostic-research",
        chemistry="LFP-graphite",
        form_factor="pouch",
        artifact_manifest=manifest,
    )
    rpt = BatteryDiagnosticRecord(
        record_id="rpt-1",
        resource_id="resource-rpt",
        row_ids=("rpt-row-0", "rpt-row-1"),
        diagnostic_role=BatteryDiagnosticRole.RPT,
        coordinate_name="time_s",
        coordinate_unit="s",
        channel_names=("capacity_c", "resistance_ohm"),
        channel_units=("C", "ohm"),
        source_coordinate_unit="s",
        source_channel_units=("C", "ohm"),
        coordinate=(0.0, 1.0),
        values=((3600.0, 0.02), (3590.0, 0.021)),
        valid_mask=((True, True), (True, True)),
        **common,
    )
    eis = BatteryDiagnosticRecord(
        record_id="eis-1",
        resource_id="resource-eis",
        row_ids=("eis-row-0", "eis-row-1"),
        diagnostic_role=BatteryDiagnosticRole.EIS,
        coordinate_name="frequency_hz",
        coordinate_unit="Hz",
        channel_names=("impedance_real_ohm", "impedance_imaginary_ohm"),
        channel_units=("ohm", "ohm"),
        source_coordinate_unit="Hz",
        source_channel_units=("ohm", "ohm"),
        coordinate=(1.0, 10.0),
        values=((0.02, -0.003), (0.018, -0.001)),
        valid_mask=((True, True), (True, True)),
        **common,
    )

    assert rpt.role is BatteryRecordRole.RPT
    assert eis.role is BatteryRecordRole.EIS
    assert rpt.diagnostic_role is BatteryDiagnosticRole.RPT
    assert eis.diagnostic_role is BatteryDiagnosticRole.EIS
    assert rpt.content_fingerprint != eis.content_fingerprint
    assert rpt.raw_digest == manifest.sha256
    assert rpt.license_id == manifest.license_id
    assert rpt.rights_id == "rights-diagnostic-research"
    assert rpt.artifact_manifest.source_uri.startswith("file:")


def test_normalized_content_fingerprint_is_deterministic_under_source_row_permutation():
    first = _raw(
        "stable",
        "cell-a",
        time=(1.0, 0.0, 1.0),
        current=(2.0, 1.0, 2.0),
        voltage=(3.1, 3.0, 3.1),
        temperature=(26.0, 25.0, 26.0),
        row_ids=("row-c", "row-a", "row-b"),
    )
    second = _raw(
        "stable",
        "cell-a",
        time=(1.0, 1.0, 0.0),
        current=(2.0, 2.0, 1.0),
        voltage=(3.1, 3.1, 3.0),
        temperature=(26.0, 26.0, 25.0),
        row_ids=("row-b", "row-c", "row-a"),
    )
    policy = _policy()
    normalized_first = preprocess_battery_time_series(first, policy)
    normalized_second = preprocess_battery_time_series(second, policy)

    assert normalized_first.content_fingerprint == normalized_second.content_fingerprint
    assert normalized_first.row_id_groups == normalized_second.row_id_groups
    assert policy.preprocessing_id == _policy().preprocessing_id


def test_whole_cell_split_is_disjoint_complete_and_binds_every_pipeline_identity():
    records = (
        _series("train-1", "cell-train"),
        _series("calibration-1", "cell-calibration"),
        _series("test-1", "cell-test"),
    )
    pipelines = _pipelines(records[0].preprocessing_id)
    split = BatteryGroupSplit(
        records,
        train_cell_ids=("cell-train",),
        calibration_cell_ids=("cell-calibration",),
        test_cell_ids=("cell-test",),
        pipeline_ids=pipelines,
    )

    assert set(split.all_record_ids) == {record.record_id for record in records}
    assert set(split.train_record_ids).isdisjoint(split.calibration_record_ids)
    assert set(split.train_record_ids).isdisjoint(split.test_record_ids)
    assert set(split.calibration_record_ids).isdisjoint(split.test_record_ids)
    assert all(
        isinstance(binding, BatteryRecordBinding) for binding in split.record_bindings
    )
    assert {
        (
            binding.record_id,
            binding.cell_id,
            binding.content_fingerprint,
            binding.raw_digest,
        )
        for binding in split.record_bindings
    } == {
        (
            record.record_id,
            record.cell_id,
            record.content_fingerprint,
            record.raw_digest,
        )
        for record in records
    }
    assert split.preprocessing_id == pipelines.preprocessing_id
    assert split.normalization_id == pipelines.normalization_id
    assert split.noise_model_id == pipelines.noise_model_id
    assert split.model_selection_id == pipelines.model_selection_id
    assert all(
        len(value) == 64
        for value in (
            split.split_id,
            split.preprocessing_id,
            split.normalization_id,
            split.noise_model_id,
            split.model_selection_id,
        )
    )

    with pytest.raises(ValueError, match="disjoint"):
        BatteryGroupSplit(
            records,
            train_cell_ids=("cell-train", "cell-calibration"),
            calibration_cell_ids=("cell-calibration",),
            test_cell_ids=("cell-test",),
            pipeline_ids=pipelines,
        )
    with pytest.raises(ValueError, match="exhaustively"):
        BatteryGroupSplit(
            records,
            train_cell_ids=("cell-train",),
            calibration_cell_ids=("cell-calibration",),
            test_cell_ids=("cell-other",),
            pipeline_ids=pipelines,
        )


def test_transformation_binds_corpus_then_fits_only_train_content():
    original = (
        _series("train-1", "cell-train", offset=0.0),
        _series("calibration-1", "cell-calibration", offset=1.0),
        _series("test-1", "cell-test", offset=2.0),
    )
    pipelines = _pipelines(original[0].preprocessing_id)
    split = BatteryGroupSplit(
        original,
        train_cell_ids=("cell-train",),
        calibration_cell_ids=("cell-calibration",),
        test_cell_ids=("cell-test",),
        pipeline_ids=pipelines,
    )
    with pytest.raises(TypeError, match="required before fitting"):
        fit_battery_transformation(original)

    fitted = BatteryChannelTransformation.fit(original, split)
    held_out_mutated = (
        original[0],
        _series("calibration-1", "cell-calibration", offset=1000.0),
        _series("test-1", "cell-test", offset=2000.0),
    )
    with pytest.raises(ValueError, match="immutable content bindings"):
        fit_battery_transformation(held_out_mutated, split)

    mutated_split = BatteryGroupSplit(
        held_out_mutated,
        train_cell_ids=("cell-train",),
        calibration_cell_ids=("cell-calibration",),
        test_cell_ids=("cell-test",),
        pipeline_ids=pipelines,
    )
    refitted = fit_battery_transformation(held_out_mutated, mutated_split)
    np.testing.assert_allclose(fitted.location, refitted.location)
    np.testing.assert_allclose(fitted.scale, refitted.scale)
    assert fitted.transformation_id != refitted.transformation_id
    assert fitted.training_record_ids == ("train-1",)

    relabeled_train_content = (
        _series("train-1", "cell-train", offset=1.0),
        original[1],
        original[2],
    )
    with pytest.raises(ValueError, match="immutable content bindings"):
        fit_battery_transformation(relabeled_train_content, split)

    transformed = fitted.apply(original[1])
    assert transformed.values.shape == (3, 3)
    np.testing.assert_array_equal(transformed.valid_mask, np.ones((3, 3), dtype=bool))


def test_canonical_arrays_and_sampling_are_jit_ready_and_records_are_immutable():
    record = _series("jit", "cell-train")
    sampled = jax.jit(lambda query: interpolate_battery_time_series(record, query))(
        jnp.asarray((0.25, 1.5))
    )
    np.testing.assert_array_equal(sampled.in_support, (True, True))

    other = (
        record,
        _series("jit-calibration", "cell-calibration"),
        _series("jit-test", "cell-test"),
    )
    split = BatteryGroupSplit(
        other,
        train_cell_ids=("cell-train",),
        calibration_cell_ids=("cell-calibration",),
        test_cell_ids=("cell-test",),
        pipeline_ids=_pipelines(record.preprocessing_id),
    )
    fitted = fit_battery_transformation(other, split)
    transformed = jax.jit(lambda transform, item: transform.apply(item))(fitted, record)
    assert transformed.values.shape == (3, 3)
    assert transformed.values.dtype == record.current_a.dtype
    with pytest.raises(AttributeError):
        record.rights_id = "changed"
    with pytest.raises(ValueError):
        _raw("immutable", "cell-a").time[0] = 10.0


def test_remote_manifests_are_refused_without_fetching_or_parsing_payloads():
    remote = ArtifactManifest(
        artifact_id="remote-battery",
        producer="unit-test",
        version="1",
        sha256="c" * 64,
        byte_size=123,
        source_uri="https://example.invalid/battery.csv",
        license_id="CC-BY-4.0",
        model="battery-table",
        coverage="none",
    )
    with pytest.raises(ValueError, match="already-local"):
        _raw("remote", "cell-a", manifest=remote)
