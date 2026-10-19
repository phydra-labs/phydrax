#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host capacity normalization and pass checkpoint contracts."""

from dataclasses import replace

import numpy as np
import pytest

from phydrax.meshing._metric_execution import (
    metric_execution_passes,
    MetricExecutionSchedule,
    MetricOperationCounts,
    validate_metric_operation_controls,
    validate_metric_phase_recorder,
    validate_metric_quality_floor,
)
from tests._support.sphere_material import accepted_sphere_material


def _schedule(passes: int) -> MetricExecutionSchedule:
    return MetricExecutionSchedule(
        maximum_passes=passes,
        maximum_vertices=8,
        maximum_cells=12,
        maximum_operations=3,
        maximum_location_pairs=24,
        maximum_arc_work_units=32,
    )


def test_schedule_normalizes_native_integer_capacities() -> None:
    schedule = MetricExecutionSchedule(
        maximum_passes=np.int64(2),
        maximum_vertices=np.int64(8),
        maximum_cells=np.int64(12),
        maximum_operations=np.int64(3),
        maximum_work_units=np.int64(48),
        maximum_location_pairs=np.int64(24),
        maximum_cavity_cells=np.int64(4),
        maximum_cavity_work=np.int64(16),
        maximum_scratch_bytes=np.int64(1024),
        maximum_arc_work_units=np.int64(32),
    )
    assert schedule.maximum_passes == 2
    assert schedule.maximum_work_units == 48
    assert schedule.maximum_cavity_cells == 4
    assert schedule.maximum_cavity_work == 16
    assert schedule.maximum_scratch_bytes == 1024
    assert type(schedule.maximum_passes) is int
    assert type(schedule.maximum_work_units) is int


def test_schedule_validation_keeps_capacity_order() -> None:
    with pytest.raises(ValueError, match="maximum_work_units"):
        MetricExecutionSchedule(
            maximum_passes=2,
            maximum_vertices=8,
            maximum_cells=12,
            maximum_operations=3,
            maximum_work_units=-1,
            maximum_location_pairs=-1,
            maximum_cavity_cells=-1,
            maximum_cavity_work=-1,
            maximum_scratch_bytes=-1,
            maximum_arc_work_units=-1,
        )


@pytest.mark.parametrize("passes", [0, 3], ids=["zero-pass", "three-pass"])
def test_driver_checkpoints_before_every_authored_pass(passes: int) -> None:
    events: list[tuple[str, int]] = []

    def checkpoint() -> None:
        events.append(("checkpoint", len(events) // 2 + 1))

    for index in metric_execution_passes(_schedule(passes), checkpoint):
        events.append(("pass", index))
    assert events == [
        event
        for index in range(1, passes + 1)
        for event in (("checkpoint", index), ("pass", index))
    ]


def test_operation_counts_preserve_evidence_slot_order_without_mutation() -> None:
    original = MetricOperationCounts()
    changed = replace(original, splits=2, collapses=3, reconnections=5, relocations=7)
    assert original.evidence_counts() == (0, 0, 0, 0)
    assert changed.evidence_counts() == (2, 3, 5, 7)


def test_operation_controls_reject_nonboolean_selector() -> None:
    with pytest.raises(TypeError, match="metric_execution.operation_controls"):
        validate_metric_operation_controls(1, False)  # ty: ignore[invalid-argument-type]


def test_phase_recorder_rejects_noncallable_sink() -> None:
    with pytest.raises(TypeError, match="metric_execution.record_phase"):
        validate_metric_phase_recorder(1)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "floor",
    [-0.01, 1.01, np.nan, np.inf],
    ids=["negative", "above-one", "nan", "infinite"],
)
def test_quality_floor_rejects_out_of_interval_values(floor: float) -> None:
    with pytest.raises(ValueError, match="metric_execution.minimum_quality"):
        validate_metric_quality_floor(floor)


def test_operation_count_record_rejects_negative_accepted_count() -> None:
    with pytest.raises(ValueError, match="counts.splits"):
        MetricOperationCounts(splits=-1)


def test_shared_sphere_preparation_rejects_nonboolean_reuse_selector() -> None:
    with pytest.raises(TypeError, match="sphere_material.reuse_preparation"):
        accepted_sphere_material(0.8, reuse_preparation=1)  # ty: ignore[invalid-argument-type]
