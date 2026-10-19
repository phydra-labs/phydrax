#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Validated capacities and pass checkpoints for metric remeshing."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import equinox as eqx
import numpy as np

from .._strict import StrictModule
from .._validation import nonnegative_integer
from ..typing import checked
from ._measurements import NativeMeshingPhaseRecorder


class MetricExecutionSchedule(StrictModule):
    """Retain the authored host capacities for one remeshing execution."""

    maximum_passes: int = eqx.field(static=True)
    maximum_vertices: int = eqx.field(static=True)
    maximum_cells: int = eqx.field(static=True)
    maximum_operations: int = eqx.field(static=True)
    maximum_work_units: int = eqx.field(static=True)
    maximum_cavity_cells: int = eqx.field(static=True)
    maximum_cavity_work: int = eqx.field(static=True)
    maximum_scratch_bytes: int = eqx.field(static=True)
    maximum_location_pairs: int = eqx.field(static=True)
    maximum_arc_work_units: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_passes: int | np.integer,
        maximum_vertices: int | np.integer,
        maximum_cells: int | np.integer,
        maximum_operations: int | np.integer,
        maximum_location_pairs: int | np.integer,
        maximum_work_units: int | np.integer | None = None,
        maximum_cavity_cells: int | np.integer | None = None,
        maximum_cavity_work: int | np.integer | None = None,
        maximum_scratch_bytes: int | np.integer | None = None,
        maximum_arc_work_units: int | np.integer,
    ) -> None:
        passes = nonnegative_integer(maximum_passes, "maximum_passes")
        vertices = nonnegative_integer(maximum_vertices, "maximum_vertices")
        cells = nonnegative_integer(maximum_cells, "maximum_cells")
        operations = nonnegative_integer(maximum_operations, "maximum_operations")
        work = (
            0
            if maximum_work_units is None
            else nonnegative_integer(maximum_work_units, "maximum_work_units")
        )
        pairs = nonnegative_integer(maximum_location_pairs, "maximum_location_pairs")
        cavity_cells = (
            0
            if maximum_cavity_cells is None
            else nonnegative_integer(maximum_cavity_cells, "maximum_cavity_cells")
        )
        cavity_work = (
            0
            if maximum_cavity_work is None
            else nonnegative_integer(maximum_cavity_work, "maximum_cavity_work")
        )
        scratch = (
            0
            if maximum_scratch_bytes is None
            else nonnegative_integer(maximum_scratch_bytes, "maximum_scratch_bytes")
        )
        arcs = nonnegative_integer(maximum_arc_work_units, "maximum_arc_work_units")
        self.maximum_passes = passes
        self.maximum_vertices = vertices
        self.maximum_cells = cells
        self.maximum_operations = operations
        self.maximum_location_pairs = pairs
        self.maximum_arc_work_units = arcs
        self.maximum_work_units = work
        self.maximum_cavity_cells = cavity_cells
        self.maximum_cavity_work = cavity_work
        self.maximum_scratch_bytes = scratch


def validate_metric_operation_controls(
    topology_operations: bool,
    relocation: bool,
    /,
    *,
    message: str = "metric_execution.operation_controls must use bool selectors; got nonboolean control",
) -> None:
    """Reject nonboolean execution selectors before numerical preparation."""
    if not isinstance(topology_operations, bool) or not isinstance(relocation, bool):
        raise TypeError(message)


def validate_metric_phase_recorder(
    recorder: NativeMeshingPhaseRecorder | None, /
) -> None:
    """Reject a noncallable host phase sink."""
    if recorder is not None and not callable(recorder):
        raise TypeError(
            "metric_execution.record_phase must be callable or None; got noncallable recorder"
        )


def validate_metric_quality_floor(
    minimum_quality: float,
    /,
    *,
    message: str = "metric_execution.minimum_quality must lie in [0, 1]; got invalid bound",
) -> None:
    """Reject an invalid dimensionless mean-ratio acceptance threshold."""
    if not 0.0 <= minimum_quality <= 1.0:
        raise ValueError(message)


@checked
def metric_execution_passes(
    schedule: MetricExecutionSchedule,
    checkpoint: Callable[[], None],
    /,
) -> Iterator[int]:
    """Visit authored passes, checking the phase ledger before each measurement."""
    for index in range(1, schedule.maximum_passes + 1):
        checkpoint()
        yield index


class MetricOperationCounts(StrictModule):
    """Name accepted topology and relocation counts in evidence order."""

    splits: int = eqx.field(static=True, default=0)
    collapses: int = eqx.field(static=True, default=0)
    reconnections: int = eqx.field(static=True, default=0)
    relocations: int = eqx.field(static=True, default=0)

    def __init__(
        self,
        splits: int | np.integer = 0,
        collapses: int | np.integer = 0,
        reconnections: int | np.integer = 0,
        relocations: int | np.integer = 0,
    ) -> None:
        accepted_splits = nonnegative_integer(splits, "counts.splits")
        accepted_collapses = nonnegative_integer(collapses, "counts.collapses")
        accepted_reconnections = nonnegative_integer(
            reconnections, "counts.reconnections"
        )
        accepted_relocations = nonnegative_integer(relocations, "counts.relocations")
        self.splits = accepted_splits
        self.collapses = accepted_collapses
        self.reconnections = accepted_reconnections
        self.relocations = accepted_relocations

    def evidence_counts(self) -> tuple[int, int, int, int]:
        return self.splits, self.collapses, self.reconnections, self.relocations
