# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Resource admission and work accounting shared by mesh-family construction."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import jax
import numpy as np

from .._meshcore import NativeBatchTicket, NativeExecutionScope
from ..discretization import CellMesh
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..optim import MinimizationResult, ProjectedLBFGS
from ..typing import checked
from ._contracts import MeshingLimits
from ._host_banks import failure


@checked
def check_family_limits(
    limits: MeshingLimits,
    cells: int,
    vertices: int,
    entries: int,
    scratch: int,
    work: int,
    /,
) -> None:
    for name, value, bound in (
        ("cells", cells, limits.maximum_cells),
        ("vertices", vertices, limits.maximum_vertices),
        ("connectivity entries", entries, limits.maximum_connectivity_entries),
        ("scratch bytes", scratch, limits.maximum_scratch_bytes),
        ("work units", work, limits.maximum_work_units),
    ):
        if value > bound:
            raise failure(
                f"check_family_limits.capacity must fit {name} limit {bound}; got {value}.",
                resource=True,
            )


@contextmanager
@checked
def coordinate_phase(
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> Iterator[CoordinateEnclosureBudget]:
    """Bind one arithmetic-only phase to the remaining request envelope."""
    if not budget.ticket.settled:
        yield budget
        return
    remaining = scope.remaining()
    work_upper = min(
        budget.maximum_work_units - budget.work_units, remaining.remaining_work_units
    )
    bytes_upper = min(budget.maximum_memory_bytes, remaining.remaining_managed_bytes)
    ticket = scope.batch(
        work_upper,
        queries_upper=remaining.remaining_geometry_queries,
        bytes_upper=bytes_upper,
    )
    with budget.bind_ticket(ticket), ticket.native_target():
        yield budget


@checked
def count_geometry_queries(
    queries: int,
    /,
    *,
    work_units: int,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Admit and charge query work against the shared coordinate ticket."""

    ticket = budget.ticket
    if (
        work_units > ticket.remaining_work_units
        or queries > ticket.remaining_geometry_queries
    ):
        ticket.refuse(work_units, queries_upper=queries)
    ticket.consume(work_units, geometry_queries=queries)


@checked
def vertex_ids(source: CellMesh, added: int, /) -> np.ndarray:
    original = np.asarray(source.vertex_global_ids, dtype=np.int64)
    start = int(np.max(original)) + 1
    if start + added > np.iinfo(np.int64).max:
        raise failure(
            f"vertex_ids.range must fit int64 before allocation; got start={start}, added={added}.",
            resource=True,
        )
    return np.concatenate((original, np.arange(start, start + added, dtype=np.int64)))


@checked
def field_work_allowance(
    method: ProjectedLBFGS,
    rows: int,
    maximum_steps: int,
    /,
    *,
    scope: NativeExecutionScope,
) -> NativeBatchTicket:
    """Reserve the declared optimizer batch before its first evaluation."""
    # Each iteration has one value/gradient pair and at most the authored
    # Armijo trials. The optimizer owns one final value/gradient pair.
    bound = rows * (maximum_steps * (method.line_search.maximum_steps + 2) + 2)
    return scope.batch(bound)


@checked
def charge_field_work(
    ticket: NativeBatchTicket,
    result: MinimizationResult,
    rows: int,
    /,
) -> None:
    """Count optimizer diagnostics locally before candidate use."""
    if not result.diagnostics.counts_complete:
        raise RuntimeError(
            f"MinimizationResult.diagnostics.counts_complete must report all optimizer work; got {result.diagnostics.counts_complete}."
        )
    objectives, gradients = jax.device_get(
        (
            result.diagnostics.objective_evaluations,
            result.diagnostics.gradient_evaluations,
        )
    )
    ticket.consume(rows * (int(objectives) + int(gradients)))


__all__ = [
    "check_family_limits",
    "coordinate_phase",
    "count_geometry_queries",
    "vertex_ids",
    "field_work_allowance",
    "charge_field_work",
]
