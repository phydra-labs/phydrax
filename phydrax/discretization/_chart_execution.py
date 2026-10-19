#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bound native and host coefficient work to one chart request."""

import math
from collections.abc import Iterator
from contextlib import contextmanager, ExitStack

from .._meshcore import NativeBatchTicket, NativeExecutionLimits, NativeExecutionScope
from .._resource_ticket import HostBatchTicket
from ..typing import checked
from ._coordinate_budget import CoordinateEnclosureBudget


@contextmanager
@checked
def chart_native_execution(
    budget: CoordinateEnclosureBudget | None,
    scope: NativeExecutionScope | None,
    /,
    *,
    maximum_work: int,
    maximum_memory: int,
    maximum_queries: int,
) -> Iterator[tuple[CoordinateEnclosureBudget, NativeBatchTicket]]:
    """Own native chart work without changing an explicit host coefficient view."""
    with ExitStack() as stack:
        native_ticket: NativeBatchTicket | None = None
        native_work_upper = 2 * maximum_work
        if budget is not None:
            match budget.ticket:
                case NativeBatchTicket() as ticket:
                    native_ticket = ticket
                case HostBatchTicket():
                    native_work_upper = maximum_work
                case other:
                    raise TypeError(
                        "chart.budget.ticket must be a native or host batch ticket; "
                        f"got {type(other).__name__}."
                    )
        if native_ticket is None:
            if scope is None:
                scope = stack.enter_context(
                    NativeExecutionScope(
                        NativeExecutionLimits(
                            2 * maximum_work,
                            maximum_queries,
                            0,
                            maximum_memory,
                            math.inf,
                        )
                    )
                )
            remaining = scope.remaining()
            native_ticket = stack.enter_context(
                scope.batch(
                    min(native_work_upper, remaining.remaining_work_units),
                    queries_upper=min(
                        maximum_queries, remaining.remaining_geometry_queries
                    ),
                    bytes_upper=min(maximum_memory, remaining.remaining_managed_bytes),
                )
            )
        if budget is None:
            budget = CoordinateEnclosureBudget(
                native_ticket,
                maximum_work_units=maximum_work,
                maximum_memory_bytes=maximum_memory,
            )
        stack.enter_context(budget.activate())
        stack.enter_context(native_ticket.native_target())
        yield budget, native_ticket
