"""Input-derived host reservations for standalone source enclosure requests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from fractions import Fraction

import jax
import numpy as np

from ..._resource_ticket import HostBatchTicket
from ...discretization._coordinate_budget import CoordinateEnclosureBudget


@contextmanager
def coordinate_source_request(
    inputs: tuple[object, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
    rows: int = 1,
    order: int = 3,
    queries_upper: int = 0,
) -> Iterator[CoordinateEnclosureBudget]:
    """Borrow a supplied view or reserve a standalone source's shape envelope.

    Let M count all source coefficients, knots and scalar inputs (at least one).
    Each degree and knot-span count is at most M. At most M squared Boehm
    insertions retain at most M squared controls per axis; a tensor bank thus
    has at most M to the fourth homogeneous coefficients. The three refinement
    streams cost at most 3 M to the fifth coefficient visits. Each queried row
    restricts at most (order + 1) squared derivative banks, with two coefficient
    streams, two de Casteljau triangles, and fourteen quotient/jet lanes.
    Storage uses the existing 8192-byte source-transfer and 4096-byte row
    overheads, eight-byte numerical lanes, and the source's declared rational
    bit envelope. This is a reservation, not a scientific stopping limit.
    """
    if budget is not None:
        yield budget
        return
    measure = 1
    bits = 2200  # Binary64 endpoint normalization uses this existing bit bound.
    pending: list[object] = list(inputs)
    seen: set[int] = set()
    while pending:
        leaf = pending.pop()
        if id(leaf) in seen:
            continue
        seen.add(id(leaf))
        if isinstance(leaf, (jax.Array, np.ndarray)):
            measure += leaf.size
        elif isinstance(leaf, Fraction):
            measure += 1
            bits = max(bits, leaf.numerator.bit_length(), leaf.denominator.bit_length())
        elif isinstance(leaf, int):
            # Static degrees and declared iteration maxima are request inputs.
            measure += max(1, abs(leaf))
        elif is_dataclass(leaf) and not isinstance(leaf, type):
            pending.extend(getattr(leaf, item.name) for item in fields(leaf))
        elif isinstance(leaf, (tuple, list)):
            pending.extend(leaf)
        elif isinstance(leaf, dict):
            pending.extend(leaf.values())
        else:
            measure += 1
    banks = (order + 1) ** 2
    coefficients = measure**4
    work_upper = (
        measure
        + 3 * measure**5
        + (rows + 1) * banks * (2 * measure**2 + 14 * measure) * (coefficients + 1)
    )
    rational_bytes = 512 + 2 * ((bits * coefficients + 7) // 8)
    bytes_upper = (
        8192
        + 4096 * (rows + 1)
        + (32 * 8 + rational_bytes) * banks * (coefficients + 1) * (rows + 1)
    )
    with HostBatchTicket(
        work_upper, bytes_upper=bytes_upper, queries_upper=queries_upper
    ) as ticket:
        owner = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=work_upper,
            maximum_memory_bytes=bytes_upper,
        )
        with owner.activate():
            yield owner
