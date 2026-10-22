# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Measured coordinate work translated into geometry-transition refusal evidence."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from . import _cell_geometry_transition_types as transition_types
from ._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)


_active_transition: ContextVar[tuple[CoordinateEnclosureBudget, int] | None] = ContextVar(
    "geometry_transition_allowance", default=None
)


@contextmanager
def charged_geometry_transition(
    policy: transition_types.CellGeometryTransitionPolicy,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Iterator[None]:
    """Charge algebraic coefficient-term work to the transition evaluation budget.

    Rational corner images and measures evaluate no quadrature points; their
    algebraic term visits consume the same allowance. Nested producer phases
    share the outer transition's origin on the caller's unchanged ledger.
    Its independent request-work and polynomial-storage limits still apply.
    """
    active = _active_transition.get()
    start = (
        active[1]
        if active is not None and active[0] is budget
        else budget.work_units
    )
    if active is not None and active[0] is budget:
        # Keep the outer origin even when a producer splits its work into phases.
        # The outer owner translates refusals using that same phase-local count.
        with (
            budget.measurement(),
            budget.bound_stage(
                policy.maximum_evaluations,
                budget.maximum_memory_bytes,
                starting_work_units=start,
            ),
            budget.activate(),
        ):
            yield
        return
    token = _active_transition.set((budget, start))
    try:
        try:
            with (
                budget.measurement(),
                budget.bound_stage(
                    policy.maximum_evaluations,
                    budget.maximum_memory_bytes,
                    starting_work_units=start,
                ),
                budget.activate(),
            ):
                yield
        except (
            CoordinateEnclosureResourceError,
            transition_types.CellGeometryTransitionError,
        ) as error:
            if isinstance(error, CoordinateEnclosureResourceError):
                refusal = error
            else:
                # A nested consumer may already have translated its error.
                # Normalize only an inherited coordinate-work cap whose counters
                # that owner deliberately left cumulative; keep its own relative
                # limits and non-coordinate resource contracts unchanged.
                refusal = error.__cause__
                if (
                    not isinstance(refusal, CoordinateEnclosureResourceError)
                    or refusal.resource != "coefficient_work"
                    or error.limit != refusal.limit
                    or refusal.completed != budget.work_units
                ):
                    raise
            work = refusal.resource == "coefficient_work"
            # Exact source arithmetic also has its declared source-local cap;
            # only the unchanged request ledger's own counters carry this prefix.
            inherited_work = work and refusal.completed == budget.work_units
            raise transition_types.CellGeometryTransitionError(
                "resource_limit",
                (
                    "geometry_transition.evaluations must fit the evaluation budget; got excess basis evaluations"
                    if work
                    else f"geometry_transition.{refusal.resource} must fit the byte budget; got excess proposed bytes"
                ),
                measured=float(
                    refusal.requested - start if inherited_work else refusal.requested
                ),
                limit=float(
                    max(0, refusal.limit - start) if inherited_work else refusal.limit
                ),
            ) from error
    finally:
        _active_transition.reset(token)
