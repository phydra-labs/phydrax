# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Measured coordinate work translated into geometry-transition refusal evidence."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from . import _cell_geometry_transition_types as transition_types
from ._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)


@contextmanager
def charged_geometry_transition(
    policy: transition_types.CellGeometryTransitionPolicy,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Iterator[None]:
    """Charge algebraic coefficient-term work to the transition evaluation budget.

    Rational corner images and measures evaluate no quadrature points; the
    coordinate-enclosure ledger reserves each operation's term visits before it
    runs, so exhaustion refuses before the work is performed. The supplied
    ledger independently owns any producer-specific polynomial-storage limit.
    """
    try:
        with budget.measurement(), budget.activate():
            yield
    except CoordinateEnclosureResourceError as error:
        raise transition_types.CellGeometryTransitionError(
            "resource_limit",
            (
                "geometry_transition.evaluations must fit the evaluation budget; got excess basis evaluations"
                if error.resource == "coefficient_work"
                else f"geometry_transition.{error.resource} must fit the byte budget; got excess proposed bytes"
            ),
            measured=float(error.requested),
            limit=float(error.limit),
        ) from error
