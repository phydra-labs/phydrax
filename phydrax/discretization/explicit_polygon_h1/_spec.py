#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ExplicitPolygonH1FieldSpec(StrictModule, NonTrainableState):
    """One point-value field over the explicit lowest-order polygon H1 space."""

    name: str = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    field_spec_id: str = eqx.field(static=True)

    def __init__(self, name: str, /, *, component_shape: Sequence[int] = ()):
        name_ = str(name)
        shape = tuple(int(value) for value in component_shape)
        if not name_:
            raise ValueError("Explicit polygon field name must be non-empty.")
        if any(value <= 0 for value in shape):
            raise ValueError("Explicit polygon component dimensions must be positive.")
        self.name = name_
        self.component_shape = shape
        self.field_spec_id = canonical_fingerprint(
            {
                "kind": "explicit-polygon-h1-field",
                "name": name_,
                "component_shape": list(shape),
            }
        )


class ExplicitPolygonH1QuadraturePolicy(StrictModule, NonTrainableState):
    """Fixed Gauss orders on each fan triangle and exterior edge."""

    cell_order: int = eqx.field(static=True)
    facet_order: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, *, cell_order: int = 3, facet_order: int = 3):
        cell = int(cell_order)
        facet = int(facet_order)
        if cell <= 0 or facet <= 0:
            raise ValueError("Explicit polygon quadrature orders must be positive.")
        self.cell_order = cell
        self.facet_order = facet
        self.policy_id = canonical_fingerprint(
            {
                "kind": "explicit-polygon-h1-quadrature",
                "cell_order": cell,
                "facet_order": facet,
            }
        )


class ExplicitPolygonH1QualificationPolicy(StrictModule, NonTrainableState):
    """Scale-aware local basis certification thresholds."""

    tolerance_multiplier: float = eqx.field(static=True)
    maximum_condition_number: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        tolerance_multiplier: float = 4096.0,
        maximum_condition_number: float = 1.0e12,
    ):
        multiplier = float(tolerance_multiplier)
        condition = float(maximum_condition_number)
        if (
            not isfinite(multiplier)
            or multiplier <= 0.0
            or not isfinite(condition)
            or condition <= 1.0
        ):
            raise ValueError(
                "Qualification multiplier and maximum condition must be finite and positive."
            )
        self.tolerance_multiplier = multiplier
        self.maximum_condition_number = condition
        self.policy_id = canonical_fingerprint(
            {
                "kind": "explicit-polygon-h1-qualification",
                "tolerance_multiplier": multiplier,
                "maximum_condition_number": condition,
            }
        )


class ExplicitPolygonH1ResourceBudget(StrictModule, NonTrainableState):
    """Bound retained basis data and preparation workspace."""

    maximum_cells: int = eqx.field(static=True)
    maximum_arity: int = eqx.field(static=True)
    maximum_retained_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    budget_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_cells: int = 1_000_000,
        maximum_arity: int = 256,
        maximum_retained_bytes: int = 1 << 30,
        maximum_workspace_bytes: int = 1 << 30,
    ):
        values = tuple(
            int(value)
            for value in (
                maximum_cells,
                maximum_arity,
                maximum_retained_bytes,
                maximum_workspace_bytes,
            )
        )
        if any(value <= 0 for value in values):
            raise ValueError("Explicit polygon resource budgets must be positive.")
        (
            self.maximum_cells,
            self.maximum_arity,
            self.maximum_retained_bytes,
            self.maximum_workspace_bytes,
        ) = values
        self.budget_id = canonical_fingerprint(
            {
                "kind": "explicit-polygon-h1-resource-budget",
                "maximum_cells": values[0],
                "maximum_arity": values[1],
                "maximum_retained_bytes": values[2],
                "maximum_workspace_bytes": values[3],
            }
        )


__all__ = [
    "ExplicitPolygonH1FieldSpec",
    "ExplicitPolygonH1QuadraturePolicy",
    "ExplicitPolygonH1QualificationPolicy",
    "ExplicitPolygonH1ResourceBudget",
]
