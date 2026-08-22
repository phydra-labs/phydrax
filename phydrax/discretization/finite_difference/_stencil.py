#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._tensor_support import GridLocation
from ._coefficients import StencilCoefficientPlan
from ._request import BoundaryClosureKind, DerivativeRequest


StencilRowKind: TypeAlias = Literal[
    "interior",
    "lower_closure",
    "upper_closure",
    "corner",
    "ghost",
    "interface",
]


class StencilFootprint(StrictModule, NonTrainableState):
    """Lower/upper integer read reach per tensor axis."""

    axis_names: tuple[str, ...] = eqx.field(static=True)
    lower: tuple[int, ...] = eqx.field(static=True)
    upper: tuple[int, ...] = eqx.field(static=True)
    footprint_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_names: Sequence[str],
        lower: Sequence[int],
        upper: Sequence[int],
        /,
    ):
        names = tuple(str(name) for name in axis_names)
        lower_ = tuple(int(value) for value in lower)
        upper_ = tuple(int(value) for value in upper)
        if (
            not names
            or len(lower_) != len(names)
            or len(upper_) != len(names)
            or any(value < 0 for value in lower_ + upper_)
        ):
            raise ValueError("Stencil footprint reaches must align and be non-negative.")
        self.axis_names = names
        self.lower = lower_
        self.upper = upper_
        self.footprint_id = canonical_fingerprint(
            {
                "kind": "stencil-footprint",
                "axis_names": list(names),
                "lower": list(lower_),
                "upper": list(upper_),
            }
        )

    def union(self, other: "StencilFootprint", /) -> "StencilFootprint":
        if not isinstance(other, StencilFootprint) or other.axis_names != self.axis_names:
            raise ValueError("Stencil footprints require aligned axes.")
        return StencilFootprint(
            self.axis_names,
            tuple(max(a, b) for a, b in zip(self.lower, other.lower, strict=True)),
            tuple(max(a, b) for a, b in zip(self.upper, other.upper, strict=True)),
        )


class StencilRowReport(StrictModule, NonTrainableState):
    """Observed width, order, conditioning, and kind for one target row."""

    kind: StencilRowKind = eqx.field(static=True)
    valid_width: int = eqx.field(static=True)
    derivative_order: int = eqx.field(static=True)
    achieved_accuracy_order: int = eqx.field(static=True)
    condition_estimate: float = eqx.field(static=True)
    coefficient_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: StencilRowKind,
        valid_width: int,
        coefficient_plan: StencilCoefficientPlan,
        /,
    ):
        if kind not in (
            "interior",
            "lower_closure",
            "upper_closure",
            "corner",
            "ghost",
            "interface",
        ):
            raise ValueError("Unknown stencil row kind.")
        width = int(valid_width)
        if width <= coefficient_plan.derivative_order:
            raise ValueError("Stencil row width must exceed derivative order.")
        self.kind = kind
        self.valid_width = width
        self.derivative_order = coefficient_plan.derivative_order
        self.achieved_accuracy_order = coefficient_plan.accuracy_order
        self.condition_estimate = coefficient_plan.condition_estimate
        self.coefficient_plan_id = coefficient_plan.plan_id


class LinearStencil(StrictModule, NonTrainableState):
    """Fixed-capacity masked gather bank from one entity layout to another."""

    request: DerivativeRequest
    axis_index: int = eqx.field(static=True)
    indices: Array
    weights: Array
    valid: Array
    row_kind_codes: Array
    row_reports: tuple[StencilRowReport, ...]
    coefficient_plans: tuple[StencilCoefficientPlan, ...]
    coefficient_plan_indices: Array
    source_location: GridLocation
    target_location: GridLocation
    footprint: StencilFootprint
    stencil_id: str = eqx.field(static=True)

    def __init__(
        self,
        request: DerivativeRequest,
        axis_index: int,
        indices: ArrayLike,
        weights: ArrayLike,
        coefficient_plans: Sequence[StencilCoefficientPlan],
        footprint: StencilFootprint,
        /,
        *,
        valid: ArrayLike | None = None,
        row_kinds: Sequence[StencilRowKind] | None = None,
        stencil_id: str | None = None,
    ):
        if not isinstance(request, DerivativeRequest):
            raise TypeError("request must be a DerivativeRequest.")
        axis = int(axis_index)
        if axis < 0 or axis >= len(request.source_location.axis_names):
            raise ValueError("axis_index is out of range.")
        indices_ = np.asarray(indices, dtype=np.int32)
        weights_ = np.asarray(weights)
        if indices_.ndim != 2 or weights_.shape != indices_.shape:
            raise ValueError("Stencil indices and weights must share rank-2 shape.")
        valid_ = (
            np.ones(indices_.shape, dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if valid_.shape != indices_.shape or np.any(indices_[valid_] < 0):
            raise ValueError(
                "Stencil validity must align and active indices be non-negative."
            )
        if np.any(~np.isfinite(weights_[valid_])):
            raise ValueError("Active stencil weights must be finite.")
        if np.any(np.count_nonzero(valid_, axis=1) == 0):
            raise ValueError(
                "Every stencil row requires at least one active coefficient."
            )
        plans = tuple(coefficient_plans)
        if len(plans) != indices_.shape[0] or not all(
            isinstance(plan, StencilCoefficientPlan) for plan in plans
        ):
            raise ValueError("One coefficient-plan reference is required per output.")
        kinds = (
            ("interior",) * indices_.shape[0] if row_kinds is None else tuple(row_kinds)
        )
        if len(kinds) != indices_.shape[0]:
            raise ValueError("row_kinds must contain one value per output row.")
        unique_by_id: dict[str, StencilCoefficientPlan] = {}
        for plan in plans:
            unique_by_id.setdefault(plan.plan_id, plan)
        unique_plans = tuple(unique_by_id.values())
        plan_lookup = {plan.plan_id: index for index, plan in enumerate(unique_plans)}
        plan_indices = np.asarray(
            [plan_lookup[plan.plan_id] for plan in plans],
            dtype=np.int32,
        )
        reports = tuple(
            StencilRowReport(kind, int(np.count_nonzero(row_valid)), plan)
            for kind, row_valid, plan in zip(kinds, valid_, plans, strict=True)
        )
        kind_values = (
            "interior",
            "lower_closure",
            "upper_closure",
            "corner",
            "ghost",
            "interface",
        )
        kind_codes = np.asarray(
            [kind_values.index(kind) for kind in kinds], dtype=np.int8
        )
        if not isinstance(footprint, StencilFootprint):
            raise TypeError("footprint must be a StencilFootprint.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "linear-stencil-bank",
                    "request": request.request_id,
                    "axis_index": axis,
                    "indices": array_tree_fingerprint(indices_),
                    "weights": array_tree_fingerprint(weights_),
                    "valid": array_tree_fingerprint(valid_),
                    "row_kinds": list(kinds),
                    "coefficient_plans": [plan.plan_id for plan in unique_plans],
                    "coefficient_plan_indices": array_tree_fingerprint(plan_indices),
                    "footprint": footprint.footprint_id,
                }
            )
            if stencil_id is None
            else str(stencil_id)
        )
        if not identifier:
            raise ValueError("stencil_id must be non-empty.")
        self.request = request
        self.axis_index = axis
        self.indices = jnp.asarray(indices_)
        self.weights = jnp.asarray(weights_)
        self.valid = jnp.asarray(valid_)
        self.row_kind_codes = jnp.asarray(kind_codes)
        self.row_reports = reports
        self.coefficient_plans = unique_plans
        self.coefficient_plan_indices = jnp.asarray(plan_indices)
        self.source_location = request.source_location
        self.target_location = request.target_location
        self.footprint = footprint
        self.stencil_id = identifier


class BoundaryStencilSet(StrictModule, NonTrainableState):
    """Interior/closure realization and observed row-level accuracy evidence."""

    stencil: LinearStencil
    kind: BoundaryClosureKind = eqx.field(static=True)
    interior_accuracy_order: int = eqx.field(static=True)
    closure_accuracy_order: int = eqx.field(static=True)
    minimum_accuracy_order: int = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        stencil: LinearStencil,
        /,
        *,
        kind: BoundaryClosureKind,
        interior_accuracy_order: int | None = None,
        closure_accuracy_order: int | None = None,
    ):
        if not isinstance(stencil, LinearStencil):
            raise TypeError("stencil must be a LinearStencil.")
        if kind not in ("periodic", "one_sided"):
            raise ValueError("Unknown boundary stencil kind.")
        interior_rows = [
            report.achieved_accuracy_order
            for report in stencil.row_reports
            if report.kind == "interior"
        ]
        closure_rows = [
            report.achieved_accuracy_order
            for report in stencil.row_reports
            if report.kind != "interior"
        ]
        interior = (
            min(interior_rows)
            if interior_accuracy_order is None and interior_rows
            else int(interior_accuracy_order or min(closure_rows))
        )
        closure = (
            min(closure_rows)
            if closure_accuracy_order is None and closure_rows
            else int(closure_accuracy_order or interior)
        )
        if interior <= 0 or closure <= 0:
            raise ValueError("Interior and closure accuracy must be positive.")
        self.stencil = stencil
        self.kind = kind
        self.interior_accuracy_order = interior
        self.closure_accuracy_order = closure
        self.minimum_accuracy_order = min(interior, closure)
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "boundary-stencil-set",
                "stencil": stencil.stencil_id,
                "boundary": kind,
                "interior_accuracy_order": interior,
                "closure_accuracy_order": closure,
            }
        )


__all__ = [
    "BoundaryStencilSet",
    "LinearStencil",
    "StencilFootprint",
    "StencilRowKind",
    "StencilRowReport",
]
