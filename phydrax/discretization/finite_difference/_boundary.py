#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import AbstractLinearOperator
from ._stencil import StencilFootprint


BoundaryConditionKind: TypeAlias = Literal[
    "periodic",
    "dirichlet",
    "neumann",
    "robin",
    "ghost",
    "one_sided",
    "sbp_sat",
    "absorbing",
]
BoundaryRealizationKind: TypeAlias = Literal[
    "periodic",
    "ghost",
    "closure",
    "sat",
    "basis",
    "absorbing",
]


class AxisBoundaryPair(StrictModule, NonTrainableState):
    """Typed lower/upper physical boundary semantics for one tensor axis."""

    axis: str = eqx.field(static=True)
    lower: BoundaryConditionKind = eqx.field(static=True)
    upper: BoundaryConditionKind = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: str,
        lower: BoundaryConditionKind,
        upper: BoundaryConditionKind,
        /,
    ):
        axis_ = str(axis)
        allowed = (
            "periodic",
            "dirichlet",
            "neumann",
            "robin",
            "ghost",
            "one_sided",
            "sbp_sat",
            "absorbing",
        )
        if not axis_ or lower not in allowed or upper not in allowed:
            raise ValueError(
                "Axis boundary values must be recognized and axis non-empty."
            )
        if (lower == "periodic") != (upper == "periodic"):
            raise ValueError("Periodicity must be declared on both sides of an axis.")
        self.axis = axis_
        self.lower = lower
        self.upper = upper
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "axis-boundary-pair",
                "axis": axis_,
                "lower": lower,
                "upper": upper,
            }
        )


class BoundaryRealizationPlan(StrictModule, NonTrainableState):
    """Numerical realization selected for one typed physical boundary pair."""

    boundary: AxisBoundaryPair
    realization: BoundaryRealizationKind = eqx.field(static=True)
    lower_width: int = eqx.field(static=True)
    upper_width: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        boundary: AxisBoundaryPair,
        realization: BoundaryRealizationKind,
        /,
        *,
        lower_width: int = 0,
        upper_width: int = 0,
    ):
        if not isinstance(boundary, AxisBoundaryPair):
            raise TypeError("boundary must be an AxisBoundaryPair.")
        if realization not in (
            "periodic",
            "ghost",
            "closure",
            "sat",
            "basis",
            "absorbing",
        ):
            raise ValueError("Unknown boundary realization.")
        lower = int(lower_width)
        upper = int(upper_width)
        if lower < 0 or upper < 0:
            raise ValueError("Boundary widths must be non-negative.")
        if realization == "periodic" and (boundary.lower != "periodic" or lower or upper):
            raise ValueError(
                "Periodic realization requires periodic sides and zero ghosts."
            )
        if realization == "ghost" and lower + upper == 0:
            raise ValueError("Ghost realization requires at least one ghost value.")
        self.boundary = boundary
        self.realization = realization
        self.lower_width = lower
        self.upper_width = upper
        self.plan_id = canonical_fingerprint(
            {
                "kind": "boundary-realization",
                "boundary": boundary.boundary_id,
                "realization": realization,
                "lower_width": lower,
                "upper_width": upper,
            }
        )


class HaloPlan(StrictModule, NonTrainableState):
    """Aggregate physical and neighbor read reach for one stencil program."""

    axis_names: tuple[str, ...] = eqx.field(static=True)
    lower_widths: tuple[int, ...] = eqx.field(static=True)
    upper_widths: tuple[int, ...] = eqx.field(static=True)
    physical_boundaries: tuple[BoundaryRealizationPlan, ...]
    same_level_neighbors: bool = eqx.field(static=True)
    coarse_fine_neighbors: bool = eqx.field(static=True)
    distributed_neighbors: bool = eqx.field(static=True)
    halo_id: str = eqx.field(static=True)

    def __init__(
        self,
        footprint: StencilFootprint,
        /,
        *,
        physical_boundaries: Sequence[BoundaryRealizationPlan] = (),
        same_level_neighbors: bool = False,
        coarse_fine_neighbors: bool = False,
        distributed_neighbors: bool = False,
    ):
        if not isinstance(footprint, StencilFootprint):
            raise TypeError("footprint must be a StencilFootprint.")
        boundaries = tuple(physical_boundaries)
        if not all(isinstance(value, BoundaryRealizationPlan) for value in boundaries):
            raise TypeError("physical_boundaries must contain BoundaryRealizationPlan.")
        axes = tuple(value.boundary.axis for value in boundaries)
        if len(set(axes)) != len(axes) or any(
            axis not in footprint.axis_names for axis in axes
        ):
            raise ValueError("Physical boundary plans require unique footprint axes.")
        lower = list(footprint.lower)
        upper = list(footprint.upper)
        for value in boundaries:
            index = footprint.axis_names.index(value.boundary.axis)
            lower[index] = max(lower[index], value.lower_width)
            upper[index] = max(upper[index], value.upper_width)
        self.axis_names = footprint.axis_names
        self.lower_widths = tuple(lower)
        self.upper_widths = tuple(upper)
        self.physical_boundaries = boundaries
        self.same_level_neighbors = bool(same_level_neighbors)
        self.coarse_fine_neighbors = bool(coarse_fine_neighbors)
        self.distributed_neighbors = bool(distributed_neighbors)
        self.halo_id = canonical_fingerprint(
            {
                "kind": "halo-plan",
                "footprint": footprint.footprint_id,
                "lower": lower,
                "upper": upper,
                "physical": [value.plan_id for value in boundaries],
                "same_level": bool(same_level_neighbors),
                "coarse_fine": bool(coarse_fine_neighbors),
                "distributed": bool(distributed_neighbors),
            }
        )


class BoundaryAffineMap(StrictModule, NonTrainableState):
    """Boundary-data lift and RHS correction for a homogeneous core operator."""

    rhs_operator: AbstractLinearOperator
    lift_operator: AbstractLinearOperator
    boundary_id: str = eqx.field(static=True)
    affine_id: str = eqx.field(static=True)

    def __init__(
        self,
        rhs_operator: AbstractLinearOperator,
        lift_operator: AbstractLinearOperator,
        boundary_id: str,
        /,
    ):
        if not isinstance(rhs_operator, AbstractLinearOperator) or not isinstance(
            lift_operator, AbstractLinearOperator
        ):
            raise TypeError("Boundary affine maps require linear RHS and lift operators.")
        boundary = str(boundary_id)
        if not boundary:
            raise ValueError("boundary_id must be non-empty.")
        if not rhs_operator.source.compatible(lift_operator.source):
            raise ValueError("Boundary RHS and lift operators require one trace space.")
        self.rhs_operator = rhs_operator
        self.lift_operator = lift_operator
        self.boundary_id = boundary
        self.affine_id = canonical_fingerprint(
            {
                "kind": "boundary-affine-map",
                "rhs": rhs_operator.operator_id,
                "lift": lift_operator.operator_id,
                "boundary": boundary,
            }
        )

    def rhs(self, values: ArrayLike, /) -> Array:
        return self.rhs_operator.mv(values)

    def lift(self, values: ArrayLike, /) -> Array:
        return self.lift_operator.mv(values)


BoundaryValue: TypeAlias = Callable[[Array, Any], ArrayLike]


__all__ = [
    "AxisBoundaryPair",
    "BoundaryAffineMap",
    "BoundaryConditionKind",
    "BoundaryRealizationKind",
    "BoundaryRealizationPlan",
    "BoundaryValue",
    "HaloPlan",
]
