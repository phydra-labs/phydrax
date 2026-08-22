#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._tensor_support import PreparedTensorGrid
from ..linalg import AbstractLinearOperator, ArraySpace


ManufacturedNorm: TypeAlias = Literal["l2", "linf"]


class ManufacturedPDECase(StrictModule):
    """Exact scalar field, spatial action, and automatically derived forcing."""

    exact_solution: Callable[[Array, Array, Any], ArrayLike] = eqx.field(static=True)
    exact_spatial_action: Callable[[Array, Array, Any], ArrayLike] = eqx.field(
        static=True
    )
    case_id: str = eqx.field(static=True)

    def __init__(
        self,
        exact_solution: Callable[[Array, Array, Any], ArrayLike],
        exact_spatial_action: Callable[[Array, Array, Any], ArrayLike],
        /,
        *,
        case_id: str | None = None,
    ):
        if not callable(exact_solution) or not callable(exact_spatial_action):
            raise TypeError(
                "Manufactured exact solution and spatial action must be callable."
            )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "manufactured-pde-case",
                    "exact_solution": repr(exact_solution),
                    "exact_spatial_action": repr(exact_spatial_action),
                }
            )
            if case_id is None
            else str(case_id)
        )
        if not identifier:
            raise ValueError("case_id must be non-empty.")
        self.exact_solution = exact_solution
        self.exact_spatial_action = exact_spatial_action
        self.case_id = identifier

    def exact_state(
        self,
        grid: PreparedTensorGrid,
        time: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        value = jnp.asarray(self.exact_solution(jnp.asarray(time), grid.points, args))
        if value.shape == (grid.size,):
            value = value.reshape(grid.shape)
        if value.shape != grid.shape:
            raise ValueError(
                "Manufactured exact solution must return one scalar per grid entity."
            )
        return value

    def exact_spatial_state(
        self,
        grid: PreparedTensorGrid,
        time: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        value = jnp.asarray(
            self.exact_spatial_action(jnp.asarray(time), grid.points, args)
        )
        if value.shape == (grid.size,):
            value = value.reshape(grid.shape)
        if value.shape != grid.shape:
            raise ValueError(
                "Manufactured spatial action must return one scalar per grid entity."
            )
        return value

    def forcing(
        self,
        grid: PreparedTensorGrid,
        time: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        time_ = jnp.asarray(time)
        _, time_derivative = jax.jvp(
            lambda value: self.exact_state(grid, value, args),
            (time_,),
            (jnp.ones_like(time_),),
        )
        return time_derivative - self.exact_spatial_state(grid, time_, args)


class ManufacturedSpatialOperator(StrictModule):
    """One prepared same-layout scalar spatial action for convergence studies."""

    grid: PreparedTensorGrid
    operator: AbstractLinearOperator
    boundary_mask: Array
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        operator: AbstractLinearOperator,
        /,
        *,
        boundary_mask: ArrayLike | None = None,
        operator_id: str | None = None,
    ):
        if not isinstance(grid, PreparedTensorGrid) or not isinstance(
            operator, AbstractLinearOperator
        ):
            raise TypeError(
                "Manufactured operator requires a prepared grid and linear operator."
            )
        if not isinstance(operator.source, ArraySpace) or not isinstance(
            operator.target, ArraySpace
        ):
            raise TypeError("Manufactured operator requires array source/target spaces.")
        if operator.source.shape != grid.shape or operator.target.shape != grid.shape:
            raise ValueError("Manufactured operator must preserve the primary grid shape.")
        if boundary_mask is None:
            layout = grid.primary_entity_layout
            mask = jnp.zeros(grid.shape, dtype=bool)
            for lower, upper in zip(
                layout.lower_boundary_masks,
                layout.upper_boundary_masks,
                strict=True,
            ):
                mask = mask | lower | upper
        else:
            mask = jnp.asarray(boundary_mask, dtype=bool)
        if mask.shape != grid.shape:
            raise ValueError("Manufactured boundary mask must match the grid shape.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "manufactured-spatial-operator",
                    "grid": grid.prepared_id,
                    "operator": operator.operator_id,
                    "boundary_mask_shape": list(mask.shape),
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        self.grid = grid
        self.operator = operator
        self.boundary_mask = mask
        self.operator_id = identifier

    def apply(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.grid.shape:
            raise ValueError("Manufactured operator state must match the grid shape.")
        result = jnp.asarray(self.operator.mv(value))
        if result.shape != self.grid.shape:
            raise ValueError("Manufactured operator action must preserve the grid shape.")
        return result


class ManufacturedConvergenceResult(StrictModule, NonTrainableState):
    """Resolution-wise errors, observed rates, and explicit order verdicts."""

    resolutions: tuple[int, ...] = eqx.field(static=True)
    spacings: Array
    total_errors: Array
    interior_errors: Array
    boundary_errors: Array
    total_rates: Array
    interior_rates: Array
    boundary_rates: Array
    total_passed: bool | None = eqx.field(static=True)
    interior_passed: bool | None = eqx.field(static=True)
    boundary_passed: bool | None = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        resolutions: tuple[int, ...],
        spacings: ArrayLike,
        total_errors: ArrayLike,
        interior_errors: ArrayLike,
        boundary_errors: ArrayLike,
        total_rates: ArrayLike,
        interior_rates: ArrayLike,
        boundary_rates: ArrayLike,
        expected_total_order: float | None,
        expected_interior_order: float | None,
        expected_boundary_order: float | None,
        rate_tolerance: float,
        plan_id: str,
    ):
        spacing_ = jnp.asarray(spacings)
        total_ = jnp.asarray(total_errors)
        interior_ = jnp.asarray(interior_errors)
        boundary_ = jnp.asarray(boundary_errors)
        total_rates_ = jnp.asarray(total_rates)
        interior_rates_ = jnp.asarray(interior_rates)
        boundary_rates_ = jnp.asarray(boundary_rates)
        tolerance = float(rate_tolerance)

        def verdict(expected: float | None, rates: Array) -> bool | None:
            if expected is None:
                return None
            return bool(np.asarray(rates[-1]) >= float(expected) - tolerance)

        self.resolutions = resolutions
        self.spacings = spacing_
        self.total_errors = total_
        self.interior_errors = interior_
        self.boundary_errors = boundary_
        self.total_rates = total_rates_
        self.interior_rates = interior_rates_
        self.boundary_rates = boundary_rates_
        self.total_passed = verdict(expected_total_order, total_rates_)
        self.interior_passed = verdict(expected_interior_order, interior_rates_)
        self.boundary_passed = verdict(expected_boundary_order, boundary_rates_)
        self.result_id = canonical_fingerprint(
            {
                "kind": "manufactured-convergence-result",
                "plan": plan_id,
                "resolutions": list(resolutions),
                "total_passed": self.total_passed,
                "interior_passed": self.interior_passed,
                "boundary_passed": self.boundary_passed,
            }
        )


class ManufacturedConvergencePlan(StrictModule):
    """Prepare and measure one scalar spatial operator over a resolution sequence."""

    resolutions: tuple[int, ...] = eqx.field(static=True)
    prepare_operator: Callable[[int], ManufacturedSpatialOperator] = eqx.field(
        static=True
    )
    norm: ManufacturedNorm = eqx.field(static=True)
    expected_total_order: float | None = eqx.field(static=True)
    expected_interior_order: float | None = eqx.field(static=True)
    expected_boundary_order: float | None = eqx.field(static=True)
    rate_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        resolutions: Sequence[int],
        prepare_operator: Callable[[int], ManufacturedSpatialOperator],
        /,
        *,
        norm: ManufacturedNorm = "l2",
        expected_total_order: float | None = None,
        expected_interior_order: float | None = None,
        expected_boundary_order: float | None = None,
        rate_tolerance: float = 0.25,
        plan_id: str | None = None,
    ):
        values = tuple(int(value) for value in resolutions)
        if len(values) < 2 or any(value <= 0 for value in values):
            raise ValueError(
                "Convergence studies require at least two positive resolutions."
            )
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("Convergence resolutions must be strictly increasing.")
        if not callable(prepare_operator) or norm not in ("l2", "linf"):
            raise ValueError("Convergence operator factory/norm is invalid.")
        tolerance = float(rate_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("rate_tolerance must be finite and non-negative.")
        expected = (
            expected_total_order,
            expected_interior_order,
            expected_boundary_order,
        )
        if any(value is not None and float(value) <= 0.0 for value in expected):
            raise ValueError("Expected convergence orders must be positive or None.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "manufactured-convergence-plan",
                    "resolutions": list(values),
                    "prepare_operator": repr(prepare_operator),
                    "norm": norm,
                    "expected_orders": list(expected),
                    "rate_tolerance": tolerance,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.resolutions = values
        self.prepare_operator = prepare_operator
        self.norm = norm
        self.expected_total_order = expected_total_order
        self.expected_interior_order = expected_interior_order
        self.expected_boundary_order = expected_boundary_order
        self.rate_tolerance = tolerance
        self.plan_id = identifier

    def _norm(
        self,
        error: Array,
        weights: Array,
        mask: Array,
        /,
    ) -> Array:
        selected = jnp.where(mask, error, 0.0)
        if self.norm == "linf":
            return jnp.max(jnp.abs(selected))
        selected_weights = jnp.where(mask, weights, 0.0)
        mass = jnp.sum(selected_weights)
        return jnp.sqrt(jnp.sum(selected_weights * jnp.abs(selected) ** 2) / mass)

    def run(
        self,
        case: ManufacturedPDECase,
        /,
        *,
        time: ArrayLike = 0.0,
        args: Any = None,
    ) -> ManufacturedConvergenceResult:
        if not isinstance(case, ManufacturedPDECase):
            raise TypeError("case must be a ManufacturedPDECase.")
        spacings = []
        total_errors = []
        interior_errors = []
        boundary_errors = []
        for resolution in self.resolutions:
            prepared = self.prepare_operator(resolution)
            if not isinstance(prepared, ManufacturedSpatialOperator):
                raise TypeError(
                    "prepare_operator must return ManufacturedSpatialOperator."
                )
            grid = prepared.grid
            exact_state = case.exact_state(grid, time, args)
            exact_action = case.exact_spatial_state(grid, time, args)
            error = prepared.apply(exact_state) - exact_action
            weights = jnp.asarray(grid.quadrature_weights)
            boundary = prepared.boundary_mask
            interior = ~boundary
            if not bool(np.any(np.asarray(interior))) or not bool(
                np.any(np.asarray(boundary))
            ):
                raise ValueError(
                    "Convergence grid requires both interior and boundary rows."
                )
            total_errors.append(self._norm(error, weights, jnp.ones_like(boundary)))
            interior_errors.append(self._norm(error, weights, interior))
            boundary_errors.append(self._norm(error, weights, boundary))
            axis_spacings = [
                float(np.max(np.asarray(axis.interval_widths)))
                for axis in grid.structured_axes
            ]
            spacings.append(max(axis_spacings))
        spacing_array = jnp.asarray(spacings)
        if not bool(np.all(np.diff(np.asarray(spacing_array)) < 0.0)):
            raise ValueError("Prepared resolution spacing must decrease monotonically.")
        total_array = jnp.asarray(total_errors)
        interior_array = jnp.asarray(interior_errors)
        boundary_array = jnp.asarray(boundary_errors)

        def rates(errors: Array) -> Array:
            return jnp.log(errors[:-1] / errors[1:]) / jnp.log(
                spacing_array[:-1] / spacing_array[1:]
            )

        return ManufacturedConvergenceResult(
            resolutions=self.resolutions,
            spacings=spacing_array,
            total_errors=total_array,
            interior_errors=interior_array,
            boundary_errors=boundary_array,
            total_rates=rates(total_array),
            interior_rates=rates(interior_array),
            boundary_rates=rates(boundary_array),
            expected_total_order=self.expected_total_order,
            expected_interior_order=self.expected_interior_order,
            expected_boundary_order=self.expected_boundary_order,
            rate_tolerance=self.rate_tolerance,
            plan_id=self.plan_id,
        )


__all__ = [
    "ManufacturedConvergencePlan",
    "ManufacturedConvergenceResult",
    "ManufacturedNorm",
    "ManufacturedPDECase",
    "ManufacturedSpatialOperator",
]
