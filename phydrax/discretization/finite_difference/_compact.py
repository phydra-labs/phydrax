#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    BasePlusLowRankLinearOperator,
    LowRankSolvePolicy,
    OperatorCapabilities,
    OperatorProperties,
    prepare_low_rank_solve,
    PreparedLowRankSolve,
    RHSLayout,
    solve_low_rank,
    TridiagonalLinearOperator,
)
from .._tensor_support import GridLocation, PreparedTensorGrid
from ._request import DerivativeRequest


CompactOperatorKind: TypeAlias = Literal["derivative", "interpolation"]


class CompactOperatorReport(StrictModule, NonTrainableState):
    """Consistency, spectral, conditioning, and resource evidence."""

    kind: CompactOperatorKind = eqx.field(static=True)
    derivative_order: int = eqx.field(static=True)
    accuracy_order: int = eqx.field(static=True)
    axis: str = eqx.field(static=True)
    implicit_alpha: float = eqx.field(static=True)
    explicit_width: int = eqx.field(static=True)
    moment_residual: float = eqx.field(static=True)
    condition_estimate: float = eqx.field(static=True)
    modified_symbol_error: float = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    solve_workspace_bytes_per_rhs: int = eqx.field(static=True)
    dense_materialization_entries: int = eqx.field(static=True)
    requires_unsharded_axis: bool = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        kind: CompactOperatorKind,
        derivative_order: int,
        accuracy_order: int,
        axis: str,
        implicit_alpha: float,
        explicit_width: int,
        moment_residual: float,
        condition_estimate: float,
        modified_symbol_error: float,
        storage_bytes: int,
        solve_workspace_bytes_per_rhs: int,
        subject_id: str,
    ):
        self.kind = kind
        self.derivative_order = int(derivative_order)
        self.accuracy_order = int(accuracy_order)
        self.axis = str(axis)
        self.implicit_alpha = float(implicit_alpha)
        self.explicit_width = int(explicit_width)
        self.moment_residual = float(moment_residual)
        self.condition_estimate = float(condition_estimate)
        self.modified_symbol_error = float(modified_symbol_error)
        self.storage_bytes = int(storage_bytes)
        self.solve_workspace_bytes_per_rhs = int(solve_workspace_bytes_per_rhs)
        self.dense_materialization_entries = 0
        self.requires_unsharded_axis = True
        self.passed = (
            np.isfinite(moment_residual)
            and moment_residual <= 5e-9
            and np.isfinite(condition_estimate)
            and condition_estimate <= 1e12
            and storage_bytes > 0
            and solve_workspace_bytes_per_rhs > 0
        )
        self.report_id = canonical_fingerprint(
            {
                "kind": "compact-operator-report-v1",
                "subject": subject_id,
                "operator_kind": kind,
                "derivative_order": int(derivative_order),
                "accuracy_order": int(accuracy_order),
                "axis": str(axis),
                "alpha": float(implicit_alpha),
                "explicit_width": int(explicit_width),
                "moment_residual": float(moment_residual),
                "condition_estimate": float(condition_estimate),
                "modified_symbol_error": float(modified_symbol_error),
                "storage_bytes": int(storage_bytes),
                "solve_workspace_bytes_per_rhs": int(
                    solve_workspace_bytes_per_rhs
                ),
            }
        )


def _monomial_derivative(degree: int, derivative: int, coordinate: float, /) -> float:
    if degree < derivative:
        return 0.0
    coefficient = math.factorial(degree) / math.factorial(degree - derivative)
    return coefficient * coordinate ** (degree - derivative)


def _implicit_alpha(kind: CompactOperatorKind, derivative: int, order: int, /) -> float:
    if kind == "interpolation":
        return 0.25 if order == 4 else 1.0 / 3.0
    if derivative == 1:
        return 0.25 if order == 4 else 1.0 / 3.0
    if derivative == 2:
        return 0.1 if order == 4 else 2.0 / 11.0
    raise ValueError("Compact derivatives support first or second derivatives.")


def _explicit_offsets(
    kind: CompactOperatorKind,
    derivative: int,
    order: int,
    shift: float,
    /,
) -> np.ndarray:
    if kind == "derivative" and abs(shift) <= 1e-12:
        if derivative == 1:
            return np.asarray((-1, 1) if order == 4 else (-2, -1, 1, 2))
        return np.asarray((-1, 0, 1) if order == 4 else (-2, -1, 0, 1, 2))
    width = order if kind == "interpolation" else order + derivative
    start = math.floor(shift - 0.5 * (width - 1))
    return np.arange(start, start + width, dtype=np.int32)


def _prepare_right_weights(
    offsets: np.ndarray,
    *,
    shift: float,
    derivative: int,
    alpha: float,
    spacing: float,
    accuracy_order: int,
) -> tuple[np.ndarray, float]:
    source_coordinates = (offsets.astype(float) - shift) * spacing
    width = int(offsets.size)
    matrix = np.asarray(
        [[coordinate**degree for coordinate in source_coordinates] for degree in range(width)]
    )
    target = np.asarray(
        [
            _monomial_derivative(degree, derivative, 0.0)
            + alpha
            * (
                _monomial_derivative(degree, derivative, -spacing)
                + _monomial_derivative(degree, derivative, spacing)
            )
            for degree in range(width)
        ]
    )
    weights = np.linalg.solve(matrix, target)
    maximum_degree = derivative + accuracy_order - 1
    residual = 0.0
    for degree in range(maximum_degree + 1):
        left = float(np.sum(weights * source_coordinates**degree))
        right = _monomial_derivative(degree, derivative, 0.0) + alpha * (
            _monomial_derivative(degree, derivative, -spacing)
            + _monomial_derivative(degree, derivative, spacing)
        )
        residual = max(residual, abs(left - right))
    return weights, residual


def _modified_symbol_error(
    offsets: np.ndarray,
    weights: np.ndarray,
    *,
    shift: float,
    derivative: int,
    alpha: float,
    spacing: float,
) -> float:
    angles = np.linspace(1e-4, 0.5 * np.pi, 128)
    numerator = np.sum(
        weights[None, :] * np.exp(1j * angles[:, None] * offsets[None, :]), axis=1
    )
    denominator = 1.0 + 2.0 * alpha * np.cos(angles)
    symbol = numerator / denominator
    exact = (
        np.exp(1j * angles * shift)
        if derivative == 0
        else (1j * angles / spacing) ** derivative
        * np.exp(1j * angles * shift)
    )
    scale = np.maximum(np.abs(exact), 1.0)
    return float(np.max(np.abs(symbol - exact) / scale))


class PreparedCompactOperator(AbstractLinearOperator):
    """Tensor-axis periodic compact action using a structured cyclic line solve."""

    source: ArraySpace
    target: ArraySpace
    grid: PreparedTensorGrid
    line_operator: BasePlusLowRankLinearOperator
    prepared_line_solve: PreparedLowRankSolve
    offsets: tuple[int, ...] = eqx.field(static=True)
    weights: Array
    axis: int = eqx.field(static=True)
    report: CompactOperatorReport
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        source_location: GridLocation,
        target_location: GridLocation,
        offsets: Sequence[int],
        weights: ArrayLike,
        /,
        *,
        kind: CompactOperatorKind,
        derivative_order: int,
        accuracy_order: int,
        axis: str,
        alpha: float,
        moment_residual: float,
        component_shape: Sequence[int] = (),
        dtype: object = float,
    ):
        axis_index = grid.axis_names.index(axis)
        source_field = grid.field_space(
            "compact_source",
            location=source_location,
            component_shape=component_shape,
            dtype=dtype,
        )
        target_field = grid.field_space(
            "compact_target",
            location=target_location,
            component_shape=component_shape,
            dtype=dtype,
        )
        if not isinstance(source_field.vector_space, ArraySpace) or not isinstance(
            target_field.vector_space, ArraySpace
        ):
            raise TypeError("Compact operators require ArraySpace field coordinates.")
        source_shape = source_field.vector_space.shape
        target_shape = target_field.vector_space.shape
        if source_shape != target_shape:
            raise ValueError(
                "Initial compact execution requires equal periodic source and target shapes."
            )
        count = target_shape[axis_index]
        coordinate_dtype = target_field.vector_space.dtype
        diagonal = jnp.ones((count,), dtype=coordinate_dtype)
        off_diagonal = jnp.full((count - 1,), alpha, dtype=coordinate_dtype)
        line_space = ArraySpace((count,), dtype=coordinate_dtype)
        base = TridiagonalLinearOperator(
            off_diagonal,
            diagonal,
            off_diagonal,
            space=line_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "compact-line-tridiagonal-base",
                    "grid": grid.prepared_id,
                    "axis": axis,
                    "count": count,
                    "alpha": alpha,
                }
            ),
        )
        left_factor = jnp.zeros((count, 2), dtype=coordinate_dtype)
        left_factor = left_factor.at[0, 0].set(1.0)
        left_factor = left_factor.at[-1, 1].set(1.0)
        right_factor = jnp.zeros((count, 2), dtype=coordinate_dtype)
        right_factor = right_factor.at[-1, 0].set(1.0)
        right_factor = right_factor.at[0, 1].set(1.0)
        core = jnp.diag(jnp.full((2,), alpha, dtype=coordinate_dtype))
        line_operator = BasePlusLowRankLinearOperator(
            base,
            left_factor,
            right_factor,
            core,
            operator_id=canonical_fingerprint(
                {
                    "kind": "compact-cyclic-line",
                    "grid": grid.prepared_id,
                    "axis": axis,
                    "count": count,
                    "alpha": alpha,
                }
            ),
        )
        prepared = prepare_low_rank_solve(
            line_operator,
            LowRankSolvePolicy(base_nonsingularity="asserted"),
        )
        offsets_ = tuple(int(value) for value in offsets)
        weights_ = jnp.asarray(weights, dtype=coordinate_dtype)
        identifier = canonical_fingerprint(
            {
                "kind": "prepared-compact-operator-v1",
                "grid": grid.prepared_id,
                "source_location": source_location.location_id,
                "target_location": target_location.location_id,
                "operator_kind": kind,
                "derivative_order": derivative_order,
                "accuracy_order": accuracy_order,
                "axis": axis,
                "alpha": alpha,
                "offsets": list(offsets_),
                "component_shape": list(component_shape),
                "dtype": coordinate_dtype.str,
            }
        )
        angles = 2.0 * np.pi * np.arange(count) / count
        eigenvalues = 1.0 + 2.0 * alpha * np.cos(angles)
        condition = float(np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues)))
        itemsize = coordinate_dtype.itemsize
        storage = int(
            (3 * count - 2 + 4 * count + 4 + len(offsets_)) * itemsize
        )
        self.source = source_field.vector_space
        self.target = target_field.vector_space
        self.properties = OperatorProperties(evidence={})
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
            diagonal_assembly=False,
        )
        self.batch_shape = ()
        self.operator_id = identifier
        self.grid = grid
        self.line_operator = line_operator
        self.prepared_line_solve = prepared
        self.offsets = offsets_
        self.weights = weights_
        self.axis = axis_index
        spacing = float(grid.structured_axes[axis_index].interval_widths[0])
        shift = float(
            target_location.offsets[axis_index] - source_location.offsets[axis_index]
        )
        self.report = CompactOperatorReport(
            kind=kind,
            derivative_order=derivative_order,
            accuracy_order=accuracy_order,
            axis=axis,
            implicit_alpha=alpha,
            explicit_width=len(offsets_),
            moment_residual=moment_residual,
            condition_estimate=condition,
            modified_symbol_error=_modified_symbol_error(
                np.asarray(offsets_),
                np.asarray(weights_),
                shift=shift,
                derivative=derivative_order,
                alpha=alpha,
                spacing=spacing,
            ),
            storage_bytes=storage,
            solve_workspace_bytes_per_rhs=prepared.plan.cost.solve_workspace_bytes_per_rhs,
            subject_id=identifier,
        )
        if not self.report.passed:
            raise RuntimeError("Prepared compact operator failed its certification.")
        self.prepared_id = identifier

    def _right_action(self, value: Array, /, *, transpose: bool) -> Array:
        result = jnp.zeros(self.target.shape if not transpose else self.source.shape, dtype=value.dtype)
        for offset, weight in zip(self.offsets, self.weights, strict=True):
            shift = offset if transpose else -offset
            result = result + weight * jnp.roll(value, shift, axis=self.axis)
        return result

    def _solve_lines(self, right_hand_side: Array, /) -> Array:
        moved = jnp.moveaxis(right_hand_side, self.axis, 0)
        count = moved.shape[0]
        trailing_shape = moved.shape[1:]
        columns = int(np.prod(trailing_shape))
        flattened = moved.reshape((count, columns))
        result = solve_low_rank(
            self.prepared_line_solve,
            flattened,
            rhs_layout=RHSLayout((columns,)),
        )
        value = eqx.error_if(
            result.value,
            jnp.any(~result.successful),
            "Compact cyclic line solve failed.",
        )
        restored = value.reshape((count,) + trailing_shape)
        return jnp.moveaxis(restored, 0, self.axis)

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(jnp.asarray(vector))
        return self._solve_lines(self._right_action(value, transpose=False))

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        solved = self._solve_lines(value)
        return self._right_action(solved, transpose=True)

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        covector = self.target.riesz(value)
        return self.source.inverse_riesz(self.transpose_mv(covector))

    def _materialize(self, /) -> Array:
        raise ValueError("Compact operators prohibit dense materialization.")


def _validate_periodic_axis(
    grid: PreparedTensorGrid,
    axis: str,
    source_location: GridLocation,
    target_location: GridLocation,
    /,
) -> tuple[int, float, float]:
    if not isinstance(grid, PreparedTensorGrid):
        raise TypeError("grid must be a PreparedTensorGrid.")
    axis_ = str(axis)
    if axis_ not in grid.axis_names:
        raise ValueError("Compact axis must belong to the prepared tensor grid.")
    axis_index = grid.axis_names.index(axis_)
    structured_axis = grid.structured_axes[axis_index]
    if not structured_axis.periodic:
        raise ValueError("Initial compact operators require a periodic axis.")
    source_layout = grid.layout_at(source_location)
    target_layout = grid.layout_at(target_location)
    if source_layout.shape != target_layout.shape:
        raise ValueError("Periodic compact source and target layouts must have equal shape.")
    widths = np.asarray(structured_axis.interval_widths, dtype=float)
    if not np.allclose(widths, widths[0], rtol=1e-10, atol=1e-12):
        raise ValueError("Periodic compact operators require uniform spacing.")
    shift = float(
        target_location.offsets[axis_index] - source_location.offsets[axis_index]
    )
    return axis_index, float(widths[0]), shift


class CompactDerivativePlan(StrictModule, NonTrainableState):
    """Periodic fourth/sixth-order implicit derivative preparation."""

    grid: PreparedTensorGrid
    request: DerivativeRequest
    component_shape: tuple[int, ...] = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        request: DerivativeRequest,
        /,
        *,
        component_shape: Sequence[int] = (),
        dtype: object = float,
    ):
        if not isinstance(request, DerivativeRequest):
            raise TypeError("request must be a DerivativeRequest.")
        if request.bias != "centered" or request.boundary != "periodic":
            raise ValueError("Compact derivatives require centered periodic requests.")
        if request.derivative_order not in (1, 2) or request.accuracy_order not in (4, 6):
            raise ValueError(
                "Compact derivatives support derivative orders one/two and accuracy four/six."
            )
        _validate_periodic_axis(
            grid,
            request.axis,
            request.source_location,
            request.target_location,
        )
        components = tuple(int(value) for value in component_shape)
        if any(value <= 0 for value in components):
            raise ValueError("component_shape dimensions must be positive.")
        dtype_name = jnp.dtype(dtype).name
        self.grid = grid
        self.request = request
        self.component_shape = components
        self.dtype = dtype_name
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compact-derivative-plan-v1",
                "grid": grid.prepared_id,
                "request": request.request_id,
                "component_shape": list(components),
                "dtype": dtype_name,
            }
        )

    def prepare(self, /) -> PreparedCompactOperator:
        _, spacing, shift = _validate_periodic_axis(
            self.grid,
            self.request.axis,
            self.request.source_location,
            self.request.target_location,
        )
        alpha = _implicit_alpha(
            "derivative",
            self.request.derivative_order,
            self.request.accuracy_order,
        )
        offsets = _explicit_offsets(
            "derivative",
            self.request.derivative_order,
            self.request.accuracy_order,
            shift,
        )
        count = self.grid.layout_at(self.request.target_location).shape[
            self.grid.axis_names.index(self.request.axis)
        ]
        if count <= 2 * int(np.max(np.abs(offsets))):
            raise ValueError("Compact derivative line is too short for its explicit support.")
        weights, residual = _prepare_right_weights(
            offsets,
            shift=shift,
            derivative=self.request.derivative_order,
            alpha=alpha,
            spacing=spacing,
            accuracy_order=self.request.accuracy_order,
        )
        return PreparedCompactOperator(
            self.grid,
            self.request.source_location,
            self.request.target_location,
            offsets,
            weights,
            kind="derivative",
            derivative_order=self.request.derivative_order,
            accuracy_order=self.request.accuracy_order,
            axis=self.request.axis,
            alpha=alpha,
            moment_residual=residual,
            component_shape=self.component_shape,
            dtype=self.dtype,
        )


class CompactInterpolationPlan(StrictModule, NonTrainableState):
    """Periodic fourth/sixth-order implicit interpolation between grid locations."""

    grid: PreparedTensorGrid
    axis: str = eqx.field(static=True)
    source_location: GridLocation
    target_location: GridLocation
    accuracy_order: int = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        axis: str,
        source_location: GridLocation,
        target_location: GridLocation,
        /,
        *,
        accuracy_order: int = 4,
        component_shape: Sequence[int] = (),
        dtype: object = float,
    ):
        order = int(accuracy_order)
        if order not in (4, 6):
            raise ValueError("Compact interpolation supports accuracy four or six.")
        _validate_periodic_axis(grid, axis, source_location, target_location)
        if source_location.location_id == target_location.location_id:
            raise ValueError("Compact interpolation requires distinct grid locations.")
        components = tuple(int(value) for value in component_shape)
        if any(value <= 0 for value in components):
            raise ValueError("component_shape dimensions must be positive.")
        dtype_name = jnp.dtype(dtype).name
        self.grid = grid
        self.axis = str(axis)
        self.source_location = source_location
        self.target_location = target_location
        self.accuracy_order = order
        self.component_shape = components
        self.dtype = dtype_name
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compact-interpolation-plan-v1",
                "grid": grid.prepared_id,
                "axis": str(axis),
                "source": source_location.location_id,
                "target": target_location.location_id,
                "accuracy_order": order,
                "component_shape": list(components),
                "dtype": dtype_name,
            }
        )

    def prepare(self, /) -> PreparedCompactOperator:
        axis_index, spacing, shift = _validate_periodic_axis(
            self.grid,
            self.axis,
            self.source_location,
            self.target_location,
        )
        alpha = _implicit_alpha("interpolation", 0, self.accuracy_order)
        offsets = _explicit_offsets(
            "interpolation", 0, self.accuracy_order, shift
        )
        count = self.grid.layout_at(self.target_location).shape[axis_index]
        if count <= 2 * int(np.max(np.abs(offsets))):
            raise ValueError(
                "Compact interpolation line is too short for its explicit support."
            )
        weights, residual = _prepare_right_weights(
            offsets,
            shift=shift,
            derivative=0,
            alpha=alpha,
            spacing=spacing,
            accuracy_order=self.accuracy_order,
        )
        return PreparedCompactOperator(
            self.grid,
            self.source_location,
            self.target_location,
            offsets,
            weights,
            kind="interpolation",
            derivative_order=0,
            accuracy_order=self.accuracy_order,
            axis=self.axis,
            alpha=alpha,
            moment_residual=residual,
            component_shape=self.component_shape,
            dtype=self.dtype,
        )


__all__ = [
    "CompactDerivativePlan",
    "CompactInterpolationPlan",
    "CompactOperatorKind",
    "CompactOperatorReport",
    "PreparedCompactOperator",
]
