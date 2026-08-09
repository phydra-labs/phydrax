#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Integral
from typing import Any, cast, Literal, TypeAlias

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainFunction, PointBatch, SampleLayout

from ..._doc import DOC_KEY0
from ..._frozendict import frozendict
from ..._interpolation import BoundsMode, bspline_evaluate, bspline_stencil, BSplineGrid
from ..._numerics import solve_weighted_least_squares
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


BSplineFitMode: TypeAlias = Literal["interpolate", "least_squares", "smooth"]
BSplineBoundaryMode: TypeAlias = Literal["open", "natural", "periodic"]


class BSplineInterpolationPlan(StrictModule, NonTrainableState):
    """Static policy for exact, regression, or Sobolev-regularized spline fitting."""

    degree: int = eqx.field(static=True)
    num_intervals: int | None = eqx.field(static=True)
    mode: BSplineFitMode = eqx.field(static=True)
    smoothing: float = eqx.field(static=True)
    regularization_order: int = eqx.field(static=True)
    bounds: BoundsMode = eqx.field(static=True)
    boundary: BSplineBoundaryMode = eqx.field(static=True)
    rcond: float | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        degree: int = 3,
        num_intervals: int | None = None,
        mode: BSplineFitMode = "interpolate",
        smoothing: float | None = None,
        regularization_order: int | None = None,
        bounds: BoundsMode = "error",
        boundary: BSplineBoundaryMode = "open",
        rcond: float | None = None,
    ):
        if isinstance(degree, bool) or not isinstance(degree, Integral):
            raise TypeError("B-spline interpolation degree must be an integer.")
        degree_ = int(degree)
        if degree_ < 0:
            raise ValueError("B-spline interpolation degree must be nonnegative.")
        if num_intervals is not None and (
            isinstance(num_intervals, bool)
            or not isinstance(num_intervals, Integral)
            or num_intervals < 1
        ):
            raise ValueError("num_intervals must be a positive integer or None.")
        if mode not in ("interpolate", "least_squares", "smooth"):
            raise ValueError(f"Unknown B-spline fit mode: {mode!r}.")
        smoothing_ = (
            (1.0e-4 if mode == "smooth" else 0.0)
            if smoothing is None
            else float(smoothing)
        )
        if not isfinite(smoothing_) or smoothing_ < 0.0:
            raise ValueError("B-spline smoothing must be finite and nonnegative.")
        if mode == "smooth" and smoothing_ == 0.0:
            raise ValueError("Smooth B-spline fitting requires positive smoothing.")
        if mode != "smooth" and smoothing_ != 0.0:
            raise ValueError("smoothing is only valid for smooth B-spline fitting.")
        order_ = min(2, degree_) if regularization_order is None else regularization_order
        if isinstance(order_, bool) or not isinstance(order_, Integral):
            raise TypeError("regularization_order must be an integer.")
        order_ = int(order_)
        if mode == "smooth" and not 1 <= order_ <= degree_:
            raise ValueError(
                "Smooth-fit regularization_order must lie between one and degree."
            )
        if mode != "smooth" and not 0 <= order_ <= degree_:
            raise ValueError("regularization_order must lie between zero and degree.")
        if bounds not in ("error", "clip", "extrapolate", "fill"):
            raise ValueError(f"Unknown B-spline bounds mode: {bounds!r}.")
        if boundary not in ("open", "natural", "periodic"):
            raise ValueError(f"Unknown B-spline boundary mode: {boundary!r}.")
        if boundary == "natural" and degree_ < 2:
            raise ValueError("Natural B-spline boundaries require degree at least two.")
        if boundary == "periodic" and degree_ < 1:
            raise ValueError("Periodic B-spline boundaries require positive degree.")
        rcond_ = None if rcond is None else float(rcond)
        if rcond_ is not None and (not isfinite(rcond_) or rcond_ < 0.0):
            raise ValueError("rcond must be finite and nonnegative or None.")
        self.degree = degree_
        self.num_intervals = None if num_intervals is None else int(num_intervals)
        self.mode = mode
        self.smoothing = smoothing_
        self.regularization_order = order_
        self.bounds = bounds
        self.boundary = boundary
        self.rcond = rcond_


class BSplineBoundaryConstraint(StrictModule, NonTrainableState):
    """One exact value or derivative jet imposed at a spline coordinate."""

    location: float | Literal["lower", "upper"] = eqx.field(static=True)
    derivative_order: int = eqx.field(static=True)
    value: Array

    def __init__(
        self,
        location: float | Literal["lower", "upper"],
        derivative_order: int,
        value: ArrayLike,
        /,
    ):
        if isinstance(location, str):
            if location not in ("lower", "upper"):
                raise ValueError(
                    "Constraint location must be 'lower', 'upper', or finite."
                )
            location_ = cast(Literal["lower", "upper"], location)
        else:
            location_ = float(location)
            if not isfinite(location_):
                raise ValueError("Constraint location must be finite.")
        if (
            isinstance(derivative_order, bool)
            or not isinstance(derivative_order, Integral)
            or derivative_order < 0
        ):
            raise ValueError("Constraint derivative_order must be a nonnegative integer.")
        value_ = jnp.asarray(value)
        if bool(jnp.any(~jnp.isfinite(value_))):
            raise ValueError("Constraint values must be finite.")
        self.location = location_
        self.derivative_order = int(derivative_order)
        self.value = value_


class BSplineFitDiagnostics(StrictModule, NonTrainableState):
    """Rank, conditioning, residual, and regularization diagnostics for one fit."""

    mode: BSplineFitMode = eqx.field(static=True)
    num_observations: int = eqx.field(static=True)
    coefficient_count: int = eqx.field(static=True)
    constraint_count: int = eqx.field(static=True)
    matrix_rank: int = eqx.field(static=True)
    condition_estimate: float = eqx.field(static=True)
    weighted_residual_norm: float = eqx.field(static=True)
    constraint_residual_norm: float = eqx.field(static=True)
    regularization_energy: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        mode: BSplineFitMode,
        num_observations: int,
        coefficient_count: int,
        constraint_count: int,
        matrix_rank: int,
        condition_estimate: float,
        weighted_residual_norm: float,
        constraint_residual_norm: float,
        regularization_energy: float,
    ):
        self.mode = mode
        self.num_observations = num_observations
        self.coefficient_count = coefficient_count
        self.constraint_count = constraint_count
        self.matrix_rank = matrix_rank
        self.condition_estimate = condition_estimate
        self.weighted_residual_norm = weighted_residual_norm
        self.constraint_residual_norm = constraint_residual_norm
        self.regularization_energy = regularization_energy


class BSplineInterpolant(StrictModule, NonTrainableState):
    """Immutable differentiable univariate B-spline in physical coordinates."""

    grid: BSplineGrid
    coefficients: Array
    diagnostics: BSplineFitDiagnostics
    bounds: BoundsMode = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        grid: BSplineGrid,
        coefficients: ArrayLike,
        diagnostics: BSplineFitDiagnostics,
        bounds: BoundsMode,
    ):
        coefficients_ = jnp.asarray(coefficients)
        if (
            coefficients_.ndim < 1
            or int(coefficients_.shape[0]) != grid.coefficient_count
        ):
            raise ValueError(
                "Interpolant coefficients must have one leading entry per B-spline basis."
            )
        self.grid = grid
        self.coefficients = coefficients_
        self.diagnostics = diagnostics
        self.bounds = bounds
        self.output_shape = tuple(int(size) for size in coefficients_.shape[1:])

    @property
    def dtype(self):
        return self.coefficients.dtype

    def __call__(
        self,
        coordinate: ArrayLike,
        /,
        *,
        derivative_order: int = 0,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array | None = None,
        **kwargs: Any,
    ) -> Array:
        del key, iter_
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"BSplineInterpolant received unsupported keywords: {names}.")
        return bspline_evaluate(
            self.grid.knots,
            self.coefficients,
            coordinate,
            degree=self.grid.degree,
            derivative_order=derivative_order,
            bounds=self.bounds,
        ).values

    def derivative(self, coordinate: ArrayLike, order: int = 1, /) -> Array:
        """Evaluate one explicit coordinate derivative."""
        return self(coordinate, derivative_order=order)


def _basis_matrix(
    grid: BSplineGrid, coordinates: Array, derivative_order: int = 0
) -> Array:
    stencil = bspline_stencil(
        grid.knots,
        coordinates,
        degree=grid.degree,
        derivative_order=derivative_order,
        bounds="error",
    )
    rows = jnp.arange(coordinates.size, dtype=jnp.int32)[:, None]
    return (
        jnp.zeros(
            (coordinates.size, grid.coefficient_count),
            dtype=stencil.weights.dtype,
        )
        .at[rows, stencil.indices]
        .add(stencil.weights)
    )


def _rank_and_condition(matrix: Array, rcond: float | None) -> tuple[int, float]:
    matrix_host = np.asarray(matrix)
    singular_values = np.linalg.svd(matrix_host, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 0, float("inf")
    tolerance = (
        max(matrix_host.shape) * np.finfo(singular_values.dtype).eps
        if rcond is None
        else rcond
    ) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == min(matrix_host.shape)
        else float("inf")
    )
    return rank, condition


def _constraint_system(
    grid: BSplineGrid,
    plan: BSplineInterpolationPlan,
    constraints: tuple[BSplineBoundaryConstraint, ...],
    output_shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    lower, upper = (float(value) for value in grid.active_interval)
    rows: list[Array] = []
    values: list[Array] = []

    def append(location: float, order: int, value: ArrayLike) -> None:
        if order > grid.degree:
            raise ValueError("Boundary derivative order cannot exceed the spline degree.")
        if not lower <= location <= upper:
            raise ValueError(
                "Boundary constraint lies outside the spline active interval."
            )
        rows.append(
            _basis_matrix(grid, jnp.asarray([location]), derivative_order=order)[0]
        )
        value_ = jnp.asarray(value, dtype=dtype)
        try_shape = tuple(int(size) for size in value_.shape)
        if try_shape != output_shape:
            if value_.ndim != 0:
                raise ValueError(
                    "Boundary constraint values must match the fitted value shape or be scalar."
                )
            value_ = jnp.broadcast_to(value_, output_shape)
        values.append(value_.reshape((-1,)))

    zero = jnp.zeros(output_shape, dtype=dtype)
    if plan.boundary == "natural":
        append(lower, 2, zero)
        append(upper, 2, zero)
    elif plan.boundary == "periodic":
        for order in range(grid.degree):
            lower_row = _basis_matrix(grid, jnp.asarray([lower]), derivative_order=order)[
                0
            ]
            upper_row = _basis_matrix(grid, jnp.asarray([upper]), derivative_order=order)[
                0
            ]
            rows.append(lower_row - upper_row)
            values.append(zero.reshape((-1,)))
    for constraint in constraints:
        location = (
            lower
            if constraint.location == "lower"
            else upper
            if constraint.location == "upper"
            else constraint.location
        )
        append(float(location), constraint.derivative_order, constraint.value)
    if not rows:
        return (
            jnp.empty((0, grid.coefficient_count), dtype=grid.knots.dtype),
            jnp.empty((0, int(np.prod(output_shape, dtype=int))), dtype=dtype),
        )
    return jnp.stack(tuple(rows)), jnp.stack(tuple(values))


def _constrained_lstsq(
    matrix: Array,
    right_hand_side: Array,
    constraints: Array,
    constraint_values: Array,
    rcond: float | None,
) -> Array:
    constraint_rank, _ = _rank_and_condition(constraints, rcond)
    if constraint_rank != int(constraints.shape[0]):
        raise ValueError("B-spline boundary constraints are rank deficient or redundant.")
    _, _, vh = np.linalg.svd(np.asarray(constraints), full_matrices=True)
    null_space = jnp.asarray(vh[constraint_rank:].T, dtype=matrix.dtype)
    particular = jnp.linalg.lstsq(constraints, constraint_values, rcond=rcond)[0]
    if null_space.shape[1] == 0:
        return particular
    reduced_matrix = matrix @ null_space
    reduced_rank, _ = _rank_and_condition(reduced_matrix, rcond)
    if reduced_rank != int(null_space.shape[1]):
        raise ValueError(
            "B-spline fitting system is underdetermined or rank deficient after constraints."
        )
    reduced_rhs = right_hand_side - matrix @ particular
    correction = jnp.linalg.lstsq(reduced_matrix, reduced_rhs, rcond=rcond)[0]
    return particular + null_space @ correction


def fit_bspline(
    nodes: ArrayLike,
    values: ArrayLike,
    /,
    *,
    plan: BSplineInterpolationPlan | None = None,
    grid: BSplineGrid | None = None,
    sample_weights: ArrayLike | None = None,
    constraints: Sequence[BSplineBoundaryConstraint] = (),
) -> BSplineInterpolant:
    """Fit an immutable B-spline to scalar nodes and arbitrary trailing payloads."""
    if grid is not None and not isinstance(grid, BSplineGrid):
        raise TypeError("grid must be a BSplineGrid or None.")
    plan_ = (
        BSplineInterpolationPlan(degree=grid.degree)
        if plan is None and grid is not None
        else BSplineInterpolationPlan()
        if plan is None
        else plan
    )
    if not isinstance(plan_, BSplineInterpolationPlan):
        raise TypeError("plan must be a BSplineInterpolationPlan or None.")
    if grid is not None and grid.degree != plan_.degree:
        raise ValueError("The fitting plan degree must match the explicit grid degree.")
    constraints_ = tuple(constraints)
    if not all(isinstance(item, BSplineBoundaryConstraint) for item in constraints_):
        raise TypeError("Every fitting constraint must be a BSplineBoundaryConstraint.")

    nodes_raw = jnp.asarray(nodes)
    values_raw = jnp.asarray(values)
    if nodes_raw.ndim != 1 or nodes_raw.size == 0:
        raise ValueError("B-spline fitting nodes must be a nonempty rank-one array.")
    if jnp.issubdtype(nodes_raw.dtype, jnp.complexfloating):
        raise TypeError("B-spline fitting nodes must be real-valued.")
    nodes_host = np.asarray(nodes_raw, dtype=float)
    if not np.all(np.isfinite(nodes_host)):
        raise ValueError("B-spline fitting nodes must be finite.")
    if values_raw.ndim < 1 or int(values_raw.shape[0]) != int(nodes_raw.size):
        raise ValueError("Fitted values must have one leading entry per node.")
    if bool(jnp.any(~jnp.isfinite(values_raw))):
        raise ValueError("Fitted values must be finite.")
    if int(nodes_raw.size) < plan_.degree + 1:
        raise ValueError("B-spline fitting requires at least degree + 1 observations.")

    permutation = np.argsort(nodes_host, kind="stable")
    nodes_ = jnp.asarray(nodes_host[permutation], dtype=jnp.result_type(nodes_raw, float))
    value_dtype = jnp.result_type(values_raw, float)
    values_ = jnp.asarray(values_raw[permutation], dtype=value_dtype)
    unique_count = int(np.unique(nodes_host).size)
    if plan_.mode == "interpolate" and unique_count != int(nodes_.size):
        raise ValueError("Exact B-spline interpolation requires distinct nodes.")

    boundary_count = len(constraints_)
    if plan_.boundary == "natural":
        boundary_count += 2
    elif plan_.boundary == "periodic":
        boundary_count += plan_.degree
    if grid is None:
        if plan_.num_intervals is None:
            coefficient_count = (
                int(nodes_.size) + boundary_count
                if plan_.mode == "interpolate"
                else unique_count
            )
            interval_count = coefficient_count - plan_.degree
            if interval_count < 1:
                raise ValueError(
                    "The observations and constraints do not define enough spline coefficients."
                )
        else:
            interval_count = plan_.num_intervals
        grid_ = BSplineGrid.open_uniform(
            plan_.degree,
            interval_count,
            interval=(float(nodes_[0]), float(nodes_[-1])),
        )
    else:
        if plan_.num_intervals is not None:
            raise ValueError("num_intervals cannot be combined with an explicit grid.")
        grid_ = grid
    lower, upper = (float(value) for value in grid_.active_interval)
    if float(nodes_[0]) < lower or float(nodes_[-1]) > upper:
        raise ValueError("B-spline fitting nodes lie outside the grid active interval.")

    weights_ = jnp.ones(nodes_.shape, dtype=nodes_.dtype)
    if sample_weights is not None:
        if plan_.mode == "interpolate":
            raise ValueError("sample_weights are not meaningful for exact interpolation.")
        weights_raw = jnp.asarray(sample_weights)
        if weights_raw.ndim != 1 or weights_raw.shape != nodes_raw.shape:
            raise ValueError("sample_weights must have one scalar entry per node.")
        weights_host = np.asarray(weights_raw, dtype=float)[permutation]
        if not np.all(np.isfinite(weights_host)) or np.any(weights_host < 0.0):
            raise ValueError("sample_weights must be finite and nonnegative.")
        if not np.any(weights_host > 0.0):
            raise ValueError("At least one sample weight must be positive.")
        weights_ = jnp.asarray(weights_host, dtype=nodes_.dtype)

    output_shape = tuple(int(size) for size in values_.shape[1:])
    payload_size = int(np.prod(output_shape, dtype=int))
    observations = values_.reshape((int(nodes_.size), payload_size))
    basis = _basis_matrix(grid_, nodes_)
    constraint_matrix, constraint_values = _constraint_system(
        grid_, plan_, constraints_, output_shape, value_dtype
    )
    regularization_matrix = jnp.empty(
        (0, grid_.coefficient_count), dtype=grid_.knots.dtype
    )

    if plan_.mode == "interpolate":
        system = jnp.concatenate((basis, constraint_matrix), axis=0)
        right_hand_side = jnp.concatenate((observations, constraint_values), axis=0)
        if system.shape[0] != grid_.coefficient_count:
            raise ValueError(
                "Exact B-spline interpolation requires observations plus constraints "
                "to equal the coefficient count."
            )
        rank, condition = _rank_and_condition(system, plan_.rcond)
        if rank != grid_.coefficient_count:
            raise ValueError("Exact B-spline interpolation system is rank deficient.")
        coefficients_flat = jnp.linalg.solve(system, right_hand_side)
    else:
        square_root_weights = jnp.sqrt(weights_)
        system = basis * square_root_weights[:, None]
        right_hand_side = observations * square_root_weights[:, None]
        if plan_.mode == "smooth":
            quadrature_points, quadrature_weights = grid_.derivative_quadrature(
                plan_.regularization_order
            )
            derivative_basis = _basis_matrix(
                grid_, quadrature_points, plan_.regularization_order
            )
            regularization_matrix = (
                jnp.sqrt(quadrature_weights)[:, None] * derivative_basis
            )
            system = jnp.concatenate(
                (system, jnp.sqrt(plan_.smoothing) * regularization_matrix), axis=0
            )
            right_hand_side = jnp.concatenate(
                (
                    right_hand_side,
                    jnp.zeros(
                        (regularization_matrix.shape[0], payload_size),
                        dtype=value_dtype,
                    ),
                ),
                axis=0,
            )
        combined = jnp.concatenate((system, constraint_matrix), axis=0)
        rank, condition = _rank_and_condition(combined, plan_.rcond)
        if rank != grid_.coefficient_count:
            raise ValueError(
                "B-spline fitting system is underdetermined or rank deficient."
            )
        if constraint_matrix.shape[0] == 0:
            regression = solve_weighted_least_squares(
                system,
                right_hand_side,
                rcond=plan_.rcond,
                min_samples=grid_.coefficient_count,
            )
            if not bool(regression.valid):
                raise ValueError("B-spline least-squares solve failed.")
            coefficients_flat = regression.coefficients
        else:
            coefficients_flat = _constrained_lstsq(
                system,
                right_hand_side,
                constraint_matrix,
                constraint_values,
                plan_.rcond,
            )

    fitted = basis @ coefficients_flat
    weighted_residual = jnp.sqrt(weights_)[:, None] * (fitted - observations)
    constraint_residual = constraint_matrix @ coefficients_flat - constraint_values
    regularized = regularization_matrix @ coefficients_flat
    diagnostics = BSplineFitDiagnostics(
        mode=plan_.mode,
        num_observations=int(nodes_.size),
        coefficient_count=grid_.coefficient_count,
        constraint_count=int(constraint_matrix.shape[0]),
        matrix_rank=rank,
        condition_estimate=condition,
        weighted_residual_norm=float(jnp.linalg.norm(weighted_residual)),
        constraint_residual_norm=float(jnp.linalg.norm(constraint_residual)),
        regularization_energy=float(
            jnp.real(jnp.sum(regularized * jnp.conj(regularized)))
        ),
    )
    coefficients = coefficients_flat.reshape((grid_.coefficient_count, *output_shape))
    return BSplineInterpolant(
        grid=grid_,
        coefficients=coefficients,
        diagnostics=diagnostics,
        bounds=plan_.bounds,
    )


def interpolate_bspline(
    function: DomainFunction,
    nodes: ArrayLike,
    /,
    *,
    plan: BSplineInterpolationPlan | None = None,
    grid: BSplineGrid | None = None,
    sample_weights: ArrayLike | None = None,
    constraints: Sequence[BSplineBoundaryConstraint] = (),
    key: Key[Array, ""] = DOC_KEY0,
) -> DomainFunction:
    """Fit a one-dependency `DomainFunction` and preserve its domain metadata."""
    if not isinstance(function, DomainFunction):
        raise TypeError("interpolate_bspline expects a DomainFunction.")
    if len(function.deps) != 1:
        raise ValueError("B-spline DomainFunction interpolation requires one dependency.")
    nodes_ = jnp.asarray(nodes)
    if nodes_.ndim != 1 or nodes_.size == 0:
        raise ValueError(
            "B-spline interpolation nodes must be a nonempty rank-one array."
        )
    dependency = function.deps[0]
    dependency_domain = function.domain.factor(dependency)
    structure = SampleLayout(((dependency,),)).canonicalize(dependency_domain.labels)
    sample_axis = structure.axis_for(dependency)
    if sample_axis is None:
        raise RuntimeError("B-spline interpolation structure has no sample axis.")
    points = PointBatch(
        frozendict({dependency: cx.Field(nodes_, dims=(sample_axis,))}),
        structure,
    )
    fitting_function = DomainFunction(
        domain=dependency_domain,
        deps=(dependency,),
        func=function.func,
        metadata={},
    )
    evaluated = fitting_function(points, key=key)
    interpolant = fit_bspline(
        nodes_,
        evaluated.data,
        plan=plan,
        grid=grid,
        sample_weights=sample_weights,
        constraints=constraints,
    )
    return DomainFunction(
        domain=function.domain,
        deps=function.deps,
        func=interpolant,
        metadata=function.metadata,
    )


__all__ = [
    "BSplineBoundaryConstraint",
    "BSplineBoundaryMode",
    "BSplineFitDiagnostics",
    "BSplineFitMode",
    "BSplineInterpolant",
    "BSplineInterpolationPlan",
    "fit_bspline",
    "interpolate_bspline",
]
