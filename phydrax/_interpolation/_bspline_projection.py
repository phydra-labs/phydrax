#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import ceil, isfinite
from numbers import Integral
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    DenseCholesky,
    DenseLinearOperator,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    solve,
)
from ._bspline import bspline_stencil
from ._bspline_grid import BSplineGrid


ProjectionMethod = Literal["auto", "exact", "l2"]


def _validate_common_interval(old_grid: BSplineGrid, new_grid: BSplineGrid) -> None:
    old_interval = np.asarray(old_grid.active_interval, dtype=float)
    new_interval = np.asarray(new_grid.active_interval, dtype=float)
    if not np.array_equal(old_interval, new_interval):
        raise ValueError("B-spline projection grids must have the same active interval.")


def _union_quadrature(
    grids: tuple[BSplineGrid, ...], polynomial_degree: int
) -> tuple[Array, Array]:
    breakpoints = np.unique(
        np.concatenate(tuple(np.asarray(grid.breakpoints) for grid in grids))
    )
    order = max(1, ceil((polynomial_degree + 1) / 2))
    reference_nodes, reference_weights = np.polynomial.legendre.leggauss(order)
    nodes: list[float] = []
    weights: list[float] = []
    for lower, upper in zip(breakpoints[:-1], breakpoints[1:], strict=True):
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        nodes.extend((midpoint + half_width * reference_nodes).tolist())
        weights.extend((half_width * reference_weights).tolist())
    dtype = jnp.result_type(*(grid.knots for grid in grids))
    return jnp.asarray(nodes, dtype=dtype), jnp.asarray(weights, dtype=dtype)


def _gram_from_stencils(
    left_grid: BSplineGrid,
    right_grid: BSplineGrid,
    nodes: Array,
    weights: Array,
) -> Array:
    left = bspline_stencil(
        left_grid.knots,
        nodes,
        degree=left_grid.degree,
        bounds="error",
    )
    right = bspline_stencil(
        right_grid.knots,
        nodes,
        degree=right_grid.degree,
        bounds="error",
    )
    left_indices = left.indices.reshape((-1, left_grid.degree + 1))
    right_indices = right.indices.reshape((-1, right_grid.degree + 1))
    left_weights = left.weights.reshape((-1, left_grid.degree + 1))
    right_weights = right.weights.reshape((-1, right_grid.degree + 1))
    contributions = (
        weights[:, None, None] * left_weights[:, :, None] * right_weights[:, None, :]
    )
    matrix = jnp.zeros(
        (left_grid.coefficient_count, right_grid.coefficient_count),
        dtype=contributions.dtype,
    )
    return matrix.at[left_indices[:, :, None], right_indices[:, None, :]].add(
        contributions
    )


def bspline_mass_matrix(grid: BSplineGrid, /) -> Array:
    """Assemble the exact coefficient-space L2 mass matrix."""
    nodes, weights = grid.quadrature(2 * grid.degree)
    return _gram_from_stencils(grid, grid, nodes, weights)


def bspline_cross_gram(old_grid: BSplineGrid, new_grid: BSplineGrid, /) -> Array:
    """Assemble ``integral N_new(x) N_old(x)^T dx`` over the shared interval."""
    _validate_common_interval(old_grid, new_grid)
    nodes, weights = _union_quadrature(
        (old_grid, new_grid), old_grid.degree + new_grid.degree
    )
    return _gram_from_stencils(new_grid, old_grid, nodes, weights)


def _solve_mass_matrix(mass: Array, right_hand_side: Array, /) -> Array:
    solve_dtype = jnp.result_type(mass, right_hand_side)
    operator = DenseLinearOperator(
        mass.astype(solve_dtype),
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    return solve(
        LinearSystem(operator),
        right_hand_side.astype(solve_dtype),
        policy=LinearSolvePolicy(
            DenseCholesky(),
            failure=FailurePolicy("error"),
        ),
    ).value


def _refinement_insertions(
    old_knots: np.ndarray, new_knots: np.ndarray
) -> list[float] | None:
    old_index = 0
    insertions: list[float] = []
    for knot in new_knots:
        if old_index < old_knots.size and knot == old_knots[old_index]:
            old_index += 1
        else:
            insertions.append(float(knot))
    return insertions if old_index == old_knots.size else None


def _insert_knot_once(
    knots: np.ndarray,
    coefficients: np.ndarray,
    degree: int,
    knot: float,
) -> tuple[np.ndarray, np.ndarray]:
    coefficient_count = coefficients.shape[0]
    span = int(np.searchsorted(knots, knot, side="right") - 1)
    span = min(max(span, degree), coefficient_count - 1)
    multiplicity = int(np.count_nonzero(knots == knot))
    if multiplicity > degree:
        raise ValueError(
            "A B-spline knot cannot be inserted beyond degree + 1 multiplicity."
        )

    refined = np.empty(
        (coefficient_count + 1, coefficients.shape[1]), dtype=coefficients.dtype
    )
    first_blend = span - degree + 1
    last_blend = span - multiplicity
    refined[:first_blend] = coefficients[:first_blend]
    refined[last_blend + 1 :] = coefficients[last_blend:]
    for index in range(first_blend, last_blend + 1):
        denominator = knots[index + degree] - knots[index]
        alpha = 0.0 if denominator == 0.0 else (knot - knots[index]) / denominator
        refined[index] = (
            alpha * coefficients[index] + (1.0 - alpha) * coefficients[index - 1]
        )
    refined_knots = np.insert(knots, span + 1, knot)
    return refined_knots, refined


def _exact_refinement_matrix(
    old_grid: BSplineGrid, new_grid: BSplineGrid
) -> Array | None:
    if old_grid.degree != new_grid.degree:
        return None
    old_knots = np.asarray(old_grid.knots)
    new_knots = np.asarray(new_grid.knots)
    insertions = _refinement_insertions(old_knots, new_knots)
    if insertions is None:
        return None

    matrix = np.eye(old_grid.coefficient_count, dtype=np.result_type(old_knots, float))
    working_knots = old_knots
    for knot in insertions:
        working_knots, matrix = _insert_knot_once(
            working_knots,
            matrix,
            old_grid.degree,
            knot,
        )
    if not np.array_equal(working_knots, new_knots):
        return None
    return jnp.asarray(matrix, dtype=jnp.result_type(old_grid.knots, new_grid.knots))


def bspline_projection_matrix(
    old_grid: BSplineGrid,
    new_grid: BSplineGrid,
    /,
    *,
    method: ProjectionMethod = "auto",
) -> Array:
    """Return the old-to-new coefficient map using exact insertion or L2 projection."""
    if method not in ("auto", "exact", "l2"):
        raise ValueError(f"Unknown B-spline projection method: {method!r}.")
    _validate_common_interval(old_grid, new_grid)
    if method != "l2":
        exact = _exact_refinement_matrix(old_grid, new_grid)
        if exact is not None:
            return exact
        if method == "exact":
            raise ValueError(
                "Exact transfer requires equal degrees and a nested knot vector."
            )
    mass = bspline_mass_matrix(new_grid)
    cross_gram = bspline_cross_gram(old_grid, new_grid)
    return _solve_mass_matrix(mass, cross_gram)


class BSplineGridTransfer(StrictModule, NonTrainableState):
    """Reusable, diagnosed map between two fixed B-spline coefficient grids."""

    old_grid: BSplineGrid
    new_grid: BSplineGrid
    matrix: Array
    method: Literal["exact", "l2"] = eqx.field(static=True)
    condition_estimate: float = eqx.field(static=True)
    projection_error_bound: float = eqx.field(static=True)

    def __init__(
        self,
        old_grid: BSplineGrid,
        new_grid: BSplineGrid,
        /,
        *,
        method: ProjectionMethod = "auto",
        maximum_condition: float = 1.0e12,
    ):
        if not isfinite(maximum_condition) or maximum_condition <= 1.0:
            raise ValueError("maximum_condition must be finite and greater than one.")
        _validate_common_interval(old_grid, new_grid)
        exact = None if method == "l2" else _exact_refinement_matrix(old_grid, new_grid)
        if method == "exact" and exact is None:
            raise ValueError(
                "Exact transfer requires equal degrees and a nested knot vector."
            )
        if method not in ("auto", "exact", "l2"):
            raise ValueError(f"Unknown B-spline projection method: {method!r}.")

        mass = bspline_mass_matrix(new_grid)
        condition = float(np.linalg.cond(np.asarray(mass)))
        if not isfinite(condition) or condition > maximum_condition:
            raise ValueError(
                "B-spline projection mass matrix is singular or ill-conditioned: "
                f"condition estimate {condition:.6g}."
            )
        if exact is not None:
            matrix = exact
            resolved_method: Literal["exact", "l2"] = "exact"
            error_bound = 0.0
        else:
            cross_gram = bspline_cross_gram(old_grid, new_grid)
            matrix = _solve_mass_matrix(mass, cross_gram)
            residual_gram = bspline_mass_matrix(old_grid) - cross_gram.T @ matrix
            residual_gram = 0.5 * (residual_gram + residual_gram.T)
            maximum_eigenvalue = float(
                np.max(np.linalg.eigvalsh(np.asarray(residual_gram)))
            )
            error_bound = float(np.sqrt(max(0.0, maximum_eigenvalue)))
            resolved_method = "l2"

        self.old_grid = old_grid
        self.new_grid = new_grid
        self.matrix = matrix
        self.method = resolved_method
        self.condition_estimate = condition
        self.projection_error_bound = error_bound

    def __call__(
        self, coefficients: ArrayLike, /, *, coefficient_axis: int = -1
    ) -> Array:
        coefficients_ = jnp.asarray(coefficients)
        if coefficients_.ndim == 0:
            raise ValueError("B-spline coefficients must have at least one axis.")
        if isinstance(coefficient_axis, bool) or not isinstance(
            coefficient_axis, Integral
        ):
            raise TypeError("coefficient_axis must be an integer.")
        axis = int(coefficient_axis) % coefficients_.ndim
        if int(coefficients_.shape[axis]) != self.old_grid.coefficient_count:
            raise ValueError(
                "B-spline coefficient axis does not match the transfer source grid."
            )
        coefficients_ = eqx.error_if(
            coefficients_,
            jnp.any(~jnp.isfinite(coefficients_)),
            "B-spline projection coefficients must be finite.",
        )
        moved = jnp.moveaxis(coefficients_, axis, -1)
        projected = oe.contract("ji,...i->...j", self.matrix, moved)
        return jnp.moveaxis(projected, -1, axis)


def project_bspline_coefficients(
    coefficients: ArrayLike,
    transfer: BSplineGridTransfer,
    /,
    *,
    coefficient_axis: int = -1,
) -> Array:
    """Apply a reusable grid transfer along one coefficient axis."""
    if not isinstance(transfer, BSplineGridTransfer):
        raise TypeError("transfer must be a BSplineGridTransfer.")
    return transfer(coefficients, coefficient_axis=coefficient_axis)


__all__ = [
    "BSplineGridTransfer",
    "ProjectionMethod",
    "bspline_cross_gram",
    "bspline_mass_matrix",
    "bspline_projection_matrix",
    "project_bspline_coefficients",
]
