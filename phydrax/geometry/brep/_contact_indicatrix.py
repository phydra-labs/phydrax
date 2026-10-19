"""Interval contact indicatrices in a common world-coordinate graph chart."""

from __future__ import annotations

from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import numpy as np

from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ...typing import parse
from .._interval_enclosure import (
    Interval,
    interval_add,
    interval_multiply,
    interval_subtract,
    prepare_interval_function,
)
from ._interval_source import evaluate_interval
from ._patches import AbstractSurfacePatch


ContactIndicatrixKind: TypeAlias = Literal[
    "tangent_point", "singular_point", "unresolved"
]
Bounds: TypeAlias = tuple[np.ndarray, np.ndarray]


@final
class ContactIndicatrix(StrictModule):
    """Certified real half-direction count; degenerate forms remain unresolved."""

    kind: ContactIndicatrixKind = eqx.field(static=True)
    branches: int = eqx.field(static=True)
    determinant_lower: float = eqx.field(static=True)
    determinant_upper: float = eqx.field(static=True)

    def __init__(
        self,
        kind: ContactIndicatrixKind,
        branches: int,
        determinant_lower: float,
        determinant_upper: float,
    ) -> None:
        self.kind = parse(kind, ContactIndicatrixKind, "kind")
        if branches not in (-1, 0, 4):
            raise ValueError(
                f"ContactIndicatrix.branches must be -1, 0 or 4; got {branches}"
            )
        self.branches = branches
        self.determinant_lower = determinant_lower
        self.determinant_upper = determinant_upper


def _entry(bounds: Bounds, *indices: int) -> Bounds:
    return bounds[0][indices], bounds[1][indices]


def _graph_directions(
    jet: Bounds,
    axes: tuple[int, int],
    determinant_margin: float,
) -> tuple[tuple[Bounds, Bounds], tuple[Bounds, Bounds]] | None:
    """Enclose native solves using a certified residual-to-forward-error bound."""
    lower, upper = jet[0][list(axes)], jet[1][list(axes)]
    midpoint = (lower + upper) / 2
    magnitude = np.maximum(np.abs(lower), np.abs(upper))
    # For a 2x2 matrix the adjugate infinity norm is bounded by the
    # largest column sum. Only the operator bound is formed, never an inverse.
    column_sums = np.nextafter(magnitude[0] + magnitude[1], np.inf)
    operator_bound = np.nextafter(np.max(column_sums) / determinant_margin, np.inf)
    if not np.isfinite(operator_bound):
        return None
    directions = []
    for axis in range(2):
        right = np.zeros(2, dtype=np.float64)
        right[axis] = 1
        result = solve_small_linear(SmallLinearSolvePlan(2), midpoint, right)
        if not bool(result.successful):
            return None
        value = np.asarray(result.value)
        residual_bounds = []
        for row in range(2):
            residual = (np.asarray(-right[row]), np.asarray(-right[row]))
            for column in range(2):
                residual = interval_add(
                    residual,
                    interval_multiply(
                        (lower[row, column], upper[row, column]),
                        (value[column], value[column]),
                    ),
                )
            residual_bounds.append(
                float(np.maximum(np.abs(residual[0]), np.abs(residual[1])))
            )
        radius = np.nextafter(operator_bound * max(residual_bounds), np.inf)
        lo, hi = (
            np.nextafter(value - radius, -np.inf),
            np.nextafter(value + radius, np.inf),
        )
        directions.append(
            (
                (np.asarray(lo[0]), np.asarray(hi[0])),
                (np.asarray(lo[1]), np.asarray(hi[1])),
            )
        )
    return directions[0], directions[1]


def _pullback_form(
    corrected: list[list[Interval]],
    directions: tuple[tuple[Bounds, Bounds], tuple[Bounds, Bounds]],
) -> Bounds:
    """Apply the enclosed chart-coordinate directions to a quadratic form."""
    lower, upper = np.zeros((2, 2)), np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            value = (np.asarray(0.0), np.asarray(0.0))
            for u in range(2):
                for v in range(2):
                    value = interval_add(
                        value,
                        interval_multiply(
                            corrected[u][v],
                            interval_multiply(directions[i][u], directions[j][v]),
                        ),
                    )
            lower[i, j], upper[i, j] = value
    return lower, upper


def _graph_form(
    jet: Bounds, hessian: Bounds, axes: tuple[int, int], normal_axis: int
) -> Bounds | None:
    """Eliminate UV derivatives in the shared graph chart (no matrix inverse)."""
    a, b = (_entry(jet, axes[0], k) for k in range(2))
    c, d = (_entry(jet, axes[1], k) for k in range(2))
    determinant = interval_subtract(interval_multiply(a, d), interval_multiply(b, c))
    if determinant[0] <= 0 <= determinant[1]:
        return None
    determinant_margin = min(abs(float(determinant[0])), abs(float(determinant[1])))
    directions = _graph_directions(jet, axes, determinant_margin)
    if directions is None:
        return None
    slope = []
    for direction in directions:
        slope.append(
            interval_add(
                interval_multiply(_entry(jet, normal_axis, 0), direction[0]),
                interval_multiply(_entry(jet, normal_axis, 1), direction[1]),
            )
        )
    corrected = []
    for i in range(2):
        row = []
        for j in range(2):
            value = _entry(hessian, normal_axis, i, j)
            for k in range(2):
                value = interval_subtract(
                    value, interval_multiply(slope[k], _entry(hessian, axes[k], i, j))
                )
            row.append(value)
        corrected.append(row)
    return _pullback_form(corrected, directions)


def contact_indicatrix(
    first: AbstractSurfacePatch,
    second: AbstractSurfacePatch,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> ContactIndicatrix:
    """Classify a certified tangency box using interval graph second forms.

    Definiteness gives an isolated contact; indefiniteness gives four outgoing
    half-branches. A zero-containing determinant never proves coincidence.
    """
    jets: list[Bounds] = []
    hessians: list[Bounds] = []
    for patch, offset in ((first, 0), (second, 2)):
        for order, destination in ((1, jets), (2, hessians)):
            derivative = jax.jacfwd(patch.evaluate)
            if order == 2:
                derivative = jax.jacfwd(derivative)
            prepared = prepare_interval_function(derivative, 2)
            lo, hi = evaluate_interval(
                prepared,
                lower[None, offset : offset + 2],
                upper[None, offset : offset + 2],
                budget=budget,
            )
            destination.append((lo[0], hi[0]))
    unresolved = ContactIndicatrix("unresolved", -1, -np.inf, np.inf)
    for normal_axis in range(3):
        axes = tuple(k for k in range(3) if k != normal_axis)
        first_form = _graph_form(jets[0], hessians[0], (axes[0], axes[1]), normal_axis)
        second_form = _graph_form(jets[1], hessians[1], (axes[0], axes[1]), normal_axis)
        if first_form is None or second_form is None:
            continue
        difference_bounds = interval_subtract(first_form, second_form)
        difference = (
            np.asarray(difference_bounds[0], dtype=np.float64),
            np.asarray(difference_bounds[1], dtype=np.float64),
        )
        determinant = interval_subtract(
            interval_multiply(_entry(difference, 0, 0), _entry(difference, 1, 1)),
            interval_multiply(_entry(difference, 0, 1), _entry(difference, 1, 0)),
        )
        lo, hi = float(determinant[0]), float(determinant[1])
        if hi < 0:
            return ContactIndicatrix("singular_point", 4, lo, hi)
        if lo > 0:
            return ContactIndicatrix("tangent_point", 0, lo, hi)
        unresolved = ContactIndicatrix("unresolved", -1, lo, hi)
    return unresolved
