#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._tree_math import tree_norm

from ...optim import (
    BoundedLevenbergMarquardt,
    Bounds,
    least_squares,
    LevenbergMarquardt,
    NonlinearLeastSquaresProblem,
    OptimizationStatus,
    OptimizationTermination,
)
from ...optim._least_squares import _prepare_residual_model

if TYPE_CHECKING:
    from ._reduced import _SketchGroundedRigidParameterization
    from ._schema import DesignState
    from ._sketch import Sketch
    from ._sketch_certificate import LineCircleConstruction, TwoCircleConstruction


@dataclass(frozen=True, slots=True)
class NativeResidualSolve:
    """Native optimizer output interpreted as a constraint-root attempt."""

    coordinates: Array
    residual: Array
    residual_norm: Array
    converged: Array
    status: Array
    iterations: Array


def _validate_solve_inputs(
    residual: Callable[[Array], Array],
    initial: ArrayLike,
    max_iterations: int,
    tolerance: float,
    damping: float,
    problem_id: str,
    bounds: tuple[ArrayLike, ArrayLike] | None,
) -> tuple[Array, tuple[np.ndarray, np.ndarray] | None]:
    if not callable(residual):
        raise TypeError("residual must be callable.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be finite and positive.")
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError("problem_id must be a non-empty string.")
    initial_ = jnp.asarray(initial, dtype=jnp.float64).reshape((-1,))
    if initial_.size == 0:
        raise ValueError("A design residual solve requires at least one variable.")
    if bounds is None:
        return initial_, None
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        raise TypeError("bounds must be a (lower, upper) pair or None.")
    lower = np.asarray(bounds[0], dtype=np.float64).reshape((-1,))
    upper = np.asarray(bounds[1], dtype=np.float64).reshape((-1,))
    initial_host = np.asarray(initial_, dtype=np.float64)
    if lower.shape != initial_host.shape or upper.shape != initial_host.shape:
        raise ValueError("Design solve bounds must match the variable vector.")
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
        raise ValueError("Design solve bounds must be ordered and contain no NaNs.")
    if np.any(initial_host < lower) or np.any(initial_host > upper):
        raise ValueError("The initial design state lies outside the solve bounds.")
    return initial_, (lower, upper)


def solve_design_residual(
    residual: Callable[[Array], Array],
    initial: ArrayLike,
    /,
    *,
    max_iterations: int,
    tolerance: float,
    damping: float,
    problem_id: str,
    bounds: tuple[ArrayLike, ArrayLike] | None = None,
) -> NativeResidualSolve:
    """Solve one real residual system with Phydrax's native least-squares methods."""
    initial_, validated_bounds = _validate_solve_inputs(
        residual, initial, max_iterations, tolerance, damping, problem_id, bounds
    )
    residual_ = jnp.asarray(residual(initial_), dtype=initial_.dtype).reshape((-1,))

    if validated_bounds is None:
        problem = NonlinearLeastSquaresProblem(
            lambda coordinates, _args: residual(coordinates),
            problem_id=problem_id,
        )
        method = LevenbergMarquardt(initial_damping=damping)
    else:
        lower, upper = validated_bounds
        problem = NonlinearLeastSquaresProblem(
            lambda coordinates, _args: residual(coordinates),
            bounds=Bounds(jnp.asarray(lower), jnp.asarray(upper)),
            problem_id=problem_id,
        )
        method = BoundedLevenbergMarquardt(damping=damping)
    if residual_.size == 0:
        finite = jnp.all(jnp.isfinite(initial_))
        return NativeResidualSolve(
            initial_,
            residual_,
            jnp.asarray(0.0, dtype=initial_.dtype),
            finite,
            jnp.where(
                finite,
                int(OptimizationStatus.SUCCESS),
                int(OptimizationStatus.NONFINITE_EVALUATION),
            ).astype(jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

    result = least_squares(
        problem,
        initial_,
        method=method,
        termination=OptimizationTermination(
            absolute_optimality=tolerance,
            relative_optimality=0.0,
            maximum_steps=max_iterations,
        ),
    )
    output_residual = jnp.asarray(result.residual, dtype=initial_.dtype).reshape((-1,))
    residual_norm = jnp.linalg.norm(output_residual)
    converged = result.successful & (residual_norm <= tolerance)
    return NativeResidualSolve(
        coordinates=jnp.asarray(result.parameters, dtype=initial_.dtype).reshape((-1,)),
        residual=output_residual,
        residual_norm=residual_norm,
        converged=converged,
        status=jnp.asarray(result.status, dtype=jnp.int32).reshape(()),
        iterations=jnp.asarray(result.diagnostics.iterations, dtype=jnp.int32).reshape(
            ()
        ),
    )


def _full_residual_optimal(
    residual: Callable[[Array], Array],
    candidate: Array,
    tolerance: float,
    bounds: tuple[ArrayLike, ArrayLike] | None,
    /,
) -> bool:
    """Apply the native least-squares stationarity criterion to the full system."""
    model = _prepare_residual_model(residual, candidate)
    gradient = model.gradient
    if bounds is not None:
        gradient = Bounds(
            jnp.asarray(bounds[0]), jnp.asarray(bounds[1])
        ).projected_gradient(candidate, gradient)
    optimality = tree_norm(gradient)
    return bool(
        np.asarray(
            jnp.isfinite(model.objective)
            & jnp.all(jnp.isfinite(model.residual))
            & jnp.all(jnp.isfinite(model.gradient))
            & jnp.isfinite(optimality)
            & (optimality <= tolerance)
        )
    )


def prepare_sketch_fixed_coordinates(
    sketch: Sketch, dimensions: DesignState, /
) -> tuple[tuple[int, ...], Array]:
    """Propagate structurally exact scalar equalities from fixed sketch values.

    Only equalities whose operands are already fixed enter this reduction. All
    original residuals remain present for root acceptance.
    """
    from ._sketch import Coincident, FixedPoint, Horizontal, Midpoint, Radius, Vertical

    point_size = sketch.points.size
    lines = np.asarray(sketch.lines, dtype=np.int32)
    bindings = {
        binding.constraint_index: binding for binding in sketch.dimension_bindings
    }
    fixed: dict[int, Array] = {}

    def place(index: int, value: Array) -> bool:
        if index in fixed or not bool(np.all(np.isfinite(np.asarray(value)))):
            return False
        fixed[index] = value
        return True

    for index, constraint in enumerate(sketch.constraints):
        if isinstance(constraint, FixedPoint):
            for axis in range(2):
                place(2 * constraint.point + axis, constraint.target[axis])
        elif isinstance(constraint, Radius):
            binding = bindings.get(index)
            value = (
                constraint.radius
                if binding is None
                else dimensions.values[binding.state_index]
            )
            if float(np.asarray(value)) >= np.finfo(np.float64).tiny:
                place(point_size + constraint.circle, value)

    changed = True
    while changed:
        changed = False
        for constraint in sketch.constraints:
            if isinstance(constraint, Coincident):
                pairs = tuple(
                    (
                        2 * constraint.first_point + axis,
                        2 * constraint.second_point + axis,
                    )
                    for axis in range(2)
                )
            elif isinstance(constraint, (Horizontal, Vertical)):
                first, second = lines[constraint.line]
                axis = 1 if isinstance(constraint, Horizontal) else 0
                pairs = ((2 * first + axis, 2 * second + axis),)
            else:
                pairs = ()
            for first, second in pairs:
                if first in fixed:
                    changed |= place(second, fixed[first])
                if second in fixed:
                    changed |= place(first, fixed[second])
            if isinstance(constraint, Midpoint):
                first, second = lines[constraint.line]
                for axis in range(2):
                    a, b = 2 * first + axis, 2 * second + axis
                    target = 2 * constraint.point + axis
                    if a in fixed and b in fixed:
                        midpoint = 0.5 * (fixed[a] + fixed[b])
                        rounded = float(np.asarray(midpoint))
                        first_value = float(np.asarray(fixed[a]))
                        second_value = float(np.asarray(fixed[b]))
                        if np.isfinite(rounded) and (
                            Fraction.from_float(rounded)
                            == (
                                Fraction.from_float(first_value)
                                + Fraction.from_float(second_value)
                            )
                            / 2
                        ):
                            changed |= place(target, midpoint)

    indices = tuple(int(index) for index in sorted(fixed))
    values = (
        jnp.stack(tuple(fixed[index] for index in indices))
        if indices
        else jnp.empty((0,), dtype=sketch.points.dtype)
    )
    return indices, values


def seed_sketch_circle_intersections(
    sketch: Sketch,
    dimensions: DesignState,
    reference: Array,
    pinned_indices: tuple[int, ...],
    pinned_values: Array,
    /,
    *,
    initial_seed: Array | None = None,
) -> tuple[Array, tuple[tuple[int, TwoCircleConstruction | LineCircleConstruction], ...]]:
    """Prepare circle and axis-aligned line-circle seeds and branch constructions.

    A construction is admitted only from exact input and branch predicates.
    Tangent and uncertain numerical placements remain seeds.
    """
    from ._sketch import Horizontal, PointDistance, Vertical
    from ._sketch_certificate import (
        LineCircleConstruction,
        certify_line_circle_construction,
        certify_two_circle_construction,
    )

    fixed = dict(zip(pinned_indices, np.asarray(pinned_values), strict=True))
    bindings = {
        binding.constraint_index: binding for binding in sketch.dimension_bindings
    }
    loci: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for index, constraint in enumerate(sketch.constraints):
        if not isinstance(constraint, PointDistance):
            continue
        binding = bindings.get(index)
        radius = float(
            np.asarray(
                constraint.distance
                if binding is None
                else dimensions.values[binding.state_index]
            )
        )
        if not np.isfinite(radius) or radius < 0.0:
            continue
        for point, center in (
            (constraint.first_point, constraint.second_point),
            (constraint.second_point, constraint.first_point),
        ):
            if 2 * point in fixed or 2 * point + 1 in fixed:
                continue
            if 2 * center not in fixed or 2 * center + 1 not in fixed:
                continue
            center_value = np.asarray(
                [fixed[2 * center], fixed[2 * center + 1]], dtype=np.float64
            )
            loci.setdefault(point, []).append((center, center_value, radius))

    reference_host = np.asarray(reference, dtype=np.float64)
    seed = np.asarray(
        reference if initial_seed is None else initial_seed, dtype=np.float64
    ).copy()
    constructions: list[tuple[int, TwoCircleConstruction | LineCircleConstruction]] = []
    for point, circles in loci.items():
        placed = False
        for first_index, (first_id, first, first_radius) in enumerate(circles):
            for second_id, second, second_radius in circles[first_index + 1 :]:
                delta = second - first
                distance = float(np.linalg.norm(delta))
                if not np.isfinite(distance) or distance == 0.0:
                    continue
                along = (
                    first_radius * first_radius
                    - second_radius * second_radius
                    + distance * distance
                ) / (2.0 * distance)
                height_squared = first_radius * first_radius - along * along
                if not np.isfinite(height_squared) or height_squared < 0.0:
                    continue
                direction = delta / distance
                base = first + along * direction
                normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
                offset = np.sqrt(height_squared) * normal
                candidates = (base + offset, base - offset)
                before = reference_host[2 * point : 2 * point + 2]
                witness = certify_two_circle_construction(
                    first,
                    second,
                    first_radius,
                    second_radius,
                    before,
                    reference_centers=(
                        reference_host[2 * first_id : 2 * first_id + 2],
                        reference_host[2 * second_id : 2 * second_id + 2],
                    ),
                )
                candidate = (
                    candidates[0 if witness.branch_sign > 0 else 1]
                    if witness is not None
                    else min(candidates, key=lambda value: np.linalg.norm(value - before))
                )
                if np.all(np.isfinite(candidate)):
                    seed[2 * point : 2 * point + 2] = candidate
                    if witness is not None:
                        constructions.append((int(point), witness))
                    placed = True
                    break
            if placed:
                break
    # Fixed-coordinate propagation identifies the line coordinate. Require an
    # actual horizontal/vertical constraint as well as a fixed opposite endpoint.
    lines = np.asarray(sketch.lines, dtype=np.int32)
    line_points: dict[int, int] = {}
    for constraint in sketch.constraints:
        if not isinstance(constraint, (Horizontal, Vertical)):
            continue
        axis = 1 if isinstance(constraint, Horizontal) else 0
        first, second = lines[constraint.line]
        for point, other in ((int(first), int(second)), (int(second), int(first))):
            if (
                2 * point + axis in fixed
                and 2 * point + 1 - axis not in fixed
                and 2 * other + axis in fixed
            ):
                line_points[point] = axis

    placed_points: set[int] = set()
    for index, constraint in enumerate(sketch.constraints):
        if not isinstance(constraint, PointDistance):
            continue
        binding = bindings.get(index)
        radius = float(
            np.asarray(
                constraint.distance
                if binding is None
                else dimensions.values[binding.state_index]
            )
        )
        if not np.isfinite(radius) or radius < 0.0:
            continue
        for point, center in (
            (constraint.first_point, constraint.second_point),
            (constraint.second_point, constraint.first_point),
        ):
            if point not in line_points or point in placed_points:
                continue
            if 2 * center not in fixed or 2 * center + 1 not in fixed:
                continue
            axis = line_points[point]
            free_axis = 1 - axis
            center_value = np.asarray(
                [fixed[2 * center], fixed[2 * center + 1]], dtype=np.float64
            )
            line_value = float(fixed[2 * point + axis])
            before = reference_host[2 * point : 2 * point + 2]
            old_center = reference_host[2 * center : 2 * center + 2]
            if not np.all(np.isfinite(center_value)) or not np.isfinite(line_value):
                continue
            rq = Fraction.from_float(radius)
            difference = Fraction.from_float(line_value) - Fraction.from_float(
                float(center_value[axis])
            )
            discriminant = rq * rq - difference * difference
            if discriminant < 0:
                continue
            witness = certify_line_circle_construction(
                center_value,
                radius,
                axis,
                line_value,
                before,
                reference_center=old_center,
            )
            sign = (
                witness.branch_sign
                if witness is not None
                else (1 if before[free_axis] >= old_center[free_axis] else -1)
            )
            try:
                offset = np.sqrt(float(discriminant))
            except OverflowError:
                continue
            candidate = center_value.copy()
            candidate[axis] = line_value
            candidate[free_axis] += sign * offset
            if np.all(np.isfinite(candidate)):
                seed[2 * point : 2 * point + 2] = candidate
                if witness is not None:
                    constructions.append((int(point), witness))
                placed_points.add(point)
    return _propagate_rational_circle_constructions(
        sketch, dimensions, reference, pinned_indices, pinned_values, seed, constructions
    )


def _rational_construction_point(
    witness: TwoCircleConstruction | LineCircleConstruction, /
) -> tuple[Fraction, Fraction] | None:
    """Recover the exact point only when its radical is rational."""
    from ._sketch_certificate import LineCircleConstruction

    radicand = (
        witness.discriminant
        if isinstance(witness, LineCircleConstruction)
        else witness.squared_height_numerator
    )
    numerator = isqrt(radicand.numerator)
    denominator = isqrt(radicand.denominator)
    if (
        numerator * numerator != radicand.numerator
        or denominator * denominator != radicand.denominator
    ):
        return None
    root = Fraction(numerator, denominator) * witness.branch_sign
    if isinstance(witness, LineCircleConstruction):
        values = list(witness.center)
        values[witness.fixed_axis] = witness.line_coordinate
        values[1 - witness.fixed_axis] += root
        return values[0], values[1]
    a, b = witness.first, witness.second
    dx, dy = b[0] - a[0], b[1] - a[1]
    q = dx * dx + dy * dy
    k = (q + witness.first_radius**2 - witness.second_radius**2) / 2
    return a[0] + (k * dx - root * dy) / q, a[1] + (k * dy + root * dx) / q


def _propagate_rational_circle_constructions(
    sketch: Sketch,
    dimensions: DesignState,
    reference: Array,
    pinned_indices: tuple[int, ...],
    pinned_values: Array,
    seed: np.ndarray,
    constructions: list[tuple[int, TwoCircleConstruction | LineCircleConstruction]],
    /,
) -> tuple[Array, tuple[tuple[int, TwoCircleConstruction | LineCircleConstruction], ...]]:
    """Propagate exact rational constructed anchors without a rounded shadow."""
    from ._sketch import PointDistance
    from ._sketch_certificate import TwoCircleConstruction

    fixed = {
        index: Fraction.from_float(float(value))
        for index, value in zip(pinned_indices, np.asarray(pinned_values), strict=True)
    }
    known = {point for point, _ in constructions}
    for point, witness in constructions:
        exact = _rational_construction_point(witness)
        if exact is not None:
            fixed[2 * point], fixed[2 * point + 1] = exact
    bindings = {item.constraint_index: item for item in sketch.dimension_bindings}
    reference_host = np.asarray(reference, dtype=np.float64)

    while True:
        loci: dict[int, list[tuple[int, tuple[Fraction, Fraction], Fraction]]] = {}
        for index, constraint in enumerate(sketch.constraints):
            if not isinstance(constraint, PointDistance):
                continue
            binding = bindings.get(index)
            value = (
                constraint.distance
                if binding is None
                else dimensions.values[binding.state_index]
            )
            radius_float = float(np.asarray(value))
            if not np.isfinite(radius_float) or radius_float <= 0:
                continue
            radius = Fraction.from_float(radius_float)
            for point, center in (
                (constraint.first_point, constraint.second_point),
                (constraint.second_point, constraint.first_point),
            ):
                if point in known or 2 * point in fixed or 2 * point + 1 in fixed:
                    continue
                if 2 * center in fixed and 2 * center + 1 in fixed:
                    loci.setdefault(point, []).append(
                        (center, (fixed[2 * center], fixed[2 * center + 1]), radius)
                    )
        admitted_rational = False
        for point, circles in loci.items():
            placed = False
            for first_index, (first_id, first, first_radius) in enumerate(circles):
                for second_id, second, second_radius in circles[first_index + 1 :]:
                    dx, dy = second[0] - first[0], second[1] - first[1]
                    q = dx * dx + dy * dy
                    if q == 0:
                        continue
                    k = (q + first_radius**2 - second_radius**2) / 2
                    height = first_radius**2 * q - k**2
                    old_first = reference_host[2 * first_id : 2 * first_id + 2]
                    old_second = reference_host[2 * second_id : 2 * second_id + 2]
                    old_point = reference_host[2 * point : 2 * point + 2]
                    if not np.all(
                        np.isfinite(np.concatenate((old_first, old_second, old_point)))
                    ):
                        continue
                    ax, ay = (Fraction.from_float(float(value)) for value in old_first)
                    bx, by = (Fraction.from_float(float(value)) for value in old_second)
                    px, py = (Fraction.from_float(float(value)) for value in old_point)
                    side = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
                    if height <= 0 or side == 0:
                        continue
                    witness = TwoCircleConstruction(
                        first,
                        second,
                        first_radius,
                        second_radius,
                        height,
                        1 if side > 0 else -1,
                    )
                    exact = _rational_construction_point(witness)
                    try:
                        if exact is not None:
                            candidate = np.asarray(
                                [float(exact[0]), float(exact[1])], dtype=np.float64
                            )
                        else:
                            root = np.sqrt(float(height)) * witness.branch_sign
                            candidate = np.asarray(
                                [
                                    float(first[0] + k * dx / q) - root * float(dy / q),
                                    float(first[1] + k * dy / q) + root * float(dx / q),
                                ],
                                dtype=np.float64,
                            )
                    except OverflowError:
                        continue
                    if not np.all(np.isfinite(candidate)):
                        continue
                    seed[2 * point : 2 * point + 2] = candidate
                    constructions.append((int(point), witness))
                    known.add(point)
                    if exact is not None:
                        fixed[2 * point], fixed[2 * point + 1] = exact
                        admitted_rational = True
                    placed = True
                    break
                if placed:
                    break
        if not admitted_rational:
            break
    return jnp.asarray(seed, dtype=reference.dtype), tuple(constructions)


def solve_prepared_sketch_residual(
    residual: Callable[[Array], Array],
    initial: Array,
    prepared_seed: Array,
    pinned_indices: tuple[int, ...],
    pinned_values: Array,
    constructions: tuple[tuple[int, TwoCircleConstruction | LineCircleConstruction], ...],
    rigid_reductions: tuple[_SketchGroundedRigidParameterization, ...],
    /,
    *,
    max_iterations: int,
    tolerance: float,
    damping: float,
    problem_id: str,
    bounds: tuple[ArrayLike, ArrayLike] | None,
) -> tuple[NativeResidualSolve, bool]:
    """Try a root-preserving coordinate reduction, then retain the full problem.

    Pinning equalities preserve their zero set, not an inconsistent weighted
    least-squares objective. A reduced candidate must satisfy both the full
    residual and the native full-system stationarity criterion; otherwise the
    original native solve runs from the original reference coordinates.
    """
    from ._sketch_certificate import LineCircleConstruction

    if not pinned_indices and not constructions and not rigid_reductions:
        return solve_design_residual(
            residual,
            prepared_seed,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
            problem_id=problem_id,
            bounds=bounds,
        ), False
    mask = np.ones(initial.size, dtype=np.bool_)
    mask[np.asarray(pinned_indices, dtype=np.int32)] = False
    for point, _ in constructions:
        mask[2 * point : 2 * point + 2] = False
    for cluster in rigid_reductions:
        for point in cluster.point_indices:
            mask[2 * point : 2 * point + 2] = False
    all_fixed = not np.any(mask) and not rigid_reductions
    free = np.flatnonzero(mask)
    pinned = jnp.asarray(pinned_indices, dtype=jnp.int32)
    free_index = jnp.asarray(free, dtype=jnp.int32)
    seed = prepared_seed.at[pinned].set(pinned_values)

    def expand(coordinates: Array) -> Array:
        result = seed.at[free_index].set(coordinates[: free.size])
        for point, witness in constructions:
            if isinstance(witness, LineCircleConstruction):
                center = jnp.asarray(
                    tuple(map(float, witness.center)), dtype=initial.dtype
                )
                value = center.at[witness.fixed_axis].set(float(witness.line_coordinate))
                value = value.at[1 - witness.fixed_axis].add(
                    witness.branch_sign
                    * jnp.sqrt(
                        jnp.asarray(float(witness.discriminant), dtype=initial.dtype)
                    )
                )
                result = result.at[2 * point : 2 * point + 2].set(value)
                continue
            first = jnp.asarray(tuple(map(float, witness.first)), dtype=initial.dtype)
            second = jnp.asarray(tuple(map(float, witness.second)), dtype=initial.dtype)
            delta = second - first
            q = jnp.dot(delta, delta)
            first_radius = jnp.asarray(float(witness.first_radius), dtype=initial.dtype)
            second_radius = jnp.asarray(float(witness.second_radius), dtype=initial.dtype)
            k = (q + first_radius**2 - second_radius**2) / 2.0
            height = first_radius**2 * q - k**2
            normal = jnp.stack((-delta[1], delta[0]))
            value = (
                first + (k * delta + witness.branch_sign * jnp.sqrt(height) * normal) / q
            )
            result = result.at[2 * point : 2 * point + 2].set(value)
        for index, cluster in enumerate(rigid_reductions):
            result = cluster.expand(result, coordinates[free.size + index])
        return result

    if all_fixed:
        _validate_solve_inputs(
            residual, initial, max_iterations, tolerance, damping, problem_id, bounds
        )
        candidate = expand(seed[free_index])
        output_residual = jnp.asarray(residual(candidate), dtype=initial.dtype).reshape(
            (-1,)
        )
        residual_norm = jnp.linalg.norm(output_residual)
        within_bounds = bounds is None or bool(
            np.all(
                (np.asarray(candidate) >= np.asarray(bounds[0]))
                & (np.asarray(candidate) <= np.asarray(bounds[1]))
            )
        )
        if (
            within_bounds
            and bool(np.all(np.isfinite(np.asarray(candidate))))
            and bool(np.isfinite(np.asarray(residual_norm)))
            and bool(np.asarray(residual_norm <= tolerance))
            and _full_residual_optimal(residual, candidate, tolerance, bounds)
        ):
            return NativeResidualSolve(
                candidate,
                output_residual,
                residual_norm,
                jnp.asarray(True),
                jnp.asarray(int(OptimizationStatus.SUCCESS), dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            ), True
        return solve_design_residual(
            residual,
            initial,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
            problem_id=problem_id,
            bounds=bounds,
        ), False

    reduced_bounds = None
    if bounds is not None:
        reduced_bounds = (
            jnp.concatenate(
                (
                    jnp.asarray(bounds[0])[free_index],
                    jnp.full((len(rigid_reductions),), -jnp.inf, dtype=initial.dtype),
                )
            ),
            jnp.concatenate(
                (
                    jnp.asarray(bounds[1])[free_index],
                    jnp.full((len(rigid_reductions),), jnp.inf, dtype=initial.dtype),
                )
            ),
        )
    reduced_initial = jnp.concatenate(
        (seed[free_index], jnp.zeros((len(rigid_reductions),), dtype=initial.dtype))
    )
    attempt = solve_design_residual(
        lambda coordinates: residual(expand(coordinates)),
        reduced_initial,
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
        problem_id=problem_id,
        bounds=reduced_bounds,
    )
    candidate = expand(attempt.coordinates)
    within_bounds = bounds is None or bool(
        np.all(
            (np.asarray(candidate) >= np.asarray(bounds[0]))
            & (np.asarray(candidate) <= np.asarray(bounds[1]))
        )
    )
    if (
        bool(np.asarray(attempt.converged))
        and within_bounds
        and bool(np.all(np.isfinite(np.asarray(candidate))))
        and _full_residual_optimal(residual, candidate, tolerance, bounds)
    ):
        return NativeResidualSolve(
            candidate,
            attempt.residual,
            attempt.residual_norm,
            attempt.converged,
            attempt.status,
            attempt.iterations,
        ), True
    return solve_design_residual(
        residual,
        initial,
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
        problem_id=problem_id,
        bounds=bounds,
    ), False


__all__ = ["NativeResidualSolve", "solve_design_residual"]
