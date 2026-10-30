#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ...optim import (
    BoundedLevenbergMarquardt,
    Bounds,
    least_squares,
    LevenbergMarquardt,
    NonlinearLeastSquaresProblem,
    OptimizationTermination,
)


@dataclass(frozen=True, slots=True)
class NativeResidualSolve:
    """Native optimizer output interpreted as a constraint-root attempt."""

    coordinates: Array
    residual: Array
    residual_norm: Array
    converged: Array
    status: Array
    iterations: Array


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
    residual_ = jnp.asarray(residual(initial_), dtype=initial_.dtype).reshape((-1,))
    if residual_.size == 0:
        raise ValueError("A design residual solve requires at least one equation.")

    if bounds is None:
        problem = NonlinearLeastSquaresProblem(
            lambda coordinates, _args: residual(coordinates),
            problem_id=problem_id,
        )
        method = LevenbergMarquardt(initial_damping=damping)
    else:
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
        problem = NonlinearLeastSquaresProblem(
            lambda coordinates, _args: residual(coordinates),
            bounds=Bounds(jnp.asarray(lower), jnp.asarray(upper)),
            problem_id=problem_id,
        )
        method = BoundedLevenbergMarquardt(damping=damping)

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


__all__ = ["NativeResidualSolve", "solve_design_residual"]
