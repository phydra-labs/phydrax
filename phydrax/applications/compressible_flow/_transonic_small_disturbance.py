#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...nonlinear import implicit_root, NonlinearSystemProblem, NonlinearTermination


class TransonicSmallDisturbanceResult(StrictModule):
    potential: Array
    residual: Array
    pressure_coefficient: Array
    maximum_residual: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class TransonicSmallDisturbancePlan(StrictModule, NonTrainableState):
    """Steady two-dimensional inviscid transonic small-disturbance model."""

    x_coordinates: Array
    y_coordinates: Array
    free_stream_mach: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    nonlinear_coefficient: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        x_coordinates: ArrayLike,
        y_coordinates: ArrayLike,
        free_stream_mach: float,
        /,
        *,
        gamma: float = 1.4,
        maximum_steps: int = 40,
        residual_tolerance: float = 1.0e-9,
    ):
        x = np.asarray(x_coordinates, dtype=float)
        y = np.asarray(y_coordinates, dtype=float)
        mach = float(free_stream_mach)
        gamma_ = float(gamma)
        steps = int(maximum_steps)
        tolerance = float(residual_tolerance)
        if (
            x.ndim != 1
            or y.ndim != 1
            or x.size < 5
            or y.size < 5
            or np.any(~np.isfinite(x))
            or np.any(~np.isfinite(y))
            or np.any(np.diff(x) <= 0.0)
            or np.any(np.diff(y) <= 0.0)
            or not np.allclose(np.diff(x), np.diff(x)[0], rtol=1.0e-10)
            or not np.allclose(np.diff(y), np.diff(y)[0], rtol=1.0e-10)
            or not 0.5 < mach < 1.0
            or not np.isfinite(gamma_)
            or gamma_ <= 1.0
            or steps <= 0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("TSD grid, flow state, or nonlinear controls are invalid.")
        self.x_coordinates = jnp.asarray(x)
        self.y_coordinates = jnp.asarray(y)
        self.free_stream_mach = mach
        self.gamma = gamma_
        self.nonlinear_coefficient = (gamma_ + 1.0) * mach * mach
        self.maximum_steps = steps
        self.residual_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "transonic-small-disturbance",
                "x": array_tree_fingerprint(x),
                "y": array_tree_fingerprint(y),
                "free_stream_mach": mach,
                "gamma": gamma_,
                "maximum_steps": steps,
                "residual_tolerance": tolerance,
                "boundary": "dirichlet",
            }
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.x_coordinates.size, self.y_coordinates.size

    def residual(
        self,
        potential: ArrayLike,
        forcing: ArrayLike,
        boundary_values: ArrayLike,
        /,
    ) -> Array:
        value = jnp.asarray(potential)
        forcing_ = jnp.asarray(forcing, dtype=value.dtype)
        boundary = jnp.asarray(boundary_values, dtype=value.dtype)
        if (
            value.shape != self.shape
            or forcing_.shape != self.shape
            or boundary.shape != self.shape
        ):
            raise ValueError(
                "TSD potential, forcing, and boundary arrays must match the grid."
            )
        dx = self.x_coordinates[1] - self.x_coordinates[0]
        dy = self.y_coordinates[1] - self.y_coordinates[0]
        phi_x = (value[2:, 1:-1] - value[:-2, 1:-1]) / (2.0 * dx)
        phi_xx = (value[2:, 1:-1] - 2.0 * value[1:-1, 1:-1] + value[:-2, 1:-1]) / (
            dx * dx
        )
        phi_yy = (value[1:-1, 2:] - 2.0 * value[1:-1, 1:-1] + value[1:-1, :-2]) / (
            dy * dy
        )
        coefficient = 1.0 - self.free_stream_mach**2 - self.nonlinear_coefficient * phi_x
        interior = coefficient * phi_xx + phi_yy - forcing_[1:-1, 1:-1]
        result = value - boundary
        return result.at[1:-1, 1:-1].set(interior)

    def pressure_coefficient(self, potential: ArrayLike, /) -> Array:
        value = jnp.asarray(potential)
        dx = self.x_coordinates[1] - self.x_coordinates[0]
        derivative = jnp.zeros_like(value)
        derivative = derivative.at[1:-1].set((value[2:] - value[:-2]) / (2.0 * dx))
        derivative = derivative.at[0].set((value[1] - value[0]) / dx)
        derivative = derivative.at[-1].set((value[-1] - value[-2]) / dx)
        return -2.0 * derivative

    def solve(
        self,
        initial_potential: ArrayLike,
        forcing: ArrayLike,
        boundary_values: ArrayLike,
        /,
    ) -> TransonicSmallDisturbanceResult:
        initial = jnp.asarray(initial_potential)
        forcing_ = jnp.asarray(forcing, dtype=initial.dtype)
        boundary = jnp.asarray(boundary_values, dtype=initial.dtype)
        problem = NonlinearSystemProblem(
            lambda potential, arguments: self.residual(
                potential, arguments[0], arguments[1]
            ),
            problem_id=f"tsd:{self.plan_id}",
        )
        potential = implicit_root(
            problem,
            initial,
            termination=NonlinearTermination(
                absolute_residual=self.residual_tolerance,
                relative_residual=0.0,
                maximum_steps=self.maximum_steps,
                maximum_evaluations=4 * self.maximum_steps,
                maximum_linear_iterations=8 * np.prod(self.shape),
            ),
            args=(forcing_, boundary),
        )
        residual = self.residual(potential, forcing_, boundary)
        pressure = self.pressure_coefficient(potential)
        maximum = jnp.max(jnp.abs(residual))
        finite = (
            jnp.all(jnp.isfinite(potential))
            & jnp.all(jnp.isfinite(residual))
            & jnp.all(jnp.isfinite(pressure))
        )
        return TransonicSmallDisturbanceResult(
            potential,
            residual,
            pressure,
            maximum,
            finite,
            finite & (maximum <= self.residual_tolerance),
            self.plan_id,
        )


__all__ = ["TransonicSmallDisturbancePlan", "TransonicSmallDisturbanceResult"]
