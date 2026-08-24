#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._structured import FiniteVolumeDiscretization


FaceVelocity = tuple[Array, ...]
MomentumPredictor = Callable[[Array, FaceVelocity, Any], FaceVelocity]


def _difference(integrated: Array, axis: int, periodic: bool, /) -> Array:
    if periodic:
        return jnp.roll(integrated, -1, axis=axis) - integrated
    lower = [slice(None)] * integrated.ndim
    upper = [slice(None)] * integrated.ndim
    lower[axis] = slice(0, integrated.shape[axis] - 1)
    upper[axis] = slice(1, integrated.shape[axis])
    return integrated[tuple(upper)] - integrated[tuple(lower)]


class PressureProjectionResult(StrictModule):
    pressure: Array
    velocity: FaceVelocity
    divergence_before: Array
    divergence_after: Array
    pressure_residual: Array
    compatible_rhs: Array
    converged: Array


class MACPressureProjectionPlan(StrictModule, NonTrainableState):
    """Compatible face-velocity/cell-pressure projection on a tensor grid."""

    discretization: FiniteVolumeDiscretization
    density: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteVolumeDiscretization,
        /,
        *,
        density: float = 1.0,
        tolerance: float = 1e-8,
        maximum_iterations: int = 500,
    ):
        if not isinstance(discretization, FiniteVolumeDiscretization):
            raise TypeError("Pressure projection requires finite-volume geometry.")
        density_ = float(density)
        tolerance_ = float(tolerance)
        iterations = int(maximum_iterations)
        if (
            not np.isfinite(density_)
            or density_ <= 0.0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or iterations <= 0
        ):
            raise ValueError("Projection density, tolerance, and iterations are invalid.")
        self.discretization = discretization
        self.density = density_
        self.tolerance = tolerance_
        self.maximum_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-pressure-projection",
                "discretization": discretization.prepared_id,
                "density": density_,
                "tolerance": tolerance_,
                "maximum_iterations": iterations,
            }
        )

    def _validate_velocity(self, velocity: FaceVelocity, /) -> FaceVelocity:
        values = tuple(jnp.asarray(component) for component in velocity)
        if len(values) != len(self.discretization.cell_shape):
            raise ValueError("MAC velocity requires one normal component per axis.")
        for axis, component in enumerate(values):
            if component.shape != self.discretization.face_layouts[axis].shape:
                raise ValueError("MAC velocity component must match its face layout.")
        return values

    def divergence(self, velocity: FaceVelocity, /) -> Array:
        values = self._validate_velocity(velocity)
        divergence = jnp.zeros(self.discretization.cell_shape)
        for axis, component in enumerate(values):
            integrated = component * self.discretization.face_measures[axis]
            divergence = (
                divergence
                + _difference(
                    integrated,
                    axis,
                    self.discretization.grid.structured_axes[axis].periodic,
                )
                / self.discretization.cell_volumes
            )
        return divergence

    def gradient(self, pressure: ArrayLike, /) -> FaceVelocity:
        value = jnp.asarray(pressure)
        if value.shape != self.discretization.cell_shape:
            raise ValueError("Pressure must match the finite-volume cell shape.")
        output = []
        for axis, structured_axis in enumerate(self.discretization.grid.structured_axes):
            moved = jnp.moveaxis(value, axis, 0)
            centers = structured_axis.interval_centers
            if structured_axis.periodic:
                period = structured_axis.bounds[1] - structured_axis.bounds[0]
                previous = jnp.roll(moved, 1, axis=0)
                previous_centers = jnp.roll(centers, 1).at[0].add(-period)
                distance = centers - previous_centers
                gradient = (moved - previous) / distance.reshape(
                    (distance.size,) + (1,) * (moved.ndim - 1)
                )
            else:
                if moved.shape[0] == 1:
                    gradient = jnp.zeros((2,) + moved.shape[1:], dtype=moved.dtype)
                else:
                    interior = (moved[1:] - moved[:-1]) / (
                        centers[1:] - centers[:-1]
                    ).reshape((-1,) + (1,) * (moved.ndim - 1))
                    gradient = jnp.concatenate(
                        (jnp.zeros_like(moved[:1]), interior, jnp.zeros_like(moved[:1])),
                        axis=0,
                    )
            output.append(jnp.moveaxis(gradient, 0, axis))
        return tuple(output)

    def laplacian(self, pressure: ArrayLike, /) -> Array:
        return self.divergence(self.gradient(pressure))

    def project(
        self,
        velocity: FaceVelocity,
        step_size: ArrayLike,
        /,
        *,
        initial_pressure: ArrayLike | None = None,
    ) -> PressureProjectionResult:
        values = self._validate_velocity(velocity)
        dt = jnp.asarray(step_size).reshape(())
        divergence_before = self.divergence(values)
        volume = self.discretization.cell_volumes
        mean = jnp.sum(volume * divergence_before) / jnp.sum(volume)
        compatible_divergence = divergence_before - mean
        rhs = -(self.density / dt) * compatible_divergence
        initial = (
            jnp.zeros(self.discretization.cell_shape, dtype=divergence_before.dtype)
            if initial_pressure is None
            else jnp.asarray(initial_pressure)
        )
        if initial.shape != self.discretization.cell_shape:
            raise ValueError("Initial pressure must match the cell shape.")

        def positive_laplacian(pressure: Array) -> Array:
            return -self.laplacian(pressure)

        rhs_norm = jnp.sqrt(jnp.sum(volume * rhs**2))

        def solve_pressure(_):
            return jax.scipy.sparse.linalg.cg(
                positive_laplacian,
                rhs,
                x0=initial,
                tol=self.tolerance,
                maxiter=self.maximum_iterations,
            )[0]

        pressure = jax.lax.cond(
            rhs_norm > self.tolerance,
            solve_pressure,
            lambda _: jnp.zeros_like(initial),
            operand=None,
        )
        pressure = pressure - jnp.sum(volume * pressure) / jnp.sum(volume)
        pressure_gradient = self.gradient(pressure)
        corrected = tuple(
            component - (dt / self.density) * gradient
            for component, gradient in zip(values, pressure_gradient, strict=True)
        )
        divergence_after = self.divergence(corrected)
        residual = positive_laplacian(pressure) - rhs
        residual_norm = jnp.sqrt(jnp.sum(volume * residual**2))
        converged = residual_norm <= self.tolerance * jnp.maximum(rhs_norm, 1.0)
        return PressureProjectionResult(
            pressure=pressure,
            velocity=corrected,
            divergence_before=divergence_before,
            divergence_after=divergence_after,
            pressure_residual=residual,
            compatible_rhs=rhs,
            converged=converged,
        )


class PressureCorrectionResult(StrictModule):
    velocity: FaceVelocity
    pressure: Array
    divergence_history: Array
    converged: Array


class FunctionalPressureCorrectionPlan(StrictModule, NonTrainableState):
    """Fixed-count predictor/projection correction loop."""

    projection: MACPressureProjectionPlan
    correctors: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, projection: MACPressureProjectionPlan, correctors: int = 2, /):
        if not isinstance(projection, MACPressureProjectionPlan):
            raise TypeError("projection must be a MACPressureProjectionPlan.")
        correctors_ = int(correctors)
        if correctors_ <= 0:
            raise ValueError("correctors must be positive.")
        self.projection = projection
        self.correctors = correctors_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "functional-pressure-correction",
                "projection": projection.plan_id,
                "correctors": correctors_,
            }
        )

    def advance(
        self,
        time: Array,
        velocity: FaceVelocity,
        step_size: ArrayLike,
        predictor: MomentumPredictor,
        args: Any = None,
        /,
    ) -> PressureCorrectionResult:
        if not callable(predictor):
            raise TypeError("predictor must be callable.")
        predicted = predictor(time, velocity, args)
        initial_pressure = jnp.zeros(self.projection.discretization.cell_shape)
        initial_history = jnp.zeros((self.correctors,))

        def body(index, carry):
            current_velocity, pressure, history, converged = carry
            result = self.projection.project(
                current_velocity,
                step_size,
                initial_pressure=pressure,
            )
            norm = jnp.sqrt(
                jnp.sum(
                    self.projection.discretization.cell_volumes
                    * result.divergence_after**2
                )
            )
            history = history.at[index].set(norm)
            return result.velocity, result.pressure, history, converged & result.converged

        corrected, pressure, history, converged = jax.lax.fori_loop(
            0,
            self.correctors,
            body,
            (predicted, initial_pressure, initial_history, jnp.asarray(True)),
        )
        return PressureCorrectionResult(
            velocity=corrected,
            pressure=pressure,
            divergence_history=history,
            converged=converged,
        )


__all__ = [
    "FunctionalPressureCorrectionPlan",
    "MACPressureProjectionPlan",
    "PressureCorrectionResult",
    "PressureProjectionResult",
]
