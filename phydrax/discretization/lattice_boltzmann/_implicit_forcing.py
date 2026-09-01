#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


LocalRootResidual = Callable[[Array, Any], Array]
VelocityDependentAcceleration = Callable[[Array, Array, Array, Any], Array]


class LocalRootSolveResult(StrictModule):
    """Minimal, backend-neutral result returned by an injected local root solver."""

    value: Array
    residual: Array
    converged: Array
    iterations: Array

    def __init__(
        self,
        value: ArrayLike,
        residual: ArrayLike,
        converged: ArrayLike,
        iterations: ArrayLike,
        /,
    ):
        value_ = jnp.asarray(value)
        residual_ = jnp.asarray(residual)
        converged_ = jnp.asarray(converged, dtype=bool)
        iterations_ = jnp.asarray(iterations, dtype=jnp.int32)
        if residual_.shape != value_.shape:
            raise ValueError("Local root value and residual must have the same shape.")
        if converged_.shape not in ((), value_.shape[:-1]):
            raise ValueError(
                "Local root convergence must be scalar or one value per point."
            )
        if iterations_.shape not in ((), value_.shape[:-1]):
            raise ValueError(
                "Local root iterations must be scalar or one value per point."
            )
        self.value = value_
        self.residual = residual_
        self.converged = converged_
        self.iterations = iterations_


class LocalRootSolver(Protocol):
    """Injected generic local nonlinear solver; no LBM-specific solver is required."""

    def solve(
        self,
        residual: LocalRootResidual,
        initial_guess: Array,
        args: Any,
        /,
    ) -> LocalRootSolveResult: ...


class VelocityDependentAccelerationProblem(StrictModule):
    """Point-local half-force closure data in lattice units."""

    time: Array
    coordinates: Array
    density: Array
    raw_momentum: Array
    parameters: Any
    initial_velocity: Array

    def __init__(
        self,
        time: ArrayLike,
        coordinates: ArrayLike,
        density: ArrayLike,
        raw_momentum: ArrayLike,
        /,
        *,
        parameters: Any = None,
        initial_velocity: ArrayLike | None = None,
    ):
        time_ = jnp.asarray(time)
        coordinates_ = jnp.asarray(coordinates)
        density_ = jnp.asarray(density)
        momentum_ = jnp.asarray(raw_momentum)
        if time_.shape != ():
            raise ValueError("Velocity-dependent acceleration time must be scalar.")
        if coordinates_.ndim < 1 or coordinates_.shape[-1] not in (2, 3):
            raise ValueError(
                "coordinates must have a trailing two- or three-vector axis."
            )
        if density_.shape != coordinates_.shape[:-1]:
            raise ValueError("density must match the acceleration coordinate support.")
        if momentum_.shape != coordinates_.shape:
            raise ValueError("raw_momentum must match the acceleration coordinates.")
        initial = (
            momentum_ / jnp.where(density_ > 0.0, density_, 1.0)[..., None]
            if initial_velocity is None
            else jnp.asarray(initial_velocity, dtype=momentum_.dtype)
        )
        if initial.shape != coordinates_.shape:
            raise ValueError("initial_velocity must match raw_momentum.")
        self.time = time_
        self.coordinates = coordinates_
        self.density = density_
        self.raw_momentum = momentum_
        self.parameters = parameters
        self.initial_velocity = initial


class VelocityDependentAccelerationResult(StrictModule):
    """Certified implicit velocity, acceleration, and force-density closure."""

    velocity: Array
    acceleration: Array
    force_density: Array
    root: LocalRootSolveResult

    def __init__(
        self,
        velocity: Array,
        acceleration: Array,
        force_density: Array,
        root: LocalRootSolveResult,
        /,
    ):
        self.velocity = velocity
        self.acceleration = acceleration
        self.force_density = force_density
        self.root = root


class VelocityDependentAccelerationPlan(StrictModule, NonTrainableState):
    """Half-force closure that delegates all nonlinear work to an injected solver."""

    acceleration: VelocityDependentAcceleration = eqx.field(static=True)
    acceleration_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        acceleration: VelocityDependentAcceleration,
        /,
        *,
        acceleration_id: str,
    ):
        if not callable(acceleration):
            raise TypeError("acceleration must be callable.")
        identifier = str(acceleration_id)
        if not identifier:
            raise ValueError("acceleration_id must be non-empty.")
        self.acceleration = acceleration
        self.acceleration_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "velocity-dependent-lattice-acceleration",
                "acceleration": identifier,
            }
        )

    def evaluate_acceleration(
        self,
        velocity: Array,
        problem: VelocityDependentAccelerationProblem,
        /,
    ) -> Array:
        value = jnp.asarray(
            self.acceleration(
                problem.time,
                problem.coordinates,
                velocity,
                problem.parameters,
            ),
            dtype=velocity.dtype,
        )
        if value.shape == (velocity.shape[-1],):
            value = jnp.broadcast_to(value, velocity.shape)
        if value.shape != velocity.shape:
            raise ValueError(
                "Velocity-dependent acceleration must return one vector or one vector per point."
            )
        return value

    def residual(
        self,
        velocity: Array,
        problem: VelocityDependentAccelerationProblem,
        /,
    ) -> Array:
        acceleration = self.evaluate_acceleration(velocity, problem)
        safe_density = jnp.where(problem.density > 0.0, problem.density, 1.0)
        return (
            velocity
            - (problem.raw_momentum + 0.5 * problem.density[..., None] * acceleration)
            / safe_density[..., None]
        )

    def solve(
        self,
        problem: VelocityDependentAccelerationProblem,
        solver: LocalRootSolver,
        /,
    ) -> VelocityDependentAccelerationResult:
        if not isinstance(problem, VelocityDependentAccelerationProblem):
            raise TypeError("problem must be a VelocityDependentAccelerationProblem.")
        density = eqx.error_if(
            problem.density,
            jnp.any(~jnp.isfinite(problem.density) | (problem.density <= 0.0)),
            "Implicit acceleration density must be finite and positive.",
        )
        initial = eqx.error_if(
            problem.initial_velocity,
            jnp.any(~jnp.isfinite(problem.initial_velocity)),
            "Implicit acceleration initial velocity must be finite.",
        )
        problem = VelocityDependentAccelerationProblem(
            problem.time,
            problem.coordinates,
            density,
            problem.raw_momentum,
            parameters=problem.parameters,
            initial_velocity=initial,
        )
        root = solver.solve(self.residual, initial, problem)
        if not isinstance(root, LocalRootSolveResult):
            raise TypeError("Local root solver must return LocalRootSolveResult.")
        if root.value.shape != problem.raw_momentum.shape:
            raise ValueError("Local root solver returned a value on the wrong support.")
        certified_residual = self.residual(root.value, problem)
        acceleration = self.evaluate_acceleration(root.value, problem)
        invalid = (
            ~jnp.all(root.converged)
            | jnp.any(~jnp.isfinite(root.value))
            | jnp.any(~jnp.isfinite(acceleration))
            | jnp.any(~jnp.isfinite(certified_residual))
            | (jnp.max(jnp.abs(certified_residual)) > 1.0e-7)
        )
        velocity = eqx.error_if(
            root.value,
            invalid,
            "Velocity-dependent acceleration root solve failed certification.",
        )
        force_density = density[..., None] * acceleration
        certified_root = LocalRootSolveResult(
            velocity,
            certified_residual,
            root.converged,
            root.iterations,
        )
        return VelocityDependentAccelerationResult(
            velocity,
            acceleration,
            force_density,
            certified_root,
        )


class DampedLocalRootSolver(StrictModule, NonTrainableState):
    """Fixed-iteration point-local residual solver with explicit convergence evidence."""

    iterations: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        iterations: int = 16,
        damping: float = 0.8,
        tolerance: float = 1.0e-9,
    ):
        count = int(iterations)
        damping_ = float(damping)
        tolerance_ = float(tolerance)
        if count <= 0 or not 0.0 < damping_ <= 1.0 or tolerance_ <= 0.0:
            raise ValueError("Local root solver configuration is invalid.")
        self.iterations = count
        self.damping = damping_
        self.tolerance = tolerance_

    def solve(
        self,
        residual: LocalRootResidual,
        initial_guess: Array,
        args: Any,
        /,
    ) -> LocalRootSolveResult:
        initial = jnp.asarray(initial_guess)

        def iteration(_, value):
            defect = residual(value, args)
            return value - self.damping * defect

        value = jax.lax.fori_loop(0, self.iterations, iteration, initial)
        defect = residual(value, args)
        norm = jnp.sqrt(jnp.sum(defect * defect, axis=-1))
        converged = jnp.isfinite(norm) & (norm <= self.tolerance)
        return LocalRootSolveResult(
            value,
            defect,
            converged,
            jnp.full(norm.shape, self.iterations, dtype=jnp.int32),
        )


__all__ = [
    "DampedLocalRootSolver",
    "LocalRootResidual",
    "LocalRootSolveResult",
    "LocalRootSolver",
    "VelocityDependentAcceleration",
    "VelocityDependentAccelerationPlan",
    "VelocityDependentAccelerationProblem",
    "VelocityDependentAccelerationResult",
]
