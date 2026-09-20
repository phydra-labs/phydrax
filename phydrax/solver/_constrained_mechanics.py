#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Holonomic index-three mechanics with explicit SHAKE/RATTLE projection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule


class ConstrainedMechanicalState(StrictModule):
    configuration: Array
    momentum: Array

    def __init__(self, configuration: ArrayLike, momentum: ArrayLike, /):
        configuration_ = jnp.asarray(configuration)
        momentum_ = jnp.asarray(momentum)
        if configuration_.ndim != 1 or momentum_.shape != configuration_.shape:
            raise ValueError("Configuration and momentum must be aligned vectors.")
        self.configuration = configuration_
        self.momentum = momentum_


class ConstrainedMechanicalStep(StrictModule):
    state: ConstrainedMechanicalState
    position_residual: Array
    velocity_residual: Array
    position_iterations: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class SHAKERATTLEPlan:
    """Fixed-iteration SHAKE/RATTLE for a regular holonomic constraint manifold."""

    inverse_mass: Array
    potential_gradient: Callable[[Array, object], Array]
    constraint: Callable[[Array, object], Array]
    maximum_projection_steps: int
    constraint_tolerance: float
    plan_id: str

    def __init__(
        self,
        inverse_mass: ArrayLike,
        potential_gradient: Callable[[Array, object], Array],
        constraint: Callable[[Array, object], Array],
        /,
        *,
        maximum_projection_steps: int = 8,
        constraint_tolerance: float = 1.0e-10,
        plan_id: str | None = None,
    ):
        inverse_mass_ = jnp.asarray(inverse_mass)
        if inverse_mass_.ndim != 1 or inverse_mass_.size == 0:
            raise ValueError("inverse_mass must be a non-empty vector.")
        host_mass = np.asarray(inverse_mass_)
        if not np.all(np.isfinite(host_mass)) or np.any(host_mass <= 0.0):
            raise ValueError("inverse_mass must be finite and strictly positive.")
        if not callable(potential_gradient) or not callable(constraint):
            raise TypeError("Potential gradient and constraint must be callable.")
        steps = int(maximum_projection_steps)
        tolerance = float(constraint_tolerance)
        if steps <= 0:
            raise ValueError("maximum_projection_steps must be positive.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("constraint_tolerance must be finite and positive.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "shake-rattle-plan",
                    "dimension": inverse_mass_.size,
                    "maximum_projection_steps": steps,
                    "constraint_tolerance": tolerance,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        object.__setattr__(self, "inverse_mass", inverse_mass_)
        object.__setattr__(self, "potential_gradient", potential_gradient)
        object.__setattr__(self, "constraint", constraint)
        object.__setattr__(self, "maximum_projection_steps", steps)
        object.__setattr__(self, "constraint_tolerance", tolerance)
        object.__setattr__(self, "plan_id", identifier)

    def _position_projection(self, trial: Array, args: object, /):
        inverse_mass = self.inverse_mass

        def project(_, carry):
            configuration, active, used = carry
            residual = jnp.atleast_1d(jnp.asarray(self.constraint(configuration, args)))
            jacobian = jax.jacfwd(
                lambda value: jnp.atleast_1d(self.constraint(value, args))
            )(configuration)
            gram = (jacobian * inverse_mass[None, :]) @ jacobian.T
            multiplier = jnp.linalg.solve(gram, -residual)
            correction = inverse_mass * (jacobian.T @ multiplier)
            candidate = configuration + correction
            norm = jnp.linalg.norm(residual)
            execute = active & jnp.isfinite(norm) & (norm > self.constraint_tolerance)
            configuration = jnp.where(execute, candidate, configuration)
            return configuration, execute, used + execute.astype(jnp.int32)

        return jax.lax.fori_loop(
            0,
            self.maximum_projection_steps,
            project,
            (trial, jnp.asarray(True), jnp.asarray(0, dtype=jnp.int32)),
        )

    def step(
        self,
        state: ConstrainedMechanicalState,
        step_size: ArrayLike,
        /,
        *,
        args: object = None,
    ) -> ConstrainedMechanicalStep:
        if not isinstance(state, ConstrainedMechanicalState):
            raise TypeError("state must be ConstrainedMechanicalState.")
        if state.configuration.shape != self.inverse_mass.shape:
            raise ValueError("State dimension does not match inverse_mass.")
        step = jnp.asarray(step_size, dtype=state.configuration.dtype)
        if step.ndim != 0:
            raise ValueError("step_size must be scalar.")
        first_force = jnp.asarray(self.potential_gradient(state.configuration, args))
        if first_force.shape != state.configuration.shape:
            raise ValueError("potential_gradient changed the configuration shape.")
        half_momentum = state.momentum - 0.5 * step * first_force
        trial = state.configuration + step * self.inverse_mass * half_momentum
        configuration, _, iterations = self._position_projection(trial, args)
        second_force = jnp.asarray(self.potential_gradient(configuration, args))
        trial_momentum = half_momentum - 0.5 * step * second_force
        jacobian = jax.jacfwd(lambda value: jnp.atleast_1d(self.constraint(value, args)))(
            configuration
        )
        gram = (jacobian * self.inverse_mass[None, :]) @ jacobian.T
        velocity_defect = jacobian @ (self.inverse_mass * trial_momentum)
        multiplier = jnp.linalg.solve(gram, velocity_defect)
        momentum = trial_momentum - jacobian.T @ multiplier
        position_residual = jnp.linalg.norm(
            jnp.atleast_1d(self.constraint(configuration, args))
        )
        velocity_residual = jnp.linalg.norm(jacobian @ (self.inverse_mass * momentum))
        finite = jnp.all(jnp.isfinite(configuration)) & jnp.all(jnp.isfinite(momentum))
        accepted = (
            finite
            & jnp.isfinite(step)
            & (step > 0.0)
            & (position_residual <= self.constraint_tolerance)
            & (velocity_residual <= self.constraint_tolerance)
        )
        safe_state = ConstrainedMechanicalState(
            jnp.where(accepted, configuration, state.configuration),
            jnp.where(accepted, momentum, state.momentum),
        )
        return ConstrainedMechanicalStep(
            safe_state,
            position_residual,
            velocity_residual,
            iterations,
            accepted,
            self.plan_id,
        )


__all__ = [
    "ConstrainedMechanicalState",
    "ConstrainedMechanicalStep",
    "SHAKERATTLEPlan",
]
