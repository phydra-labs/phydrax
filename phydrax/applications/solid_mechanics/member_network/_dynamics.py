#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ....linalg import DenseLinearOperator, LinearSystem, solve


class MemberDynamicState(StrictModule):
    displacement: Array
    velocity: Array
    acceleration: Array
    kinetic_energy: Array
    strain_energy: Array
    damping_dissipation: Array


class NewmarkPolicy(StrictModule):
    beta: float
    gamma: float

    def __init__(self, *, beta: float = 0.25, gamma: float = 0.5):
        if beta <= 0.0 or gamma <= 0.0:
            raise ValueError("Newmark beta and gamma must be positive.")
        self.beta = float(beta)
        self.gamma = float(gamma)


def newmark_step(
    mass: ArrayLike,
    damping: ArrayLike,
    tangent: ArrayLike,
    force: ArrayLike,
    state: MemberDynamicState,
    time_step: float,
    /,
    *,
    policy: NewmarkPolicy | None = None,
) -> MemberDynamicState:
    """Advance one average-acceleration step with native dense linear solve."""
    policy_ = NewmarkPolicy() if policy is None else policy
    mass_ = jnp.asarray(mass)
    damping_ = jnp.asarray(damping, dtype=mass_.dtype)
    tangent_ = jnp.asarray(tangent, dtype=mass_.dtype)
    force_ = jnp.asarray(force, dtype=mass_.dtype)
    dt = jnp.asarray(time_step, dtype=mass_.dtype)
    shape = mass_.shape
    if mass_.ndim != 2 or mass_.shape[0] != mass_.shape[1]:
        raise ValueError("Dynamic matrices must be square.")
    if damping_.shape != shape or tangent_.shape != shape or force_.shape != (shape[0],):
        raise ValueError("Dynamic matrices and force vector are incompatible.")
    beta, gamma = policy_.beta, policy_.gamma
    predicted_displacement = (
        state.displacement
        + dt * state.velocity
        + dt**2 * (0.5 - beta) * state.acceleration
    )
    predicted_velocity = state.velocity + dt * (1.0 - gamma) * state.acceleration
    effective = mass_ + gamma * dt * damping_ + beta * dt**2 * tangent_
    rhs = force_ - damping_ @ predicted_velocity - tangent_ @ predicted_displacement
    acceleration = solve(LinearSystem(DenseLinearOperator(effective)), rhs).value
    displacement = predicted_displacement + beta * dt**2 * acceleration
    velocity = predicted_velocity + gamma * dt * acceleration
    kinetic = 0.5 * velocity @ mass_ @ velocity
    strain = 0.5 * displacement @ tangent_ @ displacement
    dissipation = dt * velocity @ damping_ @ velocity
    return MemberDynamicState(
        displacement,
        velocity,
        acceleration,
        kinetic,
        strain,
        state.damping_dissipation + dissipation,
    )


__all__ = ["MemberDynamicState", "NewmarkPolicy", "newmark_step"]
