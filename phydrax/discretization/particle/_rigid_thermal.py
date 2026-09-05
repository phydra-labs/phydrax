#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Fixed-cell COM presentation and exact conditional rigid velocity heat baths.

States retain unwrapped COM coordinates. Images are presentation data, never
independent wrapped marker coordinates. Rotational friction is isotropic in
angular velocity; inertia is COM-centred and angular velocities are world-frame.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from .._periodic_cell import PeriodicCell
from ._rigid_body import (
    PreparedRigidBodySet,
    quaternion_rotation_matrix,
    rigid_body_kick_drift_kick,
    RigidBodyKinematics,
    RigidBodyLoad,
    RigidBodyStepResult,
)


class RigidPeriodicPresentation(StrictModule):
    position: Array
    images: Array


def rigid_periodic_presentation(
    cell: PeriodicCell, state: RigidBodyKinematics, /
) -> RigidPeriodicPresentation:
    """Present unwrapped body COMs through the native cell/image contract."""
    position, images = cell.wrap(state.position)
    return RigidPeriodicPresentation(position, images)


class PreparedRigidHeatBath(StrictModule, NonTrainableState):
    """Exact OU velocity transition with covariance kT/m and kT I_world^-1.

    kT is in mechanical mass*length^2/time^2 units, not an arbitrary energy
    scale. Frictions are nonnegative inverse-time rates. All validation and
    inertia factorization are host-only. No centre-of-mass velocity removal is
    performed: that would change the ensemble. Keys are consumed explicitly.
    """

    bodies: PreparedRigidBodySet
    inverse_inertia_sqrt: Array
    thermal_energy: Array
    translation_friction: Array
    rotation_friction: Array

    def __init__(
        self, bodies, thermal_energy, translation_friction, rotation_friction, /
    ):
        values = np.asarray(
            [thermal_energy, translation_friction, rotation_friction], dtype=float
        )
        if values.shape != (3,) or not np.all(np.isfinite(values) & (values >= 0)):
            raise ValueError(
                "Thermal energy and scalar frictions must be finite and nonnegative."
            )
        inverse = np.asarray(bodies.inverse_inertia_body)
        if bodies.ambient_dimension == 3:
            eigenvalues, eigenvectors = np.linalg.eigh(inverse)
            root = (eigenvectors * np.sqrt(eigenvalues)[:, None, :]) @ np.swapaxes(
                eigenvectors, -1, -2
            )
        else:
            root = np.sqrt(inverse)
        self.bodies = bodies
        self.inverse_inertia_sqrt = jnp.asarray(root)
        self.thermal_energy = jnp.asarray(values[0])
        self.translation_friction = jnp.asarray(values[1])
        self.rotation_friction = jnp.asarray(values[2])

    def apply(self, state: RigidBodyKinematics, step_size, key, /) -> RigidBodyKinematics:
        dt = jnp.asarray(step_size, dtype=state.position.dtype)
        translation_key, rotation_key = jax.random.split(key)
        ct = jnp.exp(-self.translation_friction * dt)
        cr = jnp.exp(-self.rotation_friction * dt)
        st = jnp.sqrt(
            -jnp.expm1(-2 * self.translation_friction * dt) * self.thermal_energy
        )
        sr = jnp.sqrt(-jnp.expm1(-2 * self.rotation_friction * dt) * self.thermal_energy)
        velocity_noise = jax.random.normal(
            translation_key, state.velocity.shape, dtype=state.velocity.dtype
        )
        angular_noise = jax.random.normal(
            rotation_key, state.angular_velocity.shape, dtype=state.angular_velocity.dtype
        )
        velocity = (
            ct * state.velocity
            + st * jnp.sqrt(self.bodies.inverse_masses)[:, None] * velocity_noise
        )
        if self.bodies.ambient_dimension == 3:
            body_noise = contract(
                "...ij,...j->...i", self.inverse_inertia_sqrt, angular_noise
            )
            world_noise = contract(
                "...ij,...j->...i",
                quaternion_rotation_matrix(state.orientation),
                body_noise,
            )
        else:
            world_noise = self.inverse_inertia_sqrt[:, None] * angular_noise
        angular = cr * state.angular_velocity + sr * world_noise
        mobile = (self.bodies.particles.active_mask & ~self.bodies.fixed_mask)[:, None]
        return RigidBodyKinematics(
            state.position,
            jnp.where(mobile, velocity, 0),
            state.orientation,
            jnp.where(mobile, angular, 0),
        )

    def step(
        self,
        state,
        load: RigidBodyLoad,
        time,
        step_size,
        load_function,
        key,
        args=None,
        /,
    ) -> RigidBodyStepResult:
        """Symmetric OU/2--native KDK--OU/2; finite dt has configurational bias.

        Canonical kinetic covariance is exact for the bath substep. Equilibrium
        claims for the composed dynamics require timestep convergence, especially
        for anisotropic free-rotation handled by the native KDK approximation.
        """
        first_key, last_key = jax.random.split(key)
        heated = self.apply(state, 0.5 * step_size, first_key)
        stepped = rigid_body_kick_drift_kick(
            self.bodies, heated, load, time, step_size, load_function, args
        )
        result = self.apply(stepped.kinematics, 0.5 * step_size, last_key)
        successful = (
            stepped.successful
            & (step_size >= 0)
            & jnp.all(jnp.isfinite(result.velocity))
            & jnp.all(jnp.isfinite(result.angular_velocity))
        )
        return RigidBodyStepResult(result, stepped.load, successful)


__all__ = [
    "rigid_periodic_presentation",
    "RigidPeriodicPresentation",
    "PreparedRigidHeatBath",
]
