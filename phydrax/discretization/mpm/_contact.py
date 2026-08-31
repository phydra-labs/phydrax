#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._numerics._compensated import compensated_sum
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState


class MPMGridConstraintResult(StrictModule):
    velocity: Array
    impulse: Array
    work: Array
    dissipation: Array
    contact_step_limit: Array
    active_mask: Array
    mode: Array
    successful: Array


class AbstractMPMFrictionPlan(StrictModule, NonTrainableState):
    coefficient: AbstractAttribute[float]
    plan_id: AbstractAttribute[str]

    @abc.abstractmethod
    def impulse_magnitude(
        self, tangential_speed: Array, normal_impulse: Array, mass: Array, /
    ) -> Array:
        raise NotImplementedError


class SharpCoulombMPMFrictionPlan(AbstractMPMFrictionPlan):
    coefficient: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, coefficient: float, /):
        value = float(coefficient)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("Coulomb coefficient must be finite and nonnegative.")
        self.coefficient = value
        self.plan_id = canonical_fingerprint(
            {"kind": "sharp-coulomb-mpm-friction", "coefficient": value}
        )

    def impulse_magnitude(self, tangential_speed, normal_impulse, mass, /):
        return jnp.minimum(mass * tangential_speed, self.coefficient * normal_impulse)


class SmoothCoulombMPMFrictionPlan(AbstractMPMFrictionPlan):
    coefficient: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, coefficient: float, /, *, regularization: float = 1.0e-4):
        value = float(coefficient)
        regularization_ = float(regularization)
        if (
            not np.isfinite(value)
            or value < 0.0
            or not np.isfinite(regularization_)
            or regularization_ <= 0.0
        ):
            raise ValueError("Smooth Coulomb parameters are invalid.")
        self.coefficient = value
        self.regularization = regularization_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "smooth-coulomb-mpm-friction",
                "coefficient": value,
                "regularization": regularization_,
            }
        )

    def impulse_magnitude(self, tangential_speed, normal_impulse, mass, /):
        cap = self.coefficient * normal_impulse
        scale = jnp.maximum(cap, self.regularization)
        return cap * jnp.tanh(mass * tangential_speed / scale)


class RigidMPMContactPlan(StrictModule, NonTrainableState):
    """Nodal rigid-obstacle projection with auditable Coulomb impulse."""

    geometry: Any
    friction: AbstractMPMFrictionPlan
    wall_velocity: Callable | None
    wall_velocity_id: str | None = eqx.field(static=True)
    contact_band: float = eqx.field(static=True)
    smooth_normal_regularization: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: Any,
        friction: AbstractMPMFrictionPlan,
        /,
        *,
        contact_band: float,
        wall_velocity: Callable | None = None,
        wall_velocity_id: str | None = None,
        smooth_normal_regularization: float | None = None,
    ):
        if geometry is None:
            raise TypeError("geometry must provide signed_distance and boundary_normal.")
        if not isinstance(friction, AbstractMPMFrictionPlan):
            raise TypeError("friction must be AbstractMPMFrictionPlan.")
        band = float(contact_band)
        regularization = (
            None
            if smooth_normal_regularization is None
            else float(smooth_normal_regularization)
        )
        if not np.isfinite(band) or band < 0.0:
            raise ValueError("contact_band must be finite and nonnegative.")
        if regularization is not None and (
            not np.isfinite(regularization) or regularization <= 0.0
        ):
            raise ValueError("Normal regularization must be finite and positive.")
        if wall_velocity is not None and not callable(wall_velocity):
            raise TypeError("wall_velocity must be callable or None.")
        if wall_velocity is not None and not wall_velocity_id:
            raise ValueError("Moving walls require wall_velocity_id.")
        self.geometry = geometry
        self.friction = friction
        self.wall_velocity = wall_velocity
        self.wall_velocity_id = (
            None if wall_velocity_id is None else str(wall_velocity_id)
        )
        self.contact_band = band
        self.smooth_normal_regularization = regularization
        self.plan_id = canonical_fingerprint(
            {
                "kind": "rigid-mpm-contact",
                "geometry_type": f"{type(geometry).__module__}.{type(geometry).__qualname__}",
                "geometry_state": array_tree_fingerprint(geometry.state),
                "friction": friction.plan_id,
                "contact_band": band,
                "wall_velocity": self.wall_velocity_id,
                "smooth_normal_regularization": regularization,
            }
        )

    def prospective_mask(self, coordinates: ArrayLike, /) -> Array:
        coordinates_ = jnp.asarray(coordinates)
        return self.geometry.signed_distance(coordinates_) <= self.contact_band

    def apply(
        self,
        coordinates: ArrayLike,
        velocity: ArrayLike,
        mass: ArrayLike,
        time: ArrayLike,
        step_size: ArrayLike,
        arguments: Any = None,
        /,
    ) -> MPMGridConstraintResult:
        coordinates_ = jnp.asarray(coordinates)
        velocity_ = jnp.asarray(velocity)
        mass_ = jnp.asarray(mass)
        if coordinates_.shape != velocity_.shape or mass_.shape != velocity_.shape[:-1]:
            raise ValueError("Rigid MPM contact grid shapes differ.")
        distance = self.geometry.signed_distance(coordinates_)
        normal = self.geometry.boundary_normal(coordinates_)
        normal_norm = jnp.sqrt(jnp.sum(normal * normal, axis=-1))
        reliable_normal = jnp.isfinite(normal_norm) & (normal_norm > 1.0e-12)
        normal = normal / jnp.where(reliable_normal, normal_norm, 1.0)[..., None]
        wall = (
            jnp.zeros_like(velocity_)
            if self.wall_velocity is None
            else jnp.asarray(
                self.wall_velocity(time, coordinates_, arguments), dtype=velocity_.dtype
            )
        )
        if wall.shape != velocity_.shape:
            raise ValueError("Wall velocity must match nodal velocity shape.")
        relative = velocity_ - wall
        normal_speed = jnp.sum(relative * normal, axis=-1)
        occupied = mass_ > 0.0
        candidate = occupied & reliable_normal & (distance <= self.contact_band)
        if self.smooth_normal_regularization is None:
            normal_impulse_magnitude = mass_ * jnp.maximum(-normal_speed, 0.0)
        else:
            epsilon = jnp.asarray(
                self.smooth_normal_regularization, dtype=velocity_.dtype
            )
            normal_impulse_magnitude = (
                mass_ * epsilon * jax.nn.softplus(-normal_speed / epsilon)
            )
        active = candidate & (normal_impulse_magnitude > 0.0)
        tangential = relative - normal_speed[..., None] * normal
        tangential_speed = jnp.sqrt(jnp.sum(tangential * tangential, axis=-1))
        tangent_direction = (
            tangential
            / jnp.where(tangential_speed > 0.0, tangential_speed, 1.0)[..., None]
        )
        tangential_impulse_magnitude = self.friction.impulse_magnitude(
            tangential_speed, normal_impulse_magnitude, mass_
        )
        impulse = (
            normal_impulse_magnitude[..., None] * normal
            - tangential_impulse_magnitude[..., None] * tangent_direction
        )
        impulse = jnp.where(active[..., None], impulse, 0.0)
        next_velocity = velocity_ + impulse / jnp.where(occupied, mass_, 1.0)[..., None]
        kinetic_change = (
            0.5
            * mass_
            * (
                jnp.sum(next_velocity * next_velocity, axis=-1)
                - jnp.sum(velocity_ * velocity_, axis=-1)
            )
        )
        work = compensated_sum(jnp.where(active, kinetic_change, 0.0))
        dissipation = compensated_sum(
            jnp.where(active, tangential_impulse_magnitude * tangential_speed, 0.0)
        )
        total_impulse = compensated_sum(
            impulse.reshape((-1, velocity_.shape[-1])), axis=0
        )
        approach_speed = jnp.maximum(-normal_speed, 0.0)
        step_limit = jnp.min(
            jnp.where(
                active & (approach_speed > 0.0),
                jnp.maximum(distance + self.contact_band, 0.0)
                / jnp.maximum(approach_speed, 1.0e-30),
                jnp.inf,
            ),
            initial=jnp.asarray(jnp.inf, dtype=velocity_.dtype),
        )
        sticking = active & (
            mass_ * tangential_speed
            <= self.friction.coefficient * normal_impulse_magnitude
        )
        mode = jnp.where(active, jnp.where(sticking, 1, 2), 0).astype(jnp.int32)
        successful = (
            jnp.all(~candidate | reliable_normal)
            & jnp.all(jnp.isfinite(next_velocity))
            & jnp.all(jnp.isfinite(total_impulse))
            & jnp.isfinite(work)
            & jnp.isfinite(dissipation)
            & jnp.isfinite(jnp.asarray(step_size))
        )
        return MPMGridConstraintResult(
            next_velocity,
            total_impulse,
            work,
            dissipation,
            step_limit,
            active,
            mode,
            successful,
        )


__all__ = [
    "AbstractMPMFrictionPlan",
    "MPMGridConstraintResult",
    "RigidMPMContactPlan",
    "SharpCoulombMPMFrictionPlan",
    "SmoothCoulombMPMFrictionPlan",
]
