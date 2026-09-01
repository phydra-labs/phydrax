#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._status import AstrodynamicsStatus


class TargetingResidualPlan(StrictModule, NonTrainableState):
    propagator: Callable
    terminal_projection: Callable
    target: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, propagator, terminal_projection, target, /, *, plan_id="mission-targeting"
    ):
        if not callable(propagator) or not callable(terminal_projection):
            raise TypeError("Targeting models must be callable.")
        self.propagator = propagator
        self.terminal_projection = terminal_projection
        self.target = jnp.asarray(target)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "targeting-residual-plan",
                "declared_id": str(plan_id),
                "target_shape": list(self.target.shape),
            }
        )

    def residual(self, decision: ArrayLike, args: Any = None, /) -> Array:
        terminal = self.propagator(jnp.asarray(decision), args)
        return self.terminal_projection(terminal) - self.target

    def jacobian(self, decision: ArrayLike, args: Any = None, /) -> Array:
        return jax.jacfwd(lambda value: self.residual(value, args))(jnp.asarray(decision))


class AccessResult(StrictModule):
    visible: Array
    elevation: Array
    range: Array
    valid: Array


class AccessPlan(StrictModule, NonTrainableState):
    horizon_elevation: Array
    maximum_range: Array

    def __init__(self, horizon_elevation=0.0, maximum_range=jnp.inf, /):
        self.horizon_elevation = jnp.asarray(horizon_elevation).reshape(())
        self.maximum_range = jnp.asarray(maximum_range).reshape(())

    def evaluate(
        self,
        station_position: ArrayLike,
        station_zenith: ArrayLike,
        spacecraft_position: ArrayLike,
        /,
    ) -> AccessResult:
        station = jnp.asarray(station_position)
        zenith = jnp.asarray(station_zenith)
        relative = jnp.asarray(spacecraft_position) - station
        distance = jnp.sqrt(jnp.sum(relative * relative, axis=-1))
        unit = relative / jnp.maximum(distance[..., None], 1.0e-30)
        zenith_unit = zenith / jnp.sqrt(jnp.sum(zenith * zenith))
        elevation = jnp.arcsin(jnp.clip(jnp.sum(unit * zenith_unit, axis=-1), -1.0, 1.0))
        valid = jnp.isfinite(distance) & (distance > 0.0)
        visible = (
            valid
            & (elevation >= self.horizon_elevation)
            & (distance <= self.maximum_range)
        )
        return AccessResult(visible, elevation, distance, valid)


class ConjunctionResult(StrictModule):
    time_of_closest_approach: Array
    miss_vector: Array
    miss_distance: Array
    relative_speed: Array
    collision_probability: Array
    valid: Array
    status: Array


class ConjunctionPlan(StrictModule, NonTrainableState):
    hard_body_radius: Array

    def __init__(self, hard_body_radius: ArrayLike, /):
        self.hard_body_radius = jnp.asarray(hard_body_radius).reshape(())

    def evaluate(
        self,
        relative_position: ArrayLike,
        relative_velocity: ArrayLike,
        covariance: ArrayLike,
        reference_time: ArrayLike = 0.0,
        /,
    ) -> ConjunctionResult:
        position = jnp.asarray(relative_position)
        velocity = jnp.asarray(relative_velocity)
        covariance_ = jnp.asarray(covariance)
        speed_squared = jnp.sum(velocity * velocity)
        offset = -jnp.sum(position * velocity) / jnp.where(
            speed_squared > 0.0, speed_squared, 1.0
        )
        miss = position + offset * velocity
        distance = jnp.sqrt(jnp.sum(miss * miss))
        speed = jnp.sqrt(speed_squared)
        radial_variance = jnp.sum(miss * (covariance_ @ miss)) / jnp.where(
            distance > 0.0, distance**2, 1.0
        )
        sigma = jnp.sqrt(jnp.maximum(radial_variance, 1.0e-30))
        probability = jnp.exp(-0.5 * (distance / sigma) ** 2) * jnp.clip(
            self.hard_body_radius**2 / (2.0 * sigma**2), 0.0, 1.0
        )
        valid = (
            jnp.all(jnp.isfinite(miss))
            & (speed_squared > 0.0)
            & (self.hard_body_radius >= 0.0)
        )
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.INVALID_DOMAIN),
        ).astype(jnp.int32)
        return ConjunctionResult(
            jnp.asarray(reference_time) + offset,
            miss,
            distance,
            speed,
            jnp.where(valid, probability, jnp.nan),
            valid,
            status,
        )


__all__ = [
    "AccessPlan",
    "AccessResult",
    "ConjunctionPlan",
    "ConjunctionResult",
    "TargetingResidualPlan",
]
