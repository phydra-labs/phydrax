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

import phydrax.linalg as la

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._status import AstrodynamicsStatus


_CONJUNCTION_COVARIANCE_POLICY = la.DensePropertyVerificationPolicy(
    require_hermitian=True,
    require_positive_semidefinite=True,
)


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

    plan_id: str = eqx.field(static=True)


class AccessPlan(StrictModule, NonTrainableState):
    horizon_elevation: Array
    maximum_range: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        horizon_elevation: ArrayLike = 0.0,
        maximum_range: ArrayLike = np.finfo(np.float32).max,
        /,
    ):
        horizon = np.asarray(horizon_elevation, dtype=np.float64)
        maximum = np.asarray(maximum_range, dtype=np.float64)
        if (
            horizon.shape != ()
            or maximum.shape != ()
            or not np.isfinite(horizon)
            or not np.isfinite(maximum)
            or maximum <= 0.0
        ):
            raise ValueError("Access limits must be finite scalars with positive range.")
        self.horizon_elevation = jnp.asarray(horizon)
        self.maximum_range = jnp.asarray(maximum)
        self.plan_id = canonical_fingerprint(
            {"kind": "access-plan", "horizon": horizon, "maximum_range": maximum}
        )

    def evaluate(
        self,
        station_position: ArrayLike,
        station_zenith: ArrayLike,
        spacecraft_position: ArrayLike,
        /,
    ) -> AccessResult:
        station = jnp.asarray(station_position)
        zenith = jnp.asarray(station_zenith)
        spacecraft = jnp.asarray(spacecraft_position)
        if station.shape != (3,) or zenith.shape != (3,) or spacecraft.shape != (3,):
            raise ValueError(
                "Access geometry requires position and zenith three-vectors."
            )
        relative = spacecraft - station
        distance = jnp.sqrt(jnp.sum(relative * relative, axis=-1))
        zenith_norm = jnp.sqrt(jnp.sum(zenith * zenith))
        unit = relative / jnp.where(distance > 0.0, distance, 1.0)
        zenith_unit = zenith / jnp.where(zenith_norm > 0.0, zenith_norm, 1.0)
        elevation = jnp.arcsin(jnp.clip(jnp.sum(unit * zenith_unit, axis=-1), -1.0, 1.0))
        valid = (
            jnp.all(jnp.isfinite(station))
            & jnp.all(jnp.isfinite(spacecraft))
            & jnp.all(jnp.isfinite(zenith))
            & jnp.isfinite(elevation)
            & (distance > 0.0)
            & (zenith_norm > 0.0)
        )
        visible = (
            valid
            & (elevation >= self.horizon_elevation)
            & (distance <= self.maximum_range)
        )
        return AccessResult(visible, elevation, distance, valid, self.plan_id)


class ConjunctionResult(StrictModule):
    time_of_closest_approach: Array
    miss_vector: Array
    miss_distance: Array
    relative_speed: Array
    collision_probability: Array
    valid: Array
    status: Array

    plan_id: str = eqx.field(static=True)


class ConjunctionPlan(StrictModule, NonTrainableState):
    hard_body_radius: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, hard_body_radius: ArrayLike, /):
        radius = np.asarray(hard_body_radius, dtype=np.float64)
        if radius.shape != () or not np.isfinite(radius) or radius < 0.0:
            raise ValueError("hard_body_radius must be a finite nonnegative scalar.")
        self.hard_body_radius = jnp.asarray(radius)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conjunction-plan",
                "hard_body_radius": radius,
                "covariance_policy": "finite-hermitian-positive-semidefinite",
            }
        )

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
        reference = jnp.asarray(reference_time).reshape(())
        if (
            position.shape != (3,)
            or velocity.shape != (3,)
            or covariance_.shape != (3, 3)
        ):
            raise ValueError(
                "Conjunction position/velocity/covariance shapes are invalid."
            )
        covariance_evidence = la.verify_dense_properties(
            covariance_,
            policy=_CONJUNCTION_COVARIANCE_POLICY,
        )
        covariance_ = covariance_evidence.matrix
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
        finite = (
            jnp.all(jnp.isfinite(position))
            & jnp.all(jnp.isfinite(velocity))
            & covariance_evidence.finite
            & jnp.isfinite(reference)
            & jnp.all(jnp.isfinite(miss))
            & jnp.isfinite(probability)
        )
        valid = (
            finite
            & covariance_evidence.successful
            & (speed_squared > 0.0)
            & (radial_variance >= 0.0)
        )
        status = jnp.where(
            ~finite,
            int(AstrodynamicsStatus.NONFINITE_INPUT),
            jnp.where(
                valid,
                int(AstrodynamicsStatus.SUCCESS),
                int(AstrodynamicsStatus.INVALID_DOMAIN),
            ),
        ).astype(jnp.int32)
        return ConjunctionResult(
            reference + offset,
            miss,
            distance,
            speed,
            jnp.where(valid, probability, jnp.nan),
            valid,
            status,
            self.plan_id,
        )


__all__ = [
    "AccessPlan",
    "AccessResult",
    "ConjunctionPlan",
    "ConjunctionResult",
    "TargetingResidualPlan",
]
