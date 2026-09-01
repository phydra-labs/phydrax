#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._context import AstrodynamicsContext
from ._status import AstrodynamicsStatus


OrbitMeasurementKind: TypeAlias = Literal[
    "range", "range_rate", "right_ascension_declination"
]


class OrbitMeasurementResult(StrictModule):
    predicted: Array
    jacobian: Array
    valid: Array
    status: Array
    measurement_id: str = eqx.field(static=True)


class OrbitMeasurementPlan(StrictModule, NonTrainableState):
    times: Array
    observer_position: Array
    observer_velocity: Array
    covariance: Array
    context: AstrodynamicsContext
    kind: OrbitMeasurementKind = eqx.field(static=True)
    measurement_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: OrbitMeasurementKind,
        times: ArrayLike,
        observer_position: ArrayLike,
        observer_velocity: ArrayLike,
        covariance: ArrayLike,
        context: AstrodynamicsContext,
        /,
        *,
        measurement_id: str,
    ):
        if kind not in ("range", "range_rate", "right_ascension_declination"):
            raise ValueError("Unknown orbit measurement kind.")
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        times_host = np.asarray(times, dtype=float)
        position_host = np.asarray(observer_position, dtype=float)
        velocity_host = np.asarray(observer_velocity, dtype=float)
        count = times_host.size
        if (
            times_host.ndim != 1
            or position_host.shape != (count, 3)
            or velocity_host.shape != (count, 3)
        ):
            raise ValueError("Observer states must have shape (num_measurements,3).")
        output_dimension = 2 if kind == "right_ascension_declination" else 1
        covariance_host = np.asarray(covariance, dtype=float)
        if covariance_host.shape not in (
            (output_dimension, output_dimension),
            (count, output_dimension, output_dimension),
        ):
            raise ValueError("Measurement covariance has incompatible shape.")
        if (
            np.any(~np.isfinite(times_host))
            or np.any(~np.isfinite(position_host))
            or np.any(~np.isfinite(velocity_host))
            or np.any(~np.isfinite(covariance_host))
        ):
            raise ValueError("Measurement schedule and covariance must be finite.")
        identifier = str(measurement_id).strip()
        if not identifier:
            raise ValueError("measurement_id must be non-empty.")
        self.times = jnp.asarray(times_host)
        self.observer_position = jnp.asarray(position_host)
        self.observer_velocity = jnp.asarray(velocity_host)
        self.covariance = jnp.asarray(covariance_host)
        self.context = context
        self.kind = kind
        self.measurement_id = canonical_fingerprint(
            {
                "kind": "orbit-measurement-plan",
                "declared_id": identifier,
                "measurement_kind": kind,
                "context": context.context_id,
                "count": count,
            }
        )

    def _one(
        self, state: Array, observer_position: Array, observer_velocity: Array, /
    ) -> Array:
        relative_position = state[:3] - observer_position
        relative_velocity = state[3:] - observer_velocity
        distance = jnp.sqrt(jnp.sum(relative_position * relative_position))
        if self.kind == "range":
            return jnp.asarray((distance,))
        if self.kind == "range_rate":
            rate = jnp.sum(relative_position * relative_velocity) / jnp.where(
                distance > 0.0, distance, 1.0
            )
            return jnp.asarray((rate,))
        right_ascension = jnp.mod(
            jnp.arctan2(relative_position[1], relative_position[0]), 2.0 * jnp.pi
        )
        declination = jnp.arcsin(
            jnp.clip(
                relative_position[2] / jnp.where(distance > 0.0, distance, 1.0),
                -1.0,
                1.0,
            )
        )
        return jnp.asarray((right_ascension, declination))

    def evaluate(self, states: ArrayLike, /) -> OrbitMeasurementResult:
        state_values = jnp.asarray(states)
        if state_values.shape != (int(self.times.size), 6):
            raise ValueError("Measurement states must have shape (num_measurements,6).")
        predicted = jax.vmap(self._one)(
            state_values, self.observer_position, self.observer_velocity
        )
        jacobian = jax.vmap(jax.jacfwd(self._one, argnums=0))(
            state_values, self.observer_position, self.observer_velocity
        )
        relative = state_values[:, :3] - self.observer_position
        distance = jnp.sqrt(jnp.sum(relative * relative, axis=-1))
        finite = jnp.all(jnp.isfinite(state_values), axis=-1) & jnp.all(
            jnp.isfinite(predicted), axis=-1
        )
        valid = finite & (distance > 0.0)
        status = jnp.where(
            ~finite,
            int(AstrodynamicsStatus.NONFINITE_INPUT),
            jnp.where(
                distance > 0.0,
                int(AstrodynamicsStatus.SUCCESS),
                int(AstrodynamicsStatus.SINGULAR_GEOMETRY),
            ),
        ).astype(jnp.int32)
        return OrbitMeasurementResult(
            jnp.where(valid[:, None], predicted, 0.0),
            jnp.where(valid[:, None, None], jacobian, 0.0),
            valid,
            status,
            self.measurement_id,
        )


__all__ = [
    "OrbitMeasurementKind",
    "OrbitMeasurementPlan",
    "OrbitMeasurementResult",
]
