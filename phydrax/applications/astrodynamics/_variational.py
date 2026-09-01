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
from ...solver import HybridEventSensitivityResult


class VariationalResult(StrictModule):
    times: Array
    states: Array
    transition_matrix: Array
    parameter_sensitivity: Array
    covariance: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


class VariationalPropagationPlan(StrictModule, NonTrainableState):
    dynamics: Callable
    times: Array
    process_noise: Array
    parameter_dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics,
        times,
        process_noise,
        /,
        *,
        parameter_dimension=0,
        dynamics_id="variational-dynamics",
    ):
        if not callable(dynamics):
            raise TypeError("dynamics must be callable.")
        times_host = np.asarray(times, dtype=float)
        noise = np.asarray(process_noise, dtype=float)
        if (
            times_host.ndim != 1
            or times_host.size < 2
            or np.any(np.diff(times_host) <= 0.0)
            or noise.ndim != 2
            or noise.shape[0] != noise.shape[1]
        ):
            raise ValueError("Variational time/noise arrays are invalid.")
        self.dynamics = dynamics
        self.times = jnp.asarray(times_host)
        self.process_noise = jnp.asarray(noise)
        self.parameter_dimension = int(parameter_dimension)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "variational-propagation",
                "dynamics": str(dynamics_id),
                "times": int(times_host.size),
                "parameters": int(parameter_dimension),
            }
        )

    def propagate(
        self,
        initial_state: ArrayLike,
        parameters: ArrayLike,
        initial_covariance: ArrayLike,
        args: Any = None,
        /,
    ) -> VariationalResult:
        state0 = jnp.asarray(initial_state)
        parameter_values = jnp.asarray(parameters)
        covariance0 = jnp.asarray(initial_covariance)
        dimension = int(state0.size)
        if (
            state0.shape != (dimension,)
            or covariance0.shape != (dimension, dimension)
            or parameter_values.shape != (self.parameter_dimension,)
        ):
            raise ValueError("Variational initial arrays have incompatible shapes.")
        phi0 = jnp.eye(dimension, dtype=state0.dtype)
        sensitivity0 = jnp.zeros(
            (dimension, self.parameter_dimension), dtype=state0.dtype
        )

        def derivative(time, combined):
            state, phi, sensitivity, covariance = combined
            state_rate = self.dynamics(time, state, parameter_values, args)
            jacobian_state = jax.jacfwd(self.dynamics, argnums=1)(
                time, state, parameter_values, args
            )
            jacobian_parameters = jax.jacfwd(self.dynamics, argnums=2)(
                time, state, parameter_values, args
            )
            return (
                state_rate,
                jacobian_state @ phi,
                jacobian_state @ sensitivity + jacobian_parameters,
                jacobian_state @ covariance
                + covariance @ jacobian_state.T
                + self.process_noise,
            )

        def add(values, rates, factor):
            return jax.tree.map(lambda value, rate: value + factor * rate, values, rates)

        def step(carry, interval):
            start, end = interval
            dt = end - start
            k1 = derivative(start, carry)
            k2 = derivative(start + 0.5 * dt, add(carry, k1, 0.5 * dt))
            k3 = derivative(start + 0.5 * dt, add(carry, k2, 0.5 * dt))
            k4 = derivative(end, add(carry, k3, dt))
            next_values = jax.tree.map(
                lambda value, a, b, c, d: value + dt / 6.0 * (a + 2.0 * b + 2.0 * c + d),
                carry,
                k1,
                k2,
                k3,
                k4,
            )
            valid = jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in next_values))
            )
            accepted = jax.tree.map(
                lambda new, old: jnp.where(valid, new, old), next_values, carry
            )
            return accepted, (*accepted, jnp.asarray(valid))

        intervals = jnp.stack((self.times[:-1], self.times[1:]), axis=-1)
        initial = (state0, phi0, sensitivity0, covariance0)
        _, outputs = jax.lax.scan(step, initial, intervals)
        return VariationalResult(
            self.times,
            jnp.concatenate((state0[None], outputs[0]), axis=0),
            jnp.concatenate((phi0[None], outputs[1]), axis=0),
            jnp.concatenate((sensitivity0[None], outputs[2]), axis=0),
            jnp.concatenate((covariance0[None], outputs[3]), axis=0),
            jnp.concatenate((jnp.asarray(True)[None], outputs[4])),
            self.plan_id,
        )


def apply_event_saltation(
    transition_matrix: ArrayLike, event: HybridEventSensitivityResult, /
) -> Array:
    matrix = jnp.asarray(transition_matrix)
    return event.saltation_matrix @ matrix


__all__ = ["VariationalPropagationPlan", "VariationalResult", "apply_event_saltation"]
