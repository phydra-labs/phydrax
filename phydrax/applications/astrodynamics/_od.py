#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...observation import CholeskyCovarianceAction, CoordinateLayout
from ._status import AstrodynamicsStatus


class OrbitDeterminationResult(StrictModule):
    estimate: Array
    covariance: Array
    residual: Array
    iterations: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class BatchOrbitDeterminationPlan(StrictModule, NonTrainableState):
    observation_model: Callable
    observed: Array
    covariance: CholeskyCovarianceAction
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        observation_model,
        observed,
        covariance_cholesky,
        /,
        *,
        maximum_iterations=12,
        tolerance=1.0e-10,
        model_id="batch-od",
    ):
        if not callable(observation_model):
            raise TypeError("observation_model must be callable.")
        observed_ = jnp.asarray(observed)
        root = jnp.asarray(covariance_cholesky)
        if root.shape != (observed_.size, observed_.size):
            raise ValueError("Observation covariance root has incompatible shape.")
        layout = CoordinateLayout(
            tuple(f"{model_id}:observation:{index}" for index in range(observed_.size))
        )
        self.observation_model = observation_model
        self.observed = observed_
        self.covariance = CholeskyCovarianceAction(root, layout)
        self.maximum_iterations = int(maximum_iterations)
        self.tolerance = float(tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "batch-orbit-determination",
                "model": str(model_id),
                "observations": int(observed_.size),
            }
        )

    @property
    def covariance_cholesky(self) -> Array:
        return self.covariance.lower_cholesky

    def solve(
        self, initial_parameters: ArrayLike, args: Any = None, /
    ) -> OrbitDeterminationResult:
        initial = jnp.asarray(initial_parameters)

        def step(index, carry):
            estimate, converged, first = carry
            predicted = self.observation_model(estimate, args).reshape(-1)
            residual = self.observed.reshape(-1) - predicted
            jacobian = jax.jacfwd(
                lambda value: self.observation_model(value, args).reshape(-1)
            )(estimate)
            whitened_residual = self.covariance.whiten(residual)
            whitened_jacobian = jax.vmap(self.covariance.whiten, in_axes=1, out_axes=1)(
                jacobian
            )
            information = whitened_jacobian.T @ whitened_jacobian
            rhs = whitened_jacobian.T @ whitened_residual
            update = jsp.linalg.solve(information, rhs, assume_a="sym")
            now = jnp.sqrt(jnp.sum(update * update)) <= self.tolerance * (
                1.0 + jnp.sqrt(jnp.sum(estimate * estimate))
            )
            candidate = estimate + update
            finite = jnp.all(jnp.isfinite(candidate))
            return (
                jnp.where(~converged & finite, candidate, estimate),
                converged | now,
                jnp.where((first < 0) & now, index + 1, first),
            )

        estimate, converged, iterations = jax.lax.fori_loop(
            0,
            self.maximum_iterations,
            step,
            (initial, jnp.asarray(False), jnp.asarray(-1, dtype=jnp.int32)),
        )
        predicted = self.observation_model(estimate, args).reshape(-1)
        residual = self.observed.reshape(-1) - predicted
        jacobian = jax.jacfwd(
            lambda value: self.observation_model(value, args).reshape(-1)
        )(estimate)
        whitened = jax.vmap(self.covariance.whiten, in_axes=1, out_axes=1)(jacobian)
        information = whitened.T @ whitened
        covariance = jsp.linalg.inv(information)
        valid = converged & jnp.all(jnp.isfinite(covariance))
        status = jnp.where(
            valid, int(AstrodynamicsStatus.SUCCESS), int(AstrodynamicsStatus.NONCONVERGED)
        ).astype(jnp.int32)
        return OrbitDeterminationResult(
            estimate,
            covariance,
            residual,
            jnp.where(iterations >= 0, iterations, self.maximum_iterations),
            valid,
            status,
            self.plan_id,
        )


class SequentialOrbitDeterminationPlan(StrictModule, NonTrainableState):
    transition: Callable
    observation: Callable
    process_covariance: Array
    measurement_covariance: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition,
        observation,
        process_covariance,
        measurement_covariance,
        /,
        *,
        model_id="sequential-od",
    ):
        if not callable(transition) or not callable(observation):
            raise TypeError("Sequential OD models must be callable.")
        self.transition = transition
        self.observation = observation
        self.process_covariance = jnp.asarray(process_covariance)
        self.measurement_covariance = jnp.asarray(measurement_covariance)
        self.plan_id = canonical_fingerprint(
            {"kind": "sequential-orbit-determination", "model": str(model_id)}
        )

    def filter(
        self, initial_state, initial_covariance, observations, times, args: Any = None, /
    ):
        state0 = jnp.asarray(initial_state)
        covariance0 = jnp.asarray(initial_covariance)
        observed = jnp.asarray(observations)
        times_ = jnp.asarray(times)

        def step(carry, item):
            state, covariance, previous_time = carry
            time, measurement = item
            predicted_state = self.transition(previous_time, time, state, args)
            transition_jacobian = jax.jacfwd(
                lambda value: self.transition(previous_time, time, value, args)
            )(state)
            predicted_covariance = (
                transition_jacobian @ covariance @ transition_jacobian.T
                + self.process_covariance
            )
            predicted_measurement = self.observation(time, predicted_state, args)
            observation_jacobian = jax.jacfwd(
                lambda value: self.observation(time, value, args)
            )(predicted_state)
            innovation_covariance = (
                observation_jacobian @ predicted_covariance @ observation_jacobian.T
                + self.measurement_covariance
            )
            gain = jsp.linalg.solve(
                innovation_covariance,
                observation_jacobian @ predicted_covariance,
                assume_a="sym",
            ).T
            innovation = measurement - predicted_measurement
            next_state = predicted_state + gain @ innovation
            identity = jnp.eye(state.size)
            next_covariance = (
                identity - gain @ observation_jacobian
            ) @ predicted_covariance @ (
                identity - gain @ observation_jacobian
            ).T + gain @ self.measurement_covariance @ gain.T
            return (next_state, next_covariance, time), (
                next_state,
                next_covariance,
                innovation,
            )

        _, outputs = jax.lax.scan(
            step, (state0, covariance0, times_[0]), (times_, observed)
        )
        return outputs


__all__ = [
    "BatchOrbitDeterminationPlan",
    "OrbitDeterminationResult",
    "SequentialOrbitDeterminationPlan",
]
