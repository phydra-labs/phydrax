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

import phydrax.linalg as la

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...observation import CholeskyCovarianceAction, CoordinateLayout
from ...optim import GaussNewton, least_squares, OptimizationTermination
from ...uq import (
    condition_gaussian_moments,
    first_order_gaussian_transform,
    gaussian_factor_from_covariance,
)
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

        def whitened_residual(parameters, context):
            predicted = self.observation_model(parameters, context).reshape(-1)
            return self.covariance.whiten(self.observed.reshape(-1) - predicted)

        optimization = least_squares(
            whitened_residual,
            initial,
            method=GaussNewton(),
            termination=OptimizationTermination(
                absolute_optimality=self.tolerance,
                relative_optimality=self.tolerance,
                absolute_step=self.tolerance,
                relative_step=self.tolerance,
                maximum_steps=self.maximum_iterations,
            ),
            args=args,
        )
        estimate = jnp.asarray(optimization.parameters)
        predicted = self.observation_model(estimate, args).reshape(-1)
        residual = self.observed.reshape(-1) - predicted
        linearization = la.prepare_linearization(
            lambda value: self.covariance.whiten(
                self.observation_model(value, args).reshape(-1)
            ),
            estimate,
        )
        entry_count = max(1, int(self.observed.size) * int(initial.size))
        jacobian = la.materialize(
            la.JacobianLinearOperator(linearization),
            la.MaterializationPolicy(
                max_entries=entry_count,
                max_bytes=max(1, entry_count * jnp.dtype(initial.dtype).itemsize),
            ),
        )
        information = jnp.conj(jacobian).T @ jacobian
        covariance_result = la.pseudoinverse(
            information,
            la.FactorizationPolicy(
                "svd",
                rank=la.RankPolicy(relative_cutoff=self.tolerance),
            ),
        )
        rank = jnp.asarray(covariance_result.diagnostics.rank).reshape((-1,))[0]
        covariance = covariance_result.value
        valid = (
            optimization.successful
            & covariance_result.successful
            & (rank == int(initial.size))
        )
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.NONCONVERGED),
        ).astype(jnp.int32)
        return OrbitDeterminationResult(
            estimate,
            covariance,
            residual,
            optimization.diagnostics.iterations,
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
            tolerance = 128.0 * int(state.size) * float(jnp.finfo(state.dtype).eps)
            state_factor = gaussian_factor_from_covariance(
                0.5 * (covariance + covariance.T),
                rank_tolerance=tolerance,
                hermitian_tolerance=tolerance,
                factor_id="sequential-od-prior",
            )
            transition_transform = first_order_gaussian_transform(
                lambda value: self.transition(previous_time, time, value, args),
                state,
                state_factor,
            )
            predicted_state = jnp.asarray(transition_transform.mean)
            predicted_covariance = (
                transition_transform.factor.covariance + self.process_covariance
            )
            predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T)
            predicted_factor = gaussian_factor_from_covariance(
                predicted_covariance,
                rank_tolerance=tolerance,
                hermitian_tolerance=tolerance,
                factor_id="sequential-od-prediction",
            )
            observation_transform = first_order_gaussian_transform(
                lambda value: self.observation(time, value, args),
                predicted_state,
                predicted_factor,
            )
            predicted_measurement = jnp.asarray(observation_transform.mean).reshape((-1,))
            observation_covariance = (
                observation_transform.factor.covariance + self.measurement_covariance
            )
            conditioning = condition_gaussian_moments(
                predicted_state,
                predicted_covariance,
                predicted_measurement,
                observation_covariance,
                observation_transform.cross_covariance,
                measurement,
                rank_tolerance=tolerance,
                moments_valid=(
                    state_factor.valid
                    & transition_transform.valid
                    & predicted_factor.valid
                    & observation_transform.valid
                ),
            )
            next_state = eqx.error_if(
                conditioning.mean,
                ~conditioning.valid,
                "Sequential orbit-determination Gaussian conditioning failed.",
            )
            next_covariance = conditioning.covariance
            innovation = conditioning.innovation
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
