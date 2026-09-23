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

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
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


_COVARIANCE_POLICY = la.DensePropertyVerificationPolicy(
    require_hermitian=True,
    require_positive_semidefinite=True,
)


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
        observed_host = np.asarray(observed, dtype=np.float64)
        root_host = np.asarray(covariance_cholesky, dtype=np.float64)
        if observed_host.size < 1 or np.any(~np.isfinite(observed_host)):
            raise ValueError("Observed values must be nonempty and finite.")
        if (
            root_host.shape != (observed_host.size, observed_host.size)
            or np.any(~np.isfinite(root_host))
            or not np.allclose(root_host, np.tril(root_host))
            or np.any(np.diag(root_host) <= 0.0)
        ):
            raise ValueError("Observation covariance root must be finite lower Cholesky.")
        if isinstance(maximum_iterations, bool) or not isinstance(
            maximum_iterations, int
        ):
            raise TypeError("maximum_iterations must be an integer.")
        tolerance_ = float(tolerance)
        identifier = str(model_id).strip()
        if maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive.")
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if not identifier:
            raise ValueError("model_id must be non-empty.")
        observed_ = jnp.asarray(observed_host)
        root = jnp.asarray(root_host)
        layout = CoordinateLayout(
            tuple(f"{identifier}:observation:{index}" for index in range(observed_.size))
        )
        self.observation_model = observation_model
        self.observed = observed_
        self.covariance = CholeskyCovarianceAction(root, layout)
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "batch-orbit-determination",
                "model": identifier,
                "observed": array_tree_fingerprint(observed_host),
                "covariance_cholesky": array_tree_fingerprint(root_host),
                "maximum_iterations": maximum_iterations,
                "tolerance": tolerance_,
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
        entry_count = max(1, self.observed.size * initial.size)
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
            & (rank == initial.size)
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
        transition_id,
        observation_id,
    ):
        if not callable(transition) or not callable(observation):
            raise TypeError("Sequential OD models must be callable.")
        process = jnp.asarray(process_covariance)
        measurement = jnp.asarray(measurement_covariance)
        if (
            process.ndim != 2
            or process.shape[0] != process.shape[1]
            or measurement.ndim != 2
            or measurement.shape[0] != measurement.shape[1]
        ):
            raise ValueError("Sequential OD covariances must be square matrices.")
        process_evidence = la.verify_dense_properties(process, policy=_COVARIANCE_POLICY)
        measurement_evidence = la.verify_dense_properties(
            measurement, policy=_COVARIANCE_POLICY
        )
        process = eqx.error_if(
            process_evidence.matrix,
            ~process_evidence.successful,
            "Sequential OD process covariance must be finite positive semidefinite.",
        )
        measurement = eqx.error_if(
            measurement_evidence.matrix,
            ~measurement_evidence.successful,
            "Sequential OD measurement covariance must be finite positive semidefinite.",
        )
        declared = tuple(
            str(value).strip() for value in (model_id, transition_id, observation_id)
        )
        if any(not value for value in declared):
            raise ValueError(
                "Sequential OD model and callable identities must be non-empty."
            )
        self.transition = transition
        self.observation = observation
        self.process_covariance = process
        self.measurement_covariance = measurement
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sequential-orbit-determination",
                "model": declared[0],
                "transition": declared[1],
                "observation": declared[2],
                "process_covariance": array_tree_fingerprint(
                    np.asarray(process_covariance)
                ),
                "measurement_covariance": array_tree_fingerprint(
                    np.asarray(measurement_covariance)
                ),
            }
        )

    def filter(
        self, initial_state, initial_covariance, observations, times, args: Any = None, /
    ):
        state0 = jnp.asarray(initial_state)
        covariance0 = jnp.asarray(initial_covariance)
        observed = jnp.asarray(observations)
        times_ = jnp.asarray(times)
        if (
            state0.ndim != 1
            or state0.size != self.process_covariance.shape[0]
            or covariance0.shape != self.process_covariance.shape
            or observed.ndim != 2
            or observed.shape[1] != self.measurement_covariance.shape[0]
            or times_.shape != (observed.shape[0],)
            or observed.shape[0] < 1
        ):
            raise ValueError(
                "Sequential OD state, covariance, observations, and times are incompatible."
            )
        initial_evidence = la.verify_dense_properties(
            covariance0, policy=_COVARIANCE_POLICY
        )
        covariance0 = eqx.error_if(
            initial_evidence.matrix,
            ~initial_evidence.successful
            | jnp.any(~jnp.isfinite(state0))
            | jnp.any(~jnp.isfinite(observed))
            | jnp.any(~jnp.isfinite(times_))
            | jnp.any(jnp.diff(times_) <= 0.0),
            "Sequential OD inputs must be finite with valid covariance and increasing times.",
        )

        def step(carry, item):
            state, covariance, previous_time = carry
            time, measurement = item
            tolerance = 128.0 * state.size * float(jnp.finfo(state.dtype).eps)
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
