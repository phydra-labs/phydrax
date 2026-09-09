#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...observation import DiagonalCovarianceAction
from ...uq import TemporalDifferencePrior
from ._workflows import EnsembleInversionResult, EnsembleKalmanInversionPlan


class MonitoringEpoch(StrictModule, NonTrainableState):
    time_s: float = eqx.field(static=True)
    observation: Array
    covariance: DiagonalCovarianceAction
    geometry_id: str = eqx.field(static=True)
    acquisition_id: str = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_s: float,
        observation: ArrayLike,
        covariance: DiagonalCovarianceAction,
        geometry_id: str,
        acquisition_id: str,
        /,
    ):
        time = float(time_s)
        values = jnp.asarray(observation)
        geometry, acquisition = str(geometry_id).strip(), str(acquisition_id).strip()
        if (
            not np.isfinite(time)
            or not isinstance(covariance, DiagonalCovarianceAction)
            or values.shape != (covariance.layout.size,)
            or bool(jnp.any(~jnp.isfinite(values)))
            or not geometry
            or not acquisition
        ):
            raise ValueError(
                "Monitoring epoch time/data/covariance/identities are invalid."
            )
        self.time_s, self.observation, self.covariance = time, values, covariance
        self.geometry_id, self.acquisition_id = geometry, acquisition
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "monitoring-epoch",
                "time_s": time,
                "geometry": geometry,
                "acquisition": acquisition,
                "covariance": covariance.action_id,
                "observation": values,
            }
        )


class TimeLapseParameterization(StrictModule, NonTrainableState):
    epoch_count: int = eqx.field(static=True)
    parameter_count: int = eqx.field(static=True)
    temporal_prior: TemporalDifferencePrior

    def __init__(
        self,
        epoch_count: int,
        parameter_count: int,
        temporal_prior: TemporalDifferencePrior,
        /,
    ):
        epochs, parameters = int(epoch_count), int(parameter_count)
        if (
            epochs < 2
            or parameters <= 0
            or not isinstance(temporal_prior, TemporalDifferencePrior)
            or temporal_prior.times.size != epochs
        ):
            raise ValueError("Time-lapse shape and temporal prior are incompatible.")
        self.epoch_count, self.parameter_count = epochs, parameters
        self.temporal_prior = temporal_prior

    def reconstruct(self, baseline: ArrayLike, increments: ArrayLike, /) -> Array:
        baseline_ = jnp.asarray(baseline)
        increments_ = jnp.asarray(increments)
        if baseline_.shape != (self.parameter_count,) or increments_.shape != (
            self.epoch_count - 1,
            self.parameter_count,
        ):
            raise ValueError("Time-lapse baseline/increment shapes are invalid.")
        return jnp.concatenate(
            (baseline_[None, :], baseline_[None, :] + jnp.cumsum(increments_, axis=0)),
            axis=0,
        )

    def log_prior(self, states: ArrayLike, /) -> Array:
        values = jnp.asarray(states)
        if values.shape != (self.epoch_count, self.parameter_count):
            raise ValueError("Time-lapse state history has wrong shape.")
        return self.temporal_prior.log_prob(values)


class MonitoringState(StrictModule):
    ensemble: Array
    calibration_drift: Array
    epoch_index: Array
    time_s: Array
    random_key: Array
    plan_id: str = eqx.field(static=True)


class MonitoringStepResult(StrictModule):
    state: MonitoringState
    inversion: EnsembleInversionResult
    forecast_ensemble: Array
    successful: Array


class SequentialMonitoringPlan(StrictModule, NonTrainableState):
    """Ensemble forecast/update with explicit acquisition calibration drift."""

    epochs: tuple[MonitoringEpoch, ...]
    dynamics: Callable[[Array, Array, Array], Array] = eqx.field(static=True)
    predictions: tuple[Callable[[Array, Array], Array], ...] = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    prediction_ids: tuple[str, ...] = eqx.field(static=True)
    calibration_random_walk_scale: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        epochs: Sequence[MonitoringEpoch],
        dynamics: Callable[[Array, Array, Array], Array],
        predictions: Sequence[Callable[[Array, Array], Array]],
        calibration_random_walk_scale: ArrayLike,
        /,
        *,
        dynamics_id: str,
        prediction_ids: Sequence[str],
    ):
        epochs_ = tuple(epochs)
        predictors = tuple(predictions)
        dynamics_identity = str(dynamics_id).strip()
        prediction_identities = tuple(str(value).strip() for value in prediction_ids)
        if (
            len(epochs_) < 2
            or any(not isinstance(value, MonitoringEpoch) for value in epochs_)
            or any(
                epochs_[index + 1].time_s <= epochs_[index].time_s
                for index in range(len(epochs_) - 1)
            )
            or not callable(dynamics)
            or not dynamics_identity
            or len(prediction_identities) != len(predictors)
            or any(not value for value in prediction_identities)
            or len(predictors) != len(epochs_)
            or any(not callable(value) for value in predictors)
        ):
            raise ValueError("Monitoring epochs/dynamics/predictors are invalid.")
        scales = jnp.asarray(calibration_random_walk_scale)
        if scales.shape != (epochs_[0].observation.size,):
            raise ValueError(
                "Calibration drift scale must match first observation layout."
            )
        if any(epoch.observation.shape != scales.shape for epoch in epochs_):
            raise ValueError(
                "Monitoring epochs must share observation layout for drift state."
            )
        scales = eqx.error_if(
            scales,
            jnp.any(~jnp.isfinite(scales)) | jnp.any(scales < 0),
            "Calibration drift scales must be finite nonnegative.",
        )
        self.epochs, self.dynamics, self.predictions = epochs_, dynamics, predictors
        self.dynamics_id = dynamics_identity
        self.prediction_ids = prediction_identities
        self.calibration_random_walk_scale = scales
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sequential-monitoring",
                "epochs": [value.epoch_id for value in epochs_],
                "dynamics": dynamics_identity,
                "predictions": prediction_identities,
                "calibration_random_walk_scale": scales,
            }
        )

    def initialize(self, ensemble: ArrayLike, key: PRNGKeyArray, /) -> MonitoringState:
        members = jnp.asarray(ensemble)
        if members.ndim != 2 or members.shape[0] < 2:
            raise ValueError("Monitoring ensemble requires at least two members.")
        members = eqx.error_if(
            members,
            jnp.any(~jnp.isfinite(members)),
            "Monitoring initial ensemble must be finite.",
        )
        return MonitoringState(
            members,
            jnp.zeros((members.shape[0], self.epochs[0].observation.size)),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(self.epochs[0].time_s),
            key,
            self.plan_id,
        )

    def step(self, state: MonitoringState, /) -> MonitoringStepResult:
        if not isinstance(state, MonitoringState) or state.plan_id != self.plan_id:
            raise ValueError("Monitoring state belongs to a different plan.")
        index = int(np.asarray(state.epoch_index)) + 1
        if index >= len(self.epochs):
            raise ValueError("Monitoring state is already at the final epoch.")
        previous, epoch = self.epochs[index - 1], self.epochs[index]
        dt = jnp.asarray(epoch.time_s - previous.time_s)
        dynamic_key, drift_key, next_key = jax.random.split(state.random_key, 3)
        member_keys = jax.random.split(dynamic_key, state.ensemble.shape[0])
        forecast = jax.vmap(self.dynamics, in_axes=(0, None, 0))(
            state.ensemble, dt, member_keys
        )
        if forecast.shape != state.ensemble.shape:
            raise ValueError("Monitoring dynamics changed the ensemble member shape.")
        forecast = eqx.error_if(
            forecast,
            jnp.any(~jnp.isfinite(forecast)),
            "Monitoring dynamics produced a nonfinite forecast.",
        )
        drift = (
            state.calibration_drift
            + self.calibration_random_walk_scale
            * jax.random.normal(drift_key, state.calibration_drift.shape)
            * jnp.sqrt(dt)
        )
        predictor = self.predictions[index]

        def predict(augmented):
            parameter_count = forecast.shape[1]
            return (
                predictor(augmented[:parameter_count], jnp.asarray(epoch.time_s))
                + augmented[parameter_count:]
            )

        augmented = jnp.concatenate((forecast, drift), axis=1)
        inversion = EnsembleKalmanInversionPlan(
            epoch.observation, epoch.covariance
        ).update(augmented, predict)
        updated = inversion.ensemble
        parameter_count = forecast.shape[1]
        next_state = MonitoringState(
            updated[:, :parameter_count],
            updated[:, parameter_count:],
            jnp.asarray(index, dtype=jnp.int32),
            jnp.asarray(epoch.time_s),
            next_key,
            self.plan_id,
        )
        successful = inversion.finite & jnp.all(jnp.isfinite(next_state.ensemble))
        return MonitoringStepResult(next_state, inversion, forecast, successful)


__all__ = [
    "MonitoringEpoch",
    "MonitoringState",
    "MonitoringStepResult",
    "SequentialMonitoringPlan",
    "TimeLapseParameterization",
]
