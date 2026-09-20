#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class SensorConfiguration(StrictModule, NonTrainableState):
    coordinates: Array
    noise_covariance: Array
    channel_names: tuple[str, ...] = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    cadence_id: str = eqx.field(static=True)
    configuration_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        noise_covariance: ArrayLike,
        /,
        *,
        channel_names: Sequence[str],
        unit_contract_id: str,
        frame_id: str,
        geometry_id: str,
        cadence_id: str,
    ):
        points = jnp.asarray(coordinates)
        covariance = jnp.asarray(noise_covariance)
        channels = tuple(str(value) for value in channel_names)
        identifiers = tuple(
            str(value) for value in (unit_contract_id, frame_id, geometry_id, cadence_id)
        )
        if (
            points.ndim != 2
            or not channels
            or covariance.shape != (len(channels), len(channels))
        ):
            raise ValueError(
                "Sensor coordinates, channels, and noise covariance are inconsistent."
            )
        if any(not value for value in (*channels, *identifiers)):
            raise ValueError("Sensor identities must be non-empty.")
        eigenvalues = np.linalg.eigvalsh(np.asarray(0.5 * (covariance + covariance.T)))
        if np.any(eigenvalues < -64.0 * np.finfo(eigenvalues.dtype).eps):
            raise ValueError("Sensor noise covariance must be positive semidefinite.")
        self.coordinates = points
        self.noise_covariance = covariance
        self.channel_names = channels
        self.unit_contract_id = identifiers[0]
        self.frame_id = identifiers[1]
        self.geometry_id = identifiers[2]
        self.cadence_id = identifiers[3]
        self.configuration_id = canonical_fingerprint(
            {
                "kind": "sensor-configuration",
                "channels": list(channels),
                "units": identifiers[0],
                "frame": identifiers[1],
                "geometry": identifiers[2],
                "cadence": identifiers[3],
                "content": array_tree_fingerprint(
                    {"coordinates": points, "noise": covariance}
                )["sha256"],
            }
        )


class ObservationHistory(StrictModule, NonTrainableState):
    values: Array
    times: Array
    valid: Array
    reset: Array
    configuration: SensorConfiguration
    history_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        times: ArrayLike,
        valid: ArrayLike,
        reset: ArrayLike,
        configuration: SensorConfiguration,
        /,
    ):
        if not isinstance(configuration, SensorConfiguration):
            raise TypeError("configuration must be SensorConfiguration.")
        observations = jnp.asarray(values)
        coordinates = jnp.asarray(times)
        mask = jnp.asarray(valid, dtype=jnp.bool_)
        resets = jnp.asarray(reset, dtype=jnp.bool_)
        if observations.ndim != 2 or observations.shape[1] != len(
            configuration.channel_names
        ):
            raise ValueError("Observation history values have invalid shape.")
        if (
            coordinates.shape != observations.shape[:1]
            or mask.shape != observations.shape
            or resets.shape != observations.shape[:1]
        ):
            raise ValueError(
                "Observation history time, validity, and reset shapes are invalid."
            )
        self.values = observations
        self.times = coordinates
        self.valid = mask
        self.reset = resets
        self.configuration = configuration
        self.history_id = canonical_fingerprint(
            {
                "kind": "observation-history",
                "configuration": configuration.configuration_id,
                "content": array_tree_fingerprint(
                    {
                        "values": observations,
                        "times": coordinates,
                        "valid": mask,
                        "reset": resets,
                    }
                )["sha256"],
            }
        )


class LinearSensorHistoryEstimator(StrictModule, NonTrainableState):
    """Causal fixed-window linear map to a Gaussian reduced-state estimate."""

    coefficient_matrix: Array
    offset: Array
    covariance: Array
    history_length: int = eqx.field(static=True)
    configuration_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    estimator_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficient_matrix: ArrayLike,
        offset: ArrayLike,
        covariance: ArrayLike,
        /,
        *,
        history_length: int,
        configuration_id: str,
        partition_id: str,
    ):
        matrix = jnp.asarray(coefficient_matrix)
        offset_ = jnp.asarray(offset)
        covariance_ = jnp.asarray(covariance)
        length = int(history_length)
        if (
            matrix.ndim != 2
            or offset_.shape != (matrix.shape[0],)
            or covariance_.shape != (matrix.shape[0], matrix.shape[0])
        ):
            raise ValueError("Sensor estimator arrays have invalid shape.")
        if length <= 0 or matrix.shape[1] % length != 0:
            raise ValueError("history_length must divide the estimator input width.")
        identifiers = tuple(str(value) for value in (configuration_id, partition_id))
        if any(not value for value in identifiers):
            raise ValueError("Estimator identities must be non-empty.")
        eigenvalues = np.linalg.eigvalsh(np.asarray(0.5 * (covariance_ + covariance_.T)))
        if np.any(eigenvalues < -64.0 * np.finfo(eigenvalues.dtype).eps):
            raise ValueError("Estimator covariance must be positive semidefinite.")
        self.coefficient_matrix = matrix
        self.offset = offset_
        self.covariance = covariance_
        self.history_length = length
        self.configuration_id = identifiers[0]
        self.partition_id = identifiers[1]
        self.estimator_id = canonical_fingerprint(
            {
                "kind": "linear-sensor-history-estimator",
                "history_length": length,
                "configuration": identifiers[0],
                "partition": identifiers[1],
                "content": array_tree_fingerprint(
                    {"matrix": matrix, "offset": offset_, "covariance": covariance_}
                )["sha256"],
            }
        )

    def estimate(self, history: ObservationHistory, /) -> tuple[Array, Array, Array]:
        if history.configuration.configuration_id != self.configuration_id:
            raise ValueError("Sensor configuration does not match the estimator.")
        if history.values.shape[0] < self.history_length:
            return self.offset, self.covariance, jnp.asarray(False)
        values = history.values[-self.history_length :]
        valid = history.valid[-self.history_length :]
        admitted = jnp.all(valid) & ~jnp.any(history.reset[-self.history_length + 1 :])
        flattened = jnp.where(valid, values, 0.0).reshape((-1,))
        mean = self.offset + self.coefficient_matrix @ flattened
        return mean, self.covariance, admitted


def select_basis_sensors(basis_matrix: ArrayLike, count: int, /) -> Array:
    """Deterministic pivoted residual-energy sensor selection."""
    basis = np.asarray(basis_matrix)
    sensors = int(count)
    if basis.ndim != 2 or sensors <= 0 or sensors > min(basis.shape):
        raise ValueError("Sensor count must be positive and not exceed basis dimensions.")
    residual = basis.copy()
    selected: list[int] = []
    for _ in range(sensors):
        norms = np.sum(np.abs(residual) ** 2, axis=1)
        norms[selected] = -np.inf
        index = int(np.argmax(norms))
        if not np.isfinite(norms[index]) or norms[index] <= 0.0:
            raise ValueError(
                "Basis does not support the requested observable sensor rank."
            )
        selected.append(index)
        row = residual[index]
        denominator = np.vdot(row, row).real
        residual = residual - np.outer(residual @ np.conj(row), row) / denominator
    return jnp.asarray(selected, dtype=jnp.int32)


class ReducedKalmanAssimilator(StrictModule, NonTrainableState):
    transition_matrix: Array
    process_covariance: Array
    observation_matrix: Array
    measurement_covariance: Array
    assimilator_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition_matrix: ArrayLike,
        process_covariance: ArrayLike,
        observation_matrix: ArrayLike,
        measurement_covariance: ArrayLike,
        /,
    ):
        transition = jnp.asarray(transition_matrix)
        process = jnp.asarray(process_covariance)
        observation = jnp.asarray(observation_matrix)
        measurement = jnp.asarray(measurement_covariance)
        rank = transition.shape[0]
        if (
            transition.shape != (rank, rank)
            or process.shape != (rank, rank)
            or observation.ndim != 2
            or observation.shape[1] != rank
            or measurement.shape != (observation.shape[0], observation.shape[0])
        ):
            raise ValueError("Assimilation matrices have incompatible shape.")
        self.transition_matrix = transition
        self.process_covariance = process
        self.observation_matrix = observation
        self.measurement_covariance = measurement
        self.assimilator_id = canonical_fingerprint(
            {
                "kind": "reduced-kalman-assimilator",
                "content": array_tree_fingerprint(
                    {
                        "transition": transition,
                        "process": process,
                        "observation": observation,
                        "measurement": measurement,
                    }
                )["sha256"],
            }
        )

    def predict(self, mean: ArrayLike, covariance: ArrayLike, /) -> tuple[Array, Array]:
        state = jnp.asarray(mean)
        uncertainty = jnp.asarray(covariance)
        predicted = self.transition_matrix @ state
        predicted_covariance = (
            self.transition_matrix @ uncertainty @ self.transition_matrix.T
            + self.process_covariance
        )
        return predicted, predicted_covariance

    def update(
        self, mean: ArrayLike, covariance: ArrayLike, observation: ArrayLike, /
    ) -> tuple[Array, Array, Array]:
        state = jnp.asarray(mean)
        uncertainty = jnp.asarray(covariance)
        measured = jnp.asarray(observation)
        innovation = measured - self.observation_matrix @ state
        innovation_covariance = (
            self.observation_matrix @ uncertainty @ self.observation_matrix.T
            + self.measurement_covariance
        )
        gain = jnp.linalg.solve(
            innovation_covariance, self.observation_matrix @ uncertainty
        ).T
        updated = state + gain @ innovation
        identity = jnp.eye(state.size, dtype=state.dtype)
        stabilized = identity - gain @ self.observation_matrix
        updated_covariance = (
            stabilized @ uncertainty @ stabilized.T
            + gain @ self.measurement_covariance @ gain.T
        )
        nis = jnp.vdot(
            innovation, jnp.linalg.solve(innovation_covariance, innovation)
        ).real
        return updated, updated_covariance, nis


__all__ = [
    "LinearSensorHistoryEstimator",
    "ObservationHistory",
    "ReducedKalmanAssimilator",
    "SensorConfiguration",
    "select_basis_sensors",
]
