#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Sparse time-averaged inference of additive periodic-flow model error.

The inferred field is an additive, divergence-free momentum-rate correction. It
is not an SGS stress and is not identifiable as an SGS stress from sparse
velocity observations: resolved-model, forcing, filtering, discretization, and
observation errors can all produce the same inferred correction.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from operator import index
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...control import (
    AbstractControlParameterization,
    PiecewiseConstantControlParameterization,
)
from ...dynamics import TimeGrid
from ._forcing import SolenoidalHermitianFourierBasis


_MODEL_ERROR_INTERPRETATION = (
    "additive-model-error-correction;not-identifiable-as-sgs-stress"
)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _real_array(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value)
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    return result.astype(jnp.result_type(result.dtype, jnp.float32))


class TimeAverageWindows(StrictModule, NonTrainableState):
    """Fixed sample times and normalized linear time-average weights.

    Each row of ``weights`` defines one average over ``sample_times``. Weights
    must be finite and nonnegative and each row must have positive mass; rows
    are normalized exactly once during construction.
    """

    sample_times: Array
    weights: Array
    labels: tuple[str, ...] = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    window_count: int = eqx.field(static=True)
    windows_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_times: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        labels: Sequence[str] | None = None,
    ):
        times = _real_array(sample_times, "Time-average sample times")
        raw_weights = _real_array(weights, "Time-average weights")
        host_times = np.asarray(times)
        host_weights = np.asarray(raw_weights)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("Time-average sample times must be a non-empty vector.")
        if not np.all(np.isfinite(host_times)) or np.any(np.diff(host_times) <= 0.0):
            raise ValueError(
                "Time-average sample times must be finite and strictly increasing."
            )
        if raw_weights.ndim != 2 or raw_weights.shape[1] != times.size:
            raise ValueError(
                "Time-average weights must have shape (windows, sample_times)."
            )
        if raw_weights.shape[0] == 0:
            raise ValueError("At least one time-average window is required.")
        row_mass = np.sum(host_weights, axis=1)
        if (
            not np.all(np.isfinite(host_weights))
            or np.any(host_weights < 0.0)
            or np.any(~np.isfinite(row_mass))
            or np.any(row_mass <= 0.0)
        ):
            raise ValueError(
                "Time-average weights must be finite, nonnegative, and nonempty per window."
            )
        normalized = host_weights / row_mass[:, None]
        if labels is None:
            labels_ = tuple(
                f"time-average-window-{window}"
                for window in range(int(raw_weights.shape[0]))
            )
        else:
            labels_ = tuple(str(label).strip() for label in labels)
        if (
            len(labels_) != raw_weights.shape[0]
            or len(set(labels_)) != len(labels_)
            or any(not label for label in labels_)
        ):
            raise ValueError("Time-average window labels must be non-empty and unique.")
        normalized_array = jax.lax.stop_gradient(
            jnp.asarray(normalized, dtype=raw_weights.dtype)
        )
        times = jax.lax.stop_gradient(times)
        self.sample_times = times
        self.weights = normalized_array
        self.labels = labels_
        self.sample_count = int(times.size)
        self.window_count = int(normalized_array.shape[0])
        self.windows_id = canonical_fingerprint(
            {
                "kind": "time-average-windows",
                "sample_times": array_tree_fingerprint(times),
                "weights": array_tree_fingerprint(normalized_array),
                "labels": list(labels_),
            }
        )

    @classmethod
    def from_bounds(
        cls,
        sample_times: ArrayLike,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
        /,
        *,
        labels: Sequence[str] | None = None,
    ) -> TimeAverageWindows:
        """Build composite-trapezoid averages with bounds on sample nodes.

        A zero-duration bound denotes an instantaneous sample. Nonzero windows
        include every supplied node between their endpoints and use the
        composite trapezoid rule, normalized by the window duration.
        """

        times = np.asarray(_real_array(sample_times, "Time-average sample times"))
        lower = np.asarray(_real_array(lower_bounds, "Time-average lower bounds"))
        upper = np.asarray(_real_array(upper_bounds, "Time-average upper bounds"))
        if lower.ndim != 1 or upper.shape != lower.shape or lower.size == 0:
            raise ValueError("Time-average bounds must be non-empty matching vectors.")
        if (
            not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(upper < lower)
        ):
            raise ValueError("Time-average bounds must be finite and ordered.")
        weights = np.zeros((lower.size, times.size), dtype=np.result_type(times, float))
        for window, (start, stop) in enumerate(zip(lower, upper, strict=True)):
            start_matches = np.flatnonzero(times == start)
            stop_matches = np.flatnonzero(times == stop)
            if start_matches.size != 1 or stop_matches.size != 1:
                raise ValueError(
                    "Every time-average bound must coincide with one sample time."
                )
            first = int(start_matches[0])
            last = int(stop_matches[0])
            if last < first:
                raise ValueError(
                    "Time-average bounds are inconsistent with sample order."
                )
            if first == last:
                weights[window, first] = 1.0
                continue
            local_times = times[first : last + 1]
            intervals = np.diff(local_times)
            weights[window, first] = 0.5 * intervals[0]
            weights[window, last] = 0.5 * intervals[-1]
            if last - first > 1:
                weights[window, first + 1 : last] = 0.5 * (intervals[:-1] + intervals[1:])
        return cls(times, weights, labels=labels)

    def average(self, samples: ArrayLike, /) -> Array:
        """Average an array whose leading dimension is the sample-time axis."""

        values = jnp.asarray(samples)
        if values.ndim < 1 or values.shape[0] != self.sample_count:
            raise ValueError(
                "Time-average samples must have the prepared leading sample axis."
            )
        return jnp.tensordot(self.weights, values, axes=((1,), (0,)))


class SparseTimeAverageObservationOperator(StrictModule, NonTrainableState):
    """Sparse channel selection after fixed linear time averaging."""

    windows: TimeAverageWindows
    window_indices: Array
    source_indices: Array
    source_size: int = eqx.field(static=True)
    observation_count: int = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        windows: TimeAverageWindows,
        source_size: int,
        window_indices: ArrayLike,
        source_indices: ArrayLike,
        /,
    ):
        if not isinstance(windows, TimeAverageWindows):
            raise TypeError("windows must be TimeAverageWindows.")
        if isinstance(source_size, bool):
            raise TypeError("source_size must be an integer.")
        source_size_ = index(source_size)
        if source_size_ <= 0:
            raise ValueError("source_size must be positive.")
        window_index = jnp.asarray(window_indices)
        source_index = jnp.asarray(source_indices)
        if (
            window_index.ndim != 1
            or source_index.shape != window_index.shape
            or window_index.size == 0
        ):
            raise ValueError(
                "Sparse observation indices must be non-empty matching vectors."
            )
        if not jnp.issubdtype(window_index.dtype, jnp.integer) or not jnp.issubdtype(
            source_index.dtype, jnp.integer
        ):
            raise TypeError("Sparse observation indices must have integer dtype.")
        host_windows = np.asarray(window_index)
        host_sources = np.asarray(source_index)
        if (
            np.any(host_windows < 0)
            or np.any(host_windows >= windows.window_count)
            or np.any(host_sources < 0)
            or np.any(host_sources >= source_size_)
        ):
            raise ValueError(
                "Sparse observation indices lie outside the prepared domains."
            )
        window_index = jax.lax.stop_gradient(window_index.astype(jnp.int32))
        source_index = jax.lax.stop_gradient(source_index.astype(jnp.int32))
        self.windows = windows
        self.window_indices = window_index
        self.source_indices = source_index
        self.source_size = source_size_
        self.observation_count = int(window_index.size)
        self.operator_id = canonical_fingerprint(
            {
                "kind": "sparse-time-average-observation-operator",
                "windows": windows.windows_id,
                "source_size": source_size_,
                "window_indices": array_tree_fingerprint(window_index),
                "source_indices": array_tree_fingerprint(source_index),
            }
        )

    def apply(self, samples: ArrayLike, /) -> Array:
        values = jnp.asarray(samples)
        expected = (self.windows.sample_count, self.source_size)
        if values.shape != expected:
            raise ValueError(
                f"Observation evaluator output must have shape {expected}; got {values.shape}."
            )
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("Observation evaluator output must be real-valued.")
        selected_values = values[:, self.source_indices]
        selected_weights = self.windows.weights[self.window_indices].T
        return jnp.sum(selected_weights * selected_values, axis=0)


class SparseTimeAverageObservationData(StrictModule, NonTrainableState):
    """Finite sparse observations with uncertainty and a fixed train split."""

    values: Array
    standard_deviations: Array
    training_mask: Array
    operator_id: str = eqx.field(static=True)
    observation_count: int = eqx.field(static=True)
    training_count: int = eqx.field(static=True)
    holdout_count: int = eqx.field(static=True)
    data_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: SparseTimeAverageObservationOperator,
        values: ArrayLike,
        standard_deviations: ArrayLike,
        training_mask: ArrayLike,
        /,
    ):
        if not isinstance(operator, SparseTimeAverageObservationOperator):
            raise TypeError("operator must be a SparseTimeAverageObservationOperator.")
        observed = _real_array(values, "Sparse observation values")
        uncertainty = _real_array(
            standard_deviations, "Sparse observation standard deviations"
        )
        mask = jnp.asarray(training_mask)
        expected = (operator.observation_count,)
        if (
            observed.shape != expected
            or uncertainty.shape != expected
            or mask.shape != expected
        ):
            raise ValueError(
                "Sparse observation data must match the observation operator."
            )
        if not jnp.issubdtype(mask.dtype, jnp.bool_):
            raise TypeError("training_mask must have boolean dtype.")
        host_observed = np.asarray(observed)
        host_uncertainty = np.asarray(uncertainty)
        host_mask = np.asarray(mask)
        if not np.all(np.isfinite(host_observed)):
            raise ValueError("Sparse observation values must be finite.")
        if not np.all(np.isfinite(host_uncertainty)) or np.any(host_uncertainty <= 0.0):
            raise ValueError(
                "Sparse observation standard deviations must be finite and positive."
            )
        training_count = int(np.count_nonzero(host_mask))
        if training_count == 0:
            raise ValueError(
                "At least one sparse observation must be assigned to training."
            )
        observed = jax.lax.stop_gradient(observed)
        uncertainty = jax.lax.stop_gradient(uncertainty)
        mask = jax.lax.stop_gradient(mask)
        self.values = observed
        self.standard_deviations = uncertainty
        self.training_mask = mask
        self.operator_id = operator.operator_id
        self.observation_count = operator.observation_count
        self.training_count = training_count
        self.holdout_count = operator.observation_count - training_count
        self.data_id = canonical_fingerprint(
            {
                "kind": "sparse-time-average-observation-data",
                "operator": operator.operator_id,
                "values": array_tree_fingerprint(observed),
                "standard_deviations": array_tree_fingerprint(uncertainty),
                "training_mask": array_tree_fingerprint(mask),
            }
        )

    @property
    def holdout_mask(self) -> Array:
        return ~self.training_mask


class ModelErrorRegularizationEvidence(StrictModule):
    amplitude: Array
    temporal_difference: Array
    total: Array
    regularization_id: str = eqx.field(static=True)


class QuadraticModelErrorRegularization(StrictModule, NonTrainableState):
    """Mean-square amplitude and adjacent-window difference regularization."""

    amplitude_weight: float = eqx.field(static=True)
    temporal_difference_weight: float = eqx.field(static=True)
    regularization_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        amplitude_weight: float = 0.0,
        temporal_difference_weight: float = 0.0,
    ):
        amplitude = float(amplitude_weight)
        temporal = float(temporal_difference_weight)
        if (
            not math.isfinite(amplitude)
            or amplitude < 0.0
            or not math.isfinite(temporal)
            or temporal < 0.0
        ):
            raise ValueError(
                "Model-error regularization weights must be finite and nonnegative."
            )
        self.amplitude_weight = amplitude
        self.temporal_difference_weight = temporal
        self.regularization_id = canonical_fingerprint(
            {
                "kind": "quadratic-model-error-regularization",
                "amplitude_weight": amplitude,
                "temporal_difference_weight": temporal,
                "normalization": "mean-square-per-term",
            }
        )

    def evaluate(self, parameters: ArrayLike, /) -> ModelErrorRegularizationEvidence:
        values = _real_array(parameters, "Model-error parameters")
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("Model-error parameters must be a non-empty rank-two array.")
        amplitude = 0.5 * self.amplitude_weight * jnp.mean(values * values)
        if values.shape[0] > 1:
            difference = jnp.diff(values, axis=0)
            temporal = (
                0.5 * self.temporal_difference_weight * jnp.mean(difference * difference)
            )
        else:
            temporal = jnp.zeros((), dtype=values.dtype)
        return ModelErrorRegularizationEvidence(
            amplitude=amplitude,
            temporal_difference=temporal,
            total=amplitude + temporal,
            regularization_id=self.regularization_id,
        )


class PeriodicModelErrorParameterization(
    AbstractControlParameterization, NonTrainableState
):
    """Piecewise-constant additive correction in an exact solenoidal basis.

    ``SolenoidalHermitianFourierBasis`` spans transverse Fourier polarizations,
    which is the exact periodic Leray subspace with Hermitian reality and the
    zero/Nyquist policies already enforced. The output is an additive modal
    momentum rate, not a constitutive SGS-stress representation.
    """

    basis: SolenoidalHermitianFourierBasis
    time_grid: TimeGrid
    coordinate_schedule: PiecewiseConstantControlParameterization
    base_forcing_id: str = eqx.field(static=True)
    model_interpretation: str = eqx.field(static=True)

    def __init__(
        self,
        basis: SolenoidalHermitianFourierBasis,
        time_grid: TimeGrid,
        /,
        *,
        base_forcing_id: str,
    ):
        if not isinstance(basis, SolenoidalHermitianFourierBasis):
            raise TypeError("basis must be a SolenoidalHermitianFourierBasis.")
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        forcing_id = _identifier(base_forcing_id, "base_forcing_id")
        parameterization_id = canonical_fingerprint(
            {
                "kind": "periodic-additive-model-error-parameterization",
                "basis": basis.basis_id,
                "time_grid": time_grid.time_id,
                "base_forcing": forcing_id,
                "schedule": "piecewise-constant-left-endpoint",
                "interpretation": _MODEL_ERROR_INTERPRETATION,
            }
        )
        schedule = PiecewiseConstantControlParameterization(
            time_grid,
            (basis.coordinate_size,),
            parameterization_id=f"model-error-coordinates:{parameterization_id}",
        )
        self.basis = basis
        self.time_grid = time_grid
        self.coordinate_schedule = schedule
        self.control_shape = basis.projector.state_shape
        self.parameter_shape = schedule.parameter_shape
        self.parameterization_id = parameterization_id
        self.approximation_id = (
            "additive-model-error:piecewise-constant:exact-periodic-leray"
        )
        self.base_forcing_id = forcing_id
        self.model_interpretation = _MODEL_ERROR_INTERPRETATION

    def validate_parameters(self, parameters: ArrayLike, /) -> Array:
        values = _real_array(parameters, "Model-error parameters")
        if values.shape != self.parameter_shape:
            raise ValueError(
                f"Model-error parameters must have shape {self.parameter_shape}; "
                f"got {values.shape}."
            )
        return eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Model-error parameters must be finite.",
        )

    def evaluate(
        self,
        coefficients: ArrayLike,
        time: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        state: ArrayLike | None = None,
    ) -> Array:
        del state
        coordinates = self.coordinate_schedule.evaluate(
            coefficients,
            time,
            case_shape=case_shape,
        )
        leading_shape = coordinates.shape[:-1]
        flat_coordinates = coordinates.reshape((-1, self.basis.coordinate_size))
        flat_corrections = jax.vmap(self.basis.evaluate)(flat_coordinates)
        return flat_corrections.reshape(leading_shape + self.control_shape)

    def sample(
        self,
        coefficients: ArrayLike,
        times: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> Array:
        return self.evaluate(coefficients, times, case_shape=case_shape)


class ModelErrorAssimilationIdentity(StrictModule, NonTrainableState):
    """Exact scientific identities required by an assimilation runtime."""

    problem_id: str = eqx.field(static=True)
    compiler_id: str = eqx.field(static=True)
    filter_id: str = eqx.field(static=True)
    forcing_id: str = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)
    identity_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem_id: str,
        compiler_id: str,
        filter_id: str,
        forcing_id: str,
        observation_id: str,
    ):
        problem = _identifier(problem_id, "problem_id")
        compiler = _identifier(compiler_id, "compiler_id")
        filter_ = _identifier(filter_id, "filter_id")
        forcing = _identifier(forcing_id, "forcing_id")
        observation = _identifier(observation_id, "observation_id")
        self.problem_id = problem
        self.compiler_id = compiler
        self.filter_id = filter_
        self.forcing_id = forcing
        self.observation_id = observation
        self.identity_id = canonical_fingerprint(
            {
                "kind": "model-error-assimilation-identity",
                "problem": problem,
                "compiler": compiler,
                "filter": filter_,
                "forcing": forcing,
                "observation": observation,
            }
        )

    def require_compatible(self, other: ModelErrorAssimilationIdentity, /) -> None:
        if not isinstance(other, ModelErrorAssimilationIdentity):
            raise TypeError("other must be a ModelErrorAssimilationIdentity.")
        expected = (
            self.problem_id,
            self.compiler_id,
            self.filter_id,
            self.forcing_id,
            self.observation_id,
        )
        supplied = (
            other.problem_id,
            other.compiler_id,
            other.filter_id,
            other.forcing_id,
            other.observation_id,
        )
        names = (
            "problem_id",
            "compiler_id",
            "filter_id",
            "forcing_id",
            "observation_id",
        )
        mismatches = tuple(
            name
            for name, expected_value, supplied_value in zip(
                names, expected, supplied, strict=True
            )
            if expected_value != supplied_value
        )
        if mismatches:
            raise ValueError(
                "Assimilation identity mismatch for " + ", ".join(mismatches) + "."
            )


class ModelErrorRolloutEvaluator(StrictModule, NonTrainableState):
    """Supplied fixed-shape rollout and observation-evaluator callbacks.

    ``rollout(parameterization, parameters, sample_times)`` must execute fixed
    control flow and return any JAX PyTree.
    ``evaluator(rollout_result, sample_times)`` must return a real array of shape
    ``(sample_count, source_size)``.
    """

    identity: ModelErrorAssimilationIdentity
    rollout: Callable[[PeriodicModelErrorParameterization, Array, Array], Any] = (
        eqx.field(static=True)
    )
    evaluator: Callable[[Any, Array], ArrayLike] = eqx.field(static=True)
    rollout_id: str = eqx.field(static=True)
    evaluator_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        identity: ModelErrorAssimilationIdentity,
        rollout: Callable[[PeriodicModelErrorParameterization, Array, Array], Any],
        evaluator: Callable[[Any, Array], ArrayLike],
        /,
        *,
        rollout_id: str,
        evaluator_id: str,
    ):
        if not isinstance(identity, ModelErrorAssimilationIdentity):
            raise TypeError("identity must be a ModelErrorAssimilationIdentity.")
        if not callable(rollout) or not callable(evaluator):
            raise TypeError("rollout and evaluator must be callable.")
        rollout_name = _identifier(rollout_id, "rollout_id")
        evaluator_name = _identifier(evaluator_id, "evaluator_id")
        self.identity = identity
        self.rollout = rollout
        self.evaluator = evaluator
        self.rollout_id = rollout_name
        self.evaluator_id = evaluator_name
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "model-error-rollout-evaluator",
                "identity": identity.identity_id,
                "rollout": rollout_name,
                "evaluator": evaluator_name,
            }
        )


class ModelErrorAssimilationEvidence(StrictModule):
    predicted_observations: Array
    residuals: Array
    standardized_residuals: Array
    training_data_misfit: Array
    holdout_data_misfit: Array
    regularization: ModelErrorRegularizationEvidence
    finite: Array
    successful: Array
    training_count: int = eqx.field(static=True)
    holdout_count: int = eqx.field(static=True)
    model_interpretation: str = eqx.field(static=True)
    identity_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    objective_id: str = eqx.field(static=True)


class ModelErrorAssimilationResult(StrictModule):
    value: Array
    evidence: ModelErrorAssimilationEvidence
    objective_id: str = eqx.field(static=True)


class ModelErrorAssimilationValueGradient(StrictModule):
    value: Array
    gradient: Array
    evidence: ModelErrorAssimilationEvidence
    objective_id: str = eqx.field(static=True)


class ModelErrorAssimilationObjective(StrictModule, NonTrainableState):
    """Differentiable training objective with separately reported holdout error."""

    parameterization: PeriodicModelErrorParameterization
    operator: SparseTimeAverageObservationOperator
    observations: SparseTimeAverageObservationData
    regularization: QuadraticModelErrorRegularization
    runtime: ModelErrorRolloutEvaluator
    identity: ModelErrorAssimilationIdentity
    objective_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameterization: PeriodicModelErrorParameterization,
        operator: SparseTimeAverageObservationOperator,
        observations: SparseTimeAverageObservationData,
        regularization: QuadraticModelErrorRegularization,
        runtime: ModelErrorRolloutEvaluator,
        /,
        *,
        problem_id: str,
        compiler_id: str,
        filter_id: str,
    ):
        if not isinstance(parameterization, PeriodicModelErrorParameterization):
            raise TypeError(
                "parameterization must be PeriodicModelErrorParameterization."
            )
        if not isinstance(operator, SparseTimeAverageObservationOperator):
            raise TypeError("operator must be SparseTimeAverageObservationOperator.")
        if not isinstance(observations, SparseTimeAverageObservationData):
            raise TypeError("observations must be SparseTimeAverageObservationData.")
        if not isinstance(regularization, QuadraticModelErrorRegularization):
            raise TypeError("regularization must be QuadraticModelErrorRegularization.")
        if not isinstance(runtime, ModelErrorRolloutEvaluator):
            raise TypeError("runtime must be ModelErrorRolloutEvaluator.")
        if observations.operator_id != operator.operator_id:
            raise ValueError(
                "Sparse observation data and operator identities do not match."
            )
        identity = ModelErrorAssimilationIdentity(
            problem_id=problem_id,
            compiler_id=compiler_id,
            filter_id=filter_id,
            forcing_id=parameterization.base_forcing_id,
            observation_id=observations.data_id,
        )
        identity.require_compatible(runtime.identity)
        self.parameterization = parameterization
        self.operator = operator
        self.observations = observations
        self.regularization = regularization
        self.runtime = runtime
        self.identity = identity
        self.objective_id = canonical_fingerprint(
            {
                "kind": "sparse-time-averaged-model-error-assimilation-objective",
                "parameterization": parameterization.parameterization_id,
                "operator": operator.operator_id,
                "observations": observations.data_id,
                "regularization": regularization.regularization_id,
                "runtime": runtime.runtime_id,
                "identity": identity.identity_id,
                "holdout_policy": "report-only",
                "interpretation": _MODEL_ERROR_INTERPRETATION,
            }
        )

    def value_with_evidence(
        self, parameters: ArrayLike, /
    ) -> tuple[Array, ModelErrorAssimilationEvidence]:
        values = self.parameterization.validate_parameters(parameters)
        rollout_result = self.runtime.rollout(
            self.parameterization,
            values,
            self.operator.windows.sample_times,
        )
        evaluated = self.runtime.evaluator(
            rollout_result,
            self.operator.windows.sample_times,
        )
        predicted = self.operator.apply(evaluated)
        residuals = predicted - self.observations.values
        standardized = residuals / self.observations.standard_deviations
        square = standardized * standardized
        training_mask = self.observations.training_mask
        training_misfit = (
            0.5
            * jnp.sum(jnp.where(training_mask, square, 0.0))
            / float(self.observations.training_count)
        )
        holdout_divisor = float(max(self.observations.holdout_count, 1))
        holdout_misfit = (
            0.5 * jnp.sum(jnp.where(~training_mask, square, 0.0)) / holdout_divisor
        )
        regularization = self.regularization.evaluate(values)
        value = training_misfit + regularization.total
        finite = (
            jnp.all(jnp.isfinite(predicted))
            & jnp.all(jnp.isfinite(residuals))
            & jnp.isfinite(training_misfit)
            & jnp.isfinite(holdout_misfit)
            & jnp.isfinite(regularization.total)
            & jnp.isfinite(value)
        )
        evidence = ModelErrorAssimilationEvidence(
            predicted_observations=predicted,
            residuals=residuals,
            standardized_residuals=standardized,
            training_data_misfit=training_misfit,
            holdout_data_misfit=holdout_misfit,
            regularization=regularization,
            finite=finite,
            successful=finite,
            training_count=self.observations.training_count,
            holdout_count=self.observations.holdout_count,
            model_interpretation=_MODEL_ERROR_INTERPRETATION,
            identity_id=self.identity.identity_id,
            runtime_id=self.runtime.runtime_id,
            objective_id=self.objective_id,
        )
        return value, evidence

    def evaluate(self, parameters: ArrayLike, /) -> ModelErrorAssimilationResult:
        value, evidence = self.value_with_evidence(parameters)
        return ModelErrorAssimilationResult(value, evidence, self.objective_id)

    def __call__(self, parameters: ArrayLike, /) -> Array:
        return self.value_with_evidence(parameters)[0]

    def value_and_gradient(
        self, parameters: ArrayLike, /
    ) -> ModelErrorAssimilationValueGradient:
        values = self.parameterization.validate_parameters(parameters)
        (value, evidence), gradient = jax.value_and_grad(
            self.value_with_evidence,
            has_aux=True,
        )(values)
        return ModelErrorAssimilationValueGradient(
            value=value,
            gradient=gradient,
            evidence=evidence,
            objective_id=self.objective_id,
        )


__all__ = [
    "ModelErrorAssimilationEvidence",
    "ModelErrorAssimilationIdentity",
    "ModelErrorAssimilationObjective",
    "ModelErrorAssimilationResult",
    "ModelErrorAssimilationValueGradient",
    "ModelErrorRegularizationEvidence",
    "ModelErrorRolloutEvaluator",
    "PeriodicModelErrorParameterization",
    "QuadraticModelErrorRegularization",
    "SparseTimeAverageObservationData",
    "SparseTimeAverageObservationOperator",
    "TimeAverageWindows",
]
