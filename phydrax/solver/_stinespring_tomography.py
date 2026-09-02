#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..metrix import (
    ComplexEuclideanManifold,
    ComplexStiefelManifold,
    faithful_density_from_cholesky,
)
from ..tensor_network import CausalProcessTensor, SequentialStinespringProcess
from ._process_tomography import (
    ProcessTomographyExperiment,
    tomography_designs_disjoint,
)


class StinespringTomographyProblem(StrictModule):
    model: SequentialStinespringProcess
    experiments: tuple[ProcessTomographyExperiment, ...]
    problem_id: str

    def __init__(
        self,
        model: SequentialStinespringProcess,
        experiments: Sequence[ProcessTomographyExperiment],
        /,
        *,
        problem_id: str = "stinespring-process-tomography",
    ):
        values = tuple(experiments)
        if not values:
            raise ValueError("At least one tomography experiment is required.")
        self.model = model
        self.experiments = values
        self.problem_id = str(problem_id)


class StinespringTomographyResult(StrictModule):
    model: SequentialStinespringProcess
    loss_history: Array
    held_out_loss: Array
    identifiability_rank: Array
    nullity: Array
    singular_values: Array
    physical_parameter_count: Array
    coordinate_count: Array
    quotient_identified: Array
    underidentified: Array
    execution_valid: Array
    valid: Array

    def __init__(
        self,
        model: SequentialStinespringProcess,
        loss_history: ArrayLike,
        held_out_loss: ArrayLike,
        identifiability_rank: ArrayLike,
        nullity: ArrayLike,
        singular_values: ArrayLike,
        physical_parameter_count: ArrayLike,
        coordinate_count: ArrayLike,
        /,
    ):
        self.model = model
        self.loss_history = jnp.asarray(loss_history)
        self.held_out_loss = jnp.asarray(held_out_loss)
        self.identifiability_rank = jnp.asarray(identifiability_rank)
        self.nullity = jnp.asarray(nullity)
        self.singular_values = jnp.asarray(singular_values)
        self.physical_parameter_count = jnp.asarray(physical_parameter_count)
        self.coordinate_count = jnp.asarray(coordinate_count)
        self.quotient_identified = (
            self.identifiability_rank == self.physical_parameter_count
        )
        self.underidentified = ~self.quotient_identified
        self.execution_valid = (
            model.materialize().valid
            & model.gauge_report().valid
            & jnp.all(jnp.isfinite(self.loss_history))
            & jnp.isfinite(self.held_out_loss)
            & jnp.all(jnp.isfinite(self.singular_values))
            & (self.identifiability_rank <= self.coordinate_count)
        )
        self.valid = self.execution_valid & self.quotient_identified


def _materialize(
    model: SequentialStinespringProcess,
    factor: Array,
    isometries: tuple[Array, ...],
) -> CausalProcessTensor:
    composite = model.spec.system_dimension * model.spec.memory_dimension
    channels = tuple(
        value.reshape((environment, composite, composite))
        for value, environment in zip(
            isometries, model.environment_dimensions, strict=True
        )
    )
    return CausalProcessTensor(
        model.spec,
        faithful_density_from_cholesky(factor),
        channels,
        process_id=model.process_id,
    )


def _nll(
    process: CausalProcessTensor,
    experiments: tuple[ProcessTomographyExperiment, ...],
) -> Array:
    if not experiments:
        raise ValueError("Stinespring fitting requires tomography experiments.")
    if any(not bool(experiment.valid) for experiment in experiments):
        raise ValueError("Stinespring fitting requires valid tomography experiments.")
    probabilities = jnp.stack(
        [experiment.probability(process) for experiment in experiments]
    )
    counts = jnp.stack([experiment.count for experiment in experiments])
    failures = jnp.stack(
        [experiment.trials - experiment.count for experiment in experiments]
    )
    support_floor = jnp.asarray(1e-6, dtype=probabilities.real.dtype)
    safe = (probabilities + support_floor) / (1.0 + 2.0 * support_floor)
    return jnp.sum(-counts * jnp.log(safe) - failures * jnp.log1p(-safe))


def _physical_parameter_count(model: SequentialStinespringProcess, /) -> int:
    return int(model.gauge_report().physical_parameter_count)


def fit_stinespring_process(
    problem: StinespringTomographyProblem,
    /,
    *,
    iterations: int = 100,
    learning_rate: float = 1e-2,
    held_out_experiments: Sequence[ProcessTomographyExperiment] = (),
    identifiability_tolerance: float = 1e-8,
    optimize_isometries: bool = True,
) -> StinespringTomographyResult:
    iteration_count = int(iterations)
    tolerance = float(identifiability_tolerance)
    learning_rate_ = float(learning_rate)
    if iteration_count <= 0:
        raise ValueError("iterations must be positive.")
    if (
        not jnp.isfinite(tolerance)
        or tolerance <= 0.0
        or not jnp.isfinite(learning_rate_)
        or learning_rate_ <= 0.0
    ):
        raise ValueError(
            "Learning rate and identifiability tolerance must be finite and positive."
        )
    factor = problem.model.initial_factor
    isometries = problem.model.isometries
    factor_manifold = ComplexEuclideanManifold(factor.shape)

    def loss(factor_value, isometry_values):
        return _nll(
            _materialize(problem.model, factor_value, isometry_values),
            problem.experiments,
        )

    value_and_grad = jax.value_and_grad(loss, argnums=(0, 1))
    history = []
    for _ in range(iteration_count):
        value, (factor_gradient, isometry_gradients) = value_and_grad(factor, isometries)
        factor_direction = factor_manifold.egrad_to_rgrad(factor, factor_gradient)
        factor = factor_manifold.retract(
            factor,
            -learning_rate_ * factor_direction,
        )
        if optimize_isometries:
            updated = []
            for isometry, gradient in zip(isometries, isometry_gradients, strict=True):
                manifold = ComplexStiefelManifold(isometry.shape[0], isometry.shape[1])
                tangent = manifold.egrad_to_rgrad(isometry, gradient)
                updated.append(manifold.retract(isometry, -learning_rate_ * tangent))
            isometries = tuple(updated)
        history.append(value)
    held_out_values = tuple(held_out_experiments)
    if held_out_values and not tomography_designs_disjoint(
        problem.experiments,
        held_out_values,
    ):
        raise ValueError(
            "Held-out tomography experiments must be disjoint from training."
        )
    model = SequentialStinespringProcess(
        problem.model.spec,
        factor,
        isometries,
        problem.model.environment_dimensions,
        process_id=problem.model.process_id,
    )
    held_out = (
        _nll(model.materialize(), held_out_values)
        if held_out_values
        else jnp.asarray(0.0)
    )
    values = (factor,) + isometries
    sizes = tuple(value.size for value in values)
    shapes = tuple(value.shape for value in values)
    coordinate_count = 2 * sum(sizes)
    coordinates = jnp.zeros((coordinate_count,), dtype=jnp.real(factor).dtype)

    def probabilities(parameters):
        candidates = []
        cursor = 0
        for index, (base, size, shape) in enumerate(
            zip(values, sizes, shapes, strict=True)
        ):
            real = parameters[cursor : cursor + size]
            imaginary = parameters[cursor + size : cursor + 2 * size]
            perturbation = (real + 1j * imaginary).reshape(shape)
            if index == 0:
                candidates.append(base + perturbation)
            else:
                manifold = ComplexStiefelManifold(base.shape[0], base.shape[1])
                candidates.append(manifold.retract(base, perturbation))
            cursor += 2 * size
        process = _materialize(problem.model, candidates[0], tuple(candidates[1:]))
        return jnp.stack(
            [experiment.probability(process) for experiment in problem.experiments]
        )

    jacobian = jax.jacfwd(probabilities)(coordinates)
    singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
    scale = jnp.max(singular_values, initial=0.0)
    rank = jnp.sum(singular_values > tolerance * jnp.maximum(scale, 1e-30))
    nullity = coordinate_count - rank
    return StinespringTomographyResult(
        model,
        jnp.stack(history),
        held_out,
        rank,
        nullity,
        singular_values,
        _physical_parameter_count(model),
        coordinate_count,
    )


class ProcessMemoryRefitResult(StrictModule):
    """Physical reduced-memory fit with training and held-out process errors."""

    process: CausalProcessTensor
    tomography: StinespringTomographyResult
    training_observed_probabilities: Array
    training_fitted_probabilities: Array
    held_out_observed_probabilities: Array
    held_out_initial_probabilities: Array
    held_out_fitted_probabilities: Array
    maximum_training_probability_error: Array
    maximum_held_out_initial_probability_error: Array
    maximum_held_out_probability_error: Array
    post_fit_to_pre_fit_error_ratio: Array
    recovery_improved: Array
    valid: Array
    source_memory_dimension: int
    retained_memory_dimension: int

    def __init__(
        self,
        source: CausalProcessTensor,
        tomography: StinespringTomographyResult,
        training_observed_probabilities: ArrayLike,
        training_fitted_probabilities: ArrayLike,
        held_out_observed_probabilities: ArrayLike,
        held_out_initial_probabilities: ArrayLike,
        held_out_fitted_probabilities: ArrayLike,
        /,
        *,
        probability_tolerance: float,
    ):
        process = tomography.model.materialize()
        training_observed = jnp.asarray(training_observed_probabilities)
        training_fitted = jnp.asarray(training_fitted_probabilities)
        held_out_observed = jnp.asarray(held_out_observed_probabilities)
        held_out_initial = jnp.asarray(held_out_initial_probabilities)
        held_out_fitted = jnp.asarray(held_out_fitted_probabilities)
        if training_observed.shape != training_fitted.shape:
            raise ValueError("Training process probabilities must share shape.")
        if (
            held_out_observed.shape != held_out_initial.shape
            or held_out_observed.shape != held_out_fitted.shape
        ):
            raise ValueError("Held-out process probabilities must share shape.")
        if training_observed.size < 1 or held_out_observed.size < 1:
            raise ValueError("Memory refit evidence requires nonempty probability sets.")
        training_error = jnp.max(jnp.abs(training_fitted - training_observed))
        initial_error = jnp.max(jnp.abs(held_out_initial - held_out_observed))
        held_out_error = jnp.max(jnp.abs(held_out_fitted - held_out_observed))
        ratio = held_out_error / jnp.maximum(initial_error, 1e-12)
        recovery_improved = (initial_error > 1e-8) & (held_out_error < initial_error)
        tolerance = float(probability_tolerance)
        self.process = process
        self.tomography = tomography
        self.training_observed_probabilities = training_observed
        self.training_fitted_probabilities = training_fitted
        self.held_out_observed_probabilities = held_out_observed
        self.held_out_initial_probabilities = held_out_initial
        self.held_out_fitted_probabilities = held_out_fitted
        self.maximum_training_probability_error = training_error
        self.maximum_held_out_initial_probability_error = initial_error
        self.maximum_held_out_probability_error = held_out_error
        self.post_fit_to_pre_fit_error_ratio = ratio
        self.recovery_improved = recovery_improved
        self.source_memory_dimension = source.spec.memory_dimension
        self.retained_memory_dimension = process.spec.memory_dimension
        self.valid = (
            source.valid
            & tomography.valid
            & process.valid
            & recovery_improved
            & jnp.all(jnp.isfinite(training_observed))
            & jnp.all(jnp.isfinite(training_fitted))
            & jnp.all(jnp.isfinite(held_out_observed))
            & jnp.all(jnp.isfinite(held_out_initial))
            & jnp.all(jnp.isfinite(held_out_fitted))
            & (training_error <= tolerance)
            & (held_out_error <= tolerance)
        )


def _experiment_probabilities(
    process: CausalProcessTensor,
    experiments: tuple[ProcessTomographyExperiment, ...],
    /,
) -> Array:
    return jnp.stack([experiment.probability(process) for experiment in experiments])


def _experiment_observed_probabilities(
    experiments: tuple[ProcessTomographyExperiment, ...],
    /,
) -> Array:
    if any(
        not bool(experiment.valid) or not bool(experiment.trials > 0.0)
        for experiment in experiments
    ):
        raise ValueError("Observed process probabilities require valid trials.")
    return jnp.stack([experiment.count / experiment.trials for experiment in experiments])


def fit_causal_process_memory(
    source: CausalProcessTensor,
    initial_model: SequentialStinespringProcess,
    training_experiments: Sequence[ProcessTomographyExperiment],
    held_out_experiments: Sequence[ProcessTomographyExperiment],
    /,
    *,
    iterations: int = 100,
    learning_rate: float = 1e-2,
    probability_tolerance: float = 1e-3,
    identifiability_tolerance: float = 1e-8,
    optimize_isometries: bool = True,
) -> ProcessMemoryRefitResult:
    """Refit a smaller physical memory model against causal interventions."""
    if not isinstance(source, CausalProcessTensor):
        raise TypeError("source must be a CausalProcessTensor.")
    if not isinstance(initial_model, SequentialStinespringProcess):
        raise TypeError("initial_model must be a SequentialStinespringProcess.")
    target_spec = initial_model.spec
    if (
        target_spec.system_dimension != source.spec.system_dimension
        or target_spec.slot_count != source.spec.slot_count
    ):
        raise ValueError("Source and refit models must share system legs and slots.")
    if target_spec.memory_dimension >= source.spec.memory_dimension:
        raise ValueError(
            "The refit model must retain fewer memory dimensions than the source."
        )
    training = tuple(training_experiments)
    held_out = tuple(held_out_experiments)
    if not training or not held_out:
        raise ValueError(
            "Causal memory refit requires training and held-out experiments."
        )
    if not tomography_designs_disjoint(training, held_out):
        raise ValueError(
            "Causal memory refit requires disjoint intervention/effect settings."
        )
    tolerance = float(probability_tolerance)
    if not jnp.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("probability_tolerance must be finite and non-negative.")
    tomography = fit_stinespring_process(
        StinespringTomographyProblem(
            initial_model,
            training,
            problem_id=f"{initial_model.process_id}:memory-refit",
        ),
        iterations=iterations,
        learning_rate=learning_rate,
        held_out_experiments=held_out,
        identifiability_tolerance=identifiability_tolerance,
        optimize_isometries=optimize_isometries,
    )
    fitted = tomography.model.materialize()
    initial_probabilities = _experiment_probabilities(
        initial_model.materialize(), held_out
    )
    return ProcessMemoryRefitResult(
        source,
        tomography,
        _experiment_observed_probabilities(training),
        _experiment_probabilities(fitted, training),
        _experiment_observed_probabilities(held_out),
        initial_probabilities,
        _experiment_probabilities(fitted, held_out),
        probability_tolerance=tolerance,
    )


__all__ = [
    "ProcessMemoryRefitResult",
    "StinespringTomographyProblem",
    "StinespringTomographyResult",
    "fit_causal_process_memory",
    "fit_stinespring_process",
]
