#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..metrix import ComplexStiefelManifold, faithful_density_from_cholesky
from ..tensor_network import CausalProcessTensor, SequentialStinespringProcess
from ._process_tomography import ProcessTomographyExperiment


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
    valid: Array

    def __init__(
        self,
        model: SequentialStinespringProcess,
        loss_history: ArrayLike,
        held_out_loss: ArrayLike,
        identifiability_rank: ArrayLike,
        nullity: ArrayLike,
        /,
    ):
        self.model = model
        self.loss_history = jnp.asarray(loss_history)
        self.held_out_loss = jnp.asarray(held_out_loss)
        self.identifiability_rank = jnp.asarray(identifiability_rank)
        self.nullity = jnp.asarray(nullity)
        self.valid = (
            model.materialize().valid
            & model.gauge_report().valid
            & jnp.all(jnp.isfinite(self.loss_history))
            & jnp.isfinite(self.held_out_loss)
        )


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
    values = []
    for experiment in experiments:
        probability = process.contract(
            experiment.instruments, experiment.outcomes
        ).probability
        values.append(
            jnp.where(
                (experiment.count > 0.0) & (probability <= 0.0),
                jnp.inf,
                -experiment.count * jnp.log(jnp.maximum(probability, 1e-30)),
            )
        )
    return jnp.sum(jnp.stack(values))


def fit_stinespring_process(
    problem: StinespringTomographyProblem,
    /,
    *,
    iterations: int = 100,
    learning_rate: float = 1e-2,
    held_out_experiments: Sequence[ProcessTomographyExperiment] = (),
) -> StinespringTomographyResult:
    factor = problem.model.initial_factor
    isometries = problem.model.isometries

    def loss(factor_value, isometry_values):
        return _nll(
            _materialize(problem.model, factor_value, isometry_values),
            problem.experiments,
        )

    value_and_grad = jax.value_and_grad(loss, argnums=(0, 1))
    history = []
    for _ in range(int(iterations)):
        value, (factor_gradient, isometry_gradients) = value_and_grad(factor, isometries)
        factor = factor - float(learning_rate) * factor_gradient
        updated = []
        for isometry, gradient in zip(isometries, isometry_gradients, strict=True):
            manifold = ComplexStiefelManifold(isometry.shape[0], isometry.shape[1])
            tangent = manifold.egrad_to_rgrad(isometry, gradient)
            updated.append(manifold.retract(isometry, -float(learning_rate) * tangent))
        isometries = tuple(updated)
        history.append(value)
    model = SequentialStinespringProcess(
        problem.model.spec,
        factor,
        isometries,
        problem.model.environment_dimensions,
        process_id=problem.model.process_id,
    )
    held_out = (
        _nll(model.materialize(), tuple(held_out_experiments))
        if held_out_experiments
        else jnp.asarray(0.0)
    )
    realified = jnp.concatenate(
        [
            jnp.concatenate((jnp.real(value).reshape(-1), jnp.imag(value).reshape(-1)))
            for value in isometries
        ]
    )

    sizes = tuple(value.size for value in isometries)
    shapes = tuple(value.shape for value in isometries)

    def probabilities(parameters):
        complex_values = []
        real_cursor = 0
        imaginary_cursor = sum(sizes)
        for size, shape in zip(sizes, shapes, strict=True):
            real = parameters[real_cursor : real_cursor + size]
            imaginary = parameters[imaginary_cursor : imaginary_cursor + size]
            complex_values.append((real + 1j * imaginary).reshape(shape))
            real_cursor += size
            imaginary_cursor += size
        process = _materialize(problem.model, factor, tuple(complex_values))
        return jnp.stack(
            [
                process.contract(experiment.instruments, experiment.outcomes).probability
                for experiment in problem.experiments
            ]
        )

    jacobian = jax.jacfwd(probabilities)(realified)
    singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
    rank = jnp.sum(singular_values > 1e-8)
    nullity = realified.size - rank
    return StinespringTomographyResult(
        model,
        jnp.stack(history),
        held_out,
        rank,
        nullity,
    )


__all__ = [
    "StinespringTomographyProblem",
    "StinespringTomographyResult",
    "fit_stinespring_process",
]
