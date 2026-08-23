#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..metrix import faithful_density_from_cholesky
from ..tensor_network import CausalProcessTensor, QuantumInstrument


class ProcessTomographyExperiment(StrictModule):
    instruments: tuple[QuantumInstrument, ...]
    outcomes: tuple[int, ...]
    count: Array
    experiment_id: str

    def __init__(
        self,
        instruments: Sequence[QuantumInstrument],
        outcomes: Sequence[int],
        count: ArrayLike,
        /,
        *,
        experiment_id: str,
    ):
        self.instruments = tuple(instruments)
        self.outcomes = tuple(int(value) for value in outcomes)
        self.count = jnp.asarray(count, dtype=float).reshape(())
        self.experiment_id = str(experiment_id)


class CausalProcessTomographyProblem(StrictModule):
    process: CausalProcessTensor
    experiments: tuple[ProcessTomographyExperiment, ...]
    problem_id: str

    def __init__(
        self,
        process: CausalProcessTensor,
        experiments: Sequence[ProcessTomographyExperiment],
        /,
        *,
        problem_id: str = "causal-process-tomography",
    ):
        values = tuple(experiments)
        if not values:
            raise ValueError("At least one process experiment is required.")
        self.process = process
        self.experiments = values
        self.problem_id = str(problem_id)

    def probabilities(self, process: CausalProcessTensor | None = None, /) -> Array:
        model = self.process if process is None else process
        return jnp.stack(
            [
                model.contract(experiment.instruments, experiment.outcomes).probability
                for experiment in self.experiments
            ]
        )

    def negative_log_likelihood(
        self, process: CausalProcessTensor | None = None, /
    ) -> Array:
        probabilities = self.probabilities(process)
        counts = jnp.stack([experiment.count for experiment in self.experiments])
        support_violation = jnp.any((counts > 0.0) & (probabilities <= 0.0))
        safe = jnp.maximum(probabilities, jnp.finfo(probabilities.dtype).tiny)
        value = -jnp.sum(counts * jnp.log(safe))
        return jnp.where(support_violation, jnp.inf, value)


class CausalProcessTomographyResult(StrictModule):
    process: CausalProcessTensor
    loss_history: Array
    support_valid: Array
    valid: Array
    problem_id: str

    def __init__(
        self,
        process: CausalProcessTensor,
        loss_history: ArrayLike,
        /,
        *,
        problem_id: str,
    ):
        self.process = process
        self.loss_history = jnp.asarray(loss_history)
        self.support_valid = jnp.all(jnp.isfinite(self.loss_history))
        self.valid = process.valid & self.support_valid
        self.problem_id = str(problem_id)


def fit_causal_process_initial_state(
    problem: CausalProcessTomographyProblem,
    initial_factor: ArrayLike,
    /,
    *,
    iterations: int = 100,
    learning_rate: float = 1e-2,
) -> CausalProcessTomographyResult:
    factor = jnp.asarray(initial_factor)
    expected = problem.process.initial_state.shape
    if factor.shape != expected:
        raise ValueError("Initial process-density factor shape is invalid.")

    def model(candidate):
        density = faithful_density_from_cholesky(candidate)
        return CausalProcessTensor(
            problem.process.spec,
            density,
            problem.process.channel_kraus,
            process_id=problem.process.process_id,
        )

    def loss(candidate):
        return problem.negative_log_likelihood(model(candidate))

    value_and_grad = jax.value_and_grad(loss)
    history = []
    for _ in range(int(iterations)):
        value, gradient = value_and_grad(factor)
        factor = factor - float(learning_rate) * gradient
        history.append(value)
    return CausalProcessTomographyResult(
        model(factor),
        jnp.stack(history),
        problem_id=problem.problem_id,
    )


__all__ = [
    "CausalProcessTomographyProblem",
    "CausalProcessTomographyResult",
    "ProcessTomographyExperiment",
    "fit_causal_process_initial_state",
]
