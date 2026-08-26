#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from itertools import product
from math import isfinite

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    FailurePolicy,
    HermitianSpectrum,
    svd as svd_linalg,
)
from ..metrix import faithful_density_from_cholesky
from ..tensor_network import CausalProcessTensor, QuantumInstrument


def _process_probability(
    process: CausalProcessTensor,
    instruments: tuple[QuantumInstrument, ...],
    outcomes: tuple[int, ...],
    terminal_effect: Array,
    /,
) -> Array:
    branch = process.contract(instruments, outcomes)
    terminal = jnp.real(jnp.trace(terminal_effect @ branch.final_system_state))
    return branch.probability * terminal


def _informationally_complete_vectors(
    dimension: int,
    design_seed: int,
    /,
) -> tuple[Array, ...]:
    basis = jnp.eye(dimension, dtype=complex)
    vectors = [basis[index] for index in range(dimension)]
    for left in range(dimension):
        for right in range(left + 1, dimension):
            vectors.append((basis[left] + basis[right]) / jnp.sqrt(2.0))
            vectors.append((basis[left] + 1j * basis[right]) / jnp.sqrt(2.0))
    if design_seed == 0:
        return tuple(vectors)
    if dimension == 1:
        raise ValueError(
            "A one-dimensional process has no distinct seeded design settings."
        )
    key = jax.random.PRNGKey(design_seed)
    real_key, imaginary_key = jax.random.split(key)
    raw = jax.random.normal(real_key, (dimension, dimension)) + 1j * jax.random.normal(
        imaginary_key, (dimension, dimension)
    )
    decomposition = svd_linalg.svd(
        svd_linalg.SVDProblem(
            DenseLinearOperator(raw),
            problem_id=f"process-design-unitary-{design_seed}",
        ),
        policy=svd_linalg.SVDSolvePolicy(
            count=dimension,
            failure=FailurePolicy("error"),
        ),
    )
    unitary = decomposition.left_vectors @ jnp.conj(decomposition.right_vectors.T)
    return tuple(unitary @ vector for vector in vectors)


def _rank_one_instrument(
    prepared: Array,
    probe: Array,
    /,
    *,
    instrument_id: str,
) -> QuantumInstrument:
    operator = jnp.outer(prepared, jnp.conj(probe))
    remainder = jnp.eye(probe.size, dtype=operator.dtype) - jnp.outer(
        probe, jnp.conj(probe)
    )
    spectrum = HermitianSpectrum(0.5 * remainder + 0.5 * jnp.conj(remainder.T))
    complement = (
        spectrum.eigenvectors * jnp.sqrt(jnp.maximum(spectrum.eigenvalues, 0.0))
    ) @ jnp.conj(spectrum.eigenvectors.T)
    return QuantumInstrument(
        jnp.stack((operator, complement))[:, None, ...],
        jnp.asarray([True, True]),
        jnp.asarray([[True], [True]]),
        instrument_id=instrument_id,
    )


def _canonical_complex_array(value: ArrayLike, /) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    rounded = np.round(array.real, decimals=12) + 1j * np.round(array.imag, decimals=12)
    return np.ascontiguousarray(rounded)


def _selected_choi(instrument: QuantumInstrument, outcome: int, /) -> np.ndarray:
    operators = np.asarray(instrument.kraus[outcome], dtype=np.complex128)
    active = np.asarray(
        instrument.outcome_active[outcome] & instrument.kraus_active[outcome],
        dtype=bool,
    )
    masked = np.where(active[:, None, None], operators, 0.0)
    vectors = masked.reshape((masked.shape[0], -1))
    choi = vectors.T @ np.conj(vectors)
    return 0.5 * (choi + np.conj(choi.T))


def _setting_fingerprint(
    instruments: tuple[QuantumInstrument, ...],
    outcomes: tuple[int, ...],
    terminal_effect: Array,
    /,
) -> str:
    digest = sha256()
    for instrument, outcome in zip(instruments, outcomes, strict=True):
        choi = _canonical_complex_array(_selected_choi(instrument, outcome))
        digest.update(np.asarray(choi.shape, dtype=np.int64).tobytes())
        digest.update(choi.tobytes(order="C"))
    effect = _canonical_complex_array(terminal_effect)
    digest.update(np.asarray(effect.shape, dtype=np.int64).tobytes())
    digest.update(effect.tobytes(order="C"))
    return digest.hexdigest()


class ProcessTomographyExperiment(StrictModule):
    instruments: tuple[QuantumInstrument, ...]
    outcomes: tuple[int, ...]
    terminal_effect: Array
    count: Array
    trials: Array
    valid: Array
    experiment_id: str
    setting_fingerprint: str

    def __init__(
        self,
        instruments: Sequence[QuantumInstrument],
        outcomes: Sequence[int],
        count: ArrayLike,
        /,
        *,
        terminal_effect: ArrayLike | None = None,
        trials: ArrayLike | None = None,
        experiment_id: str,
    ):
        operations = tuple(instruments)
        selected = tuple(int(value) for value in outcomes)
        if not operations or len(operations) != len(selected):
            raise ValueError("Tomography experiments require one outcome per instrument.")
        if any(not isinstance(value, QuantumInstrument) for value in operations):
            raise TypeError("instruments must contain QuantumInstrument values.")
        dimension = operations[0].dimension
        if any(value.dimension != dimension for value in operations):
            raise ValueError("Every tomography instrument must share dimension.")
        if any(
            not 0 <= outcome < instrument.kraus.shape[0]
            for instrument, outcome in zip(operations, selected, strict=True)
        ):
            raise ValueError("A tomography outcome is outside its instrument.")
        if any(
            not bool(instrument.outcome_active[outcome])
            for instrument, outcome in zip(operations, selected, strict=True)
        ):
            raise ValueError("A tomography experiment selected an inactive outcome.")
        effect = (
            jnp.eye(dimension, dtype=operations[0].kraus.dtype)
            if terminal_effect is None
            else jnp.asarray(terminal_effect)
        )
        if effect.shape != (dimension, dimension):
            raise ValueError("terminal_effect has incompatible system shape.")
        hermitian = 0.5 * (effect + jnp.conj(effect.T))
        hermitian_residual = jnp.max(jnp.abs(effect - hermitian))
        eigenvalues = jnp.linalg.eigvalsh(hermitian)
        effect_valid = (
            jnp.all(jnp.isfinite(effect))
            & (hermitian_residual <= 1e-8)
            & (jnp.min(eigenvalues) >= -1e-8)
            & (jnp.max(eigenvalues) <= 1.0 + 1e-8)
        )
        successes = jnp.asarray(count, dtype=float).reshape(())
        attempts = (
            successes if trials is None else jnp.asarray(trials, dtype=float).reshape(())
        )
        counts_valid = (
            jnp.isfinite(successes)
            & jnp.isfinite(attempts)
            & (successes >= 0.0)
            & (attempts > 0.0)
            & (attempts >= successes)
        )
        identifier = str(experiment_id)
        if not identifier:
            raise ValueError("experiment_id must be non-empty.")
        self.instruments = operations
        self.outcomes = selected
        self.terminal_effect = effect
        self.count = successes
        self.trials = attempts
        self.valid = (
            effect_valid
            & counts_valid
            & jnp.all(jnp.stack([value.valid for value in operations]))
        )
        self.experiment_id = identifier
        self.setting_fingerprint = _setting_fingerprint(operations, selected, effect)

    def probability(self, process: CausalProcessTensor, /) -> Array:
        if not isinstance(process, CausalProcessTensor):
            raise TypeError("process must be a CausalProcessTensor.")
        if (
            len(self.instruments) != process.spec.slot_count
            or self.instruments[0].dimension != process.spec.system_dimension
        ):
            raise ValueError("Experiment and process dimensions are incompatible.")
        value = _process_probability(
            process, self.instruments, self.outcomes, self.terminal_effect
        )
        near_probability = (value >= -1e-8) & (value <= 1.0 + 1e-8)
        return jnp.where(near_probability, jnp.clip(value, 0.0, 1.0), value)

    def negative_log_likelihood(self, process: CausalProcessTensor, /) -> Array:
        probability = self.probability(process)
        failures = self.trials - self.count
        support_violation = (
            ~self.valid
            | ~jnp.isfinite(probability)
            | (probability < 0.0)
            | (probability > 1.0)
            | ((self.count > 0.0) & (probability <= 0.0))
            | ((failures > 0.0) & (probability >= 1.0))
        )
        tiny = jnp.finfo(probability.dtype).tiny
        safe = jnp.clip(probability, tiny, 1.0 - jnp.finfo(probability.dtype).eps)
        value = -self.count * jnp.log(safe) - failures * jnp.log1p(-safe)
        return jnp.where(support_violation, jnp.inf, value)

    def same_setting(self, other: ProcessTomographyExperiment, /) -> bool:
        """Return whether canonical selected maps and effects are identical."""
        return (
            isinstance(other, ProcessTomographyExperiment)
            and self.setting_fingerprint == other.setting_fingerprint
        )


def tomography_designs_disjoint(
    first: Sequence[ProcessTomographyExperiment],
    second: Sequence[ProcessTomographyExperiment],
    /,
) -> bool:
    """Check seeded tomography designs in O(n + m) fingerprint work."""
    first_fingerprints = {experiment.setting_fingerprint for experiment in first}
    second_fingerprints = {experiment.setting_fingerprint for experiment in second}
    return first_fingerprints.isdisjoint(second_fingerprints)


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
            [experiment.probability(model) for experiment in self.experiments]
        )

    def negative_log_likelihood(
        self, process: CausalProcessTensor | None = None, /
    ) -> Array:
        model = self.process if process is None else process
        return jnp.sum(
            jnp.stack(
                [
                    experiment.negative_log_likelihood(model)
                    for experiment in self.experiments
                ]
            )
        )

    def initial_state_identifiability(
        self,
        factor: ArrayLike,
        /,
        *,
        tolerance: float = 1e-8,
    ) -> tuple[Array, Array]:
        value = jnp.asarray(factor)
        size = value.size
        realified = jnp.concatenate(
            (jnp.real(value).reshape(-1), jnp.imag(value).reshape(-1))
        )

        def probabilities(parameters):
            candidate = parameters[:size].reshape(value.shape) + 1j * parameters[
                size:
            ].reshape(value.shape)
            density = faithful_density_from_cholesky(candidate)
            model = CausalProcessTensor(
                self.process.spec,
                density,
                self.process.channel_kraus,
                process_id=self.process.process_id,
            )
            return self.probabilities(model)

        jacobian = jax.jacfwd(probabilities)(realified)
        singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
        scale = jnp.max(singular_values, initial=0.0)
        rank = jnp.sum(singular_values > tolerance * jnp.maximum(scale, 1e-30))
        physical_parameter_count = value.shape[0] ** 2 - 1
        return rank, jnp.maximum(physical_parameter_count - rank, 0)


class CausalProcessTomographyResult(StrictModule):
    process: CausalProcessTensor
    loss_history: Array
    support_valid: Array
    identifiability_rank: Array
    nullity: Array
    underidentified: Array
    valid: Array
    problem_id: str

    def __init__(
        self,
        process: CausalProcessTensor,
        loss_history: ArrayLike,
        identifiability_rank: ArrayLike,
        nullity: ArrayLike,
        /,
        *,
        problem_id: str,
    ):
        self.process = process
        self.loss_history = jnp.asarray(loss_history)
        self.support_valid = jnp.all(jnp.isfinite(self.loss_history))
        self.identifiability_rank = jnp.asarray(identifiability_rank)
        self.nullity = jnp.asarray(nullity)
        self.underidentified = self.nullity > 0
        self.valid = process.valid & self.support_valid & ~self.underidentified
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
    count = int(iterations)
    rate = float(learning_rate)
    if count < 1 or not jnp.isfinite(rate) or rate <= 0.0:
        raise ValueError("iterations and learning_rate must be positive and finite.")
    for _ in range(count):
        value, gradient = value_and_grad(factor)
        direction = jnp.conj(gradient) if jnp.iscomplexobj(gradient) else gradient
        factor = factor - rate * direction
        history.append(value)
    rank, nullity = problem.initial_state_identifiability(factor)
    return CausalProcessTomographyResult(
        model(factor),
        jnp.stack(history),
        rank,
        nullity,
        problem_id=problem.problem_id,
    )


def informationally_complete_process_experiments(
    process: CausalProcessTensor,
    /,
    *,
    shots: float,
    design_seed: int = 0,
    experiment_id: str = "informationally-complete-process",
    maximum_experiments: int = 100_000,
) -> tuple[ProcessTomographyExperiment, ...]:
    """Generate a bounded seeded spanning intervention/effect design."""
    if not isinstance(design_seed, int) or design_seed < 0:
        raise ValueError("design_seed must be a non-negative integer.")
    if not isinstance(process, CausalProcessTensor):
        raise TypeError("process must be a CausalProcessTensor.")
    shot_count = float(shots)
    if not isfinite(shot_count) or shot_count <= 0.0:
        raise ValueError("shots must be finite and positive.")
    capacity = int(maximum_experiments)
    if capacity <= 0:
        raise ValueError("maximum_experiments must be positive.")
    identifier = str(experiment_id)
    if not identifier:
        raise ValueError("experiment_id must be non-empty.")
    vectors = _informationally_complete_vectors(
        process.spec.system_dimension, design_seed
    )
    instrument_values = []
    for prepared_index, prepared in enumerate(vectors):
        for probe_index, probe in enumerate(vectors):
            instrument_values.append(
                _rank_one_instrument(
                    prepared,
                    probe,
                    instrument_id=(
                        f"{identifier}:prepare-{prepared_index}:probe-{probe_index}"
                    ),
                )
            )
    instruments = tuple(instrument_values)
    effects = tuple(jnp.outer(value, jnp.conj(value)) for value in vectors)
    count = len(instruments) ** process.spec.slot_count * len(effects)
    if count > capacity:
        raise ValueError(
            f"Informationally complete design requires {count} experiments; "
            f"capacity is {capacity}."
        )
    experiments = []
    for instrument_indices in product(
        range(len(instruments)), repeat=process.spec.slot_count
    ):
        operations = tuple(instruments[index] for index in instrument_indices)
        outcomes = (0,) * process.spec.slot_count
        setting = "-".join(str(index) for index in instrument_indices)
        for effect_index, effect in enumerate(effects):
            probability = jnp.clip(
                _process_probability(process, operations, outcomes, effect),
                0.0,
                1.0,
            )
            experiments.append(
                ProcessTomographyExperiment(
                    operations,
                    outcomes,
                    shot_count * probability,
                    terminal_effect=effect,
                    trials=shot_count,
                    experiment_id=f"{identifier}:maps-{setting}:effect-{effect_index}",
                )
            )
    return tuple(experiments)


__all__ = [
    "CausalProcessTomographyProblem",
    "CausalProcessTomographyResult",
    "ProcessTomographyExperiment",
    "fit_causal_process_initial_state",
    "informationally_complete_process_experiments",
    "tomography_designs_disjoint",
]
