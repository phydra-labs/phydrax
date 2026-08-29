#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key
from opt_einsum import contract

from .._doc import DOC_KEY0
from .._fingerprint import array_tree_signature, canonical_fingerprint
from .._sampling import (
    MarkovSampleResult,
    MarkovState,
    MetropolisHastings,
    sample_markov,
)
from .._strict import StrictModule
from ..linalg import (
    ArraySpace,
    EmpiricalGramLinearOperator,
    JacobianLinearOperator,
    LinearSolveResult,
    LinearSystem,
    prepare_linearization,
    solve,
)
from ..linalg.eigen import (
    block_rayleigh_trace,
    BlockRayleighEvaluation,
    ReducedRitzResult,
    solve_reduced_ritz,
)
from ..nn.parameters import ParameterSubspace
from ..operators.quantum import (
    AbstractDiscreteQuantumOperator,
    ComplexParameterMode,
    LogAmplitude,
)
from ._variational_monte_carlo import VariationalMonteCarloPolicy


if TYPE_CHECKING:
    from ..uq._diagnostics import MCMCDiagnostics


VMCSubspaceStatus: TypeAlias = Literal[0, 1, 2, 3, 4, 5]
VMC_SUBSPACE_SUCCESS: VMCSubspaceStatus = 0
VMC_SUBSPACE_INVALID_SAMPLES: VMCSubspaceStatus = 1
VMC_SUBSPACE_NONFINITE: VMCSubspaceStatus = 2
VMC_SUBSPACE_SINGULAR_SPAN: VMCSubspaceStatus = 3
VMC_SUBSPACE_RITZ_FAILURE: VMCSubspaceStatus = 4
VMC_SUBSPACE_LINEAR_FAILURE: VMCSubspaceStatus = 5


def vmc_subspace_status_name(status: int | Array, /) -> str:
    """Return the stable public name of a subspace-VMC status code."""
    code = int(status)
    names = (
        "success",
        "invalid_samples",
        "nonfinite",
        "singular_span",
        "ritz_failure",
        "linear_failure",
    )
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown subspace VMC status {code}.")
    return names[code]


def _scalar_amplitude(model: Any, configuration: Array, /) -> LogAmplitude:
    value = model(configuration)
    if not isinstance(value, LogAmplitude):
        raise TypeError("Every subspace VMC model must return LogAmplitude.")
    if value.log_abs.shape != ():
        raise ValueError(
            "Every subspace VMC model must return one scalar amplitude per "
            "configuration."
        )
    return value


def _batched_amplitude(model: Any, configurations: Array, /) -> LogAmplitude:
    value = jax.vmap(model)(configurations)
    if not isinstance(value, LogAmplitude):
        raise TypeError("Every subspace VMC model must return LogAmplitude.")
    expected = (int(configurations.shape[0]),)
    if value.log_abs.shape != expected:
        raise ValueError(
            "Every subspace VMC model must return one scalar amplitude per "
            f"configuration; expected {expected}, got {value.log_abs.shape}."
        )
    return value


def _parameter_mode(value: ComplexParameterMode, /) -> ComplexParameterMode:
    if value not in ("real", "holomorphic", "nonholomorphic"):
        raise ValueError(
            "complex parameter modes must be 'real', 'holomorphic', or "
            "'nonholomorphic'."
        )
    return value


def _coordinates_from_vector(vector: Array, mode: ComplexParameterMode, /) -> Array:
    values = jnp.asarray(vector)
    if mode == "real":
        if jnp.iscomplexobj(values):
            raise TypeError("real parameter mode requires real trainable parameters.")
        return values
    if mode == "holomorphic":
        if not jnp.iscomplexobj(values):
            raise TypeError("holomorphic parameter mode requires complex parameters.")
        return values
    if not jnp.iscomplexobj(values):
        raise TypeError("nonholomorphic parameter mode requires complex parameters.")
    return jnp.concatenate((jnp.real(values), jnp.imag(values)))


def _vector_from_coordinates(
    coordinates: Array,
    exemplar: Array,
    mode: ComplexParameterMode,
    /,
) -> Array:
    values = jnp.asarray(coordinates)
    if mode != "nonholomorphic":
        return values.astype(exemplar.dtype)
    size = int(exemplar.shape[0])
    if values.shape != (2 * size,):
        raise ValueError(
            f"Nonholomorphic coordinates must have shape ({2 * size},); "
            f"got {values.shape}."
        )
    return (values[:size] + 1j * values[size:]).astype(exemplar.dtype)


def _surrogate(amplitude: LogAmplitude, /) -> Array:
    safe_phase = jax.lax.stop_gradient(amplitude.phase)
    value = amplitude.log_abs + amplitude.phase / safe_phase
    return jnp.where(amplitude.nonzero, value, jnp.zeros((), dtype=value.dtype))


def _resolve_modes(
    modes: ComplexParameterMode | Sequence[ComplexParameterMode],
    count: int,
    /,
) -> tuple[ComplexParameterMode, ...]:
    if isinstance(modes, str):
        return tuple(_parameter_mode(modes) for _ in range(count))
    resolved = tuple(_parameter_mode(mode) for mode in modes)
    if len(resolved) != count:
        raise ValueError("complex_parameter_modes must have one entry per model.")
    return resolved


def _default_parameter_subspace(model: Any, /) -> ParameterSubspace | None:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    if not leaves:
        return None
    return ParameterSubspace(model, eqx.is_inexact_array)


def _mixture_components(
    amplitudes: tuple[LogAmplitude, ...],
    /,
) -> tuple[Array, Array, Array]:
    log_abs = jnp.stack(tuple(value.log_abs for value in amplitudes), axis=-1)
    phase = jnp.stack(tuple(value.phase for value in amplitudes), axis=-1)
    valid = jnp.stack(tuple(value.valid for value in amplitudes), axis=-1)
    nonzero = valid & jnp.isfinite(log_abs)
    any_nonzero = jnp.any(nonzero, axis=-1)
    safe_log_abs = jnp.where(nonzero, log_abs, -jnp.inf)
    maximum = jnp.max(safe_log_abs, axis=-1)
    safe_maximum = jnp.where(any_nonzero, maximum, 0.0)
    scaled_magnitude = jnp.where(
        nonzero,
        jnp.exp(safe_log_abs - safe_maximum[..., None]),
        0.0,
    )
    squared_norm = jnp.sum(scaled_magnitude**2, axis=-1)
    safe_norm = jnp.sqrt(jnp.where(any_nonzero, squared_norm, 1.0))
    relative = scaled_magnitude * phase / safe_norm[..., None]
    log_norm = jnp.where(
        any_nonzero,
        safe_maximum + 0.5 * jnp.log(squared_norm),
        -jnp.inf,
    )
    all_valid = jnp.all(valid, axis=-1)
    return relative, log_norm, all_valid & any_nonzero


def _mixture_log_target(models: tuple[Any, ...], /):
    def log_target(configuration):
        amplitudes = tuple(_scalar_amplitude(model, configuration) for model in models)
        _relative, log_norm, valid = _mixture_components(amplitudes)
        return jnp.where(valid, 2.0 * log_norm, -jnp.inf)

    return log_target


def _batched_mixture_log_weight(
    models: tuple[Any, ...], configurations: Array, /
) -> Array:
    amplitudes = tuple(_batched_amplitude(model, configurations) for model in models)
    _relative, log_norm, valid = _mixture_components(amplitudes)
    return jnp.where(valid, 2.0 * log_norm, -jnp.inf)


class VariationalMonteCarloSubspaceProblem(StrictModule):
    """Discrete model block, connected operator, and one mixture sampler."""

    models: tuple[Any, ...]
    operator: AbstractDiscreteQuantumOperator
    kernel: MetropolisHastings
    initial_configurations: Array
    parameter_subspaces: tuple[ParameterSubspace | None, ...]
    initial_parameter_vectors: tuple[Array, ...]
    initial_coordinates: tuple[Array, ...]
    complex_parameter_modes: tuple[ComplexParameterMode, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        models: Sequence[Any],
        operator: AbstractDiscreteQuantumOperator,
        kernel: MetropolisHastings,
        initial_configurations: Array,
        /,
        *,
        complex_parameter_modes: ComplexParameterMode
        | Sequence[ComplexParameterMode] = "real",
        parameter_subspaces: Sequence[ParameterSubspace | None] | None = None,
        problem_id: str | None = None,
    ):
        models_ = tuple(models)
        if len(models_) < 2:
            raise ValueError("Subspace VMC requires at least two amplitude models.")
        if any(not callable(model) for model in models_):
            raise TypeError("Every subspace VMC model must be callable.")
        if not isinstance(operator, AbstractDiscreteQuantumOperator):
            raise TypeError("operator must implement AbstractDiscreteQuantumOperator.")
        if not isinstance(kernel, MetropolisHastings):
            raise TypeError("kernel must be a MetropolisHastings instance.")
        configurations = jnp.asarray(initial_configurations)
        expected_rank = 1 + len(operator.configuration_shape)
        if (
            configurations.ndim != expected_rank
            or tuple(configurations.shape[1:]) != operator.configuration_shape
        ):
            raise ValueError(
                "initial_configurations must have shape (chain,) + "
                f"{operator.configuration_shape}; got {configurations.shape}."
            )
        if int(configurations.shape[0]) < 1:
            raise ValueError("At least one initial Markov chain is required.")
        exemplars = tuple(
            _scalar_amplitude(model, configurations[0]) for model in models_
        )
        if not all(bool(jnp.asarray(value.valid)) for value in exemplars):
            raise ValueError("Every model must be valid at the exemplar configuration.")
        if not any(bool(jnp.asarray(value.nonzero)) for value in exemplars):
            raise ValueError(
                "At least one model must be nonzero at the exemplar configuration."
            )
        modes = _resolve_modes(complex_parameter_modes, len(models_))
        if parameter_subspaces is None:
            subspaces = tuple(_default_parameter_subspace(model) for model in models_)
        else:
            subspaces = tuple(parameter_subspaces)
            if len(subspaces) != len(models_):
                raise ValueError("parameter_subspaces must have one entry per model.")
            if any(
                subspace is not None
                and not isinstance(subspace, ParameterSubspace)
                for subspace in subspaces
            ):
                raise TypeError(
                    "parameter_subspaces entries must be ParameterSubspace or None."
                )
        vectors = tuple(
            jnp.empty((0,), dtype=float) if subspace is None else subspace.pack()
            for subspace in subspaces
        )
        coordinates = tuple(
            vector
            if subspace is None
            else _coordinates_from_vector(vector, mode)
            for vector, mode, subspace in zip(
                vectors, modes, subspaces, strict=True
            )
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "variational-monte-carlo-subspace",
                    "operator": operator.operator_id,
                    "kernel": kernel.kernel_id,
                    "proposal": kernel.proposal.proposal_id,
                    "model_count": len(models_),
                    "parameter_paths": [
                        None if subspace is None else list(subspace.leaf_paths)
                        for subspace in subspaces
                    ],
                    "complex_parameter_modes": list(modes),
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.models = models_
        self.operator = operator
        self.kernel = kernel
        self.initial_configurations = configurations
        self.parameter_subspaces = subspaces
        self.initial_parameter_vectors = vectors
        self.initial_coordinates = coordinates
        self.complex_parameter_modes = modes
        self.problem_id = identifier

    @property
    def state_count(self) -> int:
        return len(self.models)

    @property
    def trainable_dimension(self) -> int:
        return sum(int(coordinates.size) for coordinates in self.initial_coordinates)

    def model_from_coordinates(self, index: int, coordinates: Array, /) -> Any:
        if index < 0 or index >= self.state_count:
            raise IndexError("model index is outside the subspace block.")
        subspace = self.parameter_subspaces[index]
        values = jnp.asarray(coordinates)
        if subspace is None:
            if values.shape != (0,):
                raise ValueError("A frozen model requires empty parameter coordinates.")
            return self.models[index]
        vector = _vector_from_coordinates(
            values,
            self.initial_parameter_vectors[index],
            self.complex_parameter_modes[index],
        )
        return subspace.reconstruct_vector(vector)

    def models_from_coordinates(
        self, coordinates: Sequence[Array], /
    ) -> tuple[Any, ...]:
        coordinates_ = tuple(coordinates)
        if len(coordinates_) != self.state_count:
            raise ValueError("coordinates must have one entry per model.")
        return tuple(
            self.model_from_coordinates(index, value)
            for index, value in enumerate(coordinates_)
        )

    def initial_state(
        self, *, key: Key[Array, ""] = DOC_KEY0
    ) -> VariationalMonteCarloSubspaceState:
        markov = self.kernel.initialize(
            _mixture_log_target(self.models),
            self.initial_configurations,
        )
        return VariationalMonteCarloSubspaceState(
            models=self.models,
            parameter_coordinates=self.initial_coordinates,
            markov_state=markov,
            iteration=0,
            root_key=key,
        )


class VariationalMonteCarloSubspaceState(StrictModule):
    """Joint model coordinates and one persistent mixture Markov ensemble."""

    models: tuple[Any, ...]
    parameter_coordinates: tuple[Array, ...]
    markov_state: MarkovState
    iteration: Array
    root_key: Array

    def __init__(
        self,
        *,
        models: Sequence[Any],
        parameter_coordinates: Sequence[Array],
        markov_state: MarkovState,
        iteration: int | Array,
        root_key: Key[Array, ""],
    ):
        models_ = tuple(models)
        coordinates = tuple(jnp.asarray(value) for value in parameter_coordinates)
        if len(models_) < 2 or len(coordinates) != len(models_):
            raise ValueError(
                "models and parameter_coordinates must describe at least two states."
            )
        if any(not callable(model) for model in models_):
            raise TypeError("Every subspace VMC state model must be callable.")
        if not isinstance(markov_state, MarkovState):
            raise TypeError("markov_state must be a MarkovState.")
        iteration_ = jnp.asarray(iteration, dtype=jnp.int32)
        if iteration_.shape != ():
            raise ValueError("iteration must be scalar.")
        if jnp.asarray(root_key).shape != ():
            raise ValueError("root_key must be one scalar PRNG key.")
        self.models = models_
        self.parameter_coordinates = coordinates
        self.markov_state = markov_state
        self.iteration = iteration_
        self.root_key = root_key


class VariationalMonteCarloSubspaceEstimate(StrictModule):
    """Block Ritz estimate with sampler, span, and Hermitian evidence."""

    objective: Array
    state_energies: Array
    state_modes: Array
    state_variances: Array
    raw_overlap_matrix: Array
    raw_hamiltonian_matrix: Array
    overlap_matrix: Array
    hamiltonian_matrix: Array
    overlap_hermiticity_residual: Array
    hamiltonian_hermiticity_residual: Array
    gram_minimum_eigenvalue: Array
    gram_condition_number: Array
    gram_numerical_rank: Array
    relative_amplitudes: Array
    local_hamiltonian_actions: Array
    acceptance_rate: Array
    active_samples: Array
    valid: Array
    status: Array
    rayleigh: BlockRayleighEvaluation | None
    ritz: ReducedRitzResult | None
    chain_diagnostics: MCMCDiagnostics | None

    @property
    def successful(self) -> Array:
        return self.status == VMC_SUBSPACE_SUCCESS


class VariationalMonteCarloSubspaceResult(StrictModule):
    """Final joint state plus complete block-training evidence."""

    final_state: VariationalMonteCarloSubspaceState
    final_estimate: VariationalMonteCarloSubspaceEstimate
    objective_history: Array
    state_energy_history: Array
    state_variance_history: Array
    overlap_hermiticity_history: Array
    hamiltonian_hermiticity_history: Array
    acceptance_history: Array
    update_norm_history: Array
    status_history: Array
    linear_results: tuple[tuple[LinearSolveResult | None, ...], ...]
    root_key: Array
    problem_id: str = eqx.field(static=True)
    completed_iterations: int = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.final_estimate.successful & jnp.all(
            self.status_history == VMC_SUBSPACE_SUCCESS
        )


def _relative_amplitudes_and_actions(
    models: tuple[Any, ...],
    operator: AbstractDiscreteQuantumOperator,
    configurations: Array,
    /,
) -> tuple[Array, Array, Array]:
    """Return ``psi_j / sqrt(W)`` and ``(H psi_j) / sqrt(W)`` at each sample.

    Computing the local action numerator directly avoids the undefined
    ``0 * (H psi_j / psi_j)`` form at nodes of an individual state.
    """
    count = int(configurations.shape[0])
    amplitudes = tuple(_batched_amplitude(model, configurations) for model in models)
    relative, log_norm, current_valid = _mixture_components(amplitudes)
    diagonal = jnp.asarray(operator.diagonal(configurations)).reshape((count,))
    connected = operator.connections(configurations)
    connection_count = connected.max_connections
    connected_configurations = connected.configurations.reshape(
        (count * connection_count,) + operator.configuration_shape
    )
    connected_amplitudes = tuple(
        _batched_amplitude(model, connected_configurations) for model in models
    )
    connected_log_abs = jnp.stack(
        tuple(
            value.log_abs.reshape((count, connection_count))
            for value in connected_amplitudes
        ),
        axis=-1,
    )
    connected_phase = jnp.stack(
        tuple(
            value.phase.reshape((count, connection_count))
            for value in connected_amplitudes
        ),
        axis=-1,
    )
    connected_valid = jnp.stack(
        tuple(
            value.valid.reshape((count, connection_count))
            for value in connected_amplitudes
        ),
        axis=-1,
    )
    matrix_elements = connected.matrix_elements.reshape((count, connection_count))
    connection_mask = connected.valid.reshape((count, connection_count))
    finite_elements = jnp.isfinite(matrix_elements)
    active_amplitude = (
        connected_valid
        & jnp.isfinite(connected_log_abs)
        & connection_mask[..., None]
        & finite_elements[..., None]
    )
    safe_connected_log_abs = jnp.where(
        active_amplitude, connected_log_abs, -jnp.inf
    )
    safe_current_log_norm = jnp.where(current_valid, log_norm, 0.0)
    relative_connected = jnp.where(
        active_amplitude,
        jnp.exp(safe_connected_log_abs - safe_current_log_norm[:, None, None])
        * connected_phase,
        0.0,
    )
    safe_elements = jnp.where(connection_mask & finite_elements, matrix_elements, 0.0)
    connected_action = contract("nc,ncm->nm", safe_elements, relative_connected)
    action = diagonal[:, None] * relative + connected_action
    valid_connected_models = jnp.all(connected_valid, axis=-1)
    invalid_active_connection = jnp.any(
        connection_mask & (~finite_elements | ~valid_connected_models), axis=-1
    )
    sample_valid = (
        current_valid & jnp.isfinite(diagonal) & ~invalid_active_connection
    )
    return relative, action, sample_valid


def _weighted_matrices(
    relative: Array, actions: Array, normalized_weights: Array, /
) -> tuple[Array, Array, Array, Array]:
    """Estimate ``S_ij/Z`` and ``K_ij/Z`` with the same mixture weights."""
    raw_overlap = contract(
        "n,ni,nj->ij", normalized_weights, jnp.conj(relative), relative
    )
    raw_hamiltonian = contract(
        "n,ni,nj->ij", normalized_weights, jnp.conj(relative), actions
    )
    overlap = 0.5 * (raw_overlap + jnp.conj(raw_overlap.T))
    hamiltonian = 0.5 * (raw_hamiltonian + jnp.conj(raw_hamiltonian.T))
    return raw_overlap, raw_hamiltonian, overlap, hamiltonian


def _hermiticity_residual(value: Array, /) -> Array:
    scale = jnp.maximum(jnp.max(jnp.abs(value)), 1.0)
    return jnp.max(jnp.abs(value - jnp.conj(value.T))) / scale


def _estimate_from_samples(
    problem: VariationalMonteCarloSubspaceProblem,
    models: tuple[Any, ...],
    samples: MarkovSampleResult,
    /,
    *,
    ritz_tolerance: float,
    compute_chain_diagnostics: bool,
) -> VariationalMonteCarloSubspaceEstimate:
    configurations = jnp.asarray(samples.samples)
    flat = configurations.reshape((-1,) + problem.operator.configuration_shape)
    relative, actions, sample_valid = _relative_amplitudes_and_actions(
        models, problem.operator, flat
    )
    active = jnp.sum(sample_valid, dtype=jnp.int32)
    normalized_weights = sample_valid.astype(float) / jnp.maximum(active, 1)
    raw_overlap, raw_hamiltonian, overlap, hamiltonian = _weighted_matrices(
        relative, actions, normalized_weights
    )
    overlap_defect = _hermiticity_residual(raw_overlap)
    hamiltonian_defect = _hermiticity_residual(raw_hamiltonian)
    finite = (
        jnp.all(jnp.isfinite(relative))
        & jnp.all(jnp.isfinite(actions))
        & jnp.all(jnp.isfinite(overlap))
        & jnp.all(jnp.isfinite(hamiltonian))
    )
    all_samples_valid = jnp.all(sample_valid)
    rayleigh: BlockRayleighEvaluation | None = None
    ritz: ReducedRitzResult | None = None
    objective = jnp.asarray(jnp.nan)
    state_count = problem.state_count
    energy_dtype = jnp.real(hamiltonian).dtype
    state_energies = jnp.full((state_count,), jnp.nan, dtype=energy_dtype)
    state_modes = jnp.full(
        (state_count, state_count),
        jnp.asarray(jnp.nan, dtype=hamiltonian.dtype),
        dtype=hamiltonian.dtype,
    )
    state_variances = jnp.full((state_count,), jnp.nan, dtype=energy_dtype)
    gram_minimum = jnp.asarray(jnp.nan)
    gram_condition = jnp.asarray(jnp.inf)
    gram_rank = jnp.asarray(0, dtype=jnp.int32)
    status_code: VMCSubspaceStatus
    if not bool(all_samples_valid):
        status_code = VMC_SUBSPACE_INVALID_SAMPLES
    elif not bool(finite):
        status_code = VMC_SUBSPACE_NONFINITE
    else:
        rayleigh = block_rayleigh_trace(
            hamiltonian,
            overlap,
            tolerance=ritz_tolerance,
        )
        objective = rayleigh.objective
        gram_minimum = rayleigh.mass_minimum_eigenvalue
        gram_condition = rayleigh.mass_condition_number
        gram_rank = rayleigh.mass_numerical_rank
        full_span = bool(
            (gram_rank == state_count)
            & (gram_minimum > ritz_tolerance)
            & jnp.isfinite(gram_condition)
        )
        if not full_span:
            status_code = VMC_SUBSPACE_SINGULAR_SPAN
        elif not bool(rayleigh.valid):
            status_code = VMC_SUBSPACE_RITZ_FAILURE
        else:
            ritz = solve_reduced_ritz(
                hamiltonian,
                overlap,
                count=state_count,
                which="smallest-algebraic",
                tolerance=ritz_tolerance,
            )
            state_energies = jnp.real(jnp.asarray(ritz.eigenvalues))
            state_modes = jnp.asarray(ritz.coefficients)
            mode_amplitudes = contract("ni,ia->na", relative, state_modes)
            mode_actions = contract("ni,ia->na", actions, state_modes)
            residuals = mode_actions - mode_amplitudes * state_energies[None, :]
            variance_numerator = jnp.mean(jnp.abs(residuals) ** 2, axis=0)
            variance_denominator = jnp.mean(jnp.abs(mode_amplitudes) ** 2, axis=0)
            state_variances = jnp.real(variance_numerator / variance_denominator)
            ritz_finite = (
                jnp.all(jnp.isfinite(state_energies))
                & jnp.all(jnp.isfinite(state_modes))
                & jnp.all(jnp.isfinite(state_variances))
            )
            status_code = (
                VMC_SUBSPACE_SUCCESS
                if bool(jnp.all(ritz.successful) & ritz_finite)
                else VMC_SUBSPACE_RITZ_FAILURE
            )
    status = jnp.asarray(status_code, dtype=jnp.int32)
    valid = status == VMC_SUBSPACE_SUCCESS
    chain_diagnostics = None
    if compute_chain_diagnostics:
        from ..uq._diagnostics import mcmc_diagnostics

        chain_diagnostics = mcmc_diagnostics(
            {
                "configuration": configurations.astype(float),
                "mixture_log_target": samples.log_target,
            },
            acceptance_rate=samples.acceptance_rate,
            divergent=jnp.zeros(samples.log_target.shape, dtype=bool),
        )
    return VariationalMonteCarloSubspaceEstimate(
        objective=objective,
        state_energies=state_energies,
        state_modes=state_modes,
        state_variances=state_variances,
        raw_overlap_matrix=raw_overlap,
        raw_hamiltonian_matrix=raw_hamiltonian,
        overlap_matrix=overlap,
        hamiltonian_matrix=hamiltonian,
        overlap_hermiticity_residual=overlap_defect,
        hamiltonian_hermiticity_residual=hamiltonian_defect,
        gram_minimum_eigenvalue=gram_minimum,
        gram_condition_number=gram_condition,
        gram_numerical_rank=gram_rank,
        relative_amplitudes=relative,
        local_hamiltonian_actions=actions,
        acceptance_rate=jnp.mean(samples.acceptance_rate),
        active_samples=active,
        valid=valid,
        status=status,
        rayleigh=rayleigh,
        ritz=ritz,
        chain_diagnostics=chain_diagnostics,
    )


def evaluate_variational_monte_carlo_subspace(
    problem: VariationalMonteCarloSubspaceProblem,
    models: Sequence[Any],
    markov_state: MarkovState,
    /,
    *,
    key: Key[Array, ""],
    num_draws: int,
    steps_per_draw: int = 1,
    warmup_steps: int = 0,
    ritz_tolerance: float = 1e-10,
    compute_chain_diagnostics: bool = False,
) -> tuple[VariationalMonteCarloSubspaceEstimate, MarkovSampleResult]:
    """Evaluate one fixed model block from a shared persistent mixture ensemble."""
    if not isinstance(problem, VariationalMonteCarloSubspaceProblem):
        raise TypeError("problem must be a VariationalMonteCarloSubspaceProblem.")
    models_ = tuple(models)
    if len(models_) != problem.state_count:
        raise ValueError("models must have the problem's fixed state count.")
    if any(not callable(model) for model in models_):
        raise TypeError("Every subspace VMC model must be callable.")
    if not isinstance(markov_state, MarkovState):
        raise TypeError("markov_state must be a MarkovState.")
    tolerance = float(ritz_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("ritz_tolerance must be finite and non-negative.")
    log_target = _mixture_log_target(models_)
    refreshed = problem.kernel.refresh(log_target, markov_state)
    samples = sample_markov(
        log_target,
        problem.kernel,
        refreshed,
        key=key,
        num_draws=num_draws,
        steps_per_draw=steps_per_draw,
        warmup_steps=warmup_steps,
    )
    estimate = _estimate_from_samples(
        problem,
        models_,
        samples,
        ritz_tolerance=tolerance,
        compute_chain_diagnostics=bool(compute_chain_diagnostics),
    )
    return estimate, samples


def _score_geometry(
    problem: VariationalMonteCarloSubspaceProblem,
    model_index: int,
    coordinates: Array,
    configurations: Array,
    responsibilities: Array,
    /,
    *,
    damping: float,
) -> EmpiricalGramLinearOperator:
    mode = problem.complex_parameter_modes[model_index]

    def features(parameter_coordinates):
        model = problem.model_from_coordinates(model_index, parameter_coordinates)
        amplitudes = _batched_amplitude(model, configurations)
        values = _surrogate(amplitudes)
        if mode in ("real", "nonholomorphic"):
            return jnp.stack((jnp.real(values), jnp.imag(values)), axis=-1)
        return values

    primal_shape = jax.eval_shape(features, coordinates)
    source = ArraySpace(coordinates.shape, dtype=coordinates.dtype)
    target = ArraySpace(primal_shape.shape, dtype=primal_shape.dtype)
    linearization = prepare_linearization(
        features,
        coordinates,
        source=source,
        target=target,
        linearization_id=f"vmc-subspace-score:{problem.problem_id}:{model_index}",
    )
    score = JacobianLinearOperator(
        linearization,
        operator_id=f"vmc-subspace-score:{problem.problem_id}:{model_index}",
    )
    return EmpiricalGramLinearOperator(
        score,
        responsibilities,
        centered=True,
        damping=damping,
        operator_id=f"vmc-subspace-metric:{problem.problem_id}:{model_index}",
    )


def _weighted_block_objective(
    problem: VariationalMonteCarloSubspaceProblem,
    coordinates: tuple[Array, ...],
    configurations: Array,
    normalized_weights: Array,
    /,
    *,
    ritz_tolerance: float,
) -> Array:
    models = problem.models_from_coordinates(coordinates)
    relative, actions, _valid = _relative_amplitudes_and_actions(
        models, problem.operator, configurations
    )
    _raw_overlap, _raw_hamiltonian, overlap, hamiltonian = _weighted_matrices(
        relative, actions, normalized_weights
    )
    return block_rayleigh_trace(
        hamiltonian,
        overlap,
        tolerance=ritz_tolerance,
    ).objective


def _score_corrected_objective(
    problem: VariationalMonteCarloSubspaceProblem,
    coordinates: tuple[Array, ...],
    configurations: Array,
    /,
    *,
    ritz_tolerance: float,
) -> Array:
    """Add the mixture-distribution score term to the frozen-sample gradient.

    Differentiating the block objective with respect to per-sample softmax logits
    gives its empirical influence function. Pairing that stopped cotangent with
    ``d log(sum_i |psi_i|^2)`` supplies the missing distribution derivative.
    Softmax shift invariance makes the unknown mixture normalization cancel.
    """
    sample_count = int(configurations.shape[0])
    zero_logits = jnp.zeros((sample_count,), dtype=float)

    def objective_from_logits(logits):
        return _weighted_block_objective(
            problem,
            coordinates,
            configurations,
            jax.nn.softmax(logits),
            ritz_tolerance=ritz_tolerance,
        )

    distribution_cotangent = jax.lax.stop_gradient(
        jax.grad(objective_from_logits)(zero_logits)
    )
    pathwise_objective = objective_from_logits(zero_logits)
    models = problem.models_from_coordinates(coordinates)
    mixture_log_weight = _batched_mixture_log_weight(models, configurations)
    zero_value_score = mixture_log_weight - jax.lax.stop_gradient(mixture_log_weight)
    score_correction = contract(
        "n,n->", distribution_cotangent, zero_value_score
    )
    return pathwise_objective + score_correction


def _validate_state(
    problem: VariationalMonteCarloSubspaceProblem,
    state: VariationalMonteCarloSubspaceState,
    /,
) -> None:
    if len(state.models) != problem.state_count:
        raise ValueError("Subspace VMC state has an incompatible model count.")
    if len(state.parameter_coordinates) != problem.state_count:
        raise ValueError("Subspace VMC state has an incompatible coordinate count.")
    if state.markov_state.num_chains != int(problem.initial_configurations.shape[0]):
        raise ValueError("Subspace VMC state chain count is incompatible.")
    for model_index, (model, current, initial) in enumerate(
        zip(
            state.models,
            state.parameter_coordinates,
            problem.initial_coordinates,
            strict=True,
        )
    ):
        if current.shape != initial.shape or current.dtype != initial.dtype:
            raise ValueError("Subspace VMC state parameter coordinates are incompatible.")
        if array_tree_signature(model) != array_tree_signature(
            problem.models[model_index]
        ):
            raise ValueError("Subspace VMC state model structure is incompatible.")
        reconstructed = problem.model_from_coordinates(model_index, current)
        actual_arrays = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
        expected_arrays = jax.tree_util.tree_leaves(
            eqx.filter(reconstructed, eqx.is_array)
        )
        if len(actual_arrays) != len(expected_arrays) or any(
            not bool(jnp.array_equal(actual, expected))
            for actual, expected in zip(
                actual_arrays, expected_arrays, strict=True
            )
        ):
            raise ValueError(
                "Subspace VMC state models and parameter coordinates are inconsistent."
            )
    if int(state.iteration) < 0:
        raise ValueError("Subspace VMC state iteration must be non-negative.")


def _raise_subspace_vmc(status: Array, role: str, /) -> None:
    raise RuntimeError(
        f"{role} failed with subspace VMC status "
        f"{vmc_subspace_status_name(status)}."
    )


def solve_variational_monte_carlo_subspace(
    problem: VariationalMonteCarloSubspaceProblem,
    policy: VariationalMonteCarloPolicy,
    /,
    *,
    key: Key[Array, ""] | None = None,
    state: VariationalMonteCarloSubspaceState | None = None,
    ritz_tolerance: float = 1e-10,
) -> VariationalMonteCarloSubspaceResult:
    """Optimize a discrete model block with score-corrected, block-diagonal SR."""
    if not isinstance(problem, VariationalMonteCarloSubspaceProblem):
        raise TypeError("problem must be a VariationalMonteCarloSubspaceProblem.")
    if not isinstance(policy, VariationalMonteCarloPolicy):
        raise TypeError("policy must be a VariationalMonteCarloPolicy.")
    tolerance = float(ritz_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("ritz_tolerance must be finite and non-negative.")
    if policy.num_iterations > 0 and problem.trainable_dimension < 1:
        raise ValueError("Subspace VMC optimization requires trainable parameters.")
    if state is None:
        if key is None:
            raise ValueError("A root key is required for a new subspace VMC run.")
        resolved_key = key
        current = problem.initial_state(key=resolved_key)
    else:
        if not isinstance(state, VariationalMonteCarloSubspaceState):
            raise TypeError(
                "state must be a VariationalMonteCarloSubspaceState or None."
            )
        resolved_key = state.root_key if key is None else key
        if not jnp.array_equal(jr.key_data(resolved_key), jr.key_data(state.root_key)):
            raise ValueError("Resume key does not match the subspace VMC state key.")
        current = state
    _validate_state(problem, current)
    objectives: list[Array] = []
    energies: list[Array] = []
    variances: list[Array] = []
    overlap_defects: list[Array] = []
    hamiltonian_defects: list[Array] = []
    acceptances: list[Array] = []
    update_norms: list[Array] = []
    statuses: list[Array] = []
    all_linear_results: list[tuple[LinearSolveResult | None, ...]] = []

    for _ in range(policy.num_iterations):
        iteration = int(current.iteration)
        iteration_key = jr.fold_in(resolved_key, iteration)
        estimate, samples = evaluate_variational_monte_carlo_subspace(
            problem,
            current.models,
            current.markov_state,
            key=iteration_key,
            num_draws=policy.draws_per_iteration,
            steps_per_draw=policy.steps_per_draw,
            warmup_steps=policy.warmup_steps if iteration == 0 else 0,
            ritz_tolerance=tolerance,
        )
        objectives.append(estimate.objective)
        energies.append(estimate.state_energies)
        variances.append(estimate.state_variances)
        overlap_defects.append(estimate.overlap_hermiticity_residual)
        hamiltonian_defects.append(estimate.hamiltonian_hermiticity_residual)
        acceptances.append(estimate.acceptance_rate)
        if not bool(estimate.successful):
            statuses.append(estimate.status)
            update_norms.append(jnp.asarray(jnp.nan))
            current = VariationalMonteCarloSubspaceState(
                models=current.models,
                parameter_coordinates=current.parameter_coordinates,
                markov_state=samples.final_state,
                iteration=iteration,
                root_key=resolved_key,
            )
            if policy.failure_mode == "raise":
                _raise_subspace_vmc(estimate.status, "Subspace VMC estimation")
            break

        flat = jnp.asarray(samples.samples).reshape(
            (-1,) + problem.operator.configuration_shape
        )
        objective_gradient = jax.grad(
            _score_corrected_objective,
            argnums=1,
        )(
            problem,
            current.parameter_coordinates,
            flat,
            ritz_tolerance=tolerance,
        )
        directions: list[Array] = []
        iteration_linear_results: list[LinearSolveResult | None] = []
        linear_success = True
        for model_index, (coordinates, gradient) in enumerate(
            zip(
                current.parameter_coordinates,
                objective_gradient,
                strict=True,
            )
        ):
            if int(coordinates.size) == 0:
                directions.append(jnp.empty((0,), dtype=coordinates.dtype))
                iteration_linear_results.append(None)
                continue
            responsibilities = jnp.abs(
                estimate.relative_amplitudes[:, model_index]
            ) ** 2
            metric = _score_geometry(
                problem,
                model_index,
                coordinates,
                flat,
                responsibilities,
                damping=policy.damping,
            )
            force = jnp.conj(gradient) if jnp.iscomplexobj(coordinates) else gradient
            linear = solve(
                LinearSystem(metric, nullspace_policy=policy.nullspace_policy),
                force,
                policy=policy.linear_policy,
            )
            direction = jnp.asarray(linear.value)
            iteration_linear_results.append(linear)
            directions.append(direction)
            linear_success = linear_success and bool(
                jnp.all(linear.successful) & jnp.all(jnp.isfinite(direction))
            )
        linear_tuple = tuple(iteration_linear_results)
        all_linear_results.append(linear_tuple)
        if not linear_success:
            status = jnp.asarray(VMC_SUBSPACE_LINEAR_FAILURE, dtype=jnp.int32)
            statuses.append(status)
            update_norms.append(jnp.asarray(jnp.nan))
            current = VariationalMonteCarloSubspaceState(
                models=current.models,
                parameter_coordinates=current.parameter_coordinates,
                markov_state=samples.final_state,
                iteration=iteration,
                root_key=resolved_key,
            )
            if policy.failure_mode == "raise":
                _raise_subspace_vmc(status, "Subspace VMC metric solve")
            break
        norm = jnp.sqrt(
            sum(
                jnp.real(jnp.vdot(direction, direction))
                for direction in directions
            )
        )
        scale = jnp.asarray(1.0)
        if policy.max_update_norm is not None:
            scale = jnp.minimum(
                1.0,
                policy.max_update_norm / jnp.maximum(norm, 1e-30),
            )
        scaled_directions = tuple(scale * direction for direction in directions)
        update_norm = scale * norm
        next_coordinates = tuple(
            coordinates - policy.learning_rate * direction
            for coordinates, direction in zip(
                current.parameter_coordinates,
                scaled_directions,
                strict=True,
            )
        )
        next_models = problem.models_from_coordinates(next_coordinates)
        statuses.append(jnp.asarray(VMC_SUBSPACE_SUCCESS, dtype=jnp.int32))
        update_norms.append(update_norm)
        current = VariationalMonteCarloSubspaceState(
            models=next_models,
            parameter_coordinates=next_coordinates,
            markov_state=samples.final_state,
            iteration=iteration + 1,
            root_key=resolved_key,
        )

    final_key = jr.fold_in(resolved_key, 0x5B5A)
    final_estimate, _final_samples = evaluate_variational_monte_carlo_subspace(
        problem,
        current.models,
        current.markov_state,
        key=final_key,
        num_draws=policy.final_evaluation_draws,
        steps_per_draw=policy.steps_per_draw,
        ritz_tolerance=tolerance,
        compute_chain_diagnostics=policy.final_chain_diagnostics,
    )
    state_count = problem.state_count
    return VariationalMonteCarloSubspaceResult(
        final_state=current,
        final_estimate=final_estimate,
        objective_history=jnp.stack(objectives)
        if objectives
        else jnp.empty((0,)),
        state_energy_history=jnp.stack(energies)
        if energies
        else jnp.empty((0, state_count)),
        state_variance_history=jnp.stack(variances)
        if variances
        else jnp.empty((0, state_count)),
        overlap_hermiticity_history=jnp.stack(overlap_defects)
        if overlap_defects
        else jnp.empty((0,)),
        hamiltonian_hermiticity_history=jnp.stack(hamiltonian_defects)
        if hamiltonian_defects
        else jnp.empty((0,)),
        acceptance_history=jnp.stack(acceptances)
        if acceptances
        else jnp.empty((0,)),
        update_norm_history=jnp.stack(update_norms)
        if update_norms
        else jnp.empty((0,)),
        status_history=jnp.stack(statuses)
        if statuses
        else jnp.empty((0,), dtype=jnp.int32),
        linear_results=tuple(all_linear_results),
        root_key=resolved_key,
        problem_id=problem.problem_id,
        completed_iterations=int(current.iteration),
    )


__all__ = [
    "VMC_SUBSPACE_INVALID_SAMPLES",
    "VMC_SUBSPACE_LINEAR_FAILURE",
    "VMC_SUBSPACE_NONFINITE",
    "VMC_SUBSPACE_RITZ_FAILURE",
    "VMC_SUBSPACE_SINGULAR_SPAN",
    "VMC_SUBSPACE_SUCCESS",
    "VMCSubspaceStatus",
    "VariationalMonteCarloSubspaceEstimate",
    "VariationalMonteCarloSubspaceProblem",
    "VariationalMonteCarloSubspaceResult",
    "VariationalMonteCarloSubspaceState",
    "evaluate_variational_monte_carlo_subspace",
    "solve_variational_monte_carlo_subspace",
    "vmc_subspace_status_name",
]
