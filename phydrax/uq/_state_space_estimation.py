#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key, PyTree

from .._strict import StrictModule
from ..optim import DifferentialEvolutionSearch
from ..stochastic._state_space import StateSpaceProblem
from ._bellman import BellmanFilterResult
from ._ensemble_filter import EnsembleFilterResult
from ._guided_particle import GuidedParticleFilterResult
from ._kalman import KalmanExecutionMethod
from ._laplace import fit_laplace, LaplaceResult
from ._laplax_backend import StructuredLaplaceResult
from ._map import find_map, MAPResult
from ._map_search import MAPSearchResult, PositionBounds, search_map
from ._particle import ParticleFilterResult
from ._posterior import ParameterSpace, PosteriorProblem
from ._posterior_terms import AbstractPosteriorTerm
from ._rao_blackwellized import (
    RaoBlackwellizedFilterResult,
    RaoBlackwellizedStateSpaceProblem,
)
from ._state_space_inference import (
    exact_state_space_log_likelihood,
    ExactStateSpaceLikelihood,
    ExactStateSpaceMethod,
)


StateSpaceEstimationProblem: TypeAlias = (
    StateSpaceProblem | RaoBlackwellizedStateSpaceProblem
)
ApproximateStateSpaceLikelihoodResult: TypeAlias = (
    BellmanFilterResult
    | ParticleFilterResult
    | GuidedParticleFilterResult
    | EnsembleFilterResult
    | RaoBlackwellizedFilterResult
)
StateSpaceLikelihoodBackend: TypeAlias = (
    ExactStateSpaceLikelihood | ApproximateStateSpaceLikelihoodResult
)
StateSpaceLikelihoodFunction: TypeAlias = Callable[
    [StateSpaceEstimationProblem], StateSpaceLikelihoodBackend
]
StateSpaceSampler: TypeAlias = Callable[..., Any]


def _name(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _case_contract(
    case_axes: Sequence[str],
    case_shape: Sequence[int],
    case_ids: Sequence[str],
    /,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
    axes = tuple(str(axis) for axis in case_axes)
    shape = tuple(int(size) for size in case_shape)
    identifiers = tuple(str(identifier) for identifier in case_ids)
    if any(not axis for axis in axes) or len(set(axes)) != len(axes):
        raise ValueError("case_axes must contain unique non-empty names.")
    if any(size <= 0 for size in shape):
        raise ValueError("case_shape dimensions must be positive.")
    if len(axes) != len(shape):
        raise ValueError("case_axes and case_shape must have equal rank.")
    count = prod(shape) if shape else 1
    if (
        len(identifiers) != count
        or any(not identifier for identifier in identifiers)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise ValueError("case_ids must contain one unique non-empty ID per case.")
    return axes, shape, identifiers


class ExperimentStateSpaceLikelihood(StrictModule):
    """Likelihood and untouched filtering diagnostics for one named experiment."""

    per_case_log_likelihood: Array
    total_log_likelihood: Array
    incremental_log_likelihood: Array
    cumulative_log_likelihood: Array
    step_valid: Array
    valid: Array
    status: Array
    backend: StateSpaceLikelihoodBackend
    problem: StateSpaceEstimationProblem
    experiment_id: str = eqx.field(static=True)
    case_axes: tuple[str, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    likelihood_id: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    temporal_method: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    sequence_id: str = eqx.field(static=True)
    input_id: str | None = eqx.field(static=True)
    model_discretization_id: str | None = eqx.field(static=True)
    observation_discretization_id: str | None = eqx.field(static=True)
    covariance_regularization: float | None = eqx.field(static=True)
    curvature_damping: float | None = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.step_valid, axis=-1)


class MultiExperimentStateSpaceLikelihoodResult(StrictModule):
    """Aggregated scalar likelihood retaining every experiment's native record."""

    experiments: tuple[ExperimentStateSpaceLikelihood, ...]
    per_experiment_log_likelihood: Array
    per_case_log_likelihood: Array
    total_log_likelihood: Array
    experiment_ids: tuple[str, ...] = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        values = tuple(jnp.all(experiment.successful) for experiment in self.experiments)
        return jnp.all(jnp.stack(values))

    def experiment(self, experiment_id: str, /) -> ExperimentStateSpaceLikelihood:
        """Return one diagnostic record by its explicit semantic experiment ID."""
        resolved = _name(experiment_id, owner="experiment_id")
        if resolved not in self.experiment_ids:
            raise KeyError(resolved)
        return self.experiments[self.experiment_ids.index(resolved)]


class StateSpaceExperiment(StrictModule):
    """Parameterized state-space problem with an explicit experiment/case contract."""

    problem_fn: Callable[[PyTree[Any]], StateSpaceEstimationProblem] = eqx.field(
        static=True
    )
    likelihood_fn: StateSpaceLikelihoodFunction | None = eqx.field(static=True)
    experiment_id: str = eqx.field(static=True)
    case_axes: tuple[str, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    likelihood_id: str = eqx.field(static=True)
    exact_method: ExactStateSpaceMethod = eqx.field(static=True)
    covariance_regularization: float = eqx.field(static=True)
    temporal_method: KalmanExecutionMethod = eqx.field(static=True)
    transform_safe: bool = eqx.field(static=True)

    def __init__(
        self,
        problem: Callable[[PyTree[Any]], StateSpaceEstimationProblem],
        /,
        *,
        experiment_id: str,
        case_axes: Sequence[str],
        case_shape: Sequence[int],
        case_ids: Sequence[str],
        likelihood: StateSpaceLikelihoodFunction | None = None,
        likelihood_id: str = "exact-state-space",
        exact_method: ExactStateSpaceMethod = "auto",
        covariance_regularization: float = 0.0,
        temporal_method: KalmanExecutionMethod = "auto",
        transform_safe: bool = False,
    ):
        if not callable(problem):
            raise TypeError("problem must be callable.")
        if likelihood is not None and not callable(likelihood):
            raise TypeError("likelihood must be callable or None.")
        if not isinstance(transform_safe, bool):
            raise TypeError("transform_safe must be a bool.")
        if exact_method not in ("auto", "kalman", "finite-state"):
            raise ValueError("exact_method must be 'auto', 'kalman', or 'finite-state'.")
        if temporal_method not in ("auto", "sequential", "parallel"):
            raise ValueError(
                "temporal_method must be 'auto', 'sequential', or 'parallel'."
            )
        regularization = float(covariance_regularization)
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("covariance_regularization must be finite and nonnegative.")
        if likelihood is not None:
            if likelihood_id == "exact-state-space":
                raise ValueError(
                    "A custom likelihood requires an explicit non-exact likelihood_id."
                )
            if (
                exact_method != "auto"
                or regularization != 0.0
                or temporal_method != "auto"
            ):
                raise ValueError(
                    "Exact likelihood options cannot be combined with a custom likelihood."
                )
        axes, shape, identifiers = _case_contract(case_axes, case_shape, case_ids)
        self.problem_fn = problem
        self.likelihood_fn = likelihood
        self.experiment_id = _name(experiment_id, owner="experiment_id")
        self.case_axes = axes
        self.case_shape = shape
        self.case_ids = identifiers
        self.likelihood_id = _name(likelihood_id, owner="likelihood_id")
        self.exact_method = exact_method
        self.covariance_regularization = regularization
        self.temporal_method = temporal_method
        self.transform_safe = transform_safe

    def problem(self, parameters: PyTree[Any], /) -> StateSpaceEstimationProblem:
        problem = self.problem_fn(parameters)
        if not isinstance(
            problem, (StateSpaceProblem, RaoBlackwellizedStateSpaceProblem)
        ):
            raise TypeError(
                "problem(parameters) must return a StateSpaceProblem or "
                "RaoBlackwellizedStateSpaceProblem."
            )
        if (
            isinstance(problem, RaoBlackwellizedStateSpaceProblem)
            and self.likelihood_fn is None
        ):
            raise TypeError(
                "A RaoBlackwellizedStateSpaceProblem requires a custom likelihood."
            )
        observations = problem.observations
        if observations.case_axes != self.case_axes:
            raise ValueError(
                f"Experiment {self.experiment_id!r} produced different case axes."
            )
        if observations.case_shape != self.case_shape:
            raise ValueError(
                f"Experiment {self.experiment_id!r} produced a different case shape."
            )
        if observations.case_ids != self.case_ids:
            raise ValueError(
                f"Experiment {self.experiment_id!r} produced different case IDs."
            )
        return problem

    def evaluate(self, parameters: PyTree[Any], /) -> ExperimentStateSpaceLikelihood:
        problem = self.problem(parameters)
        if self.likelihood_fn is None:
            if not isinstance(problem, StateSpaceProblem):
                raise TypeError("Exact likelihood requires a StateSpaceProblem.")
            backend = exact_state_space_log_likelihood(
                problem,
                method=self.exact_method,
                covariance_regularization=self.covariance_regularization,
                temporal_method=self.temporal_method,
            )
        else:
            backend = self.likelihood_fn(problem)
        return _experiment_likelihood(self, problem, backend)


def _experiment_likelihood(
    experiment: StateSpaceExperiment,
    problem: StateSpaceEstimationProblem,
    backend: StateSpaceLikelihoodBackend,
    /,
) -> ExperimentStateSpaceLikelihood:
    if isinstance(backend, ExactStateSpaceLikelihood):
        per_case = backend.per_case_log_likelihood
        total = backend.total_log_likelihood
        incremental = backend.incremental_log_likelihood
        cumulative = backend.cumulative_log_likelihood
        step_valid = backend.step_valid
        valid = backend.valid
        status = backend.status
        method = backend.method
        temporal_method = backend.temporal_method
        approximation_id = backend.problem.model.approximation_id
        covariance_regularization = experiment.covariance_regularization
        curvature_damping = None
    elif isinstance(backend, ParticleFilterResult):
        per_case = jnp.where(
            backend.successful,
            backend.cumulative_log_likelihood[..., -1],
            -jnp.inf,
        )
        total = jnp.sum(per_case).reshape(())
        incremental = backend.incremental_log_likelihood
        cumulative = backend.cumulative_log_likelihood
        step_valid = backend.step_valid
        valid = backend.valid
        status = backend.status
        method = "bootstrap-particle"
        temporal_method = "sequential"
        approximation_id = f"particle:{backend.num_particles}"
        covariance_regularization = None
        curvature_damping = None
    elif isinstance(backend, BellmanFilterResult):
        per_case = jnp.where(
            backend.successful,
            backend.cumulative_pseudo_log_likelihood[..., -1],
            -jnp.inf,
        )
        total = jnp.sum(per_case).reshape(())
        incremental = backend.incremental_pseudo_log_likelihood
        cumulative = backend.cumulative_pseudo_log_likelihood
        step_valid = backend.step_valid
        valid = backend.valid
        status = backend.status
        method = "bellman-pseudo"
        temporal_method = "sequential"
        approximation_id = (
            f"bellman:{backend.execution_method}:{backend.curvature_method}"
        )
        covariance_regularization = None
        curvature_damping = backend.curvature_damping
    elif isinstance(backend, GuidedParticleFilterResult):
        per_case = jnp.where(
            backend.successful,
            backend.cumulative_log_likelihood[..., -1],
            -jnp.inf,
        )
        total = jnp.sum(per_case).reshape(())
        incremental = backend.incremental_log_likelihood
        cumulative = backend.cumulative_log_likelihood
        step_valid = backend.step_valid
        valid = backend.valid
        status = backend.status
        method = "guided-particle"
        temporal_method = "sequential"
        approximation_id = (
            f"guided-particle:{backend.proposal_id}:{backend.num_particles}"
        )
        covariance_regularization = None
        curvature_damping = None
    elif isinstance(backend, EnsembleFilterResult):
        per_case = jnp.where(
            backend.successful,
            backend.cumulative_log_likelihood[..., -1],
            -jnp.inf,
        )
        total = jnp.sum(per_case).reshape(())
        incremental = backend.incremental_log_likelihood
        cumulative = backend.cumulative_log_likelihood
        step_valid = backend.step_valid
        valid = backend.valid
        status = backend.status
        method = "ensemble-transform-kalman"
        temporal_method = "sequential"
        approximation_id = f"ensemble:{backend.ensemble_size}"
        covariance_regularization = backend.covariance_regularization
        curvature_damping = None
    elif isinstance(backend, RaoBlackwellizedFilterResult):
        per_case = jnp.where(
            backend.successful,
            backend.cumulative_log_likelihood[..., -1],
            -jnp.inf,
        )
        total = jnp.sum(per_case).reshape(())
        incremental = backend.incremental_log_likelihood
        cumulative = backend.cumulative_log_likelihood
        step_valid = backend.step_valid
        valid = backend.valid
        status = backend.status
        method = "rao-blackwellized-particle"
        temporal_method = "sequential"
        approximation_id = (
            "rao-blackwellized:"
            f"{backend.problem.model.nonlinear_transition.approximation_id}:"
            f"{backend.num_particles}"
        )
        covariance_regularization = None
        curvature_damping = None
    else:
        raise TypeError(
            "likelihood(problem) must return ExactStateSpaceLikelihood, "
            "BellmanFilterResult, ParticleFilterResult, GuidedParticleFilterResult, "
            "EnsembleFilterResult, or RaoBlackwellizedFilterResult."
        )
    if backend.problem is not problem:
        raise ValueError(
            f"Experiment {experiment.experiment_id!r} likelihood backend must retain "
            "the exact evaluated StateSpaceProblem or RaoBlackwellizedStateSpaceProblem, "
            "including its model, input, and observation schedule; cached or relabelled "
            "diagnostics are not accepted."
        )
    expected_case_shape = experiment.case_shape
    if per_case.shape != expected_case_shape:
        raise ValueError(
            f"Experiment {experiment.experiment_id!r} likelihood returned per-case "
            f"shape {per_case.shape}, expected {expected_case_shape}."
        )
    expected_step_shape = expected_case_shape + (problem.observations.num_steps,)
    for owner, value in (
        ("incremental_log_likelihood", incremental),
        ("cumulative_log_likelihood", cumulative),
        ("step_valid", step_valid),
        ("valid", valid),
        ("status", status),
    ):
        if value.shape != expected_step_shape:
            raise ValueError(
                f"Experiment {experiment.experiment_id!r} likelihood returned "
                f"{owner} shape {value.shape}, expected {expected_step_shape}."
            )
    return ExperimentStateSpaceLikelihood(
        per_case_log_likelihood=per_case,
        total_log_likelihood=total,
        incremental_log_likelihood=incremental,
        cumulative_log_likelihood=cumulative,
        step_valid=step_valid,
        valid=valid,
        status=status,
        backend=backend,
        problem=problem,
        experiment_id=experiment.experiment_id,
        case_axes=experiment.case_axes,
        case_shape=experiment.case_shape,
        case_ids=experiment.case_ids,
        likelihood_id=experiment.likelihood_id,
        method=method,
        temporal_method=temporal_method,
        approximation_id=approximation_id,
        model_id=problem.model.model_id,
        problem_id=problem.problem_id,
        sequence_id=problem.observations.sequence_id,
        input_id=(
            None if problem.input_signal is None else problem.input_signal.input_id
        ),
        model_discretization_id=(
            problem.model.discretization_id
            if isinstance(problem, StateSpaceProblem)
            else None
        ),
        observation_discretization_id=problem.observations.discretization_id,
        covariance_regularization=covariance_regularization,
        curvature_damping=curvature_damping,
    )


class MultiExperimentStateSpaceLikelihood(AbstractPosteriorTerm):
    """Normalized likelihood term summing named experiments and physical cases."""

    experiments: tuple[StateSpaceExperiment, ...]

    def __init__(
        self,
        experiments: Sequence[StateSpaceExperiment],
        /,
        *,
        label: str = "multi_experiment_state_space",
    ):
        resolved = tuple(experiments)
        if not resolved:
            raise ValueError("At least one state-space experiment is required.")
        if any(not isinstance(value, StateSpaceExperiment) for value in resolved):
            raise TypeError("Every experiment must be a StateSpaceExperiment.")
        identifiers = tuple(experiment.experiment_id for experiment in resolved)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("experiment_id values must be unique.")
        qualified_cases = tuple(
            (experiment.experiment_id, case_id)
            for experiment in resolved
            for case_id in experiment.case_ids
        )
        if len(set(qualified_cases)) != len(qualified_cases):
            raise ValueError("Qualified experiment/case identities must be unique.")
        self.experiments = resolved
        self.label = _name(label, owner="label")

    def evaluate(
        self, parameters: PyTree[Any], /
    ) -> MultiExperimentStateSpaceLikelihoodResult:
        diagnostics = tuple(
            experiment.evaluate(parameters) for experiment in self.experiments
        )
        per_experiment = jnp.stack(
            tuple(result.total_log_likelihood for result in diagnostics)
        )
        per_case = jnp.concatenate(
            tuple(result.per_case_log_likelihood.reshape((-1,)) for result in diagnostics)
        )
        return MultiExperimentStateSpaceLikelihoodResult(
            experiments=diagnostics,
            per_experiment_log_likelihood=per_experiment,
            per_case_log_likelihood=per_case,
            total_log_likelihood=jnp.sum(per_experiment).reshape(()),
            experiment_ids=tuple(
                experiment.experiment_id for experiment in self.experiments
            ),
        )

    def per_case_log_prob(self, parameters: PyTree[Any], /) -> Array:
        return self.evaluate(parameters).per_case_log_likelihood


class StateSpaceMAPWorkflowResult(StrictModule):
    """Existing global/local MAP records plus diagnostics at the selected mode."""

    likelihood: MultiExperimentStateSpaceLikelihoodResult
    global_search: MAPSearchResult | None
    local_map: MAPResult | None
    workflow: Literal["global", "local", "global-local"] = eqx.field(static=True)

    @property
    def position(self) -> PyTree[Array]:
        if self.local_map is not None:
            return self.local_map.position
        if self.global_search is None:
            raise RuntimeError("MAP workflow has no optimization result.")
        return self.global_search.position

    @property
    def parameters(self) -> PyTree[Array]:
        if self.local_map is not None:
            return self.local_map.parameters
        if self.global_search is None:
            raise RuntimeError("MAP workflow has no optimization result.")
        return self.global_search.parameters

    @property
    def log_density(self) -> Array:
        if self.local_map is not None:
            return self.local_map.log_density
        if self.global_search is None:
            raise RuntimeError("MAP workflow has no optimization result.")
        return self.global_search.log_density


class StateSpaceLaplaceWorkflowResult(StrictModule):
    """Existing Laplace record with per-experiment diagnostics at its mode."""

    approximation: LaplaceResult | StructuredLaplaceResult
    likelihood: MultiExperimentStateSpaceLikelihoodResult
    source_map: StateSpaceMAPWorkflowResult | None


class StateSpaceSamplingWorkflowResult(StrictModule):
    """Existing sampler output with declared reference-point diagnostics."""

    result: Any
    reference_likelihood: MultiExperimentStateSpaceLikelihoodResult
    reference_position: PyTree[Array]
    sampler_id: str = eqx.field(static=True)


class StateSpaceEstimation(StrictModule):
    """Multi-experiment composition over posterior, MAP, Laplace, and samplers."""

    likelihood: MultiExperimentStateSpaceLikelihood
    posterior: PosteriorProblem

    def __init__(
        self,
        parameter_space: ParameterSpace,
        experiments: Sequence[StateSpaceExperiment],
        /,
        *,
        label: str = "multi_experiment_state_space",
    ):
        if not isinstance(parameter_space, ParameterSpace):
            raise TypeError("parameter_space must be a ParameterSpace.")
        likelihood = MultiExperimentStateSpaceLikelihood(experiments, label=label)
        self.likelihood = likelihood
        self.posterior = PosteriorProblem(parameter_space, likelihood)

    def evaluate_likelihood(
        self, physical_parameters: PyTree[Any], /
    ) -> MultiExperimentStateSpaceLikelihoodResult:
        return self.likelihood.evaluate(physical_parameters)

    def _require_transform_safe_likelihoods(self, workflow: str, /) -> None:
        unsupported = tuple(
            experiment
            for experiment in self.likelihood.experiments
            if experiment.likelihood_fn is not None and not experiment.transform_safe
        )
        if not unsupported:
            return
        details = ", ".join(
            f"{experiment.experiment_id!r} ({experiment.likelihood_id!r})"
            for experiment in unsupported
        )
        raise ValueError(
            f"{workflow} requires explicitly transform-safe likelihood backends; "
            "custom likelihood experiments do not declare that capability: "
            f"{details}. Use evaluate_likelihood or global_map for non-gradient "
            "likelihood evaluation."
        )

    def local_map(
        self,
        initial_position: PyTree[Array] | None = None,
        /,
        *,
        max_steps: int = 500,
        gradient_tolerance: float = 1e-6,
        objective_tolerance: float | None = None,
        learning_rate: float = 1.0,
        memory: int = 10,
        raise_on_failure: bool = True,
    ) -> StateSpaceMAPWorkflowResult:
        self._require_transform_safe_likelihoods("local_map")
        local = find_map(
            self.posterior,
            initial_position,
            max_steps=max_steps,
            gradient_tolerance=gradient_tolerance,
            objective_tolerance=objective_tolerance,
            learning_rate=learning_rate,
            memory=memory,
            raise_on_failure=raise_on_failure,
        )
        return StateSpaceMAPWorkflowResult(
            likelihood=self.evaluate_likelihood(local.parameters),
            global_search=None,
            local_map=local,
            workflow="local",
        )

    def global_map(
        self,
        search: DifferentialEvolutionSearch,
        /,
        *,
        key: Key[Array, ""],
        position_bounds: PositionBounds,
        initial_position: PyTree[Array] | None = None,
    ) -> StateSpaceMAPWorkflowResult:
        global_result = search_map(
            self.posterior,
            search,
            key=key,
            position_bounds=position_bounds,
            initial_position=initial_position,
        )
        return StateSpaceMAPWorkflowResult(
            likelihood=self.evaluate_likelihood(global_result.parameters),
            global_search=global_result,
            local_map=None,
            workflow="global",
        )

    def global_then_local_map(
        self,
        search: DifferentialEvolutionSearch,
        /,
        *,
        key: Key[Array, ""],
        position_bounds: PositionBounds,
        initial_position: PyTree[Array] | None = None,
        max_steps: int = 500,
        gradient_tolerance: float = 1e-6,
        objective_tolerance: float | None = None,
        learning_rate: float = 1.0,
        memory: int = 10,
        raise_on_failure: bool = True,
    ) -> StateSpaceMAPWorkflowResult:
        self._require_transform_safe_likelihoods("global_then_local_map")
        global_result = search_map(
            self.posterior,
            search,
            key=key,
            position_bounds=position_bounds,
            initial_position=initial_position,
        )
        local = find_map(
            self.posterior,
            global_result.position,
            max_steps=max_steps,
            gradient_tolerance=gradient_tolerance,
            objective_tolerance=objective_tolerance,
            learning_rate=learning_rate,
            memory=memory,
            raise_on_failure=raise_on_failure,
        )
        return StateSpaceMAPWorkflowResult(
            likelihood=self.evaluate_likelihood(local.parameters),
            global_search=global_result,
            local_map=local,
            workflow="global-local",
        )

    def laplace(
        self,
        source_map: StateSpaceMAPWorkflowResult
        | MAPResult
        | MAPSearchResult
        | None = None,
        /,
        *,
        curvature: Literal["exact", "full", "diagonal", "lanczos", "lobpcg"] = "exact",
        damping: float = 0.0,
        stationarity_tolerance: float | None = 1e-4,
        max_dimension: int = 256,
        prior_precision: float | None = None,
        rank: int = 20,
        key: Array | None = None,
        tolerance: float = 1e-6,
        mv_jit: bool = True,
        likelihood_curvature: Literal["hessian", "ggn"] = "hessian",
    ) -> StateSpaceLaplaceWorkflowResult:
        workflow_map: StateSpaceMAPWorkflowResult | None
        if source_map is None:
            position = self.posterior.initial_position
            workflow_map = None
        elif isinstance(source_map, StateSpaceMAPWorkflowResult):
            position = source_map.position
            workflow_map = source_map
        elif isinstance(source_map, MAPResult):
            position = source_map.position
            workflow_map = None
        elif isinstance(source_map, MAPSearchResult):
            position = source_map.position
            workflow_map = None
        else:
            raise TypeError(
                "source_map must be a state-space workflow, MAPResult, "
                "MAPSearchResult, or None."
            )
        self._require_transform_safe_likelihoods("laplace")
        approximation = fit_laplace(
            self.posterior,
            position,
            curvature=curvature,
            damping=damping,
            stationarity_tolerance=stationarity_tolerance,
            max_dimension=max_dimension,
            prior_precision=prior_precision,
            rank=rank,
            key=key,
            tolerance=tolerance,
            mv_jit=mv_jit,
            likelihood_curvature=likelihood_curvature,
        )
        physical = self.posterior.parameter_space.constrain(position)
        return StateSpaceLaplaceWorkflowResult(
            approximation=approximation,
            likelihood=self.evaluate_likelihood(physical),
            source_map=workflow_map,
        )

    def sample(
        self,
        sampler: StateSpaceSampler,
        /,
        *,
        sampler_id: str,
        reference_position: PyTree[Array] | None = None,
        **kwargs: Any,
    ) -> StateSpaceSamplingWorkflowResult:
        """Call an existing posterior sampler without changing its algorithm contract."""
        if not callable(sampler):
            raise TypeError("sampler must be callable.")
        position = (
            self.posterior.initial_position
            if reference_position is None
            else reference_position
        )
        physical = self.posterior.parameter_space.constrain(position)
        resolved_sampler_id = _name(sampler_id, owner="sampler_id")
        reference_likelihood = self.evaluate_likelihood(physical)
        result = sampler(self.posterior, **kwargs)
        return StateSpaceSamplingWorkflowResult(
            result=result,
            reference_likelihood=reference_likelihood,
            reference_position=position,
            sampler_id=resolved_sampler_id,
        )


__all__ = [
    "ApproximateStateSpaceLikelihoodResult",
    "ExperimentStateSpaceLikelihood",
    "MultiExperimentStateSpaceLikelihood",
    "MultiExperimentStateSpaceLikelihoodResult",
    "StateSpaceEstimationProblem",
    "StateSpaceEstimation",
    "StateSpaceExperiment",
    "StateSpaceLaplaceWorkflowResult",
    "StateSpaceLikelihoodBackend",
    "StateSpaceLikelihoodFunction",
    "StateSpaceMAPWorkflowResult",
    "StateSpaceSampler",
    "StateSpaceSamplingWorkflowResult",
]
