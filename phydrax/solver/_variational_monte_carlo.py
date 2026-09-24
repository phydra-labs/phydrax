#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Mapping
from math import isfinite
from pathlib import Path
from typing import Any, ClassVar, final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key

import phydrax.axes as cx

from .._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from .._fingerprint import array_tree_signature, canonical_fingerprint
from .._identity import callable_payload
from .._sampling import (
    FullMarkovTarget,
    IncrementalMarkovTarget,
    MarkovSampleResult,
    MarkovState,
    MetropolisHastings,
    sample_markov,
)
from .._strict import StrictModule
from .._trainable import (
    ArrayRole,
    combine_parameters,
    ExplicitFreeze,
    parameter_field,
    require_parameter_roles,
)
from .._training import TrainingController, TrainingIterationKind, TrainingProgress
from .._training_kernel import (
    AbstractKernelUpdateRule,
    KernelObjective,
    KernelUpdateContext,
    prepare_training_kernel,
    PreparedTrainingKernel,
    run_training_attempt,
    training_site_key,
    TrainingAttemptEvidence,
    TrainingAttemptOutcome,
    TrainingKernelSpec,
    TrainingKernelState,
    TrainingRejectionBudgetError,
)
from .._training_objective import _ObjectiveContribution
from .._tree_math import tree_norm
from ..integration import integrate, markov_chain_measure
from ..linalg import (
    ArraySpace,
    EmpiricalGramLinearOperator,
    JacobianLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    NullspacePolicy,
    prepare_linearization,
    solve,
)
from ..nn.parameters import ParameterSubspace
from ..operators.quantum import (
    AbstractLocalQuantumOperator,
    ComplexParameterMode,
    evaluate_local_operator,
    LocalOperatorEstimate,
    LogAmplitude,
    sampling_log_weight,
)


if TYPE_CHECKING:
    from ..uq._diagnostics import MCMCDiagnostics


VMCStatus: TypeAlias = Literal[0, 1, 2, 3, 4]
VMC_SUCCESS: VMCStatus = 0
VMC_INVALID_SAMPLES: VMCStatus = 1
VMC_NONFINITE: VMCStatus = 2
VMC_IMAGINARY_ENERGY: VMCStatus = 3
VMC_LINEAR_FAILURE: VMCStatus = 4
FailureMode: TypeAlias = Literal["raise", "record"]

_VMC_CHECKPOINT_KIND = "variational-monte-carlo-state"
_VMC_CHECKPOINT_STATE_FIELDS = frozenset(
    {
        "iteration",
        "attempt_cursor",
        "model_tree",
        "position_tree",
        "parameter_coordinates_array",
        "markov_step_index_array",
        "root_key_data_array",
        "markov_valid_array",
    }
)
_VMC_OBJECTIVE_ID = "variational-monte-carlo"


def vmc_status_name(status: int | Array, /) -> str:
    code = int(status)
    names = (
        "success",
        "invalid_samples",
        "nonfinite",
        "imaginary_energy",
        "linear_failure",
    )
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown VMC status {code}.")
    return names[code]


def _amplitude(model: Any, configuration: Array, /) -> LogAmplitude:
    value = model(configuration)
    if not isinstance(value, LogAmplitude):
        raise TypeError("The VMC model must return LogAmplitude.")
    if value.log_abs.shape != ():
        raise ValueError(
            "The VMC model must return one scalar amplitude per configuration."
        )
    return value


def _model_log_target(model: Any):
    def log_target(configuration):
        return sampling_log_weight(_amplitude(model, configuration))

    return log_target


def _explicit_model_target(target: Any, /) -> FullMarkovTarget | IncrementalMarkovTarget:
    if not isinstance(target, (FullMarkovTarget, IncrementalMarkovTarget)):
        raise TypeError(
            "target_factory must return FullMarkovTarget or IncrementalMarkovTarget."
        )
    return target


def _target_kind(target: FullMarkovTarget | IncrementalMarkovTarget, /) -> str:
    return "incremental" if isinstance(target, IncrementalMarkovTarget) else "full"


def _parameter_mode(value: ComplexParameterMode, /) -> ComplexParameterMode:
    if value not in ("real", "holomorphic", "nonholomorphic"):
        raise ValueError(
            "complex_parameter_mode must be 'real', 'holomorphic', or 'nonholomorphic'."
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
    size = exemplar.shape[0]
    if values.shape != (2 * size,):
        raise ValueError(
            f"Nonholomorphic coordinates must have shape ({2 * size},); got {values.shape}."
        )
    return (values[:size] + 1j * values[size:]).astype(exemplar.dtype)


def _surrogate(amplitude: LogAmplitude, /) -> Array:
    safe_phase = jax.lax.stop_gradient(amplitude.phase)
    return amplitude.log_abs + amplitude.phase / safe_phase


def _default_parameter_subspace(model: Any, /) -> ParameterSubspace:
    resolution = require_parameter_roles(model, context="VariationalMonteCarloProblem")
    return ParameterSubspace(model, resolution.filter_spec(ArrayRole.PARAMETER))


class VariationalMonteCarloProblem(StrictModule):
    """Amplitude model, local operator, sampler, and selected parameter space."""

    model: Any
    operator: AbstractLocalQuantumOperator
    kernel: MetropolisHastings
    initial_configurations: Array
    parameter_subspace: ParameterSubspace
    initial_parameter_vector: Array
    initial_coordinates: Array
    target_factory: Callable[[Any], Any] | None = eqx.field(static=True)
    complex_parameter_mode: ComplexParameterMode = eqx.field(static=True)
    target_factory_id: str | None = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    target_kind: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: Any,
        operator: AbstractLocalQuantumOperator,
        kernel: MetropolisHastings,
        initial_configurations: Array,
        /,
        *,
        complex_parameter_mode: ComplexParameterMode = "real",
        parameter_subspace: ParameterSubspace | None = None,
        problem_id: str | None = None,
        target_factory: Callable[[Any], Any] | None = None,
        target_factory_id: str | None = None,
    ):
        if not callable(model):
            raise TypeError("model must be callable.")
        if not isinstance(operator, AbstractLocalQuantumOperator):
            raise TypeError("operator must implement AbstractLocalQuantumOperator.")
        if not isinstance(kernel, MetropolisHastings):
            raise TypeError("kernel must be a MetropolisHastings instance.")
        configs = jnp.asarray(initial_configurations)
        expected_rank = 1 + len(operator.configuration_shape)
        if (
            configs.ndim != expected_rank
            or tuple(configs.shape[1:]) != operator.configuration_shape
        ):
            raise ValueError(
                "initial_configurations must have shape (chain,) + "
                f"{operator.configuration_shape}; got {configs.shape}."
            )
        if configs.shape[0] < 1:
            raise ValueError("At least one initial chain is required.")
        exemplar = _amplitude(model, configs[0])
        if not bool(jnp.asarray(exemplar.valid & exemplar.nonzero)):
            raise ValueError(
                "The exemplar initial configuration must have nonzero amplitude."
            )
        if target_factory is None:
            if target_factory_id is not None:
                raise ValueError("target_factory_id requires target_factory.")
            factory_identity = None
            target_identity = None
            target_kind = "full"
        else:
            if not callable(target_factory):
                raise TypeError("target_factory must be callable or None.")
            if not isinstance(target_factory_id, str) or not target_factory_id.strip():
                raise ValueError(
                    "target_factory_id must be a nonempty string when target_factory is provided."
                )
            factory_identity = target_factory_id.strip()
            initial_target = _explicit_model_target(target_factory(model))
            target_identity = initial_target.target_id
            target_kind = _target_kind(initial_target)
        subspace = (
            _default_parameter_subspace(model)
            if parameter_subspace is None
            else parameter_subspace
        )
        if not isinstance(subspace, ParameterSubspace):
            raise TypeError("parameter_subspace must be a ParameterSubspace or None.")
        vector = subspace.pack()
        if vector.size < 1:
            raise ValueError(
                "The VMC model must expose at least one trainable parameter."
            )
        mode = _parameter_mode(complex_parameter_mode)
        coordinates = _coordinates_from_vector(vector, mode)
        identity = {
            "kind": "variational-monte-carlo",
            "operator": operator.operator_id,
            "kernel": kernel.kernel_id,
            "proposal": kernel.proposal.proposal_id,
            "parameter_paths": list(subspace.leaf_paths),
            "complex_parameter_mode": mode,
        }
        if factory_identity is not None:
            identity.update(
                {
                    "target_factory": factory_identity,
                    "target": target_identity,
                    "target_kind": target_kind,
                }
            )
        identifier = (
            canonical_fingerprint(identity) if problem_id is None else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.model = model
        self.operator = operator
        self.kernel = kernel
        self.initial_configurations = configs
        self.parameter_subspace = subspace
        self.initial_parameter_vector = vector
        self.initial_coordinates = coordinates
        self.target_factory = target_factory
        self.complex_parameter_mode = mode
        self.target_factory_id = factory_identity
        self.target_id = (
            f"{identifier}:amplitude-squared"
            if target_identity is None
            else target_identity
        )
        self.target_kind = target_kind
        self.problem_id = identifier

    def model_from_coordinates(self, coordinates: Array, /) -> Any:
        vector = _vector_from_coordinates(
            coordinates,
            self.initial_parameter_vector,
            self.complex_parameter_mode,
        )
        return self.parameter_subspace.reconstruct_vector(vector)

    def target_for_model(self, model: Any, /):
        """Bind the problem's declared sampling target to one frozen model."""
        if self.target_factory is None:
            return FullMarkovTarget(
                _model_log_target(model),
                target_id=self.target_id,
            )
        target = _explicit_model_target(self.target_factory(model))
        if target.target_id != self.target_id or _target_kind(target) != self.target_kind:
            raise ValueError(
                "target_factory must preserve its target identity and target kind."
            )
        return target

    def initial_state(
        self, *, key: Key[Array, ""] = jr.key(0)
    ) -> VariationalMonteCarloState:
        markov = self.kernel.initialize(
            self.target_for_model(self.model),
            self.initial_configurations,
        )
        return VariationalMonteCarloState(
            model=self.model,
            parameter_coordinates=self.initial_coordinates,
            markov_state=markov,
            iteration=0,
            root_key=key,
        )


class VariationalMonteCarloPolicy(StrictModule):
    """Complete fixed-sampler stochastic-reconfiguration training policy."""

    num_iterations: int = eqx.field(static=True)
    draws_per_iteration: int = eqx.field(static=True)
    steps_per_draw: int = eqx.field(static=True)
    warmup_steps: int = eqx.field(static=True)
    final_evaluation_draws: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    max_update_norm: float | None = eqx.field(static=True)
    energy_imag_tolerance: float = eqx.field(static=True)
    failure_mode: FailureMode = eqx.field(static=True)
    final_chain_diagnostics: bool = eqx.field(static=True)
    linear_policy: LinearSolvePolicy | None
    nullspace_policy: NullspacePolicy | None

    def __init__(
        self,
        *,
        num_iterations: int,
        draws_per_iteration: int,
        steps_per_draw: int = 1,
        warmup_steps: int = 0,
        final_evaluation_draws: int | None = None,
        learning_rate: float = 0.05,
        damping: float = 1e-3,
        max_update_norm: float | None = None,
        energy_imag_tolerance: float = 1e-8,
        failure_mode: FailureMode = "raise",
        final_chain_diagnostics: bool = True,
        linear_policy: LinearSolvePolicy | None = None,
        nullspace_policy: NullspacePolicy | None = None,
    ):
        iterations = int(num_iterations)
        draws = int(draws_per_iteration)
        transitions = int(steps_per_draw)
        warmup = int(warmup_steps)
        final_draws = (
            draws if final_evaluation_draws is None else int(final_evaluation_draws)
        )
        if iterations < 0:
            raise ValueError("num_iterations must be non-negative.")
        if draws <= 0 or final_draws <= 0:
            raise ValueError("draw counts must be positive.")
        if transitions <= 0 or warmup < 0:
            raise ValueError(
                "steps_per_draw must be positive and warmup_steps non-negative."
            )
        learning_rate_ = float(learning_rate)
        damping_ = float(damping)
        tolerance = float(energy_imag_tolerance)
        if not isfinite(learning_rate_) or learning_rate_ <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        if not isfinite(damping_) or damping_ < 0.0:
            raise ValueError("damping must be finite and non-negative.")
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("energy_imag_tolerance must be finite and non-negative.")
        if max_update_norm is None:
            update_limit = None
        else:
            update_limit = float(max_update_norm)
            if not isfinite(update_limit) or update_limit <= 0.0:
                raise ValueError("max_update_norm must be finite and positive.")
        if failure_mode not in ("raise", "record"):
            raise ValueError("failure_mode must be 'raise' or 'record'.")
        if linear_policy is not None and not isinstance(linear_policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
        if nullspace_policy is not None and not isinstance(
            nullspace_policy, NullspacePolicy
        ):
            raise TypeError("nullspace_policy must be a NullspacePolicy or None.")
        self.num_iterations = iterations
        self.draws_per_iteration = draws
        self.steps_per_draw = transitions
        self.warmup_steps = warmup
        self.final_evaluation_draws = final_draws
        self.learning_rate = learning_rate_
        self.damping = damping_
        self.max_update_norm = update_limit
        self.energy_imag_tolerance = tolerance
        self.failure_mode = failure_mode
        self.final_chain_diagnostics = bool(final_chain_diagnostics)
        self.linear_policy = linear_policy
        self.nullspace_policy = nullspace_policy


class VariationalMonteCarloState(StrictModule):
    """Restartable model coordinates and persistent Markov state.

    `iteration` counts accepted parameter updates and `attempt_cursor` counts
    training attempts (accepted or rejected); together with `root_key` they
    address every training sample key, so a continuation never repeats one.
    Omitting `attempt_cursor` declares a state without rejected attempts.
    """

    model: Any
    parameter_coordinates: Array
    markov_state: MarkovState
    iteration: Array
    attempt_cursor: Array
    root_key: Array

    def __init__(
        self,
        *,
        model: Any,
        parameter_coordinates: Array,
        markov_state: MarkovState,
        iteration: int | Array,
        root_key: Key[Array, ""],
        attempt_cursor: int | Array | None = None,
    ):
        if not isinstance(markov_state, MarkovState):
            raise TypeError("markov_state must be a MarkovState.")
        iteration_ = jnp.asarray(iteration, dtype=jnp.int32)
        if iteration_.shape != ():
            raise ValueError("iteration must be scalar.")
        attempt_ = (
            iteration_
            if attempt_cursor is None
            else jnp.asarray(attempt_cursor, dtype=jnp.int32)
        )
        if attempt_.shape != ():
            raise ValueError("attempt_cursor must be scalar.")
        if jnp.asarray(root_key).shape != ():
            raise ValueError("root_key must be one scalar PRNG key.")
        self.model = model
        self.parameter_coordinates = jnp.asarray(parameter_coordinates)
        self.markov_state = markov_state
        self.iteration = iteration_
        self.attempt_cursor = attempt_
        self.root_key = root_key


class VariationalMonteCarloEstimate(StrictModule):
    """Energy, variance, sampler, and validity evidence from one frozen model."""

    energy: Array
    physical_energy: Array
    imaginary_energy: Array
    variance: Array
    acceptance_rate: Array
    active_samples: Array
    valid: Array
    status: Array
    local: LocalOperatorEstimate
    chain_diagnostics: MCMCDiagnostics | None

    @property
    def successful(self) -> Array:
        return self.status == VMC_SUCCESS


class VariationalMonteCarloResult(StrictModule):
    """Final model plus complete training and frozen-evaluation histories."""

    final_state: VariationalMonteCarloState
    final_estimate: VariationalMonteCarloEstimate
    energy_history: Array
    variance_history: Array
    acceptance_history: Array
    update_norm_history: Array
    status_history: Array
    linear_results: tuple[LinearSolveResult, ...]
    root_key: Array
    problem_id: str = eqx.field(static=True)
    completed_iterations: int = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.final_estimate.successful & jnp.all(
            self.status_history == VMC_SUCCESS
        )


def _extract_scalar(value: Any, /) -> Array:
    if isinstance(value, cx.AxisArray):
        return jnp.asarray(value.data).reshape(())
    return jnp.asarray(value).reshape(())


def _estimate_from_samples(
    problem: VariationalMonteCarloProblem,
    model: Any,
    samples: MarkovSampleResult,
    /,
    *,
    energy_imag_tolerance: float,
    compute_chain_diagnostics: bool,
) -> VariationalMonteCarloEstimate:
    configurations = jnp.asarray(samples.samples)
    local = evaluate_local_operator(model, problem.operator, configurations)
    target = markov_chain_measure(samples)
    energy_result = integrate(local.value, target)
    energy = _extract_scalar(energy_result.value)
    centered = local.value - energy
    variance_result = integrate(jnp.abs(centered) ** 2, target)
    variance = jnp.real(_extract_scalar(variance_result.value))
    all_local_valid = jnp.all(local.successful)
    all_chain_valid = jnp.all(samples.final_state.valid)
    finite = jnp.isfinite(energy) & jnp.isfinite(variance)
    imaginary = jnp.abs(jnp.imag(energy))
    status = jnp.where(
        ~(all_local_valid & all_chain_valid),
        VMC_INVALID_SAMPLES,
        jnp.where(
            ~finite,
            VMC_NONFINITE,
            jnp.where(
                imaginary > energy_imag_tolerance,
                VMC_IMAGINARY_ENERGY,
                VMC_SUCCESS,
            ),
        ),
    ).astype(jnp.int32)
    valid = status == VMC_SUCCESS
    chain_diagnostics = None
    if compute_chain_diagnostics:
        from ..uq._diagnostics import mcmc_diagnostics

        chain_diagnostics = mcmc_diagnostics(
            {
                "configuration": configurations.astype("float64"),
                "local_energy_real": jnp.real(local.value),
                "local_energy_imag": jnp.imag(local.value),
            },
            acceptance_rate=samples.acceptance_rate,
            divergent=jnp.zeros(samples.log_target.shape, dtype=jnp.bool_),
        )
    return VariationalMonteCarloEstimate(
        energy=energy,
        physical_energy=jnp.where(valid, jnp.real(energy), jnp.nan),
        imaginary_energy=imaginary,
        variance=variance,
        acceptance_rate=jnp.mean(samples.acceptance_rate),
        active_samples=jnp.sum(local.successful, dtype=jnp.int32),
        valid=valid,
        status=status,
        local=local,
        chain_diagnostics=chain_diagnostics,
    )


def _sample_frozen_model(
    problem: VariationalMonteCarloProblem,
    model: Any,
    markov_state: MarkovState,
    /,
    *,
    key: Key[Array, ""],
    num_draws: int,
    steps_per_draw: int,
    warmup_steps: int,
) -> MarkovSampleResult:
    target = problem.target_for_model(model)
    rebound = problem.kernel.rebind(target, markov_state)
    return sample_markov(
        target,
        problem.kernel,
        rebound,
        key=key,
        num_draws=num_draws,
        steps_per_draw=steps_per_draw,
        warmup_steps=warmup_steps,
    )


def evaluate_variational_monte_carlo(
    problem: VariationalMonteCarloProblem,
    model: Any,
    markov_state: MarkovState,
    /,
    *,
    key: Key[Array, ""],
    num_draws: int,
    steps_per_draw: int = 1,
    warmup_steps: int = 0,
    energy_imag_tolerance: float = 1e-8,
    compute_chain_diagnostics: bool = False,
) -> tuple[VariationalMonteCarloEstimate, MarkovSampleResult]:
    """Sample and evaluate one fixed amplitude model without updating parameters."""
    if not isinstance(problem, VariationalMonteCarloProblem):
        raise TypeError("problem must be a VariationalMonteCarloProblem.")
    samples = _sample_frozen_model(
        problem,
        model,
        markov_state,
        key=key,
        num_draws=num_draws,
        steps_per_draw=steps_per_draw,
        warmup_steps=warmup_steps,
    )
    estimate = _estimate_from_samples(
        problem,
        model,
        samples,
        energy_imag_tolerance=energy_imag_tolerance,
        compute_chain_diagnostics=bool(compute_chain_diagnostics),
    )
    return estimate, samples


def _score_geometry(
    problem: VariationalMonteCarloProblem,
    coordinates: Array,
    configurations: Array,
    /,
    *,
    damping: float,
) -> tuple[JacobianLinearOperator, EmpiricalGramLinearOperator]:
    shape = problem.operator.configuration_shape
    flat = jnp.asarray(configurations).reshape((-1,) + shape)
    mode = problem.complex_parameter_mode

    def features(parameter_coordinates):
        model = problem.model_from_coordinates(parameter_coordinates)
        amplitudes = jax.vmap(model)(flat)
        if not isinstance(amplitudes, LogAmplitude):
            raise TypeError("The VMC model must return LogAmplitude.")
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
        linearization_id=f"vmc-score:{problem.problem_id}",
    )
    score = JacobianLinearOperator(
        linearization,
        operator_id=f"vmc-score:{problem.problem_id}",
    )
    metric = EmpiricalGramLinearOperator(
        score,
        jnp.ones((flat.shape[0],), dtype=jnp.float64),
        centered=True,
        damping=damping,
        operator_id=f"vmc-metric:{problem.problem_id}",
    )
    return score, metric


def _energy_force(
    score: JacobianLinearOperator,
    local_energy: Array,
    mean_energy: Array,
    mode: ComplexParameterMode,
    /,
) -> Array:
    residual = jnp.asarray(local_energy).reshape((-1,)) - mean_energy
    count = residual.shape[0]
    if mode == "holomorphic":
        cotangent = residual / count
    else:
        cotangent = (2.0 / count) * jnp.stack(
            (jnp.real(residual), jnp.imag(residual)), axis=-1
        )
    return jnp.asarray(score.adjoint_mv(cotangent))


# Training-kernel lowering ----------------------------------------------------------


@final
class _FrozenVMCRun(StrictModule, ExplicitFreeze):
    """Problem and policy of one VMC run, FIXED in the kernel tree."""

    problem: Any
    policy: VariationalMonteCarloPolicy
    ritz_tolerance: float | None = eqx.field(static=True, default=None)


@final
class _VMCCoordinateTree(StrictModule):
    """Kernel tree training the VMC parameter coordinates (array or tuple)."""

    coordinates: Any = parameter_field()
    run: _FrozenVMCRun


@final
class _SRStep(StrictModule):
    """Stochastic-reconfiguration direction formed at the current parameters.

    `direction` is congruent with the kernel's PARAMETER lane; `successful`
    holds when the estimate is valid and every metric solve succeeded with a
    finite direction.
    """

    direction: Any
    successful: Array


def _clipped_direction(
    direction: Any, max_update_norm: float | None, /
) -> tuple[Any, Array]:
    """Scale a direction onto the update-norm ball; return it and its norm."""
    norm = tree_norm(direction)
    if max_update_norm is None:
        return direction, norm
    scale = jnp.minimum(1.0, max_update_norm / jnp.maximum(norm, 1e-30))
    scaled = jax.tree.map(lambda value: (scale * value).astype(value.dtype), direction)
    return scaled, scale * norm


@final
class _StochasticReconfigurationRule(AbstractKernelUpdateRule):
    """Fixed-rate SR step along the objective's sample-formed metric solve.

    The VMC objectives form the SR direction from their frozen Monte Carlo
    samples (score geometry and metric solve) and emit it as `_SRStep`
    diagnostics, so the kernel skips autodiff (`forms_own_derivatives`). The rule
    clips the direction to `max_update_norm`, steps by `learning_rate`, and
    accepts exactly when the step is successful. The rule is stateless: a finite
    rejection commits nothing.
    """

    rejection_commit_policy: ClassVar[tuple[str, ...]] = ()
    learning_rate: float = eqx.field(static=True)
    max_update_norm: float | None = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(self, policy: VariationalMonteCarloPolicy, /):
        self.learning_rate = policy.learning_rate
        self.max_update_norm = policy.max_update_norm
        self.rule_id = canonical_fingerprint(
            {
                "kind": "variational-monte-carlo-stochastic-reconfiguration",
                "learning_rate": policy.learning_rate,
                "max_update_norm": policy.max_update_norm,
            }
        )

    @property
    def forms_own_derivatives(self) -> bool:
        return True

    def init(self, parameters: Any, /) -> None:
        del parameters
        return None

    def propose(
        self,
        parameters: Any,
        gradients: Any,
        value: Array,
        rule_state: None,
        context: KernelUpdateContext,
        /,
    ) -> tuple[Any, None, None, Array]:
        del gradients, value
        step = context.diagnostics[0].step
        direction, _norm = _clipped_direction(step.direction, self.max_update_norm)
        candidate = jax.tree.map(
            lambda current, delta: current - self.learning_rate * delta,
            parameters,
            direction,
        )
        return candidate, rule_state, rule_state, step.successful


def _prepare_sr_kernel(
    coordinates: Any,
    run: _FrozenVMCRun,
    objective: Callable[..., Any],
    /,
    *,
    objective_id: str,
    root_key: Key[Array, ""],
    iteration: Array,
    attempt_cursor: Array,
) -> tuple[PreparedTrainingKernel, TrainingKernelState]:
    """Prepare the SR kernel and rebuild the continuation's kernel state.

    The SR rule is stateless and the budget is zero, so a continuation is fully
    described by its coordinates, root key, and accepted and attempt cursors.
    """
    tree = _VMCCoordinateTree(coordinates=coordinates, run=run)
    kernel = prepare_training_kernel(
        tree,
        (
            KernelObjective(
                objective_id=objective_id,
                kind=ObjectiveKind.PHYSICAL_RESIDUAL,
                route=DerivativeRoute.DIRECT,
                fn=objective,
            ),
        ),
        # Every rejected attempt stops the run (raised or recorded).
        TrainingKernelSpec(
            _StochasticReconfigurationRule(run.policy),
            context=objective_id,
            rejection_budget=0,
        ),
        root_authority=ComponentAuthority.SURROGATE,
    )
    state = dataclasses.replace(
        kernel.init(tree, root_key),
        accepted_cursor=jnp.asarray(iteration, dtype=jnp.int32),
        attempt_cursor=jnp.asarray(attempt_cursor, dtype=jnp.int32),
    )
    return kernel, state


def _run_sr_attempt(
    kernel: PreparedTrainingKernel, state: TrainingKernelState, payload: Any, /
) -> tuple[TrainingKernelState, TrainingAttemptEvidence, bool]:
    """One kernel attempt; returns the committed state, evidence, and acceptance.

    With a zero budget every rejection raises; the error carries the rolled-back
    state and the attempt evidence the frontend records.
    """
    try:
        next_state, evidence = run_training_attempt(kernel, state, payload)
    except TrainingRejectionBudgetError as error:
        return error.state, error.evidence, False
    return next_state, evidence, True


@final
class _VMCAttemptDiagnostics(StrictModule):
    estimate: VariationalMonteCarloEstimate
    linear: LinearSolveResult
    step: _SRStep


def _vmc_objective(
    parameters: Any,
    model_state: Any,
    fixed: Any,
    samples: MarkovSampleResult,
    keys: Any,
    /,
) -> tuple[_ObjectiveContribution, Any, _VMCAttemptDiagnostics]:
    """Frozen-sample energy at the parameters plus its SR direction."""
    del keys
    tree = combine_parameters(parameters, model_state, fixed)
    problem = tree.run.problem
    policy = tree.run.policy
    coordinates = tree.coordinates
    model = problem.model_from_coordinates(coordinates)
    estimate = _estimate_from_samples(
        problem,
        model,
        samples,
        energy_imag_tolerance=policy.energy_imag_tolerance,
        compute_chain_diagnostics=False,
    )
    score, metric = _score_geometry(
        problem, coordinates, samples.samples, damping=policy.damping
    )
    force = _energy_force(
        score, estimate.local.value, estimate.energy, problem.complex_parameter_mode
    )
    linear = solve(
        LinearSystem(metric, nullspace_policy=policy.nullspace_policy),
        force,
        policy=policy.linear_policy,
    )
    direction = jnp.asarray(linear.value)
    step = _SRStep(
        eqx.tree_at(lambda lane: lane.coordinates, parameters, direction),
        estimate.successful
        & jnp.all(linear.successful)
        & jnp.all(jnp.isfinite(direction)),
    )
    value = jnp.real(estimate.energy)
    contribution = _ObjectiveContribution(
        value, jnp.ones((), value.dtype), jnp.zeros((), value.dtype)
    )
    return contribution, model_state, _VMCAttemptDiagnostics(estimate, linear, step)


def _raise_vmc(status: Array, role: str, /) -> None:
    raise RuntimeError(f"{role} failed with VMC status {vmc_status_name(status)}.")


def _model_static_payload(value: Any, path: str = "model", /) -> Any:
    if eqx.is_array(value):
        array = jnp.asarray(value)
        return {
            "kind": "array-leaf",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} contains a nonfinite static value.")
        return value
    if isinstance(value, complex):
        if not isfinite(value.real) or not isfinite(value.imag):
            raise ValueError(f"{path} contains a nonfinite static value.")
        return {"kind": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, np.dtype):
        return {"kind": "dtype", "value": value.str}
    if isinstance(value, type):
        return {
            "kind": "type",
            "module": value.__module__,
            "qualname": value.__qualname__,
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} static mapping keys must be strings.")
        return {
            key: _model_static_payload(value[key], f"{path}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, (tuple, list)):
        return [
            _model_static_payload(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": "model-node",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _model_static_payload(
                    getattr(value, field.name), f"{path}.{field.name}"
                )
                for field in dataclasses.fields(value)
            },
        }
    if callable(value):
        return {
            "kind": "callable",
            "semantic_content_id": callable_payload(value)["semantic_content_id"],
        }
    raise TypeError(
        f"{path} contains unsupported static checkpoint identity type "
        f"{type(value).__name__}."
    )


def _model_structure_id(model: Any, /) -> str:
    return canonical_fingerprint(_model_static_payload(model))


def _policy_checkpoint_compatibility(
    policy: VariationalMonteCarloPolicy, /
) -> dict[str, Any]:
    return {
        "draws_per_iteration": policy.draws_per_iteration,
        "steps_per_draw": policy.steps_per_draw,
        "warmup_steps": policy.warmup_steps,
        "learning_rate": policy.learning_rate,
        "damping": policy.damping,
        "max_update_norm": policy.max_update_norm,
        "energy_imag_tolerance": policy.energy_imag_tolerance,
        "failure_mode": policy.failure_mode,
        "linear_policy": repr(policy.linear_policy),
        "nullspace_policy": repr(policy.nullspace_policy),
    }


def _checkpoint_compatibility(
    problem: VariationalMonteCarloProblem,
    policy: VariationalMonteCarloPolicy,
    /,
) -> dict[str, Any]:
    return {
        "problem_id": problem.problem_id,
        "operator_id": problem.operator.operator_id,
        "kernel_id": problem.kernel.kernel_id,
        "proposal_id": problem.kernel.proposal.proposal_id,
        "complex_parameter_mode": problem.complex_parameter_mode,
        "target_factory_id": problem.target_factory_id,
        "target_id": problem.target_id,
        "target_kind": problem.target_kind,
        "configuration_shape": list(problem.operator.configuration_shape),
        "initial_configuration_signature": array_tree_signature(
            problem.initial_configurations
        ),
        "model_signature": array_tree_signature(problem.model),
        "model_structure_id": _model_structure_id(problem.model),
        "parameter_paths": list(problem.parameter_subspace.leaf_paths),
        "policy_fingerprint": canonical_fingerprint(
            _policy_checkpoint_compatibility(policy)
        ),
    }


def _validate_state_compatibility(
    problem: VariationalMonteCarloProblem,
    state: VariationalMonteCarloState,
    /,
) -> None:
    expected_coordinates = problem.initial_coordinates
    coordinates = jnp.asarray(state.parameter_coordinates)
    if (
        coordinates.shape != expected_coordinates.shape
        or coordinates.dtype != expected_coordinates.dtype
    ):
        raise ValueError("VMC state parameter coordinates are incompatible.")
    if state.markov_state.num_chains != problem.initial_configurations.shape[0]:
        raise ValueError("VMC state chain count is incompatible with the problem.")
    if state.markov_state.target_id != problem.target_id:
        raise ValueError("VMC state target identity is incompatible with the problem.")
    if _model_structure_id(state.model) != _model_structure_id(problem.model):
        raise ValueError("VMC state model static structure is incompatible.")
    if int(state.iteration) < 0:
        raise ValueError("VMC state iteration must be non-negative.")
    if int(state.attempt_cursor) < int(state.iteration):
        raise ValueError("VMC state attempt cursor must not precede its iteration.")


def _validate_model_coordinates(
    problem: VariationalMonteCarloProblem,
    state: VariationalMonteCarloState,
    /,
) -> None:
    reconstructed = problem.model_from_coordinates(state.parameter_coordinates)
    state_leaves = jax.tree_util.tree_leaves(state.model)
    reconstructed_leaves = jax.tree_util.tree_leaves(reconstructed)
    if len(state_leaves) != len(reconstructed_leaves) or any(
        not bool(jnp.array_equal(jnp.asarray(actual), jnp.asarray(expected)))
        for actual, expected in zip(state_leaves, reconstructed_leaves, strict=True)
    ):
        raise ValueError("VMC state model and parameter coordinates are inconsistent.")


def write_variational_monte_carlo_checkpoint(
    path: str | os.PathLike[str],
    problem: VariationalMonteCarloProblem,
    policy: VariationalMonteCarloPolicy,
    state: VariationalMonteCarloState,
    /,
) -> Path:
    """Atomically write one pickle-free, compatibility-checked VMC state archive."""
    if not isinstance(problem, VariationalMonteCarloProblem):
        raise TypeError("problem must be a VariationalMonteCarloProblem.")
    if not isinstance(policy, VariationalMonteCarloPolicy):
        raise TypeError("policy must be a VariationalMonteCarloPolicy.")
    if not isinstance(state, VariationalMonteCarloState):
        raise TypeError("state must be a VariationalMonteCarloState.")
    _validate_state_compatibility(problem, state)
    _validate_model_coordinates(problem, state)
    from ..uq._checkpoint import pack_array_tree, write_checkpoint_archive

    arrays: dict[str, Any] = {
        "parameter_coordinates": state.parameter_coordinates,
        "markov_step_index": state.markov_state.step_index,
        "root_key_data": jr.key_data(state.root_key),
        "markov_valid": state.markov_state.valid,
    }
    checkpoint_state = {
        "iteration": int(state.iteration),
        "attempt_cursor": int(state.attempt_cursor),
        "model_tree": pack_array_tree("model", state.model, arrays),
        "position_tree": pack_array_tree(
            "markov_position", state.markov_state.position, arrays
        ),
        "parameter_coordinates_array": "parameter_coordinates",
        "markov_step_index_array": "markov_step_index",
        "root_key_data_array": "root_key_data",
        "markov_valid_array": "markov_valid",
    }
    return write_checkpoint_archive(
        path,
        kind=_VMC_CHECKPOINT_KIND,
        compatibility=_checkpoint_compatibility(problem, policy),
        state=checkpoint_state,
        arrays=arrays,
    )


def _checkpoint_array(
    arrays: dict[str, Array],
    name: Any,
    template: Array,
    /,
) -> Array:
    if not isinstance(name, str) or name not in arrays:
        raise ValueError("VMC checkpoint array inventory is invalid.")
    value = jnp.asarray(arrays[name])
    expected = jnp.asarray(template)
    if value.shape != expected.shape or value.dtype != expected.dtype:
        raise ValueError("VMC checkpoint array shape or dtype is incompatible.")
    return value


def read_variational_monte_carlo_checkpoint(
    path: str | os.PathLike[str],
    problem: VariationalMonteCarloProblem,
    policy: VariationalMonteCarloPolicy,
    /,
) -> VariationalMonteCarloState:
    """Restore one portable VMC continuation state against a live problem and policy."""
    if not isinstance(problem, VariationalMonteCarloProblem):
        raise TypeError("problem must be a VariationalMonteCarloProblem.")
    if not isinstance(policy, VariationalMonteCarloPolicy):
        raise TypeError("policy must be a VariationalMonteCarloPolicy.")
    from ..uq._checkpoint import read_checkpoint_archive, unpack_array_tree

    checkpoint_state, arrays = read_checkpoint_archive(
        path,
        kind=_VMC_CHECKPOINT_KIND,
        compatibility=_checkpoint_compatibility(problem, policy),
    )
    if set(checkpoint_state) != _VMC_CHECKPOINT_STATE_FIELDS:
        raise ValueError("VMC checkpoint state must use the current canonical fields.")
    iteration = checkpoint_state["iteration"]
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("VMC checkpoint iteration is invalid.")
    attempt = checkpoint_state["attempt_cursor"]
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < iteration:
        raise ValueError("VMC checkpoint attempt cursor is invalid.")
    model_spec = checkpoint_state.get("model_tree")
    position_spec = checkpoint_state.get("position_tree")
    if not isinstance(model_spec, dict) or not isinstance(position_spec, dict):
        raise ValueError("VMC checkpoint tree metadata is invalid.")
    model = unpack_array_tree(model_spec, arrays, problem.model)
    position = unpack_array_tree(position_spec, arrays, problem.initial_configurations)
    coordinates = _checkpoint_array(
        arrays,
        checkpoint_state.get("parameter_coordinates_array"),
        problem.initial_coordinates,
    )
    step_index = _checkpoint_array(
        arrays,
        checkpoint_state.get("markov_step_index_array"),
        jnp.asarray(0, dtype=jnp.uint32),
    )
    saved_valid = _checkpoint_array(
        arrays,
        checkpoint_state.get("markov_valid_array"),
        jnp.ones(
            (problem.initial_configurations.shape[0],),
            dtype=jnp.bool_,
        ),
    )
    key_data = _checkpoint_array(
        arrays,
        checkpoint_state.get("root_key_data_array"),
        jr.key_data(jr.key(0)),
    )
    rebuilt_markov = problem.kernel.initialize(
        problem.target_for_model(model),
        position,
    )
    state = VariationalMonteCarloState(
        model=model,
        parameter_coordinates=coordinates,
        markov_state=MarkovState(
            rebuilt_markov.position,
            rebuilt_markov.log_target,
            cache=rebuilt_markov.cache,
            valid=rebuilt_markov.valid & saved_valid,
            step_index=step_index,
            target_id=rebuilt_markov.target_id,
        ),
        iteration=iteration,
        attempt_cursor=attempt,
        root_key=jr.wrap_key_data(key_data),
    )
    _validate_state_compatibility(problem, state)
    _validate_model_coordinates(problem, state)
    return state


def solve_variational_monte_carlo(
    problem: VariationalMonteCarloProblem,
    policy: VariationalMonteCarloPolicy,
    /,
    *,
    key: Key[Array, ""] | None = None,
    state: VariationalMonteCarloState | None = None,
) -> VariationalMonteCarloResult:
    """Optimize a local-operator amplitude model with persistent-chain SR updates.

    Each iteration samples the persistent chains on the host at the attempt's
    semantic key and runs one training-kernel attempt: the objective estimates the
    frozen-sample energy and forms the SR direction; the stateless SR rule steps.
    An invalid estimate or a failed metric solve is a rejected attempt that
    rolls the parameters back; the chains keep their advanced state. Every
    rejection stops the run (`failure_mode="raise"` raises, `"record"` returns).
    """
    if not isinstance(problem, VariationalMonteCarloProblem):
        raise TypeError("problem must be a VariationalMonteCarloProblem.")
    if not isinstance(policy, VariationalMonteCarloPolicy):
        raise TypeError("policy must be a VariationalMonteCarloPolicy.")
    if state is None:
        if key is None:
            raise ValueError("A root key is required for a new VMC run.")
        resolved_key = key
        current = problem.initial_state(key=resolved_key)
    else:
        if not isinstance(state, VariationalMonteCarloState):
            raise TypeError("state must be a VariationalMonteCarloState or None.")
        resolved_key = state.root_key if key is None else key
        if not jnp.array_equal(jr.key_data(resolved_key), jr.key_data(state.root_key)):
            raise ValueError("Resume key does not match the VMC state root key.")
        current = state
    _validate_state_compatibility(problem, current)
    _validate_model_coordinates(problem, current)
    kernel, training = _prepare_sr_kernel(
        current.parameter_coordinates,
        _FrozenVMCRun(problem, policy),
        _vmc_objective,
        objective_id=_VMC_OBJECTIVE_ID,
        root_key=current.root_key,
        iteration=current.iteration,
        attempt_cursor=current.attempt_cursor,
    )
    energies: list[Array] = []
    variances: list[Array] = []
    acceptances: list[Array] = []
    update_norms: list[Array] = []
    statuses: list[Array] = []
    linear_results: list[LinearSolveResult] = []
    control = TrainingController(
        total_steps=int(current.iteration) + policy.num_iterations,
        algorithm_id="variational-monte-carlo-training",
        progress=TrainingProgress(update_step=int(current.iteration)),
    )
    control.emit(
        TrainingIterationKind.RUN_START,
        metrics={
            "resumed_from_step": int(current.iteration),
            "total_steps": int(current.iteration) + policy.num_iterations,
        },
    )

    for _ in range(policy.num_iterations):
        first_attempt = int(training.attempt_cursor) == 0
        samples = _sample_frozen_model(
            problem,
            current.model,
            current.markov_state,
            key=training_site_key(
                training.root_key,
                objective_id=_VMC_OBJECTIVE_ID,
                site="samples",
                attempt=training.attempt_cursor,
                microstep=training.microstep,
            ),
            num_draws=policy.draws_per_iteration,
            steps_per_draw=policy.steps_per_draw,
            warmup_steps=policy.warmup_steps if first_attempt else 0,
        )
        training, evidence, accepted = _run_sr_attempt(kernel, training, samples)
        diagnostics: _VMCAttemptDiagnostics = evidence.diagnostics[0]
        estimate = diagnostics.estimate
        energies.append(estimate.energy)
        variances.append(estimate.variance)
        acceptances.append(estimate.acceptance_rate)
        estimated = bool(estimate.successful)
        if estimated:
            linear_results.append(diagnostics.linear)
        if not accepted:
            if not estimated:
                status, role = estimate.status, "VMC estimation"
                failure_metrics = {
                    "acceptance_rate": estimate.acceptance_rate,
                    "status": estimate.status,
                }
            else:
                status = jnp.asarray(
                    VMC_NONFINITE
                    if int(evidence.outcome) == TrainingAttemptOutcome.NONFINITE
                    else VMC_LINEAR_FAILURE,
                    dtype=jnp.int32,
                )
                role = "VMC metric solve"
                failure_metrics = {"status": status}
            control.emit(TrainingIterationKind.FAILURE, metrics=failure_metrics)
            statuses.append(status)
            update_norms.append(jnp.asarray(jnp.nan))
            current = VariationalMonteCarloState(
                model=current.model,
                parameter_coordinates=current.parameter_coordinates,
                markov_state=samples.final_state,
                iteration=training.accepted_cursor,
                attempt_cursor=training.attempt_cursor,
                root_key=training.root_key,
            )
            if policy.failure_mode == "raise":
                _raise_vmc(status, role)
            break

        coordinates = training.parameters.coordinates
        _direction, norm = _clipped_direction(
            diagnostics.step.direction, policy.max_update_norm
        )
        statuses.append(jnp.asarray(VMC_SUCCESS, dtype=jnp.int32))
        update_norms.append(norm)
        current = VariationalMonteCarloState(
            model=problem.model_from_coordinates(coordinates),
            parameter_coordinates=coordinates,
            markov_state=samples.final_state,
            iteration=training.accepted_cursor,
            attempt_cursor=training.attempt_cursor,
            root_key=training.root_key,
        )

        control.complete_update(int(training.accepted_cursor))
        control.emit(
            TrainingIterationKind.UPDATE,
            metrics={
                "acceptance_rate": estimate.acceptance_rate,
                "energy": estimate.physical_energy,
                "update_norm": norm,
                "variance": estimate.variance,
            },
        )
    control.emit(
        TrainingIterationKind.RUN_TERMINAL,
        metrics={"completed_steps": control.progress.update_step},
    )
    final_estimate, _final_samples = evaluate_variational_monte_carlo(
        problem,
        current.model,
        current.markov_state,
        key=training_site_key(
            training.root_key,
            objective_id=_VMC_OBJECTIVE_ID,
            site="final-estimate",
            attempt=training.attempt_cursor,
            microstep=training.microstep,
        ),
        num_draws=policy.final_evaluation_draws,
        steps_per_draw=policy.steps_per_draw,
        energy_imag_tolerance=policy.energy_imag_tolerance,
        compute_chain_diagnostics=policy.final_chain_diagnostics,
    )
    return VariationalMonteCarloResult(
        final_state=current,
        final_estimate=final_estimate,
        energy_history=jnp.stack(energies)
        if energies
        else jnp.empty((0,), dtype=jnp.complex128),
        variance_history=jnp.stack(variances) if variances else jnp.empty((0,)),
        acceptance_history=jnp.stack(acceptances) if acceptances else jnp.empty((0,)),
        update_norm_history=jnp.stack(update_norms) if update_norms else jnp.empty((0,)),
        status_history=jnp.stack(statuses)
        if statuses
        else jnp.empty((0,), dtype=jnp.int32),
        linear_results=tuple(linear_results),
        root_key=current.root_key,
        problem_id=problem.problem_id,
        completed_iterations=int(current.iteration),
    )


__all__ = [
    "VMC_IMAGINARY_ENERGY",
    "VMC_INVALID_SAMPLES",
    "VMC_LINEAR_FAILURE",
    "VMC_NONFINITE",
    "VMC_SUCCESS",
    "VMCStatus",
    "VariationalMonteCarloEstimate",
    "VariationalMonteCarloPolicy",
    "VariationalMonteCarloProblem",
    "VariationalMonteCarloResult",
    "VariationalMonteCarloState",
    "evaluate_variational_monte_carlo",
    "read_variational_monte_carlo_checkpoint",
    "solve_variational_monte_carlo",
    "write_variational_monte_carlo_checkpoint",
    "vmc_status_name",
]
