#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pathwise evidence for the open-loop stochastic maximum principle."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from math import isfinite
from typing import Any, Literal, Protocol, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ...dynamics import DiscreteStepContext, TimeGrid
from ._evaluation import ControlledPathBatch


_SAMPLE_ROLES = ("training", "holdout")
_METHOD_ID = "pathwise-euler-open-loop-stochastic-maximum-principle-v1"
_CERTIFICATE = "OPEN_LOOP_SMP_STATIONARY"

SampleRole: TypeAlias = Literal["training", "holdout"]


class SMPStageVector(Protocol):
    """A state-shaped stage callback evaluated on supplied physical actions."""

    def __call__(
        self,
        context: DiscreteStepContext,
        state: Array,
        action: Array,
        args: Any,
        /,
    ) -> ArrayLike: ...


class SMPStageMatrix(Protocol):
    """A stage Jacobian callback with its declared tensor shape."""

    def __call__(
        self,
        context: DiscreteStepContext,
        state: Array,
        action: Array,
        args: Any,
        /,
    ) -> ArrayLike: ...


class SMPTerminalGradient(Protocol):
    """A terminal-cost gradient callback."""

    def __call__(
        self,
        terminal_time: Array,
        terminal_state: Array,
        args: Any,
        /,
    ) -> ArrayLike: ...


class SMPAdjointPredictor(Protocol):
    """An adjoint predictor receiving only current time and state."""

    def __call__(self, time: Array, state: Array, args: Any, /) -> ArrayLike: ...


class SMPMartingaleIntegrandPredictor(Protocol):
    """A BSDE martingale-integrand predictor, distinct from physical action."""

    def __call__(
        self,
        context: DiscreteStepContext,
        state: Array,
        action: Array,
        args: Any,
        /,
    ) -> ArrayLike: ...


AdjointPrediction: TypeAlias = SMPAdjointPredictor | ArrayLike
MartingaleIntegrandPrediction: TypeAlias = SMPMartingaleIntegrandPredictor | ArrayLike


class StochasticMaximumPrincipleStatus(IntEnum):
    """Stable path-local status codes for stochastic maximum-principle evidence."""

    SUCCESS = 0
    INVALID_FORWARD_PATH = 1
    NONFINITE_ADJOINT = 2
    NONFINITE_MARTINGALE_INTEGRAND = 3
    NONFINITE_DYNAMICS = 4
    NONFINITE_DERIVATIVE = 5
    NONFINITE_TERMINAL_GRADIENT = 6
    NONCAUSAL_INFORMATION = 7
    NO_VALID_PATHS = 8


class StochasticMaximumPrincipleProblem(StrictModule):
    """Continuous-time SMP ingredients evaluated on a supplied Euler path batch.

    The controlled SDE is ``dX = b(t, X, a) dt + sigma(t, X, a) dW`` and the
    minimization Hamiltonian is
    ``H = running_cost + p·b + q:sigma``. Callers explicitly supply derivatives
    of ``b``, ``sigma``, and the running-cost gradients; no automatic
    differentiation or numerical fallback is used. Jacobian shapes are
    ``b_x=(n,n)``, ``b_a=(n,m)``, ``sigma_x=(n,w,n)``, and
    ``sigma_a=(n,w,m)``. In particular, the evaluated action gradient contains
    the complete ``q:sigma_a`` term.
    """

    time_grid: TimeGrid
    drift: SMPStageVector
    diffusion: SMPStageMatrix
    drift_state_jacobian: SMPStageMatrix
    drift_action_jacobian: SMPStageMatrix
    diffusion_state_jacobian: SMPStageMatrix
    diffusion_action_jacobian: SMPStageMatrix
    running_cost_state_gradient: SMPStageVector
    running_cost_action_gradient: SMPStageVector
    terminal_cost_gradient: SMPTerminalGradient
    args: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    action_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    action_size: int = eqx.field(static=True)
    noise_size: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_grid: TimeGrid,
        drift: SMPStageVector,
        diffusion: SMPStageMatrix,
        drift_state_jacobian: SMPStageMatrix,
        drift_action_jacobian: SMPStageMatrix,
        diffusion_state_jacobian: SMPStageMatrix,
        diffusion_action_jacobian: SMPStageMatrix,
        running_cost_state_gradient: SMPStageVector,
        running_cost_action_gradient: SMPStageVector,
        terminal_cost_gradient: SMPTerminalGradient,
        /,
        *,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        noise_shape: Sequence[int],
        args: Any = None,
        problem_id: str,
    ):
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        callbacks = (
            drift,
            diffusion,
            drift_state_jacobian,
            drift_action_jacobian,
            diffusion_state_jacobian,
            diffusion_action_jacobian,
            running_cost_state_gradient,
            running_cost_action_gradient,
            terminal_cost_gradient,
        )
        if any(not callable(callback) for callback in callbacks):
            raise TypeError(
                "Every stochastic maximum-principle callback must be callable."
            )
        states = _vector_shape(state_shape, "state_shape")
        actions = _vector_shape(action_shape, "action_shape")
        noises = _vector_shape(noise_shape, "noise_shape")
        self.time_grid = time_grid
        self.drift = drift
        self.diffusion = diffusion
        self.drift_state_jacobian = drift_state_jacobian
        self.drift_action_jacobian = drift_action_jacobian
        self.diffusion_state_jacobian = diffusion_state_jacobian
        self.diffusion_action_jacobian = diffusion_action_jacobian
        self.running_cost_state_gradient = running_cost_state_gradient
        self.running_cost_action_gradient = running_cost_action_gradient
        self.terminal_cost_gradient = terminal_cost_gradient
        self.args = args
        self.state_shape = states
        self.action_shape = actions
        self.noise_shape = noises
        self.state_size = states[0]
        self.action_size = actions[0]
        self.noise_size = noises[0]
        self.problem_id = _identifier(problem_id, "problem_id")


class SMPCausalInformationEvidence(StrictModule):
    """Empirical adaptedness and conditioning cells for one supplied sample.

    Integer ``information_labels[path, step]`` identify paths having the same
    available information immediately before the current noise increment.
    Constancy residuals audit that actions, adjoints, and martingale integrands
    are measurable on those cells. ``externally_checked`` records the separate
    caller assertion that labels themselves were built without future noise.
    """

    information_labels: Array
    action_measurability_residuals: Array
    adjoint_measurability_residuals: Array
    martingale_integrand_measurability_residuals: Array
    conditional_cluster_counts: Array
    measurable: Array
    information_id: str = eqx.field(static=True)
    externally_checked: bool = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return jnp.asarray(self.externally_checked) & jnp.all(self.measurable)

    @property
    def causal(self) -> Array:
        return self.valid


class SMPPathClusterEvidence(StrictModule):
    """Path eligibility and independent-cluster provenance for SMP residuals."""

    path_valid: Array
    valid_path_count: Array
    independent_cluster_count: Array
    independence_labels: Array
    path_ids: tuple[str, ...] = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    sample_role: SampleRole = eqx.field(static=True)
    sample_id: str = eqx.field(static=True)

    @property
    def realization_ids(self) -> tuple[str, ...]:
        return self.path_ids


class StochasticMaximumPrincipleResult(StrictModule):
    """Auditable necessary-condition evidence for one supplied open-loop sample.

    This object does not turn pathwise controls into a feedback law. Finite-batch
    conditional residuals are empirical necessary-condition diagnostics only:
    even checked convexity metadata cannot establish population conditional
    stationarity, sufficiency, feedback optimality, or global optimality.
    """

    paths: ControlledPathBatch
    causal_information: SMPCausalInformationEvidence
    path_evidence: SMPPathClusterEvidence
    adjoint_values: Array
    martingale_integrands: Array
    drift_values: Array
    diffusion_values: Array
    hamiltonian_state_gradients: Array
    hamiltonian_action_gradients: Array
    forward_residuals: Array
    terminal_adjoint_residuals: Array
    backward_martingale_residuals: Array
    conditional_stationarity_residuals: Array
    forward_rms_norms: Array
    terminal_adjoint_rms_norms: Array
    backward_martingale_rms_norms: Array
    conditional_stationarity_rms_norms: Array
    maximum_residual_norms: Array
    status: Array
    valid: Array
    stationary: Array
    tolerance: float = eqx.field(static=True)
    certificate: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    predictor_id: str = eqx.field(static=True)
    convexity_checked: bool = eqx.field(static=True)
    convexity_evidence: str | None = eqx.field(static=True)
    sufficient: bool = eqx.field(static=True)
    population_stationarity_claim: bool = eqx.field(static=True)
    feedback_claim: bool = eqx.field(static=True)
    markov_perfect_claim: bool = eqx.field(static=True)
    global_optimality_claim: bool = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.any(self.valid) & jnp.all(self.valid)

    @property
    def label(self) -> str:
        return self.certificate

    @property
    def forward_residual(self) -> Array:
        return self.forward_residuals

    @property
    def terminal_adjoint_residual(self) -> Array:
        return self.terminal_adjoint_residuals

    @property
    def backward_martingale_residual(self) -> Array:
        return self.backward_martingale_residuals

    @property
    def conditional_stationarity(self) -> Array:
        return self.conditional_stationarity_residuals


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _vector_shape(value: Sequence[int], owner: str, /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if len(shape) != 1 or shape[0] <= 0:
        raise ValueError(f"{owner} must be one positive vector dimension.")
    return shape


def _positive_tolerance(value: float, owner: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{owner} must be finite and strictly positive.")
    return result


def _sample_role(value: str, /) -> SampleRole:
    if value not in _SAMPLE_ROLES:
        raise ValueError("sample_role must be 'training' or 'holdout'.")
    return value  # type: ignore[return-value]


def _real_array(value: ArrayLike, owner: str, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.number) or jnp.issubdtype(
        array.dtype, jnp.complexfloating
    ):
        raise TypeError(f"{owner} must be a real numeric array.")
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _prediction_values(
    problem: StochasticMaximumPrincipleProblem,
    paths: ControlledPathBatch,
    adjoint_prediction: AdjointPrediction,
    martingale_integrand_prediction: MartingaleIntegrandPrediction,
    /,
) -> tuple[Array, Array]:
    count = paths.path_count
    steps = problem.time_grid.num_steps
    safe_states = jnp.where(jnp.isfinite(paths.states), paths.states, 0.0)
    safe_actions = jnp.where(jnp.isfinite(paths.actions), paths.actions, 0.0)

    if callable(adjoint_prediction):
        adjoints = []
        for node in range(steps + 1):
            values = jax.vmap(
                lambda state: jnp.asarray(
                    adjoint_prediction(problem.time_grid.times[node], state, problem.args)
                )
            )(safe_states[:, node])
            adjoints.append(values)
        adjoint = _real_array(jnp.stack(adjoints, axis=1), "adjoint predictor")
    else:
        adjoint = _real_array(adjoint_prediction, "adjoint predictions")
    expected_adjoint = (count, steps + 1, problem.state_size)
    if tuple(adjoint.shape) != expected_adjoint:
        raise ValueError(f"adjoint predictions must have shape {expected_adjoint}.")

    if callable(martingale_integrand_prediction):
        integrands = []
        for step in range(steps):
            context = DiscreteStepContext(
                problem.time_grid.times[step],
                problem.time_grid.times[step + 1],
                jnp.asarray(step, dtype=jnp.int32),
            )
            values = jax.vmap(
                lambda state, action: jnp.asarray(
                    martingale_integrand_prediction(context, state, action, problem.args)
                )
            )(safe_states[:, step], safe_actions[:, step])
            integrands.append(values)
        integrand = _real_array(
            jnp.stack(integrands, axis=1), "martingale-integrand predictor"
        )
    else:
        integrand = _real_array(
            martingale_integrand_prediction, "martingale-integrand predictions"
        )
    expected_integrand = (
        count,
        steps,
        problem.state_size,
        problem.noise_size,
    )
    if tuple(integrand.shape) != expected_integrand:
        raise ValueError(
            "martingale-integrand predictions must have shape "
            f"{expected_integrand}; physical actions are not BSDE integrands."
        )
    return adjoint, integrand


def _batched_stage_callback(
    callback: Callable,
    context: DiscreteStepContext,
    states: Array,
    actions: Array,
    args: Any,
    expected_shape: tuple[int, ...],
    owner: str,
    /,
) -> Array:
    value = jax.vmap(
        lambda state, action: jnp.asarray(callback(context, state, action, args))
    )(states, actions)
    result = _real_array(value, owner)
    expected = (int(states.shape[0]),) + expected_shape
    if tuple(result.shape) != expected:
        raise ValueError(
            f"{owner} must return {expected_shape} per path; got {result.shape}."
        )
    return result


def _terminal_gradients(
    problem: StochasticMaximumPrincipleProblem,
    states: Array,
    /,
) -> Array:
    values = jax.vmap(
        lambda state: jnp.asarray(
            problem.terminal_cost_gradient(
                problem.time_grid.times[-1], state, problem.args
            )
        )
    )(states)
    result = _real_array(values, "terminal_cost_gradient")
    expected = (int(states.shape[0]), problem.state_size)
    if tuple(result.shape) != expected:
        raise ValueError(f"terminal_cost_gradient must return {problem.state_shape}.")
    return result


def _labels(value: ArrayLike, count: int, steps: int, /) -> Array:
    labels = jnp.asarray(value)
    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise TypeError("information_labels must have an integer dtype.")
    if tuple(labels.shape) != (count, steps):
        raise ValueError(f"information_labels must have shape {(count, steps)}.")
    host = np.asarray(labels)
    if np.any(host < 0) or np.any(host > np.iinfo(np.int32).max):
        raise ValueError("information_labels must be nonnegative int32 values.")
    return labels.astype(jnp.int32)


def _cell_measurability_residuals(
    values: Array,
    labels: Array,
    eligible: Array,
    /,
) -> Array:
    host_values = np.asarray(jax.device_get(values))
    host_labels = np.asarray(jax.device_get(labels))
    host_eligible = np.asarray(jax.device_get(eligible), dtype=bool)
    count, steps = host_labels.shape
    residual = np.full((count, steps), np.inf, dtype=host_values.dtype)
    for step in range(steps):
        for label in np.unique(host_labels[host_eligible, step]):
            members = np.flatnonzero(host_eligible & (host_labels[:, step] == label))
            cell = host_values[members, step].reshape((len(members), -1))
            center = np.mean(cell, axis=0)
            deviation = float(np.max(np.abs(cell - center)))
            residual[members, step] = deviation
    return jnp.asarray(residual)


def _conditional_cluster_means(
    values: Array,
    labels: Array,
    independence_labels: Array,
    eligible: Array,
    /,
) -> tuple[Array, Array]:
    host_values = np.asarray(jax.device_get(values))
    host_labels = np.asarray(jax.device_get(labels))
    host_clusters = np.asarray(jax.device_get(independence_labels))
    host_eligible = np.asarray(jax.device_get(eligible), dtype=bool)
    count, steps = host_labels.shape
    result = np.full_like(host_values, np.nan)
    cluster_counts = np.zeros((count, steps), dtype=np.int32)
    for step in range(steps):
        for label in np.unique(host_labels[host_eligible, step]):
            members = np.flatnonzero(host_eligible & (host_labels[:, step] == label))
            clusters = np.unique(host_clusters[members])
            cluster_values = np.stack(
                [
                    np.mean(
                        host_values[members[host_clusters[members] == cluster], step],
                        axis=0,
                    )
                    for cluster in clusters
                ]
            )
            conditional = np.mean(cluster_values, axis=0)
            result[members, step] = conditional
            cluster_counts[members, step] = len(clusters)
    return jnp.asarray(result), jnp.asarray(cluster_counts)


def _finite_over(value: Array, axes: tuple[int, ...], /) -> Array:
    return jnp.all(jnp.isfinite(value), axis=axes)


def _path_rms(value: Array, /) -> Array:
    axes = tuple(range(1, value.ndim))
    finite = _finite_over(value, axes)
    safe = jnp.where(jnp.isfinite(value), value, 0.0)
    rms = jnp.sqrt(jnp.mean(jnp.square(safe), axis=axes))
    return jnp.where(finite, rms, jnp.asarray(jnp.inf, dtype=rms.dtype))


def _set_first_status(
    status: Array,
    failed: Array,
    code: StochasticMaximumPrincipleStatus,
    /,
) -> Array:
    return jnp.where(
        (status == int(StochasticMaximumPrincipleStatus.SUCCESS)) & failed,
        int(code),
        status,
    ).astype(jnp.int32)


def _validate_alignment(
    problem: StochasticMaximumPrincipleProblem,
    paths: ControlledPathBatch,
    /,
) -> None:
    if not isinstance(problem, StochasticMaximumPrincipleProblem):
        raise TypeError("problem must be a StochasticMaximumPrincipleProblem.")
    if not isinstance(paths, ControlledPathBatch):
        raise TypeError("paths must be a ControlledPathBatch.")
    if paths.problem_id != problem.problem_id:
        raise ValueError("paths and problem must carry the same problem_id.")
    if paths.state_shape != problem.state_shape:
        raise ValueError("paths state_shape does not match the SMP problem.")
    if paths.action_shape != problem.action_shape:
        raise ValueError("paths action_shape does not match the SMP problem.")
    if paths.noise_shape != problem.noise_shape:
        raise ValueError("paths noise_shape does not match the SMP problem.")
    if paths.time_grid.time_id != problem.time_grid.time_id or not bool(
        jnp.array_equal(paths.time_grid.times, problem.time_grid.times)
    ):
        raise ValueError("paths and problem must use the same time grid.")


def evaluate_stochastic_maximum_principle(
    problem: StochasticMaximumPrincipleProblem,
    paths: ControlledPathBatch,
    adjoint_prediction: AdjointPrediction,
    martingale_integrand_prediction: MartingaleIntegrandPrediction,
    information_labels: ArrayLike,
    /,
    *,
    information_id: str,
    predictor_id: str,
    sample_id: str,
    sample_role: SampleRole = "holdout",
    causal_information_checked: bool = False,
    tolerance: float = 1e-6,
    measurability_tolerance: float | None = None,
    convexity_checked: bool = False,
    convexity_evidence: str | None = None,
) -> StochasticMaximumPrincipleResult:
    """Evaluate open-loop stochastic maximum-principle residual evidence.

    The supplied path controls remain physical controls. The separately supplied
    martingale-integrand prediction is the BSDE ``q`` tensor and is never used as
    an action. Conditional stationarity is an equal-independent-cluster empirical
    mean within each caller-declared pre-increment information cell. Training and
    holdout samples retain separate explicit identities; no coverage, feedback,
    Markov-perfect, or population-optimality claim is inferred.
    """

    _validate_alignment(problem, paths)
    residual_tolerance = _positive_tolerance(tolerance, "tolerance")
    measurable_tolerance = _positive_tolerance(
        tolerance if measurability_tolerance is None else measurability_tolerance,
        "measurability_tolerance",
    )
    information_name = _identifier(information_id, "information_id")
    predictor_name = _identifier(predictor_id, "predictor_id")
    sample_name = _identifier(sample_id, "sample_id")
    role = _sample_role(sample_role)
    if not isinstance(causal_information_checked, bool):
        raise TypeError("causal_information_checked must be a bool.")
    if not isinstance(convexity_checked, bool):
        raise TypeError("convexity_checked must be a bool.")
    if convexity_checked:
        convexity_evidence = _identifier(convexity_evidence, "convexity_evidence")  # type: ignore[arg-type]
    elif convexity_evidence is not None:
        raise ValueError(
            "convexity_evidence requires convexity_checked=True; unchecked text is not evidence."
        )

    count = paths.path_count
    steps = problem.time_grid.num_steps
    labels = _labels(information_labels, count, steps)
    adjoint, integrand = _prediction_values(
        problem, paths, adjoint_prediction, martingale_integrand_prediction
    )

    forward_data_finite = (
        jnp.all(jnp.isfinite(paths.states), axis=(1, 2))
        & jnp.all(jnp.isfinite(paths.actions), axis=(1, 2))
        & jnp.all(jnp.isfinite(paths.noise_paths), axis=(1, 2))
    )
    base_forward_valid = paths.valid & paths.noise_valid & forward_data_finite
    safe_states = jnp.where(jnp.isfinite(paths.states), paths.states, 0.0)
    safe_actions = jnp.where(jnp.isfinite(paths.actions), paths.actions, 0.0)

    drifts = []
    diffusions = []
    drift_states = []
    drift_actions = []
    diffusion_states = []
    diffusion_actions = []
    running_states = []
    running_actions = []
    for step in range(steps):
        context = DiscreteStepContext(
            problem.time_grid.times[step],
            problem.time_grid.times[step + 1],
            jnp.asarray(step, dtype=jnp.int32),
        )
        state = safe_states[:, step]
        action = safe_actions[:, step]
        drifts.append(
            _batched_stage_callback(
                problem.drift,
                context,
                state,
                action,
                problem.args,
                problem.state_shape,
                "drift",
            )
        )
        diffusions.append(
            _batched_stage_callback(
                problem.diffusion,
                context,
                state,
                action,
                problem.args,
                (problem.state_size, problem.noise_size),
                "diffusion",
            )
        )
        drift_states.append(
            _batched_stage_callback(
                problem.drift_state_jacobian,
                context,
                state,
                action,
                problem.args,
                (problem.state_size, problem.state_size),
                "drift_state_jacobian",
            )
        )
        drift_actions.append(
            _batched_stage_callback(
                problem.drift_action_jacobian,
                context,
                state,
                action,
                problem.args,
                (problem.state_size, problem.action_size),
                "drift_action_jacobian",
            )
        )
        diffusion_states.append(
            _batched_stage_callback(
                problem.diffusion_state_jacobian,
                context,
                state,
                action,
                problem.args,
                (problem.state_size, problem.noise_size, problem.state_size),
                "diffusion_state_jacobian",
            )
        )
        diffusion_actions.append(
            _batched_stage_callback(
                problem.diffusion_action_jacobian,
                context,
                state,
                action,
                problem.args,
                (problem.state_size, problem.noise_size, problem.action_size),
                "diffusion_action_jacobian",
            )
        )
        running_states.append(
            _batched_stage_callback(
                problem.running_cost_state_gradient,
                context,
                state,
                action,
                problem.args,
                problem.state_shape,
                "running_cost_state_gradient",
            )
        )
        running_actions.append(
            _batched_stage_callback(
                problem.running_cost_action_gradient,
                context,
                state,
                action,
                problem.args,
                problem.action_shape,
                "running_cost_action_gradient",
            )
        )

    drift = jnp.stack(drifts, axis=1)
    diffusion = jnp.stack(diffusions, axis=1)
    drift_state = jnp.stack(drift_states, axis=1)
    drift_action = jnp.stack(drift_actions, axis=1)
    diffusion_state = jnp.stack(diffusion_states, axis=1)
    diffusion_action = jnp.stack(diffusion_actions, axis=1)
    running_state = jnp.stack(running_states, axis=1)
    running_action = jnp.stack(running_actions, axis=1)
    terminal_gradient = _terminal_gradients(problem, safe_states[:, -1])

    p_stage = adjoint[:, :-1]
    hamiltonian_state = (
        running_state
        + ein.contract("ptij,pti->ptj", drift_state, p_stage)
        + ein.contract("ptiwj,ptiw->ptj", diffusion_state, integrand)
    )
    hamiltonian_action = (
        running_action
        + ein.contract("ptim,pti->ptm", drift_action, p_stage)
        + ein.contract("ptiwm,ptiw->ptm", diffusion_action, integrand)
    )

    durations = problem.time_grid.durations.reshape((1, steps, 1))
    stochastic_forward = ein.contract("ptiw,ptw->pti", diffusion, paths.noise_paths)
    forward_residual = (
        paths.states[:, 1:]
        - paths.states[:, :-1]
        - drift * durations
        - stochastic_forward
    )
    terminal_residual = adjoint[:, -1] - terminal_gradient
    martingale_increment = ein.contract("ptiw,ptw->pti", integrand, paths.noise_paths)
    backward_residual = (
        adjoint[:, 1:]
        - adjoint[:, :-1]
        + hamiltonian_state * durations
        - martingale_increment
    )

    adjoint_finite = jnp.all(jnp.isfinite(adjoint), axis=(1, 2))
    integrand_finite = jnp.all(jnp.isfinite(integrand), axis=(1, 2, 3))
    dynamics_finite = jnp.all(jnp.isfinite(drift), axis=(1, 2)) & jnp.all(
        jnp.isfinite(diffusion), axis=(1, 2, 3)
    )
    derivative_finite = (
        jnp.all(jnp.isfinite(drift_state), axis=(1, 2, 3))
        & jnp.all(jnp.isfinite(drift_action), axis=(1, 2, 3))
        & jnp.all(jnp.isfinite(diffusion_state), axis=(1, 2, 3, 4))
        & jnp.all(jnp.isfinite(diffusion_action), axis=(1, 2, 3, 4))
        & jnp.all(jnp.isfinite(running_state), axis=(1, 2))
        & jnp.all(jnp.isfinite(running_action), axis=(1, 2))
    )
    terminal_finite = jnp.all(jnp.isfinite(terminal_gradient), axis=1)
    numerical_valid = (
        base_forward_valid
        & adjoint_finite
        & integrand_finite
        & dynamics_finite
        & derivative_finite
        & terminal_finite
    )

    action_measurability = _cell_measurability_residuals(
        paths.actions, labels, numerical_valid
    )
    adjoint_measurability = _cell_measurability_residuals(
        adjoint[:, :-1], labels, numerical_valid
    )
    integrand_measurability = _cell_measurability_residuals(
        integrand, labels, numerical_valid
    )
    measurable = (
        (action_measurability <= measurable_tolerance)
        & (adjoint_measurability <= measurable_tolerance)
        & (integrand_measurability <= measurable_tolerance)
    )
    conditional_stationarity, cluster_counts = _conditional_cluster_means(
        hamiltonian_action,
        labels,
        paths.independence_labels,
        numerical_valid,
    )
    causal_evidence = SMPCausalInformationEvidence(
        information_labels=labels,
        action_measurability_residuals=action_measurability,
        adjoint_measurability_residuals=adjoint_measurability,
        martingale_integrand_measurability_residuals=integrand_measurability,
        conditional_cluster_counts=cluster_counts,
        measurable=measurable,
        information_id=information_name,
        externally_checked=causal_information_checked,
        tolerance=measurable_tolerance,
    )
    path_causal = jnp.asarray(causal_information_checked) & jnp.all(measurable, axis=1)

    status = jnp.where(
        base_forward_valid,
        int(StochasticMaximumPrincipleStatus.SUCCESS),
        int(StochasticMaximumPrincipleStatus.INVALID_FORWARD_PATH),
    ).astype(jnp.int32)
    status = _set_first_status(
        status,
        ~adjoint_finite,
        StochasticMaximumPrincipleStatus.NONFINITE_ADJOINT,
    )
    status = _set_first_status(
        status,
        ~integrand_finite,
        StochasticMaximumPrincipleStatus.NONFINITE_MARTINGALE_INTEGRAND,
    )
    status = _set_first_status(
        status,
        ~dynamics_finite,
        StochasticMaximumPrincipleStatus.NONFINITE_DYNAMICS,
    )
    status = _set_first_status(
        status,
        ~derivative_finite,
        StochasticMaximumPrincipleStatus.NONFINITE_DERIVATIVE,
    )
    status = _set_first_status(
        status,
        ~terminal_finite,
        StochasticMaximumPrincipleStatus.NONFINITE_TERMINAL_GRADIENT,
    )
    status = _set_first_status(
        status,
        ~path_causal,
        StochasticMaximumPrincipleStatus.NONCAUSAL_INFORMATION,
    )
    valid = status == int(StochasticMaximumPrincipleStatus.SUCCESS)

    forward_norm = _path_rms(forward_residual)
    terminal_norm = _path_rms(terminal_residual)
    backward_norm = _path_rms(backward_residual)
    stationarity_norm = _path_rms(conditional_stationarity)
    maximum_norm = jnp.maximum(
        jnp.maximum(forward_norm, terminal_norm),
        jnp.maximum(backward_norm, stationarity_norm),
    )
    maximum_norm = jnp.where(
        valid, maximum_norm, jnp.asarray(jnp.inf, dtype=maximum_norm.dtype)
    )
    stationary = valid & (maximum_norm <= residual_tolerance)

    eligible_labels = np.asarray(
        jax.device_get(paths.independence_labels[numerical_valid & path_causal])
    )
    independent_count = len(np.unique(eligible_labels))
    path_evidence = SMPPathClusterEvidence(
        path_valid=valid,
        valid_path_count=jnp.sum(valid, dtype=jnp.int32),
        independent_cluster_count=jnp.asarray(independent_count, dtype=jnp.int32),
        independence_labels=paths.independence_labels,
        path_ids=paths.realization_ids,
        coupling_id=paths.coupling_id,
        sample_role=role,
        sample_id=sample_name,
    )

    return StochasticMaximumPrincipleResult(
        paths=paths,
        causal_information=causal_evidence,
        path_evidence=path_evidence,
        adjoint_values=adjoint,
        martingale_integrands=integrand,
        drift_values=drift,
        diffusion_values=diffusion,
        hamiltonian_state_gradients=hamiltonian_state,
        hamiltonian_action_gradients=hamiltonian_action,
        forward_residuals=forward_residual,
        terminal_adjoint_residuals=terminal_residual,
        backward_martingale_residuals=backward_residual,
        conditional_stationarity_residuals=conditional_stationarity,
        forward_rms_norms=forward_norm,
        terminal_adjoint_rms_norms=terminal_norm,
        backward_martingale_rms_norms=backward_norm,
        conditional_stationarity_rms_norms=stationarity_norm,
        maximum_residual_norms=maximum_norm,
        status=status,
        valid=valid,
        stationary=stationary,
        tolerance=residual_tolerance,
        certificate=_CERTIFICATE,
        method_id=_METHOD_ID,
        predictor_id=predictor_name,
        convexity_checked=convexity_checked,
        convexity_evidence=convexity_evidence,
        sufficient=False,
        population_stationarity_claim=False,
        feedback_claim=False,
        markov_perfect_claim=False,
        global_optimality_claim=False,
    )


__all__ = [
    "SMPAdjointPredictor",
    "SMPCausalInformationEvidence",
    "SMPMartingaleIntegrandPredictor",
    "SMPPathClusterEvidence",
    "StochasticMaximumPrincipleProblem",
    "StochasticMaximumPrincipleResult",
    "StochasticMaximumPrincipleStatus",
    "evaluate_stochastic_maximum_principle",
]
