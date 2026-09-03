#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Open-loop stochastic maximum-principle evidence for simultaneous games."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ...dynamics import DiscreteStepContext, TimeGrid
from ..stochastic._evaluation import ControlledPathBatch
from ..stochastic._smp import (
    _batched_stage_callback,
    _cell_measurability_residuals,
    _conditional_cluster_means,
    _identifier,
    _path_rms,
    _positive_tolerance,
    _real_array,
    _sample_role,
    _vector_shape,
    AdjointPrediction,
    MartingaleIntegrandPrediction,
    SampleRole,
    SMPStageMatrix,
    SMPStageVector,
    SMPTerminalGradient,
)
from ._layout import PlayerControlPartition


_METHOD_ID = "pathwise-euler-open-loop-stochastic-game-smp-v1"
_CERTIFICATE = "OPEN_LOOP_NASH_SMP_STATIONARY"


class OpenLoopStochasticGameSMPStatus(IntEnum):
    """Stable player-and-path status codes for stochastic game SMP evidence."""

    SUCCESS = 0
    INVALID_FORWARD_PATH = 1
    NONFINITE_ADJOINT = 2
    NONFINITE_MARTINGALE_INTEGRAND = 3
    NONFINITE_DYNAMICS = 4
    NONFINITE_DERIVATIVE = 5
    NONFINITE_TERMINAL_GRADIENT = 6
    NONCAUSAL_INFORMATION = 7
    NO_VALID_PATHS = 8


class OpenLoopStochasticGameSMPProblem(StrictModule):
    """Continuous-time SMP ingredients for a supplied simultaneous-game path.

    ``partition`` fixes contiguous physical-action ownership. Every player has
    its own running and terminal gradient callbacks and its own adjoint pair
    ``(p_i, q_i)``. Running action gradients return the full joint-action
    gradient, but evaluation retains only the rows owned by that player. The
    controlled drift and diffusion and all their state/action Jacobians are
    supplied explicitly; the owned Hamiltonian row includes ``q_i:sigma_a``.
    """

    time_grid: TimeGrid
    partition: PlayerControlPartition
    drift: SMPStageVector
    diffusion: SMPStageMatrix
    drift_state_jacobian: SMPStageMatrix
    drift_action_jacobian: SMPStageMatrix
    diffusion_state_jacobian: SMPStageMatrix
    diffusion_action_jacobian: SMPStageMatrix
    running_cost_state_gradients: tuple[SMPStageVector, ...] = eqx.field(static=True)
    running_cost_action_gradients: tuple[SMPStageVector, ...] = eqx.field(static=True)
    terminal_cost_gradients: tuple[SMPTerminalGradient, ...] = eqx.field(static=True)
    args: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    action_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    action_size: int = eqx.field(static=True)
    noise_size: int = eqx.field(static=True)
    num_players: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_grid: TimeGrid,
        partition: PlayerControlPartition,
        drift: SMPStageVector,
        diffusion: SMPStageMatrix,
        drift_state_jacobian: SMPStageMatrix,
        drift_action_jacobian: SMPStageMatrix,
        diffusion_state_jacobian: SMPStageMatrix,
        diffusion_action_jacobian: SMPStageMatrix,
        /,
        *,
        running_cost_state_gradients: Sequence[SMPStageVector],
        running_cost_action_gradients: Sequence[SMPStageVector],
        terminal_cost_gradients: Sequence[SMPTerminalGradient],
        state_shape: Sequence[int],
        noise_shape: Sequence[int],
        args: Any = None,
        problem_id: str,
    ):
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        if not isinstance(partition, PlayerControlPartition):
            raise TypeError("partition must be a PlayerControlPartition.")
        shared_callbacks = (
            drift,
            diffusion,
            drift_state_jacobian,
            drift_action_jacobian,
            diffusion_state_jacobian,
            diffusion_action_jacobian,
        )
        if any(not callable(callback) for callback in shared_callbacks):
            raise TypeError("Every shared stochastic game callback must be callable.")
        running_state = tuple(running_cost_state_gradients)
        running_action = tuple(running_cost_action_gradients)
        terminal = tuple(terminal_cost_gradients)
        for owner, callbacks in (
            ("running_cost_state_gradients", running_state),
            ("running_cost_action_gradients", running_action),
            ("terminal_cost_gradients", terminal),
        ):
            if len(callbacks) != partition.num_players:
                raise ValueError(f"{owner} must contain one callback per player.")
            if any(not callable(callback) for callback in callbacks):
                raise TypeError(f"Every {owner} entry must be callable.")
        states = _vector_shape(state_shape, "state_shape")
        noises = _vector_shape(noise_shape, "noise_shape")
        actions = (partition.joint_control_size,)
        self.time_grid = time_grid
        self.partition = partition
        self.drift = drift
        self.diffusion = diffusion
        self.drift_state_jacobian = drift_state_jacobian
        self.drift_action_jacobian = drift_action_jacobian
        self.diffusion_state_jacobian = diffusion_state_jacobian
        self.diffusion_action_jacobian = diffusion_action_jacobian
        self.running_cost_state_gradients = running_state
        self.running_cost_action_gradients = running_action
        self.terminal_cost_gradients = terminal
        self.args = args
        self.state_shape = states
        self.action_shape = actions
        self.noise_shape = noises
        self.state_size = states[0]
        self.action_size = actions[0]
        self.noise_size = noises[0]
        self.num_players = partition.num_players
        self.problem_id = _identifier(problem_id, "problem_id")


class GameSMPCausalInformationEvidence(StrictModule):
    """Per-player empirical adaptedness under declared pre-increment information."""

    information_labels: Array
    owned_action_measurability_residuals: Array
    adjoint_measurability_residuals: Array
    martingale_integrand_measurability_residuals: Array
    conditional_cluster_counts: Array
    measurable: Array
    causal: Array
    information_ids: tuple[str, ...] = eqx.field(static=True)
    externally_checked: tuple[bool, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return self.causal


class GameSMPPathClusterEvidence(StrictModule):
    """Player-local path eligibility with shared stochastic provenance."""

    path_valid: Array
    valid_path_counts: Array
    independent_cluster_counts: Array
    independence_labels: Array
    path_ids: tuple[str, ...] = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    sample_role: SampleRole = eqx.field(static=True)
    sample_id: str = eqx.field(static=True)

    @property
    def realization_ids(self) -> tuple[str, ...]:
        return self.path_ids


class OpenLoopStochasticGameSMPResult(StrictModule):
    """Owned-row open-loop Nash SMP evidence on supplied stochastic paths.

    The result contains empirical first-order evidence only. It does not construct
    a feedback strategy. Finite-batch conditional residuals and checked convexity
    metadata cannot establish population conditional stationarity, open-loop Nash
    sufficiency, feedback Nash, or a Markov-perfect equilibrium.
    """

    paths: ControlledPathBatch
    partition: PlayerControlPartition
    causal_information: GameSMPCausalInformationEvidence
    path_evidence: GameSMPPathClusterEvidence
    player_adjoint_values: Array
    player_martingale_integrands: Array
    drift_values: Array
    diffusion_values: Array
    player_hamiltonian_state_gradients: Array
    owned_hamiltonian_action_gradients: Array
    forward_residuals: Array
    terminal_adjoint_residuals: Array
    backward_martingale_residuals: Array
    conditional_owned_stationarity_residuals: Array
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
    predictor_ids: tuple[str, ...] = eqx.field(static=True)
    convexity_checked: tuple[bool, ...] = eqx.field(static=True)
    convexity_evidence: tuple[str | None, ...] = eqx.field(static=True)
    sufficient: bool = eqx.field(static=True)
    population_stationarity_claim: bool = eqx.field(static=True)
    population_nash_claim: bool = eqx.field(static=True)
    open_loop_nash_claim: bool = eqx.field(static=True)
    feedback_claim: bool = eqx.field(static=True)
    feedback_nash_claim: bool = eqx.field(static=True)
    markov_perfect_claim: bool = eqx.field(static=True)

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
        return self.conditional_owned_stationarity_residuals


def _validate_alignment(
    problem: OpenLoopStochasticGameSMPProblem,
    paths: ControlledPathBatch,
    /,
) -> None:
    if not isinstance(problem, OpenLoopStochasticGameSMPProblem):
        raise TypeError("problem must be an OpenLoopStochasticGameSMPProblem.")
    if not isinstance(paths, ControlledPathBatch):
        raise TypeError("paths must be a ControlledPathBatch.")
    if paths.problem_id != problem.problem_id:
        raise ValueError("paths and problem must carry the same problem_id.")
    if paths.state_shape != problem.state_shape:
        raise ValueError("paths state_shape does not match the game SMP problem.")
    if paths.action_shape != problem.action_shape:
        raise ValueError("paths action_shape does not match the game SMP problem.")
    if paths.noise_shape != problem.noise_shape:
        raise ValueError("paths noise_shape does not match the game SMP problem.")
    if paths.time_grid.time_id != problem.time_grid.time_id or not bool(
        jnp.array_equal(paths.time_grid.times, problem.time_grid.times)
    ):
        raise ValueError("paths and problem must use the same time grid.")


def _player_predictions(
    problem: OpenLoopStochasticGameSMPProblem,
    paths: ControlledPathBatch,
    adjoint_predictions: Sequence[AdjointPrediction],
    martingale_integrand_predictions: Sequence[MartingaleIntegrandPrediction],
    /,
) -> tuple[Array, Array]:
    adjoint_inputs = tuple(adjoint_predictions)
    integrand_inputs = tuple(martingale_integrand_predictions)
    if len(adjoint_inputs) != problem.num_players:
        raise ValueError("adjoint_predictions must contain one predictor per player.")
    if len(integrand_inputs) != problem.num_players:
        raise ValueError(
            "martingale_integrand_predictions must contain one predictor per player."
        )
    count = paths.path_count
    steps = problem.time_grid.num_steps
    safe_states = jnp.where(jnp.isfinite(paths.states), paths.states, 0.0)
    safe_actions = jnp.where(jnp.isfinite(paths.actions), paths.actions, 0.0)
    adjoints = []
    integrands = []
    for player, prediction in enumerate(adjoint_inputs):
        if callable(prediction):
            nodes = []
            for node in range(steps + 1):
                nodes.append(
                    jax.vmap(
                        lambda state: jnp.asarray(
                            prediction(problem.time_grid.times[node], state, problem.args)
                        )
                    )(safe_states[:, node])
                )
            value = _real_array(
                jnp.stack(nodes, axis=1), f"adjoint_predictions[{player}]"
            )
        else:
            value = _real_array(prediction, f"adjoint_predictions[{player}]")
        expected = (count, steps + 1, problem.state_size)
        if tuple(value.shape) != expected:
            raise ValueError(f"adjoint_predictions[{player}] must have shape {expected}.")
        adjoints.append(value)

    for player, prediction in enumerate(integrand_inputs):
        if callable(prediction):
            stages = []
            for step in range(steps):
                context = DiscreteStepContext(
                    problem.time_grid.times[step],
                    problem.time_grid.times[step + 1],
                    jnp.asarray(step, dtype=jnp.int32),
                )
                stages.append(
                    jax.vmap(
                        lambda state, action: jnp.asarray(
                            prediction(context, state, action, problem.args)
                        )
                    )(safe_states[:, step], safe_actions[:, step])
                )
            value = _real_array(
                jnp.stack(stages, axis=1),
                f"martingale_integrand_predictions[{player}]",
            )
        else:
            value = _real_array(prediction, f"martingale_integrand_predictions[{player}]")
        expected = (count, steps, problem.state_size, problem.noise_size)
        if tuple(value.shape) != expected:
            raise ValueError(
                f"martingale_integrand_predictions[{player}] must have shape "
                f"{expected}; physical actions are not BSDE integrands."
            )
        integrands.append(value)
    return jnp.stack(adjoints, axis=0), jnp.stack(integrands, axis=0)


def _game_labels(
    value: ArrayLike,
    players: int,
    count: int,
    steps: int,
    /,
) -> Array:
    labels = jnp.asarray(value)
    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise TypeError("information_labels must have an integer dtype.")
    expected = (players, count, steps)
    if tuple(labels.shape) != expected:
        raise ValueError(f"information_labels must have shape {expected}.")
    host = np.asarray(labels)
    if np.any(host < 0) or np.any(host > np.iinfo(np.int32).max):
        raise ValueError("information_labels must be nonnegative int32 values.")
    return labels.astype(jnp.int32)


def _bools(value: bool | Sequence[bool], count: int, owner: str, /) -> tuple[bool, ...]:
    if isinstance(value, bool):
        return (value,) * count
    values = tuple(value)
    if len(values) != count or any(not isinstance(item, bool) for item in values):
        raise TypeError(f"{owner} must be a bool or contain one bool per player.")
    return values


def _optional_evidence(
    checked: tuple[bool, ...],
    evidence: Sequence[str | None] | None,
    /,
) -> tuple[str | None, ...]:
    values = (None,) * len(checked) if evidence is None else tuple(evidence)
    if len(values) != len(checked):
        raise ValueError("convexity_evidence must contain one entry per player.")
    resolved: list[str | None] = []
    for player, (is_checked, item) in enumerate(zip(checked, values, strict=True)):
        if is_checked:
            resolved.append(_identifier(item, f"convexity_evidence[{player}]"))  # type: ignore[arg-type]
        elif item is not None:
            raise ValueError(
                "convexity evidence requires the corresponding checked flag."
            )
        else:
            resolved.append(None)
    return tuple(resolved)


def _set_first_status(
    status: Array,
    failed: Array,
    code: OpenLoopStochasticGameSMPStatus,
    /,
) -> Array:
    return jnp.where(
        (status == int(OpenLoopStochasticGameSMPStatus.SUCCESS)) & failed,
        int(code),
        status,
    ).astype(jnp.int32)


def evaluate_open_loop_stochastic_game_smp(
    problem: OpenLoopStochasticGameSMPProblem,
    paths: ControlledPathBatch,
    adjoint_predictions: Sequence[AdjointPrediction],
    martingale_integrand_predictions: Sequence[MartingaleIntegrandPrediction],
    information_labels: ArrayLike,
    /,
    *,
    information_ids: Sequence[str],
    predictor_ids: Sequence[str],
    sample_id: str,
    sample_role: SampleRole = "holdout",
    causal_information_checked: bool | Sequence[bool] = False,
    tolerance: float = 1e-6,
    measurability_tolerance: float | None = None,
    convexity_checked: bool | Sequence[bool] = False,
    convexity_evidence: Sequence[str | None] | None = None,
) -> OpenLoopStochasticGameSMPResult:
    """Evaluate player adjoints and only their owned stochastic-Hamiltonian rows.

    Information labels are player-specific pre-increment conditioning cells.
    Conditional stationarity uses equal weighting of the declared independent
    clusters in each cell. The evaluator never interprets physical actions as
    martingale integrands and never promotes supplied open-loop paths to a
    feedback or Markov-perfect equilibrium.
    """

    _validate_alignment(problem, paths)
    residual_tolerance = _positive_tolerance(tolerance, "tolerance")
    measurable_tolerance = _positive_tolerance(
        tolerance if measurability_tolerance is None else measurability_tolerance,
        "measurability_tolerance",
    )
    player_information_ids = tuple(
        _identifier(item, f"information_ids[{player}]")
        for player, item in enumerate(information_ids)
    )
    player_predictor_ids = tuple(
        _identifier(item, f"predictor_ids[{player}]")
        for player, item in enumerate(predictor_ids)
    )
    if len(player_information_ids) != problem.num_players:
        raise ValueError("information_ids must contain one ID per player.")
    if len(player_predictor_ids) != problem.num_players:
        raise ValueError("predictor_ids must contain one ID per player.")
    sample_name = _identifier(sample_id, "sample_id")
    role = _sample_role(sample_role)
    causal_checked = _bools(
        causal_information_checked,
        problem.num_players,
        "causal_information_checked",
    )
    convex_checked = _bools(convexity_checked, problem.num_players, "convexity_checked")
    convex_evidence = _optional_evidence(convex_checked, convexity_evidence)

    count = paths.path_count
    steps = problem.time_grid.num_steps
    labels = _game_labels(information_labels, problem.num_players, count, steps)
    adjoint, integrand = _player_predictions(
        problem,
        paths,
        adjoint_predictions,
        martingale_integrand_predictions,
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
    running_states: list[list[Array]] = [[] for _ in range(problem.num_players)]
    running_actions: list[list[Array]] = [[] for _ in range(problem.num_players)]
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
        for player in range(problem.num_players):
            running_states[player].append(
                _batched_stage_callback(
                    problem.running_cost_state_gradients[player],
                    context,
                    state,
                    action,
                    problem.args,
                    problem.state_shape,
                    f"running_cost_state_gradients[{player}]",
                )
            )
            running_actions[player].append(
                _batched_stage_callback(
                    problem.running_cost_action_gradients[player],
                    context,
                    state,
                    action,
                    problem.args,
                    problem.action_shape,
                    f"running_cost_action_gradients[{player}]",
                )
            )

    drift = jnp.stack(drifts, axis=1)
    diffusion = jnp.stack(diffusions, axis=1)
    drift_state = jnp.stack(drift_states, axis=1)
    drift_action = jnp.stack(drift_actions, axis=1)
    diffusion_state = jnp.stack(diffusion_states, axis=1)
    diffusion_action = jnp.stack(diffusion_actions, axis=1)
    running_state = jnp.stack(
        tuple(jnp.stack(values, axis=1) for values in running_states), axis=0
    )
    running_action = jnp.stack(
        tuple(jnp.stack(values, axis=1) for values in running_actions), axis=0
    )

    terminal_gradients = []
    for player, callback in enumerate(problem.terminal_cost_gradients):
        values = jax.vmap(
            lambda state: jnp.asarray(
                callback(problem.time_grid.times[-1], state, problem.args)
            )
        )(safe_states[:, -1])
        value = _real_array(values, f"terminal_cost_gradients[{player}]")
        expected = (count, problem.state_size)
        if tuple(value.shape) != expected:
            raise ValueError(
                f"terminal_cost_gradients[{player}] must return {problem.state_shape}."
            )
        terminal_gradients.append(value)
    terminal_gradient = jnp.stack(terminal_gradients, axis=0)

    hamiltonian_state = (
        running_state
        + ein.contract("ptij,apti->aptj", drift_state, adjoint[:, :, :-1])
        + ein.contract("ptiwj,aptiw->aptj", diffusion_state, integrand)
    )
    full_hamiltonian_action = (
        running_action
        + ein.contract("ptim,apti->aptm", drift_action, adjoint[:, :, :-1])
        + ein.contract("ptiwm,aptiw->aptm", diffusion_action, integrand)
    )
    owned_hamiltonian_action = jnp.zeros(
        (count, steps, problem.action_size), dtype=full_hamiltonian_action.dtype
    )
    for player, (start, stop) in enumerate(problem.partition.control_slices):
        owned_hamiltonian_action = owned_hamiltonian_action.at[..., start:stop].set(
            full_hamiltonian_action[player, ..., start:stop]
        )

    durations = problem.time_grid.durations.reshape((1, steps, 1))
    stochastic_forward = ein.contract("ptiw,ptw->pti", diffusion, paths.noise_paths)
    forward_residual = (
        paths.states[:, 1:]
        - paths.states[:, :-1]
        - drift * durations
        - stochastic_forward
    )
    terminal_residual = adjoint[:, :, -1] - terminal_gradient
    martingale_increment = ein.contract("aptiw,ptw->apti", integrand, paths.noise_paths)
    backward_residual = (
        adjoint[:, :, 1:]
        - adjoint[:, :, :-1]
        + hamiltonian_state * durations[None, ...]
        - martingale_increment
    )

    shared_dynamics_finite = jnp.all(jnp.isfinite(drift), axis=(1, 2)) & jnp.all(
        jnp.isfinite(diffusion), axis=(1, 2, 3)
    )
    state_derivative_finite = jnp.all(
        jnp.isfinite(drift_state), axis=(1, 2, 3)
    ) & jnp.all(jnp.isfinite(diffusion_state), axis=(1, 2, 3, 4))
    adjoint_finite = jnp.all(jnp.isfinite(adjoint), axis=(2, 3))
    integrand_finite = jnp.all(jnp.isfinite(integrand), axis=(2, 3, 4))
    terminal_finite = jnp.all(jnp.isfinite(terminal_gradient), axis=2)
    derivative_finite_players = []
    for player, (start, stop) in enumerate(problem.partition.control_slices):
        action_derivative_finite = jnp.all(
            jnp.isfinite(drift_action[..., start:stop]), axis=(1, 2, 3)
        ) & jnp.all(jnp.isfinite(diffusion_action[..., start:stop]), axis=(1, 2, 3, 4))
        running_finite = jnp.all(
            jnp.isfinite(running_state[player]), axis=(1, 2)
        ) & jnp.all(jnp.isfinite(running_action[player, ..., start:stop]), axis=(1, 2))
        derivative_finite_players.append(
            state_derivative_finite & action_derivative_finite & running_finite
        )
    derivative_finite = jnp.stack(derivative_finite_players, axis=0)
    numerical_valid = (
        base_forward_valid[None, :]
        & shared_dynamics_finite[None, :]
        & adjoint_finite
        & integrand_finite
        & derivative_finite
        & terminal_finite
    )

    action_measurability = []
    adjoint_measurability = []
    integrand_measurability = []
    measurable = []
    conditional_counts = []
    conditional_owned = jnp.full_like(owned_hamiltonian_action, jnp.nan)
    for player, (start, stop) in enumerate(problem.partition.control_slices):
        action_residual = _cell_measurability_residuals(
            paths.actions[..., start:stop], labels[player], numerical_valid[player]
        )
        adjoint_residual = _cell_measurability_residuals(
            adjoint[player, :, :-1], labels[player], numerical_valid[player]
        )
        integrand_residual = _cell_measurability_residuals(
            integrand[player], labels[player], numerical_valid[player]
        )
        player_measurable = (
            (action_residual <= measurable_tolerance)
            & (adjoint_residual <= measurable_tolerance)
            & (integrand_residual <= measurable_tolerance)
        )
        conditional, counts = _conditional_cluster_means(
            full_hamiltonian_action[player, ..., start:stop],
            labels[player],
            paths.independence_labels,
            numerical_valid[player],
        )
        conditional_owned = conditional_owned.at[..., start:stop].set(conditional)
        action_measurability.append(action_residual)
        adjoint_measurability.append(adjoint_residual)
        integrand_measurability.append(integrand_residual)
        measurable.append(player_measurable)
        conditional_counts.append(counts)

    action_measurability_array = jnp.stack(action_measurability, axis=0)
    adjoint_measurability_array = jnp.stack(adjoint_measurability, axis=0)
    integrand_measurability_array = jnp.stack(integrand_measurability, axis=0)
    measurable_array = jnp.stack(measurable, axis=0)
    conditional_counts_array = jnp.stack(conditional_counts, axis=0)
    causal_flags = jnp.asarray(causal_checked, dtype=bool)
    causal = causal_flags & jnp.all(measurable_array, axis=(1, 2))
    path_causal = causal_flags[:, None] & jnp.all(measurable_array, axis=2)
    causal_evidence = GameSMPCausalInformationEvidence(
        information_labels=labels,
        owned_action_measurability_residuals=action_measurability_array,
        adjoint_measurability_residuals=adjoint_measurability_array,
        martingale_integrand_measurability_residuals=integrand_measurability_array,
        conditional_cluster_counts=conditional_counts_array,
        measurable=measurable_array,
        causal=causal,
        information_ids=player_information_ids,
        externally_checked=causal_checked,
        tolerance=measurable_tolerance,
    )

    status = jnp.where(
        base_forward_valid[None, :],
        int(OpenLoopStochasticGameSMPStatus.SUCCESS),
        int(OpenLoopStochasticGameSMPStatus.INVALID_FORWARD_PATH),
    ).astype(jnp.int32)
    status = jnp.broadcast_to(status, (problem.num_players, count))
    status = _set_first_status(
        status,
        ~adjoint_finite,
        OpenLoopStochasticGameSMPStatus.NONFINITE_ADJOINT,
    )
    status = _set_first_status(
        status,
        ~integrand_finite,
        OpenLoopStochasticGameSMPStatus.NONFINITE_MARTINGALE_INTEGRAND,
    )
    status = _set_first_status(
        status,
        jnp.broadcast_to(~shared_dynamics_finite, (problem.num_players, count)),
        OpenLoopStochasticGameSMPStatus.NONFINITE_DYNAMICS,
    )
    status = _set_first_status(
        status,
        ~derivative_finite,
        OpenLoopStochasticGameSMPStatus.NONFINITE_DERIVATIVE,
    )
    status = _set_first_status(
        status,
        ~terminal_finite,
        OpenLoopStochasticGameSMPStatus.NONFINITE_TERMINAL_GRADIENT,
    )
    status = _set_first_status(
        status,
        ~path_causal,
        OpenLoopStochasticGameSMPStatus.NONCAUSAL_INFORMATION,
    )
    valid = status == int(OpenLoopStochasticGameSMPStatus.SUCCESS)

    forward_norm = _path_rms(forward_residual)
    terminal_norm = jnp.stack(
        tuple(
            _path_rms(terminal_residual[player]) for player in range(problem.num_players)
        )
    )
    backward_norm = jnp.stack(
        tuple(
            _path_rms(backward_residual[player]) for player in range(problem.num_players)
        )
    )
    stationarity_norms = []
    for start, stop in problem.partition.control_slices:
        stationarity_norms.append(_path_rms(conditional_owned[..., start:stop]))
    stationarity_norm = jnp.stack(stationarity_norms, axis=0)
    maximum_norm = jnp.maximum(
        jnp.maximum(forward_norm[None, :], terminal_norm),
        jnp.maximum(backward_norm, stationarity_norm),
    )
    maximum_norm = jnp.where(
        valid, maximum_norm, jnp.asarray(jnp.inf, dtype=maximum_norm.dtype)
    )
    stationary = valid & (maximum_norm <= residual_tolerance)

    independent_counts = []
    for player in range(problem.num_players):
        player_labels = np.asarray(
            jax.device_get(paths.independence_labels[valid[player]])
        )
        independent_counts.append(len(np.unique(player_labels)))
    path_evidence = GameSMPPathClusterEvidence(
        path_valid=valid,
        valid_path_counts=jnp.sum(valid, axis=1, dtype=jnp.int32),
        independent_cluster_counts=jnp.asarray(independent_counts, dtype=jnp.int32),
        independence_labels=paths.independence_labels,
        path_ids=paths.realization_ids,
        coupling_id=paths.coupling_id,
        sample_role=role,
        sample_id=sample_name,
    )

    return OpenLoopStochasticGameSMPResult(
        paths=paths,
        partition=problem.partition,
        causal_information=causal_evidence,
        path_evidence=path_evidence,
        player_adjoint_values=adjoint,
        player_martingale_integrands=integrand,
        drift_values=drift,
        diffusion_values=diffusion,
        player_hamiltonian_state_gradients=hamiltonian_state,
        owned_hamiltonian_action_gradients=owned_hamiltonian_action,
        forward_residuals=forward_residual,
        terminal_adjoint_residuals=terminal_residual,
        backward_martingale_residuals=backward_residual,
        conditional_owned_stationarity_residuals=conditional_owned,
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
        predictor_ids=player_predictor_ids,
        convexity_checked=convex_checked,
        convexity_evidence=convex_evidence,
        sufficient=False,
        population_stationarity_claim=False,
        population_nash_claim=False,
        open_loop_nash_claim=False,
        feedback_claim=False,
        feedback_nash_claim=False,
        markov_perfect_claim=False,
    )


__all__ = [
    "GameSMPCausalInformationEvidence",
    "GameSMPPathClusterEvidence",
    "OpenLoopStochasticGameSMPProblem",
    "OpenLoopStochasticGameSMPResult",
    "OpenLoopStochasticGameSMPStatus",
    "evaluate_open_loop_stochastic_game_smp",
]
