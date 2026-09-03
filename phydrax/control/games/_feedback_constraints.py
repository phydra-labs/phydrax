#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-active-set constrained local feedback quasi-Nash models.

The solve in this module is deliberately local.  It augments the stagewise
control-value equations of an already valid :class:`LocalAffineGameSuggestion`
with the KKT rows of one declared active set.  It neither searches for another
active set nor certifies the nonlinear game away from the nominal trajectory.
"""

from __future__ import annotations

import math
from enum import IntEnum
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    DifferentiationPolicy,
    FactorizationPolicy,
    factorize,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    RankPolicy,
    RHSLayout,
    solve,
    TolerancePolicy,
)
from ._constraints import (
    GameConstraintScope,
    GameConstraintSite,
    OpenLoopGameConstraints,
)
from ._local_lq import LocalAffineGamePolicy, LocalAffineGameSuggestion


CONSTRAINED_FEEDBACK_QUASI_NASH_MODEL = "CONSTRAINED_FEEDBACK_QUASI_NASH_MODEL"
_METHOD_ID = "control:games:fixed-active-feedback-quasi-nash-model"


class FeedbackQuasiNashStatus(IntEnum):
    """Stable case-local outcomes for the fixed-active-set local model."""

    SUCCESS = 0
    INVALID_LOCAL_SUGGESTION = 1
    NONFINITE_INPUT = 2
    LICQ_FAILURE = 3
    STRICT_COMPLEMENTARITY_FAILURE = 4
    OWN_CURVATURE_FAILURE = 5
    COUPLED_KKT_RANK_DEFICIENT = 6
    CONDITION_LIMIT_REACHED = 7
    LINEAR_SOLVE_FAILED = 8
    NONFINITE_OUTPUT = 9
    ACTIVE_RESIDUAL_TOO_LARGE = 10
    INACTIVE_CONSTRAINT_VIOLATION = 11
    KKT_RESIDUAL_TOO_LARGE = 12
    DEPENDENCY_FAILED = 13


class ConstrainedFeedbackGameProblem(StrictModule):
    """One stagewise linearized constraint model around a local LQ game.

    All blocks must be path blocks.  Their residual components are concatenated
    in block order on the final axis of the numeric arrays.  At every stage the
    supplied linearization means

    ``residual + state_jacobian @ dx + control_jacobian @ du <= 0``

    for inequalities (and equality to zero for equality blocks).  ``active_set``
    declares the one fixed active set to use; equality components must be active.

    A non-variational shared block allocates one player-specific multiplier copy
    per participant and repeats the same physical feasibility row for every
    copy.  A variational shared block allocates one common copy.  Player-local
    and player-owned-coupled blocks always retain an owner-private multiplier.
    Duplicate generic-shared active equations are not normalized or collapsed:
    if they make the feedback-sensitivity KKT nonisolated, the solve reports a
    rank-deficient KKT.  That status does not reject existence of a generic GNE.
    """

    suggestion: LocalAffineGameSuggestion
    constraints: OpenLoopGameConstraints
    constraint_residuals: Array
    constraint_state_jacobians: Array
    constraint_control_jacobians: Array
    active_set: Array
    equality_mask: Array
    multiplier_constraint_indices_array: Array
    multiplier_equality_mask: Array
    stationarity_incidence: Array
    constraint_row_incidence: Array
    licq_row_incidence: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    num_constraints: int = eqx.field(static=True)
    num_multipliers: int = eqx.field(static=True)
    variational: bool = eqx.field(static=True)
    equilibrium_concept: str = eqx.field(static=True)
    block_constraint_slices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    multiplier_constraint_indices: tuple[int, ...] = eqx.field(static=True)
    multiplier_player_indices: tuple[int, ...] = eqx.field(static=True)
    multiplier_block_indices: tuple[int, ...] = eqx.field(static=True)
    player_multiplier_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    private_multiplier_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    shared_player_multiplier_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    variational_multiplier_indices: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        suggestion: LocalAffineGameSuggestion,
        constraints: OpenLoopGameConstraints,
        constraint_residuals: ArrayLike | None = None,
        constraint_state_jacobians: ArrayLike | None = None,
        constraint_control_jacobians: ArrayLike | None = None,
        active_set: ArrayLike | None = None,
        /,
        *,
        variational: bool = False,
        problem_id: str = "constrained-feedback-quasi-nash",
    ):
        if not isinstance(suggestion, LocalAffineGameSuggestion):
            raise TypeError("suggestion must be a LocalAffineGameSuggestion.")
        if not isinstance(constraints, OpenLoopGameConstraints):
            raise TypeError("constraints must be OpenLoopGameConstraints.")
        if not isinstance(variational, bool):
            raise TypeError("variational must be a bool.")
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError("problem_id must be a non-empty string.")
        model = suggestion.model
        partition = model.partition
        if constraints.partition.partition_id != partition.partition_id:
            raise ValueError(
                "constraints and the local suggestion must use the same player "
                "control partition."
            )
        if any(block.site is not GameConstraintSite.PATH for block in constraints.blocks):
            raise ValueError(
                "Constrained feedback models require stagewise PATH constraint blocks."
            )

        cases = suggestion.case_shape
        horizon = model.time_grid.num_steps
        state_size = model.state_size
        control_size = model.control_size
        block_slices: list[tuple[int, int]] = []
        equality: list[bool] = []
        cursor = 0
        for block in constraints.blocks:
            size = prod(block.residual_shape)
            block_slices.append((cursor, cursor + size))
            equality.extend((block.equality,) * size)
            cursor += size
        num_constraints = cursor
        expected_residual = cases + (horizon, num_constraints)
        expected_state = cases + (horizon, num_constraints, state_size)
        expected_control = cases + (horizon, num_constraints, control_size)
        if num_constraints and any(
            value is None
            for value in (
                constraint_residuals,
                constraint_state_jacobians,
                constraint_control_jacobians,
                active_set,
            )
        ):
            raise ValueError(
                "Nonempty constrained feedback models require residuals, state "
                "Jacobians, control Jacobians, and an explicit active_set."
            )

        residuals = _optional_array(
            constraint_residuals,
            expected_residual,
            "constraint_residuals",
            default_dtype=model.nominal_controls.dtype,
        )
        state_jacobians = _optional_array(
            constraint_state_jacobians,
            expected_state,
            "constraint_state_jacobians",
            default_dtype=residuals.dtype,
        )
        control_jacobians = _optional_array(
            constraint_control_jacobians,
            expected_control,
            "constraint_control_jacobians",
            default_dtype=residuals.dtype,
        )
        if active_set is None:
            active = jnp.zeros(expected_residual, dtype=bool)
        else:
            active = jnp.asarray(active_set)
            if active.dtype != jnp.bool_:
                raise TypeError("active_set must be a boolean array.")
            if tuple(active.shape) != expected_residual:
                raise ValueError(
                    f"active_set must have shape {expected_residual}; got {active.shape}."
                )
        equality_mask = jnp.asarray(equality, dtype=bool)
        if num_constraints:
            missing_equalities = jnp.any(equality_mask & ~active, axis=-1)
            active = eqx.error_if(
                active,
                jnp.any(missing_equalities),
                "Every equality constraint component must be active at every stage.",
            )

        (
            copy_constraints,
            copy_players,
            copy_blocks,
            player_indices,
            private_indices,
            shared_player_indices,
            variational_indices,
        ) = _multiplier_metadata(constraints, block_slices, variational=variational)
        num_multipliers = len(copy_constraints)
        control_owner = jnp.asarray(partition.control_owner, dtype=jnp.int32)
        stationarity_incidence = []
        licq_incidence = []
        for player in copy_players:
            if player < 0:
                stationarity_incidence.append(jnp.ones((control_size,), dtype=bool))
                licq_incidence.append(jnp.ones((control_size,), dtype=bool))
            else:
                owned_controls = control_owner == player
                stationarity_incidence.append(owned_controls)
                licq_incidence.append(owned_controls)
        stationarity_mask = (
            jnp.stack(stationarity_incidence, axis=-1)
            if stationarity_incidence
            else jnp.zeros((control_size, 0), dtype=bool)
        )
        licq_mask = (
            jnp.stack(licq_incidence, axis=0)
            if licq_incidence
            else jnp.zeros((0, control_size), dtype=bool)
        )
        constraint_mask = jnp.ones((num_multipliers, control_size), dtype=bool)
        multiplier_equalities = (
            equality_mask[jnp.asarray(copy_constraints, dtype=jnp.int32)]
            if copy_constraints
            else jnp.zeros((0,), dtype=bool)
        )

        dtype = jnp.result_type(
            residuals,
            state_jacobians,
            control_jacobians,
            model.nominal_controls,
            float,
        )
        self.suggestion = suggestion
        self.constraints = constraints
        self.constraint_residuals = residuals.astype(dtype)
        self.constraint_state_jacobians = state_jacobians.astype(dtype)
        self.constraint_control_jacobians = control_jacobians.astype(dtype)
        self.active_set = active
        self.equality_mask = equality_mask
        self.multiplier_constraint_indices_array = jnp.asarray(
            copy_constraints, dtype=jnp.int32
        )
        self.multiplier_equality_mask = multiplier_equalities
        self.stationarity_incidence = stationarity_mask
        self.constraint_row_incidence = constraint_mask
        self.licq_row_incidence = licq_mask
        self.case_shape = cases
        self.horizon = horizon
        self.state_size = state_size
        self.control_size = control_size
        self.num_constraints = num_constraints
        self.num_multipliers = num_multipliers
        self.variational = variational
        self.equilibrium_concept = (
            "variational-shared-common-multiplier"
            if variational
            else "generic-shared-player-multiplier-copies"
        )
        self.block_constraint_slices = tuple(block_slices)
        self.multiplier_constraint_indices = copy_constraints
        self.multiplier_player_indices = copy_players
        self.multiplier_block_indices = copy_blocks
        self.player_multiplier_indices = player_indices
        self.private_multiplier_indices = private_indices
        self.shared_player_multiplier_indices = shared_player_indices
        self.variational_multiplier_indices = variational_indices
        self.problem_id = problem_id


class FeedbackQuasiNashPlan(StrictModule):
    """Numerical acceptance policy for one fixed-active-set local KKT solve."""

    residual_tolerance: float = eqx.field(static=True)
    feasibility_tolerance: float = eqx.field(static=True)
    strict_complementarity_tolerance: float = eqx.field(static=True)
    curvature_tolerance: float = eqx.field(static=True)
    rank_relative_tolerance: float = eqx.field(static=True)
    rank_absolute_tolerance: float = eqx.field(static=True)
    maximum_condition: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        residual_tolerance: float = 1.0e-8,
        feasibility_tolerance: float = 1.0e-8,
        strict_complementarity_tolerance: float = 1.0e-10,
        curvature_tolerance: float = 1.0e-10,
        rank_relative_tolerance: float = 1.0e-10,
        rank_absolute_tolerance: float = 0.0,
        maximum_condition: float | None = None,
    ):
        residual = _positive(residual_tolerance, "residual_tolerance")
        feasibility = _nonnegative(feasibility_tolerance, "feasibility_tolerance")
        strict = _nonnegative(
            strict_complementarity_tolerance,
            "strict_complementarity_tolerance",
        )
        curvature = _nonnegative(curvature_tolerance, "curvature_tolerance")
        rank_relative = _nonnegative(rank_relative_tolerance, "rank_relative_tolerance")
        rank_absolute = _nonnegative(rank_absolute_tolerance, "rank_absolute_tolerance")
        condition = None if maximum_condition is None else float(maximum_condition)
        if condition is not None and (not math.isfinite(condition) or condition <= 1.0):
            raise ValueError(
                "maximum_condition must be finite and exceed one or be None."
            )
        payload = {
            "residual_tolerance": residual,
            "feasibility_tolerance": feasibility,
            "strict_complementarity_tolerance": strict,
            "curvature_tolerance": curvature,
            "rank_relative_tolerance": rank_relative,
            "rank_absolute_tolerance": rank_absolute,
            "maximum_condition": condition,
        }
        self.residual_tolerance = residual
        self.feasibility_tolerance = feasibility
        self.strict_complementarity_tolerance = strict
        self.curvature_tolerance = curvature
        self.rank_relative_tolerance = rank_relative
        self.rank_absolute_tolerance = rank_absolute
        self.maximum_condition = condition
        self.plan_id = f"feedback-quasi-nash-plan:{canonical_fingerprint(payload)}"


class FeedbackQuasiNashResult(StrictModule):
    """Piecewise-affine suggestion and fixed-active-set numerical evidence.

    ``multipliers`` and ``multiplier_feedback_gain`` use the stage-local copy
    order declared by ``problem.multiplier_*``.  No derivative through an active
    switch is supplied.  ``policy_authoritative`` is false whenever the local
    feedback sensitivity is not isolated, including a duplicate-row generic
    shared KKT.  Such a failure does not reject generic-GNE existence.  Even
    successful output remains a local model suggestion; it is not an exact
    nonlinear feedback-Nash or global-GNE certificate.
    """

    problem: ConstrainedFeedbackGameProblem
    plan: FeedbackQuasiNashPlan
    policy: LocalAffineGamePolicy
    multipliers: Array
    multiplier_feedback_gain: Array
    active_set: Array
    linearized_constraint_residuals: Array
    linearized_constraint_feedback_residuals: Array
    active_residuals: Array
    inactive_residuals: Array
    maximum_active_residuals: Array
    maximum_active_feedback_residuals: Array
    maximum_inactive_violations: Array
    stationarity_residuals: Array
    active_kkt_residuals: Array
    linear_relative_residuals: Array
    licq_ranks: Array
    active_multiplier_counts: Array
    kkt_ranks: Array
    rank_cutoffs: Array
    minimum_singular_values: Array
    maximum_singular_values: Array
    kkt_condition_numbers: Array
    own_minimum_curvatures: Array
    stage_status: Array
    linear_status: Array
    first_failed_stage: Array
    valid: Array
    status: Array
    policy_authoritative: Array
    unique_feedback_sensitivity_available: Array
    model_label: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    equilibrium_concept: str = eqx.field(static=True)
    fixed_active_set: bool = eqx.field(static=True)
    local_piecewise_affine_suggestion: bool = eqx.field(static=True)
    exact_nonlinear_feedback_nash_claim: bool = eqx.field(static=True)
    global_gne_claim: bool = eqx.field(static=True)
    off_trajectory_feasibility_claim: bool = eqx.field(static=True)
    active_switch_derivative_available: bool = eqx.field(static=True)
    generic_gne_existence_rejected: bool = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(FeedbackQuasiNashStatus.SUCCESS))

    @property
    def feedback_gain(self) -> Array:
        return self.policy.feedback_gain

    @property
    def feedforward(self) -> Array:
        return self.policy.feedforward

    @property
    def player_multipliers(self) -> tuple[Array, ...]:
        return tuple(
            jnp.take(
                self.multipliers,
                jnp.asarray(indices, dtype=jnp.int32),
                axis=-1,
            )
            for indices in self.problem.player_multiplier_indices
        )

    @property
    def private_multipliers(self) -> tuple[Array, ...]:
        return tuple(
            jnp.take(
                self.multipliers,
                jnp.asarray(indices, dtype=jnp.int32),
                axis=-1,
            )
            for indices in self.problem.private_multiplier_indices
        )

    @property
    def shared_player_multipliers(self) -> tuple[Array, ...]:
        return tuple(
            jnp.take(
                self.multipliers,
                jnp.asarray(indices, dtype=jnp.int32),
                axis=-1,
            )
            for indices in self.problem.shared_player_multiplier_indices
        )

    @property
    def variational_multipliers(self) -> Array:
        return jnp.take(
            self.multipliers,
            jnp.asarray(self.problem.variational_multiplier_indices, dtype=jnp.int32),
            axis=-1,
        )


def _optional_array(
    value: ArrayLike | None,
    shape: tuple[int, ...],
    name: str,
    /,
    *,
    default_dtype,
) -> Array:
    if value is None:
        return jnp.zeros(shape, dtype=default_dtype)
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _positive(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _nonnegative(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _multiplier_metadata(
    constraints: OpenLoopGameConstraints,
    block_slices: list[tuple[int, int]],
    /,
    *,
    variational: bool,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
]:
    players = constraints.partition.player_ids
    copy_constraints: list[int] = []
    copy_players: list[int] = []
    copy_blocks: list[int] = []
    player_indices: list[list[int]] = [[] for _ in players]
    private_indices: list[list[int]] = [[] for _ in players]
    shared_player_indices: list[list[int]] = [[] for _ in players]

    def append_block(block_index: int, player: int, generic: bool) -> None:
        start, stop = block_slices[block_index]
        for constraint_index in range(start, stop):
            multiplier_index = len(copy_constraints)
            copy_constraints.append(constraint_index)
            copy_players.append(player)
            copy_blocks.append(block_index)
            if player >= 0:
                player_indices[player].append(multiplier_index)
                if generic:
                    shared_player_indices[player].append(multiplier_index)
                else:
                    private_indices[player].append(multiplier_index)

    for player, player_id in enumerate(players):
        for block_index, block in enumerate(constraints.blocks):
            owned = (
                block.scope is not GameConstraintScope.SHARED and block.owner == player_id
            )
            shared_copy = (
                block.scope is GameConstraintScope.SHARED
                and not variational
                and player_id in block.participants
            )
            if owned or shared_copy:
                append_block(block_index, player, shared_copy)

    variational_indices: list[int] = []
    if variational:
        for block_index, block in enumerate(constraints.blocks):
            if block.scope is GameConstraintScope.SHARED:
                before = len(copy_constraints)
                append_block(block_index, -1, False)
                variational_indices.extend(range(before, len(copy_constraints)))

    return (
        tuple(copy_constraints),
        tuple(copy_players),
        tuple(copy_blocks),
        tuple(tuple(indices) for indices in player_indices),
        tuple(tuple(indices) for indices in private_indices),
        tuple(tuple(indices) for indices in shared_player_indices),
        tuple(variational_indices),
    )


def _all_finite(value: Array, payload_rank: int, /) -> Array:
    axes = tuple(range(value.ndim - payload_rank, value.ndim))
    return jnp.all(jnp.isfinite(value), axis=axes) if axes else jnp.isfinite(value)


def _masked_maximum(value: Array, mask: Array, /) -> Array:
    if value.shape[-1] == 0:
        return jnp.zeros(value.shape[:-1], dtype=value.dtype)
    return jnp.max(jnp.where(mask, value, 0.0), axis=-1)


def _case_where(mask: Array, on_true: Array, on_false: Array, /) -> Array:
    extra = on_true.ndim - mask.ndim
    return jnp.where(mask.reshape(mask.shape + (1,) * extra), on_true, on_false)


def _symmetric(matrix: Array, /) -> Array:
    return 0.5 * (matrix + jnp.swapaxes(matrix, -1, -2))


def _diagnostic_singular_values(
    matrix: Array,
    plan: FeedbackQuasiNashPlan,
    /,
    *,
    operator_id: str,
) -> Array:
    policy = FactorizationPolicy(
        "svd",
        rank=RankPolicy(
            relative_cutoff=plan.rank_relative_tolerance,
            absolute_cutoff=plan.rank_absolute_tolerance,
            require_full_rank=False,
        ),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
    )
    factorization = factorize(
        DenseLinearOperator(
            jax.lax.stop_gradient(matrix),
            operator_id=operator_id,
        ),
        policy,
    )
    return factorization.singular_values()


def _rank_evidence(
    singular_values: Array,
    plan: FeedbackQuasiNashPlan,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    if singular_values.shape[-1] == 0:
        zeros = jnp.zeros(singular_values.shape[:-1], dtype=singular_values.dtype)
        return zeros.astype(jnp.int32), zeros, zeros, zeros, jnp.ones_like(zeros)
    maximum = jnp.max(singular_values, axis=-1)
    minimum = jnp.min(singular_values, axis=-1)
    cutoff = plan.rank_absolute_tolerance + plan.rank_relative_tolerance * maximum
    rank = jnp.sum(singular_values > cutoff[..., None], axis=-1).astype(jnp.int32)
    condition = jnp.where(
        minimum > 0.0,
        maximum / minimum,
        jnp.asarray(jnp.inf, dtype=singular_values.dtype),
    )
    return rank, cutoff, minimum, maximum, condition


def solve_feedback_quasi_nash_model(
    problem: ConstrainedFeedbackGameProblem,
    /,
    *,
    plan: FeedbackQuasiNashPlan | None = None,
) -> FeedbackQuasiNashResult:
    """Solve one declared active set of the local control-value KKT equations.

    The returned policy is the affine branch associated with exactly
    ``problem.active_set``.  There is no active-set search, clipping, repair,
    pseudoinverse, or fallback solve.
    """

    if not isinstance(problem, ConstrainedFeedbackGameProblem):
        raise TypeError("problem must be a ConstrainedFeedbackGameProblem.")
    if plan is None:
        plan = FeedbackQuasiNashPlan()
    elif not isinstance(plan, FeedbackQuasiNashPlan):
        raise TypeError("plan must be a FeedbackQuasiNashPlan or None.")

    suggestion = problem.suggestion
    model = suggestion.model
    partition = model.partition
    cases = problem.case_shape
    case_rank = len(cases)
    horizon = problem.horizon
    n = problem.state_size
    m = problem.control_size
    multipliers_count = problem.num_multipliers
    dtype = problem.constraint_residuals.dtype
    owner = jnp.asarray(partition.control_owner, dtype=jnp.int32)
    rows = jnp.arange(m, dtype=jnp.int32)
    copy_indices = problem.multiplier_constraint_indices_array

    linear_policy = LinearSolvePolicy(
        DenseLU(),
        tolerance=TolerancePolicy(
            relative=plan.residual_tolerance,
            absolute=plan.residual_tolerance,
        ),
        rank=RankPolicy(require_full_rank=False),
        differentiation=DifferentiationPolicy("mathematical"),
        failure=FailurePolicy("status"),
    )
    rhs_layout = RHSLayout((n + 1,), names=("feedback-and-affine",))

    z_next = _symmetric(model.terminal_Q)
    linear_next = model.terminal_q
    constant_next = model.terminal_constants
    continuation_valid = suggestion.successful
    causal_stage = jnp.full(cases, -1, dtype=jnp.int32)
    causal_status = jnp.full(cases, int(FeedbackQuasiNashStatus.SUCCESS), dtype=jnp.int32)

    feedback_values: list[Array] = []
    feedforward_values: list[Array] = []
    multiplier_values: list[Array] = []
    multiplier_gain_values: list[Array] = []
    constraint_values: list[Array] = []
    constraint_feedback_values: list[Array] = []
    active_residual_values: list[Array] = []
    inactive_residual_values: list[Array] = []
    maximum_active_values: list[Array] = []
    maximum_active_feedback_values: list[Array] = []
    maximum_inactive_values: list[Array] = []
    stationarity_values: list[Array] = []
    active_kkt_values: list[Array] = []
    linear_relative_values: list[Array] = []
    licq_rank_values: list[Array] = []
    active_count_values: list[Array] = []
    kkt_rank_values: list[Array] = []
    rank_cutoff_values: list[Array] = []
    minimum_singular_values: list[Array] = []
    maximum_singular_values: list[Array] = []
    condition_values: list[Array] = []
    curvature_values: list[Array] = []
    stage_status_values: list[Array] = []
    linear_status_values: list[Array] = []

    for stage in range(horizon - 1, -1, -1):
        a = model.A[..., stage, :, :]
        b = model.B[..., stage, :, :]
        dynamics_bias = model.dynamics_bias[..., stage, :]
        q = _symmetric(model.Q[..., :, stage, :, :])
        r = _symmetric(model.R[..., :, stage, :, :])
        cross = model.N[..., :, stage, :, :]
        q_linear = model.q[..., :, stage, :]
        r_linear = model.r[..., :, stage, :]
        stage_constant = model.stage_constants[..., :, stage]
        residual = problem.constraint_residuals[..., stage, :]
        state_jacobian = problem.constraint_state_jacobians[..., stage, :, :]
        control_jacobian = problem.constraint_control_jacobians[..., stage, :, :]
        active = problem.active_set[..., stage, :]

        z_b = z_next @ b[..., None, :, :]
        b_transpose = jnp.swapaxes(b, -1, -2)[..., None, :, :]
        h = r + b_transpose @ z_b
        w = b_transpose @ z_next @ a[..., None, :, :] + jnp.swapaxes(cross, -1, -2)
        affine_next = (
            ein.contract("...pij,...j->...pi", z_next, dynamics_bias) + linear_next
        )
        g = r_linear + ein.contract("...ji,...pj->...pi", b, affine_next)
        coupled_h = h[..., owner, rows, :]
        coupled_w = w[..., owner, rows, :]
        coupled_g = g[..., owner, rows]

        own_curvatures = []
        for player, (start, stop) in enumerate(partition.control_slices):
            own = _symmetric(h[..., player, start:stop, start:stop])
            eigenvalues = jnp.linalg.eigvalsh(jax.lax.stop_gradient(own))
            own_curvatures.append(jnp.min(eigenvalues, axis=-1))
        own_minimum = jnp.stack(own_curvatures, axis=-1)

        copied_control = jnp.take(control_jacobian, copy_indices, axis=-2)
        copied_state = jnp.take(state_jacobian, copy_indices, axis=-2)
        copied_residual = jnp.take(residual, copy_indices, axis=-1)
        copied_active = jnp.take(active, copy_indices, axis=-1)
        stationarity_columns = jnp.swapaxes(
            copied_control, -1, -2
        ) * problem.stationarity_incidence.astype(dtype)
        constraint_rows = copied_control * problem.constraint_row_incidence.astype(dtype)
        licq_rows = copied_control * problem.licq_row_incidence.astype(dtype)
        stationarity_columns = jnp.where(
            copied_active[..., None, :], stationarity_columns, 0.0
        )
        active_constraint_rows = jnp.where(
            copied_active[..., :, None], constraint_rows, 0.0
        )
        inactive_diagonal = (
            jnp.eye(multipliers_count, dtype=dtype)
            * (~copied_active).astype(dtype)[..., None, :]
        )
        top = jnp.concatenate((coupled_h, stationarity_columns), axis=-1)
        bottom = jnp.concatenate((active_constraint_rows, inactive_diagonal), axis=-1)
        kkt = jnp.concatenate((top, bottom), axis=-2)
        stationarity_rhs = jnp.concatenate((-coupled_w, -coupled_g[..., None]), axis=-1)
        copied_constraint_rhs = jnp.concatenate(
            (-copied_state, -copied_residual[..., None]), axis=-1
        )
        copied_constraint_rhs = jnp.where(
            copied_active[..., :, None], copied_constraint_rhs, 0.0
        )
        rhs = jnp.concatenate((stationarity_rhs, copied_constraint_rhs), axis=-2)

        solve_result = solve(
            LinearSystem(
                DenseLinearOperator(
                    kkt,
                    operator_id="control-games:feedback-quasi-nash:lu",
                ),
                problem_id="control-games:feedback-quasi-nash:stage",
            ),
            rhs,
            policy=linear_policy,
            rhs_layout=rhs_layout,
        )
        solved = solve_result.value
        feedback = solved[..., :m, :n]
        feedforward = solved[..., :m, n]
        multiplier_gain = solved[..., m:, :n]
        multiplier = solved[..., m:, n]

        if multipliers_count:
            active_licq_rows = jnp.where(copied_active[..., :, None], licq_rows, 0.0)
            licq_singular = _diagnostic_singular_values(
                active_licq_rows,
                plan,
                operator_id="control-games:feedback-quasi-nash:licq-svd",
            )
            licq_rank, _, _, _, _ = _rank_evidence(licq_singular, plan)
        else:
            licq_rank = jnp.zeros(cases, dtype=jnp.int32)
        active_count = jnp.sum(copied_active, axis=-1).astype(jnp.int32)
        kkt_singular = _diagnostic_singular_values(
            kkt,
            plan,
            operator_id="control-games:feedback-quasi-nash:kkt-svd",
        )
        (
            kkt_rank,
            rank_cutoff,
            minimum_singular,
            maximum_singular,
            condition,
        ) = _rank_evidence(kkt_singular, plan)

        kkt_residual = kkt @ solved - rhs
        stationarity_residual = jnp.max(jnp.abs(kkt_residual[..., :m, :]), axis=(-2, -1))
        if multipliers_count:
            active_kkt_residual = jnp.max(
                jnp.where(
                    copied_active[..., :, None],
                    jnp.abs(kkt_residual[..., m:, :]),
                    0.0,
                ),
                axis=(-2, -1),
            )
        else:
            active_kkt_residual = jnp.zeros(cases, dtype=dtype)
        linear_status = solve_result.status.astype(jnp.int32)
        linear_valid = jnp.all(linear_status == int(LinearSolveStatus.SUCCESS), axis=-1)
        linear_relative = jnp.max(solve_result.diagnostics.relative_residual, axis=-1)

        physical_residual = residual + ein.contract(
            "...ij,...j->...i", control_jacobian, feedforward
        )
        physical_feedback_residual = state_jacobian + control_jacobian @ feedback
        inequality = ~problem.equality_mask
        active_physical = active
        inactive_inequality = ~active & inequality
        active_residual = jnp.where(active_physical, physical_residual, 0.0)
        inactive_residual = jnp.where(inactive_inequality, physical_residual, 0.0)
        maximum_active = _masked_maximum(jnp.abs(physical_residual), active_physical)
        if problem.num_constraints:
            feedback_row_norm = jnp.max(jnp.abs(physical_feedback_residual), axis=-1)
        else:
            feedback_row_norm = jnp.zeros(cases + (0,), dtype=dtype)
        maximum_active_feedback = _masked_maximum(feedback_row_norm, active_physical)
        maximum_inactive = _masked_maximum(
            jnp.maximum(physical_residual, 0.0), inactive_inequality
        )

        copied_inequality = ~problem.multiplier_equality_mask
        strict_complementarity = jnp.all(
            ~copied_active
            | ~copied_inequality
            | (multiplier > plan.strict_complementarity_tolerance),
            axis=-1,
        )
        input_finite = (
            _all_finite(a, 2)
            & _all_finite(b, 2)
            & _all_finite(dynamics_bias, 1)
            & _all_finite(q, 3)
            & _all_finite(r, 3)
            & _all_finite(cross, 3)
            & _all_finite(q_linear, 2)
            & _all_finite(r_linear, 2)
            & _all_finite(stage_constant, 1)
            & _all_finite(residual, 1)
            & _all_finite(state_jacobian, 2)
            & _all_finite(control_jacobian, 2)
        )
        licq_valid = licq_rank == active_count
        curvature_valid = jnp.all(own_minimum > plan.curvature_tolerance, axis=-1)
        kkt_rank_valid = kkt_rank == (m + multipliers_count)
        condition_valid = (
            jnp.ones_like(condition, dtype=bool)
            if plan.maximum_condition is None
            else condition <= plan.maximum_condition
        )
        output_finite = (
            _all_finite(feedback, 2)
            & _all_finite(feedforward, 1)
            & _all_finite(multiplier_gain, 2)
            & _all_finite(multiplier, 1)
            & jnp.isfinite(stationarity_residual)
            & jnp.isfinite(active_kkt_residual)
            & jnp.isfinite(maximum_active)
            & jnp.isfinite(maximum_active_feedback)
            & jnp.isfinite(maximum_inactive)
        )
        active_valid = (maximum_active <= plan.feasibility_tolerance) & (
            maximum_active_feedback <= plan.feasibility_tolerance
        )
        inactive_valid = maximum_inactive <= plan.feasibility_tolerance
        residual_valid = (
            (stationarity_residual <= plan.residual_tolerance)
            & (active_kkt_residual <= plan.residual_tolerance)
            & (linear_relative <= plan.residual_tolerance)
        )

        direct_status = jnp.where(
            ~input_finite,
            int(FeedbackQuasiNashStatus.NONFINITE_INPUT),
            jnp.where(
                ~licq_valid,
                int(FeedbackQuasiNashStatus.LICQ_FAILURE),
                jnp.where(
                    ~curvature_valid,
                    int(FeedbackQuasiNashStatus.OWN_CURVATURE_FAILURE),
                    jnp.where(
                        ~kkt_rank_valid,
                        int(FeedbackQuasiNashStatus.COUPLED_KKT_RANK_DEFICIENT),
                        jnp.where(
                            ~condition_valid,
                            int(FeedbackQuasiNashStatus.CONDITION_LIMIT_REACHED),
                            jnp.where(
                                ~linear_valid,
                                int(FeedbackQuasiNashStatus.LINEAR_SOLVE_FAILED),
                                jnp.where(
                                    ~output_finite,
                                    int(FeedbackQuasiNashStatus.NONFINITE_OUTPUT),
                                    jnp.where(
                                        ~active_valid,
                                        int(
                                            FeedbackQuasiNashStatus.ACTIVE_RESIDUAL_TOO_LARGE
                                        ),
                                        jnp.where(
                                            ~strict_complementarity,
                                            int(
                                                FeedbackQuasiNashStatus.STRICT_COMPLEMENTARITY_FAILURE
                                            ),
                                            jnp.where(
                                                ~inactive_valid,
                                                int(
                                                    FeedbackQuasiNashStatus.INACTIVE_CONSTRAINT_VIOLATION
                                                ),
                                                jnp.where(
                                                    ~residual_valid,
                                                    int(
                                                        FeedbackQuasiNashStatus.KKT_RESIDUAL_TOO_LARGE
                                                    ),
                                                    int(FeedbackQuasiNashStatus.SUCCESS),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        stage_status = jnp.where(
            continuation_valid,
            direct_status,
            int(FeedbackQuasiNashStatus.DEPENDENCY_FAILED),
        ).astype(jnp.int32)
        direct_valid = direct_status == int(FeedbackQuasiNashStatus.SUCCESS)
        stage_valid = continuation_valid & direct_valid
        direct_failure = continuation_valid & ~direct_valid
        causal_stage = jnp.where(direct_failure, stage, causal_stage)
        causal_status = jnp.where(direct_failure, direct_status, causal_status)

        closed_loop = a + b @ feedback
        closed_bias = dynamics_bias + ein.contract("...ij,...j->...i", b, feedforward)
        feedback_player = feedback[..., None, :, :]
        z_raw = (
            q
            + cross @ feedback_player
            + jnp.swapaxes(feedback, -1, -2)[..., None, :, :]
            @ jnp.swapaxes(cross, -1, -2)
            + jnp.swapaxes(feedback, -1, -2)[..., None, :, :] @ r @ feedback_player
            + jnp.swapaxes(closed_loop, -1, -2)[..., None, :, :]
            @ z_next
            @ closed_loop[..., None, :, :]
        )
        z_current = _symmetric(z_raw)
        linear_current = (
            q_linear
            + ein.contract("...pij,...j->...pi", cross, feedforward)
            + ein.contract(
                "...ji,...pj->...pi",
                feedback,
                ein.contract("...pij,...j->...pi", r, feedforward) + r_linear,
            )
            + ein.contract(
                "...ji,...pj->...pi",
                closed_loop,
                ein.contract("...pij,...j->...pi", z_next, closed_bias) + linear_next,
            )
        )
        constant_current = (
            stage_constant
            + constant_next
            + 0.5
            * ein.contract("...i,...pij,...j->...p", closed_bias, z_next, closed_bias)
            + ein.contract("...pi,...i->...p", linear_next, closed_bias)
            + 0.5 * ein.contract("...i,...pij,...j->...p", feedforward, r, feedforward)
            + ein.contract("...pi,...i->...p", r_linear, feedforward)
        )
        z_next = _case_where(stage_valid, z_current, jnp.full_like(z_current, jnp.nan))
        linear_next = _case_where(
            stage_valid, linear_current, jnp.full_like(linear_current, jnp.nan)
        )
        constant_next = _case_where(
            stage_valid, constant_current, jnp.full_like(constant_current, jnp.nan)
        )
        continuation_valid = stage_valid

        feedback_values.insert(0, feedback)
        feedforward_values.insert(0, feedforward)
        multiplier_values.insert(0, multiplier)
        multiplier_gain_values.insert(0, multiplier_gain)
        constraint_values.insert(0, physical_residual)
        constraint_feedback_values.insert(0, physical_feedback_residual)
        active_residual_values.insert(0, active_residual)
        inactive_residual_values.insert(0, inactive_residual)
        maximum_active_values.insert(0, maximum_active)
        maximum_active_feedback_values.insert(0, maximum_active_feedback)
        maximum_inactive_values.insert(0, maximum_inactive)
        stationarity_values.insert(0, stationarity_residual)
        active_kkt_values.insert(0, active_kkt_residual)
        linear_relative_values.insert(0, linear_relative)
        licq_rank_values.insert(0, licq_rank)
        active_count_values.insert(0, active_count)
        kkt_rank_values.insert(0, kkt_rank)
        rank_cutoff_values.insert(0, rank_cutoff)
        minimum_singular_values.insert(0, minimum_singular)
        maximum_singular_values.insert(0, maximum_singular)
        condition_values.insert(0, condition)
        curvature_values.insert(0, own_minimum)
        stage_status_values.insert(0, stage_status)
        linear_status_values.insert(0, linear_status)

    def stack(values: list[Array]) -> Array:
        return jnp.stack(values, axis=case_rank)

    feedback_gain = stack(feedback_values)
    feedforward = stack(feedforward_values)
    valid = continuation_valid & suggestion.successful
    status = jnp.where(
        suggestion.successful,
        causal_status,
        int(FeedbackQuasiNashStatus.INVALID_LOCAL_SUGGESTION),
    ).astype(jnp.int32)
    first_failed_stage = jnp.where(
        suggestion.successful,
        causal_stage,
        jnp.asarray(horizon, dtype=jnp.int32),
    )
    policy = LocalAffineGamePolicy(
        model.nominal_states,
        model.nominal_controls,
        feedback_gain,
        feedforward,
        feedforward_scale=jnp.asarray(1.0, dtype=dtype),
        time_grid=model.time_grid,
        input_layout=suggestion.policy.input_layout,
        partition=partition,
        case_shape=cases,
        policy_id=f"{problem.problem_id}:fixed-active-policy",
    )
    return FeedbackQuasiNashResult(
        problem=problem,
        plan=plan,
        policy=policy,
        multipliers=stack(multiplier_values),
        multiplier_feedback_gain=stack(multiplier_gain_values),
        active_set=problem.active_set,
        linearized_constraint_residuals=stack(constraint_values),
        linearized_constraint_feedback_residuals=stack(constraint_feedback_values),
        active_residuals=stack(active_residual_values),
        inactive_residuals=stack(inactive_residual_values),
        maximum_active_residuals=stack(maximum_active_values),
        maximum_active_feedback_residuals=stack(maximum_active_feedback_values),
        maximum_inactive_violations=stack(maximum_inactive_values),
        stationarity_residuals=stack(stationarity_values),
        active_kkt_residuals=stack(active_kkt_values),
        linear_relative_residuals=stack(linear_relative_values),
        licq_ranks=stack(licq_rank_values),
        active_multiplier_counts=stack(active_count_values),
        kkt_ranks=stack(kkt_rank_values),
        rank_cutoffs=stack(rank_cutoff_values),
        minimum_singular_values=stack(minimum_singular_values),
        maximum_singular_values=stack(maximum_singular_values),
        kkt_condition_numbers=stack(condition_values),
        own_minimum_curvatures=stack(curvature_values),
        stage_status=stack(stage_status_values),
        linear_status=stack(linear_status_values),
        first_failed_stage=first_failed_stage,
        valid=valid,
        status=status,
        policy_authoritative=valid,
        unique_feedback_sensitivity_available=valid,
        model_label=CONSTRAINED_FEEDBACK_QUASI_NASH_MODEL,
        method_id=_METHOD_ID,
        equilibrium_concept=problem.equilibrium_concept,
        fixed_active_set=True,
        local_piecewise_affine_suggestion=True,
        exact_nonlinear_feedback_nash_claim=False,
        global_gne_claim=False,
        off_trajectory_feasibility_claim=False,
        active_switch_derivative_available=False,
        generic_gne_existence_rejected=False,
    )


__all__ = [
    "CONSTRAINED_FEEDBACK_QUASI_NASH_MODEL",
    "ConstrainedFeedbackGameProblem",
    "FeedbackQuasiNashPlan",
    "FeedbackQuasiNashResult",
    "FeedbackQuasiNashStatus",
    "solve_feedback_quasi_nash_model",
]
