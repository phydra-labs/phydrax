#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..equations._mac_incompressible import CompiledMACIncompressibleDynamics
from ..linalg import DenseLinearOperator, FactorizationPolicy, factorize
from ..linalg._block_banded_least_squares import solve_block_banded_least_squares
from ..linalg._materialization import MaterializationPolicy
from ..linalg.backends._jax_dense import DenseSVDState
from ..typing import checked, parse
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult
from ._mac_adaptive import (
    MACAcceptedGridTrace,
    MACFrozenGridReplayPlan,
    MACFrozenGridReplayResult,
)


MACDerivativeMode: TypeAlias = Literal["smooth", "branchwise", "unsupported"]
MACNeutralMode: TypeAlias = Literal["none", "flow"]
MACShadowingSolveRoute: TypeAlias = Literal["auto", "block-banded", "dense-svd"]
_SegmentCarry: TypeAlias = tuple[Array, Array, Array, Array]
_SegmentInputs: TypeAlias = tuple[Array, Array, Array, Array]
_SegmentOutput: TypeAlias = tuple[Array, Array, Array, Array, Array, Array]
_StepCarry: TypeAlias = tuple[Array, Array, Array]


class MACShadowingStatus(IntEnum):
    SUCCESS = 0
    REPLAY_FAILED = 1
    DERIVATIVE_UNSUPPORTED = 2
    NONFINITE = 3
    TANGENT_RANK_DEFICIENT = 4
    ILL_CONDITIONED = 5
    CONTINUITY_RESIDUAL_FAILED = 6
    CONVERGENCE_FAILED = 7
    NEUTRAL_DIRECTION_FAILED = 8


def _dense_shadowing_workspace_bytes(
    count: int, rows: int, width: int, itemsize: int, regularization: float
) -> int:
    columns = count * width
    weighted_rows = count * rows + (count - 1) * width
    solve_rows = weighted_rows + (columns if regularization > 0.0 else 0)
    return (8 * solve_rows * columns + 4 * columns**2) * itemsize


def _resolve_shadowing_solve_route(
    route: MACShadowingSolveRoute, required_bytes: int, budget_bytes: int
) -> MACShadowingSolveRoute:
    match route:
        case "auto":
            return "dense-svd" if required_bytes <= budget_bytes else "block-banded"
        case "block-banded" | "dense-svd":
            return route
        case _:
            raise ValueError("Invalid MAC shadowing solve route.")


def _dense_shadowing_least_squares(
    samples: Array,
    targets: Array,
    factors: Array,
    offsets: Array,
    continuity_weight: float,
    regularization: float,
    workspace_bytes: int,
) -> tuple[Array, Array, Array, Array]:
    count, rows, width = samples.shape
    columns = count * width
    weighted_rows = count * rows + (count - 1) * width
    solve_rows = weighted_rows + (columns if regularization > 0.0 else 0)
    required_bytes = _dense_shadowing_workspace_bytes(
        count, rows, width, samples.dtype.itemsize, regularization
    )
    if required_bytes > workspace_bytes:
        raise ValueError(
            f"MAC dense-svd workspace requires {required_bytes} bytes, exceeding {workspace_bytes}; select block-banded."
        )
    segment_eye = jnp.eye(count, dtype=samples.dtype)
    sample_design = contract("smk,sr->smrk", samples, segment_eye).reshape((-1, columns))
    continuity = -contract("eij,er->eirj", factors[:-1], segment_eye[:-1]) + contract(
        "ij,er->eirj", jnp.eye(width, dtype=samples.dtype), segment_eye[1:]
    )
    weighted = jnp.concatenate(
        (
            sample_design,
            continuity_weight * continuity.reshape(((count - 1) * width, columns)),
        ),
        axis=0,
    )
    target = jnp.concatenate(
        (targets.reshape((-1,)), continuity_weight * offsets[:-1].reshape((-1,)))
    )
    design = weighted
    if regularization > 0.0:
        design = jnp.concatenate(
            (
                weighted,
                jnp.sqrt(jnp.asarray(regularization, dtype=samples.dtype))
                * jnp.eye(columns, dtype=samples.dtype),
            ),
            axis=0,
        )
        target = jnp.concatenate((target, jnp.zeros((columns,), dtype=samples.dtype)))
    policy = FactorizationPolicy(
        "svd",
        materialization=MaterializationPolicy(
            max_entries=solve_rows * columns, max_bytes=workspace_bytes
        ),
    )
    factor = factorize(DenseLinearOperator(design), policy)
    state = factor.prepared_solve.state
    if not isinstance(state, DenseSVDState):
        raise TypeError("MAC dense-svd requires native dense SVD factors.")
    tolerance = (
        jnp.finfo(samples.dtype).eps
        * max(design.shape)
        * jnp.maximum(state.singular_values[0], 1.0)
    )
    inverse = jnp.where(
        state.singular_values > tolerance, 1.0 / state.singular_values, 0.0
    )
    coefficients = state.vh.T @ (inverse * (state.u.T @ target))
    diagnostic = state.singular_values
    if regularization > 0.0:
        diagnostic_factor = factorize(DenseLinearOperator(weighted), policy)
        diagnostic_state = diagnostic_factor.prepared_solve.state
        if not isinstance(diagnostic_state, DenseSVDState):
            raise TypeError("MAC diagnostics require native dense SVD factors.")
        diagnostic = diagnostic_state.singular_values
    cutoff = (
        jnp.finfo(samples.dtype).eps
        * max(weighted.shape)
        * jnp.maximum(diagnostic[0], 1.0)
    )
    full_rank = jnp.sum(diagnostic > cutoff, dtype=jnp.int32) == columns
    minimum = jnp.min(diagnostic)
    condition = jnp.where(
        full_rank,
        diagnostic[0] / jnp.maximum(minimum, jnp.finfo(samples.dtype).tiny),
        jnp.inf,
    )
    return coefficients.reshape((count, width)), full_rank, minimum, condition


def _zero_tangent(value: Any, /) -> Any:
    return jax.tree.map(
        lambda leaf: jnp.zeros_like(leaf) if eqx.is_inexact_array(leaf) else None,
        value,
        is_leaf=lambda leaf: leaf is None,
    )


def _tree_finite(value: Any, /) -> Array:
    leaves = tuple(leaf for leaf in jax.tree.leaves(value) if eqx.is_inexact_array(leaf))
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)))


def _validate_transition(result: FixedStepResult, reference: Array, /) -> None:
    if not isinstance(result, FixedStepResult):
        raise TypeError("MAC sensitivity transition must return FixedStepResult.")
    if (
        not eqx.is_array(result.accepted_state)
        or result.accepted_state.shape != reference.shape
        or result.accepted_state.dtype != reference.dtype
    ):
        raise TypeError("MAC sensitivity transition changed state shape or dtype.")


class MACReplayCertification(StrictModule):
    """Primal identity and derivative-semantics evidence for one frozen replay."""

    supported: Array
    branchwise: Array
    decisions_frozen: Array
    grid_valid: Array
    replay_completed: Array
    primal_matches: Array
    maximum_absolute_error: Array
    tolerance: Array
    finite: Array
    successful: Array
    derivative_mode: MACDerivativeMode = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    source_plan_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


class MACTerminalJVPResult(StrictModule):
    terminal_state: Array
    terminal_tangent: Array
    replay: MACFrozenGridReplayResult
    certification: MACReplayCertification
    finite: Array
    successful: Array
    sensitivity_id: str = eqx.field(static=True)


class MACTerminalVJPResult(StrictModule):
    terminal_state: Array
    initial_state_cotangent: Array
    args_cotangent: Any
    replay: MACFrozenGridReplayResult
    certification: MACReplayCertification
    finite: Array
    successful: Array
    sensitivity_id: str = eqx.field(static=True)


class MACFixedGridSensitivityPlan(StrictModule):
    """Checkpointed discrete differentiation on one frozen accepted grid."""

    replay_plan: MACFrozenGridReplayPlan
    derivative_mode: MACDerivativeMode = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    sensitivity_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        method: AbstractFixedStepMethod,
        /,
        *,
        derivative_mode: MACDerivativeMode,
        checkpointing: Literal["full", "step", "block"] = "block",
        block_size: int | None = 16,
        absolute_tolerance: float = 1e-8,
        relative_tolerance: float = 1e-6,
    ) -> None:
        derivative_mode = parse(derivative_mode, MACDerivativeMode, "derivative_mode")
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if (
            not isfinite(absolute)
            or absolute < 0.0
            or not isfinite(relative)
            or relative < 0.0
        ):
            raise ValueError("MAC replay tolerances must be finite and nonnegative.")
        replay = MACFrozenGridReplayPlan(
            dynamics,
            method,
            checkpointing=checkpointing,
            block_size=block_size,
        )
        self.replay_plan = replay
        self.derivative_mode = derivative_mode
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.sensitivity_id = canonical_fingerprint(
            {
                "kind": "mac-fixed-grid-sensitivity",
                "replay": replay.replay_id,
                "derivative_mode": derivative_mode,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
            }
        )

    def rollout(
        self,
        initial_state: Array,
        grid: MACAcceptedGridTrace,
        args: Any = None,
        /,
    ) -> MACFrozenGridReplayResult:
        return self.replay_plan.replay(initial_state, grid, args)

    @checked
    def certify(
        self,
        replay: MACFrozenGridReplayResult,
        reference_final_state: Array | None = None,
        /,
    ) -> MACReplayCertification:
        if replay.replay_id != self.replay_plan.replay_id:
            raise ValueError("MAC replay plan identity changed before certification.")
        if reference_final_state is None:
            error = jnp.zeros((), dtype=replay.final_state.dtype)
            tolerance = jnp.asarray(
                self.absolute_tolerance, dtype=replay.final_state.dtype
            )
            matches = jnp.asarray(True)
        else:
            reference = self.replay_plan.dynamics.validate_state(reference_final_state)
            error = jnp.max(jnp.abs(replay.final_state - reference))
            scale = jnp.maximum(jnp.max(jnp.abs(reference)), 1.0)
            tolerance = (
                jnp.asarray(self.absolute_tolerance, dtype=replay.final_state.dtype)
                + jnp.asarray(self.relative_tolerance, dtype=replay.final_state.dtype)
                * scale
            )
            matches = error <= tolerance
        supported = jnp.asarray(self.derivative_mode != "unsupported")
        branchwise = jnp.asarray(self.derivative_mode == "branchwise")
        finite = replay.finite & jnp.isfinite(error) & jnp.isfinite(tolerance)
        successful = supported & replay.grid_valid & replay.completed & matches & finite
        return MACReplayCertification(
            supported,
            branchwise,
            jnp.asarray(True),
            replay.grid_valid,
            replay.completed,
            matches,
            error,
            tolerance,
            finite,
            successful,
            self.derivative_mode,
            replay.replay_id,
            replay.source_plan_id,
            replay.dynamics_id,
            replay.method_id,
        )

    def terminal_jvp(
        self,
        initial_state: Array,
        grid: MACAcceptedGridTrace,
        args: Any = None,
        /,
        *,
        initial_tangent: Array | None = None,
        args_tangent: Any = None,
        reference_final_state: Array | None = None,
    ) -> MACTerminalJVPResult:
        state0 = self.replay_plan.dynamics.validate_state(initial_state)
        state_tangent = (
            jnp.zeros_like(state0)
            if initial_tangent is None
            else self.replay_plan.dynamics.validate_state(initial_tangent)
        )
        parameter_tangent = _zero_tangent(args) if args_tangent is None else args_tangent
        replay = self.rollout(state0, grid, args)
        certification = self.certify(replay, reference_final_state)
        if self.derivative_mode == "unsupported":
            tangent = jnp.zeros_like(state0)
        else:
            _, tangent = eqx.filter_jvp(
                lambda state, parameters: (
                    self.rollout(state, grid, parameters).final_state
                ),
                (state0, args),
                (state_tangent, parameter_tangent),
            )
        finite = jnp.all(jnp.isfinite(tangent))
        successful = certification.successful & finite
        return MACTerminalJVPResult(
            replay.final_state,
            tangent,
            replay,
            certification,
            finite,
            successful,
            self.sensitivity_id,
        )

    def terminal_vjp(
        self,
        initial_state: Array,
        grid: MACAcceptedGridTrace,
        terminal_cotangent: Array,
        args: Any = None,
        /,
        *,
        reference_final_state: Array | None = None,
    ) -> MACTerminalVJPResult:
        state0 = self.replay_plan.dynamics.validate_state(initial_state)
        cotangent = self.replay_plan.dynamics.validate_state(terminal_cotangent)
        replay = self.rollout(state0, grid, args)
        certification = self.certify(replay, reference_final_state)
        if self.derivative_mode == "unsupported":
            state_cotangent = jnp.zeros_like(state0)
            args_cotangent = _zero_tangent(args)
        else:
            _, pullback = eqx.filter_vjp(
                lambda state, parameters: (
                    self.rollout(state, grid, parameters).final_state
                ),
                state0,
                args,
            )
            state_cotangent, args_cotangent = pullback(cotangent)
        finite = jnp.all(jnp.isfinite(state_cotangent)) & _tree_finite(args_cotangent)
        successful = certification.successful & finite
        return MACTerminalVJPResult(
            replay.final_state,
            state_cotangent,
            args_cotangent,
            replay,
            certification,
            finite,
            successful,
            self.sensitivity_id,
        )


class MACShadowingSensitivityResult(StrictModule):
    sensitivity: Array
    response_samples: Array
    shadow_tangent: Array
    time_dilation: Array
    segment_coefficients: Array
    inhomogeneous_tangents: Array
    homogeneous_bases: Array
    qr_factors: Array
    continuity_residual: Array
    continuity_residual_norm: Array
    least_squares_residual_norm: Array
    condition_number: Array
    minimum_singular_value: Array
    qr_condition_number: Array
    neutral_minimum_norm: Array
    convergence_error: Array
    convergence_evaluated: Array
    converged: Array
    conditioned: Array
    residual_converged: Array
    finite: Array
    successful: Array
    status: Array
    replay: MACFrozenGridReplayResult
    certification: MACReplayCertification
    neutral_mode: MACNeutralMode = eqx.field(static=True)
    solve_route: MACShadowingSolveRoute = eqx.field(static=True)
    observable_id: str = eqx.field(static=True)
    shadowing_id: str = eqx.field(static=True)


class MACSegmentedShadowingPlan(StrictModule):
    """QR-stabilized segmented LSS/NILSS-style MAC sensitivity solve."""

    sensitivity_plan: MACFixedGridSensitivityPlan
    segment_length: int = eqx.field(static=True)
    tangent_dimension: int = eqx.field(static=True)
    neutral_mode: MACNeutralMode = eqx.field(static=True)
    continuity_weight: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    solve_route: MACShadowingSolveRoute = eqx.field(static=True)
    dense_workspace_bytes: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    qr_tolerance: float = eqx.field(static=True)
    neutral_tolerance: float = eqx.field(static=True)
    convergence_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        sensitivity_plan: MACFixedGridSensitivityPlan,
        /,
        *,
        segment_length: int,
        tangent_dimension: int,
        neutral_mode: MACNeutralMode = "flow",
        continuity_weight: float = 100.0,
        regularization: float = 0.0,
        solve_route: MACShadowingSolveRoute = "auto",
        dense_workspace_bytes: int = 512 * 1024 * 1024,
        residual_tolerance: float = 1e-6,
        condition_limit: float = 1e10,
        qr_tolerance: float = 1e-10,
        neutral_tolerance: float = 1e-12,
        convergence_tolerance: float = 1e-3,
    ) -> None:
        length = int(segment_length)
        dimension = int(tangent_dimension)
        state_size = sensitivity_plan.replay_plan.dynamics.state_shape[0]
        if length <= 0 or dimension <= 0 or dimension > state_size:
            raise ValueError("MAC shadowing segment and tangent dimensions are invalid.")
        neutral_mode = parse(neutral_mode, MACNeutralMode, "neutral_mode")
        solve_route = parse(solve_route, MACShadowingSolveRoute, "solve_route")
        if isinstance(dense_workspace_bytes, bool) or not isinstance(
            dense_workspace_bytes, int
        ):
            raise TypeError("dense_workspace_bytes must be an integer.")
        if dense_workspace_bytes < 1:
            raise ValueError("dense_workspace_bytes must be positive.")
        if neutral_mode == "flow" and dimension >= state_size:
            raise ValueError(
                "Flow-neutral shadowing needs tangent_dimension below state dimension."
            )
        numeric = (
            float(continuity_weight),
            float(regularization),
            float(residual_tolerance),
            float(condition_limit),
            float(qr_tolerance),
            float(neutral_tolerance),
            float(convergence_tolerance),
        )
        if (
            any(not isfinite(value) for value in numeric)
            or numeric[0] <= 0.0
            or numeric[1] < 0.0
            or any(value <= 0.0 for value in numeric[2:])
        ):
            raise ValueError("MAC shadowing numerical gates are invalid.")
        self.sensitivity_plan = sensitivity_plan
        self.segment_length = length
        self.tangent_dimension = dimension
        self.neutral_mode = neutral_mode
        self.continuity_weight = numeric[0]
        self.regularization = numeric[1]
        self.solve_route = solve_route
        self.dense_workspace_bytes = dense_workspace_bytes
        self.residual_tolerance = numeric[2]
        self.condition_limit = numeric[3]
        self.qr_tolerance = numeric[4]
        self.neutral_tolerance = numeric[5]
        self.convergence_tolerance = numeric[6]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-segmented-shadowing",
                "sensitivity": sensitivity_plan.sensitivity_id,
                "segment_length": length,
                "tangent_dimension": dimension,
                "neutral_mode": neutral_mode,
                "continuity_weight": numeric[0],
                "regularization": numeric[1],
                "solve_route": solve_route,
                "dense_workspace_bytes": dense_workspace_bytes,
                "residual_tolerance": numeric[2],
                "condition_limit": numeric[3],
                "qr_tolerance": numeric[4],
                "neutral_tolerance": numeric[5],
                "convergence_tolerance": numeric[6],
            }
        )

    def solve(
        self,
        initial_state: Array,
        grid: MACAcceptedGridTrace,
        args: Any,
        parameter_tangent: Any,
        observable: Callable[[Array, Array, Any], Array],
        /,
        *,
        observable_id: str,
        initial_basis: Array | None = None,
        comparison_sensitivity: Array | None = None,
    ) -> MACShadowingSensitivityResult:
        if not callable(observable):
            raise TypeError("MAC shadowing observable must be callable.")
        observable_id_ = str(observable_id)
        if not observable_id_:
            raise ValueError("observable_id must be non-empty.")
        replay = self.sensitivity_plan.rollout(initial_state, grid, args)
        certification = self.sensitivity_plan.certify(replay)
        parameter_direction = (
            _zero_tangent(args) if parameter_tangent is None else parameter_tangent
        )
        state_size = self.sensitivity_plan.replay_plan.dynamics.state_shape[0]
        capacity = grid.capacity
        if capacity % self.segment_length:
            raise ValueError(
                "Frozen MAC grid capacity must be divisible by segment_length."
            )
        segment_count = capacity // self.segment_length
        tangent_dimension = self.tangent_dimension
        dtype = replay.final_state.dtype
        required_bytes = _dense_shadowing_workspace_bytes(
            segment_count,
            self.segment_length * state_size,
            tangent_dimension,
            dtype.itemsize,
            self.regularization,
        )
        solve_route = _resolve_shadowing_solve_route(
            self.solve_route, required_bytes, self.dense_workspace_bytes
        )
        shadowing_id = canonical_fingerprint(
            {
                "kind": "mac-shadowing-result",
                "plan": self.plan_id,
                "observable": observable_id_,
                "grid_source": grid.source_plan_id,
                "solve_route": solve_route,
            }
        )
        if initial_basis is None:
            basis0 = jnp.eye(state_size, tangent_dimension, dtype=dtype)
        else:
            basis0 = jnp.asarray(initial_basis, dtype=dtype)
            if basis0.shape != (state_size, tangent_dimension):
                raise ValueError("initial_basis has the wrong MAC tangent shape.")
        if self.sensitivity_plan.derivative_mode == "unsupported":
            nan = jnp.asarray(jnp.nan, dtype=dtype)
            zero = jnp.zeros
            status = jnp.where(
                replay.completed,
                int(MACShadowingStatus.DERIVATIVE_UNSUPPORTED),
                int(MACShadowingStatus.REPLAY_FAILED),
            ).astype(jnp.int32)
            return MACShadowingSensitivityResult(
                nan,
                zero((capacity,), dtype=dtype),
                zero((capacity, state_size), dtype=dtype),
                zero((capacity,), dtype=dtype),
                zero((segment_count, tangent_dimension), dtype=dtype),
                zero(
                    (
                        segment_count,
                        self.segment_length,
                        state_size,
                    ),
                    dtype=dtype,
                ),
                zero(
                    (
                        segment_count,
                        self.segment_length,
                        state_size,
                        tangent_dimension,
                    ),
                    dtype=dtype,
                ),
                zero(
                    (segment_count, tangent_dimension, tangent_dimension),
                    dtype=dtype,
                ),
                zero(
                    (max(segment_count - 1, 0), tangent_dimension),
                    dtype=dtype,
                ),
                nan,
                nan,
                jnp.asarray(jnp.inf, dtype=dtype),
                zero((), dtype=dtype),
                jnp.asarray(jnp.inf, dtype=dtype),
                nan,
                nan,
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(False),
                replay.finite,
                jnp.asarray(False),
                status,
                replay,
                certification,
                self.neutral_mode,
                solve_route,
                observable_id_,
                shadowing_id,
            )
        states = replay.states[:-1].reshape(
            (segment_count, self.segment_length, state_size)
        )
        times = jax.lax.stop_gradient(grid.times[:-1]).reshape(
            (segment_count, self.segment_length)
        )
        step_sizes = jax.lax.stop_gradient(grid.step_sizes).reshape(
            (segment_count, self.segment_length)
        )
        step_indices = jnp.arange(capacity, dtype=jnp.int32).reshape(
            (segment_count, self.segment_length)
        )
        dynamics = self.sensitivity_plan.replay_plan.dynamics
        method = self.sensitivity_plan.replay_plan.method
        neutral_tolerance = jnp.asarray(self.neutral_tolerance, dtype=dtype)
        qr_tolerance = jnp.asarray(self.qr_tolerance, dtype=dtype)

        def flow_direction(time: Array, state: Array) -> Array:
            return dynamics(time, state, args)

        def project_neutral(
            basis: Array, tangent: Array, time: Array, state: Array
        ) -> tuple[Array, Array, Array, Array]:
            if self.neutral_mode == "none":
                return (
                    basis,
                    tangent,
                    jnp.asarray(jnp.inf, dtype=dtype),
                    jnp.asarray(True),
                )
            flow = flow_direction(time, state)
            norm_squared = jnp.vdot(flow, flow).real
            safe = jnp.maximum(norm_squared, neutral_tolerance**2)
            basis_ = basis - flow[:, None] * (flow @ basis)[None, :] / safe
            tangent_ = tangent - flow * jnp.vdot(flow, tangent).real / safe
            valid = jnp.isfinite(norm_squared) & (norm_squared > neutral_tolerance**2)
            return basis_, tangent_, jnp.sqrt(jnp.maximum(norm_squared, 0.0)), valid

        basis0, tangent0, initial_neutral_norm, initial_neutral_valid = project_neutral(
            basis0,
            jnp.zeros((state_size,), dtype=dtype),
            times[0, 0],
            states[0, 0],
        )
        basis0, initial_r = jnp.linalg.qr(basis0, mode="reduced")
        initial_diagonal = jnp.diag(initial_r)
        initial_sign = jnp.where(initial_diagonal < 0.0, -1.0, 1.0)
        basis0 = basis0 * initial_sign[None, :]
        initial_rank_valid = jnp.all(jnp.abs(initial_diagonal) > qr_tolerance)

        def transition(
            index: Array,
            time: Array,
            state: Array,
            step_size: Array,
            parameters: object,
        ) -> Array:
            result = method.step(index, time, state, step_size, parameters)
            _validate_transition(result, state)
            return result.accepted_state

        def run_segment(
            carry: _SegmentCarry, segment_inputs: _SegmentInputs
        ) -> tuple[_SegmentCarry, _SegmentOutput]:
            basis, inhomogeneous, previous_ok, minimum_neutral = carry
            segment_states, segment_times, segment_steps, segment_indices = segment_inputs

            def run_step(
                local_carry: _StepCarry, local_inputs: _SegmentInputs
            ) -> tuple[_StepCarry, tuple[Array, Array]]:
                current_basis, current_inhomogeneous, local_ok = local_carry
                state, time, step_size, index = local_inputs

                def homogeneous_tangent(column: Array) -> Array:
                    _, tangent = eqx.filter_jvp(
                        lambda current: transition(index, time, current, step_size, args),
                        (state,),
                        (column,),
                    )
                    return tangent

                propagated_basis = jax.vmap(homogeneous_tangent, in_axes=1, out_axes=1)(
                    current_basis
                )
                _, propagated_inhomogeneous = eqx.filter_jvp(
                    lambda current, parameters: transition(
                        index, time, current, step_size, parameters
                    ),
                    (state, args),
                    (current_inhomogeneous, parameter_direction),
                )
                finite = jnp.all(jnp.isfinite(propagated_basis)) & jnp.all(
                    jnp.isfinite(propagated_inhomogeneous)
                )
                return (
                    propagated_basis,
                    propagated_inhomogeneous,
                    local_ok & finite,
                ), (current_basis, current_inhomogeneous)

            (basis_end, tangent_end, segment_ok), samples = jax.lax.scan(
                run_step,
                (basis, inhomogeneous, previous_ok),
                (segment_states, segment_times, segment_steps, segment_indices),
            )
            end_time = segment_times[-1] + segment_steps[-1]
            end_state = transition(
                segment_indices[-1],
                segment_times[-1],
                segment_states[-1],
                segment_steps[-1],
                args,
            )
            basis_end, tangent_end, neutral_norm, neutral_valid = project_neutral(
                basis_end, tangent_end, end_time, end_state
            )
            next_basis, factor = jnp.linalg.qr(basis_end, mode="reduced")
            diagonal = jnp.diag(factor)
            signs = jnp.where(diagonal < 0.0, -1.0, 1.0)
            next_basis = next_basis * signs[None, :]
            factor = signs[:, None] * factor
            rank_valid = jnp.all(jnp.abs(diagonal) > qr_tolerance)
            offset = next_basis.T @ tangent_end
            next_inhomogeneous = tangent_end - next_basis @ offset
            next_ok = segment_ok & rank_valid & neutral_valid
            next_minimum = jnp.minimum(minimum_neutral, neutral_norm)
            segment_output = (
                samples[0],
                samples[1],
                factor,
                offset,
                next_ok,
                neutral_norm,
            )
            return (
                next_basis,
                next_inhomogeneous,
                next_ok,
                next_minimum,
            ), segment_output

        initial_minimum = initial_neutral_norm
        _, segment_outputs = jax.lax.scan(
            run_segment,
            (
                basis0,
                tangent0,
                initial_neutral_valid & initial_rank_valid,
                initial_minimum,
            ),
            (states, times, step_sizes, step_indices),
        )
        (
            homogeneous_bases,
            inhomogeneous_tangents,
            qr_factors,
            offsets,
            tangent_valid_by_segment,
            segment_neutral_norms,
        ) = segment_outputs
        flat_states = states.reshape((capacity, state_size))
        flat_times = times.reshape((capacity,))
        flat_steps = step_sizes.reshape((capacity,))
        if self.neutral_mode == "flow":
            flow_directions = jax.vmap(flow_direction)(flat_times, flat_states).reshape(
                (segment_count, self.segment_length, state_size)
            )
            flow_norm_squared = jnp.sum(flow_directions**2, axis=-1)
            safe_flow_norm = jnp.maximum(flow_norm_squared, neutral_tolerance**2)
            basis_projection = contract(
                "sln,slnk->slk", flow_directions, homogeneous_bases
            )
            tangent_projection = contract(
                "slm,slm->sl", flow_directions, inhomogeneous_tangents
            )
            projected_bases = (
                homogeneous_bases
                - flow_directions[..., None]
                * (basis_projection / safe_flow_norm[..., None])[..., None, :]
            )
            projected_inhomogeneous = (
                inhomogeneous_tangents
                - flow_directions * (tangent_projection / safe_flow_norm)[..., None]
            )
            neutral_valid = jnp.all(
                jnp.isfinite(flow_norm_squared)
                & (flow_norm_squared > neutral_tolerance**2)
            )
            neutral_minimum = jnp.minimum(
                jnp.min(jnp.sqrt(jnp.maximum(flow_norm_squared, 0.0))),
                jnp.min(segment_neutral_norms),
            )
        else:
            flow_directions = jnp.zeros_like(homogeneous_bases[..., 0])
            flow_norm_squared = jnp.ones(
                (segment_count, self.segment_length), dtype=dtype
            )
            projected_bases = homogeneous_bases
            projected_inhomogeneous = inhomogeneous_tangents
            neutral_valid = jnp.asarray(True)
            neutral_minimum = jnp.asarray(jnp.inf, dtype=dtype)

        sample_blocks = projected_bases.reshape((segment_count, -1, tangent_dimension))
        sample_targets = -projected_inhomogeneous.reshape((segment_count, -1))
        weight = jnp.asarray(self.continuity_weight, dtype=dtype)
        match solve_route:
            case "block-banded":
                banded = solve_block_banded_least_squares(
                    sample_blocks,
                    sample_targets,
                    -weight * qr_factors[:-1],
                    jnp.broadcast_to(
                        weight * jnp.eye(tangent_dimension, dtype=dtype),
                        qr_factors[:-1].shape,
                    ),
                    weight * offsets[:-1],
                    regularization=self.regularization,
                )
                coefficients = banded.value
                full_rank = (
                    banded.rank_evidence.lower_bound == segment_count * tangent_dimension
                )
                minimum_singular = banded.minimum_singular_value
                condition_number = banded.condition_number
            case "dense-svd":
                coefficients, full_rank, minimum_singular, condition_number = (
                    _dense_shadowing_least_squares(
                        sample_blocks,
                        sample_targets,
                        qr_factors,
                        offsets,
                        self.continuity_weight,
                        self.regularization,
                        self.dense_workspace_bytes,
                    )
                )
            case _:
                raise ValueError("Invalid MAC shadowing solve route.")
        if segment_count > 1:
            continuity_residual = (
                contract("sij,sj->si", qr_factors[:-1], coefficients[:-1])
                + offsets[:-1]
                - coefficients[1:]
            )
            continuity_scale = jnp.maximum(jnp.linalg.norm(offsets[:-1]), 1.0)
            continuity_residual_norm = (
                jnp.linalg.norm(continuity_residual) / continuity_scale
            )
        else:
            continuity_residual = jnp.zeros((0, tangent_dimension), dtype=dtype)
            continuity_residual_norm = jnp.zeros((), dtype=dtype)
        least_squares_residual = (
            contract("smk,sk->sm", sample_blocks, coefficients) - sample_targets
        )
        least_squares_residual_norm = jnp.linalg.norm(
            least_squares_residual
        ) / jnp.maximum(jnp.linalg.norm(sample_targets), 1.0)
        qr_singular = jax.vmap(lambda value: jnp.linalg.svd(value, compute_uv=False))(
            qr_factors
        )
        qr_condition = jnp.max(
            qr_singular[:, 0] / jnp.maximum(qr_singular[:, -1], jnp.finfo(dtype).tiny)
        )
        raw_tangent = inhomogeneous_tangents + contract(
            "slnk,sk->sln", homogeneous_bases, coefficients
        )
        if self.neutral_mode == "flow":
            time_dilation = contract(
                "sln,sln->sl", flow_directions, raw_tangent
            ) / jnp.maximum(flow_norm_squared, neutral_tolerance**2)
            shadow_tangent = raw_tangent - time_dilation[..., None] * flow_directions
        else:
            time_dilation = jnp.zeros((segment_count, self.segment_length), dtype=dtype)
            shadow_tangent = raw_tangent

        def observable_response(
            time: Array, state: Array, tangent: Array
        ) -> tuple[Array, Array]:
            primal, response = eqx.filter_jvp(
                lambda current, parameters: jnp.asarray(
                    observable(time, current, parameters)
                ).reshape(()),
                (state, args),
                (tangent, parameter_direction),
            )
            return primal, response

        observable_values, directional_samples = jax.vmap(observable_response)(
            flat_times,
            flat_states,
            shadow_tangent.reshape((capacity, state_size)),
        )
        total_time = jnp.sum(flat_steps)
        safe_total_time = jnp.maximum(total_time, jnp.finfo(dtype).tiny)
        mean_observable = jnp.sum(flat_steps * observable_values) / safe_total_time
        response_samples = directional_samples + time_dilation.reshape((capacity,)) * (
            observable_values - mean_observable
        )
        sensitivity = jnp.sum(flat_steps * response_samples) / safe_total_time
        if comparison_sensitivity is None:
            convergence_evaluated = jnp.asarray(False)
            convergence_error = jnp.asarray(jnp.nan, dtype=dtype)
            converged = jnp.asarray(False)
            convergence_gate = jnp.asarray(True)
        else:
            comparison = jnp.asarray(comparison_sensitivity, dtype=dtype).reshape(())
            convergence_evaluated = jnp.asarray(True)
            convergence_error = jnp.abs(sensitivity - comparison) / jnp.maximum(
                jnp.abs(comparison), 1.0
            )
            converged = convergence_error <= self.convergence_tolerance
            convergence_gate = converged
        conditioned = (
            full_rank
            & jnp.isfinite(condition_number)
            & (condition_number <= self.condition_limit)
            & jnp.isfinite(qr_condition)
            & (qr_condition <= self.condition_limit)
        )
        residual_converged = jnp.isfinite(continuity_residual_norm) & (
            continuity_residual_norm <= self.residual_tolerance
        )
        full_grid = grid.accepted_step_count == capacity
        finite = (
            replay.finite
            & jnp.isfinite(sensitivity)
            & jnp.all(jnp.isfinite(observable_values))
            & jnp.all(jnp.isfinite(response_samples))
            & jnp.all(jnp.isfinite(shadow_tangent))
            & jnp.all(jnp.isfinite(coefficients))
            & jnp.isfinite(least_squares_residual_norm)
            & jnp.isfinite(continuity_residual_norm)
            & jnp.where(convergence_evaluated, jnp.isfinite(convergence_error), True)
        )
        derivative_supported = certification.supported
        tangent_valid = jnp.all(tangent_valid_by_segment)
        successful = (
            certification.successful
            & full_grid
            & derivative_supported
            & tangent_valid
            & neutral_valid
            & conditioned
            & residual_converged
            & convergence_gate
            & finite
        )
        status = jnp.where(
            ~certification.replay_completed | ~full_grid,
            int(MACShadowingStatus.REPLAY_FAILED),
            jnp.where(
                ~derivative_supported,
                int(MACShadowingStatus.DERIVATIVE_UNSUPPORTED),
                jnp.where(
                    ~finite,
                    int(MACShadowingStatus.NONFINITE),
                    jnp.where(
                        ~tangent_valid,
                        int(MACShadowingStatus.TANGENT_RANK_DEFICIENT),
                        jnp.where(
                            ~neutral_valid,
                            int(MACShadowingStatus.NEUTRAL_DIRECTION_FAILED),
                            jnp.where(
                                ~conditioned,
                                int(MACShadowingStatus.ILL_CONDITIONED),
                                jnp.where(
                                    ~residual_converged,
                                    int(MACShadowingStatus.CONTINUITY_RESIDUAL_FAILED),
                                    jnp.where(
                                        ~convergence_gate,
                                        int(MACShadowingStatus.CONVERGENCE_FAILED),
                                        int(MACShadowingStatus.SUCCESS),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        safe_sensitivity = jnp.where(successful, sensitivity, jnp.nan)
        return MACShadowingSensitivityResult(
            safe_sensitivity,
            response_samples,
            shadow_tangent.reshape((capacity, state_size)),
            time_dilation.reshape((capacity,)),
            coefficients,
            inhomogeneous_tangents,
            homogeneous_bases,
            qr_factors,
            continuity_residual,
            continuity_residual_norm,
            least_squares_residual_norm,
            condition_number,
            minimum_singular,
            qr_condition,
            neutral_minimum,
            convergence_error,
            convergence_evaluated,
            converged,
            conditioned,
            residual_converged,
            finite,
            successful,
            status,
            replay,
            certification,
            self.neutral_mode,
            solve_route,
            observable_id_,
            shadowing_id,
        )


__all__ = [
    "MACDerivativeMode",
    "MACFixedGridSensitivityPlan",
    "MACNeutralMode",
    "MACShadowingSolveRoute",
    "MACReplayCertification",
    "MACSegmentedShadowingPlan",
    "MACShadowingSensitivityResult",
    "MACShadowingStatus",
    "MACTerminalJVPResult",
    "MACTerminalVJPResult",
]
