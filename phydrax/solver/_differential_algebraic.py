#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import math
from enum import IntEnum
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
    RealizedTemporalMesh,
)
from ..dynamics import AbstractInputPolicy, DifferentialAlgebraicSystem, TimeGrid
from ..nonlinear import (
    AbstractNonlinearMethod,
    implicit_root_result,
    NewtonKrylov,
    NewtonTrustRegion,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
)
from ._bdf_method import (
    bdf_predict as _general_bdf_predict,
    bdf_rate as _general_bdf_rate,
    bdf_shift_offset as _general_bdf_shift_offset,
    BDFMethod,
)
from ._dae_initialization import (
    _DAEInitializationArguments,
    _initialize_dae,
    _masked_rms,
    _prepare_dae_initialization,
    _PreparedDAEInitialization,
    _scaled_space,
    _unknown_guess,
    DAEInitializationResult,
    DAEInitializationSpec,
)
from ._implicit_stage import ImplicitStageArguments, ImplicitStageResidual
from ._solution_validation import validate_solution_arrays
from ._theta import (
    endpoint_theta_rate,
    endpoint_theta_stage_arguments,
    ThetaMethod,
)


DAEFailureMode: TypeAlias = Literal["status", "error"]
DAEReplayMode: TypeAlias = Literal["full", "chunked"]
DAERegularityMode: TypeAlias = Literal["solver-evidence", "periodic"]
DAERegularityFailureMode: TypeAlias = Literal["record", "status"]
_DEFAULT_ARGS = object()


class DAEStatus(IntEnum):
    SUCCESS = 0
    INITIALIZATION_FAILED = 1
    NONLINEAR_FAILED = 2
    LINEAR_FAILED = 3
    NONFINITE = 4
    RESIDUAL_TOO_LARGE = 5
    NOT_RUN = 6


class DAEAttemptStatus(IntEnum):
    ACCEPTED = 0
    LOCAL_ERROR_REJECTED = 1
    NONLINEAR_REJECTED = 2
    LINEAR_REJECTED = 3
    NONFINITE_REJECTED = 4
    RESIDUAL_REJECTED = 5
    CONSTRAINT_REJECTED = 6
    STALE_JACOBIAN_RETRY = 7
    REGULARITY_REJECTED = 8
    NOT_RUN = 9


class DAETerminationStatus(IntEnum):
    SUCCESS = 0
    INITIALIZATION_FAILED = 1
    CONTINUATION_INCONSISTENT = 2
    MAXIMUM_ACCEPTED_STEPS_REACHED = 3
    MAXIMUM_ATTEMPTS_REACHED = 4
    MINIMUM_STEP_REACHED = 5
    REPEATED_REJECTIONS = 6
    NONLINEAR_FAILURE = 7
    RESIDUAL_CERTIFICATION_FAILED = 8
    REGULARITY_FAILED = 9


class DAERegularityStatus(IntEnum):
    VERIFIED = 0
    ESTIMATED = 1
    INCONCLUSIVE = 2
    NUMERICALLY_SINGULAR = 3
    NOT_RUN = 4


def _identifier(value: str | None, payload: object, prefix: str, /) -> str:
    if value is not None:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{prefix} identifier must be a non-empty string or None.")
        return value
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _inexact(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _method_identity(method: NewtonKrylov | NewtonTrustRegion, /) -> tuple[str, ...]:
    globalization = (
        method.line_search if isinstance(method, NewtonKrylov) else method.trust_region
    )
    return (
        method.method_id,
        method.jacobian_policy.policy_id,
        repr(method.linear_policy),
        repr(method.forcing_policy),
        repr(method.jacobian_refresh),
        repr(globalization),
    )


def _termination_identity(termination: NonlinearTermination, /) -> tuple[Any, ...]:
    return (
        termination.absolute_residual,
        termination.relative_residual,
        termination.absolute_step,
        termination.relative_step,
        termination.maximum_steps,
        termination.maximum_evaluations,
        termination.maximum_linear_iterations,
        termination.divergence_factor,
    )


class DAEAdaptivePolicy(StrictModule):
    """Accepted-step controller for adaptive BDF1/BDF2 integration."""

    relative_tolerance: Array
    absolute_tolerance: Array
    initial_step: float | None = eqx.field(static=True)
    minimum_step: float | None = eqx.field(static=True)
    maximum_step: float | None = eqx.field(static=True)
    safety: float = eqx.field(static=True)
    accepted_growth_minimum: float = eqx.field(static=True)
    accepted_growth_maximum: float = eqx.field(static=True)
    rejected_shrink_minimum: float = eqx.field(static=True)
    rejected_shrink_maximum: float = eqx.field(static=True)
    nonlinear_failure_shrink: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    maximum_accepted_steps: int = eqx.field(static=True)
    maximum_attempts: int = eqx.field(static=True)
    maximum_consecutive_rejections: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative_tolerance: ArrayLike = 1e-5,
        absolute_tolerance: ArrayLike = 1e-8,
        initial_step: float | None = None,
        minimum_step: float | None = None,
        maximum_step: float | None = None,
        safety: float = 0.9,
        accepted_growth_minimum: float = 0.2,
        accepted_growth_maximum: float = 5.0,
        rejected_shrink_minimum: float = 0.1,
        rejected_shrink_maximum: float = 0.5,
        nonlinear_failure_shrink: float = 0.25,
        residual_tolerance: float = 1e-8,
        constraint_tolerance: float = 1e-8,
        maximum_accepted_steps: int = 4096,
        maximum_attempts: int = 8192,
        maximum_consecutive_rejections: int = 12,
    ):
        relative = _inexact(relative_tolerance)
        absolute = _inexact(absolute_tolerance)
        relative_host = np.asarray(relative, dtype=float)
        absolute_host = np.asarray(absolute, dtype=float)
        if (
            not np.all(np.isfinite(relative_host))
            or not np.all(np.isfinite(absolute_host))
            or np.any(relative_host < 0.0)
            or np.any(absolute_host < 0.0)
        ):
            raise ValueError("Adaptive tolerances must be finite and non-negative.")
        if np.all(relative_host == 0.0) and np.all(absolute_host == 0.0):
            raise ValueError("At least one adaptive tolerance must be positive.")
        steps = tuple(
            None if value is None else float(value)
            for value in (initial_step, minimum_step, maximum_step)
        )
        if any(
            value is not None and (not math.isfinite(value) or value <= 0.0)
            for value in steps
        ):
            raise ValueError("Adaptive step bounds must be finite and positive.")
        initial, minimum, maximum = steps
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("minimum_step cannot exceed maximum_step.")
        if initial is not None and minimum is not None and initial < minimum:
            raise ValueError("initial_step cannot be smaller than minimum_step.")
        if initial is not None and maximum is not None and initial > maximum:
            raise ValueError("initial_step cannot exceed maximum_step.")
        controller = tuple(
            float(value)
            for value in (
                safety,
                accepted_growth_minimum,
                accepted_growth_maximum,
                rejected_shrink_minimum,
                rejected_shrink_maximum,
                nonlinear_failure_shrink,
                residual_tolerance,
                constraint_tolerance,
            )
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in controller):
            raise ValueError("Adaptive controller values must be finite and positive.")
        (
            safety_,
            growth_minimum,
            growth_maximum,
            shrink_minimum,
            shrink_maximum,
            nonlinear_shrink,
            residual_threshold,
            constraint_threshold,
        ) = controller
        if not 0.0 < safety_ <= 1.0:
            raise ValueError("safety must lie in (0, 1].")
        if not 0.0 < growth_minimum <= 1.0 <= growth_maximum:
            raise ValueError("Accepted growth bounds must straddle one.")
        if not 0.0 < shrink_minimum <= shrink_maximum < 1.0:
            raise ValueError("Rejected shrink bounds must lie in (0, 1).")
        if not 0.0 < nonlinear_shrink < 1.0:
            raise ValueError("nonlinear_failure_shrink must lie in (0, 1).")
        capacities = tuple(
            int(value)
            for value in (
                maximum_accepted_steps,
                maximum_attempts,
                maximum_consecutive_rejections,
            )
        )
        if any(value < 1 for value in capacities):
            raise ValueError("Adaptive capacities must be positive.")
        if capacities[1] < capacities[0]:
            raise ValueError("maximum_attempts must cover maximum_accepted_steps.")
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.initial_step = initial
        self.minimum_step = minimum
        self.maximum_step = maximum
        self.safety = safety_
        self.accepted_growth_minimum = growth_minimum
        self.accepted_growth_maximum = growth_maximum
        self.rejected_shrink_minimum = shrink_minimum
        self.rejected_shrink_maximum = shrink_maximum
        self.nonlinear_failure_shrink = nonlinear_shrink
        self.residual_tolerance = residual_threshold
        self.constraint_tolerance = constraint_threshold
        self.maximum_accepted_steps = capacities[0]
        self.maximum_attempts = capacities[1]
        self.maximum_consecutive_rejections = capacities[2]


class DAETemporalReusePolicy(StrictModule):
    """Cross-step modified-Newton reuse and mandatory refresh safeguards."""

    enabled: bool = eqx.field(static=True)
    maximum_jacobian_age: int = eqx.field(static=True)
    maximum_alpha_ratio: float = eqx.field(static=True)
    refresh_after_iterations: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        enabled: bool = True,
        maximum_jacobian_age: int = 2,
        maximum_alpha_ratio: float = 1.25,
        refresh_after_iterations: int | None = 3,
    ):
        age = int(maximum_jacobian_age)
        ratio = float(maximum_alpha_ratio)
        iterations = (
            None if refresh_after_iterations is None else int(refresh_after_iterations)
        )
        if age < 1:
            raise ValueError("maximum_jacobian_age must be positive.")
        if not math.isfinite(ratio) or ratio < 1.0:
            raise ValueError("maximum_alpha_ratio must be finite and at least one.")
        if iterations is not None and iterations < 1:
            raise ValueError("refresh_after_iterations must be positive or None.")
        self.enabled = bool(enabled)
        self.maximum_jacobian_age = age
        self.maximum_alpha_ratio = ratio
        self.refresh_after_iterations = iterations


class DAEReplayPolicy(StrictModule):
    """Frozen-grid replay checkpointing policy."""

    checkpointing: DAEReplayMode = eqx.field(static=True)
    chunk_size: int | None = eqx.field(static=True)
    memory_budget_bytes: int | None = eqx.field(static=True)

    def __init__(
        self,
        checkpointing: DAEReplayMode = "full",
        /,
        *,
        chunk_size: int | None = None,
        memory_budget_bytes: int | None = None,
    ):
        if checkpointing not in ("full", "chunked"):
            raise ValueError("checkpointing must be 'full' or 'chunked'.")
        chunk = None if chunk_size is None else int(chunk_size)
        budget = None if memory_budget_bytes is None else int(memory_budget_bytes)
        if chunk is not None and chunk < 1:
            raise ValueError("chunk_size must be positive or None.")
        if budget is not None and budget < 1:
            raise ValueError("memory_budget_bytes must be positive or None.")
        if checkpointing == "full" and (chunk is not None or budget is not None):
            raise ValueError("Full replay does not accept chunk planning inputs.")
        if checkpointing == "chunked" and (chunk is None) == (budget is None):
            raise ValueError(
                "Chunked replay requires exactly one of chunk_size or "
                "memory_budget_bytes."
            )
        self.checkpointing = checkpointing
        self.chunk_size = chunk
        self.memory_budget_bytes = budget


class DAERegularityPolicy(StrictModule):
    """Local numerical evidence policy; never a global DAE index claim."""

    mode: DAERegularityMode = eqx.field(static=True)
    interval: int = eqx.field(static=True)
    condition_limit: float | None = eqx.field(static=True)
    failure: DAERegularityFailureMode = eqx.field(static=True)

    def __init__(
        self,
        mode: DAERegularityMode = "solver-evidence",
        /,
        *,
        interval: int = 1,
        condition_limit: float | None = None,
        failure: DAERegularityFailureMode = "record",
    ):
        if mode not in ("solver-evidence", "periodic"):
            raise ValueError("mode must be 'solver-evidence' or 'periodic'.")
        interval_ = int(interval)
        limit = None if condition_limit is None else float(condition_limit)
        if interval_ < 1:
            raise ValueError("interval must be positive.")
        if limit is not None and (not math.isfinite(limit) or limit <= 1.0):
            raise ValueError("condition_limit must be finite and exceed one.")
        if failure not in ("record", "status"):
            raise ValueError("failure must be 'record' or 'status'.")
        self.mode = mode
        self.interval = interval_
        self.condition_limit = limit
        self.failure = failure


class DAESolvePolicy(StrictModule):
    """Fixed or adaptive implicit integration with explicit numerical policies."""

    nonlinear_method: NewtonKrylov | NewtonTrustRegion
    nonlinear_termination: NonlinearTermination
    initialization_method: NewtonKrylov | NewtonTrustRegion
    initialization_termination: NonlinearTermination
    adaptive: DAEAdaptivePolicy | None
    temporal_reuse: DAETemporalReusePolicy
    replay: DAEReplayPolicy
    regularity: DAERegularityPolicy
    method: BDFMethod | ThetaMethod
    max_step_ratio: float = eqx.field(static=True)
    failure: DAEFailureMode = eqx.field(static=True)

    def __init__(
        self,
        *,
        method: BDFMethod | ThetaMethod | None = None,
        nonlinear_method: AbstractNonlinearMethod | None = None,
        nonlinear_termination: NonlinearTermination | None = None,
        initialization_method: AbstractNonlinearMethod | None = None,
        initialization_termination: NonlinearTermination | None = None,
        adaptive: DAEAdaptivePolicy | None = None,
        temporal_reuse: DAETemporalReusePolicy | None = None,
        replay: DAEReplayPolicy | None = None,
        regularity: DAERegularityPolicy | None = None,
        max_step_ratio: float = 2.0,
        failure: DAEFailureMode = "status",
    ):
        temporal_method = BDFMethod() if method is None else method
        if not isinstance(temporal_method, (BDFMethod, ThetaMethod)):
            raise TypeError("method must be BDFMethod, ThetaMethod, or None.")
        stage_method = NewtonKrylov() if nonlinear_method is None else nonlinear_method
        initial_method = (
            NewtonKrylov() if initialization_method is None else initialization_method
        )
        if not isinstance(stage_method, (NewtonKrylov, NewtonTrustRegion)):
            raise TypeError(
                "nonlinear_method must be NewtonKrylov, NewtonTrustRegion, or None."
            )
        if not isinstance(initial_method, (NewtonKrylov, NewtonTrustRegion)):
            raise TypeError(
                "initialization_method must be NewtonKrylov, NewtonTrustRegion, or None."
            )
        stage_termination = (
            NonlinearTermination(
                absolute_residual=1e-8,
                relative_residual=0.0,
                maximum_steps=12,
            )
            if nonlinear_termination is None
            else nonlinear_termination
        )
        initial_termination = (
            NonlinearTermination(
                absolute_residual=1e-8,
                relative_residual=0.0,
                maximum_steps=32,
            )
            if initialization_termination is None
            else initialization_termination
        )
        if not isinstance(stage_termination, NonlinearTermination):
            raise TypeError(
                "nonlinear_termination must be a NonlinearTermination or None."
            )
        if not isinstance(initial_termination, NonlinearTermination):
            raise TypeError(
                "initialization_termination must be a NonlinearTermination or None."
            )
        if adaptive is not None and not isinstance(adaptive, DAEAdaptivePolicy):
            raise TypeError("adaptive must be a DAEAdaptivePolicy or None.")
        reuse_policy = (
            DAETemporalReusePolicy() if temporal_reuse is None else temporal_reuse
        )
        replay_policy = DAEReplayPolicy() if replay is None else replay
        regularity_policy = DAERegularityPolicy() if regularity is None else regularity
        if not isinstance(reuse_policy, DAETemporalReusePolicy):
            raise TypeError("temporal_reuse must be a DAETemporalReusePolicy or None.")
        if not isinstance(replay_policy, DAEReplayPolicy):
            raise TypeError("replay must be a DAEReplayPolicy or None.")
        if not isinstance(regularity_policy, DAERegularityPolicy):
            raise TypeError("regularity must be a DAERegularityPolicy or None.")
        ratio = float(max_step_ratio)
        if not math.isfinite(ratio) or ratio < 1.0:
            raise ValueError("max_step_ratio must be finite and at least one.")
        if failure not in ("status", "error"):
            raise ValueError("failure must be 'status' or 'error'.")
        self.nonlinear_method = stage_method
        self.nonlinear_termination = stage_termination
        self.initialization_method = initial_method
        self.initialization_termination = initial_termination
        self.adaptive = adaptive
        self.temporal_reuse = reuse_policy
        self.replay = replay_policy
        self.regularity = regularity_policy
        self.method = temporal_method
        self.max_step_ratio = ratio
        self.failure = failure


class DifferentialAlgebraicProblem(StrictModule):
    """Implicit initial-value problem with an explicit consistency contract."""

    system: DifferentialAlgebraicSystem
    input_policy: AbstractInputPolicy | None
    initial_state: Array
    initial_state_rate: Array
    args: Any
    initialization: DAEInitializationSpec
    problem_id: str = eqx.field(static=True)
    discretization_bundle: DiscretizationBundle | None
    discretization_bundle_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        system: DifferentialAlgebraicSystem | Any,
        initial_state: ArrayLike,
        /,
        *,
        initial_state_rate: ArrayLike | None = None,
        args: Any = None,
        input_policy: AbstractInputPolicy | None = None,
        initialization: DAEInitializationSpec | Literal["structural"] | None = None,
        discretization_bundle: DiscretizationBundle | None = None,
        problem_id: str | None = None,
    ):
        from ..dynamics._dae_structural import ReducedDAECompilation

        structural = system if isinstance(system, ReducedDAECompilation) else None
        if structural is not None:
            system = structural.system
        if not isinstance(system, DifferentialAlgebraicSystem):
            raise TypeError(
                "system must be DifferentialAlgebraicSystem or ReducedDAECompilation."
            )
        if system.input_layout is None:
            if input_policy is not None:
                raise ValueError(
                    "An autonomous DAE problem does not accept input_policy."
                )
        else:
            if input_policy is None:
                raise ValueError("An input-aware DAE problem requires input_policy.")
            if not isinstance(input_policy, AbstractInputPolicy):
                raise TypeError("input_policy must be an AbstractInputPolicy or None.")
            if input_policy.input_layout.layout_id != system.input_layout.layout_id:
                raise ValueError(
                    "input_policy layout must exactly match the DAE system input layout."
                )
        state = _inexact(initial_state)
        state_rate = (
            jnp.zeros_like(state)
            if initial_state_rate is None
            else _inexact(initial_state_rate)
        )
        if state.shape != system.state_shape or state_rate.shape != system.state_shape:
            raise ValueError(
                f"Initial state and rate must both have shape {system.state_shape}."
            )
        if state.dtype != state_rate.dtype:
            raise TypeError("Initial state and rate must have the same dtype.")
        state = eqx.error_if(
            state,
            jnp.any(~jnp.isfinite(state)) | jnp.any(~jnp.isfinite(state_rate)),
            "DAE initial state and rate must be finite.",
        )
        state = eqx.error_if(
            state,
            ~jnp.asarray(system.state_geometry.contains(state), dtype=bool),
            "DAE initial state is outside its state geometry.",
        )
        if initialization == "structural":
            if structural is None:
                raise ValueError(
                    "initialization='structural' requires ReducedDAECompilation."
                )
            initial_spec = DAEInitializationSpec.from_masks(
                structural.fixed_state_mask,
                structural.fixed_rate_mask,
            )
        else:
            initial_spec = (
                DAEInitializationSpec.index_one()
                if initialization is None
                else initialization
            )
        if not isinstance(initial_spec, DAEInitializationSpec):
            raise TypeError(
                "initialization must be DAEInitializationSpec, 'structural', or None."
            )
        if discretization_bundle is not None and not isinstance(
            discretization_bundle,
            DiscretizationBundle,
        ):
            raise TypeError(
                "discretization_bundle must be a DiscretizationBundle or None."
            )
        bundle_id = (
            None if discretization_bundle is None else discretization_bundle.bundle_id
        )
        self.system = system
        self.input_policy = input_policy
        self.initial_state = state
        self.initial_state_rate = state_rate
        self.args = args
        self.initialization = initial_spec
        self.discretization_bundle_id = bundle_id
        self.discretization_bundle = discretization_bundle
        self.problem_id = _identifier(
            problem_id,
            (
                system.system_id,
                system.state_shape,
                np.dtype(state.dtype).str,
                initial_spec.initialization_id,
                None if input_policy is None else input_policy.policy_id,
                bundle_id,
            ),
            "dae-problem",
        )


def discretized_dae_problem(
    compiled: Any,
    initial_state: ArrayLike,
    /,
    *,
    initial_state_rate: ArrayLike | None = None,
    args: Any = None,
    input_policy: AbstractInputPolicy | None = None,
    initialization: DAEInitializationSpec | None = None,
    problem_id: str | None = None,
) -> DifferentialAlgebraicProblem:
    """Bind a compiled discrete residual to the native DAE lifecycle."""
    from ..equations import CompiledDiscreteResidual

    if not isinstance(compiled, CompiledDiscreteResidual):
        raise TypeError("compiled must be a CompiledDiscreteResidual.")
    return DifferentialAlgebraicProblem(
        compiled.system,
        initial_state,
        initial_state_rate=initial_state_rate,
        args=args,
        initialization=initialization,
        input_policy=input_policy,
        discretization_bundle=compiled.discretization_bundle,
        problem_id=problem_id,
    )


class DAESolvePlan(StrictModule):
    """Validated fixed/adaptive DAE policy and structural execution identity."""

    policy: DAESolvePolicy
    system_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    time_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_dtype: str = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    maximum_accepted_steps: int = eqx.field(static=True)
    maximum_attempts: int = eqx.field(static=True)
    replay_chunk_size: int = eqx.field(static=True)
    replay_memory_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: DifferentialAlgebraicProblem,
        time_grid: TimeGrid,
        policy: DAESolvePolicy,
        /,
    ):
        if not isinstance(problem, DifferentialAlgebraicProblem):
            raise TypeError("problem must be a DifferentialAlgebraicProblem.")
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        if not isinstance(policy, DAESolvePolicy):
            raise TypeError("policy must be a DAESolvePolicy.")
        if isinstance(policy.method, ThetaMethod):
            if policy.adaptive is not None:
                raise ValueError("ThetaMethod currently requires a fixed TimeGrid.")
            if not policy.method.endpoint:
                raise ValueError(
                    "Native residual integration requires the stiffly accurate "
                    "endpoint theta form; implicit midpoint is provided by "
                    "GaussLegendreIRK."
                )
        durations = np.asarray(time_grid.durations, dtype=float)
        if (
            isinstance(policy.method, BDFMethod)
            and policy.adaptive is None
            and policy.method.maximum_order >= 2
            and durations.size > 1
        ):
            ratios = durations[1:] / durations[:-1]
            if np.any(ratios > policy.max_step_ratio) or np.any(
                ratios < 1.0 / policy.max_step_ratio
            ):
                raise ValueError(
                    "BDF2 adjacent step ratios exceed the declared max_step_ratio."
                )
        adaptive = policy.adaptive
        if adaptive is not None:
            np.broadcast_to(
                np.asarray(adaptive.relative_tolerance),
                problem.system.state_shape,
            )
            np.broadcast_to(
                np.asarray(adaptive.absolute_tolerance),
                problem.system.state_shape,
            )
            if adaptive.maximum_accepted_steps < time_grid.num_steps:
                raise ValueError(
                    "maximum_accepted_steps must cover every requested save interval."
                )
        accepted_capacity = (
            time_grid.num_steps if adaptive is None else adaptive.maximum_accepted_steps
        )
        attempt_capacity = (
            time_grid.num_steps if adaptive is None else adaptive.maximum_attempts
        )
        state_bytes = int(
            problem.initial_state.size * problem.initial_state.dtype.itemsize
        )
        bytes_per_step = max(8 * state_bytes + 256, 1)
        replay_policy = policy.replay
        if replay_policy.checkpointing == "full":
            replay_chunk_size = max(accepted_capacity, 1)
            replay_memory_bytes = bytes_per_step * accepted_capacity
        elif replay_policy.chunk_size is not None:
            replay_chunk_size = min(replay_policy.chunk_size, accepted_capacity)
            replay_memory_bytes = (
                bytes_per_step * replay_chunk_size
                + 3 * state_bytes * math.ceil(accepted_capacity / replay_chunk_size)
            )
        else:
            budget = replay_policy.memory_budget_bytes
            if budget is None:
                raise ValueError(
                    "Chunked replay planning requires a chunk size or memory budget."
                )
            checkpoint_bytes = 3 * state_bytes
            discriminant = (
                budget * budget
                - 4 * bytes_per_step * checkpoint_bytes * accepted_capacity
            )
            if discriminant < 0:
                raise ValueError(
                    "Replay memory budget is below the minimum feasible checkpoint "
                    "footprint."
                )
            replay_chunk_size = min(
                accepted_capacity,
                max(
                    1,
                    int((budget + math.sqrt(discriminant)) // (2 * bytes_per_step)),
                ),
            )
            replay_memory_bytes = (
                bytes_per_step * replay_chunk_size
                + checkpoint_bytes * math.ceil(accepted_capacity / replay_chunk_size)
            )
            while replay_chunk_size > 1 and replay_memory_bytes > budget:
                replay_chunk_size -= 1
                replay_memory_bytes = (
                    bytes_per_step * replay_chunk_size
                    + checkpoint_bytes * math.ceil(accepted_capacity / replay_chunk_size)
                )
            if replay_memory_bytes > budget:
                raise ValueError(
                    "Replay memory budget is below the minimum feasible checkpoint "
                    "footprint."
                )
        self.policy = policy
        self.system_id = problem.system.system_id
        self.problem_id = problem.problem_id
        self.time_id = time_grid.time_id
        self.discretization_bundle_id = problem.discretization_bundle_id
        self.state_shape = problem.system.state_shape
        self.state_dtype = np.dtype(problem.initial_state.dtype).str
        self.num_steps = time_grid.num_steps
        self.maximum_accepted_steps = accepted_capacity
        self.maximum_attempts = attempt_capacity
        self.replay_chunk_size = replay_chunk_size
        self.replay_memory_bytes = replay_memory_bytes
        self.plan_id = _identifier(
            None,
            (
                self.system_id,
                self.problem_id,
                self.time_id,
                self.discretization_bundle_id,
                self.state_shape,
                self.state_dtype,
                policy.method.method_id,
                policy.max_step_ratio,
                _method_identity(policy.nonlinear_method),
                _termination_identity(policy.nonlinear_termination),
                _method_identity(policy.initialization_method),
                _termination_identity(policy.initialization_termination),
                repr(policy.adaptive),
                repr(policy.temporal_reuse),
                repr(policy.replay),
                repr(policy.regularity),
                replay_chunk_size,
            ),
            "dae-plan",
        )


def _bdf_affine_rate(
    previous: Array,
    previous_previous: Array,
    step_size: Array,
    previous_step_size: Array,
    order: Array,
    /,
) -> tuple[Array, Array]:
    def first_order(_):
        shift = 1.0 / step_size
        return shift, -shift * previous

    def second_order(_):
        ratio = step_size / previous_step_size
        shift = ((1.0 + 2.0 * ratio) / (1.0 + ratio)) / step_size
        offset = (
            -(1.0 + ratio) * previous
            + (ratio * ratio / (1.0 + ratio)) * previous_previous
        ) / step_size
        return shift, offset

    return lax.cond(order == 1, first_order, second_order, operand=None)


def _bdf_rate(
    state: Array,
    previous: Array,
    previous_previous: Array,
    step_size: Array,
    previous_step_size: Array,
    order: Array,
    /,
) -> Array:
    shift, offset = _bdf_affine_rate(
        previous,
        previous_previous,
        step_size,
        previous_step_size,
        order,
    )
    return shift * state + offset


def _predict(
    previous: Array,
    previous_previous: Array,
    previous_rate: Array,
    step_size: Array,
    previous_step_size: Array,
    order: Array,
    /,
) -> Array:
    return lax.cond(
        order == 1,
        lambda _: previous + step_size * previous_rate,
        lambda _: (
            previous + (step_size / previous_step_size) * (previous - previous_previous)
        ),
        operand=None,
    )


def _stage_arguments(
    *,
    time: Array,
    previous: Array,
    previous_previous: Array,
    step_size: Array,
    previous_step_size: Array,
    order: Array,
    model_args: Any,
    active: ArrayLike = True,
) -> ImplicitStageArguments:
    shift, offset = _bdf_affine_rate(
        previous,
        previous_previous,
        step_size,
        previous_step_size,
        order,
    )
    return ImplicitStageArguments(
        time=time,
        shift=shift,
        rate_offset=offset,
        explicit_value=jnp.zeros_like(previous),
        fallback_state=previous,
        active=active,
        model_args=model_args,
    )


def _history_stage_arguments(
    *,
    target_time: Array,
    state_history: Array,
    history_times: Array,
    order: Array,
    model_args: Any,
    active: ArrayLike = True,
) -> ImplicitStageArguments:
    shift, offset = _general_bdf_shift_offset(
        state_history,
        history_times,
        target_time,
        order,
    )
    return ImplicitStageArguments(
        time=target_time,
        shift=shift,
        rate_offset=offset,
        explicit_value=jnp.zeros_like(state_history[0]),
        fallback_state=state_history[0],
        active=active,
        model_args=model_args,
    )


class PreparedDAESolve(StrictModule):
    """DAE problem, grid, consistency root, and reusable implicit-stage root."""

    problem: DifferentialAlgebraicProblem
    time_grid: TimeGrid
    plan: DAESolvePlan
    initialization: _PreparedDAEInitialization
    stage_problem: NonlinearSystemProblem
    stage_solve: PreparedNonlinearSolve
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: DifferentialAlgebraicProblem,
        time_grid: TimeGrid,
        plan: DAESolvePlan,
        initialization: _PreparedDAEInitialization,
        stage_problem: NonlinearSystemProblem,
        stage_solve: PreparedNonlinearSolve,
        /,
    ):
        if (
            plan.problem_id != problem.problem_id
            or plan.system_id != problem.system.system_id
        ):
            raise ValueError("DAE plan and problem identities must match.")
        if plan.time_id != time_grid.time_id or plan.num_steps != time_grid.num_steps:
            raise ValueError("DAE plan and TimeGrid identities must match.")
        self.problem = problem
        self.time_grid = time_grid
        self.plan = plan
        self.initialization = initialization
        self.stage_problem = stage_problem
        self.stage_solve = stage_solve
        self.prepared_id = _identifier(
            None,
            (
                plan.plan_id,
                initialization.preparation_id,
                stage_solve.linear_template_id,
            ),
            "prepared-dae",
        )

    @property
    def stage_linear_plan_id(self) -> str:
        return self.stage_solve.linear_plan_id

    @property
    def initialization_linear_plan_id(self) -> str:
        nonlinear_solve = self.initialization.nonlinear_solve
        return "" if nonlinear_solve is None else nonlinear_solve.linear_plan_id


class DAEStepHistory(StrictModule):
    """Fixed-capacity evidence for accepted internal integration steps."""

    accepted_times: Array
    step_sizes: Array
    orders: Array
    error_ratios: Array
    source_attempt_indices: Array
    valid: Array
    count: Array
    save_step_indices: Array

    def __init__(
        self,
        *,
        accepted_times: Array,
        step_sizes: Array,
        orders: Array,
        error_ratios: Array,
        source_attempt_indices: Array,
        valid: Array,
        count: Array,
        save_step_indices: Array,
    ):
        capacity = int(jnp.asarray(step_sizes).size)
        shape = (capacity,)
        for values, name in (
            (accepted_times, "accepted_times"),
            (orders, "orders"),
            (error_ratios, "error_ratios"),
            (source_attempt_indices, "source_attempt_indices"),
            (valid, "valid"),
        ):
            if jnp.asarray(values).shape != shape:
                raise ValueError(f"DAE step history {name} must have shape {shape}.")
        self.accepted_times = jnp.asarray(accepted_times)
        self.step_sizes = jnp.asarray(step_sizes)
        self.orders = jnp.asarray(orders, dtype=jnp.int32)
        self.error_ratios = jnp.asarray(error_ratios)
        self.source_attempt_indices = jnp.asarray(source_attempt_indices, dtype=jnp.int32)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.count = jnp.asarray(count, dtype=jnp.int32)
        self.save_step_indices = jnp.asarray(save_step_indices, dtype=jnp.int32)


class DAEAttemptHistory(StrictModule):
    """Fixed-capacity evidence for every accepted or rejected attempt."""

    times: Array
    proposed_step_sizes: Array
    orders: Array
    status: Array
    error_ratios: Array
    nonlinear_status: Array
    nonlinear_iterations: Array
    residual_evaluations: Array
    jacobian_preparations: Array
    linear_solves: Array
    linear_iterations: Array
    globalization_rejections: Array
    setup_refreshes: Array
    numeric_refreshes: Array
    stale_jacobian_retries: Array
    linear_rejections: Array
    residual_certifications: Array
    valid: Array
    count: Array

    def __init__(
        self,
        *,
        times: Array,
        proposed_step_sizes: Array,
        orders: Array,
        status: Array,
        error_ratios: Array,
        nonlinear_status: Array,
        nonlinear_iterations: Array,
        residual_evaluations: Array,
        jacobian_preparations: Array,
        linear_solves: Array,
        linear_iterations: Array,
        globalization_rejections: Array,
        setup_refreshes: Array,
        numeric_refreshes: Array,
        stale_jacobian_retries: Array,
        linear_rejections: Array,
        residual_certifications: Array,
        valid: Array,
        count: Array,
    ):
        shape = jnp.asarray(proposed_step_sizes).shape
        if len(shape) != 1:
            raise ValueError("DAE attempt history must be one-dimensional.")
        values = (
            times,
            orders,
            status,
            error_ratios,
            nonlinear_status,
            nonlinear_iterations,
            residual_evaluations,
            jacobian_preparations,
            linear_solves,
            linear_iterations,
            globalization_rejections,
            setup_refreshes,
            numeric_refreshes,
            stale_jacobian_retries,
            linear_rejections,
            residual_certifications,
            valid,
        )
        if any(jnp.asarray(value).shape != shape for value in values):
            raise ValueError("Every DAE attempt history field must share one shape.")
        self.times = jnp.asarray(times)
        self.proposed_step_sizes = jnp.asarray(proposed_step_sizes)
        self.orders = jnp.asarray(orders, dtype=jnp.int32)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.error_ratios = jnp.asarray(error_ratios)
        self.nonlinear_status = jnp.asarray(nonlinear_status, dtype=jnp.int32)
        self.nonlinear_iterations = jnp.asarray(nonlinear_iterations, dtype=jnp.int32)
        self.residual_evaluations = jnp.asarray(residual_evaluations, dtype=jnp.int32)
        self.jacobian_preparations = jnp.asarray(jacobian_preparations, dtype=jnp.int32)
        self.linear_solves = jnp.asarray(linear_solves, dtype=jnp.int32)
        self.linear_iterations = jnp.asarray(linear_iterations, dtype=jnp.int32)
        self.globalization_rejections = jnp.asarray(
            globalization_rejections, dtype=jnp.int32
        )
        self.setup_refreshes = jnp.asarray(setup_refreshes, dtype=jnp.int32)
        self.numeric_refreshes = jnp.asarray(numeric_refreshes, dtype=jnp.int32)
        self.stale_jacobian_retries = jnp.asarray(stale_jacobian_retries, dtype=jnp.int32)
        self.linear_rejections = jnp.asarray(linear_rejections, dtype=jnp.int32)
        self.residual_certifications = jnp.asarray(
            residual_certifications, dtype=jnp.int32
        )
        self.valid = jnp.asarray(valid, dtype=bool)
        self.count = jnp.asarray(count, dtype=jnp.int32)


class DAERegularityEvidence(StrictModule):
    """Local consistency/stage operator evidence without a global index claim."""

    consistency_status: Array
    consistency_rank: Array
    consistency_condition_estimate: Array
    stage_status: Array
    stage_rank: Array
    stage_condition_estimate: Array
    stage_valid: Array
    consistency_operator: str = eqx.field(static=True)
    stage_operator: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        consistency_status: Array,
        consistency_rank: Array,
        consistency_condition_estimate: Array,
        stage_status: Array,
        stage_rank: Array,
        stage_condition_estimate: Array,
        stage_valid: Array,
        consistency_operator: str,
        stage_operator: str,
    ):
        shape = jnp.asarray(stage_status).shape
        if (
            jnp.asarray(stage_rank).shape != shape
            or jnp.asarray(stage_condition_estimate).shape != shape
            or jnp.asarray(stage_valid).shape != shape
        ):
            raise ValueError("DAE stage regularity evidence shapes must match.")
        self.consistency_status = jnp.asarray(consistency_status, dtype=jnp.int32)
        self.consistency_rank = jnp.asarray(consistency_rank, dtype=jnp.int32)
        self.consistency_condition_estimate = jnp.asarray(consistency_condition_estimate)
        self.stage_status = jnp.asarray(stage_status, dtype=jnp.int32)
        self.stage_rank = jnp.asarray(stage_rank, dtype=jnp.int32)
        self.stage_condition_estimate = jnp.asarray(stage_condition_estimate)
        self.stage_valid = jnp.asarray(stage_valid, dtype=bool)
        self.consistency_operator = str(consistency_operator)
        self.stage_operator = str(stage_operator)


class DAEReplayEvidence(StrictModule):
    """Frozen-grid replay storage plan and realized accepted-step count."""

    accepted_steps: Array
    selected_chunk_size: int = eqx.field(static=True)
    estimated_memory_bytes: int = eqx.field(static=True)
    checkpointing: DAEReplayMode = eqx.field(static=True)

    def __init__(
        self,
        *,
        accepted_steps: Array,
        selected_chunk_size: int,
        estimated_memory_bytes: int,
        checkpointing: DAEReplayMode,
    ):
        self.accepted_steps = jnp.asarray(accepted_steps, dtype=jnp.int32)
        self.selected_chunk_size = int(selected_chunk_size)
        self.estimated_memory_bytes = int(estimated_memory_bytes)
        self.checkpointing = checkpointing


class DAEContinuation(StrictModule):
    """Exact accepted-history boundary state for segmented integration."""

    time: Array
    states: Array
    state_rates: Array
    times: Array
    step_sizes: Array
    history_depth: Array
    accepted_order: Array
    previous_error_ratio: Array
    proposed_step_size: Array
    jacobian_age: Array
    last_alpha: Array
    nonlinear_solve: PreparedNonlinearSolve | None
    problem_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    input_policy_id: str | None = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_dtype: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    initialization_id: str = eqx.field(static=True)
    nonlinear_method_id: str = eqx.field(static=True)
    stage_linear_plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        time: Array,
        states: Array,
        state_rates: Array,
        times: Array,
        step_sizes: Array,
        history_depth: Array,
        accepted_order: Array,
        previous_error_ratio: Array,
        proposed_step_size: Array,
        jacobian_age: Array,
        last_alpha: Array,
        nonlinear_solve: PreparedNonlinearSolve | None,
        problem_id: str,
        system_id: str,
        input_policy_id: str | None,
        method_id: str,
        initialization_id: str,
        nonlinear_method_id: str,
        stage_linear_plan_id: str,
    ):
        states_ = jnp.asarray(states)
        rates_ = jnp.asarray(state_rates)
        if states_.shape != rates_.shape or states_.shape[0] != 6:
            raise ValueError("DAE continuation must retain six state/rate slots.")
        if jnp.asarray(times).shape != (6,) or jnp.asarray(step_sizes).shape != (5,):
            raise ValueError("DAE continuation time history has invalid shape.")
        if nonlinear_solve is not None and not isinstance(
            nonlinear_solve, PreparedNonlinearSolve
        ):
            raise TypeError("nonlinear_solve must be a PreparedNonlinearSolve or None.")
        self.time = jnp.asarray(time)
        self.states = states_
        self.state_rates = rates_
        self.times = jnp.asarray(times)
        self.step_sizes = jnp.asarray(step_sizes)
        self.history_depth = jnp.asarray(history_depth, dtype=jnp.int32)
        self.accepted_order = jnp.asarray(accepted_order, dtype=jnp.int32)
        self.previous_error_ratio = jnp.asarray(previous_error_ratio)
        self.proposed_step_size = jnp.asarray(proposed_step_size)
        self.jacobian_age = jnp.asarray(jacobian_age, dtype=jnp.int32)
        self.last_alpha = jnp.asarray(last_alpha)
        self.nonlinear_solve = nonlinear_solve
        self.problem_id = str(problem_id)
        self.input_policy_id = None if input_policy_id is None else str(input_policy_id)
        self.system_id = str(system_id)
        self.state_shape = tuple(states_.shape[1:])
        self.state_dtype = np.dtype(states_.dtype).str
        self.method_id = str(method_id)
        self.initialization_id = str(initialization_id)
        self.nonlinear_method_id = str(nonlinear_method_id)
        self.stage_linear_plan_id = str(stage_linear_plan_id)

    @property
    def state(self) -> Array:
        return self.states[0]

    @property
    def state_rate(self) -> Array:
        return self.state_rates[0]


class DifferentialAlgebraicSolution(StrictModule):
    """Requested DAE samples plus accepted-step, attempt, and restart evidence."""

    times: Array
    states: Array
    state_rates: Array
    valid: Array
    rate_valid: Array
    status: Array
    residual_norm: Array
    residual_threshold: Array
    differential_residual_norm: Array
    constraint_norm: Array
    step_history: DAEStepHistory
    attempt_history: DAEAttemptHistory
    initialization: DAEInitializationResult
    continuation: DAEContinuation
    regularity: DAERegularityEvidence
    replay: DAEReplayEvidence
    termination_status: Array
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    input_policy_id: str | None = eqx.field(static=True)
    time_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    source_discretization_bundle_id: str | None = eqx.field(static=True)
    discretization_bundle: DiscretizationBundle
    temporal_mesh: RealizedTemporalMesh
    discretization_bundle_id: str = eqx.field(static=True)
    nonlinear_method_id: str = eqx.field(static=True)
    stage_linear_plan_id: str = eqx.field(static=True)
    initialization_linear_plan_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    differentiation_mode: str = eqx.field(static=True)
    grid_origin: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        times: Array,
        states: Array,
        state_rates: Array,
        valid: Array,
        rate_valid: Array,
        status: Array,
        residual_norm: Array,
        residual_threshold: Array,
        differential_residual_norm: Array,
        constraint_norm: Array,
        step_history: DAEStepHistory,
        attempt_history: DAEAttemptHistory,
        initialization: DAEInitializationResult,
        continuation: DAEContinuation,
        regularity: DAERegularityEvidence,
        replay: DAEReplayEvidence,
        termination_status: Array,
        problem_id: str,
        system_id: str,
        time_id: str,
        plan_id: str,
        input_policy_id: str | None,
        prepared_id: str,
        source_discretization_bundle: DiscretizationBundle | None,
        nonlinear_method_id: str,
        stage_linear_plan_id: str,
        initialization_linear_plan_id: str,
        method_id: str,
        adaptive: bool,
    ):
        validated = validate_solution_arrays(
            times,
            states,
            valid,
            sample_shape=(),
            state_shape=tuple(state_rates.shape[1:]),
            time_layout="shared",
            owner="DifferentialAlgebraicSolution",
        )
        if state_rates.shape != validated.states.shape:
            raise ValueError("DAE state_rates must have the same shape as states.")
        if rate_valid.shape != validated.states.shape:
            raise ValueError("DAE rate_valid must have the same shape as states.")
        node_shape = (int(validated.times.size),)
        for values, name in (
            (status, "status"),
            (residual_norm, "residual_norm"),
            (residual_threshold, "residual_threshold"),
            (differential_residual_norm, "differential_residual_norm"),
            (constraint_norm, "constraint_norm"),
        ):
            if jnp.asarray(values).shape != node_shape:
                raise ValueError(f"DAE {name} must have shape {node_shape}.")
        if not isinstance(step_history, DAEStepHistory):
            raise TypeError("step_history must be a DAEStepHistory.")
        if not isinstance(attempt_history, DAEAttemptHistory):
            raise TypeError("attempt_history must be a DAEAttemptHistory.")
        if not isinstance(continuation, DAEContinuation):
            raise TypeError("continuation must be a DAEContinuation.")
        if not isinstance(regularity, DAERegularityEvidence):
            raise TypeError("regularity must be DAERegularityEvidence.")
        if not isinstance(replay, DAEReplayEvidence):
            raise TypeError("replay must be DAEReplayEvidence.")
        self.times = validated.times
        self.states = validated.states
        self.state_rates = jnp.asarray(state_rates)
        self.valid = validated.valid
        self.rate_valid = jnp.asarray(rate_valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.residual_norm = jnp.asarray(residual_norm)
        self.residual_threshold = jnp.asarray(residual_threshold)
        self.differential_residual_norm = jnp.asarray(differential_residual_norm)
        self.constraint_norm = jnp.asarray(constraint_norm)
        self.step_history = step_history
        self.attempt_history = attempt_history
        self.initialization = initialization
        self.continuation = continuation
        self.regularity = regularity
        self.replay = replay
        self.termination_status = jnp.asarray(termination_status, dtype=jnp.int32)
        self.sample_shape = validated.sample_shape
        self.state_shape = validated.state_shape
        self.problem_id = str(problem_id)
        self.system_id = str(system_id)
        self.time_id = str(time_id)
        self.plan_id = str(plan_id)
        self.input_policy_id = None if input_policy_id is None else str(input_policy_id)
        self.prepared_id = str(prepared_id)
        if source_discretization_bundle is not None and not isinstance(
            source_discretization_bundle,
            DiscretizationBundle,
        ):
            raise TypeError(
                "source_discretization_bundle must be a DiscretizationBundle or None."
            )
        self.source_discretization_bundle_id = (
            None
            if source_discretization_bundle is None
            else source_discretization_bundle.bundle_id
        )
        temporal_mesh = RealizedTemporalMesh(
            validated.times[0],
            step_history.accepted_times,
            step_history.valid,
            step_history.count,
            adaptive=adaptive,
            source_plan_id=self.plan_id,
            requested_time_id=self.time_id,
        )
        temporal_key = DiscretizationKey(
            "dae_internal_time",
            DiscretizationRole.TEMPORAL,
            domain_labels=("time",),
        )
        temporal_record = DiscretizationRecord(
            temporal_key,
            "realized-temporal-mesh",
            temporal_mesh.mesh_id,
            realization_id=temporal_mesh.mesh_id,
        )
        source_records = (
            ()
            if source_discretization_bundle is None
            else source_discretization_bundle.records
        )
        source_transfers = (
            ()
            if source_discretization_bundle is None
            else source_discretization_bundle.transfers
        )
        source_couplings = (
            ()
            if source_discretization_bundle is None
            else source_discretization_bundle.stochastic_coupling_ids
        )
        bundle = DiscretizationBundle(
            source_records + (temporal_record,),
            transfers=source_transfers,
            stochastic_coupling_ids=source_couplings,
        )
        self.temporal_mesh = temporal_mesh
        self.discretization_bundle = bundle
        self.discretization_bundle_id = bundle.bundle_id
        self.nonlinear_method_id = str(nonlinear_method_id)
        self.stage_linear_plan_id = str(stage_linear_plan_id)
        self.initialization_linear_plan_id = str(initialization_linear_plan_id)
        self.method_id = str(method_id)
        self.differentiation_mode = (
            "frozen-accepted-grid-discrete-implicit"
            if adaptive
            else "fixed-grid-discrete-implicit"
        )
        self.grid_origin = "controller" if adaptive else "user"
        mesh_kind = "frozen-accepted-grid" if adaptive else "fixed-grid"
        self.approximation_id = f"{mesh_kind}:{self.method_id}"

    @property
    def successful(self) -> Array:
        return (self.termination_status == int(DAETerminationStatus.SUCCESS)) & jnp.all(
            self.valid
        )


def plan_dae(
    problem: DifferentialAlgebraicProblem,
    time_grid: TimeGrid,
    /,
    *,
    policy: DAESolvePolicy | None = None,
) -> DAESolvePlan:
    resolved_policy = DAESolvePolicy() if policy is None else policy
    return DAESolvePlan(problem, time_grid, resolved_policy)


def prepare_dae(
    problem: DifferentialAlgebraicProblem,
    time_grid: TimeGrid,
    /,
    *,
    policy: DAESolvePolicy | DAESolvePlan | None = None,
) -> PreparedDAESolve:
    if isinstance(policy, DAESolvePlan):
        plan = policy
    else:
        plan = plan_dae(problem, time_grid, policy=policy)
    if plan.problem_id != problem.problem_id:
        raise ValueError("A supplied DAE plan must match the problem.")
    resolved_policy = plan.policy
    initialization = _prepare_dae_initialization(
        problem.system,
        problem.initial_state,
        problem.initial_state_rate,
        time_grid.times[0],
        args=problem.args,
        input_policy=problem.input_policy,
        spec=problem.initialization,
        method=resolved_policy.initialization_method,
        termination=resolved_policy.initialization_termination,
    )
    step_size = time_grid.durations[0]
    if isinstance(resolved_policy.method, ThetaMethod):
        predictor = problem.initial_state + step_size * problem.initial_state_rate
        stage_arguments = endpoint_theta_stage_arguments(
            resolved_policy.method,
            target_time=time_grid.times[1],
            previous=problem.initial_state,
            previous_rate=problem.initial_state_rate,
            step_size=step_size,
            model_args=problem.args,
        )
    else:
        order = jnp.asarray(1, dtype=jnp.int32)
        predictor = _predict(
            problem.initial_state,
            problem.initial_state,
            problem.initial_state_rate,
            step_size,
            step_size,
            order,
        )
        stage_arguments = _stage_arguments(
            time=time_grid.times[1],
            previous=problem.initial_state,
            previous_previous=problem.initial_state,
            step_size=step_size,
            previous_step_size=step_size,
            order=order,
            model_args=problem.args,
        )
    state_space = _scaled_space(
        problem.system.state_shape,
        problem.initial_state.dtype,
        problem.system.state_scale,
        space_id=f"{problem.system.system_id}:implicit-state",
    )
    residual_space = _scaled_space(
        problem.system.state_shape,
        problem.initial_state.dtype,
        jnp.ones_like(problem.system.residual_scale),
        space_id=f"{problem.system.system_id}:implicit-residual",
    )
    stage_problem = NonlinearSystemProblem(
        ImplicitStageResidual(problem.system, problem.input_policy),
        state_space=state_space,
        residual_space=residual_space,
        problem_id=f"{problem.system.system_id}:implicit-stage-root",
    )
    stage_solve = prepare_nonlinear(
        stage_problem,
        predictor,
        method=resolved_policy.nonlinear_method,
        termination=resolved_policy.nonlinear_termination,
        args=stage_arguments,
    )
    return PreparedDAESolve(
        problem,
        time_grid,
        plan,
        initialization,
        stage_problem,
        stage_solve,
    )


def initialize_dae(
    problem: DifferentialAlgebraicProblem,
    time: ArrayLike,
    /,
    *,
    policy: DAESolvePolicy | None = None,
    args: Any = _DEFAULT_ARGS,
    initial_state: ArrayLike | None = None,
    initial_state_rate: ArrayLike | None = None,
) -> DAEInitializationResult:
    if not isinstance(problem, DifferentialAlgebraicProblem):
        raise TypeError("problem must be a DifferentialAlgebraicProblem.")
    resolved_policy = DAESolvePolicy() if policy is None else policy
    if not isinstance(resolved_policy, DAESolvePolicy):
        raise TypeError("policy must be a DAESolvePolicy or None.")
    runtime_args = problem.args if args is _DEFAULT_ARGS else args
    state = problem.initial_state if initial_state is None else initial_state
    state_rate = (
        problem.initial_state_rate if initial_state_rate is None else initial_state_rate
    )
    prepared = _prepare_dae_initialization(
        problem.system,
        state,
        state_rate,
        time,
        args=runtime_args,
        input_policy=problem.input_policy,
        spec=problem.initialization,
        method=resolved_policy.initialization_method,
        termination=resolved_policy.initialization_termination,
    )
    return _initialize_dae(
        prepared,
        state,
        state_rate,
        time,
        args=runtime_args,
        termination=resolved_policy.initialization_termination,
    )


def _linear_failure(status: Array, /) -> Array:
    return (
        (status == int(NonlinearStatus.LINEAR_SOLVE_FAILED))
        | (status == int(NonlinearStatus.SINGULAR_JACOBIAN))
        | (status == int(NonlinearStatus.MAXIMUM_LINEAR_ITERATIONS_REACHED))
    )


def _regularity_status(
    rank: Array,
    condition_estimate: Array,
    converged: Array,
    dimension: int,
    condition_limit: float | None,
    /,
) -> Array:
    rank_known = rank >= 0
    condition_known = jnp.isfinite(condition_estimate)
    singular = rank_known & (rank < dimension)
    if condition_limit is not None:
        singular = singular | (condition_known & (condition_estimate > condition_limit))
    verified = (
        jnp.asarray(converged, dtype=bool)
        & rank_known
        & (rank == dimension)
        & condition_known
    )
    estimated = jnp.asarray(converged, dtype=bool) & ~verified
    return jnp.where(
        singular,
        int(DAERegularityStatus.NUMERICALLY_SINGULAR),
        jnp.where(
            verified,
            int(DAERegularityStatus.VERIFIED),
            jnp.where(
                estimated,
                int(DAERegularityStatus.ESTIMATED),
                int(DAERegularityStatus.INCONCLUSIVE),
            ),
        ),
    ).astype(jnp.int32)


def _matrix_regularity(matrix: Array, /):
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    largest = singular_values[0]
    smallest = singular_values[-1]
    tolerance = (
        max(matrix.shape)
        * jnp.finfo(singular_values.real.dtype).eps
        * jnp.maximum(largest, 1.0)
    )
    rank = jnp.sum(singular_values > tolerance, dtype=jnp.int32)
    condition = jnp.where(smallest > tolerance, largest / smallest, jnp.inf)
    return rank, condition, jnp.all(jnp.isfinite(matrix))


def _dense_initial_regularity(
    prepared,
    initialization,
    time,
    args,
    /,
):
    nonlinear_problem = prepared.initialization.nonlinear_problem
    if nonlinear_problem is None:
        return (
            jnp.asarray(int(DAERegularityStatus.NOT_RUN), dtype=jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(jnp.nan, dtype=initialization.residual_norm.dtype),
        )
    unknown = _unknown_guess(
        initialization.state,
        initialization.state_rate,
        prepared.initialization.state_indices,
        prepared.initialization.rate_indices,
    )
    arguments = _DAEInitializationArguments(
        time,
        initialization.state,
        initialization.state_rate,
        args,
    )
    source = nonlinear_problem.state_space
    target = nonlinear_problem.residual_space

    def residual(current):
        return target.flatten(
            nonlinear_problem.residual(source.unflatten(current), arguments)
        )

    matrix = jax.jacfwd(residual)(source.flatten(unknown))
    rank, condition, finite = _matrix_regularity(matrix)
    status = _regularity_status(
        rank,
        condition,
        finite,
        int(prepared.problem.initial_state.size),
        prepared.plan.policy.regularity.condition_limit,
    )
    return status, rank, condition


def _initial_regularity(initialization, dimension: int, condition_limit, /):
    nonlinear_result = initialization.nonlinear_result
    if nonlinear_result is None:
        rank = jnp.asarray(-1, dtype=jnp.int32)
        condition = jnp.asarray(jnp.nan, dtype=initialization.residual_norm.dtype)
        converged = initialization.valid
    else:
        diagnostics = nonlinear_result.diagnostics
        rank = diagnostics.final_linear_rank
        condition = diagnostics.final_linear_condition_estimate
        converged = diagnostics.final_linear_converged
    status = _regularity_status(
        rank,
        condition,
        converged,
        dimension,
        condition_limit,
    )
    return status, rank, condition


def _dense_stage_regularity(prepared, state, arguments, /):
    source = prepared.stage_problem.state_space
    target = prepared.stage_problem.residual_space
    coordinates = source.flatten(state)

    def residual(current):
        return target.flatten(
            prepared.stage_problem.residual(source.unflatten(current), arguments)
        )

    matrix = jax.jacfwd(residual)(coordinates)
    return _matrix_regularity(matrix)


def _solve_prepared(
    prepared: PreparedDAESolve,
    /,
    *,
    args: Any,
    initial_state: ArrayLike | None,
    initial_state_rate: ArrayLike | None,
) -> DifferentialAlgebraicSolution:
    problem = prepared.problem
    system = problem.system
    policy = prepared.plan.policy
    times = lax.stop_gradient(prepared.time_grid.times)
    state_guess = problem.initial_state if initial_state is None else initial_state
    rate_guess = (
        problem.initial_state_rate if initial_state_rate is None else initial_state_rate
    )
    initialization = _initialize_dae(
        prepared.initialization,
        state_guess,
        rate_guess,
        times[0],
        args=args,
        termination=policy.initialization_termination,
    )
    differential_equations = system.structure.differential_equation_mask(
        system.state_shape
    )
    algebraic_equations = system.structure.algebraic_equation_mask(system.state_shape)
    indices = jnp.arange(prepared.time_grid.num_steps, dtype=jnp.int32)
    step_sizes = jnp.diff(times)

    def scan_step(carry, inputs):
        (
            previous,
            previous_previous,
            previous_rate,
            previous_step,
            prior_valid,
            state_history,
            rate_history,
            history_times,
        ) = carry
        index, target_time, step_size = inputs
        if isinstance(policy.method, ThetaMethod):
            order = jnp.asarray(policy.method.capabilities.order, dtype=jnp.int32)
            predictor = previous + step_size * previous_rate
        else:
            order = jnp.minimum(
                jnp.asarray(policy.method.maximum_order, dtype=jnp.int32),
                index + 1,
            )
            predictor = _general_bdf_predict(
                state_history,
                rate_history,
                history_times,
                target_time,
                order,
                index + 1,
            )

        def solve_step(_):
            if isinstance(policy.method, ThetaMethod):
                arguments = endpoint_theta_stage_arguments(
                    policy.method,
                    target_time=target_time,
                    previous=previous,
                    previous_rate=previous_rate,
                    step_size=step_size,
                    model_args=args,
                )
            else:
                arguments = _history_stage_arguments(
                    target_time=target_time,
                    state_history=state_history,
                    history_times=history_times,
                    order=order,
                    model_args=args,
                )
            refreshed = refresh_nonlinear(
                prepared.stage_solve,
                prepared.stage_problem,
                predictor,
                args=arguments,
            )
            nonlinear_result = implicit_root_result(refreshed)
            state = jnp.asarray(nonlinear_result.state)
            if isinstance(policy.method, ThetaMethod):
                state_rate = endpoint_theta_rate(
                    policy.method,
                    state,
                    previous,
                    previous_rate,
                    step_size,
                )
            else:
                state_rate = _general_bdf_rate(
                    state,
                    state_history,
                    history_times,
                    target_time,
                    order,
                )
            inputs = (
                None
                if problem.input_policy is None
                else problem.input_policy.evaluate(target_time, state, args)
            )
            scaled = system.scaled_residual(
                target_time,
                state,
                state_rate,
                args,
                inputs=inputs,
            )
            residual_norm = _masked_rms(
                scaled,
                jnp.ones(system.state_shape, dtype=bool),
            )
            differential_norm = _masked_rms(scaled, differential_equations)
            constraint_norm = _masked_rms(scaled, algebraic_equations)
            residual_threshold = policy.nonlinear_termination.residual_threshold(
                nonlinear_result.diagnostics.initial_residual_norm
            )
            nonlinear_status = nonlinear_result.status
            nonlinear_success = nonlinear_status == int(NonlinearStatus.SUCCESS)
            finite = (
                jnp.all(jnp.isfinite(state))
                & jnp.all(jnp.isfinite(state_rate))
                & jnp.isfinite(residual_norm)
            )
            residual_accepted = residual_norm <= residual_threshold
            valid = nonlinear_success & finite & residual_accepted
            status = jnp.where(
                ~finite,
                int(DAEStatus.NONFINITE),
                jnp.where(
                    _linear_failure(nonlinear_status),
                    int(DAEStatus.LINEAR_FAILED),
                    jnp.where(
                        ~nonlinear_success,
                        int(DAEStatus.NONLINEAR_FAILED),
                        jnp.where(
                            ~residual_accepted,
                            int(DAEStatus.RESIDUAL_TOO_LARGE),
                            int(DAEStatus.SUCCESS),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            diagnostics = nonlinear_result.diagnostics
            return (
                state,
                state_rate,
                valid,
                status,
                residual_norm,
                residual_threshold,
                differential_norm,
                constraint_norm,
                nonlinear_status,
                jnp.asarray(True),
                diagnostics.iterations,
                diagnostics.residual_evaluations,
                diagnostics.jacobian_preparations,
                diagnostics.linear_solves,
                diagnostics.linear_iterations,
                diagnostics.rejected_steps,
                diagnostics.setup_refreshes,
                diagnostics.numeric_refreshes,
                diagnostics.final_linear_status,
                diagnostics.final_linear_rank,
                diagnostics.final_linear_condition_estimate,
                diagnostics.final_linear_residual_norm,
                diagnostics.final_linear_converged,
            )

        def skip_step(_):
            nan_state = jnp.full_like(previous, jnp.nan)
            zero = jnp.asarray(0, dtype=jnp.int32)
            infinity = jnp.asarray(jnp.inf, dtype=previous.real.dtype)
            return (
                nan_state,
                nan_state,
                jnp.asarray(False),
                jnp.asarray(int(DAEStatus.NOT_RUN), dtype=jnp.int32),
                infinity,
                infinity,
                infinity,
                infinity,
                zero,
                jnp.asarray(False),
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                jnp.asarray(-1, dtype=jnp.int32),
                jnp.asarray(-1, dtype=jnp.int32),
                jnp.asarray(jnp.nan, dtype=previous.real.dtype),
                jnp.asarray(jnp.nan, dtype=previous.real.dtype),
                jnp.asarray(False),
            )

        output = lax.cond(prior_valid, solve_step, skip_step, operand=None)
        state, state_rate, valid, *_ = output
        next_state_history = jnp.concatenate(
            (state[None, ...], state_history[:-1]), axis=0
        )
        next_rate_history = jnp.concatenate(
            (state_rate[None, ...], rate_history[:-1]), axis=0
        )
        next_history_times = jnp.concatenate(
            (target_time[None], history_times[:-1]), axis=0
        )
        next_carry = (
            state,
            previous,
            state_rate,
            step_size,
            valid,
            next_state_history,
            next_rate_history,
            next_history_times,
        )
        return next_carry, output

    initial_step = step_sizes[0]
    state_history = jnp.broadcast_to(
        initialization.state,
        (5,) + initialization.state.shape,
    )
    rate_history = jnp.broadcast_to(
        initialization.state_rate,
        (5,) + initialization.state_rate.shape,
    )
    history_times = jnp.full((5,), times[0], dtype=times.dtype)
    initial_carry = (
        initialization.state,
        initialization.state,
        initialization.state_rate,
        initial_step,
        initialization.valid,
        state_history,
        rate_history,
        history_times,
    )
    _, outputs = lax.scan(
        scan_step,
        initial_carry,
        (indices, times[1:], step_sizes),
    )
    (
        step_states,
        step_rates,
        step_valid,
        step_status,
        step_residual_norm,
        step_residual_threshold,
        step_differential_norm,
        step_constraint_norm,
        nonlinear_status,
        nonlinear_status_valid,
        nonlinear_iterations,
        residual_evaluations,
        jacobian_preparations,
        linear_solves,
        linear_iterations,
        globalization_rejections,
        setup_refreshes,
        numeric_refreshes,
        final_linear_status,
        final_linear_rank,
        final_linear_condition,
        final_linear_residual,
        final_linear_converged,
    ) = outputs
    initial_status = jnp.where(
        initialization.valid,
        int(DAEStatus.SUCCESS),
        int(DAEStatus.INITIALIZATION_FAILED),
    ).astype(jnp.int32)
    states = jnp.concatenate((initialization.state[None, ...], step_states), axis=0)
    state_rates = jnp.concatenate(
        (initialization.state_rate[None, ...], step_rates),
        axis=0,
    )
    valid = jnp.concatenate((initialization.valid[None], step_valid), axis=0)
    status = jnp.concatenate((initial_status[None], step_status), axis=0)
    residual_norm = jnp.concatenate(
        (initialization.residual_norm[None], step_residual_norm),
        axis=0,
    )
    residual_threshold = jnp.concatenate(
        (initialization.residual_threshold[None], step_residual_threshold),
        axis=0,
    )
    differential_norm = jnp.concatenate(
        (
            initialization.differential_residual_norm[None],
            step_differential_norm,
        ),
        axis=0,
    )
    constraint_norm = jnp.concatenate(
        (initialization.constraint_norm[None], step_constraint_norm),
        axis=0,
    )
    step_rate_valid = jnp.broadcast_to(
        step_valid.reshape((-1,) + (1,) * len(system.state_shape)),
        step_rates.shape,
    )
    rate_valid = jnp.concatenate(
        (initialization.rate_valid[None, ...], step_rate_valid),
        axis=0,
    )
    if isinstance(policy.method, ThetaMethod):
        assert policy.method.capabilities.order is not None
        orders = jnp.full_like(indices, policy.method.capabilities.order)
    else:
        orders = jnp.minimum(
            jnp.asarray(policy.method.maximum_order, dtype=jnp.int32),
            indices + 1,
        )
    accepted_count = jnp.sum(step_valid, dtype=jnp.int32)
    save_step_indices = jnp.concatenate(
        (
            jnp.asarray((-1,), dtype=jnp.int32),
            jnp.where(step_valid, indices, -2),
        )
    )
    step_history = DAEStepHistory(
        accepted_times=jnp.where(step_valid, times[1:], jnp.nan),
        step_sizes=step_sizes,
        orders=orders,
        error_ratios=jnp.where(step_valid, 0.0, jnp.inf),
        source_attempt_indices=indices,
        valid=step_valid,
        count=accepted_count,
        save_step_indices=save_step_indices,
    )
    attempt_status = jnp.where(
        ~nonlinear_status_valid,
        int(DAEAttemptStatus.NOT_RUN),
        jnp.where(
            step_valid,
            int(DAEAttemptStatus.ACCEPTED),
            jnp.where(
                step_status == int(DAEStatus.LINEAR_FAILED),
                int(DAEAttemptStatus.LINEAR_REJECTED),
                jnp.where(
                    step_status == int(DAEStatus.NONFINITE),
                    int(DAEAttemptStatus.NONFINITE_REJECTED),
                    jnp.where(
                        step_status == int(DAEStatus.RESIDUAL_TOO_LARGE),
                        int(DAEAttemptStatus.RESIDUAL_REJECTED),
                        int(DAEAttemptStatus.NONLINEAR_REJECTED),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    attempt_history = DAEAttemptHistory(
        times=times[:-1],
        proposed_step_sizes=step_sizes,
        orders=orders,
        status=attempt_status,
        error_ratios=jnp.where(step_valid, 0.0, jnp.inf),
        nonlinear_status=nonlinear_status,
        nonlinear_iterations=nonlinear_iterations,
        residual_evaluations=residual_evaluations,
        jacobian_preparations=jacobian_preparations,
        linear_solves=linear_solves,
        linear_iterations=linear_iterations,
        globalization_rejections=globalization_rejections,
        setup_refreshes=setup_refreshes,
        numeric_refreshes=numeric_refreshes,
        stale_jacobian_retries=jnp.zeros_like(indices),
        linear_rejections=_linear_failure(nonlinear_status).astype(jnp.int32),
        residual_certifications=nonlinear_status_valid.astype(jnp.int32),
        valid=nonlinear_status_valid,
        count=jnp.sum(nonlinear_status_valid, dtype=jnp.int32),
    )
    if policy.regularity.mode == "periodic":
        (
            consistency_status,
            consistency_rank,
            consistency_condition,
        ) = _dense_initial_regularity(
            prepared,
            initialization,
            times[0],
            args,
        )

        def probe_regularity(index, state):
            requested = step_valid[index] & ((index % policy.regularity.interval) == 0)
            previous_index = jnp.maximum(index - 1, 0)
            arguments = _stage_arguments(
                time=times[index + 1],
                previous=states[index],
                previous_previous=states[previous_index],
                step_size=step_sizes[index],
                previous_step_size=step_sizes[previous_index],
                order=orders[index],
                model_args=args,
            )

            def probe(_):
                rank, condition, finite = _dense_stage_regularity(
                    prepared,
                    state,
                    arguments,
                )
                status = _regularity_status(
                    rank,
                    condition,
                    finite,
                    int(problem.initial_state.size),
                    policy.regularity.condition_limit,
                )
                return status, rank, condition, jnp.asarray(True)

            def skip(_):
                return (
                    jnp.asarray(
                        int(DAERegularityStatus.NOT_RUN),
                        dtype=jnp.int32,
                    ),
                    jnp.asarray(-1, dtype=jnp.int32),
                    jnp.asarray(jnp.nan, dtype=state.real.dtype),
                    jnp.asarray(False),
                )

            return lax.cond(requested, probe, skip, operand=None)

        (
            stage_regularity_status,
            stage_regularity_rank,
            stage_regularity_condition,
            stage_regularity_valid,
        ) = jax.vmap(probe_regularity)(indices, step_states)
    else:
        (
            consistency_status,
            consistency_rank,
            consistency_condition,
        ) = _initial_regularity(
            initialization,
            int(problem.initial_state.size),
            policy.regularity.condition_limit,
        )
        stage_regularity_status = _regularity_status(
            final_linear_rank,
            final_linear_condition,
            final_linear_converged,
            int(problem.initial_state.size),
            policy.regularity.condition_limit,
        )
        stage_regularity_rank = final_linear_rank
        stage_regularity_condition = final_linear_condition
        stage_regularity_valid = step_valid
    regularity = DAERegularityEvidence(
        consistency_status=consistency_status,
        consistency_rank=consistency_rank,
        consistency_condition_estimate=consistency_condition,
        stage_status=stage_regularity_status,
        stage_rank=stage_regularity_rank,
        stage_condition_estimate=stage_regularity_condition,
        stage_valid=stage_regularity_valid,
        consistency_operator="configured-consistency-coordinate-jacobian",
        stage_operator="implicit-stage:F_y+shift*F_ydot",
    )
    valid_count = jnp.sum(valid, dtype=jnp.int32)
    last_node = jnp.maximum(valid_count - 1, 0)
    history_indices = jnp.maximum(
        last_node - jnp.arange(6, dtype=jnp.int32),
        0,
    )
    history_step_indices = jnp.maximum(
        last_node - 1 - jnp.arange(5, dtype=jnp.int32),
        0,
    )
    history_steps = step_sizes[history_step_indices]
    history_steps = jnp.where(
        jnp.arange(5, dtype=jnp.int32) < last_node,
        history_steps,
        initial_step,
    )
    last_step_index = jnp.maximum(last_node - 1, 0)
    last_step = jnp.where(
        last_node > 0,
        step_sizes[last_step_index],
        initial_step,
    )
    accepted_order = jnp.where(
        last_node > 0,
        orders[last_step_index],
        jnp.asarray(1, dtype=jnp.int32),
    )
    if isinstance(policy.method, ThetaMethod):
        last_alpha = 1.0 / (policy.method.theta * last_step)
    else:
        prior_indices = jnp.maximum(
            last_node - 1 - jnp.arange(5, dtype=jnp.int32),
            0,
        )
        last_alpha, _ = _general_bdf_shift_offset(
            states[prior_indices],
            times[prior_indices],
            times[last_node],
            accepted_order,
        )
    continuation = DAEContinuation(
        time=times[last_node],
        states=states[history_indices],
        state_rates=state_rates[history_indices],
        times=times[history_indices],
        step_sizes=history_steps,
        history_depth=jnp.minimum(valid_count, 6),
        accepted_order=accepted_order,
        previous_error_ratio=jnp.asarray(1.0, dtype=states.real.dtype),
        proposed_step_size=last_step,
        jacobian_age=jnp.asarray(0, dtype=jnp.int32),
        last_alpha=last_alpha,
        nonlinear_solve=None,
        problem_id=problem.problem_id,
        input_policy_id=(
            None if problem.input_policy is None else problem.input_policy.policy_id
        ),
        system_id=system.system_id,
        method_id=policy.method.method_id,
        initialization_id=problem.initialization.initialization_id,
        nonlinear_method_id=policy.nonlinear_method.method_id,
        stage_linear_plan_id=prepared.stage_linear_plan_id,
    )
    regularity_failed = jnp.any(
        (stage_regularity_status == int(DAERegularityStatus.NUMERICALLY_SINGULAR))
        & stage_regularity_valid
    ) | (consistency_status == int(DAERegularityStatus.NUMERICALLY_SINGULAR))
    termination_status = jnp.where(
        jnp.all(valid),
        int(DAETerminationStatus.SUCCESS),
        jnp.where(
            ~initialization.valid,
            int(DAETerminationStatus.INITIALIZATION_FAILED),
            int(DAETerminationStatus.NONLINEAR_FAILURE),
        ),
    ).astype(jnp.int32)
    if policy.regularity.failure == "status":
        termination_status = jnp.where(
            regularity_failed,
            int(DAETerminationStatus.REGULARITY_FAILED),
            termination_status,
        ).astype(jnp.int32)
    if policy.failure == "error":
        states = eqx.error_if(
            states,
            termination_status != int(DAETerminationStatus.SUCCESS),
            "DAE solve failed.",
        )
    return DifferentialAlgebraicSolution(
        times=times,
        states=states,
        state_rates=state_rates,
        valid=valid,
        rate_valid=rate_valid,
        status=status,
        residual_norm=residual_norm,
        residual_threshold=residual_threshold,
        differential_residual_norm=differential_norm,
        constraint_norm=constraint_norm,
        step_history=step_history,
        attempt_history=attempt_history,
        initialization=initialization,
        continuation=continuation,
        regularity=regularity,
        replay=DAEReplayEvidence(
            accepted_steps=accepted_count,
            selected_chunk_size=prepared.plan.replay_chunk_size,
            estimated_memory_bytes=prepared.plan.replay_memory_bytes,
            checkpointing=policy.replay.checkpointing,
        ),
        termination_status=termination_status,
        problem_id=problem.problem_id,
        system_id=system.system_id,
        time_id=prepared.time_grid.time_id,
        input_policy_id=(
            None if problem.input_policy is None else problem.input_policy.policy_id
        ),
        plan_id=prepared.plan.plan_id,
        prepared_id=prepared.prepared_id,
        source_discretization_bundle=problem.discretization_bundle,
        nonlinear_method_id=policy.nonlinear_method.method_id,
        stage_linear_plan_id=prepared.stage_linear_plan_id,
        initialization_linear_plan_id=prepared.initialization_linear_plan_id,
        method_id=policy.method.method_id,
        adaptive=False,
    )


def solve_dae(
    problem_or_prepared: DifferentialAlgebraicProblem | PreparedDAESolve,
    time_grid: TimeGrid | None = None,
    /,
    *,
    policy: DAESolvePolicy | None = None,
    args: Any = _DEFAULT_ARGS,
    initial_state: ArrayLike | None = None,
    initial_state_rate: ArrayLike | None = None,
    continuation: DAEContinuation | None = None,
) -> DifferentialAlgebraicSolution:
    """Solve one regular index-1 DAE on a fixed or adaptive time grid."""
    if isinstance(problem_or_prepared, PreparedDAESolve):
        if time_grid is not None or policy is not None:
            raise ValueError("time_grid and policy must be omitted for a prepared solve.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, DifferentialAlgebraicProblem):
        if time_grid is None:
            raise ValueError("time_grid is required for an unprepared DAE problem.")
        prepared = prepare_dae(problem_or_prepared, time_grid, policy=policy)
    else:
        raise TypeError("Expected a DifferentialAlgebraicProblem or PreparedDAESolve.")
    if continuation is not None and (
        initial_state is not None or initial_state_rate is not None
    ):
        raise ValueError(
            "continuation cannot be combined with initial_state or initial_state_rate."
        )
    runtime_args = prepared.problem.args if args is _DEFAULT_ARGS else args
    if prepared.plan.policy.adaptive is not None:
        from ._dae_adaptive import solve_adaptive_dae

        return solve_adaptive_dae(
            prepared,
            runtime_args,
            initial_state,
            initial_state_rate,
            continuation,
        )
    if continuation is not None:
        raise ValueError("DAE continuation requires an adaptive solve policy.")
    return _solve_prepared(
        prepared,
        args=runtime_args,
        initial_state=initial_state,
        initial_state_rate=initial_state_rate,
    )


__all__ = [
    "DAEAdaptivePolicy",
    "DAEAttemptHistory",
    "DAEAttemptStatus",
    "DAEContinuation",
    "DAEFailureMode",
    "DAERegularityEvidence",
    "DAERegularityFailureMode",
    "DAERegularityMode",
    "DAERegularityPolicy",
    "DAERegularityStatus",
    "DAEReplayEvidence",
    "DAEReplayMode",
    "DAEReplayPolicy",
    "DAESolvePlan",
    "DAESolvePolicy",
    "DAEStatus",
    "DAEStepHistory",
    "DAETemporalReusePolicy",
    "DAETerminationStatus",
    "DifferentialAlgebraicProblem",
    "DifferentialAlgebraicSolution",
    "PreparedDAESolve",
    "initialize_dae",
    "plan_dae",
    "prepare_dae",
    "solve_dae",
]
