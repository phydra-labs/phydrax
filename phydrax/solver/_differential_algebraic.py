#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import math
from enum import IntEnum
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..dynamics import DifferentialAlgebraicSystem, TimeGrid
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
from ._dae_initialization import (
    _initialize_dae,
    _masked_rms,
    _prepare_dae_initialization,
    _PreparedDAEInitialization,
    _scaled_space,
    DAEInitializationResult,
    DAEInitializationSpec,
)
from ._solution_validation import validate_solution_arrays


DAEIntegrationMethod: TypeAlias = Literal["bdf1", "bdf2"]
DAEFailureMode: TypeAlias = Literal["status", "error"]
_DEFAULT_ARGS = object()


class DAEStatus(IntEnum):
    SUCCESS = 0
    INITIALIZATION_FAILED = 1
    NONLINEAR_FAILED = 2
    LINEAR_FAILED = 3
    NONFINITE = 4
    RESIDUAL_TOO_LARGE = 5
    NOT_RUN = 6


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


class DAESolvePolicy(StrictModule):
    """Fixed-grid BDF integration and native nonlinear solve policy."""

    nonlinear_method: NewtonKrylov | NewtonTrustRegion
    nonlinear_termination: NonlinearTermination
    initialization_method: NewtonKrylov | NewtonTrustRegion
    initialization_termination: NonlinearTermination
    integration_method: DAEIntegrationMethod = eqx.field(static=True)
    max_step_ratio: float = eqx.field(static=True)
    failure: DAEFailureMode = eqx.field(static=True)

    def __init__(
        self,
        *,
        integration_method: DAEIntegrationMethod = "bdf2",
        nonlinear_method: AbstractNonlinearMethod | None = None,
        nonlinear_termination: NonlinearTermination | None = None,
        initialization_method: AbstractNonlinearMethod | None = None,
        initialization_termination: NonlinearTermination | None = None,
        max_step_ratio: float = 2.0,
        failure: DAEFailureMode = "status",
    ):
        if integration_method not in ("bdf1", "bdf2"):
            raise ValueError("integration_method must be 'bdf1' or 'bdf2'.")
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
        ratio = float(max_step_ratio)
        if not math.isfinite(ratio) or ratio < 1.0:
            raise ValueError("max_step_ratio must be finite and at least one.")
        if failure not in ("status", "error"):
            raise ValueError("failure must be 'status' or 'error'.")
        self.nonlinear_method = stage_method
        self.nonlinear_termination = stage_termination
        self.initialization_method = initial_method
        self.initialization_termination = initial_termination
        self.integration_method = integration_method
        self.max_step_ratio = ratio
        self.failure = failure


class DifferentialAlgebraicProblem(StrictModule):
    """Implicit initial-value problem with an explicit consistency contract."""

    system: DifferentialAlgebraicSystem
    initial_state: Array
    initial_state_rate: Array
    args: Any
    initialization: DAEInitializationSpec
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: DifferentialAlgebraicSystem,
        initial_state: ArrayLike,
        /,
        *,
        initial_state_rate: ArrayLike | None = None,
        args: Any = None,
        initialization: DAEInitializationSpec | None = None,
        problem_id: str | None = None,
    ):
        if not isinstance(system, DifferentialAlgebraicSystem):
            raise TypeError("system must be a DifferentialAlgebraicSystem.")
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
        initial_spec = (
            DAEInitializationSpec.index_one()
            if initialization is None
            else initialization
        )
        if not isinstance(initial_spec, DAEInitializationSpec):
            raise TypeError("initialization must be a DAEInitializationSpec or None.")
        self.system = system
        self.initial_state = state
        self.initial_state_rate = state_rate
        self.args = args
        self.initialization = initial_spec
        self.problem_id = _identifier(
            problem_id,
            (
                system.system_id,
                system.state_shape,
                np.dtype(state.dtype).str,
                initial_spec.initialization_id,
            ),
            "dae-problem",
        )


class DAESolvePlan(StrictModule):
    """Validated fixed-grid DAE integration policy and structural identity."""

    policy: DAESolvePolicy
    system_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    time_id: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_dtype: str = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
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
        durations = np.asarray(time_grid.durations, dtype=float)
        if policy.integration_method == "bdf2" and durations.size > 1:
            ratios = durations[1:] / durations[:-1]
            if np.any(ratios > policy.max_step_ratio) or np.any(
                ratios < 1.0 / policy.max_step_ratio
            ):
                raise ValueError(
                    "BDF2 adjacent step ratios exceed the declared max_step_ratio."
                )
        self.policy = policy
        self.system_id = problem.system.system_id
        self.problem_id = problem.problem_id
        self.time_id = time_grid.time_id
        self.state_shape = problem.system.state_shape
        self.state_dtype = np.dtype(problem.initial_state.dtype).str
        self.num_steps = time_grid.num_steps
        self.plan_id = _identifier(
            None,
            (
                self.system_id,
                self.problem_id,
                self.time_id,
                self.state_shape,
                self.state_dtype,
                policy.integration_method,
                policy.max_step_ratio,
                _method_identity(policy.nonlinear_method),
                _termination_identity(policy.nonlinear_termination),
                _method_identity(policy.initialization_method),
                _termination_identity(policy.initialization_termination),
            ),
            "dae-plan",
        )


def _bdf_rate(
    state: Array,
    previous: Array,
    previous_previous: Array,
    step_size: Array,
    previous_step_size: Array,
    order: Array,
    /,
) -> Array:
    def first_order(_):
        return (state - previous) / step_size

    def second_order(_):
        ratio = step_size / previous_step_size
        return (
            ((1.0 + 2.0 * ratio) / (1.0 + ratio)) * state
            - (1.0 + ratio) * previous
            + (ratio * ratio / (1.0 + ratio)) * previous_previous
        ) / step_size

    return lax.cond(order == 1, first_order, second_order, operand=None)


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


class _DAEStageArguments(StrictModule):
    time: Array
    previous: Array
    previous_previous: Array
    step_size: Array
    previous_step_size: Array
    order: Array
    model_args: Any


class _DAEStageResidual(StrictModule):
    system: DifferentialAlgebraicSystem

    def __call__(self, state: Array, arguments: _DAEStageArguments, /) -> Array:
        state_rate = _bdf_rate(
            state,
            arguments.previous,
            arguments.previous_previous,
            arguments.step_size,
            arguments.previous_step_size,
            arguments.order,
        )
        return self.system.scaled_residual(
            arguments.time,
            state,
            state_rate,
            arguments.model_args,
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
) -> _DAEStageArguments:
    return _DAEStageArguments(
        time,
        previous,
        previous_previous,
        step_size,
        previous_step_size,
        order,
        model_args,
    )


class PreparedDAESolve(StrictModule):
    """DAE problem, grid, consistency root, and reusable BDF-stage root."""

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


class DifferentialAlgebraicSolution(StrictModule):
    """Fixed-grid DAE trajectory with node, step, and nonlinear evidence."""

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
    step_sizes: Array
    orders: Array
    nonlinear_status: Array
    nonlinear_status_valid: Array
    nonlinear_iterations: Array
    residual_evaluations: Array
    jacobian_preparations: Array
    linear_solves: Array
    linear_iterations: Array
    globalization_rejections: Array
    setup_refreshes: Array
    numeric_refreshes: Array
    initialization: DAEInitializationResult
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    time_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    nonlinear_method_id: str = eqx.field(static=True)
    stage_linear_plan_id: str = eqx.field(static=True)
    initialization_linear_plan_id: str = eqx.field(static=True)
    integration_method: DAEIntegrationMethod = eqx.field(static=True)
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
        step_sizes: Array,
        orders: Array,
        nonlinear_status: Array,
        nonlinear_status_valid: Array,
        nonlinear_iterations: Array,
        residual_evaluations: Array,
        jacobian_preparations: Array,
        linear_solves: Array,
        linear_iterations: Array,
        globalization_rejections: Array,
        setup_refreshes: Array,
        numeric_refreshes: Array,
        initialization: DAEInitializationResult,
        problem_id: str,
        system_id: str,
        time_id: str,
        plan_id: str,
        prepared_id: str,
        nonlinear_method_id: str,
        stage_linear_plan_id: str,
        initialization_linear_plan_id: str,
        integration_method: DAEIntegrationMethod,
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
        step_shape = (int(validated.times.size) - 1,)
        step_values = (
            (step_sizes, "step_sizes"),
            (orders, "orders"),
            (nonlinear_status, "nonlinear_status"),
            (nonlinear_status_valid, "nonlinear_status_valid"),
            (nonlinear_iterations, "nonlinear_iterations"),
            (residual_evaluations, "residual_evaluations"),
            (jacobian_preparations, "jacobian_preparations"),
            (linear_solves, "linear_solves"),
            (linear_iterations, "linear_iterations"),
            (globalization_rejections, "globalization_rejections"),
            (setup_refreshes, "setup_refreshes"),
            (numeric_refreshes, "numeric_refreshes"),
        )
        for values, name in step_values:
            if jnp.asarray(values).shape != step_shape:
                raise ValueError(f"DAE {name} must have shape {step_shape}.")
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
        self.step_sizes = jnp.asarray(step_sizes)
        self.orders = jnp.asarray(orders, dtype=jnp.int32)
        self.nonlinear_status = jnp.asarray(nonlinear_status, dtype=jnp.int32)
        self.nonlinear_status_valid = jnp.asarray(nonlinear_status_valid, dtype=bool)
        self.nonlinear_iterations = jnp.asarray(nonlinear_iterations, dtype=jnp.int32)
        self.residual_evaluations = jnp.asarray(residual_evaluations, dtype=jnp.int32)
        self.jacobian_preparations = jnp.asarray(
            jacobian_preparations,
            dtype=jnp.int32,
        )
        self.linear_solves = jnp.asarray(linear_solves, dtype=jnp.int32)
        self.linear_iterations = jnp.asarray(linear_iterations, dtype=jnp.int32)
        self.globalization_rejections = jnp.asarray(
            globalization_rejections,
            dtype=jnp.int32,
        )
        self.setup_refreshes = jnp.asarray(setup_refreshes, dtype=jnp.int32)
        self.numeric_refreshes = jnp.asarray(numeric_refreshes, dtype=jnp.int32)
        self.initialization = initialization
        self.sample_shape = validated.sample_shape
        self.state_shape = validated.state_shape
        self.problem_id = str(problem_id)
        self.system_id = str(system_id)
        self.time_id = str(time_id)
        self.plan_id = str(plan_id)
        self.prepared_id = str(prepared_id)
        self.nonlinear_method_id = str(nonlinear_method_id)
        self.stage_linear_plan_id = str(stage_linear_plan_id)
        self.initialization_linear_plan_id = str(initialization_linear_plan_id)
        self.integration_method = integration_method
        self.differentiation_mode = "fixed-grid-discrete-implicit"
        self.grid_origin = "user"
        self.approximation_id = f"fixed-grid-{integration_method}"

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid)


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
        spec=problem.initialization,
        method=resolved_policy.initialization_method,
        termination=resolved_policy.initialization_termination,
    )
    step_size = time_grid.durations[0]
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
        space_id=f"{problem.system.system_id}:bdf-state",
    )
    residual_space = _scaled_space(
        problem.system.state_shape,
        problem.initial_state.dtype,
        jnp.ones_like(problem.system.residual_scale),
        space_id=f"{problem.system.system_id}:bdf-residual",
    )
    stage_problem = NonlinearSystemProblem(
        _DAEStageResidual(problem.system),
        state_space=state_space,
        residual_space=residual_space,
        problem_id=f"{problem.system.system_id}:bdf-stage-root",
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
        ) = carry
        index, target_time, step_size = inputs
        order = (
            jnp.asarray(1, dtype=jnp.int32)
            if policy.integration_method == "bdf1"
            else jnp.where(index == 0, 1, 2).astype(jnp.int32)
        )
        predictor = _predict(
            previous,
            previous_previous,
            previous_rate,
            step_size,
            previous_step,
            order,
        )

        def solve_step(_):
            arguments = _stage_arguments(
                time=target_time,
                previous=previous,
                previous_previous=previous_previous,
                step_size=step_size,
                previous_step_size=previous_step,
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
            state_rate = _bdf_rate(
                state,
                previous,
                previous_previous,
                step_size,
                previous_step,
                order,
            )
            scaled = system.scaled_residual(
                target_time,
                state,
                state_rate,
                args,
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
            )

        output = lax.cond(prior_valid, solve_step, skip_step, operand=None)
        state, state_rate, valid, *_ = output
        next_carry = (
            state,
            previous,
            state_rate,
            step_size,
            valid,
        )
        return next_carry, output

    initial_step = step_sizes[0]
    initial_carry = (
        initialization.state,
        initialization.state,
        initialization.state_rate,
        initial_step,
        initialization.valid,
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
    if policy.failure == "error":
        states = eqx.error_if(states, jnp.any(~valid), "DAE solve failed.")
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
        step_sizes=step_sizes,
        orders=(
            jnp.ones_like(indices)
            if policy.integration_method == "bdf1"
            else jnp.where(indices == 0, 1, 2).astype(jnp.int32)
        ),
        nonlinear_status=nonlinear_status,
        nonlinear_status_valid=nonlinear_status_valid,
        nonlinear_iterations=nonlinear_iterations,
        residual_evaluations=residual_evaluations,
        jacobian_preparations=jacobian_preparations,
        linear_solves=linear_solves,
        linear_iterations=linear_iterations,
        globalization_rejections=globalization_rejections,
        setup_refreshes=setup_refreshes,
        numeric_refreshes=numeric_refreshes,
        initialization=initialization,
        problem_id=problem.problem_id,
        system_id=system.system_id,
        time_id=prepared.time_grid.time_id,
        plan_id=prepared.plan.plan_id,
        prepared_id=prepared.prepared_id,
        nonlinear_method_id=policy.nonlinear_method.method_id,
        stage_linear_plan_id=prepared.stage_linear_plan_id,
        initialization_linear_plan_id=prepared.initialization_linear_plan_id,
        integration_method=policy.integration_method,
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
) -> DifferentialAlgebraicSolution:
    """Solve one regular index-1 DAE on every node of a fixed ``TimeGrid``."""
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
    runtime_args = prepared.problem.args if args is _DEFAULT_ARGS else args
    return _solve_prepared(
        prepared,
        args=runtime_args,
        initial_state=initial_state,
        initial_state_rate=initial_state_rate,
    )


__all__ = [
    "DAEFailureMode",
    "DAEIntegrationMethod",
    "DAESolvePlan",
    "DAESolvePolicy",
    "DAEStatus",
    "DifferentialAlgebraicProblem",
    "DifferentialAlgebraicSolution",
    "PreparedDAESolve",
    "initialize_dae",
    "plan_dae",
    "prepare_dae",
    "solve_dae",
]
