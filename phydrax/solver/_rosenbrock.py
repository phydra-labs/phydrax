#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import RealizedTemporalMesh
from ..dynamics import TimeGrid
from ..linalg import (
    ArraySpace,
    FGMRES,
    FunctionLinearOperator,
    LinearSolvePlan,
    LinearSolvePolicy,
    LinearSystem,
    solve,
    TolerancePolicy,
)
from ._differential import DifferentialProblem, DifferentialSolution
from ._temporal_method import (
    configuration_id,
    native_differentiation_evidence,
    TemporalMethodCapabilities,
    TemporalSolveEvidence,
)
from ._temporal_precision import TemporalPrecisionPolicy


_DEFAULT_ARGS = object()


class RosenbrockWMethod(StrictModule, NonTrainableState):
    """Four-stage, third-order L-stable RA34PW2 Rosenbrock-W method."""

    capabilities: TemporalMethodCapabilities
    propagation: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    stage: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    weights: tuple[float, ...] = eqx.field(static=True)
    embedded_weights: tuple[float, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(self):
        self.propagation = (
            (0.0, 0.0, 0.0, 0.0),
            (8.7173304301691801e-01, 0.0, 0.0, 0.0),
            (8.4457060015369423e-01, -1.1299064236484185e-01, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        )
        self.stage = (
            (4.3586652150845900e-01, 0.0, 0.0, 0.0),
            (-8.7173304301691801e-01, 4.3586652150845900e-01, 0.0, 0.0),
            (
                -9.0338057013044082e-01,
                5.4180672388095326e-02,
                4.3586652150845900e-01,
                0.0,
            ),
            (
                2.4212380706095346e-01,
                -1.2232505839045147,
                5.4526025533510214e-01,
                4.3586652150845900e-01,
            ),
        )
        self.weights = (
            2.4212380706095346e-01,
            -1.2232505839045147,
            1.5452602553351020,
            4.3586652150845900e-01,
        )
        self.embedded_weights = (
            3.7810903145819369e-01,
            -9.6042292212423178e-02,
            5.0000000000000000e-01,
            2.1793326075422950e-01,
        )
        self.method_id = "temporal:rosenbrock-w:ra34pw2"
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("explicit-ode",),
            method_class="rosenbrock-w",
            order=3,
            embedded_order=2,
            dense_order=None,
            adaptive=True,
            history_depth=1,
            stage_abscissae=tuple(sum(row) for row in self.propagation),
            causal_stage_extent=1.0,
            a_stable=True,
            l_stable=True,
            stiffly_accurate=True,
            verified=True,
            method_id=self.method_id,
        )


class RosenbrockAdaptivePolicy(StrictModule, NonTrainableState):
    """Bounded accept/reject control for the embedded RA34PW2 pair."""

    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    initial_step: float = eqx.field(static=True)
    minimum_step: float = eqx.field(static=True)
    maximum_step: float | None = eqx.field(static=True)
    safety: float = eqx.field(static=True)
    minimum_factor: float = eqx.field(static=True)
    maximum_factor: float = eqx.field(static=True)
    maximum_accepted_steps: int = eqx.field(static=True)
    maximum_attempts: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative_tolerance: float = 1e-5,
        absolute_tolerance: float = 1e-8,
        initial_step: float = 1e-2,
        minimum_step: float = 1e-12,
        maximum_step: float | None = None,
        safety: float = 0.9,
        minimum_factor: float = 0.2,
        maximum_factor: float = 5.0,
        maximum_accepted_steps: int = 4096,
        maximum_attempts: int = 8192,
    ):
        values = tuple(
            float(value)
            for value in (
                relative_tolerance,
                absolute_tolerance,
                initial_step,
                minimum_step,
                safety,
                minimum_factor,
                maximum_factor,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Rosenbrock adaptive controls must be finite and positive.")
        relative, absolute, initial, minimum, safety_, factor_min, factor_max = values
        maximum = None if maximum_step is None else float(maximum_step)
        if maximum is not None and (
            not isfinite(maximum) or maximum <= 0.0 or maximum < minimum
        ):
            raise ValueError("maximum_step must be finite and at least minimum_step.")
        if initial < minimum or (maximum is not None and initial > maximum):
            raise ValueError("initial_step must lie inside the configured step bounds.")
        if not 0.0 < safety_ <= 1.0 or not 0.0 < factor_min <= 1.0 <= factor_max:
            raise ValueError("Adaptive safety and factor bounds are invalid.")
        accepted = int(maximum_accepted_steps)
        attempts = int(maximum_attempts)
        if accepted < 1 or attempts < accepted:
            raise ValueError(
                "Adaptive capacities must be positive and attempts cover accepts."
            )
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.initial_step = initial
        self.minimum_step = minimum
        self.maximum_step = maximum
        self.safety = safety_
        self.minimum_factor = factor_min
        self.maximum_factor = factor_max
        self.maximum_accepted_steps = accepted
        self.maximum_attempts = attempts
        self.policy_id = canonical_fingerprint(
            {
                "kind": "rosenbrock-adaptive-policy",
                "error_norm": "componentwise-wrms",
                "relative_tolerance": relative,
                "absolute_tolerance": absolute,
                "initial_step": initial,
                "minimum_step": minimum,
                "maximum_step": maximum,
                "safety": safety_,
                "minimum_factor": factor_min,
                "maximum_factor": factor_max,
                "maximum_accepted_steps": accepted,
                "maximum_attempts": attempts,
            }
        )


class _JacobianAction(eqx.Module):
    drift: Any
    time: Array
    state: Array
    args: Any

    def __call__(self, direction: Array, /) -> Array:
        return jax.jvp(
            lambda value: jnp.asarray(self.drift(self.time, value, self.args)),
            (self.state,),
            (direction,),
        )[1]


class _ShiftedJacobianAction(eqx.Module):
    jacobian: _JacobianAction
    scale: Array

    def __call__(self, direction: Array, /) -> Array:
        return direction - self.scale * self.jacobian(direction)


def _time_derivative(problem: DifferentialProblem, time: Array, state: Array, args: Any):
    return jax.jvp(
        lambda value: jnp.asarray(problem.drift(value, state, args)),
        (time,),
        (jnp.ones_like(time),),
    )[1]


def _default_linear_policy(state: Array, /) -> LinearSolvePolicy:
    restart = max(1, min(20, int(state.size) if state.shape else 1))
    return LinearSolvePolicy(
        FGMRES(restart=restart),
        tolerance=TolerancePolicy(relative=1e-8, absolute=1e-10, max_steps=64),
    )


class _RosenbrockStepResult(StrictModule):
    next_state: Array
    defect: Array
    successful: Array
    stage_status: Array
    residual_norm: Array
    relative_residual: Array
    stage_finite: Array
    stage_converged: Array
    iterations: Array


def _rosenbrock_step(
    problem: DifferentialProblem,
    method: RosenbrockWMethod,
    policy: LinearSolvePolicy | LinearSolvePlan,
    space: ArraySpace,
    time: Array,
    state: Array,
    step_size: Array,
    args: Any,
    precision: TemporalPrecisionPolicy,
    /,
) -> _RosenbrockStepResult:
    propagation = jnp.asarray(method.propagation, dtype=state.real.dtype)
    stage_matrix = jnp.asarray(method.stage, dtype=state.real.dtype)
    weights = jnp.asarray(method.weights, dtype=state.real.dtype)
    embedded = jnp.asarray(method.embedded_weights, dtype=state.real.dtype)
    jacobian = _JacobianAction(problem.drift, time, state, args)
    time_derivative = _time_derivative(problem, time, state, args)
    increments: list[Array] = []
    statuses: list[Array] = []
    residual_norms: list[Array] = []
    relative_residuals: list[Array] = []
    finite_stages: list[Array] = []
    converged_stages: list[Array] = []
    iteration_counts: list[Array] = []
    successful = jnp.asarray(True)
    for index in range(4):
        stage_state = state
        correction = jnp.zeros_like(state)
        for previous in range(index):
            stage_state = (
                stage_state + propagation[index, previous] * increments[previous]
            )
            correction = correction + stage_matrix[index, previous] * increments[previous]
        stage_time = time + step_size * jnp.sum(propagation[index])
        rhs = precision.residual(
            step_size * jnp.asarray(problem.drift(stage_time, stage_state, args))
        )
        rhs = precision.residual(rhs + step_size * jacobian(correction))
        rhs = precision.residual(
            rhs + step_size**2 * jnp.sum(stage_matrix[index]) * time_derivative
        )
        gamma = stage_matrix[index, index]
        operator = FunctionLinearOperator(
            _ShiftedJacobianAction(jacobian, step_size * gamma),
            source=space,
            target=space,
            operator_id=f"{method.method_id}:shifted-jacobian",
        )
        linear_result = solve(LinearSystem(operator), rhs, policy=policy)
        increments.append(jnp.asarray(linear_result.value, dtype=state.dtype))
        statuses.append(jnp.asarray(linear_result.status, dtype=jnp.int32))
        residual_norms.append(
            jnp.asarray(linear_result.diagnostics.residual_norm, dtype=state.real.dtype)
        )
        relative_residuals.append(
            jnp.asarray(
                linear_result.diagnostics.relative_residual,
                dtype=state.real.dtype,
            )
        )
        finite_stages.append(jnp.asarray(linear_result.diagnostics.finite, dtype=bool))
        converged_stages.append(
            jnp.asarray(linear_result.diagnostics.converged, dtype=bool)
        )
        iteration_counts.append(
            jnp.asarray(linear_result.diagnostics.iterations, dtype=jnp.int32)
        )
        successful = (
            successful
            & linear_result.successful
            & linear_result.diagnostics.finite
            & linear_result.diagnostics.converged
        )
    stacked = jnp.stack(increments)
    accumulated_state = precision.accumulation(state)
    accumulated_stages = precision.accumulation(stacked)
    accumulated_weights = precision.accumulation(weights)
    accumulated_embedded = precision.accumulation(embedded)
    next_state = (
        accumulated_state + jnp.tensordot(accumulated_weights, accumulated_stages, axes=1)
    ).astype(state.dtype)
    embedded_state = (
        accumulated_state
        + jnp.tensordot(accumulated_embedded, accumulated_stages, axes=1)
    ).astype(state.dtype)
    defect = precision.accumulation(next_state - embedded_state)
    finite = jnp.all(jnp.isfinite(next_state)) & jnp.all(jnp.isfinite(defect))
    return _RosenbrockStepResult(
        next_state=next_state,
        defect=defect,
        successful=successful & finite,
        stage_status=jnp.stack(statuses),
        residual_norm=jnp.stack(residual_norms),
        relative_residual=jnp.stack(relative_residuals),
        stage_finite=jnp.stack(finite_stages),
        stage_converged=jnp.stack(converged_stages),
        iterations=jnp.stack(iteration_counts),
    )


def _defect_norm(
    defect: Array,
    precision: TemporalPrecisionPolicy,
    /,
) -> Array:
    return precision.decision(jnp.sqrt(jnp.mean(jnp.abs(defect) ** 2)))


def _error_ratio(
    state: Array,
    next_state: Array,
    defect: Array,
    controller: RosenbrockAdaptivePolicy,
    precision: TemporalPrecisionPolicy,
    /,
) -> Array:
    scale = precision.decision(
        controller.absolute_tolerance
        + controller.relative_tolerance * jnp.maximum(jnp.abs(state), jnp.abs(next_state))
    )
    safe_scale = jnp.maximum(scale, jnp.finfo(scale.dtype).tiny)
    scaled = precision.decision(defect) / safe_scale
    return precision.decision(jnp.sqrt(jnp.mean(jnp.abs(scaled) ** 2)))


def _solve_rosenbrock_fixed(
    problem: DifferentialProblem,
    time_grid: TimeGrid,
    /,
    *,
    method: RosenbrockWMethod | None = None,
    linear_policy: LinearSolvePolicy | LinearSolvePlan | None = None,
    args: Any = _DEFAULT_ARGS,
    precision: TemporalPrecisionPolicy | None = None,
) -> DifferentialSolution:
    """Integrate one deterministic ODE on a fixed grid with matrix-free RA34PW2."""
    if not isinstance(problem, DifferentialProblem) or problem.stochastic:
        raise TypeError("solve_rosenbrock requires a deterministic DifferentialProblem.")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid.")
    geometry = problem.state_geometry
    if geometry is not None and not geometry.trivial:
        raise ValueError("Rosenbrock-W currently requires Euclidean state geometry.")
    times = lax.stop_gradient(time_grid.times)
    times = eqx.error_if(
        times,
        ~jnp.isclose(times[0], problem.t0) | ~jnp.isclose(times[-1], problem.t1),
        "TimeGrid endpoints must match the differential problem.",
    )
    selected = RosenbrockWMethod() if method is None else method
    if not isinstance(selected, RosenbrockWMethod):
        raise TypeError("method must be RosenbrockWMethod or None.")
    policy = (
        _default_linear_policy(problem.initial_state)
        if linear_policy is None
        else linear_policy
    )
    if not isinstance(policy, (LinearSolvePolicy, LinearSolvePlan)):
        raise TypeError("linear_policy must be a LinearSolvePolicy, plan, or None.")
    precision_ = TemporalPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, TemporalPrecisionPolicy):
        raise TypeError("precision must be a TemporalPrecisionPolicy or None.")
    precision_.validate_implicit_state(problem.initial_state)
    runtime_args = problem.args if args is _DEFAULT_ARGS else args
    space = ArraySpace(problem.initial_state.shape, dtype=problem.initial_state.dtype)

    def advance(carry, values):
        state, prior_valid = carry
        time, step_size = values

        def solve_step(_):
            result = _rosenbrock_step(
                problem,
                selected,
                policy,
                space,
                time,
                state,
                step_size,
                runtime_args,
                precision_,
            )
            return (
                result.next_state,
                result.successful,
                _defect_norm(result.defect, precision_),
                jnp.sum(result.iterations, dtype=jnp.int32),
            )

        def skip_step(_):
            return (
                jnp.full_like(state, jnp.nan),
                jnp.asarray(False),
                jnp.asarray(jnp.inf, dtype=state.real.dtype),
                jnp.asarray(0, dtype=jnp.int32),
            )

        next_state, valid, error, iterations = lax.cond(
            prior_valid, solve_step, skip_step, operand=None
        )
        return (next_state, valid), (next_state, valid, error, iterations)

    (_, _), (step_states, step_valid, errors, iterations) = lax.scan(
        advance,
        (problem.initial_state, jnp.asarray(True)),
        (times[:-1], jnp.diff(times)),
    )
    states = jnp.concatenate((problem.initial_state[None, ...], step_states), axis=0)
    valid = jnp.concatenate((jnp.asarray([True]), step_valid))
    configuration = configuration_id(
        (selected, policy, precision_.policy_id, time_grid.time_id),
        prefix="temporal-configuration",
    )
    mesh = RealizedTemporalMesh(
        times[0],
        times[1:],
        jnp.ones((time_grid.num_steps,), dtype=bool),
        time_grid.num_steps,
        adaptive=False,
        source_plan_id=configuration,
        requested_time_id=time_grid.time_id,
    )
    evidence = TemporalSolveEvidence(
        selected.capabilities,
        native_differentiation_evidence(
            "adjoint:jax-discrete-linear-solves",
            adaptive=False,
        ),
        equation_form="explicit-ode",
        backend_id="backend:phydrax:rosenbrock-w",
        configuration_id=configuration,
        controller_id=f"controller:fixed-grid:{time_grid.time_id}",
        event_id=None,
        adaptive=False,
        dense=False,
        maximum_steps=time_grid.num_steps,
        precision_evidence=precision_.evidence_for(problem.initial_state, times),
    )
    successful = jnp.all(valid)
    output_states = jax.vmap(precision_.output)(states)
    return DifferentialSolution(
        times=times,
        states=output_states,
        valid=valid,
        terminal_time=times[-1],
        terminal_state=output_states[-1],
        backend_result=jnp.where(successful, 0, 1),
        stats={
            "num_steps": jnp.asarray(time_grid.num_steps, dtype=jnp.int32),
            "linear_iterations": jnp.sum(iterations),
            "embedded_defect_norm": errors,
        },
        solver_name="RA34PW2",
        interpretation=problem.interpretation,
        state_geometry_id=problem.state_geometry_id,
        solver_id=selected.method_id,
        resolved_method="RA34PW2:matrix-free-exact-jacobian",
        discretization_bundle=problem.discretization_bundle,
        temporal_mesh=mesh,
        backend_successful=successful,
        temporal_evidence=evidence,
        problem_id=problem.problem_id,
    )


__all__ = [
    "RosenbrockAdaptivePolicy",
    "RosenbrockWMethod",
]
