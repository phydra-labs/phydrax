#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any, get_origin, Literal, TypeAlias

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array

from .._strict import StrictModule
from ..solver import AbstractGeometricSolver, DifferentialProblem, solve_diffrax
from ..stochastic._solver_transition import DifferentialTransitionKernel
from ..stochastic._state_space import (
    GaussianObservationModel,
    GaussianStatePrior,
    LinearGaussianObservationModel,
    LinearGaussianTransitionKernel,
    StateSpaceProblem,
    StateSpaceStepContext,
)
from ._gaussian_factor import (
    gaussian_factor_from_covariance,
    gaussian_factor_log_determinant,
    gaussian_factor_quadratic_form,
)
from ._nonlinear_gaussian import (
    first_order_gaussian_transform,
    scaled_unscented_transform,
    spherical_radial_cubature,
)


ContinuousDiscreteGaussianMethod: TypeAlias = Literal["extended", "cubature", "unscented"]
ContinuousDiscreteGaussianStatus: TypeAlias = Literal[
    "success", "solver_failure", "transform_failure", "nonfinite"
]
CONTINUOUS_DISCRETE_GAUSSIAN_SUCCESS = 0
CONTINUOUS_DISCRETE_GAUSSIAN_SOLVER_FAILURE = 1
CONTINUOUS_DISCRETE_GAUSSIAN_TRANSFORM_FAILURE = 2
CONTINUOUS_DISCRETE_GAUSSIAN_NONFINITE = 3
CONTINUOUS_DISCRETE_MAX_DENSE_DIMENSION = 64


def continuous_discrete_gaussian_status_name(
    value: int, /
) -> ContinuousDiscreteGaussianStatus:
    code = int(value)
    if code == CONTINUOUS_DISCRETE_GAUSSIAN_SUCCESS:
        return "success"
    if code == CONTINUOUS_DISCRETE_GAUSSIAN_SOLVER_FAILURE:
        return "solver_failure"
    if code == CONTINUOUS_DISCRETE_GAUSSIAN_TRANSFORM_FAILURE:
        return "transform_failure"
    if code == CONTINUOUS_DISCRETE_GAUSSIAN_NONFINITE:
        return "nonfinite"
    raise ValueError(f"Unknown continuous-discrete Gaussian status code {code}.")


class ContinuousDiscreteGaussianFilterResult(StrictModule):
    """Gaussian continuous-discrete history with physical-axis provenance."""

    predicted_means: Array
    predicted_covariances: Array
    filtered_means: Array
    filtered_covariances: Array
    transition_cross_covariances: Array
    predicted_observation_means: Array
    predicted_observation_covariances: Array
    state_observation_cross_covariances: Array
    innovations: Array
    normalized_innovation_squared: Array
    incremental_log_likelihood: Array
    cumulative_log_likelihood: Array
    observed_counts: Array
    step_valid: Array
    valid: Array
    status: Array
    solver_status: Array
    times: Array
    problem: StateSpaceProblem
    state_shape: tuple[int, ...] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    case_axes: tuple[str, ...] = eqx.field(static=True)
    observation_axes: tuple[str, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    sequence_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)
    sensor_id: str | None = eqx.field(static=True)
    input_id: str | None = eqx.field(static=True)
    parameter_id: str | None = eqx.field(static=True)
    basis_id: str | None = eqx.field(static=True)
    discretization_id: str | None = eqx.field(static=True)
    method: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    transition_method: str = eqx.field(static=True)
    observation_transform_method: str = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    solver_method: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    adjoint_method: str = eqx.field(static=True)
    stepsize_controller_method: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    transition_approximation_id: str = eqx.field(static=True)
    observation_approximation_id: str | None = eqx.field(static=True)
    covariance_regularization: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    unscented_alpha: float = eqx.field(static=True)
    unscented_beta: float = eqx.field(static=True)
    unscented_kappa: float = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.step_valid, axis=-1)


class ContinuousDiscreteGaussianSmootherResult(StrictModule):
    """Fixed-interval Gaussian marginals from stored transition cross-moments."""

    smoothed_means: Array
    smoothed_covariances: Array
    smoothing_gains: Array
    valid: Array
    status: Array
    filter_result: ContinuousDiscreteGaussianFilterResult
    method_id: str = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid | ~self.filter_result.step_valid, axis=-1)


def _validate_configuration(
    problem: StateSpaceProblem,
    method: ContinuousDiscreteGaussianMethod,
    covariance_regularization: float,
    rank_tolerance: float,
    unscented_alpha: float,
    unscented_beta: float,
    unscented_kappa: float,
) -> tuple[
    GaussianStatePrior,
    LinearGaussianTransitionKernel | DifferentialTransitionKernel,
    GaussianObservationModel | LinearGaussianObservationModel,
    int,
    int,
]:
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    prior = problem.model.prior
    transition = problem.model.transition
    observation = problem.model.observation
    if not isinstance(prior, GaussianStatePrior):
        raise TypeError(
            "Continuous-discrete Gaussian filtering requires GaussianStatePrior."
        )
    if not isinstance(
        transition, (LinearGaussianTransitionKernel, DifferentialTransitionKernel)
    ):
        raise TypeError(
            "Continuous-discrete Gaussian filtering requires a declared affine "
            "transition or DifferentialTransitionKernel."
        )
    if not isinstance(
        observation, (GaussianObservationModel, LinearGaussianObservationModel)
    ):
        raise TypeError(
            "Continuous-discrete Gaussian filtering requires a Gaussian observation model."
        )
    if isinstance(transition, DifferentialTransitionKernel):
        if transition.interpretation != "ito":
            raise ValueError(
                "Continuous-discrete Gaussian moment propagation requires Itô dynamics."
            )
        solver = _resolved_moment_solver(transition)
        if isinstance(solver, AbstractGeometricSolver):
            raise ValueError(
                "Continuous-discrete augmented moments do not support geometric solvers."
            )
        if not _moment_solver_supports_ode(solver):
            raise ValueError(
                "Continuous-discrete augmented moments require a deterministic "
                f"ODE-compatible Diffrax solver; got {type(solver).__name__}."
            )
    if method not in ("extended", "cubature", "unscented"):
        raise ValueError("method must be 'extended', 'cubature', or 'unscented'.")
    regularization = float(covariance_regularization)
    tolerance = float(rank_tolerance)
    alpha = float(unscented_alpha)
    beta = float(unscented_beta)
    kappa = float(unscented_kappa)
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("covariance_regularization must be finite and nonnegative.")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("rank_tolerance must be finite and nonnegative.")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("unscented_alpha must be finite and positive.")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("unscented_beta must be finite and nonnegative.")
    if not np.isfinite(kappa):
        raise ValueError("unscented_kappa must be finite.")
    state_size = prod(problem.model.state_shape) if problem.model.state_shape else 1
    observation_size = (
        prod(problem.model.observation_shape) if problem.model.observation_shape else 1
    )
    if (
        state_size > CONTINUOUS_DISCRETE_MAX_DENSE_DIMENSION
        or observation_size > CONTINUOUS_DISCRETE_MAX_DENSE_DIMENSION
    ):
        raise ValueError(
            "Continuous-discrete dense Gaussian filtering supports state and "
            f"observation dimensions up to {CONTINUOUS_DISCRETE_MAX_DENSE_DIMENSION}."
        )
    return prior, transition, observation, state_size, observation_size


def _diffusion_covariance(
    transition: DifferentialTransitionKernel,
    time: Array,
    state: Array,
    context: StateSpaceStepContext,
    /,
) -> Array:
    size = prod(transition.state_shape) if transition.state_shape else 1
    covariance = jnp.zeros((size, size), dtype=state.dtype)
    for term in transition.wiener_terms:
        coefficient = jnp.asarray(term.coefficient(time, state, context))
        matrix = coefficient.reshape((size, term.noise_size))
        covariance = covariance + matrix @ jnp.conj(matrix.T)
    return covariance


def _resolved_moment_solver(transition: DifferentialTransitionKernel, /) -> Any:
    if transition.solver is not None:
        return transition.solver
    return dfx.Tsit5()


def _moment_solver_supports_ode(solver: Any, /) -> bool:
    structure = get_origin(solver.term_structure) or solver.term_structure
    return isinstance(structure, type) and issubclass(dfx.ODETerm, structure)


def _resolved_moment_controller(transition: DifferentialTransitionKernel, /) -> Any:
    if transition.stepsize_controller is not None:
        return transition.stepsize_controller
    solver = _resolved_moment_solver(transition)
    return (
        None if isinstance(solver, dfx.AbstractAdaptiveSolver) else dfx.ConstantStepSize()
    )


def _solve_differential_flow(
    transition: DifferentialTransitionKernel,
    initial: Array,
    start: Array,
    end: Array,
    context: StateSpaceStepContext,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    state_shape = transition.state_shape
    size = prod(state_shape) if state_shape else 1
    flat_initial = jnp.asarray(initial).reshape((size,))
    augmented_initial = jnp.concatenate(
        (flat_initial, jnp.zeros((size * size,), dtype=flat_initial.dtype))
    )

    def augmented_drift(time, augmented, args):
        state_flat = augmented[:size]
        state = state_flat.reshape(state_shape)
        covariance = augmented[size:].reshape((size, size))

        def flat_drift(value):
            return jnp.asarray(
                transition.drift(time, value.reshape(state_shape), args)
            ).reshape((size,))

        drift = flat_drift(state_flat)
        jacobian = jax.jacfwd(flat_drift)(state_flat)
        covariance_rate = (
            jacobian @ covariance
            + covariance @ jnp.conj(jacobian.T)
            + _diffusion_covariance(transition, time, state, args)
        )
        return jnp.concatenate((drift, covariance_rate.reshape((-1,))))

    def solve_segment(augmented, bounds):
        left, right = bounds
        differential = DifferentialProblem(
            augmented_drift,
            augmented,
            t0=left,
            t1=right,
            args=context,
        )
        solution = solve_diffrax(
            differential,
            save_times=jnp.asarray([right]),
            solver=_resolved_moment_solver(transition),
            stepsize_controller=_resolved_moment_controller(transition),
            adjoint=(
                dfx.DirectAdjoint() if transition.adjoint is None else transition.adjoint
            ),
            dt0=transition.dt0,
            rtol=transition.rtol,
            atol=transition.atol,
            max_steps=transition.max_steps,
            throw=False,
        )
        solved = solution.states[-1]
        backend_status = solution.backend_result._value.astype(jnp.int32)
        solver_valid = solution.backend_result == dfx.RESULTS.successful
        finite = solution.valid[-1] & jnp.all(jnp.isfinite(solved))
        return solved, solver_valid, finite, backend_status

    breakpoints = jnp.asarray(context.input_breakpoints, dtype=start.dtype)
    breakpoint_valid = jnp.asarray(context.input_breakpoint_valid, dtype=bool)
    padding_step = jnp.abs(end - start) + jnp.asarray(1.0, dtype=start.dtype)
    padding = end + padding_step * (
        jnp.arange(breakpoints.shape[0], dtype=start.dtype) + 1.0
    )
    candidates = jnp.sort(
        jnp.concatenate((jnp.where(breakpoint_valid, breakpoints, padding), end[None]))
    )

    def segment_step(carry, candidate):
        augmented, left, solver_valid, finite, backend_status = carry
        should_solve = (
            context.input_valid
            & solver_valid
            & finite
            & (candidate > left)
            & (candidate <= end)
        )

        def apply(values):
            state, lower, previous_solver_valid, previous_finite, previous_status = values
            safe_right = jnp.where(candidate > lower, candidate, lower + padding_step)
            solved, segment_solver_valid, segment_finite, segment_status = solve_segment(
                state, (lower, safe_right)
            )
            first_status = jnp.where(
                (previous_status == 0) & (segment_status != 0),
                segment_status,
                previous_status,
            )
            return (
                solved,
                candidate,
                previous_solver_valid & segment_solver_valid,
                previous_finite & segment_finite,
                first_status,
            )

        next_carry = jax.lax.cond(should_solve, apply, lambda values: values, carry)
        return next_carry, None

    (final, _, solver_valid, finite, backend_status), _ = jax.lax.scan(
        segment_step,
        (
            augmented_initial,
            start,
            jnp.asarray(True),
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
        ),
        candidates,
    )
    state = final[:size]
    process_covariance = final[size:].reshape((size, size))
    process_covariance = 0.5 * (process_covariance + jnp.conj(process_covariance.T))
    finite = (
        finite & jnp.all(jnp.isfinite(state)) & jnp.all(jnp.isfinite(process_covariance))
    )
    return (
        state,
        process_covariance,
        solver_valid,
        context.input_valid,
        finite,
        backend_status,
    )


def _nonlinear_transform(
    function,
    mean: Array,
    covariance: Array,
    method: ContinuousDiscreteGaussianMethod,
    /,
    *,
    rank_tolerance: float,
    unscented_alpha: float,
    unscented_beta: float,
    unscented_kappa: float,
):
    factor = gaussian_factor_from_covariance(
        covariance,
        rank_tolerance=rank_tolerance,
        factor_id="continuous-discrete-input",
    )
    if method == "extended":
        transformed = first_order_gaussian_transform(function, mean, factor)
    elif method == "cubature":
        transformed = spherical_radial_cubature(function, mean, factor)
    else:
        transformed = scaled_unscented_transform(
            function,
            mean,
            factor,
            alpha=unscented_alpha,
            beta=unscented_beta,
            kappa=unscented_kappa,
        )
    return factor, transformed


def _analytic_transition(
    transition: LinearGaussianTransitionKernel,
    mean: Array,
    covariance: Array,
    start: Array,
    end: Array,
    context: StateSpaceStepContext,
    /,
    *,
    rank_tolerance: float,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    parameters = transition.parameters(start, end, context)
    matrix = jnp.asarray(parameters.transition).reshape((mean.shape[0], mean.shape[0]))
    offset = jnp.asarray(parameters.offset).reshape(mean.shape)
    process_covariance = jnp.asarray(parameters.covariance).reshape(covariance.shape)
    input_factor = gaussian_factor_from_covariance(
        covariance,
        rank_tolerance=rank_tolerance,
        factor_id="continuous-discrete-analytic-input",
    )
    process_factor = gaussian_factor_from_covariance(
        process_covariance,
        rank_tolerance=rank_tolerance,
        factor_id="continuous-discrete-analytic-process",
    )
    predicted_mean = matrix @ mean + offset
    predicted_covariance_raw = (
        matrix @ covariance @ jnp.conj(matrix.T) + process_covariance
    )
    computed_covariance = 0.5 * (
        predicted_covariance_raw + jnp.conj(predicted_covariance_raw.T)
    )
    operands_valid = input_factor.valid & process_factor.valid
    predicted_covariance = jnp.where(
        operands_valid,
        computed_covariance,
        predicted_covariance_raw,
    )
    cross_covariance = covariance @ jnp.conj(matrix.T)
    predicted_factor = gaussian_factor_from_covariance(
        predicted_covariance,
        rank_tolerance=rank_tolerance,
        factor_id="continuous-discrete-analytic-prediction",
    )
    finite = (
        jnp.all(jnp.isfinite(matrix))
        & jnp.all(jnp.isfinite(offset))
        & jnp.all(jnp.isfinite(process_covariance))
        & jnp.all(jnp.isfinite(predicted_mean))
        & jnp.all(jnp.isfinite(predicted_covariance_raw))
        & jnp.all(jnp.isfinite(cross_covariance))
    )
    transform_valid = operands_valid & predicted_factor.valid
    return (
        predicted_mean,
        predicted_covariance,
        cross_covariance,
        jnp.asarray(True),
        transform_valid,
        finite,
        jnp.asarray(0, dtype=jnp.int32),
    )


def _differential_transition(
    transition: DifferentialTransitionKernel,
    mean: Array,
    covariance: Array,
    start: Array,
    end: Array,
    context: StateSpaceStepContext,
    method: ContinuousDiscreteGaussianMethod,
    /,
    *,
    rank_tolerance: float,
    unscented_alpha: float,
    unscented_beta: float,
    unscented_kappa: float,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    size = mean.shape[0]
    payload_size = size + size * size
    input_factor = gaussian_factor_from_covariance(
        covariance,
        rank_tolerance=rank_tolerance,
        factor_id="continuous-discrete-flow-input",
    )

    def flow_payload(point):
        (
            state,
            process_covariance,
            solver_valid,
            input_valid,
            finite,
            backend_status,
        ) = _solve_differential_flow(
            transition,
            point.reshape(transition.state_shape),
            start,
            end,
            context,
        )
        metadata = jnp.stack(
            (
                solver_valid.astype(state.dtype),
                input_valid.astype(state.dtype),
                finite.astype(state.dtype),
                backend_status.astype(state.dtype),
            )
        )
        return jnp.concatenate((state, process_covariance.reshape((-1,)), metadata))

    def evaluate(_):
        factor = input_factor.factor
        rank = input_factor.rank
        if method == "extended":
            output, pushforward = jax.linearize(flow_payload, mean)
            if rank == 0:
                state_directions = jnp.zeros((size, 0), dtype=mean.dtype)
            else:
                output_directions = jax.vmap(pushforward)(factor.T)
                state_directions = output_directions[:, :size].T
            transformed_covariance = state_directions @ jnp.conj(state_directions.T)
            cross_covariance = factor @ jnp.conj(state_directions.T)
            statuses = jnp.real(output[payload_size + 3]).astype(jnp.int32)[None]
            solver_valid = jnp.real(output[payload_size]) > 0.5
            flow_input_valid = jnp.real(output[payload_size + 1]) > 0.5
            flow_finite = jnp.real(output[payload_size + 2]) > 0.5
            output_mean = output[:payload_size]
        else:
            real_dtype = jnp.real(mean).dtype
            if rank == 0:
                canonical_points = jnp.zeros((1, 0), dtype=real_dtype)
                mean_weights = jnp.ones((1,), dtype=real_dtype)
                covariance_weights = mean_weights
            elif method == "cubature":
                scale = jnp.sqrt(jnp.asarray(rank, dtype=real_dtype))
                identity = jnp.eye(rank, dtype=real_dtype)
                canonical_points = jnp.concatenate(
                    (scale * identity, -scale * identity),
                    axis=0,
                )
                mean_weights = jnp.full(
                    (2 * rank,),
                    1.0 / (2.0 * rank),
                    dtype=real_dtype,
                )
                covariance_weights = mean_weights
            else:
                lambda_ = unscented_alpha**2 * (rank + unscented_kappa) - rank
                scaling = rank + lambda_
                if not np.isfinite(scaling) or scaling <= 0.0:
                    raise ValueError(
                        "Scaled unscented parameters require n + lambda > 0."
                    )
                radius = jnp.sqrt(jnp.asarray(scaling, dtype=real_dtype))
                identity = jnp.eye(rank, dtype=real_dtype)
                canonical_points = jnp.concatenate(
                    (
                        jnp.zeros((1, rank), dtype=real_dtype),
                        radius * identity,
                        -radius * identity,
                    ),
                    axis=0,
                )
                side_weight = 1.0 / (2.0 * scaling)
                mean_weights = jnp.asarray(
                    (lambda_ / scaling, *([side_weight] * (2 * rank))),
                    dtype=real_dtype,
                )
                covariance_weights = jnp.asarray(
                    (
                        lambda_ / scaling + 1.0 - unscented_alpha**2 + unscented_beta,
                        *([side_weight] * (2 * rank)),
                    ),
                    dtype=real_dtype,
                )
            physical_points = mean[None, :] + canonical_points @ factor.T
            outputs = jax.vmap(flow_payload)(physical_points)
            output_mean = oe.contract("p,po->o", mean_weights, outputs[:, :payload_size])
            state_points = outputs[:, :size]
            state_centered = state_points - output_mean[None, :size]
            input_centered = physical_points - mean[None, :]
            transformed_covariance = oe.contract(
                "p,pi,pj->ij",
                covariance_weights,
                state_centered,
                jnp.conj(state_centered),
            )
            cross_covariance = oe.contract(
                "p,pi,pj->ij",
                covariance_weights,
                input_centered,
                jnp.conj(state_centered),
            )
            statuses = jnp.real(outputs[:, payload_size + 3]).astype(jnp.int32)
            solver_valid = jnp.all(jnp.real(outputs[:, payload_size]) > 0.5)
            flow_input_valid = jnp.all(jnp.real(outputs[:, payload_size + 1]) > 0.5)
            flow_finite = jnp.all(jnp.real(outputs[:, payload_size + 2]) > 0.5)

        predicted_mean = output_mean[:size]
        process_covariance = output_mean[size:payload_size].reshape((size, size))
        predicted_covariance = transformed_covariance + process_covariance
        predicted_covariance = 0.5 * (
            predicted_covariance + jnp.conj(predicted_covariance.T)
        )
        predicted_factor = gaussian_factor_from_covariance(
            predicted_covariance,
            rank_tolerance=rank_tolerance,
            factor_id="continuous-discrete-flow-prediction",
        )
        finite = (
            flow_finite
            & jnp.all(jnp.isfinite(output_mean[:payload_size]))
            & jnp.all(jnp.isfinite(predicted_covariance))
            & jnp.all(jnp.isfinite(cross_covariance))
        )
        failure_mask = statuses != 0
        first_failure = jnp.argmax(failure_mask.astype(jnp.int32))
        backend_status = jnp.where(
            jnp.any(failure_mask),
            statuses[first_failure],
            jnp.asarray(0, dtype=jnp.int32),
        )
        transform_valid = input_factor.valid & flow_input_valid & predicted_factor.valid
        return (
            predicted_mean,
            predicted_covariance,
            cross_covariance,
            solver_valid,
            transform_valid,
            finite,
            backend_status,
        )

    def invalid_input(_):
        finite = jnp.all(jnp.isfinite(mean)) & jnp.all(jnp.isfinite(covariance))
        return (
            mean,
            covariance,
            covariance,
            jnp.asarray(True),
            jnp.asarray(False),
            finite,
            jnp.asarray(0, dtype=jnp.int32),
        )

    return jax.lax.cond(input_factor.valid, evaluate, invalid_input, None)


def _observation_moments(
    observation: GaussianObservationModel | LinearGaussianObservationModel,
    mean: Array,
    covariance: Array,
    time: Array,
    context: StateSpaceStepContext,
    method: ContinuousDiscreteGaussianMethod,
    state_shape: tuple[int, ...],
    observation_shape: tuple[int, ...],
    /,
    *,
    rank_tolerance: float,
    unscented_alpha: float,
    unscented_beta: float,
    unscented_kappa: float,
) -> tuple[Array, Array, Array, Array, Array]:
    state_size = mean.shape[0]
    observation_size = prod(observation_shape) if observation_shape else 1
    if isinstance(observation, LinearGaussianObservationModel):
        input_factor = gaussian_factor_from_covariance(
            covariance,
            rank_tolerance=rank_tolerance,
            factor_id="continuous-discrete-observation-input",
        )
        matrix, offset, noise_covariance = observation.parameters(time, context)
        matrix = jnp.asarray(matrix).reshape((observation_size, state_size))
        offset = jnp.asarray(offset).reshape((observation_size,))
        noise_covariance = jnp.asarray(noise_covariance).reshape(
            (observation_size, observation_size)
        )
        noise_factor = gaussian_factor_from_covariance(
            noise_covariance,
            rank_tolerance=rank_tolerance,
            factor_id="continuous-discrete-linear-observation-noise",
        )
        observation_mean = matrix @ mean + offset
        observation_covariance_raw = (
            matrix @ covariance @ jnp.conj(matrix.T) + noise_covariance
        )
        operands_valid = input_factor.valid & noise_factor.valid
        observation_covariance = jnp.where(
            operands_valid,
            0.5 * (observation_covariance_raw + jnp.conj(observation_covariance_raw.T)),
            observation_covariance_raw,
        )
        cross_covariance = covariance @ jnp.conj(matrix.T)
        observation_factor = gaussian_factor_from_covariance(
            observation_covariance,
            rank_tolerance=rank_tolerance,
            factor_id="continuous-discrete-linear-observation",
        )
        finite = (
            jnp.all(jnp.isfinite(matrix))
            & jnp.all(jnp.isfinite(offset))
            & jnp.all(jnp.isfinite(noise_covariance))
            & jnp.all(jnp.isfinite(observation_mean))
            & jnp.all(jnp.isfinite(observation_covariance_raw))
            & jnp.all(jnp.isfinite(cross_covariance))
        )
        return (
            observation_mean,
            observation_covariance,
            cross_covariance,
            operands_valid & observation_factor.valid,
            finite,
        )

    def location(point):
        return observation.location(point.reshape(state_shape), time, context).reshape(
            (observation_size,)
        )

    input_factor, transformed = _nonlinear_transform(
        location,
        mean,
        covariance,
        method,
        rank_tolerance=rank_tolerance,
        unscented_alpha=unscented_alpha,
        unscented_beta=unscented_beta,
        unscented_kappa=unscented_kappa,
    )
    noise_covariance = jnp.asarray(observation.covariance_at(time, context)).reshape(
        (observation_size, observation_size)
    )
    noise_factor = gaussian_factor_from_covariance(
        noise_covariance,
        rank_tolerance=rank_tolerance,
        factor_id="continuous-discrete-nonlinear-observation-noise",
    )
    observation_covariance_raw = transformed.factor.covariance + noise_covariance
    operands_valid = input_factor.valid & transformed.valid & noise_factor.valid
    observation_covariance = jnp.where(
        operands_valid,
        0.5 * (observation_covariance_raw + jnp.conj(observation_covariance_raw.T)),
        observation_covariance_raw,
    )
    observation_factor = gaussian_factor_from_covariance(
        observation_covariance,
        rank_tolerance=rank_tolerance,
        factor_id="continuous-discrete-nonlinear-observation",
    )
    finite = (
        jnp.all(jnp.isfinite(noise_covariance))
        & jnp.all(jnp.isfinite(transformed.mean))
        & jnp.all(jnp.isfinite(observation_covariance_raw))
        & jnp.all(jnp.isfinite(transformed.cross_covariance))
    )
    return (
        transformed.mean,
        observation_covariance,
        transformed.cross_covariance,
        operands_valid & observation_factor.valid,
        finite,
    )


def _observation_update(
    observation: GaussianObservationModel | LinearGaussianObservationModel,
    predicted_mean: Array,
    predicted_covariance: Array,
    value: Array,
    mask: Array,
    time: Array,
    context: StateSpaceStepContext,
    method: ContinuousDiscreteGaussianMethod,
    state_shape: tuple[int, ...],
    observation_shape: tuple[int, ...],
    /,
    *,
    covariance_regularization: float,
    rank_tolerance: float,
    unscented_alpha: float,
    unscented_beta: float,
    unscented_kappa: float,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    observation_size = prod(observation_shape) if observation_shape else 1
    (
        observation_mean,
        observation_covariance,
        cross_covariance,
        observation_transform_valid,
        observation_finite,
    ) = _observation_moments(
        observation,
        predicted_mean,
        predicted_covariance,
        time,
        context,
        method,
        state_shape,
        observation_shape,
        rank_tolerance=rank_tolerance,
        unscented_alpha=unscented_alpha,
        unscented_beta=unscented_beta,
        unscented_kappa=unscented_kappa,
    )
    mask_flat = jnp.asarray(mask, dtype=bool).reshape((observation_size,))
    active = mask_flat.astype(predicted_mean.dtype)
    innovation = jnp.where(
        mask_flat,
        jnp.asarray(value).reshape((observation_size,)) - observation_mean,
        0.0,
    )
    identity = jnp.eye(observation_size, dtype=predicted_mean.dtype)
    effective_covariance_raw = (
        observation_covariance * active[:, None] * active[None, :]
        + identity * (1.0 - active[:, None])
        + covariance_regularization * identity * active[:, None]
    )
    effective_covariance = jnp.where(
        observation_transform_valid,
        0.5 * (effective_covariance_raw + jnp.conj(effective_covariance_raw.T)),
        effective_covariance_raw,
    )
    effective_cross = cross_covariance * active[None, :]
    innovation_factor = gaussian_factor_from_covariance(
        effective_covariance,
        rank_tolerance=rank_tolerance,
        factor_id="continuous-discrete-innovation",
    )
    full_rank = innovation_factor.numerical_rank == observation_size
    observed_count = jnp.sum(mask_flat, dtype=jnp.int32)
    can_solve = (
        observation_transform_valid
        & observation_finite
        & innovation_factor.valid
        & full_rank
    )

    def solve_update(_):
        gain = jnp.conj(
            jnp.linalg.solve(
                effective_covariance,
                jnp.conj(effective_cross.T),
            ).T
        )
        filtered_mean = predicted_mean + gain @ innovation
        filtered_covariance_raw = predicted_covariance - gain @ jnp.conj(
            effective_cross.T
        )
        filtered_covariance = 0.5 * (
            filtered_covariance_raw + jnp.conj(filtered_covariance_raw.T)
        )
        filtered_factor = gaussian_factor_from_covariance(
            filtered_covariance,
            rank_tolerance=rank_tolerance,
            factor_id="continuous-discrete-filtered",
        )
        quadratic = gaussian_factor_quadratic_form(
            innovation_factor,
            innovation,
            rank_tolerance=rank_tolerance,
            support_tolerance=rank_tolerance,
        )
        logdet = gaussian_factor_log_determinant(
            innovation_factor,
            rank_tolerance=rank_tolerance,
        )
        log_likelihood = -0.5 * (
            quadratic + logdet + observed_count * jnp.log(2.0 * jnp.pi)
        )
        finite = (
            observation_finite
            & jnp.all(jnp.isfinite(filtered_mean))
            & jnp.all(jnp.isfinite(filtered_covariance_raw))
            & jnp.isfinite(log_likelihood)
        )
        return (
            filtered_mean,
            filtered_covariance,
            jnp.where(observed_count > 0, quadratic, 0.0),
            log_likelihood,
            filtered_factor.valid,
            finite,
        )

    def skip_update(_):
        return (
            predicted_mean,
            predicted_covariance,
            jnp.asarray(0.0, dtype=predicted_mean.dtype),
            jnp.asarray(0.0, dtype=jnp.real(predicted_mean).dtype),
            jnp.asarray(False),
            observation_finite
            & jnp.all(jnp.isfinite(innovation))
            & jnp.all(jnp.isfinite(effective_covariance_raw)),
        )

    (
        filtered_mean,
        filtered_covariance,
        nis,
        log_likelihood,
        filtered_transform_valid,
        finite,
    ) = jax.lax.cond(can_solve, solve_update, skip_update, None)
    transform_valid = (
        observation_transform_valid
        & innovation_factor.valid
        & full_rank
        & filtered_transform_valid
    )
    return (
        filtered_mean,
        filtered_covariance,
        observation_mean,
        observation_covariance,
        cross_covariance,
        innovation,
        nis,
        log_likelihood,
        transform_valid,
        finite,
    )


def _stack_case_histories(
    histories: list[tuple[Array, ...]],
    case_shape: tuple[int, ...],
    trailing_shapes: tuple[tuple[int, ...], ...],
    /,
) -> tuple[Array, ...]:
    num_steps = histories[0][0].shape[0]
    outputs = []
    for position, trailing in enumerate(trailing_shapes):
        stacked = jnp.stack([history[position] for history in histories], axis=0)
        outputs.append(stacked.reshape(case_shape + (num_steps,) + trailing))
    return tuple(outputs)


def _solver_provenance(
    transition: LinearGaussianTransitionKernel | DifferentialTransitionKernel, /
) -> tuple[str, str, str]:
    if isinstance(transition, LinearGaussianTransitionKernel):
        return (
            "analytic-linear-gaussian",
            transition.resolved_method,
            transition.resolved_method,
        )
    solver = _resolved_moment_solver(transition)
    solver_name = type(solver).__name__
    solver_id = f"solver:diffrax:{solver_name}"
    return solver_id, solver_name, "differential-flow-moment-propagation"


def continuous_discrete_gaussian_filter(
    problem: StateSpaceProblem,
    /,
    *,
    method: ContinuousDiscreteGaussianMethod = "extended",
    covariance_regularization: float = 0.0,
    rank_tolerance: float = 0.0,
    unscented_alpha: float = 1.0,
    unscented_beta: float = 2.0,
    unscented_kappa: float = 0.0,
    raise_on_failure: bool = False,
) -> ContinuousDiscreteGaussianFilterResult:
    """Filter irregular discrete observations of affine or differential dynamics.

    Differential transitions are propagated through the canonical Diffrax-backed
    transition configuration. Declared affine Gaussian transitions bypass the
    numerical flow and use their exact interval discretization.
    """
    (
        prior,
        transition,
        observation,
        state_size,
        observation_size,
    ) = _validate_configuration(
        problem,
        method,
        covariance_regularization,
        rank_tolerance,
        unscented_alpha,
        unscented_beta,
        unscented_kappa,
    )
    regularization = float(covariance_regularization)
    tolerance = float(rank_tolerance)
    alpha = float(unscented_alpha)
    beta = float(unscented_beta)
    kappa = float(unscented_kappa)
    sequence = problem.observations
    case_shape = sequence.case_shape
    case_count = prod(case_shape) if case_shape else 1
    num_steps = sequence.num_steps
    flat_times = sequence.times.reshape((case_count, num_steps))
    flat_values = sequence.values.reshape(
        (case_count, num_steps) + problem.model.observation_shape
    )
    flat_masks = sequence.observation_mask.reshape(
        (case_count, num_steps) + problem.model.observation_shape
    )
    flat_active = sequence.step_valid.reshape((case_count, num_steps))
    flat_initial_times = problem.initial_time.reshape((case_count,))
    flat_prior_means = prior.mean.reshape((case_count, state_size))
    flat_prior_covariances = prior.covariance.reshape(
        (case_count, state_size, state_size)
    )
    case_histories: list[tuple[Array, ...]] = []

    for case_index in range(case_count):
        mean = flat_prior_means[case_index]
        covariance = flat_prior_covariances[case_index]
        previous_time = flat_initial_times[case_index]
        carry_valid = jnp.asarray(True)
        carry_status = jnp.asarray(
            CONTINUOUS_DISCRETE_GAUSSIAN_SUCCESS,
            dtype=jnp.int32,
        )
        carry_solver_status = jnp.asarray(0, dtype=jnp.int32)
        cumulative = jnp.asarray(0.0, dtype=mean.dtype)
        records: list[list[Array]] = [[] for _ in range(17)]

        for step_index in range(num_steps):
            active = flat_active[case_index, step_index]
            target_time = flat_times[case_index, step_index]
            context = problem.step_context(case_index, step_index)

            def propagate(_):
                if isinstance(transition, LinearGaussianTransitionKernel):
                    return _analytic_transition(
                        transition,
                        mean,
                        covariance,
                        previous_time,
                        target_time,
                        context,
                        rank_tolerance=tolerance,
                    )
                return _differential_transition(
                    transition,
                    mean,
                    covariance,
                    previous_time,
                    target_time,
                    context,
                    method,
                    rank_tolerance=tolerance,
                    unscented_alpha=alpha,
                    unscented_beta=beta,
                    unscented_kappa=kappa,
                )

            def skip_propagation(_):
                return (
                    mean,
                    covariance,
                    covariance,
                    jnp.asarray(True),
                    jnp.asarray(True),
                    jnp.asarray(True),
                    jnp.asarray(0, dtype=jnp.int32),
                )

            (
                predicted_mean,
                predicted_covariance,
                transition_cross,
                solver_valid,
                transition_transform_valid,
                transition_finite,
                transition_backend_status,
            ) = jax.lax.cond(
                active & carry_valid,
                propagate,
                skip_propagation,
                None,
            )

            def update(_):
                return _observation_update(
                    observation,
                    predicted_mean,
                    predicted_covariance,
                    flat_values[case_index, step_index],
                    flat_masks[case_index, step_index],
                    target_time,
                    context,
                    method,
                    problem.model.state_shape,
                    problem.model.observation_shape,
                    covariance_regularization=regularization,
                    rank_tolerance=tolerance,
                    unscented_alpha=alpha,
                    unscented_beta=beta,
                    unscented_kappa=kappa,
                )

            def skip_update(_):
                return (
                    predicted_mean,
                    predicted_covariance,
                    jnp.zeros((observation_size,), dtype=mean.dtype),
                    jnp.zeros((observation_size, observation_size), dtype=mean.dtype),
                    jnp.zeros((state_size, observation_size), dtype=mean.dtype),
                    jnp.zeros((observation_size,), dtype=mean.dtype),
                    jnp.asarray(0.0, dtype=mean.dtype),
                    jnp.asarray(0.0, dtype=jnp.real(mean).dtype),
                    jnp.asarray(True),
                    jnp.asarray(True),
                )

            should_update = (
                active
                & carry_valid
                & solver_valid
                & transition_transform_valid
                & transition_finite
            )
            (
                filtered_mean,
                filtered_covariance,
                observation_mean,
                observation_covariance,
                observation_cross,
                innovation,
                nis,
                likelihood,
                update_transform_valid,
                update_finite,
            ) = jax.lax.cond(should_update, update, skip_update, None)
            accepted = should_update & update_transform_valid & update_finite
            record_valid = jnp.where(active, accepted, True)
            finite = (
                transition_finite
                & update_finite
                & jnp.all(jnp.isfinite(predicted_mean))
                & jnp.all(jnp.isfinite(predicted_covariance))
                & jnp.all(jnp.isfinite(filtered_mean))
                & jnp.all(jnp.isfinite(filtered_covariance))
            )
            local_status = jnp.where(
                ~finite,
                CONTINUOUS_DISCRETE_GAUSSIAN_NONFINITE,
                jnp.where(
                    ~solver_valid,
                    CONTINUOUS_DISCRETE_GAUSSIAN_SOLVER_FAILURE,
                    jnp.where(
                        ~(transition_transform_valid & update_transform_valid),
                        CONTINUOUS_DISCRETE_GAUSSIAN_TRANSFORM_FAILURE,
                        CONTINUOUS_DISCRETE_GAUSSIAN_SUCCESS,
                    ),
                ),
            ).astype(jnp.int32)
            status = jnp.where(
                ~active,
                CONTINUOUS_DISCRETE_GAUSSIAN_SUCCESS,
                jnp.where(~carry_valid, carry_status, local_status),
            ).astype(jnp.int32)
            solver_status = jnp.where(
                ~active,
                0,
                jnp.where(
                    ~carry_valid,
                    carry_solver_status,
                    transition_backend_status,
                ),
            ).astype(jnp.int32)
            increment = jnp.where(
                ~active,
                0.0,
                jnp.where(accepted, likelihood, -jnp.inf),
            )
            cumulative = cumulative + increment
            records[0].append(predicted_mean)
            records[1].append(predicted_covariance)
            records[2].append(filtered_mean)
            records[3].append(filtered_covariance)
            records[4].append(transition_cross)
            records[5].append(observation_mean)
            records[6].append(observation_covariance)
            records[7].append(observation_cross)
            records[8].append(innovation)
            records[9].append(nis)
            records[10].append(increment)
            records[11].append(cumulative)
            records[12].append(
                jnp.sum(flat_masks[case_index, step_index], dtype=jnp.int32)
            )
            records[13].append(record_valid)
            records[14].append(status)
            records[15].append(solver_status)
            records[16].append(target_time)
            new_failure = active & carry_valid & ~record_valid
            mean = jnp.where(active & accepted, filtered_mean, mean)
            covariance = jnp.where(
                active & accepted,
                filtered_covariance,
                covariance,
            )
            previous_time = jnp.where(active & accepted, target_time, previous_time)
            carry_status = jnp.where(new_failure, status, carry_status)
            carry_solver_status = jnp.where(
                new_failure,
                solver_status,
                carry_solver_status,
            )
            carry_valid = carry_valid & record_valid

        case_histories.append(tuple(jnp.stack(record) for record in records))

    trailing_shapes = (
        (state_size,),
        (state_size, state_size),
        (state_size,),
        (state_size, state_size),
        (state_size, state_size),
        (observation_size,),
        (observation_size, observation_size),
        (state_size, observation_size),
        (observation_size,),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
    )
    stacked = _stack_case_histories(case_histories, case_shape, trailing_shapes)
    solver_id, solver_method, transition_method = _solver_provenance(transition)
    nonlinear_transform_method = {
        "extended": "first-order-jvp-vjp",
        "cubature": "spherical-radial-cubature",
        "unscented": "scaled-unscented",
    }[method]
    if isinstance(transition, DifferentialTransitionKernel):
        transition_method = f"{transition_method}:{nonlinear_transform_method}"
    observation_transform_method = (
        "exact-affine"
        if isinstance(observation, LinearGaussianObservationModel)
        else nonlinear_transform_method
    )
    approximation_id = (
        f"continuous-discrete-gaussian:{method}:{transition.approximation_id}"
    )
    resolved_controller = (
        None
        if isinstance(transition, LinearGaussianTransitionKernel)
        else _resolved_moment_controller(transition)
    )
    controller_method = (
        "not-applicable"
        if isinstance(transition, LinearGaussianTransitionKernel)
        else (
            "PIDController"
            if resolved_controller is None
            else type(resolved_controller).__name__
        )
    )
    result = ContinuousDiscreteGaussianFilterResult(
        predicted_means=stacked[0].reshape(
            case_shape + (num_steps,) + problem.model.state_shape
        ),
        predicted_covariances=stacked[1],
        filtered_means=stacked[2].reshape(
            case_shape + (num_steps,) + problem.model.state_shape
        ),
        filtered_covariances=stacked[3],
        transition_cross_covariances=stacked[4],
        predicted_observation_means=stacked[5].reshape(
            case_shape + (num_steps,) + problem.model.observation_shape
        ),
        predicted_observation_covariances=stacked[6],
        state_observation_cross_covariances=stacked[7],
        innovations=stacked[8].reshape(
            case_shape + (num_steps,) + problem.model.observation_shape
        ),
        normalized_innovation_squared=stacked[9],
        incremental_log_likelihood=stacked[10],
        cumulative_log_likelihood=stacked[11],
        observed_counts=stacked[12],
        step_valid=sequence.step_valid,
        valid=stacked[13],
        status=stacked[14],
        solver_status=stacked[15],
        times=stacked[16],
        problem=problem,
        state_shape=problem.model.state_shape,
        observation_shape=problem.model.observation_shape,
        case_shape=case_shape,
        case_axes=sequence.case_axes,
        observation_axes=sequence.observation_axes,
        case_ids=sequence.case_ids,
        model_id=problem.model.model_id,
        problem_id=problem.problem_id,
        sequence_id=sequence.sequence_id,
        process_id=transition.process_id,
        observation_id=observation.observation_id,
        sensor_id=sequence.sensor_id,
        input_id=None if problem.input_signal is None else problem.input_signal.input_id,
        parameter_id=problem.model.parameter_id,
        basis_id=problem.model.basis_id,
        discretization_id=(
            problem.model.discretization_id
            if problem.model.discretization_id is not None
            else sequence.discretization_id
        ),
        method=method,
        method_id=f"continuous-discrete-gaussian-filter:{method}",
        transition_method=transition_method,
        observation_transform_method=observation_transform_method,
        solver_id=solver_id,
        solver_method=solver_method,
        backend_id=(
            "analytic-linear-gaussian"
            if isinstance(transition, LinearGaussianTransitionKernel)
            else "diffrax"
        ),
        adjoint_method=(
            "not-applicable"
            if isinstance(transition, LinearGaussianTransitionKernel)
            else (
                "DirectAdjoint"
                if transition.adjoint is None
                else type(transition.adjoint).__name__
            )
        ),
        stepsize_controller_method=controller_method,
        approximation_id=approximation_id,
        transition_approximation_id=transition.approximation_id,
        observation_approximation_id=sequence.approximation_id,
        covariance_regularization=regularization,
        rank_tolerance=tolerance,
        unscented_alpha=alpha,
        unscented_beta=beta,
        unscented_kappa=kappa,
    )
    if raise_on_failure and not bool(jnp.all(result.successful)):
        raise RuntimeError(
            "Continuous-discrete Gaussian filtering failed for at least one case."
        )
    return result


def continuous_discrete_gaussian_smoother(
    result: ContinuousDiscreteGaussianFilterResult,
    /,
    *,
    rank_tolerance: float | None = None,
    raise_on_failure: bool = False,
) -> ContinuousDiscreteGaussianSmootherResult:
    """Apply a fixed-interval Gaussian backward recursion to filter moments."""
    if not isinstance(result, ContinuousDiscreteGaussianFilterResult):
        raise TypeError("result must be a ContinuousDiscreteGaussianFilterResult.")
    tolerance = result.rank_tolerance if rank_tolerance is None else float(rank_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("rank_tolerance must be finite and nonnegative.")
    state_size = prod(result.state_shape) if result.state_shape else 1
    case_count = prod(result.case_shape) if result.case_shape else 1
    num_steps = result.times.shape[-1]
    filtered_means = result.filtered_means.reshape((case_count, num_steps, state_size))
    filtered_covariances = result.filtered_covariances.reshape(
        (case_count, num_steps, state_size, state_size)
    )
    predicted_means = result.predicted_means.reshape((case_count, num_steps, state_size))
    predicted_covariances = result.predicted_covariances.reshape(
        (case_count, num_steps, state_size, state_size)
    )
    transition_cross = result.transition_cross_covariances.reshape(
        (case_count, num_steps, state_size, state_size)
    )
    step_valid = result.step_valid.reshape((case_count, num_steps))
    filter_valid = result.valid.reshape((case_count, num_steps))
    smoothed_case_means = []
    smoothed_case_covariances = []
    case_gains = []
    smoothed_case_validity = []
    smoothed_case_statuses = []

    for case_index in range(case_count):
        means = [filtered_means[case_index, index] for index in range(num_steps)]
        covariances = [
            filtered_covariances[case_index, index] for index in range(num_steps)
        ]
        gains = [
            jnp.zeros((state_size, state_size), dtype=filtered_means.dtype)
            for _ in range(num_steps)
        ]
        validity = [filter_valid[case_index, index] for index in range(num_steps)]
        statuses = [
            result.status.reshape((case_count, num_steps))[case_index, index]
            for index in range(num_steps)
        ]
        for index in range(num_steps - 2, -1, -1):
            dependent = (
                step_valid[case_index, index]
                & step_valid[case_index, index + 1]
                & filter_valid[case_index, index]
            )
            next_valid = validity[index + 1]
            predicted_covariance = predicted_covariances[case_index, index + 1]
            predicted_factor = gaussian_factor_from_covariance(
                predicted_covariance,
                rank_tolerance=tolerance,
                factor_id="continuous-discrete-smoother-prediction",
            )
            full_rank = predicted_factor.numerical_rank == state_size
            operands_finite = (
                jnp.all(jnp.isfinite(predicted_covariance))
                & jnp.all(jnp.isfinite(transition_cross[case_index, index + 1]))
                & jnp.all(jnp.isfinite(filtered_means[case_index, index]))
                & jnp.all(jnp.isfinite(filtered_covariances[case_index, index]))
                & jnp.all(jnp.isfinite(means[index + 1]))
                & jnp.all(jnp.isfinite(covariances[index + 1]))
            )
            should_smooth = (
                dependent
                & next_valid
                & predicted_factor.valid
                & full_rank
                & operands_finite
            )

            def smooth(_):
                gain = jnp.conj(
                    jnp.linalg.solve(
                        predicted_covariance,
                        jnp.conj(transition_cross[case_index, index + 1].T),
                    ).T
                )
                smoothed_mean = filtered_means[case_index, index] + gain @ (
                    means[index + 1] - predicted_means[case_index, index + 1]
                )
                smoothed_covariance_raw = filtered_covariances[
                    case_index, index
                ] + gain @ (covariances[index + 1] - predicted_covariance) @ jnp.conj(
                    gain.T
                )
                smoothed_covariance = 0.5 * (
                    smoothed_covariance_raw + jnp.conj(smoothed_covariance_raw.T)
                )
                smoothed_factor = gaussian_factor_from_covariance(
                    smoothed_covariance,
                    rank_tolerance=tolerance,
                    factor_id="continuous-discrete-smoothed",
                )
                finite = (
                    operands_finite
                    & jnp.all(jnp.isfinite(gain))
                    & jnp.all(jnp.isfinite(smoothed_mean))
                    & jnp.all(jnp.isfinite(smoothed_covariance_raw))
                )
                return (
                    smoothed_mean,
                    smoothed_covariance,
                    gain,
                    smoothed_factor.valid,
                    finite,
                )

            def skip(_):
                return (
                    filtered_means[case_index, index],
                    filtered_covariances[case_index, index],
                    jnp.zeros(
                        (state_size, state_size),
                        dtype=filtered_means.dtype,
                    ),
                    jnp.asarray(False),
                    operands_finite,
                )

            (
                smoothed_mean,
                smoothed_covariance,
                gain,
                output_factor_valid,
                finite,
            ) = jax.lax.cond(should_smooth, smooth, skip, None)
            local_valid = (
                predicted_factor.valid & full_rank & output_factor_valid & finite
            )
            accept = should_smooth & local_valid
            means[index] = jnp.where(accept, smoothed_mean, means[index])
            covariances[index] = jnp.where(
                accept,
                smoothed_covariance,
                covariances[index],
            )
            gains[index] = jnp.where(accept, gain, gains[index])
            inherited_failure = dependent & ~next_valid
            local_failure = dependent & next_valid & ~local_valid
            validity[index] = jnp.where(
                dependent,
                next_valid & local_valid,
                validity[index],
            )
            local_status = jnp.where(
                ~finite,
                CONTINUOUS_DISCRETE_GAUSSIAN_NONFINITE,
                CONTINUOUS_DISCRETE_GAUSSIAN_TRANSFORM_FAILURE,
            )
            statuses[index] = jnp.where(
                inherited_failure,
                statuses[index + 1],
                jnp.where(
                    local_failure,
                    local_status,
                    statuses[index],
                ),
            ).astype(jnp.int32)
        smoothed_case_means.append(jnp.stack(means))
        smoothed_case_covariances.append(jnp.stack(covariances))
        case_gains.append(jnp.stack(gains))
        smoothed_case_validity.append(jnp.stack(validity))
        smoothed_case_statuses.append(jnp.stack(statuses))

    smoothed_means = jnp.stack(smoothed_case_means).reshape(
        result.case_shape + (num_steps,) + result.state_shape
    )
    smoothed_covariances = jnp.stack(smoothed_case_covariances).reshape(
        result.case_shape + (num_steps, state_size, state_size)
    )
    smoothing_gains = jnp.stack(case_gains).reshape(
        result.case_shape + (num_steps, state_size, state_size)
    )
    valid = jnp.stack(smoothed_case_validity).reshape(result.case_shape + (num_steps,))
    status = jnp.stack(smoothed_case_statuses).reshape(result.case_shape + (num_steps,))
    smoothed = ContinuousDiscreteGaussianSmootherResult(
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covariances,
        smoothing_gains=smoothing_gains,
        valid=valid,
        status=status,
        filter_result=result,
        method_id=f"continuous-discrete-gaussian-smoother:{result.method}",
        rank_tolerance=tolerance,
    )
    if raise_on_failure and not bool(jnp.all(smoothed.successful)):
        raise RuntimeError(
            "Continuous-discrete Gaussian smoothing failed for at least one case."
        )
    return smoothed


__all__ = [
    "CONTINUOUS_DISCRETE_GAUSSIAN_NONFINITE",
    "CONTINUOUS_DISCRETE_GAUSSIAN_SOLVER_FAILURE",
    "CONTINUOUS_DISCRETE_GAUSSIAN_SUCCESS",
    "CONTINUOUS_DISCRETE_GAUSSIAN_TRANSFORM_FAILURE",
    "CONTINUOUS_DISCRETE_MAX_DENSE_DIMENSION",
    "ContinuousDiscreteGaussianFilterResult",
    "ContinuousDiscreteGaussianMethod",
    "ContinuousDiscreteGaussianSmootherResult",
    "ContinuousDiscreteGaussianStatus",
    "continuous_discrete_gaussian_filter",
    "continuous_discrete_gaussian_smoother",
    "continuous_discrete_gaussian_status_name",
]
