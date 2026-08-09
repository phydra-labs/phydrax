#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import factorial, isfinite, prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule
from ..uq._gaussian_factor import gaussian_factor_from_covariance, GaussianFactor
from ._differential import DifferentialProblem
from ._save_schedule import validate_save_times


ProbabilisticODEUpdate: TypeAlias = Literal["ek0", "ek1"]
ProbabilisticODEFactorization: TypeAlias = Literal["dense", "block_diagonal"]
ProbabilisticODECovarianceOutput: TypeAlias = Literal["dense", "matrix_free"]
ProbabilisticODECalibration: TypeAlias = Literal["none", "quasi_mle"]
ProbabilisticODEStatus: TypeAlias = Literal[
    "success", "stiff", "nonfinite", "step_limit_reached"
]

PROBABILISTIC_ODE_SUCCESS = 0
PROBABILISTIC_ODE_STIFF = 1
PROBABILISTIC_ODE_NONFINITE = 2
PROBABILISTIC_ODE_STEP_LIMIT_REACHED = 3

_UNCERTAINTY_NAMES = (
    "numerical",
    "process",
    "observation",
    "initial_condition",
    "parameter",
)
_NUMERICAL = 0
_PROCESS = 1
_OBSERVATION = 2
_INITIAL_CONDITION = 3
_PARAMETER = 4


def probabilistic_ode_status_name(value: int, /) -> ProbabilisticODEStatus:
    """Return the stable public name corresponding to a solver status code."""
    code = int(value)
    if code == PROBABILISTIC_ODE_SUCCESS:
        return "success"
    if code == PROBABILISTIC_ODE_STIFF:
        return "stiff"
    if code == PROBABILISTIC_ODE_NONFINITE:
        return "nonfinite"
    if code == PROBABILISTIC_ODE_STEP_LIMIT_REACHED:
        return "step_limit_reached"
    raise ValueError(f"Unknown probabilistic ODE status code {code}.")


class ProbabilisticODEMethod(StrictModule):
    """Configuration of an integrated-Wiener probabilistic ODE method.

    ``order`` is the number of integrated derivatives in the Gauss--Markov
    prior. EK0 freezes the vector field in the residual linearization, whereas
    EK1 includes its state Jacobian. A fixed-capacity grid makes execution and
    checkpoint replay shape-stable under JAX transformations. When ``adaptive``
    is enabled, a uniform pilot pass redistributes this fixed work according to
    its dimensionless residuals.
    """

    order: int = eqx.field(static=True)
    update: ProbabilisticODEUpdate = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    adaptive: bool = eqx.field(static=True)
    smoothing: bool = eqx.field(static=True)
    factorization: ProbabilisticODEFactorization = eqx.field(static=True)
    covariance_output: ProbabilisticODECovarianceOutput = eqx.field(static=True)
    diffusion_calibration: ProbabilisticODECalibration = eqx.field(static=True)
    base_diffusion: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    covariance_regularization: float = eqx.field(static=True)
    stiffness_threshold: float = eqx.field(static=True)
    max_dense_dimension: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        order: int = 1,
        update: ProbabilisticODEUpdate = "ek1",
        num_steps: int = 32,
        adaptive: bool = False,
        smoothing: bool = True,
        factorization: ProbabilisticODEFactorization = "dense",
        covariance_output: ProbabilisticODECovarianceOutput = "matrix_free",
        diffusion_calibration: ProbabilisticODECalibration = "quasi_mle",
        base_diffusion: float = 1.0,
        relative_tolerance: float = 1e-3,
        absolute_tolerance: float = 1e-6,
        covariance_regularization: float = 0.0,
        stiffness_threshold: float = 50.0,
        max_dense_dimension: int = 512,
        method_id: str | None = None,
    ):
        if not isinstance(order, int) or isinstance(order, bool) or order < 1:
            raise ValueError("order must be a positive integer.")
        if order > 4:
            raise ValueError("Integrated-Wiener orders above four are unsupported.")
        if update not in ("ek0", "ek1"):
            raise ValueError("update must be 'ek0' or 'ek1'.")
        if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer.")
        if not isinstance(adaptive, bool) or not isinstance(smoothing, bool):
            raise TypeError("adaptive and smoothing must be bool values.")
        if factorization not in ("dense", "block_diagonal"):
            raise ValueError("factorization must be 'dense' or 'block_diagonal'.")
        if covariance_output not in ("dense", "matrix_free"):
            raise ValueError("covariance_output must be 'dense' or 'matrix_free'.")
        if diffusion_calibration not in ("none", "quasi_mle"):
            raise ValueError("diffusion_calibration must be 'none' or 'quasi_mle'.")
        scalar_values = {
            "base_diffusion": base_diffusion,
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
            "covariance_regularization": covariance_regularization,
            "stiffness_threshold": stiffness_threshold,
        }
        for name, value in scalar_values.items():
            if not isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if base_diffusion == 0.0 and covariance_regularization == 0.0:
            raise ValueError(
                "base_diffusion and covariance_regularization cannot both be zero."
            )
        if relative_tolerance == 0.0 and absolute_tolerance == 0.0:
            raise ValueError(
                "relative_tolerance and absolute_tolerance cannot both be zero."
            )
        if stiffness_threshold == 0.0:
            raise ValueError("stiffness_threshold must be positive.")
        if (
            not isinstance(max_dense_dimension, int)
            or isinstance(max_dense_dimension, bool)
            or max_dense_dimension < 1
        ):
            raise ValueError("max_dense_dimension must be a positive integer.")
        resolved_id = (
            "probabilistic-ode:"
            f"order={order}:update={update}:num_steps={num_steps}:"
            f"adaptive={adaptive}:smoothing={smoothing}:"
            f"factorization={factorization}:covariance_output={covariance_output}:"
            f"diffusion_calibration={diffusion_calibration}:"
            f"base_diffusion={float(base_diffusion).hex()}:"
            f"relative_tolerance={float(relative_tolerance).hex()}:"
            f"absolute_tolerance={float(absolute_tolerance).hex()}:"
            f"covariance_regularization={float(covariance_regularization).hex()}:"
            f"stiffness_threshold={float(stiffness_threshold).hex()}:"
            f"max_dense_dimension={max_dense_dimension}"
            if method_id is None
            else method_id
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("method_id must be a non-empty string or None.")
        self.order = order
        self.update = update
        self.num_steps = num_steps
        self.adaptive = adaptive
        self.smoothing = smoothing
        self.factorization = factorization
        self.covariance_output = covariance_output
        self.diffusion_calibration = diffusion_calibration
        self.base_diffusion = float(base_diffusion)
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self.covariance_regularization = float(covariance_regularization)
        self.stiffness_threshold = float(stiffness_threshold)
        self.max_dense_dimension = max_dense_dimension
        self.method_id = resolved_id


class _ProbabilisticODECheckpoint(StrictModule):
    time: Array
    integrated_mean: Array
    source_covariances: Array
    parameter_sensitivity: Array
    quasi_mle_sum: Array
    quasi_mle_count: Array
    quasi_log_likelihood_sum: Array
    grid_origin: Array
    step_index: Array
    nominal_step_size: Array
    method_id: str = eqx.field(static=True)
    factorization: ProbabilisticODEFactorization = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)


class ProbabilisticODESolution(StrictModule):
    """Smoothed posterior trajectory and decomposed uncertainty provenance."""

    times: Array
    means: Array
    standard_deviations: Array
    covariances: Array | None
    covariance_factor: GaussianFactor
    source_covariances: frozendict[str, Array]
    valid: Array
    status: Array
    residuals: Array
    normalized_residuals: Array
    step_sizes: Array
    diffusion_scale: Array
    log_quasi_likelihood: Array
    stats: frozendict[str, Any]
    checkpoint: _ProbabilisticODECheckpoint
    method: ProbabilisticODEMethod
    state_shape: tuple[int, ...] = eqx.field(static=True)
    uncertainty_sources: tuple[str, ...] = eqx.field(static=True)
    covariance_representation: ProbabilisticODECovarianceOutput = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Whether integration completed without stiffness or numerical failure."""
        return (self.status == PROBABILISTIC_ODE_SUCCESS) & jnp.all(self.valid)

    def covariance_matvec(
        self,
        vector: ArrayLike,
        /,
        *,
        source: str | None = None,
    ) -> Array:
        """Apply saved marginal covariances without materializing dense block output."""
        value = jnp.asarray(vector)
        expected = self.times.shape + self.state_shape
        if value.shape != expected:
            raise ValueError(
                f"vector must have saved trajectory shape {expected}; got {value.shape}."
            )
        flat = value.reshape((self.times.shape[0], -1))
        if source is None:
            factor = self.covariance_factor.factor
            if self.method.factorization == "block_diagonal":
                result = factor[..., 0, 0] ** 2 * flat
            else:
                coefficients = jnp.einsum("tir,ti->tr", factor, flat)
                result = jnp.einsum("tir,tr->ti", factor, coefficients)
        else:
            if source not in _UNCERTAINTY_NAMES:
                raise ValueError(
                    f"source must be one of {_UNCERTAINTY_NAMES}; got {source!r}."
                )
            component = self.source_covariances[source]
            if self.method.factorization == "block_diagonal":
                result = component.reshape(flat.shape) * flat
            else:
                result = jnp.einsum("tij,tj->ti", component, flat)
        return result.reshape(expected)

    def dense_covariance(self, /, *, source: str | None = None) -> Array:
        """Materialize saved physical-state covariance after an explicit size check."""
        dimension = prod(self.state_shape) if self.state_shape else 1
        if dimension > self.method.max_dense_dimension:
            raise ValueError(
                "Dense covariance materialization exceeds max_dense_dimension; "
                "use covariance_matvec instead."
            )
        if source is not None and source not in _UNCERTAINTY_NAMES:
            raise ValueError(
                f"source must be one of {_UNCERTAINTY_NAMES}; got {source!r}."
            )
        names = _UNCERTAINTY_NAMES if source is None else (source,)
        if self.method.factorization == "dense":
            return sum(
                (self.source_covariances[name] for name in names),
                jnp.zeros(
                    (self.times.shape[0], dimension, dimension),
                    dtype=self.means.dtype,
                ),
            )
        diagonal = sum(
            (
                self.source_covariances[name].reshape((self.times.shape[0], dimension))
                for name in names
            ),
            jnp.zeros((self.times.shape[0], dimension), dtype=self.means.dtype),
        )
        return jax.vmap(jnp.diag)(diagonal)


def _covariance_matrix(
    value: ArrayLike | None,
    size: int,
    dtype: jnp.dtype,
    /,
    *,
    name: str,
) -> Array:
    if value is None:
        return jnp.zeros((size, size), dtype=dtype)
    array = jnp.asarray(value, dtype=dtype)
    if array.shape == ():
        matrix = array * jnp.eye(size, dtype=dtype)
    elif array.shape == (size,):
        matrix = jnp.diag(array)
    elif array.shape == (size, size):
        matrix = array
    else:
        raise ValueError(
            f"{name} must be scalar, length {size}, or shape {(size, size)}; "
            f"got {array.shape}."
        )
    matrix = eqx.error_if(
        matrix,
        ~jnp.all(jnp.isfinite(matrix)),
        f"{name} must be finite.",
    )
    matrix = eqx.error_if(
        matrix,
        ~jnp.allclose(matrix, matrix.T),
        f"{name} must be symmetric.",
    )
    matrix = eqx.error_if(
        matrix,
        jnp.min(jnp.linalg.eigvalsh(matrix)) < 0.0,
        f"{name} must be positive semidefinite.",
    )
    return matrix


def _diagonal_covariance(matrix: Array, /, *, name: str) -> Array:
    diagonal = jnp.diag(jnp.diag(matrix))
    return eqx.error_if(
        matrix,
        jnp.any(matrix != diagonal),
        f"{name} must be diagonal for block_diagonal factorization.",
    )


def _transition(order: int, step: Array, dtype: jnp.dtype, /) -> Array:
    result = jnp.zeros((order + 1, order + 1), dtype=dtype)
    for row in range(order + 1):
        for column in range(row, order + 1):
            result = result.at[row, column].set(
                step ** (column - row) / factorial(column - row)
            )
    return result


def _iwp_covariance(order: int, step: Array, dtype: jnp.dtype, /) -> Array:
    result = jnp.zeros((order + 1, order + 1), dtype=dtype)
    for row in range(order + 1):
        for column in range(order + 1):
            exponent = 2 * order + 1 - row - column
            denominator = exponent * factorial(order - row) * factorial(order - column)
            result = result.at[row, column].set(step**exponent / denominator)
    return result


def _startup_mean(
    problem: DifferentialProblem,
    order: int,
    args: Any,
    /,
) -> Array:
    time = problem.t0
    state = problem.initial_state
    derivatives = [state]

    def first_derivative(t, y, parameters):
        return jnp.asarray(problem.drift(t, y, parameters))

    derivative_function = first_derivative
    for _ in range(order):
        value = derivative_function(time, state, args)
        derivatives.append(value)
        previous = derivative_function

        def total_derivative(t, y, parameters, previous=previous):
            tangent = jnp.asarray(problem.drift(t, y, parameters))
            return jax.jvp(
                lambda query_time, query_state: previous(
                    query_time, query_state, parameters
                ),
                (t, y),
                (jnp.ones_like(t), tangent),
            )[1]

        derivative_function = total_derivative
    return jnp.stack(tuple(derivatives), axis=0)


def _parameter_jacobian(
    problem: DifferentialProblem,
    flat_args: Array,
    unravel_args: Any,
    time: Array,
    state: Array,
    /,
) -> Array:
    if flat_args.shape[0] == 0:
        return jnp.zeros((state.size, 0), dtype=state.dtype)
    return jax.jacfwd(
        lambda values: jnp.asarray(
            problem.drift(time, state, unravel_args(values))
        ).reshape(-1)
    )(flat_args)


def _dense_filter(
    problem: DifferentialProblem,
    method: ProbabilisticODEMethod,
    initial_mean: Array,
    initial_sources: Array,
    initial_sensitivity: Array,
    steps: Array,
    initial_time: Array,
    observation_covariance: Array,
    process_covariance: Array,
    flat_args: Array,
    unravel_args: Any,
    initial_quasi_sum: Array,
    initial_quasi_count: Array,
    initial_quasi_log_sum: Array,
    /,
):
    order = method.order
    derivative_count, state_size = initial_mean.shape
    augmented_size = derivative_count * state_size
    parameter_size = flat_args.shape[0]
    eye_augmented = jnp.eye(augmented_size, dtype=initial_mean.dtype)
    eye_state = jnp.eye(state_size, dtype=initial_mean.dtype)

    def one_step(carry, step):
        mean, sources, sensitivity, time, quasi_sum, quasi_count, quasi_log_sum = carry
        active = step > 0.0

        def advance(values):
            (
                mean,
                sources,
                sensitivity,
                time,
                quasi_sum,
                quasi_count,
                quasi_log_sum,
            ) = values
            next_time = time + step
            transition_small = _transition(order, step, mean.dtype)
            transition = jnp.kron(transition_small, eye_state)
            iwp = _iwp_covariance(order, step, mean.dtype)
            predicted_mean = (transition @ mean.reshape(-1)).reshape(mean.shape)
            predicted_sensitivity = (
                transition @ sensitivity.reshape((augmented_size, parameter_size))
            ).reshape(sensitivity.shape)
            predicted_sources = jnp.einsum(
                "ij,sjk,lk->sil", transition, sources, transition
            )
            predicted_sources = predicted_sources.at[_NUMERICAL].add(
                method.base_diffusion * jnp.kron(iwp, eye_state)
            )
            predicted_sources = predicted_sources.at[_PROCESS].add(
                jnp.kron(iwp, process_covariance)
            )
            predicted_state = predicted_mean[0].reshape(problem.initial_state.shape)
            drift = jnp.asarray(
                problem.drift(next_time, predicted_state, problem.args)
            ).reshape(-1)
            residual = predicted_mean[1] - drift
            observation_matrix = jnp.zeros((state_size, augmented_size), dtype=mean.dtype)
            observation_matrix = observation_matrix.at[
                :, state_size : 2 * state_size
            ].set(eye_state)
            drift_jacobian = jax.jacfwd(
                lambda state: jnp.asarray(
                    problem.drift(
                        next_time,
                        state.reshape(problem.initial_state.shape),
                        problem.args,
                    )
                ).reshape(-1)
            )(predicted_mean[0])
            if method.update == "ek1":
                observation_matrix = observation_matrix.at[:, :state_size].set(
                    -drift_jacobian
                )
            parameter_jacobian = _parameter_jacobian(
                problem,
                flat_args,
                unravel_args,
                next_time,
                predicted_state,
            )
            residual_sensitivity = (
                predicted_sensitivity[1]
                - drift_jacobian @ predicted_sensitivity[0]
                - parameter_jacobian
            )
            predicted_covariance = jnp.sum(predicted_sources, axis=0)
            innovation_covariance = (
                observation_matrix @ predicted_covariance @ observation_matrix.T
                + observation_covariance
                + method.covariance_regularization * eye_state
            )
            solved_residual = jnp.linalg.solve(innovation_covariance, residual)
            cross_covariance = predicted_covariance @ observation_matrix.T
            gain = jnp.linalg.solve(innovation_covariance, cross_covariance.T).T
            updated_mean = (predicted_mean.reshape(-1) - gain @ residual).reshape(
                mean.shape
            )
            updated_sensitivity = (
                predicted_sensitivity.reshape((augmented_size, parameter_size))
                - gain @ residual_sensitivity
            ).reshape(sensitivity.shape)
            joseph = eye_augmented - gain @ observation_matrix
            updated_sources = jnp.einsum(
                "ij,sjk,lk->sil", joseph, predicted_sources, joseph
            )
            updated_sources = updated_sources.at[_OBSERVATION].add(
                gain @ observation_covariance @ gain.T
            )
            updated_sources = updated_sources.at[_NUMERICAL].add(
                method.covariance_regularization * gain @ gain.T
            )
            updated_sources = 0.5 * (
                updated_sources + jnp.swapaxes(updated_sources, -1, -2)
            )
            mahalanobis = residual @ solved_residual
            normalized = mahalanobis / state_size
            log_determinant = jnp.linalg.slogdet(innovation_covariance)[1]
            quasi_term = (
                log_determinant
                + mahalanobis
                + state_size * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=mean.dtype))
            )
            tolerance = (
                method.absolute_tolerance
                + method.relative_tolerance
                * jnp.maximum(
                    jnp.linalg.norm(predicted_mean[0]),
                    jnp.linalg.norm(updated_mean[0]),
                )
            )
            residual_norm = jnp.linalg.norm(gain[:state_size] @ residual) / tolerance
            stiffness = step * jnp.linalg.norm(drift_jacobian, ord=jnp.inf)
            next_carry = (
                updated_mean,
                updated_sources,
                updated_sensitivity,
                next_time,
                quasi_sum + normalized,
                quasi_count + jnp.asarray(1, dtype=jnp.int32),
                quasi_log_sum + quasi_term,
            )
            record = (
                predicted_mean,
                predicted_sources,
                updated_mean,
                updated_sources,
                transition,
                residual,
                normalized,
                residual_norm,
                stiffness,
                jnp.asarray(True),
                next_time,
                predicted_sensitivity,
                updated_sensitivity,
            )
            return next_carry, record

        def inactive(values):
            mean, sources, sensitivity, time, _, _, _ = values
            transition = jnp.eye(augmented_size, dtype=mean.dtype)
            zero_state = jnp.zeros((state_size,), dtype=mean.dtype)
            record = (
                mean,
                sources,
                mean,
                sources,
                transition,
                zero_state,
                jnp.zeros((), dtype=mean.dtype),
                jnp.zeros((), dtype=mean.dtype),
                jnp.zeros((), dtype=mean.dtype),
                jnp.asarray(False),
                time,
                sensitivity,
                sensitivity,
            )
            return values, record

        return jax.lax.cond(active, advance, inactive, carry)

    initial_carry = (
        initial_mean,
        initial_sources,
        initial_sensitivity,
        initial_time,
        initial_quasi_sum,
        initial_quasi_count,
        initial_quasi_log_sum,
    )
    return jax.lax.scan(one_step, initial_carry, steps)


def _block_filter(
    problem: DifferentialProblem,
    method: ProbabilisticODEMethod,
    initial_mean: Array,
    initial_sources: Array,
    initial_sensitivity: Array,
    steps: Array,
    initial_time: Array,
    observation_covariance: Array,
    process_covariance: Array,
    flat_args: Array,
    unravel_args: Any,
    initial_quasi_sum: Array,
    initial_quasi_count: Array,
    initial_quasi_log_sum: Array,
    /,
):
    order = method.order
    derivative_count, state_size = initial_mean.shape
    eye_derivative = jnp.eye(derivative_count, dtype=initial_mean.dtype)
    observation_diagonal = jnp.diag(observation_covariance)
    process_diagonal = jnp.diag(process_covariance)

    def one_step(carry, step):
        mean, sources, sensitivity, time, quasi_sum, quasi_count, quasi_log_sum = carry
        active = step > 0.0

        def advance(values):
            (
                mean,
                sources,
                sensitivity,
                time,
                quasi_sum,
                quasi_count,
                quasi_log_sum,
            ) = values
            next_time = time + step
            transition = _transition(order, step, mean.dtype)
            iwp = _iwp_covariance(order, step, mean.dtype)
            predicted_mean = transition @ mean
            predicted_sensitivity = jnp.einsum("ij,jdp->idp", transition, sensitivity)
            predicted_sources = jnp.einsum(
                "ij,sdjk,lk->sdil", transition, sources, transition
            )
            predicted_sources = predicted_sources.at[_NUMERICAL].add(
                method.base_diffusion
                * jnp.broadcast_to(iwp, (state_size, derivative_count, derivative_count))
            )
            predicted_sources = predicted_sources.at[_PROCESS].add(
                process_diagonal[:, None, None] * iwp[None, :, :]
            )
            predicted_state = predicted_mean[0].reshape(problem.initial_state.shape)
            drift = jnp.asarray(
                problem.drift(next_time, predicted_state, problem.args)
            ).reshape(-1)
            residual = predicted_mean[1] - drift

            def flattened_drift(flat_state):
                return jnp.asarray(
                    problem.drift(
                        next_time,
                        flat_state.reshape(problem.initial_state.shape),
                        problem.args,
                    )
                ).reshape(-1)

            drift_jacobian = jax.jacfwd(flattened_drift)(predicted_mean[0])
            drift_jacobian_diagonal = jnp.diag(drift_jacobian)
            observation_rows = (
                jnp.zeros((state_size, derivative_count), dtype=mean.dtype)
                .at[:, 1]
                .set(1.0)
            )
            if method.update == "ek1":
                observation_rows = observation_rows.at[:, 0].set(-drift_jacobian_diagonal)
            parameter_jacobian = _parameter_jacobian(
                problem,
                flat_args,
                unravel_args,
                next_time,
                predicted_state,
            )
            residual_sensitivity = (
                predicted_sensitivity[1]
                - drift_jacobian @ predicted_sensitivity[0]
                - parameter_jacobian
            )
            predicted_covariance = jnp.sum(predicted_sources, axis=0)
            innovation_variance = (
                jnp.einsum(
                    "di,dij,dj->d",
                    observation_rows,
                    predicted_covariance,
                    observation_rows,
                )
                + observation_diagonal
                + method.covariance_regularization
            )
            cross_covariance = jnp.einsum(
                "dij,dj->di", predicted_covariance, observation_rows
            )
            gain = cross_covariance / innovation_variance[:, None]
            updated_mean = predicted_mean - (gain * residual[:, None]).T
            updated_sensitivity = predicted_sensitivity - jnp.einsum(
                "dq,dp->qdp", gain, residual_sensitivity
            )
            joseph = eye_derivative[None, :, :] - jnp.einsum(
                "di,dj->dij", gain, observation_rows
            )
            updated_sources = jnp.einsum(
                "dij,sdjk,dlk->sdil", joseph, predicted_sources, joseph
            )
            outer_gain = jnp.einsum("di,dj->dij", gain, gain)
            updated_sources = updated_sources.at[_OBSERVATION].add(
                observation_diagonal[:, None, None] * outer_gain
            )
            updated_sources = updated_sources.at[_NUMERICAL].add(
                method.covariance_regularization * outer_gain
            )
            updated_sources = 0.5 * (
                updated_sources + jnp.swapaxes(updated_sources, -1, -2)
            )
            mahalanobis = jnp.sum(residual**2 / innovation_variance)
            normalized = mahalanobis / state_size
            quasi_term = (
                jnp.sum(jnp.log(innovation_variance))
                + mahalanobis
                + state_size * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=mean.dtype))
            )
            tolerance = (
                method.absolute_tolerance
                + method.relative_tolerance
                * jnp.maximum(
                    jnp.linalg.norm(predicted_mean[0]),
                    jnp.linalg.norm(updated_mean[0]),
                )
            )
            residual_norm = jnp.linalg.norm(gain[:, 0] * residual) / tolerance
            stiffness = step * jnp.max(jnp.abs(drift_jacobian_diagonal))
            next_carry = (
                updated_mean,
                updated_sources,
                updated_sensitivity,
                next_time,
                quasi_sum + normalized,
                quasi_count + jnp.asarray(1, dtype=jnp.int32),
                quasi_log_sum + quasi_term,
            )
            record = (
                predicted_mean,
                predicted_sources,
                updated_mean,
                updated_sources,
                transition,
                residual,
                normalized,
                residual_norm,
                stiffness,
                jnp.asarray(True),
                next_time,
                predicted_sensitivity,
                updated_sensitivity,
            )
            return next_carry, record

        def inactive(values):
            mean, sources, sensitivity, time, _, _, _ = values
            zero_state = jnp.zeros((state_size,), dtype=mean.dtype)
            record = (
                mean,
                sources,
                mean,
                sources,
                eye_derivative,
                zero_state,
                jnp.zeros((), dtype=mean.dtype),
                jnp.zeros((), dtype=mean.dtype),
                jnp.zeros((), dtype=mean.dtype),
                jnp.asarray(False),
                time,
                sensitivity,
                sensitivity,
            )
            return values, record

        return jax.lax.cond(active, advance, inactive, carry)

    initial_carry = (
        initial_mean,
        initial_sources,
        initial_sensitivity,
        initial_time,
        initial_quasi_sum,
        initial_quasi_count,
        initial_quasi_log_sum,
    )
    return jax.lax.scan(one_step, initial_carry, steps)


def _dense_smooth(
    initial_mean,
    initial_sources,
    initial_sensitivity,
    records,
    /,
):
    (
        predicted_means,
        predicted_sources,
        filtered_means,
        filtered_sources,
        transitions,
        _,
        _,
        _,
        _,
        active,
        _,
        predicted_sensitivities,
        filtered_sensitivities,
    ) = records
    filtered_means_all = jnp.concatenate((initial_mean[None], filtered_means), axis=0)
    filtered_sources_all = jnp.concatenate(
        (initial_sources[None], filtered_sources), axis=0
    )
    filtered_sensitivities_all = jnp.concatenate(
        (initial_sensitivity[None], filtered_sensitivities), axis=0
    )

    def step(carry, values):
        next_mean, next_sources, next_sensitivity = carry
        (
            filtered_mean,
            filtered_source,
            filtered_sensitivity,
            predicted_mean,
            predicted_source,
            predicted_sensitivity,
            transition,
            enabled,
        ) = values

        def smooth(_):
            filtered_covariance = jnp.sum(filtered_source, axis=0)
            predicted_covariance = jnp.sum(predicted_source, axis=0)
            cross = filtered_covariance @ transition.T
            gain = jnp.linalg.solve(predicted_covariance, cross.T).T
            mean = (
                filtered_mean.reshape(-1)
                + gain @ (next_mean - predicted_mean).reshape(-1)
            ).reshape(filtered_mean.shape)
            sources = filtered_source + jnp.einsum(
                "ij,sjk,lk->sil",
                gain,
                next_sources - predicted_source,
                gain,
            )
            sources = 0.5 * (sources + jnp.swapaxes(sources, -1, -2))
            sensitivity = (
                filtered_sensitivity.reshape(
                    (gain.shape[0], filtered_sensitivity.shape[-1])
                )
                + gain
                @ (next_sensitivity - predicted_sensitivity).reshape(
                    (gain.shape[0], filtered_sensitivity.shape[-1])
                )
            ).reshape(filtered_sensitivity.shape)
            return mean, sources, sensitivity

        result = jax.lax.cond(
            enabled,
            smooth,
            lambda _: (filtered_mean, filtered_source, filtered_sensitivity),
            operand=None,
        )
        return result, result

    inputs = (
        filtered_means_all[:-1],
        filtered_sources_all[:-1],
        filtered_sensitivities_all[:-1],
        predicted_means,
        predicted_sources,
        predicted_sensitivities,
        transitions,
        active,
    )
    (
        (_, _, _),
        (
            reverse_means,
            reverse_sources,
            reverse_sensitivities,
        ),
    ) = jax.lax.scan(
        step,
        (
            filtered_means_all[-1],
            filtered_sources_all[-1],
            filtered_sensitivities_all[-1],
        ),
        inputs,
        reverse=True,
    )
    return (
        jnp.concatenate((reverse_means, filtered_means_all[-1:]), axis=0),
        jnp.concatenate((reverse_sources, filtered_sources_all[-1:]), axis=0),
        jnp.concatenate((reverse_sensitivities, filtered_sensitivities_all[-1:]), axis=0),
    )


def _block_smooth(
    initial_mean,
    initial_sources,
    initial_sensitivity,
    records,
    /,
):
    (
        predicted_means,
        predicted_sources,
        filtered_means,
        filtered_sources,
        transitions,
        _,
        _,
        _,
        _,
        active,
        _,
        predicted_sensitivities,
        filtered_sensitivities,
    ) = records
    filtered_means_all = jnp.concatenate((initial_mean[None], filtered_means), axis=0)
    filtered_sources_all = jnp.concatenate(
        (initial_sources[None], filtered_sources), axis=0
    )
    filtered_sensitivities_all = jnp.concatenate(
        (initial_sensitivity[None], filtered_sensitivities), axis=0
    )

    def step(carry, values):
        next_mean, next_sources, next_sensitivity = carry
        (
            filtered_mean,
            filtered_source,
            filtered_sensitivity,
            predicted_mean,
            predicted_source,
            predicted_sensitivity,
            transition,
            enabled,
        ) = values

        def smooth(_):
            filtered_covariance = jnp.sum(filtered_source, axis=0)
            predicted_covariance = jnp.sum(predicted_source, axis=0)
            cross = jnp.einsum("dij,kj->dik", filtered_covariance, transition)
            gain = jax.vmap(lambda p, c: jnp.linalg.solve(p, c.T).T)(
                predicted_covariance, cross
            )
            delta = (next_mean - predicted_mean).T
            mean = filtered_mean + jnp.einsum("dij,dj->di", gain, delta).T
            sources = filtered_source + jnp.einsum(
                "dij,sdjk,dlk->sdil",
                gain,
                next_sources - predicted_source,
                gain,
            )
            sources = 0.5 * (sources + jnp.swapaxes(sources, -1, -2))
            sensitivity = filtered_sensitivity + jnp.einsum(
                "dij,jdp->idp",
                gain,
                next_sensitivity - predicted_sensitivity,
            )
            return mean, sources, sensitivity

        result = jax.lax.cond(
            enabled,
            smooth,
            lambda _: (filtered_mean, filtered_source, filtered_sensitivity),
            operand=None,
        )
        return result, result

    inputs = (
        filtered_means_all[:-1],
        filtered_sources_all[:-1],
        filtered_sensitivities_all[:-1],
        predicted_means,
        predicted_sources,
        predicted_sensitivities,
        transitions,
        active,
    )
    (
        (_, _, _),
        (
            reverse_means,
            reverse_sources,
            reverse_sensitivities,
        ),
    ) = jax.lax.scan(
        step,
        (
            filtered_means_all[-1],
            filtered_sources_all[-1],
            filtered_sensitivities_all[-1],
        ),
        inputs,
        reverse=True,
    )
    return (
        jnp.concatenate((reverse_means, filtered_means_all[-1:]), axis=0),
        jnp.concatenate((reverse_sources, filtered_sources_all[-1:]), axis=0),
        jnp.concatenate((reverse_sensitivities, filtered_sensitivities_all[-1:]), axis=0),
    )


def _fixed_steps(
    start: Array,
    end: Array,
    count: int,
    step_size: ArrayLike | None,
    grid_origin: Array,
    step_index: Array,
    /,
) -> Array:
    if step_size is None:
        uniform = jnp.full((count,), (end - start) / count, dtype=start.dtype)
        return jnp.concatenate(
            (
                uniform[:-1],
                (end - start - jnp.sum(uniform[:-1]))[None],
            )
        )
    nominal = jnp.asarray(step_size, dtype=start.dtype)
    if nominal.shape != ():
        raise ValueError("step_size must be scalar or None.")
    nominal = eqx.error_if(
        nominal,
        ~(jnp.isfinite(nominal) & (nominal > 0.0)),
        "step_size must be finite and positive.",
    )
    indices = step_index + jnp.arange(1, count + 1, dtype=step_index.dtype)
    targets = grid_origin + nominal * indices.astype(start.dtype)
    ends = jnp.minimum(targets, end)
    starts = jnp.concatenate((start[None], ends[:-1]))
    return jnp.where(starts < end, ends - starts, 0.0)


def _saved_dense_marginals(
    times,
    grid,
    smoothed_means,
    smoothed_sources,
    smoothed_sensitivities,
    filtered_means,
    filtered_sources,
    filtered_sensitivities,
    predicted_means,
    predicted_sources,
    predicted_sensitivities,
    order,
    smoothing,
    base_diffusion,
    process_covariance,
    /,
):
    indices = jnp.searchsorted(grid, times, side="right") - 1
    indices = jnp.where(times == grid[-1], grid.shape[0] - 1, indices)
    record_indices = jnp.minimum(indices, predicted_means.shape[0] - 1)
    right_indices = jnp.minimum(indices + 1, grid.shape[0] - 1)
    state_size = filtered_means.shape[-1]
    eye_state = jnp.eye(state_size, dtype=filtered_means.dtype)

    def evaluate(time, index, record_index, right_index):
        exact = time == grid[index]

        def exact_knot(_):
            return (
                smoothed_means[index, 0],
                smoothed_sources[index, :, :state_size, :state_size],
                smoothed_sensitivities[index, 0],
            )

        def off_grid(_):
            delta = time - grid[index]
            transition_small = _transition(order, delta, filtered_means.dtype)
            transition = jnp.kron(transition_small, eye_state)
            mean = (transition @ filtered_means[index].reshape(-1)).reshape(
                filtered_means[index].shape
            )
            sensitivity = (
                transition
                @ filtered_sensitivities[index].reshape(
                    (transition.shape[0], filtered_sensitivities.shape[-1])
                )
            ).reshape(filtered_sensitivities[index].shape)
            sources = jnp.einsum(
                "ij,sjk,lk->sil",
                transition,
                filtered_sources[index],
                transition,
            )
            iwp = _iwp_covariance(order, delta, filtered_means.dtype)
            sources = sources.at[_NUMERICAL].add(
                base_diffusion * jnp.kron(iwp, eye_state)
            )
            sources = sources.at[_PROCESS].add(jnp.kron(iwp, process_covariance))
            if smoothing:

                def bridge(values):
                    mean, sources, sensitivity = values
                    remaining = grid[right_index] - time
                    remaining_transition = jnp.kron(
                        _transition(order, remaining, filtered_means.dtype),
                        eye_state,
                    )
                    covariance = jnp.sum(sources, axis=0)
                    cross = covariance @ remaining_transition.T
                    predicted_right_covariance = jnp.sum(
                        predicted_sources[record_index], axis=0
                    )
                    gain = jnp.linalg.solve(predicted_right_covariance, cross.T).T
                    mean = (
                        mean.reshape(-1)
                        + gain
                        @ (
                            smoothed_means[right_index] - predicted_means[record_index]
                        ).reshape(-1)
                    ).reshape(mean.shape)
                    sources = sources + jnp.einsum(
                        "ij,sjk,lk->sil",
                        gain,
                        smoothed_sources[right_index] - predicted_sources[record_index],
                        gain,
                    )
                    sources = 0.5 * (sources + jnp.swapaxes(sources, -1, -2))
                    sensitivity = (
                        sensitivity.reshape((gain.shape[0], sensitivity.shape[-1]))
                        + gain
                        @ (
                            smoothed_sensitivities[right_index]
                            - predicted_sensitivities[record_index]
                        ).reshape((gain.shape[0], sensitivity.shape[-1]))
                    ).reshape(sensitivity.shape)
                    return mean, sources, sensitivity

                mean, sources, sensitivity = jax.lax.cond(
                    (right_index > index) & (time < grid[right_index]),
                    bridge,
                    lambda values: values,
                    (mean, sources, sensitivity),
                )
            return (
                mean[0],
                sources[:, :state_size, :state_size],
                sensitivity[0],
            )

        return jax.lax.cond(exact, exact_knot, off_grid, operand=None)

    saved_means, saved_sources, saved_sensitivities = jax.vmap(evaluate)(
        times,
        indices,
        record_indices,
        right_indices,
    )
    return (
        saved_means,
        jnp.swapaxes(saved_sources, 0, 1),
        saved_sensitivities,
    )


def _saved_block_marginals(
    times,
    grid,
    smoothed_means,
    smoothed_sources,
    smoothed_sensitivities,
    filtered_means,
    filtered_sources,
    filtered_sensitivities,
    predicted_means,
    predicted_sources,
    predicted_sensitivities,
    order,
    smoothing,
    base_diffusion,
    process_covariance,
    /,
):
    indices = jnp.searchsorted(grid, times, side="right") - 1
    indices = jnp.where(times == grid[-1], grid.shape[0] - 1, indices)
    record_indices = jnp.minimum(indices, predicted_means.shape[0] - 1)
    right_indices = jnp.minimum(indices + 1, grid.shape[0] - 1)
    process_diagonal = jnp.diag(process_covariance)

    def evaluate(time, index, record_index, right_index):
        exact = time == grid[index]

        def exact_knot(_):
            return (
                smoothed_means[index, 0],
                smoothed_sources[index, :, :, 0, 0],
                smoothed_sensitivities[index, 0],
            )

        def off_grid(_):
            delta = time - grid[index]
            transition = _transition(order, delta, filtered_means.dtype)
            mean = transition @ filtered_means[index]
            sensitivity = jnp.einsum(
                "ij,jdp->idp", transition, filtered_sensitivities[index]
            )
            sources = jnp.einsum(
                "ij,sdjk,lk->sdil",
                transition,
                filtered_sources[index],
                transition,
            )
            iwp = _iwp_covariance(order, delta, filtered_means.dtype)
            sources = sources.at[_NUMERICAL].add(
                base_diffusion * jnp.broadcast_to(iwp, sources[_NUMERICAL].shape)
            )
            sources = sources.at[_PROCESS].add(
                process_diagonal[:, None, None] * iwp[None, :, :]
            )
            if smoothing:

                def bridge(values):
                    mean, sources, sensitivity = values
                    remaining_transition = _transition(
                        order,
                        grid[right_index] - time,
                        filtered_means.dtype,
                    )
                    covariance = jnp.sum(sources, axis=0)
                    cross = jnp.einsum("dij,kj->dik", covariance, remaining_transition)
                    predicted_right_covariance = jnp.sum(
                        predicted_sources[record_index], axis=0
                    )
                    gain = jax.vmap(
                        lambda predicted, component: (
                            jnp.linalg.solve(predicted, component.T).T
                        )
                    )(predicted_right_covariance, cross)
                    mean = (
                        mean
                        + jnp.einsum(
                            "dij,dj->di",
                            gain,
                            (
                                smoothed_means[right_index]
                                - predicted_means[record_index]
                            ).T,
                        ).T
                    )
                    sources = sources + jnp.einsum(
                        "dij,sdjk,dlk->sdil",
                        gain,
                        smoothed_sources[right_index] - predicted_sources[record_index],
                        gain,
                    )
                    sources = 0.5 * (sources + jnp.swapaxes(sources, -1, -2))
                    sensitivity = sensitivity + jnp.einsum(
                        "dij,jdp->idp",
                        gain,
                        smoothed_sensitivities[right_index]
                        - predicted_sensitivities[record_index],
                    )
                    return mean, sources, sensitivity

                mean, sources, sensitivity = jax.lax.cond(
                    (right_index > index) & (time < grid[right_index]),
                    bridge,
                    lambda values: values,
                    (mean, sources, sensitivity),
                )
            return mean[0], sources[:, :, 0, 0], sensitivity[0]

        return jax.lax.cond(exact, exact_knot, off_grid, operand=None)

    saved_means, saved_sources, saved_sensitivities = jax.vmap(evaluate)(
        times,
        indices,
        record_indices,
        right_indices,
    )
    return (
        saved_means,
        jnp.transpose(saved_sources, (1, 0, 2)),
        saved_sensitivities,
    )


def solve_probabilistic_ode(
    problem: DifferentialProblem,
    /,
    *,
    save_times: ArrayLike,
    method: ProbabilisticODEMethod | None = None,
    step_size: ArrayLike | None = None,
    initial_covariance: ArrayLike | None = None,
    process_covariance: ArrayLike | None = None,
    observation_covariance: ArrayLike | None = None,
    parameter_covariance: ArrayLike | None = None,
    checkpoint: _ProbabilisticODECheckpoint | None = None,
) -> ProbabilisticODESolution:
    """Solve a deterministic ``DifferentialProblem`` by Gaussian ODE filtering.

    The implementation is native JAX and never dispatches to Diffrax. Numerical
    IWP diffusion, process discrepancy, residual observation noise, initial
    condition uncertainty, and parameter uncertainty remain separate covariance
    components through filtering and Rauch--Tung--Striebel smoothing.
    """
    if not isinstance(problem, DifferentialProblem):
        raise TypeError("problem must be a DifferentialProblem.")
    if problem.stochastic:
        raise ValueError(
            "Probabilistic ODE integration requires a deterministic problem; "
            "Wiener process uncertainty must be solved as an SDE."
        )
    if problem.state_geometry is not None and not problem.state_geometry.trivial:
        raise ValueError(
            "Integrated-Wiener priors currently require Euclidean state geometry."
        )
    selected = ProbabilisticODEMethod() if method is None else method
    if not isinstance(selected, ProbabilisticODEMethod):
        raise TypeError("method must be a ProbabilisticODEMethod or None.")
    requested_times = validate_save_times(problem.t0, problem.t1, save_times)
    state = jnp.asarray(problem.initial_state)
    if not jnp.issubdtype(state.dtype, jnp.inexact):
        state = state.astype(float)
    if jnp.iscomplexobj(state):
        raise TypeError("Probabilistic ODE integration requires a real-valued state.")
    state_shape = tuple(state.shape)
    state_size = int(state.size)
    derivative_count = selected.order + 1
    augmented_size = derivative_count * state_size
    if (
        selected.factorization == "dense"
        and augmented_size > selected.max_dense_dimension
    ):
        raise ValueError(
            "Dense probabilistic ODE state exceeds max_dense_dimension; select "
            "factorization='block_diagonal' explicitly."
        )
    if (
        selected.factorization == "block_diagonal"
        and selected.covariance_output == "dense"
        and state_size > selected.max_dense_dimension
    ):
        raise ValueError(
            "Dense covariance output exceeds max_dense_dimension; use "
            "covariance_output='matrix_free' and dense_covariance() explicitly."
        )

    initial_state_covariance = _covariance_matrix(
        initial_covariance,
        state_size,
        state.dtype,
        name="initial_covariance",
    )
    model_process_covariance = _covariance_matrix(
        process_covariance,
        state_size,
        state.dtype,
        name="process_covariance",
    )
    residual_observation_covariance = _covariance_matrix(
        observation_covariance,
        state_size,
        state.dtype,
        name="observation_covariance",
    )
    if selected.factorization == "block_diagonal":
        initial_state_covariance = _diagonal_covariance(
            initial_state_covariance,
            name="initial_covariance",
        )
        model_process_covariance = _diagonal_covariance(
            model_process_covariance,
            name="process_covariance",
        )
        residual_observation_covariance = _diagonal_covariance(
            residual_observation_covariance,
            name="observation_covariance",
        )
    parameter_uncertainty = parameter_covariance is not None
    flat_args, unravel_args = ravel_pytree(problem.args)
    if problem.args is None:
        if parameter_uncertainty:
            raise ValueError("parameter_covariance requires DifferentialProblem args.")
        parameter_matrix = jnp.zeros((0, 0), dtype=state.dtype)
    else:
        if parameter_uncertainty and not jnp.issubdtype(flat_args.dtype, jnp.inexact):
            raise TypeError(
                "parameter uncertainty requires inexact DifferentialProblem args."
            )
        parameter_matrix = (
            _covariance_matrix(
                parameter_covariance,
                int(flat_args.size),
                state.dtype,
                name="parameter_covariance",
            )
            if parameter_uncertainty
            else jnp.zeros((0, 0), dtype=state.dtype)
        )
    uncertain_flat_args = (
        flat_args if parameter_uncertainty else jnp.zeros((0,), dtype=state.dtype)
    )

    if checkpoint is None:
        initial_mean = _startup_mean(problem, selected.order, problem.args).reshape(
            (derivative_count, state_size)
        )
        if parameter_uncertainty:
            initial_sensitivity = jax.jacfwd(
                lambda values: _startup_mean(
                    problem,
                    selected.order,
                    unravel_args(values),
                ).reshape((derivative_count, state_size))
            )(flat_args)
        else:
            initial_sensitivity = jnp.zeros(
                (derivative_count, state_size, 0), dtype=state.dtype
            )
        if selected.factorization == "dense":
            initial_sources = jnp.zeros(
                (len(_UNCERTAINTY_NAMES), augmented_size, augmented_size),
                dtype=state.dtype,
            )
            initial_sources = initial_sources.at[
                _INITIAL_CONDITION, :state_size, :state_size
            ].set(initial_state_covariance)
        else:
            initial_sources = jnp.zeros(
                (
                    len(_UNCERTAINTY_NAMES),
                    state_size,
                    derivative_count,
                    derivative_count,
                ),
                dtype=state.dtype,
            )
            initial_sources = initial_sources.at[_INITIAL_CONDITION, :, 0, 0].set(
                jnp.diag(initial_state_covariance)
            )
        initial_time = problem.t0
        initial_quasi_sum = jnp.zeros((), dtype=state.dtype)
        initial_quasi_count = jnp.zeros((), dtype=jnp.int32)
        initial_quasi_log_sum = jnp.zeros((), dtype=state.dtype)
        grid_origin = problem.t0
        initial_step_index = jnp.zeros((), dtype=jnp.int32)
        nominal_step_size = (
            jnp.zeros((), dtype=state.dtype)
            if step_size is None
            else jnp.asarray(step_size, dtype=state.dtype)
        )
    else:
        if not isinstance(checkpoint, _ProbabilisticODECheckpoint):
            raise TypeError("checkpoint must come from a ProbabilisticODESolution.")
        if checkpoint.method_id != selected.method_id:
            raise ValueError("checkpoint and method IDs do not match.")
        if checkpoint.factorization != selected.factorization:
            raise ValueError("checkpoint and factorization do not match.")
        if checkpoint.state_shape != state_shape:
            raise ValueError("checkpoint and problem state shapes do not match.")
        if step_size is None:
            raise ValueError("Checkpoint resume requires an explicit step_size.")
        initial_time = eqx.error_if(
            checkpoint.time,
            checkpoint.time != problem.t0,
            "checkpoint time must equal DifferentialProblem t0.",
        )
        initial_mean = checkpoint.integrated_mean
        initial_sources = checkpoint.source_covariances
        initial_sensitivity = checkpoint.parameter_sensitivity
        if initial_sensitivity.shape[-1] != parameter_matrix.shape[0]:
            raise ValueError(
                "checkpoint and parameter covariance dimensions do not match."
            )
        initial_quasi_sum = checkpoint.quasi_mle_sum
        initial_quasi_count = checkpoint.quasi_mle_count
        initial_quasi_log_sum = checkpoint.quasi_log_likelihood_sum
        grid_origin = checkpoint.grid_origin
        initial_step_index = checkpoint.step_index
        nominal_step_size = eqx.error_if(
            checkpoint.nominal_step_size,
            checkpoint.nominal_step_size != jnp.asarray(step_size, dtype=state.dtype),
            "checkpoint and resume step_size values must match.",
        )

    steps = _fixed_steps(
        initial_time,
        problem.t1,
        selected.num_steps,
        step_size,
        grid_origin,
        initial_step_index,
    )
    filter_function = (
        _dense_filter if selected.factorization == "dense" else _block_filter
    )
    filter_arguments = (
        problem,
        selected,
        initial_mean,
        initial_sources,
        initial_sensitivity,
        steps,
        initial_time,
        residual_observation_covariance,
        model_process_covariance,
        uncertain_flat_args,
        unravel_args,
        initial_quasi_sum,
        initial_quasi_count,
        initial_quasi_log_sum,
    )
    if selected.adaptive:
        if step_size is not None:
            raise ValueError("adaptive integration does not accept step_size.")
        _, pilot_records = filter_function(*filter_arguments)
        pilot_errors = pilot_records[7]
        weights = (1.0 + pilot_errors) ** (-1.0 / (selected.order + 1))
        proposed_steps = (problem.t1 - initial_time) * weights / jnp.sum(weights)
        steps = jnp.concatenate(
            (
                proposed_steps[:-1],
                (problem.t1 - initial_time - jnp.sum(proposed_steps[:-1]))[None],
            )
        )
        filter_arguments = (
            problem,
            selected,
            initial_mean,
            initial_sources,
            initial_sensitivity,
            steps,
            initial_time,
            residual_observation_covariance,
            model_process_covariance,
            uncertain_flat_args,
            unravel_args,
            initial_quasi_sum,
            initial_quasi_count,
            initial_quasi_log_sum,
        )
    final_carry, records = filter_function(*filter_arguments)
    (
        final_mean,
        final_sources,
        final_sensitivity,
        final_time,
        quasi_sum,
        quasi_count,
        quasi_log_sum,
    ) = final_carry
    filtered_means = jnp.concatenate((initial_mean[None], records[2]), axis=0)
    filtered_sources = jnp.concatenate((initial_sources[None], records[3]), axis=0)
    filtered_sensitivities = jnp.concatenate(
        (initial_sensitivity[None], records[12]), axis=0
    )
    if selected.factorization == "dense":
        if selected.smoothing:
            (
                integrated_means,
                integrated_sources,
                integrated_sensitivities,
            ) = _dense_smooth(
                initial_mean,
                initial_sources,
                initial_sensitivity,
                records,
            )
        else:
            integrated_means = filtered_means
            integrated_sources = filtered_sources
            integrated_sensitivities = filtered_sensitivities
    else:
        if selected.smoothing:
            (
                integrated_means,
                integrated_sources,
                integrated_sensitivities,
            ) = _block_smooth(
                initial_mean,
                initial_sources,
                initial_sensitivity,
                records,
            )
        else:
            integrated_means = filtered_means
            integrated_sources = filtered_sources
            integrated_sensitivities = filtered_sensitivities

    active_count = jnp.maximum(quasi_count, jnp.asarray(1, dtype=jnp.int32))
    diffusion_multiplier = quasi_sum / active_count
    diffusion_scale = selected.base_diffusion * jnp.where(
        selected.diffusion_calibration == "quasi_mle",
        diffusion_multiplier,
        jnp.ones((), dtype=state.dtype),
    )
    grid = jnp.concatenate((initial_time[None], initial_time + jnp.cumsum(steps)))
    marginal_arguments = (
        requested_times,
        grid,
        integrated_means,
        integrated_sources,
        integrated_sensitivities,
        filtered_means,
        filtered_sources,
        filtered_sensitivities,
        records[0],
        records[1],
        records[11],
        selected.order,
        selected.smoothing,
        selected.base_diffusion,
        model_process_covariance,
    )
    if selected.factorization == "dense":
        saved_means, saved_sources, saved_sensitivities = _saved_dense_marginals(
            *marginal_arguments
        )
        saved_sources = 0.5 * (saved_sources + jnp.swapaxes(saved_sources, -1, -2))
        parameter_sources = jnp.einsum(
            "tdp,pq,teq->tde",
            saved_sensitivities,
            parameter_matrix,
            saved_sensitivities,
        )
        saved_sources = saved_sources.at[_PARAMETER].set(parameter_sources)
        if selected.diffusion_calibration == "quasi_mle":
            saved_sources = saved_sources.at[_NUMERICAL].multiply(diffusion_multiplier)
        factor_covariance = jnp.sum(saved_sources, axis=0)
        diagonal = jnp.diagonal(factor_covariance, axis1=-2, axis2=-1)
        covariance_factor = gaussian_factor_from_covariance(
            factor_covariance,
            factor_id=f"{selected.method_id}:posterior",
        )
        covariance_value = (
            factor_covariance if selected.covariance_output == "dense" else None
        )
    else:
        saved_means, saved_sources, saved_sensitivities = _saved_block_marginals(
            *marginal_arguments
        )
        parameter_sources = jnp.einsum(
            "tdp,pq,tdq->td",
            saved_sensitivities,
            parameter_matrix,
            saved_sensitivities,
        )
        saved_sources = saved_sources.at[_PARAMETER].set(parameter_sources)
        if selected.diffusion_calibration == "quasi_mle":
            saved_sources = saved_sources.at[_NUMERICAL].multiply(diffusion_multiplier)
        diagonal = jnp.sum(saved_sources, axis=0)
        covariance_value = (
            jax.vmap(jnp.diag)(diagonal)
            if selected.covariance_output == "dense"
            else None
        )
        covariance_factor = GaussianFactor(
            jnp.sqrt(diagonal)[..., None, None],
            factor_id=f"{selected.method_id}:posterior",
            resolved_method="block-diagonal-square-root",
        )
    means = saved_means.reshape(requested_times.shape + state_shape)
    standard_deviations = jnp.sqrt(diagonal).reshape(requested_times.shape + state_shape)
    factor_valid = (
        covariance_factor.valid
        if selected.factorization == "dense"
        else jnp.all(covariance_factor.valid, axis=-1)
    )
    finite = (
        jnp.all(jnp.isfinite(saved_means), axis=-1)
        & jnp.all(jnp.isfinite(diagonal), axis=-1)
        & factor_valid
    )
    endpoint_scale = jnp.maximum(
        jnp.abs(problem.t1),
        jnp.maximum(jnp.abs(initial_time), jnp.abs(problem.t1 - initial_time)),
    )
    endpoint_tolerance = (
        8.0 * jnp.finfo(state.dtype).eps * selected.num_steps * endpoint_scale
    )
    reached_end = jnp.abs(final_time - problem.t1) <= endpoint_tolerance
    final_time = jnp.where(reached_end, problem.t1, final_time)
    any_stiff = jnp.any(records[8] > selected.stiffness_threshold)
    all_finite = (
        jnp.all(jnp.isfinite(final_mean))
        & jnp.all(jnp.isfinite(final_sources))
        & jnp.all(jnp.isfinite(final_sensitivity))
        & jnp.all(finite)
    )
    status = jnp.where(
        ~all_finite,
        PROBABILISTIC_ODE_NONFINITE,
        jnp.where(
            ~reached_end,
            PROBABILISTIC_ODE_STEP_LIMIT_REACHED,
            jnp.where(
                any_stiff,
                PROBABILISTIC_ODE_STIFF,
                PROBABILISTIC_ODE_SUCCESS,
            ),
        ),
    ).astype(jnp.int32)
    valid = finite & (status == PROBABILISTIC_ODE_SUCCESS)
    source_mapping = frozendict(
        {
            name: (
                saved_sources[index]
                if selected.factorization == "dense"
                else saved_sources[index].reshape(requested_times.shape + state_shape)
            )
            for index, name in enumerate(_UNCERTAINTY_NAMES)
        }
    )
    residual_index = jnp.searchsorted(grid[1:], requested_times, side="left")
    residual_index = jnp.minimum(residual_index, selected.num_steps - 1)
    saved_residuals = records[5][residual_index].reshape(
        requested_times.shape + state_shape
    )
    saved_normalized = records[6][residual_index]
    log_quasi_likelihood = -0.5 * quasi_log_sum
    checkpoint_value = _ProbabilisticODECheckpoint(
        time=final_time,
        integrated_mean=final_mean,
        source_covariances=final_sources,
        parameter_sensitivity=final_sensitivity,
        quasi_mle_sum=quasi_sum,
        quasi_mle_count=quasi_count,
        quasi_log_likelihood_sum=quasi_log_sum,
        grid_origin=grid_origin,
        step_index=initial_step_index + jnp.sum(records[9], dtype=jnp.int32),
        nominal_step_size=nominal_step_size,
        method_id=selected.method_id,
        factorization=selected.factorization,
        state_shape=state_shape,
    )
    stats = frozendict(
        {
            "num_steps": jnp.sum(records[9], dtype=jnp.int32),
            "num_drift_evaluations": (
                (2 if selected.update == "ek1" else 1)
                * jnp.sum(records[9], dtype=jnp.int32)
                * (2 if selected.adaptive else 1)
            ),
            "max_normalized_residual": jnp.max(records[7]),
            "normalized_chi_square": diffusion_multiplier,
            "max_stiffness_indicator": jnp.max(records[8]),
            "pilot_used": selected.adaptive,
            "prior": f"integrated-wiener-{selected.order}",
            "update": selected.update,
            "factorization": selected.factorization,
            "calibration": selected.diffusion_calibration,
            "covariance_factor_status": covariance_factor.status,
        }
    )
    return ProbabilisticODESolution(
        times=requested_times,
        means=means,
        standard_deviations=standard_deviations,
        covariances=covariance_value,
        covariance_factor=covariance_factor,
        source_covariances=source_mapping,
        valid=valid,
        status=status,
        residuals=saved_residuals,
        normalized_residuals=saved_normalized,
        step_sizes=steps,
        diffusion_scale=diffusion_scale,
        log_quasi_likelihood=log_quasi_likelihood,
        stats=stats,
        checkpoint=checkpoint_value,
        method=selected,
        state_shape=state_shape,
        uncertainty_sources=_UNCERTAINTY_NAMES,
        covariance_representation=selected.covariance_output,
        method_id=selected.method_id,
        approximation_id=f"iwn{selected.order}-{selected.update}",
        discretization_id=(
            f"residual-adaptive:{selected.num_steps}"
            if selected.adaptive
            else f"fixed:{selected.num_steps}"
        ),
        backend="phydrax-native-jax",
    )


__all__ = [
    "PROBABILISTIC_ODE_NONFINITE",
    "PROBABILISTIC_ODE_STEP_LIMIT_REACHED",
    "PROBABILISTIC_ODE_STIFF",
    "PROBABILISTIC_ODE_SUCCESS",
    "ProbabilisticODECalibration",
    "ProbabilisticODECovarianceOutput",
    "ProbabilisticODEFactorization",
    "ProbabilisticODEMethod",
    "ProbabilisticODESolution",
    "ProbabilisticODEStatus",
    "ProbabilisticODEUpdate",
    "probabilistic_ode_status_name",
    "solve_probabilistic_ode",
]
