#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact one-dimensional Matérn Gaussian processes through state-space inference."""

from __future__ import annotations

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from phydrax.kernels import (
    AbstractPositiveDefiniteKernel,
    Matern32Kernel,
    Matern52Kernel,
    ScaleKernel,
)

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._precision import (
    precision_dtype_name,
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
)
from .._strict import StrictModule
from ..stochastic import (
    GaussianStatePrior,
    LinearGaussianParameterization,
    LinearGaussianObservationModel,
    LinearGaussianTransitionKernel,
    ObservationSequence,
    StateSpaceModel,
    StateSpaceProblem,
)
from ._kalman import (
    kalman_filter,
    kalman_status_name,
    KALMAN_SUCCESS,
    KalmanFilterResult,
    KalmanSmootherResult,
    rts_smoother,
)


_STATE_SPACE_GP_METHOD_ID = "exact-matern-sequential-square-root-kalman-rts"
_REPEATED_TIME_POLICY = (
    "training times must be unique; repeated query times and train-query overlaps "
    "share one latent schedule state"
)
STATE_SPACE_GP_SMOOTHER_FAILURE = 3


class _StateSpaceGaussianProcessArguments(StrictModule):
    observation_variance: Array
    drift_matrix: Array
    stationary_covariance: Array
    process_noise_factor: Array
    decay_rate: Array


class StateSpaceGaussianProcessPlan(StrictModule):
    """Prepared exact Matérn state-space coefficients and one sorted schedule.

    Plans are constructed by :func:`compile_state_space_kernel`. Training times must
    be unique. Repeated query times, including train-query overlaps, are represented
    by one schedule state and restored to the caller's original query order.
    """

    kernel: AbstractPositiveDefiniteKernel
    initial_time: Array
    schedule_times: Array
    inference_times: Array
    train_schedule_indices: Array
    query_schedule_indices: Array
    train_sort_indices: Array
    train_inverse_permutation: Array
    query_sort_indices: Array
    query_inverse_permutation: Array
    train_mask: Array
    schedule_observation_mask: Array
    drift_matrix: Array
    stationary_covariance: Array
    stationary_factor: Array
    process_noise: Array
    process_noise_factor: Array
    observation_map: Array
    lyapunov_residual: Array
    _problem_template: StateSpaceProblem
    compute_dtype: str = eqx.field(static=True)
    state_dimension: int = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    kernel_content_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    repeated_time_policy: str = eqx.field(static=True)
    max_schedule_size: int = eqx.field(static=True)

    @property
    def schedule_size(self) -> int:
        """Number of unique latent states in the prepared schedule."""
        return int(self.schedule_times.shape[0])

    @property
    def train_size(self) -> int:
        """Number of training positions, including masked positions."""
        return int(self.train_schedule_indices.shape[0])

    @property
    def query_size(self) -> int:
        """Number of requested query positions, including repeated positions."""
        return int(self.query_schedule_indices.shape[0])


class StateSpaceGaussianProcessResult(StrictModule):
    """Exact masked GP likelihood and query marginals with state-space provenance."""

    posterior_times: Array
    posterior_mean: Array
    posterior_variance: Array
    predictive_mean: Array
    predictive_variance: Array
    log_marginal_likelihood: Array
    active_observation_count: Array
    valid: Array
    status: Array
    query_valid: Array
    train_mask: Array
    schedule_times: Array
    schedule_observation_mask: Array
    evaluated_length_scale: Array
    evaluated_scale: Array
    filter_result: KalmanFilterResult
    smoother_result: KalmanSmootherResult
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    state_dimension: int = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    kernel_content_id: str | None = eqx.field(static=True)
    prepared_kernel_content_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    repeated_time_policy: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Whether every active filter step and every requested marginal is valid."""
        return self.valid


class _KernelStateSpace(NamedTuple):
    drift: Array
    stationary: Array
    stationary_factor: Array
    process_factor: Array
    observation_map: Array
    length_scale: Array
    variance: Array
    decay_rate: Array


def _kernel_state_space(kernel: AbstractPositiveDefiniteKernel, /) -> _KernelStateSpace:
    if isinstance(kernel, ScaleKernel):
        base = kernel.kernel
        variance = kernel.scale
    else:
        base = kernel
        variance = jnp.ones_like(base.length_scale)
    length_scale = base.length_scale
    dtype = jnp.result_type(length_scale, variance)
    length_scale = jnp.asarray(length_scale, dtype=dtype).reshape(())
    variance = jnp.asarray(variance, dtype=dtype).reshape(())
    one = jnp.ones((), dtype=dtype)
    zero = jnp.zeros((), dtype=dtype)
    amplitude = jnp.sqrt(variance)

    if isinstance(base, Matern32Kernel):
        rate = jnp.sqrt(jnp.asarray(3.0, dtype=dtype)) / length_scale
        drift = jnp.stack(
            (
                jnp.stack((zero, one)),
                jnp.stack((-rate**2, -2.0 * rate)),
            )
        )
        stationary = variance * jnp.stack(
            (
                jnp.stack((one, zero)),
                jnp.stack((zero, rate**2)),
            )
        )
        stationary_factor = amplitude * jnp.stack(
            (
                jnp.stack((one, zero)),
                jnp.stack((zero, rate)),
            )
        )
        spectral_density = 4.0 * variance * rate**3
        process_factor = jnp.stack(
            (zero, jnp.sqrt(spectral_density))
        ).reshape((2, 1))
        observation_map = jnp.stack((one, zero)).reshape((1, 2))
        return _KernelStateSpace(
            drift,
            stationary,
            stationary_factor,
            process_factor,
            observation_map,
            length_scale,
            variance,
            rate,
        )

    if isinstance(base, Matern52Kernel):
        rate = jnp.sqrt(jnp.asarray(5.0, dtype=dtype)) / length_scale
        rate_squared = rate**2
        drift = jnp.stack(
            (
                jnp.stack((zero, one, zero)),
                jnp.stack((zero, zero, one)),
                jnp.stack((-rate**3, -3.0 * rate_squared, -3.0 * rate)),
            )
        )
        stationary = variance * jnp.stack(
            (
                jnp.stack((one, zero, -rate_squared / 3.0)),
                jnp.stack((zero, rate_squared / 3.0, zero)),
                jnp.stack((-rate_squared / 3.0, zero, rate_squared**2)),
            )
        )
        stationary_factor = amplitude * jnp.stack(
            (
                jnp.stack((one, zero, zero)),
                jnp.stack((zero, rate / jnp.sqrt(3.0), zero)),
                jnp.stack(
                    (
                        -rate_squared / 3.0,
                        zero,
                        2.0 * jnp.sqrt(2.0) * rate_squared / 3.0,
                    )
                ),
            )
        )
        spectral_density = (16.0 / 3.0) * variance * rate**5
        process_factor = jnp.stack(
            (zero, zero, jnp.sqrt(spectral_density))
        ).reshape((3, 1))
        observation_map = jnp.stack((one, zero, zero)).reshape((1, 3))
        return _KernelStateSpace(
            drift,
            stationary,
            stationary_factor,
            process_factor,
            observation_map,
            length_scale,
            variance,
            rate,
        )

    raise TypeError(
        "State-space GP compilation supports Matern32Kernel and Matern52Kernel, "
        "directly or as the child of one ScaleKernel."
    )


def _transition_matrix(start: Array, end: Array, context: Any, /) -> Array:
    duration = end - start
    return jax.scipy.linalg.expm(context.args.drift_matrix * duration)


def _small_interval_covariance(
    duration: Array,
    drift: Array,
    process_factor: Array,
    /,
) -> Array:
    size = drift.shape[0]
    covariance_rate = process_factor @ process_factor.T
    block = jnp.zeros((2 * size, 2 * size), dtype=drift.dtype)
    block = block.at[:size, :size].set(drift)
    block = block.at[:size, size:].set(covariance_rate)
    block = block.at[size:, size:].set(-drift.T)
    exponential = jax.scipy.linalg.expm(block * duration)
    transition = exponential[:size, :size]
    covariance = exponential[:size, size:] @ transition.T
    return 0.5 * (covariance + covariance.T)


def _stationary_interval_covariance(
    duration: Array,
    drift: Array,
    stationary: Array,
    /,
) -> Array:
    transition = jax.scipy.linalg.expm(drift * duration)
    covariance = stationary - transition @ stationary @ transition.T
    return 0.5 * (covariance + covariance.T)


def _transition_covariance(start: Array, end: Array, context: Any, /) -> Array:
    duration = end - start
    arguments = context.args
    normalized_duration = arguments.decay_rate * duration
    return jax.lax.cond(
        normalized_duration <= jnp.asarray(4.0, dtype=duration.dtype),
        lambda value: _small_interval_covariance(
            value,
            arguments.drift_matrix,
            arguments.process_noise_factor,
        ),
        lambda value: _stationary_interval_covariance(
            value,
            arguments.drift_matrix,
            arguments.stationary_covariance,
        ),
        duration,
    )


def _kernel_content_id(kernel: AbstractPositiveDefiniteKernel, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "state-space-gp-kernel",
            "kernel_id": kernel.kernel_id,
            "content": array_tree_fingerprint(kernel),
        }
    )


def _evaluated_kernel_content_id(
    kernel: AbstractPositiveDefiniteKernel,
    /,
) -> str | None:
    leaves = tuple(leaf for leaf in jax.tree.leaves(kernel) if eqx.is_array(leaf))
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
        return None
    return _kernel_content_id(kernel)


def state_space_gaussian_process_status_name(value: int, /) -> str:
    """Return the public GP-level status name for one result code."""
    code = int(value)
    if code == STATE_SPACE_GP_SMOOTHER_FAILURE:
        return "smoother_failure"
    return kalman_status_name(code)


def _state_space_gp_status(
    filter_success: Array,
    filter_status: Array,
    requested_valid: Array,
    /,
) -> Array:
    requested_success = jnp.all(requested_valid)
    return jnp.where(
        ~filter_success,
        jnp.max(filter_status),
        jnp.where(
            ~requested_success,
            STATE_SPACE_GP_SMOOTHER_FAILURE,
            KALMAN_SUCCESS,
        ),
    ).astype(jnp.int32)


def _observation_covariance(time: Array, context: Any, /) -> Array:
    del time
    variance = context.args.observation_variance[context.step_index]
    return variance.reshape((1, 1))


def _as_time_vector(value: ArrayLike, /, *, name: str) -> Array:
    times = jnp.asarray(value, dtype=float)
    if times.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional time vector.")
    host = np.asarray(jax.device_get(times))
    if not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must contain only finite times.")
    return times


def _validated_kernel(kernel: AbstractPositiveDefiniteKernel, /) -> None:
    if isinstance(kernel, ScaleKernel):
        if isinstance(kernel.kernel, ScaleKernel):
            raise TypeError("Nested ScaleKernel wrappers are not supported.")
        base = kernel.kernel
        scale = np.asarray(jax.device_get(kernel.scale))
        if scale.ndim != 0 or not np.isfinite(scale) or scale < 0.0:
            raise ValueError("ScaleKernel scale must be finite and nonnegative.")
    else:
        base = kernel
    if not isinstance(base, (Matern32Kernel, Matern52Kernel)):
        raise TypeError(
            "kernel must be Matern32Kernel or Matern52Kernel, directly or inside "
            "one ScaleKernel."
        )
    length_scale = np.asarray(jax.device_get(base.length_scale))
    if length_scale.ndim != 0:
        raise ValueError("State-space Matérn length_scale must be scalar.")
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("State-space Matérn length_scale must be finite and positive.")


def compile_state_space_kernel(
    kernel: AbstractPositiveDefiniteKernel,
    train_times: ArrayLike,
    query_times: ArrayLike,
    /,
    *,
    train_mask: ArrayLike | None = None,
    max_schedule_size: int = 1_000_000,
) -> StateSpaceGaussianProcessPlan:
    """Prepare one exact scalar Matérn state-space GP schedule.

    Preparation is a host-time operation: it validates kernel support, performs a
    stable sort, rejects repeated training times, deduplicates repeated query times,
    and verifies the continuous stationary Lyapunov equation. ``train_mask`` marks
    exact missing observations; query-only positions are always unobserved.
    """
    if not isinstance(kernel, AbstractPositiveDefiniteKernel):
        raise TypeError("kernel must be a positive-definite kernel.")
    _validated_kernel(kernel)
    limit = int(max_schedule_size)
    if limit <= 0:
        raise ValueError("max_schedule_size must be positive.")
    train = _as_time_vector(train_times, name="train_times")
    query = _as_time_vector(query_times, name="query_times")
    if int(train.size + query.size) == 0:
        raise ValueError("At least one training or query time is required.")

    train_host = np.asarray(jax.device_get(train))
    query_host = np.asarray(jax.device_get(query))
    train_sort_host = np.argsort(train_host, kind="stable")
    query_sort_host = np.argsort(query_host, kind="stable")
    if train_host.size > 1 and np.any(np.diff(train_host[train_sort_host]) == 0.0):
        raise ValueError(
            "Repeated training times are unsupported; combine observations before "
            "state-space GP preparation."
        )
    combined = np.concatenate((train_host, query_host))
    schedule_host, combined_inverse = np.unique(combined, return_inverse=True)
    if schedule_host.size > limit:
        raise ValueError(
            f"Unique schedule size {schedule_host.size} exceeds max_schedule_size={limit}."
        )
    train_schedule_host = combined_inverse[: train_host.size]
    query_schedule_host = combined_inverse[train_host.size :]

    if train_mask is None:
        mask = jnp.ones(train.shape, dtype=bool)
    else:
        mask = jnp.asarray(train_mask, dtype=bool)
        if mask.shape != train.shape:
            raise ValueError("train_mask must have the same shape as train_times.")
    mask_host = np.asarray(jax.device_get(mask))
    schedule_mask_host = np.zeros((schedule_host.size,), dtype=bool)
    schedule_mask_host[train_schedule_host] = mask_host

    dtype = train.dtype if train.size else query.dtype
    schedule = jnp.asarray(schedule_host, dtype=dtype)
    train_schedule = jnp.asarray(train_schedule_host, dtype=jnp.int32)
    query_schedule = jnp.asarray(query_schedule_host, dtype=jnp.int32)
    train_sort = jnp.asarray(train_sort_host, dtype=jnp.int32)
    query_sort = jnp.asarray(query_sort_host, dtype=jnp.int32)
    train_inverse = jnp.argsort(train_sort)
    query_inverse = jnp.argsort(query_sort)
    schedule_mask = jnp.asarray(schedule_mask_host, dtype=bool)

    components = _kernel_state_space(kernel)
    if schedule.dtype != components.drift.dtype:
        raise TypeError(
            "Kernel coefficients and schedule times must use one identical compute "
            "dtype; construct both under the same JAX precision context."
        )
    inference_schedule = schedule - schedule[0]
    initial_time = -components.length_scale
    process_noise = components.process_factor @ components.process_factor.T
    lyapunov = (
        components.drift @ components.stationary
        + components.stationary @ components.drift.T
        + process_noise
    )
    residual_host = np.asarray(jax.device_get(lyapunov))
    scale_host = max(
        1.0,
        float(np.max(np.abs(np.asarray(jax.device_get(process_noise))))),
        float(np.max(np.abs(np.asarray(jax.device_get(components.stationary))))),
    )
    tolerance = 256.0 * np.finfo(np.asarray(residual_host).dtype).eps * scale_host
    if not np.all(np.isfinite(residual_host)) or np.max(np.abs(residual_host)) > tolerance:
        raise ValueError(
            "Prepared Matérn coefficients fail the stationary Lyapunov residual check."
        )

    kernel_content_id = _kernel_content_id(kernel)
    schedule_id = canonical_fingerprint(
        {
            "kind": "state-space-gp-schedule",
            "times": schedule_host.tolist(),
            "train_schedule_indices": train_schedule_host.tolist(),
            "query_schedule_indices": query_schedule_host.tolist(),
            "train_mask": mask_host.tolist(),
            "repeated_time_policy": _REPEATED_TIME_POLICY,
        }
    )
    state_dimension = int(components.drift.shape[0])
    prior = GaussianStatePrior(
        jnp.zeros((state_dimension,), dtype=components.drift.dtype),
        components.stationary,
        state_shape=(state_dimension,),
        prior_id="state-space-gp-stationary-prior",
    )
    transition = LinearGaussianTransitionKernel(
        LinearGaussianParameterization(
            _transition_matrix,
            _transition_covariance,
            state_shape=(state_dimension,),
            parameterization_id="state-space-gp-stationary-interval",
            resolved_method="hybrid-van-loan/stationary-covariance",
        ),
        process_id="state-space-gp-matern-process",
        approximation_id="exact-matern-lti",
    )
    observation = LinearGaussianObservationModel(
        components.observation_map,
        _observation_covariance,
        state_shape=(state_dimension,),
        observation_shape=(1,),
        observation_id="state-space-gp-value-observation",
    )
    model = StateSpaceModel(
        prior,
        transition,
        observation,
        model_id="state-space-gp-model",
        metadata={
            "kernel_id": kernel.kernel_id,
            "method_id": _STATE_SPACE_GP_METHOD_ID,
        },
    )
    observations = ObservationSequence(
        inference_schedule,
        jnp.zeros((schedule.size, 1), dtype=components.drift.dtype),
        observation_axes=("value",),
        observation_mask=schedule_mask[:, None],
        sequence_id=f"state-space-gp-schedule:{schedule_id}",
        sensor_id="state-space-gp-values",
        discretization_id="exact-sorted-temporal-schedule",
        approximation_id="exact-mask",
    )
    arguments = _StateSpaceGaussianProcessArguments(
        observation_variance=jnp.zeros(
            (schedule.size,), dtype=components.drift.dtype
        ),
        drift_matrix=components.drift,
        stationary_covariance=components.stationary,
        process_noise_factor=components.process_factor,
        decay_rate=components.decay_rate,
    )
    problem = StateSpaceProblem(
        model,
        observations,
        initial_time=initial_time,
        problem_id=f"state-space-gp-problem:{schedule_id}",
        args=arguments,
    )
    problem = eqx.tree_at(
        lambda node: node.model.prior.factor,
        problem,
        components.stationary_factor,
    )

    return StateSpaceGaussianProcessPlan(
        kernel=kernel,
        initial_time=initial_time,
        schedule_times=schedule,
        inference_times=inference_schedule,
        train_schedule_indices=train_schedule,
        query_schedule_indices=query_schedule,
        train_sort_indices=train_sort,
        train_inverse_permutation=train_inverse,
        query_sort_indices=query_sort,
        query_inverse_permutation=query_inverse,
        train_mask=mask,
        schedule_observation_mask=schedule_mask,
        drift_matrix=components.drift,
        stationary_covariance=components.stationary,
        stationary_factor=components.stationary_factor,
        process_noise=process_noise,
        process_noise_factor=components.process_factor,
        observation_map=components.observation_map,
        lyapunov_residual=lyapunov,
        _problem_template=problem,
        compute_dtype=components.drift.dtype.name,
        state_dimension=state_dimension,
        kernel_id=kernel.kernel_id,
        kernel_content_id=kernel_content_id,
        schedule_id=schedule_id,
        method_id=_STATE_SPACE_GP_METHOD_ID,
        repeated_time_policy=_REPEATED_TIME_POLICY,
        max_schedule_size=limit,
    )


def _precision_evidence(dtype: Any, /) -> PrecisionEvidenceEnvelope:
    name = precision_dtype_name(dtype)
    request = PrecisionRequest(
        "state-space-gaussian-process",
        {
            "storage": name,
            "compute": name,
            "factorization": name,
            "output": name,
        },
    )
    resolution = PrecisionResolution(
        request,
        "phydrax-sequential-square-root-kalman-rts",
        dict(request.requested),
    )
    return PrecisionEvidenceEnvelope(resolution, dict(resolution.effective))


def fit_state_space_gaussian_process(
    plan: StateSpaceGaussianProcessPlan,
    train_values: ArrayLike,
    /,
    *,
    noise_scale: ArrayLike = 0.0,
) -> StateSpaceGaussianProcessResult:
    """Evaluate an exact masked Matérn GP likelihood and query marginals.

    ``noise_scale`` is the scalar observation standard deviation. The execution is
    always the canonical sequential square-root Kalman filter followed by its
    matching square-root RTS smoother. No covariance sentinel, regularization,
    projection, clipping, or missing-data approximation is introduced.
    """
    if not isinstance(plan, StateSpaceGaussianProcessPlan):
        raise TypeError("plan must be a StateSpaceGaussianProcessPlan.")
    values = jnp.asarray(train_values, dtype=plan.schedule_times.dtype)
    if values.shape != (plan.train_size,):
        raise ValueError("train_values must have shape (plan.train_size,).")
    values = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)),
        "train_values must be finite; represent missing values with train_mask.",
    )
    noise = jnp.asarray(noise_scale, dtype=values.dtype)
    if noise.ndim != 0:
        raise ValueError("noise_scale must be scalar.")
    noise = eqx.error_if(
        noise,
        ~jnp.isfinite(noise) | (noise < 0.0),
        "noise_scale must be finite and nonnegative.",
    )

    components = _kernel_state_space(plan.kernel)
    if components.drift.dtype.name != plan.compute_dtype:
        raise TypeError(
            "The evaluated kernel dtype must match the plan compute dtype; recompile "
            "the plan under the intended JAX precision context."
        )
    initial_time = eqx.error_if(
        -components.length_scale,
        ~jnp.isfinite(components.length_scale) | (components.length_scale <= 0.0),
        "The evaluated length scale must be finite and strictly positive.",
    )
    schedule_values = jnp.zeros(
        (plan.schedule_size, 1), dtype=components.drift.dtype
    ).at[plan.train_schedule_indices, 0].set(values)
    observation_variance = jnp.full(
        (plan.schedule_size,), noise**2, dtype=components.drift.dtype
    )
    arguments = _StateSpaceGaussianProcessArguments(
        observation_variance=observation_variance,
        drift_matrix=components.drift,
        stationary_covariance=components.stationary,
        process_noise_factor=components.process_factor,
        decay_rate=components.decay_rate,
    )
    problem = eqx.tree_at(
        lambda node: (
            node.model.prior.covariance,
            node.model.prior.factor,
            node.model.observation.matrix,
            node.observations.values,
            node.initial_time,
            node.args,
        ),
        plan._problem_template,
        (
            components.stationary,
            components.stationary_factor,
            components.observation_map,
            schedule_values,
            initial_time,
            arguments,
        ),
    )
    filtered = kalman_filter(
        problem,
        method="sequential",
        covariance_form="square_root",
        covariance_regularization=0.0,
        raise_on_failure=False,
    )
    smoothed = rts_smoother(
        filtered,
        method="sequential",
        covariance_form="square_root",
    )
    query_states = smoothed.means[plan.query_schedule_indices]
    query_covariances = smoothed.covariances[plan.query_schedule_indices]
    posterior_mean = oe.contract(
        "qi,ji->q", query_states, components.observation_map
    )
    posterior_variance = oe.contract(
        "ai,qij,aj->q",
        components.observation_map,
        query_covariances,
        components.observation_map,
    )
    query_valid = smoothed.valid[plan.query_schedule_indices]
    filter_success = jnp.all(filtered.successful)
    query_success = jnp.all(query_valid)
    valid = filter_success & query_success
    status = _state_space_gp_status(
        filter_success,
        filtered.status,
        query_valid,
    )

    return StateSpaceGaussianProcessResult(
        posterior_times=plan.schedule_times[plan.query_schedule_indices],
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
        predictive_mean=posterior_mean,
        predictive_variance=posterior_variance + noise**2,
        log_marginal_likelihood=filtered.final_state.log_likelihood.reshape(()),
        active_observation_count=jnp.sum(plan.train_mask, dtype=jnp.int32),
        valid=valid,
        status=status,
        query_valid=query_valid,
        train_mask=plan.train_mask,
        schedule_times=plan.schedule_times,
        schedule_observation_mask=plan.schedule_observation_mask,
        evaluated_length_scale=components.length_scale,
        evaluated_scale=components.variance,
        filter_result=filtered,
        smoother_result=smoothed,
        precision_evidence=_precision_evidence(posterior_mean.dtype),
        state_dimension=plan.state_dimension,
        kernel_id=plan.kernel_id,
        kernel_content_id=_evaluated_kernel_content_id(plan.kernel),
        prepared_kernel_content_id=plan.kernel_content_id,
        schedule_id=plan.schedule_id,
        method_id=plan.method_id,
        repeated_time_policy=plan.repeated_time_policy,
    )


__all__ = [
    "STATE_SPACE_GP_SMOOTHER_FAILURE",
    "StateSpaceGaussianProcessPlan",
    "StateSpaceGaussianProcessResult",
    "compile_state_space_kernel",
    "fit_state_space_gaussian_process",
    "state_space_gaussian_process_status_name",
]
