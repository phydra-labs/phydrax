#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded exact state-space Gaussian processes on prepared finite designs."""

from __future__ import annotations

from numbers import Integral
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein
from phydrax.kernels import (
    AbstractPositiveDefiniteKernel,
    AmplitudeKernel,
    CARMAKernel,
    Matern32Kernel,
    Matern52Kernel,
    ScaleKernel,
    SHOKernel,
    SumKernel,
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
    LinearGaussianObservationModel,
    LinearGaussianParameterization,
    LinearGaussianTransitionKernel,
    ObservationSequence,
    StateSpaceModel,
    StateSpaceProblem,
)
from ._gp_functional import functional_kernel_matrix, FunctionalDesign
from ._kalman import (
    kalman_filter,
    kalman_status_name,
    KALMAN_SUCCESS,
    KalmanFilterResult,
    KalmanSmootherResult,
    rts_smoother,
)


_STATE_SPACE_GP_METHOD_ID = "bounded-rational-separable-kalman-rts"
_REPEATED_TIME_POLICY = (
    "observations at equal times occupy distinct fixed-capacity rows; repeated query "
    "times share one latent schedule state"
)
STATE_SPACE_GP_SMOOTHER_FAILURE = 3


class StateSpaceGaussianProcessDesign(StrictModule):
    """Finite train/query schedule and optional certified spatial functionals."""

    train_times: Array
    query_times: Array
    train_time_derivative_order: Array
    query_time_derivative_order: Array
    train_mask: Array
    train_spatial: FunctionalDesign | None
    query_spatial: FunctionalDesign | None

    def __init__(
        self,
        train_times: ArrayLike,
        query_times: ArrayLike,
        /,
        *,
        train_spatial: FunctionalDesign | None = None,
        query_spatial: FunctionalDesign | None = None,
        train_time_derivative_order: ArrayLike | None = None,
        query_time_derivative_order: ArrayLike | None = None,
        train_mask: ArrayLike | None = None,
    ):
        train = _as_time_vector(train_times, name="train_times")
        query = _as_time_vector(query_times, name="query_times")
        if int(train.size + query.size) == 0:
            raise ValueError("At least one training or query time is required.")
        train_orders = _derivative_orders(
            train_time_derivative_order,
            size=int(train.size),
            name="train_time_derivative_order",
        )
        query_orders = _derivative_orders(
            query_time_derivative_order,
            size=int(query.size),
            name="query_time_derivative_order",
        )
        if train_mask is None:
            mask = jnp.ones(train.shape, dtype=bool)
        else:
            mask = jnp.asarray(train_mask, dtype=bool)
            if mask.shape != train.shape:
                raise ValueError("train_mask must align with train_times.")
        if train_spatial is not None:
            if not isinstance(train_spatial, FunctionalDesign):
                raise TypeError("train_spatial must be a FunctionalDesign or None.")
            if train_spatial.num_observations != int(train.size):
                raise ValueError("train_spatial must align with train_times.")
        if query_spatial is not None:
            if not isinstance(query_spatial, FunctionalDesign):
                raise TypeError("query_spatial must be a FunctionalDesign or None.")
            if query_spatial.num_observations != int(query.size):
                raise ValueError("query_spatial must align with query_times.")
        self.train_times = train
        self.query_times = query
        self.train_time_derivative_order = train_orders
        self.query_time_derivative_order = query_orders
        self.train_mask = mask
        self.train_spatial = train_spatial
        self.query_spatial = query_spatial

    @property
    def train_size(self) -> int:
        return int(self.train_times.shape[0])

    @property
    def query_size(self) -> int:
        return int(self.query_times.shape[0])


class _StateSpaceGaussianProcessArguments(StrictModule):
    observation_covariance: Array
    observation_matrices: Array
    drift_matrix: Array
    stationary_covariance: Array
    process_noise_factor: Array
    characteristic_rate: Array


class StateSpaceGaussianProcessPlan(StrictModule):
    """Immutable finite-capacity state-space GP execution plan."""

    temporal_kernel: AbstractPositiveDefiniteKernel
    spatial_kernel: AbstractPositiveDefiniteKernel | None
    design: StateSpaceGaussianProcessDesign
    initial_time: Array
    schedule_times: Array
    inference_times: Array
    train_gather_indices: Array
    query_schedule_indices: Array
    train_mask: Array
    schedule_observation_mask: Array
    observation_matrices: Array
    query_rows: Array
    spatial_factor: Array
    drift_matrix: Array
    stationary_covariance: Array
    stationary_factor: Array
    process_noise: Array
    process_noise_factor: Array
    lyapunov_residual: Array
    _problem_template: StateSpaceProblem
    compute_dtype: str = eqx.field(static=True)
    temporal_state_dimension: int = eqx.field(static=True)
    spatial_rank: int = eqx.field(static=True)
    state_dimension: int = eqx.field(static=True)
    max_observations_per_time: int = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    kernel_content_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    repeated_time_policy: str = eqx.field(static=True)
    max_schedule_size: int = eqx.field(static=True)
    max_state_dimension: int = eqx.field(static=True)

    @property
    def schedule_size(self) -> int:
        return int(self.schedule_times.shape[0])

    @property
    def train_size(self) -> int:
        return self.design.train_size

    @property
    def query_size(self) -> int:
        return self.design.query_size


class StateSpaceGaussianProcessResult(StrictModule):
    """Exact Gaussian finite-design marginals with Kalman provenance."""

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
    spatial_rank: int = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    kernel_content_id: str | None = eqx.field(static=True)
    prepared_kernel_content_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    repeated_time_policy: str = eqx.field(static=True)
    temporal_method: str = eqx.field(static=True)
    covariance_form: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid


class _KernelStateSpace(NamedTuple):
    drift: Array
    stationary: Array
    stationary_factor: Array
    process_factor: Array
    observation_row: Array
    characteristic_rate: Array
    component_scales: Array


class _ExpandedStateSpace(NamedTuple):
    drift: Array
    stationary: Array
    stationary_factor: Array
    process_factor: Array
    observation_matrices: Array
    query_rows: Array
    characteristic_rate: Array
    component_scales: Array


def _kernel_state_space(kernel: AbstractPositiveDefiniteKernel, /) -> _KernelStateSpace:
    if isinstance(kernel, ScaleKernel):
        return _scaled_state_space(_kernel_state_space(kernel.kernel), kernel.scale)
    if isinstance(kernel, AmplitudeKernel):
        return _scaled_state_space(
            _kernel_state_space(kernel.kernel), kernel.variance_scale
        )
    if isinstance(kernel, SumKernel):
        return _sum_state_spaces(
            tuple(_kernel_state_space(child) for child in kernel.kernels)
        )
    if isinstance(kernel, (Matern32Kernel, Matern52Kernel)):
        length_scale = jnp.asarray(kernel.length_scale)
        if length_scale.ndim != 0:
            raise ValueError("State-space Matérn length_scale must be scalar.")
        dtype = length_scale.dtype
        variance = jnp.ones((), dtype=dtype)
        if isinstance(kernel, Matern32Kernel):
            rate = jnp.sqrt(jnp.asarray(3.0, dtype=dtype)) / length_scale
            drift = rate * jnp.asarray(((0.0, 1.0), (-1.0, -2.0)), dtype=dtype)
            stationary = jnp.eye(2, dtype=dtype)
            factor = stationary
            process_factor = jnp.asarray(((0.0,), (jnp.sqrt(4.0 * rate),)), dtype=dtype)
            observation = jnp.asarray((1.0, 0.0), dtype=dtype)
        else:
            rate = jnp.sqrt(jnp.asarray(5.0, dtype=dtype)) / length_scale
            drift = rate * jnp.asarray(
                ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, -3.0, -3.0)),
                dtype=dtype,
            )
            stationary = jnp.asarray(
                ((1.0, 0.0, -1.0 / 3.0), (0.0, 1.0 / 3.0, 0.0), (-1.0 / 3.0, 0.0, 1.0)),
                dtype=dtype,
            )
            factor = jnp.asarray(
                (
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0 / jnp.sqrt(3.0), 0.0),
                    (-1.0 / 3.0, 0.0, 2.0 * jnp.sqrt(2.0) / 3.0),
                ),
                dtype=dtype,
            )
            process_factor = jnp.asarray(
                ((0.0,), (0.0,), (jnp.sqrt((16.0 / 3.0) * rate),)), dtype=dtype
            )
            observation = jnp.asarray((1.0, 0.0, 0.0), dtype=dtype)
        return _KernelStateSpace(
            drift,
            stationary,
            factor,
            process_factor,
            observation,
            rate,
            jnp.asarray((variance,), dtype=dtype),
        )
    if isinstance(kernel, SHOKernel):
        variance = kernel.variance
        omega = kernel.frequency
        stationary = kernel.stationary_covariance
        factor = jnp.diag(jnp.asarray((jnp.sqrt(variance), omega * jnp.sqrt(variance))))
        process_factor = jnp.asarray(
            ((0.0,), (jnp.sqrt(kernel.diffusion_intensity),)), dtype=omega.dtype
        )
        return _KernelStateSpace(
            kernel.drift_matrix,
            stationary,
            factor,
            process_factor,
            kernel.observation_row,
            omega / kernel.quality_factor,
            jnp.asarray((variance,), dtype=omega.dtype),
        )
    if isinstance(kernel, CARMAKernel):
        factor = jax.scipy.linalg.cholesky(kernel.stationary_covariance, lower=True)
        process_factor = (
            jnp.zeros((kernel.order, 1), dtype=kernel.stationary_covariance.dtype)
            .at[-1, 0]
            .set(kernel.innovation_scale)
        )
        variance = (
            kernel.observation_row @ kernel.stationary_covariance @ kernel.observation_row
        )
        return _KernelStateSpace(
            kernel.drift_matrix,
            kernel.stationary_covariance,
            factor,
            process_factor,
            kernel.observation_row,
            jnp.asarray(
                kernel.stability_margin, dtype=kernel.stationary_covariance.dtype
            ),
            jnp.asarray((variance,), dtype=kernel.stationary_covariance.dtype),
        )
    raise TypeError(
        "State-space GP compilation supports Matérn-3/2, Matérn-5/2, SHO, stable "
        "CARMA, nonnegative scale/amplitude wrappers, and finite sums of them."
    )


def _scaled_state_space(
    component: _KernelStateSpace, scale: Array, /
) -> _KernelStateSpace:
    scale_array = jnp.asarray(scale, dtype=component.drift.dtype).reshape(())
    root = jnp.sqrt(scale_array)
    return _KernelStateSpace(
        component.drift,
        scale_array * component.stationary,
        root * component.stationary_factor,
        root * component.process_factor,
        component.observation_row,
        component.characteristic_rate,
        scale_array * component.component_scales,
    )


def _sum_state_spaces(components: tuple[_KernelStateSpace, ...], /) -> _KernelStateSpace:
    if not components:
        raise ValueError("At least one temporal Markov component is required.")
    dtype = components[0].drift.dtype
    if any(component.drift.dtype != dtype for component in components[1:]):
        raise TypeError("Temporal sum components must use one compute dtype.")
    drift = jax.scipy.linalg.block_diag(*(component.drift for component in components))
    stationary = jax.scipy.linalg.block_diag(
        *(component.stationary for component in components)
    )
    stationary_factor = jax.scipy.linalg.block_diag(
        *(component.stationary_factor for component in components)
    )
    process_factor = jax.scipy.linalg.block_diag(
        *(component.process_factor for component in components)
    )
    observation = jnp.concatenate(
        tuple(component.observation_row for component in components)
    )
    rates = jnp.stack(tuple(component.characteristic_rate for component in components))
    scales = jnp.concatenate(
        tuple(component.component_scales for component in components)
    )
    return _KernelStateSpace(
        drift,
        stationary,
        stationary_factor,
        process_factor,
        observation,
        jnp.min(rates),
        scales,
    )


def _transition_matrix(start: Array, end: Array, context: Any, /) -> Array:
    return jax.scipy.linalg.expm(context.args.drift_matrix * (end - start))


def _small_interval_covariance(
    duration: Array, drift: Array, process_factor: Array, /
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


def _transition_covariance(start: Array, end: Array, context: Any, /) -> Array:
    duration = end - start
    arguments = context.args
    return jax.lax.cond(
        arguments.characteristic_rate * duration <= 4.0,
        lambda value: _small_interval_covariance(
            value, arguments.drift_matrix, arguments.process_noise_factor
        ),
        lambda value: _stationary_interval_covariance(
            value, arguments.drift_matrix, arguments.stationary_covariance
        ),
        duration,
    )


def _stationary_interval_covariance(
    duration: Array, drift: Array, stationary: Array, /
) -> Array:
    transition = jax.scipy.linalg.expm(drift * duration)
    covariance = stationary - transition @ stationary @ transition.T
    return 0.5 * (covariance + covariance.T)


def _observation_matrix(time: Array, context: Any, /) -> Array:
    del time
    return context.args.observation_matrices[context.step_index]


def _observation_covariance(time: Array, context: Any, /) -> Array:
    del time
    return context.args.observation_covariance[context.step_index]


def _expanded_state_space(
    temporal: _KernelStateSpace,
    spatial_factor: Array,
    train_orders: Array,
    query_orders: Array,
    train_gather: Array,
    schedule_size: int,
    row_capacity: int,
    /,
) -> _ExpandedStateSpace:
    rank = int(spatial_factor.shape[1])
    identity = jnp.eye(rank, dtype=temporal.drift.dtype)
    drift = jnp.kron(identity, temporal.drift)
    stationary = jnp.kron(identity, temporal.stationary)
    stationary_factor = jnp.kron(identity, temporal.stationary_factor)
    process_factor = jnp.kron(identity, temporal.process_factor)
    state_size = int(drift.shape[0])
    observation_matrices = jnp.zeros(
        (schedule_size, row_capacity, state_size), dtype=drift.dtype
    )
    train_count = int(train_orders.shape[0])
    for index in range(train_count):
        temporal_row = _time_observation_row(
            temporal, int(np.asarray(jax.device_get(train_orders[index])))
        )
        row = jnp.kron(spatial_factor[index], temporal_row)
        time_index = int(np.asarray(jax.device_get(train_gather[index, 0])))
        row_index = int(np.asarray(jax.device_get(train_gather[index, 1])))
        observation_matrices = observation_matrices.at[time_index, row_index].set(row)
    query_rows = []
    offset = train_count
    for index in range(int(query_orders.shape[0])):
        temporal_row = _time_observation_row(
            temporal, int(np.asarray(jax.device_get(query_orders[index])))
        )
        query_rows.append(jnp.kron(spatial_factor[offset + index], temporal_row))
    query_matrix = (
        jnp.stack(tuple(query_rows))
        if query_rows
        else jnp.zeros((0, state_size), dtype=drift.dtype)
    )
    return _ExpandedStateSpace(
        drift,
        stationary,
        stationary_factor,
        process_factor,
        observation_matrices,
        query_matrix,
        temporal.characteristic_rate,
        temporal.component_scales,
    )


def _time_observation_row(component: _KernelStateSpace, order: int, /) -> Array:
    if order < 0:
        raise ValueError("Time derivative orders must be nonnegative.")
    row = component.observation_row
    tolerance = 256.0 * jnp.finfo(component.drift.dtype).eps
    for _ in range(order):
        feedthrough = row @ component.process_factor
        host = np.asarray(jax.device_get(feedthrough))
        if not np.all(np.isfinite(host)) or np.max(np.abs(host)) > float(tolerance):
            raise ValueError(
                "Requested time derivative lacks the required mean-square differentiability certificate."
            )
        row = row @ component.drift
    return row


def _spatial_factor(
    design: StateSpaceGaussianProcessDesign,
    spatial_kernel: AbstractPositiveDefiniteKernel | None,
    /,
) -> Array:
    total = design.train_size + design.query_size
    if spatial_kernel is None:
        if design.train_spatial is not None or design.query_spatial is not None:
            raise ValueError(
                "A spatial_kernel is required for spatial functional designs."
            )
        return jnp.ones((total, 1), dtype=design.train_times.dtype)
    if not isinstance(spatial_kernel, AbstractPositiveDefiniteKernel):
        raise TypeError("spatial_kernel must be a positive-definite kernel or None.")
    if design.train_size and design.train_spatial is None:
        raise ValueError("train_spatial is required when spatial_kernel is supplied.")
    if design.query_size and design.query_spatial is None:
        raise ValueError("query_spatial is required when spatial_kernel is supplied.")
    designs = []
    if design.train_size:
        designs.append(design.train_spatial)
    if design.query_size:
        designs.append(design.query_spatial)
    if len(designs) == 1:
        covariance = functional_kernel_matrix(spatial_kernel, designs[0], designs[0])
    else:
        train_covariance = functional_kernel_matrix(
            spatial_kernel, designs[0], designs[0]
        )
        train_query = functional_kernel_matrix(spatial_kernel, designs[0], designs[1])
        query_covariance = functional_kernel_matrix(
            spatial_kernel, designs[1], designs[1]
        )
        covariance = jnp.block(
            ((train_covariance, train_query), (train_query.T, query_covariance))
        )
    host = np.asarray(jax.device_get(0.5 * (covariance + covariance.T)))
    eigenvalues, eigenvectors = np.linalg.eigh(host)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = 256.0 * np.finfo(host.dtype).eps * max(1, total) * scale
    if np.any(~np.isfinite(eigenvalues)) or float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("Spatial functional covariance is not positive semidefinite.")
    active = eigenvalues > tolerance
    if not np.any(active):
        raise ValueError("Spatial functional covariance has zero numerical rank.")
    factor = eigenvectors[:, active] * np.sqrt(eigenvalues[active])[None, :]
    return jnp.asarray(factor, dtype=covariance.dtype)


def compile_state_space_kernel(
    temporal_kernel: AbstractPositiveDefiniteKernel,
    design: StateSpaceGaussianProcessDesign,
    /,
    *,
    spatial_kernel: AbstractPositiveDefiniteKernel | None = None,
    max_state_dimension: int = 512,
    max_observations_per_time: int = 8,
    max_schedule_size: int = 1_000_000,
    precision: PrecisionRequest | None = None,
) -> StateSpaceGaussianProcessPlan:
    """Prepare a bounded Markov/separable GP epoch outside transformed execution."""
    if not isinstance(temporal_kernel, AbstractPositiveDefiniteKernel):
        raise TypeError("temporal_kernel must be a positive-definite kernel.")
    if not isinstance(design, StateSpaceGaussianProcessDesign):
        raise TypeError("design must be a StateSpaceGaussianProcessDesign.")
    if temporal_kernel.input_ndim != 1:
        raise ValueError("Temporal kernels must consume scalar time inputs.")
    if precision is not None and not isinstance(precision, PrecisionRequest):
        raise TypeError("precision must be a PrecisionRequest or None.")
    state_limit = _positive_integer(max_state_dimension, name="max_state_dimension")
    row_capacity = _positive_integer(
        max_observations_per_time, name="max_observations_per_time"
    )
    schedule_limit = _positive_integer(max_schedule_size, name="max_schedule_size")

    train_host = np.asarray(jax.device_get(design.train_times))
    query_host = np.asarray(jax.device_get(design.query_times))
    combined = np.concatenate((train_host, query_host))
    schedule_host, combined_inverse = np.unique(combined, return_inverse=True)
    if schedule_host.size > schedule_limit:
        raise ValueError(
            f"Unique schedule size {schedule_host.size} exceeds max_schedule_size={schedule_limit}."
        )
    train_schedule = combined_inverse[: design.train_size]
    query_schedule = combined_inverse[design.train_size :]
    row_counts = np.zeros((schedule_host.size,), dtype=np.int32)
    train_gather = np.zeros((design.train_size, 2), dtype=np.int32)
    for index, schedule_index in enumerate(train_schedule):
        row = int(row_counts[schedule_index])
        if row >= row_capacity:
            raise ValueError(
                "Observations sharing a time exceed max_observations_per_time."
            )
        train_gather[index] = (schedule_index, row)
        row_counts[schedule_index] += 1
    schedule_mask = np.zeros((schedule_host.size, row_capacity), dtype=bool)
    mask_host = np.asarray(jax.device_get(design.train_mask))
    for index, (schedule_index, row) in enumerate(train_gather):
        schedule_mask[schedule_index, row] = bool(mask_host[index])

    temporal = _kernel_state_space(temporal_kernel)
    supported_order = temporal_kernel.max_derivative_order
    requested_order = max(
        _host_max(design.train_time_derivative_order),
        _host_max(design.query_time_derivative_order),
    )
    if supported_order is not None and requested_order > supported_order:
        raise ValueError(
            f"{temporal_kernel.kernel_id} certifies time derivative order {supported_order}, "
            f"but the design requires {requested_order}."
        )
    spatial_factor = _spatial_factor(design, spatial_kernel)
    rank = int(spatial_factor.shape[1])
    temporal_size = int(temporal.drift.shape[0])
    state_size = rank * temporal_size
    if state_size > state_limit:
        raise ValueError(
            f"Expanded state dimension {state_size} exceeds max_state_dimension={state_limit}."
        )
    schedule = jnp.asarray(schedule_host, dtype=design.train_times.dtype)
    if schedule.dtype != temporal.drift.dtype:
        raise TypeError(
            "Kernel coefficients and schedule times must use one compute dtype."
        )
    train_gather_array = jnp.asarray(train_gather, dtype=jnp.int32)
    query_schedule_array = jnp.asarray(query_schedule, dtype=jnp.int32)
    expanded = _expanded_state_space(
        temporal,
        spatial_factor,
        design.train_time_derivative_order,
        design.query_time_derivative_order,
        train_gather_array,
        int(schedule_host.size),
        row_capacity,
    )
    process_noise = expanded.process_factor @ expanded.process_factor.T
    lyapunov = (
        expanded.drift @ expanded.stationary
        + expanded.stationary @ expanded.drift.T
        + process_noise
    )
    residual_host = np.asarray(jax.device_get(lyapunov))
    scale_host = max(
        1.0,
        float(np.max(np.abs(np.asarray(jax.device_get(process_noise))))),
        float(np.max(np.abs(np.asarray(jax.device_get(expanded.stationary))))),
    )
    tolerance = 512.0 * np.finfo(residual_host.dtype).eps * scale_host * state_size
    if (
        not np.all(np.isfinite(residual_host))
        or np.max(np.abs(residual_host)) > tolerance
    ):
        raise ValueError("Prepared coefficients fail the stationary Lyapunov check.")

    inference_times = schedule - schedule[0]
    initial_time = jnp.zeros((), dtype=schedule.dtype)
    schedule_mask_array = jnp.asarray(schedule_mask, dtype=bool)
    prior = GaussianStatePrior(
        jnp.zeros((state_size,), dtype=schedule.dtype),
        expanded.stationary,
        state_shape=(state_size,),
        prior_id="state-space-gp-stationary-prior",
    )
    transition = LinearGaussianTransitionKernel(
        LinearGaussianParameterization(
            _transition_matrix,
            _transition_covariance,
            state_shape=(state_size,),
            parameterization_id="state-space-gp-stationary-lti",
            resolved_method="matrix-exponential/stationary-covariance",
        ),
        process_id="state-space-gp-rational-process",
        approximation_id="exact-finite-markov",
    )
    observation = LinearGaussianObservationModel(
        _observation_matrix,
        _observation_covariance,
        state_shape=(state_size,),
        observation_shape=(row_capacity,),
        observation_id="state-space-gp-packed-functional-observations",
    )
    kernel_content_id = _kernel_content_id(temporal_kernel)
    schedule_id = canonical_fingerprint(
        {
            "kind": "state-space-gp-schedule",
            "times": schedule_host.tolist(),
            "train_gather": train_gather.tolist(),
            "query_schedule": query_schedule.tolist(),
            "train_mask": mask_host.tolist(),
            "spatial_rank": rank,
        }
    )
    model = StateSpaceModel(
        prior,
        transition,
        observation,
        model_id="state-space-gp-model",
        metadata={
            "kernel_id": temporal_kernel.kernel_id,
            "method_id": _STATE_SPACE_GP_METHOD_ID,
        },
    )
    observations = ObservationSequence(
        inference_times,
        jnp.zeros((schedule.size, row_capacity), dtype=schedule.dtype),
        observation_axes=("row",),
        observation_mask=schedule_mask_array,
        sequence_id=f"state-space-gp-schedule:{schedule_id}",
        sensor_id="state-space-gp-functionals",
        discretization_id="exact-sorted-temporal-schedule",
        approximation_id="exact-mask",
    )
    arguments = _StateSpaceGaussianProcessArguments(
        observation_covariance=jnp.zeros(
            (schedule.size, row_capacity, row_capacity), dtype=schedule.dtype
        ),
        observation_matrices=expanded.observation_matrices,
        drift_matrix=expanded.drift,
        stationary_covariance=expanded.stationary,
        process_noise_factor=expanded.process_factor,
        characteristic_rate=expanded.characteristic_rate,
    )
    problem = StateSpaceProblem(
        model,
        observations,
        initial_time=initial_time,
        problem_id=f"state-space-gp-problem:{schedule_id}",
        args=arguments,
    )
    problem = eqx.tree_at(
        lambda node: node.model.prior.factor, problem, expanded.stationary_factor
    )
    return StateSpaceGaussianProcessPlan(
        temporal_kernel=temporal_kernel,
        spatial_kernel=spatial_kernel,
        design=design,
        initial_time=initial_time,
        schedule_times=schedule,
        inference_times=inference_times,
        train_gather_indices=train_gather_array,
        query_schedule_indices=query_schedule_array,
        train_mask=design.train_mask,
        schedule_observation_mask=schedule_mask_array,
        observation_matrices=expanded.observation_matrices,
        query_rows=expanded.query_rows,
        spatial_factor=spatial_factor,
        drift_matrix=expanded.drift,
        stationary_covariance=expanded.stationary,
        stationary_factor=expanded.stationary_factor,
        process_noise=process_noise,
        process_noise_factor=expanded.process_factor,
        lyapunov_residual=lyapunov,
        _problem_template=problem,
        compute_dtype=schedule.dtype.name,
        temporal_state_dimension=temporal_size,
        spatial_rank=rank,
        state_dimension=state_size,
        max_observations_per_time=row_capacity,
        kernel_id=temporal_kernel.kernel_id,
        kernel_content_id=kernel_content_id,
        schedule_id=schedule_id,
        method_id=_STATE_SPACE_GP_METHOD_ID,
        repeated_time_policy=_REPEATED_TIME_POLICY,
        max_schedule_size=schedule_limit,
        max_state_dimension=state_limit,
    )


def fit_state_space_gaussian_process(
    plan: StateSpaceGaussianProcessPlan,
    train_values: ArrayLike,
    /,
    *,
    noise_scale: ArrayLike = 0.0,
    temporal_method: str = "sequential",
    covariance_form: str = "square_root",
) -> StateSpaceGaussianProcessResult:
    """Execute exact Gaussian filtering/smoothing for one prepared finite epoch."""
    if not isinstance(plan, StateSpaceGaussianProcessPlan):
        raise TypeError("plan must be a StateSpaceGaussianProcessPlan.")
    if temporal_method not in ("sequential", "parallel", "auto"):
        raise ValueError("temporal_method must be 'sequential', 'parallel', or 'auto'.")
    if covariance_form not in ("square_root", "covariance"):
        raise ValueError("covariance_form must be 'square_root' or 'covariance'.")
    if temporal_method == "parallel" and covariance_form == "square_root":
        raise ValueError("Parallel square-root filtering is not implemented.")
    values = jnp.asarray(train_values, dtype=plan.schedule_times.dtype)
    if values.shape != (plan.train_size,):
        raise ValueError("train_values must have shape (plan.train_size,).")
    values = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)),
        "train_values must be finite; represent missing values with train_mask.",
    )
    noise = _noise_vector(noise_scale, size=plan.train_size, dtype=values.dtype)
    temporal = _kernel_state_space(plan.temporal_kernel)
    if plan.drift_matrix.dtype.name != plan.compute_dtype:
        raise TypeError("The evaluated kernel dtype must match the plan compute dtype.")
    schedule_values = jnp.zeros(
        (plan.schedule_size, plan.max_observations_per_time), dtype=values.dtype
    )
    observation_covariance = jnp.zeros(
        (
            plan.schedule_size,
            plan.max_observations_per_time,
            plan.max_observations_per_time,
        ),
        dtype=values.dtype,
    )
    for index in range(plan.train_size):
        time_index = plan.train_gather_indices[index, 0]
        row_index = plan.train_gather_indices[index, 1]
        schedule_values = schedule_values.at[time_index, row_index].set(values[index])
        observation_covariance = observation_covariance.at[
            time_index, row_index, row_index
        ].set(noise[index] * noise[index])
    arguments = _StateSpaceGaussianProcessArguments(
        observation_covariance=observation_covariance,
        observation_matrices=plan.observation_matrices,
        drift_matrix=plan.drift_matrix,
        stationary_covariance=plan.stationary_covariance,
        process_noise_factor=plan.process_noise_factor,
        characteristic_rate=temporal.characteristic_rate,
    )
    problem = eqx.tree_at(
        lambda node: (
            node.model.prior.covariance,
            node.model.prior.factor,
            node.observations.values,
            node.args,
        ),
        plan._problem_template,
        (plan.stationary_covariance, plan.stationary_factor, schedule_values, arguments),
    )
    filtered = kalman_filter(
        problem,
        method=temporal_method,
        covariance_form=covariance_form,
        covariance_regularization=0.0,
        raise_on_failure=False,
    )
    smoothed = rts_smoother(
        filtered, method=temporal_method, covariance_form=covariance_form
    )
    query_states = smoothed.means[plan.query_schedule_indices]
    query_covariances = smoothed.covariances[plan.query_schedule_indices]
    posterior_mean = ein.contract("qi,qi->q", query_states, plan.query_rows)
    posterior_variance = ein.contract(
        "qi,qij,qj->q", plan.query_rows, query_covariances, plan.query_rows
    )
    query_valid = smoothed.valid[plan.query_schedule_indices]
    filter_success = jnp.all(filtered.successful)
    valid = filter_success & jnp.all(query_valid)
    status = _state_space_gp_status(filter_success, filtered.status, query_valid)
    return StateSpaceGaussianProcessResult(
        posterior_times=plan.design.query_times,
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
        predictive_mean=posterior_mean,
        predictive_variance=posterior_variance,
        log_marginal_likelihood=filtered.final_state.log_likelihood.reshape(()),
        active_observation_count=jnp.sum(plan.train_mask, dtype=jnp.int32),
        valid=valid,
        status=status,
        query_valid=query_valid,
        train_mask=plan.train_mask,
        schedule_times=plan.schedule_times,
        schedule_observation_mask=plan.schedule_observation_mask,
        evaluated_length_scale=1.0 / jnp.atleast_1d(temporal.characteristic_rate),
        evaluated_scale=temporal.component_scales,
        filter_result=filtered,
        smoother_result=smoothed,
        precision_evidence=_precision_evidence(
            posterior_mean.dtype, filtered.execution_method
        ),
        state_dimension=plan.state_dimension,
        spatial_rank=plan.spatial_rank,
        kernel_id=plan.kernel_id,
        kernel_content_id=_evaluated_kernel_content_id(plan.temporal_kernel),
        prepared_kernel_content_id=plan.kernel_content_id,
        schedule_id=plan.schedule_id,
        method_id=plan.method_id,
        repeated_time_policy=plan.repeated_time_policy,
        temporal_method=filtered.execution_method,
        covariance_form=filtered.covariance_form,
    )


def _training_marginals(
    plan: StateSpaceGaussianProcessPlan,
    result: StateSpaceGaussianProcessResult,
    /,
) -> tuple[Array, Array]:
    observation_matrices = plan.observation_matrices
    means = []
    variances = []
    for index in range(plan.train_size):
        time_index = plan.train_gather_indices[index, 0]
        row_index = plan.train_gather_indices[index, 1]
        row = observation_matrices[time_index, row_index]
        means.append(row @ result.smoother_result.means[time_index])
        variances.append(row @ result.smoother_result.covariances[time_index] @ row)
    if not means:
        empty = jnp.empty((0,), dtype=plan.schedule_times.dtype)
        return empty, empty
    return jnp.stack(tuple(means)), jnp.stack(tuple(variances))


def _precision_evidence(dtype: Any, method: str, /) -> PrecisionEvidenceEnvelope:
    name = precision_dtype_name(dtype)
    request = PrecisionRequest(
        "state-space-gaussian-process",
        {"storage": name, "compute": name, "factorization": name, "output": name},
    )
    resolution = PrecisionResolution(
        request, f"phydrax-{method}-kalman-rts", dict(request.requested)
    )
    return PrecisionEvidenceEnvelope(resolution, dict(resolution.effective))


def _kernel_content_id(kernel: AbstractPositiveDefiniteKernel, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "state-space-gp-kernel",
            "kernel_id": kernel.kernel_id,
            "content": array_tree_fingerprint(kernel),
        }
    )


def _evaluated_kernel_content_id(kernel: AbstractPositiveDefiniteKernel, /) -> str | None:
    leaves = tuple(leaf for leaf in jax.tree.leaves(kernel) if eqx.is_array(leaf))
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
        return None
    return _kernel_content_id(kernel)


def _state_space_gp_status(
    filter_success: Array, filter_status: Array, requested_valid: Array, /
) -> Array:
    return jnp.where(
        ~filter_success,
        jnp.max(filter_status),
        jnp.where(
            jnp.all(requested_valid), KALMAN_SUCCESS, STATE_SPACE_GP_SMOOTHER_FAILURE
        ),
    ).astype(jnp.int32)


def state_space_gaussian_process_status_name(value: int, /) -> str:
    code = int(value)
    if code == STATE_SPACE_GP_SMOOTHER_FAILURE:
        return "smoother_failure"
    return kalman_status_name(code)


def _as_time_vector(value: ArrayLike, /, *, name: str) -> Array:
    times = jnp.asarray(value, dtype=float)
    if times.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional time vector.")
    host = np.asarray(jax.device_get(times))
    if not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must contain only finite times.")
    return times


def _derivative_orders(value: ArrayLike | None, /, *, size: int, name: str) -> Array:
    if value is None:
        return jnp.zeros((size,), dtype=jnp.int32)
    orders = jnp.asarray(value)
    if orders.shape != (size,) or not jnp.issubdtype(orders.dtype, jnp.integer):
        raise ValueError(f"{name} must be an integer vector of shape ({size},).")
    host = np.asarray(jax.device_get(orders))
    if np.any(host < 0):
        raise ValueError(f"{name} must contain nonnegative orders.")
    return orders.astype(jnp.int32)


def _noise_vector(value: ArrayLike, /, *, size: int, dtype: jnp.dtype) -> Array:
    noise = jnp.asarray(value, dtype=dtype)
    if noise.ndim == 0:
        noise = jnp.broadcast_to(noise, (size,))
    if noise.shape != (size,):
        raise ValueError("noise_scale must be scalar or align with train_values.")
    return eqx.error_if(
        noise,
        jnp.any(~jnp.isfinite(noise)) | jnp.any(noise < 0.0),
        "noise_scale must be finite and nonnegative.",
    )


def _positive_integer(value: int, /, *, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def _host_max(value: Array, /) -> int:
    host = np.asarray(jax.device_get(value))
    return int(np.max(host)) if host.size else 0


__all__ = [
    "STATE_SPACE_GP_SMOOTHER_FAILURE",
    "StateSpaceGaussianProcessDesign",
    "StateSpaceGaussianProcessPlan",
    "StateSpaceGaussianProcessResult",
    "compile_state_space_kernel",
    "fit_state_space_gaussian_process",
    "state_space_gaussian_process_status_name",
]
