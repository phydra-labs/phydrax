#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntFlag

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax.ein import contract

from .._admissibility import AdmissibilityHeader, AdmissibilityReason
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ..geometry._wall_frame import PlanarWallFramePlan
from ._dynamics import AtomisticDynamicsState, PreparedAtomisticDynamics
from ._observer import AbstractAtomisticObserverPlan


class NanoflowObservableReason(IntFlag):
    EMPTY_BIN = 1 << 8
    INSUFFICIENT_ORIGINS = 1 << 9
    NONSTATIONARY_FIT = 1 << 10
    DEGENERATE_SHEAR = 1 << 11


class PlanarWallProfileState(StrictModule):
    count_sum: Array
    mass_sum: Array
    charge_sum: Array
    mass_velocity_sum: Array
    mass_velocity_outer_sum: Array
    boltzmann_constant: Array
    samples: Array
    observer_id: str = eqx.field(static=True)


class PlanarWallProfileResult(StrictModule):
    bin_centers: Array
    number_density: Array
    mass_density: Array
    charge_density: Array
    tangential_velocity: Array
    tangential_temperature_tensor: Array
    counts: Array
    empty_bins: Array
    samples: Array
    header: AdmissibilityHeader
    observer_id: str = eqx.field(static=True)


class PlanarWallProfileObserverPlan(AbstractAtomisticObserverPlan, NonTrainableState):
    frame: PlanarWallFramePlan
    group_masks: Array
    group_names: tuple[str, ...] = eqx.field(static=True)
    bin_count: int = eqx.field(static=True)
    minimum_count_per_bin: int = eqx.field(static=True)
    observer_id: str = eqx.field(static=True)

    def __init__(
        self,
        frame: PlanarWallFramePlan,
        group_masks: Array,
        group_names: Sequence[str],
        /,
        *,
        bin_count: int,
        minimum_count_per_bin: int = 1,
    ) -> None:
        masks = np.asarray(group_masks, dtype=bool)
        names = tuple(str(value) for value in group_names)
        bins = int(bin_count)
        minimum = int(minimum_count_per_bin)
        if (
            not isinstance(frame, PlanarWallFramePlan)
            or frame.dimension != 3
            or masks.ndim != 2
            or masks.shape[0] != len(names)
            or not names
            or any(not value for value in names)
            or bins <= 0
            or minimum <= 0
        ):
            raise ValueError("Planar wall profile observer inputs are invalid.")
        self.frame = frame
        self.group_masks = jnp.asarray(masks)
        self.group_names = names
        self.bin_count = bins
        self.minimum_count_per_bin = minimum
        self.observer_id = canonical_fingerprint(
            {
                "kind": "planar-wall-profile-observer",
                "frame": frame.frame_id,
                "group_masks": array_tree_fingerprint(masks),
                "group_names": names,
                "bin_count": bins,
                "minimum_count_per_bin": minimum,
            }
        )

    def initialize(
        self,
        dynamics: PreparedAtomisticDynamics,
        state: AtomisticDynamicsState,
        /,
    ) -> PlanarWallProfileState:
        del state
        if self.group_masks.shape[1] != dynamics.system.capacity:
            raise ValueError("Profile group masks must match atomistic capacity.")
        dtype = dynamics.system.plan.masses.dtype
        shape = (len(self.group_names), self.bin_count)
        tangent_count = self.frame.dimension - 1
        return PlanarWallProfileState(
            jnp.zeros(shape, dtype=dtype),
            jnp.zeros(shape, dtype=dtype),
            jnp.zeros(shape, dtype=dtype),
            jnp.zeros(shape + (tangent_count,), dtype=dtype),
            jnp.zeros(shape + (tangent_count, tangent_count), dtype=dtype),
            jnp.asarray(dynamics.system.plan.units.boltzmann_constant, dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
            self.observer_id,
        )

    def update(
        self,
        observer_state: PlanarWallProfileState,
        dynamics: PreparedAtomisticDynamics,
        state: AtomisticDynamicsState,
        accepted: Array,
        /,
    ) -> PlanarWallProfileState:
        if observer_state.observer_id != self.observer_id:
            raise ValueError("Profile observer state belongs to another plan.")
        position = dynamics._unwrapped(state.kinematics, state.cell_vectors)
        coordinates = self.frame.coordinates(position)
        velocity = dynamics.velocity(state)
        tangential_velocity = contract(
            "ni,ji->nj", velocity, self.frame.tangential_basis, backend="jax"
        )
        scaled = coordinates.normal_coordinate / self.frame.gap * self.bin_count
        index = jnp.floor(scaled).astype(jnp.int32)
        valid_bin = coordinates.inside_gap & (index >= 0) & (index < self.bin_count)
        safe_index = jnp.clip(index, 0, self.bin_count - 1)
        active = dynamics.system.active_mask
        mass = dynamics.system.plan.masses.astype(position.dtype)
        charge = dynamics.system.plan.charges.astype(position.dtype)
        count_sum = observer_state.count_sum
        mass_sum = observer_state.mass_sum
        charge_sum = observer_state.charge_sum
        velocity_sum = observer_state.mass_velocity_sum
        outer_sum = observer_state.mass_velocity_outer_sum
        for group in range(len(self.group_names)):
            valid = active & self.group_masks[group] & valid_bin
            count_sum = count_sum.at[group, safe_index].add(valid.astype(position.dtype))
            mass_sum = mass_sum.at[group, safe_index].add(jnp.where(valid, mass, 0.0))
            charge_sum = charge_sum.at[group, safe_index].add(
                jnp.where(valid, charge, 0.0)
            )
            velocity_sum = velocity_sum.at[group, safe_index].add(
                jnp.where(valid[:, None], mass[:, None] * tangential_velocity, 0.0)
            )
            outer = mass[:, None, None] * contract(
                "ni,nj->nij", tangential_velocity, tangential_velocity, backend="jax"
            )
            outer_sum = outer_sum.at[group, safe_index].add(
                jnp.where(valid[:, None, None], outer, 0.0)
            )
        candidate = PlanarWallProfileState(
            count_sum,
            mass_sum,
            charge_sum,
            velocity_sum,
            outer_sum,
            observer_state.boltzmann_constant,
            observer_state.samples + jnp.asarray(1, dtype=jnp.int32),
            self.observer_id,
        )
        return tree_where(jnp.asarray(accepted, dtype=bool), candidate, observer_state)

    def finalize(
        self, observer_state: PlanarWallProfileState, /
    ) -> PlanarWallProfileResult:
        if observer_state.observer_id != self.observer_id:
            raise ValueError("Profile observer state belongs to another plan.")
        dtype = observer_state.mass_sum.dtype
        samples = jnp.maximum(observer_state.samples, 1).astype(dtype)
        width = self.frame.gap / self.bin_count
        bin_volume = self.frame.cross_section_area * width
        number_density = observer_state.count_sum / (samples * bin_volume)
        mass_density = observer_state.mass_sum / (samples * bin_volume)
        charge_density = observer_state.charge_sum / (samples * bin_volume)
        mean_velocity = observer_state.mass_velocity_sum / jnp.maximum(
            observer_state.mass_sum[..., None], jnp.finfo(dtype).tiny
        )
        centered_outer = observer_state.mass_velocity_outer_sum - (
            observer_state.mass_sum[..., None, None]
            * contract("...i,...j->...ij", mean_velocity, mean_velocity, backend="jax")
        )
        temperature = centered_outer / jnp.maximum(
            observer_state.boltzmann_constant * observer_state.count_sum[..., None, None],
            jnp.finfo(dtype).tiny,
        )
        enough = observer_state.count_sum >= self.minimum_count_per_bin
        reasons = jnp.where(
            enough,
            jnp.asarray(0, dtype=jnp.uint32),
            jnp.asarray(int(NanoflowObservableReason.EMPTY_BIN), dtype=jnp.uint32),
        )
        header = AdmissibilityHeader(
            observer_state.count_sum - self.minimum_count_per_bin,
            reasons,
            self.observer_id,
            canonical_fingerprint(
                {"kind": "planar-wall-profile-evidence", "plan": self.observer_id}
            ),
        )
        centers = (jnp.arange(self.bin_count, dtype=dtype) + 0.5) * width
        return PlanarWallProfileResult(
            centers,
            number_density,
            mass_density,
            charge_density,
            mean_velocity,
            temperature,
            observer_state.count_sum,
            ~enough,
            observer_state.samples,
            header,
            self.observer_id,
        )


class MultiOriginCorrelationState(StrictModule):
    origin_positions: Array
    origin_velocities: Array
    origin_times: Array
    origin_steps: Array
    origin_valid: Array
    next_origin: Array
    accepted_index: Array
    msd_sum: Array
    msd_outer_sum: Array
    vacf_sum: Array
    lag_time_sum: Array
    counts: Array
    observer_id: str = eqx.field(static=True)


class MultiOriginCorrelationResult(StrictModule):
    lag_time: Array
    mean_squared_displacement_tensor: Array
    mean_squared_displacement_covariance: Array
    velocity_autocorrelation_tensor: Array
    counts: Array
    header: AdmissibilityHeader
    observer_id: str = eqx.field(static=True)


class MultiOriginCorrelationObserverPlan(
    AbstractAtomisticObserverPlan, NonTrainableState
):
    origin_stride: int = eqx.field(static=True)
    origin_capacity: int = eqx.field(static=True)
    lag_count: int = eqx.field(static=True)
    minimum_origins: int = eqx.field(static=True)
    observer_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        origin_stride: int,
        origin_capacity: int,
        lag_count: int,
        minimum_origins: int = 2,
    ) -> None:
        stride = int(origin_stride)
        capacity = int(origin_capacity)
        lags = int(lag_count)
        minimum = int(minimum_origins)
        required = int(np.ceil(lags / max(stride, 1))) + 1
        if stride <= 0 or lags <= 0 or capacity < required or minimum < 2:
            raise ValueError(
                "Correlation origin capacity must cover every live lag window."
            )
        self.origin_stride = stride
        self.origin_capacity = capacity
        self.lag_count = lags
        self.minimum_origins = minimum
        self.observer_id = canonical_fingerprint(
            {
                "kind": "multi-origin-correlation-observer",
                "origin_stride": stride,
                "origin_capacity": capacity,
                "lag_count": lags,
                "minimum_origins": minimum,
            }
        )

    def initialize(
        self,
        dynamics: PreparedAtomisticDynamics,
        state: AtomisticDynamicsState,
        /,
    ) -> MultiOriginCorrelationState:
        position = dynamics._unwrapped(state.kinematics, state.cell_vectors)
        velocity = dynamics.velocity(state)
        dtype = position.dtype
        origins = jnp.zeros((self.origin_capacity,) + position.shape, dtype=dtype)
        origin_velocity = jnp.zeros_like(origins)
        origin_times = jnp.zeros((self.origin_capacity,), dtype=dtype)
        origin_steps = -jnp.ones((self.origin_capacity,), dtype=jnp.int32)
        origin_valid = jnp.zeros((self.origin_capacity,), dtype=bool)
        origins = origins.at[0].set(position)
        origin_velocity = origin_velocity.at[0].set(velocity)
        origin_times = origin_times.at[0].set(state.time)
        origin_steps = origin_steps.at[0].set(0)
        origin_valid = origin_valid.at[0].set(True)
        tensor_shape = (self.lag_count, 3, 3)
        return MultiOriginCorrelationState(
            origins,
            origin_velocity,
            origin_times,
            origin_steps,
            origin_valid,
            jnp.asarray(1 % self.origin_capacity, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.zeros(tensor_shape, dtype=dtype),
            jnp.zeros((self.lag_count, 9, 9), dtype=dtype),
            jnp.zeros(tensor_shape, dtype=dtype),
            jnp.zeros((self.lag_count,), dtype=dtype),
            jnp.zeros((self.lag_count,), dtype=jnp.int32),
            self.observer_id,
        )

    def update(
        self,
        observer_state: MultiOriginCorrelationState,
        dynamics: PreparedAtomisticDynamics,
        state: AtomisticDynamicsState,
        accepted: Array,
        /,
    ) -> MultiOriginCorrelationState:
        if observer_state.observer_id != self.observer_id:
            raise ValueError("Correlation observer state belongs to another plan.")
        position = dynamics._unwrapped(state.kinematics, state.cell_vectors)
        velocity = dynamics.velocity(state)
        active = dynamics.system.active_mask
        active_count = jnp.maximum(jnp.sum(active), 1)
        step = observer_state.accepted_index + 1
        lag = step - observer_state.origin_steps
        valid_origin = observer_state.origin_valid & (lag >= 1) & (lag <= self.lag_count)
        safe_lag = jnp.clip(lag - 1, 0, self.lag_count - 1)
        displacement = position[None, ...] - observer_state.origin_positions
        displacement_tensor = (
            contract(
                "oni,onj->oij",
                jnp.where(active[None, :, None], displacement, 0.0),
                jnp.where(active[None, :, None], displacement, 0.0),
                backend="jax",
            )
            / active_count
        )
        vacf_tensor = (
            contract(
                "oni,onj->oij",
                jnp.where(active[None, :, None], observer_state.origin_velocities, 0.0),
                jnp.where(active[None, :, None], velocity[None, ...], 0.0),
                backend="jax",
            )
            / active_count
        )
        flattened = displacement_tensor.reshape((self.origin_capacity, 9))
        msd_outer = contract("oi,oj->oij", flattened, flattened, backend="jax")
        msd_sum = observer_state.msd_sum.at[safe_lag].add(
            jnp.where(valid_origin[:, None, None], displacement_tensor, 0.0)
        )
        msd_outer_sum = observer_state.msd_outer_sum.at[safe_lag].add(
            jnp.where(valid_origin[:, None, None], msd_outer, 0.0)
        )
        vacf_sum = observer_state.vacf_sum.at[safe_lag].add(
            jnp.where(valid_origin[:, None, None], vacf_tensor, 0.0)
        )
        lag_time_sum = observer_state.lag_time_sum.at[safe_lag].add(
            jnp.where(valid_origin, state.time - observer_state.origin_times, 0.0)
        )
        counts = observer_state.counts.at[safe_lag].add(valid_origin.astype(jnp.int32))
        due = jnp.mod(step, self.origin_stride) == 0
        slot = observer_state.next_origin
        origin_positions = jax.lax.cond(
            due,
            lambda value: value.at[slot].set(position),
            lambda value: value,
            observer_state.origin_positions,
        )
        origin_velocities = jax.lax.cond(
            due,
            lambda value: value.at[slot].set(velocity),
            lambda value: value,
            observer_state.origin_velocities,
        )
        origin_times = jax.lax.cond(
            due,
            lambda value: value.at[slot].set(state.time),
            lambda value: value,
            observer_state.origin_times,
        )
        origin_steps = jax.lax.cond(
            due,
            lambda value: value.at[slot].set(step),
            lambda value: value,
            observer_state.origin_steps,
        )
        origin_valid = jax.lax.cond(
            due,
            lambda value: value.at[slot].set(True),
            lambda value: value,
            observer_state.origin_valid,
        )
        next_origin = jnp.where(
            due,
            jnp.mod(slot + 1, self.origin_capacity),
            slot,
        )
        candidate = MultiOriginCorrelationState(
            origin_positions,
            origin_velocities,
            origin_times,
            origin_steps,
            origin_valid,
            next_origin,
            step,
            msd_sum,
            msd_outer_sum,
            vacf_sum,
            lag_time_sum,
            counts,
            self.observer_id,
        )
        return tree_where(jnp.asarray(accepted, dtype=bool), candidate, observer_state)

    def finalize(
        self, observer_state: MultiOriginCorrelationState, /
    ) -> MultiOriginCorrelationResult:
        if observer_state.observer_id != self.observer_id:
            raise ValueError("Correlation observer state belongs to another plan.")
        dtype = observer_state.msd_sum.dtype
        count = jnp.maximum(observer_state.counts, 1).astype(dtype)
        msd = observer_state.msd_sum / count[:, None, None]
        vacf = observer_state.vacf_sum / count[:, None, None]
        flat_mean = msd.reshape((self.lag_count, 9))
        covariance = (
            observer_state.msd_outer_sum
            - count[:, None, None]
            * contract("li,lj->lij", flat_mean, flat_mean, backend="jax")
        ) / jnp.maximum(count - 1.0, 1.0)[:, None, None]
        lag_time = observer_state.lag_time_sum / count
        enough = observer_state.counts >= self.minimum_origins
        reasons = jnp.where(
            enough,
            jnp.asarray(0, dtype=jnp.uint32),
            jnp.asarray(
                int(NanoflowObservableReason.INSUFFICIENT_ORIGINS), dtype=jnp.uint32
            ),
        )
        header = AdmissibilityHeader(
            (observer_state.counts - self.minimum_origins).astype(dtype),
            reasons,
            self.observer_id,
            canonical_fingerprint(
                {"kind": "multi-origin-correlation-evidence", "plan": self.observer_id}
            ),
        )
        return MultiOriginCorrelationResult(
            lag_time,
            msd,
            covariance,
            vacf,
            observer_state.counts,
            header,
            self.observer_id,
        )


class DiffusionTensorFitResult(StrictModule):
    diffusion_tensor: Array
    covariance: Array
    intercept: Array
    residual_norm: Array
    stationarity_defect: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class DiffusionTensorFitPlan(StrictModule, NonTrainableState):
    fit_start: int = eqx.field(static=True)
    fit_stop: int = eqx.field(static=True)
    minimum_origins: int = eqx.field(static=True)
    stationarity_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fit_start: int,
        fit_stop: int,
        /,
        *,
        minimum_origins: int = 4,
        stationarity_tolerance: float = 0.25,
    ) -> None:
        start = int(fit_start)
        stop = int(fit_stop)
        minimum = int(minimum_origins)
        tolerance = float(stationarity_tolerance)
        if start < 0 or stop - start < 3 or minimum < 2 or tolerance <= 0.0:
            raise ValueError("Diffusion fit window or evidence controls are invalid.")
        self.fit_start = start
        self.fit_stop = stop
        self.minimum_origins = minimum
        self.stationarity_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "diffusion-tensor-fit",
                "window": (start, stop),
                "minimum_origins": minimum,
                "stationarity_tolerance": tolerance,
            }
        )

    @staticmethod
    def _linear_weights(time: Array, /) -> tuple[Array, Array]:
        mean = jnp.mean(time)
        centered = time - mean
        denominator = jnp.sum(centered**2)
        slope_weights = centered / denominator
        intercept_weights = jnp.ones_like(time) / time.size - mean * slope_weights
        return slope_weights, intercept_weights

    def evaluate(
        self, correlation: MultiOriginCorrelationResult, /
    ) -> DiffusionTensorFitResult:
        if not isinstance(correlation, MultiOriginCorrelationResult):
            raise TypeError("correlation must be MultiOriginCorrelationResult.")
        if self.fit_stop > correlation.lag_time.shape[0]:
            raise ValueError("Diffusion fit window exceeds available lags.")
        section = slice(self.fit_start, self.fit_stop)
        time = correlation.lag_time[section]
        msd = correlation.mean_squared_displacement_tensor[section]
        weights, intercept_weights = self._linear_weights(time)
        slope = contract("l,lij->ij", weights, msd, backend="jax")
        intercept = contract("l,lij->ij", intercept_weights, msd, backend="jax")
        diffusion = 0.5 * slope
        fitted = intercept + time[:, None, None] * slope
        residual = jnp.sqrt(jnp.mean((msd - fitted) ** 2))
        covariance = 0.25 * jnp.sum(
            weights[:, None, None] ** 2
            * correlation.mean_squared_displacement_covariance[section],
            axis=0,
        )
        midpoint = time.shape[0] // 2
        first_weights, _ = self._linear_weights(time[: midpoint + 1])
        second_weights, _ = self._linear_weights(time[midpoint:])
        first_slope = contract(
            "l,lij->ij", first_weights, msd[: midpoint + 1], backend="jax"
        )
        second_slope = contract(
            "l,lij->ij", second_weights, msd[midpoint:], backend="jax"
        )
        stationarity = jnp.max(jnp.abs(first_slope - second_slope)) / jnp.maximum(
            jnp.max(jnp.abs(slope)), jnp.finfo(msd.dtype).tiny
        )
        enough = jnp.all(correlation.counts[section] >= self.minimum_origins)
        stationary = stationarity <= self.stationarity_tolerance
        finite = (
            jnp.all(jnp.isfinite(diffusion))
            & jnp.all(jnp.isfinite(covariance))
            & jnp.isfinite(residual)
        )
        supported = finite & enough & stationary & jnp.all(jnp.diag(diffusion) >= 0.0)
        reasons = jnp.asarray(0, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            enough,
            reasons,
            reasons
            | jnp.asarray(
                int(NanoflowObservableReason.INSUFFICIENT_ORIGINS), dtype=jnp.uint32
            ),
        )
        reasons = jnp.where(
            stationary,
            reasons,
            reasons
            | jnp.asarray(
                int(NanoflowObservableReason.NONSTATIONARY_FIT), dtype=jnp.uint32
            ),
        )
        header = AdmissibilityHeader(
            jnp.where(
                supported,
                self.stationarity_tolerance - stationarity,
                -1.0,
            ),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "diffusion-fit-evidence", "plan": self.plan_id}
            ),
        )
        return DiffusionTensorFitResult(
            diffusion,
            covariance,
            intercept,
            residual,
            stationarity,
            header,
            self.plan_id,
        )


class DrivenSlipFitResult(StrictModule):
    slip_lengths: Array
    covariance: Array
    velocity_gradient: Array
    intercept: Array
    residual_norm: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class DrivenSlipFitPlan(StrictModule, NonTrainableState):
    """Fit bulk linear shear and extrapolate slip at both exact wall planes."""

    group_index: int = eqx.field(static=True)
    tangential_component: int = eqx.field(static=True)
    fit_start: int = eqx.field(static=True)
    fit_stop: int = eqx.field(static=True)
    minimum_gradient: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        group_index: int,
        tangential_component: int,
        fit_start: int,
        fit_stop: int,
        /,
        *,
        minimum_gradient: float = 1.0e-12,
    ) -> None:
        group = int(group_index)
        component = int(tangential_component)
        start = int(fit_start)
        stop = int(fit_stop)
        minimum = float(minimum_gradient)
        if (
            group < 0
            or component < 0
            or start < 0
            or stop - start < 3
            or not np.isfinite(minimum)
            or minimum <= 0.0
        ):
            raise ValueError("Driven slip fit indices or gradient floor are invalid.")
        self.group_index = group
        self.tangential_component = component
        self.fit_start = start
        self.fit_stop = stop
        self.minimum_gradient = minimum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "driven-slip-fit",
                "group_index": group,
                "tangential_component": component,
                "fit_window": (start, stop),
                "minimum_gradient": minimum,
            }
        )

    def evaluate(
        self,
        profile: PlanarWallProfileResult,
        lower_wall_velocity: Array,
        upper_wall_velocity: Array,
        gap: Array,
        /,
    ) -> DrivenSlipFitResult:
        if not isinstance(profile, PlanarWallProfileResult):
            raise TypeError("profile must be PlanarWallProfileResult.")
        if (
            self.group_index >= profile.tangential_velocity.shape[0]
            or self.tangential_component >= profile.tangential_velocity.shape[-1]
            or self.fit_stop > profile.bin_centers.size
        ):
            raise ValueError("Driven slip fit indices exceed the profile layout.")
        section = slice(self.fit_start, self.fit_stop)
        coordinate = profile.bin_centers[section]
        velocity = profile.tangential_velocity[
            self.group_index, section, self.tangential_component
        ]
        count = coordinate.size
        sum_x = jnp.sum(coordinate)
        sum_xx = jnp.sum(coordinate**2)
        determinant = count * sum_xx - sum_x**2
        slope = (
            count * jnp.sum(coordinate * velocity) - sum_x * jnp.sum(velocity)
        ) / determinant
        intercept = (jnp.sum(velocity) - slope * sum_x) / count
        fitted = intercept + slope * coordinate
        residual = velocity - fitted
        variance = jnp.sum(residual**2) / (count - 2)
        intercept_variance = variance * sum_xx / determinant
        slope_variance = variance * count / determinant
        intercept_slope_covariance = -variance * sum_x / determinant
        lower_velocity = jnp.asarray(lower_wall_velocity, dtype=velocity.dtype)
        upper_velocity = jnp.asarray(upper_wall_velocity, dtype=velocity.dtype)
        gap_ = jnp.asarray(gap, dtype=velocity.dtype)
        lower_numerator = intercept - lower_velocity
        upper_numerator = upper_velocity - (intercept + slope * gap_)
        lower_slip = lower_numerator / slope
        upper_slip = upper_numerator / slope
        lower_gradient = jnp.asarray((1.0 / slope, -lower_numerator / slope**2))
        upper_gradient = jnp.asarray(
            (-1.0 / slope, -gap_ / slope - upper_numerator / slope**2)
        )
        parameter_covariance = jnp.asarray(
            (
                (intercept_variance, intercept_slope_covariance),
                (intercept_slope_covariance, slope_variance),
            )
        )
        jacobian = jnp.stack((lower_gradient, upper_gradient))
        slip_covariance = contract(
            "ia,ab,jb->ij",
            jacobian,
            parameter_covariance,
            jacobian,
            backend="jax",
        )
        bins_admitted = jnp.all(profile.header.eligible[self.group_index, section])
        finite = (
            jnp.all(jnp.isfinite(jnp.asarray((lower_slip, upper_slip))))
            & jnp.all(jnp.isfinite(slip_covariance))
            & jnp.isfinite(variance)
        )
        gradient_ok = jnp.abs(slope) >= self.minimum_gradient
        supported = finite & bins_admitted & gradient_ok & (gap_ > 0.0)
        reasons = jnp.asarray(0, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            bins_admitted,
            reasons,
            reasons
            | jnp.asarray(int(AdmissibilityReason.UNCERTAINTY_UNRESOLVED), jnp.uint32),
        )
        reasons = jnp.where(
            gradient_ok,
            reasons,
            reasons
            | jnp.asarray(
                int(NanoflowObservableReason.DEGENERATE_SHEAR), dtype=jnp.uint32
            ),
        )
        header = AdmissibilityHeader(
            jnp.where(supported, jnp.abs(slope) - self.minimum_gradient, -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "driven-slip-fit-evidence", "plan": self.plan_id}
            ),
        )
        return DrivenSlipFitResult(
            jnp.asarray((lower_slip, upper_slip)),
            slip_covariance,
            slope,
            intercept,
            jnp.sqrt(jnp.mean(residual**2)),
            header,
            self.plan_id,
        )


class WallForceCorrelationResult(StrictModule):
    lag_time: Array
    autocorrelation: Array
    friction_coefficient: Array
    covariance: Array
    pair_counts: Array
    header: AdmissibilityHeader
    force_source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class WallForceCorrelationPlan(StrictModule, NonTrainableState):
    """Green-Kubo wall friction from an explicitly exact tangential wall force."""

    area: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    boltzmann_constant: float = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    lag_count: int = eqx.field(static=True)
    minimum_pairs: int = eqx.field(static=True)
    force_source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        area: float,
        temperature: float,
        boltzmann_constant: float,
        time_step: float,
        lag_count: int,
        minimum_pairs: int,
        force_source_id: str,
    ) -> None:
        values = tuple(
            float(value) for value in (area, temperature, boltzmann_constant, time_step)
        )
        lags = int(lag_count)
        minimum = int(minimum_pairs)
        source = str(force_source_id)
        if (
            any(not np.isfinite(value) or value <= 0.0 for value in values)
            or lags <= 1
            or minimum < 2
            or not source
        ):
            raise ValueError("Wall-force correlation controls are invalid.")
        self.area, self.temperature, self.boltzmann_constant, self.time_step = values
        self.lag_count = lags
        self.minimum_pairs = minimum
        self.force_source_id = source
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wall-force-correlation",
                "physical": values,
                "lag_count": lags,
                "minimum_pairs": minimum,
                "force_source": source,
            }
        )

    def evaluate(self, tangential_wall_force: Array, /) -> WallForceCorrelationResult:
        force = jnp.asarray(tangential_wall_force)
        if force.ndim != 2 or force.shape[1] not in (1, 2):
            raise ValueError(
                "Tangential wall force must have sample/tangential-component shape."
            )
        if force.shape[0] <= self.lag_count:
            raise ValueError("Wall-force series must exceed the requested lag count.")
        fluctuation = force - jnp.mean(force, axis=0)
        correlations = []
        variances = []
        counts = []
        for lag in range(self.lag_count):
            product = (
                jnp.sum(
                    fluctuation[: force.shape[0] - lag] * fluctuation[lag:],
                    axis=-1,
                )
                / force.shape[1]
            )
            correlations.append(jnp.mean(product))
            variances.append(jnp.var(product, ddof=1) / product.size)
            counts.append(product.size)
        correlation = jnp.stack(tuple(correlations))
        correlation_variance = jnp.stack(tuple(variances))
        pair_counts = jnp.asarray(counts, dtype=jnp.int32)
        integration_weights = jnp.ones((self.lag_count,), dtype=force.dtype)
        integration_weights = integration_weights.at[0].set(0.5)
        integration_weights = integration_weights.at[-1].set(0.5)
        scale = self.time_step / (self.area * self.boltzmann_constant * self.temperature)
        friction = scale * jnp.sum(integration_weights * correlation)
        variance = scale**2 * jnp.sum(integration_weights**2 * correlation_variance)
        finite = (
            jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(correlation))
            & jnp.isfinite(friction)
            & jnp.isfinite(variance)
        )
        enough = jnp.all(pair_counts >= self.minimum_pairs)
        supported = finite & enough & (friction >= 0.0) & (variance >= 0.0)
        reasons = jnp.asarray(0, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            enough,
            reasons,
            reasons
            | jnp.asarray(int(AdmissibilityReason.UNCERTAINTY_UNRESOLVED), jnp.uint32),
        )
        header = AdmissibilityHeader(
            jnp.where(supported, friction, -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "wall-force-correlation-evidence", "plan": self.plan_id}
            ),
        )
        return WallForceCorrelationResult(
            jnp.arange(self.lag_count, dtype=force.dtype) * self.time_step,
            correlation,
            friction,
            jnp.asarray(((variance,),)),
            pair_counts,
            header,
            self.force_source_id,
            self.plan_id,
        )


__all__ = [
    "DiffusionTensorFitPlan",
    "DiffusionTensorFitResult",
    "DrivenSlipFitPlan",
    "DrivenSlipFitResult",
    "MultiOriginCorrelationObserverPlan",
    "MultiOriginCorrelationResult",
    "MultiOriginCorrelationState",
    "NanoflowObservableReason",
    "PlanarWallProfileObserverPlan",
    "PlanarWallProfileResult",
    "PlanarWallProfileState",
    "WallForceCorrelationPlan",
    "WallForceCorrelationResult",
]
