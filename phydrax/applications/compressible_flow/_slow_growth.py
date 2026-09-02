#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._contracts import CompressibleFlowCaseSpec


WallThermalMode = Literal["adiabatic", "isothermal"]
SlowGrowthCoordinate = Literal["temporal", "modeled-spatial"]


def _sum_axes(value: Array, axes: tuple[int, ...], /) -> Array:
    result = value
    for axis in sorted(axes, reverse=True):
        result = jnp.sum(result, axis=axis)
    return result


def _weighted_mean(value: Array, weights: Array, axes: tuple[int, ...], /) -> Array:
    if not axes:
        return value
    extra = value.ndim - weights.ndim
    numerator = _sum_axes(value * weights.reshape(weights.shape + (1,) * extra), axes)
    denominator = _sum_axes(weights, axes)
    return numerator / denominator.reshape(denominator.shape + (1,) * extra)


def _profile_gradient(value: Array, coordinates: Array, /) -> Array:
    spacing = jnp.diff(coordinates)
    slopes = jnp.diff(value, axis=0) / spacing.reshape(
        spacing.shape + (1,) * (value.ndim - 1)
    )
    if value.shape[0] == 2:
        return jnp.stack((slopes[0], slopes[0]), axis=0)
    left_spacing = spacing[:-1].reshape(spacing[:-1].shape + (1,) * (value.ndim - 1))
    right_spacing = spacing[1:].reshape(spacing[1:].shape + (1,) * (value.ndim - 1))
    interior = (right_spacing * slopes[:-1] + left_spacing * slopes[1:]) / (
        left_spacing + right_spacing
    )
    return jnp.concatenate((slopes[:1], interior, slopes[-1:]), axis=0)


def _trapezoid(value: Array, coordinates: Array, /) -> Array:
    spacing = jnp.diff(coordinates)
    extra = value.ndim - 1
    return jnp.sum(
        0.5 * (value[1:] + value[:-1]) * spacing.reshape(spacing.shape + (1,) * extra),
        axis=0,
    )


def _integral_thicknesses(primitive: Array, coordinates: Array, /) -> tuple[Array, Array]:
    density = primitive[:, 0]
    streamwise_velocity = primitive[:, 1]
    edge_mass_flux = density[-1] * streamwise_velocity[-1]
    valid = jnp.abs(edge_mass_flux) > 64.0 * jnp.finfo(primitive.dtype).eps
    safe_edge_mass_flux = jnp.where(valid, edge_mass_flux, jnp.ones_like(edge_mass_flux))
    mass_flux_ratio = density * streamwise_velocity / safe_edge_mass_flux
    edge_velocity = jnp.where(
        jnp.abs(streamwise_velocity[-1]) > 64.0 * jnp.finfo(primitive.dtype).eps,
        streamwise_velocity[-1],
        jnp.ones_like(streamwise_velocity[-1]),
    )
    displacement = _trapezoid(1.0 - mass_flux_ratio, coordinates)
    momentum = _trapezoid(
        mass_flux_ratio * (1.0 - streamwise_velocity / edge_velocity), coordinates
    )
    zero = jnp.zeros_like(displacement)
    return jnp.where(valid, displacement, zero), jnp.where(valid, momentum, zero)


def _integral_thickness_rates(
    primitive: Array,
    primitive_source: Array,
    coordinates: Array,
    /,
) -> tuple[Array, Array]:
    _, rates = jax.jvp(
        lambda value: jnp.stack(_integral_thicknesses(value, coordinates)),
        (primitive,),
        (primitive_source,),
    )
    return rates[0], rates[1]


def _thermal_rates(
    case: CompressibleFlowCaseSpec,
    primitive: Array,
    primitive_source: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    density = primitive[..., 0]
    pressure = primitive[..., -1]
    density_source = primitive_source[..., 0]
    pressure_source = primitive_source[..., -1]
    internal, internal_source = jax.jvp(
        case.material.specific_internal_energy,
        (density, pressure),
        (density_source, pressure_source),
    )
    temperature, temperature_source = jax.jvp(
        case.material.temperature,
        (density, pressure),
        (density_source, pressure_source),
    )
    return internal, internal_source, temperature, temperature_source


def _conservative_source(
    case: CompressibleFlowCaseSpec,
    primitive: Array,
    primitive_source: Array,
    /,
) -> Array:
    return jax.jvp(
        case.primitive_to_conserved,
        (primitive,),
        (primitive_source,),
    )[1]


def _normalize_wall_indices(
    wall_indices: tuple[int, ...], profile_size: int, /
) -> tuple[int, ...]:
    normalized = tuple(
        profile_size - 1 if index == -1 else index for index in wall_indices
    )
    if any(index not in (0, profile_size - 1) for index in normalized):
        raise ValueError("Slow-growth thermal walls must be profile endpoints.")
    return normalized


def _apply_wall_thermal_condition(
    case: CompressibleFlowCaseSpec,
    primitive: Array,
    primitive_source: Array,
    mode: WallThermalMode,
    wall_indices: tuple[int, ...],
    /,
) -> tuple[Array, Array, Array]:
    result = primitive_source
    normalized = _normalize_wall_indices(wall_indices, primitive.shape[0])
    for index in normalized:
        neighbor = 1 if index == 0 else primitive.shape[0] - 2
        _, _, _, temperature_source = _thermal_rates(case, primitive, result)
        desired = (
            jnp.zeros_like(temperature_source[index])
            if mode == "isothermal"
            else temperature_source[neighbor]
        )
        density = primitive[:, 0]
        pressure = primitive[:, -1]
        pressure_direction = jnp.ones_like(pressure)
        pressure_sensitivity = jax.jvp(
            lambda value: case.material.temperature(density, value),
            (pressure,),
            (pressure_direction,),
        )[1]
        correction = (desired - temperature_source[index]) / pressure_sensitivity[index]
        result = result.at[index, -1].add(correction)
    _, _, _, temperature_source = _thermal_rates(case, primitive, result)
    residuals = []
    for index in normalized:
        neighbor = 1 if index == 0 else primitive.shape[0] - 2
        residuals.append(
            temperature_source[index]
            if mode == "isothermal"
            else temperature_source[index] - temperature_source[neighbor]
        )
    residual = (
        jnp.max(jnp.abs(jnp.stack(residuals)))
        if residuals
        else jnp.asarray(0.0, dtype=primitive.dtype)
    )
    return result, temperature_source, residual


def _apply_integral_constraints(
    primitive: Array,
    primitive_source: Array,
    coordinates: Array,
    displacement_target: float | None,
    momentum_target: float | None,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    displacement_rate, momentum_rate = _integral_thickness_rates(
        primitive, primitive_source, coordinates
    )
    constrained = tuple(
        index
        for index, target in enumerate((displacement_target, momentum_target))
        if target is not None
    )
    if not constrained:
        return (
            primitive_source,
            displacement_rate,
            momentum_rate,
            displacement_rate,
            momentum_rate,
        )

    extent = coordinates[-1] - coordinates[0]
    eta = (coordinates - coordinates[0]) / extent
    bubble = 4.0 * eta * (1.0 - eta)
    velocity_scale = jnp.maximum(
        jnp.abs(primitive[-1, 1]), jnp.asarray(1.0, primitive.dtype)
    )
    direction_0 = jnp.zeros_like(primitive).at[:, 1].set(velocity_scale * bubble)
    direction_1 = jnp.zeros_like(primitive).at[:, 1].set(velocity_scale * eta)
    columns = []
    for direction in (direction_0, direction_1):
        columns.append(
            jnp.stack(_integral_thickness_rates(primitive, direction, coordinates))
        )
    jacobian = jnp.stack(columns, axis=1)
    current = jnp.stack((displacement_rate, momentum_rate))
    target = jnp.asarray(
        (
            displacement_rate if displacement_target is None else displacement_target,
            momentum_rate if momentum_target is None else momentum_target,
        ),
        dtype=primitive.dtype,
    )
    selected = jnp.asarray(constrained, dtype=jnp.int32)
    selected_jacobian = jacobian[selected, :]
    selected_delta = (target - current)[selected]
    coefficients = jnp.linalg.pinv(selected_jacobian) @ selected_delta
    corrected = (
        primitive_source + coefficients[0] * direction_0 + coefficients[1] * direction_1
    )
    displacement_rate, momentum_rate = _integral_thickness_rates(
        primitive, corrected, coordinates
    )
    return corrected, displacement_rate, momentum_rate, target[0], target[1]


def _expanded_profile(profile: Array, state: Array, wall_normal_axis: int, /) -> Array:
    spatial_rank = state.ndim - 1
    shape = [1] * spatial_rank + [profile.shape[-1]]
    shape[wall_normal_axis] = profile.shape[0]
    return jnp.broadcast_to(profile.reshape(tuple(shape)), state.shape)


class CompressiblePlaneBaseflowSnapshot(StrictModule, NonTrainableState):
    """Immutable homogeneous statistics and their one-dimensional baseflow."""

    case: CompressibleFlowCaseSpec
    coordinates: Array
    mean_conserved: Array
    base_primitive: Array
    base_conserved: Array
    mean_density: Array
    density_variance: Array
    reynolds_mean_velocity: Array
    favre_mean_velocity: Array
    reynolds_stress: Array
    favre_stress: Array
    mean_pressure: Array
    mean_temperature: Array
    wall_normal_base_derivative: Array
    streamwise_base_derivative: Array | None
    displacement_thickness: Array
    momentum_thickness: Array
    finite: Array
    admissible: Array
    dimension: int = eqx.field(static=True)
    wall_normal_axis: int = eqx.field(static=True)
    homogeneous_axes: tuple[int, ...] = eqx.field(static=True)
    sample_index: int = eqx.field(static=True)
    sample_time: float | None = eqx.field(static=True)
    streamwise_location: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)


class CompressiblePlaneBaseflowPlan(StrictModule, NonTrainableState):
    """Freeze Favre-consistent plane statistics as a baseflow snapshot."""

    case: CompressibleFlowCaseSpec
    coordinates: Array
    dimension: int = eqx.field(static=True)
    wall_normal_axis: int = eqx.field(static=True)
    homogeneous_axes: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        case: CompressibleFlowCaseSpec,
        wall_normal_coordinates: ArrayLike,
        /,
        *,
        wall_normal_axis: int = 0,
        homogeneous_axes: Sequence[int] | None = None,
    ):
        if not isinstance(case, CompressibleFlowCaseSpec):
            raise TypeError(
                "Compressible baseflow preparation requires a case specification."
            )
        coordinates = jnp.asarray(wall_normal_coordinates)
        wall_axis = int(wall_normal_axis)
        axes = (
            tuple(axis for axis in range(case.dimension) if axis != wall_axis)
            if homogeneous_axes is None
            else tuple(int(axis) for axis in homogeneous_axes)
        )
        expected = tuple(axis for axis in range(case.dimension) if axis != wall_axis)
        values = np.asarray(coordinates)
        if (
            coordinates.ndim != 1
            or coordinates.shape[0] < 2
            or wall_axis not in range(case.dimension)
            or axes != expected
            or not np.all(np.isfinite(values))
            or not np.all(np.diff(values) > 0.0)
        ):
            raise ValueError("Compressible plane-baseflow geometry is invalid.")
        self.case = case
        self.coordinates = coordinates
        self.dimension = case.dimension
        self.wall_normal_axis = wall_axis
        self.homogeneous_axes = axes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compressible-plane-baseflow",
                "case_id": case.case_id,
                "coordinates": values.tolist(),
                "wall_normal_axis": wall_axis,
                "homogeneous_axes": axes,
            }
        )

    def evaluate(
        self,
        conserved: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
        streamwise_base_derivative: ArrayLike | None = None,
        sample_index: int = 0,
        sample_time: float | None = None,
        streamwise_location: float | None = None,
    ) -> CompressiblePlaneBaseflowSnapshot:
        state = jnp.asarray(conserved)
        if (
            state.ndim != self.dimension + 1
            or state.shape[-1] != self.dimension + 2
            or state.shape[self.wall_normal_axis] != self.coordinates.shape[0]
        ):
            raise ValueError("Compressible baseflow state has the wrong shape.")
        spatial_shape = state.shape[:-1]
        weights_ = (
            jnp.ones(spatial_shape, dtype=state.dtype)
            if weights is None
            else jnp.asarray(weights)
        )
        try:
            weights_ = jnp.broadcast_to(weights_, spatial_shape)
        except ValueError as error:
            raise ValueError(
                "Compressible baseflow weights do not broadcast to the state."
            ) from error
        primitive = self.case.conserved_to_primitive(state)
        density = primitive[..., 0]
        velocity = primitive[..., 1 : 1 + self.dimension]
        pressure = primitive[..., -1]
        temperature = self.case.material.temperature(density, pressure)
        mean_density = _weighted_mean(density, weights_, self.homogeneous_axes)
        mean_density_squared = _weighted_mean(
            density * density, weights_, self.homogeneous_axes
        )
        reynolds_velocity = _weighted_mean(velocity, weights_, self.homogeneous_axes)
        mean_momentum = _weighted_mean(
            density[..., None] * velocity, weights_, self.homogeneous_axes
        )
        favre_velocity = mean_momentum / mean_density[..., None]
        reynolds_fluctuation = velocity - reynolds_velocity.reshape(
            tuple(
                1 if axis in self.homogeneous_axes else size
                for axis, size in enumerate(spatial_shape)
            )
            + (self.dimension,)
        )
        favre_fluctuation = velocity - favre_velocity.reshape(
            tuple(
                1 if axis in self.homogeneous_axes else size
                for axis, size in enumerate(spatial_shape)
            )
            + (self.dimension,)
        )
        reynolds_outer = (
            reynolds_fluctuation[..., :, None] * reynolds_fluctuation[..., None, :]
        )
        favre_outer = favre_fluctuation[..., :, None] * favre_fluctuation[..., None, :]
        reynolds_stress = _weighted_mean(reynolds_outer, weights_, self.homogeneous_axes)
        favre_stress = (
            _weighted_mean(
                density[..., None, None] * favre_outer, weights_, self.homogeneous_axes
            )
            / mean_density[..., None, None]
        )
        mean_pressure = _weighted_mean(pressure, weights_, self.homogeneous_axes)
        mean_temperature = _weighted_mean(temperature, weights_, self.homogeneous_axes)
        mean_conserved = _weighted_mean(state, weights_, self.homogeneous_axes)
        base_primitive = jnp.concatenate(
            (mean_density[..., None], favre_velocity, mean_pressure[..., None]), axis=-1
        )
        base_conserved = self.case.primitive_to_conserved(base_primitive)
        wall_derivative = _profile_gradient(base_primitive, self.coordinates)
        streamwise_derivative = (
            None
            if streamwise_base_derivative is None
            else jnp.asarray(streamwise_base_derivative, dtype=state.dtype)
        )
        if (
            streamwise_derivative is not None
            and streamwise_derivative.shape != base_primitive.shape
        ):
            raise ValueError(
                "Streamwise base derivatives must match the primitive baseflow."
            )
        displacement, momentum = _integral_thicknesses(base_primitive, self.coordinates)
        plane_weight = _sum_axes(weights_, self.homogeneous_axes)
        finite = (
            jnp.all(jnp.isfinite(state))
            & jnp.all(jnp.isfinite(weights_))
            & jnp.all(weights_ >= 0.0)
            & jnp.all(plane_weight > 0.0)
            & jnp.all(jnp.isfinite(base_primitive))
            & jnp.all(jnp.isfinite(wall_derivative))
            & (
                True
                if streamwise_derivative is None
                else jnp.all(jnp.isfinite(streamwise_derivative))
            )
        )
        admissible = jnp.all(self.case.material.admissible(density, pressure))
        index = int(sample_index)
        time = None if sample_time is None else float(sample_time)
        location = None if streamwise_location is None else float(streamwise_location)
        if (
            index < 0
            or (time is not None and not np.isfinite(time))
            or (location is not None and not np.isfinite(location))
        ):
            raise ValueError("Compressible baseflow sample coordinates are invalid.")
        fingerprint = array_tree_fingerprint(
            {
                "base_primitive": base_primitive,
                "mean_conserved": mean_conserved,
                "streamwise_base_derivative": streamwise_derivative,
            }
        )
        snapshot_id = canonical_fingerprint(
            {
                "kind": "compressible-plane-baseflow-snapshot",
                "plan_id": self.plan_id,
                "sample_index": index,
                "sample_time": time,
                "streamwise_location": location,
                "content": fingerprint,
            }
        )
        return CompressiblePlaneBaseflowSnapshot(
            self.case,
            self.coordinates,
            mean_conserved,
            base_primitive,
            base_conserved,
            mean_density,
            mean_density_squared - mean_density * mean_density,
            reynolds_velocity,
            favre_velocity,
            reynolds_stress,
            favre_stress,
            mean_pressure,
            mean_temperature,
            wall_derivative,
            streamwise_derivative,
            displacement,
            momentum,
            finite,
            admissible,
            self.dimension,
            self.wall_normal_axis,
            self.homogeneous_axes,
            index,
            time,
            location,
            self.plan_id,
            snapshot_id,
        )

    __call__ = evaluate


class SlowGrowthSource(StrictModule):
    """Primitive and conservative forms of one frozen slow-growth source."""

    primitive: Array
    conservative: Array
    mass: Array
    momentum: Array
    total_energy: Array
    specific_internal_energy: Array
    temperature: Array
    specific_entropy: Array
    finite: Array
    prepared_id: str = eqx.field(static=True)


class SlowGrowthBudget(StrictModule):
    """Mass, momentum, energy, and integral-thickness source ledger."""

    profile_mass_rate: Array
    profile_momentum_rate: Array
    profile_total_energy_rate: Array
    displacement_thickness_rate: Array
    momentum_thickness_rate: Array
    target_displacement_thickness_rate: Array
    target_momentum_thickness_rate: Array
    displacement_constraint_residual: Array
    momentum_constraint_residual: Array
    prepared_id: str = eqx.field(static=True)


class SlowGrowthEvidence(StrictModule):
    """Named algebraic, wall, integral, energy, and entropy checks."""

    zero_source_residual: Array
    base_residual: Array
    integral_constraint_residual: Array
    wall_thermal_residual: Array
    energy_identity_residual: Array
    entropy_identity_residual: Array
    finite: Array
    admissible: Array
    coordinate: SlowGrowthCoordinate = eqx.field(static=True)
    model_label: str = eqx.field(static=True)
    claims_spatial_dns: bool = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class SlowGrowthEvaluation(StrictModule):
    source: SlowGrowthSource
    budget: SlowGrowthBudget
    evidence: SlowGrowthEvidence


class SlowGrowthFiniteXEvidence(StrictModule, NonTrainableState):
    """Comparison against supplied finite-x evidence; never a fidelity relabel."""

    l2_error: Array
    relative_l2_error: Array
    maximum_error: Array
    reference_l2_norm: Array
    admission_threshold: Array
    admitted: Array
    finite: Array
    compared_values: int = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)
    reference_label: str = eqx.field(static=True)
    model_label: str = eqx.field(static=True)
    claims_spatial_dns: bool = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class PreparedSlowGrowthSource(StrictModule, NonTrainableState):
    """One pre-step snapshot shared unchanged by every RK or IMEX stage."""

    snapshot: CompressiblePlaneBaseflowSnapshot
    primitive_source_profile: Array
    base_conservative_source_profile: Array
    wall_temperature_source_profile: Array
    displacement_thickness_rate: Array
    momentum_thickness_rate: Array
    target_displacement_thickness_rate: Array
    target_momentum_thickness_rate: Array
    wall_thermal_residual: Array
    evidence_tolerance: float = eqx.field(static=True)
    displacement_constrained: bool = eqx.field(static=True)
    momentum_constrained: bool = eqx.field(static=True)
    zero_source_expected: bool = eqx.field(static=True)
    wall_thermal_mode: WallThermalMode = eqx.field(static=True)
    wall_indices: tuple[int, ...] = eqx.field(static=True)
    coordinate: SlowGrowthCoordinate = eqx.field(static=True)
    model_label: str = eqx.field(static=True)
    parent_step: int = eqx.field(static=True)
    parent_continuation_id: str | None = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @property
    def claims_spatial_dns(self) -> bool:
        return False

    def _check_state(self, conserved: Array, /) -> None:
        if (
            conserved.ndim != self.snapshot.dimension + 1
            or conserved.shape[-1] != self.snapshot.dimension + 2
            or conserved.shape[self.snapshot.wall_normal_axis]
            != self.snapshot.coordinates.shape[0]
        ):
            raise ValueError("Slow-growth state does not match its frozen snapshot.")

    def primitive_source(self, conserved: ArrayLike, /) -> Array:
        state = jnp.asarray(conserved)
        self._check_state(state)
        return _expanded_profile(
            self.primitive_source_profile, state, self.snapshot.wall_normal_axis
        )

    def conservative_source(self, conserved: ArrayLike, /) -> Array:
        state = jnp.asarray(conserved)
        self._check_state(state)
        primitive = self.snapshot.case.conserved_to_primitive(state)
        primitive_source = _expanded_profile(
            self.primitive_source_profile, state, self.snapshot.wall_normal_axis
        )
        return _conservative_source(self.snapshot.case, primitive, primitive_source)

    def evaluate(self, conserved: ArrayLike, /) -> SlowGrowthEvaluation:
        state = jnp.asarray(conserved)
        self._check_state(state)
        primitive = self.snapshot.case.conserved_to_primitive(state)
        primitive_source = _expanded_profile(
            self.primitive_source_profile, state, self.snapshot.wall_normal_axis
        )
        conservative_source = _conservative_source(
            self.snapshot.case, primitive, primitive_source
        )
        internal, internal_source, temperature, temperature_source = _thermal_rates(
            self.snapshot.case, primitive, primitive_source
        )
        density = primitive[..., 0]
        velocity = primitive[..., 1 : 1 + self.snapshot.dimension]
        pressure = primitive[..., -1]
        density_source = primitive_source[..., 0]
        velocity_source = primitive_source[..., 1 : 1 + self.snapshot.dimension]
        kinetic_source = 0.5 * jnp.sum(
            velocity * velocity, axis=-1
        ) * density_source + density * jnp.sum(velocity * velocity_source, axis=-1)
        expected_energy_source = (
            internal * density_source + density * internal_source + kinetic_source
        )
        energy_residual = conservative_source[..., -1] - expected_energy_source
        entropy_source = (
            internal_source - pressure * density_source / (density * density)
        ) / temperature
        entropy_residual = (
            temperature * entropy_source
            - internal_source
            + pressure * density_source / (density * density)
        )
        base_check = _conservative_source(
            self.snapshot.case,
            self.snapshot.base_primitive,
            self.primitive_source_profile,
        )
        base_residual = jnp.max(
            jnp.abs(base_check - self.base_conservative_source_profile)
        )
        displacement_residual = (
            self.displacement_thickness_rate - self.target_displacement_thickness_rate
            if self.displacement_constrained
            else jnp.asarray(0.0, dtype=state.dtype)
        )
        momentum_residual = (
            self.momentum_thickness_rate - self.target_momentum_thickness_rate
            if self.momentum_constrained
            else jnp.asarray(0.0, dtype=state.dtype)
        )
        integral_residual = jnp.maximum(
            jnp.abs(displacement_residual), jnp.abs(momentum_residual)
        )
        zero_residual = (
            jnp.max(jnp.abs(conservative_source))
            if self.zero_source_expected
            else jnp.asarray(0.0, dtype=state.dtype)
        )
        profile_mass = _trapezoid(
            self.base_conservative_source_profile[:, 0], self.snapshot.coordinates
        )
        profile_momentum = _trapezoid(
            self.base_conservative_source_profile[:, 1 : 1 + self.snapshot.dimension],
            self.snapshot.coordinates,
        )
        profile_energy = _trapezoid(
            self.base_conservative_source_profile[:, -1], self.snapshot.coordinates
        )
        finite = jnp.all(
            jnp.isfinite(primitive_source) & jnp.isfinite(conservative_source)
        ) & jnp.all(jnp.isfinite(entropy_source))
        maximum_identity_residual = jnp.maximum(
            jnp.max(jnp.abs(energy_residual)), jnp.max(jnp.abs(entropy_residual))
        )
        state_admissible = jnp.all(
            self.snapshot.case.material.admissible(density, pressure)
        )
        admissible = (
            finite
            & self.snapshot.finite
            & self.snapshot.admissible
            & state_admissible
            & (zero_residual <= self.evidence_tolerance)
            & (base_residual <= self.evidence_tolerance)
            & (integral_residual <= self.evidence_tolerance)
            & (self.wall_thermal_residual <= self.evidence_tolerance)
            & (maximum_identity_residual <= self.evidence_tolerance)
        )
        source = SlowGrowthSource(
            primitive_source,
            conservative_source,
            conservative_source[..., 0],
            conservative_source[..., 1 : 1 + self.snapshot.dimension],
            conservative_source[..., -1],
            internal_source,
            temperature_source,
            entropy_source,
            finite,
            self.prepared_id,
        )
        budget = SlowGrowthBudget(
            profile_mass,
            profile_momentum,
            profile_energy,
            self.displacement_thickness_rate,
            self.momentum_thickness_rate,
            self.target_displacement_thickness_rate,
            self.target_momentum_thickness_rate,
            displacement_residual,
            momentum_residual,
            self.prepared_id,
        )
        evidence = SlowGrowthEvidence(
            zero_residual,
            base_residual,
            integral_residual,
            self.wall_thermal_residual,
            jnp.max(jnp.abs(energy_residual)),
            jnp.max(jnp.abs(entropy_residual)),
            finite,
            admissible,
            self.coordinate,
            self.model_label,
            False,
            self.snapshot.snapshot_id,
            self.prepared_id,
        )
        return SlowGrowthEvaluation(source, budget, evidence)

    def evaluate_primitive(self, primitive: ArrayLike, /) -> SlowGrowthEvaluation:
        value = jnp.asarray(primitive)
        return self.evaluate(self.snapshot.case.primitive_to_conserved(value))

    def jvp(self, conserved: ArrayLike, tangent: ArrayLike, /) -> Array:
        state = jnp.asarray(conserved)
        direction = jnp.asarray(tangent)
        self._check_state(state)
        if direction.shape != state.shape:
            raise ValueError("Slow-growth JVP tangent has the wrong shape.")
        return jax.jvp(self.conservative_source, (state,), (direction,))[1]

    def vjp(self, conserved: ArrayLike, cotangent: ArrayLike, /) -> Array:
        state = jnp.asarray(conserved)
        dual = jnp.asarray(cotangent)
        self._check_state(state)
        if dual.shape != state.shape:
            raise ValueError("Slow-growth VJP cotangent has the wrong shape.")
        _, pullback = jax.vjp(self.conservative_source, state)
        return pullback(dual)[0]

    def compare_finite_x(
        self,
        conserved: ArrayLike,
        finite_x_conservative_source: ArrayLike,
        /,
        *,
        reference_id: str,
        relative_tolerance: float,
        absolute_tolerance: float = 0.0,
    ) -> SlowGrowthFiniteXEvidence:
        if not reference_id:
            raise ValueError(
                "Finite-x comparison evidence requires a reference identifier."
            )
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        if (
            not np.isfinite(relative)
            or not np.isfinite(absolute)
            or relative < 0.0
            or absolute < 0.0
        ):
            raise ValueError(
                "Finite-x comparison tolerances must be finite and nonnegative."
            )
        modeled = self.conservative_source(conserved)
        reference = jnp.asarray(finite_x_conservative_source, dtype=modeled.dtype)
        if reference.shape != modeled.shape:
            raise ValueError("Finite-x reference source has the wrong shape.")
        difference = modeled - reference
        l2_error = jnp.sqrt(jnp.mean(difference * difference))
        reference_norm = jnp.sqrt(jnp.mean(reference * reference))
        threshold = absolute + relative * reference_norm
        relative_error = l2_error / jnp.maximum(
            reference_norm, jnp.asarray(jnp.finfo(modeled.dtype).tiny, modeled.dtype)
        )
        maximum_error = jnp.max(jnp.abs(difference))
        finite = jnp.all(jnp.isfinite(modeled)) & jnp.all(jnp.isfinite(reference))
        admitted = finite & (l2_error <= threshold)
        return SlowGrowthFiniteXEvidence(
            l2_error,
            relative_error,
            maximum_error,
            reference_norm,
            threshold,
            admitted,
            finite,
            modeled.size,
            str(reference_id),
            "finite-x-dns",
            self.model_label,
            False,
            self.snapshot.snapshot_id,
            self.prepared_id,
        )

    __call__ = evaluate


class SlowGrowthStepEvidence(StrictModule, NonTrainableState):
    """Parent-step acceptance or rejected-step rollback evidence."""

    accepted: bool = eqx.field(static=True)
    continuation_advanced: bool = eqx.field(static=True)
    parent_step: int = eqx.field(static=True)
    resulting_step: int = eqx.field(static=True)
    parent_snapshot_id: str = eqx.field(static=True)
    resulting_snapshot_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    parent_continuation_id: str = eqx.field(static=True)
    resulting_continuation_id: str = eqx.field(static=True)


class SlowGrowthRestart(StrictModule, NonTrainableState):
    snapshot: CompressiblePlaneBaseflowSnapshot
    accepted_step: int = eqx.field(static=True)
    accepted_time: float = eqx.field(static=True)
    continuation_id: str = eqx.field(static=True)


class SlowGrowthContinuation(StrictModule, NonTrainableState):
    """Accepted baseflow state; rejected parent attempts leave it unchanged."""

    snapshot: CompressiblePlaneBaseflowSnapshot
    accepted_step: int = eqx.field(static=True)
    accepted_time: float = eqx.field(static=True)
    continuation_id: str = eqx.field(static=True)

    def __init__(
        self,
        snapshot: CompressiblePlaneBaseflowSnapshot,
        /,
        *,
        accepted_step: int = 0,
        accepted_time: float = 0.0,
        continuation_id: str | None = None,
    ):
        if not isinstance(snapshot, CompressiblePlaneBaseflowSnapshot):
            raise TypeError("Slow-growth continuation requires a baseflow snapshot.")
        step = int(accepted_step)
        time = float(accepted_time)
        if step < 0 or not np.isfinite(time):
            raise ValueError("Slow-growth continuation coordinates are invalid.")
        expected_id = canonical_fingerprint(
            {
                "kind": "slow-growth-continuation",
                "snapshot_id": snapshot.snapshot_id,
                "accepted_step": step,
                "accepted_time": time,
            }
        )
        if continuation_id is not None and str(continuation_id) != expected_id:
            raise ValueError(
                "Restart continuation identifier does not bind its exact state."
            )
        self.snapshot = snapshot
        self.accepted_step = step
        self.accepted_time = time
        self.continuation_id = expected_id

    def _check_prepared(self, prepared: PreparedSlowGrowthSource, /) -> None:
        if (
            not isinstance(prepared, PreparedSlowGrowthSource)
            or prepared.parent_step != self.accepted_step
            or prepared.parent_continuation_id != self.continuation_id
            or prepared.snapshot.snapshot_id != self.snapshot.snapshot_id
        ):
            raise ValueError(
                "Prepared slow-growth source does not belong to this parent step."
            )

    def accept(
        self,
        prepared: PreparedSlowGrowthSource,
        next_snapshot: CompressiblePlaneBaseflowSnapshot,
        /,
        *,
        accepted_time: float | None = None,
    ) -> "SlowGrowthContinuation":
        self._check_prepared(prepared)
        if not isinstance(next_snapshot, CompressiblePlaneBaseflowSnapshot):
            raise TypeError("Accepted slow-growth continuation requires a new snapshot.")
        if next_snapshot.plan_id != self.snapshot.plan_id:
            raise ValueError(
                "Accepted slow-growth snapshot belongs to a different baseflow plan."
            )
        time = (
            next_snapshot.sample_time
            if accepted_time is None and next_snapshot.sample_time is not None
            else self.accepted_time
            if accepted_time is None
            else float(accepted_time)
        )
        if time < self.accepted_time:
            raise ValueError("Accepted slow-growth time cannot move backwards.")
        return SlowGrowthContinuation(
            next_snapshot,
            accepted_step=self.accepted_step + 1,
            accepted_time=time,
        )

    def reject(self, prepared: PreparedSlowGrowthSource, /) -> "SlowGrowthContinuation":
        self._check_prepared(prepared)
        return self

    def acceptance_evidence(
        self,
        prepared: PreparedSlowGrowthSource,
        continuation: "SlowGrowthContinuation",
        /,
    ) -> SlowGrowthStepEvidence:
        self._check_prepared(prepared)
        if continuation.accepted_step != self.accepted_step + 1:
            raise ValueError("Acceptance evidence requires the next continuation step.")
        return SlowGrowthStepEvidence(
            True,
            True,
            self.accepted_step,
            continuation.accepted_step,
            self.snapshot.snapshot_id,
            continuation.snapshot.snapshot_id,
            prepared.prepared_id,
            self.continuation_id,
            continuation.continuation_id,
        )

    def rejection_evidence(
        self, prepared: PreparedSlowGrowthSource, /
    ) -> SlowGrowthStepEvidence:
        self._check_prepared(prepared)
        return SlowGrowthStepEvidence(
            False,
            False,
            self.accepted_step,
            self.accepted_step,
            self.snapshot.snapshot_id,
            self.snapshot.snapshot_id,
            prepared.prepared_id,
            self.continuation_id,
            self.continuation_id,
        )

    def checkpoint(self) -> SlowGrowthRestart:
        return SlowGrowthRestart(
            self.snapshot,
            self.accepted_step,
            self.accepted_time,
            self.continuation_id,
        )

    @classmethod
    def from_restart(cls, restart: SlowGrowthRestart, /) -> "SlowGrowthContinuation":
        if not isinstance(restart, SlowGrowthRestart):
            raise TypeError("Slow-growth restart has the wrong record type.")
        return cls(
            restart.snapshot,
            accepted_step=restart.accepted_step,
            accepted_time=restart.accepted_time,
            continuation_id=restart.continuation_id,
        )


def _model_options(
    wall_thermal_mode: WallThermalMode,
    wall_indices: Sequence[int],
    displacement_thickness_rate: float | None,
    momentum_thickness_rate: float | None,
    evidence_tolerance: float,
    /,
) -> tuple[WallThermalMode, tuple[int, ...], float | None, float | None, float]:
    indices = tuple(int(index) for index in wall_indices)
    displacement = (
        None
        if displacement_thickness_rate is None
        else float(displacement_thickness_rate)
    )
    momentum = None if momentum_thickness_rate is None else float(momentum_thickness_rate)
    tolerance = float(evidence_tolerance)
    scalars = tuple(
        value for value in (displacement, momentum, tolerance) if value is not None
    )
    if (
        wall_thermal_mode not in ("adiabatic", "isothermal")
        or len(set(indices)) != len(indices)
        or any(index not in (0, -1) for index in indices)
        or any(not np.isfinite(value) for value in scalars)
        or tolerance <= 0.0
    ):
        raise ValueError("Slow-growth model options are invalid.")
    return wall_thermal_mode, indices, displacement, momentum, tolerance


def _prepare_source(
    snapshot: CompressiblePlaneBaseflowSnapshot,
    preliminary_source: Array,
    *,
    coordinate: SlowGrowthCoordinate,
    model_label: str,
    model_id: str,
    wall_thermal_mode: WallThermalMode,
    wall_indices: tuple[int, ...],
    displacement_thickness_rate: float | None,
    momentum_thickness_rate: float | None,
    evidence_tolerance: float,
    zero_source_expected: bool,
    continuation: SlowGrowthContinuation | None,
) -> PreparedSlowGrowthSource:
    if continuation is not None:
        if continuation.snapshot.snapshot_id != snapshot.snapshot_id:
            raise ValueError(
                "Slow-growth preparation must use the accepted parent snapshot."
            )
        parent_step = continuation.accepted_step
        parent_id = continuation.continuation_id
    else:
        parent_step = snapshot.sample_index
        parent_id = None
    thermal_source, _, wall_residual = _apply_wall_thermal_condition(
        snapshot.case,
        snapshot.base_primitive,
        preliminary_source,
        wall_thermal_mode,
        wall_indices,
    )
    (
        constrained_source,
        displacement_rate,
        momentum_rate,
        displacement_target,
        momentum_target,
    ) = _apply_integral_constraints(
        snapshot.base_primitive,
        thermal_source,
        snapshot.coordinates,
        displacement_thickness_rate,
        momentum_thickness_rate,
    )
    _, _, _, wall_temperature_source = _thermal_rates(
        snapshot.case, snapshot.base_primitive, constrained_source
    )
    base_conservative_source = _conservative_source(
        snapshot.case, snapshot.base_primitive, constrained_source
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-compressible-slow-growth-source",
            "model_id": model_id,
            "snapshot_id": snapshot.snapshot_id,
            "parent_step": parent_step,
            "parent_continuation_id": parent_id,
        }
    )
    return PreparedSlowGrowthSource(
        snapshot,
        constrained_source,
        base_conservative_source,
        wall_temperature_source,
        displacement_rate,
        momentum_rate,
        displacement_target,
        momentum_target,
        wall_residual,
        evidence_tolerance,
        displacement_thickness_rate is not None,
        momentum_thickness_rate is not None,
        zero_source_expected,
        wall_thermal_mode,
        wall_indices,
        coordinate,
        model_label,
        parent_step,
        parent_id,
        model_id,
        prepared_id,
    )


class TemporalSlowGrowthModelPlan(StrictModule, NonTrainableState):
    """Temporal dilation model using wall-normal, never streamwise, derivatives."""

    growth_rate: float = eqx.field(static=True)
    wall_thermal_mode: WallThermalMode = eqx.field(static=True)
    wall_indices: tuple[int, ...] = eqx.field(static=True)
    displacement_thickness_rate: float | None = eqx.field(static=True)
    momentum_thickness_rate: float | None = eqx.field(static=True)
    evidence_tolerance: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        growth_rate: float,
        /,
        *,
        wall_thermal_mode: WallThermalMode = "adiabatic",
        wall_indices: Sequence[int] = (0,),
        displacement_thickness_rate: float | None = None,
        momentum_thickness_rate: float | None = None,
        evidence_tolerance: float = 1e-6,
    ):
        rate = float(growth_rate)
        if not np.isfinite(rate):
            raise ValueError("Temporal slow-growth rate must be finite.")
        mode, indices, displacement, momentum, tolerance = _model_options(
            wall_thermal_mode,
            wall_indices,
            displacement_thickness_rate,
            momentum_thickness_rate,
            evidence_tolerance,
        )
        self.growth_rate = rate
        self.wall_thermal_mode = mode
        self.wall_indices = indices
        self.displacement_thickness_rate = displacement
        self.momentum_thickness_rate = momentum
        self.evidence_tolerance = tolerance
        self.model_id = canonical_fingerprint(
            {
                "kind": "temporal-compressible-slow-growth",
                "growth_rate": rate,
                "wall_thermal_mode": mode,
                "wall_indices": indices,
                "displacement_thickness_rate": displacement,
                "momentum_thickness_rate": momentum,
                "evidence_tolerance": tolerance,
            }
        )

    @property
    def coordinate(self) -> SlowGrowthCoordinate:
        return "temporal"

    @property
    def claims_spatial_dns(self) -> bool:
        return False

    def prepare(
        self,
        snapshot: CompressiblePlaneBaseflowSnapshot,
        /,
        *,
        continuation: SlowGrowthContinuation | None = None,
    ) -> PreparedSlowGrowthSource:
        if not isinstance(snapshot, CompressiblePlaneBaseflowSnapshot):
            raise TypeError("Temporal slow growth requires a baseflow snapshot.")
        dilation_coordinate = snapshot.coordinates - snapshot.coordinates[0]
        preliminary = (
            -self.growth_rate
            * dilation_coordinate[:, None]
            * snapshot.wall_normal_base_derivative
        )
        zero_expected = (
            self.growth_rate == 0.0
            and (self.displacement_thickness_rate in (None, 0.0))
            and (self.momentum_thickness_rate in (None, 0.0))
        )
        return _prepare_source(
            snapshot,
            preliminary,
            coordinate="temporal",
            model_label="temporal-slow-growth-model",
            model_id=self.model_id,
            wall_thermal_mode=self.wall_thermal_mode,
            wall_indices=self.wall_indices,
            displacement_thickness_rate=self.displacement_thickness_rate,
            momentum_thickness_rate=self.momentum_thickness_rate,
            evidence_tolerance=self.evidence_tolerance,
            zero_source_expected=zero_expected,
            continuation=continuation,
        )


class SpatialSlowGrowthModelPlan(StrictModule, NonTrainableState):
    """Modeled-spatial source requiring supplied finite streamwise derivatives."""

    streamwise_convection_velocity: float = eqx.field(static=True)
    wall_thermal_mode: WallThermalMode = eqx.field(static=True)
    wall_indices: tuple[int, ...] = eqx.field(static=True)
    displacement_thickness_rate: float | None = eqx.field(static=True)
    momentum_thickness_rate: float | None = eqx.field(static=True)
    evidence_tolerance: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        streamwise_convection_velocity: float = 1.0,
        /,
        *,
        wall_thermal_mode: WallThermalMode = "adiabatic",
        wall_indices: Sequence[int] = (0,),
        displacement_thickness_rate: float | None = None,
        momentum_thickness_rate: float | None = None,
        evidence_tolerance: float = 1e-6,
    ):
        velocity = float(streamwise_convection_velocity)
        if not np.isfinite(velocity):
            raise ValueError("Modeled-spatial convection velocity must be finite.")
        mode, indices, displacement, momentum, tolerance = _model_options(
            wall_thermal_mode,
            wall_indices,
            displacement_thickness_rate,
            momentum_thickness_rate,
            evidence_tolerance,
        )
        self.streamwise_convection_velocity = velocity
        self.wall_thermal_mode = mode
        self.wall_indices = indices
        self.displacement_thickness_rate = displacement
        self.momentum_thickness_rate = momentum
        self.evidence_tolerance = tolerance
        self.model_id = canonical_fingerprint(
            {
                "kind": "modeled-spatial-compressible-slow-growth",
                "streamwise_convection_velocity": velocity,
                "wall_thermal_mode": mode,
                "wall_indices": indices,
                "displacement_thickness_rate": displacement,
                "momentum_thickness_rate": momentum,
                "evidence_tolerance": tolerance,
            }
        )

    @property
    def coordinate(self) -> SlowGrowthCoordinate:
        return "modeled-spatial"

    @property
    def claims_spatial_dns(self) -> bool:
        return False

    def prepare(
        self,
        snapshot: CompressiblePlaneBaseflowSnapshot,
        /,
        *,
        continuation: SlowGrowthContinuation | None = None,
    ) -> PreparedSlowGrowthSource:
        if not isinstance(snapshot, CompressiblePlaneBaseflowSnapshot):
            raise TypeError("Modeled-spatial slow growth requires a baseflow snapshot.")
        if snapshot.streamwise_base_derivative is None:
            raise ValueError(
                "Modeled-spatial slow growth requires supplied streamwise base derivatives."
            )
        preliminary = (
            -self.streamwise_convection_velocity * snapshot.streamwise_base_derivative
        )
        zero_expected = (
            self.streamwise_convection_velocity == 0.0
            and (self.displacement_thickness_rate in (None, 0.0))
            and (self.momentum_thickness_rate in (None, 0.0))
        )
        return _prepare_source(
            snapshot,
            preliminary,
            coordinate="modeled-spatial",
            model_label="modeled-spatial-slow-growth",
            model_id=self.model_id,
            wall_thermal_mode=self.wall_thermal_mode,
            wall_indices=self.wall_indices,
            displacement_thickness_rate=self.displacement_thickness_rate,
            momentum_thickness_rate=self.momentum_thickness_rate,
            evidence_tolerance=self.evidence_tolerance,
            zero_source_expected=zero_expected,
            continuation=continuation,
        )


__all__ = [
    "CompressiblePlaneBaseflowPlan",
    "CompressiblePlaneBaseflowSnapshot",
    "PreparedSlowGrowthSource",
    "SlowGrowthBudget",
    "SlowGrowthContinuation",
    "SlowGrowthEvaluation",
    "SlowGrowthEvidence",
    "SlowGrowthFiniteXEvidence",
    "SlowGrowthRestart",
    "SlowGrowthSource",
    "SlowGrowthStepEvidence",
    "SpatialSlowGrowthModelPlan",
    "TemporalSlowGrowthModelPlan",
]
