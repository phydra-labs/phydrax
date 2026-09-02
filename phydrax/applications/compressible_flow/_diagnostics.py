#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


def _sum_axes(value: Array, axes: tuple[int, ...], /) -> Array:
    result = value
    for axis in sorted(axes, reverse=True):
        result = jnp.sum(result, axis=axis)
    return result


def _weighted_sum(value: Array, weights: Array, axes: tuple[int, ...], /) -> Array:
    extra = value.ndim - weights.ndim
    weighted = value * weights.reshape(weights.shape + (1,) * extra)
    return _sum_axes(weighted, axes)


class CompressibleBudget(StrictModule):
    mass: Array
    momentum: Array
    total_energy: Array
    kinetic_energy: Array
    internal_energy: Array
    mass_rate: Array
    momentum_rate: Array
    total_energy_rate: Array
    kinetic_energy_rate: Array
    internal_energy_rate: Array
    pressure_dilatation: Array
    viscous: Array
    thermal: Array
    entropy: Array
    interface: Array
    filter: Array
    limiter: Array
    sponge: Array
    forcing: Array
    boundary: Array
    decomposition_residual: Array
    complete: Array
    plan_id: str = eqx.field(static=True)


class CompressibleBudgetPlan(StrictModule, NonTrainableState):
    """Conservative compressible totals and a complete named work ledger."""

    dimension: int = eqx.field(static=True)
    accumulation: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, /, *, accumulation: str = "deterministic"):
        dimension_ = int(dimension)
        accumulation_ = str(accumulation)
        if dimension_ not in (1, 2, 3) or accumulation_ not in (
            "deterministic",
            "compensated",
        ):
            raise ValueError("Compressible budget plan is invalid.")
        self.dimension = dimension_
        self.accumulation = accumulation_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compressible-budget-plan",
                "dimension": dimension_,
                "accumulation": accumulation_,
                "terms": (
                    "mass",
                    "momentum",
                    "total",
                    "kinetic",
                    "internal",
                    "pressure-dilatation",
                    "viscous",
                    "thermal",
                    "entropy",
                    "interface",
                    "filter",
                    "limiter",
                    "sponge",
                    "forcing",
                    "boundary",
                ),
            }
        )

    def evaluate(
        self,
        conserved: ArrayLike,
        total_rate: ArrayLike,
        pressure: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
        velocity_gradient: ArrayLike | None = None,
        viscous_stress: ArrayLike | None = None,
        thermal_rate: ArrayLike | None = None,
        entropy_rate: ArrayLike | None = None,
        interface_rate: ArrayLike | None = None,
        filter_rate: ArrayLike | None = None,
        limiter_rate: ArrayLike | None = None,
        sponge_rate: ArrayLike | None = None,
        forcing_rate: ArrayLike | None = None,
        boundary_rate: ArrayLike | None = None,
    ) -> CompressibleBudget:
        state = jnp.asarray(conserved)
        rate = jnp.asarray(total_rate)
        pressure_ = jnp.asarray(pressure)
        if state.shape != rate.shape or state.shape[-1:] != (self.dimension + 2,):
            raise ValueError("Compressible budget state/rate shapes are incompatible.")
        spatial_shape = state.shape[:-1]
        if pressure_.shape != spatial_shape:
            raise ValueError("Pressure must match the budget spatial shape.")
        state = eqx.error_if(
            state,
            jnp.any(~jnp.isfinite(state) | (state[..., :1] <= 0.0))
            | jnp.any(~jnp.isfinite(pressure_)),
            "Compressible budget inputs must be finite with positive density.",
        )
        weights_ = (
            jnp.ones(spatial_shape, dtype=state.dtype)
            if weights is None
            else jnp.asarray(weights, dtype=state.dtype)
        )
        if weights_.shape != spatial_shape:
            raise ValueError("Budget weights must match the spatial shape.")
        axes = tuple(range(len(spatial_shape)))
        density = state[..., 0]
        momentum_density = state[..., 1 : 1 + self.dimension]
        total_density = state[..., -1]
        velocity = momentum_density / density[..., None]
        speed_squared = oe.contract("...d,...d->...", velocity, velocity, backend="jax")
        kinetic_density = 0.5 * density * speed_squared
        internal_density = total_density - kinetic_density
        mass_rate_density = rate[..., 0]
        momentum_rate_density = rate[..., 1 : 1 + self.dimension]
        total_rate_density = rate[..., -1]
        kinetic_rate_density = (
            oe.contract("...d,...d->...", velocity, momentum_rate_density, backend="jax")
            - 0.5 * speed_squared * mass_rate_density
        )
        internal_rate_density = total_rate_density - kinetic_rate_density

        def scalar_term(value: ArrayLike | None) -> Array:
            if value is None:
                return jnp.zeros((), dtype=state.dtype)
            array = jnp.asarray(value, dtype=state.dtype)
            if array.shape != spatial_shape:
                raise ValueError("Scalar budget term must match the spatial shape.")
            return _weighted_sum(array, weights_, axes)

        def boundary_term(value: ArrayLike | None) -> Array:
            if value is None:
                return jnp.zeros((self.dimension + 2,), dtype=state.dtype)
            array = jnp.asarray(value, dtype=state.dtype)
            if array.shape == spatial_shape:
                return _weighted_sum(array, weights_, axes)
            if array.shape == state.shape:
                return _weighted_sum(array, weights_, axes)
            raise ValueError(
                "Boundary budget must be scalar work or a conserved-component field."
            )

        has_gradient = velocity_gradient is not None
        has_stress = viscous_stress is not None
        if has_gradient != has_stress:
            raise ValueError("Viscous budget requires both velocity gradient and stress.")
        if velocity_gradient is None:
            pressure_dilatation = jnp.zeros((), dtype=state.dtype)
            viscous = jnp.zeros((), dtype=state.dtype)
        else:
            gradient = jnp.asarray(velocity_gradient, dtype=state.dtype)
            stress = jnp.asarray(viscous_stress, dtype=state.dtype)
            expected = spatial_shape + (self.dimension, self.dimension)
            if gradient.shape != expected or stress.shape != expected:
                raise ValueError(
                    "Velocity gradient and viscous stress have wrong shapes."
                )
            divergence = jnp.trace(gradient, axis1=-2, axis2=-1)
            pressure_dilatation = _weighted_sum(pressure_ * divergence, weights_, axes)
            viscous_density = oe.contract(
                "...ij,...ij->...", stress, gradient, backend="jax"
            )
            viscous = _weighted_sum(viscous_density, weights_, axes)
        mass = _weighted_sum(density, weights_, axes)
        momentum = _weighted_sum(momentum_density, weights_, axes)
        total = _weighted_sum(total_density, weights_, axes)
        kinetic = _weighted_sum(kinetic_density, weights_, axes)
        internal = _weighted_sum(internal_density, weights_, axes)
        mass_rate_total = _weighted_sum(mass_rate_density, weights_, axes)
        momentum_rate_total = _weighted_sum(momentum_rate_density, weights_, axes)
        total_rate_total = _weighted_sum(total_rate_density, weights_, axes)
        kinetic_rate = _weighted_sum(kinetic_rate_density, weights_, axes)
        internal_rate = _weighted_sum(internal_rate_density, weights_, axes)
        decomposition = total_rate_total - kinetic_rate - internal_rate
        complete = jnp.asarray(
            velocity_gradient is not None
            and thermal_rate is not None
            and entropy_rate is not None
            and interface_rate is not None
            and filter_rate is not None
            and limiter_rate is not None
            and sponge_rate is not None
            and forcing_rate is not None
            and boundary_rate is not None
        )
        return CompressibleBudget(
            mass,
            momentum,
            total,
            kinetic,
            internal,
            mass_rate_total,
            momentum_rate_total,
            total_rate_total,
            kinetic_rate,
            internal_rate,
            pressure_dilatation,
            viscous,
            scalar_term(thermal_rate),
            scalar_term(entropy_rate),
            scalar_term(interface_rate),
            scalar_term(filter_rate),
            scalar_term(limiter_rate),
            scalar_term(sponge_rate),
            scalar_term(forcing_rate),
            boundary_term(boundary_rate),
            decomposition,
            complete,
            self.plan_id,
        )

    __call__ = evaluate


class CompressibleRawMoments(StrictModule):
    """Additive plane sums; blocks merge without reconstructing samples."""

    weight: Array
    density: Array
    density_squared: Array
    velocity: Array
    velocity_outer: Array
    density_velocity: Array
    density_velocity_outer: Array
    pressure: Array
    pressure_squared: Array
    temperature: Array
    temperature_squared: Array

    def merge(self, other: "CompressibleRawMoments", /) -> "CompressibleRawMoments":
        if not isinstance(other, CompressibleRawMoments):
            raise TypeError(
                "Raw compressible moments can merge only with their own type."
            )
        left = (
            self.weight,
            self.density,
            self.density_squared,
            self.velocity,
            self.velocity_outer,
            self.density_velocity,
            self.density_velocity_outer,
            self.pressure,
            self.pressure_squared,
            self.temperature,
            self.temperature_squared,
        )
        right = (
            other.weight,
            other.density,
            other.density_squared,
            other.velocity,
            other.velocity_outer,
            other.density_velocity,
            other.density_velocity_outer,
            other.pressure,
            other.pressure_squared,
            other.temperature,
            other.temperature_squared,
        )
        if any(a.shape != b.shape for a, b in zip(left, right, strict=True)):
            raise ValueError("Raw compressible moment blocks have different shapes.")
        return CompressibleRawMoments(*(a + b for a, b in zip(left, right, strict=True)))


class CompressiblePlaneStatistics(StrictModule):
    raw_moments: CompressibleRawMoments
    mean_density: Array
    reynolds_mean_velocity: Array
    favre_mean_velocity: Array
    reynolds_stress: Array
    favre_stress: Array
    mean_pressure: Array
    mean_temperature: Array
    mean_mach: Array
    mean_reynolds: Array
    solenoidal_spectrum: Array
    dilatational_spectrum: Array
    wall_shear: Array
    wall_heat_flux: Array
    wall_friction_velocity: Array
    wall_viscous_length: Array
    wall_y_plus: Array
    favre_identity_residual: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


class CompressiblePlaneStatisticsPlan(StrictModule, NonTrainableState):
    """Plane Reynolds/Favre statistics, mode split, and wall thermal units."""

    wall_normal_coordinates: Array | None
    dimension: int = eqx.field(static=True)
    wall_normal_axis: int | None = eqx.field(static=True)
    plane_axes: tuple[int, ...] = eqx.field(static=True)
    periodic_lengths: tuple[float, ...] = eqx.field(static=True)
    characteristic_length: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        wall_normal_axis: int | None = None,
        wall_normal_coordinates: ArrayLike | None = None,
        plane_axes: Sequence[int] | None = None,
        periodic_lengths: Sequence[float] | None = None,
        characteristic_length: float = 1.0,
    ):
        dimension_ = int(dimension)
        wall_axis = None if wall_normal_axis is None else int(wall_normal_axis)
        axes = (
            tuple(axis for axis in range(dimension_) if axis != wall_axis)
            if plane_axes is None
            else tuple(int(axis) for axis in plane_axes)
        )
        expected_axes = tuple(axis for axis in range(dimension_) if axis != wall_axis)
        lengths = (
            (1.0,) * len(axes)
            if periodic_lengths is None
            else tuple(float(value) for value in periodic_lengths)
        )
        length = float(characteristic_length)
        coordinates = (
            None
            if wall_normal_coordinates is None
            else jnp.asarray(wall_normal_coordinates)
        )
        if (
            dimension_ not in (1, 2, 3)
            or (wall_axis is not None and not 0 <= wall_axis < dimension_)
            or len(set(axes)) != len(axes)
            or any(not 0 <= axis < dimension_ or axis == wall_axis for axis in axes)
            or axes != expected_axes
            or len(lengths) != len(axes)
            or any(not np.isfinite(value) or value <= 0.0 for value in lengths)
            or not np.isfinite(length)
            or length <= 0.0
            or (
                wall_axis is not None
                and (
                    coordinates is None
                    or coordinates.ndim != 1
                    or coordinates.shape[0] < 2
                )
            )
            or (wall_axis is None and coordinates is not None)
        ):
            raise ValueError("Compressible plane-statistics plan is invalid.")
        if coordinates is not None:
            values = np.asarray(coordinates)
            if not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0.0):
                raise ValueError("Wall-normal coordinates must be finite and increasing.")
        self.dimension = dimension_
        self.wall_normal_axis = wall_axis
        self.wall_normal_coordinates = coordinates
        self.plane_axes = axes
        self.periodic_lengths = lengths
        self.characteristic_length = length
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compressible-plane-statistics",
                "dimension": dimension_,
                "wall_normal_axis": wall_axis,
                "wall_normal_coordinates": None
                if coordinates is None
                else tuple(float(value) for value in np.asarray(coordinates)),
                "plane_axes": axes,
                "periodic_lengths": lengths,
                "characteristic_length": length,
            }
        )

    def _spectra(self, velocity: Array, /) -> tuple[Array, Array]:
        if not self.plane_axes:
            shape = velocity.shape[:-1]
            return jnp.zeros(shape, dtype=velocity.dtype), jnp.zeros(
                shape, dtype=velocity.dtype
            )
        transformed = jnp.fft.fftn(velocity, axes=self.plane_axes)
        spatial_shape = velocity.shape[:-1]
        wave_components = []
        length_by_axis = dict(zip(self.plane_axes, self.periodic_lengths, strict=True))
        for component in range(self.dimension):
            if component in self.plane_axes:
                size = spatial_shape[component]
                frequency = (
                    2.0
                    * jnp.pi
                    * jnp.fft.fftfreq(size, d=length_by_axis[component] / size)
                )
                shape = [1] * self.dimension
                shape[component] = size
                wave_components.append(
                    jnp.broadcast_to(frequency.reshape(shape), spatial_shape)
                )
            else:
                wave_components.append(jnp.zeros(spatial_shape, dtype=velocity.dtype))
        wave = jnp.stack(wave_components, axis=-1)
        wave_squared = oe.contract("...d,...d->...", wave, wave, backend="jax")
        projection = oe.contract("...d,...d->...", transformed, wave, backend="jax")
        dilatational = jnp.where(
            (wave_squared > 0.0)[..., None],
            projection[..., None]
            * wave
            / jnp.where(wave_squared > 0.0, wave_squared, 1.0)[..., None],
            0.0,
        )
        solenoidal = transformed - dilatational
        normalization = (
            float(
                np.prod(tuple(spatial_shape[axis] for axis in self.plane_axes), dtype=int)
            )
            ** 2
        )
        solenoidal_energy = (
            0.5
            * jnp.real(
                oe.contract(
                    "...d,...d->...", jnp.conj(solenoidal), solenoidal, backend="jax"
                )
            )
            / normalization
        )
        dilatational_energy = (
            0.5
            * jnp.real(
                oe.contract(
                    "...d,...d->...", jnp.conj(dilatational), dilatational, backend="jax"
                )
            )
            / normalization
        )
        return solenoidal_energy, dilatational_energy

    def evaluate(
        self,
        conserved: ArrayLike,
        pressure: ArrayLike,
        temperature: ArrayLike,
        sound_speed: ArrayLike,
        dynamic_viscosity: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
        velocity_gradient: ArrayLike | None = None,
        thermal_conductivity: ArrayLike | None = None,
        temperature_gradient: ArrayLike | None = None,
    ) -> CompressiblePlaneStatistics:
        state = jnp.asarray(conserved)
        if state.ndim != self.dimension + 1 or state.shape[-1] != self.dimension + 2:
            raise ValueError("Statistics state must have one grid axis per dimension.")
        spatial_shape = state.shape[:-1]
        scalar_fields = tuple(
            jnp.asarray(value, dtype=state.dtype)
            for value in (pressure, temperature, sound_speed, dynamic_viscosity)
        )
        if any(value.shape != spatial_shape for value in scalar_fields):
            raise ValueError("Thermodynamic statistics fields must match the grid.")
        pressure_, temperature_, sound, viscosity = scalar_fields
        state = eqx.error_if(
            state,
            jnp.any(~jnp.isfinite(state) | (state[..., :1] <= 0.0))
            | jnp.any(~jnp.isfinite(pressure_))
            | jnp.any(~jnp.isfinite(temperature_))
            | jnp.any(~jnp.isfinite(sound) | (sound <= 0.0))
            | jnp.any(~jnp.isfinite(viscosity) | (viscosity <= 0.0)),
            "Statistics fields must be finite with positive density, sound speed, and viscosity.",
        )
        weights_ = (
            jnp.ones(spatial_shape, dtype=state.dtype)
            if weights is None
            else jnp.asarray(weights, dtype=state.dtype)
        )
        if weights_.shape != spatial_shape:
            raise ValueError("Statistics weights must match the grid.")
        weights_ = eqx.error_if(
            weights_,
            jnp.any(~jnp.isfinite(weights_) | (weights_ < 0.0)),
            "Statistics weights must be finite and nonnegative.",
        )
        density = state[..., 0]
        velocity = state[..., 1 : 1 + self.dimension] / density[..., None]
        axes = self.plane_axes
        weight_sum = _sum_axes(weights_, axes)
        weight_sum = eqx.error_if(
            weight_sum,
            jnp.any(weight_sum <= 0.0),
            "Every retained statistics plane must have positive total weight.",
        )
        density_sum = _weighted_sum(density, weights_, axes)
        density_squared_sum = _weighted_sum(density * density, weights_, axes)
        velocity_sum = _weighted_sum(velocity, weights_, axes)
        velocity_outer = velocity[..., :, None] * velocity[..., None, :]
        velocity_outer_sum = _weighted_sum(velocity_outer, weights_, axes)
        density_velocity_sum = _weighted_sum(
            density[..., None] * velocity, weights_, axes
        )
        density_velocity_outer_sum = _weighted_sum(
            density[..., None, None] * velocity_outer, weights_, axes
        )
        pressure_sum = _weighted_sum(pressure_, weights_, axes)
        temperature_sum = _weighted_sum(temperature_, weights_, axes)
        raw = CompressibleRawMoments(
            weight_sum,
            density_sum,
            density_squared_sum,
            velocity_sum,
            velocity_outer_sum,
            density_velocity_sum,
            density_velocity_outer_sum,
            pressure_sum,
            _weighted_sum(pressure_ * pressure_, weights_, axes),
            temperature_sum,
            _weighted_sum(temperature_ * temperature_, weights_, axes),
        )
        reynolds_mean = velocity_sum / weight_sum[..., None]
        favre_mean = density_velocity_sum / density_sum[..., None]
        reynolds_stress = velocity_outer_sum / weight_sum[..., None, None] - (
            reynolds_mean[..., :, None] * reynolds_mean[..., None, :]
        )
        favre_stress = density_velocity_outer_sum / density_sum[..., None, None] - (
            favre_mean[..., :, None] * favre_mean[..., None, :]
        )
        mean_density = density_sum / weight_sum
        mean_pressure = pressure_sum / weight_sum
        mean_temperature = temperature_sum / weight_sum
        speed = jnp.sqrt(oe.contract("...d,...d->...", velocity, velocity, backend="jax"))
        mean_mach = _weighted_sum(speed / sound, weights_, axes) / weight_sum
        reynolds_field = density * speed * self.characteristic_length / viscosity
        mean_reynolds = _weighted_sum(reynolds_field, weights_, axes) / weight_sum
        favre_residual = density_velocity_sum - density_sum[..., None] * favre_mean
        solenoidal, dilatational = self._spectra(velocity)
        if self.wall_normal_axis is None:
            wall_shear = jnp.zeros((0, self.dimension), dtype=state.dtype)
            wall_heat_flux = jnp.zeros((0,), dtype=state.dtype)
            friction_velocity = jnp.zeros((0,), dtype=state.dtype)
            viscous_length = jnp.zeros((0,), dtype=state.dtype)
            wall_y_plus = jnp.zeros((0, 0), dtype=state.dtype)
        else:
            if (
                velocity_gradient is None
                or thermal_conductivity is None
                or temperature_gradient is None
            ):
                raise ValueError(
                    "Wall statistics require velocity/temperature gradients and conductivity."
                )
            gradient = jnp.asarray(velocity_gradient, dtype=state.dtype)
            conductivity = jnp.asarray(thermal_conductivity, dtype=state.dtype)
            temperature_gradient_ = jnp.asarray(temperature_gradient, dtype=state.dtype)
            expected_gradient = spatial_shape + (self.dimension, self.dimension)
            if (
                gradient.shape != expected_gradient
                or conductivity.shape != spatial_shape
                or temperature_gradient_.shape != spatial_shape + (self.dimension,)
            ):
                raise ValueError("Wall-gradient fields have incompatible shapes.")
            divergence = jnp.trace(gradient, axis1=-2, axis2=-1)
            identity = jnp.eye(self.dimension, dtype=state.dtype)
            stress = viscosity[..., None, None] * (
                gradient
                + jnp.swapaxes(gradient, -1, -2)
                - (2.0 / 3.0) * divergence[..., None, None] * identity
            )
            wall_axis = self.wall_normal_axis
            reduced_axes = tuple(axis - int(axis > wall_axis) for axis in axes)

            def wall_mean(field: Array, index: int) -> Array:
                wall_field = jnp.take(field, index, axis=wall_axis)
                wall_weight = jnp.take(weights_, index, axis=wall_axis)
                denominator = _sum_axes(wall_weight, reduced_axes)
                return _weighted_sum(
                    wall_field, wall_weight, reduced_axes
                ) / denominator.reshape(
                    denominator.shape + (1,) * (wall_field.ndim - wall_weight.ndim)
                )

            shear_values = []
            heat_values = []
            friction_values = []
            length_values = []
            for index, outward_sign in ((0, -1.0), (-1, 1.0)):
                traction = outward_sign * stress[..., :, wall_axis]
                traction = traction.at[..., wall_axis].set(0.0)
                mean_traction = wall_mean(traction, index)
                temperature_normal = temperature_gradient_[..., wall_axis]
                heat_outward = -outward_sign * conductivity * temperature_normal
                mean_heat = wall_mean(heat_outward, index)
                density_wall = wall_mean(density, index)
                viscosity_wall = wall_mean(viscosity, index)
                shear_magnitude = jnp.sqrt(
                    oe.contract(
                        "...d,...d->...", mean_traction, mean_traction, backend="jax"
                    )
                )
                friction = jnp.sqrt(shear_magnitude / density_wall)
                shear_values.append(mean_traction)
                heat_values.append(mean_heat)
                friction_values.append(friction)
                length_values.append(viscosity_wall / (density_wall * friction))
            wall_shear = jnp.stack(shear_values)
            wall_heat_flux = jnp.stack(heat_values)
            friction_velocity = jnp.stack(friction_values)
            viscous_length = jnp.stack(length_values)
            coordinates = self.wall_normal_coordinates.astype(state.dtype)
            lower_distance = coordinates - coordinates[0]
            upper_distance = coordinates[-1] - coordinates
            wall_y_plus = jnp.stack(
                (
                    lower_distance / viscous_length[0],
                    upper_distance / viscous_length[1],
                )
            )
        finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value))
                    for value in (
                        mean_density,
                        reynolds_mean,
                        favre_mean,
                        reynolds_stress,
                        favre_stress,
                        mean_pressure,
                        mean_temperature,
                        mean_mach,
                        mean_reynolds,
                        solenoidal,
                        dilatational,
                        wall_shear,
                        wall_heat_flux,
                        friction_velocity,
                        viscous_length,
                        wall_y_plus,
                    )
                )
            )
        )
        return CompressiblePlaneStatistics(
            raw,
            mean_density,
            reynolds_mean,
            favre_mean,
            reynolds_stress,
            favre_stress,
            mean_pressure,
            mean_temperature,
            mean_mach,
            mean_reynolds,
            solenoidal,
            dilatational,
            wall_shear,
            wall_heat_flux,
            friction_velocity,
            viscous_length,
            wall_y_plus,
            favre_residual,
            finite,
            self.plan_id,
        )

    __call__ = evaluate


__all__ = [
    "CompressibleBudget",
    "CompressibleBudgetPlan",
    "CompressiblePlaneStatistics",
    "CompressiblePlaneStatisticsPlan",
    "CompressibleRawMoments",
]
