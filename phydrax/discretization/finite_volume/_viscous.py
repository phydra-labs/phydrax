#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._halo import PreparedFiniteVolumeHaloPlan
from ._mapped import MappedFiniteVolumeDiscretization
from ._physical_boundaries import PrescribedHeatFluxWallBoundary
from ._structured import FiniteVolumeDiscretization


def _cell_gradient(
    values: Array,
    coordinates: Array,
    axis: int,
    periodic: bool,
    /,
) -> Array:
    moved = jnp.moveaxis(values, axis, 0)
    coordinate = jnp.asarray(coordinates)
    if periodic:
        period = coordinate[-1] - coordinate[0] + (coordinate[1] - coordinate[0])
        previous_coordinate = jnp.roll(coordinate, 1).at[0].add(-period)
        next_coordinate = jnp.roll(coordinate, -1).at[-1].add(period)
        denominator = next_coordinate - previous_coordinate
        gradient = (
            jnp.roll(moved, -1, axis=0) - jnp.roll(moved, 1, axis=0)
        ) / denominator.reshape((denominator.size,) + (1,) * (moved.ndim - 1))
    else:
        if moved.shape[0] == 1:
            gradient = jnp.zeros_like(moved)
        else:
            forward = (moved[1:] - moved[:-1]) / (
                coordinate[1:] - coordinate[:-1]
            ).reshape((-1,) + (1,) * (moved.ndim - 1))
            interior = (
                0.5 * (forward[:-1] + forward[1:])
                if moved.shape[0] > 2
                else jnp.empty((0,) + moved.shape[1:], dtype=moved.dtype)
            )
            gradient = jnp.concatenate((forward[:1], interior, forward[-1:]), axis=0)
    return jnp.moveaxis(gradient, 0, axis)


def _cell_to_faces(values: Array, axis: int, periodic: bool, /) -> Array:
    moved = jnp.moveaxis(values, axis, 0)
    if periodic:
        faces = 0.5 * (moved + jnp.roll(moved, 1, axis=0))
    else:
        interior = 0.5 * (moved[:-1] + moved[1:])
        faces = jnp.concatenate((moved[:1], interior, moved[-1:]), axis=0)
    return jnp.moveaxis(faces, 0, axis)


def _ghosted_center_gradient(
    values: Array,
    coordinates: Array,
    axis: int,
    depth: int,
    interior_count: int,
    /,
) -> Array:
    moved = jnp.moveaxis(values, axis, 0)
    denominator = (
        coordinates[depth + 1 : depth + interior_count + 1]
        - coordinates[depth - 1 : depth + interior_count - 1]
    )
    shape = (interior_count,) + (1,) * (moved.ndim - 1)
    gradient = (
        moved[depth + 1 : depth + interior_count + 1]
        - moved[depth - 1 : depth + interior_count - 1]
    ) / denominator.reshape(shape)
    return jnp.moveaxis(gradient, 0, axis)


def _mapped_cell_gradient(values: Array, centers: Array, /) -> Array:
    spatial_shape = centers.shape[:-1]
    dimension = centers.shape[-1]
    value = jnp.asarray(values)
    scalar = value.ndim == len(spatial_shape)
    components = value[..., None] if scalar else value
    matrix = jnp.zeros(spatial_shape + (dimension, dimension), dtype=value.dtype)
    right_hand_side = jnp.zeros(
        spatial_shape + (components.shape[-1], dimension), dtype=value.dtype
    )
    for axis in range(dimension):
        for shift in (-1, 1):
            neighbor_centers = jnp.roll(centers, shift, axis=axis)
            neighbor_values = jnp.roll(components, shift, axis=axis)
            if shift < 0:
                boundary_index = centers.shape[axis] - 1
            else:
                boundary_index = 0
            selector: list[slice | int] = [slice(None)] * centers.ndim
            selector[axis] = boundary_index
            neighbor_centers = neighbor_centers.at[tuple(selector)].set(
                centers[tuple(selector)]
            )
            value_selector: list[slice | int] = [slice(None)] * components.ndim
            value_selector[axis] = boundary_index
            neighbor_values = neighbor_values.at[tuple(value_selector)].set(
                components[tuple(value_selector)]
            )
            displacement = neighbor_centers - centers
            difference = neighbor_values - components
            matrix = matrix + displacement[..., :, None] * displacement[..., None, :]
            right_hand_side = (
                right_hand_side + difference[..., :, None] * displacement[..., None, :]
            )
    regularization = jnp.finfo(value.dtype).eps * jnp.eye(dimension)
    gradient = jnp.linalg.solve(
        matrix + regularization,
        jnp.swapaxes(right_hand_side, -1, -2),
    )
    gradient = jnp.swapaxes(gradient, -1, -2)
    return gradient[..., 0, :] if scalar else gradient


def _mapped_halo_gradient(
    system: Any,
    time: Array,
    state: Array,
    discretization: MappedFiniteVolumeDiscretization,
    halo: PreparedFiniteVolumeHaloPlan,
    args: Any,
    field: str,
    /,
) -> Array:
    dimension = system.dimension
    scalar = field == "temperature"
    components = 1 if scalar else dimension
    matrix = jnp.zeros(
        discretization.cell_shape + (dimension, dimension),
        dtype=state.dtype,
    )
    right_hand_side = jnp.zeros(
        discretization.cell_shape + (components, dimension),
        dtype=state.dtype,
    )
    for axis in range(dimension):
        ghosted = halo.materialize_axis(system, time, state, axis, args)
        primitive = system.conserved_to_primitive(ghosted.values)
        values = (
            system.temperature(ghosted.values)[..., None]
            if scalar
            else primitive[..., 1:-1]
        )
        moved_values = jnp.moveaxis(values, axis, 0)
        moved_centers = jnp.moveaxis(ghosted.physical_centers, axis, 0)
        depth = ghosted.depth
        count = discretization.cell_shape[axis]
        current_values = moved_values[depth : depth + count]
        current_centers = moved_centers[depth : depth + count]
        for offset in (-1, 1):
            neighbor_values = moved_values[depth + offset : depth + count + offset]
            neighbor_centers = moved_centers[depth + offset : depth + count + offset]
            displacement = jnp.moveaxis(neighbor_centers - current_centers, 0, axis)
            difference = jnp.moveaxis(neighbor_values - current_values, 0, axis)
            matrix = matrix + displacement[..., :, None] * displacement[..., None, :]
            right_hand_side = (
                right_hand_side + difference[..., :, None] * displacement[..., None, :]
            )
    regularization = jnp.finfo(state.dtype).eps * jnp.eye(dimension)
    gradient = jnp.linalg.solve(
        matrix + regularization,
        jnp.swapaxes(right_hand_side, -1, -2),
    )
    gradient = jnp.swapaxes(gradient, -1, -2)
    return gradient[..., 0, :] if scalar else gradient


class ViscousStabilityReport(StrictModule):
    maximum_momentum_rate: Array
    maximum_thermal_rate: Array
    momentum_step: Array
    thermal_step: Array
    selected_step: Array
    limiting_cell_flat_index: Array


class ViscousFluxPlan(StrictModule, NonTrainableState):
    """Material-owned Newtonian stress and Fourier heat flux."""

    plan_id: str = eqx.field(static=True)

    def __init__(self):
        self.plan_id = canonical_fingerprint({"kind": "material-viscous-flux"})

    def _apply_prescribed_heat_flux(
        self,
        system: Any,
        time: Array,
        state: Array,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        halo: PreparedFiniteVolumeHaloPlan,
        fluxes: tuple[Array, ...],
        args: Any,
        /,
    ) -> tuple[Array, ...]:
        output = list(fluxes)
        for axis, pair in enumerate(halo.plan.boundaries.pairs):
            if pair is None:
                continue
            for side, boundary in (
                ("lower", pair.lower),
                ("upper", pair.upper),
            ):
                if not isinstance(boundary, PrescribedHeatFluxWallBoundary):
                    continue
                face_index = 0 if side == "lower" else output[axis].shape[axis] - 1
                cell_index = 0 if side == "lower" else state.shape[axis] - 1
                interior = jnp.take(state, cell_index, axis=axis)
                coordinates = jnp.take(
                    discretization.face_centers[axis],
                    face_index,
                    axis=axis,
                )
                normal = discretization.outward_normal(axis, side)
                outward_heat = boundary.normal_heat_flux(
                    time, interior, coordinates, normal, args
                )
                face_flux = jnp.take(output[axis], face_index, axis=axis)
                traction = face_flux[..., 1 : 1 + system.dimension]
                mechanical = jnp.sum(boundary.wall_velocity * traction, axis=-1)
                sign = -1.0 if side == "lower" else 1.0
                replacement = face_flux.at[..., -1].set(mechanical + sign * outward_heat)
                index: list[slice | int] = [slice(None)] * output[axis].ndim
                index[axis] = face_index
                output[axis] = output[axis].at[tuple(index)].set(replacement)
        return tuple(output)

    def _mapped_face_fluxes(
        self,
        system: Any,
        time: Array,
        value: Array,
        discretization: MappedFiniteVolumeDiscretization,
        halo: PreparedFiniteVolumeHaloPlan,
        args: Any,
        /,
    ) -> tuple[Array, ...]:
        if any(axis.periodic for axis in discretization.grid.structured_axes):
            raise ValueError(
                "Mapped viscous flux currently requires bounded structured axes."
            )
        primitive = system.conserved_to_primitive(value)
        velocity = primitive[..., 1:-1]
        temperature = system.temperature(value)
        transport = system.transport.properties(temperature, value, args)
        velocity_gradient = _mapped_halo_gradient(
            system, time, value, discretization, halo, args, "velocity"
        )
        temperature_gradient = _mapped_halo_gradient(
            system, time, value, discretization, halo, args, "temperature"
        )
        divergence = jnp.trace(velocity_gradient, axis1=-2, axis2=-1)
        identity = jnp.eye(system.dimension, dtype=value.dtype)
        stress = (
            transport.dynamic_viscosity[..., None, None]
            * (velocity_gradient + jnp.swapaxes(velocity_gradient, -1, -2))
            + (transport.bulk_viscosity - 2.0 * transport.dynamic_viscosity / 3.0)[
                ..., None, None
            ]
            * divergence[..., None, None]
            * identity
        )
        output = []
        for axis in range(system.dimension):
            stress_face = _cell_to_faces(stress, axis, False)
            velocity_face = _cell_to_faces(velocity, axis, False)
            temperature_gradient_face = _cell_to_faces(temperature_gradient, axis, False)
            conductivity_face = _cell_to_faces(
                transport.thermal_conductivity, axis, False
            )
            normal = (
                discretization.face_area_vectors[axis]
                / discretization.face_measures[axis][..., None]
            )
            traction = oe.contract("...ij,...j->...i", stress_face, normal)
            normal_temperature_gradient = jnp.sum(
                temperature_gradient_face * normal, axis=-1
            )
            energy_flux = (
                jnp.sum(velocity_face * traction, axis=-1)
                + conductivity_face * normal_temperature_gradient
            )
            output.append(
                jnp.concatenate(
                    (
                        jnp.zeros_like(energy_flux)[..., None],
                        traction,
                        energy_flux[..., None],
                    ),
                    axis=-1,
                )
            )
        return self._apply_prescribed_heat_flux(
            system, time, value, discretization, halo, tuple(output), args
        )

    def face_fluxes(
        self,
        system: Any,
        time: Array,
        state: ArrayLike,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        halo: PreparedFiniteVolumeHaloPlan,
        args: Any = None,
        /,
    ) -> tuple[Array, ...]:
        value = jnp.asarray(state)
        if isinstance(discretization, MappedFiniteVolumeDiscretization):
            return self._mapped_face_fluxes(
                system, time, value, discretization, halo, args
            )
        if system.component_count != system.dimension + 2:
            raise TypeError("Viscous flux requires a compressible-flow state layout.")
        primitive = system.conserved_to_primitive(value)
        velocity = primitive[..., 1:-1]
        temperature = system.temperature(value)
        transport = system.transport.properties(temperature, value, args)
        dimension = system.dimension
        velocity_gradients = []
        temperature_gradients = []
        for axis in range(dimension):
            ghosted = halo.materialize_axis(system, time, value, axis, args)
            ghosted_primitive = system.conserved_to_primitive(ghosted.values)
            velocity_gradients.append(
                _ghosted_center_gradient(
                    ghosted_primitive[..., 1:-1],
                    ghosted.axis_coordinates,
                    axis,
                    ghosted.depth,
                    discretization.cell_shape[axis],
                )
            )
            temperature_gradients.append(
                _ghosted_center_gradient(
                    system.temperature(ghosted.values),
                    ghosted.axis_coordinates,
                    axis,
                    ghosted.depth,
                    discretization.cell_shape[axis],
                )
            )
        velocity_gradients = tuple(velocity_gradients)
        temperature_gradients = tuple(temperature_gradients)
        divergence = jnp.sum(
            jnp.stack(
                tuple(velocity_gradients[axis][..., axis] for axis in range(dimension)),
                axis=0,
            ),
            axis=0,
        )
        output = []
        for normal_axis in range(dimension):
            periodic = discretization.grid.structured_axes[normal_axis].periodic
            velocity_face = _cell_to_faces(velocity, normal_axis, periodic)
            divergence_face = _cell_to_faces(divergence, normal_axis, periodic)
            viscosity_face = _cell_to_faces(
                transport.dynamic_viscosity, normal_axis, periodic
            )
            bulk_face = _cell_to_faces(transport.bulk_viscosity, normal_axis, periodic)
            conductivity_face = _cell_to_faces(
                transport.thermal_conductivity, normal_axis, periodic
            )
            lambda_face = bulk_face - 2.0 * viscosity_face / 3.0
            stress_components = []
            for component in range(dimension):
                derivative_normal = _cell_to_faces(
                    velocity_gradients[normal_axis][..., component],
                    normal_axis,
                    periodic,
                )
                derivative_component = _cell_to_faces(
                    velocity_gradients[component][..., normal_axis],
                    normal_axis,
                    periodic,
                )
                stress = viscosity_face * (derivative_normal + derivative_component)
                if component == normal_axis:
                    stress = stress + lambda_face * divergence_face
                stress_components.append(stress)
            stress_vector = jnp.stack(tuple(stress_components), axis=-1)
            heat_gradient = _cell_to_faces(
                temperature_gradients[normal_axis], normal_axis, periodic
            )
            energy_flux = (
                jnp.sum(velocity_face * stress_vector, axis=-1)
                + conductivity_face * heat_gradient
            )
            output.append(
                jnp.concatenate(
                    (
                        jnp.zeros_like(energy_flux)[..., None],
                        stress_vector,
                        energy_flux[..., None],
                    ),
                    axis=-1,
                )
            )
        return self._apply_prescribed_heat_flux(
            system, time, value, discretization, halo, tuple(output), args
        )

    def stability_report(
        self,
        system: Any,
        state: ArrayLike,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        args: Any = None,
        /,
        *,
        safety: float = 0.45,
    ) -> ViscousStabilityReport:
        value = jnp.asarray(state)
        primitive = system.conserved_to_primitive(value)
        density = primitive[..., 0]
        pressure = primitive[..., -1]
        temperature = system.temperature(value)
        transport = system.transport.properties(temperature, value, args)
        heat_capacity = system.material.specific_heat_cp(density, pressure)
        momentum_diffusivity = transport.dynamic_viscosity / density
        thermal_diffusivity = transport.thermal_conductivity / (density * heat_capacity)
        momentum_rate = jnp.zeros(discretization.cell_shape, dtype=value.dtype)
        thermal_rate = jnp.zeros_like(momentum_rate)
        for axis in range(system.dimension):
            periodic = discretization.grid.structured_axes[axis].periodic
            momentum_face = _cell_to_faces(momentum_diffusivity, axis, periodic)
            thermal_face = _cell_to_faces(thermal_diffusivity, axis, periodic)
            measure = discretization.face_measures[axis]
            if periodic:
                widths = discretization.grid.structured_axes[axis].interval_widths
                distance = 0.5 * (widths + jnp.roll(widths, 1))
                shape = [1] * measure.ndim
                shape[axis] = distance.size
                distance = jnp.broadcast_to(distance.reshape(tuple(shape)), measure.shape)
            else:
                centers = jnp.moveaxis(discretization.cell_centers, axis, 0)
                face_centers = jnp.moveaxis(discretization.face_centers[axis], axis, 0)
                normals = jnp.moveaxis(
                    discretization.face_area_vectors[axis] / measure[..., None],
                    axis,
                    0,
                )
                lower_distance = 2.0 * jnp.abs(
                    jnp.sum(
                        (centers[0] - face_centers[0]) * normals[0],
                        axis=-1,
                    )
                )
                upper_distance = 2.0 * jnp.abs(
                    jnp.sum(
                        (face_centers[-1] - centers[-1]) * normals[-1],
                        axis=-1,
                    )
                )
                interior_distance = jnp.abs(
                    jnp.sum(
                        (centers[1:] - centers[:-1]) * normals[1:-1],
                        axis=-1,
                    )
                )
                distance = jnp.moveaxis(
                    jnp.concatenate(
                        (
                            lower_distance[None, ...],
                            interior_distance,
                            upper_distance[None, ...],
                        ),
                        axis=0,
                    ),
                    0,
                    axis,
                )
            distance = eqx.error_if(
                distance,
                jnp.any(~jnp.isfinite(distance) | (distance <= 0.0)),
                "Viscous stability requires finite positive face distance.",
            )
            momentum_weight = 2.0 * measure * momentum_face / distance
            thermal_weight = 2.0 * measure * thermal_face / distance
            if periodic:
                momentum_contribution = momentum_weight + jnp.roll(
                    momentum_weight, -1, axis=axis
                )
                thermal_contribution = thermal_weight + jnp.roll(
                    thermal_weight, -1, axis=axis
                )
            else:
                lower: list[slice | int] = [slice(None)] * measure.ndim
                upper: list[slice | int] = [slice(None)] * measure.ndim
                lower[axis] = slice(0, measure.shape[axis] - 1)
                upper[axis] = slice(1, measure.shape[axis])
                momentum_contribution = (
                    momentum_weight[tuple(lower)] + momentum_weight[tuple(upper)]
                )
                thermal_contribution = (
                    thermal_weight[tuple(lower)] + thermal_weight[tuple(upper)]
                )
            momentum_rate = (
                momentum_rate + momentum_contribution / discretization.cell_volumes
            )
            thermal_rate = (
                thermal_rate + thermal_contribution / discretization.cell_volumes
            )
        maximum_momentum = jnp.max(momentum_rate)
        maximum_thermal = jnp.max(thermal_rate)
        safety_ = jnp.asarray(safety, dtype=value.dtype)
        momentum_step = jnp.where(
            maximum_momentum > 0.0,
            safety_ / maximum_momentum,
            jnp.inf,
        )
        thermal_step = jnp.where(
            maximum_thermal > 0.0,
            safety_ / maximum_thermal,
            jnp.inf,
        )
        combined_rate = jnp.maximum(momentum_rate, thermal_rate)
        return ViscousStabilityReport(
            maximum_momentum_rate=maximum_momentum,
            maximum_thermal_rate=maximum_thermal,
            momentum_step=momentum_step,
            thermal_step=thermal_step,
            selected_step=jnp.minimum(momentum_step, thermal_step),
            limiting_cell_flat_index=jnp.argmax(combined_rate),
        )

    def stable_step(
        self,
        system: Any,
        state: ArrayLike,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        args: Any = None,
        /,
        *,
        safety: float = 0.45,
    ) -> Array:
        return self.stability_report(
            system, state, discretization, args, safety=safety
        ).selected_step

    def residual(
        self,
        system: Any,
        time: Array,
        state: ArrayLike,
        discretization: FiniteVolumeDiscretization | MappedFiniteVolumeDiscretization,
        halo: PreparedFiniteVolumeHaloPlan,
        args: Any = None,
        /,
    ) -> Array:
        fluxes = self.face_fluxes(system, time, state, discretization, halo, args)
        residual = jnp.zeros_like(jnp.asarray(state))
        for axis, flux in enumerate(fluxes):
            integrated = flux * discretization.face_measures[axis][..., None]
            if discretization.grid.structured_axes[axis].periodic:
                difference = jnp.roll(integrated, -1, axis=axis) - integrated
            else:
                lower: list[slice | int] = [slice(None)] * integrated.ndim
                upper: list[slice | int] = [slice(None)] * integrated.ndim
                lower[axis] = slice(0, integrated.shape[axis] - 1)
                upper[axis] = slice(1, integrated.shape[axis])
                difference = integrated[tuple(upper)] - integrated[tuple(lower)]
            residual = residual + difference / discretization.cell_volumes[..., None]
        return residual


__all__ = ["ViscousFluxPlan", "ViscousStabilityReport"]
