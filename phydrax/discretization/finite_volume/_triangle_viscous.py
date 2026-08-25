#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._physical_boundaries import PrescribedHeatFluxWallBoundary
from ._triangle_fv import TriangleFiniteVolumeDiscretization
from ._triangle_reconstruction import PreparedTriangleWLSQ


class TriangleViscousStabilityReport(StrictModule):
    maximum_momentum_rate: Array
    maximum_thermal_rate: Array
    selected_step: Array
    limiting_cell: Array


class TriangleViscousFluxPlan(StrictModule, NonTrainableState):
    gradient: PreparedTriangleWLSQ
    plan_id: str = eqx.field(static=True)

    def __init__(self, gradient: PreparedTriangleWLSQ, /):
        if not isinstance(gradient, PreparedTriangleWLSQ):
            raise TypeError("gradient must be PreparedTriangleWLSQ.")
        self.gradient = gradient
        self.plan_id = canonical_fingerprint(
            {"kind": "triangle-viscous-flux", "gradient": gradient.prepared_id}
        )

    def face_fluxes(
        self,
        system: Any,
        time: Array,
        state: Array,
        discretization: TriangleFiniteVolumeDiscretization,
        boundaries: Any,
        args: Any = None,
        /,
    ) -> Array:
        value = jnp.asarray(state)
        primitive = system.conserved_to_primitive(value)
        velocity = primitive[..., 1:-1]
        temperature = system.temperature(value)
        transport = system.transport.properties(temperature, value, args)
        velocity_gradient = self.gradient.gradient(velocity)
        temperature_gradient = self.gradient.gradient(temperature)
        owner = discretization.owner_cells
        neighbour = discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        interior = neighbour >= 0
        area_vectors = discretization.area_vectors.astype(value.dtype)
        face_measures = discretization.face_measures.astype(value.dtype)
        cell_centers = discretization.cell_centers.astype(value.dtype)
        face_centers = discretization.face_centers.astype(value.dtype)
        normal = area_vectors / face_measures[:, None]
        owner_center = cell_centers[owner]
        neighbour_center = cell_centers[safe_neighbour]
        connector = neighbour_center - owner_center
        projected_distance = jnp.sum(connector * normal, axis=-1)
        projected_distance = eqx.error_if(
            jnp.where(interior, projected_distance, 1.0),
            jnp.any(interior & (projected_distance <= 0.0)),
            "Triangle viscous interior face has nonpositive normal projection.",
        )
        average_velocity_gradient = 0.5 * (
            velocity_gradient[owner] + velocity_gradient[safe_neighbour]
        )
        average_temperature_gradient = 0.5 * (
            temperature_gradient[owner] + temperature_gradient[safe_neighbour]
        )
        tangential_connector = connector - projected_distance[:, None] * normal
        normal_velocity_derivative = (
            velocity[safe_neighbour]
            - velocity[owner]
            - oe.contract(
                "fij,fj->fi",
                average_velocity_gradient,
                tangential_connector,
            )
        ) / projected_distance[:, None]
        normal_temperature_derivative = (
            temperature[safe_neighbour]
            - temperature[owner]
            - jnp.sum(
                average_temperature_gradient * tangential_connector,
                axis=-1,
            )
        ) / projected_distance
        current_velocity_normal = oe.contract(
            "fij,fj->fi", average_velocity_gradient, normal
        )
        current_temperature_normal = jnp.sum(
            average_temperature_gradient * normal, axis=-1
        )
        face_velocity_gradient = (
            average_velocity_gradient
            + (normal_velocity_derivative - current_velocity_normal)[..., :, None]
            * normal[:, None, :]
        )
        face_temperature_gradient = (
            average_temperature_gradient
            + (normal_temperature_derivative - current_temperature_normal)[:, None]
            * normal
        )
        velocity_face = 0.5 * (velocity[owner] + velocity[safe_neighbour])
        viscosity_face = 0.5 * (
            transport.dynamic_viscosity[owner]
            + transport.dynamic_viscosity[safe_neighbour]
        )
        bulk_face = 0.5 * (
            transport.bulk_viscosity[owner] + transport.bulk_viscosity[safe_neighbour]
        )
        conductivity_face = 0.5 * (
            transport.thermal_conductivity[owner]
            + transport.thermal_conductivity[safe_neighbour]
        )
        boundary_mask = ~interior
        for patch_id, policy in enumerate(boundaries.boundaries):
            patch_mask = boundary_mask & (discretization.boundary_patch_ids == patch_id)
            exterior_state = policy.exterior_state(
                system,
                time,
                value[owner],
                face_centers,
                normal,
                0,
                args,
            )
            exterior_primitive = system.conserved_to_primitive(exterior_state)
            exterior_velocity = exterior_primitive[..., 1:-1]
            exterior_temperature = system.temperature(exterior_state)
            wall_projection = jnp.sum(
                (face_centers - owner_center) * normal,
                axis=-1,
            )
            wall_projection = eqx.error_if(
                wall_projection,
                jnp.any(patch_mask & (wall_projection <= 0.0)),
                "Triangle viscous boundary has nonpositive normal projection.",
            )
            full_distance = 2.0 * wall_projection
            boundary_velocity_derivative = (
                exterior_velocity - velocity[owner]
            ) / full_distance[:, None]
            boundary_temperature_derivative = (
                exterior_temperature - temperature[owner]
            ) / full_distance
            owner_velocity_gradient = velocity_gradient[owner]
            owner_temperature_gradient = temperature_gradient[owner]
            boundary_velocity_gradient = (
                owner_velocity_gradient
                + (
                    boundary_velocity_derivative
                    - oe.contract("fij,fj->fi", owner_velocity_gradient, normal)
                )[..., :, None]
                * normal[:, None, :]
            )
            boundary_temperature_gradient = (
                owner_temperature_gradient
                + (
                    boundary_temperature_derivative
                    - jnp.sum(owner_temperature_gradient * normal, axis=-1)
                )[:, None]
                * normal
            )
            face_velocity_gradient = jnp.where(
                patch_mask[:, None, None],
                boundary_velocity_gradient,
                face_velocity_gradient,
            )
            face_temperature_gradient = jnp.where(
                patch_mask[:, None],
                boundary_temperature_gradient,
                face_temperature_gradient,
            )
            velocity_face = jnp.where(
                patch_mask[:, None],
                0.5 * (velocity[owner] + exterior_velocity),
                velocity_face,
            )
            viscosity_face = jnp.where(
                patch_mask,
                transport.dynamic_viscosity[owner],
                viscosity_face,
            )
            bulk_face = jnp.where(patch_mask, transport.bulk_viscosity[owner], bulk_face)
            conductivity_face = jnp.where(
                patch_mask,
                transport.thermal_conductivity[owner],
                conductivity_face,
            )
        divergence = jnp.trace(face_velocity_gradient, axis1=-2, axis2=-1)
        identity = jnp.eye(system.dimension, dtype=value.dtype)
        stress = (
            viscosity_face[:, None, None]
            * (face_velocity_gradient + jnp.swapaxes(face_velocity_gradient, -1, -2))
            + (bulk_face - 2.0 * viscosity_face / 3.0)[:, None, None]
            * divergence[:, None, None]
            * identity
        )
        traction = oe.contract("fij,fj->fi", stress, normal)
        heat = conductivity_face * jnp.sum(face_temperature_gradient * normal, axis=-1)
        energy = jnp.sum(velocity_face * traction, axis=-1) + heat
        for patch_id, policy in enumerate(boundaries.boundaries):
            if not isinstance(policy, PrescribedHeatFluxWallBoundary):
                continue
            patch_mask = boundary_mask & (discretization.boundary_patch_ids == patch_id)
            prescribed = jnp.asarray(
                policy.normal_heat_flux(
                    time,
                    value[owner],
                    face_centers,
                    normal,
                    args,
                ),
                dtype=value.dtype,
            )
            mechanical = jnp.sum(
                policy.wall_velocity.astype(value.dtype) * traction,
                axis=-1,
            )
            energy = jnp.where(patch_mask, mechanical + prescribed, energy)
        return jnp.concatenate(
            (jnp.zeros_like(energy)[:, None], traction, energy[:, None]),
            axis=-1,
        )

    def residual(
        self,
        system: Any,
        time: Array,
        state: Array,
        discretization: TriangleFiniteVolumeDiscretization,
        boundaries: Any,
        args: Any = None,
        /,
    ) -> Array:
        flux = self.face_fluxes(system, time, state, discretization, boundaries, args)
        face_measures = discretization.face_measures.astype(state.dtype)
        cell_volumes = discretization.cell_volumes.astype(state.dtype)
        integrated = flux * face_measures[:, None]
        residual = jnp.zeros_like(state)
        residual = residual.at[discretization.owner_cells].add(integrated)
        neighbour = discretization.neighbour_cells
        residual = residual.at[jnp.maximum(neighbour, 0)].add(
            jnp.where((neighbour >= 0)[:, None], -integrated, 0.0)
        )
        return residual / cell_volumes[:, None]

    def stability_report(
        self,
        system: Any,
        state: Array,
        discretization: TriangleFiniteVolumeDiscretization,
        args: Any = None,
        /,
        *,
        safety: float = 0.45,
    ) -> TriangleViscousStabilityReport:
        primitive = system.conserved_to_primitive(state)
        density = primitive[..., 0]
        pressure = primitive[..., -1]
        temperature = system.temperature(state)
        transport = system.transport.properties(temperature, state, args)
        heat_capacity = system.material.specific_heat_cp(density, pressure)
        momentum_diffusivity = transport.dynamic_viscosity / density
        thermal_diffusivity = transport.thermal_conductivity / (density * heat_capacity)
        owner = discretization.owner_cells
        neighbour = discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        interior = neighbour >= 0
        area_vectors = discretization.area_vectors.astype(state.dtype)
        face_measures = discretization.face_measures.astype(state.dtype)
        cell_centers = discretization.cell_centers.astype(state.dtype)
        face_centers = discretization.face_centers.astype(state.dtype)
        cell_volumes = discretization.cell_volumes.astype(state.dtype)
        normal = area_vectors / face_measures[:, None]
        owner_to_neighbour = cell_centers[safe_neighbour] - cell_centers[owner]
        owner_to_face = face_centers - cell_centers[owner]
        interior_projection = jnp.sum(owner_to_neighbour * normal, axis=-1)
        boundary_projection = 2.0 * jnp.sum(owner_to_face * normal, axis=-1)
        distance = jnp.where(interior, interior_projection, boundary_projection)
        distance = eqx.error_if(
            distance,
            jnp.any(~jnp.isfinite(distance) | (distance <= 0.0)),
            "Triangle viscous stability requires positive normal distance.",
        )
        momentum_face = jnp.where(
            interior,
            0.5 * (momentum_diffusivity[owner] + momentum_diffusivity[safe_neighbour]),
            momentum_diffusivity[owner],
        )
        thermal_face = jnp.where(
            interior,
            0.5 * (thermal_diffusivity[owner] + thermal_diffusivity[safe_neighbour]),
            thermal_diffusivity[owner],
        )
        momentum_weight = 2.0 * face_measures * momentum_face / distance
        thermal_weight = 2.0 * face_measures * thermal_face / distance
        momentum_rate = jnp.zeros((discretization.cell_count,), dtype=state.dtype)
        thermal_rate = jnp.zeros_like(momentum_rate)
        momentum_rate = momentum_rate.at[owner].add(momentum_weight)
        thermal_rate = thermal_rate.at[owner].add(thermal_weight)
        momentum_rate = momentum_rate.at[safe_neighbour].add(
            jnp.where(interior, momentum_weight, 0.0)
        )
        thermal_rate = thermal_rate.at[safe_neighbour].add(
            jnp.where(interior, thermal_weight, 0.0)
        )
        momentum_rate /= cell_volumes
        thermal_rate /= cell_volumes
        maximum_momentum = jnp.max(momentum_rate)
        maximum_thermal = jnp.max(thermal_rate)
        selected = float(safety) / jnp.maximum(
            jnp.maximum(maximum_momentum, maximum_thermal),
            jnp.finfo(state.dtype).tiny,
        )
        return TriangleViscousStabilityReport(
            maximum_momentum_rate=maximum_momentum,
            maximum_thermal_rate=maximum_thermal,
            selected_step=selected,
            limiting_cell=jnp.argmax(jnp.maximum(momentum_rate, thermal_rate)),
        )


__all__ = ["TriangleViscousFluxPlan", "TriangleViscousStabilityReport"]
