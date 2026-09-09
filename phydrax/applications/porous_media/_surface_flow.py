#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class SurfaceFlowState(StrictModule):
    water_volume_m3: Array
    momentum_m4_s: Array
    time_s: Array
    plan_id: str = eqx.field(static=True)


class SurfaceFlowStepResult(StrictModule):
    state: SurfaceFlowState
    edge_volume_rate_m3_s: Array
    volume_balance_m3: Array
    successful: Array
    limited: Array
    derivative_available: Array


class UnstructuredShallowWaterPlan(StrictModule, NonTrainableState):
    """Hydrostatically reconstructed finite-volume shallow water on surface cells."""

    cell_areas_m2: Array
    bed_elevation_m: Array
    edge_owner: Array
    edge_neighbour: Array
    edge_outward_normal: Array
    edge_length_m: Array
    boundary_kind: Array
    boundary_depth_m: Array
    boundary_velocity_m_s: Array
    gravity_m_s2: float = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_areas_m2: ArrayLike,
        bed_elevation_m: ArrayLike,
        edge_owner: ArrayLike,
        edge_neighbour: ArrayLike,
        edge_outward_normal: ArrayLike,
        edge_length_m: ArrayLike,
        /,
        *,
        boundary_kind: ArrayLike | None = None,
        boundary_depth_m: ArrayLike = 0.0,
        boundary_velocity_m_s: ArrayLike = 0.0,
        gravity_m_s2: float = 9.80665,
    ):
        area, bed = (
            np.asarray(cell_areas_m2, dtype=float),
            np.asarray(bed_elevation_m, dtype=float),
        )
        owner, neighbour = np.asarray(edge_owner), np.asarray(edge_neighbour)
        normal, length = (
            np.asarray(edge_outward_normal, dtype=float),
            np.asarray(edge_length_m, dtype=float),
        )
        cells, edges = area.size, owner.size
        kind = (
            np.where(neighbour < 0, 1, 0).astype(np.int32)
            if boundary_kind is None
            else np.asarray(boundary_kind)
        )
        boundary_depth = np.broadcast_to(
            np.asarray(boundary_depth_m, dtype=float), (edges,)
        )
        boundary_velocity = np.broadcast_to(
            np.asarray(boundary_velocity_m_s, dtype=float), (edges, 2)
        )
        gravity = float(gravity_m_s2)
        if (
            area.ndim != 1
            or bed.shape != area.shape
            or cells == 0
            or owner.ndim != 1
            or neighbour.shape != owner.shape
            or normal.shape != (edges, 2)
            or length.shape != (edges,)
            or kind.shape != (edges,)
            or not np.issubdtype(owner.dtype, np.integer)
            or not np.issubdtype(neighbour.dtype, np.integer)
            or not np.issubdtype(kind.dtype, np.integer)
            or np.any(owner < 0)
            or np.any(owner >= cells)
            or np.any(neighbour >= cells)
            or np.any(neighbour == owner)
            or np.any(~np.isfinite(area))
            or np.any(area <= 0)
            or np.any(~np.isfinite(bed))
            or np.any(~np.isfinite(normal))
            or not np.allclose(np.sum(normal**2, axis=1), 1.0)
            or np.any(~np.isfinite(length))
            or np.any(length <= 0)
            or np.any((neighbour >= 0) & (kind != 0))
            or np.any((neighbour < 0) & ~np.isin(kind, (1, 2, 3)))
            or np.any(~np.isfinite(boundary_depth))
            or np.any(boundary_depth < 0)
            or np.any(~np.isfinite(boundary_velocity))
            or not np.isfinite(gravity)
            or gravity <= 0
        ):
            raise ValueError("Unstructured surface geometry/boundary data are invalid.")
        self.cell_areas_m2, self.bed_elevation_m = jnp.asarray(area), jnp.asarray(bed)
        self.edge_owner, self.edge_neighbour = jnp.asarray(owner), jnp.asarray(neighbour)
        self.edge_outward_normal, self.edge_length_m = (
            jnp.asarray(normal),
            jnp.asarray(length),
        )
        self.boundary_kind = jnp.asarray(kind, dtype=jnp.int32)
        self.boundary_depth_m, self.boundary_velocity_m_s = (
            jnp.asarray(boundary_depth),
            jnp.asarray(boundary_velocity),
        )
        self.gravity_m_s2, self.cell_count, self.edge_count = gravity, cells, edges
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-shallow-water-plan",
                "cell_areas_m2": area,
                "bed_elevation_m": bed,
                "edge_owner": owner,
                "edge_neighbour": neighbour,
                "edge_outward_normal": normal,
                "edge_length_m": length,
                "boundary_kind": kind,
                "boundary_depth_m": boundary_depth,
                "boundary_velocity_m_s": boundary_velocity,
                "gravity_m_s2": gravity,
            }
        )

    def initial_state(
        self, depth_m: ArrayLike, velocity_m_s: ArrayLike = 0.0, /
    ) -> SurfaceFlowState:
        depth = jnp.broadcast_to(jnp.asarray(depth_m), (self.cell_count,))
        velocity = jnp.broadcast_to(jnp.asarray(velocity_m_s), (self.cell_count, 2))
        depth = eqx.error_if(
            depth,
            jnp.any(~jnp.isfinite(depth))
            | jnp.any(depth < 0)
            | jnp.any(~jnp.isfinite(velocity)),
            "Surface depth/velocity must be finite and depth nonnegative.",
        )
        volume = depth * self.cell_areas_m2
        momentum = volume[:, None] * velocity
        return SurfaceFlowState(volume, momentum, jnp.asarray(0.0), self.plan_id)

    def _edge_states(self, state: SurfaceFlowState):
        depth = state.water_volume_m3 / self.cell_areas_m2
        velocity = state.momentum_m4_s / jnp.where(
            state.water_volume_m3[:, None] > 0,
            state.water_volume_m3[:, None],
            1.0,
        )
        owner = self.edge_owner
        interior = self.edge_neighbour >= 0
        neighbour = jnp.where(interior, self.edge_neighbour, owner)
        left_depth, right_depth = depth[owner], depth[neighbour]
        left_velocity, right_velocity = velocity[owner], velocity[neighbour]
        wall = self.boundary_kind == 1
        outflow = self.boundary_kind == 2
        inflow = self.boundary_kind == 3
        normal_velocity = jnp.sum(left_velocity * self.edge_outward_normal, axis=1)
        reflected = (
            left_velocity - 2.0 * normal_velocity[:, None] * self.edge_outward_normal
        )
        right_velocity = jnp.where(wall[:, None], reflected, right_velocity)
        right_velocity = jnp.where(outflow[:, None], left_velocity, right_velocity)
        right_velocity = jnp.where(
            inflow[:, None], self.boundary_velocity_m_s, right_velocity
        )
        right_depth = jnp.where((wall | outflow), left_depth, right_depth)
        right_depth = jnp.where(inflow, self.boundary_depth_m, right_depth)
        left_bed = self.bed_elevation_m[owner]
        right_bed = self.bed_elevation_m[neighbour]
        right_bed = jnp.where(interior, right_bed, left_bed)
        interface_bed = jnp.maximum(left_bed, right_bed)
        left_star = jnp.maximum(0.0, left_depth + left_bed - interface_bed)
        right_star = jnp.maximum(0.0, right_depth + right_bed - interface_bed)
        return (
            left_depth,
            right_depth,
            left_star,
            right_star,
            left_velocity,
            right_velocity,
            interior,
            neighbour,
        )

    def _fluxes(self, state: SurfaceFlowState):
        (
            left_depth,
            right_depth,
            left,
            right,
            left_velocity,
            right_velocity,
            interior,
            neighbour,
        ) = self._edge_states(state)
        normal = self.edge_outward_normal
        left_normal = jnp.sum(left_velocity * normal, axis=1)
        right_normal = jnp.sum(right_velocity * normal, axis=1)
        left_state = jnp.concatenate(
            (left[:, None], left[:, None] * left_velocity), axis=1
        )
        right_state = jnp.concatenate(
            (right[:, None], right[:, None] * right_velocity), axis=1
        )
        left_flux = jnp.concatenate(
            (
                (left * left_normal)[:, None],
                left[:, None] * left_normal[:, None] * left_velocity
                + 0.5 * self.gravity_m_s2 * left[:, None] ** 2 * normal,
            ),
            axis=1,
        )
        right_flux = jnp.concatenate(
            (
                (right * right_normal)[:, None],
                right[:, None] * right_normal[:, None] * right_velocity
                + 0.5 * self.gravity_m_s2 * right[:, None] ** 2 * normal,
            ),
            axis=1,
        )
        wave = jnp.maximum(
            jnp.abs(left_normal) + jnp.sqrt(self.gravity_m_s2 * left),
            jnp.abs(right_normal) + jnp.sqrt(self.gravity_m_s2 * right),
        )
        common = 0.5 * (left_flux + right_flux) - 0.5 * wave[:, None] * (
            right_state - left_state
        )
        common = common * self.edge_length_m[:, None]
        left_correction = (
            0.5 * self.gravity_m_s2 * (left_depth**2 - left**2) * self.edge_length_m
        )[:, None] * normal
        right_correction = (
            0.5 * self.gravity_m_s2 * (right_depth**2 - right**2) * self.edge_length_m
        )[:, None] * normal
        owner_flux = common.at[:, 1:].add(left_correction)
        neighbour_flux = (-common).at[:, 1:].add(-right_correction)
        return owner_flux, neighbour_flux, interior, neighbour

    def step(
        self,
        state: SurfaceFlowState,
        dt_s: ArrayLike,
        /,
        *,
        rainfall_m_s: ArrayLike = 0.0,
        infiltration_m3_s: ArrayLike = 0.0,
    ) -> SurfaceFlowStepResult:
        if not isinstance(state, SurfaceFlowState) or state.plan_id != self.plan_id:
            raise ValueError("Surface-flow state belongs to a different plan.")
        dt = jnp.asarray(dt_s)
        rain = jnp.broadcast_to(jnp.asarray(rainfall_m_s), (self.cell_count,))
        infiltration = jnp.broadcast_to(
            jnp.asarray(infiltration_m3_s), (self.cell_count,)
        )
        dt = eqx.error_if(
            dt,
            ~jnp.isfinite(dt)
            | (dt <= 0)
            | jnp.any(~jnp.isfinite(rain))
            | jnp.any(rain < 0)
            | jnp.any(~jnp.isfinite(infiltration)),
            "Surface timestep/rainfall/infiltration must be finite and physical.",
        )
        owner_flux, neighbour_flux, interior, neighbour = self._fluxes(state)
        outgoing = jnp.zeros((self.cell_count,))
        outgoing = outgoing.at[self.edge_owner].add(jnp.maximum(owner_flux[:, 0], 0.0))
        outgoing = outgoing.at[neighbour].add(
            jnp.where(interior, jnp.maximum(neighbour_flux[:, 0], 0.0), 0.0)
        )
        available_rate = (
            state.water_volume_m3 / dt + rain * self.cell_areas_m2 - infiltration
        )
        available_rate = eqx.error_if(
            available_rate,
            jnp.any(available_rate < 0),
            "Infiltration demand exceeds available surface storage and rainfall.",
        )
        scale = jnp.minimum(1.0, available_rate / jnp.where(outgoing > 0, outgoing, 1.0))
        owner_scale = jnp.where(owner_flux[:, 0] > 0, scale[self.edge_owner], 1.0)
        neighbour_scale = jnp.where(
            interior & (neighbour_flux[:, 0] > 0), scale[neighbour], 1.0
        )
        edge_scale = jnp.minimum(owner_scale, neighbour_scale)
        owner_flux, neighbour_flux = (
            owner_flux * edge_scale[:, None],
            neighbour_flux * edge_scale[:, None],
        )
        content_rate = jnp.zeros((self.cell_count, 3))
        content_rate = content_rate.at[self.edge_owner].add(owner_flux)
        content_rate = content_rate.at[neighbour].add(
            jnp.where(interior[:, None], neighbour_flux, 0.0)
        )
        source_volume = rain * self.cell_areas_m2 - infiltration
        volume = state.water_volume_m3 + dt * (source_volume - content_rate[:, 0])
        momentum = state.momentum_m4_s - dt * content_rate[:, 1:]
        tolerance = (
            100
            * jnp.finfo(volume.dtype).eps
            * jnp.maximum(jnp.max(state.water_volume_m3), 1.0)
        )
        successful = (
            jnp.all(jnp.isfinite(volume))
            & jnp.all(jnp.isfinite(momentum))
            & jnp.all(volume >= -tolerance)
        )
        balance = jnp.sum(volume - state.water_volume_m3) - dt * (
            jnp.sum(source_volume)
            - jnp.sum(jnp.where(self.edge_neighbour < 0, owner_flux[:, 0], 0.0))
        )
        limited = jnp.any(edge_scale < 1.0)
        return SurfaceFlowStepResult(
            SurfaceFlowState(volume, momentum, state.time_s + dt, self.plan_id),
            owner_flux[:, 0],
            balance,
            successful,
            limited,
            successful & ~limited & jnp.all(volume > tolerance),
        )


__all__ = ["SurfaceFlowState", "SurfaceFlowStepResult", "UnstructuredShallowWaterPlan"]
