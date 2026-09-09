#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ...discretization._boundary_trace import BoundarySurfaceTrace
from ...units import METER


class SurfaceWaterState(StrictModule):
    """Incompressible liquid volume (m³) and sensible energy (J)."""

    volume: Array
    energy: Array


class SurfaceExchangeResult(StrictModule):
    state: SurfaceWaterState
    infiltration_rate: Array
    infiltration_energy_rate: Array
    lateral_rate: Array
    volume_residual: Array
    energy_residual: Array
    limited: Array
    derivative_available: Array
    successful: Array


class OrthogonalDiffusiveWaveSurfacePlan(StrictModule):
    """Conservative diffusive-wave runoff on qualified projected-orthogonal cells.

    Depth is vertical and storage uses projected horizontal area. The donor
    budget limits *rates*, never clips inventories: each accepted edge or
    interface rate is used unchanged on both sides. The lateral perimeter is
    impermeable. Exchange conductance is m²/s, head and depth are meters.
    This explicit reservoir model admits mathematically one-way partitioning;
    shared pressure unknowns require the monolithic coupling below.
    """

    trace: BoundarySurfaceTrace
    projected_areas: Array
    bed: Array
    edge_owner: Array
    edge_neighbour: Array
    edge_width: Array
    edge_distance: Array
    manning: Array
    density: Array
    heat_capacity: Array
    reference_temperature: Array
    lateral: bool = eqx.field(static=True)

    def __init__(
        self,
        trace: BoundarySurfaceTrace,
        *,
        spatial: SpatialCoordinateContract,
        manning: ArrayLike = 0.03,
        density: float = 1000.0,
        heat_capacity: float = 4180.0,
        reference_temperature: float = 273.15,
        lateral: bool = True,
        orthogonality_tolerance: float = 1e-8,
    ):
        if not isinstance(trace, BoundarySurfaceTrace):
            raise TypeError("Surface runoff requires a BoundarySurfaceTrace.")
        if (
            spatial.length_unit.unit_id != METER.unit_id
            or spatial.length_coordinate_kind != "physical"
            or spatial.coordinate_system != "cartesian"
        ):
            raise ValueError(
                "Surface runoff requires physical Cartesian meter coordinates."
            )
        if not np.isfinite(orthogonality_tolerance) or orthogonality_tolerance <= 0:
            raise ValueError("Orthogonality tolerance must be positive and finite.")
        normals = np.asarray(trace.normals)
        if np.any(normals[:, 2] <= 0):
            raise ValueError("Surface storage requires upward-facing boundary cells.")
        centers = np.asarray(trace.centers)
        edge_cells = np.asarray(trace.edge_cells)
        interior = edge_cells[:, 1] >= 0
        pairs = edge_cells[interior]
        vertices = np.asarray(trace.vertices)
        endpoints = np.asarray(trace.edge_vertices)[interior]
        vectors = vertices[endpoints[:, 1], :2] - vertices[endpoints[:, 0], :2]
        widths = np.sqrt(np.sum(vectors * vectors, axis=1))
        displacements = centers[pairs[:, 1], :2] - centers[pairs[:, 0], :2]
        distances = np.sqrt(np.sum(displacements * displacements, axis=1))
        if np.any(widths <= 0) or np.any(distances <= 0):
            raise ValueError(
                "Surface edges and projected dual distances must be positive."
            )
        if lateral:
            tangential = np.abs(np.sum(vectors * displacements, axis=1))
            if np.any(tangential > orthogonality_tolerance * widths * distances):
                raise ValueError(
                    "Manning two-point runoff requires projected-orthogonal dual edges."
                )
        n = np.broadcast_to(np.asarray(manning, dtype=float), (centers.shape[0],))
        if np.any(~np.isfinite(n)) or np.any(n <= 0):
            raise ValueError("Manning roughness must be positive and finite.")
        if (
            not np.isfinite(density)
            or density <= 0
            or not np.isfinite(heat_capacity)
            or heat_capacity <= 0
            or not np.isfinite(reference_temperature)
            or reference_temperature <= 0
        ):
            raise ValueError("Surface thermal parameters must be positive and finite.")
        self.trace = trace
        self.projected_areas = trace.areas * trace.normals[:, 2]
        self.bed = trace.centers[:, 2]
        self.edge_owner = jnp.asarray(pairs[:, 0], dtype=jnp.int32)
        self.edge_neighbour = jnp.asarray(pairs[:, 1], dtype=jnp.int32)
        self.edge_width = jnp.asarray(widths)
        self.edge_distance = jnp.asarray(distances)
        self.manning = jnp.asarray(n)
        self.density = jnp.asarray(density)
        self.heat_capacity = jnp.asarray(heat_capacity)
        self.reference_temperature = jnp.asarray(reference_temperature)
        self.lateral = bool(lateral)

    def _vector(self, values, name):
        values = jnp.broadcast_to(jnp.asarray(values), self.bed.shape)
        return eqx.error_if(
            values, jnp.any(~jnp.isfinite(values)), name + " must be finite."
        )

    def initial_state(self, depth: ArrayLike, temperature: ArrayLike = 293.15):
        depth = self._vector(depth, "Surface depth")
        temperature = self._vector(temperature, "Surface temperature")
        depth = eqx.error_if(
            depth,
            jnp.any(depth < 0) | jnp.any(temperature <= 0),
            "Surface depth must be nonnegative and temperature positive.",
        )
        volume = depth * self.projected_areas
        return SurfaceWaterState(
            volume,
            self.density
            * self.heat_capacity
            * volume
            * (temperature - self.reference_temperature),
        )

    def temperature(self, state: SurfaceWaterState):
        wet = state.volume > 0
        denominator = (
            self.density * self.heat_capacity * jnp.where(wet, state.volume, 1.0)
        )
        return self.reference_temperature + jnp.where(
            wet, state.energy / denominator, 0.0
        )

    def lateral_rates(self, volume: ArrayLike):
        volume = jnp.asarray(volume)
        if not self.lateral:
            return jnp.zeros_like(self.edge_width)
        depth = jnp.maximum(volume / self.projected_areas, 0.0)
        head = self.bed + depth
        difference = head[self.edge_owner] - head[self.edge_neighbour]
        donor = jnp.where(difference >= 0, self.edge_owner, self.edge_neighbour)
        # Reconstruction above the higher bed prevents flow through dry uphill cells.
        sill = jnp.maximum(self.bed[self.edge_owner], self.bed[self.edge_neighbour])
        hydraulic_depth = jnp.maximum(head[donor] - sill, 0.0)
        nonzero = difference != 0
        magnitude = jnp.sqrt(
            jnp.where(nonzero, jnp.abs(difference), 1.0) / self.edge_distance
        )
        return jnp.where(
            nonzero,
            jnp.sign(difference)
            * self.edge_width
            * hydraulic_depth ** (5.0 / 3.0)
            * magnitude
            / self.manning[donor],
            0.0,
        )

    def divergence(self, rates: ArrayLike):
        rates = jnp.asarray(rates)
        return (
            jnp.zeros_like(self.bed)
            .at[self.edge_owner]
            .add(rates)
            .at[self.edge_neighbour]
            .add(-rates)
        )

    def step(
        self,
        state: SurfaceWaterState,
        dt: ArrayLike,
        *,
        infiltration_demand: ArrayLike = 0.0,
        rainfall: ArrayLike = 0.0,
        rain_temperature: ArrayLike = 293.15,
        subsurface_temperature: ArrayLike = 293.15,
    ) -> SurfaceExchangeResult:
        """Advance accepted shared rates; positive infiltration enters the volume.

        ``rainfall`` is m/s; infiltration is integrated m³/s per trace cell.
        Exfiltration must be an already accepted supply from the porous solver.
        Return rates must be routed into that solver without recomputation.
        """
        dt = jnp.asarray(dt)
        if dt.ndim != 0:
            raise ValueError("Surface time step must be scalar.")
        dt = eqx.error_if(
            dt, ~jnp.isfinite(dt) | (dt <= 0), "Time step must be positive."
        )
        if state.volume.shape != self.bed.shape or state.energy.shape != self.bed.shape:
            raise ValueError("Surface state does not match the prepared trace.")
        volume = eqx.error_if(
            state.volume,
            jnp.any(~jnp.isfinite(state.volume))
            | jnp.any(state.volume < 0)
            | jnp.any(~jnp.isfinite(state.energy))
            | jnp.any((state.volume == 0) & (state.energy != 0)),
            "Surface inventories must be finite, nonnegative water, and dry energy zero.",
        )
        rain = self._vector(rainfall, "Rainfall")
        rain = eqx.error_if(rain, jnp.any(rain < 0), "Rainfall must be nonnegative.")
        rain_t = self._vector(rain_temperature, "Rain temperature")
        sub_t = self._vector(subsurface_temperature, "Subsurface temperature")
        rain_t = eqx.error_if(
            rain_t,
            jnp.any(rain_t <= 0) | jnp.any(sub_t <= 0),
            "Incoming liquid temperatures must be positive.",
        )
        demand = self._vector(infiltration_demand, "Infiltration demand")
        rain_rate = rain * self.projected_areas
        exfiltration = jnp.minimum(demand, 0.0)
        infiltration = jnp.maximum(demand, 0.0)
        factor = self.density * self.heat_capacity
        rain_energy = factor * rain_rate * (rain_t - self.reference_temperature)
        exfiltration_energy = factor * exfiltration * (sub_t - self.reference_temperature)
        available = volume + dt * (rain_rate - exfiltration)
        available_energy = state.energy + dt * (rain_energy - exfiltration_energy)
        specific = available_energy / jnp.where(available > 0, available, 1.0)
        lateral = self.lateral_rates(available)
        donor = jnp.where(lateral >= 0, self.edge_owner, self.edge_neighbour)
        total_out = infiltration + jnp.zeros_like(volume).at[donor].add(jnp.abs(lateral))
        budget = jnp.where(total_out > 0, total_out, 1.0)
        scale = jnp.minimum(1.0, available / (dt * budget))
        scale = jnp.where(total_out > 0, scale, 1.0)
        lateral = lateral * scale[donor]
        accepted = infiltration * scale + exfiltration
        lateral_energy = lateral * specific[donor]
        accepted_energy = infiltration * scale * specific + exfiltration_energy
        next_volume = available - dt * (infiltration * scale + self.divergence(lateral))
        next_energy = available_energy - dt * (
            infiltration * scale * specific + self.divergence(lateral_energy)
        )
        volume_residual = jnp.sum(next_volume - volume) - dt * jnp.sum(
            rain_rate - accepted
        )
        energy_residual = jnp.sum(next_energy - state.energy) - dt * jnp.sum(
            rain_energy - accepted_energy
        )
        tolerance = (
            64 * jnp.finfo(next_volume.dtype).eps * jnp.maximum(jnp.max(available), 1.0)
        )
        successful = (
            jnp.all(jnp.isfinite(next_volume))
            & jnp.all(jnp.isfinite(next_energy))
            & jnp.all(next_volume >= -tolerance)
        )
        limited = jnp.any(scale < 1)
        derivative_available = (
            successful & ~limited & jnp.all(volume > 0) & jnp.all(next_volume > 0)
        )
        return SurfaceExchangeResult(
            SurfaceWaterState(next_volume, next_energy),
            accepted,
            accepted_energy,
            lateral,
            volume_residual,
            energy_residual,
            limited,
            derivative_available,
            successful,
        )


__all__ = [
    "SurfaceExchangeResult",
    "OrthogonalDiffusiveWaveSurfacePlan",
    "SurfaceWaterState",
]
