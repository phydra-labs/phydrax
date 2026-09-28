"""Gravity drainage and conservative film exchange on a double-bubble rim."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax.applications.foams import (
    PlateauBorderBoundaryFlux,
    PlateauBorderPlan,
    PlateauBorderStatus,
)
from phydrax.geometry.multiregion_surface import (
    PreparedMultiRegionSurface,
    seed_double_bubble,
)
from phydrax.interfacial_transport import prepare_film_sheet_slots


def run() -> dict[str, float | int | bool]:
    seed = seed_double_bubble(1.0, 0.8, ring_points=12)
    topology = seed.topology(
        seed.capacity_plan(resource_id="plateau-border-example")
    )
    surface_state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, surface_state)
    film_slots = prepare_film_sheet_slots(surface, surface_state)
    valence = np.sum(
        np.asarray(topology.edge_faces[: topology.edge_count]) >= 0, axis=1
    )
    physical_edges = np.asarray(topology.edges[: topology.edge_count])[valence == 3]
    points = np.asarray(surface_state.positions)
    physical_midpoint = 0.5 * (
        points[physical_edges[:, 0]] + points[physical_edges[:, 1]]
    )
    gravity_axis = int(np.argmax(np.ptp(physical_midpoint, axis=0)))
    gravity = np.zeros((3,), dtype=np.float64)
    gravity[gravity_axis] = -9.81
    slot_area = surface.slot_areas(surface_state.positions)
    sheet_liquid = jnp.where(topology.slot_active, 2.0e-6 * slot_area, 0.0)
    sheet_surfactant = jnp.where(topology.slot_active, 1.0e-7 * slot_area, 0.0)
    prepared = PlateauBorderPlan(
        border_edge_capacity=12,
        quad_point_capacity=0,
        density_kg_m3=1000.0,
        viscosity_pa_s=1.0e-3,
        surface_tension_n_m=0.03,
        gravity_m_s2=gravity,
        hydraulic_shape_factor=50.0,
        time_step_s=1.0e-5,
        evaporation_declared=True,
        resource_id="plateau-border-example",
    ).prepare(surface, film_slots, surface_state)
    state = prepared.initial_state(
        sheet_liquid,
        sheet_surfactant,
        1.0e-8,
        border_surfactant_concentration_mol_m3=2.0e-4,
    )
    route = int(np.flatnonzero(np.asarray(prepared.boundary_supported))[0])
    liquid_flux = np.zeros((film_slots.boundary_route_capacity,))
    surfactant_flux = np.zeros_like(liquid_flux)
    liquid_flux[route] = 1.0e-11
    surfactant_flux[route] = 2.0e-15
    sheet_evaporation = np.zeros(
        (topology.vertex_capacity, topology.slot_width)
    )
    border_evaporation = np.full((prepared.plan.border_edge_capacity,), 1.0e-14)
    boundary = PlateauBorderBoundaryFlux(
        liquid_flux,
        surfactant_flux,
        sheet_evaporation,
        border_evaporation,
    )
    maximum_liquid_residual = 0.0
    maximum_surfactant_residual = 0.0
    status = PlateauBorderStatus.ACCEPTED
    for _ in range(40):
        result = prepared.step(state, boundary)
        status = PlateauBorderStatus(int(result.evidence.status))
        if status is not PlateauBorderStatus.ACCEPTED:
            break
        state = result.state
        maximum_liquid_residual = max(
            maximum_liquid_residual,
            abs(float(result.evidence.liquid_conservation_residual_m3)),
        )
        maximum_surfactant_residual = max(
            maximum_surfactant_residual,
            abs(float(result.evidence.surfactant_conservation_residual_mol)),
        )
    edges = np.asarray(prepared.border_edges)
    positions = np.asarray(prepared.positions_m)
    midpoint = 0.5 * (positions[edges[:, 0]] + positions[edges[:, 1]])
    cross_section = np.asarray(result.rates.cross_section_area_m2)
    order = np.argsort(midpoint[:, gravity_axis], kind="stable")
    lower = cross_section[order[: prepared.border_count // 2]]
    upper = cross_section[order[prepared.border_count // 2 :]]
    return {
        "status": int(status),
        "accepted": status is PlateauBorderStatus.ACCEPTED,
        "sheet_count": len(film_slots.surfaces),
        "border_edge_count": prepared.border_count,
        "boundary_route_count": film_slots.evidence.boundary_route_count,
        "gravity_axis": gravity_axis,
        "mean_lower_cross_section_m2": float(np.mean(lower)),
        "mean_upper_cross_section_m2": float(np.mean(upper)),
        "gravity_drainage_observed": bool(np.mean(lower) > np.mean(upper)),
        "maximum_liquid_residual_m3": maximum_liquid_residual,
        "maximum_surfactant_residual_mol": maximum_surfactant_residual,
        "evaporation_sink_m3": float(result.evidence.liquid_evaporation_sink_m3),
        "maximum_courant_number": float(result.evidence.maximum_courant_number),
    }


if __name__ == "__main__":
    print(run())
