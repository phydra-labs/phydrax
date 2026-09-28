import jax.numpy as jnp
import numpy as np

from phydrax.applications.foams import (
    apply_foam_rupture_with_borders,
    FoamRupturePlan,
    PlateauBorderBoundaryFlux,
    PlateauBorderPlan,
    PlateauBorderRimStatus,
    PlateauBorderStatus,
)
from phydrax.geometry.multiregion_surface import (
    MultiRegionSurfaceState,
    PreparedMultiRegionSurface,
    seed_double_bubble,
)
from phydrax.interfacial_transport import (
    FilmStepStatus,
    prepare_film_sheet_slots,
    SurfaceFilmEvidence,
)


def test_b_film_slots_drain_to_e_borders_move_and_resolve_a_burst() -> None:
    seed = seed_double_bubble(1.0, 0.8, ring_points=6)
    topology = seed.topology(seed.capacity_plan(resource_id="plateau-film-workflow"))
    base = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, base)
    slot_area = surface.slot_areas(base.positions)
    liquid = jnp.where(topology.slot_active, 2.0e-6 * slot_area, 0.0)
    surfactant = jnp.where(topology.slot_active, 1.0e-7 * slot_area, 0.0)
    sheet_fields = np.zeros((topology.vertex_capacity, topology.slot_width, 1))
    sheet_fields[..., 0] = np.asarray(liquid)
    surface_state = MultiRegionSurfaceState(
        topology,
        base.positions,
        sheet_fields=sheet_fields,
        sheet_field_names=("film_liquid_volume",),
    )
    surface = PreparedMultiRegionSurface(topology, surface_state)
    film_slots = prepare_film_sheet_slots(surface, surface_state)
    prepared = PlateauBorderPlan(
        border_edge_capacity=6,
        quad_point_capacity=0,
        density_kg_m3=1000.0,
        viscosity_pa_s=1.0e-3,
        surface_tension_n_m=0.03,
        gravity_m_s2=np.zeros((3,), dtype=np.float64),
        time_step_s=1.0e-4,
        evaporation_declared=True,
    ).prepare(surface, film_slots, surface_state)
    state = prepared.initial_state(
        liquid,
        surfactant,
        1.0e-6,
        border_surfactant_concentration_mol_m3=2.0e-4,
    )
    route = int(np.flatnonzero(np.asarray(prepared.boundary_supported))[0])
    liquid_flux = np.zeros((film_slots.boundary_route_capacity,))
    surfactant_flux = np.zeros_like(liquid_flux)
    liquid_flux[route] = 1.0e-8
    surfactant_flux[route] = 2.0e-12
    boundary = PlateauBorderBoundaryFlux(
        liquid_flux,
        surfactant_flux,
        np.zeros_like(np.asarray(liquid)),
        np.full((prepared.plan.border_edge_capacity,), 1.0e-11),
    )
    drained = prepared.step(state, boundary)
    assert int(drained.evidence.status) == PlateauBorderStatus.ACCEPTED
    assert abs(float(drained.evidence.liquid_conservation_residual_m3)) < 1.0e-18
    assert abs(float(drained.evidence.surfactant_conservation_residual_mol)) <= (
        1.0e-14 * float(state.total_surfactant_mol())
    )

    moved_surface_state = MultiRegionSurfaceState(
        topology,
        1.001 * surface_state.positions,
        sheet_fields=surface_state.sheet_fields,
        sheet_field_names=surface_state.sheet_field_names,
    )
    moved_surface = PreparedMultiRegionSurface(topology, moved_surface_state)
    moved_slots = film_slots.refresh(
        moved_surface, moved_surface_state, geometry_revision=1
    )
    moved_border = prepared.refresh(
        moved_surface,
        moved_slots,
        moved_surface_state,
        geometry_revision=1,
    )
    assert moved_border.prepared_id == prepared.prepared_id
    assert int(moved_slots.geometry_revision) == 1

    pairs = np.asarray(topology.region_pairs[: topology.region_pair_count])
    finite = np.asarray(topology.finite_region_indices)
    pair_index = int(
        np.flatnonzero(np.all(pairs == np.sort(finite)[None, :], axis=1))[0]
    )
    pair_slots = np.asarray(topology.vertex_pair_slots) == pair_index
    thickness = np.full((topology.vertex_capacity, topology.slot_width), 1.0e-6)
    thickness[pair_slots] = 5.0e-8
    film = SurfaceFilmEvidence(
        liquid_volume_residual_m3=jnp.asarray(0.0),
        boundary_exchange_m3=jnp.asarray(0.0),
        minimum_thickness_m=jnp.asarray(5.0e-8),
        rupture_mask=jnp.asarray(pair_slots),
        energy_change_j=jnp.asarray(-1.0),
        dissipation_guaranteed=jnp.asarray(True),
        positivity_guaranteed=jnp.asarray(True),
        conductance_admissible=jnp.asarray(True),
        nonlinear_status=jnp.asarray(0, dtype=jnp.int32),
        nonlinear_iterations=jnp.asarray(2, dtype=jnp.int32),
        nonlinear_residual_norm=jnp.asarray(1.0e-12),
        converged=jnp.asarray(True),
        finite=jnp.asarray(True),
        geometry_revision=jnp.asarray(0, dtype=jnp.int32),
    )
    rupture_fields = np.zeros_like(np.asarray(surface_state.sheet_fields))
    rupture_fields[..., 0] = np.asarray(drained.state.sheet_liquid_m3)
    rupture_surface_state = MultiRegionSurfaceState(
        topology,
        surface_state.positions,
        sheet_fields=rupture_fields,
        sheet_field_names=surface_state.sheet_field_names,
    )
    burst = apply_foam_rupture_with_borders(
        FoamRupturePlan(5.0e-8),
        prepared,
        drained.state,
        topology,
        rupture_surface_state,
        thickness,
        FilmStepStatus.ACCEPTED,
        film,
        0,
    )
    assert burst.evidence.status is PlateauBorderRimStatus.RESOLVED
    assert abs(float(burst.evidence.liquid_conservation_residual_m3)) < 1.0e-20
    assert float(burst.unresolved_rim_content_m3) == 0.0
