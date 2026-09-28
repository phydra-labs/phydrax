import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.foams import (
    apply_foam_rupture_with_borders,
    FoamRupturePlan,
    PlateauBorderBoundaryFlux,
    PlateauBorderPlan,
    PlateauBorderPreparationError,
    PlateauBorderPreparationStatus,
    PlateauBorderRimStatus,
    PlateauBorderState,
    PlateauBorderStatus,
    PreparedPlateauBorder,
)
from phydrax.geometry.multiregion_surface import (
    MultiRegionSurfaceSeed,
    MultiRegionSurfaceState,
    MultiRegionSurfaceTopology,
    PreparedMultiRegionSurface,
    seed_double_bubble,
    seed_sphere,
)
from phydrax.interfacial_transport import (
    FilmStepStatus,
    prepare_film_sheet_slots,
    PreparedFilmSheetSlots,
    SurfaceFilmEvidence,
)


def _quad_seed() -> MultiRegionSurfaceSeed:
    outer = np.asarray(
        (
            (1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
        )
    )
    points = np.concatenate((np.zeros((1, 3)), outer), axis=0)
    cell_centers = -0.25 * outer
    faces: list[list[int]] = []
    labels: list[tuple[int, int]] = []
    for first in range(4):
        for second in range(first + 1, 4):
            remaining = [value for value in range(4) if value not in (first, second)]
            triangle = [0, first + 1, second + 1]
            corners = points[triangle]
            normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
            left, right = remaining
            if np.dot(normal, cell_centers[right] - cell_centers[left]) < 0.0:
                left, right = right, left
            faces.append(triangle)
            labels.append((left, right))
    return MultiRegionSurfaceSeed(
        points,
        np.asarray(faces),
        np.asarray(labels),
        ("q0", "q1", "q2", "q3"),
        ("boundary",) * 4,
        source="quad-point-star",
    )


def _prepared_quad(
    *,
    border_capacity: int = 4,
    quad_capacity: int = 1,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    evaporation: bool = False,
    time_step: float = 1.0e-6,
) -> tuple[
    MultiRegionSurfaceTopology,
    MultiRegionSurfaceState,
    PreparedFilmSheetSlots,
    PreparedPlateauBorder,
]:
    seed = _quad_seed()
    topology = seed.topology(seed.capacity_plan(resource_id="plateau-quad"))
    state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, state)
    adapter = prepare_film_sheet_slots(surface, state)
    plan = PlateauBorderPlan(
        border_edge_capacity=border_capacity,
        quad_point_capacity=quad_capacity,
        density_kg_m3=1000.0,
        viscosity_pa_s=1.0e-3,
        surface_tension_n_m=0.03,
        gravity_m_s2=np.asarray(gravity, dtype=np.float64),
        hydraulic_shape_factor=50.0,
        time_step_s=time_step,
        evaporation_declared=evaporation,
        resource_id="plateau-quad",
    )
    return topology, state, adapter, plan.prepare(surface, adapter, state)


def _initial(
    prepared: PreparedPlateauBorder, *, area: float = 1.0e-6
) -> PlateauBorderState:
    topology = prepared.surface.topology
    sheet_liquid = jnp.where(topology.slot_active, 2.0e-7, 0.0)
    sheet_surfactant = jnp.where(topology.slot_active, 3.0e-12, 0.0)
    return prepared.initial_state(
        sheet_liquid,
        sheet_surfactant,
        area,
        border_surfactant_concentration_mol_m3=2.0e-3,
    )


def test_manufactured_gravity_drainage_and_quad_balance() -> None:
    _, _, _, prepared = _prepared_quad()
    state = _initial(prepared)
    rates = prepared.rates(state, prepared.zero_boundary_flux())

    edges = np.asarray(prepared.border_edges)
    points = np.asarray(prepared.positions_m)
    length = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    midpoint = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    area = np.asarray(state.border_liquid_m3) / length
    pressure = -0.03 / np.sqrt(area)
    potential = pressure - 1000.0 * (midpoint @ np.asarray((0.0, 0.0, -9.81)))
    conductance = 2.0 * area**2 / (50.0 * 1.0e-3 * length)
    node_potential = np.sum(conductance * potential) / np.sum(conductance)
    expected = -conductance * (potential - node_potential)

    np.testing.assert_allclose(rates.border_liquid_m3_s, expected, rtol=2.0e-13)
    assert abs(float(jnp.sum(rates.border_liquid_m3_s))) < 1.0e-20
    result = prepared.step(state, prepared.zero_boundary_flux())
    assert int(result.evidence.status) == PlateauBorderStatus.ACCEPTED
    assert float(result.evidence.maximum_quad_mass_residual_m3_s) < 1.0e-20
    assert float(result.evidence.maximum_quad_pressure_residual_pa) < 1.0e-9
    assert abs(float(result.evidence.liquid_conservation_residual_m3)) < 1.0e-18
    assert np.asarray(result.state.border_liquid_m3)[np.argmin(midpoint[:, 2])] > np.asarray(
        state.border_liquid_m3
    )[np.argmin(midpoint[:, 2])]


def test_sheet_border_liquid_surfactant_and_declared_evaporation_ledgers() -> None:
    topology, _, adapter, prepared = _prepared_quad(
        gravity=(0.0, 0.0, 0.0), evaporation=True, time_step=1.0e-3
    )
    state = _initial(prepared)
    route = int(np.flatnonzero(np.asarray(prepared.boundary_supported))[0])
    liquid = np.zeros((adapter.boundary_route_capacity,))
    surfactant = np.zeros_like(liquid)
    liquid[route] = 2.0e-8
    surfactant[route] = 3.0e-13
    sheet_evaporation = np.zeros((topology.vertex_capacity, topology.slot_width))
    border_evaporation = np.zeros((prepared.plan.border_edge_capacity,))
    sheet_evaporation[np.asarray(topology.slot_active)] = 1.0e-11
    border_evaporation[0] = 2.0e-11
    boundary = PlateauBorderBoundaryFlux(
        liquid,
        surfactant,
        sheet_evaporation,
        border_evaporation,
    )

    result = prepared.step(state, boundary)

    assert int(result.evidence.status) == PlateauBorderStatus.ACCEPTED
    assert abs(float(result.evidence.liquid_conservation_residual_m3)) < 1.0e-18
    assert abs(float(result.evidence.surfactant_conservation_residual_mol)) < 1.0e-24
    assert float(result.evidence.sheet_border_liquid_transfer_m3) == pytest.approx(
        prepared.plan.time_step_s * liquid[route]
    )
    assert float(result.evidence.sheet_border_surfactant_transfer_mol) == pytest.approx(
        prepared.plan.time_step_s * surfactant[route]
    )
    flat_slot = int(np.asarray(adapter.boundary_slot_indices)[route])
    vertex, slot = divmod(flat_slot, topology.slot_width)
    border = int(np.asarray(prepared.boundary_to_borders.target_indices)[route])
    dt = float(prepared.plan.time_step_s)
    assert float(
        result.state.sheet_liquid_m3[vertex, slot]
        - state.sheet_liquid_m3[vertex, slot]
    ) == pytest.approx(-dt * (liquid[route] + sheet_evaporation[vertex, slot]))
    assert float(
        result.state.sheet_surfactant_mol[vertex, slot]
        - state.sheet_surfactant_mol[vertex, slot]
    ) == pytest.approx(-dt * surfactant[route])
    assert float(
        result.state.border_liquid_m3[border] - state.border_liquid_m3[border]
    ) == pytest.approx(
        dt * (liquid[route] - border_evaporation[border]), abs=1.0e-20
    )
    expected_sink = float(prepared.plan.time_step_s) * (
        np.sum(sheet_evaporation) + np.sum(border_evaporation)
    )
    assert float(result.evidence.liquid_evaporation_sink_m3) == pytest.approx(
        expected_sink
    )


def test_support_resource_and_undeclared_sink_refusals() -> None:
    seed = _quad_seed()
    topology = seed.topology(seed.capacity_plan(resource_id="plateau-refusal"))
    state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, state)
    adapter = prepare_film_sheet_slots(surface, state)
    with pytest.raises(PlateauBorderPreparationError) as capacity:
        PlateauBorderPlan(
            border_edge_capacity=3,
            quad_point_capacity=1,
            density_kg_m3=1000.0,
            viscosity_pa_s=1.0e-3,
            surface_tension_n_m=0.03,
            time_step_s=1.0e-6,
        ).prepare(surface, adapter, state)
    assert (
        capacity.value.evidence.status
        is PlateauBorderPreparationStatus.BORDER_CAPACITY_EXCEEDED
    )

    sphere_seed = seed_sphere(1.0, subdivisions=0)
    sphere_topology = sphere_seed.topology(
        sphere_seed.capacity_plan(resource_id="plateau-no-border")
    )
    sphere_state = sphere_seed.state(sphere_topology)
    sphere_surface = PreparedMultiRegionSurface(sphere_topology, sphere_state)
    sphere_adapter = prepare_film_sheet_slots(sphere_surface, sphere_state)
    with pytest.raises(PlateauBorderPreparationError) as no_border:
        PlateauBorderPlan(
            border_edge_capacity=1,
            quad_point_capacity=0,
            density_kg_m3=1000.0,
            viscosity_pa_s=1.0e-3,
            surface_tension_n_m=0.03,
            time_step_s=1.0e-6,
        ).prepare(sphere_surface, sphere_adapter, sphere_state)
    assert (
        no_border.value.evidence.status
        is PlateauBorderPreparationStatus.NO_PHYSICAL_BORDERS
    )

    _, _, adapter, prepared = _prepared_quad(gravity=(0.0, 0.0, 0.0))
    initial = _initial(prepared)
    wire_route = int(
        np.flatnonzero(
            np.asarray(
                adapter.boundary_route_valid & ~adapter.boundary_plateau_supported
            )
        )[0]
    )
    liquid = np.zeros((adapter.boundary_route_capacity,))
    liquid[wire_route] = 1.0e-9
    unsupported = PlateauBorderBoundaryFlux(
        liquid,
        np.zeros_like(liquid),
        np.zeros_like(np.asarray(initial.sheet_liquid_m3)),
        np.zeros_like(np.asarray(initial.border_liquid_m3)),
    )
    assert int(prepared.step(initial, unsupported).evidence.status) == (
        PlateauBorderStatus.UNSUPPORTED_BOUNDARY_FLUX
    )

    evaporation = PlateauBorderBoundaryFlux(
        np.zeros_like(liquid),
        np.zeros_like(liquid),
        np.where(
            np.asarray(prepared.surface.topology.slot_active),
            1.0e-12,
            0.0,
        ),
        np.zeros_like(np.asarray(initial.border_liquid_m3)),
    )
    assert int(prepared.step(initial, evaporation).evidence.status) == (
        PlateauBorderStatus.EVAPORATION_UNDECLARED
    )
    unchanged, added, count = prepared.resolve_rim_content(
        initial, ("absent-a", "absent-b"), 1.0e-9
    )
    assert unchanged is initial
    assert count == 0
    np.testing.assert_array_equal(added, jnp.zeros_like(initial.border_liquid_m3))


def test_fixed_topology_border_rates_jvp_vjp_duality() -> None:
    _, _, _, prepared = _prepared_quad(gravity=(0.0, 0.0, 0.0))
    state = _initial(prepared)
    boundary = prepared.zero_boundary_flux()

    def action(volume: jax.Array) -> jax.Array:
        candidate = PlateauBorderState(
            state.sheet_liquid_m3,
            state.sheet_surfactant_mol,
            volume,
            state.border_surfactant_mol,
            unresolved_rim_content_m3=state.unresolved_rim_content_m3,
            geometry_revision=state.geometry_revision,
            topology_id=state.topology_id,
        )
        return prepared.rates(candidate, boundary).border_liquid_m3_s

    tangent = jnp.asarray((0.2, -0.1, 0.3, -0.4)) * 1.0e-7
    cotangent = jnp.asarray((0.7, -0.2, 0.4, 0.1))
    _, jvp = jax.jvp(action, (state.border_liquid_m3,), (tangent,))
    _, pullback = jax.vjp(action, state.border_liquid_m3)
    vjp = pullback(cotangent)[0]
    np.testing.assert_allclose(
        jnp.vdot(cotangent, jvp), jnp.vdot(vjp, tangent), rtol=2.0e-12
    )
    epsilon = 1.0e-4
    finite_difference = (
        action(state.border_liquid_m3 + epsilon * tangent)
        - action(state.border_liquid_m3 - epsilon * tangent)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(jvp, finite_difference, rtol=2.0e-7, atol=1.0e-16)


def _film_evidence(thickness: np.ndarray, mask: np.ndarray) -> SurfaceFilmEvidence:
    return SurfaceFilmEvidence(
        liquid_volume_residual_m3=jnp.asarray(0.0),
        boundary_exchange_m3=jnp.asarray(0.0),
        minimum_thickness_m=jnp.asarray(np.min(thickness)),
        rupture_mask=jnp.asarray(mask),
        energy_change_j=jnp.asarray(-1.0),
        dissipation_guaranteed=jnp.asarray(True),
        positivity_guaranteed=jnp.asarray(True),
        conductance_admissible=jnp.asarray(True),
        nonlinear_status=jnp.asarray(0, dtype=jnp.int32),
        nonlinear_iterations=jnp.asarray(2, dtype=jnp.int32),
        nonlinear_residual_norm=jnp.asarray(1.0e-12),
        converged=jnp.asarray(True),
        finite=jnp.asarray(True),
        geometry_revision=jnp.asarray(3, dtype=jnp.int32),
    )


def _prepared_burst_case() -> tuple[
    MultiRegionSurfaceTopology,
    MultiRegionSurfaceState,
    PreparedPlateauBorder,
    PlateauBorderState,
]:
    seed = seed_double_bubble(1.0, 0.8, ring_points=6)
    topology = seed.topology(seed.capacity_plan(resource_id="plateau-burst"))
    base = seed.state(topology)
    slot_values = np.arange(
        topology.vertex_capacity * topology.slot_width, dtype=np.float64
    ).reshape((topology.vertex_capacity, topology.slot_width))
    liquid = np.where(np.asarray(topology.slot_active), (slot_values + 1.0) * 1.0e-11, 0.0)
    surfactant = np.where(
        np.asarray(topology.slot_active), (slot_values + 1.0) * 2.0e-14, 0.0
    )
    surface_state = MultiRegionSurfaceState(
        topology,
        base.positions,
        sheet_fields=liquid[..., None],
        sheet_field_names=("film_liquid_volume",),
    )
    surface = PreparedMultiRegionSurface(topology, surface_state)
    adapter = prepare_film_sheet_slots(surface, surface_state)
    prepared = PlateauBorderPlan(
        border_edge_capacity=6,
        quad_point_capacity=0,
        density_kg_m3=1000.0,
        viscosity_pa_s=1.0e-3,
        surface_tension_n_m=0.03,
        gravity_m_s2=np.zeros((3,), dtype=np.float64),
        time_step_s=1.0e-6,
    ).prepare(surface, adapter, surface_state)
    border_state = prepared.initial_state(
        liquid,
        surfactant,
        1.0e-6,
        unresolved_rim_content_m3=2.0e-9,
    )
    return topology, surface_state, prepared, border_state


def _pair_slot_mask(
    topology: MultiRegionSurfaceTopology, region_ids: tuple[str, str]
) -> np.ndarray:
    region_indices = tuple(topology.region_ids.index(value) for value in region_ids)
    pair = np.sort(np.asarray(region_indices, dtype=np.int64))
    pairs = np.asarray(topology.region_pairs[: topology.region_pair_count])
    pair_index = int(np.flatnonzero(np.all(pairs == pair[None, :], axis=1))[0])
    return np.asarray(topology.slot_active) & (
        np.asarray(topology.vertex_pair_slots) == pair_index
    )


def test_burst_content_resolves_to_supported_rim_once_and_conservatively() -> None:
    topology, surface_state, prepared, border_state = _prepared_burst_case()
    finite_ids = tuple(
        topology.region_ids[index]
        for index in np.asarray(topology.finite_region_indices)
    )
    if len(finite_ids) != 2:
        raise ValueError("The double-bubble fixture must have two finite regions.")
    pair_ids = (finite_ids[0], finite_ids[1])
    pair_slots = _pair_slot_mask(topology, pair_ids)
    thickness = np.full((topology.vertex_capacity, topology.slot_width), 1.0e-6)
    thickness[pair_slots] = 5.0e-8
    source_liquid = np.asarray(border_state.sheet_liquid_m3)
    source_surfactant = np.asarray(border_state.sheet_surfactant_mol)
    liquid_before = border_state.total_liquid_m3()
    surfactant_before = border_state.total_surfactant_mol()

    result = apply_foam_rupture_with_borders(
        FoamRupturePlan(5.0e-8),
        prepared,
        border_state,
        topology,
        surface_state,
        thickness,
        FilmStepStatus.ACCEPTED,
        _film_evidence(thickness, pair_slots),
        3,
    )

    assert result.evidence.status is PlateauBorderRimStatus.RESOLVED
    assert result.evidence.resolved
    assert result.evidence.supporting_border_count == 6
    assert float(result.unresolved_rim_content_m3) == pytest.approx(2.0e-9)
    np.testing.assert_array_equal(result.border_state.sheet_liquid_m3[pair_slots], 0.0)
    np.testing.assert_array_equal(
        result.border_state.sheet_surfactant_mol[pair_slots], 0.0
    )
    np.testing.assert_array_equal(
        result.border_state.sheet_liquid_m3[~pair_slots], source_liquid[~pair_slots]
    )
    np.testing.assert_array_equal(
        result.border_state.sheet_surfactant_mol[~pair_slots],
        source_surfactant[~pair_slots],
    )
    np.testing.assert_allclose(
        jnp.sum(result.border_content_added_m3),
        jnp.sum(border_state.sheet_liquid_m3[pair_slots]),
        rtol=2.0e-15,
    )
    np.testing.assert_allclose(
        jnp.sum(
            result.border_state.border_surfactant_mol
            - border_state.border_surfactant_mol
        ),
        jnp.sum(border_state.sheet_surfactant_mol[pair_slots]),
        rtol=2.0e-15,
    )
    np.testing.assert_allclose(
        result.border_state.total_liquid_m3(), liquid_before, rtol=2.0e-15
    )
    np.testing.assert_allclose(
        result.border_state.total_surfactant_mol(),
        surfactant_before,
        rtol=2.0e-15,
    )
    np.testing.assert_allclose(
        result.evidence.liquid_conservation_residual_m3,
        result.border_state.total_liquid_m3() - liquid_before,
        rtol=0.0,
        atol=0.0,
    )

    depleted_surface_state = MultiRegionSurfaceState(
        topology,
        surface_state.positions,
        sheet_fields=result.border_state.sheet_liquid_m3[..., None],
        sheet_field_names=("film_liquid_volume",),
    )
    repeated = apply_foam_rupture_with_borders(
        FoamRupturePlan(5.0e-8),
        prepared,
        result.border_state,
        topology,
        depleted_surface_state,
        thickness,
        FilmStepStatus.ACCEPTED,
        _film_evidence(thickness, pair_slots),
        3,
    )
    np.testing.assert_array_equal(
        repeated.border_state.border_liquid_m3,
        result.border_state.border_liquid_m3,
    )
    np.testing.assert_array_equal(
        repeated.border_state.border_surfactant_mol,
        result.border_state.border_surfactant_mol,
    )
    np.testing.assert_array_equal(
        repeated.border_content_added_m3,
        jnp.zeros_like(repeated.border_content_added_m3),
    )


def test_unsupported_burst_rim_keeps_source_border_state_unchanged() -> None:
    topology, surface_state, prepared, border_state = _prepared_burst_case()
    pair_ids = ("ambient", "bubble-1")
    pair_slots = _pair_slot_mask(topology, pair_ids)
    thickness = np.full((topology.vertex_capacity, topology.slot_width), 1.0e-6)
    thickness[pair_slots] = 5.0e-8

    result = apply_foam_rupture_with_borders(
        FoamRupturePlan(5.0e-8),
        prepared,
        border_state,
        topology,
        surface_state,
        thickness,
        FilmStepStatus.ACCEPTED,
        _film_evidence(thickness, pair_slots),
        3,
    )

    assert result.evidence.status is PlateauBorderRimStatus.BORDER_SUPPORT_UNAVAILABLE
    assert result.border_state is border_state
    np.testing.assert_array_equal(
        result.border_content_added_m3, jnp.zeros_like(border_state.border_liquid_m3)
    )
    np.testing.assert_array_equal(
        result.border_state.sheet_liquid_m3, border_state.sheet_liquid_m3
    )
    np.testing.assert_array_equal(
        result.border_state.sheet_surfactant_mol,
        border_state.sheet_surfactant_mol,
    )
