import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cellular_mechanics import (
    polyhedral_vertex_tissue_plan,
    VertexTissuePlan,
)
from phydrax.applications.foams import (
    apply_foam_rupture,
    FoamDynamicsPlan,
    FoamDynamicsState,
    FoamEquilibriumPlan,
    FoamEquilibriumStatus,
    FoamKKTStatus,
    FoamMaterialPlan,
    FoamRelaxationPlan,
    FoamRelaxationStatus,
    FoamRupturePlan,
    PreparedFoamDynamics,
    PreparedFoamEquilibrium,
    PreparedFoamRelaxation,
    RegionPressureAirPlan,
    StandardDoubleBubble,
)
from phydrax.geometry.multiregion_surface import (
    apply_surface_events,
    BoundedFieldReconstruction,
    ConservativeFieldTransfer,
    multiregion_cell_complex,
    multiregion_sheet_views,
    MultiRegionRemeshPlan,
    MultiRegionSurfaceState,
    MultiRegionSurfaceStatus,
    MultiRegionSurfaceValidationPolicy,
    PreparedMultiRegionSurface,
    propose_remesh,
    seed_double_bubble,
    seed_from_vertex_tissue,
    SurfaceEventPolicy,
    validate_multiregion_surface,
)
from phydrax.interfacial_transport import FilmStepStatus, SurfaceFilmEvidence


SIGMA = 0.03


def _two_cell_tissue() -> tuple[VertexTissuePlan, np.ndarray]:
    def vertex(i: int, j: int, k: int) -> int:
        return 4 * i + 2 * j + k

    positions = np.asarray(
        [(i, j, k) for i in range(3) for j in range(2) for k in range(2)],
        dtype=np.float64,
    )
    loops = [
        [vertex(0, 0, 0), vertex(0, 0, 1), vertex(0, 1, 1), vertex(0, 1, 0)],
        [vertex(1, 0, 0), vertex(1, 1, 0), vertex(1, 1, 1), vertex(1, 0, 1)],
        [vertex(2, 0, 0), vertex(2, 1, 0), vertex(2, 1, 1), vertex(2, 0, 1)],
    ]
    for a in (0, 1):
        loops += [
            [vertex(a, 0, 0), vertex(a + 1, 0, 0), vertex(a + 1, 0, 1), vertex(a, 0, 1)],
            [vertex(a, 1, 0), vertex(a, 1, 1), vertex(a + 1, 1, 1), vertex(a + 1, 1, 0)],
            [vertex(a, 0, 0), vertex(a, 1, 0), vertex(a + 1, 1, 0), vertex(a + 1, 0, 0)],
            [vertex(a, 0, 1), vertex(a + 1, 0, 1), vertex(a + 1, 1, 1), vertex(a, 1, 1)],
        ]
    faces = np.asarray(loops)
    edges = np.asarray(
        sorted(
            {
                (min(loop[i], loop[(i + 1) % 4]), max(loop[i], loop[(i + 1) % 4]))
                for loop in loops
                for i in range(4)
            }
        )
    )
    first, second = [0, 1, 3, 4, 5, 6], [1, 2, 7, 8, 9, 10]
    owners = np.asarray([(0, -1) if face in first else (1, -1) for face in range(11)])
    owners[1] = (0, 1)
    plan = polyhedral_vertex_tissue_plan(
        np.arange(12),
        np.arange(edges.shape[0]),
        edges,
        np.arange(11),
        faces,
        np.asarray((7, 9)),
        np.asarray((first, second)),
        np.asarray(((1,) * 6, (-1, 1, 1, 1, 1, 1))),
        owners,
        1.0,
        1.0,
        6.0,
        0.0,
    )
    return plan, positions


def test_vertex_tissue_seeds_a_quasistatic_double_bubble_workflow() -> None:
    tissue, positions = _two_cell_tissue()
    seed = seed_from_vertex_tissue(tissue, positions).subdivided(2)
    assert seed.region_ids == ("cell-7", "cell-9", "ambient")
    plan = seed.capacity_plan(resource_id="tissue-foam", headroom=1.1)
    topology = seed.topology(plan)
    state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, state)
    assert surface.evidence.signed_volumes == pytest.approx((1.0, 1.0), rel=1e-12)
    material = FoamMaterialPlan.soap_film(topology.region_ids, SIGMA)
    equilibrium = PreparedFoamEquilibrium(FoamEquilibriumPlan(), surface, material, state)
    result = equilibrium.solve(state, equilibrium.parameters(jnp.asarray((1.0, 1.0))))
    evidence = result.evidence
    assert int(evidence.status) == FoamEquilibriumStatus.CONVERGED
    assert int(evidence.kkt_status) == FoamKKTStatus.REGULAR
    assert abs(float(evidence.virial_residual)) < 3e-8
    reference = StandardDoubleBubble(1.0, 1.0, 2.0 * SIGMA)
    scale = (1.0 / reference.volume_first) ** (1.0 / 3.0)
    pressures = np.asarray(result.pressures[:2])
    assert pressures[0] == pytest.approx(pressures[1], rel=1e-6)
    assert pressures[0] == pytest.approx(reference.pressures[0] / scale, rel=0.02)

    relaxed = validate_multiregion_surface(topology, result.state)
    assert relaxed.status is MultiRegionSurfaceStatus.ACCEPTED
    assert relaxed.signed_volumes == pytest.approx((1.0, 1.0), rel=1e-7)
    assert multiregion_cell_complex(topology).entity_sets[3].count == 2
    views = multiregion_sheet_views(surface, result.state)
    wall = next(view for view in views.views if view.region_ids == ("cell-7", "cell-9"))
    assert wall.mesh.topology.euler_characteristic == 1
    junction = np.flatnonzero(np.sum(np.asarray(topology.slot_active), axis=1) == 3)
    boundary = np.asarray(wall.vertex_indices)[
        np.asarray(wall.mesh.topology.boundary_loop_vertices)
    ]
    assert set(boundary.tolist()) == set(junction.tolist())

    # Uniform film thickness h0: slot liquid content h0 * A_slot moves conservatively
    # to per-sheet totals, and the per-sheet thickness reconstructs within bounds.
    thickness = 2.0e-6
    slot_areas = np.asarray(surface.slot_areas(result.state.positions)).reshape(-1)
    slot_pairs = np.asarray(topology.vertex_pair_slots).reshape(-1)
    active = slot_pairs >= 0
    to_sheets = ConservativeFieldTransfer(
        np.flatnonzero(active),
        slot_pairs[active],
        np.ones(int(np.count_nonzero(active))),
        source_active=active,
        target_active=np.asarray(topology.pair_active),
    )
    content = jnp.asarray(thickness * slot_areas)
    sheets = to_sheets.apply(content)
    transfer_evidence = to_sheets.evidence(content, sheets)
    assert bool(transfer_evidence.successful)
    pair_areas = np.asarray(surface.pair_areas(result.state.positions))
    np.testing.assert_allclose(
        np.asarray(sheets)[: topology.region_pair_count],
        thickness * pair_areas[: topology.region_pair_count],
        rtol=1e-12,
    )
    back = BoundedFieldReconstruction(
        slot_pairs[active],
        np.flatnonzero(active),
        np.ones(int(np.count_nonzero(active))),
        source_active=np.asarray(topology.pair_active),
        target_active=active,
    )
    sheet_thickness = jnp.asarray(np.asarray(sheets) / np.maximum(pair_areas, 1e-300))
    slot_thickness = back.apply(sheet_thickness)
    assert bool(back.evidence(sheet_thickness, slot_thickness).successful)
    np.testing.assert_allclose(np.asarray(slot_thickness)[active], thickness, rtol=1e-12)


def test_curved_double_bubble_relaxes_through_remeshing_to_equal_pressures() -> None:
    reference = StandardDoubleBubble(1.0, 1.0, 2.0 * SIGMA)
    targets = jnp.asarray((reference.volume_first, reference.volume_second))
    seed = seed_double_bubble(1.1, 0.9, ring_points=12)
    topology = seed.topology(
        seed.capacity_plan(resource_id="curved", headroom=2.0, event_capacity=32)
    )
    base = seed.state(topology)
    gas = np.zeros((topology.region_capacity, 1))
    gas[:2, 0] = np.asarray(targets)
    state = MultiRegionSurfaceState(
        topology, base.positions, region_fields=gas, region_field_names=("gas",)
    )
    validation = MultiRegionSurfaceValidationPolicy(profile="dry_foam")
    relaxation = FoamRelaxationPlan(friction=1.0, time_step=0.5, steps=60)
    remesh = MultiRegionRemeshPlan(minimum_edge_length=0.15, maximum_edge_length=0.7)
    for epoch in range(3):
        surface = PreparedMultiRegionSurface(topology, state, policy=validation)
        material = FoamMaterialPlan.soap_film(topology.region_ids, SIGMA)
        relaxed = PreparedFoamRelaxation(relaxation, surface, material, state).relax(
            state, targets
        )
        assert int(relaxed.evidence.status) == FoamRelaxationStatus.COMPLETED
        assert float(relaxed.evidence.volume_residual) < 1e-9
        if epoch:
            # The first call also projects the unequal seed volumes onto the
            # equal targets, which may raise the area; later calls only relax.
            assert float(relaxed.evidence.final_energy) < float(
                relaxed.evidence.initial_energy
            )
        surface = PreparedMultiRegionSurface(topology, relaxed.state, policy=validation)
        result = apply_surface_events(
            topology,
            relaxed.state,
            propose_remesh(surface, relaxed.state, remesh),
            policy=SurfaceEventPolicy(validation=validation),
        )
        np.testing.assert_array_equal(np.asarray(result.state.region_fields), gas)
        topology, state = result.topology, result.state
    surface = PreparedMultiRegionSurface(topology, state, policy=validation)
    material = FoamMaterialPlan.soap_film(topology.region_ids, SIGMA)
    equilibrium = PreparedFoamEquilibrium(
        FoamEquilibriumPlan(method="sqp", maximum_steps=300), surface, material, state
    )
    polished = equilibrium.solve(state, equilibrium.parameters(targets))
    evidence = polished.evidence
    assert int(evidence.status) == FoamEquilibriumStatus.CONVERGED
    assert bool(evidence.optimizer_successful)
    assert evidence.tangential_gauge_dimension > 0
    assert int(evidence.kkt_status) == FoamKKTStatus.REGULAR
    assert int(evidence.kkt_rank) == (
        evidence.primal_dimension + evidence.constraint_dimension
    )
    assert int(evidence.kkt_positive) == evidence.primal_dimension
    assert int(evidence.kkt_negative) == evidence.constraint_dimension
    assert int(evidence.kkt_zero) == 0
    assert float(evidence.stationarity_residual) < 2.0e-8
    pressures = np.asarray(polished.pressures[:2])
    assert pressures[0] == pytest.approx(pressures[1], rel=0.02)
    assert pressures[0] == pytest.approx(reference.pressures[0], rel=0.03)
    assert float(polished.energy) == pytest.approx(reference.energy, rel=0.01)
    final_validation = validate_multiregion_surface(
        topology, polished.state, policy=validation
    )
    assert final_validation.self_intersection_checked
    assert final_validation.uncertain_pair_count == 0
    np.testing.assert_allclose(
        final_validation.signed_volumes, np.asarray(targets), rtol=1e-8
    )


def test_accepted_drained_sheet_bursts_then_advances_merged_cell() -> None:
    seed = seed_double_bubble(1.0, 0.8, ring_points=12)
    topology = seed.topology(seed.capacity_plan(resource_id="foam-burst-workflow"))
    base = seed.state(topology)
    sheet = np.zeros((topology.vertex_capacity, topology.slot_width, 1), dtype=np.float64)
    sheet[np.asarray(topology.slot_active), 0] = 1.0e-10
    region = np.zeros((topology.region_capacity, 2), dtype=np.float64)
    region[: topology.region_count, 0] = (1.0, 0.8, 0.0)
    region[: topology.region_count, 1] = (3.0, 2.4, 0.0)
    state = MultiRegionSurfaceState(
        topology,
        base.positions,
        sheet_fields=sheet,
        region_fields=region,
        sheet_field_names=("film_liquid_volume",),
        region_field_names=("gas_amount_mol", "gas_internal_energy_j"),
    )
    finite = np.flatnonzero(np.asarray(topology.region_finite))
    pairs = np.asarray(topology.region_pairs[: topology.region_pair_count])
    pair_index = int(np.flatnonzero(np.all(pairs == np.sort(finite)[None, :], axis=1))[0])
    pair_slots = np.asarray(topology.vertex_pair_slots) == pair_index
    thickness = np.full(
        (topology.vertex_capacity, topology.slot_width), 1.0e-6, dtype=np.float64
    )
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
        nonlinear_iterations=jnp.asarray(4, dtype=jnp.int32),
        nonlinear_residual_norm=jnp.asarray(1.0e-12),
        converged=jnp.asarray(True),
        finite=jnp.asarray(True),
        geometry_revision=jnp.asarray(2, dtype=jnp.int32),
    )
    burst = apply_foam_rupture(
        FoamRupturePlan(5.0e-8),
        topology,
        state,
        thickness,
        FilmStepStatus.ACCEPTED,
        film,
        2,
        0.0,
    )
    assert burst.successful
    surface = PreparedMultiRegionSurface(burst.topology, burst.state)
    finite_slots = jnp.asarray(burst.topology.finite_region_indices, dtype=jnp.int32)
    target = surface.region_volumes(burst.state.positions)[finite_slots]
    air = RegionPressureAirPlan.incompressible(target)
    dynamics_state = FoamDynamicsState(
        burst.state, unresolved_rim_content=burst.unresolved_rim_content
    )
    dynamics = PreparedFoamDynamics(
        FoamDynamicsPlan(time_step=1.0e-4, friction=5.0),
        surface,
        FoamMaterialPlan.soap_film(burst.topology.region_ids, SIGMA),
        air,
        dynamics_state,
    ).advance(dynamics_state)

    assert dynamics.successful
    assert float(dynamics.evidence.volume_residual) < 1.0e-9
    assert abs(float(burst.evidence.liquid_conservation_residual)) < 1.0e-18
    assert abs(float(burst.evidence.gas_amount_residual)) < 1.0e-14
    assert abs(float(burst.evidence.gas_energy_residual)) < 1.0e-14
    assert burst.evidence.gas_region_lineage
