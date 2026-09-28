import jax.numpy as jnp
import numpy as np

from phydrax.geometry.multiregion_surface import (
    MultiRegionSurfaceSeed,
    MultiRegionSurfaceState,
    PreparedMultiRegionSurface,
)
from phydrax.interfacial_transport import (
    FilmStepStatus,
    LangmuirSurfactantLaw,
    prepare_film_sheet_slots,
    SurfaceLubricationPlan,
    SurfaceMeshMotion,
    SymmetricFilmSurfactantPlan,
)


def _y_junction() -> MultiRegionSurfaceSeed:
    angles = 2.0 * np.pi * np.arange(3) / 3.0
    points = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    for angle in angles:
        points.extend(
            (
                (np.cos(angle), np.sin(angle), 0.0),
                (np.cos(angle), np.sin(angle), 1.0),
            )
        )
    faces: list[tuple[int, int, int]] = []
    labels: list[tuple[int, int]] = []
    for sheet in range(3):
        lower = 2 + 2 * sheet
        upper = lower + 1
        faces.extend(((0, lower, upper), (0, upper, 1)))
        labels.extend(((sheet, (sheet - 1) % 3),) * 2)
    return MultiRegionSurfaceSeed(
        np.asarray(points),
        np.asarray(faces),
        np.asarray(labels),
        ("a", "b", "c"),
        ("boundary",) * 3,
        source="adapter-y-junction",
    )


def test_sheet_slots_are_independent_manifold_B_operators_with_sparse_roundtrip() -> None:
    seed = _y_junction()
    topology = seed.topology(seed.capacity_plan(resource_id="film-slot-adapter"))
    state = seed.state(topology)
    multiregion = PreparedMultiRegionSurface(topology, state)
    adapter = prepare_film_sheet_slots(multiregion, state)

    assert adapter.evidence.accepted
    assert len(adapter.surfaces) == 3
    assert all(surface.topology.watertight is False for surface in adapter.surfaces)
    assert all(surface.topology.num_vertices == 4 for surface in adapter.surfaces)
    assert int(np.count_nonzero(np.asarray(adapter.boundary_plateau_supported))) == 6

    content = jnp.arange(topology.vertex_capacity * topology.slot_width).reshape(
        (topology.vertex_capacity, topology.slot_width)
    )
    packed = adapter.gather_slot_content(content)
    scattered = adapter.scatter_slot_content(packed)
    np.testing.assert_array_equal(
        np.asarray(scattered)[np.asarray(topology.slot_active)],
        np.asarray(content)[np.asarray(topology.slot_active)],
    )
    area = adapter.scatter_slot_content(adapter.packed_vertex_area_m2)
    np.testing.assert_allclose(
        np.asarray(area)[np.asarray(topology.slot_active)],
        np.asarray(multiregion.slot_areas(state.positions))[
            np.asarray(topology.slot_active)
        ],
        rtol=2.0e-15,
    )


def test_lubrication_surfactant_and_moving_geometry_use_the_sheet_adapter() -> None:
    seed = _y_junction()
    topology = seed.topology(seed.capacity_plan(resource_id="film-slot-coupling"))
    state = seed.state(topology)
    multiregion = PreparedMultiRegionSurface(topology, state)
    adapter = prepare_film_sheet_slots(multiregion, state)
    sheet_index = 0
    sheet = adapter.surfaces[sheet_index]
    view = adapter.views.views[sheet_index]

    lubrication = SurfaceLubricationPlan(
        sheet,
        mobility_law="immobile-free-film",
        surface_tension_n_m=0.03,
        viscosity_pa_s=1.0e-3,
        boundary_policy="fixed-thickness",
        boundary_thickness_m=5.0e-5,
    ).prepare()
    liquid = lubrication.initial_state(1.0e-4)
    drained = lubrication.step(liquid, 1.0e-4)
    assert int(drained.status) == FilmStepStatus.ACCEPTED
    assert float(jnp.sum(drained.boundary_exchange_m3)) < 0.0

    slot_exchange = jnp.zeros((topology.vertex_capacity * topology.slot_width,))
    slot_exchange = slot_exchange.at[view.slot_indices].set(
        drained.boundary_exchange_m3
    )
    slot_exchange = slot_exchange.reshape(
        (topology.vertex_capacity, topology.slot_width)
    )
    route_exchange = adapter.distribute_slot_boundary_rate(slot_exchange)
    reconstructed = adapter.boundary_slot_rate(route_exchange)
    np.testing.assert_allclose(
        np.asarray(reconstructed)[np.asarray(topology.slot_active)],
        np.asarray(slot_exchange)[np.asarray(topology.slot_active)],
        rtol=2.0e-15,
        atol=1.0e-20,
    )

    law = LangmuirSurfactantLaw(0.03, 298.15, 2.0e-6)
    surfactant = SymmetricFilmSurfactantPlan(
        sheet,
        law,
        surface_diffusivity_m2_s=1.0e-4,
    ).prepare()
    surfactant_state = surfactant.initial_state(
        1.0e-4,
        jnp.asarray((0.8, 0.4, 0.2, 0.6)) * 1.0e-6,
    )
    transported = surfactant.step(surfactant_state, 1.0e-3)
    assert int(transported.status) == FilmStepStatus.ACCEPTED
    np.testing.assert_allclose(
        transported.state.total_surfactant_mol(),
        surfactant_state.total_surfactant_mol(),
        rtol=2.0e-14,
    )

    motion = SurfaceMeshMotion(sheet, 1.01 * sheet.coordinates, 1.0e-2)
    moved = motion.transport(liquid.liquid_volume_m3, motion.mesh_velocity)
    assert bool(moved.accepted)
    np.testing.assert_allclose(
        jnp.sum(moved.content), jnp.sum(liquid.liquid_volume_m3), rtol=2.0e-15
    )
    moved_surface_state = MultiRegionSurfaceState(
        topology,
        1.01 * state.positions,
    )
    refreshed_multiregion = PreparedMultiRegionSurface(topology, moved_surface_state)
    refreshed = adapter.refresh(
        refreshed_multiregion, moved_surface_state, geometry_revision=1
    )
    assert int(refreshed.geometry_revision) == 1
    assert refreshed.surfaces[sheet_index].topology.operator_id == (
        sheet.topology.operator_id
    )
    assert int(refreshed.surfaces[sheet_index].geometry_revision) == 1
