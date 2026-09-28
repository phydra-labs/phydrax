import jax.numpy as jnp
import numpy as np

from phydrax import array_tree_fingerprint
from phydrax.applications.foams import PlateauBorderPlan
from phydrax.geometry.multiregion_surface import (
    MultiRegionSurfaceState,
    PreparedMultiRegionSurface,
    seed_double_bubble,
)
from phydrax.interfacial_transport import prepare_film_sheet_slots
from phydrax.optics.wave import ThinFilmInterferencePlan
from phydrax.rendering import (
    SpectralColorimetryPlan,
    SpectralIlluminant,
    thin_film_surface_colors,
    ThinFilmAppearancePlan,
    ThinFilmSurfaceColorStatus,
)


def test_plateau_sheet_slots_render_without_mutating_physics_state() -> None:
    seed = seed_double_bubble(1.0, 0.8, ring_points=6)
    topology = seed.topology(seed.capacity_plan(resource_id="foam-iridescence-test"))
    base = seed.state(topology)
    prepared_surface = PreparedMultiRegionSurface(topology, base)
    area = np.asarray(prepared_surface.slot_areas(base.positions))
    slots = np.asarray(topology.vertex_pair_slots)
    active = np.asarray(topology.slot_active)
    thickness = np.where(active, (220.0 + 90.0 * np.maximum(slots, 0)) * 1.0e-9, 0.0)
    liquid = thickness * area
    state = MultiRegionSurfaceState(
        topology,
        base.positions,
        sheet_fields=liquid[..., None],
        sheet_field_names=("film_liquid_volume",),
    )
    prepared_surface = PreparedMultiRegionSurface(topology, state)
    film_slots = prepare_film_sheet_slots(prepared_surface, state)
    valence = np.sum(
        np.asarray(topology.edge_faces[: topology.edge_count]) >= 0, axis=1
    )
    border_count = int(np.count_nonzero(valence == 3))
    borders = PlateauBorderPlan(
        border_edge_capacity=border_count,
        quad_point_capacity=0,
        density_kg_m3=1000.0,
        viscosity_pa_s=1.0e-3,
        surface_tension_n_m=0.03,
        time_step_s=1.0e-4,
        resource_id="foam-iridescence-test",
    ).prepare(prepared_surface, film_slots, state)
    border_state = borders.initial_state(
        liquid,
        np.zeros_like(liquid),
        np.full((border_count,), 1.0e-4),
    )
    physics = {
        "surface": state.positions,
        "sheet": border_state.sheet_liquid_m3,
        "border": border_state.border_liquid_m3,
    }
    before = array_tree_fingerprint(physics)

    wavelengths = np.arange(380.0, 781.0, 10.0) * 1.0e-9
    illuminant = SpectralIlluminant(
        wavelengths,
        np.ones_like(wavelengths),
        illuminant_id="equal-energy-integration-test",
    )
    appearance = ThinFilmAppearancePlan(
        ThinFilmInterferencePlan(wavelengths, 1.0, 1.33, 1.0),
        SpectralColorimetryPlan(wavelengths, illuminant, exposure=4.0),
        two_sided=True,
    )
    colors = []
    for sheet_index, surface in enumerate(film_slots.surfaces):
        sheet_thickness = film_slots.sheet_content(
            border_state.sheet_liquid_m3, sheet_index
        ) / surface.vertex_area
        support = jnp.ones(sheet_thickness.shape, dtype=jnp.bool_)
        if sheet_index == 0:
            support = support.at[0].set(False)
        result = thin_film_surface_colors(
            appearance,
            sheet_thickness,
            surface.vertex_normal,
            surface.vertex_normal,
            support,
        )
        rejected = support & ~result.accepted
        assert bool(jnp.any(result.accepted))
        assert bool(jnp.all(jnp.isfinite(result.encoded_srgb[result.accepted])))
        assert bool(jnp.all(jnp.isnan(result.encoded_srgb[~result.accepted])))
        assert bool(
            jnp.all(
                result.status[rejected]
                == ThinFilmSurfaceColorStatus.APPEARANCE_REJECTED
            )
        )
        accepted_index = int(np.flatnonzero(np.asarray(result.accepted))[0])
        colors.append(np.asarray(result.linear_srgb[accepted_index]))
        if sheet_index == 0:
            assert int(result.status[0]) == ThinFilmSurfaceColorStatus.UNSUPPORTED

    after = array_tree_fingerprint(physics)
    assert before == after
    assert max(
        np.linalg.norm(first - second)
        for first, second in zip(colors[:-1], colors[1:], strict=True)
    ) > 0.1
