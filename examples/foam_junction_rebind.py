#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Foam sheets drain into a Plateau-border junction across one accepted rebind.

A centimeter double bubble carries film liquid and surfactant on its E
``(vertex, region-pair)`` sheet slots and extensive liquid/surfactant on its
ring of valence-three Plateau borders. The three sheets meeting the ring are
declared once as one P1 ``"junction"`` interface binding; no pairwise sheet
laws are generated.

1. Exchange: every sheet boundary cell drains through its explicit B-on-E
   half-edge routes into the incident border cells. The closure is a declared
   first-order suction ``q = k V_slot`` (surfactant carried at the slot's
   amount per liquid volume) split uniformly over the slot's routes; it is not
   a lubrication model. The border owner applies equal and opposite sheet and
   border rates and its own hydraulic network flux.
2. Refinement: one Plateau border is split at its midpoint by the E event pass.
   One ``phx.lifecycle`` rebind reprepares the surface geometry, the manifold
   sheet views, the junction binding, the border network, and the rendering
   observation. Sheet content crosses through E's conservative sheet-slot
   transition and border content through
   ``PreparedPlateauBorder.reprepare_after_events``: retained borders keep
   their content and the split border shares it by child length.
3. Rendering: thin-film colors are evaluated from the published thickness
   ``V_slot / A_slot``; the physics state is fingerprinted before and after.
4. Refusal: bursting the separating sheet (a rupture) has no declared border
   content rule, so staging raises ``PlateauBorderTransportError`` and the
   published composition keeps exchanging unchanged.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.applications.foams import (
    PlateauBorderBoundaryFlux,
    PlateauBorderPlan,
    PlateauBorderResult,
    PlateauBorderState,
    PlateauBorderStatus,
    PlateauBorderTransportError,
    PreparedPlateauBorder,
)
from phydrax.geometry.multiregion_surface import (
    apply_surface_burst,
    apply_surface_events,
    EdgeSplitProposal,
    multiregion_topology_epoch,
    MultiRegionSurfaceSeed,
    MultiRegionSurfaceState,
    MultiRegionSurfaceValidationPolicy,
    PreparedMultiRegionSurface,
    seed_double_bubble,
    SurfaceBurstResult,
    SurfaceEventPassResult,
    SurfaceEventPolicy,
)
from phydrax.interfacial_transport import prepare_film_sheet_slots, PreparedFilmSheetSlots
from phydrax.rendering import ThinFilmAppearancePlan, ThinFilmSurfaceColorResult
from phydrax.solver.coupling import (
    InterfaceBinding,
    InterfaceEndpoint,
    InterfaceSource,
    SheetViewAttachment,
)


lc = phx.lifecycle

LIQUID = "film_liquid_volume"
SURFACTANT = "film_surfactant_amount"
THICKNESS_M = 600.0e-9
SURFACTANT_MOL_M2 = 6.0e-6  # both leaflets
BORDER_AREA_M2 = 4.0e-10
BORDER_CONCENTRATION_MOL_M3 = 1.0
SUCTION_RATE_S = 0.2
TIME_STEP_S = 0.05
EXCHANGE_STEPS = 6
CAMERA_M = (0.0, 0.0, 0.2)
VALIDATION = MultiRegionSurfaceValidationPolicy(profile="dry_foam")
EVENT_POLICY = SurfaceEventPolicy(validation=VALIDATION)
DERIVED = (
    "border/network",
    "film/sheet-views",
    "junction/plateau-border",
    "render/thin-film",
    "surface/geometry",
)
TRANSPORTED = ("border/liquid", "border/surfactant", "film/liquid", "film/surfactant")

type SurfaceEvent = SurfaceEventPassResult | SurfaceBurstResult


class Foam(NamedTuple):
    """Owner artifacts and extensive state of one accepted foam boundary."""

    geometry: MultiRegionSurfaceState
    film_slots: PreparedFilmSheetSlots
    junction: InterfaceBinding
    borders: PreparedPlateauBorder
    state: PlateauBorderState
    appearance: ThinFilmAppearancePlan


class Rendering(NamedTuple):
    thickness_m: tuple[Array, ...]
    colors: tuple[ThinFilmSurfaceColorResult, ...]


def appearance() -> ThinFilmAppearancePlan:
    wavelengths = np.arange(380.0, 781.0, 10.0, dtype=np.float64) * 1.0e-9
    illuminant = phx.rendering.SpectralIlluminant(
        wavelengths, np.ones_like(wavelengths), illuminant_id="equal-energy"
    )
    return ThinFilmAppearancePlan(
        phx.optics.wave.ThinFilmInterferencePlan(wavelengths, 1.0, 1.33, 1.0),
        phx.rendering.SpectralColorimetryPlan(wavelengths, illuminant, exposure=4.0),
        two_sided=True,
    )


def junction_binding(film_slots: PreparedFilmSheetSlots, /) -> InterfaceBinding:
    """One explicit junction of every sheet with Plateau-border support."""
    supported = np.asarray(
        film_slots.boundary_route_valid & film_slots.boundary_plateau_supported
    )
    sheets = np.unique(np.asarray(film_slots.boundary_sheet_indices)[supported])
    views = film_slots.views
    pairs = tuple(views.views[index].region_ids for index in sheets.tolist())
    endpoints = tuple(
        InterfaceEndpoint(
            f"sheet:{pair[0]}|{pair[1]}",
            SheetViewAttachment(views, pair),
            fields={"liquid": "film/liquid", "surfactant": "film/surfactant"},
        )
        for pair in pairs
    )
    return InterfaceBinding(
        "plateau-border-network",
        InterfaceSource.sheet_views(views, pairs),
        "junction",
        endpoints,
    )


def prepare_foam(
    seed: MultiRegionSurfaceSeed,
    /,
    *,
    resource_id: str,
    border_edge_capacity: int,
    quad_point_capacity: int,
) -> Foam:
    """Uniform films and borders on one validated dry-foam epoch."""
    topology = seed.topology(
        seed.capacity_plan(resource_id=resource_id, headroom=2.0, event_capacity=16)
    )
    geometry = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, geometry, policy=VALIDATION)
    film_slots = prepare_film_sheet_slots(surface, geometry)
    slot_area = surface.slot_areas(geometry.positions)
    valence = np.sum(np.asarray(topology.edge_faces[: topology.edge_count]) >= 0, axis=1)
    edges = np.asarray(topology.edges[: topology.edge_count])[valence == 3]
    points = np.asarray(geometry.positions)
    midpoint = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    gravity = np.zeros((3,), dtype=np.float64)
    gravity[int(np.argmax(np.ptp(midpoint, axis=0)))] = -9.81
    borders = PlateauBorderPlan(
        border_edge_capacity=border_edge_capacity,
        quad_point_capacity=quad_point_capacity,
        density_kg_m3=1000.0,
        viscosity_pa_s=1.0e-3,
        surface_tension_n_m=0.03,
        gravity_m_s2=gravity,
        time_step_s=TIME_STEP_S,
        resource_id=resource_id,
    ).prepare(surface, film_slots, geometry)
    state = borders.initial_state(
        jnp.where(topology.slot_active, THICKNESS_M * slot_area, 0.0),
        jnp.where(topology.slot_active, SURFACTANT_MOL_M2 * slot_area, 0.0),
        BORDER_AREA_M2,
        border_surfactant_concentration_mol_m3=BORDER_CONCENTRATION_MOL_M3,
    )
    return Foam(
        geometry, film_slots, junction_binding(film_slots), borders, state, appearance()
    )


def build(*, ring_points: int = 8) -> Foam:
    return prepare_foam(
        seed_double_bubble(1.0e-2, 0.8e-2, ring_points=ring_points),
        resource_id="foam-junction-rebind",
        border_edge_capacity=2 * ring_points,
        quad_point_capacity=0,
    )


def boundary_flux(foam: Foam, /) -> PlateauBorderBoundaryFlux:
    """Declared first-order suction of every sheet boundary cell into its borders."""
    slots = foam.film_slots
    routes = slots.boundary_route_valid & slots.boundary_plateau_supported
    boundary = slots.boundary_slot_rate(routes.astype(jnp.float64)) > 0.0
    liquid = jnp.where(boundary, SUCTION_RATE_S * foam.state.sheet_liquid_m3, 0.0)
    surfactant = jnp.where(
        boundary, SUCTION_RATE_S * foam.state.sheet_surfactant_mol, 0.0
    )
    return PlateauBorderBoundaryFlux(
        slots.distribute_slot_boundary_rate(liquid),
        slots.distribute_slot_boundary_rate(surfactant),
        jnp.zeros_like(foam.state.sheet_liquid_m3),
        jnp.zeros_like(foam.state.border_liquid_m3),
    )


def exchange(foam: Foam, steps: int, /) -> tuple[Foam, tuple[PlateauBorderResult, ...]]:
    """Advance accepted border steps under the declared sheet-to-border exchange."""
    results: list[PlateauBorderResult] = []
    for _ in range(steps):
        result = foam.borders.step(foam.state, boundary_flux(foam))
        status = PlateauBorderStatus(int(result.evidence.status))
        if status is not PlateauBorderStatus.ACCEPTED:
            raise RuntimeError(f"Plateau-border exchange was refused: {status.name}.")
        foam = foam._replace(state=result.state)
        results.append(result)
    return foam, tuple(results)


def _content(
    value: Array,
    entry_id: str,
    semantics_id: str,
    owner_id: str,
    structure_id: str,
    dependency: lc.CompositionDependency,
    /,
) -> lc.CompositionEntry:
    return lc.CompositionEntry(
        value,
        entry_id=entry_id,
        role="physical-state",
        owner_id=owner_id,
        structure_id=structure_id,
        revision_id=phx.array_tree_fingerprint(value)["sha256"],
        semantics_id=semantics_id,
        dependencies=(dependency,),
    )


def entries(foam: Foam, /) -> dict[str, lc.CompositionEntry]:
    """Explicitly identified owner artifacts and state of one foam boundary."""
    topology = foam.borders.surface.topology
    epoch = multiregion_topology_epoch(topology, foam.geometry.positions).epoch_id
    adapter = foam.film_slots.adapter_id
    geometry = lc.CompositionEntry(
        foam.geometry,
        entry_id="surface/geometry",
        role="topology",
        owner_id="multiregion-surface",
        structure_id=epoch,
        revision_id=epoch,
        semantics_id="multiregion-surface-geometry",
    )
    film = lc.CompositionEntry(
        foam.film_slots,
        entry_id="film/sheet-views",
        role="discretization",
        owner_id="film",
        structure_id=adapter,
        revision_id=adapter,
        semantics_id="b-on-e-film-sheet-slots",
        dependencies=(geometry.binding("structure"),),
    )
    junction = lc.CompositionEntry(
        foam.junction,
        entry_id="junction/plateau-border",
        role="interface-route",
        owner_id="foam",
        structure_id=foam.junction.binding_id,
        revision_id=foam.junction.binding_id,
        semantics_id="plateau-border-junction",
        dependencies=(film.binding("structure"),),
    )
    network = lc.CompositionEntry(
        foam.borders,
        entry_id="border/network",
        role="interface-route",
        owner_id="plateau-border",
        structure_id=foam.borders.prepared_id,
        revision_id=foam.borders.prepared_id,
        semantics_id=foam.borders.plan.plan_id,
        dependencies=(film.binding("structure"), junction.binding("structure")),
    )
    render = lc.CompositionEntry(
        foam.appearance,
        entry_id="render/thin-film",
        role="observation",
        owner_id="rendering",
        structure_id=adapter,
        revision_id=adapter,
        semantics_id=foam.appearance.plan_id,
        dependencies=(film.binding("structure"),),
    )
    rim = foam.state.unresolved_rim_content_m3
    sheet = geometry.binding("structure")
    border = network.binding("structure")
    state = foam.state
    return {
        item.entry_id: item
        for item in (
            geometry,
            film,
            junction,
            network,
            render,
            _content(
                state.sheet_liquid_m3,
                "film/liquid",
                "sheet-slot-liquid-volume-m3",
                "multiregion-surface",
                epoch,
                sheet,
            ),
            _content(
                state.sheet_surfactant_mol,
                "film/surfactant",
                "sheet-slot-surfactant-amount-mol",
                "multiregion-surface",
                epoch,
                sheet,
            ),
            _content(
                state.border_liquid_m3,
                "border/liquid",
                "plateau-border-liquid-volume-m3",
                "plateau-border",
                epoch,
                border,
            ),
            _content(
                state.border_surfactant_mol,
                "border/surfactant",
                "plateau-border-surfactant-amount-mol",
                "plateau-border",
                epoch,
                border,
            ),
            lc.CompositionEntry(
                rim,
                entry_id="border/rim",
                role="physical-state",
                owner_id="plateau-border",
                structure_id="plateau-border-rim-ledger",
                revision_id=phx.array_tree_fingerprint(rim)["sha256"],
                semantics_id="unresolved-rim-liquid-volume-m3",
            ),
        )
    }


def compose(foam: Foam, /) -> lc.Composition:
    return lc.Composition(tuple(entries(foam).values()), boundary_id="foam-boundary")


def from_composition(composition: lc.Composition, /) -> Foam:
    borders: PreparedPlateauBorder = composition.value("border/network")
    state = PlateauBorderState(
        composition.value("film/liquid"),
        composition.value("film/surfactant"),
        composition.value("border/liquid"),
        composition.value("border/surfactant"),
        unresolved_rim_content_m3=composition.value("border/rim"),
        geometry_revision=borders.geometry_revision,
        topology_id=borders.surface.topology.topology_id,
    )
    return Foam(
        composition.value("surface/geometry"),
        composition.value("film/sheet-views"),
        composition.value("junction/plateau-border"),
        borders,
        state,
        composition.value("render/thin-film"),
    )


def surface_state(composition: lc.Composition, /) -> MultiRegionSurfaceState:
    """E state carrying the composition's sheet content as extensive fields."""
    foam = from_composition(composition)
    return MultiRegionSurfaceState(
        foam.borders.surface.topology,
        foam.geometry.positions,
        sheet_fields=jnp.stack(
            (foam.state.sheet_liquid_m3, foam.state.sheet_surfactant_mol), axis=-1
        ),
        sheet_field_names=(LIQUID, SURFACTANT),
    )


def split_event(
    composition: lc.Composition, border: int = 0, /
) -> SurfaceEventPassResult:
    """Midpoint split of one Plateau border (all three films refine with it)."""
    borders: PreparedPlateauBorder = composition.value("border/network")
    topology = borders.surface.topology
    edge = np.asarray(borders.border_edges)[border]
    ids = np.asarray(topology.vertex_global_ids)[edge]
    return apply_surface_events(
        topology,
        surface_state(composition),
        [EdgeSplitProposal((int(ids[0]), int(ids[1])))],
        policy=EVENT_POLICY,
    )


def burst_event(composition: lc.Composition, /) -> SurfaceBurstResult:
    """Rupture of the double bubble's separating sheet through the E5 burst."""
    borders: PreparedPlateauBorder = composition.value("border/network")
    topology = borders.surface.topology
    finite = np.flatnonzero(np.asarray(topology.region_finite[: topology.region_count]))
    pairs = np.asarray(topology.region_pairs[: topology.region_pair_count])
    pair = int(np.flatnonzero(np.all(pairs == np.sort(finite)[None, :], axis=1))[0])
    faces = np.asarray(topology.face_active) & (np.asarray(topology.face_pairs) == pair)
    labels = pairs[pair]
    return apply_surface_burst(
        topology,
        surface_state(composition),
        (topology.region_ids[labels[0]], topology.region_ids[labels[1]]),
        np.asarray(topology.face_global_ids)[faces].tolist(),
        policy=EVENT_POLICY,
    )


def stage_rebind(
    composition: lc.Composition, event: SurfaceEvent, /
) -> lc.CompositionRebind:
    """Stage every owner's target artifact and explicit state transport.

    Raises ``PlateauBorderTransportError`` for events without a declared
    border-content rule; nothing is published and ``composition`` is unchanged.
    """
    transition = event.transition
    if not event.committed or transition is None:
        raise ValueError("Only a committed surface event can be rebound.")
    source = from_composition(composition)
    topology, state = event.topology, event.state
    surface = PreparedMultiRegionSurface(topology, state, policy=VALIDATION)
    film_slots = prepare_film_sheet_slots(surface, state)
    adaptation = source.borders.reprepare_after_events(
        event.evidence, surface, film_slots, state
    )
    border = adaptation.transition
    moved = PlateauBorderState(
        state.sheet_fields[..., state.sheet_field_names.index(LIQUID)],
        state.sheet_fields[..., state.sheet_field_names.index(SURFACTANT)],
        border.apply(source.state.border_liquid_m3).values,
        border.apply(source.state.border_surfactant_mol).values,
        unresolved_rim_content_m3=source.state.unresolved_rim_content_m3,
        geometry_revision=adaptation.prepared.geometry_revision,
        topology_id=topology.topology_id,
    )
    target = Foam(
        MultiRegionSurfaceState(topology, state.positions),
        film_slots,
        junction_binding(film_slots),
        adaptation.prepared,
        moved,
        source.appearance,
    )
    staged = entries(target)
    routes = {"film": transition, "border": border}
    transports = tuple(
        routes[item.split("/")[0]].composition_transport(
            composition.entry(item), staged[item]
        )
        for item in TRANSPORTED
    )
    return lc.CompositionRebind(
        composition,
        retain=("border/rim",),
        reprepare=tuple(staged[item] for item in DERIVED),
        transports=transports,
    )


def render(composition: lc.Composition, /) -> Rendering:
    """Thin-film colors of the published sheet thickness (one-way consumer)."""
    film_slots: PreparedFilmSheetSlots = composition.value("film/sheet-views")
    plan: ThinFilmAppearancePlan = composition.value("render/thin-film")
    liquid = composition.value("film/liquid")
    camera = jnp.asarray(CAMERA_M)
    thickness: list[Array] = []
    colors: list[ThinFilmSurfaceColorResult] = []
    for index, surface in enumerate(film_slots.surfaces):
        sheet = film_slots.sheet_content(liquid, index) / surface.vertex_area
        thickness.append(sheet)
        colors.append(
            phx.rendering.thin_film_surface_colors(
                plan,
                sheet,
                surface.vertex_normal,
                camera - surface.coordinates,
                jnp.ones(sheet.shape, dtype=jnp.bool_),
            )
        )
    return Rendering(tuple(thickness), tuple(colors))


def physics_fingerprint(composition: lc.Composition, /) -> str:
    state = {
        entry.entry_id: entry.value
        for entry in composition.entries
        if entry.role == "physical-state"
    }
    return phx.array_tree_fingerprint(state)["sha256"]


def totals(foam: Foam, /) -> tuple[float, float]:
    """Film plus border liquid volume and surfactant amount."""
    return float(foam.state.total_liquid_m3()), float(foam.state.total_surfactant_mol())


def run() -> dict[str, object]:
    foam, exchanged = exchange(build(), EXCHANGE_STEPS)
    source = compose(foam)
    staged = stage_rebind(source, split_event(source))
    receipt = lc.commit_composition_rebind(staged, accepted_boundary=True)
    if not receipt.published:
        raise RuntimeError("The accepted refinement rebind was not published.")
    published = receipt.composition
    before = physics_fingerprint(published)
    rendering = render(published)
    after = physics_fingerprint(published)
    refined = from_composition(published)
    try:
        stage_rebind(published, burst_event(published))
    except PlateauBorderTransportError:
        rupture_refused = True
    else:
        rupture_refused = False
    continued, more = exchange(refined, EXCHANGE_STEPS)
    liquid = [abs(float(item.evidence.liquid_conservation_residual_m3)) for item in more]
    accepted = [bool(item.accepted) for item in (*exchanged, *more)]
    supported = [np.asarray(item.accepted) for item in rendering.colors]
    return {
        "published": receipt.published,
        "reprepared": list(receipt.reprepared),
        "remapped": list(receipt.remapped),
        "retained": list(receipt.retained),
        "border_count": [foam.borders.border_count, refined.borders.border_count],
        "junction_sheets": len(refined.junction.endpoints),
        "totals_before_rebind": totals(foam),
        "totals_after_rebind": totals(refined),
        "totals_after_exchange": totals(continued),
        "maximum_liquid_step_residual_m3": max(liquid),
        "exchange_steps_accepted": all(accepted),
        "rendered_samples": int(sum(np.count_nonzero(item) for item in supported)),
        "thickness_range_m": [
            float(min(jnp.min(item) for item in rendering.thickness_m)),
            float(max(jnp.max(item) for item in rendering.thickness_m)),
        ],
        "physics_unchanged_by_rendering": before == after,
        "rupture_refused": rupture_refused,
    }


if __name__ == "__main__":
    print(run())
