#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Foam sheets, Plateau-border junction, refinement rebind, and one-way rendering.

Workflow: `examples/foam_junction_rebind.py`. A double bubble's three sheets
drain through explicit half-edge routes into its ring of Plateau borders; one
border is split through a single cross-owner composition rebind; the published
thickness is rendered. References are independent of the border and film
owners: the junction reference walks the multiregion edge-face incidence and
sheet slots directly, and the split reference is the uniform cross-section and
concentration of a finite-volume border cell cut at its midpoint.
"""

from collections import Counter
from typing import NamedTuple

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from examples import foam_junction_rebind as ex
from phydrax.applications.foams import PlateauBorderResult, PlateauBorderTransportError
from phydrax.geometry.multiregion_surface import (
    apply_surface_events,
    EdgeCollapseProposal,
    PreparedMultiRegionSurface,
    propose_t1_pops,
    SurfaceEventPassResult,
    T1PopProposal,
)
from phydrax.solver.coupling import SheetViewAttachment
from tests._support.multiregion_foams import t1_cluster


lc = phx.lifecycle
_EPS = float(np.finfo(np.float64).eps)


class Rebound(NamedTuple):
    initial: ex.Foam
    exchanged: tuple[PlateauBorderResult, ...]
    foam: ex.Foam
    source: lc.Composition
    event: SurfaceEventPassResult
    staged: lc.CompositionRebind
    receipt: lc.CompositionRebindReceipt


@pytest.fixture(scope="module")
def rebound() -> Rebound:
    initial = ex.build()
    foam, exchanged = ex.exchange(initial, 3)
    source = ex.compose(foam)
    event = ex.split_event(source)
    staged = ex.stage_rebind(source, event)
    receipt = lc.commit_composition_rebind(staged, accepted_boundary=True)
    return Rebound(initial, exchanged, foam, source, event, staged, receipt)


def _junction_reference(
    foam: ex.Foam, slot_rate: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Border inflow and sheet loss of the declared suction from raw E incidence."""
    topology = foam.borders.surface.topology
    edge_faces = np.asarray(topology.edge_faces)
    face_pairs = np.asarray(topology.face_pairs)
    vertex_slots = np.asarray(topology.vertex_pair_slots)
    edges = np.asarray(topology.edges)
    borders = np.asarray(foam.borders.border_global_edge_indices)[
        : foam.borders.border_count
    ]
    sheets = {
        int(edge): sorted(
            {int(face_pairs[face]) for face in edge_faces[edge] if face >= 0}
        )
        for edge in borders
    }
    halves = Counter(
        (int(vertex), pair)
        for edge in borders
        for vertex in edges[edge]
        for pair in sheets[int(edge)]
    )
    inflow = np.zeros((borders.size,))
    loss = np.zeros_like(slot_rate)
    for local, edge in enumerate(borders):
        for pair in sheets[int(edge)]:
            for vertex in edges[edge]:
                slot = int(np.flatnonzero(vertex_slots[vertex] == pair)[0])
                inflow[local] += slot_rate[vertex, slot] / halves[(int(vertex), pair)]
                loss[vertex, slot] = slot_rate[vertex, slot]
    return inflow, loss, [len(pairs) for pairs in sheets.values()]


@pytest.mark.parametrize(
    ("sheet_field", "border_rate", "internal_flux"),
    (
        ("sheet_liquid_m3", "border_liquid_m3_s", "border_internal_flux_m3_s"),
        (
            "sheet_surfactant_mol",
            "border_surfactant_mol_s",
            "border_internal_surfactant_flux_mol_s",
        ),
    ),
    ids=("liquid", "surfactant"),
)
def test_sheet_half_edge_fluxes_feed_the_junction_equal_and_opposite(
    rebound: Rebound, sheet_field: str, border_rate: str, internal_flux: str
) -> None:
    foam = rebound.initial
    rates = foam.borders.rates(foam.state, ex.boundary_flux(foam))
    content = np.asarray(getattr(foam.state, sheet_field))
    inflow, loss, incidence = _junction_reference(foam, ex.SUCTION_RATE_S * content)
    assert incidence == [3] * foam.borders.border_count
    count = foam.borders.border_count
    endpoint_flux = np.asarray(getattr(rates, internal_flux)).reshape((-1, 2))
    exchanged = np.asarray(getattr(rates, border_rate)) + endpoint_flux.sum(axis=1)
    np.testing.assert_allclose(exchanged[:count], inflow, rtol=1e-13, atol=0.0)
    assert np.all(exchanged[count:] == 0.0)
    sheet_rate = np.asarray(
        rates.sheet_liquid_m3_s
        if sheet_field == "sheet_liquid_m3"
        else rates.sheet_surfactant_mol_s
    )
    np.testing.assert_allclose(sheet_rate, -loss, rtol=1e-14, atol=0.0)
    scale = np.abs(sheet_rate).sum()
    assert abs(sheet_rate.sum() + exchanged.sum()) <= 16 * _EPS * scale
    # The border network's own junctions pass the hydraulic flux through: the
    # eliminated node balance is a pressure residual at roundoff of the potential.
    active = np.asarray(foam.borders.network_vertex_active)
    residual = np.asarray(rates.node_pressure_balance_pa)[active]
    potential = np.asarray(rates.node_hydraulic_potential_pa)[active]
    assert np.max(np.abs(residual)) <= 64 * _EPS * np.max(np.abs(potential))


def test_junction_is_one_explicit_incidence_of_the_three_border_sheets(
    rebound: Rebound,
) -> None:
    foam = rebound.initial
    junction = foam.junction
    assert junction.incidence == "junction" and len(junction.endpoints) == 3
    junction.require_current(foam.film_slots.views)
    topology = foam.borders.surface.topology
    borders = np.asarray(foam.borders.border_global_edge_indices)[
        : foam.borders.border_count
    ]
    faces = np.asarray(topology.edge_faces)[borders]
    pairs = np.asarray(topology.region_pairs)[
        np.unique(np.asarray(topology.face_pairs)[faces[faces >= 0]])
    ]
    expected = {tuple(topology.region_ids[label] for label in row) for row in pairs}
    attached = {
        endpoint.attachment.region_ids
        for endpoint in junction.endpoints
        if isinstance(endpoint.attachment, SheetViewAttachment)
    }
    assert attached == expected


def test_exchange_conserves_liquid_and_surfactant_to_roundoff(rebound: Rebound) -> None:
    previous = rebound.initial.state
    for result in rebound.exchanged:
        assert bool(result.accepted)
        state = result.state
        for total in ("total_liquid_m3", "total_surfactant_mol"):
            before = float(getattr(previous, total)())
            assert abs(float(getattr(state, total)()) - before) <= 16 * _EPS * before
        moved = float(result.evidence.sheet_border_liquid_transfer_m3)
        gained = float(np.sum(state.border_liquid_m3 - previous.border_liquid_m3))
        assert moved > 0.0
        np.testing.assert_allclose(gained, moved, rtol=1e-11)
        previous = state


def test_split_publishes_reprepared_owners_and_explicit_transports(
    rebound: Rebound,
) -> None:
    receipt = rebound.receipt
    assert receipt.published and receipt.boundary_accepted
    assert all(receipt.transport_accepted)
    assert receipt.reprepared == ex.DERIVED
    assert receipt.remapped == ex.TRANSPORTED
    assert receipt.retained == ("border/rim",)
    assert receipt.invalidated == () and receipt.consumed == ()
    assert receipt.composition.structure_id != receipt.source_structure_id
    before = ex.from_composition(rebound.source)
    after = ex.from_composition(receipt.composition)
    assert (before.borders.border_count, after.borders.border_count) == (8, 9)
    after.junction.require_current(after.film_slots.views)
    with pytest.raises(ValueError, match="No owner of interface endpoint"):
        before.junction.require_current(after.film_slots.views)


def _borders(
    composition: lc.Composition, entry_id: str, /
) -> dict[tuple[int, int], tuple[float, float]]:
    """Border content and independently measured length keyed by stable vertex IDs."""
    network = composition.value("border/network")
    ids = np.asarray(network.surface.topology.vertex_global_ids)
    points = np.asarray(composition.value("surface/geometry").positions)
    edges = np.asarray(network.border_edges)[: network.border_count]
    content = np.asarray(composition.value(entry_id))
    return {
        (int(min(ids[a], ids[b])), int(max(ids[a], ids[b]))): (
            float(content[index]),
            float(np.linalg.norm(points[b] - points[a])),
        )
        for index, (a, b) in enumerate(edges)
    }


def test_split_conserves_content_and_keeps_the_border_cross_section(
    rebound: Rebound,
) -> None:
    source, published = rebound.source, rebound.receipt.composition
    for entry_id in ex.TRANSPORTED:
        before = np.asarray(source.value(entry_id))
        after = np.asarray(published.value(entry_id))
        assert np.all(after >= 0.0)
        assert abs(after.sum() - before.sum()) <= 16 * _EPS * np.abs(before).sum()
    liquid = (_borders(source, "border/liquid"), _borders(published, "border/liquid"))
    amount = (
        _borders(source, "border/surfactant"),
        _borders(published, "border/surfactant"),
    )
    retained = liquid[0].keys() & liquid[1].keys()
    assert len(retained) == len(liquid[0]) - 1
    assert all(liquid[1][key][0] == liquid[0][key][0] for key in retained)
    assert all(amount[1][key][0] == amount[0][key][0] for key in retained)
    ((parent_volume, parent_length),) = [
        liquid[0][key] for key in liquid[0].keys() - retained
    ]
    ((parent_amount, _),) = [amount[0][key] for key in amount[0].keys() - retained]
    children = liquid[1].keys() - retained
    assert len(children) == 2
    for key in children:
        volume, length = liquid[1][key]
        np.testing.assert_allclose(
            volume / length, parent_volume / parent_length, rtol=1e-13
        )
        np.testing.assert_allclose(
            amount[1][key][0] / volume, parent_amount / parent_volume, rtol=1e-13
        )


def test_refined_network_continues_the_exchange(rebound: Rebound) -> None:
    refined = ex.from_composition(rebound.receipt.composition)
    continued, results = ex.exchange(refined, 2)
    before = float(refined.state.total_liquid_m3())
    assert all(bool(result.accepted) for result in results)
    assert abs(float(continued.state.total_liquid_m3()) - before) <= 16 * _EPS * before


def test_rendering_the_published_thickness_leaves_physics_bitwise_unchanged(
    rebound: Rebound,
) -> None:
    published = rebound.receipt.composition
    before = phx.array_tree_fingerprint(eqx.filter(published, eqx.is_array))
    rendering = ex.render(published)
    after = phx.array_tree_fingerprint(eqx.filter(published, eqx.is_array))
    assert before == after
    film_slots = published.value("film/sheet-views")
    liquid = np.asarray(published.value("film/liquid")).reshape(-1)
    areas = np.asarray(
        film_slots.multiregion.slot_areas(published.value("surface/geometry").positions)
    ).reshape(-1)
    for view, thickness, colors in zip(
        film_slots.views.views, rendering.thickness_m, rendering.colors, strict=True
    ):
        slots = np.asarray(view.slot_indices)
        np.testing.assert_allclose(
            np.asarray(thickness), liquid[slots] / areas[slots], rtol=1e-12
        )
        accepted = np.asarray(colors.accepted)
        assert accepted.any()
        assert np.all(np.isfinite(np.asarray(colors.encoded_srgb)[accepted]))


def _continues_bitwise(composition: lc.Composition, reference: ex.Foam, /) -> bool:
    """The old composition keeps exchanging exactly as the never-rebound foam."""
    resumed, _ = ex.exchange(ex.from_composition(composition), 2)
    untouched, _ = ex.exchange(reference, 2)
    return bool(eqx.tree_equal(resumed.state, untouched.state))


def test_rupture_without_a_border_content_rule_is_refused(rebound: Rebound) -> None:
    source = rebound.source
    identity = source.composition_id
    burst = ex.burst_event(source)
    assert burst.committed
    with pytest.raises(PlateauBorderTransportError, match="BURST"):
        ex.stage_rebind(source, burst)
    assert source.composition_id == identity
    assert _continues_bitwise(source, rebound.foam)


def test_t1_pop_without_a_border_content_rule_is_refused() -> None:
    foam = ex.prepare_foam(
        t1_cluster(),
        resource_id="t1-foam",
        border_edge_capacity=64,
        quad_point_capacity=16,
    )
    composition = ex.compose(foam)
    state = ex.surface_state(composition)
    surface = PreparedMultiRegionSurface(
        foam.borders.surface.topology, state, policy=ex.VALIDATION
    )
    (proposal,) = propose_t1_pops(surface, state, maximum_film_diameter=0.2).proposals
    assert isinstance(proposal, T1PopProposal)
    event = apply_surface_events(
        surface.topology, state, [proposal], policy=ex.EVENT_POLICY
    )
    assert event.committed
    with pytest.raises(PlateauBorderTransportError, match="T1_POP"):
        ex.stage_rebind(composition, event)
    assert _continues_bitwise(composition, foam)


def test_collapsing_a_border_has_no_declared_coarsening_rule(rebound: Rebound) -> None:
    source = rebound.source
    borders = source.value("border/network")
    ids = np.asarray(borders.surface.topology.vertex_global_ids)[
        np.asarray(borders.border_edges)[0]
    ]
    event = apply_surface_events(
        borders.surface.topology,
        ex.surface_state(source),
        [EdgeCollapseProposal((int(ids[0]), int(ids[1])))],
        policy=ex.EVENT_POLICY,
    )
    assert event.committed
    with pytest.raises(PlateauBorderTransportError, match="no physical content source"):
        ex.stage_rebind(source, event)
    assert _continues_bitwise(source, rebound.foam)


@pytest.mark.parametrize(
    ("variant", "message"),
    (
        ("omitted", "lack an explicit rebind disposition: border/surfactant"),
        ("invalidated", "'border/surfactant' .* cannot be invalidated"),
        ("retained", "'border/surfactant' is stale against the structure"),
    ),
    ids=("omitted", "invalidated", "retained"),
)
def test_unknown_border_state_transport_is_refused(
    rebound: Rebound, variant: str, message: str
) -> None:
    staged = rebound.staged
    kept = tuple(
        route
        for route in staged.transports
        if route.source_entry_ids != ("border/surfactant",)
    )
    options: dict[str, dict[str, tuple[str, ...]]] = {
        "omitted": {},
        "invalidated": {"invalidate": ("border/surfactant",)},
        "retained": {"retain": ("border/rim", "border/surfactant")},
    }
    with pytest.raises(ValueError, match=message):
        lc.CompositionRebind(
            rebound.source,
            retain=options[variant].get("retain", staged.retained),
            reprepare=tuple(staged.candidate.entry(item) for item in staged.reprepared),
            transports=kept,
            invalidate=options[variant].get("invalidate", ()),
        )
    assert _continues_bitwise(rebound.source, rebound.foam)


def test_rejected_boundary_publishes_nothing(rebound: Rebound) -> None:
    receipt = lc.commit_composition_rebind(rebound.staged, accepted_boundary=False)
    assert not receipt.published and not receipt.boundary_accepted
    assert receipt.composition is rebound.source
