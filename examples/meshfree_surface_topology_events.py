# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Meshfree surfactant across committed physical foam topology events.

A meshfree surfactant concentration lives on the active sheet slots of a
multiregion foam (one support point per vertex and region pair, measured by
its barycentric slot area). The multiregion surface remains the physical event
authority: a merge zips two facing bubble films into one shared wall, and a
catenoid past its stability limit pinches into two disks while its core region
splits into two child regions. Every committed pass becomes one meshfree epoch
change whose lineage carries the event kinds, CCD and volume evidence, and
region parents. One staged composition transaction remaps the current field and
every live history through the authority's own conservative transfer and
rebuilds the support; a refused route would leave the source composition
unchanged. Concentration values stay differentiable through each frozen
transfer, while the event selection has no derivative.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.meshfree import (
    commit_meshfree_epoch,
    MeshfreeEpochReceipt,
    MeshfreeEventLineage,
    MeshfreeSheetSupport,
    MeshfreeSurfaceEventEpoch,
    stage_meshfree_epoch,
    surface_event_epoch,
)
from phydrax.geometry.multiregion_surface import (
    apply_surface_events,
    MergeProposal,
    MultiRegionRemeshPlan,
    MultiRegionSurfaceCapacityPlan,
    MultiRegionSurfaceSeed,
    MultiRegionSurfaceState,
    MultiRegionSurfaceTopology,
    PreparedMultiRegionSurface,
    propose_merges,
    propose_pinches,
    propose_remesh,
    RegionSplitProposal,
    seed_catenoid,
    seed_sphere,
    SurfaceEventPassResult,
    SurfaceEventPolicy,
    SurfaceMergePolicy,
)
from phydrax.lifecycle import Composition, CompositionEntry


OWNER = "meshfree-surfactant"
EPOCH, SUPPORT = "foam/epoch", "foam/support"
CURRENT, CLOCK = "foam/surfactant", "foam/clock"
HISTORIES = ("foam/surfactant-history/0", "foam/surfactant-history/1")


def two_bubbles() -> tuple[MultiRegionSurfaceTopology, MultiRegionSurfaceState]:
    """Two unit bubbles separated by a thin gap of ambient gas."""
    first = seed_sphere(1.0, subdivisions=1)
    second = seed_sphere(1.0, center=(2.04, 0.0, 0.0), subdivisions=1)
    count = first.positions.shape[0]
    seed = MultiRegionSurfaceSeed(
        np.concatenate((first.positions, second.positions)),
        np.concatenate((first.faces, second.faces + count)),
        np.concatenate(
            (
                np.tile((0, 2), (first.faces.shape[0], 1)),
                np.tile((1, 2), (second.faces.shape[0], 1)),
            )
        ),
        ("left", "right", "ambient"),
        ("finite", "finite", "boundary"),
        source="two-bubbles",
    )
    counts = seed.counts()
    plan = MultiRegionSurfaceCapacityPlan(
        vertex_capacity=2 * counts.vertex,
        edge_capacity=2 * counts.edge,
        face_capacity=2 * counts.face,
        region_capacity=counts.region + 2,
        region_pair_capacity=2 * counts.region_pair + 2,
        maximum_edge_valence=3,
        maximum_vertex_region_pairs=9,
        resource_id="meshfree-merge",
        event_capacity=16,
    )
    topology = seed.topology(plan)
    return topology, _surfactant_state(topology, seed.state(topology))


def _surfactant_state(
    topology: MultiRegionSurfaceTopology, base: MultiRegionSurfaceState, /
) -> MultiRegionSurfaceState:
    """Extensive surfactant amount: a nonuniform concentration times slot area."""
    areas = np.asarray(
        PreparedMultiRegionSurface(topology, base).slot_areas(base.positions)
    )
    x = np.asarray(base.positions)[:, 0]
    concentration = 3.0e-7 * (1.0 + 0.1 * x)
    return MultiRegionSurfaceState(
        topology,
        base.positions,
        sheet_fields=(concentration[:, None] * areas)[:, :, None],
        sheet_field_names=("surfactant",),
    )


def surfactant_composition(
    topology: MultiRegionSurfaceTopology, state: MultiRegionSurfaceState, /
) -> Composition:
    """Epoch, support, current surfactant, two live histories and a controller."""
    support = MeshfreeSheetSupport(topology, state)
    epoch = CompositionEntry(
        support.epoch,
        entry_id=EPOCH,
        role="topology",
        owner_id=OWNER,
        structure_id=support.epoch.epoch_id,
        revision_id=support.epoch.epoch_id,
        semantics_id="foam-sheet-epoch",
    )
    bound = (epoch.binding("structure"),)
    current = support.concentration(state.sheet_fields[..., 0])
    entries = [
        epoch,
        CompositionEntry(
            support,
            entry_id=SUPPORT,
            role="discretization",
            owner_id=OWNER,
            structure_id=support.support_id,
            revision_id=support.support_id,
            semantics_id="sheet-slot-support",
            dependencies=bound,
        ),
    ]
    for entry_id, scale in ((CURRENT, 1.0), (HISTORIES[0], 0.95), (HISTORIES[1], 0.9)):
        entries.append(
            CompositionEntry(
                scale * current,
                entry_id=entry_id,
                role="physical-state" if entry_id == CURRENT else "history",
                owner_id=OWNER,
                structure_id=support.epoch.epoch_id,
                revision_id=f"{entry_id}:{support.epoch.epoch_id}",
                semantics_id="surfactant-concentration",
                dependencies=bound,
            )
        )
    entries.append(
        CompositionEntry(
            jnp.asarray(1.0e-3),
            entry_id=CLOCK,
            role="optimizer-state",
            owner_id=OWNER,
            structure_id="scalar-step",
            revision_id="step-0",
            semantics_id="step-size-controller",
        )
    )
    return Composition(entries, boundary_id="accepted-step")


def advance_epoch(
    composition: Composition,
    topology: MultiRegionSurfaceTopology,
    state: MultiRegionSurfaceState,
    result: SurfaceEventPassResult,
    /,
) -> tuple[MeshfreeSurfaceEventEpoch, MeshfreeEpochReceipt]:
    """Consume one committed pass as one staged meshfree epoch transaction."""
    event = surface_event_epoch(topology, state, result)
    transition = event.transition()
    target_epoch = CompositionEntry(
        event.change.target,
        entry_id=EPOCH,
        role="topology",
        owner_id=OWNER,
        structure_id=event.change.target.epoch_id,
        revision_id=event.change.change_id,
        semantics_id="foam-sheet-epoch",
    )
    support = CompositionEntry(
        event.target,
        entry_id=SUPPORT,
        role="discretization",
        owner_id=OWNER,
        structure_id=event.target.support_id,
        revision_id=event.target.support_id,
        semantics_id="sheet-slot-support",
        dependencies=(target_epoch.binding("structure"),),
    )
    candidate = stage_meshfree_epoch(
        composition,
        event.change,
        epoch_entry=EPOCH,
        remap={name: transition for name in (CURRENT, *HISTORIES)},
        reprepare=(support,),
    )
    return event, commit_meshfree_epoch(candidate, accepted_boundary=True)


def _content(composition: Composition, entry_id: str, /) -> float:
    support = composition.value(SUPPORT)
    return float(jnp.vdot(support.measures, composition.value(entry_id)))


def _drift(composition: Composition, entry_id: str, expected: float, /) -> float:
    return abs(_content(composition, entry_id) - expected) / expected


@dataclass(frozen=True)
class MergeMetrics:
    published: bool
    kinds: tuple[str, ...]
    topology_changed: bool
    ccd_certified: bool
    constant_preserving: bool
    area_defect: float
    authority_mismatch: float
    wall_slots: int
    continuation_published: bool
    continuation_kinds: tuple[str, ...]
    epoch_index: int
    content_drift: float
    history_drift: float
    value_jvp_finite: bool


@dataclass(frozen=True)
class PinchMetrics:
    pinched: bool
    epochs: int
    region_ids: tuple[str, ...]
    region_parents: tuple[str, ...]
    removed_regions: tuple[str, ...]
    child_labels: tuple[str, ...]
    content_drift: float
    history_drift: float


def _lineage(event: MeshfreeSurfaceEventEpoch, /) -> MeshfreeEventLineage:
    lineage = event.change.lineage
    if lineage is None:
        raise RuntimeError("A surface-event epoch always carries its lineage.")
    return lineage


def run_merge_workflow() -> MergeMetrics:
    """Merge two facing films, then continue through a refinement pass."""
    topology, state = two_bubbles()
    composition = surfactant_composition(topology, state)
    initial = _content(composition, CURRENT)
    prepared = PreparedMultiRegionSurface(topology, state)
    search = propose_merges(
        prepared, state, SurfaceMergePolicy(("ambient",), merge_distance=0.4)
    )
    merges = [item for item in search.proposals if isinstance(item, MergeProposal)]
    merged = apply_surface_events(topology, state, merges[:1])
    event, receipt = advance_epoch(composition, topology, state, merged)
    composition = receipt.composition
    authority = event.target.concentration(merged.state.sheet_fields[..., 0])
    mismatch = float(
        jnp.max(jnp.abs(composition.value(CURRENT) - authority))
        / jnp.max(jnp.abs(authority))
    )
    # Continuation on the merged epoch: one refinement pass of the same surface.
    refine = MultiRegionRemeshPlan(
        minimum_edge_length=0.05, maximum_edge_length=0.55, operations=("split",)
    )
    refined = apply_surface_events(
        merged.topology,
        merged.state,
        propose_remesh(
            PreparedMultiRegionSurface(merged.topology, merged.state),
            merged.state,
            refine,
        ),
    )
    second, continued = advance_epoch(composition, merged.topology, merged.state, refined)
    transition = second.transition()
    values = jnp.asarray(composition.value(CURRENT))
    _, jvp = jax.jvp(
        lambda v: transition.apply(v).values, (values,), (jnp.ones_like(values),)
    )
    lineage = _lineage(event)
    final = continued.composition
    return MergeMetrics(
        published=receipt.published,
        kinds=lineage.event_kinds,
        topology_changed=lineage.topology_changed,
        ccd_certified=bool(jnp.all(lineage.ccd_certified)),
        constant_preserving=event.transfer.evidence.constant_preserving,
        area_defect=event.transfer.evidence.obstruction_defect,
        authority_mismatch=mismatch,
        wall_slots=sum(label == ("left", "right") for label in event.target.labels),
        continuation_published=continued.published,
        continuation_kinds=tuple(sorted(set(_lineage(second).event_kinds))),
        epoch_index=final.value(EPOCH).index,
        content_drift=_drift(final, CURRENT, initial),
        history_drift=_drift(final, HISTORIES[1], 0.9 * initial),
        value_jvp_finite=bool(jnp.all(jnp.isfinite(jvp))),
    )


def run_pinch_workflow(*, maximum_passes: int = 12) -> PinchMetrics:
    """Drive a catenoid neck to pinch-off, chaining every committed pass."""
    seed = seed_catenoid(1.0, 0.7, ring_points=12, rows=8, neck_radius=0.04)
    topology = seed.topology(
        seed.capacity_plan(resource_id="meshfree-pinch", headroom=2.0, event_capacity=32)
    )
    state = _surfactant_state(topology, seed.state(topology))
    composition = surfactant_composition(topology, state)
    initial = _content(composition, CURRENT)
    policy = SurfaceEventPolicy(
        fixed_vertex_ids=seed.vertex_set("ring-lower") + seed.vertex_set("ring-upper")
    )
    collapse = MultiRegionRemeshPlan(
        minimum_edge_length=0.06, maximum_edge_length=1.0, operations=("collapse",)
    )
    epochs = 0
    for _ in range(maximum_passes):
        prepared = PreparedMultiRegionSurface(topology, state)
        proposals = [
            *propose_pinches(prepared, state, maximum_neck_perimeter=0.3),
            RegionSplitProposal("core"),
            *propose_remesh(prepared, state, collapse),
        ]
        result = apply_surface_events(topology, state, proposals, policy=policy)
        if not result.committed:
            continue
        event, receipt = advance_epoch(composition, topology, state, result)
        if not receipt.published:
            raise RuntimeError("A committed surface event failed its meshfree epoch.")
        composition, topology, state = receipt.composition, result.topology, result.state
        epochs += 1
        lineage = _lineage(event)
        if "PINCH" in lineage.event_kinds:
            names = {name for pair in event.target.labels for name in pair}
            return PinchMetrics(
                pinched=True,
                epochs=epochs,
                region_ids=topology.region_ids,
                region_parents=tuple(
                    f"{child}<-{'+'.join(parents)}"
                    for child, parents in lineage.region_parents
                ),
                removed_regions=lineage.removed_region_ids,
                child_labels=tuple(sorted(names - {"ambient"})),
                content_drift=_drift(composition, CURRENT, initial),
                history_drift=_drift(composition, HISTORIES[0], 0.95 * initial),
            )
    raise RuntimeError(f"No pinch committed within {maximum_passes} passes.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", choices=("merge", "pinch", "both"), default="both")
    options = parser.parse_args()
    if options.workflow in ("merge", "both"):
        print(run_merge_workflow())
    if options.workflow in ("pinch", "both"):
        print(run_pinch_workflow())


if __name__ == "__main__":
    main()
