#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transactional event passes over a multiregion surface.

`apply_surface_events` owns remeshing and local physical transitions:

1. proposals are ordered canonically (physical transitions, then collapses,
   splits and flips, each by priority and support ids);
2. each proposal's vertex 2-ring must be disjoint from every accepted one;
3. the central fail-closed dispatch drafts the local edit with stable lineage;
4. finite-region volumes are restored locally and exact guards run
   (duplicates, degeneracy, labels/orientation, valence, fans, region
   graph, extinction, folding, local intersections);
5. every motion leg and the restoration motion are certified by inclusion CCD;
6. accepted edits are applied to the host working copy; regions left in
   several components by physical events (or requested explicitly) are split;
7. the candidate topology, sheet-slot/face/region/velocity transfers and the
   full exact validation are built;
8. the dynamic payload commits through `phydrax.lifecycle` only when the
   validation and every transfer certificate pass, otherwise the source
   topology and state are returned unchanged.

`apply_surface_burst` is the typed whole-sheet transaction. It deletes the
complete separating sheet, merges its region labels and returns vanished
sheet-slot content explicitly so the foam owner can commit its rim ledger.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never, final, Literal, TypeAlias, TypeVar

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...discretization._topology_epoch import TopologyEpoch, TopologyEpochTransition
from ...discretization._transfer import TransferGeometryBinding
from ...lifecycle import commit_candidate, TransactionalCandidate
from ...typing import Dim, Float, parse
from ._contracts import (
    MultiRegionKind,
    MultiRegionSurfaceCapacityEvidence,
    MultiRegionSurfaceCounts,
    MultiRegionSurfaceEvidence,
)
from ._edits import (
    _apply_edit,
    _capacity_admits,
    _certify_legs,
    _duplicate_faces,
    _Edit,
    _local_guards,
    _restoration_leg,
    _restore_volumes,
    _ring,
    _slots_of,
    _working_mesh,
    _WorkingMesh,
)
from ._events import (
    MultiRegionSurfaceLineage,
    SurfaceEventKind,
    SurfaceEventPassEvidence,
    SurfaceEventPassResult,
    SurfaceEventPassStatus,
    SurfaceEventPolicy,
    SurfaceEventRecord,
    SurfaceEventStatus,
)
from ._remesh import (
    _collapse_edit,
    _flip_edit,
    _split_edit,
    EdgeCollapseProposal,
    EdgeFlipProposal,
    EdgeSplitProposal,
)
from ._state import MultiRegionSurfaceState
from ._topology import _host_incidence, MultiRegionSurfaceTopology
from ._topology_transitions import (
    _label_components,
    _merge_edit,
    _pinch_edit,
    _t1_edit,
    MergeProposal,
    PinchProposal,
    RegionSplitProposal,
    T1PopProposal,
)
from ._transfers import (
    _combined_routes,
    _pooled_slot_routes,
    BoundedFieldReconstruction,
    ConservativeFieldTransfer,
    ExtensiveTransferEvidence,
    IntensiveReconstructionEvidence,
)
from ._validation import _component_volumes, _geometry_id, validate_multiregion_surface


SurfaceEventProposal: TypeAlias = (
    EdgeSplitProposal
    | EdgeCollapseProposal
    | EdgeFlipProposal
    | T1PopProposal
    | PinchProposal
    | MergeProposal
    | RegionSplitProposal
)

_Proposal = TypeVar("_Proposal")


class _PayloadVertexDim(Dim, minimum=1):
    """Vertex slots."""


class _PayloadSlotDim(Dim, minimum=1):
    """Sheet slots per vertex."""


class _PayloadFieldDim(Dim):
    """Sheet fields."""


class _PayloadRegionDim(Dim, minimum=2):
    """Region slots."""


class _PayloadRegionFieldDim(Dim):
    """Region fields."""


@final
class _SurfacePayload(StrictModule):
    """Capacity-shaped dynamic arrays committed atomically by one pass."""

    __strict_contract__ = True

    positions: Float[_PayloadVertexDim, Literal[3]]
    velocities: Float[_PayloadVertexDim, Literal[3]]
    sheet_fields: Float[_PayloadVertexDim, _PayloadSlotDim, _PayloadFieldDim]
    region_fields: Float[_PayloadRegionDim, _PayloadRegionFieldDim]


def _payload(state: MultiRegionSurfaceState, /) -> _SurfacePayload:
    return _SurfacePayload(
        positions=state.positions,
        velocities=state.velocities,
        sheet_fields=state.sheet_fields,
        region_fields=state.region_fields,
    )


def _expect(proposal: object, kind: type[_Proposal], /) -> _Proposal:
    if not isinstance(proposal, kind):
        raise TypeError(f"Proposal kind does not match {kind.__name__}.")
    return proposal


_EdgeProposal: TypeAlias = EdgeSplitProposal | EdgeCollapseProposal | EdgeFlipProposal


def _expect_edge(proposal: object, /) -> _EdgeProposal:
    if not isinstance(
        proposal, (EdgeSplitProposal, EdgeCollapseProposal, EdgeFlipProposal)
    ):
        raise TypeError("Proposal kind does not match an edge event.")
    return proposal


def _order_key(
    proposal: SurfaceEventProposal, /
) -> tuple[int, int, float, tuple[object, ...]]:
    """Canonical evaluation order: phase, kind rank, priority, identity."""
    match proposal.kind:
        case SurfaceEventKind.T1_POP:
            rank = (0, 0)
            identity: tuple[object, ...] = _expect(
                proposal, T1PopProposal
            ).film_vertex_ids
        case SurfaceEventKind.PINCH:
            rank = (0, 1)
            identity = _expect(proposal, PinchProposal).loop_vertex_ids
        case SurfaceEventKind.MERGE:
            rank = (0, 2)
            identity = _expect(proposal, MergeProposal).face_ids
        case SurfaceEventKind.COLLAPSE:
            rank = (1, 0)
            identity = _expect(proposal, EdgeCollapseProposal).edge_vertex_ids
        case SurfaceEventKind.SPLIT:
            rank = (1, 1)
            identity = _expect(proposal, EdgeSplitProposal).edge_vertex_ids
        case SurfaceEventKind.FLIP:
            rank = (1, 2)
            identity = _expect(proposal, EdgeFlipProposal).edge_vertex_ids
        case SurfaceEventKind.REGION_SPLIT:
            rank = (2, 0)
            identity = (_expect(proposal, RegionSplitProposal).region_id,)
        case SurfaceEventKind.BURST:
            raise RuntimeError("Burst uses the dedicated whole-sheet transaction.")
        case _:
            assert_never(proposal.kind)
    return rank[0], rank[1], proposal.priority, identity


def _merge_support(
    mesh: _WorkingMesh, proposal: MergeProposal, /
) -> tuple[int, ...] | None:
    faces = []
    for face_id in proposal.face_ids:
        match = np.flatnonzero(
            (mesh.face_ids[: mesh.face_count] == face_id)
            & mesh.face_alive[: mesh.face_count]
        )
        if match.size != 1:
            return None
        faces.append(int(match[0]))
    return tuple(faces)


def _support(
    mesh: _WorkingMesh, proposal: SurfaceEventProposal, /
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """``(core entity slots, core vertex slots)`` of a geometric proposal."""
    match proposal.kind:
        case SurfaceEventKind.SPLIT | SurfaceEventKind.COLLAPSE | SurfaceEventKind.FLIP:
            edge = _expect_edge(proposal).edge_vertex_ids
            slots = _slots_of(mesh, edge)
            return None if slots is None else (slots, slots)
        case SurfaceEventKind.T1_POP:
            slots = _slots_of(mesh, _expect(proposal, T1PopProposal).film_vertex_ids)
            return None if slots is None else (slots, slots)
        case SurfaceEventKind.PINCH:
            slots = _slots_of(mesh, _expect(proposal, PinchProposal).loop_vertex_ids)
            return None if slots is None else (slots, slots)
        case SurfaceEventKind.MERGE:
            faces = _merge_support(mesh, _expect(proposal, MergeProposal))
            if faces is None:
                return None
            return faces, tuple(sorted(set(mesh.faces[list(faces)].reshape(-1).tolist())))
        case SurfaceEventKind.REGION_SPLIT:
            raise RuntimeError("Region splits have no vertex support.")
        case SurfaceEventKind.BURST:
            raise RuntimeError("Burst uses the dedicated whole-sheet transaction.")
        case _:
            assert_never(proposal.kind)


def _draft(
    mesh: _WorkingMesh, proposal: SurfaceEventProposal, core: tuple[int, ...], /
) -> _Edit | SurfaceEventStatus:
    """The one fail-closed dispatch over the closed geometric event kinds."""
    match proposal.kind:
        case SurfaceEventKind.SPLIT:
            return _split_edit(mesh, core[0], core[1])
        case SurfaceEventKind.COLLAPSE:
            return _collapse_edit(mesh, core[0], core[1])
        case SurfaceEventKind.FLIP:
            return _flip_edit(mesh, core[0], core[1])
        case SurfaceEventKind.T1_POP:
            return _t1_edit(mesh, _expect(proposal, T1PopProposal), core)
        case SurfaceEventKind.PINCH:
            return _pinch_edit(mesh, _expect(proposal, PinchProposal), core)
        case SurfaceEventKind.MERGE:
            return _merge_edit(mesh, _expect(proposal, MergeProposal), core)
        case SurfaceEventKind.REGION_SPLIT:
            raise RuntimeError("Region splits are applied after the geometric events.")
        case SurfaceEventKind.BURST:
            raise RuntimeError("Burst uses the dedicated whole-sheet transaction.")
        case _:
            assert_never(proposal.kind)


def _evaluate(
    mesh: _WorkingMesh,
    proposal: SurfaceEventProposal,
    core: tuple[int, ...],
    policy: SurfaceEventPolicy,
    /,
) -> tuple[SurfaceEventStatus, _Edit | None, float, float]:
    """Draft, restore, guard and certify one proposal (no mutation)."""
    draft = _draft(mesh, proposal, core)
    if isinstance(draft, SurfaceEventStatus):
        return draft, None, float("nan"), 0.0
    if not _capacity_admits(mesh, draft):
        return SurfaceEventStatus.CAPACITY_EXCEEDED, None, float("nan"), 0.0
    if _duplicate_faces(mesh, draft):
        return SurfaceEventStatus.DUPLICATE_FACE, None, float("nan"), 0.0
    restored, residual, restored_ok = _restore_volumes(mesh, draft, policy)
    if not restored_ok:
        return SurfaceEventStatus.VOLUME_RESTORATION_FAILED, None, float("nan"), residual
    status = _local_guards(mesh, restored, policy)
    if status is not SurfaceEventStatus.ACCEPTED:
        return status, None, float("nan"), residual
    restoration = _restoration_leg(mesh, draft, restored)
    legs = draft.legs + (() if restoration is None else (restoration,))
    certified, impact = _certify_legs(mesh, restored, legs, policy)
    if not certified:
        return SurfaceEventStatus.CCD_NOT_CERTIFIED, None, impact, residual
    return SurfaceEventStatus.ACCEPTED, restored, impact, residual


class _Ledger:
    """Accumulated lineage and transfer groups of the accepted events of one pass."""

    def __init__(self) -> None:
        self.vertex_parents: list[tuple[int, tuple[int, ...]]] = []
        self.face_parents: list[tuple[int, tuple[int, ...]]] = []
        self.removed_vertex_ids: list[int] = []
        self.removed_face_ids: list[int] = []
        self.groups: list[tuple[np.ndarray, np.ndarray]] = []
        self.velocity_parents: dict[int, tuple[int, ...]] = {}
        self.touched_regions: set[int] = set()


def _record_edit(
    mesh: _WorkingMesh,
    proposal: SurfaceEventProposal,
    support_ids: Sequence[int],
    edit: _Edit,
    impact: float,
    residual: float,
    ledger: _Ledger,
    /,
) -> SurfaceEventRecord:
    """Apply an accepted edit to the working mesh and record its lineage."""
    removed_vertex_ids = [int(mesh.vertex_ids[v]) for v in edit.removed_vertices]
    removed_face_ids = [int(mesh.face_ids[f]) for f in edit.removed_faces]
    parent_vertex_ids = [
        tuple(int(mesh.vertex_ids[p]) for p in parents)
        for parents in edit.new_vertex_parents
    ]
    parent_face_ids = [
        tuple(int(mesh.face_ids[p]) for p in parents) for parents in edit.new_face_parents
    ]
    survivors = [
        (int(mesh.vertex_ids[slot]), tuple(int(mesh.vertex_ids[p]) for p in parents))
        for slot, parents in edit.survivors
    ]
    base = mesh.vertex_count
    created_vertices, created_faces, created_slots = _apply_edit(mesh, edit)
    ledger.vertex_parents += list(zip(created_vertices, parent_vertex_ids, strict=True))
    ledger.vertex_parents += survivors
    ledger.face_parents += list(zip(created_faces, parent_face_ids, strict=True))
    ledger.removed_vertex_ids += removed_vertex_ids
    ledger.removed_face_ids += removed_face_ids
    for offset, parents in enumerate(edit.new_vertex_parents):
        ledger.velocity_parents[base + offset] = parents
    for slot, parents in edit.survivors:
        ledger.velocity_parents[slot] = parents
    slots = np.asarray(created_slots, dtype=np.int64)
    if edit.groups is None:
        ledger.groups.append((np.asarray(edit.removed_faces, dtype=np.int64), slots))
    else:
        for source, target in edit.groups:
            ledger.groups.append(
                (
                    np.asarray(source, dtype=np.int64),
                    slots[np.asarray(target, dtype=np.int64)],
                )
            )
    if proposal.kind in (
        SurfaceEventKind.T1_POP,
        SurfaceEventKind.PINCH,
        SurfaceEventKind.MERGE,
    ):
        ledger.touched_regions |= set(edit.new_labels.reshape(-1).tolist())
    return SurfaceEventRecord(
        proposal.kind,
        SurfaceEventStatus.ACCEPTED,
        support_ids,
        priority=proposal.priority,
        removed_vertex_ids=removed_vertex_ids,
        created_vertex_ids=created_vertices,
        removed_face_ids=removed_face_ids,
        created_face_ids=created_faces,
        ccd_time_of_impact=impact,
        ccd_certified=True,
        volume_residual=residual,
    )


def _support_ids(mesh: _WorkingMesh, faces: Sequence[int], /) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(mesh.vertex_ids[v])
                for v in mesh.faces[list(faces)].reshape(-1).tolist()
            }
        )
    )


def _proposal_ids(proposal: SurfaceEventProposal, /) -> tuple[int, ...]:
    match proposal.kind:
        case SurfaceEventKind.SPLIT | SurfaceEventKind.COLLAPSE | SurfaceEventKind.FLIP:
            return _expect_edge(proposal).edge_vertex_ids
        case SurfaceEventKind.T1_POP:
            return _expect(proposal, T1PopProposal).film_vertex_ids
        case SurfaceEventKind.PINCH:
            return _expect(proposal, PinchProposal).loop_vertex_ids
        case SurfaceEventKind.MERGE | SurfaceEventKind.REGION_SPLIT:
            return ()
        case SurfaceEventKind.BURST:
            raise RuntimeError("Burst uses the dedicated whole-sheet transaction.")
        case _:
            assert_never(proposal.kind)


def _geometric_phase(
    mesh: _WorkingMesh,
    proposals: Sequence[SurfaceEventProposal],
    policy: SurfaceEventPolicy,
    ledger: _Ledger,
    /,
) -> tuple[list[SurfaceEventRecord], int]:
    records: list[SurfaceEventRecord] = []
    claimed: set[int] = set()
    budget = mesh.topology.plan.event_capacity
    accepted = 0
    for proposal in proposals:
        support = _support(mesh, proposal)
        if support is None:
            records.append(
                SurfaceEventRecord(
                    proposal.kind,
                    SurfaceEventStatus.SUPPORT_INVALID,
                    _proposal_ids(proposal),
                    priority=proposal.priority,
                )
            )
            continue
        core, vertices = support
        ids = (
            _support_ids(mesh, core)
            if proposal.kind is SurfaceEventKind.MERGE
            else (_proposal_ids(proposal))
        )
        ring = _ring(mesh, vertices, 2)
        if ring & claimed:
            status, edit, impact, residual = (
                SurfaceEventStatus.CONFLICT,
                None,
                float("nan"),
                0.0,
            )
        elif accepted >= budget:
            status, edit, impact, residual = (
                SurfaceEventStatus.EVENT_BUDGET_EXCEEDED,
                None,
                float("nan"),
                0.0,
            )
        else:
            status, edit, impact, residual = _evaluate(mesh, proposal, core, policy)
        if edit is None:
            records.append(
                SurfaceEventRecord(
                    proposal.kind,
                    status,
                    ids,
                    priority=proposal.priority,
                    ccd_time_of_impact=impact,
                    volume_residual=residual,
                )
            )
            continue
        records.append(_record_edit(mesh, proposal, ids, edit, impact, residual, ledger))
        claimed |= ring
        accepted += 1
    return records, accepted


class _Regions:
    """Region table after the region-split phase."""

    def __init__(self, topology: MultiRegionSurfaceTopology, /) -> None:
        self.ids: list[str] = list(topology.region_ids)
        self.kinds: list[str] = list(topology.region_kinds)
        self.parent: list[int] = list(range(topology.region_count))
        self.region_parents: list[tuple[str, tuple[str, ...]]] = []
        self.removed: list[str] = []


def _split_region(
    regions: _Regions,
    region: int,
    parts: np.ndarray,
    sides: np.ndarray,
    component: np.ndarray,
    side_labels: np.ndarray,
    /,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Relabel the ordered components ``parts`` of ``region`` into child regions."""
    parent_id = regions.ids[region]
    children: list[str] = []
    suffix = 0
    for part in parts.tolist():
        while f"{parent_id}/{suffix}" in regions.ids:
            suffix += 1
        child = f"{parent_id}/{suffix}"
        suffix += 1
        if children:
            index = len(regions.ids)
            regions.ids.append(child)
            regions.kinds.append(regions.kinds[region])
            regions.parent.append(region)
        else:
            index = region
            regions.ids[region] = child
        children.append(child)
        side_labels[sides[component[sides] == part]] = index
    regions.removed.append(parent_id)
    parentage = tuple((child, (parent_id,)) for child in children)
    regions.region_parents += list(parentage)
    return parentage


def _region_phase(
    mesh: _WorkingMesh,
    requested: Sequence[RegionSplitProposal],
    ledger: _Ledger,
    budget: int,
    /,
) -> tuple[list[SurfaceEventRecord], _Regions, np.ndarray]:
    """Split labels left in several components; returns records, table and face labels.

    Finite regions touched by physical events split automatically when they
    consist of several positive shells (a cavity-bounding shell refuses the
    split); any label splits by face-side components on explicit request.
    """
    topology = mesh.topology
    regions = _Regions(topology)
    faces_alive = np.flatnonzero(mesh.face_alive[: mesh.face_count])
    labels = mesh.labels[faces_alive].copy()
    records: list[SurfaceEventRecord] = []
    targets = {
        int(r): False for r in sorted(ledger.touched_regions) if mesh.region_finite[r]
    }
    for proposal in requested:
        if proposal.region_id in topology.region_ids:
            targets[topology.region_index(proposal.region_id)] = True
        else:
            records.append(_region_record(SurfaceEventStatus.SUPPORT_INVALID))
    if not targets:
        return records, regions, labels
    vertices = np.flatnonzero(mesh.vertex_alive[: mesh.vertex_count])
    compact = np.full((mesh.vertex_count,), -1, dtype=np.int64)
    compact[vertices] = np.arange(vertices.size)
    faces = compact[mesh.faces[faces_alive]]
    points = mesh.positions[vertices]
    _, component = _label_components(
        points, faces, labels, mesh.region_finite, topology.region_count, mesh.mode
    )
    volumes, _ = _component_volumes(points, faces, labels, component)
    side_labels = labels.reshape(-1)
    side_faces = np.repeat(mesh.face_ids[faces_alive], 2)
    explicit_accepted = 0
    for region, explicit in sorted(targets.items()):
        sides = np.flatnonzero(side_labels == region)
        parts = np.unique(component[sides])
        if parts.size <= 1:
            if explicit:
                records.append(_region_record(SurfaceEventStatus.NOT_TRIGGERED))
            continue
        if bool(mesh.region_finite[region]) and np.any(volumes[parts] <= 0.0):
            records.append(_region_record(SurfaceEventStatus.SUPPORT_INVALID))
            continue
        if explicit and explicit_accepted >= budget:
            records.append(_region_record(SurfaceEventStatus.EVENT_BUDGET_EXCEEDED))
            continue
        explicit_accepted += int(explicit)
        first_face = [
            int(np.min(side_faces[sides[component[sides] == part]])) for part in parts
        ]
        ordered = parts[np.argsort(first_face, kind="stable")]
        parentage = _split_region(regions, region, ordered, sides, component, side_labels)
        records.append(_region_record(SurfaceEventStatus.ACCEPTED, parentage))
    return records, regions, side_labels.reshape((-1, 2))


def _region_record(
    status: SurfaceEventStatus,
    parentage: tuple[tuple[str, tuple[str, ...]], ...] = (),
    /,
) -> SurfaceEventRecord:
    return SurfaceEventRecord(
        SurfaceEventKind.REGION_SPLIT, status, (), priority=0.0, region_parents=parentage
    )


def _region_order(regions: _Regions, /) -> np.ndarray:
    """Children follow their parent in the region table (stable)."""
    keys = [(regions.parent[index], index) for index in range(len(regions.ids))]
    return np.asarray(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int64)


def _areas(points: np.ndarray, faces: np.ndarray, /) -> np.ndarray:
    corners = points[faces]
    return 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )


@dataclass(frozen=True, slots=True)
class _Transfers:
    """Transfers and their certificates from the source to a validated candidate."""

    state: MultiRegionSurfaceState
    sheet: ConservativeFieldTransfer
    face: ConservativeFieldTransfer | None
    sheet_evidence: ExtensiveTransferEvidence | None
    region_evidence: ExtensiveTransferEvidence | None
    velocity_evidence: IntensiveReconstructionEvidence


@dataclass(frozen=True, slots=True)
class _Candidate:
    """Assembled candidate epoch of one pass (transfers only when it validated)."""

    topology: MultiRegionSurfaceTopology
    validation: MultiRegionSurfaceEvidence
    transfers: _Transfers | None


def _sheet_transfer(
    source: MultiRegionSurfaceTopology,
    source_points: np.ndarray,
    target: MultiRegionSurfaceTopology,
    target_points: np.ndarray,
    mesh: _WorkingMesh,
    faces_alive: np.ndarray,
    parents: np.ndarray,
    ledger: _Ledger,
    /,
) -> ConservativeFieldTransfer:
    """Corner-pooled sheet-slot transfer from the source to the candidate epoch."""
    width = source.slot_width
    source_faces = source.host_faces()
    target_faces = target.host_faces()
    source_slots = source_faces * width + np.asarray(
        source.face_corner_slots[: source.face_count], dtype=np.int64
    )
    target_slots = target_faces * width + np.asarray(
        target.face_corner_slots[: target.face_count], dtype=np.int64
    )
    region_count = source.region_count

    def pair_keys(labels: np.ndarray, /) -> np.ndarray:
        ordered = np.sort(labels, axis=1)
        return ordered[:, 0] * region_count + ordered[:, 1]

    source_pairs = pair_keys(source.host_face_labels())
    target_pairs = pair_keys(parents[target.host_face_labels()])
    compact_face = np.full((mesh.face_count,), -1, dtype=np.int64)
    compact_face[faces_alive] = np.arange(faces_alive.size)
    corner_targets = np.full(source_faces.shape, -1, dtype=np.int64)
    kept = np.flatnonzero(compact_face[: source.face_count] >= 0)
    vertex_ids = np.asarray(source.vertex_global_ids[: source.vertex_count])
    target_ids = np.asarray(target.vertex_global_ids[: target.vertex_count])
    rows = compact_face[kept]
    same = (
        vertex_ids[source_faces[kept]][:, :, None]
        == target_ids[target_faces[rows]][:, None, :]
    )
    corner_targets[kept] = np.take_along_axis(
        target_slots[rows], np.argmax(same, axis=2), axis=1
    )
    groups = [
        (source_rows, compact_face[target_rows])
        for source_rows, target_rows in ledger.groups
    ]
    sources, targets, weights = _pooled_slot_routes(
        source_slots,
        _areas(source_points, source_faces),
        source_pairs,
        target_slots,
        _areas(target_points, target_faces),
        target_pairs,
        corner_targets,
        groups,
    )
    return ConservativeFieldTransfer(
        sources,
        targets,
        weights,
        source_active=np.asarray(source.slot_active).reshape(-1),
        target_active=np.asarray(target.slot_active).reshape(-1),
    )


def _face_transfer(
    source: MultiRegionSurfaceTopology,
    target: MultiRegionSurfaceTopology,
    target_points: np.ndarray,
    mesh: _WorkingMesh,
    faces_alive: np.ndarray,
    parents: np.ndarray,
    ledger: _Ledger,
    /,
) -> ConservativeFieldTransfer:
    """Sparse conservative transfer of face-integrated application fields."""
    compact_face = np.full((mesh.face_count,), -1, dtype=np.int64)
    compact_face[faces_alive] = np.arange(faces_alive.size)
    kept = np.flatnonzero(compact_face[: source.face_count] >= 0)
    sources = [kept]
    targets = [compact_face[kept]]
    weights = [np.ones((kept.size,), dtype=np.float64)]
    region_count = source.region_count

    def pair_keys(labels: np.ndarray, /) -> np.ndarray:
        ordered = np.sort(labels, axis=1)
        return ordered[:, 0] * region_count + ordered[:, 1]

    source_pairs = pair_keys(source.host_face_labels())
    target_pairs = pair_keys(parents[target.host_face_labels()])
    target_areas = _areas(target_points, target.host_faces())
    for replaced, replacement in ledger.groups:
        target_rows = compact_face[replacement]
        target_rows = target_rows[target_rows >= 0]
        if target_rows.size == 0:
            raise RuntimeError("An event group lost every replacement face.")
        for pair in np.unique(source_pairs[replaced]):
            members = replaced[source_pairs[replaced] == pair]
            receivers = target_rows[target_pairs[target_rows] == pair]
            receivers = target_rows if receivers.size == 0 else receivers
            share = target_areas[receivers] / np.sum(target_areas[receivers])
            sources.append(np.repeat(members, receivers.size))
            targets.append(np.tile(receivers, members.size))
            weights.append(np.tile(share, members.size))
    source_routes, target_routes, route_weights = _combined_routes(
        np.concatenate(sources), np.concatenate(targets), np.concatenate(weights)
    )
    return ConservativeFieldTransfer(
        source_routes,
        target_routes,
        route_weights,
        source_active=np.asarray(source.face_active),
        target_active=np.asarray(target.face_active),
    )


def _region_transfer(
    source: MultiRegionSurfaceTopology,
    target: MultiRegionSurfaceTopology,
    target_points: np.ndarray,
    parents: np.ndarray,
    /,
) -> ConservativeFieldTransfer:
    """Identity for kept regions; component volume (or area) shares for splits."""
    faces = target.host_faces()
    labels = target.host_face_labels()
    corners = target_points[faces] - np.mean(target_points, axis=0)
    volume = np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2]), axis=1) / 6.0
    area = _areas(target_points, faces)
    count = target.region_count
    volumes = np.zeros((count,), dtype=np.float64)
    np.add.at(volumes, labels[:, 0], volume)
    np.add.at(volumes, labels[:, 1], -volume)
    areas = np.bincount(labels.reshape(-1), weights=np.repeat(area, 2), minlength=count)
    finite = np.asarray(target.region_finite[:count])
    measure = np.where(finite, volumes, areas)
    shares = (
        measure
        / np.bincount(parents[:count], weights=measure, minlength=source.region_count)[
            parents[:count]
        ]
    )
    return ConservativeFieldTransfer(
        parents[:count],
        np.arange(count),
        shares,
        source_active=np.asarray(source.region_active),
        target_active=np.asarray(target.region_active),
    )


def _velocity_reconstruction(
    source: MultiRegionSurfaceTopology,
    target: MultiRegionSurfaceTopology,
    vertices: np.ndarray,
    ledger: _Ledger,
    /,
) -> BoundedFieldReconstruction:
    sources, targets = [], []
    for compact, slot in enumerate(vertices.tolist()):
        parents = ledger.velocity_parents.get(slot, (slot,))
        sources += list(parents)
        targets += [compact] * len(parents)
    return BoundedFieldReconstruction(
        np.asarray(sources, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.ones((len(sources),), dtype=np.float64),
        source_active=np.asarray(source.vertex_active),
        target_active=np.asarray(target.vertex_active),
    )


def _assemble(
    topology: MultiRegionSurfaceTopology,
    state: MultiRegionSurfaceState,
    mesh: _WorkingMesh,
    regions: _Regions,
    face_labels: np.ndarray,
    ledger: _Ledger,
    policy: SurfaceEventPolicy,
    /,
) -> _Candidate | MultiRegionSurfaceCounts:
    """Candidate topology, exact validation and transfers (or refused counts)."""
    order = _region_order(regions)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    vertices = np.flatnonzero(mesh.vertex_alive[: mesh.vertex_count])
    compact = np.full((mesh.vertex_count,), -1, dtype=np.int64)
    compact[vertices] = np.arange(vertices.size)
    faces_alive = np.flatnonzero(mesh.face_alive[: mesh.face_count])
    faces = compact[mesh.faces[faces_alive]]
    labels = rank[face_labels]
    incidence = _host_incidence(faces, labels, vertices.size)
    counts = MultiRegionSurfaceCounts(
        vertex=vertices.size,
        edge=incidence.edges.shape[0],
        face=faces.shape[0],
        region=order.size,
        region_pair=incidence.region_pairs.shape[0],
        edge_valence=int(np.max(np.sum(incidence.edge_faces >= 0, axis=1))),
        vertex_region_pairs=int(np.max(incidence.vertex_slot_count)),
    )
    if not topology.plan.capacity_evidence(counts).admitted:
        return counts
    kinds = tuple(regions.kinds[index] for index in order.tolist())
    candidate = MultiRegionSurfaceTopology(
        topology.plan,
        faces,
        labels,
        [regions.ids[index] for index in order.tolist()],
        [parse(kind, MultiRegionKind, "region_kinds") for kind in kinds],
        vertex_count=vertices.size,
        vertex_global_ids=mesh.vertex_ids[vertices],
        face_global_ids=mesh.face_ids[faces_alive],
        epoch=topology.epoch + 1,
        domain=topology.domain,
    )
    points = mesh.positions[vertices]
    padded = np.zeros((topology.vertex_capacity, 3), dtype=np.float64)
    padded[: vertices.size] = points
    validation = validate_multiregion_surface(
        candidate, MultiRegionSurfaceState(candidate, padded), policy=policy.validation
    )
    if not validation.accepted:
        return _Candidate(candidate, validation, None)
    source_points = np.asarray(state.positions[: topology.vertex_count], dtype=np.float64)
    parents = np.zeros((candidate.region_capacity,), dtype=np.int64)
    parents[: order.size] = np.asarray(regions.parent, dtype=np.int64)[order]
    sheet = _sheet_transfer(
        topology, source_points, candidate, points, mesh, faces_alive, parents, ledger
    )
    face = _face_transfer(topology, candidate, points, mesh, faces_alive, parents, ledger)
    region = _region_transfer(topology, candidate, points, parents)
    velocity = _velocity_reconstruction(topology, candidate, vertices, ledger)
    width, fields = topology.slot_width, len(state.sheet_field_names)
    source_sheet = state.sheet_fields.reshape((topology.vertex_capacity * width, fields))
    target_sheet = sheet.apply(source_sheet)
    target_region = region.apply(state.region_fields)
    target_velocity = velocity.apply(state.velocities)
    new_state = MultiRegionSurfaceState(
        candidate,
        padded,
        velocities=target_velocity,
        sheet_fields=target_sheet.reshape((topology.vertex_capacity, width, fields)),
        region_fields=target_region,
        sheet_field_names=state.sheet_field_names,
        region_field_names=state.region_field_names,
    )
    transfers = _Transfers(
        state=new_state,
        sheet=sheet,
        face=face,
        sheet_evidence=sheet.evidence(source_sheet, target_sheet) if fields else None,
        region_evidence=region.evidence(state.region_fields, target_region)
        if state.region_field_names
        else None,
        velocity_evidence=velocity.evidence(state.velocities, target_velocity),
    )
    return _Candidate(candidate, validation, transfers)


def _accepted_flag(transfers: _Transfers, /) -> Array:
    flag = transfers.velocity_evidence.successful
    for evidence in (transfers.sheet_evidence, transfers.region_evidence):
        if evidence is not None:
            flag = flag & evidence.successful
    return flag


def _epoch(topology: MultiRegionSurfaceTopology, points: np.ndarray, /) -> TopologyEpoch:
    return TopologyEpoch(
        topology.epoch,
        _geometry_id(topology, points),
        topology.topology_id,
        topology.lineage_id,
    )


def multiregion_topology_epoch(
    topology: MultiRegionSurfaceTopology, positions: ArrayLike, /
) -> TopologyEpoch:
    """The topology epoch an event pass certifies for ``topology`` at ``positions``.

    The identity binds the incidence, the stable lineage, and the exact active
    vertex positions, so it equals ``SurfaceEventPassEvidence.source_epoch`` of
    a pass applied to this geometry. Consumers use its ``epoch_id`` as the
    structure identity of epoch-owned state that crosses an event pass.
    """
    if not isinstance(topology, MultiRegionSurfaceTopology):
        raise TypeError("topology must be a MultiRegionSurfaceTopology.")
    points = np.asarray(positions, dtype=np.float64)
    if points.shape != (topology.vertex_capacity, 3):
        raise ValueError("positions must have shape (vertex_capacity, 3).")
    return _epoch(topology, points[: topology.vertex_count])


def _evidence(
    status: SurfaceEventPassStatus,
    records: Sequence[SurfaceEventRecord],
    policy: SurfaceEventPolicy,
    source_epoch: TopologyEpoch,
    /,
    *,
    lineage: MultiRegionSurfaceLineage | None = None,
    candidate: _Candidate | None = None,
    capacity: MultiRegionSurfaceCapacityEvidence | None = None,
    target_epoch: TopologyEpoch | None = None,
    transaction_id: str | None = None,
) -> SurfaceEventPassEvidence:
    committed = status is SurfaceEventPassStatus.COMMITTED
    transfers = None if candidate is None else candidate.transfers
    if candidate is not None:
        capacity = candidate.validation.capacity
    return SurfaceEventPassEvidence(
        status=status,
        committed=committed,
        records=tuple(records),
        accepted_count=sum(record.accepted for record in records),
        lineage=lineage,
        capacity=capacity,
        validation=None if candidate is None else candidate.validation,
        sheet_transfer=None if transfers is None else transfers.sheet_evidence,
        region_transfer=None if transfers is None else transfers.region_evidence,
        velocity_reconstruction=None
        if transfers is None
        else transfers.velocity_evidence,
        source_epoch=source_epoch,
        target_epoch=target_epoch,
        transaction_id=transaction_id,
        derivative_available=False,
        policy_id=policy.policy_id,
        evidence_id=canonical_fingerprint(
            {
                "kind": "surface-event-pass-evidence",
                "status": status.name,
                "records": [record.record_id for record in records],
                "source_epoch": source_epoch.epoch_id,
                "target_epoch": None if target_epoch is None else target_epoch.epoch_id,
                "lineage": None if lineage is None else lineage.lineage_id,
                "capacity": None if capacity is None else list(capacity.exceeded),
                "policy": policy.policy_id,
            }
        ),
    )


def apply_surface_events(
    topology: MultiRegionSurfaceTopology,
    state: MultiRegionSurfaceState,
    proposals: Sequence[SurfaceEventProposal],
    /,
    *,
    policy: SurfaceEventPolicy | None = None,
) -> SurfaceEventPassResult:
    """Apply one transactional pass of topology events (host boundary).

    Proposals of any kind may be mixed; they are evaluated in canonical order
    independent of the input order. At most ``topology.plan.event_capacity``
    events are accepted per pass. The result holds the committed candidate or
    the unchanged source objects, and evidence for every proposal.
    """
    if not isinstance(topology, MultiRegionSurfaceTopology):
        raise TypeError("topology must be a MultiRegionSurfaceTopology.")
    if not isinstance(state, MultiRegionSurfaceState):
        raise TypeError("state must be a MultiRegionSurfaceState.")
    policy_ = SurfaceEventPolicy() if policy is None else policy
    if not isinstance(policy_, SurfaceEventPolicy):
        raise TypeError("policy must be a SurfaceEventPolicy.")
    state.require_topology(topology)
    admissible = (
        EdgeSplitProposal,
        EdgeCollapseProposal,
        EdgeFlipProposal,
        T1PopProposal,
        PinchProposal,
        MergeProposal,
        RegionSplitProposal,
    )
    if isinstance(proposals, (str, bytes)) or not all(
        isinstance(proposal, admissible) for proposal in proposals
    ):
        raise TypeError("proposals must be surface event proposals.")
    ordered = sorted(proposals, key=_order_key)
    geometric = [p for p in ordered if not isinstance(p, RegionSplitProposal)]
    requested = [p for p in ordered if isinstance(p, RegionSplitProposal)]
    mesh = _working_mesh(topology, state, policy_)
    ledger = _Ledger()
    records, accepted = _geometric_phase(mesh, geometric, policy_, ledger)
    region_records, regions, face_labels = _region_phase(
        mesh, requested, ledger, topology.plan.event_capacity - accepted
    )
    records += region_records
    source_points = np.asarray(state.positions[: topology.vertex_count], dtype=np.float64)
    source_epoch = _epoch(topology, source_points)
    if not any(record.accepted for record in records):
        evidence = _evidence(
            SurfaceEventPassStatus.NO_ACCEPTED_EVENTS, records, policy_, source_epoch
        )
        return SurfaceEventPassResult(topology, state, evidence, None, None, None, None)
    assembled = _assemble(topology, state, mesh, regions, face_labels, ledger, policy_)
    if isinstance(assembled, MultiRegionSurfaceCounts):
        evidence = _evidence(
            SurfaceEventPassStatus.CAPACITY_EXCEEDED,
            records,
            policy_,
            source_epoch,
            capacity=topology.plan.capacity_evidence(assembled),
        )
        return SurfaceEventPassResult(topology, state, evidence, None, None, None, None)
    candidate = assembled
    target_points = np.asarray(
        mesh.positions[np.flatnonzero(mesh.vertex_alive[: mesh.vertex_count])],
        dtype=np.float64,
    )
    target_epoch = _epoch(candidate.topology, target_points)
    lineage = MultiRegionSurfaceLineage(
        source_lineage_id=topology.lineage_id,
        target_lineage_id=candidate.topology.lineage_id,
        source_epoch=topology.epoch,
        target_epoch=candidate.topology.epoch,
        vertex_parents=ledger.vertex_parents,
        face_parents=ledger.face_parents,
        region_parents=regions.region_parents,
        removed_vertex_ids=ledger.removed_vertex_ids,
        removed_face_ids=ledger.removed_face_ids,
        removed_region_ids=regions.removed,
    )
    transfers = candidate.transfers
    if transfers is None:
        evidence = _evidence(
            SurfaceEventPassStatus.CANDIDATE_VALIDATION_FAILED,
            records,
            policy_,
            source_epoch,
            lineage=lineage,
            candidate=candidate,
            target_epoch=target_epoch,
        )
        return SurfaceEventPassResult(topology, state, evidence, None, None, None, None)
    transaction = TransactionalCandidate(
        _payload(state),
        _payload(transfers.state),
        candidate.validation,
        _accepted_flag(transfers),
        topology.lineage_id,
    )
    commit = commit_candidate(transaction)
    # Host boundary: the static topology epoch follows the committed flag.
    committed = bool(commit.committed)
    status = (
        SurfaceEventPassStatus.COMMITTED
        if committed
        else SurfaceEventPassStatus.TRANSFER_FAILED
    )
    evidence = _evidence(
        status,
        records,
        policy_,
        source_epoch,
        lineage=lineage,
        candidate=candidate,
        target_epoch=target_epoch,
        transaction_id=transaction.candidate_id,
    )
    if not committed:
        return SurfaceEventPassResult(topology, state, evidence, None, None, None, None)
    committed_state = MultiRegionSurfaceState(
        candidate.topology,
        commit.state.positions,
        velocities=commit.state.velocities,
        sheet_fields=commit.state.sheet_fields,
        region_fields=commit.state.region_fields,
        sheet_field_names=state.sheet_field_names,
        region_field_names=state.region_field_names,
    )
    geometry = TransferGeometryBinding(
        _geometry_id(topology, source_points),
        _geometry_id(
            candidate.topology,
            np.asarray(
                committed_state.positions[: candidate.topology.vertex_count],
                dtype=np.float64,
            ),
        ),
        "topology-correspondence",
        source_topology_id=topology.topology_id,
        target_topology_id=candidate.topology.topology_id,
        coverage_defect=None,
    )
    transition = transfers.sheet.epoch_transition(
        source_epoch, target_epoch, field_name="sheet-slot-content", geometry=geometry
    )
    face_transfer = transfers.face
    if face_transfer is None:
        raise RuntimeError("A remeshing transaction must prepare a face transfer.")
    face_transition = face_transfer.epoch_transition(
        source_epoch, target_epoch, field_name="face-content", geometry=geometry
    )
    return SurfaceEventPassResult(
        candidate.topology,
        committed_state,
        evidence,
        transfers.sheet,
        face_transfer,
        transition,
        face_transition,
    )


@final
class SurfaceBurstResult(StrictModule):
    """Atomic whole-sheet burst candidate and its explicitly dropped content.

    ``dropped_sheet_content`` has one value per named extensive sheet field. It
    is deliberately outside the surviving-sheet transfer: an application must
    assign it to a physical ledger (the foam owner uses unresolved rim content)
    before exposing the committed result.
    """

    topology: MultiRegionSurfaceTopology
    state: MultiRegionSurfaceState
    dropped_sheet_content: Array
    evidence: SurfaceEventPassEvidence
    transition: TopologyEpochTransition | None

    @property
    def committed(self) -> bool:
        return self.evidence.committed


def _burst_record(
    status: SurfaceEventStatus,
    support_vertex_ids: Sequence[int],
    face_ids: Sequence[int],
    region_parents: Sequence[tuple[str, tuple[str, ...]]],
    priority: float,
    /,
    *,
    removed_vertex_ids: Sequence[int] = (),
) -> SurfaceEventRecord:
    return SurfaceEventRecord(
        SurfaceEventKind.BURST,
        status,
        support_vertex_ids,
        priority=priority,
        removed_vertex_ids=removed_vertex_ids,
        removed_face_ids=face_ids,
        region_parents=region_parents,
        ccd_time_of_impact=1.0 if status is SurfaceEventStatus.ACCEPTED else np.nan,
        ccd_certified=status is SurfaceEventStatus.ACCEPTED,
    )


def _burst_failure(
    topology: MultiRegionSurfaceTopology,
    state: MultiRegionSurfaceState,
    policy: SurfaceEventPolicy,
    record: SurfaceEventRecord,
    /,
) -> SurfaceBurstResult:
    source_points = np.asarray(state.positions[: topology.vertex_count], dtype=np.float64)
    source_epoch = _epoch(topology, source_points)
    evidence = _evidence(
        SurfaceEventPassStatus.NO_ACCEPTED_EVENTS,
        (record,),
        policy,
        source_epoch,
    )
    dropped = jnp.zeros((len(state.sheet_field_names),), dtype=state.sheet_fields.dtype)
    return SurfaceBurstResult(topology, state, dropped, evidence, None)


def _burst_survivor(
    topology: MultiRegionSurfaceTopology, first: int, second: int, /
) -> tuple[int, int]:
    first_finite = bool(np.asarray(topology.region_finite)[first])
    second_finite = bool(np.asarray(topology.region_finite)[second])
    if first_finite != second_finite:
        return (second, first) if first_finite else (first, second)
    first_id, second_id = topology.region_ids[first], topology.region_ids[second]
    return (first, second) if first_id < second_id else (second, first)


def _burst_topology(
    topology: MultiRegionSurfaceTopology,
    state: MultiRegionSurfaceState,
    first: int,
    second: int,
    face_ids: tuple[int, ...],
    /,
) -> (
    tuple[
        MultiRegionSurfaceTopology,
        np.ndarray,
        np.ndarray,
        tuple[int, ...],
        tuple[int, ...],
    ]
    | None
):
    faces = topology.host_faces()
    labels = topology.host_face_labels()
    global_faces = np.asarray(
        topology.face_global_ids[: topology.face_count], dtype=np.int64
    )
    pair = np.asarray(sorted((first, second)), dtype=np.int64)
    pair_mask = np.all(np.sort(labels, axis=1) == pair[None, :], axis=1)
    expected = tuple(sorted(int(value) for value in global_faces[pair_mask]))
    if expected != face_ids:
        return None
    survivor, removed = _burst_survivor(topology, first, second)
    keep_face = ~pair_mask
    relabeled = labels.copy()
    relabeled[relabeled == removed] = survivor
    keep_face &= relabeled[:, 0] != relabeled[:, 1]
    if np.count_nonzero(keep_face) < 1 or topology.region_count - 1 < 2:
        return None
    kept_faces = faces[keep_face]
    kept_labels = relabeled[keep_face]
    used_vertices = np.unique(kept_faces)
    compact = np.full((topology.vertex_count,), -1, dtype=np.int64)
    compact[used_vertices] = np.arange(used_vertices.size)
    remaining_regions = tuple(
        index for index in range(topology.region_count) if index != removed
    )
    region_rank = np.full((topology.region_count,), -1, dtype=np.int64)
    region_rank[np.asarray(remaining_regions)] = np.arange(len(remaining_regions))
    candidate = MultiRegionSurfaceTopology(
        topology.plan,
        compact[kept_faces],
        region_rank[kept_labels],
        tuple(topology.region_ids[index] for index in remaining_regions),
        tuple(topology.region_kinds[index] for index in remaining_regions),
        vertex_count=used_vertices.size,
        vertex_global_ids=np.asarray(
            topology.vertex_global_ids[: topology.vertex_count], dtype=np.int64
        )[used_vertices],
        face_global_ids=global_faces[keep_face],
        epoch=topology.epoch + 1,
        domain=topology.domain,
    )
    removed_vertices = tuple(
        int(value)
        for value in np.asarray(
            topology.vertex_global_ids[: topology.vertex_count], dtype=np.int64
        )[np.setdiff1d(np.arange(topology.vertex_count), used_vertices)]
    )
    removed_faces = tuple(sorted(int(value) for value in global_faces[~keep_face]))
    points = np.asarray(state.positions, dtype=np.float64)
    padded = np.zeros_like(points)
    padded[: used_vertices.size] = points[used_vertices]
    return candidate, padded, used_vertices, removed_vertices, removed_faces


def _burst_sheet_transfer(
    source: MultiRegionSurfaceTopology,
    target: MultiRegionSurfaceTopology,
    survivor_id: str,
    removed_id: str,
    /,
) -> tuple[ConservativeFieldTransfer, np.ndarray]:
    width = source.slot_width
    source_active = np.asarray(source.slot_active).reshape(-1)
    kept_active = np.zeros_like(source_active)
    dropped = np.zeros_like(source_active)
    target_lookup: dict[tuple[int, tuple[str, str]], int] = {}
    target_vertex_ids = np.asarray(
        target.vertex_global_ids[: target.vertex_count], dtype=np.int64
    )
    target_pairs = np.asarray(
        target.region_pairs[: target.region_pair_count], dtype=np.int64
    )
    target_slots = np.asarray(
        target.vertex_pair_slots[: target.vertex_count], dtype=np.int64
    )
    for vertex in range(target.vertex_count):
        for slot in range(target.slot_width):
            pair_index = int(target_slots[vertex, slot])
            if pair_index < 0:
                continue
            labels = target_pairs[pair_index]
            ids = tuple(
                sorted((target.region_ids[labels[0]], target.region_ids[labels[1]]))
            )
            target_lookup[(int(target_vertex_ids[vertex]), ids)] = (
                vertex * target.slot_width + slot
            )
    source_vertex_ids = np.asarray(
        source.vertex_global_ids[: source.vertex_count], dtype=np.int64
    )
    source_pairs = np.asarray(
        source.region_pairs[: source.region_pair_count], dtype=np.int64
    )
    source_slots = np.asarray(
        source.vertex_pair_slots[: source.vertex_count], dtype=np.int64
    )
    sources: list[int] = []
    targets: list[int] = []
    for vertex in range(source.vertex_count):
        for slot in range(source.slot_width):
            flat = vertex * width + slot
            pair_index = int(source_slots[vertex, slot])
            if pair_index < 0:
                continue
            labels = source_pairs[pair_index]
            mapped = tuple(
                sorted(
                    survivor_id
                    if source.region_ids[label] == removed_id
                    else source.region_ids[label]
                    for label in labels
                )
            )
            target_flat = target_lookup.get((int(source_vertex_ids[vertex]), mapped))
            if mapped[0] == mapped[1] or target_flat is None:
                dropped[flat] = True
                continue
            kept_active[flat] = True
            sources.append(flat)
            targets.append(target_flat)
    if not sources:
        raise ValueError("A burst candidate retained no sheet-slot support.")
    transfer = ConservativeFieldTransfer(
        np.asarray(sources, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.ones((len(sources),), dtype=np.float64),
        source_active=kept_active,
        target_active=np.asarray(target.slot_active).reshape(-1),
    )
    return transfer, dropped


def _burst_region_transfer(
    source: MultiRegionSurfaceTopology,
    target: MultiRegionSurfaceTopology,
    survivor_id: str,
    removed_id: str,
    /,
) -> ConservativeFieldTransfer:
    target_index = {region_id: index for index, region_id in enumerate(target.region_ids)}
    sources = np.arange(source.region_count, dtype=np.int64)
    targets = np.asarray(
        [
            target_index[survivor_id if region_id == removed_id else region_id]
            for region_id in source.region_ids
        ],
        dtype=np.int64,
    )
    return ConservativeFieldTransfer(
        sources,
        targets,
        np.ones((source.region_count,), dtype=np.float64),
        source_active=np.asarray(source.region_active),
        target_active=np.asarray(target.region_active),
    )


def _burst_velocity_reconstruction(
    source: MultiRegionSurfaceTopology,
    target: MultiRegionSurfaceTopology,
    /,
) -> BoundedFieldReconstruction:
    lookup = {
        int(value): index
        for index, value in enumerate(
            np.asarray(source.vertex_global_ids[: source.vertex_count], dtype=np.int64)
        )
    }
    target_ids = np.asarray(
        target.vertex_global_ids[: target.vertex_count], dtype=np.int64
    )
    sources = np.asarray([lookup[int(value)] for value in target_ids], dtype=np.int64)
    return BoundedFieldReconstruction(
        sources,
        np.arange(target.vertex_count, dtype=np.int64),
        np.ones((target.vertex_count,), dtype=np.float64),
        source_active=np.asarray(source.vertex_active),
        target_active=np.asarray(target.vertex_active),
    )


def _commit_surface_burst(
    topology: MultiRegionSurfaceTopology,
    state: MultiRegionSurfaceState,
    candidate_topology: MultiRegionSurfaceTopology,
    padded: np.ndarray,
    survivor_id: str,
    removed_id: str,
    validation: MultiRegionSurfaceEvidence,
    policy: SurfaceEventPolicy,
    record: SurfaceEventRecord,
    lineage: MultiRegionSurfaceLineage,
    source_epoch: TopologyEpoch,
    target_epoch: TopologyEpoch,
    /,
) -> SurfaceBurstResult:
    sheet, dropped_mask = _burst_sheet_transfer(
        topology,
        candidate_topology,
        survivor_id,
        removed_id,
    )
    region = _burst_region_transfer(
        topology,
        candidate_topology,
        survivor_id,
        removed_id,
    )
    velocity = _burst_velocity_reconstruction(topology, candidate_topology)
    width = topology.slot_width
    source_sheet = state.sheet_fields.reshape(
        (topology.vertex_capacity * width, len(state.sheet_field_names))
    )
    target_sheet = sheet.apply(source_sheet)
    target_region = region.apply(state.region_fields)
    target_velocity = velocity.apply(state.velocities)
    candidate_state = MultiRegionSurfaceState(
        candidate_topology,
        padded,
        velocities=target_velocity,
        sheet_fields=target_sheet.reshape(
            (
                candidate_topology.vertex_capacity,
                candidate_topology.slot_width,
                len(state.sheet_field_names),
            )
        ),
        region_fields=target_region,
        sheet_field_names=state.sheet_field_names,
        region_field_names=state.region_field_names,
    )
    transfers = _Transfers(
        state=candidate_state,
        sheet=sheet,
        face=None,
        sheet_evidence=(
            sheet.evidence(source_sheet, target_sheet)
            if state.sheet_field_names
            else None
        ),
        region_evidence=(
            region.evidence(state.region_fields, target_region)
            if state.region_field_names
            else None
        ),
        velocity_evidence=velocity.evidence(state.velocities, target_velocity),
    )
    candidate = _Candidate(candidate_topology, validation, transfers)
    transaction = TransactionalCandidate(
        _payload(state),
        _payload(candidate_state),
        validation,
        _accepted_flag(transfers),
        topology.lineage_id,
    )
    commit = commit_candidate(transaction)
    committed = bool(commit.committed)
    status = (
        SurfaceEventPassStatus.COMMITTED
        if committed
        else SurfaceEventPassStatus.TRANSFER_FAILED
    )
    evidence = _evidence(
        status,
        (record,),
        policy,
        source_epoch,
        lineage=lineage,
        candidate=candidate,
        target_epoch=target_epoch,
        transaction_id=transaction.candidate_id,
    )
    if not committed:
        dropped = jnp.zeros(
            (len(state.sheet_field_names),), dtype=state.sheet_fields.dtype
        )
        return SurfaceBurstResult(topology, state, dropped, evidence, None)
    committed_state = MultiRegionSurfaceState(
        candidate_topology,
        commit.state.positions,
        velocities=commit.state.velocities,
        sheet_fields=commit.state.sheet_fields,
        region_fields=commit.state.region_fields,
        sheet_field_names=state.sheet_field_names,
        region_field_names=state.region_field_names,
    )
    dropped = jnp.sum(
        jnp.where(dropped_mask[:, None], source_sheet, 0.0),
        axis=0,
    )
    transition = sheet.epoch_transition(
        source_epoch,
        target_epoch,
        field_name="surviving-sheet-slot-content",
        geometry=TransferGeometryBinding(
            _geometry_id(
                topology,
                np.asarray(state.positions[: topology.vertex_count], dtype=np.float64),
            ),
            _geometry_id(
                candidate_topology,
                np.asarray(
                    committed_state.positions[: candidate_topology.vertex_count],
                    dtype=np.float64,
                ),
            ),
            "topology-correspondence",
            source_topology_id=topology.topology_id,
            target_topology_id=candidate_topology.topology_id,
            coverage_defect=None,
        ),
    )
    return SurfaceBurstResult(
        candidate_topology,
        committed_state,
        dropped,
        evidence,
        transition,
    )


def apply_surface_burst(
    topology: MultiRegionSurfaceTopology,
    state: MultiRegionSurfaceState,
    region_ids: tuple[str, str],
    face_ids: Sequence[int],
    /,
    *,
    policy: SurfaceEventPolicy | None = None,
    priority: float = 0.0,
) -> SurfaceBurstResult:
    """Delete one complete separating sheet and merge its region labels.

    The supplied stable face IDs must equal *all* active faces of the declared
    region pair. Surviving sheet and region content transfer conservatively;
    content on vanished sheet slots is returned separately and never hidden in
    another field. No point moves, so the CCD certificate is the exact unit
    interval, while the complete candidate still undergoes the policy's full
    self-intersection validation.
    """
    if not isinstance(topology, MultiRegionSurfaceTopology):
        raise TypeError("topology must be MultiRegionSurfaceTopology.")
    if not isinstance(state, MultiRegionSurfaceState):
        raise TypeError("state must be MultiRegionSurfaceState.")
    state.require_topology(topology)
    policy_ = SurfaceEventPolicy() if policy is None else policy
    if not isinstance(policy_, SurfaceEventPolicy):
        raise TypeError("policy must be SurfaceEventPolicy or None.")
    if (
        not isinstance(region_ids, tuple)
        or len(region_ids) != 2
        or not all(isinstance(value, str) and value for value in region_ids)
        or region_ids[0] == region_ids[1]
    ):
        raise ValueError("region_ids must be two distinct stable region identifiers.")
    if any(region_id not in topology.region_ids for region_id in region_ids):
        record = _burst_record(SurfaceEventStatus.SUPPORT_INVALID, (), (), (), priority)
        return _burst_failure(topology, state, policy_, record)
    first = topology.region_index(region_ids[0])
    second = topology.region_index(region_ids[1])
    canonical_faces = tuple(sorted(int(value) for value in face_ids))
    labels = topology.host_face_labels()
    pair = np.asarray(sorted((first, second)), dtype=np.int64)
    pair_rows = np.flatnonzero(np.all(np.sort(labels, axis=1) == pair[None, :], axis=1))
    support_slots = (
        np.unique(topology.host_faces()[pair_rows])
        if pair_rows.size
        else np.zeros((0,), dtype=np.int64)
    )
    support_ids = tuple(
        int(value)
        for value in np.asarray(
            topology.vertex_global_ids[: topology.vertex_count], dtype=np.int64
        )[support_slots]
    )
    built = _burst_topology(topology, state, first, second, canonical_faces)
    if built is None:
        record = _burst_record(
            SurfaceEventStatus.SUPPORT_INVALID,
            support_ids,
            canonical_faces,
            (),
            priority,
        )
        return _burst_failure(topology, state, policy_, record)
    candidate_topology, padded, used_vertices, removed_vertices, removed_faces = built
    survivor, removed = _burst_survivor(topology, first, second)
    survivor_id = topology.region_ids[survivor]
    removed_id = topology.region_ids[removed]
    parentage = ((survivor_id, tuple(sorted((survivor_id, removed_id)))),)
    record = _burst_record(
        SurfaceEventStatus.ACCEPTED,
        support_ids,
        removed_faces,
        parentage,
        priority,
        removed_vertex_ids=removed_vertices,
    )
    empty_candidate_state = MultiRegionSurfaceState(candidate_topology, padded)
    validation = validate_multiregion_surface(
        candidate_topology,
        empty_candidate_state,
        policy=policy_.validation,
    )
    source_points = np.asarray(state.positions[: topology.vertex_count], dtype=np.float64)
    source_epoch = _epoch(topology, source_points)
    target_epoch = _epoch(
        candidate_topology,
        padded[: candidate_topology.vertex_count],
    )
    lineage = MultiRegionSurfaceLineage(
        source_lineage_id=topology.lineage_id,
        target_lineage_id=candidate_topology.lineage_id,
        source_epoch=topology.epoch,
        target_epoch=candidate_topology.epoch,
        vertex_parents=(),
        face_parents=(),
        region_parents=parentage,
        removed_vertex_ids=removed_vertices,
        removed_face_ids=removed_faces,
        removed_region_ids=(removed_id,),
    )
    if not validation.accepted:
        candidate = _Candidate(candidate_topology, validation, None)
        evidence = _evidence(
            SurfaceEventPassStatus.CANDIDATE_VALIDATION_FAILED,
            (record,),
            policy_,
            source_epoch,
            lineage=lineage,
            candidate=candidate,
            target_epoch=target_epoch,
        )
        dropped = jnp.zeros(
            (len(state.sheet_field_names),), dtype=state.sheet_fields.dtype
        )
        return SurfaceBurstResult(topology, state, dropped, evidence, None)
    return _commit_surface_burst(
        topology,
        state,
        candidate_topology,
        padded,
        survivor_id,
        removed_id,
        validation,
        policy_,
        record,
        lineage,
        source_epoch,
        target_epoch,
    )


__all__ = [
    "SurfaceBurstResult",
    "SurfaceEventProposal",
    "apply_surface_burst",
    "apply_surface_events",
    "multiregion_topology_epoch",
]
