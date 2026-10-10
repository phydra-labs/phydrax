# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""One staged dependency transaction per meshfree topology epoch.

A meshfree epoch change replaces every artifact prepared on the source
support (geometry, measures, points, routes, stencils, metric, hierarchy,
coupling queries, compiled derivative plans) and transports every state
payload bound to it (current fields, every live history, predictor and
controller state) through its own frozen conservative route. The dependency
graph of a ``phydrax.lifecycle`` composition decides what is bound to the
epoch: everything reachable from the epoch entry must be reprepared by its
owner (derived artifacts) or remapped by an explicit route (state). Nothing is
published unless every route reports success and conserved content, so a
failure in any history or dependency returns the source composition object
unchanged.

The change has exactly one cause. ``sample-repair`` follows an accepted
:class:`SurfaceResamplingResult` of an unchanged surface; ``surface-event``
follows a committed multiregion event pass, whose split/merge/pinch/region
lineage and CCD/volume/validation evidence are consumed, never manufactured;
``adaptive-refinement`` follows an admitted :class:`MeshfreeAdaptationProposal`
(a bounded insertion/removal/degree/support change of a bulk or surface cloud).
Values remain differentiable through the frozen transfers; selection of the
epoch has no derivative.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...geometry.multiregion_surface import (
    multiregion_topology_epoch,
    MultiRegionSurfaceState,
    MultiRegionSurfaceTopology,
    PreparedMultiRegionSurface,
    SurfaceEventKind,
    SurfaceEventPassResult,
)
from ...lifecycle import (
    commit_composition_rebind,
    Composition,
    CompositionDependency,
    CompositionEntry,
    CompositionRebind,
    CompositionRebindReceipt,
    CompositionTransport,
)
from ...lifecycle._composition_rebind import _carries_state
from ...sparse import EdgeRelation
from ...typing import Bool, Dim, Float64, Int32, Int64
from .._topology_epoch import (
    TopologyEpoch,
    TopologyEpochTransition,
    TopologyEpochTransitionResult,
)
from ._adaptivity import MeshfreeAdaptationProposal
from ._resampling import SurfaceResamplingResult
from .._transfer import TransferGeometryBinding
from ._transfer import PointTransferPlan, PointTransferRequest, PreparedPointTransfer


MeshfreeEpochCause: TypeAlias = Literal[
    "sample-repair", "surface-event", "adaptive-refinement"
]

# Kinds that change the physical surface topology (not only its sampling).
_TOPOLOGY_EVENTS = frozenset(
    {
        SurfaceEventKind.T1_POP,
        SurfaceEventKind.PINCH,
        SurfaceEventKind.MERGE,
        SurfaceEventKind.REGION_SPLIT,
        SurfaceEventKind.BURST,
    }
)


class EpochEventDim(Dim):
    """Accepted records of one committed event pass."""


class EpochHistoryDim(Dim):
    """Live histories remapped together."""


class SheetSupportDim(Dim):
    """Active sheet slots of one multiregion epoch."""


class SheetAmbientDim(Dim):
    """Ambient coordinates of a sheet support."""


@final
class MeshfreeHistoryRemap(StrictModule):
    """All live histories transferred by their own frozen routes, or none.

    ``values`` follow the input order; ``successful`` holds only when every
    route succeeded with its content ledger, and ``failed`` lists the others.
    Value derivatives cross the frozen routes: ``values`` are differentiable
    in every input history (each JVP is its own route's primal action), and
    ``pullback``/``adjoint`` publish every history's coordinate dual and
    Hilbert adjoint for adjoint sweeps outside one JAX trace. A failed remap
    publishes NaN values (with NaN derivatives) and refuses both reverse
    maps; route selection itself has no derivative.
    """

    __strict_contract__ = True
    values: tuple[Array, ...]
    conservation_residuals: Float64[EpochHistoryDim]
    content_tolerances: Float64[EpochHistoryDim]
    routes: tuple[TopologyEpochTransition, ...]
    successful: bool = eqx.field(static=True)
    failed: tuple[int, ...] = eqx.field(static=True)

    @property
    def value_derivative_available(self) -> bool:
        return self.successful

    def _reverse(
        self, cotangents: Sequence[ArrayLike], adjoint: bool, /
    ) -> tuple[Array, ...]:
        if not self.successful:
            raise ValueError(
                f"Live-history remap failed for histories {self.failed}; no value "
                "derivative crosses a refused epoch."
            )
        items = tuple(cotangents)
        if len(items) != len(self.routes):
            raise ValueError("Every live history requires exactly one cotangent.")
        return tuple(
            route.adjoint(item) if adjoint else route.pullback(item)
            for route, item in zip(self.routes, items, strict=True)
        )

    def pullback(self, cotangents: Sequence[ArrayLike], /) -> tuple[Array, ...]:
        """Every history's coordinate dual ``P_i^T w_i`` (the VJP of its remap)."""
        return self._reverse(cotangents, False)

    def adjoint(self, values: Sequence[ArrayLike], /) -> tuple[Array, ...]:
        """Every history's Hilbert adjoint ``P_i^*`` in its own epoch pairings."""
        return self._reverse(values, True)


def _same_change(transitions: Sequence[TopologyEpochTransition]) -> None:
    if not transitions or any(
        not isinstance(item, TopologyEpochTransition) for item in transitions
    ):
        raise TypeError("Live-history remap needs TopologyEpochTransition routes.")
    first = transitions[0]
    if any(
        item.source.epoch_id != first.source.epoch_id
        or item.target.epoch_id != first.target.epoch_id
        for item in transitions
    ):
        raise ValueError("Every live-history route must cross the same epoch change.")


def remap_live_histories(
    transitions: Sequence[TopologyEpochTransition],
    concentrations: Sequence[ArrayLike],
    /,
) -> MeshfreeHistoryRemap:
    """Transfer every live history through its own route at one host boundary.

    Each history keeps its own source/target measures; one history's route is
    never applied to another's values. Values are differentiable through the
    frozen routes under eager JAX differentiation (the acceptance decision is
    the host boundary and carries no derivative).
    """
    routes = tuple(transitions)
    _same_change(routes)
    fields = tuple(concentrations)
    if len(fields) != len(routes):
        raise ValueError("Every live history requires exactly one route.")
    results = tuple(
        route.apply(field) for route, field in zip(routes, fields, strict=True)
    )
    # Host boundary: the epoch decision is one explicit synchronization.
    accepted = np.asarray(jnp.stack([item.successful for item in results]))
    successful = bool(np.all(accepted))
    # All or none: a failed remap is NaN, multiplicatively so that its
    # tangents and cotangents are NaN rather than a plausible derivative.
    refused = 1.0 if successful else np.nan
    return MeshfreeHistoryRemap(
        tuple(item.values * refused for item in results),
        jnp.stack([item.conservation_residual for item in results]),
        jnp.stack([item.content_tolerance for item in results]),
        routes,
        successful,
        tuple(int(index) for index in np.flatnonzero(~accepted)),
    )


@final
class MeshfreeEventLineage(StrictModule):
    """Committed physical surface-event lineage consumed by a meshfree epoch.

    Built only from a committed multiregion event pass; per accepted record it
    retains the event kind, CCD certification and time of impact, and the
    finite-region volume residual, together with region lineage and the pass
    validation/transaction identities.
    """

    __strict_contract__ = True
    ccd_time_of_impact: Float64[EpochEventDim]
    volume_residual: Float64[EpochEventDim]
    ccd_certified: Bool[EpochEventDim]
    source_epoch: TopologyEpoch
    target_epoch: TopologyEpoch
    event_kinds: tuple[str, ...] = eqx.field(static=True)
    record_ids: tuple[str, ...] = eqx.field(static=True)
    region_parents: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True)
    removed_region_ids: tuple[str, ...] = eqx.field(static=True)
    multiregion_lineage_id: str = eqx.field(static=True)
    validation_id: str = eqx.field(static=True)
    transaction_id: str = eqx.field(static=True)
    topology_changed: bool = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)

    def __init__(self, result: SurfaceEventPassResult, /) -> None:
        if not isinstance(result, SurfaceEventPassResult):
            raise TypeError("Physical lineage needs a SurfaceEventPassResult.")
        evidence = result.evidence
        lineage = evidence.lineage
        if (
            not result.committed
            or lineage is None
            or evidence.target_epoch is None
            or evidence.validation is None
            or evidence.transaction_id is None
        ):
            raise ValueError(
                "A meshfree surface event requires a committed multiregion pass; "
                "sample insertion or removal never manufactures a physical event."
            )
        records = tuple(record for record in evidence.records if record.accepted)
        self.ccd_time_of_impact = jnp.asarray(
            [record.ccd_time_of_impact for record in records], dtype=jnp.float64
        )
        self.volume_residual = jnp.asarray(
            [record.volume_residual for record in records], dtype=jnp.float64
        )
        self.ccd_certified = jnp.asarray(
            [record.ccd_certified for record in records], dtype=jnp.bool_
        )
        self.source_epoch, self.target_epoch = (
            evidence.source_epoch,
            evidence.target_epoch,
        )
        self.event_kinds = tuple(record.kind.name for record in records)
        self.record_ids = tuple(record.record_id for record in records)
        self.region_parents = tuple(lineage.region_parents)
        self.removed_region_ids = tuple(lineage.removed_region_ids)
        self.multiregion_lineage_id = lineage.lineage_id
        self.validation_id = evidence.validation.evidence_id
        self.transaction_id = evidence.transaction_id
        self.topology_changed = any(record.kind in _TOPOLOGY_EVENTS for record in records)
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "meshfree-event-lineage",
                "pass": evidence.evidence_id,
                "lineage": lineage.lineage_id,
                "records": list(self.record_ids),
            }
        )


@final
class MeshfreeEpochChange(StrictModule):
    """One meshfree epoch change with exactly one declared cause."""

    source: TopologyEpoch
    target: TopologyEpoch
    lineage: MeshfreeEventLineage | None
    cause: MeshfreeEpochCause = eqx.field(static=True)
    proposal_id: str | None = eqx.field(static=True)
    change_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: TopologyEpoch,
        target: TopologyEpoch,
        /,
        *,
        cause: MeshfreeEpochCause,
        lineage: MeshfreeEventLineage | None = None,
        proposal: SurfaceResamplingResult | MeshfreeAdaptationProposal | None = None,
    ) -> None:
        if not isinstance(source, TopologyEpoch) or not isinstance(target, TopologyEpoch):
            raise TypeError("Epoch changes connect TopologyEpoch values.")
        if target.index != source.index + 1 or source.epoch_id == target.epoch_id:
            raise ValueError("Epoch changes connect consecutive distinct epochs.")
        match cause:
            case "sample-repair":
                if lineage is not None or not isinstance(
                    proposal, SurfaceResamplingResult
                ):
                    raise ValueError(
                        "A sample repair follows one resampling proposal and carries "
                        "no physical lineage."
                    )
                if not proposal.converged or proposal.capacity_refused:
                    raise ValueError(
                        "Only a converged, capacity-admitted resampling proposal can "
                        "become an epoch."
                    )
                identity = proposal.proposal_id
            case "surface-event":
                if proposal is not None or not isinstance(lineage, MeshfreeEventLineage):
                    raise ValueError(
                        "A surface event follows committed multiregion lineage only."
                    )
                if (
                    lineage.source_epoch.epoch_id != source.epoch_id
                    or lineage.target_epoch.epoch_id != target.epoch_id
                ):
                    raise ValueError("Event lineage belongs to different epochs.")
                identity = lineage.lineage_id
            case "adaptive-refinement":
                if lineage is not None or not isinstance(
                    proposal, MeshfreeAdaptationProposal
                ):
                    raise ValueError(
                        "An adaptive refinement follows one adaptation proposal and "
                        "carries no physical lineage."
                    )
                if not proposal.admitted:
                    raise ValueError(
                        "Only an admitted adaptation proposal can become an epoch."
                    )
                identity = proposal.proposal_id
            case unknown:
                assert_never(unknown)
        self.source, self.target, self.lineage = source, target, lineage
        self.cause = cause
        self.proposal_id = None if proposal is None else proposal.proposal_id
        self.change_id = canonical_fingerprint(
            {
                "kind": "meshfree-epoch-change",
                "source": source.epoch_id,
                "target": target.epoch_id,
                "cause": cause,
                "identity": identity,
            }
        )


def _dependency_closure(source: Composition, root: str, /) -> tuple[str, ...]:
    """Every entry prepared, directly or transitively, against ``root``."""
    seen: set[str] = set()
    frontier = [root]
    while frontier:
        for dependent in source.dependents(frontier.pop()):
            if dependent not in seen:
                seen.add(dependent)
                frontier.append(dependent)
    return tuple(sorted(seen))


def _rebound(
    dependencies: Sequence[CompositionDependency],
    replaced: Mapping[str, CompositionEntry],
    /,
) -> tuple[CompositionDependency, ...]:
    return tuple(
        replaced[item.entry_id].binding(item.facet) if item.entry_id in replaced else item
        for item in dependencies
    )


@final
class MeshfreeEpochCandidate(StrictModule):
    """Validated, unpublished epoch transaction with every route's ledger."""

    change: MeshfreeEpochChange
    rebind: CompositionRebind
    remap_results: tuple[TopologyEpochTransitionResult, ...]
    remapped: tuple[str, ...] = eqx.field(static=True)
    reprepared: tuple[str, ...] = eqx.field(static=True)


@final
class MeshfreeEpochReceipt(StrictModule):
    """Published (or refused) epoch with its complete account.

    On refusal ``composition`` is the unchanged source object. Value
    derivatives through the frozen routes are available exactly when the epoch
    was published; selection of the epoch never has a derivative.
    """

    change: MeshfreeEpochChange
    receipt: CompositionRebindReceipt
    conservation_residuals: Array
    content_tolerances: Array
    published: bool = eqx.field(static=True)
    remapped: tuple[str, ...] = eqx.field(static=True)
    failed: tuple[str, ...] = eqx.field(static=True)
    value_derivative_available: bool = eqx.field(static=True)

    @property
    def composition(self) -> Composition:
        return self.receipt.composition

    def require_differentiable_selection(self) -> None:
        raise ValueError(
            "Meshfree epoch selection is nondifferentiable; differentiate values "
            "through its frozen transfers or within one fixed epoch."
        )


def _check_epoch_entry(
    source: Composition, change: MeshfreeEpochChange, epoch_entry: str, /
) -> CompositionEntry:
    entry = source.entry(epoch_entry)
    value = entry.value
    if (
        entry.role != "topology"
        or not isinstance(value, TopologyEpoch)
        or value.epoch_id != change.source.epoch_id
        or entry.structure_id != change.source.epoch_id
    ):
        raise ValueError(
            f"Entry {epoch_entry!r} is not the source topology epoch of this change."
        )
    return entry


def _dispositions(
    source: Composition,
    closure: tuple[str, ...],
    remap: Mapping[str, TopologyEpochTransition],
    staged: Mapping[str, CompositionEntry],
    change: MeshfreeEpochChange,
    /,
) -> None:
    """Every epoch-bound state has its own route and every artifact a rebuild."""
    unrouted, unbuilt = [], []
    for entry_id in closure:
        state = _carries_state(source.entry(entry_id).role)
        if state and entry_id not in remap:
            unrouted.append(entry_id)
        if not state and entry_id not in staged:
            unbuilt.append(entry_id)
    if unrouted or unbuilt:
        raise ValueError(
            "Epoch-bound entries lack a disposition; state needs its own route "
            f"{unrouted} and derived artifacts a target-epoch rebuild {unbuilt}."
        )
    for entry_id, route in remap.items():
        if entry_id not in closure:
            raise ValueError(f"Remapped entry {entry_id!r} is not bound to the epoch.")
        if not isinstance(route, TopologyEpochTransition):
            raise TypeError("Remap routes must be TopologyEpochTransition values.")
        if (
            route.source.epoch_id != change.source.epoch_id
            or route.target.epoch_id != change.target.epoch_id
        ):
            raise ValueError(f"Route of {entry_id!r} crosses another epoch change.")


def stage_meshfree_epoch(
    source: Composition,
    change: MeshfreeEpochChange,
    /,
    *,
    epoch_entry: str,
    remap: Mapping[str, TopologyEpochTransition],
    reprepare: Sequence[CompositionEntry],
) -> MeshfreeEpochCandidate:
    """Stage one complete epoch change without publishing anything.

    ``remap`` assigns every epoch-bound state entry (current fields, every
    live history, predictors) its own frozen route; ``reprepare`` holds the
    owners' target-epoch rebuilds of every epoch-bound derived artifact.
    Entries independent of the epoch are retained. Missing routes or rebuilds
    are refused before staging.
    """
    if not isinstance(source, Composition) or not isinstance(change, MeshfreeEpochChange):
        raise TypeError("Staging needs a Composition and a MeshfreeEpochChange.")
    epoch = _check_epoch_entry(source, change, epoch_entry)
    staged = {item.entry_id: item for item in reprepare}
    if len(staged) != len(tuple(reprepare)) or epoch_entry in staged:
        raise ValueError("Rebuilds must be unique and exclude the epoch entry.")
    if any(item in staged for item in remap):
        raise ValueError("An entry is either remapped or reprepared, not both.")
    closure = _dependency_closure(source, epoch_entry)
    _dispositions(source, closure, remap, staged, change)
    target_epoch = CompositionEntry(
        change.target,
        entry_id=epoch_entry,
        role="topology",
        owner_id=epoch.owner_id,
        structure_id=change.target.epoch_id,
        revision_id=change.change_id,
        semantics_id=epoch.semantics_id,
        dependencies=epoch.dependencies,
    )
    replaced: dict[str, CompositionEntry] = {epoch_entry: target_epoch, **staged}
    remapped = tuple(sorted(remap))
    results = tuple(
        remap[entry_id].apply(source.value(entry_id)) for entry_id in remapped
    )
    for entry_id, result in zip(remapped, results, strict=True):
        entry = source.entry(entry_id)
        replaced[entry_id] = CompositionEntry(
            result.values,
            entry_id=entry_id,
            role=entry.role,
            owner_id=entry.owner_id,
            structure_id=change.target.epoch_id,
            revision_id=canonical_fingerprint(
                {
                    "kind": "meshfree-epoch-remap",
                    "source": entry.revision_id,
                    "route": remap[entry_id].transition_id,
                }
            ),
            semantics_id=entry.semantics_id,
        )
    # Dependencies are rebound once every replacement identity is known.
    targets = {
        entry_id: CompositionEntry(
            item.value,
            entry_id=entry_id,
            role=item.role,
            owner_id=item.owner_id,
            structure_id=item.structure_id,
            revision_id=item.revision_id,
            semantics_id=item.semantics_id,
            dependencies=_rebound(source.entry(entry_id).dependencies, replaced),
        )
        for entry_id, item in replaced.items()
        if entry_id in remap
    }
    transports: list[CompositionTransport] = [
        remap[entry_id].composition_transport(source.entry(entry_id), targets[entry_id])
        for entry_id in remapped
    ]
    retained = tuple(
        entry_id
        for entry_id in source.entry_ids
        if entry_id != epoch_entry and entry_id not in closure and entry_id not in staged
    )
    rebind = CompositionRebind(
        source,
        retain=retained,
        reprepare=(target_epoch, *staged.values()),
        transports=transports,
    )
    return MeshfreeEpochCandidate(
        change, rebind, results, remapped, tuple(sorted(staged))
    )


def commit_meshfree_epoch(
    candidate: MeshfreeEpochCandidate, /, *, accepted_boundary: bool
) -> MeshfreeEpochReceipt:
    """Publish the staged epoch only if every route and the boundary are accepted."""
    if not isinstance(candidate, MeshfreeEpochCandidate):
        raise TypeError("candidate must be a MeshfreeEpochCandidate.")
    receipt = commit_composition_rebind(
        candidate.rebind, accepted_boundary=accepted_boundary
    )
    failed = tuple(
        entry_id
        for entry_id, accepted in zip(
            candidate.remapped, receipt.transport_accepted, strict=True
        )
        if not accepted
    )
    empty = jnp.zeros((0,), dtype=jnp.float64)
    return MeshfreeEpochReceipt(
        candidate.change,
        receipt,
        jnp.stack([item.conservation_residual for item in candidate.remap_results])
        if candidate.remap_results
        else empty,
        jnp.stack([item.content_tolerance for item in candidate.remap_results])
        if candidate.remap_results
        else empty,
        receipt.published,
        candidate.remapped,
        failed,
        receipt.published,
    )


@final
class MeshfreeSheetSupport(StrictModule):
    """Meshfree support on the active sheet slots of one multiregion epoch.

    Each support point is one ``(vertex, region-pair)`` slot at its vertex
    position with its barycentric slot area as measure; ``slots`` index the
    flattened ``vertex * slot_width + slot`` capacity layout of the sheet
    fields and ``labels`` name each slot's region pair.
    """

    __strict_contract__ = True
    points: Float64[SheetSupportDim, SheetAmbientDim]
    measures: Float64[SheetSupportDim]
    slots: Int32[SheetSupportDim]
    vertex_ids: Int64[SheetSupportDim]
    epoch: TopologyEpoch
    labels: tuple[tuple[str, str], ...] = eqx.field(static=True)
    slot_capacity: int = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self, topology: MultiRegionSurfaceTopology, state: MultiRegionSurfaceState, /
    ) -> None:
        if not isinstance(topology, MultiRegionSurfaceTopology) or not isinstance(
            state, MultiRegionSurfaceState
        ):
            raise TypeError("Sheet supports need a multiregion topology and state.")
        positions = np.asarray(state.positions, dtype=np.float64)
        areas = np.asarray(
            PreparedMultiRegionSurface(topology, state).slot_areas(state.positions)
        ).reshape(-1)
        active = np.asarray(topology.slot_active).reshape(-1)
        slots = np.flatnonzero(active)
        if np.any(areas[slots] <= 0) or not np.all(np.isfinite(areas[slots])):
            raise ValueError("Active sheet slots need positive finite areas.")
        width = topology.slot_width
        vertices = slots // width
        pairs = np.asarray(topology.vertex_pair_slots).reshape(-1)[slots]
        regions = np.asarray(topology.region_pairs)[pairs]
        self.points = jnp.asarray(positions[vertices])
        self.measures = jnp.asarray(areas[slots])
        self.slots = jnp.asarray(slots, dtype=jnp.int32)
        self.vertex_ids = jnp.asarray(
            np.asarray(topology.vertex_global_ids)[vertices], dtype=jnp.int64
        )
        self.epoch = multiregion_topology_epoch(topology, state.positions)
        self.labels = tuple(
            (topology.region_ids[int(left)], topology.region_ids[int(right)])
            for left, right in regions
        )
        self.slot_capacity = active.size
        self.support_id = canonical_fingerprint(
            {"kind": "meshfree-sheet-support", "epoch": self.epoch.epoch_id}
        )

    def concentration(self, sheet_content: ArrayLike, /) -> Array:
        """Compact concentration of capacity-shaped ``(vertex, slot)`` content."""
        content = jnp.asarray(sheet_content).reshape(-1)
        if content.shape != (self.slot_capacity,):
            raise ValueError("Sheet content must match the slot capacity layout.")
        return content[self.slots] / self.measures


@final
class MeshfreeSurfaceEventEpoch(StrictModule):
    """A committed multiregion event pass consumed as one meshfree epoch change.

    ``transfer`` is the audited concentration form ``diag(1/m_new) P diag(m_old)``
    of the authority's own extensive sheet transfer ``P``: conservative and
    nonnegative by construction. Constant reproduction is measured, not
    claimed; a physical event that changes area cannot have it.
    """

    change: MeshfreeEpochChange
    source: MeshfreeSheetSupport
    target: MeshfreeSheetSupport
    transfer: PreparedPointTransfer

    def transition(self) -> TopologyEpochTransition:
        return self.transfer.epoch_transition(self.change.source, self.change.target)


def surface_event_epoch(
    source_topology: MultiRegionSurfaceTopology,
    source_state: MultiRegionSurfaceState,
    result: SurfaceEventPassResult,
    /,
    *,
    tolerance: float = 1e-10,
) -> MeshfreeSurfaceEventEpoch:
    """Adapt one committed event pass of ``(source_topology, source_state)``."""
    lineage = MeshfreeEventLineage(result)
    sheet = result.sheet_transfer
    if sheet is None:
        raise ValueError("A committed event pass must expose its sheet transfer.")
    source = MeshfreeSheetSupport(source_topology, source_state)
    target = MeshfreeSheetSupport(result.topology, result.state)
    if (
        source.epoch.epoch_id != lineage.source_epoch.epoch_id
        or target.epoch.epoch_id != lineage.target_epoch.epoch_id
    ):
        raise ValueError("The event pass was not applied to this source geometry.")
    valid = np.asarray(sheet.relation.valid)
    routes = np.asarray(sheet.relation.source_indices)[valid]
    receivers = np.asarray(sheet.relation.target_indices)[valid]
    weights = np.asarray(sheet.weights)[valid]
    source_compact = np.full(source.slot_capacity, -1, dtype=np.int64)
    source_compact[np.asarray(source.slots)] = np.arange(source.slots.shape[0])
    target_compact = np.full(target.slot_capacity, -1, dtype=np.int64)
    target_compact[np.asarray(target.slots)] = np.arange(target.slots.shape[0])
    columns, rows = source_compact[routes], target_compact[receivers]
    if np.any(columns < 0) or np.any(rows < 0):
        raise ValueError("The sheet transfer routes inactive slots.")
    old, new = np.asarray(source.measures), np.asarray(target.measures)
    transfer = PointTransferPlan(
        EdgeRelation(columns, rows, source_size=old.size, target_size=new.size),
        weights * old[columns] / new[rows],
        old,
        new,
        source_id=source.support_id,
        target_id=target.support_id,
        request=PointTransferRequest("conservative-positive"),
        tolerance=tolerance,
        geometry=TransferGeometryBinding(
            source.epoch.geometry_id,
            target.epoch.geometry_id,
            "topology-correspondence",
            source_topology_id=source.epoch.topology_id,
            target_topology_id=target.epoch.topology_id,
            coverage_defect=None,
        ),
    ).prepare()
    change = MeshfreeEpochChange(
        lineage.source_epoch, lineage.target_epoch, cause="surface-event", lineage=lineage
    )
    return MeshfreeSurfaceEventEpoch(change, source, target, transfer)


__all__ = [
    "MeshfreeEpochCandidate",
    "MeshfreeEpochCause",
    "MeshfreeEpochChange",
    "MeshfreeEpochReceipt",
    "MeshfreeEventLineage",
    "MeshfreeHistoryRemap",
    "MeshfreeSheetSupport",
    "MeshfreeSurfaceEventEpoch",
    "commit_meshfree_epoch",
    "remap_live_histories",
    "stage_meshfree_epoch",
    "surface_event_epoch",
]
