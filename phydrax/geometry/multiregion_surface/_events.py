#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Contracts of transactional topology events on multiregion surfaces.

Topology mutations use one host transaction per pass. `apply_surface_events`
orders local proposals canonically, selects disjoint vertex 2-rings, drafts
local edits with stable lineage, runs exact combinatorial/geometric guards and
local continuous collision detection, constructs conservative sparse field
transfers, and commits only after complete candidate validation.
`apply_surface_burst` owns the typed whole-sheet deletion whose vanished
content must leave the surviving-sheet transfer explicitly; either route
returns the source topology and state unchanged on failure. Event kinds form one
closed set: quality remeshing (``SPLIT``, ``COLLAPSE``, ``FLIP``) and physical
transitions (``T1_POP``, ``PINCH``, ``MERGE``, ``REGION_SPLIT``, ``BURST``).
Derivatives are never available across an event pass.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum
from typing import final, TYPE_CHECKING

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import nonnegative_integer, positive_finite_float, positive_integer
from ...discretization._topology_epoch import TopologyEpoch, TopologyEpochTransition
from ._contracts import (
    MultiRegionSurfaceCapacityEvidence,
    MultiRegionSurfaceEvidence,
    MultiRegionSurfaceValidationPolicy,
)
from ._state import MultiRegionSurfaceState
from ._topology import MultiRegionSurfaceTopology
from ._transfers import (
    ConservativeFieldTransfer,
    ExtensiveTransferEvidence,
    IntensiveReconstructionEvidence,
)


if TYPE_CHECKING:
    from ...discretization.contact import InclusionCCDPlan


class SurfaceEventKind(IntEnum):
    """Closed set of multiregion topology events."""

    SPLIT = 0
    COLLAPSE = 1
    FLIP = 2
    T1_POP = 3
    PINCH = 4
    MERGE = 5
    REGION_SPLIT = 6
    BURST = 7


class SurfaceEventStatus(IntEnum):
    """Outcome of one proposal: acceptance or the first failed guard."""

    ACCEPTED = 0
    NOT_TRIGGERED = 1
    CONFLICT = 2
    EVENT_BUDGET_EXCEEDED = 3
    CAPACITY_EXCEEDED = 4
    SUPPORT_INVALID = 5
    FEATURE_NOT_PRESERVED = 6
    LINK_CONDITION_VIOLATED = 7
    FAN_STRUCTURE_INVALID = 8
    DUPLICATE_FACE = 9
    DEGENERATE_FACE = 10
    LABEL_ORIENTATION_INCONSISTENT = 11
    NONPHYSICAL_VALENCE = 12
    REGION_GRAPH_INCOMPLETE = 13
    REGION_EXTINCTION = 14
    NORMAL_INVERSION = 15
    VOLUME_RESTORATION_FAILED = 16
    LOCAL_INTERSECTION = 17
    CCD_NOT_CERTIFIED = 18
    CERTIFICATE_CAPACITY_EXCEEDED = 19


class SurfaceEventPassStatus(IntEnum):
    """Outcome of one event pass (commit or the reason for rollback)."""

    COMMITTED = 0
    NO_ACCEPTED_EVENTS = 1
    CAPACITY_EXCEEDED = 2
    CANDIDATE_VALIDATION_FAILED = 3
    TRANSFER_FAILED = 4


def _id_tuple(values: Sequence[int], name: str, /) -> tuple[int, ...]:
    return tuple(nonnegative_integer(value, name) for value in values)


@final
class SurfaceEventRecord(StrictModule, NonTrainableState):
    """Host evidence of one proposal inside one pass.

    ``support_vertex_ids`` are the proposal's defining vertices; the removed and
    created entity ids and ``region_parents`` (``(child, parents)``) are its
    lineage when accepted. ``ccd_time_of_impact`` is the certified collision
    time of the local motion (``1.0`` means collision-free over the whole
    motion; ``nan`` when no motion was certified) and
    ``volume_residual`` the largest relative finite-region volume change left
    after local volume restoration.
    """

    kind: SurfaceEventKind = eqx.field(static=True)
    status: SurfaceEventStatus = eqx.field(static=True)
    support_vertex_ids: tuple[int, ...] = eqx.field(static=True)
    removed_vertex_ids: tuple[int, ...] = eqx.field(static=True)
    created_vertex_ids: tuple[int, ...] = eqx.field(static=True)
    removed_face_ids: tuple[int, ...] = eqx.field(static=True)
    created_face_ids: tuple[int, ...] = eqx.field(static=True)
    region_parents: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True)
    ccd_time_of_impact: float = eqx.field(static=True)
    ccd_certified: bool = eqx.field(static=True)
    volume_residual: float = eqx.field(static=True)
    priority: float = eqx.field(static=True)
    record_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: SurfaceEventKind,
        status: SurfaceEventStatus,
        support_vertex_ids: Sequence[int],
        /,
        *,
        priority: float,
        removed_vertex_ids: Sequence[int] = (),
        created_vertex_ids: Sequence[int] = (),
        removed_face_ids: Sequence[int] = (),
        created_face_ids: Sequence[int] = (),
        region_parents: Sequence[tuple[str, tuple[str, ...]]] = (),
        ccd_time_of_impact: float = math.nan,
        ccd_certified: bool = False,
        volume_residual: float = 0.0,
    ) -> None:
        kind_ = SurfaceEventKind(kind)
        status_ = SurfaceEventStatus(status)
        support = _id_tuple(support_vertex_ids, "support_vertex_ids")
        removed_vertices = _id_tuple(removed_vertex_ids, "removed_vertex_ids")
        created_vertices = _id_tuple(created_vertex_ids, "created_vertex_ids")
        removed_faces = _id_tuple(removed_face_ids, "removed_face_ids")
        created_faces = _id_tuple(created_face_ids, "created_face_ids")
        parents = tuple(
            (str(child), tuple(str(p) for p in ps)) for child, ps in region_parents
        )
        if not isinstance(ccd_certified, bool):
            raise TypeError("ccd_certified must be a bool.")
        toi = float(ccd_time_of_impact)
        residual = float(volume_residual)
        weight = float(priority)
        if not math.isfinite(weight):
            raise ValueError("priority must be finite.")
        self.kind = kind_
        self.status = status_
        self.support_vertex_ids = support
        self.removed_vertex_ids = removed_vertices
        self.created_vertex_ids = created_vertices
        self.removed_face_ids = removed_faces
        self.created_face_ids = created_faces
        self.region_parents = parents
        self.ccd_time_of_impact = toi
        self.ccd_certified = ccd_certified
        self.volume_residual = residual
        self.priority = weight
        self.record_id = canonical_fingerprint(
            {
                "kind": "surface-event-record",
                "event": kind_.name,
                "status": status_.name,
                "support": list(support),
                "removed_vertices": list(removed_vertices),
                "created_vertices": list(created_vertices),
                "removed_faces": list(removed_faces),
                "created_faces": list(created_faces),
                "region_parents": [[child, list(ps)] for child, ps in parents],
                "priority": weight.hex(),
            }
        )

    @property
    def accepted(self) -> bool:
        return self.status is SurfaceEventStatus.ACCEPTED


@final
class MultiRegionSurfaceLineage(StrictModule, NonTrainableState):
    """Stable identity map from one topology epoch to the next.

    ``vertex_parents``/``face_parents`` list ``(child_id, parent_ids)`` for every
    created or merged entity (a collapse survivor lists itself and the removed
    vertex). An empty ``parent_ids`` tuple is the canonical representation of a
    genuinely parentless created entity; absence from both the parent map and
    the removed sets means that an entity preserved its identity.
    ``region_parents`` lists ``(child_region, parent_regions)`` for region splits.
    """

    source_lineage_id: str = eqx.field(static=True)
    target_lineage_id: str = eqx.field(static=True)
    source_epoch: int = eqx.field(static=True)
    target_epoch: int = eqx.field(static=True)
    vertex_parents: tuple[tuple[int, tuple[int, ...]], ...] = eqx.field(static=True)
    face_parents: tuple[tuple[int, tuple[int, ...]], ...] = eqx.field(static=True)
    region_parents: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True)
    removed_vertex_ids: tuple[int, ...] = eqx.field(static=True)
    removed_face_ids: tuple[int, ...] = eqx.field(static=True)
    removed_region_ids: tuple[str, ...] = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_lineage_id: str,
        target_lineage_id: str,
        source_epoch: int,
        target_epoch: int,
        vertex_parents: Sequence[tuple[int, tuple[int, ...]]],
        face_parents: Sequence[tuple[int, tuple[int, ...]]],
        region_parents: Sequence[tuple[str, tuple[str, ...]]],
        removed_vertex_ids: Sequence[int],
        removed_face_ids: Sequence[int],
        removed_region_ids: Sequence[str],
    ) -> None:
        source = nonnegative_integer(source_epoch, "source_epoch")
        target = nonnegative_integer(target_epoch, "target_epoch")
        if target != source + 1:
            raise ValueError("A lineage joins consecutive topology epochs.")
        vertices = tuple(
            sorted(
                (int(child), tuple(sorted(int(p) for p in ps)))
                for child, ps in vertex_parents
            )
        )
        faces = tuple(
            sorted(
                (int(child), tuple(sorted(int(p) for p in ps)))
                for child, ps in face_parents
            )
        )
        regions = tuple(
            sorted(
                (str(child), tuple(sorted(str(p) for p in ps)))
                for child, ps in region_parents
            )
        )
        removed_vertices = tuple(sorted(int(value) for value in removed_vertex_ids))
        removed_faces = tuple(sorted(int(value) for value in removed_face_ids))
        removed_regions = tuple(sorted(str(value) for value in removed_region_ids))
        self.source_lineage_id = str(source_lineage_id)
        self.target_lineage_id = str(target_lineage_id)
        self.source_epoch = source
        self.target_epoch = target
        self.vertex_parents = vertices
        self.face_parents = faces
        self.region_parents = regions
        self.removed_vertex_ids = removed_vertices
        self.removed_face_ids = removed_faces
        self.removed_region_ids = removed_regions
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "multiregion-surface-event-lineage",
                "source": self.source_lineage_id,
                "target": self.target_lineage_id,
                "vertex_parents": [[child, list(ps)] for child, ps in vertices],
                "face_parents": [[child, list(ps)] for child, ps in faces],
                "region_parents": [[child, list(ps)] for child, ps in regions],
                "removed_vertices": list(removed_vertices),
                "removed_faces": list(removed_faces),
                "removed_regions": list(removed_regions),
            }
        )


@final
class SurfaceEventPolicy(StrictModule, NonTrainableState):
    """Guards, certification and restoration settings of event passes.

    ``validation`` certifies the complete candidate (its profile also governs
    the local valence guards); event policies require its self-intersection
    check because no topology mutation may bypass exact candidate validation.
    ``ccd`` certifies every local motion with conservative inclusion CCD;
    ``ccd_candidate_capacity`` bounds the collision stencils examined per
    motion (overflow refuses the event).
    ``fixed_vertex_ids`` (wire frames) are never moved or removed.
    ``restore_region_volumes`` restores the finite-region volumes changed by an
    event with a minimum-norm displacement of the event's free vertices to the
    relative tolerance ``volume_tolerance`` (relative to the cube of the local
    edge length). ``maximum_normal_rotation`` (radians) refuses events that turn
    a surviving or reconnected face by more than that angle.
    """

    validation: MultiRegionSurfaceValidationPolicy
    ccd: InclusionCCDPlan
    ccd_candidate_capacity: int = eqx.field(static=True)
    fixed_vertex_ids: tuple[int, ...] = eqx.field(static=True)
    restore_region_volumes: bool = eqx.field(static=True)
    volume_tolerance: float = eqx.field(static=True)
    maximum_restoration_iterations: int = eqx.field(static=True)
    maximum_normal_rotation: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        validation: MultiRegionSurfaceValidationPolicy | None = None,
        ccd: InclusionCCDPlan | None = None,
        ccd_candidate_capacity: int = 20_000,
        fixed_vertex_ids: Sequence[int] = (),
        restore_region_volumes: bool = True,
        volume_tolerance: float = 1.0e-12,
        maximum_restoration_iterations: int = 8,
        maximum_normal_rotation: float = 0.5 * math.pi,
    ) -> None:
        validation_ = (
            MultiRegionSurfaceValidationPolicy() if validation is None else validation
        )
        if not isinstance(validation_, MultiRegionSurfaceValidationPolicy):
            raise TypeError("validation must be a MultiRegionSurfaceValidationPolicy.")
        if not validation_.check_self_intersection:
            raise ValueError(
                "Surface event candidates require self-intersection validation."
            )
        # The contact package imports geometry; it is resolved at construction.
        from ...discretization.contact import InclusionCCDPlan

        ccd_ = InclusionCCDPlan() if ccd is None else ccd
        if not isinstance(ccd_, InclusionCCDPlan):
            raise TypeError("ccd must be an InclusionCCDPlan.")
        capacity = positive_integer(ccd_candidate_capacity, "ccd_candidate_capacity")
        fixed = tuple(sorted(set(_id_tuple(fixed_vertex_ids, "fixed_vertex_ids"))))
        if not isinstance(restore_region_volumes, bool):
            raise TypeError("restore_region_volumes must be a bool.")
        tolerance = float(positive_finite_float(volume_tolerance, "volume_tolerance"))
        iterations = positive_integer(
            maximum_restoration_iterations, "maximum_restoration_iterations"
        )
        rotation = float(
            positive_finite_float(maximum_normal_rotation, "maximum_normal_rotation")
        )
        if rotation > math.pi:
            raise ValueError("maximum_normal_rotation must not exceed pi.")
        self.validation = validation_
        self.ccd = ccd_
        self.ccd_candidate_capacity = capacity
        self.fixed_vertex_ids = fixed
        self.restore_region_volumes = restore_region_volumes
        self.volume_tolerance = tolerance
        self.maximum_restoration_iterations = iterations
        self.maximum_normal_rotation = rotation
        self.policy_id = canonical_fingerprint(
            {
                "kind": "surface-event-policy",
                "validation": validation_.policy_id,
                "ccd": ccd_.plan_id,
                "ccd_candidate_capacity": capacity,
                "fixed_vertex_ids": list(fixed),
                "restore_region_volumes": restore_region_volumes,
                "volume_tolerance": tolerance.hex(),
                "maximum_restoration_iterations": iterations,
                "maximum_normal_rotation": rotation.hex(),
            }
        )


@final
class SurfaceEventPassEvidence(StrictModule):
    """Evidence of one event pass.

    ``records`` holds every proposal in canonical evaluation order with its
    reason code. When a candidate was assembled, ``capacity`` and
    ``validation`` certify it, ``sheet_transfer``/``region_transfer`` carry the
    conservation residuals of the extensive sheet-slot and region fields and
    ``velocity_reconstruction`` the boundedness of the reconstructed vertex
    velocities. ``derivative_available`` is always ``False``: event passes are
    nondifferentiable epoch transitions.
    """

    status: SurfaceEventPassStatus = eqx.field(static=True)
    committed: bool = eqx.field(static=True)
    records: tuple[SurfaceEventRecord, ...]
    accepted_count: int = eqx.field(static=True)
    lineage: MultiRegionSurfaceLineage | None
    capacity: MultiRegionSurfaceCapacityEvidence | None
    validation: MultiRegionSurfaceEvidence | None
    sheet_transfer: ExtensiveTransferEvidence | None
    region_transfer: ExtensiveTransferEvidence | None
    velocity_reconstruction: IntensiveReconstructionEvidence | None
    source_epoch: TopologyEpoch
    target_epoch: TopologyEpoch | None
    transaction_id: str | None = eqx.field(static=True)
    derivative_available: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def records_of(self, kind: SurfaceEventKind, /) -> tuple[SurfaceEventRecord, ...]:
        """Records of one event kind in evaluation order."""
        kind_ = SurfaceEventKind(kind)
        return tuple(record for record in self.records if record.kind is kind_)


@final
class SurfaceEventPassResult(StrictModule):
    """Committed candidate or the unchanged source, with sparse transfers.

    ``transition`` and ``face_transition`` are nondifferentiable epoch
    transitions for sheet-slot and face extensive content. The owning sparse
    transfers are exposed so an application can apply one prepared route to
    all of its field components; every transfer field is ``None`` unless the
    pass committed.
    """

    topology: MultiRegionSurfaceTopology
    state: MultiRegionSurfaceState
    evidence: SurfaceEventPassEvidence
    sheet_transfer: ConservativeFieldTransfer | None
    face_transfer: ConservativeFieldTransfer | None
    transition: TopologyEpochTransition | None
    face_transition: TopologyEpochTransition | None

    @property
    def committed(self) -> bool:
        return self.evidence.committed


__all__ = [
    "MultiRegionSurfaceLineage",
    "SurfaceEventKind",
    "SurfaceEventPassEvidence",
    "SurfaceEventPassResult",
    "SurfaceEventPassStatus",
    "SurfaceEventPolicy",
    "SurfaceEventRecord",
    "SurfaceEventStatus",
]
