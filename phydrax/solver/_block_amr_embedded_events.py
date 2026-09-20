#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Accepted-boundary moving embedded-boundary topology transactions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import TopologyEpoch
from ..discretization.amr import VariablePatchGeometryState
from ._finite_volume_topology_events import (
    FiniteVolumeTopologyArtifacts,
    FiniteVolumeTopologyEventJournal,
    FiniteVolumeTopologyEventRequest,
    TopologyEventKind,
    TopologyEventStatus,
)


EmbeddedMotionField = Callable[[Array, Array, Any], ArrayLike]


class MovingEmbeddedBoundaryEventEvidence(StrictModule, NonTrainableState):
    """Time-slab sign-margin, swept-budget, and successor-epoch evidence."""

    sign_margin: float = eqx.field(static=True)
    sign_changed: bool = eqx.field(static=True)
    symmetric_difference_volume: float = eqx.field(static=True)
    swept_wall_volume: float = eqx.field(static=True)
    budget_defect: float = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        sign_margin: float,
        sign_changed: bool,
        symmetric_difference_volume: float,
        swept_wall_volume: float,
        budget_defect: float,
        successful: bool,
        /,
    ):
        values = tuple(
            float(value)
            for value in (
                sign_margin,
                symmetric_difference_volume,
                swept_wall_volume,
                budget_defect,
            )
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Moving EB event evidence values must be finite/nonnegative."
            )
        self.sign_margin = values[0]
        self.sign_changed = bool(sign_changed)
        self.symmetric_difference_volume = values[1]
        self.swept_wall_volume = values[2]
        self.budget_defect = values[3]
        self.successful = bool(successful)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "moving-eb-event-evidence",
                "sign_margin": values[0],
                "sign_changed": bool(sign_changed),
                "symmetric_difference_volume": values[1],
                "swept_wall_volume": values[2],
                "budget_defect": values[3],
                "successful": bool(successful),
            }
        )


class MovingEmbeddedBoundaryEventResult(StrictModule, NonTrainableState):
    journal: FiniteVolumeTopologyEventJournal
    epoch: TopologyEpoch
    committed: bool = eqx.field(static=True)
    evidence: MovingEmbeddedBoundaryEventEvidence
    result_id: str = eqx.field(static=True)


class MovingEmbeddedBoundaryEventPlan(StrictModule, NonTrainableState):
    """Host transaction for one moving-body sign-topology crossing.

    Continuous fixed-sign motion remains a geometry-version update.  Any sign change is
    represented by one consecutive topology epoch and committed only when the declared
    symmetric-difference/swept-volume budget closes.
    """

    level_set: EmbeddedMotionField = eqx.field(static=True)
    body_id: str = eqx.field(static=True)
    sign_tolerance: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        level_set: EmbeddedMotionField,
        body_id: str,
        /,
        *,
        sign_tolerance: float = 1.0e-10,
        conservation_tolerance: float = 1.0e-10,
    ):
        body = str(body_id)
        sign = float(sign_tolerance)
        conservation = float(conservation_tolerance)
        if (
            not callable(level_set)
            or not body
            or not np.isfinite(sign)
            or sign <= 0.0
            or not np.isfinite(conservation)
            or conservation < 0.0
        ):
            raise ValueError("Moving embedded-boundary event plan is invalid.")
        self.level_set = level_set
        self.body_id = body
        self.sign_tolerance = sign
        self.conservation_tolerance = conservation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "moving-eb-event-plan",
                "body": body,
                "sign_tolerance": sign,
                "conservation_tolerance": conservation,
            }
        )

    @staticmethod
    def _vertex_values(
        geometry: VariablePatchGeometryState,
        level_set: EmbeddedMotionField,
        args: Any,
        /,
    ) -> tuple[np.ndarray, ...]:
        result = []
        for level_vertices, level_masks in zip(
            geometry.vertex_coordinates,
            geometry.plan.active_vertex_masks,
            strict=True,
        ):
            for vertices, active in zip(level_vertices, level_masks, strict=True):
                flat = vertices.reshape((-1, vertices.shape[-1]))
                values = np.asarray(
                    level_set(flat, geometry.time, args), dtype=np.float64
                )
                if values.shape != (flat.shape[0],):
                    raise ValueError(
                        "Moving EB level set must return one value per vertex."
                    )
                mask = np.asarray(active, dtype=np.bool_).reshape((-1,))
                if np.any(~np.isfinite(values[mask])):
                    raise ValueError("Moving EB active vertex values must be finite.")
                result.append(values[mask])
        return tuple(result)

    def transact(
        self,
        journal: FiniteVolumeTopologyEventJournal,
        source_geometry: VariablePatchGeometryState,
        target_geometry: VariablePatchGeometryState,
        accepted_step: int,
        symmetric_difference_volume: float,
        swept_wall_volume: float,
        /,
        *,
        args: Any = None,
    ) -> MovingEmbeddedBoundaryEventResult:
        if not isinstance(journal, FiniteVolumeTopologyEventJournal):
            raise TypeError("Moving EB transaction requires a topology event journal.")
        if (
            not isinstance(source_geometry, VariablePatchGeometryState)
            or not isinstance(target_geometry, VariablePatchGeometryState)
            or source_geometry.plan.topology.epoch.epoch_id != journal.current_epoch_id
            or target_geometry.plan.topology.topology_id
            != source_geometry.plan.topology.topology_id
        ):
            raise ValueError("Moving EB transaction geometry has stale topology.")
        source_values = self._vertex_values(source_geometry, self.level_set, args)
        target_values = self._vertex_values(target_geometry, self.level_set, args)
        if len(source_values) != len(target_values) or any(
            source.shape != target.shape
            for source, target in zip(source_values, target_values, strict=True)
        ):
            raise ValueError("Moving EB geometry changed vertex capacity during a step.")
        margin = min(
            float(np.min(np.abs(values), initial=np.inf))
            for values in (*source_values, *target_values)
        )
        sign_changed = any(
            np.any((source > 0.0) != (target > 0.0))
            for source, target in zip(source_values, target_values, strict=True)
        )
        symmetric = float(symmetric_difference_volume)
        swept = float(swept_wall_volume)
        defect = abs(symmetric - swept)
        successful = (
            sign_changed
            and margin > self.sign_tolerance
            and defect <= self.conservation_tolerance * max(symmetric, swept, 1.0)
        )
        evidence = MovingEmbeddedBoundaryEventEvidence(
            margin,
            sign_changed,
            symmetric,
            swept,
            defect,
            successful,
        )
        request = FiniteVolumeTopologyEventRequest(
            TopologyEventKind.AMR_REGRID,
            journal.current_epoch_id,
            self.plan_id,
            payload_id=evidence.evidence_id,
            reason="moving embedded-boundary sign topology changed",
        )
        requested = journal.append_requested(
            request,
            accepted_step,
            target_geometry.time,
        )
        sequence = int(np.asarray(requested.next_sequence)) - 1
        if not successful:
            failed = requested.fail(
                sequence,
                status=(
                    TopologyEventStatus.FAILED_COVERAGE
                    if sign_changed
                    else TopologyEventStatus.FAILED_STALE_EPOCH
                ),
                result_id=evidence.evidence_id,
                payload_id=evidence.evidence_id,
            )
            return MovingEmbeddedBoundaryEventResult(
                failed,
                source_geometry.plan.topology.epoch,
                False,
                evidence,
                evidence.evidence_id,
            )
        source_epoch = source_geometry.plan.topology.epoch
        topology_id = canonical_fingerprint(
            {
                "kind": "moving-eb-cut-topology",
                "source_topology": source_epoch.topology_id,
                "body": self.body_id,
                "target_signs": [
                    array_tree_fingerprint(values > 0.0) for values in target_values
                ],
            }
        )
        epoch = TopologyEpoch(
            source_epoch.index + 1,
            source_epoch.geometry_id,
            topology_id,
            source_epoch.partition_id,
        )
        artifacts = FiniteVolumeTopologyArtifacts(
            epoch,
            self.plan_id,
            topology_artifact_id=evidence.evidence_id,
            metrics_artifact_id=target_geometry.plan.plan_id,
        )
        committed = requested.commit(
            sequence,
            epoch,
            artifacts,
            result_id=epoch.epoch_id,
            payload_id=evidence.evidence_id,
        )
        return MovingEmbeddedBoundaryEventResult(
            committed,
            epoch,
            True,
            evidence,
            canonical_fingerprint(
                {
                    "kind": "moving-eb-event-result",
                    "epoch": epoch.epoch_id,
                    "evidence": evidence.evidence_id,
                }
            ),
        )


__all__ = [
    "MovingEmbeddedBoundaryEventEvidence",
    "MovingEmbeddedBoundaryEventPlan",
    "MovingEmbeddedBoundaryEventResult",
]
