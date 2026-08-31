#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum
from typing import Any

import equinox as eqx

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class EvidenceStatus(IntEnum):
    CERTIFIED = 0
    FAILED = 1
    INCOMPLETE = 2
    AMBIGUOUS = 3
    OUTSIDE_APPLICABILITY = 4


class EvidenceAcquisitionAction(StrictModule, NonTrainableState):
    action_id: str = eqx.field(static=True)
    description: str = eqx.field(static=True)
    resolves: tuple[str, ...] = eqx.field(static=True)
    cost: float = eqx.field(static=True)
    expected_information_gain: float = eqx.field(static=True)


class EvidenceNode(StrictModule):
    evidence_id: str = eqx.field(static=True)
    dependencies: tuple[str, ...] = eqx.field(static=True)
    fidelity: str = eqx.field(static=True)
    status: int = eqx.field(static=True)
    result: Any
    missing_inputs: tuple[str, ...] = eqx.field(static=True)
    actions: tuple[EvidenceAcquisitionAction, ...]


class EvidenceGraph(StrictModule):
    nodes: tuple[EvidenceNode, ...]
    graph_id: str = eqx.field(static=True)

    def __init__(self, nodes: Sequence[EvidenceNode], /):
        nodes_ = tuple(nodes)
        identifiers = {value.evidence_id for value in nodes_}
        if len(identifiers) != len(nodes_):
            raise ValueError("Evidence node IDs must be unique.")
        if any(
            dependency not in identifiers
            for node in nodes_
            for dependency in node.dependencies
        ):
            raise ValueError("Every evidence dependency must identify one node.")
        self.nodes = nodes_
        self._assert_acyclic()
        self.graph_id = canonical_fingerprint(
            {
                "kind": "structural-evidence-graph",
                "nodes": [
                    {
                        "id": value.evidence_id,
                        "dependencies": list(value.dependencies),
                        "fidelity": value.fidelity,
                    }
                    for value in nodes_
                ],
            }
        )

    def _assert_acyclic(self) -> None:
        completed: set[str] = set()
        while len(completed) < len(self.nodes):
            available = [
                value.evidence_id
                for value in self.nodes
                if value.evidence_id not in completed
                and set(value.dependencies).issubset(completed)
            ]
            if not available:
                raise ValueError("Evidence dependencies contain a cycle.")
            completed.update(available)

    def node(self, evidence_id: str, /) -> EvidenceNode:
        matches = tuple(value for value in self.nodes if value.evidence_id == evidence_id)
        if len(matches) != 1:
            raise KeyError(evidence_id)
        return matches[0]

    def required_actions(
        self, evidence_id: str, /
    ) -> tuple[EvidenceAcquisitionAction, ...]:
        node = self.node(evidence_id)
        pending = [node] + [self.node(value) for value in node.dependencies]
        actions = {
            action.action_id: action
            for value in pending
            if value.status
            in (
                int(EvidenceStatus.INCOMPLETE),
                int(EvidenceStatus.AMBIGUOUS),
            )
            for action in value.actions
        }
        return tuple(
            sorted(
                actions.values(),
                key=lambda value: (
                    -value.expected_information_gain / max(value.cost, 1.0e-15),
                    value.action_id,
                ),
            )
        )


class StructuralTwinSnapshot(StrictModule):
    snapshot_id: str = eqx.field(static=True)
    parent_snapshot_id: str | None = eqx.field(static=True)
    design_state: Any
    as_built_state: Any
    calibration_state: Any
    service_state: Any
    evidence_graph: EvidenceGraph
    metadata: tuple[tuple[str, str], ...] = eqx.field(static=True)

    @classmethod
    def create(
        cls,
        evidence_graph: EvidenceGraph,
        /,
        *,
        design_state: Any = None,
        as_built_state: Any = None,
        calibration_state: Any = None,
        service_state: Any = None,
        parent_snapshot_id: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> StructuralTwinSnapshot:
        metadata_ = tuple(
            sorted((str(key), str(value)) for key, value in (metadata or {}).items())
        )
        identifier = canonical_fingerprint(
            {
                "kind": "structural-twin-snapshot",
                "parent": parent_snapshot_id,
                "evidence": evidence_graph.graph_id,
                "metadata": list(metadata_),
            }
        )
        return cls(
            identifier,
            parent_snapshot_id,
            design_state,
            as_built_state,
            calibration_state,
            service_state,
            evidence_graph,
            metadata_,
        )


__all__ = [
    "EvidenceAcquisitionAction",
    "EvidenceGraph",
    "EvidenceNode",
    "EvidenceStatus",
    "StructuralTwinSnapshot",
]
