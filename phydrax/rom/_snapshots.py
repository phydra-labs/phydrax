#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..data_utils import CasePartitionManifest


SnapshotRole: TypeAlias = Literal[
    "state",
    "nonlinear-term",
    "time-discrete-residual",
    "jacobian-action",
    "element-contribution",
    "primal-trajectory",
    "dual-trajectory",
]


class SnapshotManifest(StrictModule, NonTrainableState):
    """Immutable scientific identity of one chunked snapshot collection."""

    role: SnapshotRole = eqx.field(static=True)
    partition: CasePartitionManifest
    case_ids: tuple[str, ...] = eqx.field(static=True)
    chunk_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    state_layout_id: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    quadrature_id: str = eqx.field(static=True)
    truth_revision_id: str = eqx.field(static=True)
    source_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        role: SnapshotRole,
        partition: CasePartitionManifest,
        case_ids: Sequence[str],
        chunk_artifact_ids: Sequence[str],
        /,
        *,
        state_layout_id: str,
        field_space_id: str,
        support_id: str,
        measure_id: str,
        geometry_id: str,
        topology_id: str,
        quadrature_id: str,
        truth_revision_id: str,
        source_artifact_ids: Sequence[str],
    ):
        roles = (
            "state",
            "nonlinear-term",
            "time-discrete-residual",
            "jacobian-action",
            "element-contribution",
            "primal-trajectory",
            "dual-trajectory",
        )
        if role not in roles:
            raise ValueError("Unknown snapshot role.")
        if not isinstance(partition, CasePartitionManifest):
            raise TypeError("partition must be a CasePartitionManifest.")
        cases = tuple(str(value) for value in case_ids)
        chunks = tuple(str(value) for value in chunk_artifact_ids)
        sources = tuple(str(value) for value in source_artifact_ids)
        identifiers = tuple(
            str(value)
            for value in (
                state_layout_id,
                field_space_id,
                support_id,
                measure_id,
                geometry_id,
                topology_id,
                quadrature_id,
                truth_revision_id,
            )
        )
        if (
            not cases
            or not chunks
            or not sources
            or any(not value for value in (*cases, *chunks, *sources, *identifiers))
        ):
            raise ValueError("Snapshot manifest identities must be non-empty.")
        if len(set(cases)) != len(cases) or len(set(chunks)) != len(chunks):
            raise ValueError("Snapshot case and chunk IDs must be unique.")
        if any(case not in partition.case_ids for case in cases):
            raise ValueError("Snapshot cases must belong to the shared partition.")
        self.role = role
        self.partition = partition
        self.case_ids = cases
        self.chunk_artifact_ids = chunks
        (
            self.state_layout_id,
            self.field_space_id,
            self.support_id,
            self.measure_id,
            self.geometry_id,
            self.topology_id,
            self.quadrature_id,
            self.truth_revision_id,
        ) = identifiers
        self.source_artifact_ids = sources
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "snapshot-manifest",
                "role": role,
                "partition": partition.partition_id,
                "cases": list(cases),
                "chunks": list(chunks),
                "state_layout": identifiers[0],
                "field_space": identifiers[1],
                "support": identifiers[2],
                "measure": identifiers[3],
                "geometry": identifiers[4],
                "topology": identifiers[5],
                "quadrature": identifiers[6],
                "truth_revision": identifiers[7],
                "sources": list(sources),
            }
        )


__all__ = ["SnapshotManifest", "SnapshotRole"]
