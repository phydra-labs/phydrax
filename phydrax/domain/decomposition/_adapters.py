#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import CellPartition
from ...geometry import CompiledGeometry
from ...metrix import AtlasCover
from .._domain import Domain
from .._function import DomainFunction
from .._geometry import GeometryDomain
from ._cover import SubdomainCover, SubdomainPatch


class CoverAdapterEvidence(StrictModule, NonTrainableState):
    adapter_kind: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    cover_id: str = eqx.field(static=True)
    patch_count: int = eqx.field(static=True)
    relation_count: int = eqx.field(static=True)
    verified: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        adapter_kind: str,
        source_id: str,
        cover_id: str,
        patch_count: int,
        relation_count: int,
        verified: bool,
    ):
        self.adapter_kind = str(adapter_kind)
        self.source_id = str(source_id)
        self.cover_id = str(cover_id)
        self.patch_count = int(patch_count)
        self.relation_count = int(relation_count)
        self.verified = bool(verified)


def geometry_subdomain_patch(
    ambient: Domain,
    geometry: CompiledGeometry,
    support: DomainFunction,
    to_local: Mapping[str, DomainFunction],
    to_ambient: Mapping[str, DomainFunction],
    /,
    *,
    patch_id: str,
    label: str = "x",
    window: DomainFunction | None = None,
) -> SubdomainPatch:
    """Adapt one compiled CSG/reconstructed region to a mapped local patch."""
    local = GeometryDomain(geometry, label=label)
    if local.labels != ambient.labels:
        raise ValueError(
            "geometry_subdomain_patch currently requires a single-factor ambient "
            "domain with the same coordinate label."
        )
    return SubdomainPatch(
        local,
        local.component(),
        support,
        to_local,
        to_ambient,
        patch_id=patch_id,
        window=window,
    )


def validate_cell_partition_cover(
    partition: CellPartition,
    cover: SubdomainCover,
    /,
) -> CoverAdapterEvidence:
    """Bind exactly-once mesh cell ownership to canonical cover patch order."""
    if not isinstance(partition, CellPartition):
        raise TypeError("partition must be a CellPartition.")
    if not isinstance(cover, SubdomainCover):
        raise TypeError("cover must be a SubdomainCover.")
    verified = partition.part_count == len(cover.patches) and cover.exact_coverage
    return CoverAdapterEvidence(
        adapter_kind="cell-partition",
        source_id=partition.partition_id,
        cover_id=cover.cover_id,
        patch_count=partition.part_count,
        relation_count=len(cover.pairings),
        verified=verified,
    )


def validate_atlas_cover_adapter(
    atlas: AtlasCover,
    cover: SubdomainCover,
    chart_patch_ids: Sequence[str],
    /,
) -> CoverAdapterEvidence:
    """Validate atlas chart/transition ownership against subdomain patch topology."""
    if not isinstance(atlas, AtlasCover):
        raise TypeError("atlas must be an AtlasCover.")
    if not isinstance(cover, SubdomainCover):
        raise TypeError("cover must be a SubdomainCover.")
    mapping = tuple(str(value) for value in chart_patch_ids)
    if len(mapping) != len(atlas.atlas.charts) or len(set(mapping)) != len(mapping):
        raise ValueError("chart_patch_ids must map every atlas chart uniquely.")
    if not set(mapping).issubset(cover.patch_ids):
        raise ValueError("Atlas adapter references an unknown subdomain patch.")
    cover_edges = {
        frozenset((pairing.left_patch_id, pairing.right_patch_id))
        for pairing in cover.pairings
    }
    transition_edges = {
        frozenset((mapping[overlap.source_index], mapping[overlap.target_index]))
        for overlap in atlas.overlaps
    }
    verified = len(mapping) == len(cover.patches) and transition_edges.issubset(
        cover_edges
    )
    return CoverAdapterEvidence(
        adapter_kind="atlas-cover",
        source_id=atlas.cover_id,
        cover_id=cover.cover_id,
        patch_count=len(mapping),
        relation_count=len(transition_edges),
        verified=verified,
    )


__all__ = [
    "CoverAdapterEvidence",
    "geometry_subdomain_patch",
    "validate_atlas_cover_adapter",
    "validate_cell_partition_cover",
]
