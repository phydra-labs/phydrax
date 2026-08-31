#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation
from ..topology import CellSubcomplex


class CellMapEnclosure(StrictModule, NonTrainableState):
    """Outer multivalued cell-map enclosure with one declared index pair."""

    neighborhood: CellSubcomplex
    exit_set: CellSubcomplex
    relation: EdgeRelation
    degree: int = eqx.field(static=True)
    isolating: bool = eqx.field(static=True)
    enclosure_id: str = eqx.field(static=True)

    def __init__(
        self,
        neighborhood: CellSubcomplex,
        exit_set: CellSubcomplex,
        relation: EdgeRelation,
        /,
        *,
        degree: int,
    ):
        degree_ = int(degree)
        if neighborhood.topology.topology_id != exit_set.topology.topology_id:
            raise ValueError("Cell enclosure index pair must share one topology.")
        if degree_ < 0 or degree_ > neighborhood.max_degree:
            raise ValueError("Cell enclosure degree is outside the topology.")
        neighborhood_mask = np.asarray(neighborhood.masks[degree_], dtype=bool)
        exit_mask = np.asarray(exit_set.masks[degree_], dtype=bool)
        if np.any(exit_mask & ~neighborhood_mask):
            raise ValueError("Cell enclosure exit set must lie in the neighborhood.")
        count = neighborhood.topology.entity_sets[degree_].count
        if relation.source_size != count or relation.target_size != count:
            raise ValueError("Cell enclosure relation size does not match entity count.")
        valid = np.asarray(relation.valid, dtype=bool)
        sources = np.asarray(relation.source_indices)[valid]
        targets = np.asarray(relation.target_indices)[valid]
        outgoing = [set() for _ in range(count)]
        for source, target in zip(sources, targets, strict=True):
            outgoing[int(source)].add(int(target))
        interior = neighborhood_mask & ~exit_mask
        isolating = True
        for cell in np.flatnonzero(interior):
            if not outgoing[int(cell)] or any(
                not neighborhood_mask[target] for target in outgoing[int(cell)]
            ):
                isolating = False
                break
        self.neighborhood = neighborhood
        self.exit_set = exit_set
        self.relation = relation
        self.degree = degree_
        self.isolating = isolating
        self.enclosure_id = canonical_fingerprint(
            {
                "kind": "cell-map-enclosure",
                "neighborhood": neighborhood.subcomplex_id,
                "exit": exit_set.subcomplex_id,
                "degree": degree_,
                "isolating": isolating,
            }
        )


__all__ = ["CellMapEnclosure"]
