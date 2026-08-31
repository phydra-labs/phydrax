#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge
from ._coefficients import PrimeField
from ._complex import CellSubcomplex
from ._filtration import cell_vertex_support, lower_star_filtration
from ._persistence import compute_persistence, PersistenceResult
from ._resources import TopologyResourcePolicy


class CubicalPersistenceResult(StrictModule, NonTrainableState):
    """Exact persistence on a canonical structured cubical bridge."""

    bridge_id: str = eqx.field(static=True)
    persistence: PersistenceResult
    result_id: str = eqx.field(static=True)

    def __init__(self, bridge_id: str, persistence: PersistenceResult, /):
        self.bridge_id = str(bridge_id)
        self.persistence = persistence
        self.result_id = canonical_fingerprint(
            {
                "kind": "cubical-persistence-result",
                "bridge": self.bridge_id,
                "persistence": persistence.result_id,
            }
        )


def _regular_vertex_support(bridge: StructuredCochainBridge, /):
    topology = bridge.cochain.topology
    supports: list[list[set[int]]] = [
        [{index} for index in range(topology.entity_sets[0].count)]
    ]
    for degree, incidence in enumerate(topology.incidences, start=1):
        current = [set() for _ in range(topology.entity_sets[degree].count)]
        valid = np.asarray(incidence.relation.valid, dtype=bool)
        lower = np.asarray(incidence.relation.source_indices)[valid]
        upper = np.asarray(incidence.relation.target_indices)[valid]
        for lower_cell, upper_cell in zip(lower, upper, strict=True):
            current[int(upper_cell)].update(supports[degree - 1][int(lower_cell)])
        supports.append(current)
    arrays = []
    for degree_support in supports:
        width = max((len(value) for value in degree_support), default=1)
        array = np.full((len(degree_support), width), -1, dtype=np.int32)
        for cell, vertices in enumerate(degree_support):
            array[cell, : len(vertices)] = sorted(vertices)
        arrays.append(array)
    return cell_vertex_support(topology, arrays)


def compute_structured_cubical_persistence(
    bridge: StructuredCochainBridge,
    vertex_values: ArrayLike,
    /,
    *,
    coefficients: PrimeField,
    source_id: str,
    max_degree: int | None = None,
    resources: TopologyResourcePolicy | None = None,
) -> CubicalPersistenceResult:
    """Compute lower-star persistence on one explicit structured cubical complex."""
    if not isinstance(bridge, StructuredCochainBridge):
        raise TypeError("Cubical persistence requires StructuredCochainBridge.")
    complex = CellSubcomplex.full(bridge.cochain.topology)
    support = _regular_vertex_support(bridge)
    filtration = lower_star_filtration(
        complex,
        support,
        vertex_values,
        source_id=source_id,
    )
    result = compute_persistence(
        filtration,
        coefficients=coefficients,
        max_degree=max_degree,
        resources=resources,
    )
    return CubicalPersistenceResult(bridge.bridge_id, result)


__all__ = ["CubicalPersistenceResult", "compute_structured_cubical_persistence"]
