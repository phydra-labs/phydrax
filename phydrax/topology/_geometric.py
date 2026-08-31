#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._coefficients import PrimeField
from ._complex import CellComplexPair, CellSubcomplex
from ._filtration import CellFiltration
from ._homology import compute_homology, HomologyResult
from ._optimized_persistence import compute_h0_persistence_union_find
from ._resources import TopologyResourcePolicy


class MergeTree(StrictModule, NonTrainableState):
    """Creator/death hierarchy derived from exact H0 elder-rule persistence."""

    component_entity_ids: Array
    merge_entity_ids: Array
    birth_values: Array
    merge_values: Array
    has_parent: Array
    filtration_id: str = eqx.field(static=True)
    tree_id: str = eqx.field(static=True)

    def __init__(self, filtration: CellFiltration, /):
        diagram = compute_h0_persistence_union_find(filtration)
        self.component_entity_ids = diagram.birth_entity_ids
        self.merge_entity_ids = diagram.death_entity_ids
        self.birth_values = diagram.birth_values
        self.merge_values = diagram.death_values
        self.has_parent = diagram.has_finite_death
        self.filtration_id = filtration.filtration_id
        self.tree_id = canonical_fingerprint(
            {
                "kind": "merge-tree",
                "filtration": filtration.filtration_id,
                "diagram": diagram.diagram_id,
            }
        )


class LocalHomologyReport(StrictModule, NonTrainableState):
    """Exact relative homology of a cell's ambient open-star neighborhood."""

    homology: HomologyResult
    degree: int = eqx.field(static=True)
    ambient_cell: int = eqx.field(static=True)
    neighborhood_pair_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        homology: HomologyResult,
        degree: int,
        ambient_cell: int,
        neighborhood_pair_id: str,
        /,
    ):
        self.homology = homology
        self.degree = int(degree)
        self.ambient_cell = int(ambient_cell)
        self.neighborhood_pair_id = str(neighborhood_pair_id)
        self.report_id = canonical_fingerprint(
            {
                "kind": "local-homology-report",
                "homology": homology.result_id,
                "degree": int(degree),
                "cell": int(ambient_cell),
                "pair": self.neighborhood_pair_id,
            }
        )


def compute_cell_local_homology(
    complex: CellSubcomplex,
    degree: int,
    ambient_cell: int,
    /,
    *,
    coefficients: PrimeField,
    resources: TopologyResourcePolicy | None = None,
) -> LocalHomologyReport:
    """Compute H(K, K minus open-star(cell)) on the declared finite complex."""
    degree_ = int(degree)
    cell = int(ambient_cell)
    if degree_ < 0 or degree_ > complex.max_degree:
        raise ValueError("Local homology degree is outside the complex.")
    if cell < 0 or cell >= complex.topology.entity_sets[degree_].count:
        raise ValueError("Local homology cell is outside the entity set.")
    if not bool(np.asarray(complex.masks[degree_])[cell]):
        raise ValueError("Local homology cell is not selected in the complex.")
    cofaces = [np.zeros_like(np.asarray(mask), dtype=bool) for mask in complex.masks]
    cofaces[degree_][cell] = True
    for current_degree in range(degree_ + 1, complex.max_degree + 1):
        incidence = complex.topology.incidences[current_degree - 1]
        valid = np.asarray(incidence.relation.valid, dtype=bool)
        lower = np.asarray(incidence.relation.source_indices)[valid]
        upper = np.asarray(incidence.relation.target_indices)[valid]
        for lower_cell, upper_cell in zip(lower, upper, strict=True):
            if cofaces[current_degree - 1][int(lower_cell)]:
                cofaces[current_degree][int(upper_cell)] = True
    complement_masks = tuple(
        np.asarray(mask, dtype=bool) & ~coface
        for mask, coface in zip(complex.masks, cofaces, strict=True)
    )
    complement = CellSubcomplex(complex.topology, complement_masks)
    pair = CellComplexPair(complex, complement)
    result = compute_homology(
        pair,
        coefficients=coefficients,
        resources=resources,
    )
    return LocalHomologyReport(result, degree_, cell, pair.pair_id)


__all__ = ["LocalHomologyReport", "MergeTree", "compute_cell_local_homology"]
