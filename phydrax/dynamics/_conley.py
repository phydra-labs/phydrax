#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..topology import (
    CellComplexPair,
    CellularPairMap,
    compute_homology,
    FiniteFieldCoordinateMap,
    HomologyResult,
    induced_homology_coordinates,
    PrimeField,
    TopologyResourcePolicy,
)
from ._cell_enclosure import CellMapEnclosure


class ConleyHomologyIndex(StrictModule, NonTrainableState):
    """Field-qualified relative homology index of a certified isolating pair."""

    enclosure: CellMapEnclosure
    homology: HomologyResult
    result_id: str = eqx.field(static=True)

    def __init__(self, enclosure: CellMapEnclosure, homology: HomologyResult, /):
        self.enclosure = enclosure
        self.homology = homology
        self.result_id = canonical_fingerprint(
            {
                "kind": "conley-homology-index",
                "enclosure": enclosure.enclosure_id,
                "homology": homology.result_id,
            }
        )


class ConleyIndexResult(StrictModule, NonTrainableState):
    """Relative homology and induced index endomorphism of an isolating pair."""

    homology_index: ConleyHomologyIndex
    index_maps: tuple[FiniteFieldCoordinateMap, ...]
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        homology_index: ConleyHomologyIndex,
        index_maps: tuple[FiniteFieldCoordinateMap, ...],
        /,
    ):
        self.homology_index = homology_index
        self.index_maps = index_maps
        self.result_id = canonical_fingerprint(
            {
                "kind": "conley-index-result",
                "homology_index": homology_index.result_id,
                "index_maps": [value.map_id for value in index_maps],
            }
        )


def compute_conley_homology_index(
    enclosure: CellMapEnclosure,
    /,
    *,
    coefficients: PrimeField,
    resources: TopologyResourcePolicy | None = None,
) -> ConleyHomologyIndex:
    """Compute relative index-pair homology after isolation evidence succeeds."""
    if not isinstance(enclosure, CellMapEnclosure):
        raise TypeError("Conley homology requires CellMapEnclosure.")
    if not enclosure.isolating:
        raise ValueError("Conley homology index requires a certified isolating pair.")
    pair = CellComplexPair(enclosure.neighborhood, enclosure.exit_set)
    homology = compute_homology(
        pair,
        coefficients=coefficients,
        resources=resources,
    )
    return ConleyHomologyIndex(enclosure, homology)


def compute_conley_index(
    enclosure: CellMapEnclosure,
    index_pair_map: CellularPairMap,
    /,
    *,
    coefficients: PrimeField,
    resources: TopologyResourcePolicy | None = None,
) -> ConleyIndexResult:
    """Compute relative homology and the induced discrete index endomorphism."""
    homology_index = compute_conley_homology_index(
        enclosure,
        coefficients=coefficients,
        resources=resources,
    )
    pair = CellComplexPair(enclosure.neighborhood, enclosure.exit_set)
    if (
        index_pair_map.source.pair_id != pair.pair_id
        or index_pair_map.target.pair_id != pair.pair_id
    ):
        raise ValueError("Conley index map must be an endomorphism of the index pair.")
    homology = compute_homology(
        pair,
        coefficients=coefficients,
        representatives="both",
        resources=resources,
    )
    maps = induced_homology_coordinates(
        index_pair_map.quotient_maps,
        pair.quotient_layout,
        pair.quotient_layout,
        homology,
        homology,
    )
    return ConleyIndexResult(homology_index, maps)


__all__ = [
    "ConleyHomologyIndex",
    "ConleyIndexResult",
    "compute_conley_homology_index",
    "compute_conley_index",
]
