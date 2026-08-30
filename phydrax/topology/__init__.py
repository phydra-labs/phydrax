#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact finite-complex topology and evidence-bearing persistence analysis."""

from ._coefficients import CoefficientDomain, PrimeField, RationalField
from ._complex import (
    CellComplexPair,
    CellSubcomplex,
    CellVertexSupport,
    compact_boundary,
    CompactBoundary,
    CompactCellLayout,
)
from ._diagram import PackedPersistenceDiagram, PersistenceDiagram
from ._filtration import (
    cell_vertex_support,
    CellFiltration,
    FiltrationDirection,
    lower_star_filtration,
    PreparedVertexFiltration,
    upper_star_filtration,
)
from ._homology import (
    BettiDimensionResult,
    compute_betti_dimensions,
    compute_homology,
    FiniteFieldBasis,
    HomologyDegreeResult,
    HomologyResult,
    RepresentativeKind,
)
from ._persistence import (
    compute_persistence,
    freeze_persistence_pairing,
    FrozenPersistenceEvaluation,
    FrozenPersistencePairing,
    PersistencePairing,
    PersistenceRepresentativeKind,
    PersistenceRepresentatives,
    PersistenceResult,
)
from ._resources import (
    TopologyReductionEvidence,
    TopologyResourceError,
    TopologyResourcePolicy,
)


__all__ = [
    "BettiDimensionResult",
    "CellComplexPair",
    "CellFiltration",
    "CellSubcomplex",
    "CellVertexSupport",
    "CoefficientDomain",
    "CompactBoundary",
    "CompactCellLayout",
    "FiltrationDirection",
    "FiniteFieldBasis",
    "FrozenPersistenceEvaluation",
    "FrozenPersistencePairing",
    "HomologyDegreeResult",
    "HomologyResult",
    "PackedPersistenceDiagram",
    "PersistenceDiagram",
    "PersistencePairing",
    "PersistenceRepresentativeKind",
    "PersistenceRepresentatives",
    "PersistenceResult",
    "PreparedVertexFiltration",
    "PrimeField",
    "RationalField",
    "RepresentativeKind",
    "TopologyReductionEvidence",
    "TopologyResourceError",
    "TopologyResourcePolicy",
    "cell_vertex_support",
    "compact_boundary",
    "compute_betti_dimensions",
    "compute_homology",
    "compute_persistence",
    "freeze_persistence_pairing",
    "lower_star_filtration",
    "upper_star_filtration",
]
