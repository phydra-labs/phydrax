#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact finite-complex topology and evidence-bearing persistence analysis."""

from ._advanced import (
    alpha_complex,
    cech_complex,
    CellDiagonalApproximation,
    CellularSheaf,
    compute_multiparameter_persistence,
    compute_spectral_sequence,
    compute_zigzag_intervals,
    cup_product,
    FilteredBicomplex,
    FilteredChainComplex,
    FinitePersistenceModule,
    MultiFiltration,
    MultiparameterPersistenceResult,
    PointCloudComplexPolicy,
    PointCloudComplexResult,
    SpectralSequenceResult,
    vietoris_rips_complex,
    ZigzagIntervalResult,
)
from ._coefficients import CoefficientDomain, PrimeField, RationalField
from ._complex import (
    CellComplexPair,
    CellSubcomplex,
    CellVertexSupport,
    compact_boundary,
    CompactBoundary,
    CompactCellLayout,
)
from ._cone import compute_mapping_cone_homology, mapping_cone, MappingConeResult
from ._cubical import compute_structured_cubical_persistence, CubicalPersistenceResult
from ._diagram import PackedPersistenceDiagram, PersistenceDiagram
from ._diagram_distance import (
    diagram_bottleneck_distance,
    diagram_sliced_wasserstein_distance,
    diagram_wasserstein_distance,
    DiagramDistanceResult,
)
from ._extended_persistence import (
    compute_extended_persistence,
    ExtendedPersistenceComponent,
    ExtendedPersistenceResult,
)
from ._features import (
    betti_curve,
    frozen_total_persistence,
    persistence_image,
    PersistenceFeatureEvidence,
    PersistenceFeaturePolicy,
    total_persistence,
)
from ._fem_transfer import finite_element_topology_transfer
from ._field import FieldTopologyPlan, FieldTopologySeries, FieldTopologySnapshot
from ._filtration import (
    cell_vertex_support,
    CellFiltration,
    FiltrationDirection,
    lower_star_filtration,
    PreparedVertexFiltration,
    upper_star_filtration,
)
from ._geometric import compute_cell_local_homology, LocalHomologyReport, MergeTree
from ._homology import (
    BettiDimensionResult,
    compute_betti_dimensions,
    compute_homology,
    FiniteFieldBasis,
    HomologyDegreeResult,
    HomologyResult,
    RepresentativeKind,
)
from ._induced import (
    compute_induced_topology_map,
    FiniteFieldCoordinateMap,
    induced_homology_coordinates,
    InducedTopologyMap,
)
from ._integer import block_matrix, ExactChainComplex, ExactIntegerCOO
from ._integral import (
    compute_integral_homology,
    IntegralHomologyDegree,
    IntegralHomologyResult,
)
from ._maps import (
    CellularChainContraction,
    CellularChainMap,
    CellularPairMap,
    chain_coordinate_id,
    FilteredCellularChainContraction,
    FilteredCellularChainMap,
)
from ._morse import cancel_unit_pair, MorseReductionResult
from ._optimized_persistence import compute_h0_persistence_union_find
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
from ._persistent_cohomology import (
    compute_persistent_cohomology,
    PersistentCohomologyResult,
    TerminalCocycleAnnotation,
)
from ._rational_basis import (
    compute_rational_homology_basis,
    RationalClassBasis,
    RationalHomologyBasisResult,
)
from ._resources import (
    TopologyReductionEvidence,
    TopologyResourceError,
    TopologyResourcePolicy,
)
from ._temporal import (
    compute_monotone_zigzag_intervals,
    compute_vineyard,
    compute_zigzag_topology,
    MonotoneZigzagIntervals,
    VineyardResult,
    ZigzagCellOperation,
    ZigzagTopologyResult,
)


__all__ = [
    "CellDiagonalApproximation",
    "CellularSheaf",
    "FilteredBicomplex",
    "FilteredChainComplex",
    "FinitePersistenceModule",
    "MultiFiltration",
    "MultiparameterPersistenceResult",
    "PointCloudComplexPolicy",
    "PointCloudComplexResult",
    "SpectralSequenceResult",
    "ZigzagIntervalResult",
    "alpha_complex",
    "cech_complex",
    "compute_multiparameter_persistence",
    "compute_spectral_sequence",
    "compute_zigzag_intervals",
    "cup_product",
    "vietoris_rips_complex",
    "BettiDimensionResult",
    "CellComplexPair",
    "CellFiltration",
    "CellSubcomplex",
    "CellVertexSupport",
    "CellularChainContraction",
    "CellularChainMap",
    "CellularPairMap",
    "CoefficientDomain",
    "CompactBoundary",
    "CompactCellLayout",
    "CubicalPersistenceResult",
    "DiagramDistanceResult",
    "ExactChainComplex",
    "ExactIntegerCOO",
    "ExtendedPersistenceComponent",
    "ExtendedPersistenceResult",
    "FieldTopologyPlan",
    "FieldTopologySeries",
    "FieldTopologySnapshot",
    "FilteredCellularChainContraction",
    "FilteredCellularChainMap",
    "FiltrationDirection",
    "FiniteFieldBasis",
    "FiniteFieldCoordinateMap",
    "FrozenPersistenceEvaluation",
    "FrozenPersistencePairing",
    "HomologyDegreeResult",
    "HomologyResult",
    "InducedTopologyMap",
    "IntegralHomologyDegree",
    "IntegralHomologyResult",
    "LocalHomologyReport",
    "MappingConeResult",
    "MonotoneZigzagIntervals",
    "MergeTree",
    "MorseReductionResult",
    "PackedPersistenceDiagram",
    "PersistenceDiagram",
    "PersistenceFeatureEvidence",
    "PersistenceFeaturePolicy",
    "PersistencePairing",
    "PersistenceRepresentativeKind",
    "PersistenceRepresentatives",
    "PersistenceResult",
    "PersistentCohomologyResult",
    "TerminalCocycleAnnotation",
    "PreparedVertexFiltration",
    "PrimeField",
    "RationalClassBasis",
    "RationalField",
    "RationalHomologyBasisResult",
    "RepresentativeKind",
    "TopologyReductionEvidence",
    "TopologyResourceError",
    "TopologyResourcePolicy",
    "VineyardResult",
    "ZigzagCellOperation",
    "ZigzagTopologyResult",
    "betti_curve",
    "block_matrix",
    "cancel_unit_pair",
    "cell_vertex_support",
    "chain_coordinate_id",
    "compact_boundary",
    "compute_betti_dimensions",
    "compute_cell_local_homology",
    "compute_extended_persistence",
    "compute_h0_persistence_union_find",
    "compute_homology",
    "compute_induced_topology_map",
    "compute_integral_homology",
    "compute_mapping_cone_homology",
    "compute_persistence",
    "compute_persistent_cohomology",
    "compute_rational_homology_basis",
    "compute_structured_cubical_persistence",
    "compute_vineyard",
    "compute_monotone_zigzag_intervals",
    "compute_zigzag_topology",
    "diagram_bottleneck_distance",
    "diagram_sliced_wasserstein_distance",
    "diagram_wasserstein_distance",
    "finite_element_topology_transfer",
    "freeze_persistence_pairing",
    "frozen_total_persistence",
    "induced_homology_coordinates",
    "lower_star_filtration",
    "mapping_cone",
    "persistence_image",
    "total_persistence",
    "upper_star_filtration",
]
