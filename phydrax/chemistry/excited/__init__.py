#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Electronic excited-state manifolds, response, tracking, and couplings."""

from ._correlated import (
    AbstractCorrelatedManifoldProvider,
    biorthogonal_manifold,
    CallableCorrelatedManifoldProvider,
    casci_manifold,
    CorrelatedManifoldPlan,
)
from ._crossing import (
    AbstractTwoStateSurfaceProvider,
    branching_plane,
    BranchingPlaneResult,
    CallableTwoStateSurfaceProvider,
    CrossingKind,
    CrossingOptimizationResult,
    MinimumEnergyCrossingPlan,
    TwoStateSurfaceEvaluation,
)
from ._derivatives import (
    AbstractExcitedDerivativeProvider,
    CallableExcitedDerivativeProvider,
    ExcitedStateDerivativeResult,
    rpa_derivative_couplings,
    rpa_eigenvalue_derivatives,
    tda_derivative_couplings,
    tda_eigenvalue_derivatives,
    tda_property_derivatives,
    TDAPropertyDerivativeResult,
)
from ._dynamics import (
    AbstractNonadiabaticSurfaceProvider,
    CallableNonadiabaticSurfaceProvider,
    DecoherenceKind,
    FewestSwitchesSurfaceHoppingPlan,
    FrustratedHopPolicy,
    NonadiabaticSurfaceEvaluation,
    SurfaceHopEvent,
    SurfaceHoppingState,
    SurfaceHoppingStep,
)
from ._hf_response import HartreeFockExcitedResponsePlan
from ._ks_response import KohnShamExcitedResponsePlan
from ._manifold import ElectronicManifoldResult
from ._representation import (
    BiorthogonalStateRepresentation,
    CIStateRepresentation,
    ExcitedStateRepresentation,
    RPAStateRepresentation,
    TDAStateRepresentation,
)
from ._rpa import RandomPhaseApproximationPlan
from ._tda import (
    ExcitedStateManifoldPlan,
    finite_difference_nonadiabatic_coupling,
    NonadiabaticCouplingResult,
    TammDancoffPlan,
)
from ._tracking import StateTrackingResult, track_excited_states


__all__ = [
    "AbstractCorrelatedManifoldProvider",
    "AbstractExcitedDerivativeProvider",
    "AbstractNonadiabaticSurfaceProvider",
    "AbstractTwoStateSurfaceProvider",
    "BiorthogonalStateRepresentation",
    "BranchingPlaneResult",
    "CIStateRepresentation",
    "CallableCorrelatedManifoldProvider",
    "CallableExcitedDerivativeProvider",
    "CallableNonadiabaticSurfaceProvider",
    "CallableTwoStateSurfaceProvider",
    "CorrelatedManifoldPlan",
    "CrossingKind",
    "CrossingOptimizationResult",
    "DecoherenceKind",
    "ElectronicManifoldResult",
    "ExcitedStateDerivativeResult",
    "ExcitedStateManifoldPlan",
    "ExcitedStateRepresentation",
    "FewestSwitchesSurfaceHoppingPlan",
    "FrustratedHopPolicy",
    "HartreeFockExcitedResponsePlan",
    "MinimumEnergyCrossingPlan",
    "KohnShamExcitedResponsePlan",
    "NonadiabaticCouplingResult",
    "NonadiabaticSurfaceEvaluation",
    "RPAStateRepresentation",
    "RandomPhaseApproximationPlan",
    "StateTrackingResult",
    "SurfaceHopEvent",
    "SurfaceHoppingState",
    "SurfaceHoppingStep",
    "TDAStateRepresentation",
    "TDAPropertyDerivativeResult",
    "TammDancoffPlan",
    "TwoStateSurfaceEvaluation",
    "biorthogonal_manifold",
    "branching_plane",
    "casci_manifold",
    "finite_difference_nonadiabatic_coupling",
    "rpa_derivative_couplings",
    "rpa_eigenvalue_derivatives",
    "tda_derivative_couplings",
    "tda_eigenvalue_derivatives",
    "tda_property_derivatives",
    "track_excited_states",
]
