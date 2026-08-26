#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Global tensor spectral spaces, operators, and pseudospectral methods."""

from ._basis import (
    AbstractSpectralBasisPlan,
    ChebyshevBasisPlan,
    CosineBasisPlan,
    FourierBasisPlan,
    LegendreBasisPlan,
    PreparedSpectralAxis,
    SineBasisPlan,
    SpectralBasisFamily,
    SpectralBoundaryKind,
    SpectralModeLayout,
)
from ._conservation import (
    PreparedSpectralConservationDynamics,
    PreparedSpectralConservationMethod,
    SpectralConservationDiagnostics,
    SpectralConservationMethodPlan,
    SpectralEntropyDiagnostics,
)
from ._constraints import (
    BoundaryLiftPlan,
    ConstrainedBasisPlan,
    EndpointConstraint,
    PreparedBoundaryLift,
    SpectralBoundaryConditionPlan,
)
from ._dealias import (
    AbstractDealiasingPlan,
    DealiasingKind,
    DealiasingReport,
    ModalFilterPlan,
    NoDealiasingPlan,
    PaddingDealiasingPlan,
    PreparedDealiasingPlan,
)
from ._galerkin import PreparedSpectralGalerkin, SpectralGalerkinMethodPlan
from ._method import (
    PreparedPseudospectralMethod,
    PseudospectralMethodPlan,
    SpectralDifferentiabilityPolicy,
    SpectralResidualDiagnostics,
)
from ._operators import (
    PreparedSpectralOperator,
    spectral_derivative_operator,
    spectral_laplacian_operator,
)
from ._precision import SpectralPrecisionPolicy
from ._space import TensorSpectralDiscretization, TensorSpectralPlan
from ._tau import GeneralizedTauPlan, PreparedTauSystem, TauSolveResult


__all__ = [
    "AbstractDealiasingPlan",
    "AbstractSpectralBasisPlan",
    "BoundaryLiftPlan",
    "ChebyshevBasisPlan",
    "ConstrainedBasisPlan",
    "CosineBasisPlan",
    "DealiasingKind",
    "DealiasingReport",
    "EndpointConstraint",
    "FourierBasisPlan",
    "GeneralizedTauPlan",
    "LegendreBasisPlan",
    "ModalFilterPlan",
    "NoDealiasingPlan",
    "PaddingDealiasingPlan",
    "PreparedBoundaryLift",
    "PreparedDealiasingPlan",
    "PreparedPseudospectralMethod",
    "PreparedSpectralAxis",
    "PreparedSpectralConservationDynamics",
    "PreparedSpectralConservationMethod",
    "PreparedSpectralGalerkin",
    "PreparedSpectralOperator",
    "PreparedTauSystem",
    "PseudospectralMethodPlan",
    "SineBasisPlan",
    "SpectralBasisFamily",
    "SpectralBoundaryConditionPlan",
    "SpectralBoundaryKind",
    "SpectralConservationDiagnostics",
    "SpectralConservationMethodPlan",
    "SpectralDifferentiabilityPolicy",
    "SpectralEntropyDiagnostics",
    "SpectralGalerkinMethodPlan",
    "SpectralModeLayout",
    "SpectralPrecisionPolicy",
    "SpectralResidualDiagnostics",
    "TauSolveResult",
    "TensorSpectralDiscretization",
    "TensorSpectralPlan",
    "spectral_derivative_operator",
    "spectral_laplacian_operator",
]
