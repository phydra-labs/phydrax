#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Global tensor spectral spaces, operators, and pseudospectral methods."""

from ..._spectral._spherical import (
    SphericalExecution,
    SphericalHarmonicPlan,
    SphericalSampling,
)
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
from ._channel import (
    ChannelMeanConstraint,
    ChannelMeanConstraintKind,
    ChannelStokesDiagnostics,
    ChannelStokesPlan,
    ChannelStokesSolveResult,
    PreparedChannelStokesSolver,
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
from ._coordinates import HermitianSpectralCoordinates
from ._dealias import (
    AbstractDealiasingPlan,
    DealiasingKind,
    DealiasingReport,
    ModalFilterPlan,
    NoDealiasingPlan,
    PaddingDealiasingPlan,
    PolynomialClosureDealiasingPlan,
    PreparedDealiasingPlan,
)
from ._galerkin import PreparedSpectralGalerkin, SpectralGalerkinMethodPlan
from ._incompressible import (
    IncompressibleSpectralDiagnostics,
    PeriodicLerayProjector,
)
from ._lattice import (
    BrillouinZonePlan,
    HarmonicTruncationKind,
    LatticeHarmonicDiscretization,
    LatticeHarmonicLayout,
    LatticeHarmonicPlan,
    PreparedBrillouinZone,
)
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
from ._spherical import (
    spherical_laplacian_operator,
    SphericalSpectralDiscretization,
    SphericalSpectralPlan,
)
from ._spherical_layout import SphericalModeLayout
from ._symmetry import (
    project_tensor_spectral_symmetries,
    TensorSpectralSymmetry,
)
from ._tau import GeneralizedTauPlan, PreparedTauSystem, TauSolveResult


__all__ = [
    "AbstractDealiasingPlan",
    "AbstractSpectralBasisPlan",
    "BoundaryLiftPlan",
    "ChebyshevBasisPlan",
    "ConstrainedBasisPlan",
    "CosineBasisPlan",
    "ChannelMeanConstraint",
    "ChannelMeanConstraintKind",
    "ChannelStokesDiagnostics",
    "ChannelStokesPlan",
    "ChannelStokesSolveResult",
    "DealiasingKind",
    "BrillouinZonePlan",
    "DealiasingReport",
    "EndpointConstraint",
    "FourierBasisPlan",
    "HermitianSpectralCoordinates",
    "IncompressibleSpectralDiagnostics",
    "GeneralizedTauPlan",
    "HarmonicTruncationKind",
    "LegendreBasisPlan",
    "LatticeHarmonicDiscretization",
    "LatticeHarmonicLayout",
    "LatticeHarmonicPlan",
    "ModalFilterPlan",
    "NoDealiasingPlan",
    "PaddingDealiasingPlan",
    "PolynomialClosureDealiasingPlan",
    "PeriodicLerayProjector",
    "PreparedBoundaryLift",
    "PreparedDealiasingPlan",
    "PreparedPseudospectralMethod",
    "PreparedSpectralAxis",
    "PreparedSpectralConservationDynamics",
    "PreparedSpectralConservationMethod",
    "PreparedChannelStokesSolver",
    "PreparedSpectralGalerkin",
    "PreparedSpectralOperator",
    "PreparedBrillouinZone",
    "PreparedTauSystem",
    "TensorSpectralSymmetry",
    "PseudospectralMethodPlan",
    "SineBasisPlan",
    "SphericalExecution",
    "SphericalHarmonicPlan",
    "SphericalModeLayout",
    "SphericalSampling",
    "SphericalSpectralDiscretization",
    "SphericalSpectralPlan",
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
    "project_tensor_spectral_symmetries",
    "SpectralResidualDiagnostics",
    "TauSolveResult",
    "TensorSpectralDiscretization",
    "TensorSpectralPlan",
    "spectral_derivative_operator",
    "spectral_laplacian_operator",
    "spherical_laplacian_operator",
]
