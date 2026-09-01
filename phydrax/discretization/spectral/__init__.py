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
    PreparedBoundaryLift,
    SpectralBoundaryConditionPlan,
    SpectralTraceConstraint,
    SpectralTraceTerm,
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
from ._diagnostics import (
    ModalDecayReport,
    PreparedSpectralModalDiagnostics,
    SpectralModalDiagnosticsPlan,
)
from ._eigen_verification import (
    compare_spectral_eigen_resolutions,
    SpectralEigenResolutionPolicy,
    SpectralEigenResolutionReport,
)
from ._fourier_shells import (
    DCPolicy,
    FinalEdgePolicy,
    FourierShellStatisticResult,
    ModeTransferCorrection,
    NyquistPolicy,
    PeriodicFourierField,
    PeriodicFourierShellPlan,
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
    spectral_hilbert_operator,
    spectral_laplacian_operator,
)
from ._precision import SpectralPrecisionPolicy
from ._rational import (
    RationalChebyshevHalfLineBasisPlan,
    RationalChebyshevLineBasisPlan,
)
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
from ._transfer import (
    prepare_spectral_modal_transfer,
    PreparedSpectralModalTransfer,
    SpectralModalTransferPlan,
    SpectralModalTransferReport,
)


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
    "DCPolicy",
    "FinalEdgePolicy",
    "FourierShellStatisticResult",
    "DealiasingKind",
    "BrillouinZonePlan",
    "DealiasingReport",
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
    "ModalDecayReport",
    "ModeTransferCorrection",
    "NyquistPolicy",
    "NoDealiasingPlan",
    "PaddingDealiasingPlan",
    "PolynomialClosureDealiasingPlan",
    "PeriodicLerayProjector",
    "PreparedBoundaryLift",
    "PreparedDealiasingPlan",
    "PreparedPseudospectralMethod",
    "PreparedSpectralModalDiagnostics",
    "PreparedSpectralModalTransfer",
    "PreparedSpectralAxis",
    "PreparedSpectralConservationDynamics",
    "PreparedSpectralConservationMethod",
    "PreparedChannelStokesSolver",
    "PreparedSpectralGalerkin",
    "PreparedSpectralOperator",
    "PreparedBrillouinZone",
    "PeriodicFourierField",
    "PeriodicFourierShellPlan",
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
    "RationalChebyshevHalfLineBasisPlan",
    "RationalChebyshevLineBasisPlan",
    "SpectralBasisFamily",
    "SpectralBoundaryConditionPlan",
    "SpectralBoundaryKind",
    "SpectralConservationDiagnostics",
    "SpectralConservationMethodPlan",
    "SpectralDifferentiabilityPolicy",
    "SpectralEntropyDiagnostics",
    "SpectralGalerkinMethodPlan",
    "SpectralModeLayout",
    "SpectralEigenResolutionPolicy",
    "SpectralEigenResolutionReport",
    "SpectralModalDiagnosticsPlan",
    "SpectralModalTransferPlan",
    "SpectralModalTransferReport",
    "SpectralPrecisionPolicy",
    "SpectralTraceConstraint",
    "SpectralTraceTerm",
    "project_tensor_spectral_symmetries",
    "compare_spectral_eigen_resolutions",
    "prepare_spectral_modal_transfer",
    "SpectralResidualDiagnostics",
    "TauSolveResult",
    "TensorSpectralDiscretization",
    "TensorSpectralPlan",
    "spectral_derivative_operator",
    "spectral_hilbert_operator",
    "spectral_laplacian_operator",
    "spherical_laplacian_operator",
]
