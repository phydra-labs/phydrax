#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Composable positive-definite kernels shared across Phydrax subsystems."""

from ._algebra import AmplitudeKernel, ProductKernel, ScaleKernel, SumKernel
from ._base import AbstractPositiveDefiniteKernel, AbstractUnitDiagonalKernel
from ._combinatorial import HammingSpectralKernel, HypercubeSpectralKernel
from ._compact import (
    GrassmannSpectralKernel,
    SpecialOrthogonalCharacterKernel,
    SpecialUnitaryCharacterKernel,
    SphereSpectralKernel,
    StiefelSpectralKernel,
)
from ._finite_feature import (
    AbstractFiniteFeatureKernel,
    FiniteFeatureKernel,
    kernel_feature_rank,
    kernel_features,
)
from ._hodge import CochainHodgeSpectralKernel
from ._noncompact import (
    hyperbolic_feature_proposal,
    HyperbolicRandomFeatureKernel,
    ImportanceFeatureDiagnostics,
    NoncompactFeatureProposal,
    spd_feature_proposal,
    SPDRandomFeatureKernel,
)
from ._operator_valued import (
    AbstractOperatorValuedKernel,
    ProjectedDifferentialFormKernel,
    ProjectedTangentKernel,
    sphere_differential_form_kernel,
    sphere_tangent_kernel,
    sphere_tangent_projector,
)
from ._spectral import (
    AbstractSpectralMultiplier,
    HeatSpectralMultiplier,
    MaternSpectralMultiplier,
    SpectralFeatureKernel,
)
from ._stationary import (
    AbstractStationaryKernel,
    InverseMultiquadricKernel,
    Matern32Kernel,
    Matern52Kernel,
    SquaredExponentialKernel,
)
from ._transforms import AffineInputTransform, InputTransformedKernel


__all__ = [
    "AbstractPositiveDefiniteKernel",
    "AbstractFiniteFeatureKernel",
    "AbstractStationaryKernel",
    "AbstractSpectralMultiplier",
    "AbstractUnitDiagonalKernel",
    "AbstractOperatorValuedKernel",
    "AffineInputTransform",
    "AmplitudeKernel",
    "FiniteFeatureKernel",
    "CochainHodgeSpectralKernel",
    "kernel_feature_rank",
    "kernel_features",
    "GrassmannSpectralKernel",
    "HammingSpectralKernel",
    "HeatSpectralMultiplier",
    "hyperbolic_feature_proposal",
    "HyperbolicRandomFeatureKernel",
    "ImportanceFeatureDiagnostics",
    "InputTransformedKernel",
    "HypercubeSpectralKernel",
    "InverseMultiquadricKernel",
    "Matern32Kernel",
    "Matern52Kernel",
    "ProductKernel",
    "MaternSpectralMultiplier",
    "NoncompactFeatureProposal",
    "ProjectedDifferentialFormKernel",
    "ProjectedTangentKernel",
    "ScaleKernel",
    "SquaredExponentialKernel",
    "SpecialOrthogonalCharacterKernel",
    "SpecialUnitaryCharacterKernel",
    "SphereSpectralKernel",
    "StiefelSpectralKernel",
    "SumKernel",
    "sphere_differential_form_kernel",
    "sphere_tangent_kernel",
    "spd_feature_proposal",
    "SPDRandomFeatureKernel",
    "sphere_tangent_projector",
    "SpectralFeatureKernel",
]
