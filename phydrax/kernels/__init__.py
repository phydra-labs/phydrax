#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Composable positive-definite kernels shared across Phydrax subsystems."""

from ._algebra import (
    AmplitudeKernel,
    NormalizedKernel,
    ProductKernel,
    ScaleKernel,
    SumKernel,
)
from ._base import AbstractPositiveDefiniteKernel, AbstractUnitDiagonalKernel
from ._combinatorial import HammingSpectralKernel, HypercubeSpectralKernel
from ._compact import (
    GrassmannSpectralKernel,
    SpecialOrthogonalCharacterKernel,
    SpecialUnitaryCharacterKernel,
    SphereSpectralKernel,
    StiefelSpectralKernel,
)
from ._compact_homogeneous import (
    CompactHomogeneousHeatKernel,
    CompactHomogeneousMaternKernel,
    GeodesicDistanceEvidence,
    GeodesicExponentialKernel,
    GeodesicRadialKernel,
    KernelEvaluationEvidence,
    PreparedCompactHomogeneousSpectrum,
)
from ._field_metric import (
    kernel_functional_diagonal,
    kernel_functional_gram,
    kernel_functional_representer,
    KernelFunctional,
    KernelFunctionalExactness,
    KernelFunctionalTerm,
    KernelGram,
    KernelGramEvidence,
    KernelInputAdapter,
    KernelMetricMode,
    KernelSection,
    ProductFieldKernelMetric,
)
from ._finite_feature import (
    AbstractFiniteFeatureKernel,
    FiniteFeatureKernel,
    kernel_feature_rank,
    kernel_features,
)
from ._hodge import CochainHodgeSpectralKernel
from ._linear import LinearKernel
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
    Coregionalization,
    IntrinsicCoregionalizationKernel,
    LinearModelCoregionalizationKernel,
    operator_kernel_feature_rank,
    operator_kernel_features,
    ProjectedDifferentialFormKernel,
    ProjectedTangentKernel,
    sphere_differential_form_kernel,
    sphere_tangent_kernel,
    sphere_tangent_projector,
)
from ._quantum import ExactQuantumStateFidelityKernel
from ._signature import SignaturePDEKernel
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
from ._temporal import CARMAKernel, SHOKernel
from ._transforms import AffineInputTransform, InputTransformedKernel


__all__ = [
    "CompactHomogeneousHeatKernel",
    "CompactHomogeneousMaternKernel",
    "GeodesicDistanceEvidence",
    "GeodesicExponentialKernel",
    "GeodesicRadialKernel",
    "KernelEvaluationEvidence",
    "PreparedCompactHomogeneousSpectrum",
    "AbstractPositiveDefiniteKernel",
    "AbstractFiniteFeatureKernel",
    "AbstractStationaryKernel",
    "AbstractSpectralMultiplier",
    "AbstractUnitDiagonalKernel",
    "AbstractOperatorValuedKernel",
    "AffineInputTransform",
    "AmplitudeKernel",
    "CARMAKernel",
    "ExactQuantumStateFidelityKernel",
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
    "LinearKernel",
    "NormalizedKernel",
    "InverseMultiquadricKernel",
    "Matern32Kernel",
    "SHOKernel",
    "SignaturePDEKernel",
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
    "Coregionalization",
    "IntrinsicCoregionalizationKernel",
    "KernelFunctional",
    "KernelFunctionalExactness",
    "KernelFunctionalTerm",
    "KernelGram",
    "KernelGramEvidence",
    "KernelInputAdapter",
    "KernelMetricMode",
    "KernelSection",
    "LinearModelCoregionalizationKernel",
    "ProductFieldKernelMetric",
    "kernel_functional_diagonal",
    "kernel_functional_gram",
    "kernel_functional_representer",
    "operator_kernel_feature_rank",
    "operator_kernel_features",
]
