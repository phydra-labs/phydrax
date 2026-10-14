#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Composable positive-definite kernels shared across Phydrax subsystems."""

from typing import TYPE_CHECKING

from .._deferred_exports import deferred_exports


# Re-exports load on first use, so importing one owner never imports the
# whole package and its import cycles.
_DEFERRED_EXPORTS = {
    "AmplitudeKernel": "._algebra",
    "NormalizedKernel": "._algebra",
    "ProductKernel": "._algebra",
    "ScaleKernel": "._algebra",
    "SumKernel": "._algebra",
    "AbstractPositiveDefiniteKernel": "._base",
    "AbstractUnitDiagonalKernel": "._base",
    "HammingSpectralKernel": "._combinatorial",
    "HypercubeSpectralKernel": "._combinatorial",
    "GrassmannSpectralKernel": "._compact",
    "SpecialOrthogonalCharacterKernel": "._compact",
    "SpecialUnitaryCharacterKernel": "._compact",
    "SphereSpectralKernel": "._compact",
    "StiefelSpectralKernel": "._compact",
    "CompactHomogeneousHeatKernel": "._compact_homogeneous",
    "CompactHomogeneousMaternKernel": "._compact_homogeneous",
    "GeodesicDistanceEvidence": "._compact_homogeneous",
    "GeodesicExponentialKernel": "._compact_homogeneous",
    "GeodesicRadialKernel": "._compact_homogeneous",
    "KernelEvaluationEvidence": "._compact_homogeneous",
    "PreparedCompactHomogeneousSpectrum": "._compact_homogeneous",
    "kernel_functional_diagonal": "._field_metric",
    "kernel_functional_gram": "._field_metric",
    "kernel_functional_representer": "._field_metric",
    "KernelFunctional": "._field_metric",
    "KernelFunctionalExactness": "._field_metric",
    "KernelFunctionalTerm": "._field_metric",
    "KernelGram": "._field_metric",
    "KernelGramEvidence": "._field_metric",
    "KernelInputAdapter": "._field_metric",
    "KernelMetricMode": "._field_metric",
    "KernelSection": "._field_metric",
    "ProductFieldKernelMetric": "._field_metric",
    "AbstractFiniteFeatureKernel": "._finite_feature",
    "FiniteFeatureKernel": "._finite_feature",
    "kernel_feature_rank": "._finite_feature",
    "kernel_features": "._finite_feature",
    "HodgeSpectralKernel": "._hodge",
    "LinearKernel": "._linear",
    "hyperbolic_feature_proposal": "._noncompact",
    "HyperbolicRandomFeatureKernel": "._noncompact",
    "ImportanceFeatureDiagnostics": "._noncompact",
    "NoncompactFeatureProposal": "._noncompact",
    "spd_feature_proposal": "._noncompact",
    "SPDRandomFeatureKernel": "._noncompact",
    "AbstractOperatorValuedKernel": "._operator_valued",
    "Coregionalization": "._operator_valued",
    "IntrinsicCoregionalizationKernel": "._operator_valued",
    "LinearModelCoregionalizationKernel": "._operator_valued",
    "operator_kernel_feature_rank": "._operator_valued",
    "operator_kernel_features": "._operator_valued",
    "ProjectedDifferentialFormKernel": "._operator_valued",
    "ProjectedTangentKernel": "._operator_valued",
    "sphere_differential_form_kernel": "._operator_valued",
    "sphere_tangent_kernel": "._operator_valued",
    "sphere_tangent_projector": "._operator_valued",
    "ExactQuantumStateFidelityKernel": "._quantum",
    "SignaturePDEKernel": "._signature",
    "AbstractSpectralMultiplier": "._spectral",
    "HeatSpectralMultiplier": "._spectral",
    "MaternSpectralMultiplier": "._spectral",
    "SpectralFeatureKernel": "._spectral",
    "AbstractStationaryKernel": "._stationary",
    "InverseMultiquadricKernel": "._stationary",
    "Matern32Kernel": "._stationary",
    "Matern52Kernel": "._stationary",
    "SquaredExponentialKernel": "._stationary",
    "CARMAKernel": "._temporal",
    "SHOKernel": "._temporal",
    "AffineInputTransform": "._transforms",
    "InputTransformedKernel": "._transforms",
}


__getattr__, __dir__ = deferred_exports(__name__, globals(), _DEFERRED_EXPORTS)


if TYPE_CHECKING:
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
    from ._hodge import HodgeSpectralKernel
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
    "HodgeSpectralKernel",
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
