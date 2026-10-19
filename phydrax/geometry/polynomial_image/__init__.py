#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Discover numerical relations and prove polynomial-image containment."""

from ._analysis import (
    PolynomialImageAnalysisPlan,
    PolynomialImageAnalysisPolicy,
    prepare_polynomial_image_analysis,
    PreparedPolynomialImageAnalysis,
    refresh_polynomial_image_analysis,
)
from ._contracts import (
    EvidenceDisposition,
    ExactCompositionRemainder,
    ExactPolynomialContainmentResult,
    JacobianRankEvidence,
    PolynomialImageAnalysisResult,
    PolynomialImageAnalysisStatus,
    PolynomialImageClaimEvidence,
    PolynomialImageResourceEvidence,
    SourceSampleKind,
    TargetMonomialSupport,
    TargetRelationEvidence,
)
from ._exact import prove_exact_containment
from ._map import SparsePolynomialMap


__all__ = [
    "EvidenceDisposition",
    "ExactCompositionRemainder",
    "ExactPolynomialContainmentResult",
    "JacobianRankEvidence",
    "PolynomialImageAnalysisPlan",
    "PolynomialImageAnalysisPolicy",
    "PolynomialImageAnalysisResult",
    "PolynomialImageAnalysisStatus",
    "PolynomialImageClaimEvidence",
    "PolynomialImageResourceEvidence",
    "PreparedPolynomialImageAnalysis",
    "SourceSampleKind",
    "SparsePolynomialMap",
    "TargetMonomialSupport",
    "TargetRelationEvidence",
    "prepare_polynomial_image_analysis",
    "prove_exact_containment",
    "refresh_polynomial_image_analysis",
]
