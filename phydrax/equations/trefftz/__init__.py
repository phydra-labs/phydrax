"""Finite exact-PDE trial spaces for boundary-only field fitting."""

from ..._holomorphic import (
    ComplexAffineNormalization,
    HolomorphicJet,
    HolomorphicMapCertificate,
    HolomorphicParameterCoverage,
    HolomorphicPotentialProvider,
)
from ._audit import audit_trial_space, trial_space_certificate
from ._complex_potential_2d import (
    BiharmonicPotential2D,
    HarmonicPotential2D,
    PlaneElasticityPotential2D,
    PlaneIsotropicMaterial,
)
from ._core import (
    AbstractTrefftzBasis,
    LinearTrefftzField,
    SimilarityNormalization,
    TrefftzResourceBudget,
    TrefftzResourceEvidence,
    TrialSpaceAuditReport,
    TrialSpaceCertificate,
    TrialValidityRegion,
)
from ._helmholtz import HelmholtzPlaneWaveBasis, sample_unit_directions
from ._holomorphic import HolomorphicPolynomialPotential
from ._holomorphic_composition import (
    HolomorphicBranchBundle,
    HolomorphicFactorGaugeReport,
    HolomorphicFactorizationEvidence,
    HolomorphicProductPotential,
)
from ._holomorphic_constraints import (
    ConstrainedHolomorphicPolynomialPotential,
    HolomorphicConstraintComponent,
    HolomorphicConstraintEvidence,
    HolomorphicPointConstraint,
    HolomorphicPolynomialConstraintPlan,
    PreparedHolomorphicPolynomialConstraints,
)
from ._monogenic import LinearMonogenicField, MonogenicPolynomialBasis
from ._polynomial import HarmonicPolynomialBasis, PolyharmonicAlmansiBasis


__all__ = [
    "AbstractTrefftzBasis",
    "audit_trial_space",
    "BiharmonicPotential2D",
    "ConstrainedHolomorphicPolynomialPotential",
    "ComplexAffineNormalization",
    "HarmonicPolynomialBasis",
    "HarmonicPotential2D",
    "HelmholtzPlaneWaveBasis",
    "HolomorphicBranchBundle",
    "HolomorphicConstraintComponent",
    "HolomorphicConstraintEvidence",
    "HolomorphicFactorGaugeReport",
    "HolomorphicFactorizationEvidence",
    "HolomorphicProductPotential",
    "HolomorphicPointConstraint",
    "HolomorphicPolynomialConstraintPlan",
    "HolomorphicPolynomialPotential",
    "HolomorphicJet",
    "HolomorphicMapCertificate",
    "HolomorphicParameterCoverage",
    "HolomorphicPotentialProvider",
    "LinearTrefftzField",
    "LinearMonogenicField",
    "MonogenicPolynomialBasis",
    "PolyharmonicAlmansiBasis",
    "PreparedHolomorphicPolynomialConstraints",
    "PlaneElasticityPotential2D",
    "PlaneIsotropicMaterial",
    "sample_unit_directions",
    "SimilarityNormalization",
    "TrefftzResourceBudget",
    "TrefftzResourceEvidence",
    "trial_space_certificate",
    "TrialSpaceAuditReport",
    "TrialSpaceCertificate",
    "TrialValidityRegion",
]
