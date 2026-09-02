#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact finite-dimensional real algebras and prepared coordinate products."""

from ._core import AbstractFiniteRealAlgebraSpec, FiniteRealAlgebraSpec
from ._derivations import (
    AlgebraDerivationConstraint,
    AlgebraSymmetryBudget,
    AlgebraSymmetryResourceEvidence,
)
from ._families import (
    CayleyDicksonAlgebraSpec,
    ComplexAlgebraSpec,
    MulticomplexAlgebraSpec,
    OctonionAlgebraSpec,
    QuaternionAlgebraSpec,
    RealAlgebraSpec,
)
from ._geometry import (
    unit_algebra_state_geometry,
    UnitComplexStateGeometry,
    UnitQuaternionStateGeometry,
)
from ._layout import AlgebraElementLayout
from ._nonassociative import (
    algebra_left_solve,
    algebra_right_solve,
    AlgebraMatrixLayout,
    AlgebraMatrixProductPlan,
    AlgebraOperatorInverse,
    AlgebraRegularSpectrum,
    BracketingPlan,
    G2GroupOperations,
    G2LocalLogResult,
    G2MatrixElement,
    MoufangLoopOperations,
    PreparedUnitOctonionEvolution,
    UnitOctonionStateGeometry,
)
from ._product import AlgebraProductBackend, AlgebraProductEvidence, AlgebraProductPlan
from ._properties import (
    AlgebraClaimEvidence,
    AlgebraClaimSource,
    AlgebraClaimStatus,
    AlgebraPropertyEvidence,
    audit_algebra_properties,
)
from ._provider import FiniteRealAlgebraProvider
from ._resources import AlgebraResourceBudget, AlgebraResourceEvidence
from ._structure import (
    AlgebraRationalMap,
    AlgebraRationalVector,
    AlgebraStructureTable,
)


__all__ = [
    "AlgebraMatrixLayout",
    "AlgebraMatrixProductPlan",
    "AlgebraOperatorInverse",
    "AlgebraRegularSpectrum",
    "BracketingPlan",
    "G2GroupOperations",
    "G2LocalLogResult",
    "G2MatrixElement",
    "MoufangLoopOperations",
    "PreparedUnitOctonionEvolution",
    "UnitOctonionStateGeometry",
    "algebra_left_solve",
    "algebra_right_solve",
    "AbstractFiniteRealAlgebraSpec",
    "AlgebraClaimEvidence",
    "FiniteRealAlgebraProvider",
    "AlgebraClaimSource",
    "AlgebraClaimStatus",
    "AlgebraDerivationConstraint",
    "AlgebraElementLayout",
    "AlgebraProductBackend",
    "AlgebraProductEvidence",
    "AlgebraProductPlan",
    "AlgebraPropertyEvidence",
    "AlgebraRationalMap",
    "AlgebraRationalVector",
    "AlgebraResourceBudget",
    "AlgebraResourceEvidence",
    "AlgebraSymmetryBudget",
    "AlgebraSymmetryResourceEvidence",
    "AlgebraStructureTable",
    "CayleyDicksonAlgebraSpec",
    "ComplexAlgebraSpec",
    "FiniteRealAlgebraSpec",
    "MulticomplexAlgebraSpec",
    "OctonionAlgebraSpec",
    "QuaternionAlgebraSpec",
    "RealAlgebraSpec",
    "audit_algebra_properties",
    "UnitComplexStateGeometry",
    "UnitQuaternionStateGeometry",
    "unit_algebra_state_geometry",
]
