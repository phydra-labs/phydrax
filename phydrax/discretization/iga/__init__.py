#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scalar isogeometric discretizations and compatible spline differential forms."""

from ..._interpolation import BSplineGrid
from ._basis import IsogeometricQuadraturePolicy
from ._compatible import (
    AssembledSplineDeRhamComplex,
    BoundarySide,
    CompatibleQualificationEvidence,
    CompatibleQualificationPolicy,
    qualify_compatible_complex,
    RelativeCohomologyEvidence,
    SplineDeRhamComplex,
    SplineDifferentialSpace,
    SplineFormComponent,
)
from ._field_view import (
    IsogeometricFieldReconstructionKernel,
    prepare_isogeometric_field_reconstruction,
)
from ._geometry import (
    IsogeometricGeometryEvidence,
    IsogeometricH1QualificationPolicy,
    IsogeometricRuntimeData,
    NURBSGeometryState,
)
from ._plan import IsogeometricPlan, PreparedIsogeometricDiscretization


__all__ = [
    "AssembledSplineDeRhamComplex",
    "BoundarySide",
    "BSplineGrid",
    "CompatibleQualificationEvidence",
    "CompatibleQualificationPolicy",
    "IsogeometricFieldReconstructionKernel",
    "IsogeometricGeometryEvidence",
    "IsogeometricH1QualificationPolicy",
    "IsogeometricPlan",
    "IsogeometricQuadraturePolicy",
    "IsogeometricRuntimeData",
    "NURBSGeometryState",
    "PreparedIsogeometricDiscretization",
    "RelativeCohomologyEvidence",
    "SplineDeRhamComplex",
    "SplineDifferentialSpace",
    "SplineFormComponent",
    "qualify_compatible_complex",
    "prepare_isogeometric_field_reconstruction",
]
