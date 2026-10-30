#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scalar isogeometric discretizations and compatible spline differential forms."""

from ..._interpolation import (
    ASTSCertificate,
    ASTSRefinement,
    ASTSTransferPlan,
    BSplineGrid,
    certify_asts,
    certify_tspline_geometry,
    LocalKnotVector2D,
    TAnchor2D,
    TCell2D,
    TEdge2D,
    TJunctionExtension2D,
    TMesh2D,
    TSplineGeometryCertificate,
    TSplineRationalJet,
    TVertex2D,
)
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
from ._cut import UVTrimLoop, UVTrimmedSurface
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
from ._tspline import (
    PreparedTSplineIsogeometricDiscretization,
    TSplineGeometryState,
    TSplineH1FieldSpec,
    TSplineIsogeometricPlan,
    TSplineIsogeometricRefinement,
    TSplineQueryRoute,
    TSplineRuntimeData,
)


__all__ = [
    "AssembledSplineDeRhamComplex",
    "BoundarySide",
    "BSplineGrid",
    "ASTSCertificate",
    "ASTSRefinement",
    "ASTSTransferPlan",
    "CompatibleQualificationEvidence",
    "CompatibleQualificationPolicy",
    "IsogeometricFieldReconstructionKernel",
    "IsogeometricGeometryEvidence",
    "IsogeometricH1QualificationPolicy",
    "IsogeometricPlan",
    "IsogeometricQuadraturePolicy",
    "IsogeometricRuntimeData",
    "NURBSGeometryState",
    "UVTrimLoop",
    "UVTrimmedSurface",
    "LocalKnotVector2D",
    "PreparedIsogeometricDiscretization",
    "PreparedTSplineIsogeometricDiscretization",
    "RelativeCohomologyEvidence",
    "SplineDeRhamComplex",
    "SplineDifferentialSpace",
    "TAnchor2D",
    "TCell2D",
    "TEdge2D",
    "TMesh2D",
    "TSplineGeometryCertificate",
    "TSplineGeometryState",
    "TSplineH1FieldSpec",
    "TSplineIsogeometricPlan",
    "TSplineIsogeometricRefinement",
    "TSplineRationalJet",
    "TSplineQueryRoute",
    "TSplineRuntimeData",
    "TVertex2D",
    "TJunctionExtension2D",
    "SplineFormComponent",
    "qualify_compatible_complex",
    "certify_asts",
    "certify_tspline_geometry",
    "prepare_isogeometric_field_reconstruction",
]
