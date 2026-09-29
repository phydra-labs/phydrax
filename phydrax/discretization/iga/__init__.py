#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Single-patch S1 isogeometric discretization."""

from ..._interpolation import BSplineGrid
from ._basis import IsogeometricQuadraturePolicy
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
    "BSplineGrid",
    "IsogeometricFieldReconstructionKernel",
    "IsogeometricGeometryEvidence",
    "IsogeometricH1QualificationPolicy",
    "IsogeometricPlan",
    "IsogeometricQuadraturePolicy",
    "IsogeometricRuntimeData",
    "NURBSGeometryState",
    "PreparedIsogeometricDiscretization",
    "prepare_isogeometric_field_reconstruction",
]
