#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Single-patch S1 isogeometric discretization."""

from ..._interpolation import BSplineGrid
from ._basis import IsogeometricQuadraturePolicy
from ._geometry import (
    IsogeometricGeometryEvidence,
    IsogeometricH1QualificationPolicy,
    IsogeometricRuntimeData,
    NURBSGeometryState,
)
from ._plan import IsogeometricPlan, PreparedIsogeometricDiscretization


__all__ = [
    "BSplineGrid",
    "IsogeometricGeometryEvidence",
    "IsogeometricH1QualificationPolicy",
    "IsogeometricPlan",
    "IsogeometricQuadraturePolicy",
    "IsogeometricRuntimeData",
    "NURBSGeometryState",
    "PreparedIsogeometricDiscretization",
]
