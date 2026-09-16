#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Research-only time-activity and internal-dosimetry contracts.

This facade is intentionally separate from :mod:`phydrax.nuclear`'s eager imports:
medical-grid dosimetry depends on imaging, while the core nuclear data package is
also consumed during equation-package initialization.
"""

from ._internal_dosimetry import (
    InternalDosimetryEvidence,
    PreparedRegionalSValuePlan,
    PreparedSpatialSValueConvolution,
    RegionalDoseResult,
    RegionalSValueEvaluation,
    RegionalSValuePlan,
    RegionalSValueTable,
    S_VALUE_UNIT,
    SpatialDoseResult,
    SpatialSValueConvolutionPlan,
    SpatialSValueEvaluation,
    SpatialSValueKernel,
)
from ._time_activity import (
    PreparedTimeActivityIntegration,
    TimeActivityIntegralEvaluation,
    TimeActivityIntegrationEvidence,
    TimeActivityIntegrationPlan,
    TimeActivityIntegrationResult,
    TimeActivitySeries,
)


__all__ = [
    "InternalDosimetryEvidence",
    "PreparedRegionalSValuePlan",
    "PreparedSpatialSValueConvolution",
    "PreparedTimeActivityIntegration",
    "RegionalDoseResult",
    "RegionalSValueEvaluation",
    "RegionalSValuePlan",
    "RegionalSValueTable",
    "S_VALUE_UNIT",
    "SpatialDoseResult",
    "SpatialSValueConvolutionPlan",
    "SpatialSValueEvaluation",
    "SpatialSValueKernel",
    "TimeActivityIntegralEvaluation",
    "TimeActivityIntegrationEvidence",
    "TimeActivityIntegrationPlan",
    "TimeActivityIntegrationResult",
    "TimeActivitySeries",
]
