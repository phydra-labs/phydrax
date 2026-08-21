#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Target-aware relative-entropy weighting of finite measures."""

from ._problem import (
    ExactMoments,
    MomentCalibrationPolicy,
    MomentCalibrationProblem,
    MomentTarget,
    QuadraticMoments,
)
from ._relative_entropy import calibrate_moments, implicit_calibrate_moments
from ._results import (
    moment_calibration_status_message,
    MomentCalibrationDiagnostics,
    MomentCalibrationProvenance,
    MomentCalibrationResult,
    MomentCalibrationStatus,
    require_converged,
)


__all__ = [
    "ExactMoments",
    "MomentCalibrationDiagnostics",
    "MomentCalibrationPolicy",
    "MomentCalibrationProblem",
    "MomentCalibrationProvenance",
    "MomentCalibrationResult",
    "MomentCalibrationStatus",
    "MomentTarget",
    "QuadraticMoments",
    "calibrate_moments",
    "implicit_calibrate_moments",
    "moment_calibration_status_message",
    "require_converged",
]
