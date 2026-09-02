#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Target-aware relative-entropy weighting of finite measures."""

from ._canonical import (
    BoundaryFaceEvidence,
    calibrate_moments_conic,
    discover_boundary_face,
    implicit_calibrate_fixed_face,
)
from ._problem import (
    BoundaryFacePolicy,
    EqualWeightSubset,
    ExactMoments,
    GroupMassConstraints,
    IntervalMoments,
    MomentCalibrationExecutionPolicy,
    MomentCalibrationPolicy,
    MomentCalibrationProblem,
    MomentTarget,
    QuadraticMoments,
    stratified_group_constraints,
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
from ._subset import calibrate_moments_subset


__all__ = [
    "BoundaryFaceEvidence",
    "BoundaryFacePolicy",
    "EqualWeightSubset",
    "ExactMoments",
    "GroupMassConstraints",
    "IntervalMoments",
    "MomentCalibrationDiagnostics",
    "MomentCalibrationExecutionPolicy",
    "MomentCalibrationPolicy",
    "MomentCalibrationProblem",
    "MomentCalibrationProvenance",
    "MomentCalibrationResult",
    "MomentCalibrationStatus",
    "MomentTarget",
    "QuadraticMoments",
    "calibrate_moments_conic",
    "calibrate_moments_subset",
    "calibrate_moments",
    "implicit_calibrate_fixed_face",
    "implicit_calibrate_moments",
    "discover_boundary_face",
    "moment_calibration_status_message",
    "require_converged",
    "stratified_group_constraints",
]
