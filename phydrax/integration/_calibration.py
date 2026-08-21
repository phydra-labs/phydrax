#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
from jaxtyping import Array

from .._strict import StrictModule
from ..optim import AbstractScalarIterativeMethod, OptimizationTermination
from ..weighting import (
    calibrate_moments,
    ExactMoments,
    MomentCalibrationPolicy,
    MomentCalibrationProblem,
    MomentCalibrationResult,
    QuadraticMoments,
    require_converged,
)
from ._measure_transform import (
    feature_matrix,
    lower_finite_measure,
    transformed_weighted_realization,
)


class MeasureCalibrationDiagnostics(StrictModule):
    """Calibration evidence and source identity for one reweighted realization."""

    calibration: MomentCalibrationResult
    source_mass: Array
    source_points: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    source_provenance: str = eqx.field(static=True)


def calibrate(
    realization: Any,
    target: ExactMoments | QuadraticMoments,
    /,
    *,
    features: Any | None = None,
    method: AbstractScalarIterativeMethod | None = None,
    termination: OptimizationTermination | None = None,
    policy: MomentCalibrationPolicy | None = None,
    initial_dual: Array | None = None,
):
    """Calibrate one realized finite measure to normalized target expectations."""

    if not isinstance(target, (ExactMoments, QuadraticMoments)):
        raise TypeError("target must be ExactMoments or QuadraticMoments.")
    measure = lower_finite_measure(realization)
    raw_features = (
        measure.samples
        if features is None
        else features(measure.samples)
        if callable(features)
        else features
    )
    feature_values = feature_matrix(raw_features, measure.axis, measure.count)
    problem = MomentCalibrationProblem(
        feature_values,
        target,
        prior_log_weights=measure.log_weights,
        mask=measure.mask,
    )
    result = require_converged(
        calibrate_moments(
            problem,
            method=method,
            termination=termination,
            policy=policy,
            initial_dual=initial_dual,
        )
    )
    diagnostics = MeasureCalibrationDiagnostics(
        calibration=result,
        source_mass=measure.physical_mass,
        source_points=measure.count,
        feature_count=int(feature_values.shape[1]),
        source_provenance=measure.source_provenance,
    )
    target_kind = "exact" if isinstance(target, ExactMoments) else "quadratic"
    provenance = f"calibrated:{target_kind}:{measure.source_provenance}"
    return transformed_weighted_realization(
        realization,
        measure,
        result.log_weights,
        transformation_kind="calibration",
        transformation_diagnostics=diagnostics,
        provenance=provenance,
    )


__all__ = ["MeasureCalibrationDiagnostics", "calibrate"]
