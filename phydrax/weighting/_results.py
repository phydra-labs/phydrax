#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import core as jax_core
from jaxtyping import Array

from .._strict import StrictModule
from ..optim import OptimizationDiagnostics, OptimizationProvenance
from ._problem import MomentCalibrationProblem


class MomentCalibrationStatus(IntEnum):
    """Terminal statuses for finite-measure moment calibration."""

    SUCCESS = 0
    AFFINE_TARGET_INCONSISTENT = 1
    TARGET_RESIDUAL_NOT_MET = 2
    REGULARITY_NOT_CERTIFIED = 3
    OPTIMIZATION_FAILED = 4
    NONFINITE_RESULT = 5


_STATUS_MESSAGES = {
    MomentCalibrationStatus.SUCCESS: "success",
    MomentCalibrationStatus.AFFINE_TARGET_INCONSISTENT: (
        "target is inconsistent with the active feature affine hull"
    ),
    MomentCalibrationStatus.TARGET_RESIDUAL_NOT_MET: (
        "exact target residual exceeds tolerance"
    ),
    MomentCalibrationStatus.REGULARITY_NOT_CERTIFIED: (
        "finite exact-dual regularity was not certified"
    ),
    MomentCalibrationStatus.OPTIMIZATION_FAILED: "underlying optimization failed",
    MomentCalibrationStatus.NONFINITE_RESULT: "calibration produced non-finite values",
}


def moment_calibration_status_message(
    status: int | MomentCalibrationStatus,
    /,
) -> str:
    """Return a stable human-readable calibration status message."""

    return _STATUS_MESSAGES[MomentCalibrationStatus(int(status))]


class MomentCalibrationDiagnostics(StrictModule):
    """Moment, regularity, concentration, and optimizer evidence."""

    optimizer_status: Array
    optimization: OptimizationDiagnostics
    prior_moments: Array
    target_residual: Array
    scaled_target_residual: Array
    maximum_absolute_residual: Array
    maximum_scaled_residual: Array
    affine_residual_norm: Array
    numerical_affine_rank: Array
    rank_cutoff: Array
    minimum_prior_eigenvalue: Array
    maximum_prior_eigenvalue: Array
    minimum_final_eigenvalue: Array
    final_condition_estimate: Array
    dual_gradient_norm: Array
    dual_norm: Array
    relative_entropy: Array
    effective_sample_size: Array
    active_support: Array
    minimum_active_weight: Array
    maximum_active_weight: Array
    maximum_log_weight_ratio: Array
    normalization_residual: Array
    geometry_finite: Array
    spectrum: Any


class MomentCalibrationProvenance(StrictModule):
    """Static numerical identity for one moment-calibration solve."""

    problem_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    target_kind: str = eqx.field(static=True)
    source_points: int = eqx.field(static=True)
    moment_count: int = eqx.field(static=True)
    execution: str = eqx.field(static=True)
    differentiation: str = eqx.field(static=True)
    optimizer: OptimizationProvenance


class MomentCalibrationResult(StrictModule):
    """Calibrated normalized weights with explicit numerical evidence."""

    problem: MomentCalibrationProblem
    log_weights: Array
    dual_variables: Array
    achieved_moments: Array
    status: Array
    diagnostics: MomentCalibrationDiagnostics
    provenance: MomentCalibrationProvenance

    @property
    def weights(self) -> Array:
        """Return normalized nonnegative weights with inactive support at zero."""

        return jnp.where(
            jnp.isfinite(self.log_weights),
            jnp.exp(self.log_weights),
            0.0,
        )

    @property
    def converged(self) -> Array:
        """Whether all calibration success criteria were satisfied."""

        return self.status == int(MomentCalibrationStatus.SUCCESS)

    @property
    def successful(self) -> Array:
        """Alias for the terminal calibration success predicate."""

        return self.converged


def require_converged(result: MomentCalibrationResult, /) -> MomentCalibrationResult:
    """Raise unless calibration succeeded, including under JAX transforms."""

    if not isinstance(result, MomentCalibrationResult):
        raise TypeError("result must be a MomentCalibrationResult.")
    failed = jnp.logical_not(result.converged)
    if not isinstance(failed, jax_core.Tracer):
        if bool(failed):
            raise eqx.EquinoxRuntimeError(
                "Moment calibration did not converge: "
                f"{moment_calibration_status_message(result.status)}."
            )
        return result
    checked = eqx.error_if(
        result.log_weights,
        failed,
        "Moment calibration did not converge.",
    )
    return eqx.tree_at(lambda item: item.log_weights, result, checked)


__all__ = [
    "MomentCalibrationDiagnostics",
    "MomentCalibrationProvenance",
    "MomentCalibrationResult",
    "MomentCalibrationStatus",
    "moment_calibration_status_message",
    "require_converged",
]
