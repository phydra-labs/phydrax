#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Observation model for calibrating relative force to physical newtons.

The measurement equation is ``y_N = s_N * f_relative + Z beta_N + error``.
``s_N`` is an explicit physical scale and ``Z beta_N`` contains protocol-named
nuisance effects.  This is a fitted observation/personalization model, never a
quantity conversion.  The separation of measurand, input quantities, and
nuisance/influence quantities follows JCGM 100:2008, clauses 4.1--4.2
(https://www.bipm.org/en/committees/jc/jcgm/publications); that reference does
not prescribe this skeletal-muscle design matrix or confer physiological
identity on its relative-force input.
"""

from __future__ import annotations

from enum import IntFlag
from math import isfinite
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._numerics import solve_weighted_least_squares
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class PhysicalRelativeForceCalibrationStatus(IntFlag):
    """Fail-closed status for one physical force-scale fit."""

    SUCCESS = 0
    NONFINITE_OBSERVATION = 1
    NONPOSITIVE_UNCERTAINTY = 2
    INSUFFICIENT_SAMPLES = 4
    RANK_DEFICIENT = 8
    SCALE_NOT_IDENTIFIABLE = 16
    NONPOSITIVE_SCALE = 32
    ILL_CONDITIONED = 64
    SOLVER_FAILURE = 128


class PhysicalRelativeForceCalibrationPlan(StrictModule):
    """Fixed protocol/asset identity and nuisance observation design."""

    nuisance_design: Array
    nuisance_names: tuple[str, ...] = eqx.field(static=True)
    protocol_id: str = eqx.field(static=True)
    asset_id: str = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    nuisance_count: int = eqx.field(static=True)
    relative_rank_cutoff: float = eqx.field(static=True)
    maximum_condition_number: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        nuisance_design: ArrayLike,
        nuisance_names: Sequence[str],
        /,
        *,
        protocol_id: str,
        asset_id: str,
        relative_rank_cutoff: float = 1.0e-10,
        maximum_condition_number: float = 1.0e10,
    ):
        design = jnp.asarray(nuisance_design)
        if design.ndim != 2 or design.shape[0] < 1:
            raise ValueError("nuisance_design must have shape (samples>=1, nuisances).")
        if not jnp.issubdtype(design.dtype, jnp.inexact):
            design = design.astype(float)
        names = tuple(nuisance_names)
        if len(names) != design.shape[1]:
            raise ValueError("nuisance_names must name every nuisance-design column.")
        if len(set(names)) != len(names) or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError("nuisance_names must be unique nonempty strings.")
        if not isinstance(protocol_id, str) or not protocol_id:
            raise ValueError("protocol_id must be a nonempty string.")
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError("asset_id must be a nonempty string.")
        cutoff = float(relative_rank_cutoff)
        maximum_condition = float(maximum_condition_number)
        if not isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("relative_rank_cutoff must be finite and nonnegative.")
        if not isfinite(maximum_condition) or maximum_condition <= 1.0:
            raise ValueError("maximum_condition_number must be finite and exceed one.")
        if not bool(np.all(np.isfinite(np.asarray(design)))):
            raise ValueError("nuisance_design must be finite.")
        self.nuisance_design = design
        self.nuisance_names = names
        self.protocol_id = protocol_id
        self.asset_id = asset_id
        self.sample_count = int(design.shape[0])
        self.nuisance_count = int(design.shape[1])
        self.relative_rank_cutoff = cutoff
        self.maximum_condition_number = maximum_condition
        self.plan_id = canonical_fingerprint(
            {
                "kind": "physical-relative-force-observation-calibration",
                "protocol_id": protocol_id,
                "asset_id": asset_id,
                "sample_count": int(design.shape[0]),
                "nuisance_names": list(names),
                "relative_rank_cutoff": cutoff.hex(),
                "maximum_condition_number": maximum_condition.hex(),
                "measurement_equation": "force_N=scale_N_per_relative*relative_force+Z*nuisance_N",
            }
        )

    def prepare(self, /) -> "PreparedPhysicalRelativeForceCalibration":
        """Prepare the fixed observation design and identity."""

        return PreparedPhysicalRelativeForceCalibration(
            self,
            canonical_fingerprint(
                {
                    "kind": "prepared-physical-relative-force-calibration",
                    "plan": self.plan_id,
                    "solver_owner": "phydrax._numerics.solve_weighted_least_squares",
                }
            ),
        )


class PhysicalRelativeForceCalibrationState(StrictModule, NonTrainableState):
    """Committed physical scale and protocol nuisance coefficients."""

    scale_newton_per_relative_force: Array
    nuisance_coefficients_newton: Array
    calibration_epoch: Array
    plan_id: str = eqx.field(static=True)
    protocol_id: str = eqx.field(static=True)
    asset_id: str = eqx.field(static=True)


class PhysicalRelativeForceCalibrationEvidence(StrictModule, NonTrainableState):
    """Fit, nuisance-confounding, and scale-identifiability evidence."""

    prediction_newton: Array
    residual_newton: Array
    valid_sample_mask: Array
    standard_uncertainty_newton: Array
    singular_values: Array
    sample_count: Array
    design_rank: Array
    condition_number: Array
    scale_information_per_newton_squared: Array
    nuisance_confounding_fraction: Array
    scale_standard_uncertainty_newton_per_relative_force: Array
    residual_standard_deviation_newton: Array
    solver_status: Array
    observations_finite: Array
    uncertainties_positive: Array
    scale_identifiable: Array
    scale_positive: Array
    condition_acceptable: Array
    status: Array
    successful: Array
    measurement_equation: str = eqx.field(static=True)
    claim_scope: str = eqx.field(static=True)


class PhysicalRelativeForceCalibrationCandidate(StrictModule):
    """Uncommitted scale fit retaining its whole source calibration."""

    source: PhysicalRelativeForceCalibrationState
    proposed: PhysicalRelativeForceCalibrationState
    evidence: PhysicalRelativeForceCalibrationEvidence


class PhysicalForceObservation(StrictModule, NonTrainableState):
    """Calibrated physical force observation in newtons."""

    relative_force: Array
    force_newton: Array
    scale_newton_per_relative_force: Array
    plan_id: str = eqx.field(static=True)
    protocol_id: str = eqx.field(static=True)
    asset_id: str = eqx.field(static=True)


class PreparedPhysicalRelativeForceCalibration(StrictModule):
    """Prepared rank-diagnosed physical force observation runtime."""

    plan: PhysicalRelativeForceCalibrationPlan
    prepared_id: str = eqx.field(static=True)

    def initialize(
        self,
        scale_newton_per_relative_force: float,
        /,
        *,
        nuisance_coefficients_newton: ArrayLike | None = None,
    ) -> PhysicalRelativeForceCalibrationState:
        """Initialize an explicit positive physical scale for transactional fitting."""

        scale = float(scale_newton_per_relative_force)
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "scale_newton_per_relative_force must be positive and finite."
            )
        if nuisance_coefficients_newton is None:
            nuisance = jnp.zeros(
                (self.plan.nuisance_count,), dtype=self.plan.nuisance_design.dtype
            )
        else:
            nuisance = jnp.asarray(
                nuisance_coefficients_newton, dtype=self.plan.nuisance_design.dtype
            )
        if nuisance.shape != (self.plan.nuisance_count,):
            raise ValueError(
                "nuisance_coefficients_newton must match the nuisance column count."
            )
        if not bool(np.all(np.isfinite(np.asarray(nuisance)))):
            raise ValueError("nuisance_coefficients_newton must be finite.")
        return PhysicalRelativeForceCalibrationState(
            jnp.asarray(scale, dtype=self.plan.nuisance_design.dtype),
            nuisance,
            jnp.asarray(0, dtype=jnp.int32),
            self.plan.plan_id,
            self.plan.protocol_id,
            self.plan.asset_id,
        )

    def evaluate(
        self,
        state: PhysicalRelativeForceCalibrationState,
        relative_force: ArrayLike,
        observed_force_newton: ArrayLike,
        standard_uncertainty_newton: ArrayLike,
        /,
        *,
        sample_mask: ArrayLike | None = None,
    ) -> PhysicalRelativeForceCalibrationCandidate:
        """Fit scale plus named nuisances with Phydrax's diagnosed SVD solve."""

        if not isinstance(state, PhysicalRelativeForceCalibrationState):
            raise TypeError("state must be PhysicalRelativeForceCalibrationState.")
        if state.plan_id != self.plan.plan_id:
            raise ValueError("state does not belong to this prepared calibration.")
        dtype = self.plan.nuisance_design.dtype
        relative = jnp.asarray(relative_force, dtype=dtype)
        observed = jnp.asarray(observed_force_newton, dtype=dtype)
        uncertainty = jnp.asarray(standard_uncertainty_newton, dtype=dtype)
        shape = (self.plan.sample_count,)
        if relative.shape != shape or observed.shape != shape or uncertainty.shape != shape:
            raise ValueError(
                "relative_force, observed_force_newton, and standard_uncertainty_newton "
                f"must have shape {shape}."
            )
        if sample_mask is None:
            requested = jnp.ones(shape, dtype=bool)
        else:
            requested = jnp.asarray(sample_mask, dtype=bool)
            if requested.shape != shape:
                raise ValueError(f"sample_mask must have shape {shape}.")
        observations_finite = jnp.isfinite(relative) & jnp.isfinite(observed)
        uncertainties_positive = jnp.isfinite(uncertainty) & (uncertainty > 0.0)
        valid = requested & observations_finite & uncertainties_positive
        safe_uncertainty = jnp.where(valid, uncertainty, 1.0)
        weights = jnp.where(valid, 1.0 / safe_uncertainty**2, 0.0)
        design = jnp.concatenate(
            (relative[:, None], self.plan.nuisance_design), axis=1
        )
        feature_count = 1 + self.plan.nuisance_count
        result = solve_weighted_least_squares(
            design,
            observed,
            mask=valid,
            weights=weights,
            center=False,
            scale=True,
            ridge=0.0,
            rcond=self.plan.relative_rank_cutoff,
            min_samples=feature_count + 1,
            max_features=feature_count,
        )
        scale = result.raw_coefficients[0]
        nuisance = result.raw_coefficients[1:]

        if self.plan.nuisance_count:
            nuisance_projection = solve_weighted_least_squares(
                self.plan.nuisance_design,
                relative,
                mask=valid,
                weights=weights,
                center=False,
                scale=True,
                ridge=0.0,
                rcond=self.plan.relative_rank_cutoff,
                min_samples=self.plan.nuisance_count,
                max_features=self.plan.nuisance_count,
            )
            scale_residual = relative - nuisance_projection.prediction
        else:
            scale_residual = relative
        scale_information = jnp.sum(
            jnp.where(valid, weights * scale_residual**2, 0.0)
        )
        total_scale_moment = jnp.sum(jnp.where(valid, weights * relative**2, 0.0))
        nuisance_confounding = jnp.where(
            total_scale_moment > 0.0,
            jnp.clip(1.0 - scale_information / total_scale_moment, 0.0, 1.0),
            1.0,
        )
        information_tolerance = (
            jnp.finfo(dtype).eps * jnp.maximum(total_scale_moment, 1.0)
        )
        scale_identifiable = (
            result.rank == feature_count
        ) & (scale_information > information_tolerance)
        scale_positive = jnp.isfinite(scale) & (scale > 0.0)
        condition_acceptable = (
            jnp.isfinite(result.condition_number)
            & (result.condition_number <= self.plan.maximum_condition_number)
        )
        degrees_of_freedom = result.sample_count - feature_count
        safe_degrees_of_freedom = jnp.maximum(
            degrees_of_freedom.astype(dtype), 1.0
        )
        normalized_residual_variance = (
            jnp.sum(jnp.where(valid, weights * result.residual**2, 0.0))
            / safe_degrees_of_freedom
        )
        residual_variance_newton_squared = (
            jnp.sum(jnp.where(valid, result.residual**2, 0.0))
            / safe_degrees_of_freedom
        )
        residual_sd = jnp.sqrt(residual_variance_newton_squared)
        scale_standard_uncertainty = jnp.sqrt(
            normalized_residual_variance
            / jnp.maximum(scale_information, jnp.finfo(dtype).tiny)
        )
        proposed = PhysicalRelativeForceCalibrationState(
            scale,
            nuisance,
            state.calibration_epoch + jnp.asarray(1, dtype=jnp.int32),
            state.plan_id,
            state.protocol_id,
            state.asset_id,
        )

        observations_all_finite = jnp.all((~requested) | observations_finite)
        uncertainties_all_positive = jnp.all((~requested) | uncertainties_positive)
        enough = result.sample_count >= feature_count + 1
        full_rank = result.rank == feature_count
        status = jnp.asarray(
            int(PhysicalRelativeForceCalibrationStatus.SUCCESS), dtype=jnp.int32
        )
        status = jnp.where(
            observations_all_finite,
            status,
            jnp.bitwise_or(status, int(PhysicalRelativeForceCalibrationStatus.NONFINITE_OBSERVATION)),
        )
        status = jnp.where(
            uncertainties_all_positive,
            status,
            jnp.bitwise_or(status, int(PhysicalRelativeForceCalibrationStatus.NONPOSITIVE_UNCERTAINTY)),
        )
        status = jnp.where(
            enough,
            status,
            jnp.bitwise_or(status, int(PhysicalRelativeForceCalibrationStatus.INSUFFICIENT_SAMPLES)),
        )
        status = jnp.where(
            full_rank,
            status,
            jnp.bitwise_or(status, int(PhysicalRelativeForceCalibrationStatus.RANK_DEFICIENT)),
        )
        status = jnp.where(
            scale_identifiable,
            status,
            jnp.bitwise_or(status, int(PhysicalRelativeForceCalibrationStatus.SCALE_NOT_IDENTIFIABLE)),
        )
        status = jnp.where(
            scale_positive,
            status,
            jnp.bitwise_or(status, int(PhysicalRelativeForceCalibrationStatus.NONPOSITIVE_SCALE)),
        )
        status = jnp.where(
            condition_acceptable,
            status,
            jnp.bitwise_or(status, int(PhysicalRelativeForceCalibrationStatus.ILL_CONDITIONED)),
        )
        status = jnp.where(
            result.valid,
            status,
            jnp.bitwise_or(status, int(PhysicalRelativeForceCalibrationStatus.SOLVER_FAILURE)),
        )
        successful = status == int(PhysicalRelativeForceCalibrationStatus.SUCCESS)
        evidence = PhysicalRelativeForceCalibrationEvidence(
            result.prediction,
            result.residual,
            valid,
            uncertainty,
            result.singular_values,
            result.sample_count,
            result.rank,
            result.condition_number,
            scale_information,
            nuisance_confounding,
            scale_standard_uncertainty,
            residual_sd,
            result.status,
            observations_all_finite,
            uncertainties_all_positive,
            scale_identifiable,
            scale_positive,
            condition_acceptable,
            status,
            successful,
            (
                "observed_force_N = scale_N_per_relative_force * relative_force "
                "+ nuisance_design * nuisance_coefficients_N"
            ),
            (
                "physical observation fit bound to the declared protocol and asset; "
                "not a quantity conversion and not evidence that relative-force "
                "models share identities"
            ),
        )
        return PhysicalRelativeForceCalibrationCandidate(state, proposed, evidence)

    def observe(
        self,
        state: PhysicalRelativeForceCalibrationState,
        relative_force: ArrayLike,
        /,
    ) -> PhysicalForceObservation:
        """Map relative force to newtons with the committed physical scale only."""

        if not isinstance(state, PhysicalRelativeForceCalibrationState):
            raise TypeError("state must be PhysicalRelativeForceCalibrationState.")
        if state.plan_id != self.plan.plan_id:
            raise ValueError("state does not belong to this prepared calibration.")
        relative = jnp.asarray(relative_force, dtype=state.scale_newton_per_relative_force.dtype)
        return PhysicalForceObservation(
            relative,
            state.scale_newton_per_relative_force * relative,
            state.scale_newton_per_relative_force,
            state.plan_id,
            state.protocol_id,
            state.asset_id,
        )


def commit_physical_relative_force_calibration(
    candidate: PhysicalRelativeForceCalibrationCandidate,
    current: PhysicalRelativeForceCalibrationState,
    /,
) -> PhysicalRelativeForceCalibrationState:
    """Commit an identifiable positive scale atomically, otherwise roll back."""

    if not isinstance(candidate, PhysicalRelativeForceCalibrationCandidate):
        raise TypeError("candidate must be PhysicalRelativeForceCalibrationCandidate.")
    if not isinstance(current, PhysicalRelativeForceCalibrationState):
        raise TypeError("current must be PhysicalRelativeForceCalibrationState.")
    source = candidate.source
    source_matches = (
        (source.plan_id == current.plan_id)
        & (source.protocol_id == current.protocol_id)
        & (source.asset_id == current.asset_id)
        & (source.scale_newton_per_relative_force == current.scale_newton_per_relative_force)
        & jnp.array_equal(
            source.nuisance_coefficients_newton,
            current.nuisance_coefficients_newton,
        )
        & (source.calibration_epoch == current.calibration_epoch)
    )
    return jax.lax.cond(
        candidate.evidence.successful & source_matches,
        lambda _: candidate.proposed,
        lambda _: current,
        operand=None,
    )


__all__ = [
    "PhysicalForceObservation",
    "PhysicalRelativeForceCalibrationCandidate",
    "PhysicalRelativeForceCalibrationEvidence",
    "PhysicalRelativeForceCalibrationPlan",
    "PhysicalRelativeForceCalibrationState",
    "PhysicalRelativeForceCalibrationStatus",
    "PreparedPhysicalRelativeForceCalibration",
    "commit_physical_relative_force_calibration",
]
