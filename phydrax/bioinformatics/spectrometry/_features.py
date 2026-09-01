#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._spectrum import IonMobilityUnit, IonPolarity, SpectrometryUnits


class FeatureMatchStatus(IntEnum):
    """Status of bounded LC-MS feature matching."""

    SUCCESS = 0
    CAPACITY_EXCEEDED = 1
    UNIT_MISMATCH = 2
    POLARITY_MISMATCH = 3
    NONFINITE = 4


class FeatureMatchEvidence(IntFlag):
    """Evidence retained by feature matching."""

    NONE = 0
    MASS_MATCH = 1
    RETENTION_TIME_MATCH = 2
    ION_MOBILITY_MATCH = 4
    AMBIGUOUS = 8
    UNMATCHED = 16


_MATCH_CONTRACT = BioinformaticsMethodContract(
    "bounded LC-MS feature matching",
    MethodKind.HEURISTIC,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.ALMOST_EVERYWHERE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Candidate pairs are gated by explicit mass, retention-time, and optional "
        "ion-mobility tolerances, then ranked by normalized squared error."
    ),
    truncation_statement=(
        "Worst-case query and reference capacities are preflighted; ambiguity is "
        "reported rather than silently collapsed."
    ),
    capacity_semantics="The pairwise comparison has fixed query_capacity × reference_capacity work.",
    assumptions=(
        "Features compared in one call share compatible acquisition conditions.",
    ),
    nondifferentiable_outputs=(
        "reference_index",
        "matched_mask",
        "ambiguous_mask",
        "status",
        "evidence",
    ),
)


class LCMSFeatureBatch(StrictModule):
    """Fixed-capacity LC-MS features with explicit missing mobility and charge."""

    feature_ids: Array
    mass_to_charge: Array
    retention_time: Array
    intensity: Array
    charge: Array
    ion_mobility: Array
    ion_mobility_mask: Array
    active_mask: Array
    units: SpectrometryUnits = eqx.field(static=True)
    polarity: IonPolarity = eqx.field(static=True)
    feature_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        feature_ids: ArrayLike,
        mass_to_charge: ArrayLike,
        retention_time: ArrayLike,
        intensity: ArrayLike,
        charge: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        ion_mobility: ArrayLike | None = None,
        ion_mobility_mask: ArrayLike | None = None,
        units: SpectrometryUnits | None = None,
        polarity: IonPolarity = IonPolarity.UNKNOWN,
    ):
        ids = np.asarray(feature_ids)
        mz = np.asarray(mass_to_charge)
        time = np.asarray(retention_time)
        signal = np.asarray(intensity)
        charges = np.asarray(charge)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("feature_ids must be a non-empty vector.")
        if any(value.shape != ids.shape for value in (mz, time, signal, charges)):
            raise ValueError("All feature vectors must have the feature_ids shape.")
        if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(
            charges.dtype, np.integer
        ):
            raise TypeError("feature_ids and charge must contain integers.")
        mask = (
            np.ones(ids.shape, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if mask.shape != ids.shape:
            raise ValueError("active_mask must match feature_ids.")
        count = int(np.count_nonzero(mask))
        if not np.all(mask[:count]) or np.any(mask[count:]):
            raise ValueError("active_mask must be a left-prefix mask.")
        dtype = np.result_type(mz.dtype, time.dtype, signal.dtype, np.float32)
        mz = mz.astype(dtype, copy=False)
        time = time.astype(dtype, copy=False)
        signal = signal.astype(dtype, copy=False)
        if np.any(~np.isfinite(mz[mask])) or np.any(mz[mask] <= 0.0):
            raise ValueError("Active mass-to-charge values must be finite and positive.")
        if np.any(~np.isfinite(time[mask])) or np.any(time[mask] < 0.0):
            raise ValueError("Active retention times must be finite and nonnegative.")
        if np.any(~np.isfinite(signal[mask])) or np.any(signal[mask] < 0.0):
            raise ValueError("Active intensities must be finite and nonnegative.")
        if np.any(np.abs(charges[mask]) > 64):
            raise ValueError("Feature charge magnitude cannot exceed 64.")
        resolved_units = SpectrometryUnits() if units is None else units
        if not isinstance(resolved_units, SpectrometryUnits):
            raise TypeError("units must be SpectrometryUnits.")
        if ion_mobility is None:
            mobility = np.zeros(ids.shape, dtype=dtype)
            mobility_mask = np.zeros(ids.shape, dtype=bool)
        else:
            mobility = np.asarray(ion_mobility, dtype=dtype)
            mobility_mask = (
                mask.copy()
                if ion_mobility_mask is None
                else np.asarray(ion_mobility_mask, dtype=bool)
            )
            if mobility.shape != ids.shape or mobility_mask.shape != ids.shape:
                raise ValueError("Ion-mobility values and mask must match feature_ids.")
        if np.any(mobility_mask & ~mask):
            raise ValueError("Ion-mobility values require active features.")
        if np.any(~np.isfinite(mobility[mobility_mask])):
            raise ValueError("Active ion-mobility values must be finite.")
        if resolved_units.ion_mobility == IonMobilityUnit.NONE and np.any(mobility_mask):
            raise ValueError("Ion-mobility values require a non-NONE unit.")
        for value in (ids, mz, time, signal, charges, mobility):
            if np.any(value[~mask] != 0):
                raise ValueError("Inactive feature entries must be zero padding.")
        self.feature_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.mass_to_charge = jnp.asarray(mz)
        self.retention_time = jnp.asarray(time)
        self.intensity = jnp.asarray(signal)
        self.charge = jnp.asarray(charges, dtype=jnp.int32)
        self.ion_mobility = jnp.asarray(mobility)
        self.ion_mobility_mask = jnp.asarray(mobility_mask)
        self.active_mask = jnp.asarray(mask)
        self.units = resolved_units
        self.polarity = IonPolarity(polarity)
        self.feature_capacity = int(ids.size)


class FeatureMatchPlan(StrictModule):
    """Static capacities and physical tolerances for feature matching."""

    mass_tolerance_ppm: float = eqx.field(static=True)
    retention_time_tolerance: float = eqx.field(static=True)
    ion_mobility_tolerance: float = eqx.field(static=True)
    require_ion_mobility: bool = eqx.field(static=True)
    query_capacity: int = eqx.field(static=True)
    reference_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        mass_tolerance_ppm: float,
        retention_time_tolerance: float,
        query_capacity: int,
        reference_capacity: int,
        ion_mobility_tolerance: float = 0.0,
        require_ion_mobility: bool = False,
    ):
        ppm = float(mass_tolerance_ppm)
        time = float(retention_time_tolerance)
        mobility = float(ion_mobility_tolerance)
        queries = int(query_capacity)
        references = int(reference_capacity)
        if not np.isfinite(ppm) or ppm <= 0.0:
            raise ValueError("mass_tolerance_ppm must be finite and positive.")
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("retention_time_tolerance must be finite and nonnegative.")
        if not np.isfinite(mobility) or mobility < 0.0:
            raise ValueError("ion_mobility_tolerance must be finite and nonnegative.")
        if bool(require_ion_mobility) and mobility <= 0.0:
            raise ValueError("Required ion mobility needs a positive tolerance.")
        if queries < 1 or references < 1:
            raise ValueError("query_capacity and reference_capacity must be positive.")
        self.mass_tolerance_ppm = ppm
        self.retention_time_tolerance = time
        self.ion_mobility_tolerance = mobility
        self.require_ion_mobility = bool(require_ion_mobility)
        self.query_capacity = queries
        self.reference_capacity = references


class FeatureMatchResult(StrictModule):
    """Best reference per query plus ambiguity and physical errors."""

    reference_index: Array
    reference_feature_id: Array
    mass_error_ppm: Array
    retention_time_error: Array
    ion_mobility_error: Array
    matched_mask: Array
    ambiguous_mask: Array
    candidate_count: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def _empty_match_result(
    query: LCMSFeatureBatch,
    status: FeatureMatchStatus,
    /,
) -> FeatureMatchResult:
    shape = (query.feature_capacity,)
    return FeatureMatchResult(
        reference_index=jnp.full(shape, -1, dtype=jnp.int32),
        reference_feature_id=jnp.full(shape, -1, dtype=jnp.int64),
        mass_error_ppm=jnp.zeros(shape, dtype=query.mass_to_charge.dtype),
        retention_time_error=jnp.zeros(shape, dtype=query.retention_time.dtype),
        ion_mobility_error=jnp.zeros(shape, dtype=query.ion_mobility.dtype),
        matched_mask=jnp.zeros(shape, dtype=bool),
        ambiguous_mask=jnp.zeros(shape, dtype=bool),
        candidate_count=jnp.zeros(shape, dtype=jnp.int32),
        valid=jnp.asarray(False),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        evidence=jnp.asarray(int(FeatureMatchEvidence.NONE), dtype=jnp.uint32),
        method_contract=_MATCH_CONTRACT,
    )


def match_lcms_features(
    query: LCMSFeatureBatch,
    reference: LCMSFeatureBatch,
    plan: FeatureMatchPlan,
    /,
) -> FeatureMatchResult:
    """Match compatible features without silent capacity or ambiguity loss."""
    if not isinstance(query, LCMSFeatureBatch) or not isinstance(
        reference, LCMSFeatureBatch
    ):
        raise TypeError("query and reference must be LCMSFeatureBatch values.")
    if not isinstance(plan, FeatureMatchPlan):
        raise TypeError("plan must be a FeatureMatchPlan.")
    if (
        query.feature_capacity > plan.query_capacity
        or reference.feature_capacity > plan.reference_capacity
    ):
        return _empty_match_result(query, FeatureMatchStatus.CAPACITY_EXCEEDED)
    require_mobility = (
        plan.require_ion_mobility
        or query.units.ion_mobility != IonMobilityUnit.NONE
        or reference.units.ion_mobility != IonMobilityUnit.NONE
    )
    units_match = query.units.compatible_with(
        reference.units,
        require_mobility=require_mobility,
    )
    if not units_match:
        return _empty_match_result(query, FeatureMatchStatus.UNIT_MISMATCH)
    if query.polarity != reference.polarity:
        return _empty_match_result(query, FeatureMatchStatus.POLARITY_MISMATCH)

    q_mz = query.mass_to_charge[:, None]
    r_mz = reference.mass_to_charge[None, :]
    ppm_error = 1.0e6 * (q_mz - r_mz) / r_mz
    time_error = query.retention_time[:, None] - reference.retention_time[None, :]
    q_mobility = query.ion_mobility[:, None]
    r_mobility = reference.ion_mobility[None, :]
    mobility_error = q_mobility - r_mobility
    both_mobility = (
        query.ion_mobility_mask[:, None] & reference.ion_mobility_mask[None, :]
    )
    mobility_gate = jnp.where(
        both_mobility,
        jnp.abs(mobility_error) <= plan.ion_mobility_tolerance,
        not plan.require_ion_mobility,
    )
    mass_gate = jnp.abs(ppm_error) <= plan.mass_tolerance_ppm
    time_gate = jnp.abs(time_error) <= plan.retention_time_tolerance
    pair_active = query.active_mask[:, None] & reference.active_mask[None, :]
    candidates = pair_active & mass_gate & time_gate & mobility_gate
    candidate_count = jnp.sum(candidates, axis=1, dtype=jnp.int32)
    mobility_scale = max(plan.ion_mobility_tolerance, 1.0)
    time_scale = max(plan.retention_time_tolerance, 1.0)
    score = (
        (ppm_error / plan.mass_tolerance_ppm) ** 2
        + (time_error / time_scale) ** 2
        + jnp.where(both_mobility, (mobility_error / mobility_scale) ** 2, 0.0)
    )
    ranked_score = jnp.where(candidates, score, jnp.inf)
    reference_index = jnp.argmin(ranked_score, axis=1)
    matched = query.active_mask & (candidate_count > 0)
    safe_index = jnp.where(matched, reference_index, 0)
    rows = jnp.arange(query.feature_capacity)
    selected_ppm = ppm_error[rows, safe_index]
    selected_time = time_error[rows, safe_index]
    selected_mobility = mobility_error[rows, safe_index]
    selected_mobility_present = both_mobility[rows, safe_index]
    ambiguous = matched & (candidate_count > 1)
    evidence = jnp.where(
        jnp.any(matched),
        int(FeatureMatchEvidence.MASS_MATCH | FeatureMatchEvidence.RETENTION_TIME_MATCH),
        int(FeatureMatchEvidence.NONE),
    ).astype(jnp.uint32)
    evidence = evidence | jnp.where(
        jnp.any(matched & selected_mobility_present),
        int(FeatureMatchEvidence.ION_MOBILITY_MATCH),
        0,
    ).astype(jnp.uint32)
    evidence = evidence | jnp.where(
        jnp.any(ambiguous), int(FeatureMatchEvidence.AMBIGUOUS), 0
    ).astype(jnp.uint32)
    evidence = evidence | jnp.where(
        jnp.any(query.active_mask & ~matched), int(FeatureMatchEvidence.UNMATCHED), 0
    ).astype(jnp.uint32)
    finite = jnp.all(jnp.isfinite(jnp.where(candidates, score, 0.0)))
    return FeatureMatchResult(
        reference_index=jnp.where(matched, reference_index, -1).astype(jnp.int32),
        reference_feature_id=jnp.where(
            matched, reference.feature_ids[safe_index], -1
        ).astype(jnp.int64),
        mass_error_ppm=jnp.where(matched, selected_ppm, 0.0),
        retention_time_error=jnp.where(matched, selected_time, 0.0),
        ion_mobility_error=jnp.where(
            matched & selected_mobility_present, selected_mobility, 0.0
        ),
        matched_mask=matched,
        ambiguous_mask=ambiguous,
        candidate_count=candidate_count,
        valid=finite,
        status=jnp.where(
            finite, int(FeatureMatchStatus.SUCCESS), int(FeatureMatchStatus.NONFINITE)
        ).astype(jnp.int32),
        evidence=evidence,
        method_contract=_MATCH_CONTRACT,
    )


__all__ = [
    "FeatureMatchEvidence",
    "FeatureMatchPlan",
    "FeatureMatchResult",
    "FeatureMatchStatus",
    "LCMSFeatureBatch",
    "match_lcms_features",
]
