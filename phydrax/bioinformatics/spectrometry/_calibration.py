#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._numerics import solve_weighted_least_squares
from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._spectrum import MassSpectrum, MassToChargeUnit


class CalibrationStatus(IntEnum):
    """Status of a mass calibration or binning operation."""

    SUCCESS = 0
    INSUFFICIENT_CALIBRANTS = 1
    RANK_DEFICIENT = 2
    NONFINITE = 3
    UNIT_MISMATCH = 4
    EXTRAPOLATION = 5
    OUT_OF_RANGE = 6


class CalibrationEvidence(IntFlag):
    """Evidence and limitations of a calibration operation."""

    NONE = 0
    CALIBRANTS_FIT = 1
    CALIBRATED = 2
    EXTRAPOLATED = 4
    BINNED = 8
    OUTSIDE_BINS = 16


_FIT_CONTRACT = BioinformaticsMethodContract(
    "polynomial mass calibration fit",
    MethodKind.APPROXIMATE_MODEL,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.ALMOST_EVERYWHERE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Weighted least squares is solved by the Phydrax diagnosed SVD substrate "
        "on a normalized mass coordinate."
    ),
    truncation_statement=(
        "The polynomial degree is explicit and no calibrant is silently removed "
        "beyond the supplied mask."
    ),
    capacity_semantics="Calibrant and polynomial capacities are fixed by input shapes and degree.",
    assumptions=(
        "Reference masses are traceable calibrants.",
        "Mass error is smooth over the calibrated interval.",
    ),
    nondifferentiable_outputs=("status", "evidence"),
)

_APPLY_CONTRACT = BioinformaticsMethodContract(
    "polynomial mass calibration application",
    MethodKind.APPROXIMATE_MODEL,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.EXACT_AD,
    OutputKind.STRUCTURED,
    conditioning_statement="Correction is evaluated in the normalized coordinate used by the fit.",
    truncation_statement="No spectral point is truncated; extrapolated points are explicitly marked.",
    capacity_semantics="Point capacity is inherited unchanged from the source spectrum.",
    assumptions=(
        "Calibration coefficients apply to the spectrum unit and acquisition regime.",
    ),
    nondifferentiable_outputs=("status", "evidence", "extrapolated_mask"),
)

_BIN_CONTRACT = BioinformaticsMethodContract(
    "bounded mass spectrum binning",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.ALMOST_EVERYWHERE,
    OutputKind.STRUCTURED,
    conditioning_statement="Bins are left-closed/right-open except that the final upper edge is included.",
    truncation_statement="Points outside the edge interval are excluded and reported; no in-range point is truncated.",
    capacity_semantics="Output capacity is exactly one less than the fixed edge capacity.",
    assumptions=("Bin edges are finite and strictly increasing.",),
    nondifferentiable_outputs=("bin_count", "status", "evidence", "out_of_range_mask"),
)


class MassCalibrationModel(StrictModule):
    """Polynomial additive mass correction over an explicit validity interval."""

    coefficients: Array
    mass_min: Array
    mass_max: Array
    residual_rms: Array
    degree: int = eqx.field(static=True)
    mass_to_charge_unit: MassToChargeUnit = eqx.field(static=True)


def mass_calibration_model(
    coefficients: ArrayLike,
    mass_min: float | ArrayLike,
    mass_max: float | ArrayLike,
    /,
    *,
    residual_rms: float | ArrayLike = 0.0,
    mass_to_charge_unit: MassToChargeUnit = MassToChargeUnit.MZ,
) -> MassCalibrationModel:
    """Validate and construct an explicit additive mass-calibration model."""
    values = np.asarray(coefficients)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("coefficients must be a non-empty vector.")
    if np.any(~np.isfinite(values)):
        raise ValueError("coefficients must be finite.")
    lower = np.asarray(mass_min, dtype=values.dtype)
    upper = np.asarray(mass_max, dtype=values.dtype)
    residual = np.asarray(residual_rms, dtype=values.dtype)
    if lower.shape != () or upper.shape != () or residual.shape != ():
        raise ValueError("mass bounds and residual_rms must be scalars.")
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or float(lower) <= 0.0
        or float(upper) <= float(lower)
    ):
        raise ValueError("mass_min and mass_max must define a finite positive interval.")
    if not np.isfinite(residual) or float(residual) < 0.0:
        raise ValueError("residual_rms must be finite and nonnegative.")
    return MassCalibrationModel(
        coefficients=jnp.asarray(values),
        mass_min=jnp.asarray(lower),
        mass_max=jnp.asarray(upper),
        residual_rms=jnp.asarray(residual),
        degree=int(values.size - 1),
        mass_to_charge_unit=MassToChargeUnit(mass_to_charge_unit),
    )


class CalibrationFitResult(StrictModule):
    """Diagnosed polynomial calibration fit."""

    model: MassCalibrationModel
    fitted_reference_mass: Array
    residual: Array
    calibrant_mask: Array
    sample_count: Array
    rank: Array
    condition_number: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def fit_mass_calibration(
    observed_mass_to_charge: ArrayLike,
    reference_mass_to_charge: ArrayLike,
    /,
    *,
    calibrant_mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    degree: int = 1,
    mass_to_charge_unit: MassToChargeUnit = MassToChargeUnit.MZ,
    ridge: float = 0.0,
) -> CalibrationFitResult:
    """Fit an additive polynomial correction with diagnosed rank and support."""
    observed = jnp.asarray(observed_mass_to_charge)
    reference = jnp.asarray(reference_mass_to_charge)
    if observed.ndim != 1 or reference.shape != observed.shape or observed.size == 0:
        raise ValueError("Observed and reference masses must be equal non-empty vectors.")
    if not jnp.issubdtype(observed.dtype, jnp.inexact):
        observed = observed.astype(float)
    reference = reference.astype(observed.dtype)
    polynomial_degree = int(degree)
    if polynomial_degree < 0:
        raise ValueError("degree must be nonnegative.")
    if polynomial_degree + 1 > observed.shape[0]:
        raise ValueError("degree preflight exceeds the calibrant vector capacity.")
    mask = (
        jnp.ones(observed.shape, dtype=bool)
        if calibrant_mask is None
        else jnp.asarray(calibrant_mask, dtype=bool)
    )
    if mask.shape != observed.shape:
        raise ValueError("calibrant_mask must match observed masses.")
    finite = jnp.isfinite(observed) & jnp.isfinite(reference)
    positive = (observed > 0.0) & (reference > 0.0)
    fit_mask = mask & finite & positive
    safe_observed = jnp.where(fit_mask, observed, 0.0)
    lower = jnp.min(jnp.where(fit_mask, observed, jnp.inf))
    upper = jnp.max(jnp.where(fit_mask, observed, -jnp.inf))
    finite_interval = jnp.isfinite(lower) & jnp.isfinite(upper) & (upper > lower)
    safe_lower = jnp.where(finite_interval, lower, 1.0)
    safe_upper = jnp.where(finite_interval, upper, 2.0)
    center = 0.5 * (safe_lower + safe_upper)
    half_width = 0.5 * (safe_upper - safe_lower)
    normalized = (safe_observed - center) / half_width
    design = jnp.stack(
        [normalized**power for power in range(polynomial_degree + 1)], axis=-1
    )
    target = reference - observed
    least_squares = solve_weighted_least_squares(
        design,
        target,
        mask=fit_mask,
        weights=weights,
        ridge=ridge,
        min_samples=polynomial_degree + 1,
        max_features=polynomial_degree + 1,
    )
    correction = design @ least_squares.raw_coefficients
    fitted = observed + correction
    residual = jnp.where(fit_mask, reference - fitted, 0.0)
    denominator = jnp.maximum(jnp.sum(fit_mask), 1)
    residual_rms = jnp.sqrt(jnp.sum(residual * residual) / denominator)
    valid = least_squares.valid & finite_interval
    insufficient = jnp.sum(fit_mask) < polynomial_degree + 1
    status = jnp.where(
        valid,
        int(CalibrationStatus.SUCCESS),
        jnp.where(
            insufficient | ~finite_interval,
            int(CalibrationStatus.INSUFFICIENT_CALIBRANTS),
            jnp.where(
                least_squares.rank < polynomial_degree + 1,
                int(CalibrationStatus.RANK_DEFICIENT),
                int(CalibrationStatus.NONFINITE),
            ),
        ),
    ).astype(jnp.int32)
    model = MassCalibrationModel(
        coefficients=least_squares.raw_coefficients,
        mass_min=safe_lower,
        mass_max=safe_upper,
        residual_rms=residual_rms,
        degree=polynomial_degree,
        mass_to_charge_unit=MassToChargeUnit(mass_to_charge_unit),
    )
    return CalibrationFitResult(
        model=model,
        fitted_reference_mass=jnp.where(fit_mask, fitted, 0.0),
        residual=residual,
        calibrant_mask=fit_mask,
        sample_count=least_squares.sample_count,
        rank=least_squares.rank,
        condition_number=least_squares.condition_number,
        valid=valid,
        status=status,
        evidence=jnp.where(
            valid,
            int(CalibrationEvidence.CALIBRANTS_FIT),
            int(CalibrationEvidence.NONE),
        ).astype(jnp.uint32),
        method_contract=_FIT_CONTRACT,
    )


class CalibrationResult(StrictModule):
    """Calibrated mass axis with pointwise extrapolation evidence."""

    calibrated_mass_to_charge: Array
    active_mask: Array
    extrapolated_mask: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def apply_mass_calibration(
    spectrum: MassSpectrum,
    model: MassCalibrationModel,
    /,
) -> CalibrationResult:
    """Apply a fitted additive correction while retaining extrapolation status."""
    if not isinstance(spectrum, MassSpectrum):
        raise TypeError("spectrum must be a MassSpectrum.")
    if not isinstance(model, MassCalibrationModel):
        raise TypeError("model must be a MassCalibrationModel.")
    unit_match = spectrum.units.mass_to_charge == model.mass_to_charge_unit
    mz = spectrum.mass_to_charge
    center = 0.5 * (model.mass_min + model.mass_max)
    half_width = 0.5 * (model.mass_max - model.mass_min)
    normalized = (mz - center) / half_width
    correction = jnp.zeros_like(mz)
    for coefficient in model.coefficients[::-1]:
        correction = correction * normalized + coefficient
    corrected = mz + correction
    extrapolated = spectrum.active_mask & ((mz < model.mass_min) | (mz > model.mass_max))
    finite = jnp.all(jnp.isfinite(jnp.where(spectrum.active_mask, corrected, 0.0)))
    any_extrapolated = jnp.any(extrapolated)
    valid = jnp.asarray(unit_match) & finite & ~any_extrapolated
    status = jnp.where(
        not unit_match,
        int(CalibrationStatus.UNIT_MISMATCH),
        jnp.where(
            ~finite,
            int(CalibrationStatus.NONFINITE),
            jnp.where(
                any_extrapolated,
                int(CalibrationStatus.EXTRAPOLATION),
                int(CalibrationStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    evidence = jnp.where(
        unit_match & finite,
        int(CalibrationEvidence.CALIBRATED),
        int(CalibrationEvidence.NONE),
    ).astype(jnp.uint32)
    evidence = evidence | jnp.where(
        any_extrapolated,
        int(CalibrationEvidence.EXTRAPOLATED),
        0,
    ).astype(jnp.uint32)
    return CalibrationResult(
        calibrated_mass_to_charge=jnp.where(
            spectrum.active_mask & unit_match, corrected, 0.0
        ),
        active_mask=spectrum.active_mask & unit_match,
        extrapolated_mask=extrapolated,
        valid=valid,
        status=status,
        evidence=evidence,
        method_contract=_APPLY_CONTRACT,
    )


class MassBinningPlan(StrictModule):
    """Fixed increasing mass edges with explicit boundary semantics."""

    edges: Array
    mass_to_charge_unit: MassToChargeUnit = eqx.field(static=True)
    bin_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        edges: ArrayLike,
        /,
        *,
        mass_to_charge_unit: MassToChargeUnit = MassToChargeUnit.MZ,
    ):
        edge_host = np.asarray(edges)
        if edge_host.ndim != 1 or edge_host.size < 2:
            raise ValueError("edges must contain at least two values.")
        if not np.issubdtype(edge_host.dtype, np.inexact):
            edge_host = edge_host.astype(float)
        if np.any(~np.isfinite(edge_host)) or np.any(edge_host <= 0.0):
            raise ValueError("Bin edges must be finite and positive.")
        if np.any(np.diff(edge_host) <= 0.0):
            raise ValueError("Bin edges must be strictly increasing.")
        self.edges = jnp.asarray(edge_host)
        self.mass_to_charge_unit = MassToChargeUnit(mass_to_charge_unit)
        self.bin_capacity = int(edge_host.size - 1)


class BinnedSpectrumResult(StrictModule):
    """Fixed-capacity binned intensity and occupancy."""

    intensity: Array
    point_count: Array
    out_of_range_mask: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def bin_mass_spectrum(
    spectrum: MassSpectrum,
    plan: MassBinningPlan,
    /,
) -> BinnedSpectrumResult:
    """Bin one spectrum with exact, documented edge ownership."""
    if not isinstance(spectrum, MassSpectrum):
        raise TypeError("spectrum must be a MassSpectrum.")
    if not isinstance(plan, MassBinningPlan):
        raise TypeError("plan must be a MassBinningPlan.")
    unit_match = spectrum.units.mass_to_charge == plan.mass_to_charge_unit
    mz = spectrum.mass_to_charge
    in_range = spectrum.active_mask & (mz >= plan.edges[0]) & (mz <= plan.edges[-1])
    raw_index = jnp.searchsorted(plan.edges, mz, side="right") - 1
    bin_index = jnp.clip(raw_index, 0, plan.bin_capacity - 1)
    contributions = jnp.where(in_range & unit_match, spectrum.intensity, 0.0)
    counts = (in_range & unit_match).astype(jnp.int32)
    binned = (
        jnp.zeros((plan.bin_capacity,), dtype=spectrum.intensity.dtype)
        .at[bin_index]
        .add(contributions)
    )
    point_count = (
        jnp.zeros((plan.bin_capacity,), dtype=jnp.int32).at[bin_index].add(counts)
    )
    outside = spectrum.active_mask & ~in_range
    any_outside = jnp.any(outside)
    valid = jnp.asarray(unit_match) & ~any_outside
    status = jnp.where(
        not unit_match,
        int(CalibrationStatus.UNIT_MISMATCH),
        jnp.where(
            any_outside,
            int(CalibrationStatus.OUT_OF_RANGE),
            int(CalibrationStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    evidence = jnp.where(
        unit_match,
        int(CalibrationEvidence.BINNED),
        int(CalibrationEvidence.NONE),
    ).astype(jnp.uint32)
    evidence = evidence | jnp.where(
        any_outside,
        int(CalibrationEvidence.OUTSIDE_BINS),
        0,
    ).astype(jnp.uint32)
    return BinnedSpectrumResult(
        intensity=binned,
        point_count=point_count,
        out_of_range_mask=outside,
        valid=valid,
        status=status,
        evidence=evidence,
        method_contract=_BIN_CONTRACT,
    )


__all__ = [
    "BinnedSpectrumResult",
    "CalibrationEvidence",
    "CalibrationFitResult",
    "CalibrationResult",
    "CalibrationStatus",
    "MassBinningPlan",
    "MassCalibrationModel",
    "apply_mass_calibration",
    "bin_mass_spectrum",
    "fit_mass_calibration",
    "mass_calibration_model",
]
