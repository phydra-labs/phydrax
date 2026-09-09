# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Reduced-form, intensity, and structural credit models.

Credit meaning remains above the generic curve and stochastic substrates.  In
particular, a survival curve is a :class:`PreparedCurve` with an explicit
survival representation, and a default clock is a native
:class:`PoissonClockRealization`.  No spread is silently interpreted as a
physical default probability.
"""

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._strict import StrictModule
from ...stochastic import JUMP_INVALID_INTENSITY, JUMP_SUCCESS, PoissonClockRealization
from ..contracts._credit import (
    CreditDefaultSwapContract,
    DefaultableBondContract,
    DefaultEventState,
    RecoveryTerms,
)
from ..core import PhysicalLaw, PricingLaw, StressLaw
from ..curves._core import CurveRepresentation, InterpolationMethod, PreparedCurve


IntensityTransformation: TypeAlias = Literal["exponential", "positive_part"]
CreditLaw: TypeAlias = PhysicalLaw | PricingLaw | StressLaw


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty string.")
    return value


def _scalar(
    value: ArrayLike,
    name: str,
    /,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Array:
    result = jnp.asarray(value, dtype=float)
    if result.shape != ():
        raise ValueError(f"{name} must be scalar.")
    host = float(np.asarray(jax.device_get(result)))
    if not isfinite(host):
        raise ValueError(f"{name} must be finite.")
    if positive and host <= 0.0:
        raise ValueError(f"{name} must be positive.")
    if nonnegative and host < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _strict_times(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value, dtype=float)
    host = np.asarray(jax.device_get(result))
    if result.ndim != 1 or result.shape[0] < 2:
        raise ValueError(f"{name} must be a vector with at least two entries.")
    if not np.all(np.isfinite(host)) or host[0] < 0.0 or np.any(np.diff(host) <= 0.0):
        raise ValueError(f"{name} must be nonnegative, finite, and strictly increasing.")
    return result


def _require_law(
    law: CreditLaw,
    /,
    *,
    factor_layout_id: str,
    pricing_measure_id: str | None = None,
    pricing_only: bool = False,
) -> None:
    if pricing_only and not isinstance(law, PricingLaw):
        raise TypeError(
            "Credit valuation requires a PricingLaw; physical and stress laws are invalid."
        )
    if not isinstance(law, (PhysicalLaw, PricingLaw, StressLaw)):
        raise TypeError("law must be a PhysicalLaw, PricingLaw, or StressLaw.")
    if law.factor_layout_id != factor_layout_id:
        raise ValueError("Law factor layout is incompatible with the credit model.")
    if pricing_measure_id is not None:
        if not isinstance(law, PricingLaw):
            raise TypeError("A pricing measure can only be bound by a PricingLaw.")
        if law.measure_id != pricing_measure_id:
            raise ValueError("Pricing-law measure is incompatible with the credit model.")


def _require_survival_curve(curve: PreparedCurve, /) -> None:
    if not isinstance(curve, PreparedCurve):
        raise TypeError("survival_curve must be a PreparedCurve.")
    if curve.definition.role != "survival":
        raise ValueError("Credit survival curves must have the explicit 'survival' role.")
    if curve.definition.representation not in (
        CurveRepresentation.LOG_SURVIVAL,
        CurveRepresentation.HAZARD_RATE,
    ):
        raise ValueError(
            "Credit survival requires log-survival or hazard-rate curve semantics."
        )


class ReducedFormCreditModel(StrictModule):
    """Deterministic survival term structure and recovery semantics under Q."""

    survival_curve: PreparedCurve
    recovery: RecoveryTerms
    reference_entity_id: str = eqx.field(static=True)
    factor_layout_id: str = eqx.field(static=True)
    pricing_measure_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    default_process_id: str = eqx.field(static=True)

    def __init__(
        self,
        survival_curve: PreparedCurve,
        recovery: RecoveryTerms,
        /,
        *,
        reference_entity_id: str,
        factor_layout_id: str,
        pricing_measure_id: str,
        model_id: str,
        default_process_id: str,
    ):
        _require_survival_curve(survival_curve)
        if not isinstance(recovery, RecoveryTerms):
            raise TypeError("recovery must be RecoveryTerms.")
        self.survival_curve = survival_curve
        self.recovery = recovery
        self.reference_entity_id = _identifier(reference_entity_id, "reference_entity_id")
        self.factor_layout_id = _identifier(factor_layout_id, "factor_layout_id")
        self.pricing_measure_id = _identifier(pricing_measure_id, "pricing_measure_id")
        self.model_id = _identifier(model_id, "model_id")
        self.default_process_id = _identifier(default_process_id, "default_process_id")


class IntensityCreditModel(StrictModule):
    """State-dependent intensity over an explicit deterministic base curve."""

    base: ReducedFormCreditModel
    factor_loadings: Array
    transformation: IntensityTransformation = eqx.field(static=True)
    factor_layout_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: ReducedFormCreditModel,
        factor_loadings: ArrayLike,
        /,
        *,
        transformation: IntensityTransformation,
        factor_layout_id: str,
        model_id: str,
    ):
        if not isinstance(base, ReducedFormCreditModel):
            raise TypeError("base must be a ReducedFormCreditModel.")
        loadings = jnp.asarray(factor_loadings, dtype=float)
        host = np.asarray(jax.device_get(loadings))
        if loadings.ndim != 1 or loadings.shape[0] == 0 or not np.all(np.isfinite(host)):
            raise ValueError("factor_loadings must be a non-empty finite vector.")
        if transformation not in ("exponential", "positive_part"):
            raise ValueError("Unsupported intensity transformation.")
        self.base = base
        self.factor_loadings = loadings
        self.transformation = transformation
        self.factor_layout_id = _identifier(factor_layout_id, "factor_layout_id")
        self.model_id = _identifier(model_id, "model_id")


class StructuralCreditModel(StrictModule):
    """Merton-style asset-value/barrier structure without an implicit measure."""

    initial_asset_value: Array
    debt_barrier: Array
    asset_volatility: Array
    horizon: Array
    reference_entity_id: str = eqx.field(static=True)
    factor_layout_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_asset_value: ArrayLike,
        debt_barrier: ArrayLike,
        asset_volatility: ArrayLike,
        horizon: ArrayLike,
        /,
        *,
        reference_entity_id: str,
        factor_layout_id: str,
        model_id: str,
    ):
        self.initial_asset_value = _scalar(
            initial_asset_value, "initial_asset_value", positive=True
        )
        self.debt_barrier = _scalar(debt_barrier, "debt_barrier", positive=True)
        self.asset_volatility = _scalar(
            asset_volatility, "asset_volatility", positive=True
        )
        self.horizon = _scalar(horizon, "horizon", positive=True)
        self.reference_entity_id = _identifier(reference_entity_id, "reference_entity_id")
        self.factor_layout_id = _identifier(factor_layout_id, "factor_layout_id")
        self.model_id = _identifier(model_id, "model_id")


def survival_probability(
    model: ReducedFormCreditModel,
    law: PricingLaw,
    times: ArrayLike,
    /,
) -> Array:
    """Pricing-measure survival probability from the native curve substrate."""

    if not isinstance(model, ReducedFormCreditModel):
        raise TypeError("model must be a ReducedFormCreditModel.")
    _require_law(
        law,
        factor_layout_id=model.factor_layout_id,
        pricing_measure_id=model.pricing_measure_id,
        pricing_only=True,
    )
    return model.survival_curve.survival_probability(times)


def hazard_rate(
    model: ReducedFormCreditModel,
    law: PricingLaw,
    times: ArrayLike,
    /,
) -> Array:
    """Pricing-measure hazard from the native curve substrate."""

    if not isinstance(model, ReducedFormCreditModel):
        raise TypeError("model must be a ReducedFormCreditModel.")
    _require_law(
        law,
        factor_layout_id=model.factor_layout_id,
        pricing_measure_id=model.pricing_measure_id,
        pricing_only=True,
    )
    return model.survival_curve.hazard_rate(times)


def default_probability(
    model: ReducedFormCreditModel,
    law: PricingLaw,
    start: ArrayLike,
    end: ArrayLike,
    /,
) -> Array:
    """Unconditional pricing-measure default probability in ``(start, end]``."""

    left = jnp.asarray(start, dtype=float)
    right = jnp.asarray(end, dtype=left.dtype)
    right = eqx.error_if(right, jnp.any(right < left), "end must not precede start.")
    return survival_probability(model, law, left) - survival_probability(
        model, law, right
    )


def intensity_from_factors(
    model: IntensityCreditModel,
    law: PricingLaw,
    times: ArrayLike,
    factor_values: ArrayLike,
    /,
) -> Array:
    """Evaluate Q-intensity using the model's base survival curve."""

    if not isinstance(model, IntensityCreditModel):
        raise TypeError("model must be an IntensityCreditModel.")
    _require_law(
        law,
        factor_layout_id=model.factor_layout_id,
        pricing_measure_id=model.base.pricing_measure_id,
        pricing_only=True,
    )
    query = jnp.asarray(times, dtype=float)
    if query.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    factors = jnp.asarray(factor_values, dtype=float)
    if factors.shape[-2:] != (query.shape[0], model.factor_loadings.shape[0]):
        raise ValueError("factor_values must end in (time, factor) axes.")
    base = model.base.survival_curve.hazard_rate(query)
    shock = ein.contract("...tf,f->...t", factors, model.factor_loadings)
    if model.transformation == "exponential":
        return base * jnp.exp(shock)
    return jnp.maximum(base + shock, 0.0)


def state_dependent_intensity(
    model: IntensityCreditModel,
    law: CreditLaw,
    base_intensity: ArrayLike,
    factor_values: ArrayLike,
    /,
) -> Array:
    """Apply intensity factors to an explicit base path under P, Q, or stress.

    Supplying the base path is mandatory outside Q, preventing a calibrated
    pricing hazard from being silently reused as a physical default law.
    """

    if not isinstance(model, IntensityCreditModel):
        raise TypeError("model must be an IntensityCreditModel.")
    _require_law(law, factor_layout_id=model.factor_layout_id)
    base = jnp.asarray(base_intensity, dtype=float)
    factors = jnp.asarray(factor_values, dtype=float)
    if (
        factors.shape[:-1] != base.shape
        or factors.shape[-1] != model.factor_loadings.shape[0]
    ):
        raise ValueError(
            "factor_values must have base_intensity.shape + (factor_count,)."
        )
    base = eqx.error_if(
        base,
        jnp.any(~jnp.isfinite(base) | (base < 0.0)),
        "base_intensity must be finite and nonnegative.",
    )
    shock = ein.contract("...f,f->...", factors, model.factor_loadings)
    if model.transformation == "exponential":
        return base * jnp.exp(shock)
    return jnp.maximum(base + shock, 0.0)


def structural_default_probability(
    model: StructuralCreditModel,
    law: PhysicalLaw | PricingLaw,
    asset_drift: ArrayLike,
    /,
) -> Array:
    """Merton terminal default probability under an explicit law and drift."""

    if not isinstance(model, StructuralCreditModel):
        raise TypeError("model must be a StructuralCreditModel.")
    if not isinstance(law, (PhysicalLaw, PricingLaw)):
        raise TypeError(
            "Structural default probability requires a PhysicalLaw or PricingLaw."
        )
    _require_law(law, factor_layout_id=model.factor_layout_id)
    drift = _scalar(asset_drift, "asset_drift")
    sigma = model.asset_volatility
    horizon = model.horizon
    d2 = (
        jnp.log(model.initial_asset_value / model.debt_barrier)
        + (drift - 0.5 * sigma**2) * horizon
    ) / (sigma * jnp.sqrt(horizon))
    return jsp.special.ndtr(-d2)


def simulate_default_events(
    model: ReducedFormCreditModel,
    law: PricingLaw,
    realization: PoissonClockRealization,
    /,
) -> DefaultEventState:
    """Invert Q-survival against native Poisson-clock thresholds."""

    if not isinstance(model, ReducedFormCreditModel):
        raise TypeError("model must be a ReducedFormCreditModel.")
    if not isinstance(realization, PoissonClockRealization):
        raise TypeError("realization must be a PoissonClockRealization.")
    _require_law(
        law,
        factor_layout_id=model.factor_layout_id,
        pricing_measure_id=model.pricing_measure_id,
        pricing_only=True,
    )
    if realization.process_id != model.default_process_id:
        raise ValueError(
            "Poisson process identity is incompatible with the credit model."
        )
    if realization.num_channels != 1:
        raise ValueError(
            "Single-name default simulation requires exactly one jump channel."
        )
    grid = model.survival_curve.definition.grid.times
    if realization.support != (float(grid[0]), float(grid[-1])):
        raise ValueError("Poisson-clock support must equal the survival-curve support.")
    interpolation = model.survival_curve.definition.interpolation.method
    representation = model.survival_curve.definition.representation
    if (
        representation is CurveRepresentation.HAZARD_RATE
        and interpolation is not InterpolationMethod.STEP_LEFT
    ) or (
        representation is CurveRepresentation.LOG_SURVIVAL
        and interpolation is not InterpolationMethod.LINEAR
    ):
        raise ValueError(
            "Default inversion requires step-left hazard or linear log-survival interpolation."
        )
    cumulative_hazard = -model.survival_curve.log_survival(grid)
    thresholds = realization.thresholds[..., 0, 0].reshape((-1,))
    occurred = thresholds <= cumulative_hazard[-1]
    interval_index = jnp.searchsorted(cumulative_hazard[1:], thresholds, side="left")
    safe_index = jnp.minimum(interval_index, grid.shape[0] - 2)
    hazard_increment = cumulative_hazard[safe_index + 1] - cumulative_hazard[safe_index]
    time_width = grid[safe_index + 1] - grid[safe_index]
    local_hazard = hazard_increment / time_width
    offset = jnp.where(
        local_hazard > 0.0,
        (thresholds - cumulative_hazard[safe_index]) / local_hazard,
        0.0,
    )
    default_time = jnp.where(occurred, grid[safe_index] + offset, 0.0)
    valid = jnp.isfinite(thresholds) & (~occurred | jnp.isfinite(default_time))
    status = jnp.where(valid, JUMP_SUCCESS, JUMP_INVALID_INTENSITY).astype(jnp.int32)
    return DefaultEventState(
        default_time,
        occurred,
        jnp.where(occurred, model.recovery.rate, 0.0),
        valid,
        status,
        reference_entity_id=model.reference_entity_id,
        law_id=law.law_id,
        realization_id=realization.realization_id,
        recovery_terms_id=model.recovery.terms_id,
        coupling_id=realization.coupling_id,
    )


def default_events_from_intensity_paths(
    times: ArrayLike,
    intensities: ArrayLike,
    law: CreditLaw,
    recovery: RecoveryTerms,
    realization: PoissonClockRealization,
    /,
    *,
    reference_entity_id: str,
    factor_layout_id: str,
) -> DefaultEventState:
    """Invert P-, Q-, or stress-law piecewise-constant intensity paths."""

    _require_law(law, factor_layout_id=factor_layout_id)
    if not isinstance(recovery, RecoveryTerms):
        raise TypeError("recovery must be RecoveryTerms.")
    if not isinstance(realization, PoissonClockRealization):
        raise TypeError("realization must be a PoissonClockRealization.")
    nodes = _strict_times(times, "times")
    rates = jnp.asarray(intensities, dtype=float)
    path_count = realization.num_paths
    if rates.shape != (path_count, nodes.shape[0] - 1):
        raise ValueError("intensities must have shape (path, time_interval).")
    if realization.num_channels != 1:
        raise ValueError(
            "Single-name default simulation requires exactly one jump channel."
        )
    if realization.support != (float(nodes[0]), float(nodes[-1])):
        raise ValueError("Poisson-clock support must equal the intensity-path support.")
    rates = eqx.error_if(
        rates,
        jnp.any(~jnp.isfinite(rates) | (rates < 0.0)),
        "Intensity paths must be finite and nonnegative.",
    )
    thresholds = realization.thresholds[..., 0, 0].reshape((-1,))
    cumulative = jnp.cumsum(rates * jnp.diff(nodes)[None, :], axis=-1)
    occurred = thresholds <= cumulative[:, -1]
    interval_index = jnp.sum(cumulative < thresholds[:, None], axis=-1)
    safe_index = jnp.minimum(interval_index, rates.shape[1] - 1)
    cumulative_before = jnp.where(
        safe_index == 0,
        0.0,
        jnp.take_along_axis(cumulative, (safe_index - 1)[:, None], axis=1)[:, 0],
    )
    selected_rate = jnp.take_along_axis(rates, safe_index[:, None], axis=1)[:, 0]
    offset = jnp.where(
        selected_rate > 0.0,
        (thresholds - cumulative_before) / selected_rate,
        0.0,
    )
    default_time = jnp.where(occurred, nodes[safe_index] + offset, 0.0)
    valid = jnp.all(jnp.isfinite(rates), axis=-1) & jnp.isfinite(thresholds)
    status = jnp.where(valid, JUMP_SUCCESS, JUMP_INVALID_INTENSITY).astype(jnp.int32)
    return DefaultEventState(
        default_time,
        occurred,
        jnp.where(occurred, recovery.rate, 0.0),
        valid,
        status,
        reference_entity_id=_identifier(reference_entity_id, "reference_entity_id"),
        law_id=law.law_id,
        realization_id=realization.realization_id,
        recovery_terms_id=recovery.terms_id,
        coupling_id=realization.coupling_id,
    )


def _face_units(amount, /) -> Array:
    return amount.atoms.astype(float) / float(amount.currency.atoms_per_unit)


def _expected_credit_inputs(
    contract: DefaultableBondContract | CreditDefaultSwapContract,
    credit: ReducedFormCreditModel,
    discount_curve: PreparedCurve,
    law: PricingLaw,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    if not isinstance(credit, ReducedFormCreditModel):
        raise TypeError("credit must be a ReducedFormCreditModel.")
    if not isinstance(discount_curve, PreparedCurve):
        raise TypeError("discount_curve must be a PreparedCurve.")
    if discount_curve.definition.role != "discount":
        raise ValueError("Credit cashflows require an explicit discount-role curve.")
    if contract.payoff.reference_entity_id != credit.reference_entity_id:
        raise ValueError("Contract payoff and credit model reference entities differ.")
    if contract.recovery.terms_id != credit.recovery.terms_id:
        raise ValueError("Contract and credit model recovery terms differ.")
    curve_currency = discount_curve.definition.currency
    if (
        curve_currency is None
        or curve_currency.currency_id
        != contract.instrument.settlement_currency.currency_id
    ):
        raise ValueError(
            "Discount curve currency must match contract settlement currency."
        )
    if contract.recovery.timing != "period_end":
        raise ValueError("Expected credit cashflows require period-end default timing.")
    settlement_lag = eqx.error_if(
        contract.recovery.settlement_lag,
        contract.recovery.settlement_lag != 0.0,
        "Expected credit cashflows require zero settlement lag.",
    )
    valid = contract.schedule.valid
    times = contract.payment_times + settlement_lag
    starts = jnp.where(
        valid,
        jnp.concatenate((jnp.zeros((1,), dtype=times.dtype), times[:-1])),
        0.0,
    )
    survival = survival_probability(credit, law, times)
    defaults = default_probability(credit, law, starts, times)
    discounts = discount_curve.discount_factor(times)
    return valid, times, survival, defaults, discounts


class DefaultableBondCashflows(StrictModule):
    """Expected promised/recovery cashflow ledger and present values."""

    payment_times: Array
    promised_if_surviving: Array
    expected_recovery: Array
    discounted_promised: Array
    discounted_recovery: Array
    valid: Array
    contract_id: str = eqx.field(static=True)
    credit_model_id: str = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)
    pricing_law_id: str = eqx.field(static=True)

    @property
    def present_value(self) -> Array:
        return jnp.sum(self.discounted_promised + self.discounted_recovery)


class CDSLegValues(StrictModule):
    """Separated CDS premium, accrued-premium, and protection ledgers."""

    payment_times: Array
    premium_cashflows: Array
    accrued_premium_cashflows: Array
    protection_cashflows: Array
    discounted_premium: Array
    discounted_accrued_premium: Array
    discounted_protection: Array
    valid: Array
    contract_id: str = eqx.field(static=True)
    credit_model_id: str = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)
    pricing_law_id: str = eqx.field(static=True)
    protection_side: str = eqx.field(static=True)

    @property
    def premium_leg(self) -> Array:
        return jnp.sum(self.discounted_premium + self.discounted_accrued_premium)

    @property
    def protection_leg(self) -> Array:
        return jnp.sum(self.discounted_protection)

    @property
    def buyer_value(self) -> Array:
        return self.protection_leg - self.premium_leg

    @property
    def contract_value(self) -> Array:
        return self.buyer_value if self.protection_side == "buy" else -self.buyer_value


def defaultable_bond_cashflows(
    contract: DefaultableBondContract,
    credit: ReducedFormCreditModel,
    discount_curve: PreparedCurve,
    law: PricingLaw,
    /,
) -> DefaultableBondCashflows:
    """Expected recovery-of-par bond cashflows under explicit Q and discounting."""

    if not isinstance(contract, DefaultableBondContract):
        raise TypeError("contract must be a DefaultableBondContract.")
    if contract.recovery.convention != "par":
        raise ValueError(
            "Bond expected cashflows currently support recovery of par only."
        )
    valid, times, survival, defaults, discounts = _expected_credit_inputs(
        contract, credit, discount_curve, law
    )
    promised = jnp.where(valid, contract.promised_amounts * survival, 0.0)
    recovered = jnp.where(
        valid,
        _face_units(contract.face) * contract.recovery.rate * defaults,
        0.0,
    )
    return DefaultableBondCashflows(
        times,
        promised,
        recovered,
        promised * discounts,
        recovered * discounts,
        valid,
        contract.contract_id,
        credit.model_id,
        discount_curve.definition.curve_id,
        law.law_id,
    )


def cds_leg_values(
    contract: CreditDefaultSwapContract,
    credit: ReducedFormCreditModel,
    discount_curve: PreparedCurve,
    law: PricingLaw,
    /,
) -> CDSLegValues:
    """Expected running-premium and protection legs on the resolved schedule."""

    if not isinstance(contract, CreditDefaultSwapContract):
        raise TypeError("contract must be a CreditDefaultSwapContract.")
    valid, times, survival, defaults, discounts = _expected_credit_inputs(
        contract, credit, discount_curve, law
    )
    notional = _face_units(contract.notional)
    premium = jnp.where(
        valid,
        notional * contract.running_spread * contract.schedule.year_fractions * survival,
        0.0,
    )
    accrued = jnp.where(
        valid & contract.accrued_on_default,
        0.5
        * notional
        * contract.running_spread
        * contract.schedule.year_fractions
        * defaults,
        0.0,
    )
    protection = jnp.where(
        valid,
        notional * (1.0 - contract.recovery.rate) * defaults,
        0.0,
    )
    return CDSLegValues(
        times,
        premium,
        accrued,
        protection,
        premium * discounts,
        accrued * discounts,
        protection * discounts,
        valid,
        contract.contract_id,
        credit.model_id,
        discount_curve.definition.curve_id,
        law.law_id,
        contract.protection_side,
    )


def cds_par_spread(
    contract: CreditDefaultSwapContract,
    credit: ReducedFormCreditModel,
    discount_curve: PreparedCurve,
    law: PricingLaw,
    /,
) -> Array:
    """Running spread equating the resolved premium and protection conventions."""

    if not isinstance(contract, CreditDefaultSwapContract):
        raise TypeError("contract must be a CreditDefaultSwapContract.")
    valid, _, survival, defaults, discounts = _expected_credit_inputs(
        contract, credit, discount_curve, law
    )
    annuity = contract.schedule.year_fractions * survival
    if contract.accrued_on_default:
        annuity = annuity + 0.5 * contract.schedule.year_fractions * defaults
    premium_annuity = jnp.sum(jnp.where(valid, discounts * annuity, 0.0))
    premium_annuity = eqx.error_if(
        premium_annuity,
        premium_annuity <= 0.0,
        "CDS premium annuity must be positive.",
    )
    protection = jnp.sum(
        jnp.where(
            valid,
            discounts * (1.0 - contract.recovery.rate) * defaults,
            0.0,
        )
    )
    return protection / premium_annuity


__all__ = [
    "CDSLegValues",
    "CreditLaw",
    "DefaultableBondCashflows",
    "IntensityCreditModel",
    "IntensityTransformation",
    "ReducedFormCreditModel",
    "StructuralCreditModel",
    "cds_leg_values",
    "cds_par_spread",
    "default_events_from_intensity_paths",
    "default_probability",
    "defaultable_bond_cashflows",
    "hazard_rate",
    "intensity_from_factors",
    "simulate_default_events",
    "state_dependent_intensity",
    "structural_default_probability",
    "survival_probability",
]
