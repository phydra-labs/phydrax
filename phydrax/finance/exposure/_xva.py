# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Separated CVA, DVA, FVA, MVA, and economic KVA computations."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._pathwise import ExposureProfile


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty string.")
    return value


def _profile_id(profile: ExposureProfile, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "exposure-profile",
            "plan_id": profile.plan_id,
            "value_state_id": profile.value_state_id,
            "weighting_id": profile.weighting_id,
            "wrong_way_link_id": profile.wrong_way_link_id,
            "time_count": int(profile.times.shape[0]),
        }
    )


def _vector(
    value: ArrayLike,
    shape: tuple[int, ...],
    name: str,
    /,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> Array:
    result = jnp.asarray(value, dtype=float)
    if result.shape == ():
        result = jnp.broadcast_to(result, shape)
    if result.shape != shape:
        raise ValueError(f"{name} must be scalar or have shape {shape}.")
    host = np.asarray(jax.device_get(result))
    if not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must be finite.")
    if nonnegative and np.any(host < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    if positive and np.any(host <= 0.0):
        raise ValueError(f"{name} must be positive.")
    return result


def _recovery(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value, dtype=float)
    if result.shape != ():
        raise ValueError(f"{name} must be scalar.")
    host = float(np.asarray(jax.device_get(result)))
    if not isfinite(host) or host < 0.0 or host > 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return result


def _discounts(value: ArrayLike, shape: tuple[int, ...], /) -> Array:
    return _vector(value, shape, "discount_factors", positive=True)


def _default_increments(value: ArrayLike, shape: tuple[int, ...], name: str, /) -> Array:
    result = _vector(value, shape, name, nonnegative=True)
    if float(np.sum(np.asarray(jax.device_get(result)))) > 1.0 + 1.0e-7:
        raise ValueError(f"{name} must have total mass no greater than one.")
    return result


def _time_integral(values: Array, times: Array, valid: Array, /) -> Array:
    """Integrate bucket-end values over the explicit exposure time grid."""

    widths = jnp.diff(times, prepend=times[0])
    active = valid & (widths > 0.0)
    return jnp.sum(jnp.where(active, values * widths, 0.0))


class FundingPolicy(StrictModule):
    """Treasury funding spreads; no accounting or liquidity inference."""

    borrowing_spreads: Array
    lending_spreads: Array
    policy_id: str = eqx.field(static=True)
    funding_curve_id: str = eqx.field(static=True)

    def __init__(
        self,
        borrowing_spreads: ArrayLike,
        lending_spreads: ArrayLike,
        /,
        *,
        policy_id: str,
        funding_curve_id: str,
    ):
        borrowing = jnp.asarray(borrowing_spreads, dtype=float)
        lending = jnp.asarray(lending_spreads, dtype=float)
        if (
            borrowing.ndim != 1
            or lending.shape != borrowing.shape
            or borrowing.shape[0] < 2
        ):
            raise ValueError(
                "Funding spreads must be equal vectors with at least two entries."
            )
        host_borrow = np.asarray(jax.device_get(borrowing))
        host_lend = np.asarray(jax.device_get(lending))
        if (
            not np.all(np.isfinite(host_borrow))
            or not np.all(np.isfinite(host_lend))
            or np.any(host_borrow < 0.0)
            or np.any(host_lend < 0.0)
        ):
            raise ValueError("Funding spreads must be finite and nonnegative.")
        self.borrowing_spreads = borrowing
        self.lending_spreads = lending
        self.policy_id = _identifier(policy_id, "policy_id")
        self.funding_curve_id = _identifier(funding_curve_id, "funding_curve_id")


class MarginFundingPolicy(StrictModule):
    """Initial-margin funding spreads supplied independently of exposure."""

    funding_spreads: Array
    policy_id: str = eqx.field(static=True)
    funding_curve_id: str = eqx.field(static=True)

    def __init__(
        self,
        funding_spreads: ArrayLike,
        /,
        *,
        funding_curve_id: str,
        policy_id: str,
    ):
        spreads = jnp.asarray(funding_spreads, dtype=float)
        host = np.asarray(jax.device_get(spreads))
        if spreads.ndim != 1 or spreads.shape[0] < 2:
            raise ValueError(
                "Margin funding spreads must be a vector with at least two entries."
            )
        if not np.all(np.isfinite(host)) or np.any(host < 0.0):
            raise ValueError("Margin funding spreads must be finite and nonnegative.")
        self.funding_spreads = spreads
        self.policy_id = _identifier(policy_id, "policy_id")
        self.funding_curve_id = _identifier(funding_curve_id, "funding_curve_id")


class EconomicCapitalPolicy(StrictModule):
    """Caller-supplied economic capital and cost; never a regulatory-capital claim."""

    capital_profile: Array
    cost_of_capital: Array
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        capital_profile: ArrayLike,
        cost_of_capital: ArrayLike,
        /,
        *,
        policy_id: str,
    ):
        capital = jnp.asarray(capital_profile, dtype=float)
        cost = jnp.asarray(cost_of_capital, dtype=float)
        if (
            capital.ndim != 1
            or capital.shape[0] < 2
            or cost.shape not in ((), capital.shape)
        ):
            raise ValueError(
                "Capital must be a vector and cost_of_capital scalar or aligned."
            )
        cost = jnp.broadcast_to(cost, capital.shape)
        host_capital = np.asarray(jax.device_get(capital))
        host_cost = np.asarray(jax.device_get(cost))
        if (
            not np.all(np.isfinite(host_capital))
            or not np.all(np.isfinite(host_cost))
            or np.any(host_capital < 0.0)
            or np.any(host_cost < 0.0)
        ):
            raise ValueError(
                "Economic capital and its cost must be finite and nonnegative."
            )
        self.capital_profile = capital
        self.cost_of_capital = cost
        self.policy_id = _identifier(policy_id, "policy_id")


class CVAResult(StrictModule):
    """Counterparty-credit adjustment, negative from the holder perspective."""

    bucket_contributions: Array
    adjustment: Array
    exposure_profile_id: str = eqx.field(static=True)
    counterparty_default_law_id: str = eqx.field(static=True)
    recovery_terms_id: str = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)


class DVAResult(StrictModule):
    """Own-credit adjustment, positive from the holder perspective."""

    bucket_contributions: Array
    adjustment: Array
    exposure_profile_id: str = eqx.field(static=True)
    own_default_law_id: str = eqx.field(static=True)
    recovery_terms_id: str = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)


class FVAResult(StrictModule):
    """Funding adjustment with borrowing costs and lending benefits separated by sign."""

    borrowing_contributions: Array
    lending_contributions: Array
    adjustment: Array
    exposure_profile_id: str = eqx.field(static=True)
    funding_policy_id: str = eqx.field(static=True)
    funding_curve_id: str = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)


class MVAResult(StrictModule):
    """Initial-margin funding cost, negative from the holder perspective."""

    bucket_contributions: Array
    adjustment: Array
    exposure_profile_id: str = eqx.field(static=True)
    margin_policy_id: str = eqx.field(static=True)
    initial_margin_profile_id: str = eqx.field(static=True)
    funding_curve_id: str = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)


class KVAResult(StrictModule):
    """Economic capital-cost adjustment; this is not regulatory KVA."""

    bucket_contributions: Array
    adjustment: Array
    exposure_profile_id: str = eqx.field(static=True)
    economic_capital_policy_id: str = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)


class XVAResult(StrictModule):
    """Auditable sum of five independently computed adjustment components."""

    cva: CVAResult
    dva: DVAResult
    fva: FVAResult
    mva: MVAResult
    kva: KVAResult
    total_adjustment: Array
    decomposition_residual: Array
    result_id: str = eqx.field(static=True)


def compute_cva(
    profile: ExposureProfile,
    counterparty_default_increments: ArrayLike,
    recovery_rate: ArrayLike,
    discount_factors: ArrayLike,
    /,
    *,
    counterparty_default_law_id: str,
    recovery_terms_id: str,
    discount_curve_id: str,
) -> CVAResult:
    """Compute unilateral CVA as a negative value adjustment."""

    if not isinstance(profile, ExposureProfile):
        raise TypeError("profile must be an ExposureProfile.")
    shape = profile.times.shape
    increments = _default_increments(
        counterparty_default_increments, shape, "counterparty_default_increments"
    )
    recovery = _recovery(recovery_rate, "counterparty recovery_rate")
    discounts = _discounts(discount_factors, shape)
    buckets = (
        -(1.0 - recovery) * discounts * profile.expected_positive_exposure * increments
    )
    return CVAResult(
        buckets,
        jnp.sum(buckets),
        _profile_id(profile),
        _identifier(counterparty_default_law_id, "counterparty_default_law_id"),
        _identifier(recovery_terms_id, "recovery_terms_id"),
        _identifier(discount_curve_id, "discount_curve_id"),
    )


def compute_dva(
    profile: ExposureProfile,
    own_default_increments: ArrayLike,
    recovery_rate: ArrayLike,
    discount_factors: ArrayLike,
    /,
    *,
    own_default_law_id: str,
    recovery_terms_id: str,
    discount_curve_id: str,
) -> DVAResult:
    """Compute own-default DVA as a positive value adjustment."""

    if not isinstance(profile, ExposureProfile):
        raise TypeError("profile must be an ExposureProfile.")
    shape = profile.times.shape
    increments = _default_increments(
        own_default_increments, shape, "own_default_increments"
    )
    recovery = _recovery(recovery_rate, "own recovery_rate")
    discounts = _discounts(discount_factors, shape)
    buckets = (
        (1.0 - recovery) * discounts * profile.expected_negative_exposure * increments
    )
    return DVAResult(
        buckets,
        jnp.sum(buckets),
        _profile_id(profile),
        _identifier(own_default_law_id, "own_default_law_id"),
        _identifier(recovery_terms_id, "recovery_terms_id"),
        _identifier(discount_curve_id, "discount_curve_id"),
    )


def compute_fva(
    profile: ExposureProfile,
    policy: FundingPolicy,
    discount_factors: ArrayLike,
    /,
    *,
    discount_curve_id: str,
) -> FVAResult:
    """Compute borrowing cost minus lending benefit on collateralized exposure."""

    if not isinstance(profile, ExposureProfile):
        raise TypeError("profile must be an ExposureProfile.")
    if not isinstance(policy, FundingPolicy):
        raise TypeError("policy must be a FundingPolicy.")
    shape = profile.times.shape
    if policy.borrowing_spreads.shape != shape:
        raise ValueError("Funding policy must align with the exposure time grid.")
    discount_id = _identifier(discount_curve_id, "discount_curve_id")
    if policy.funding_curve_id == discount_id:
        raise ValueError("Funding and discount curves must have distinct roles and IDs.")
    discounts = _discounts(discount_factors, shape)
    borrowing = -discounts * policy.borrowing_spreads * profile.expected_positive_exposure
    lending = discounts * policy.lending_spreads * profile.expected_negative_exposure
    valid = jnp.ones(shape, dtype=bool)
    adjustment = _time_integral(borrowing + lending, profile.times, valid)
    return FVAResult(
        borrowing,
        lending,
        adjustment,
        _profile_id(profile),
        policy.policy_id,
        policy.funding_curve_id,
        discount_id,
    )


def compute_mva(
    profile: ExposureProfile,
    expected_initial_margin: ArrayLike,
    policy: MarginFundingPolicy,
    discount_factors: ArrayLike,
    /,
    *,
    initial_margin_profile_id: str,
    discount_curve_id: str,
) -> MVAResult:
    """Compute initial-margin funding cost without inferring a margin methodology."""

    if not isinstance(profile, ExposureProfile):
        raise TypeError("profile must be an ExposureProfile.")
    if not isinstance(policy, MarginFundingPolicy):
        raise TypeError("policy must be a MarginFundingPolicy.")
    shape = profile.times.shape
    if policy.funding_spreads.shape != shape:
        raise ValueError("Margin funding policy must align with exposure times.")
    discount_id = _identifier(discount_curve_id, "discount_curve_id")
    if policy.funding_curve_id == discount_id:
        raise ValueError(
            "Margin funding and discount curves must have distinct roles and IDs."
        )
    margin = _vector(
        expected_initial_margin, shape, "expected_initial_margin", nonnegative=True
    )
    discounts = _discounts(discount_factors, shape)
    buckets = -discounts * policy.funding_spreads * margin
    adjustment = _time_integral(buckets, profile.times, jnp.ones(shape, dtype=bool))
    return MVAResult(
        buckets,
        adjustment,
        _profile_id(profile),
        policy.policy_id,
        _identifier(initial_margin_profile_id, "initial_margin_profile_id"),
        policy.funding_curve_id,
        discount_id,
    )


def compute_kva(
    profile: ExposureProfile,
    policy: EconomicCapitalPolicy,
    discount_factors: ArrayLike,
    /,
    *,
    discount_curve_id: str,
) -> KVAResult:
    """Compute caller-defined economic capital cost, not regulatory capital."""

    if not isinstance(profile, ExposureProfile):
        raise TypeError("profile must be an ExposureProfile.")
    if not isinstance(policy, EconomicCapitalPolicy):
        raise TypeError("policy must be an EconomicCapitalPolicy.")
    shape = profile.times.shape
    if policy.capital_profile.shape != shape:
        raise ValueError("Economic capital policy must align with exposure times.")
    discounts = _discounts(discount_factors, shape)
    buckets = -discounts * policy.cost_of_capital * policy.capital_profile
    adjustment = _time_integral(buckets, profile.times, jnp.ones(shape, dtype=bool))
    return KVAResult(
        buckets,
        adjustment,
        _profile_id(profile),
        policy.policy_id,
        _identifier(discount_curve_id, "discount_curve_id"),
    )


def assemble_xva(
    cva: CVAResult,
    dva: DVAResult,
    fva: FVAResult,
    mva: MVAResult,
    kva: KVAResult,
    /,
    *,
    result_id: str,
) -> XVAResult:
    """Assemble already-separated adjustments and reject incompatible ledgers."""

    if not isinstance(cva, CVAResult) or not isinstance(dva, DVAResult):
        raise TypeError("cva and dva must be their separated result types.")
    if (
        not isinstance(fva, FVAResult)
        or not isinstance(mva, MVAResult)
        or not isinstance(kva, KVAResult)
    ):
        raise TypeError("fva, mva, and kva must be their separated result types.")
    profile_ids = {
        cva.exposure_profile_id,
        dva.exposure_profile_id,
        fva.exposure_profile_id,
        mva.exposure_profile_id,
        kva.exposure_profile_id,
    }
    discount_ids = {
        cva.discount_curve_id,
        dva.discount_curve_id,
        fva.discount_curve_id,
        mva.discount_curve_id,
        kva.discount_curve_id,
    }
    if len(profile_ids) != 1:
        raise ValueError("XVA components refer to different exposure profiles.")
    if len(discount_ids) != 1:
        raise ValueError("XVA components refer to different discount curves.")
    total = (
        cva.adjustment + dva.adjustment + fva.adjustment + mva.adjustment + kva.adjustment
    )
    recomposed = jnp.sum(cva.bucket_contributions) + jnp.sum(dva.bucket_contributions)
    recomposed = recomposed + fva.adjustment + mva.adjustment + kva.adjustment
    return XVAResult(
        cva,
        dva,
        fva,
        mva,
        kva,
        total,
        total - recomposed,
        _identifier(result_id, "result_id"),
    )


__all__ = [
    "CVAResult",
    "DVAResult",
    "EconomicCapitalPolicy",
    "FVAResult",
    "FundingPolicy",
    "KVAResult",
    "MVAResult",
    "MarginFundingPolicy",
    "XVAResult",
    "assemble_xva",
    "compute_cva",
    "compute_dva",
    "compute_fva",
    "compute_kva",
    "compute_mva",
]
