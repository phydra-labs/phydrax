# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Host-side independent replay for exposure XVA ledgers."""

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
from ._xva import (
    EconomicCapitalPolicy,
    FundingPolicy,
    MarginFundingPolicy,
    XVAResult,
)


_COMPONENTS = ("cva", "dva", "fva", "mva", "kva")


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty string.")
    return value


def _host_vector(value: ArrayLike, shape: tuple[int, ...], name: str, /) -> np.ndarray:
    result = np.asarray(jax.device_get(jnp.asarray(value, dtype=float)))
    if result.shape == ():
        result = np.broadcast_to(result, shape)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite and have shape {shape}.")
    return np.asarray(result, dtype=float)


def _host_recovery(value: ArrayLike, name: str, /) -> float:
    result = np.asarray(jax.device_get(jnp.asarray(value, dtype=float)))
    if result.shape != ():
        raise ValueError(f"{name} must be scalar.")
    scalar = float(result)
    if not isfinite(scalar) or scalar < 0.0 or scalar > 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return scalar


def _independent_profile_id(profile: ExposureProfile, /) -> str:
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


class XVAReplay(StrictModule):
    """Independent component and bucket reconciliation evidence."""

    recorded_adjustments: Array
    replayed_adjustments: Array
    maximum_bucket_errors: Array
    component_matches: Array
    recorded_total: Array
    replayed_total: Array
    total_matches: Array
    decomposition_matches: Array
    valid: Array
    failure_ids: tuple[str, ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)


def replay_xva(
    result: XVAResult,
    profile: ExposureProfile,
    counterparty_default_increments: ArrayLike,
    counterparty_recovery_rate: ArrayLike,
    own_default_increments: ArrayLike,
    own_recovery_rate: ArrayLike,
    discount_factors: ArrayLike,
    funding_policy: FundingPolicy,
    expected_initial_margin: ArrayLike,
    margin_policy: MarginFundingPolicy,
    economic_capital_policy: EconomicCapitalPolicy,
    /,
    *,
    counterparty_default_law_id: str,
    counterparty_recovery_terms_id: str,
    own_default_law_id: str,
    own_recovery_terms_id: str,
    initial_margin_profile_id: str,
    discount_curve_id: str,
    absolute_tolerance: float = 1.0e-10,
    relative_tolerance: float = 1.0e-8,
) -> XVAReplay:
    """Recompute every component from decoded economic inputs, not result buckets."""

    if not isinstance(result, XVAResult):
        raise TypeError("result must be an XVAResult.")
    if not isinstance(profile, ExposureProfile):
        raise TypeError("profile must be an ExposureProfile.")
    if not isinstance(funding_policy, FundingPolicy):
        raise TypeError("funding_policy must be a FundingPolicy.")
    if not isinstance(margin_policy, MarginFundingPolicy):
        raise TypeError("margin_policy must be a MarginFundingPolicy.")
    if not isinstance(economic_capital_policy, EconomicCapitalPolicy):
        raise TypeError("economic_capital_policy must be an EconomicCapitalPolicy.")
    atol = float(absolute_tolerance)
    rtol = float(relative_tolerance)
    if not isfinite(atol) or not isfinite(rtol) or atol < 0.0 or rtol < 0.0:
        raise ValueError("Replay tolerances must be finite and nonnegative.")

    shape = tuple(profile.times.shape)
    times = _host_vector(profile.times, shape, "profile times")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("Profile times must be strictly increasing.")
    positive = _host_vector(
        profile.expected_positive_exposure, shape, "expected positive exposure"
    )
    negative = _host_vector(
        profile.expected_negative_exposure, shape, "expected negative exposure"
    )
    cp_default = _host_vector(
        counterparty_default_increments, shape, "counterparty default increments"
    )
    own_default = _host_vector(own_default_increments, shape, "own default increments")
    if np.any(cp_default < 0.0) or np.sum(cp_default) > 1.0 + 1.0e-7:
        raise ValueError("Counterparty default increments are invalid.")
    if np.any(own_default < 0.0) or np.sum(own_default) > 1.0 + 1.0e-7:
        raise ValueError("Own-default increments are invalid.")
    cp_recovery = _host_recovery(counterparty_recovery_rate, "counterparty recovery rate")
    own_recovery = _host_recovery(own_recovery_rate, "own recovery rate")
    discount = _host_vector(discount_factors, shape, "discount factors")
    if np.any(discount <= 0.0):
        raise ValueError("Discount factors must be positive.")
    borrow = _host_vector(funding_policy.borrowing_spreads, shape, "borrowing spreads")
    lend = _host_vector(funding_policy.lending_spreads, shape, "lending spreads")
    margin = _host_vector(expected_initial_margin, shape, "expected initial margin")
    margin_spread = _host_vector(
        margin_policy.funding_spreads, shape, "margin funding spreads"
    )
    capital = _host_vector(
        economic_capital_policy.capital_profile, shape, "economic capital"
    )
    capital_cost = _host_vector(
        economic_capital_policy.cost_of_capital, shape, "cost of capital"
    )
    if any(
        np.any(values < 0.0)
        for values in (borrow, lend, margin, margin_spread, capital, capital_cost)
    ):
        raise ValueError(
            "Funding, margin, and economic-capital replay inputs are nonnegative."
        )

    widths = np.diff(times, prepend=times[0])
    replay_buckets = (
        -(1.0 - cp_recovery) * discount * positive * cp_default,
        (1.0 - own_recovery) * discount * negative * own_default,
        discount * (-borrow * positive + lend * negative),
        -discount * margin_spread * margin,
        -discount * capital_cost * capital,
    )
    replayed = np.asarray(
        (
            np.sum(replay_buckets[0]),
            np.sum(replay_buckets[1]),
            np.sum(replay_buckets[2] * widths),
            np.sum(replay_buckets[3] * widths),
            np.sum(replay_buckets[4] * widths),
        ),
        dtype=float,
    )
    recorded_buckets = (
        np.asarray(jax.device_get(result.cva.bucket_contributions)),
        np.asarray(jax.device_get(result.dva.bucket_contributions)),
        np.asarray(
            jax.device_get(
                result.fva.borrowing_contributions + result.fva.lending_contributions
            )
        ),
        np.asarray(jax.device_get(result.mva.bucket_contributions)),
        np.asarray(jax.device_get(result.kva.bucket_contributions)),
    )
    recorded = np.asarray(
        tuple(
            float(np.asarray(jax.device_get(component.adjustment)))
            for component in (result.cva, result.dva, result.fva, result.mva, result.kva)
        ),
        dtype=float,
    )
    bucket_errors = np.asarray(
        [
            np.max(np.abs(left - right))
            for left, right in zip(recorded_buckets, replay_buckets, strict=True)
        ],
        dtype=float,
    )
    component_matches = np.isclose(recorded, replayed, rtol=rtol, atol=atol)
    bucket_matches = np.asarray(
        [
            np.allclose(left, right, rtol=rtol, atol=atol)
            for left, right in zip(recorded_buckets, replay_buckets, strict=True)
        ],
        dtype=bool,
    )
    component_matches &= bucket_matches
    recorded_total = float(np.asarray(jax.device_get(result.total_adjustment)))
    replayed_total = float(np.sum(replayed))
    total_matches = bool(np.isclose(recorded_total, replayed_total, rtol=rtol, atol=atol))
    decomposition_matches = bool(
        np.isclose(recorded_total, np.sum(recorded), rtol=rtol, atol=atol)
        and np.isclose(
            float(np.asarray(jax.device_get(result.decomposition_residual))),
            0.0,
            rtol=rtol,
            atol=atol,
        )
    )

    profile_id = _independent_profile_id(profile)
    expected_discount_id = _identifier(discount_curve_id, "discount_curve_id")
    failures = [
        f"xva:{name}:value-or-buckets"
        for name, matched in zip(_COMPONENTS, component_matches, strict=True)
        if not matched
    ]
    if not total_matches:
        failures.append("xva:total")
    if not decomposition_matches:
        failures.append("xva:decomposition")
    identity_checks = (
        (result.cva.exposure_profile_id == profile_id, "xva:cva:exposure-profile"),
        (result.dva.exposure_profile_id == profile_id, "xva:dva:exposure-profile"),
        (result.fva.exposure_profile_id == profile_id, "xva:fva:exposure-profile"),
        (result.mva.exposure_profile_id == profile_id, "xva:mva:exposure-profile"),
        (result.kva.exposure_profile_id == profile_id, "xva:kva:exposure-profile"),
        (
            result.cva.counterparty_default_law_id
            == _identifier(counterparty_default_law_id, "counterparty_default_law_id"),
            "xva:cva:default-law",
        ),
        (
            result.cva.recovery_terms_id
            == _identifier(
                counterparty_recovery_terms_id, "counterparty_recovery_terms_id"
            ),
            "xva:cva:recovery-terms",
        ),
        (
            result.dva.own_default_law_id
            == _identifier(own_default_law_id, "own_default_law_id"),
            "xva:dva:default-law",
        ),
        (
            result.dva.recovery_terms_id
            == _identifier(own_recovery_terms_id, "own_recovery_terms_id"),
            "xva:dva:recovery-terms",
        ),
        (
            result.fva.funding_policy_id == funding_policy.policy_id,
            "xva:fva:funding-policy",
        ),
        (
            result.fva.funding_curve_id == funding_policy.funding_curve_id,
            "xva:fva:funding-curve",
        ),
        (
            result.mva.margin_policy_id == margin_policy.policy_id,
            "xva:mva:margin-policy",
        ),
        (
            result.mva.funding_curve_id == margin_policy.funding_curve_id,
            "xva:mva:funding-curve",
        ),
        (
            result.mva.initial_margin_profile_id
            == _identifier(initial_margin_profile_id, "initial_margin_profile_id"),
            "xva:mva:margin-profile",
        ),
        (
            result.kva.economic_capital_policy_id == economic_capital_policy.policy_id,
            "xva:kva:capital-policy",
        ),
    )
    for component in (result.cva, result.dva, result.fva, result.mva, result.kva):
        identity_checks += (
            (
                component.discount_curve_id == expected_discount_id,
                f"xva:{type(component).__name__.removesuffix('Result').lower()}:discount-curve",
            ),
        )
    failures.extend(failure for matched, failure in identity_checks if not matched)
    failure_ids = tuple(failures)
    valid = not failure_ids
    replay_id = canonical_fingerprint(
        {
            "kind": "independent-xva-replay",
            "result_id": result.result_id,
            "profile_id": profile_id,
            "failure_ids": list(failure_ids),
            "absolute_tolerance": atol,
            "relative_tolerance": rtol,
        }
    )
    return XVAReplay(
        jnp.asarray(recorded),
        jnp.asarray(replayed),
        jnp.asarray(bucket_errors),
        jnp.asarray(component_matches),
        jnp.asarray(recorded_total),
        jnp.asarray(replayed_total),
        jnp.asarray(total_matches),
        jnp.asarray(decomposition_matches),
        jnp.asarray(valid),
        failure_ids,
        result.result_id,
        replay_id,
    )


__all__ = ["XVAReplay", "replay_xva"]
