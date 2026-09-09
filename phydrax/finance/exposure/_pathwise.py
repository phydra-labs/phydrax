# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Fixed-grid pathwise exposure with explicit default dependence and weighting."""

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...stochastic import (
    DiffusionMeasureChange,
    JumpMeasureChange,
    measure_changed_target,
)
from ..contracts._credit import DefaultEventState
from ..core import Currency, FinanceEvidenceBinding
from ._collateral import (
    CloseoutConvention,
    CloseoutPath,
    CollateralPath,
    evolve_collateral,
    net_trade_values,
    NettingSet,
    PreparedCollateralAgreement,
    resolve_closeout,
)


DefaultDependence: TypeAlias = Literal["independent", "wrong_way"]
WrongWayRiskMode: TypeAlias = Literal["shared_factor", "measure_change"]
PathMeasureChange = DiffusionMeasureChange | JumpMeasureChange


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty string.")
    return value


def _time_grid(value: ArrayLike, /) -> Array:
    result = jnp.asarray(value, dtype=float)
    host = np.asarray(jax.device_get(result))
    if result.ndim != 1 or result.shape[0] < 2:
        raise ValueError("times must be a vector with at least two nodes.")
    if not np.all(np.isfinite(host)) or host[0] < 0.0 or np.any(np.diff(host) <= 0.0):
        raise ValueError("times must be nonnegative, finite, and strictly increasing.")
    return result


class PathwiseTradeValues(StrictModule):
    """Base-currency trade marks before netting."""

    times: Array
    values: Array
    valid: Array
    trade_ids: tuple[str, ...] = eqx.field(static=True)
    base_currency: Currency = eqx.field(static=True)
    pricing_law_id: str = eqx.field(static=True)
    factor_layout_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    value_state_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        values: ArrayLike,
        valid: ArrayLike,
        trade_ids: tuple[str, ...],
        base_currency: Currency,
        /,
        *,
        pricing_law_id: str,
        factor_layout_id: str,
        realization_id: str,
        coupling_id: str,
        value_state_id: str,
    ):
        nodes = _time_grid(times)
        marks = jnp.asarray(values, dtype=float)
        if marks.ndim != 3 or marks.shape[1] != nodes.shape[0]:
            raise ValueError("values must have shape (path, time, trade).")
        ids = tuple(_identifier(value, "trade_id") for value in trade_ids)
        if len(ids) != marks.shape[2] or len(set(ids)) != len(ids):
            raise ValueError("trade_ids must uniquely identify the trade axis.")
        if not isinstance(base_currency, Currency):
            raise TypeError("base_currency must be a Currency.")
        path_valid = jnp.asarray(valid)
        if path_valid.dtype != jnp.dtype(bool):
            raise TypeError("valid must have a boolean dtype.")
        if path_valid.shape != (marks.shape[0],):
            raise ValueError("valid must have shape (path,).")
        finite = jnp.all(jnp.isfinite(marks), axis=(1, 2))
        path_valid = path_valid & finite
        self.times = nodes
        self.values = jnp.where(path_valid[:, None, None], marks, 0.0)
        self.valid = path_valid
        self.trade_ids = ids
        self.base_currency = base_currency
        self.pricing_law_id = _identifier(pricing_law_id, "pricing_law_id")
        self.factor_layout_id = _identifier(factor_layout_id, "factor_layout_id")
        self.realization_id = _identifier(realization_id, "realization_id")
        self.coupling_id = _identifier(coupling_id, "coupling_id")
        self.value_state_id = _identifier(value_state_id, "value_state_id")


class PathWeighting(StrictModule):
    """Normalized path weights and explicit sampling-dependence labels."""

    weights: Array
    valid: Array
    independence_labels: tuple[str, ...] = eqx.field(static=True)
    iid: bool = eqx.field(static=True)
    weighting_id: str = eqx.field(static=True)

    def __init__(
        self,
        weights: ArrayLike,
        valid: ArrayLike,
        independence_labels: tuple[str, ...],
        /,
        *,
        iid: bool,
        weighting_id: str,
    ):
        values = jnp.asarray(weights, dtype=float)
        mask = jnp.asarray(valid)
        if mask.dtype != jnp.dtype(bool):
            raise TypeError("Path weighting valid must have a boolean dtype.")
        host = np.asarray(jax.device_get(values))
        host_mask = np.asarray(jax.device_get(mask))
        if values.ndim != 1 or values.shape[0] == 0 or mask.shape != values.shape:
            raise ValueError("weights and valid must be equal non-empty path vectors.")
        labels = tuple(
            _identifier(value, "independence_label") for value in independence_labels
        )
        if len(labels) != values.shape[0]:
            raise ValueError("independence_labels must identify every path.")
        if type(iid) is not bool:
            raise TypeError("iid must be bool.")
        if iid and len(set(labels)) != len(labels):
            raise ValueError("IID paths require unique independence labels.")
        if not np.any(host_mask):
            raise ValueError("At least one path must be valid.")
        if np.any(~np.isfinite(host)) or np.any(host < 0.0):
            raise ValueError("Path weights must be finite and nonnegative.")
        if np.any(host[~host_mask] != 0.0):
            raise ValueError("Invalid paths must have exactly zero weight.")
        if not np.isclose(np.sum(host), 1.0, rtol=1.0e-7, atol=1.0e-8):
            raise ValueError("Valid path weights must sum to one.")
        self.weights = values
        self.valid = mask
        self.independence_labels = labels
        self.iid = iid
        self.weighting_id = _identifier(weighting_id, "weighting_id")

    @property
    def effective_sample_size(self) -> Array:
        return 1.0 / jnp.sum(self.weights**2)


class WrongWayRiskLink(StrictModule):
    """Audited joint-law link; correlation is never inferred from path values."""

    exposure_law_id: str = eqx.field(static=True)
    default_law_id: str = eqx.field(static=True)
    mode: WrongWayRiskMode = eqx.field(static=True)
    factor_ids: tuple[str, ...] = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    measure_change_id: str | None = eqx.field(static=True)
    link_id: str = eqx.field(static=True)

    def __init__(
        self,
        exposure_law_id: str,
        default_law_id: str,
        mode: WrongWayRiskMode,
        factor_ids: tuple[str, ...],
        coupling_id: str,
        /,
        *,
        measure_change_id: str | None = None,
        link_id: str,
    ):
        if mode not in ("shared_factor", "measure_change"):
            raise ValueError("Unsupported wrong-way-risk mode.")
        factors = tuple(_identifier(value, "factor_id") for value in factor_ids)
        if not factors or len(set(factors)) != len(factors):
            raise ValueError("factor_ids must be non-empty and unique.")
        if (mode == "measure_change") != (measure_change_id is not None):
            raise ValueError("Exactly measure-change WWR requires measure_change_id.")
        self.exposure_law_id = _identifier(exposure_law_id, "exposure_law_id")
        self.default_law_id = _identifier(default_law_id, "default_law_id")
        self.mode = mode
        self.factor_ids = factors
        self.coupling_id = _identifier(coupling_id, "coupling_id")
        self.measure_change_id = (
            None
            if measure_change_id is None
            else _identifier(measure_change_id, "measure_change_id")
        )
        self.link_id = _identifier(link_id, "link_id")


class WrongWayRiskResult(StrictModule):
    """Reweighted joint-path measure with retained likelihood and link identity."""

    weighting: PathWeighting
    likelihood_ratio: Array
    source_weighting_id: str = eqx.field(static=True)
    link_id: str = eqx.field(static=True)


class ExposureSimulationPlan(StrictModule):
    """Resolved topology for trade value -> netting -> collateral -> closeout."""

    netting_set: NettingSet
    collateral: PreparedCollateralAgreement
    closeout: CloseoutConvention
    default_dependence: DefaultDependence = eqx.field(static=True)
    wrong_way_risk: WrongWayRiskLink | None = eqx.field(static=True)
    counterparty_default_law_id: str = eqx.field(static=True)
    own_default_law_id: str = eqx.field(static=True)
    pricing_law_id: str = eqx.field(static=True)
    discount_curve_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        netting_set: NettingSet,
        collateral: PreparedCollateralAgreement,
        closeout: CloseoutConvention,
        /,
        *,
        default_dependence: DefaultDependence,
        wrong_way_risk: WrongWayRiskLink | None,
        counterparty_default_law_id: str,
        own_default_law_id: str,
        pricing_law_id: str,
        discount_curve_id: str,
        plan_id: str,
    ):
        if not isinstance(netting_set, NettingSet):
            raise TypeError("netting_set must be a NettingSet.")
        if not isinstance(collateral, PreparedCollateralAgreement):
            raise TypeError("collateral must be a PreparedCollateralAgreement.")
        if not isinstance(closeout, CloseoutConvention):
            raise TypeError("closeout must be a CloseoutConvention.")
        if collateral.agreement.netting_set_id != netting_set.netting_set_id:
            raise ValueError("Collateral agreement and netting set identities differ.")
        if (
            collateral.agreement.collateral_currency.currency_id
            != netting_set.base_currency.currency_id
        ):
            raise ValueError(
                "Single-currency collateral must match the netting-set base currency."
            )
        if default_dependence not in ("independent", "wrong_way"):
            raise ValueError("Unsupported default dependence.")
        if (default_dependence == "wrong_way") != (wrong_way_risk is not None):
            raise ValueError("Exactly wrong-way-dependent exposure requires a WWR link.")
        pricing_id = _identifier(pricing_law_id, "pricing_law_id")
        counterparty_id = _identifier(
            counterparty_default_law_id, "counterparty_default_law_id"
        )
        if wrong_way_risk is not None:
            if wrong_way_risk.exposure_law_id != pricing_id:
                raise ValueError("WWR exposure law differs from the plan pricing law.")
            if wrong_way_risk.default_law_id != counterparty_id:
                raise ValueError("WWR default law differs from the counterparty law.")
        self.netting_set = netting_set
        self.collateral = collateral
        self.closeout = closeout
        self.default_dependence = default_dependence
        self.wrong_way_risk = wrong_way_risk
        self.counterparty_default_law_id = counterparty_id
        self.own_default_law_id = _identifier(own_default_law_id, "own_default_law_id")
        self.pricing_law_id = pricing_id
        self.discount_curve_id = _identifier(discount_curve_id, "discount_curve_id")
        self.plan_id = _identifier(plan_id, "plan_id")


class PathwiseExposure(StrictModule):
    """Ordered exposure ledger after closeout and before aggregation."""

    times: Array
    trade_values: Array
    netted_values: Array
    collateral_balances: Array
    residual_values: Array
    positive_exposure: Array
    negative_exposure: Array
    discounted_positive_exposure: Array
    discounted_negative_exposure: Array
    discount_factors: Array
    path_valid: Array
    path_weights: Array
    collateral: CollateralPath
    closeout: CloseoutPath
    plan_id: str = eqx.field(static=True)
    value_state_id: str = eqx.field(static=True)
    weighting_id: str = eqx.field(static=True)
    wrong_way_link_id: str | None = eqx.field(static=True)
    iid: bool = eqx.field(static=True)


class ExposureProfile(StrictModule):
    """Weighted EE/ENE/PFE profile with sampling diagnostics."""

    times: Array
    expected_positive_exposure: Array
    expected_negative_exposure: Array
    discounted_expected_positive_exposure: Array
    discounted_expected_negative_exposure: Array
    potential_future_exposure: Array
    positive_standard_error: Array
    negative_standard_error: Array
    effective_sample_size: Array
    standard_error_valid: Array
    quantile: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    value_state_id: str = eqx.field(static=True)
    weighting_id: str = eqx.field(static=True)
    wrong_way_link_id: str | None = eqx.field(static=True)


class ExposureEvidence(StrictModule):
    """Path/statistical evidence bound to four disjoint finance evidence groups."""

    valid_path_count: Array
    effective_sample_size: Array
    standard_error_valid: Array
    binding: FinanceEvidenceBinding = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    weighting_id: str = eqx.field(static=True)
    wrong_way_link_id: str | None = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def link_wrong_way_risk(
    weighting: PathWeighting,
    link: WrongWayRiskLink,
    /,
    *,
    shared_factor_likelihood: ArrayLike | None = None,
    measure_change: PathMeasureChange | None = None,
) -> WrongWayRiskResult:
    """Construct explicit WWR weights from shared factors or a native measure change."""

    if not isinstance(weighting, PathWeighting):
        raise TypeError("weighting must be a PathWeighting.")
    if not isinstance(link, WrongWayRiskLink):
        raise TypeError("link must be a WrongWayRiskLink.")
    if link.mode == "shared_factor":
        if shared_factor_likelihood is None or measure_change is not None:
            raise ValueError("Shared-factor WWR requires only shared_factor_likelihood.")
        likelihood = jnp.asarray(shared_factor_likelihood, dtype=float)
        support_valid = weighting.valid
    else:
        if shared_factor_likelihood is not None or not isinstance(
            measure_change, (DiffusionMeasureChange, JumpMeasureChange)
        ):
            raise ValueError("Measure-change WWR requires only a native measure change.")
        if (
            measure_change.proposal_model_id != link.exposure_law_id
            or measure_change.target_model_id != link.default_law_id
        ):
            raise ValueError("Measure-change model identities do not match the WWR link.")
        target = measure_changed_target(
            weighting.weights,
            measure_change,
            sample_axes=0,
            independent=weighting.iid,
        )
        likelihood = jnp.exp(target.log_weights)
        support_valid = weighting.valid & jnp.asarray(target.mask, dtype=bool)
    if likelihood.shape != weighting.weights.shape:
        raise ValueError("WWR likelihood must have one value per path.")
    likelihood = eqx.error_if(
        likelihood,
        jnp.any(~jnp.isfinite(likelihood) | (likelihood < 0.0)),
        "WWR likelihood must be finite and nonnegative.",
    )
    raw = jnp.where(support_valid, weighting.weights * likelihood, 0.0)
    mass = jnp.sum(raw)
    mass = eqx.error_if(mass, mass <= 0.0, "WWR weighted support has zero mass.")
    normalized = raw / mass
    result_id = canonical_fingerprint(
        {
            "kind": "wrong-way-risk-weighting",
            "source_weighting_id": weighting.weighting_id,
            "link_id": link.link_id,
        }
    )
    linked = PathWeighting(
        normalized,
        support_valid,
        weighting.independence_labels,
        iid=weighting.iid,
        weighting_id=result_id,
    )
    return WrongWayRiskResult(
        linked,
        likelihood,
        weighting.weighting_id,
        link.link_id,
    )


def simulate_exposure(
    plan: ExposureSimulationPlan,
    trade_values: PathwiseTradeValues,
    counterparty_default: DefaultEventState,
    own_default: DefaultEventState,
    discount_factors: ArrayLike,
    weighting: PathWeighting,
    /,
    *,
    wrong_way_result: WrongWayRiskResult | None = None,
    replacement_values: ArrayLike | None = None,
) -> PathwiseExposure:
    """Execute the invariant order trade value -> netting -> collateral -> closeout."""

    if not isinstance(plan, ExposureSimulationPlan):
        raise TypeError("plan must be an ExposureSimulationPlan.")
    if not isinstance(trade_values, PathwiseTradeValues):
        raise TypeError("trade_values must be PathwiseTradeValues.")
    if not isinstance(weighting, PathWeighting):
        raise TypeError("weighting must be a PathWeighting.")
    if trade_values.pricing_law_id != plan.pricing_law_id:
        raise ValueError("Trade-value pricing law differs from the exposure plan.")
    if (
        tuple(
            trade_values.trade_ids[index]
            for index in np.asarray(plan.netting_set.trade_indices)
        )
        != plan.netting_set.trade_ids
    ):
        raise ValueError("Netting-set membership does not match the trade value layout.")
    if (
        trade_values.base_currency.currency_id
        != plan.netting_set.base_currency.currency_id
    ):
        raise ValueError("Trade values and netting set use different base currencies.")
    if not np.array_equal(
        np.asarray(jax.device_get(trade_values.times)),
        np.asarray(jax.device_get(plan.collateral.times)),
    ):
        raise ValueError("Trade-value and collateral exposure grids differ.")
    if counterparty_default.law_id != plan.counterparty_default_law_id:
        raise ValueError("Counterparty default law differs from the exposure plan.")
    if own_default.law_id != plan.own_default_law_id:
        raise ValueError("Own-default law differs from the exposure plan.")
    path_count, time_count, _ = trade_values.values.shape
    if counterparty_default.default_times.shape != (
        path_count,
    ) or own_default.default_times.shape != (path_count,):
        raise ValueError("Default event paths must match the trade-value path axis.")
    if weighting.weights.shape != (path_count,):
        raise ValueError("Path weighting must match the trade-value path axis.")
    if plan.default_dependence == "wrong_way":
        if (
            wrong_way_result is None
            or wrong_way_result.link_id != plan.wrong_way_risk.link_id
        ):
            raise ValueError(
                "Wrong-way exposure requires weighting from the plan's WWR link."
            )
        active_weighting = wrong_way_result.weighting
    else:
        if wrong_way_result is not None:
            raise ValueError("Independent exposure does not accept WWR weighting.")
        active_weighting = weighting
    if plan.default_dependence == "wrong_way":
        link = plan.wrong_way_risk
        if (
            trade_values.coupling_id != link.coupling_id
            or counterparty_default.coupling_id != link.coupling_id
        ):
            raise ValueError("Shared WWR coupling identities do not match the plan link.")
    elif (
        trade_values.coupling_id == counterparty_default.coupling_id
        or trade_values.coupling_id == own_default.coupling_id
    ):
        raise ValueError(
            "Independent exposure requires distinct market/default couplings."
        )
    discounts = jnp.asarray(discount_factors, dtype=float)
    if discounts.shape == (time_count,):
        discounts = jnp.broadcast_to(discounts, (path_count, time_count))
    if discounts.shape != (path_count, time_count):
        raise ValueError("discount_factors must have shape (time,) or (path, time).")
    discounts = eqx.error_if(
        discounts,
        jnp.any(~jnp.isfinite(discounts) | (discounts <= 0.0)),
        "Discount factors must be finite and positive.",
    )

    # The order below is intentional and externally observable.
    netted = net_trade_values(trade_values.values, plan.netting_set)
    collateral = evolve_collateral(
        plan.collateral,
        netted,
        path_valid=trade_values.valid & active_weighting.valid,
    )
    closeout = resolve_closeout(
        trade_values.times,
        netted,
        collateral,
        plan.collateral.agreement,
        plan.closeout,
        counterparty_default,
        own_default,
        replacement_values=replacement_values,
    )
    residual = netted - collateral.balances
    grid_index = jnp.arange(time_count)[None, :]
    pre_default = (~closeout.occurred[:, None]) | (
        grid_index < closeout.default_indices[:, None]
    )
    residual = jnp.where(pre_default, residual, 0.0)
    closeout_slot = grid_index == closeout.closeout_indices[:, None]
    residual = jnp.where(
        closeout.occurred[:, None] & closeout_slot,
        closeout.residual_values[:, None],
        residual,
    )
    path_valid = (
        trade_values.valid
        & active_weighting.valid
        & collateral.valid
        & closeout.valid
        & jnp.all(jnp.isfinite(residual), axis=-1)
    )
    validated_weights = eqx.error_if(
        active_weighting.weights,
        jnp.any(~path_valid & (active_weighting.weights != 0.0)),
        "An invalid exposure path has positive aggregation weight.",
    )
    residual = jnp.where(path_valid[:, None], residual, 0.0)
    positive = jnp.maximum(residual, 0.0)
    negative = jnp.maximum(-residual, 0.0)
    return PathwiseExposure(
        trade_values.times,
        trade_values.values,
        netted,
        collateral.balances,
        residual,
        positive,
        negative,
        positive * discounts,
        negative * discounts,
        discounts,
        path_valid,
        validated_weights,
        collateral,
        closeout,
        plan.plan_id,
        trade_values.value_state_id,
        active_weighting.weighting_id,
        None if plan.wrong_way_risk is None else plan.wrong_way_risk.link_id,
        active_weighting.iid,
    )


def aggregate_exposure(
    exposure: PathwiseExposure,
    /,
    *,
    quantile: float = 0.95,
) -> ExposureProfile:
    """Aggregate pathwise exposure without treating dependent paths as IID."""

    if not isinstance(exposure, PathwiseExposure):
        raise TypeError("exposure must be a PathwiseExposure.")
    probability = float(quantile)
    if not isfinite(probability) or probability <= 0.0 or probability >= 1.0:
        raise ValueError("quantile must lie strictly between zero and one.")
    weights = exposure.path_weights
    weights = eqx.error_if(
        weights,
        jnp.any(~jnp.isfinite(weights) | (weights < 0.0))
        | jnp.any(~exposure.path_valid & (weights != 0.0))
        | ~jnp.isclose(jnp.sum(weights), 1.0, rtol=1.0e-6, atol=1.0e-7),
        "Exposure path weights are invalid.",
    )
    positive = exposure.positive_exposure
    negative = exposure.negative_exposure
    expected_positive = ein.contract("p,pt->t", weights, positive)
    expected_negative = ein.contract("p,pt->t", weights, negative)
    discounted_positive = ein.contract(
        "p,pt->t", weights, exposure.discounted_positive_exposure
    )
    discounted_negative = ein.contract(
        "p,pt->t", weights, exposure.discounted_negative_exposure
    )
    order = jnp.argsort(positive, axis=0)
    ordered_values = jnp.take_along_axis(positive, order, axis=0)
    ordered_weights = jnp.take_along_axis(
        jnp.broadcast_to(weights[:, None], positive.shape), order, axis=0
    )
    cumulative = jnp.cumsum(ordered_weights, axis=0)
    quantile_index = jnp.argmax(cumulative >= probability, axis=0)
    pfe = jnp.take_along_axis(ordered_values, quantile_index[None, :], axis=0)[0]
    ess = 1.0 / jnp.sum(weights**2)
    positive_variance = ein.contract(
        "p,pt->t", weights, (positive - expected_positive[None, :]) ** 2
    )
    negative_variance = ein.contract(
        "p,pt->t", weights, (negative - expected_negative[None, :]) ** 2
    )
    # Error bars require IID labels; dependent paths remain valid for the mean
    # but are not mislabeled as independent observations.
    unique_positive_weights = jnp.sum(weights > 0.0)
    standard_error_valid = exposure.iid & (unique_positive_weights > 1)
    positive_se = jnp.where(
        standard_error_valid, jnp.sqrt(positive_variance / ess), jnp.nan
    )
    negative_se = jnp.where(
        standard_error_valid, jnp.sqrt(negative_variance / ess), jnp.nan
    )
    return ExposureProfile(
        exposure.times,
        expected_positive,
        expected_negative,
        discounted_positive,
        discounted_negative,
        pfe,
        positive_se,
        negative_se,
        ess,
        standard_error_valid,
        probability,
        exposure.plan_id,
        exposure.value_state_id,
        exposure.weighting_id,
        exposure.wrong_way_link_id,
    )


def exposure_evidence(
    exposure: PathwiseExposure,
    profile: ExposureProfile,
    binding: FinanceEvidenceBinding,
    /,
) -> ExposureEvidence:
    """Bind path evidence without collapsing data, model, numerical, and use evidence."""

    if not isinstance(exposure, PathwiseExposure):
        raise TypeError("exposure must be a PathwiseExposure.")
    if not isinstance(profile, ExposureProfile):
        raise TypeError("profile must be an ExposureProfile.")
    if not isinstance(binding, FinanceEvidenceBinding):
        raise TypeError("binding must be a FinanceEvidenceBinding.")
    if (
        exposure.plan_id != profile.plan_id
        or exposure.weighting_id != profile.weighting_id
        or exposure.wrong_way_link_id != profile.wrong_way_link_id
    ):
        raise ValueError("Exposure profile identities differ from the pathwise result.")
    evidence_id = canonical_fingerprint(
        {
            "kind": "pathwise-exposure-evidence",
            "plan_id": profile.plan_id,
            "weighting_id": profile.weighting_id,
            "wrong_way_link_id": profile.wrong_way_link_id,
            "binding_id": binding.binding_id,
        }
    )
    return ExposureEvidence(
        jnp.sum(exposure.path_valid, dtype=jnp.int32),
        profile.effective_sample_size,
        profile.standard_error_valid,
        binding,
        profile.plan_id,
        profile.weighting_id,
        profile.wrong_way_link_id,
        evidence_id,
    )


__all__ = [
    "DefaultDependence",
    "ExposureEvidence",
    "ExposureProfile",
    "ExposureSimulationPlan",
    "PathWeighting",
    "PathwiseExposure",
    "PathwiseTradeValues",
    "WrongWayRiskLink",
    "WrongWayRiskMode",
    "WrongWayRiskResult",
    "aggregate_exposure",
    "exposure_evidence",
    "link_wrong_way_risk",
    "simulate_exposure",
]
