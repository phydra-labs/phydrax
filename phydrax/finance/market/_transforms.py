#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..core import Currency, FinancialTimestamp
from ._lineage import DataLineage
from ._quotes import QuoteTiePolicy
from ._status import MarketStatus


class CorporateActionKind(str, Enum):
    CASH_DIVIDEND = "cash_dividend"
    SPLIT = "split"


class ReturnKind(str, Enum):
    SIMPLE = "simple"
    LOG = "log"


class RealizedMeasureKind(str, Enum):
    VARIANCE = "variance"
    VOLATILITY = "volatility"
    BIPOWER_VARIATION = "bipower_variation"


class CorporateAction(StrictModule, NonTrainableState):
    """One immutable bitemporal corporate-action vintage."""

    action_id: str = eqx.field(static=True)
    kind: CorporateActionKind = eqx.field(static=True)
    value: Array
    timestamp: FinancialTimestamp = eqx.field(static=True)
    lineage: DataLineage
    currency: Currency | None = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)

    def __init__(
        self,
        action_id: str,
        kind: CorporateActionKind,
        value: ArrayLike,
        timestamp: FinancialTimestamp,
        lineage: DataLineage,
        /,
        *,
        currency: Currency | None = None,
    ):
        if not isinstance(action_id, str) or not action_id.strip():
            raise ValueError("action_id must be a non-empty string.")
        if not isinstance(kind, CorporateActionKind):
            raise TypeError("kind must be a CorporateActionKind.")
        host = np.asarray(value)
        if host.shape != () or host.dtype.kind not in "fiu" or not np.isfinite(host):
            raise ValueError("Corporate-action value must be one finite real scalar.")
        if kind is CorporateActionKind.SPLIT and float(host) <= 0.0:
            raise ValueError("A split ratio must be positive.")
        if kind is CorporateActionKind.CASH_DIVIDEND and float(host) < 0.0:
            raise ValueError("A cash dividend must be nonnegative.")
        if not isinstance(timestamp, FinancialTimestamp):
            raise TypeError("timestamp must be a FinancialTimestamp.")
        if not isinstance(lineage, DataLineage):
            raise TypeError("lineage must be DataLineage.")
        if currency is not None and not isinstance(currency, Currency):
            raise TypeError("currency must be Currency or None.")
        if kind is CorporateActionKind.CASH_DIVIDEND and currency is None:
            raise ValueError("A cash dividend requires an explicit currency.")
        if kind is CorporateActionKind.SPLIT and currency is not None:
            raise ValueError("A split ratio must not declare a currency.")
        identifier = action_id.strip()
        self.action_id = identifier
        self.kind = kind
        self.value = jnp.asarray(host)
        self.timestamp = timestamp
        self.lineage = lineage
        self.currency = currency
        self.observation_id = canonical_fingerprint(
            {
                "kind": "financial-corporate-action",
                "action_id": identifier,
                "action_kind": kind.value,
                "value": float(host),
                "event_time_ns": int(timestamp.epoch_nanoseconds),
                "available_time_ns": int(timestamp.available_ns),
                "vintage_id": timestamp.vintage_id,
                "currency": None if currency is None else currency.code,
                "lineage": lineage.lineage_id,
            }
        )


class CorporateActionSeries(StrictModule, NonTrainableState):
    """Corporate-action archive retaining corrections as separate vintages."""

    actions: tuple[CorporateAction, ...]
    series_id: str = eqx.field(static=True)

    def __init__(self, actions: Sequence[CorporateAction], /):
        values = tuple(actions)
        if not values or not all(isinstance(value, CorporateAction) for value in values):
            raise TypeError("actions must contain at least one CorporateAction.")
        if len({value.observation_id for value in values}) != len(values):
            raise ValueError("The same corporate-action vintage cannot appear twice.")
        ordered = tuple(
            sorted(
                values,
                key=lambda value: (
                    value.action_id,
                    value.timestamp.available_ns,
                    value.timestamp.vintage_id,
                    value.observation_id,
                ),
            )
        )
        self.actions = ordered
        self.series_id = canonical_fingerprint(
            {
                "kind": "financial-corporate-action-series",
                "actions": [value.observation_id for value in ordered],
            }
        )

    def available_at(
        self,
        decision_time: FinancialTimestamp,
        /,
        *,
        tie_policy: QuoteTiePolicy = QuoteTiePolicy.REJECT,
    ) -> tuple[CorporateAction, ...]:
        if not isinstance(decision_time, FinancialTimestamp):
            raise TypeError("decision_time must be a FinancialTimestamp.")
        if not isinstance(tie_policy, QuoteTiePolicy):
            raise TypeError("tie_policy must be a QuoteTiePolicy.")
        decision_ns = int(decision_time.epoch_nanoseconds)
        grouped: dict[str, list[CorporateAction]] = {}
        for action in self.actions:
            if action.timestamp.available_ns <= decision_ns:
                grouped.setdefault(action.action_id, []).append(action)
        selected: list[CorporateAction] = []
        for action_id in sorted(grouped):
            candidates = grouped[action_id]
            newest_time = max(value.timestamp.available_ns for value in candidates)
            newest = [
                value
                for value in candidates
                if value.timestamp.available_ns == newest_time
            ]
            if len(newest) > 1 and tie_policy is QuoteTiePolicy.REJECT:
                raise ValueError(
                    f"Corporate action {action_id!r} has duplicate latest vintages."
                )
            if tie_policy is QuoteTiePolicy.EARLIEST_VINTAGE:
                choice = min(
                    newest,
                    key=lambda value: (value.timestamp.vintage_id, value.observation_id),
                )
            else:
                choice = max(
                    newest,
                    key=lambda value: (value.timestamp.vintage_id, value.observation_id),
                )
            selected.append(choice)
        return tuple(
            sorted(
                selected,
                key=lambda value: (
                    value.timestamp.epoch_nanoseconds,
                    value.action_id,
                ),
            )
        )


class CorporateActionAdjustment(StrictModule, NonTrainableState):
    adjusted_prices: Array
    valid_mask: Array
    status: Array
    lineage: DataLineage
    adjustment_id: str = eqx.field(static=True)


def adjust_for_corporate_actions(
    event_time_ns: ArrayLike,
    prices: ArrayLike,
    valid_mask: ArrayLike,
    series: CorporateActionSeries,
    decision_time: FinancialTimestamp,
    input_lineage: DataLineage,
    /,
    *,
    price_currency: Currency,
    tie_policy: QuoteTiePolicy = QuoteTiePolicy.REJECT,
) -> CorporateActionAdjustment:
    """Back-adjust earlier prices using only action vintages available to the decision."""

    times = np.asarray(event_time_ns, dtype=np.int64)
    values = np.asarray(prices)
    valid = np.asarray(valid_mask, dtype=bool)
    if times.ndim != 1 or values.shape != times.shape or valid.shape != times.shape:
        raise ValueError("Corporate-action inputs must be equal-length vectors.")
    if values.dtype.kind not in "fiu":
        raise ValueError("prices must be real.")
    if np.any(np.diff(times) <= 0):
        raise ValueError("event_time_ns must be strictly increasing.")
    if not isinstance(series, CorporateActionSeries):
        raise TypeError("series must be a CorporateActionSeries.")
    if not isinstance(input_lineage, DataLineage):
        raise TypeError("input_lineage must be DataLineage.")
    if not isinstance(price_currency, Currency):
        raise TypeError("price_currency must be Currency.")
    actions = tuple(
        action
        for action in series.available_at(decision_time, tie_policy=tie_policy)
        if action.timestamp.epoch_nanoseconds <= decision_time.epoch_nanoseconds
    )
    adjusted = values.astype(np.result_type(values.dtype, np.float64), copy=True)
    status = np.zeros(times.shape, dtype=np.int32)
    output_valid = valid & np.isfinite(values)
    status[valid & ~np.isfinite(values)] |= int(MarketStatus.NONFINITE)
    for action in actions:
        affected = times < action.timestamp.epoch_nanoseconds
        amount = float(np.asarray(action.value))
        if action.kind is CorporateActionKind.SPLIT:
            adjusted[affected] /= amount
        else:
            if (
                action.currency is None
                or action.currency.currency_id != price_currency.currency_id
            ):
                status[affected] |= int(MarketStatus.CURRENCY_MISMATCH)
                output_valid[affected] = False
            else:
                adjusted[affected] -= amount
    output_valid &= status == int(MarketStatus.SUCCESS)
    adjusted[~output_valid] = 0.0
    action_lineages = tuple(action.lineage for action in actions)
    upstream = (input_lineage,) + action_lineages
    lineage = DataLineage.derived(
        "corporate-action-back-adjustment",
        upstream,
        source_id="phydrax.finance.market",
        dataset_id=canonical_fingerprint(
            {
                "kind": "corporate-action-adjusted-price-series",
                "input_lineage": input_lineage.lineage_id,
                "actions": [value.observation_id for value in actions],
            }
        ),
    )
    adjustment_id = canonical_fingerprint(
        {
            "kind": "financial-corporate-action-adjustment",
            "lineage": lineage.lineage_id,
            "event_time_ns": times.tolist(),
            "valid_mask": output_valid.tolist(),
        }
    )
    return CorporateActionAdjustment(
        jnp.asarray(adjusted),
        jnp.asarray(output_valid),
        jnp.asarray(status),
        lineage,
        adjustment_id,
    )


class ReturnBatch(StrictModule, NonTrainableState):
    interval_start_ns: Array
    interval_end_ns: Array
    values: Array
    valid_mask: Array
    status: Array
    kind: ReturnKind = eqx.field(static=True)
    lineage: DataLineage
    return_id: str = eqx.field(static=True)


def price_returns(
    event_time_ns: ArrayLike,
    prices: ArrayLike,
    valid_mask: ArrayLike,
    lineage: DataLineage,
    /,
    *,
    kind: ReturnKind = ReturnKind.SIMPLE,
) -> ReturnBatch:
    """Compute adjacent returns; any missing endpoint invalidates the interval."""

    times = np.asarray(event_time_ns, dtype=np.int64)
    values = np.asarray(prices)
    valid = np.asarray(valid_mask, dtype=bool)
    if (
        times.ndim != 1
        or times.size < 2
        or values.shape != times.shape
        or valid.shape != times.shape
    ):
        raise ValueError(
            "Return inputs must be equal-length vectors with at least two points."
        )
    if values.dtype.kind not in "fiu" or np.any(np.diff(times) <= 0):
        raise ValueError("Return prices must be real and times strictly increasing.")
    if not isinstance(lineage, DataLineage):
        raise TypeError("lineage must be DataLineage.")
    if not isinstance(kind, ReturnKind):
        raise TypeError("kind must be a ReturnKind.")
    left = values[:-1]
    right = values[1:]
    accepted = valid[:-1] & valid[1:] & np.isfinite(left) & np.isfinite(right)
    domain = (
        (left != 0.0) if kind is ReturnKind.SIMPLE else ((left > 0.0) & (right > 0.0))
    )
    accepted &= domain
    status = np.zeros(left.shape, dtype=np.int32)
    status[~(np.isfinite(left) & np.isfinite(right))] |= int(MarketStatus.NONFINITE)
    status[~domain] |= int(MarketStatus.INVALID_DOMAIN)
    status[~(valid[:-1] & valid[1:])] |= int(MarketStatus.MISSING_FACTOR)
    safe_left = np.where(domain, left, 1.0)
    safe_right = np.where((right <= 0.0) if kind is ReturnKind.LOG else False, 1.0, right)
    result = safe_right / safe_left - 1.0
    if kind is ReturnKind.LOG:
        result = np.log(safe_right) - np.log(safe_left)
    result[~accepted] = 0.0
    derived = DataLineage.derived(
        f"{kind.value}-return",
        (lineage,),
        source_id="phydrax.finance.market",
        dataset_id=canonical_fingerprint(
            {
                "kind": "financial-return-series",
                "source": lineage.lineage_id,
                "return_kind": kind.value,
                "interval_start_ns": times[:-1].tolist(),
                "interval_end_ns": times[1:].tolist(),
            }
        ),
    )
    return_id = canonical_fingerprint(
        {
            "kind": "financial-return-batch",
            "lineage": derived.lineage_id,
            "valid_mask": accepted.tolist(),
        }
    )
    return ReturnBatch(
        jnp.asarray(times[:-1]),
        jnp.asarray(times[1:]),
        jnp.asarray(result),
        jnp.asarray(accepted),
        jnp.asarray(status),
        kind,
        derived,
        return_id,
    )


class BarBatch(StrictModule, NonTrainableState):
    start_time_ns: Array
    end_time_ns: Array
    open: Array
    high: Array
    low: Array
    close: Array
    volume: Array
    observation_count: Array
    valid_mask: Array
    status: Array
    lineage: DataLineage
    bar_id: str = eqx.field(static=True)


def build_time_bars(
    event_time_ns: ArrayLike,
    prices: ArrayLike,
    volumes: ArrayLike,
    valid_mask: ArrayLike,
    starts_ns: ArrayLike,
    ends_ns: ArrayLike,
    lineage: DataLineage,
    /,
) -> BarBatch:
    """Aggregate exact half-open bars ``[start, end)`` without filling gaps."""

    times = np.asarray(event_time_ns, dtype=np.int64)
    price_values = np.asarray(prices)
    volume_values = np.asarray(volumes)
    source_valid = np.asarray(valid_mask, dtype=bool)
    starts = np.asarray(starts_ns, dtype=np.int64)
    ends = np.asarray(ends_ns, dtype=np.int64)
    if (
        times.ndim != 1
        or price_values.shape != times.shape
        or volume_values.shape != times.shape
        or source_valid.shape != times.shape
    ):
        raise ValueError("Bar observations must be equal-length vectors.")
    if price_values.dtype.kind not in "fiu" or volume_values.dtype.kind not in "fiu":
        raise ValueError("Bar prices and volumes must be real.")
    if starts.ndim != 1 or ends.shape != starts.shape or np.any(ends <= starts):
        raise ValueError("Bar windows must be equal-length vectors with start < end.")
    if np.any(starts[1:] < ends[:-1]):
        raise ValueError("Bar windows must not overlap.")
    if not isinstance(lineage, DataLineage):
        raise TypeError("lineage must be DataLineage.")
    count = starts.size
    opens = np.zeros((count,), dtype=price_values.dtype)
    highs = np.zeros((count,), dtype=price_values.dtype)
    lows = np.zeros((count,), dtype=price_values.dtype)
    closes = np.zeros((count,), dtype=price_values.dtype)
    aggregated_volume = np.zeros((count,), dtype=volume_values.dtype)
    observation_count = np.zeros((count,), dtype=np.int32)
    valid = np.zeros((count,), dtype=bool)
    status = np.zeros((count,), dtype=np.int32)
    finite = np.isfinite(price_values) & np.isfinite(volume_values)
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        in_window = (times >= start) & (times < end)
        selected = in_window & source_valid & finite
        positions = np.flatnonzero(selected)
        if positions.size == 0:
            status[index] = int(MarketStatus.OUTSIDE_WINDOW)
            if np.any(in_window & source_valid & ~finite):
                status[index] |= int(MarketStatus.NONFINITE)
            continue
        selected_prices = price_values[positions]
        opens[index] = selected_prices[0]
        highs[index] = np.max(selected_prices)
        lows[index] = np.min(selected_prices)
        closes[index] = selected_prices[-1]
        aggregated_volume[index] = np.sum(volume_values[positions])
        observation_count[index] = positions.size
        valid[index] = True
    derived = DataLineage.derived(
        "half-open-time-bar",
        (lineage,),
        source_id="phydrax.finance.market",
        dataset_id=canonical_fingerprint(
            {
                "kind": "financial-time-bars",
                "source": lineage.lineage_id,
                "starts_ns": starts.tolist(),
                "ends_ns": ends.tolist(),
            }
        ),
    )
    bar_id = canonical_fingerprint(
        {
            "kind": "financial-bar-batch",
            "lineage": derived.lineage_id,
            "observation_count": observation_count.tolist(),
            "valid_mask": valid.tolist(),
        }
    )
    return BarBatch(
        jnp.asarray(starts),
        jnp.asarray(ends),
        jnp.asarray(opens),
        jnp.asarray(highs),
        jnp.asarray(lows),
        jnp.asarray(closes),
        jnp.asarray(aggregated_volume),
        jnp.asarray(observation_count),
        jnp.asarray(valid),
        jnp.asarray(status),
        derived,
        bar_id,
    )


class RealizedMeasure(StrictModule, NonTrainableState):
    value: Array
    valid: Array
    status: Array
    kind: RealizedMeasureKind = eqx.field(static=True)
    observation_count: Array
    lineage: DataLineage
    measure_id: str = eqx.field(static=True)


def realized_measure(
    returns: ReturnBatch,
    /,
    *,
    kind: RealizedMeasureKind,
    annualization: float = 1.0,
) -> RealizedMeasure:
    """Compute one realized measure and fail closed when any return is missing."""

    if not isinstance(returns, ReturnBatch):
        raise TypeError("returns must be a ReturnBatch.")
    if not isinstance(kind, RealizedMeasureKind):
        raise TypeError("kind must be a RealizedMeasureKind.")
    if not np.isfinite(annualization) or annualization <= 0.0:
        raise ValueError("annualization must be finite and positive.")
    values = returns.values
    accepted = returns.valid_mask & (returns.status == int(MarketStatus.SUCCESS))
    complete = jnp.all(accepted) & (values.size > 0)
    variance = jnp.sum(jnp.where(accepted, values * values, 0.0)) * annualization
    if kind is RealizedMeasureKind.VARIANCE:
        result = variance
    elif kind is RealizedMeasureKind.VOLATILITY:
        result = jnp.sqrt(variance)
    else:
        products = jnp.abs(values[1:] * values[:-1])
        pair_valid = accepted[1:] & accepted[:-1]
        complete = complete & (values.size >= 2) & jnp.all(pair_valid)
        result = (
            (jnp.pi / 2.0) * jnp.sum(jnp.where(pair_valid, products, 0.0)) * annualization
        )
    status = jnp.where(
        complete,
        jnp.asarray(int(MarketStatus.SUCCESS), dtype=jnp.int32),
        jnp.asarray(int(MarketStatus.MISSING_FACTOR), dtype=jnp.int32),
    )
    result = jnp.where(complete, result, 0.0)
    derived = DataLineage.derived(
        f"realized-{kind.value}",
        (returns.lineage,),
        source_id="phydrax.finance.market",
        dataset_id=canonical_fingerprint(
            {
                "kind": "financial-realized-measure",
                "returns": returns.return_id,
                "measure_kind": kind.value,
                "annualization": annualization,
            }
        ),
    )
    measure_id = canonical_fingerprint(
        {
            "kind": "financial-realized-measure-result",
            "lineage": derived.lineage_id,
            "measure_kind": kind.value,
        }
    )
    return RealizedMeasure(
        result,
        complete,
        status,
        kind,
        jnp.sum(accepted.astype(jnp.int32)),
        derived,
        measure_id,
    )


__all__ = [
    "BarBatch",
    "CorporateAction",
    "CorporateActionAdjustment",
    "CorporateActionKind",
    "CorporateActionSeries",
    "RealizedMeasure",
    "RealizedMeasureKind",
    "ReturnBatch",
    "ReturnKind",
    "adjust_for_corporate_actions",
    "build_time_bars",
    "price_returns",
    "realized_measure",
]
