#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..core import Currency, FinancialTimestamp
from ..market import DataLineage, QuoteTiePolicy
from ..market._transforms import (
    adjust_for_corporate_actions,
    CorporateActionAdjustment,
    CorporateActionSeries,
    price_returns,
    realized_measure,
    RealizedMeasure,
    RealizedMeasureKind as MarketRealizedMeasureKind,
    ReturnBatch,
    ReturnKind as MarketReturnKind,
)
from ._datasets import PreparedPointInTimePanel


ReturnKind: TypeAlias = Literal["simple", "log"]
RealizedMeasureKind: TypeAlias = Literal["variance", "volatility", "bipower-variation"]


class CorporateActionBinding(StrictModule):
    """Bind canonical market corporate actions to one econometric panel channel."""

    series: CorporateActionSeries
    price_currency: Currency = eqx.field(static=True)
    input_lineage: DataLineage
    decision_time: FinancialTimestamp = eqx.field(static=True)
    tie_policy: QuoteTiePolicy = eqx.field(static=True)
    quote_key_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        quote_key_id: str,
        series: CorporateActionSeries,
        price_currency: Currency,
        input_lineage: DataLineage,
        decision_time: FinancialTimestamp,
        /,
        *,
        tie_policy: QuoteTiePolicy = QuoteTiePolicy.REJECT,
    ):
        if not isinstance(quote_key_id, str) or not quote_key_id.strip():
            raise ValueError("quote_key_id must be nonempty.")
        if not isinstance(series, CorporateActionSeries):
            raise TypeError("series must be a canonical CorporateActionSeries.")
        if not isinstance(price_currency, Currency):
            raise TypeError("price_currency must be a Currency.")
        if not isinstance(input_lineage, DataLineage):
            raise TypeError("input_lineage must be a DataLineage.")
        if not isinstance(decision_time, FinancialTimestamp):
            raise TypeError("decision_time must be a FinancialTimestamp.")
        if not isinstance(tie_policy, QuoteTiePolicy):
            raise TypeError("tie_policy must be a QuoteTiePolicy.")
        self.series = series
        self.price_currency = price_currency
        self.input_lineage = input_lineage
        self.decision_time = decision_time
        self.tie_policy = tie_policy
        self.quote_key_id = quote_key_id.strip()
        self.binding_id = canonical_fingerprint(
            {
                "kind": "econometric-corporate-action-binding",
                "quote_key": self.quote_key_id,
                "series": series.series_id,
                "currency": price_currency.code,
                "lineage": input_lineage.lineage_id,
                "decision_time_ns": int(decision_time.epoch_nanoseconds),
                "tie_policy": tie_policy.value,
            }
        )


class CorporateActionAdjustmentResult(StrictModule):
    """Fixed-shape aggregation of canonical per-channel market adjustments."""

    adjusted_values: Array
    adjusted_valid_mask: Array
    status: Array
    adjustments: tuple[CorporateActionAdjustment, ...]
    source_panel_id: str = eqx.field(static=True)
    binding_ids: tuple[str, ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def _observation_count(observation_ids: Sequence[str], /) -> int:
    active = np.asarray([bool(identifier) for identifier in observation_ids], dtype=bool)
    count = int(np.sum(active))
    if not np.array_equal(active, np.arange(active.size) < count):
        raise ValueError("market transform inputs require an observation prefix.")
    return count


def apply_corporate_actions(
    panel: PreparedPointInTimePanel,
    bindings: Sequence[CorporateActionBinding],
    /,
) -> CorporateActionAdjustmentResult:
    """Call the canonical market adjustment separately for every bound panel channel."""

    if not isinstance(panel, PreparedPointInTimePanel):
        raise TypeError("panel must be a PreparedPointInTimePanel.")
    bindings_ = tuple(bindings)
    if not all(isinstance(binding, CorporateActionBinding) for binding in bindings_):
        raise TypeError("bindings must contain CorporateActionBinding values.")
    bindings_ = tuple(sorted(bindings_, key=lambda binding: binding.quote_key_id))
    if len({binding.quote_key_id for binding in bindings_}) != len(bindings_):
        raise ValueError("each quote channel may have at most one action binding.")
    by_key = {binding.quote_key_id: binding for binding in bindings_}
    unknown = set(by_key) - set(panel.quote_key_ids)
    if unknown:
        raise ValueError("every corporate-action binding must reference the panel.")
    values = panel.values
    valid = panel.valid_mask
    status = jnp.zeros(panel.values.shape, dtype=jnp.int32)
    adjustments = []
    for channel, key_id in enumerate(panel.quote_key_ids):
        if key_id not in by_key:
            continue
        binding = by_key[key_id]
        count = _observation_count(panel.observation_ids[channel])
        if count < 1:
            continue
        adjustment = adjust_for_corporate_actions(
            panel.event_times_ns[channel, :count],
            panel.values[channel, :count],
            panel.valid_mask[channel, :count],
            binding.series,
            binding.decision_time,
            binding.input_lineage,
            price_currency=binding.price_currency,
            tie_policy=binding.tie_policy,
        )
        values = values.at[channel, :count].set(adjustment.adjusted_prices)
        valid = valid.at[channel, :count].set(adjustment.valid_mask)
        status = status.at[channel, :count].set(adjustment.status)
        adjustments.append(adjustment)
    result_id = canonical_fingerprint(
        {
            "kind": "econometric-corporate-action-adjustment",
            "panel": panel.prepared_id,
            "bindings": [binding.binding_id for binding in bindings_],
            "adjustments": [adjustment.adjustment_id for adjustment in adjustments],
        }
    )
    return CorporateActionAdjustmentResult(
        adjusted_values=values,
        adjusted_valid_mask=valid,
        status=status,
        adjustments=tuple(adjustments),
        source_panel_id=panel.prepared_id,
        binding_ids=tuple(binding.binding_id for binding in bindings_),
        result_id=result_id,
    )


class ReturnDefinition(StrictModule):
    """Econometric panel return convention layered over canonical market returns."""

    kind: ReturnKind = eqx.field(static=True)
    maximum_gap_ns: int | None = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(self, *, kind: ReturnKind = "simple", maximum_gap_ns: int | None = None):
        if kind not in ("simple", "log"):
            raise ValueError("kind must be 'simple' or 'log'.")
        gap = None if maximum_gap_ns is None else int(maximum_gap_ns)
        if gap is not None and gap <= 0:
            raise ValueError("maximum_gap_ns must be positive or None.")
        self.kind = kind
        self.maximum_gap_ns = gap
        self.definition_id = canonical_fingerprint(
            {
                "kind": "econometric-return-definition",
                "return_kind": kind,
                "maximum_gap_ns": gap,
            }
        )


class ReturnResult(StrictModule):
    """Fixed-shape panel of canonical market ReturnBatch results."""

    values: Array
    valid_mask: Array
    status: Array
    interval_start_ns: Array
    interval_end_ns: Array
    batches: tuple[ReturnBatch, ...]
    batch_channels: tuple[int, ...] = eqx.field(static=True)
    source_panel_id: str = eqx.field(static=True)
    adjustment_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    kind: ReturnKind = eqx.field(static=True)


def compute_returns(
    panel: PreparedPointInTimePanel,
    definition: ReturnDefinition,
    lineages: Sequence[DataLineage],
    /,
    *,
    adjustment: CorporateActionAdjustmentResult | None = None,
) -> ReturnResult:
    """Call canonical price returns per channel and preserve fixed panel capacity."""

    if not isinstance(panel, PreparedPointInTimePanel):
        raise TypeError("panel must be a PreparedPointInTimePanel.")
    if not isinstance(definition, ReturnDefinition):
        raise TypeError("definition must be a ReturnDefinition.")
    lineages_ = tuple(lineages)
    if len(lineages_) != panel.series_count or not all(
        isinstance(lineage, DataLineage) for lineage in lineages_
    ):
        raise TypeError("lineages must contain one DataLineage per panel channel.")
    if adjustment is not None and (
        not isinstance(adjustment, CorporateActionAdjustmentResult)
        or adjustment.source_panel_id != panel.prepared_id
    ):
        raise ValueError("adjustment must be bound to this panel or None.")
    prices = panel.values if adjustment is None else adjustment.adjusted_values
    source_valid = (
        panel.valid_mask if adjustment is None else adjustment.adjusted_valid_mask
    )
    edge_capacity = panel.capacity - 1
    values = jnp.zeros((panel.series_count, edge_capacity), dtype=prices.dtype)
    valid = jnp.zeros((panel.series_count, edge_capacity), dtype=bool)
    status = jnp.zeros((panel.series_count, edge_capacity), dtype=jnp.int32)
    starts = jnp.zeros((panel.series_count, edge_capacity), dtype=jnp.int64)
    ends = jnp.zeros((panel.series_count, edge_capacity), dtype=jnp.int64)
    batches = []
    batch_channels = []
    market_kind = (
        MarketReturnKind.SIMPLE if definition.kind == "simple" else MarketReturnKind.LOG
    )
    for channel in range(panel.series_count):
        count = _observation_count(panel.observation_ids[channel])
        if count < 2:
            continue
        batch = price_returns(
            panel.event_times_ns[channel, :count],
            prices[channel, :count],
            source_valid[channel, :count],
            lineages_[channel],
            kind=market_kind,
        )
        batch_valid = batch.valid_mask
        if definition.maximum_gap_ns is not None:
            batch_valid = batch_valid & (
                batch.interval_end_ns - batch.interval_start_ns
                <= definition.maximum_gap_ns
            )
        size = count - 1
        values = values.at[channel, :size].set(jnp.where(batch_valid, batch.values, 0.0))
        valid = valid.at[channel, :size].set(batch_valid)
        status = status.at[channel, :size].set(batch.status)
        starts = starts.at[channel, :size].set(batch.interval_start_ns)
        ends = ends.at[channel, :size].set(batch.interval_end_ns)
        batches.append(batch)
        batch_channels.append(channel)
    adjustment_id = "raw" if adjustment is None else adjustment.result_id
    result_id = canonical_fingerprint(
        {
            "kind": "econometric-return-result",
            "panel": panel.prepared_id,
            "definition": definition.definition_id,
            "adjustment": adjustment_id,
            "batches": [batch.return_id for batch in batches],
        }
    )
    return ReturnResult(
        values=values,
        valid_mask=valid,
        status=status,
        interval_start_ns=starts,
        interval_end_ns=ends,
        batches=tuple(batches),
        batch_channels=tuple(batch_channels),
        source_panel_id=panel.prepared_id,
        adjustment_id=adjustment_id,
        definition_id=definition.definition_id,
        result_id=result_id,
        kind=definition.kind,
    )


class RealizedMeasureDefinition(StrictModule):
    kind: RealizedMeasureKind = eqx.field(static=True)
    window: int = eqx.field(static=True)
    annualization: float = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        kind: RealizedMeasureKind = "variance",
        window: int,
        annualization: float = 1.0,
    ):
        if kind not in ("variance", "volatility", "bipower-variation"):
            raise ValueError("unsupported realized-measure kind.")
        window_ = int(window)
        annualization_ = float(annualization)
        if window_ < 2 or not np.isfinite(annualization_) or annualization_ <= 0.0:
            raise ValueError(
                "window must be at least two and annualization positive/finite."
            )
        self.kind = kind
        self.window = window_
        self.annualization = annualization_
        self.definition_id = canonical_fingerprint(
            {
                "kind": "econometric-realized-measure-definition",
                "measure": kind,
                "window": window_,
                "annualization": annualization_,
            }
        )


class RealizedMeasureResult(StrictModule):
    values: Array
    valid_mask: Array
    status: Array
    effective_counts: Array
    window_start_ns: Array
    window_end_ns: Array
    measures: tuple[RealizedMeasure, ...]
    return_result_id: str = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    kind: RealizedMeasureKind = eqx.field(static=True)


def compute_realized_measures(
    returns: ReturnResult,
    definition: RealizedMeasureDefinition,
    /,
) -> RealizedMeasureResult:
    """Apply the canonical realized measure to every complete trailing panel window."""

    if not isinstance(returns, ReturnResult):
        raise TypeError("returns must be a ReturnResult.")
    if not isinstance(definition, RealizedMeasureDefinition):
        raise TypeError("definition must be a RealizedMeasureDefinition.")
    series_count, edge_count = returns.values.shape
    if edge_count < definition.window:
        raise ValueError("return capacity is smaller than the realized-measure window.")
    output_count = edge_count - definition.window + 1
    values = jnp.zeros((series_count, output_count), dtype=returns.values.dtype)
    valid = jnp.zeros((series_count, output_count), dtype=bool)
    status = jnp.zeros((series_count, output_count), dtype=jnp.int32)
    counts = jnp.zeros((series_count, output_count), dtype=jnp.int32)
    measures = []
    market_kind = {
        "variance": MarketRealizedMeasureKind.VARIANCE,
        "volatility": MarketRealizedMeasureKind.VOLATILITY,
        "bipower-variation": MarketRealizedMeasureKind.BIPOWER_VARIATION,
    }[definition.kind]
    by_channel = {
        channel: batch
        for channel, batch in zip(returns.batch_channels, returns.batches, strict=True)
    }
    for channel in range(series_count):
        lineage = by_channel[channel].lineage if channel in by_channel else None
        if lineage is None:
            continue
        for output in range(output_count):
            end = output + definition.window
            batch = ReturnBatch(
                returns.interval_start_ns[channel, output:end],
                returns.interval_end_ns[channel, output:end],
                returns.values[channel, output:end],
                returns.valid_mask[channel, output:end],
                returns.status[channel, output:end],
                MarketReturnKind.SIMPLE
                if returns.kind == "simple"
                else MarketReturnKind.LOG,
                lineage,
                canonical_fingerprint(
                    {
                        "kind": "econometric-return-window",
                        "returns": returns.result_id,
                        "channel": channel,
                        "start": output,
                        "end": end,
                    }
                ),
            )
            measure = realized_measure(
                batch,
                kind=market_kind,
                annualization=definition.annualization,
            )
            values = values.at[channel, output].set(measure.value)
            valid = valid.at[channel, output].set(measure.valid)
            status = status.at[channel, output].set(measure.status)
            counts = counts.at[channel, output].set(measure.observation_count)
            measures.append(measure)
    result_id = canonical_fingerprint(
        {
            "kind": "econometric-realized-measure-result",
            "returns": returns.result_id,
            "definition": definition.definition_id,
            "measures": [measure.measure_id for measure in measures],
        }
    )
    return RealizedMeasureResult(
        values=values,
        valid_mask=valid,
        status=status,
        effective_counts=counts,
        window_start_ns=returns.interval_start_ns[:, :output_count],
        window_end_ns=returns.interval_end_ns[:, definition.window - 1 :],
        measures=tuple(measures),
        return_result_id=returns.result_id,
        definition_id=definition.definition_id,
        result_id=result_id,
        kind=definition.kind,
    )


__all__ = [
    "CorporateActionAdjustmentResult",
    "CorporateActionBinding",
    "RealizedMeasureDefinition",
    "RealizedMeasureKind",
    "RealizedMeasureResult",
    "ReturnDefinition",
    "ReturnKind",
    "ReturnResult",
    "apply_corporate_actions",
    "compute_realized_measures",
    "compute_returns",
]
