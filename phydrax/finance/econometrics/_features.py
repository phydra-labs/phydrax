#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._datasets import PreparedPointInTimePanel
from ._returns import ReturnResult


FeatureAggregation: TypeAlias = Literal["point", "mean", "sum"]
LabelAggregation: TypeAlias = Literal["sum", "mean", "compound"]
DecisionClock: TypeAlias = Literal["event", "available"]


class FeatureDefinition(StrictModule):
    """One explicitly lagged return feature; lag one is the last completed edge."""

    name: str = eqx.field(static=True)
    series_index: int = eqx.field(static=True)
    lags: tuple[int, ...] = eqx.field(static=True)
    aggregation: FeatureAggregation = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        series_index: int,
        lags: Sequence[int],
        /,
        *,
        aggregation: FeatureAggregation = "point",
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("feature name must be nonempty.")
        series = int(series_index)
        if series < 0:
            raise ValueError("series_index must be nonnegative.")
        lags_ = tuple(int(lag) for lag in lags)
        if not lags_ or any(lag < 1 for lag in lags_):
            raise ValueError(
                "feature lags must be a nonempty sequence of positive integers."
            )
        if len(set(lags_)) != len(lags_):
            raise ValueError("feature lags must be unique.")
        if aggregation == "point" and len(lags_) != 1:
            raise ValueError("point aggregation requires exactly one lag.")
        if aggregation not in ("point", "mean", "sum"):
            raise ValueError("aggregation must be 'point', 'mean', or 'sum'.")
        self.name = name.strip()
        self.series_index = series
        self.lags = tuple(sorted(lags_))
        self.aggregation = aggregation
        self.definition_id = canonical_fingerprint(
            {
                "kind": "feature-definition",
                "name": self.name,
                "series_index": series,
                "lags": self.lags,
                "aggregation": aggregation,
            }
        )


class LabelDefinition(StrictModule):
    """A future return label with a complete realization interval."""

    name: str = eqx.field(static=True)
    series_index: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    aggregation: LabelAggregation = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        series_index: int,
        /,
        *,
        horizon: int,
        aggregation: LabelAggregation = "sum",
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("label name must be nonempty.")
        series = int(series_index)
        horizon_ = int(horizon)
        if series < 0 or horizon_ < 1:
            raise ValueError("series_index must be nonnegative and horizon positive.")
        if aggregation not in ("sum", "mean", "compound"):
            raise ValueError("aggregation must be 'sum', 'mean', or 'compound'.")
        self.name = name.strip()
        self.series_index = series
        self.horizon = horizon_
        self.aggregation = aggregation
        self.definition_id = canonical_fingerprint(
            {
                "kind": "label-definition",
                "name": self.name,
                "series_index": series,
                "horizon": horizon_,
                "aggregation": aggregation,
            }
        )


class FeatureLabelContract(StrictModule):
    """Point-in-time feature and label definitions bound to one decision clock."""

    features: tuple[FeatureDefinition, ...] = eqx.field(static=True)
    label: LabelDefinition = eqx.field(static=True)
    decision_clock: DecisionClock = eqx.field(static=True)
    row_capacity: int = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        features: Sequence[FeatureDefinition],
        label: LabelDefinition,
        /,
        *,
        row_capacity: int,
        decision_clock: DecisionClock = "available",
    ):
        features_ = tuple(features)
        if not features_ or not all(
            isinstance(feature, FeatureDefinition) for feature in features_
        ):
            raise TypeError(
                "features must be a nonempty sequence of FeatureDefinition values."
            )
        if len({feature.name for feature in features_}) != len(features_):
            raise ValueError("feature names must be unique.")
        if not isinstance(label, LabelDefinition):
            raise TypeError("label must be a LabelDefinition.")
        capacity = int(row_capacity)
        if capacity < 1:
            raise ValueError("row_capacity must be positive.")
        if decision_clock not in ("event", "available"):
            raise ValueError("decision_clock must be 'event' or 'available'.")
        self.features = features_
        self.label = label
        self.decision_clock = decision_clock
        self.row_capacity = capacity
        self.contract_id = canonical_fingerprint(
            {
                "kind": "feature-label-contract",
                "features": [feature.definition_id for feature in features_],
                "label": label.definition_id,
                "decision_clock": decision_clock,
                "row_capacity": capacity,
            }
        )


class PreparedFeatureLabelDataset(StrictModule):
    """Fixed-row supervised dataset with full information and outcome intervals."""

    features: Array
    labels: Array
    row_valid: Array
    decision_times_ns: Array
    feature_start_times_ns: Array
    feature_available_times_ns: Array
    label_start_times_ns: Array
    label_end_times_ns: Array
    label_available_times_ns: Array
    asset_indices: Array
    feature_names: tuple[str, ...] = eqx.field(static=True)
    target_quote_key_id: str = eqx.field(static=True)
    panel_id: str = eqx.field(static=True)
    return_result_id: str = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    row_capacity: int = eqx.field(static=True)

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


class FeatureLabelEvidence(StrictModule):
    """Leakage, completeness, and replay evidence for a prepared dataset."""

    active_row_count: Array
    masked_row_count: Array
    future_feature_count: Array
    incomplete_label_count: Array
    overflow: Array
    replay_equal: Array
    dataset_id: str = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)
    panel_id: str = eqx.field(static=True)


def _aggregate_feature(values: Array, aggregation: FeatureAggregation) -> Array:
    if aggregation == "point":
        return values[0]
    if aggregation == "mean":
        return jnp.mean(values)
    return jnp.sum(values)


def _aggregate_label(
    values: Array,
    aggregation: LabelAggregation,
    return_kind: str,
) -> Array:
    if aggregation == "mean":
        return jnp.mean(values)
    if aggregation == "sum":
        return jnp.sum(values)
    if return_kind == "simple":
        return jnp.prod(1.0 + values) - 1.0
    return jnp.expm1(jnp.sum(values))


def prepare_feature_labels(
    panel: PreparedPointInTimePanel,
    returns: ReturnResult,
    contract: FeatureLabelContract,
    /,
) -> tuple[PreparedFeatureLabelDataset, FeatureLabelEvidence]:
    """Materialize only features measurable by each row's declared decision time."""

    if not isinstance(panel, PreparedPointInTimePanel):
        raise TypeError("panel must be a PreparedPointInTimePanel.")
    if not isinstance(returns, ReturnResult):
        raise TypeError("returns must be a ReturnResult.")
    if not isinstance(contract, FeatureLabelContract):
        raise TypeError("contract must be a FeatureLabelContract.")
    if returns.source_panel_id != panel.prepared_id:
        raise ValueError("returns and panel identities must match.")
    series_count, edge_count = returns.values.shape
    referenced = tuple(feature.series_index for feature in contract.features) + (
        contract.label.series_index,
    )
    if any(series >= series_count for series in referenced):
        raise ValueError("feature and label series indices must exist in the panel.")
    maximum_lag = max(max(feature.lags) for feature in contract.features)
    first_decision_edge = maximum_lag - 1
    last_decision_edge = edge_count - contract.label.horizon - 1
    candidate_count = max(last_decision_edge - first_decision_edge + 1, 0)
    if candidate_count < 1:
        raise ValueError("panel is too short for the feature lags and label horizon.")
    overflow = candidate_count > contract.row_capacity
    used = min(candidate_count, contract.row_capacity)
    feature_array = jnp.zeros(
        (contract.row_capacity, len(contract.features)), dtype=returns.values.dtype
    )
    labels = jnp.zeros((contract.row_capacity,), dtype=returns.values.dtype)
    row_valid = jnp.zeros((contract.row_capacity,), dtype=bool)
    times = [jnp.zeros((contract.row_capacity,), dtype=jnp.int64) for _ in range(6)]
    asset_indices = jnp.full(
        (contract.row_capacity,), contract.label.series_index, dtype=jnp.int32
    )
    future_count = jnp.asarray(0, dtype=jnp.int32)
    incomplete_count = jnp.asarray(0, dtype=jnp.int32)
    target_series = contract.label.series_index
    for row in range(used):
        decision_edge = first_decision_edge + row
        end_node = decision_edge + 1
        decision_time = (
            panel.event_times_ns[target_series, end_node]
            if contract.decision_clock == "event"
            else panel.available_times_ns[target_series, end_node]
        )
        feature_values = []
        feature_valid = jnp.asarray(True)
        feature_starts = []
        feature_availability = []
        for feature in contract.features:
            indices = jnp.asarray(
                [decision_edge - lag + 1 for lag in feature.lags], dtype=jnp.int32
            )
            selected = returns.values[feature.series_index, indices]
            selected_valid = returns.valid_mask[feature.series_index, indices]
            availability = panel.available_times_ns[feature.series_index, indices + 1]
            available = jnp.all(availability <= decision_time)
            feature_valid = feature_valid & jnp.all(selected_valid) & available
            feature_values.append(_aggregate_feature(selected, feature.aggregation))
            feature_starts.append(
                jnp.min(returns.interval_start_ns[feature.series_index, indices])
            )
            feature_availability.append(jnp.max(availability))
            future_count = future_count + (~available).astype(jnp.int32)
        label_indices = jnp.arange(
            decision_edge + 1,
            decision_edge + 1 + contract.label.horizon,
            dtype=jnp.int32,
        )
        label_values = returns.values[target_series, label_indices]
        label_valid_components = returns.valid_mask[target_series, label_indices]
        label_start = returns.interval_start_ns[target_series, label_indices[0]]
        label_end = returns.interval_end_ns[target_series, label_indices[-1]]
        label_availability = jnp.max(
            panel.available_times_ns[target_series, label_indices + 1]
        )
        label_complete = jnp.all(label_valid_components) & (label_end > label_start)
        valid_row = feature_valid & label_complete
        incomplete_count = incomplete_count + (~label_complete).astype(jnp.int32)
        feature_array = feature_array.at[row].set(jnp.stack(feature_values))
        labels = labels.at[row].set(
            _aggregate_label(label_values, contract.label.aggregation, returns.kind)
        )
        row_valid = row_valid.at[row].set(valid_row)
        times[0] = times[0].at[row].set(decision_time)
        times[1] = times[1].at[row].set(jnp.min(jnp.stack(feature_starts)))
        times[2] = times[2].at[row].set(jnp.max(jnp.stack(feature_availability)))
        times[3] = times[3].at[row].set(label_start)
        times[4] = times[4].at[row].set(label_end)
        times[5] = times[5].at[row].set(label_availability)
    dataset_id = canonical_fingerprint(
        {
            "kind": "prepared-feature-label-dataset",
            "panel": panel.prepared_id,
            "returns": returns.result_id,
            "contract": contract.contract_id,
            "candidate_count": candidate_count,
            "used_count": used,
        }
    )
    dataset = PreparedFeatureLabelDataset(
        features=feature_array,
        labels=labels,
        row_valid=row_valid,
        decision_times_ns=times[0],
        feature_start_times_ns=times[1],
        feature_available_times_ns=times[2],
        label_start_times_ns=times[3],
        label_end_times_ns=times[4],
        label_available_times_ns=times[5],
        asset_indices=asset_indices,
        feature_names=tuple(feature.name for feature in contract.features),
        target_quote_key_id=panel.quote_key_ids[target_series],
        panel_id=panel.prepared_id,
        return_result_id=returns.result_id,
        contract_id=contract.contract_id,
        dataset_id=dataset_id,
        row_capacity=contract.row_capacity,
    )
    evidence = FeatureLabelEvidence(
        active_row_count=jnp.sum(row_valid).astype(jnp.int32),
        masked_row_count=jnp.sum(~row_valid).astype(jnp.int32),
        future_feature_count=future_count,
        incomplete_label_count=incomplete_count,
        overflow=jnp.asarray(overflow),
        replay_equal=jnp.asarray(True),
        dataset_id=dataset_id,
        contract_id=contract.contract_id,
        panel_id=panel.prepared_id,
    )
    return dataset, evidence


def replay_feature_labels(
    dataset: PreparedFeatureLabelDataset,
    panel: PreparedPointInTimePanel,
    returns: ReturnResult,
    contract: FeatureLabelContract,
    /,
) -> FeatureLabelEvidence:
    """Independently reconstruct and exactly compare a feature-label dataset."""

    if not isinstance(dataset, PreparedFeatureLabelDataset):
        raise TypeError("dataset must be a PreparedFeatureLabelDataset.")
    replayed, evidence = prepare_feature_labels(panel, returns, contract)
    equal = (
        replayed.dataset_id == dataset.dataset_id
        and bool(jnp.array_equal(replayed.row_valid, dataset.row_valid))
        and bool(jnp.array_equal(replayed.features, dataset.features))
        and bool(jnp.array_equal(replayed.labels, dataset.labels))
        and bool(jnp.array_equal(replayed.label_end_times_ns, dataset.label_end_times_ns))
    )
    return FeatureLabelEvidence(
        active_row_count=evidence.active_row_count,
        masked_row_count=evidence.masked_row_count,
        future_feature_count=evidence.future_feature_count,
        incomplete_label_count=evidence.incomplete_label_count,
        overflow=evidence.overflow,
        replay_equal=jnp.asarray(equal),
        dataset_id=evidence.dataset_id,
        contract_id=evidence.contract_id,
        panel_id=evidence.panel_id,
    )


__all__ = [
    "DecisionClock",
    "FeatureAggregation",
    "FeatureDefinition",
    "FeatureLabelContract",
    "FeatureLabelEvidence",
    "LabelAggregation",
    "LabelDefinition",
    "PreparedFeatureLabelDataset",
    "prepare_feature_labels",
    "replay_feature_labels",
]
