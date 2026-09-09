import jax.numpy as jnp

from phydrax.finance.core import (
    AssetReference,
    Currency,
    FinancialIdentifier,
    FinancialTimestamp,
    TemporalAdmissibilityPolicy,
)
from phydrax.finance.econometrics._datasets import (
    PointInTimePanelDefinition,
    prepare_point_in_time_panel,
    PreparedPointInTimePanel,
    resolve_point_in_time_panel,
)
from phydrax.finance.econometrics._experiments import (
    prepare_walk_forward,
    WalkForwardDefinition,
)
from phydrax.finance.econometrics._features import (
    FeatureDefinition,
    FeatureLabelContract,
    LabelDefinition,
    prepare_feature_labels,
    PreparedFeatureLabelDataset,
)
from phydrax.finance.econometrics._returns import (
    apply_corporate_actions,
    compute_returns,
    CorporateActionBinding,
    ReturnDefinition,
    ReturnResult,
)
from phydrax.finance.market._lineage import DataLineage
from phydrax.finance.market._quotes import QuoteKey, QuoteObservation
from phydrax.finance.market._snapshots import MarketDataSnapshot, ReferenceDataSnapshot
from phydrax.finance.market._transforms import (
    CorporateAction,
    CorporateActionKind,
    CorporateActionSeries,
)


_POLICY = TemporalAdmissibilityPolicy(True, True, False)


def _time(event: int, available: int, vintage: str) -> FinancialTimestamp:
    return FinancialTimestamp(event, event, event, available, vintage, _POLICY)


def test_future_revision_is_excluded_and_canonical_split_adjustment_is_reused():
    currency = Currency("USD", 2)
    asset = AssetReference(
        FinancialIdentifier("test", "A"), "equity", currency, "synthetic asset"
    )
    key = QuoteKey(asset.asset_id, "close", currency=currency)
    lineage = DataLineage("synthetic", "prices")
    old = QuoteObservation(key, 100.0, _time(10, 11, "old"), lineage)
    future_revision = QuoteObservation(key, 400.0, _time(10, 30, "revised"), lineage)
    post_split = QuoteObservation(key, 55.0, _time(20, 21, "close20"), lineage)
    reference = ReferenceDataSnapshot(
        (asset,), (), (currency,), as_of=_time(40, 40, "reference"), lineage=lineage
    )
    market = MarketDataSnapshot(
        (old, future_revision, post_split),
        snapshot_time=_time(40, 40, "market"),
        reference_data_id=reference.snapshot_id,
    )
    definition = PointInTimePanelDefinition((key,), _time(22, 22, "analysis"), capacity=4)

    resolved = resolve_point_in_time_panel(definition, reference, market)
    panel, evidence = prepare_point_in_time_panel(resolved)

    assert jnp.array_equal(panel.values[0, :2], jnp.asarray([100.0, 55.0]))
    assert panel.vintage_ids[0][:2] == ("old", "close20")
    assert evidence.active_counts[0] == 2
    assert evidence.future_revision_count[0] == 1
    assert evidence.clocks_admissible[0]

    action_lineage = DataLineage("synthetic", "actions")
    split = CorporateAction(
        "split-20",
        CorporateActionKind.SPLIT,
        2.0,
        _time(20, 21, "split"),
        action_lineage,
    )
    corrected_later = CorporateAction(
        "split-20",
        CorporateActionKind.SPLIT,
        4.0,
        _time(20, 30, "split-revised"),
        action_lineage,
    )
    binding = CorporateActionBinding(
        key.key_id,
        CorporateActionSeries((split, corrected_later)),
        currency,
        lineage,
        _time(22, 22, "decision"),
    )
    adjusted = apply_corporate_actions(panel, (binding,))
    returns = compute_returns(
        panel, ReturnDefinition(kind="simple"), (lineage,), adjustment=adjusted
    )

    assert jnp.allclose(adjusted.adjusted_values[0, :2], jnp.asarray([50.0, 55.0]))
    assert returns.valid_mask[0, 0]
    assert jnp.isclose(returns.values[0, 0], 0.1)


def test_feature_preparation_rejects_future_available_inputs_and_preserves_label_intervals():
    event = jnp.arange(8, dtype=jnp.int64) * 10
    available = event.at[2].set(100)
    panel = PreparedPointInTimePanel(
        values=jnp.arange(100.0, 108.0)[None, :],
        valid_mask=jnp.ones((1, 8), dtype=bool),
        event_times_ns=event[None, :],
        published_times_ns=event[None, :],
        received_times_ns=event[None, :],
        available_times_ns=available[None, :],
        observation_ids=(tuple(f"q-{index}" for index in range(8)),),
        vintage_ids=(tuple(f"v-{index}" for index in range(8)),),
        quote_key_ids=("close",),
        analysis_time_ns=100,
        clock="event",
        reference_data_id="reference",
        market_data_id="market",
        resolved_id="resolved",
        prepared_id="panel",
        capacity=8,
    )
    returns = ReturnResult(
        values=jnp.full((1, 7), 0.01),
        valid_mask=jnp.ones((1, 7), dtype=bool),
        status=jnp.zeros((1, 7), dtype=jnp.int32),
        interval_start_ns=event[:-1][None, :],
        interval_end_ns=event[1:][None, :],
        batches=(),
        batch_channels=(),
        source_panel_id="panel",
        adjustment_id="raw",
        definition_id="return-definition",
        result_id="returns",
        kind="simple",
    )
    contract = FeatureLabelContract(
        (FeatureDefinition("last-return", 0, (1,)),),
        LabelDefinition("forward-return", 0, horizon=2, aggregation="compound"),
        row_capacity=5,
        decision_clock="event",
    )

    dataset, evidence = prepare_feature_labels(panel, returns, contract)

    assert dataset.label_start_times_ns[0] == dataset.decision_times_ns[0]
    assert dataset.label_end_times_ns[0] == 30
    assert jnp.isclose(dataset.labels[0], (1.01**2) - 1.0)
    assert not dataset.row_valid[1]
    assert evidence.future_feature_count == 1


def test_walk_forward_purges_complete_label_overlap_and_embargoes_by_time():
    decision = jnp.arange(12, dtype=jnp.int64) * 10
    dataset = PreparedFeatureLabelDataset(
        features=jnp.arange(24.0).reshape(12, 2),
        labels=jnp.arange(12.0),
        row_valid=jnp.ones((12,), dtype=bool),
        decision_times_ns=decision,
        feature_start_times_ns=decision - 20,
        feature_available_times_ns=decision,
        label_start_times_ns=decision,
        label_end_times_ns=decision + 25,
        label_available_times_ns=decision + 25,
        asset_indices=jnp.zeros((12,), dtype=jnp.int32),
        feature_names=("lag-1", "lag-2"),
        target_quote_key_id="synthetic-close",
        panel_id="panel",
        return_result_id="returns",
        contract_id="features",
        dataset_id="dataset",
        row_capacity=12,
    )
    definition = WalkForwardDefinition(
        training_span_ns=60,
        validation_span_ns=20,
        test_span_ns=20,
        step_ns=20,
        purge_ns=0,
        embargo_ns=10,
        minimum_training_rows=2,
        fold_capacity=1,
    )

    plan, evidence = prepare_walk_forward(dataset, definition)
    fold = plan.fold(0)

    assert fold.valid
    assert jnp.all(dataset.label_end_times_ns[fold.train_mask] <= 60)
    assert jnp.all(dataset.decision_times_ns[fold.train_mask] < 50)
    assert fold.purged_mask[4]
    assert fold.embargoed_mask[5]
    assert jnp.array_equal(
        dataset.decision_times_ns[fold.validation_mask], jnp.asarray([60, 70])
    )
    assert jnp.array_equal(
        dataset.decision_times_ns[fold.test_mask], jnp.asarray([80, 90])
    )
    assert evidence.label_overlap_removed[0]
    assert evidence.embargo_respected[0]
