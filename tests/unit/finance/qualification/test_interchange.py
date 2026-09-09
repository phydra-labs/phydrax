#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from phydrax.finance.core import (
    Currency,
    FinancialTimestamp,
    TemporalAdmissibilityPolicy,
)
from phydrax.finance.interchange import (
    export_fpml_contract,
    FinanceRecordBatch,
    import_fpml_contract,
    market_records_to_polars,
    market_snapshot_to_records,
    polars_to_market_records,
    records_to_market_snapshot,
)
from phydrax.finance.market import (
    DataLineage,
    MarketDataSnapshot,
    QuoteKey,
    QuoteObservation,
)


_FPML = """<dataDocument xmlns="http://www.fpml.org/FpML-5/confirmation">
  <trade>
      <tradeHeader>
        <partyTradeIdentifier><tradeId>FX-1</tradeId></partyTradeIdentifier>
        <tradeDate>2026-01-02</tradeDate>
      </tradeHeader>
      <fxSingleLeg>
        <exchangedCurrency1>
          <payerPartyReference href="party-a"/>
          <receiverPartyReference href="party-b"/>
          <paymentAmount><currency>EUR</currency><amount>1000000</amount></paymentAmount>
        </exchangedCurrency1>
        <exchangedCurrency2>
          <payerPartyReference href="party-b"/>
          <receiverPartyReference href="party-a"/>
          <paymentAmount><currency>USD</currency><amount>1102500</amount></paymentAmount>
        </exchangedCurrency2>
        <valueDate>2026-01-06</valueDate>
      </fxSingleLeg>
  </trade>
</dataDocument>"""


def _timestamp(event: int, available: int, vintage: str) -> FinancialTimestamp:
    policy = TemporalAdmissibilityPolicy(True, True, False)
    return FinancialTimestamp(event, event + 1, event + 2, available, vintage, policy)


def test_market_records_round_trip_all_identity_bearing_children():
    usd = Currency("USD", 2)
    key = QuoteKey("SYNTH-EQUITY-1", "close", venue="SYNTHETIC", currency=usd)
    lineage = DataLineage(
        "authored-synthetic",
        "market-close",
        publisher_id="phydra-labs",
        transformation_ids=("none",),
    )
    observation = QuoteObservation(key, 100.0, _timestamp(100, 103, "a"), lineage)
    snapshot = MarketDataSnapshot(
        (observation,),
        snapshot_time=_timestamp(200, 203, "snapshot"),
        reference_data_id="synthetic-reference",
    )

    records = market_snapshot_to_records(snapshot)
    restored = records_to_market_snapshot(records)
    assert restored.snapshot_id == snapshot.snapshot_id
    assert restored.observations[0].observation_id == observation.observation_id

    frame = market_records_to_polars(records)
    reframed = polars_to_market_records(
        frame,
        primary_key=("observation_id",),
        context=records.context(),
    )
    assert reframed.batch_id == records.batch_id


def test_finance_record_batch_is_primary_key_deterministic():
    first = FinanceRecordBatch(
        "market-observation",
        ({"id": "b", "value": 2.0}, {"id": "a", "value": 1.0}),
        primary_key=("id",),
    )
    second = FinanceRecordBatch(
        "market-observation",
        ({"value": 1.0, "id": "a"}, {"value": 2.0, "id": "b"}),
        primary_key=("id",),
    )
    assert first.batch_id == second.batch_id
    assert [record["id"] for record in first.to_records()] == ["a", "b"]


def test_fpml_subset_round_trips_and_unknown_terms_refuse_without_partial_contract():
    imported = import_fpml_contract(_FPML)
    assert imported.accepted
    assert imported.contract_record()["contract_type"] == "fx-forward"
    replay = import_fpml_contract(export_fpml_contract(imported))
    assert replay.accepted
    assert replay.contract_record() == imported.contract_record()

    unknown = "<calculationAgent>party-a</calculationAgent><valueDate>"
    refused = import_fpml_contract(_FPML.replace("<valueDate>", unknown))
    assert not refused.accepted
    assert refused.contract_type is None
    assert any("calculationAgent" in term for term in refused.unsupported_terms)
