# Finance cookbook

All values below are synthetic. No network, feed, vendor file, or executable provider is used.

## Resolve host semantics, then prepare device arrays

```python
from phydrax.finance.core import (
    BusinessDayRule,
    CalendarSnapshot,
    Currency,
    DayCount,
    FinanceDate,
    FinancialTimestamp,
    PricingLaw,
    ScheduleRule,
    TemporalAdmissibilityPolicy,
    resolve_schedule,
)
from phydrax.finance.market import (
    DataLineage,
    MarketDataSnapshot,
    QuoteKey,
    QuoteObservation,
    RiskFactorKey,
    RiskFactorLayout,
)

usd = Currency("USD", 2)
calendar = CalendarSnapshot(
    "synthetic-nyc-2026",
    (FinanceDate.from_iso("2026-01-01"), FinanceDate.from_iso("2026-12-25")),
    (5, 6),
    "Phydra-authored synthetic dates; CC0-1.0",
)
rule = ScheduleRule(
    FinanceDate.from_iso("2026-01-02"),
    FinanceDate.from_iso("2027-01-02"),
    3,
    calendar.calendar_id,
    BusinessDayRule.MODIFIED_FOLLOWING,
    "none",
    False,
    2,
    DayCount.ACT_360,
)
schedule = resolve_schedule(rule, calendar, capacity=8)

clock_policy = TemporalAdmissibilityPolicy(True, True, False)
quote_time = FinancialTimestamp(100, 101, 102, 103, "close-a", clock_policy)
snapshot_time = FinancialTimestamp(200, 201, 202, 203, "snapshot-a", clock_policy)
lineage = DataLineage(
    "phydra-authored-synthetic",
    "synthetic-close",
    publisher_id="phydra-labs",
)
quote_key = QuoteKey(
    "SYNTH-EQUITY-1",
    "close",
    venue="SYNTHETIC",
    currency=usd,
)
observation = QuoteObservation(quote_key, 100.0, quote_time, lineage)
snapshot = MarketDataSnapshot(
    (observation,),
    snapshot_time=snapshot_time,
    reference_data_id="synthetic-reference",
)
layout = RiskFactorLayout((RiskFactorKey("spot", quote_key),))
market_state = snapshot.prepare(layout, snapshot_time, capacity=1)

pricing_law = PricingLaw(
    "synthetic-usd-q",
    "authored synthetic pricing-law example",
    layout.layout_id,
    "synthetic-filtration",
    "synthetic-risk-neutral-measure",
    "synthetic-money-market-numeraire",
    "uncollateralized",
)
```

`calendar`, `schedule`, contract resolution, quote selection, and layout construction remain on the host. `market_state` and route-specific prepared values contain fixed-shape arrays, validity masks, and status suitable for JAX computation. The model structure and `pricing_law` are passed separately.

## Deterministic market interchange

```python
from phydrax.finance.interchange import (
    market_records_to_polars,
    market_snapshot_to_records,
    records_to_market_snapshot,
)

records = market_snapshot_to_records(snapshot)
frame = market_records_to_polars(records)
restored = records_to_market_snapshot(records)
assert restored.snapshot_id == snapshot.snapshot_id
```

The dataframe is a host interchange surface. It is not captured by a compiled valuation.

## Exact route qualification

```python
from phydrax.finance.qualification import (
    build_finance_qualification_matrix,
    evaluate_finance_campaign,
    valuation_support,
)

support = valuation_support(
    "analytic",
    product="european-option",
    model="black-scholes",
    pricing_law=pricing_law.law_id,
)
matrix = build_finance_qualification_matrix((support,))
campaign = evaluate_finance_campaign(matrix, (), at_time=1_800_000_000)
assert campaign.coverage.outcome == "inconclusive"
```

The empty evidence sequence is intentionally inconclusive. A passing campaign requires current reviewed reference, scientific, unit, and operational evidence for the exact support-tuple ID.
