# Finance interchange

## Canonical records and tables

`FinanceRecordBatch` stores finite JSON cells, canonical column order, an explicit primary key, and a content address. Row order at input is irrelevant; duplicate primary keys are rejected. `market_snapshot_to_records` retains quote-key, timestamp-policy, lineage, and snapshot identities. `records_to_market_snapshot` reconstructs typed records and verifies every content address.

Polars is only a table surface:

```python
from phydrax.finance.interchange import (
    FinanceRecordBatch,
    market_records_to_polars,
    polars_to_market_records,
)

batch = FinanceRecordBatch(
    "market-observation",
    (
        {"observation_id": "synthetic-a", "value": 100.0},
        {"observation_id": "synthetic-b", "value": 101.0},
    ),
    primary_key=("observation_id",),
    context={"source": "authored-synthetic"},
)
frame = market_records_to_polars(batch)
restored_batch = polars_to_market_records(
    frame,
    primary_key=("observation_id",),
    context=batch.context(),
)
assert restored_batch.batch_id == batch.batch_id
```

Dataframes do not enter compiled finance kernels. Convert on the host, validate identity, then prepare a `MarketState` with fixed-shape JAX arrays, masks, and status.

## Supported FpML subset

`import_fpml_contract` deliberately supports one inert contract subset: an FpML confirmation document containing exactly one trade and one `fxSingleLeg`. The accepted fields are:

- one trade identifier and trade date;
- exactly two exchanged currencies;
- payer and receiver references for each exchange;
- positive finite decimal amount and three-letter uppercase currency for each exchange;
- one value date.

The importer rejects oversized input, DTD/entity declarations, malformed XML, unknown elements, unknown attributes, missing or duplicate required terms, unsupported products, invalid dates, invalid currencies, and invalid amounts. Refusal is returned as an `FpMLImportResult` with `accepted=False`, no partial contract, and sorted `unsupported_terms`.

`export_fpml_contract` emits only the same FX-forward record. It does not preserve unsupported extensions and will not reinterpret them. This is not a general FpML implementation, a schema-validation service, or a claim of confirmation-system interoperability.
