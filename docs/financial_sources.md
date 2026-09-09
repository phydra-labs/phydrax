# Financial sources, provenance, and rights

## Included authored fixtures

The repository ships only the following authored synthetic finance fixtures:

| Fixture | Content | Source | Rights |
| --- | --- | --- | --- |
| `tests/fixtures/finance/calendar_synthetic.json` | finite 2026 example holiday set | manually authored by Phydra Labs on 2026-09-08 | CC0-1.0; commercial use, redistribution, and training permitted |
| `tests/fixtures/finance/market_synthetic.json` | fictional rates, FX, and equity quote vintages | manually authored finite values; no feed or vendor input | CC0-1.0; commercial use, redistribution, and training permitted |
| `tests/fixtures/finance/reference_synthetic.json` | fictional identifiers, currencies, assets, and an instrument | manually authored fictional records | CC0-1.0; commercial use, redistribution, and training permitted |

Each fixture embeds its own provenance and rights record. The files contain no copied market observations, proprietary holiday service output, customer records, or claims of currentness. Their values are suitable for deterministic examples only.

## External source admission

Phydrax does not download or bundle external financial data. A caller admitting an external source is responsible for supplying and retaining:

- source, dataset, publisher, acquisition, and transformation identities;
- event, publication, receipt, and availability clocks for each vintage;
- an immutable checksum and byte count for any retained artifact;
- license identifier and explicit commercial-use, redistribution, training, and export permissions;
- uncertainty, missing-data, correction, and survivorship treatment;
- the exact calendar, convention, identifier mapping, and unit interpretation;
- requalification triggers when source content, permissions, or interpretation changes.

Data-source evidence remains separate from model, numerical, and use evidence. Permission to use a dataset does not establish accuracy, and numerical agreement does not establish permission.

## Convention and document boundaries

Currency codes and minor units are caller-supplied `Currency` records. Calendars are caller-supplied `CalendarSnapshot` records. Day-count and business-day labels select explicit implemented mathematics; they do not certify agreement with a particular contract unless that contract was resolved against the same records.

The FpML adapter accepts only the documented single-leg FX-forward subset. It is not a general document validator and does not claim conformance of unsupported products or extensions. Unknown terms are refused without a partial contract.

Nothing in the source register is legal, accounting, tax, regulatory, or compliance advice, and no fixture represents a live or tradeable market.
