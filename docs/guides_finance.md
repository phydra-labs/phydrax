# Financial mathematics

Phydrax finance is a mathematical substrate, not a data service, venue gateway, accounting engine, or universal pricer. It keeps contractual meaning, observed data, probabilistic law, numerical method, and qualification evidence as separate objects so that a successful computation cannot silently imply a broader claim.

## Ownership boundaries

- **Core** owns stable identifiers, currencies, exact minor-unit amounts, floating monetary arrays, dates, timestamps, calendars, schedules, P/Q/stress laws, scenario sets, and evidence-plane bindings.
- **Market** owns data lineage, assets and instruments, quote/fixing vintages, bitemporal selection, risk-factor layouts, and prepared market states.
- **Contracts** owns obligations and their host resolution into cashflows, exercise schedules, and settlement terms. Trades and positions add business identity without changing payoff mathematics.
- **Curves** owns curve representations, interpolation/extrapolation policy, bootstrap equations, calibration residuals, masks, and status.
- **Models** owns measure-neutral rates and credit structures. It does not decide whether parameters describe historical dynamics or pricing dynamics.
- **Calibration and valuation** own route-specific objective/preparation/result/replay records. Valuation routes are analytic, Fourier, PDE, simulation, exercise, or BSDE; there is no universal `price()`.
- **Econometrics** owns point-in-time historical datasets, estimators, filters, diagnostics, and replay under a physical law.
- **Portfolio** owns ledgers, objectives, constraints, compilation, and optimization.
- **Risk and exposure** own sensitivities, scenarios, risk measures, legal netting/collateral resolution, and separately identified XVA components.
- **Execution** owns order-flow semantics, impact, market-making and control policies. Its models do not establish venue access or achievable fills.
- **Arbitrage** owns static-arbitrage diagnostics and candidate martingale-transport computations.
- **Qualification and interchange** own exact support evidence, archives, deterministic records/tables, and the narrow FpML boundary.

## P, Q, and stress are not interchangeable

`PhysicalLaw` identifies a P-law used for historical estimation, forecasting, controls, and backtesting. `PricingLaw` identifies a Q-law together with its measure, numeraire, and collateral convention. `StressLaw` identifies deliberately imposed scenarios without pretending that they are historical or risk-neutral probabilities.

Model structures are measure-neutral. Pass the structure and the appropriate typed law together. Do not estimate a P-law, rename it as a pricing law, or reuse stress weights as probabilities. `FinanceEvidenceBinding` likewise keeps data, model, numerical, and use evidence in four distinct collections.

## Host resolution and device computation

```mermaid
flowchart LR
    A[Authored or admitted records] --> B[Host calendar and contract resolution]
    B --> C[Host quote selection and layout]
    C --> D[Fixed-shape arrays, masks, status]
    D --> E[JAX preparation]
    E --> F[Device solve]
    F --> G[Replay and qualification]
    G --> H[Checksum-bound named-array archive]
```

Calendars, identifiers, contract terms, legal sets, quote-vintage selection, and capacity choice are host concerns. Cross that boundary once: materialize fixed-shape JAX arrays with explicit validity masks and status. Compiled computation must not parse documents, inspect calendars, call a feed, grow shapes, or resolve contracts.

A route follows its own definition → resolved input → plan → prepared state → result → replay sequence. Preparation and solve timings are reported separately; neither timing is an acceptance criterion.

## Correctness-first benchmark suites

`benchmarks/finance_native.py` provides synthetic `valuation`, `econometrics`, `portfolio`, `exposure`, `execution`, and `advanced` suites. Each report has four separate phases: preparation, first solve, replay, and repeated timing. Correctness is checked against `benchmarks/finance_reference.json`, replay is checked independently, and either failure produces a nonzero exit. Timing samples are descriptive only and never affect `passed`.

## Candidate ceilings

Martingale transport and rough, deep, operator, or tensor finance routes are candidates. `advanced_finance_support` enforces that ceiling in its exact `SupportTuple`. Candidate evidence can demonstrate a named finite problem and replay, but cannot be promoted by wording into a live-trading, production, regulatory, or universal-capability claim.

## Non-goals

The finance package does not include live or vendor feeds, network lookup, venue connectivity, arbitrary executable oracles, legal advice, accounting policy, regulatory reporting, compliance certification, guaranteed hedging, vendor parity, timing-based release gates, or claims about tradeability. The checked-in examples and fixtures are synthetic and carry explicit provenance and rights.
