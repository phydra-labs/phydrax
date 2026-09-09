# Finance market data and curves

## Causal market ownership

`DataLineage` identifies an admitted source and transformations. `QuoteKey` identifies a field; every `QuoteObservation` preserves event, publication, receipt, and availability clocks through `FinancialTimestamp`. `MarketDataSnapshot` retains all vintages and resolves only observations available at the decision clock. `ReferenceDataSnapshot` separately owns assets, instruments, and currencies.

On the host, define a `RiskFactorLayout` and ask the snapshot to prepare it. The resulting `MarketState` is the device boundary: fixed-shape values, validity mask, status, event times, availability times, and observation IDs. Missing, stale, duplicate, causally unavailable, nonfinite, or crossed quotes remain explicit; they are never replaced by a hidden fallback.

`FixingSeries`, `MarketEventStream`, `PointInTimePanel`, and `MarketReplay` preserve historical ordering rather than overwriting vintages. `build_time_bars`, `price_returns`, `realized_measure`, and `adjust_for_corporate_actions` return new typed records. FX conversion uses `FXConversionGraph`; `fx_triangle_consistency` reports a diagnostic rather than implying executable arbitrage.

## Curve lifecycle

A curve begins with `CurveDefinition`, `CurveGrid`, and `InterpolationPolicy`. Preparing it yields `PreparedCurve`, whose query quantities are explicitly named by `CurveQuantity`; extrapolation is governed by `ExtrapolationMode`. `CurveSet` preserves the identity of every component curve, and `CurveSensitivity` binds derivatives to the same representation and grid.

Bootstrapping is route-specific. `SingleCurveBootstrapPlan` and `MultiCurveBootstrapPlan` consume typed instruments such as `DepositBootstrapInstrument`, `ForwardRateBootstrapInstrument`, `ParSwapBootstrapInstrument`, `BasisSwapBootstrapInstrument`, `ZeroRateBootstrapInstrument`, `SurvivalProbabilityBootstrapInstrument`, and `HazardRateBootstrapInstrument`. `BootstrapSolverPolicy` controls numerical work. `CurveBootstrapResult` reports residual, validity, status, and identity; `CurveBootstrapReplay` checks the prepared route independently.

Calendars, schedules, quote selection, instrument resolution, and curve-grid construction stay on the host. Prepared curve arrays can enter JAX valuation. A successful finite bootstrap does not establish data rights, model adequacy, extrapolation validity, or support outside the exact `curve_support` tuple.
