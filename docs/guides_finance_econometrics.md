# Financial econometrics

Econometric routes estimate historical P-law behavior. They require an explicit `PhysicalLaw`; they do not infer a `PricingLaw` from historical fit.

## Point-in-time data first

`PointInTimePanelDefinition` and `MarketEventStreamDefinition` define host-side data semantics. `resolve_point_in_time_panel`, `prepare_point_in_time_panel`, and `prepare_market_event_stream` preserve availability clocks, masks, and lineage. `replay_market_data` checks the selected vintages. `ReturnDefinition`, `compute_returns`, `apply_corporate_actions`, and `compute_realized_measures` keep return convention, corporate-action treatment, and realized-measure definition explicit.

Feature/label construction uses `FeatureDefinition`, `LabelDefinition`, `FeatureLabelContract`, and `prepare_feature_labels`. `DecisionClock` and aggregation records prevent future information from entering a feature. `replay_feature_labels` is independent evidence against accidental look-ahead.

## Estimator families

- `fit_arima`, `fit_var`, `test_cointegration`, and `fit_vecm` cover linear time-series definitions and return typed fits; forecasting uses `forecast_arima` or `forecast_var`.
- `fit_garch`, `fit_gjr_garch`, `fit_egarch`, `fit_har`, and `fit_stochastic_volatility` cover explicitly named volatility assumptions.
- `prepare_financial_state_space`, `fit_financial_state_space`, and `fit_regime_model` retain state, regime, law, and numerical identities.
- `prepare_covariance`, `fit_covariance`, `fit_factor_model`, and `clean_covariance_rmt` produce typed covariance evidence rather than silently repairing an input.
- `prepare_point_process`, `fit_hawkes`, and `score_market_point_process` retain point-process support and stability diagnostics.

## Experiments and inference

`WalkForwardDefinition` and `prepare_walk_forward` define chronological folds; `evaluate_walk_forward` reports fold evidence without using timing as acceptance. `compare_forecasts` and `evaluate_multiple_testing` require explicit comparison and hypothesis-family definitions. Data evidence, model evidence, numerical evidence, and intended-use evidence remain separate throughout.

No econometric result establishes stationarity outside its checks, causal identification, future performance, pricing-law validity, data permission, or live-trading fitness.
