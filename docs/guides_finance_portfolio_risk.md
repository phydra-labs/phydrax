# Finance portfolio and risk

## Portfolio definition and compilation

`PortfolioLedger` and `portfolio_ledger_snapshot` own holdings, cash, lots, and trade history. A `PortfolioProblem` combines a `ForecastLaw`, `PortfolioScaling`, one public objective, and `PortfolioConstraints`. Objectives include `MeanVarianceObjective`, `CVaRObjective`, `EVaRObjective`, `SpectralRiskObjective`, `TrackingErrorObjective`, `DrawdownRiskObjective`, `FiniteScenarioKellyObjective`, `BlackLittermanObjective`, and `KLDivergenceRobustObjective`. Robust second-order and scenario-tree structure is explicit through `RobustSOCConstraint` and `ScenarioTree`.

`compile_portfolio_problem` lowers that definition to a native canonical program and returns `PortfolioCompiled` plus a deterministic `PortfolioPlan`. A native optimizer solves that program; `decode_portfolio_decision` independently checks finite shape and feasibility before producing `PortfolioDecision`. `portfolio_result_from_native` binds the decision and `PortfolioOptimizerCertificate` into `PortfolioResult`. `refresh_portfolio_compilation` accepts numerical refreshes only when structure is unchanged.

Realized performance is not an optimizer echo. `replay_self_financing` uses `PortfolioReplayMarket` and `ReplayCostInputs` to return a separate `RealizedPortfolioResult`.

## Risk measures and scenarios

Historical and Gaussian VaR/ES are explicit functions (`historical_var_es`, `gaussian_var_es`, `value_at_risk`, `expected_shortfall`). `cvar_atoms`, `entropic_value_at_risk`, `spectral_risk`, `drawdown_risk`, and `kelly_risk` retain their distinct definitions and assumptions.

Scenario work starts from `FinancialScenarioSet` and its law ID. `evaluate_scenarios`, `reweight_scenarios`, `entropy_tilt_scenarios`, and `reduce_scenarios` return new evidence-bearing records. A stress law is not silently treated as P or Q. `market_factor_risk`, `explain_factor_pnl`, `explain_pnl`, `brinson_attribution`, and `stress_test` separate exposure, realized P&L explanation, attribution, and stress results.

`walk_forward_splits`, `nested_backtest_splits`, and `validate_no_lookahead` establish chronological evaluation boundaries; `evaluate_walk_forward` and `evaluate_nested_backtest` report their own decisions and results. Backtests do not establish future performance, data rights, or a compliant production process. Qualification covers only the exact `portfolio_support` tuple and measured scale.
