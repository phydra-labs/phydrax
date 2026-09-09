# Rates, credit, exposure, and XVA

## Resolve obligations before modeling

Rates and credit products live under `phydrax.finance.contracts`. Host resolution uses `ContractResolutionContext`, `ContractResolutionPlan`, and `resolve_contract`, yielding `ResolvedContract` or an explicit `ContractResolutionStatus`. Resolved deposits, FRAs, futures, fixed/floating/inflation legs, OIS/IBOR/basis/cross-currency swaps, and fixed/floating/zero-coupon bonds expose known `CashflowBatch` obligations. `ContractResolutionReplay` checks the same calendar, fixing, reference-data, and convention identities.

Credit obligations are separate from models. `CreditDefaultSwapContract` and `DefaultableBondContract` resolve contractual terms; `cds_leg_values`, `cds_par_spread`, `credit_event_cashflows`, and `defaultable_bond_cashflows` expose typed premium, protection, recovery, and timing semantics.

## Measure-neutral rates and credit models

`VasicekModel`, `CIRModel`, `CIRPlusPlusModel`, `HullWhiteModel`, `FiniteFactorHJMModel`, and `LiborMarketModel` are model structures. `RatesLaw` and an explicit `PricingLaw` determine interpretation. Public zero-coupon and drift functions retain the selected measure and tenor assumptions; `simulate_short_rate_paths` returns a `RatesPathBatch` rather than changing the model.

`IntensityCreditModel`, `ReducedFormCreditModel`, and `StructuralCreditModel` keep default mechanism, recovery, intensity transformation, and timing convention explicit. `survival_probability`, `default_probability`, `hazard_rate`, `intensity_from_factors`, `state_dependent_intensity`, and `simulate_default_events` require the corresponding law/model inputs. Historical calibration under P does not become Q calibration by reuse.

## Exposure and separate XVA components

`NettingSet` and `CollateralAgreement` are host legal/operational inputs. `prepare_collateral_agreement`, `collateral_target`, `evolve_collateral`, `net_trade_values`, and `resolve_closeout` retain timing, thresholds, disputes, simultaneous-default rule, and close-out source. `ExposureProfile` carries fixed-shape pathwise values, masks, and identities.

`compute_cva`, `compute_dva`, `compute_fva`, `compute_mva`, and `compute_kva` produce distinct typed results. `FundingPolicy`, `MarginFundingPolicy`, and `EconomicCapitalPolicy` are caller-supplied assumptions. `assemble_xva` sums only compatible components; economic KVA is not a regulatory-capital claim. `evaluate_exposure_bsde` is a separately identified route, not a replacement for legal-set resolution.

XVA qualification uses `exposure_xva_support` and names netting, collateral, default model, wrong-way-risk treatment, backend, and precision. Results do not claim enforceability, accounting treatment, regulatory approval, or completeness beyond the supplied contracts and paths.
