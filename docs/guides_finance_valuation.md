# Finance valuation

Valuation always combines a resolved contract, prepared market inputs, a measure-neutral model structure, and an explicit `PricingLaw`. The pricing law owns measure, numeraire, collateral convention, factor layout, and filtration. A `PhysicalLaw` is not accepted as a pricing-law substitute, and a `StressLaw` remains a stress descriptor rather than a probability law.

## Results and evidence

All routes return `ValuationResult` with a typed `ValuationStatus` and `ValuationEvidence`. `ValuationReplayEvidence` is separate from the first solve. Validity, numerical status, law identity, and route identity are observable outputs; a scalar number alone is not a valuation record.

## Analytic route

The public analytic functions are `evaluate_black_scholes_european`, `evaluate_black_scholes_digital`, `evaluate_black76_european`, `evaluate_black76_digital`, `evaluate_bachelier_european`, and `evaluate_bachelier_digital`. Their matching implied-volatility functions return `ImpliedVolatilityResult`, including failure status for inadmissible prices or unsuccessful inversion. These functions cover only their stated payoff/model assumptions.

## Requested sensitivities

`evaluate_aad_greeks` computes first derivatives for every parameter and only
the second-order diagonal entries named by `GreekRequest.second_order_names`.
An empty second-order request performs no curvature evaluation. Otherwise,
one linearization shares the value and gradient calculation with requested
basis-vector Hessian actions; no full Hessian is constructed. Unrequested
diagonal entries remain NaN, and non-finite requested derivatives are refused.
The diagonal workspace grows linearly with the parameter count rather than
quadratically, including when every diagonal entry is requested.


## Candidate learned and compressed routes

`assess_deep_bsde_candidate`, `evaluate_operator_valuation_candidate`, and `assess_tensor_valuation_candidate` retain explicit applicability, causality, resource, approximation, and support evidence. `deep_bsde_independent_validation`, `operator_independent_validation`, and `tensor_independent_validation` require independently supplied validation rather than training echoes. Tensor routes additionally expose `tensor_rank_evidence`.

Deep BSDE, operator, and tensor results remain candidates. Training loss, replay agreement, low tensor rank, or benchmark speed cannot promote them to universal pricing, continuum accuracy, executable hedging, or live use. Qualification is attached to an exact `valuation_support` or `advanced_finance_support` tuple and requires independent data, model, numerical, and use evidence.
