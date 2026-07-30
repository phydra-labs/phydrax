# Uncertainty quantification

`phydrax.uq` provides explicit Bayesian posterior problems, MAP optimization,
BlackJAX NUTS/HMC, Pathfinder, adaptive tempered SMC, dense and structured
Laplace approximations, exact/sparse/correlated-output Gaussian-process model
discrepancy, predictive fields, coherent stochastic models, likelihoods and
proper scores, conformal calibration, uncertain-input propagation, and global
sensitivity.

Neural-operator predictions use an operator-aware layer over the same protocol so
source/query geometry, physical case axes, masks, quadrature, and channel metadata
survive stochastic reduction, posterior prediction, calibration, and scoring.

All predictive samples retain explicit `coordax.Field` dimensions. Source axes are
labeled as epistemic, input, observation, process, or numerical uncertainty; no
method infers a source from axis position. Complete-field Gaussian and conditional
flow operator distributions use the same labels and query metadata.

Stochastic predictions carry a source-axis `valid` mask. Reductions ignore invalid
realizations; evaluators can instead request fail-fast validation.

- [Predictive results](predictive.md)
- [Inference and ensembles](inference.md)
- [Likelihoods and calibration](calibration.md)
- [Neural-operator uncertainty](operator.md)
- [Uncertain-input propagation](propagation.md)
- [Global sensitivity](sensitivity.md)
