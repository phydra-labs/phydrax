# Uncertainty quantification

`phydrax.uq` provides explicit Bayesian posterior problems, MAP optimization,
BlackJAX NUTS/HMC, Pathfinder, adaptive tempered SMC, dense and structured
Laplace approximations, exact/sparse/correlated-output Gaussian-process model
discrepancy, predictive fields, coherent stochastic models, likelihoods and
proper scores, conformal calibration, uncertain-input propagation, and global
sensitivity.

All predictive samples retain explicit `coordax.Field` dimensions. Source axes are
labeled as epistemic, input, or observation uncertainty; no method infers a source
from axis position.

Stochastic predictions carry a source-axis `valid` mask. Reductions ignore invalid
realizations; evaluators can instead request fail-fast validation.

- [Predictive results](predictive.md)
- [Inference and ensembles](inference.md)
- [Likelihoods and calibration](calibration.md)
- [Uncertain-input propagation](propagation.md)
- [Global sensitivity](sensitivity.md)
