# Uncertainty quantification

`phydrax.uq` provides explicit exact and factorized posterior problems, bounded
global and gradient-based local MAP optimization, BlackJAX NUTS/HMC, fixed-step
SGLD/SGNHT with control variates, flow-assisted NUTS, Pathfinder, adaptive tempered
SMC, dense and structured Laplace approximations, exact, sparse, and
correlated-output Gaussian-process model discrepancy,
predictive fields, coherent stochastic models, likelihoods and proper scores,
matrix-free first-order covariance propagation, normalized errors-in-variables
inference, conformal calibration, uncertain-input propagation, and global sensitivity.

Stochastic-gradient samplers consume deterministic, content-addressed minibatch
sources and preserve chain/draw structure, replay state, throughput, and mixing
evidence. Their production draws are explicitly labeled unadjusted fixed-step
approximations; they do not substitute for NUTS or Laplace reference checks when
full-data inference is feasible.

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
- [Likelihoods, process diagnostics, calibration, and retention](calibration.md)
- [Filtering, smoothing, and state estimation](filtering.md)
- [Neural-operator uncertainty](operator.md)
- [Uncertain-input propagation](propagation.md)
- [Global sensitivity](sensitivity.md)
