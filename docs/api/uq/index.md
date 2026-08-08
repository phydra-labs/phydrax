# Uncertainty quantification

`phydrax.uq` provides covariance-factor and conditional Gaussian algebra, guarded
nonlinear moment transforms, covariance- and square-root-form Kalman methods,
continuous-discrete Gaussian filtering and smoothing, and stationary stochastic
spectra. Its sensitivity layer covers global Sobol indices, stochastic pathwise and
score gradients, matrix-free Fisher and Gauss--Newton actions, empirical directions,
and guarded information-design objectives.

The same namespace also provides explicit exact and factorized posterior problems,
bounded global and gradient-based local MAP optimization, BlackJAX NUTS/HMC,
fixed-step SGLD/SGNHT with control variates, flow-assisted NUTS, Pathfinder, adaptive
tempered SMC, dense and structured Laplace approximations, exact, sparse, and
correlated-output Gaussian-process model discrepancy, predictive fields, coherent
stochastic models, likelihoods and proper scores, first-order covariance propagation,
normalized errors-in-variables inference, conformal calibration, and uncertain-input
propagation.

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

Gaussian results expose validity and status rather than hiding a numerical repair.
Regularization is always explicit. Filter histories restore physical case/event axes,
masks, schedule validity, IDs, and solver/approximation/backend provenance; method IDs
distinguish covariance, square-root, nonlinear-transform, continuous-discrete, and
spectral routes. Singular positive-semidefinite factors remain singular, and complex
covariances and spectra use conjugate adjoints.

- [Predictive results](predictive.md)
- [Shared positive-definite kernels](../kernels.md)
- [Inference and ensembles](inference.md)
- [Likelihoods, process diagnostics, calibration, and retention](calibration.md)
- [Filtering, smoothing, and state estimation](filtering.md)
- [Neural-operator uncertainty](operator.md)
- [Uncertain-input propagation](propagation.md)
- [Global and local sensitivity, information actions, and design](sensitivity.md)
