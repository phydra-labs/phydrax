# Filtering, smoothing, and state estimation

All native filters consume one `phydrax.stochastic.StateSpaceProblem` and preserve its
physical case axes, observation schedule, masks, case IDs, model ID, problem ID, and
sequence ID. Batch functions call the same streaming step used by online execution.
Algorithm-specific status arrays and validity masks distinguish inactive schedule
entries from numerical or transition failures.

## Exact linear-Gaussian filtering

`kalman_filter` and `kalman_filter_step` use Joseph-form covariance updates and expose
innovation covariances, normalized innovation squared values, incremental likelihoods,
and failure status. `rts_smoother` computes the fixed-interval backward recursion;
`sample_kalman_smoother_paths` draws coherent conditional paths rather than independent
time marginals.

::: phydrax.uq.initialize_kalman_filter

---

::: phydrax.uq.kalman_filter_step

---

::: phydrax.uq.kalman_filter

---

::: phydrax.uq.rts_smoother

---

::: phydrax.uq.sample_kalman_smoother_paths

---

::: phydrax.uq.kalman_innovation_diagnostics

## Bootstrap particle filtering

Particle states retain normalized log weights, root-key lineage, algorithm settings,
and streaming position. Histories retain forecast particles, posterior weights,
post-resampling particles, ancestor indices, transition validity, effective sample
sizes, and likelihood increments. Ancestor tracing and backward simulation provide two
explicit smoothing semantics.

::: phydrax.uq.initialize_particle_filter

---

::: phydrax.uq.particle_filter_step

---

::: phydrax.uq.bootstrap_particle_filter

---

::: phydrax.uq.sample_particle_ancestry_paths

---

::: phydrax.uq.sample_particle_backward_paths

---

::: phydrax.uq.particle_filter_predictive

---

::: phydrax.uq.particle_filter_diagnostics

`particle_posterior_measure` exposes forecast particles and posterior log
weights as a planless weighted integration target. It retains case and filtering
time axes, masks inactive or failed particles, preserves ancestor indices, and
does not report an IID standard error.

::: phydrax.uq.particle_posterior_measure

## Ensemble transform filtering

The deterministic ensemble transform Kalman filter performs analysis solves in
observation and ensemble-member space. It does not construct a state covariance matrix,
which keeps the method useful for large fields and operator states. Inflation and
covariance regularization are explicit run settings. The smoother performs member-space
regression over the retained forecast and analysis ensembles.

::: phydrax.uq.initialize_ensemble_filter

---

::: phydrax.uq.ensemble_filter_step

---

::: phydrax.uq.ensemble_transform_kalman_filter

---

::: phydrax.uq.ensemble_kalman_smoother

---

::: phydrax.uq.ensemble_filter_predictive

---

::: phydrax.uq.ensemble_filter_diagnostics

## Exact state-space likelihoods

`exact_state_space_log_likelihood` dispatches only to a mathematically exact
finite-state forward recursion or linear-Gaussian Kalman innovation likelihood.
`StateSpaceMarginalLikelihood` exposes that result to posterior inference without
resampling latent paths. `state_space_identifiability` reports effective observation
rank and prior-to-posterior contraction; it is a diagnostic, not an automatic
regularizer.

::: phydrax.uq.exact_state_space_log_likelihood

---

::: phydrax.uq.StateSpaceMarginalLikelihood

---

::: phydrax.uq.ExactStateSpaceLikelihood

---

::: phydrax.uq.state_space_identifiability

## Guided, Rao--Blackwellized, and fixed-lag particle methods

Guided filtering separates the proposal from the canonical transition. Every proposal
returns the sampled state, proposal log density, look-ahead score, and validity needed
for the exact importance correction. The bootstrap and fully adapted
linear-Gaussian proposals are built in; custom proposals implement the same contract.

`rao_blackwellized_particle_filter` samples a nonlinear state while propagating the
conditionally linear state by Kalman recursion. It therefore requires an explicit
conditional-linear model rather than attempting to discover one from arbitrary
callables. Fixed-lag smoothers expose the deliberate memory/latency approximation for
both Kalman and particle histories.

::: phydrax.uq.AbstractParticleProposal

---

::: phydrax.uq.LinearGaussianGuidedParticleProposal

---

::: phydrax.uq.guided_particle_filter

---

::: phydrax.uq.RaoBlackwellizedStateSpaceProblem

---

::: phydrax.uq.rao_blackwellized_particle_filter

---

::: phydrax.uq.fixed_lag_kalman_smoother

---

::: phydrax.uq.fixed_lag_particle_smoother

## Checkpoint and portable result contracts

Filter checkpoints are atomic, pickle-free archives with checksummed arrays and exact
compatibility metadata. A checkpoint resumes only when the live model, observation
schedule, physical case layout, algorithm, and numerical settings match. The unified
entry points dispatch among Kalman, particle, and ensemble state formats; explicit
algorithm-specific readers remain available.

Complete filter, smoother, and BSDE evaluations can be written with `export_result` and
read without importing the original model through `read_result_archive`.

::: phydrax.uq.write_filter_checkpoint

---

::: phydrax.uq.read_filter_checkpoint

---

::: phydrax.uq.write_kalman_filter_checkpoint

---

::: phydrax.uq.read_kalman_filter_checkpoint

---

::: phydrax.uq.write_particle_filter_checkpoint

---

::: phydrax.uq.read_particle_filter_checkpoint

---

::: phydrax.uq.write_ensemble_filter_checkpoint

---

::: phydrax.uq.read_ensemble_filter_checkpoint
