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
