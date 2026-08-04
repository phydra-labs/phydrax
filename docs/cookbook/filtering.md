# Filtering and smoothing stochastic state

This recipe uses one state-space contract with exact Kalman, bootstrap-particle, and
ensemble-transform filters. The scalar example is intentionally small; the same case,
time, state, mask, status, and provenance semantics apply to vector fields and
solver-backed transition kernels.

## 1. Declare observations and a model

```python
from pathlib import Path
import tempfile
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

observations = phx.stochastic.ObservationSequence(
    jnp.asarray([0.25, 0.5, 0.75, 1.0]),
    jnp.asarray([[0.2], [0.4], [0.7], [0.9]]),
    observation_mask=jnp.asarray(
        [[True], [True], [False], [True]]
    ),
    case_ids=("experiment-0",),
    sequence_id="position-sensors-v1",
)
prior = phx.stochastic.GaussianStatePrior(
    jnp.asarray([0.0]),
    jnp.asarray([[1.0]]),
    state_shape=(1,),
    prior_id="initial-position",
)
transition = phx.stochastic.LinearGaussianTransitionKernel(
    jnp.asarray([[1.0]]),
    jnp.asarray([[0.05]]),
    state_shape=(1,),
    process_id="random-walk",
)
observation = phx.stochastic.LinearGaussianObservationModel(
    jnp.asarray([[1.0]]),
    jnp.asarray([[0.1]]),
    state_shape=(1,),
    observation_shape=(1,),
)
model = phx.stochastic.StateSpaceModel(
    prior,
    transition,
    observation,
    model_id="position-model",
)
problem = phx.stochastic.StateSpaceProblem(
    model,
    observations,
    initial_time=0.0,
    problem_id="position-filtering",
)
```

The missing third sensor remains part of the fixed schedule but contributes no
likelihood term. Use `step_valid` to represent a physically absent or padded time step;
use `observation_mask` for missing channels at an otherwise valid time.

## 2. Exact filtering and smoothing

```python
kalman = phx.uq.kalman_filter(problem)
kalman_diagnostics = phx.uq.kalman_innovation_diagnostics(kalman)
smoothed = phx.uq.rts_smoother(kalman)
conditional_paths = phx.uq.sample_kalman_smoother_paths(
    jr.key(1),
    smoothed,
    sample_shape=(16,),
)
```

Use the exact path only when both the transition and observation models are linear
Gaussian. The sampled smoother paths preserve cross-time dependence; drawing each
smoothed marginal independently would not.

Batch and streaming execution are identical:

```python
state = phx.uq.initialize_kalman_filter(problem)
records = []
for _ in range(observations.num_steps):
    state, record = phx.uq.kalman_filter_step(problem, state)
    records.append(record)

assert jnp.allclose(state.mean, kalman.final_state.mean)
```

## 3. Nonlinear or non-Gaussian particle filtering

The bootstrap filter uses the same problem. A solver-backed
`DifferentialTransitionKernel`, `JumpTransitionKernel`, or
`JumpDifferentialTransitionKernel` can replace the analytic transition without changing
the filtering call.

```python
particles = phx.uq.bootstrap_particle_filter(
    jr.key(2),
    problem,
    num_particles=128,
    resampling_method="systematic",
    resampling_policy="ess",
    resampling_threshold=0.5,
)
particle_diagnostics = phx.uq.particle_filter_diagnostics(particles)
ancestral_paths = phx.uq.sample_particle_ancestry_paths(
    jr.key(3),
    particles,
    sample_shape=(16,),
)
backward_paths = phx.uq.sample_particle_backward_paths(
    jr.key(4),
    particles,
    sample_shape=(16,),
)
particle_prediction = phx.uq.particle_filter_predictive(
    jr.key(5), particles
)
```

Ancestor tracing follows the realized genealogy. Backward simulation instead uses the
transition density and requires a transition kernel with `log_prob`. These are distinct
smoothing algorithms, not interchangeable output formats.

## 4. High-dimensional ensemble filtering

The ensemble transform method performs member- and observation-space solves and avoids
a dense state covariance:

```python
ensemble = phx.uq.ensemble_transform_kalman_filter(
    jr.key(6),
    problem,
    ensemble_size=32,
    inflation=1.01,
    covariance_regularization=1e-8,
)
ensemble_diagnostics = phx.uq.ensemble_filter_diagnostics(ensemble)
ensemble_smoothed = phx.uq.ensemble_kalman_smoother(ensemble)
ensemble_prediction = phx.uq.ensemble_filter_predictive(ensemble_smoothed)
```

For a field state, keep `state_shape` as the physical tensor shape. The ensemble axis is
inserted explicitly before those event axes and is labeled as process uncertainty in the
predictive result.

## 5. Checkpoint streaming state and export results

```python
checkpoint_directory = tempfile.TemporaryDirectory()
checkpoint_path = Path(checkpoint_directory.name) / "filter-state.phxckpt"
result_path = Path(checkpoint_directory.name) / "particle-result.phxresult"
state = phx.uq.initialize_particle_filter(
    jr.key(7),
    problem,
    num_particles=128,
)
state, _ = phx.uq.particle_filter_step(problem, state)
phx.uq.write_filter_checkpoint(checkpoint_path, problem, state)

restored = phx.uq.read_filter_checkpoint(
    checkpoint_path,
    problem,
    "particle",
    num_particles=128,
)
restored, _ = phx.uq.particle_filter_step(problem, restored)

phx.uq.export_result(particles, result_path)
portable = phx.uq.read_result_archive(result_path)
checkpoint_directory.cleanup()
```

A checkpoint is a resumable algorithm state and therefore requires an exact live
problem and settings match. A portable result archive is read-only output and can be
inspected without reconstructing the original model.

## 6. Failure rules

- Inspect `valid`, `status`, and algorithm diagnostics before reducing a result.
- Do not treat padded schedule entries as successful observations.
- A particle transition failure invalidates that particle and is not repaired with an
  unconditional prior draw.
- A failed event stream or insufficient jump capacity remains a failed transition.
- Compare particle or ensemble settings on common semantic root keys when measuring
  algorithmic differences.
