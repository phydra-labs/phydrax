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

## 2. Drive filter parameters with a typed input

The input can parameterize both sides of an exact linear-Gaussian model. Callback
context is always the final positional argument:

```python
forcing = phx.stochastic.SampledStateSpaceInput(
    jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
    jnp.asarray([[0.0], [0.5], [1.0], [1.5], [2.0]]),
    interpolation="linear",
    input_id="position-forcing-v1",
)

def transition_matrix(t0, t1, context):
    del t0, t1
    return (1.0 + 0.05 * context.transition_end_input[0]).reshape((1, 1))

def transition_offset(t0, t1, context):
    return context.args["gain"] * (t1 - t0) * context.transition_start_input

def observation_matrix(time, context):
    del time
    return (1.0 + 0.1 * context.observation_input[0]).reshape((1, 1))

def observation_offset(time, context):
    del time
    return context.args["sensor_bias"] * context.observation_input

driven_transition = phx.stochastic.LinearGaussianTransitionKernel(
    transition_matrix,
    jnp.asarray([[0.05]]),
    state_shape=(1,),
    offset=transition_offset,
    process_id="driven-random-walk",
)
driven_observation = phx.stochastic.LinearGaussianObservationModel(
    observation_matrix,
    jnp.asarray([[0.1]]),
    state_shape=(1,),
    observation_shape=(1,),
    offset=observation_offset,
)
driven_model = phx.stochastic.StateSpaceModel(
    prior,
    driven_transition,
    driven_observation,
    model_id="driven-position-model",
)
driven_problem = phx.stochastic.StateSpaceProblem(
    driven_model,
    observations,
    initial_time=0.0,
    problem_id="driven-position-filtering",
    args={"gain": jnp.asarray(0.2), "sensor_bias": jnp.asarray(0.01)},
    input_signal=forcing,
)
driven_kalman = phx.uq.kalman_filter(driven_problem, method="sequential")

assert driven_kalman.input_id == "position-forcing-v1"
assert driven_kalman.filtered_means.shape == (4, 1)
```

`StateSpaceProblem` prevalidates the signal at both endpoints of every active
transition; the endpoint is also the observation time. Construction fails if an
active endpoint lies outside input support. Sampled-input support is closed. Valid
knots must form a prefix when `knot_valid` pads storage; zero-order hold is
right-continuous, while the example uses linear interpolation.

Keep the three kinds of validity separate:

- `knot_valid` describes real input knots, not observations;
- observation `step_valid` describes real schedule entries, and `observation_mask`
  selects observed channels;
- `InputEvaluation.valid` and `context.input_valid` describe input support and finite
  evaluation.

For multiple physical cases, the leading dimensions of sampled-input times and values
must equal `observations.case_shape`; the knot axis follows them, then the input event
axes. Callbacks receive the flattened `context.case_index`, while results restore the
declared case axes. The context also supplies `args`, `step_index`, start/end and
observation input values, masked interval breakpoints, the typed `input_signal`, and
`context.evaluate_input(time)` for internal solver times.

The same context-last rule applies beyond the parameter callbacks above. A callable
transition uses `sample_fn(key, state, t0, t1, context)` and
`log_prob_fn(next_state, state, t0, t1, context)`. A callable observation uses
`location_fn(state, time, context)`,
`log_prob_fn(value, state, time, mask, context)`, and
`sample_fn(key, state, time, sample_shape, context)`.

The sequential Kalman batch call fuses the one-step recursion with `jax.lax.scan`;
streaming through `kalman_filter_step` has the same step semantics. Parallel temporal
execution is limited to the exact linear-Gaussian Kalman filter and smoother; it is not
a particle- or ensemble-filter mode. It resolves the same context-dependent parameters
before associatively composing the Gaussian steps, so masks, padding, failure freeze,
and results keep their existing meaning. `auto` chooses between these implementations
and the result records the resolved choice.

The signal's `input_id` stays on filter results, inside the filter result retained by
smoothers, and in portable exports. Checkpoint compatibility also includes it,
preventing a resume with a differently identified input. The ID records provenance;
values still live only in the typed input.

## 3. Exact filtering and smoothing

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

## 4. Complete a finite-state posterior

The exact forward result is also the input to finite-state smoothing, Viterbi
decoding, expected transition counts, and arbitrary expected transition statistics:

```python
finite_states = jnp.asarray([[0], [1]])

def chain_rates(time, state, args):
    del time, args
    return jnp.asarray([
        jnp.where(state[0] == 0, 0.8, 0.0),
        jnp.where(state[0] == 1, 0.5, 0.0),
    ])

def chain_jump(state, channel, mark, args):
    del state, mark, args
    return jnp.where(channel == 0, jnp.asarray([1]), jnp.asarray([0]))

chain = phx.stochastic.JumpProcess(
    chain_rates,
    chain_jump,
    state_shape=(1,),
    num_channels=2,
    process_id="two-state-chain",
)
finite_transition = phx.stochastic.FiniteStateTransitionKernel(
    phx.solver.finite_state_generator(chain, finite_states)
)
finite_prior = phx.stochastic.CategoricalStatePrior(
    finite_states,
    jnp.asarray([0.6, 0.4]),
    prior_id="two-state-prior",
)

def finite_log_prob(value, state, time, mask, context):
    del time, context
    residual = (value - state.astype(float)) / 0.25
    terms = -0.5 * residual**2 - jnp.log(0.25 * jnp.sqrt(2.0 * jnp.pi))
    return jnp.sum(jnp.where(mask, terms, 0.0))

finite_observation = phx.stochastic.CallableObservationModel(
    lambda state, time, context: state.astype(float),
    finite_log_prob,
    lambda key, state, time, sample_shape, context: (
        state.astype(float)
        + 0.25 * jr.normal(key, tuple(sample_shape) + (1,))
    ),
    state_shape=(1,),
    observation_shape=(1,),
    observation_id="two-state-sensor",
)
finite_problem = phx.stochastic.StateSpaceProblem(
    phx.stochastic.StateSpaceModel(
        finite_prior,
        finite_transition,
        finite_observation,
        model_id="two-state-model",
    ),
    observations,
    initial_time=0.0,
    problem_id="two-state-completion",
)
finite_filter = phx.uq.exact_state_space_log_likelihood(
    finite_problem, method="finite-state"
).backend
finite_smoother = phx.uq.finite_state_backward_smoother(finite_filter)
finite_path = phx.uq.finite_state_viterbi(finite_filter)
finite_counts = phx.uq.finite_state_expected_transition_counts(finite_smoother)

def changed(previous_state, state, t0, t1, context):
    del t0, t1
    return {
        "changed": (previous_state[0] != state[0]).astype(float),
        "case_index": context.case_index.astype(float),
    }

finite_statistics = phx.uq.finite_state_expected_sufficient_statistics(
    finite_smoother, changed
)
```

The adjacent-state posterior at step zero describes the physical interval from
`initial_time` to the first observation. A zero-probability transition remains exactly
zero in the smoother and contributes negative infinity to a Viterbi candidate; no
epsilon floor is added. Masked observation channels contribute no emission likelihood.
`step_valid` padding has zero adjacent-transition mass and therefore contributes zero
to counts and statistics. Each result retains validity/status, physical cases,
`input_id` when present, and its source filter provenance.

`changed(previous_state, state, t0, t1, context)` follows the context-last callback
contract. Its PyTree structure and leaf shapes must remain constant across state pairs,
steps, and cases.

## 5. Nonlinear or non-Gaussian particle filtering

The bootstrap filter uses the same problem. A solver-backed
`DifferentialTransitionKernel`, `JumpTransitionKernel`, or
`JumpDifferentialTransitionKernel` can replace the analytic transition without changing
the filtering call.

```python
particles = phx.uq.bootstrap_particle_filter(
    jr.key(2),
    problem,
    num_particles=32,
    resampling_method="systematic",
    resampling_policy="ess",
    resampling_threshold=0.5,
)
particle_diagnostics = phx.uq.particle_filter_diagnostics(particles)
genealogical = phx.uq.full_particle_smoother(particles)
fixed_lag = phx.uq.fixed_lag_particle_smoother(particles, 2)
ancestral_paths = phx.uq.sample_particle_ancestry_paths(
    jr.key(3),
    particles,
    sample_shape=(4,),
)
backward_simulated = phx.uq.particle_backward_simulation(
    jr.key(4),
    particles,
    sample_shape=(4,),
)
particle_prediction = phx.uq.particle_filter_predictive(
    jr.key(5), particles
)
```

`full_particle_smoother` uses the complete realized genealogy and needs no transition
density, but it can expose genealogical collapse. A fixed-lag smoother conditions on
only the declared number of future steps and is not a full-interval result.
`particle_backward_smoother` and `particle_backward_simulation` instead require a
normalized transition `log_prob`; a sampler-only transition is rejected rather than
silently approximated. Backward simulation returns coherent paths plus their sampled
particle indices. Resampling ancestors and sampled indices are nondifferentiable, so
genealogical and FFBSi results record stop-gradient ancestry. All variants preserve
physical case IDs and axes, masks/padding, validity/status, `input_id`, and method,
process, approximation, or resampling provenance as applicable.

## 6. High-dimensional ensemble filtering

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

## 7. Checkpoint streaming state and export results

```python
checkpoint_directory = tempfile.TemporaryDirectory()
checkpoint_path = Path(checkpoint_directory.name) / "filter-state.phxckpt"
result_path = Path(checkpoint_directory.name) / "particle-result.phxresult"
state = phx.uq.initialize_particle_filter(
    jr.key(7),
    problem,
    num_particles=32,
)
state, _ = phx.uq.particle_filter_step(problem, state)
phx.uq.write_filter_checkpoint(checkpoint_path, problem, state)

restored = phx.uq.read_filter_checkpoint(
    checkpoint_path,
    problem,
    "particle",
    num_particles=32,
)
restored, _ = phx.uq.particle_filter_step(problem, restored)

phx.uq.export_result(particles, result_path)
portable = phx.uq.read_result_archive(result_path)
checkpoint_directory.cleanup()
```

A checkpoint is a resumable algorithm state and therefore requires an exact live
problem and settings match. A portable result archive is read-only output and can be
inspected without reconstructing the original model.

## 8. Failure rules

- Inspect `valid`, `status`, and algorithm diagnostics before reducing a result.
- Do not treat padded schedule entries as successful observations.
- A particle transition failure invalidates that particle and is not repaired with an
  unconditional prior draw.
- A failed event stream or insufficient jump capacity remains a failed transition.
- Compare particle or ensemble settings on common semantic root keys when measuring
  algorithmic differences.
