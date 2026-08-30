# Filtering, smoothing, and state estimation

All native filters consume one `phydrax.stochastic.StateSpaceProblem` and preserve its
physical case axes, observation schedule, masks, case IDs, model ID, problem ID, and
sequence ID. Streaming APIs retain their one-step semantics. Batch Kalman execution
additionally supports an associative temporal scan without changing masks, padding,
failure freeze, result fields, or semantic random keys. Algorithm-specific status
arrays and validity masks distinguish inactive schedule entries from numerical or
transition failures.

## Typed exogenous inputs and callback context

Attach an `AbstractStateSpaceInput` to `StateSpaceProblem(input_signal=...)`. The
problem constructor checks that the signal has exactly the observation `case_shape`
and supports the start and end of every active transition (the end is also the
observation time). Unsupported active schedules are rejected before a filter runs.
`SampledStateSpaceInput` uses leading case axes, a trailing knot axis, and then input
event axes. Its valid knots must be a prefix in every case. Support is closed;
`"zero-order-hold"` is right-continuous and `"linear"` includes both endpoints.

Every model callback takes the canonical `StateSpaceStepContext` as its final
positional argument:

- linear-Gaussian transition parameters: `(t0, t1, context)`;
- linear-Gaussian observation parameters: `(time, context)`;
- callable transition `sample_fn`: `(key, state, t0, t1, context)`, and
  `log_prob_fn`: `(next_state, state, t0, t1, context)`;
- Gaussian observation location: `(state, time, context)`, and covariance:
  `(time, context)`;
- callable observation `location_fn`: `(state, time, context)`, `log_prob_fn`:
  `(value, state, time, mask, context)`, and `sample_fn`:
  `(key, state, time, sample_shape, context)`.

The context exposes `args`, flattened `case_index`, `step_index`,
`transition_start_input`, `transition_end_input`, `observation_input`,
`input_breakpoints`, `input_breakpoint_valid`, `input_valid`, and `input_signal`.
Use `context.evaluate_input(time)` when a solver needs the typed input at an internal
time; it returns an `InputEvaluation` with `value` and `valid`.

This complete two-case path makes both linear-Gaussian parameterizations depend on a
sampled input:

```python
import jax.numpy as jnp
import phydrax as phx

observations = phx.stochastic.ObservationSequence(
    jnp.asarray([[0.5, 1.0], [0.5, 1.0]]),
    jnp.zeros((2, 2, 1)),
    case_axes=("case",),
    case_shape=(2,),
    case_ids=("low", "high"),
    sequence_id="driven-observations",
)
forcing = phx.stochastic.SampledStateSpaceInput(
    jnp.asarray([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]),
    jnp.asarray([[[1.0], [2.0], [3.0]], [[2.0], [3.0], [4.0]]]),
    interpolation="linear",
    input_id="forcing-v1",
)
prior = phx.stochastic.GaussianStatePrior(
    jnp.zeros((2, 1)),
    jnp.asarray([[1.0]]),
    state_shape=(1,),
    prior_id="driven-prior",
)


def transition_matrix(t0, t1, context):
    del t0, t1
    return (1.0 + 0.05 * context.transition_end_input[0]).reshape((1, 1))


def transition_offset(t0, t1, context):
    return context.args["gain"] * (t1 - t0) * context.transition_start_input


def observation_matrix(time, context):
    del time
    return (1.0 + 0.1 * context.observation_input[0]).reshape((1, 1))


transition = phx.stochastic.LinearGaussianTransitionKernel(
    transition_matrix,
    jnp.asarray([[0.05]]),
    state_shape=(1,),
    offset=transition_offset,
    process_id="input-driven-transition",
)
observation = phx.stochastic.LinearGaussianObservationModel(
    observation_matrix,
    jnp.asarray([[0.1]]),
    state_shape=(1,),
    observation_shape=(1,),
)
model = phx.stochastic.StateSpaceModel(
    prior,
    transition,
    observation,
    model_id="input-driven-model",
)
problem = phx.stochastic.StateSpaceProblem(
    model,
    observations,
    initial_time=0.0,
    problem_id="input-driven-filter",
    args={"gain": jnp.asarray(0.2)},
    input_signal=forcing,
)
filtered = phx.uq.kalman_filter(problem, method="sequential")

assert filtered.filtered_means.shape == (2, 2, 1)
assert filtered.input_id == "forcing-v1"
```

`knot_valid` only identifies the prefix of real input knots in padded input storage.
It is independent of observation `step_valid` (whether a scheduled step exists) and
`observation_mask` (which channels contribute a likelihood). Input support is a third
concern: `InputEvaluation.valid` and `context.input_valid` say whether an input
evaluation is supported and finite. Filter outputs restore the declared physical case
axes even though callbacks receive a flattened `case_index`.

The sequential Kalman batch path runs the one-step recursion in one fused
`jax.lax.scan`; `kalman_filter_step` remains the equivalent streaming primitive. The
parallel method is only a temporal implementation of exact linear-Gaussian Kalman
filtering and smoothing, not a particle- or ensemble-filter option. It first resolves
the same case/step contexts and model parameters, then associatively composes affine
Gaussian elements. Padding, masks, failure freeze, result fields, and callback meaning
do not change. `auto` retains its documented long-chain, modest-factor selection
policy. The resolved method is recorded separately from model dispatch.

The input ID is provenance, not a numerical array axis. Problems retain the typed
signal; filter results retain `input_id`; smoother results retain that filter result;
portable exports preserve the ID; and checkpoint compatibility includes it, so a
checkpoint cannot silently resume against a differently identified input.

## Gaussian factors and conditional algebra

`GaussianFactor` stores a possibly rectangular covariance root `F` with event rows
and factor-direction columns, representing `F Fᴴ`. The adjoint is conjugate
transpose, so covariance, cross-covariance `F Gᴴ`, solves, and quadratic forms remain
correct for complex arrays. A zero-column factor is exactly deterministic, and a
rank-deficient factor remains a singular positive-semidefinite Gaussian rather than
being made full rank.

`gaussian_factor_from_covariance` uses a Hermitian eigendecomposition. It never clips,
symmetrizes, jitters, or otherwise repairs an invalid covariance. `regularization` is
the sole diagonal modification and is retained in the factor; `rank_tolerance` only
classifies small negative eigenvalues as numerical null directions. Larger negative
eigenvalues, non-Hermitian inputs, nonfinite values, and invalid tolerances are visible
through `valid` and `status`. Rank-aware log determinants, triangular solves, and
quadratic forms preserve singular behavior: the log determinant is `-inf` when the
covariance lacks full row rank, and a residual outside singular Gaussian support has
infinite quadratic form.

`ConditionalGaussianMoments.cross_covariance` is always oriented
`Cov[input, output]`. Affine prediction pushes factor directions through the matrix
and concatenates only declared independent noise directions. Conditioning uses
rank-aware moment algebra; an observation outside the predicted singular support is
reported as `CONDITIONAL_GAUSSIAN_INCONSISTENT_CONDITION`, not projected or repaired.
`factor_id`, `moments_id`, `regression_id`, and every `resolved_method` identify the
represented input and algebraic path.

::: phydrax.uq.GaussianFactor

---


::: phydrax.uq.gaussian_covariance

---

::: phydrax.uq.gaussian_factor_from_covariance

---

::: phydrax.uq.compress_gaussian_factor

---

::: phydrax.uq.add_independent_gaussian_factors

---

::: phydrax.uq.gaussian_cross_covariance

---

::: phydrax.uq.solve_triangular_rank_aware

---

::: phydrax.uq.gaussian_factor_log_determinant

---

::: phydrax.uq.gaussian_factor_quadratic_form

---

::: phydrax.uq.GAUSSIAN_FACTOR_SUCCESS

---

::: phydrax.uq.GAUSSIAN_FACTOR_NONFINITE

---

::: phydrax.uq.GAUSSIAN_FACTOR_INVALID_REGULARIZATION

---

::: phydrax.uq.GAUSSIAN_FACTOR_NON_HERMITIAN

---

::: phydrax.uq.GAUSSIAN_FACTOR_NOT_POSITIVE_SEMIDEFINITE

---

::: phydrax.uq.ConditionalGaussianMoments

---


::: phydrax.uq.GaussianRegression

---

::: phydrax.uq.predict_affine_gaussian

---

::: phydrax.uq.condition_gaussian

---

::: phydrax.uq.compose_gaussian_regressions

---

::: phydrax.uq.CONDITIONAL_GAUSSIAN_SUCCESS

---

::: phydrax.uq.CONDITIONAL_GAUSSIAN_NONFINITE

---

::: phydrax.uq.CONDITIONAL_GAUSSIAN_INVALID_FACTOR

---

::: phydrax.uq.CONDITIONAL_GAUSSIAN_INCONSISTENT_CONDITION

## Exact linear-Gaussian filtering

`kalman_filter`, `rts_smoother`, and `sample_kalman_smoother_paths` accept
`method="sequential"`, `"parallel"`, or `"auto"`. The parallel path composes
covariance-form affine Gaussian elements with solves and
`jax.lax.associative_scan`; `auto` selects it only for long chains with modest
state and observation factors. Result objects and portable archives record the
resolved execution method. `kalman_filter_step` remains the streaming primitive.
Filtering uses Joseph-form covariance updates and exposes innovation covariances,
normalized innovation squared values, incremental likelihoods, and failure status.
Backward path samples are coherent conditionals, not independent time marginals,
and retain case/step/member key prefix stability.

`covariance_form="covariance"` is the default dispatch.
`covariance_form="square_root"` selects QR-based forecast, measurement, and backward
factor updates through the same batch entry points. The square-root path is sequential:
`method="auto"` resolves to `"sequential"`, while `method="parallel"` is rejected for
both filtering and smoothing. The streaming `initialize_kalman_filter` and
`kalman_filter_step` remain covariance-form APIs. Returned objects still expose the
common covariance fields and record `covariance_form` and the resolved
`execution_method`; no parallel fallback, covariance repair, or undeclared jitter is
used. `covariance_regularization` is the only requested diagonal observation
regularization.

Using the `problem` constructed above:

```python
square_root_filtered = phx.uq.kalman_filter(
    problem,
    method="sequential",
    covariance_form="square_root",
)
square_root_smoothed = phx.uq.rts_smoother(
    square_root_filtered,
    method="sequential",
    covariance_form="square_root",
)

assert square_root_filtered.covariance_form == "square_root"
assert square_root_smoothed.execution_method == "sequential"
```


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

## Exact temporal Matérn Gaussian processes

`compile_state_space_kernel` prepares a scalar one-dimensional `Matern32Kernel` or
`Matern52Kernel`, directly or inside one `ScaleKernel`, on fixed training and query
times. Preparation resolves the exact continuous drift, stationary covariance and
factor, process-noise covariance and factor, observation map, state dimension, and
content ID. It verifies the stationary Lyapunov residual before constructing a
canonical `StateSpaceProblem`.

Training and query times may arrive in any order. Preparation makes one strictly
increasing schedule and retains stable sort indices, inverse permutations, and
original-order schedule indices. Training times must be unique. Repeated query
times and train-query overlaps share one latent schedule state and are repeated in
the returned original query order. Query-only and explicitly missing training
positions are represented by `ObservationSequence.observation_mask`; no large-noise
sentinel is used. `max_schedule_size` is a preparation-time resource guard.

```python
kernel = phx.kernels.ScaleKernel(
    phx.kernels.Matern52Kernel(length_scale=0.4),
    1.5,
)
plan = phx.uq.compile_state_space_kernel(
    kernel,
    jnp.asarray([0.8, 0.0, 0.35]),
    jnp.asarray([1.2, -0.2, 0.35]),
    train_mask=jnp.asarray([True, True, False]),
)
gp = phx.uq.fit_state_space_gaussian_process(
    plan,
    jnp.asarray([0.1, -0.3, 0.0]),
    noise_scale=0.02,
)
```

`noise_scale` is an observation standard deviation. Kernel coefficients and schedule
times must use one identical compute dtype; mixed plans are rejected rather than
silently rounded. Execution is always the canonical sequential square-root Kalman
filter and matching square-root RTS smoother with zero covariance regularization.
The stationary prior begins one dynamically evaluated length scale before the
earliest schedule state, avoiding a zero-process-root derivative. Interval process
covariance uses bounded Van Loan evaluation only for normalized short gaps and the
stationary identity `P∞ - Φ P∞ Φᵀ` for long gaps, so large extrapolation intervals
do not evaluate the exponentially growing auxiliary block. The RTS recursion is one
reverse `jax.lax.scan`, not a schedule-sized unrolled graph.

`posterior_mean` and `posterior_variance` are latent query marginals;
`predictive_variance` adds the declared observation variance. The result also
retains the exact active-observation log marginal likelihood, active count, masks,
filter/smoother histories, validity/status, kernel/schedule/method IDs, and precision
evidence. A concrete transformed fit records both prepared and evaluated kernel
content IDs. Under JAX tracing, where a host content hash is unavailable, the
evaluated ID is `None` and the exact evaluated length scale and covariance scale
remain exported arrays. Smoother invalidity has its own non-success GP status. The
complete result exports through `export_result`.

The returned query contract is marginal and therefore remains linear in schedule
storage; it does not materialize a dense query-by-query posterior covariance.
Unsupported regimes are rejected during preparation: kernel sums/products,
amplitude wrappers, vector length scales, multidimensional inputs, derivative
observations, SHO/CARMA models, non-Gaussian likelihoods, and parallel square-root
execution. Covariances are not projected, clipped, repaired, or given implicit
jitter. A degenerate zero-signal, zero-noise active observation consequently reports
the canonical filtering failure instead of manufacturing a posterior.

`tools/state_space_gp_benchmarks.py` compares this path with independent dense GP
algebra over increasing schedules. Its report retains per-size accuracy, complete
unique retained-array storage for both returned results, compilation and steady
execution times, scaling summaries, an admission gate, environment details, and
source provenance. Repeated PyTree aliases count once.

::: phydrax.uq.StateSpaceGaussianProcessPlan

---

::: phydrax.uq.StateSpaceGaussianProcessResult

---

::: phydrax.uq.compile_state_space_kernel

---

::: phydrax.uq.fit_state_space_gaussian_process

---

::: phydrax.uq.state_space_gaussian_process_status_name

---

::: phydrax.uq.STATE_SPACE_GP_SMOOTHER_FAILURE

## Nonlinear Gaussian moment transforms

The nonlinear transforms accept one unbatched Gaussian mean PyTree and one
`GaussianFactor` whose event size equals the flattened mean size; use `jax.vmap` for
batched factors. They return a mean with the function's PyTree structure, an output
factor, `Cov[input, output]`, validity/status, exact method ID, point count, dimensions,
method parameters, and explicit regularization.

- `spherical_radial_cubature` uses factor rank, not ambient event size, and evaluates
  `2 rank` points (one point for rank zero).
- `scaled_unscented_transform` uses `2 rank + 1` points. When its covariance weights
  are not all nonnegative it must form a dense output covariance, guarded by
  `max_output_dimension`.
- `gauss_hermite_transform` is a tensor rule with explicit `max_dimension` and
  `max_points` guards.
- `first_order_gaussian_transform` pushes factor columns with JVPs and forms the
  cross-covariance with VJP actions without materializing a Jacobian.

Singular and zero-rank input factors are valid. Any requested `regularization` is
implemented and recorded explicitly; there is no hidden fallback, clipping, or
positive-semidefinite repair. Nonfinite evaluations and invalid input or output
factors remain distinguishable through the nonlinear status codes.

`gaussian_expectation` evaluates an arbitrary PyTree-valued function without
materializing an output covariance. It shares the same deterministic cubature,
unscented, and Gauss--Hermite rules and adds fixed-sample Monte Carlo with an explicit
key. This is the expectation-only primitive used by SING.

::: phydrax.uq.GaussianExpectationResult

---

::: phydrax.uq.gaussian_expectation

---

::: phydrax.uq.NonlinearGaussianTransformResult

---


::: phydrax.uq.spherical_radial_cubature

---

::: phydrax.uq.scaled_unscented_transform

---

::: phydrax.uq.gauss_hermite_transform

---

::: phydrax.uq.first_order_gaussian_transform

---

::: phydrax.uq.NONLINEAR_GAUSSIAN_SUCCESS

---

::: phydrax.uq.NONLINEAR_GAUSSIAN_INPUT_FACTOR_INVALID

---

::: phydrax.uq.NONLINEAR_GAUSSIAN_NONFINITE

---

::: phydrax.uq.NONLINEAR_GAUSSIAN_OUTPUT_FACTOR_INVALID

## SING latent-SDE variational smoothing

SING represents the complete latent path as a block-tridiagonal Gaussian information
law and applies natural-gradient updates to its sufficient statistics. The objective
is the Euler-discretized SDE ELBO. Linear drift with Gaussian observations reaches
the exact posterior in one unit natural step; nonlinear factors retain the declared
expectation approximation.

`initialize_sing` builds a compact, case-aligned latent grid. An observation at
`initial_time` maps to the initial node; padded rows do not create transitions.
`sing_step` constructs a natural target and applies per-case monotone backtracking.
`sing_smoother` executes a fixed-capacity JAX scan and records ELBO, accepted-step,
natural-residual, convergence, and status histories. `sample_sing_paths` draws
forward-conditional paths and returns them on the original observation schedule.

The automatic path requires a full-rank `GaussianStatePrior`, an
`EulerMaruyamaTransitionKernel`, explicitly additive `WienerTerm` objects, and
full-rank process covariance on every active interval. Observation likelihoods may be
arbitrary differentiable normalized `AbstractObservationModel` implementations.
Irregular time gaps, physical cases, typed inputs, masks, and prefix padding preserve
their state-space semantics. Multiplicative noise, solver-backed transitions, and
singular process covariance have no hidden approximation or regularization path.

`expectation_method` is one of `"cubature"`, `"unscented"`, `"gauss-hermite"`, or
keyed `"monte-carlo"`. `method` independently selects sequential, associative, or
automatic Gaussian-chain conversion. `sing_elbo` holds the supplied `SINGState`
fixed, so model-parameter gradients are suitable for an explicit outer optimization
loop; `sing_smoother` does not update drift or observation parameters.

::: phydrax.uq.SINGGrid

---

::: phydrax.uq.SINGState

---

::: phydrax.uq.SINGELBOResult

---

::: phydrax.uq.SINGStepResult

---

::: phydrax.uq.SINGResult

---

::: phydrax.uq.initialize_sing

---

::: phydrax.uq.sing_elbo

---

::: phydrax.uq.sing_step

---

::: phydrax.uq.sing_smoother

---

::: phydrax.uq.sample_sing_paths

---

::: phydrax.uq.sing_status_name

---

## Continuous-discrete Gaussian filtering

`continuous_discrete_gaussian_filter` propagates Gaussian moments between irregular
discrete observations. It accepts a declared affine Gaussian transition or an Itô
`DifferentialTransitionKernel` with a Gaussian prior and Gaussian observation model;
geometric solvers and other stochastic interpretations are rejected. Affine
transitions use their exact interval discretization and affine observations use exact
moments. Otherwise `"extended"`, `"cubature"`, and `"unscented"` select first-order
JVP/VJP, spherical-radial, and scaled-unscented moment propagation respectively.
For nonlinear transitions or observations these are Gaussian moment approximations,
not exact filters; the selected method never falls back to another transform.

This is a guarded dense method: both flattened state and observation dimensions must
not exceed `CONTINUOUS_DISCRETE_MAX_DENSE_DIMENSION` (64). There is no sparse or
matrix-free fallback. `covariance_regularization`, `rank_tolerance`, and unscented
parameters are explicit settings. The fixed-interval smoother consumes stored
transition cross-moments; an applied backward step requires a full-rank predicted
covariance and reports a transform failure instead of silently regularizing a
singular one.

Results restore physical case and event shapes and preserve case axes, observation
axes, masks, step validity, case IDs, model/problem/sequence/process/observation IDs,
and sensor, input, parameter, basis, discretization, and approximation provenance.
State-space callbacks receive the same context-last typed inputs described above.
`status` distinguishes solver, transform, and nonfinite failures; `solver_status`
retains the underlying solver code separately. `method_id`, transition and
observation transform methods, `solver_id`, solver/backend/adjoint/step-controller
methods, and approximation IDs state exactly which route produced every history.
Inactive schedule entries remain distinct from failed active entries.

::: phydrax.uq.ContinuousDiscreteGaussianFilterResult

---

::: phydrax.uq.ContinuousDiscreteGaussianSmootherResult

---


::: phydrax.uq.continuous_discrete_gaussian_filter

---

::: phydrax.uq.continuous_discrete_gaussian_smoother

---

::: phydrax.uq.continuous_discrete_gaussian_status_name

---

::: phydrax.uq.CONTINUOUS_DISCRETE_MAX_DENSE_DIMENSION

---

::: phydrax.uq.CONTINUOUS_DISCRETE_GAUSSIAN_SUCCESS

---

::: phydrax.uq.CONTINUOUS_DISCRETE_GAUSSIAN_SOLVER_FAILURE

---

::: phydrax.uq.CONTINUOUS_DISCRETE_GAUSSIAN_TRANSFORM_FAILURE

---

::: phydrax.uq.CONTINUOUS_DISCRETE_GAUSSIAN_NONFINITE

## Stationary stochastic spectra

The spectral helpers reuse the diagnosed control resolvent instead of introducing a
second state-space representation. `linear_gaussian_transfer_function` returns
input-to-state, input-to-output, process-to-state, and process-to-output transfers
with the original frequency-response diagnostics. `linear_gaussian_spectral_densities`
then propagates declared input, state-coordinate process, and output-coordinate
measurement spectral-density matrices.

Each supplied spectrum is a density matrix, not a factor to be squared. It must be
finite, Hermitian, positive semidefinite, and shape-compatible; nothing is clipped or
repaired. Stationary spectra require diagnosed stability and nonsingular resolvents.
For complex frequency responses every cross-spectrum uses the fixed orientation
`S_ab = E[a bᴴ]`, including the conjugate adjoint. The full result retains the
frequency diagnostics, per-frequency validity, supplied spectra, and
`method_id="control-resolvent/conjugate-spectral-propagation"`.

::: phydrax.uq.LinearGaussianTransferResult

---

::: phydrax.uq.LinearGaussianSpectra

---

::: phydrax.uq.linear_gaussian_transfer_function

---

::: phydrax.uq.linear_gaussian_spectral_densities

---

::: phydrax.uq.state_spectral_density

---

::: phydrax.uq.output_spectral_density

---

::: phydrax.uq.state_output_cross_spectral_density

---

::: phydrax.uq.state_input_cross_spectral_density

---

::: phydrax.uq.output_input_cross_spectral_density


## Bellman posterior-mode filtering

`bellman_filter` performs deterministic posterior-mode filtering for a normalized,
twice-differentiable prior, transition density, and observation density. It is not a
Bayesian posterior sampler: its covariance is a local Laplace covariance at the mode,
and `pseudo_log_likelihood` is the Bellman filter's explicitly penalized criterion
rather than a marginal log likelihood.

`method="auto"` selects the exact Kalman engine only for a fully linear-Gaussian model
and otherwise uses the optimization engine. `method="analytic"` rejects any other model.
The optimization path records the requested and resolved modes, predicted and filtered
information matrices and covariances, raw curvature eigenvalues, score outer products,
solver diagnostics, and separate prediction, update, and pseudo-likelihood validity.
`curvature="observed"` uses the objective Hessian; `"score-outer-product"` uses the
aggregate observation score outer product for the update information. Damping is
explicit and recorded. Indefinite or non-finite curvature fails visibly instead of
triggering an adaptive jitter fallback.

`bellman_smoother` is exact RTS smoothing on the analytic linear-Gaussian path. On an
optimization result it is available only when the transition is a
`LinearGaussianTransitionKernel`; nonlinear or state-dependent transitions are
rejected because their local-Gaussian smoothing semantics would be approximate.

::: phydrax.uq.initialize_bellman_filter

---

::: phydrax.uq.bellman_filter_step

---

::: phydrax.uq.bellman_filter

---

::: phydrax.uq.bellman_smoother


## Bootstrap particle filtering

Particle states retain normalized log weights, root-key lineage, algorithm settings,
and streaming position. Histories retain forecast particles, posterior weights,
post-resampling particles, ancestor indices, transition validity, effective sample
sizes, and likelihood increments. Ancestor tracing and backward simulation provide two
explicit smoothing semantics.

`ParticlePrecisionPolicy` separates particle-state storage, log-weight/statistical
accumulation, resampling and ESS decisions, and optional output placement.
Transition samples are stored in state precision; likelihood normalization,
evidence, and cumulative log likelihood use statistics precision; ESS thresholds
and resampling CDFs use decision precision. Results retain precision evidence.
Particle checkpoint kind `particle-filter-state-v2` includes the policy identity
and rejects state/statistic dtype mismatches on restore.

::: phydrax.uq.ParticlePrecisionPolicy

---

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
Its backend `method` and Kalman `temporal_method` are separate choices, so selecting
parallel temporal execution cannot change model dispatch. `StateSpaceMarginalLikelihood`
threads both choices into posterior inference without resampling latent paths.
`state_space_identifiability` reports effective observation rank and
prior-to-posterior contraction; it is a diagnostic, not an automatic regularizer.

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
callables. An observation callback may return `DiagonalCovariance`; this uses
factor-space conditioning and avoids constructing or factorizing a dense observation
covariance while preserving the dense result.

`rao_blackwellized_particle_smoother` performs full-interval FFBSi over the nonlinear
particle system, then reruns the exact conditional Kalman/RTS recursion along each
sampled nonlinear path. It retains the initial nonlinear particle cloud because the
first conditional transition may depend on both the sampled initial and first
observation-time nonlinear states. The output is a conditional mixture: sampled
nonlinear paths plus one smoothed Gaussian mean, covariance, and lag-one covariance per
path. It deliberately does not collapse those path-conditioned Gaussians into one
purported Gaussian. Fixed-lag smoothers expose a separate, explicit memory/latency
approximation for Kalman and particle histories.

::: phydrax.uq.AbstractParticleProposal

---

::: phydrax.uq.LinearGaussianGuidedParticleProposal

---

::: phydrax.uq.guided_particle_filter

---

::: phydrax.uq.RaoBlackwellizedStateSpaceModel

---

::: phydrax.uq.RaoBlackwellizedStateSpaceProblem

---

::: phydrax.uq.rao_blackwellized_particle_filter

---

::: phydrax.uq.rao_blackwellized_backward_simulation

---

::: phydrax.uq.rao_blackwellized_particle_smoother

---

::: phydrax.uq.sample_rao_blackwellized_backward_paths


---

::: phydrax.uq.fixed_lag_kalman_smoother

---

::: phydrax.uq.fixed_lag_particle_smoother

## Checkpoint and portable result contracts

Filter checkpoints are atomic, pickle-free archives with checksummed arrays and exact
compatibility metadata. A checkpoint resumes only when the live model, observation
schedule, physical case layout, algorithm, and numerical settings match. The unified
entry points dispatch among Bellman, Kalman, particle, and ensemble state formats;
explicit algorithm-specific readers remain available. Bellman compatibility includes
the execution and curvature methods, damping, dimension guard, and optimizer settings.

Complete Bellman, Rao--Blackwellized, Kalman, particle, ensemble, smoother, and BSDE
evaluations can be written with `export_result` and read without importing the original
model through `read_result_archive`.

::: phydrax.uq.write_filter_checkpoint

---

::: phydrax.uq.read_filter_checkpoint

---

::: phydrax.uq.write_bellman_filter_checkpoint

---

::: phydrax.uq.read_bellman_filter_checkpoint

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
