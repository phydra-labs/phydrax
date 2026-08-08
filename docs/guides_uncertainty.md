# Uncertainty quantification

Phydrax provides a native, dependency-light UQ layer under `phydrax.uq`. It uses
JAX arrays, Equinox PyTrees, `coordax.Field` dimensions, and the existing domain and
solver contracts. The core result type is `PredictiveField`: every stochastic source
has an explicit named sample dimension and a source label.

## Uncertainty sources

Phydrax distinguishes five sources:

- **epistemic**: variation across posterior draws, independently fitted models, or
  latent model-discrepancy functions;
- **input**: variation caused by uncertain parameters, coefficients, forcing, or
  initial conditions;
- **observation**: explicitly sampled measurement noise;
- **process**: intrinsic stochastic forcing or a learned stochastic transition law;
- **numerical**: uncertainty attributed to a discretization, truncation, or solver
  approximation.

A `PredictiveField` may contain several source axes. `mean`, `variance`, `std`,
`quantile`, and `interval` accept a source selection. `decompose_variance()` keeps
the five meanings separate. Total variance reduces every declared sample source and
adds mean conditional observation variance when supplied; it never silently labels
or merges unidentified axes. The source is a provenance statement, not an estimator:
a Diffrax path ensemble supplies `process` draws but does not estimate `numerical`
error automatically.

```python
import coordax as cx
import jax.numpy as jnp
import phydrax as phx

samples = cx.Field(
    jnp.zeros((16, 32)),
    dims=("member", "x"),
)
prediction = phx.uq.PredictiveField(
    samples,
    (phx.uq.SampleAxis("member", "epistemic"),),
)
mean = prediction.mean()
variance = prediction.epistemic_variance()
```

## SDE and semidiscrete SPDE path ensembles

`DifferentialProblem` and `solve_diffrax_ensemble` generate finite-dimensional
Itô or Stratonovich path ensembles. For spatial stochastic dynamics, first
choose an `AbstractSpatialDiscretization`, then a finite-rank
`SpatialNoiseBasis`, and compose a `SemidiscreteSPDE`. The resulting leading
path axis is intrinsic `process` variation:

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

axis = phx.domain.FourierAxisSpec(32).materialize(0.0, 1.0)
space = phx.solver.TensorGridDiscretization((axis,))
noise = phx.solver.SpatialNoiseBasis.from_spectrum(
    space,
    lambda eigenvalue: 0.01 * jnp.exp(-0.1 * eigenvalue),
    rank=6,
)
spde = phx.solver.semidiscretize_reaction_diffusion(
    jnp.sin(2.0 * jnp.pi * axis.nodes),
    space,
    t0=0.0,
    t1=0.2,
    kappa=0.02,
    noise_basis=noise,
)
realization = spde.wiener_realization(
    jr.key(0),
    sample_shape=(128,),
    tolerance=1e-4,
    label="heat-0",
)
solution = phx.solver.solve_diffrax_ensemble(
    spde.problem,
    save_times=jnp.linspace(0.0, 0.2, 21),
    realization=realization,
    dt0=1e-3,
)
prediction = solution.to_predictive(
    sample_dim="path",
    time_dim="time",
    state_dims=("space",),
)
```

Reusing the same realization reproduces every Brownian path, including across
subinterval solves. Changing its root key changes the realization. `noise_id`
records the spatial noise modes, eigenvalues, quadrature, and discretization identity.
A refined grid or changed truncation therefore receives a different fingerprint and
cannot be paired accidentally.

For tensor grids, `space.eigenpairs(rank=...)` forms exact separable modes and
selects only the requested lowest tensor sums. Kernel-defined covariance uses
`SpatialNoiseBasis.from_kernel_covariance`, whose Matfree pivoted-Cholesky path
queries entries on demand. A covariance available only as a state-shaped matvec
uses `SpatialNoiseBasis.from_covariance_operator(..., key=..., oversampling=...)`
and randomized Nyström. Both routes expose their method, rank, tolerance,
residual estimate, convergence flag, and seed/sketch provenance through
`noise.approximation`; check that record before treating a truncation as
numerically adequate.

The path axis alone says nothing about numerical uncertainty. To quantify
spatial truncation, time stepping, or solver error, run an explicit discretization
ensemble and label that additional axis `numerical`. Do not merge those runs
into the process axis: process covariance and numerical sensitivity answer
different questions.

For the full method-of-lines and noise-basis contracts, see
[API → Solver → Differential equations](api/solver/differential.md).


### Semilinear integration, trajectories, and convergence

When the semidiscrete drift has a meaningful split
\(F_h(U)=A_hU+N_h(U)\), preserve it with
`semidiscretize_semilinear_spde`. `solve_semilinear_spde` then selects a
fixed-step exponential-Euler path when all of the following are declared:

- Itô interpretation and additive finite-rank noise;
- an explicit `SemilinearDrift` with a stable `operator_id`;
- a matrix-function policy for applying \(\exp(hA_h)\) and
  \(\varphi_1(hA_h)\) without assembling a global operator matrix;
- compatible noise/operator eigenvalues for exact modal stochastic convolution.

Exact tensor/spectral action is used when supplied. Otherwise the matrix-function
policy selects Chebyshev for bounded self-adjoint operators, Lanczos for
mass-self-adjoint operators, or Arnoldi for general operators. Unsupported
specializations lower to the validated Diffrax path by default; set
`fallback="error"` when silent lowering would invalidate an experiment.

`DifferentialSolution.to_stochastic_trajectory()` converts solver output into
`StochasticTrajectory`, whose array contract is
`case_shape + realization_shape + (time,) + state_shape`. It retains physical
case IDs, parameter IDs, realization/coupling IDs, discretization/basis IDs,
validity masks, and named axes. `adjacent_transitions()` and
`transitions(source_indices, target_indices)` are lazy views; their
`operator_dataset()` adapter groups train/validation/test splits by physical
case, realization, and coupling identity rather than leaking neighboring states
across splits.

Convergence claims use `SPDEConvergenceStudy`, not one mixed refinement sweep.
Refine exactly one of time step, spatial resolution, noise rank, or ensemble
size while holding the other contracts fixed. Strong/pathwise levels must share
one explicit `coupling_id`. Weak estimates report Monte Carlo standard error and
confidence intervals. `NoiseTruncationStudy.from_compatible_spectrum` reports
raw covariance tails separately from finite-horizon and stationary
solution-weighted tails; these are generally different orderings.

### Weak, mild, and density-equation physics

`SPDESolutionSpec` declares `strong`, `weak`, or `mild` semantics together with
the forcing regularization and cutoff identity. Pointwise strong stochastic
residuals reject rough space-time-white forcing; selecting `weak` or `mild`
changes the mathematical contract rather than suppressing the validation.

For finite-dimensional generators, a supplied diffusion factor makes
`kolmogorov_generator(..., contraction="auto")` use exact
factor-Hessian-vector products. It never constructs the covariance or Hessian.
`StochasticTracePolicy` and `estimate_stochastic_trace` provide an explicit
matrix-free Hutchinson alternative with probe count, distribution, and Monte
Carlo standard error. `directional_stratonovich_correction` computes the
Stratonovich-to-Itô correction by JVPs.

`probability_current` returns the advective-diffusive Fokker--Planck current.
`ContinuousProbabilityFluxBoundaryConstraint` constrains its outward normal
component; the default target is zero and therefore represents a reflecting
boundary. Positivity, normalization, initial data, and flux remain separate
constraints.

## Static random fields

`phydrax.stochastic` separates a Gaussian coefficient realization from spatial
synthesis and semantic use:

```python
basis = phx.solver.SpatialNoiseBasis.from_spectrum(
    space,
    lambda eigenvalue: 0.2 / (1.0 + eigenvalue),
    rank=8,
)
synthesis = phx.stochastic.SpatialBasisSynthesis.from_spatial_noise_basis(
    basis,
    mean=0.0,
)
log_conductivity = phx.stochastic.StaticGaussianRandomField(
    synthesis,
    role="coefficient",
    source="log-conductivity",
)
realization = log_conductivity.realize(
    jr.key(1),
    sample_shape=(256,),
    label="material-inputs",
)
sample = log_conductivity.sample(realization)
```

Roles are explicit: `input`, `initial_condition`, `coefficient`,
`boundary_data`, `forcing`, or `observation`. A nonlinear transform is also
explicit:

```python
conductivity = log_conductivity.transform(
    jnp.exp,
    transform_id="exp-lognormal-v1",
)
positive_sample = conductivity.sample(realization)
```

The transformed sample keeps the latent realization and coupling identities but
receives a distinct field identity. `gaussian_field_diagnostics` checks latent
coefficient covariance, spatial pointwise variance, and exact replay.

Cross-resolution common random numbers require
`GaussianFieldCoupling((coarse_field, fine_field))`. It aligns standardized
coefficients by stable `mode_ids` and samples the union of required modes.
Passing the same PRNG key to two unrelated fields is intentionally insufficient:
different bases receive different coupling identities, and a realization
missing required mode IDs is rejected.

## Finite latent stochastic processes

Pathwise stochastic flow and marginal transition density are distinct
interfaces:

- `AbstractPathwiseTransition` consumes an explicit driver segment and defines
  how adjacent segments compose;
- `AbstractMarginalTransitionLaw` returns an
  `AbstractProcessDistribution` after integrating out the driver.

`LatentGaussianCoefficientProcess` implements both for
\(dC_t=\mu\,dt+L\,dW_t\). Its `ProcessRealization` owns one global
`WienerRealization`, so repeated or refined queries evaluate the same path:

```python
process = phx.stochastic.LatentGaussianCoefficientProcess(
    jnp.asarray([0.1, -0.2]),
    jnp.asarray([[0.3, 0.0], [0.1, 0.25]]),
)
process_realization = process.realize(
    jr.key(2),
    jnp.zeros((2,)),
    support=(0.0, 1.0),
    sample_shape=(512,),
)
trajectory = process.evaluate(
    process_realization,
    jnp.asarray([0.0, 0.25, 0.5, 1.0]),
)
marginal = process.marginal_transition(
    jnp.zeros((2,)),
    t0=0.0,
    t1=1.0,
)
```

`process_query_consistency` compares shared times across two query grids.
`cocycle_objective` tests pathwise composition with the same driver segments.
`semigroup_objective` is a Monte Carlo Chapman--Kolmogorov loss for a marginal
law. `process_sample_statistics` and `gaussian_process_diagnostics` expose
finite-sample moments, log density, replay, query consistency, cocycle error,
and empirical Gaussian-moment error.

`phx.nn.conditional_coupling_flow_process` builds a
`LatentFlowJAXCoefficientProcess` for non-Gaussian latent transition marginals.
It intentionally implements only `AbstractMarginalTransitionLaw`: independent
FlowJAX transition draws do not identify a common driving path and must not be
used to claim cocycle consistency.

### Finite-activity jump processes

`AbstractJumpProcess` separates a process law from a stochastic realization.
`JumpProcess` accepts callable channel intensities, state updates, and optional
mark sampling. `MassActionJumpProcess` provides combinatorial propensities,
stoichiometric updates, and conservation residuals for reaction systems.

`PoissonClockRealization` stores prefix-stable unit-rate thresholds and mark
keys for each channel and path. Increasing `max_events_per_channel` with
`extend()` preserves every existing threshold and mark. A solver result uses
`JumpEventBatch`: valid events are identified by an explicit prefix mask, and
each path has a success, capacity, invalid-intensity, or solver-failure status.
Never infer validity from a padding time or channel.

```python
process = phx.stochastic.JumpProcess(
    lambda t, state, args: jnp.asarray([2.0]),
    lambda state, channel, mark, args: state + jnp.asarray([1.0]),
    state_shape=(1,),
    num_channels=1,
    process_id="counting-process",
)
clock = phx.stochastic.PoissonClockRealization(
    jr.key(4),
    1,
    support=(0.0, 1.0),
    max_events_per_channel=16,
    sample_shape=(512,),
    process_id=process.process_id,
)
jump_solution = phx.solver.solve_next_reaction(
    process,
    clock,
    jnp.asarray([0.0]),
    t0=0.0,
    t1=1.0,
    save_times=jnp.linspace(0.0, 1.0, 11),
)
```

`solve_next_reaction` advances one internal clock per channel.
`solve_direct_ssa` samples the total rate and channel directly. Both are exact
for the finite-activity pure jump contract, where the state and rates remain
constant between events.

`JumpDifferentialProblem` combines a jump process with a
`DifferentialProblem`. `solve_jump_differential` augments the continuous state
with cumulative hazards, localizes threshold crossings, applies the jump map,
and continues. The continuous part may be an ODE or an SDE. An SDE solve uses
one global `WienerRealization` across every event restart and returns a
`CompositeStochasticRealization` with named `"wiener"` and `"jump"`
components. This is the supported route for state-dependent integrated
hazards and globally coupled jump diffusions.

`finite_state_generator` constructs an explicit generator only for a supplied
finite state set. Use it as a small-system reference, not as the simulation
backend for an unbounded state space.

### Martingale problems and stopping-time diagnostics

`MartingaleProblem` evaluates a declared observable against its continuous and
finite-activity jump generator on a canonical `StochasticTrajectory`.
`martingale_increments` supports left, midpoint, and trapezoid quadrature;
`stopped_martingale_increments` truncates those same paths at explicit stopping
indices. Predictable brackets and quadratic covariation remain aligned to the
trajectory intervals.

Use `martingale_validation_report` to combine cluster-aware moment tests,
quadratic-variation checks, and optional jump-compensator diagnostics. The
realization-independence label is the statistical unit: antithetic or otherwise
coupled paths do not become independent merely because they occupy different
array rows. See [API → Stochastic → Martingales](api/stochastic/martingales.md).

### State-space filtering and smoothing

`StateSpaceProblem` combines an explicit state prior, transition kernel,
observation model, and `ObservationSequence`. The sequence retains physical
case axes, timestamps, channel masks, padded-step validity, stable case IDs,
and a sequence ID. Analytic marginal kernels and solver-backed differential,
jump, hybrid, finite-state, or neural-operator kernels implement the same
transition-sample contract.

`phydrax.uq` supplies exact Kalman/RTS inference for linear-Gaussian models,
bootstrap particle filtering with genealogy and backward simulation, and an
ensemble transform Kalman filter/smoother that works in member and observation
space. Every filter has streaming and batch execution, status-aware
diagnostics, predictive conversion, and pickle-free compatible checkpoints.
Complete histories can be exported as portable results.

See the [filtering cookbook](cookbook/filtering.md),
[state-space API](api/stochastic/state_space.md), and
[filtering API](api/uq/filtering.md).

### Backward stochastic equations

`BSDEProblem` binds a forward-path sampler, drift, diffusion, backward
generator, and terminal condition. `evaluate_bsde` returns terminal,
interval-local, and global trajectory residuals with an explicit quadrature
rule. A supplied control model represents $Z$ directly; autodiff instead
computes $Z=\nabla_xu\,\sigma$ and exposes the corresponding semilinear PDE
residual. `BSDEObjective` attaches the same decomposition to
`FunctionalSolver` with fixed or resampled paths.

`solve_coupled_fbsde_explicit` supports forward drift and diffusion depending
on current value/control predictions. `JumpBSDEProblem` adds finite-activity
jump controls and exact user-declared compensator rates. Composite Wiener and
Poisson provenance is validated, and event-capacity or solver failures remain
invalid paths rather than being dropped.

See the [BSDE cookbook](cookbook/bsde.md) and
[BSDE API](api/stochastic/bsde.md).

### Process-consistent neural-operator transitions

`OperatorTransitionSpec` binds a canonical `OperatorBatch` to the evolving
state, duration, optional source time, optional driver, output query, and output
field. Every other input remains fixed forcing, geometry, or parameter
conditioning when the state advances. The state samples and output query must
use identical geometry.

A probabilistic complete-field operator becomes a marginal stochastic law
through `OperatorMarginalTransition`:

```py
spec = phx.nn.OperatorTransitionSpec(
    phx.nn.OperatorOutputSpec("scalar"),
    state_input="state",
    duration_input="duration",
)
law = phx.nn.OperatorMarginalTransition(
    probabilistic_operator,
    transition_batch,
    spec,
    process_id="stochastic-heat",
)
rollout = phx.nn.marginal_operator_rollout(
    law,
    jnp.asarray([0.0, 0.1, 0.25, 0.5]),
    key=jr.key(3),
    num_realizations=512,
)
predictive = rollout.to_predictive()
```

The model distribution must declare `uncertainty_source="process"`.
`marginal_operator_rollout` samples a Markov chain and records a replayable
chain ID, but deliberately stores no `WienerRealization`: its independent
transition samples do not establish a common driving path. The returned
`StochasticTrajectory` has explicit physical-case, realization, time, query,
and channel axes. `to_predictive()` maps each realization axis to a
`PredictiveField` process-sample axis without merging it with epistemic, input,
observation, or numerical uncertainty.

An operator conditioned on explicit additive Wiener increments instead uses
`OperatorPathwiseTransition` and one typed driver binding:

```py
path_spec = phx.nn.OperatorTransitionSpec(
    phx.nn.OperatorOutputSpec("scalar"),
    driver_bindings=(
        phx.nn.OperatorDriverBinding(
            "driver",
            "wiener",
            kind="wiener",
            quantity="increment",
        ),
    ),
)
flow = phx.nn.OperatorPathwiseTransition(
    driver_conditioned_operator,
    transition_batch_with_driver,
    path_spec,
    process_id="driven-stochastic-heat",
)
pathwise = phx.nn.pathwise_operator_rollout(
    flow,
    wiener_realization,
    jnp.asarray([0.0, 0.1, 0.25, 0.5]),
)
```

This rollout queries every segment from one global `WienerRealization`,
preserves its realization and coupling IDs, and supports a genuine cocycle
check. A driver whose `noise_shape` equals the operator driver-event shape is
shared deliberately across physical cases; prefixing that shape with the
operator `case_shape` supplies case-specific driver fields. The rollout records
which mode was used. Its default segment-composition rule is addition; a
different driver algebra requires a different `AbstractPathwiseTransition`.

Multiple stochastic drivers use `OperatorProcessTransition`. Each
`OperatorDriverBinding` declares a model input, named realization component,
driver kind, and quantity. Wiener components provide increments; jump
components provide event times, offsets, channels, marks, masks, or
per-channel counts. `OperatorJumpTransition` is the jump-only specialization.

```py
mixed_spec = phx.nn.OperatorTransitionSpec(
    phx.nn.OperatorOutputSpec("scalar"),
    driver_bindings=(
        phx.nn.OperatorDriverBinding(
            "noise",
            "wiener",
            kind="wiener",
            quantity="increment",
        ),
        phx.nn.OperatorDriverBinding(
            "counts",
            "jump",
            kind="jump",
            quantity="channel_counts",
        ),
    ),
)
mixed_law = phx.nn.OperatorProcessTransition(
    mixed_driver_operator,
    transition_batch_with_drivers,
    mixed_spec,
    process_id="mixed-stochastic-operator",
)
driver = phx.stochastic.CompositeStochasticRealization(
    {"wiener": wiener_realization, "jump": poisson_realization}
)
mixed_rollout = phx.nn.process_operator_rollout(
    mixed_law,
    driver,
    times,
    jump_events={"jump": jump_solution.events},
)
```

`jump_generator_observable` evaluates the declared nonlocal jump generator on
an observable, including marked transitions.
`operator_jump_generator_objective` compares that target with the
small-time observable increment of a learned marginal law.

Use `operator_markov_chain_nll` for teacher-forced adjacent transitions and
`direct_operator_horizon_nll` when every horizon is supervised directly from
the same initial field. `operator_weak_generator_objective` matches observable
increments to a declared infinitesimal generator. `semigroup_objective` tests
the marginal Chapman--Kolmogorov law, while `cocycle_objective` is reserved for
pathwise transitions with explicit driver segments. These objectives are
complementary; a low one-step likelihood does not imply temporal consistency.

Random initial conditions remain an `input` uncertainty axis outside a process
realization. Driver variation is always `process`. Model ensembles remain
`epistemic`, and discretization studies remain `numerical`; none of these axes
is inferred or merged automatically.

### Process diagnostics, calibration, and retention

Stochastic validation must preserve complete realization paths:

- `horizon_score_diagnostics` reports horizon-indexed marginal CRPS and energy
  scores without collapsing time;
- `trajectory_score_diagnostics` treats each complete trajectory as one
  multivariate event and adds a dependence-sensitive variogram score;
- `observable_rank_diagnostics` and `pit_diagnostics` evaluate declared
  observables rather than flattening unrelated coordinates;
- `temporal_moment_diagnostics` reports means, covariance,
  cross-covariance, correlation, and lag autocorrelation;
- `semigroup_mc_diagnostics` reports candidate, reference, and excess
  Monte Carlo uncertainty instead of treating a noisy estimate as exact;
- `jump_event_diagnostics` reports successful-path counts, interarrival
  moments, channel frequencies, and channel-conditional mark moments;
- `first_passage_diagnostics` retains right censoring and checks an analytic
  CDF with a simultaneous finite-sample bound.

Numerical error is not process uncertainty.
`paired_refinement_uncertainty` requires coupled coarse/fine paths or physical
cases and applies an explicit Richardson correction.
`predictive_variance_decomposition` accepts that result as a separate
outer `"numerical"` component; a predictive sample axis merely labeled
`"numerical"` is rejected as insufficient evidence.

`ProcessValidationSplit` requires disjoint physical case identities for
training, calibration, and test sets. `HorizonScaleCalibrator` fits
horizon-specific scales. `ProcessConformalCalibrator` supports pointwise,
simultaneous trajectory, and weighted trajectory scores.
`process_calibration_report` always retains raw and calibrated scores.
`process_shift_evaluation_matrix` requires in-distribution, rollout-horizon,
covariance, initial-condition, and parameter-regime scenarios with paired
seeds. Finally, `process_retention_report` combines replay, provenance,
temporal, semigroup, calibration, shift, and uncertainty-decomposition gates
into one explicit pass/fail artifact.

## Stochastic evaluation keys and dropout

Deterministic models accept `key=None`. Active `Dropout` layers require an explicit
key. One root key identifies one complete function draw:

- hidden layers receive distinct folded-in subkeys;
- feature dropout broadcasts one channel mask over leading spatial, time, grid,
  node, and batch axes;
- separable factors and DeepONet branch/trunk networks receive distinct keys;
- an FNO mask is channel-wise and shared over its spatial grid.

This is different from resampling a mask independently at every collocation point.
The latter produces marginal values, not a coherent random function, and is not the
Phydrax default.

```python
import jax.random as jr

model = phx.nn.MLP(
    in_size=2,
    out_size="scalar",
    width_size=64,
    depth=4,
    dropout=0.1,
    key=jr.key(0),
)
```

Use `phx.nn.inference_mode(model)` to return an immutable copy with dropout disabled.


## Deep ensembles and randomized priors

`HomogeneousFunctionEnsemble` stores one member-axis-stacked PyTree and evaluates it
with `equinox.filter_vmap`. Every array leaf must carry the member axis and static
configuration must be shared. `HeterogeneousFunctionEnsemble` is the tuple fallback
for different architectures, graph topologies, constraints, or solver settings.

Train members independently with `fit_ensemble`; do not vectorize high-level solver
logging or adaptive-collocation state. `RandomizedPriorModel` adds an independently
initialized prior network to a learned network. `FrozenModel` keeps the prior outside
Phydrax trainable partitions.

Set `return_diagnostics=True` on `fit_ensemble` to receive an
`EnsembleFitResult`. It records each member index, deterministic initialization and
solver seeds, elapsed fit time, and any `training_diagnostics` exposed by the fitted
solver. A failed fit raises `EnsembleFitError` with the failed member and all
completed diagnostics.

Ensemble spread is epistemic variation. It is not a confidence interval until a
calibration method supplies that interpretation.

The opt-in learned inverse-Poisson benchmark compares a deterministic network,
a deep ensemble, and a randomized-prior ensemble. It records coefficient recovery,
field error, PDE residuals, NLL, CRPS, calibrated simultaneous coverage, interval
width, runtime, and sample memory:

```bash
PHYDRAX_RUN_SLOW_BENCHMARKS=1 uv run pytest -q \
  tests/integration/test_uq_learned_inverse_benchmark.py
```

The repeated stress benchmark fits three independent sparse-sensor trials on
$x\in[0.05,0.65]$ and evaluates extrapolation on $x\in[0.70,1]$. It uses paired
proper-score wins, extrapolation error, stability, coverage efficiency, and an
uncertainty signal to emit `promote`, `keep_experimental`, or `remove_candidate`.
Set `PHYDRAX_UQ_STRESS_REPORT` to write its JSON result:

```bash
PHYDRAX_RUN_UQ_STRESS_BENCHMARKS=1 \
PHYDRAX_UQ_STRESS_REPORT=/tmp/phydrax-uq-stress.json \
uv run pytest -q \
  tests/integration/test_uq_learned_inverse_benchmark.py::test_sparse_sensor_extrapolation_retention_benchmark
```

The model-form benchmark generates observations from
$u(x)=2x(1-x)+0.03\sin(2\pi x)$ while every fitted solver incorrectly assumes a
constant forcing. It compares predictive scores, assumed- and true-physics residuals,
and whether epistemic scale follows the omitted mode:

```bash
PHYDRAX_RUN_UQ_MISSPEC_BENCHMARKS=1 \
PHYDRAX_UQ_MISSPEC_REPORT=/tmp/phydrax-uq-misspec.json \
uv run pytest -q \
  tests/integration/test_uq_learned_inverse_benchmark.py::test_model_misspecification_retention_benchmark
```

## Neural-operator uncertainty

`OperatorPredictiveField` wraps `PredictiveField` without flattening the operator
contract. It retains:

- named physical case axes;
- tensor-grid `OperatorAxis` names or point-cloud query coordinates;
- query masks and quadrature;
- scalar or channel-valued `OperatorOutputSpec` metadata;
- explicit epistemic, input, observation, process, and numerical sample axes.

This distinction is load-bearing. A stochastic realization represents one coherent
output function over its full query set. Query points are not independent predictive
draws, and padded points are not observations. Point-cloud coordinates may vary by
physical case, but they must be shared across a sample axis before pointwise moments
or quantiles are defined.

### Recommended method hierarchy

| Goal | First method | Escalation | Main caution |
| --- | --- | --- | --- |
| General operator epistemic uncertainty | Independently trained deep ensemble | Architecture-specific posterior subspace | Shared model-form error is invisible to ensemble spread |
| Cheap stochastic diagnostic | MC dropout with coherent full-function keys | Deep ensemble | Dropout spread is not calibrated automatically |
| Random forcing, coefficients, geometry, or initial state | Preserve named input sample axes | Joint input/epistemic predictive design | Output query geometry must align across draws |
| Small physical or calibration parameter posterior | NUTS/HMC or dense Laplace | Pathfinder or tempered SMC when justified | Likelihood must be normalized and deterministic |
| Large factorized dataset or operator-case posterior | Fixed-step SGLD with a control variate | SGNHT after reference validation | Unadjusted draws require step-halving and exact-reference checks |
| Selected neural-operator weights | Exact last-projection Laplace reference | Diagonal/Lanczos/LOBPCG Laplace | Full-weight inference is usually too large |
| Distribution-free whole-field bands | `OperatorFunctionalConformal` | Score stratification or recalibration | Exchangeability does not survive arbitrary shifts |
| Learned stochastic transition law | `GaussianFunctionOperator` with `uncertainty_source="process"` | Fixed-query `ConditionalFlowFunctionOperator` for demonstrated non-Gaussian residuals | A learned density is not a drift/diffusion identification |

Use `HomogeneousFunctionEnsemble.predict_operator` when every member has one static
architecture and output contract. Use
`HeterogeneousFunctionEnsemble.predict_operator` for different widths, families, or
external adapters; it rejects geometry or output mismatches. Use
`sample_operator_predictive` for keyed stochastic operators such as MC dropout.
`operator_input_predictive` reclassifies explicitly named physical case axes as
input draws, and ensemble prediction can retain crossed epistemic/input axes.

### Distributional operator models

`AbstractProbabilisticOperatorModel.distribution(batch)` returns one
`AbstractOperatorDistribution` per physical operator case. The distribution event
is the complete valid query field, not one point. `location`, `sample`, and
`log_prob` therefore preserve case, query, mask, output-channel, and uncertainty
source metadata.

`GaussianFunctionOperator` is the default transition-density baseline. Its wrapped
operator emits the location, an optional learned diagonal scale, and optional
low-rank loadings shared across the output field. Set `scale_mode="fixed"` to
represent only a declared noise floor plus learned factors; set
`uncertainty_source="process"` for stochastic dynamics and `"observation"` only
when the distribution represents sensor noise. `OperatorDistributionNLL` evaluates
the exact masked complete-field density during `fit_operator`.

`ConditionalFlowFunctionOperator` uses a FlowJAX conditional coupling flow for a
non-Gaussian residual around a deterministic location operator. An
`OperatorBatchConditioner` encodes named source functions into the condition vector.
The output event, mask, and physical query geometry are constructor-fixed. Loader
case broadcasting of that same geometry is accepted; changed nodes, weights, masks,
event size, or output channels are rejected. Use this path only after held-out NLL,
energy distance, tail behavior, or basin probabilities show a gain over the Gaussian
baseline.

`DistributionalSemigroupObjective` compares independently sampled direct and
composed transition laws with whole-field energy distance. It requires
`uncertainty_source="process"` and separate keys for the direct, first, and second
transitions. It tests equality in distribution; it does not assert common Brownian
paths or identify a continuous-time SDE.

Use `operator_ensemble_energy_distance` for two process ensembles. It applies the
query mask and either physical quadrature or a declared uniform measure. Continue to
report marginal CRPS and calibration separately: distributional proximity,
pointwise calibration, and simultaneous field coverage are different contracts.

### Likelihood, calibration, and scores

`FixedOperatorObservationLikelihood` defines a finite sensor likelihood over one
fixed `OperatorBatch`. It combines query and observation masks, rejects unobserved
physical cases, and sum-reduces all observed query/channel log densities. It does not
insert quadrature weights: a continuum training norm and a finite-dimensional
observation density are different mathematical objects.

For neural weights, select exact subtrees with `ParameterSubspace`. Examples include
an FNO projection, selected spectral blocks, every DeepONet branch/trunk output head,
a local-operator decoder, or a graph readout. Never use a hard-coded global
“last layer” count for branched models. Disable dropout before evaluating a posterior
density.

`OperatorFunctionalConformal` calibrates complete physical source/output cases. Its
maximum score yields simultaneous field bands. Its quadrature-weighted L2 score
yields a calibrated norm radius rather than pointwise bounds. Report marginal CRPS
and the whole-field energy score together: they answer different questions. Report
both pointwise and simultaneous coverage, interval width, and the exact physical
measure used by each reduction.

Resolution transfer, changed geometry, input noise, sensor dropout, and longer
rollouts are distribution shifts. Preserve them in result metadata and evaluate them
separately. Split-conformal nominal coverage applies to exchangeable held-out
in-distribution cases, not automatically to any shifted row.

See the [neural-operator uncertainty cookbook](cookbook/operator_uncertainty.md) and
[operator-UQ API](api/uq/operator.md). The reproducible benchmark writes separate
JSON and Parquet artifacts under
`tools/operator_benchmarks/reference/converged/operator_uq_benchmarks.*`.

## Observation likelihoods and proper scores

Native likelihoods include fixed-scale Gaussian, heteroscedastic Gaussian, and
Student-t observations. `SupervisedLikelihoodConstraint` aligns targets through a
`DatasetDomain` and can score a transformed physical observable:

```python
dataset = phx.domain.DatasetDomain(jnp.linspace(0.0, 1.0, 64)[:, None])
observed_flux = jnp.zeros((64,))

constraint = phx.constraints.SupervisedLikelihoodConstraint(
    "u",
    dataset.component(),
    observed_flux,
    phx.uq.GaussianLikelihood(0.05),
    sampling=phx.domain.PointSampling(64, design="uniform"),
    observation_operator=lambda u: phx.operators.grad(u, var="data"),
)
```

This supports state values, derivatives, fluxes, stresses, integrals, and sensor
transforms without treating a PDE residual as measurement noise.

Report held-out negative log likelihood, CRPS (Gaussian, Student-t, or empirical
ensemble), energy score for multivariate fields, interval coverage, and interval
width. `GaussianScaleCalibrator.fit` estimates one positive multiplier by the
closed-form held-out Gaussian-NLL optimum. It calibrates scale under a Gaussian
likelihood; it does not provide a finite-sample coverage guarantee.

## Uncertain predictors and normalized measurement error

When both a measured predictor \(x_i\) and response \(y_i\) are uncertain, an
observation-only residual model is generally wrong. For a smooth prediction
\(f(\theta, x_i)\), `LinearizedGaussianMeasurementLikelihood` uses the local
input Jacobian \(J_i\) and normalized effective covariance

\[
\Sigma_i(\theta)
= \Sigma_{y,i}(\theta)
+ J_i(\theta)\Sigma_{x,i}(\theta)J_i(\theta)^\mathsf{T}.
\]

```python
measured_x = jnp.linspace(0.1, 1.0, 24)[:, None]
measured_y = 1.8 * measured_x[:, 0]

measurement_term = phx.uq.LinearizedGaussianMeasurementLikelihood(
    lambda parameters, x: parameters["slope"] * x[0],
    measured_x,
    measured_y,
    input_covariance=jnp.asarray([[0.03**2]]),
    observation_covariance=jnp.asarray([[0.02**2]]),
)
measurement_space = phx.uq.ParameterSpace(
    {"slope": jnp.asarray(1.5)},
    priors={"slope": phx.uq.Normal(0.0, 3.0)},
)
measurement_posterior = phx.uq.PosteriorProblem.from_terms(
    measurement_space,
    (measurement_term,),
)
```

The quadratic residual and `log(det(Sigma_i))` terms are both mandatory. The
normalization changes with \(\theta\) whenever the model sensitivity or a
covariance callback changes; dropping it defines a different objective.
Covariances may be shared or explicitly `per_case`, and may be fixed arrays or
functions of physical parameters. No covariance shape is inferred from array
rank.

This likelihood is intentionally local and Gaussian. Use it for small or
moderate output events with differentiable predictors. Compare against an
explicit latent-input model when nonlinear input effects are material. Use
`log_prob_cases(...)` with an `ArrayMinibatchSource` containing measured inputs,
targets, and original case indices to reuse this exact term under SG-MCMC.


## Posterior contract

Bayesian inference starts from one explicit `PosteriorProblem`. It owns:

- a `ParameterSpace` whose leaves are the sampled unconstrained coordinates;
- a normalized physical-space prior or a custom joint log prior;
- invertible bijectors from unconstrained to physical parameters;
- one deterministic, scalar log likelihood over fixed observations;
- optional latent prediction, conditional observation variance, and observation
  sampling callbacks;
- an optional normalized residual callback for Gauss--Newton/Fisher curvature and EKI.

For unconstrained position $z$ and physical parameters $\theta=T(z)$, Phydrax
evaluates

$$
\log p(z\mid y)
= \log p(y\mid\theta)
+ \log p(\theta)
+ \log\left|\det J_T(z)\right|.
$$

The Jacobian term is mandatory. Use `ExpBijector` for positive parameters and
`SigmoidIntervalBijector` for bounded parameters. `ParameterSubspace` explicitly
partitions a model PyTree into sampled leaves and a frozen complement.
`from_leaf_paths(...)` selects exact array leaves. `last_layer(...)` is deliberately
generic: it selects the globally final array leaves in deterministic PyTree order;
it does not inspect model architecture or select one final layer per branch.

For a `SeparableMLP`, there is one internal MLP per input factor and no single
shared affine output head. Select every factor's final layer explicitly with
`from_subtree_paths(...)`:

```python
separable = phx.nn.inference_mode(
    phx.nn.SeparableMLP(
        in_size=2,
        out_size="scalar",
        latent_size=8,
        width_size=16,
        depth=2,
        key=jr.key(9),
    )
)
final_layer_paths = tuple(
    f".model.models[{index}].layers[{len(factor.layers) - 1}]"
    for index, factor in enumerate(separable.model.models)
)
separable_subspace = phx.uq.ParameterSubspace.from_subtree_paths(
    separable,
    final_layer_paths,
)
```

This selects every inexact array below each final `Linear`, including RWF scales
when configured and omitting absent biases naturally. If a skip projection should
also be sampled, name its `_residual_proj` subtree separately. Do not approximate
this selection with `last_layer(num_leaves=2 * num_factors)`: leaves are grouped by
factor, so that selects earlier layers from the globally last factor.

NUTS and HMC accept the resulting selected PyTree. Sampling one factor's final
layer while freezing the other factors is linear in that factor's output
parameters when its final activation is the identity. Sampling all factor heads
jointly introduces multiplicative interactions through the separable contraction;
NUTS/HMC remain valid, but this is not a conventional linear Bayesian last layer.
Disable dropout with `inference_mode` before constructing any posterior.

```python
sensor_x = jnp.linspace(0.05, 0.95, 24)
basis = 0.5 * sensor_x * (1.0 - sensor_x)
observed = 4.0 * basis
noise_scale = 0.02
observation_likelihood = phx.uq.GaussianLikelihood(noise_scale)

space = phx.uq.ParameterSpace(
    {"source": jnp.asarray(3.8)},
    priors={"source": phx.uq.Normal(0.0, 3.0)},
)
posterior = phx.uq.PosteriorProblem(
    space,
    lambda p: jnp.sum(
        observation_likelihood.log_prob(p["source"] * basis, observed)
    ),
    predict=lambda p, x: cx.Field(
        p["source"] * 0.5 * x * (1.0 - x),
        dims=("x",),
    ),
)
```

`FunctionalSolver.loss()` is a training objective, not a posterior density. Arbitrary
constraint weights, changing collocation samples, and mean reductions do not define
likelihood normalization. For a `SupervisedLikelihoodConstraint`, call
`observed_batch()` once and sum its unreduced `log_prob(...)` values inside the
posterior likelihood. Never call random `sample()` from a posterior density.

Use `FixedObservationLikelihood`, `FixedResidualLikelihood`, and
`FixedConstraintLikelihood` to construct deterministic, sum-reduced normalized
posterior terms. `PosteriorProblem.from_terms(...)` combines them without routing
through `FunctionalSolver.loss()`.

Inspect the posterior contract before starting an expensive inference run:

```python
inspection = phx.uq.diagnose_posterior(
    posterior,
    key=jr.key(9),
    num_prior_samples=16,
)
if not inspection.passed:
    raise ValueError(inspection.as_dict())
```

The report evaluates the initial target and gradient, reports exact nonfinite
gradient locations, checks unconstrained-to-physical round trips, compares eager,
repeated, JIT, and VMAP evaluation, and optionally probes declared-prior samples.
`inspection.capabilities` records only static semantic capabilities: factorized
prior sampling, latent prediction, observation variance/sampling, and a normalized
Gauss--Newton residual. It does not probe prediction callbacks with invented,
domain-specific query arguments.


## MAP estimation

### Bounded global initialization

`search_map` minimizes `PosteriorProblem.negative_log_density(...)` over an explicit
finite box when the posterior is multimodal, nonsmooth, or poorly served by one local
initialization. The box is defined in the `ParameterSpace`'s unconstrained position
coordinates, not in physical parameter coordinates. Lower and upper bounds must be
PyTrees matching `space.initial`; a scalar bound may broadcast only within its
corresponding leaf. Every resolved value must be finite and strictly ordered.

```python
search = phx.optim.DifferentialEvolutionSearch(
    32,
    100,
    design=phx.sampling.SobolDesign(scrambled=True),
)
global_mode = phx.uq.search_map(
    posterior,
    search,
    key=jr.key(10),
    position_bounds=(
        {"source": -6.0},
        {"source": 6.0},
    ),
)
```

The search evaluates the same full posterior density used by local MAP, including
priors and bijector Jacobians. Bounded unconstrained positions therefore reconstruct
valid positive or interval-valued physical parameters through their declared
bijectors. Bounds are never inferred from priors or physical bijector endpoints.

`MAPSearchResult` preserves the best unconstrained position and constrained
parameters, final population positions and objectives, resolved bound PyTrees, root
key, search configuration, initialization design signature, best-objective history,
and exact generation, objective-evaluation, and invalid-evaluation counts.
Non-finite candidates are counted and excluded from selection. The population is an
optimizer state shaped by selection; it is not a posterior sample. Population
dispersion convergence is not stationarity evidence or proof of a global optimum.

### Local MAP refinement

`find_map` compiles the complete JAX-native strong-Wolfe L-BFGS transition and
reports both unconstrained and physical positions, final log density and gradient
norm, objective evaluations, compilation time, warm execution time, total runtime,
and a termination reason. Pass the global position explicitly when local refinement
is wanted:

```python
mode = phx.uq.find_map(
    posterior,
    global_mode.position,
    gradient_tolerance=1e-7,
    max_steps=500,
)
```

`search_map` never runs L-BFGS or Laplace implicitly. This keeps the global
population criterion, local gradient criterion, extra evaluations, and failure modes
separate and observable.

`MAPResult.compilation_seconds`, `execution_seconds`, and `mean_step_seconds`
separate compiler cost from numerical optimization. Repeated problems reuse the
compiled initial evaluation and L-BFGS step when their PyTree shapes and static
callables match. Keep shared callbacks stable and put changing observations in
structured posterior-term array fields rather than creating a new likelihood lambda
for each dataset. `GaussianProcessMarginalLikelihood` provides this contract for
exact and FITC discrepancy models.

Pass `mode.position` to `fit_laplace`. A failed or non-stationary optimization is
never silently accepted as a Laplace center.

## NUTS and fixed-trajectory HMC

`sample_nuts` and `sample_hmc` use BlackJAX. Each chain adapts independently during
warmup, then samples with a frozen kernel. Results preserve separate leading chain
and draw axes instead of silently pooling chains.

```python
posterior_draws = phx.uq.sample_nuts(
    posterior,
    key=jr.key(10),
    num_chains=4,
    num_warmup=1000,
    num_samples=1000,
    target_acceptance_rate=0.9,
    chain_method="vectorized",
)

prediction = posterior_draws.predict(
    jnp.linspace(0.0, 1.0, 65),
    batch_size=128,
)

report = posterior_draws.convergence_report(
    max_rhat=1.01,
    min_bulk_ess=400,
    min_tail_ess=400,
)
report.raise_for_failure()
```

`chain_method="sequential"` is the conservative low-compilation-memory path.
`"vectorized"` compiles one batched transition and synchronizes chains after every
draw. `"interleaved"` retains vectorized warmup, then advances every active production
chain by one velocity-Verlet step per scheduler tick. A chain that finishes a draw can
begin its next draw while another chain continues a longer trajectory. All methods
preserve the same indexed chain keys and separate chain/draw axes.

Interleaving is intended for many-chain accelerator workloads where trajectory
lengths vary and each log-density gradient is expensive. It can be slower for a few
chains, cheap targets, or nearly uniform trajectory lengths, so it is explicit rather
than automatic. Checkpoints remain equal-draw boundaries: a small
`checkpoint_every`, especially one draw, removes most of the scheduling benefit.
Flow-assisted NUTS does not currently accept the interleaved method. The scheduling
construction follows [Efficiently Vectorized MCMC on Modern
Accelerators](https://arxiv.org/abs/2503.17405).

`convergence_report(...)` applies caller-controlled release gates and reports exact
failing PyTree leaves. `MCMCResult` retains tuned warmup parameters, final chain
states, energies, integration depths, deterministic keys, adaptation and sampling
runtimes, throughput, and sample-memory size. A repeated root key reproduces samples
and diagnostics.

Pass `initial_positions` with one leading chain axis when chains must begin at
different represented modes; `initial_position` retains the replicated-start
convenience. The two arguments are mutually exclusive.


Long MCMC runs can checkpoint after a fixed number of completed draws per chain:
pass `checkpoint_path`, `checkpoint_every`, and a stable caller-owned
`checkpoint_id`. Resume with `resume_from` and the same sampling configuration.
Warmup is not repeated. Phydrax validates the problem fingerprint, parameter PyTree,
sampler settings, package versions, checksums, and archive schema before resuming.
The indexed random-key schedule makes uninterrupted and resumed draws identical on
the same backend.

Posterior evaluation must be deterministic for a fixed position. Do not place
adaptive collocation, minibatch likelihoods, active dropout, or other random sampling
inside `log_density`. NUTS is the default for low-dimensional physical parameters,
noise scales, and explicitly selected small subspaces. Sampling every neural-network
weight is deliberately not the default.

## Fixed-step stochastic-gradient MCMC

Use stochastic-gradient MCMC only when the likelihood is a sum over many
statistical factors and full-density transitions are the measured bottleneck.
`MinibatchPosteriorProblem` makes that algebra explicit. A factor must be one
independent likelihood contribution; it is not an arbitrary slice of a residual
array. `ArrayMinibatchSource` emits deterministic shuffled epochs, retains the
padded final batch, and excludes padding through `factor_mask`:

```python
minibatch_source = phx.uq.ArrayMinibatchSource(
    {
        "basis": basis,
        "observation": observed,
    },
    batch_size=8,
    seed=17,
)


def likelihood_factors(parameters, batch):
    prediction = parameters["source"] * batch.data["basis"]
    return observation_likelihood.log_prob(
        prediction,
        batch.data["observation"],
    )


minibatch_posterior = phx.uq.MinibatchPosteriorProblem(
    space,
    likelihood_factors,
    num_factors=basis.size,
    full_log_likelihood=posterior.log_likelihood,
    predict=lambda parameters, x: cx.Field(
        parameters["source"] * 0.5 * x * (1.0 - x),
        dims=("x",),
    ),
)
minibatch_inspection = phx.uq.diagnose_minibatch_posterior(
    minibatch_posterior,
    minibatch_source,
)
if not minibatch_inspection.passed:
    raise ValueError(minibatch_inspection.as_dict())
```

The inspection sums one complete epoch and compares both its value and gradient
with `full_log_likelihood` when supplied. This catches incorrect population
scaling, duplicated priors, missing factors, and source/problem mismatches before
sampling.

`sample_sgld` implements fixed-step overdamped Langevin updates.
`sample_sgnht` adds momentum and a scalar thermostat to absorb stochastic-gradient
noise:

```python
control = phx.uq.build_sgmcmc_control_variate(
    minibatch_posterior,
    minibatch_source,
    mode.position,
)

sgld = phx.uq.sample_sgld(
    minibatch_posterior,
    minibatch_source,
    key=jr.key(20),
    step_size=1e-4,
    num_chains=4,
    num_burnin=1000,
    num_samples=2000,
    steps_per_sample=2,
    control_variate=control,
    chain_method="vectorized",
)
sgnht = phx.uq.sample_sgnht(
    minibatch_posterior,
    minibatch_source,
    key=jr.key(21),
    step_size=5e-4,
    diffusion=0.01,
    num_chains=4,
    num_burnin=1000,
    num_samples=2000,
    steps_per_sample=2,
    control_variate=control,
    chain_method="vectorized",
)

mixing = sgld.mixing_report(
    max_rhat=1.05,
    min_bulk_ess=200,
    min_tail_ess=200,
)
mixing.raise_for_failure()
```

Both samplers preserve separate chain/draw axes, nested PyTrees, constrained
physical samples, source configuration and fingerprint, deterministic keys,
gradient/update throughput, memory, gradient-norm traces, and nonfinite-update
locations. SGNHT additionally retains thermostat and momentum-norm traces.
Checkpointing resumes the indexed source/transition schedule exactly and rejects
changed problem, source, control-variate, PyTree, or sampler identities.

These are unadjusted fixed-step approximations. Burn-in discards early states; it
is not adaptation. Rank diagnostics measure between-chain mixing and do not detect
stationary discretization bias. For every scientific use:

1. rerun at half the step size and compare posterior and predictive moments;
2. inspect stochastic-gradient variance, with and without a control variate;
3. compare against NUTS or dense Laplace on a tractable reduced/reference problem;
4. report the batch definition, population size, step size, burn-in, thinning,
   update count, and approximation label.

Prefer NUTS or Laplace when full-data inference is feasible. Current SG-MCMC
sources support uniform factor subsampling only: no decreasing-step schedule,
automatic step-size adaptation, query-anchor subsampling, likelihood-dependent
sampling, multi-host source execution, SGHMC, pSGLD/RMSProp geometry, or online
gradient-noise covariance estimation is implied.

## Flow-assisted NUTS

`sample_flow_nuts` combines independently adapted NUTS chains with a shared
FlowJAX normalizing flow. It is intended for posteriors with nonlinear global
geometry or multiple modes that are already represented by the initial chains.
It is not a mode-discovery algorithm and does not estimate evidence.

```python
flow_config = phx.uq.FlowNUTSConfig(
    num_adaptation_rounds=4,
    num_local_adaptation_steps=100,
    num_global_adaptation_steps=20,
    num_local_steps=3,
    num_global_steps=1,
    history_capacity_per_chain=1000,
    flow_layers=6,
    max_epochs=100,
)
flow_draws = phx.uq.sample_flow_nuts(
    posterior,
    key=jr.key(11),
    num_chains=4,
    num_warmup=1000,
    num_samples=1000,
    initial_positions={
        "source": jnp.asarray([-3.0, -1.0, 1.0, 3.0]),
    },
    target_acceptance_rate=0.9,
    config=flow_config,
    chain_method="vectorized",
)
```

All kernels act in flattened unconstrained coordinates. During adaptation, local
NUTS transitions populate a fixed-capacity, chain-stratified reservoir; the flow is
refit from pooled reservoir samples and tested with exact independence
Metropolis--Hastings proposals. For current state `x`, proposal `y`, target `π`, and
normalized flow density `q`, acceptance is
`min(1, π(y) q(x) / (π(x) q(y)))`. Both proposal-density terms are mandatory.
Nonfinite states or densities are rejected and counted.

After the last adaptation round, Phydrax freezes both the flow and tuned NUTS
parameters, runs an optional local-only stabilization phase, then returns draws from
one fixed composite kernel. No returned draw trains the flow. `FlowNUTSResult`
preserves the ordinary MCMC prediction and convergence interfaces and adds
training/validation losses, adaptation proposal ESS, local/global acceptance,
nonfinite global proposal counts, phase timings, bounded-memory accounting, and the
frozen flow.

Automatic chain initialization requires declared factorized priors. A custom joint
log prior requires explicit `initial_positions`. Checkpointing commits complete
adaptation rounds, stabilization chunks, and production chunks; resume reconstructs
the dynamic FlowJAX arrays against a locally rebuilt static template and rejects
configuration, package-version, shape, dtype, or flow-fingerprint mismatches.

This implementation is native Phydrax orchestration, not a wrapper around
[`flowMC`](https://github.com/kazewong/flowMC). The flow-assisted sampling rationale
follows that work, while Phydrax retains its own posterior, exact-kernel,
checkpoint, diagnostics, and result contracts.


## Dense and structured Laplace approximation

`fit_laplace` approximates one posterior mode. The default `curvature="exact"` path
forms the full Hessian of the complete transformed negative log posterior, checks
stationarity and positive definiteness, and uses a Cholesky factor. It is the
correctness reference for small parameter spaces.

```python
# This conjugate example has an analytic MAP.
posterior_variance = 1.0 / (
    1.0 / 3.0**2 + jnp.vdot(basis, basis) / noise_scale**2
)
posterior_mean = posterior_variance * jnp.vdot(basis, observed) / noise_scale**2

laplace = phx.uq.fit_laplace(
    posterior,
    {"source": posterior_mean},
)
approximate_prediction = laplace.predict(
    jr.key(11),
    jnp.linspace(0.0, 1.0, 65),
    num_samples=512,
)
```

For cheap local prediction moments, propagate the Laplace covariance without
drawing:

```python
linearized_prediction = laplace.linearized_predict(
    jnp.linspace(0.0, 1.0, 65)
)
linearized_variance = linearized_prediction.exact_variance()
```

`linearized_predict` differentiates the complete prediction path from
unconstrained parameters, so parameter bijectors are included automatically.
Dense Laplace supports exact first-order diagonals. Structured Laplace retains
its covariance-vector product and can materialize only a caller-bounded small
output covariance or estimate its diagonal with keyed Hutchinson probes.
Continue to use `predict(...)` when nonlinear posterior-predictive shape,
mean shifts, skewness, or tails matter.

For larger subspaces, the same entry point dispatches to a Phydrax adapter around
Laplax:

```python
diagonal = phx.uq.fit_laplace(
    posterior,
    mode.position,
    curvature="diagonal",
)
```

When every declared prior is Gaussian in unconstrained coordinates
(`Normal` + identity or `LogNormal` + exponential bijector), structured Laplace
automatically whitens to a standard-normal prior. An explicit scalar
`prior_precision` remains available only for identity-transformed isotropic models.
`physical_covariance_vector_product(...)` applies the delta-method covariance after
bijectors; dense Laplace exposes the corresponding `physical_covariance()` and
`physical_correlation()`.

Supported structured modes are `full`, `diagonal`, `lanczos`, and `lobpcg`.
Set `likelihood_curvature="ggn"` only after declaring a normalized
`PosteriorProblem.gauss_newton_residual` callback; this computes matrix-free
$J^\top J$ curvature and avoids indefinite likelihood Hessians. Lanczos and LOBPCG
require `rank < dimension` and a key. Results report retained rank, backend factor
memory, and curvature type.

`full` still has dense memory and factorization cost. `diagonal` loses parameter
correlations. Low-rank modes keep only leading curvature directions and use the
prior in the complement. All Laplace modes are local to one mode; they do not
represent multimodality, and their intervals are approximate rather than calibrated
coverage guarantees.

## Pathfinder and tempered SMC

`fit_pathfinder` selects the highest-ELBO local Gaussian found along one L-BFGS
trajectory. It is useful for rapid diagnostics and approximate initialization, not
as a calibrated replacement for NUTS:

```python
pathfinder = phx.uq.fit_pathfinder(
    posterior,
    key=jr.key(12),
    num_samples=1000,
)
pathfinder_prediction = pathfinder.predict(jnp.linspace(0.0, 1.0, 65))
```

The result retains the optimization path, ELBO, target and approximation densities,
importance log ratios, runtime, and sample memory.

Use `sample_tempered_smc` for demonstrated low-dimensional multimodal posteriors.
It draws particles from declared priors, adaptively chooses likelihood temperatures
by ESS, applies fixed-trajectory HMC rejuvenation, and performs a final unweighted
resample:

```python
particles = phx.uq.sample_tempered_smc(
    posterior,
    key=jr.key(13),
    num_particles=1000,
    target_ess=0.8,
)
```

Inspect its temperature schedule, per-stage ESS, acceptance/divergence rates,
unique surviving initial particles, and log-evidence estimate. Custom joint priors
need an explicit `prior_position_sampler`; no sampler is inferred from a log density.

Tempered SMC accepts `checkpoint_path`, `checkpoint_id`, and `resume_from`.
It commits after every complete temperature stage, preserving particles, weights,
ancestry, evidence increments, rejuvenation state, and the deterministic key
schedule. Resume never invokes the prior sampler again.

## Ensemble Kalman inversion

`fit_eki` is a derivative-free, tempered ensemble inverse solver for problems that
declare a fixed normalized residual
$r(\theta)=\Gamma^{-1/2}(\mathcal G(\theta)-y)$. It evaluates that residual for each
ensemble member, chooses likelihood-temperature increments by effective sample
size, and solves updates in ensemble space:

```python
eki_posterior = phx.uq.PosteriorProblem(
    space,
    lambda p: -0.5
    * jnp.sum(((p["source"] * basis - observed) / noise_scale) ** 2),
    gauss_newton_residual=lambda p: (
        p["source"] * basis - observed
    ) / noise_scale,
    predict=lambda p, x: cx.Field(
        p["source"] * 0.5 * x * (1.0 - x),
        dims=("x",),
    ),
)
eki = phx.uq.fit_eki(
    eki_posterior,
    key=jr.key(14),
    ensemble_size=128,
    target_ess=0.8,
)
```

The result retains initial and final ensembles in unconstrained and physical
coordinates, the temperature schedule, residual norms, ensemble spread, effective
rank, parameter-update norms, forward-solve count, termination reason, and
predictive methods. Bijectors remain active, so positive and bounded physical
parameters stay valid.

EKI is exact only for an ideal linear-Gaussian inverse problem in the
infinite-ensemble limit. For nonlinear problems it is an approximate ensemble
inverse method, not an asymptotically exact posterior sampler. Its affine-subspace
rank is at most `ensemble_size - 1`; finite ensembles can collapse and underestimate
uncertainty. Use it for expensive forward models, physical coefficients,
reduced-basis parameter fields, and selected neural subspaces—not unrestricted
full-network weights. `inflation` is explicit algorithmic regularization, not
measurement likelihood. Benchmark nonlinear EKI results against NUTS or Laplace
where feasible.

## Checkpoints and portable results

Checkpoints are private resumable run state. Portable result archives are a separate
public format:

```python
result_path = phx.uq.export_result(eki, "/tmp/eki.phxuq")
portable = phx.uq.read_result_archive(result_path)
```

Both are ZIP containers with JSON metadata, individual NumPy array members,
SHA-256 checksums, atomic replacement, and no pickle or Python object arrays.
Portable archives export representable result arrays and explicitly list excluded
live callables. `MAPSearchResult` archives retain the reconstructed best position,
full final population, resolved bounds, deterministic key, convergence history,
search configuration, design provenance, and exact evaluation accounting.
`FlowNUTSResult` archives include frozen flow parameters, loss histories, local and
global sampler statistics, deterministic keys, and phase timings. `SGMCMCResult`
archives include algorithm and approximation identities, source configuration and
fingerprint, control-variate metadata, gradient/update accounting, thermostat or
momentum traces when present, and deterministic replay keys.
`phx.uq.to_arviz(posterior_draws)` accepts ordinary, flow-assisted, or
stochastic-gradient MCMC and retains separate `chain` and `draw` dimensions.
Method-specific sample statistics are added when present. Generic observed-data and
pointwise-log-likelihood groups are omitted because neither posterior contract
promises that metadata.

## Gaussian-process model discrepancy

`ExactGaussianProcessDiscrepancy` models an additive scalar discrepancy
$\delta(x)$ rather than mislabeling neural-network spread as model-form uncertainty:

$$
y = u_\theta(x) + \delta(x) + \epsilon,\qquad
\delta\sim\mathcal{GP}(0,K),\quad
\epsilon\sim\mathcal N(0,\sigma^2).
$$

Its log marginal likelihood analytically integrates the latent discrepancy values.
This makes physical parameters and kernel hyperparameters ordinary
`PosteriorProblem` leaves for NUTS or dense Laplace:

```python
discrepancy = phx.uq.ExactGaussianProcessDiscrepancy(
    sensor_x,
    observed,
    kernel="matern32",
)

log_likelihood = discrepancy.log_marginal_likelihood(
    4.0 * basis,
    amplitude=0.03,
    length_scale=0.25,
    noise_scale=0.005,
)
conditioned = discrepancy.condition(
    4.0 * basis,
    jnp.linspace(0.0, 1.0, 65),
    amplitude=0.03,
    length_scale=0.25,
    noise_scale=0.005,
    output_dim="x",
)
predictive = conditioned.predictive_field(
    4.0 * 0.5 * jnp.linspace(0.0, 1.0, 65) * (1.0 - jnp.linspace(0.0, 1.0, 65)),
    jr.key(12),
    num_samples=256,
    observation_variance=0.005**2,
)
```

When the kernel hyperparameters and point designs are fixed, factor the observation
covariance once instead of rebuilding it for every physical-parameter evaluation:

```python
factor = discrepancy.factor(
    amplitude=0.03,
    length_scale=0.25,
    noise_scale=0.005,
)
conditioner = factor.conditioner(
    jnp.linspace(0.0, 1.0, 65),
    output_dim="x",
)

residual = discrepancy.residual(4.0 * basis)
log_likelihood = factor.log_probability(residual)
conditioned = conditioner.condition(residual)
```

`ExactGaussianProcessFactor` retains the dense Cholesky factor.
`SparseGaussianProcessFactor` retains the FITC feature, diagonal, and correction
factors. A `GaussianProcessConditioner` additionally precomputes the query/observation
projection and conditional covariance, so changing a residual only performs a
matrix-vector product. Factor construction is intended to be amortized over repeated
evaluations. Hyperparameters passed to `factor(...)` are fixed: use
`log_marginal_likelihood(...)` when amplitude, length scale, or noise remains an
inferred parameter.

Conditioned samples are coherent functions over all query points. Latent GP
variation is an epistemic sample axis; independent measurement noise remains
conditional observation variance.

`ExactGaussianProcessDiscrepancy` is the dense scalar-output reference.
`SparseGaussianProcessDiscrepancy` uses an explicit FITC inducing set with
$O(nm^2+m^3)$ work and $O(nm+m^2)$ factor storage; use it only after dense
conditioning is the measured bottleneck. `factor_storage_elements` makes that
tradeoff visible.

`MultiOutputGaussianProcessDiscrepancy` implements a separable intrinsic
coregionalization covariance $K_x\otimes B$. The positive-definite output covariance
$B$ is required—outputs are never made independent by default. Its conditioned
samples preserve both point and output dimensions.

Physical parameters and a flexible discrepancy can explain the same signal.
`discrepancy_identifiability_report(...)` therefore requires repeated baseline,
fixed-hyperparameter GP, and jointly inferred GP comparisons. It gates physical
parameter bias, held-out NLL/CRPS, coverage, and maximum physical/GP posterior
correlation, returning every exact failure rather than a generic warning.

## Method boundaries

Phydrax currently recommends:

1. NUTS for low-dimensional, effectively unimodal physical inverse problems.
2. Flow-assisted NUTS for represented multimodality or nonlinear global geometry at
   moderate dimension; initialize chains across known modes and inspect exact global
   acceptance and ordinary rank diagnostics.
3. Exact dense Laplace as the small-problem Gaussian reference.
4. Whitened GGN, diagonal, or low-rank Laplax for selected larger subspaces.
5. EKI for derivative-free physical or reduced-coordinate inverse problems,
   benchmarked against NUTS or Laplace where feasible.
6. Pathfinder for rapid local diagnostics, always benchmarked against NUTS.
7. Tempered SMC for low-dimensional mode discovery and evidence estimation.
8. Fixed-step SGLD, optionally with an exact-center control variate, for large
   uniformly factorized likelihoods after step-halving and exact-reference checks.
   Use SGNHT only when its momentum/thermostat dynamics improve measured mixing.
9. Deep ensembles for independently trained neural-model epistemic variation.
10. Exact GP discrepancy for moderate scalar data, explicit coregionalization for
    correlated outputs, and FITC only when dense scaling fails.

Mean-field VI, standalone variational normalizing-flow posteriors, SWAG, SGHMC,
pSGLD/RMSProp geometry, decreasing-step stochastic approximation,
non-Gaussian/sparse variational GPs, and full-network HMC remain unsupported. None
is silently approximated by the methods above.

## Conformal calibration

Always split empirical cases into train, calibration, and test sets. For trajectories,
ragged rows, and graphs, a case is the complete independent trajectory or graph—not a
time point or node.

```python
num_cases = 64

train, calibration, test = (
    phx.data_utils.train_calibration_test_split_indices(
        num_cases,
        calibration_fraction=0.2,
        test_fraction=0.2,
        key=jr.key(2),
    )
)
```

- `SplitConformal`: scalar absolute-residual intervals.
- `NormalizedConformal`: residuals divided by predicted scale.
- `FunctionalConformal`: one maximum normalized score per complete field,
  trajectory, ragged case, or graph. Its maximum-score interval is a simultaneous
  band over the declared physical dimensions.
  `score="l2"` calibrates a weighted norm ball; `interval()` rejects that score
  because a norm ball is not representable as exact coordinatewise bounds.

All calibrators use the exact finite-sample rank
$k=\lceil(n+1)(1-\alpha)\rceil$. If $k>n$, Phydrax rejects the requested interval
instead of silently clamping the rank. Coverage requires exchangeable calibration
and test cases. Pointwise and simultaneous coverage are separate contracts.

## Local covariance propagation

`propagate_linearized` is the matrix-free first-order path for a smooth
scientific map. It evaluates the nominal output once, keeps JVP and Hermitian
VJP actions, and applies \(J C_x J^\mathrm{H}\) without constructing a
Jacobian.

```python
def forward(diffusivity, source):
    return diffusivity + source

center = {
    "diffusivity": jnp.asarray(0.2),
    "source": jnp.asarray([0.9, 1.0, 1.1]),
}
covariance = phx.uq.DiagonalCovariance(
    {
        "diffusivity": jnp.asarray(0.01),
        "source": jnp.full((3,), 0.02),
    }
)
local = phx.uq.propagate_linearized(
    lambda value: forward(value["diffusivity"], value["source"]),
    center,
    covariance,
)
output_variance = local.exact_variance()
```

| Situation | First method | Escalation |
| --- | --- | --- |
| Smooth map, small perturbations, cheap moments | `propagate_linearized` | QMC or Monte Carlo validation |
| Dense small input covariance | `DenseCovariance` | Factorize if repeatedly reused |
| Low-rank prior or empirical modes | `FactorCovariance` | Increase retained rank |
| Matrix-free covariance or PDE inverse action | `CovarianceOperator` | Keyed Hutchinson diagonal |
| Small output diagnostic | Guarded `materialize_covariance` | Keep covariance-vector products for large fields |
| Strong nonlinearity, discontinuity, threshold, or multimodality | Joint QMC or posterior draws | Explicit latent/reference model |

`exact_variance()` means exact under the first-order approximation, not exact
for the nonlinear model. `estimate_variance(...)` additionally carries
Hutchinson Monte Carlo error. Preserve `coordax.Field` dimensions and uncertainty
source labels in downstream summaries. Complex maps must be genuinely
complex-linear and request `complex_linear=True`; otherwise represent real and
imaginary parts explicitly.


## Uncertain inputs and joint QMC

`Uniform`, `Normal`, `LogNormal`, and `EmpiricalDistribution` implement sampling,
inverse CDFs, log densities, moments, and support. `ProbabilityDomain` embeds one
random variable in a labeled product domain with probability measure one. Unbounded
distributions reject endpoint components.

Use `sample_joint` for propagation and sensitivity. It generates one scrambled-Sobol
(or other supported) design in $d$ dimensions and transforms each column through its
marginal inverse CDF. Never create $d$ unrelated one-dimensional Sobol sequences;
that destroys the joint low-discrepancy design.

The same reference-design engine powers paired domain sampling and Monte Carlo
integration. String shorthands such as `"sobol_scrambled"` and typed
`phx.sampling.SobolDesign(scrambled=True)` therefore share one scrambling,
reproducibility, and capability contract.

```python
inputs = phx.uq.sample_joint(
    {
        "diffusivity": phx.uq.LogNormal(-2.0, 0.25),
        "source": phx.uq.Normal(1.0, 0.1),
    },
    num_samples=128,
    key=jr.key(3),
)
prediction = phx.uq.propagate(
    lambda diffusivity, source: forward(diffusivity, source),
    inputs,
    batch_size=64,
)
```

`propagate` records non-finite realizations in `PredictiveField.valid`, or raises with
`valid_policy="raise"`. Chunked and unchunked evaluation preserve the same samples.

### Reusable sparse parameter surrogates

For repeatedly evaluated low- or moderate-effective-dimensional observables,
fit a non-trainable Smolyak surrogate over the labeled uncertain-input domain:

```python
diffusivity = phx.domain.ProbabilityDomain(
    phx.uq.LogNormal(-2.0, 0.25),
    label="diffusivity",
)
source = phx.domain.ProbabilityDomain(
    phx.uq.Normal(1.0, 0.1),
    label="source",
)
parameter_domain = diffusivity @ source

@parameter_domain.Function("diffusivity", "source")
def observable(diffusivity, source):
    return jnp.stack(
        (
            diffusivity + source,
            diffusivity * source,
        )
    )

surrogate = phx.operators.interpolate_smolyak(
    observable,
    phx.operators.SmolyakInterpolationPlan(
        2,
        5,
        axis_rules="auto",
    ),
)
```

`Normal` and `LogNormal` parameters are interpolated in standard-normal
reference coordinates. `Uniform` parameters use a bounded uniform reference.
The surrogate snapshots one coupled batch of source evaluations, preserves
vector/tensor outputs, and can subsequently be differentiated or integrated
under an explicit measure. It is a deterministic approximation and does not
introduce posterior or sampling uncertainty. See
[Smolyak interpolation](api/operators/interpolation.md).

## Sobol sensitivity

`sobol_indices` uses Saltelli first-order and Jansen total-order estimators. Base
matrices $A$ and $B$ come from one $2d$-dimensional design; hybrid matrices replace
one column at a time. Scalar, vector, field, trajectory, and graph-array outputs are
supported. Output reduction is explicit and accepts masks and non-negative weights.

```python
distributions = inputs.distributions

result = phx.uq.sobol_indices(
    forward,
    distributions,
    num_samples=256,
    key=jr.key(4),
    batch_size=256,
)
first_order = result.first_order
total_order = result.total_order
```

Zero-variance and non-finite outputs are rejected because their indices are undefined.
Sobol estimators assume independent input marginals; dependence requires a different
sensitivity design.
