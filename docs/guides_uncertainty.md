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
driver = spde.wiener_driver(jr.key(0), realization_id="heat-0")
solution = phx.solver.solve_diffrax_ensemble(
    spde.problem,
    save_times=jnp.linspace(0.0, 0.2, 21),
    driver=driver,
    num_paths=128,
    dt0=1e-3,
)
prediction = solution.to_predictive(
    sample_dim="path",
    time_dim="time",
    state_dims=("space",),
)
```

Reusing the same driver reproduces every Brownian path; changing its key changes
the realization. `basis_id` records the spatial noise modes, eigenvalues,
quadrature, and discretization identity. A refined grid or changed truncation
therefore receives a different fingerprint.

The path axis alone says nothing about numerical uncertainty. To quantify
spatial truncation, time stepping, or solver error, run an explicit discretization
ensemble and label that additional axis `numerical`. Do not merge those runs
into the process axis: process covariance and numerical sensitivity answer
different questions.

For the full method-of-lines and noise-basis contracts, see
[API → Solver → Differential equations](api/solver/differential.md).

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
    num_cases=64,
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

space = phx.uq.ParameterSpace(
    {"source": jnp.asarray(3.8)},
    priors={"source": phx.uq.Normal(0.0, 3.0)},
)
posterior = phx.uq.PosteriorProblem(
    space,
    lambda p: -0.5
    * jnp.sum(((observed - p["source"] * basis) / noise_scale) ** 2),
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

## MAP estimation

`find_map` compiles the complete JAX-native strong-Wolfe L-BFGS transition and
reports both unconstrained and physical positions, final log density and gradient
norm, objective evaluations, compilation time, warm execution time, total runtime,
and a termination reason:

```python
mode = phx.uq.find_map(
    posterior,
    gradient_tolerance=1e-7,
    max_steps=500,
)
```

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

`chain_method="sequential"` is the conservative low-compilation-memory path;
`"vectorized"` compiles one batched transition while preserving identical chain keys
and separate chain/draw axes. `convergence_report(...)` applies caller-controlled
release gates and reports exact failing PyTree leaves. `MCMCResult` retains tuned
warmup parameters, final chain states, energies, integration depths, deterministic
keys, adaptation and sampling runtimes, throughput, and sample-memory size. A
repeated root key reproduces samples and diagnostics.

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
SHA-256 checksums, atomic replacement, and no pickle or Python object
arrays. Portable archives export representable result arrays and explicitly list
excluded live callables. `phx.uq.to_arviz(posterior_draws)` converts MCMC chains to
ArviZ `posterior` and `sample_stats` groups while retaining separate `chain` and
`draw` dimensions. Generic observed-data and pointwise-log-likelihood groups are
omitted because `PosteriorProblem` does not promise that metadata.

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
2. Exact dense Laplace as the small-problem Gaussian reference.
3. Whitened GGN, diagonal, or low-rank Laplax for selected larger subspaces.
4. EKI for derivative-free physical or reduced-coordinate inverse problems,
   benchmarked against NUTS or Laplace where feasible.
5. Pathfinder for rapid local diagnostics, always benchmarked against NUTS.
6. Tempered SMC only for demonstrated low-dimensional multimodal posteriors.
7. Deep ensembles for independently trained neural-model epistemic variation.
8. Exact GP discrepancy for moderate scalar data, explicit coregionalization for
   correlated outputs, and FITC only when dense scaling fails.

Mean-field VI, SWAG, SGMCMC, normalizing-flow posteriors, non-Gaussian/sparse
variational GPs, and full-network HMC remain unsupported. None is silently
approximated by the methods above.

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

## Uncertain inputs and joint QMC

`Uniform`, `Normal`, `LogNormal`, and `EmpiricalDistribution` implement sampling,
inverse CDFs, log densities, moments, and support. `ProbabilityDomain` embeds one
random variable in a labeled product domain with probability measure one. Unbounded
distributions reject endpoint components.

Use `sample_joint` for propagation and sensitivity. It generates one scrambled-Sobol
(or other supported) design in $d$ dimensions and transforms each column through its
marginal inverse CDF. Never create $d$ unrelated one-dimensional Sobol sequences;
that destroys the joint low-discrepancy design.

```python
def forward(diffusivity, source):
    return diffusivity + source

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
