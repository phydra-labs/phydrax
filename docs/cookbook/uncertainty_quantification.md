# Uncertainty quantification

This recipe covers independently initialized ensembles, proper scoring, split
conformal calibration, uncertain-input propagation, singular covariance-factor
Gaussian moments, global and matrix-free local sensitivity, explicit Bayesian
physical parameters, MAP/NUTS/flow-assisted NUTS/Laplace/Pathfinder inference,
fixed-step SGLD/SGNHT for factorized data, and scalable Gaussian-process model
discrepancy.

For geometry-aware output functions, independent source/query discretizations, and
whole-field calibration, use the dedicated
[neural-operator uncertainty recipe](operator_uncertainty.md).

## 1. Build an ensemble

```python
import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx


def model_factory(key):
    return phx.nn.models.MLP(
        in_size=1,
        out_size="scalar",
        width_size=64,
        depth=4,
        key=key,
    )


ensemble = phx.uq.HomogeneousFunctionEnsemble.from_factory(
    model_factory,
    num_members=4,
    key=jr.key(0),
    source_dim="member",
)
```

The compact construction above illustrates the predictive contract. For a trained
ensemble, fit every member independently through its own `FunctionalSolver` and use
`phx.uq.fit_ensemble`; do not vectorize solver logging or adaptive-collocation state.

## 2. Predict with aligned members

One evaluation key deterministically identifies all member evaluations. Member
predictions retain an explicit epistemic source axis.

```python
query = jnp.linspace(0.0, 1.0, 256)[:, None]
prediction = ensemble.predict(query, key=jr.key(1))
mean = prediction.mean()
spread = prediction.epistemic_variance()
```

## 3. Evaluate held-out scores

```python
ensemble_values = prediction.samples.data
held_out_target = jnp.sin(query[:, 0])

target = held_out_target
crps = phx.uq.ensemble_crps(ensemble_values, target, sample_axis=0)
```

Also report interval width and empirical coverage. Spread alone does not imply a
coverage level.

## 4. Calibrate independent cases

Create train/calibration/test splits before fitting. Do not calibrate on training
residuals.

```python
num_cases = 32
calibration_center = jnp.linspace(0.0, 1.0, 16)
calibration_scale = jnp.full((16,), 0.1)
calibration_target = calibration_center + 0.05 * jnp.sin(calibration_center)
test_center = jnp.linspace(0.0, 1.0, 8)
test_scale = jnp.full((8,), 0.1)
test_target = test_center + 0.05 * jnp.sin(test_center)

train_idx, calibration_idx, test_idx = (
    phx.data_utils.train_calibration_test_split_indices(
        num_cases,
        calibration_fraction=0.2,
        test_fraction=0.2,
        key=jr.key(2),
    )
)

calibrator = phx.uq.NormalizedConformal.calibrate(
    calibration_center,
    calibration_scale,
    calibration_target,
    alpha=0.1,
    case_dim=0,
)
interval = calibrator.interval(test_center, test_scale)
coverage = phx.uq.interval_coverage(
    interval.lower.data,
    interval.upper.data,
    test_target,
)
```

For a full spatial field or trajectory, use `FunctionalConformal` with one independent
case on the leading case axis. The default maximum score produces a simultaneous band.
Masks exclude ragged padding.

## 5. Propagate uncertain coefficients

```python
def solve_forward(coefficient, forcing):
    return coefficient + forcing


samples = phx.uq.sample_joint(
    {
        "coefficient": phx.uq.LogNormal(-1.0, 0.2),
        "forcing": phx.uq.Normal(1.0, 0.1),
    },
    num_samples=128,
    key=jr.key(3),
)
input_prediction = phx.uq.propagate(
    lambda coefficient, forcing: solve_forward(coefficient, forcing),
    samples,
    batch_size=32,
)
```

The resulting source is `"input"`, distinct from ensemble epistemic variation. Keep
these axes separate when both are present.

### 5a. Use a local covariance when full draws are unnecessary

```python
local_center = {
    "coefficient": jnp.asarray(0.4),
    "forcing": jnp.asarray([0.8, 1.0, 1.2]),
}
local_covariance = phx.uq.FactorCovariance(
    {
        "coefficient": jnp.asarray([0.05, 0.00]),
        "forcing": jnp.asarray([[0.02, 0.02, 0.02], [0.01, -0.01, 0.00]]),
    }
)
local_prediction = phx.uq.propagate_linearized(
    lambda value: solve_forward(value["coefficient"], value["forcing"]),
    local_center,
    local_covariance,
)
local_variance = local_prediction.exact_variance(batch_size=1)
```

The leading axis of every factor leaf is the shared covariance rank. This path
uses JVP/VJP actions and never materializes a Jacobian. Validate it against the
joint-QMC workflow above at several shrinking covariance scales before relying
on a nonlinear model.



### 5b. Reuse fixed Gaussian evaluations with Bayesian quadrature

When the uncertain input is one normalized Gaussian variable and forward solves
have already been assigned to a fixed design, Bayesian quadrature provides a
kernel-conditioned expectation through the ordinary integration interface:

```python
coefficient = phx.domain.ProbabilityDomain(
    phx.uq.Normal(0.4, 0.05),
    label="coefficient",
)
target = phx.integration.expectation(
    coefficient,
    target_id="coefficient-expectation",
)
kernel_mean = phx.integration.GaussianKernelMean(
    target,
    phx.kernels.SquaredExponentialKernel(length_scale=0.08),
)
plan = phx.integration.BayesianQuadraturePlan(
    kernel_mean,
    phx.domain.PointSampling(24, design="hammersley"),
    observation_noise=0.0,
    solve_regularization=1e-10,
)
realization = phx.integration.materialize(target, plan)
prediction_mean = phx.integration.reduce(
    coefficient.Function("coefficient")(
        lambda value: solve_forward(value, forcing=1.0)
    ),
    realization,
)
```

The fixed realization can reduce scalar, array, field, or PyTree outputs without
rebuilding the kernel system. `prediction_mean.error_estimate` is the GP
posterior integral standard deviation, with
`error_kind="bayesian-posterior-standard-deviation"`. It is **not a
deterministic or frequentist error bound** and should not be combined with
ensemble, aleatoric, or conformal uncertainties as though they had the same
meaning.

Keep observation noise separate from numerical solve regularization. Inspect
`prediction_mean.diagnostics.solve` before using the result; singular designs,
non-finite forward outputs, target-identity mismatch, and invalid posterior
variance fail closed. This initial path does not perform active acquisition and
does not support unnormalized evidence, non-Gaussian measures, or arbitrary
kernel algebra.

### 5c. Fit a reusable nonintrusive polynomial expansion

For independent scalar Uniform and Normal inputs, project a finite orthonormal
expansion with an explicit product quadrature:

```python
coefficient_factor = phx.domain.ProbabilityDomain(
    phx.uq.Uniform(0.2, 0.6), label="coefficient"
)
forcing_factor = phx.domain.ProbabilityDomain(
    phx.uq.Normal(1.0, 0.1), label="forcing"
)
pce_basis = phx.uq.PolynomialChaosBasis(
    (coefficient_factor, forcing_factor), 3
)
pce_quadrature = phx.integration.ProductIntegrationPlan(
    {
        "coefficient": phx.integration.FixedQuadraturePlan(
            phx.integration.GaussLegendreRule(5)
        ),
        "forcing": phx.integration.FixedQuadraturePlan(
            phx.integration.GaussHermiteRule(5)
        ),
    }
)
pce_fit = phx.uq.PolynomialChaosProjectionPlan(
    pce_basis, pce_quadrature
).fit(solve_forward)

pce_prediction = pce_fit.expansion(
    {"coefficient": jnp.asarray(0.4), "forcing": jnp.asarray(1.1)}
)
pce_mean = pce_fit.expansion.mean
pce_variance = pce_fit.expansion.variance
pce_first_order = pce_fit.expansion.first_order_sobol
pce_total_order = pce_fit.expansion.total_order_sobol
```

Use `PolynomialChaosRegressionPlan` instead when an existing finite design owns the
model values. Do not call that regression fit Galerkin: no stochastic residual
equations are assembled. Increase degree only after measuring truncation error
against withheld points or a higher-order projection; feature and storage ceilings
stop combinatorial basis growth rather than truncating it.

## 6. Rank global effects

```python
sensitivity = phx.uq.sobol_indices(
    solve_forward,
    samples.distributions,
    num_samples=256,
    key=jr.key(4),
    batch_size=32,
)
```

`first_order` measures each input's main effect. `total_order` also includes all
interactions involving that input. Validate the estimator on a reference problem such
as Ishigami before using a new expensive forward workflow.

### 6a. Transform singular Gaussian moments explicitly

Represent covariance by its factor directions. Here the two-dimensional input has
rank one; the transform uses that rank and does not invent a second direction:

```python
nonlinear_input_factor = phx.uq.GaussianFactor(
    jnp.asarray([[0.2], [0.0]]),
    factor_id="rank-one-coefficient",
)
nonlinear_moments = phx.uq.spherical_radial_cubature(
    lambda value: jnp.asarray([value[0] ** 2, jnp.sin(value[0] + value[1])]),
    jnp.asarray([0.5, -0.1]),
    nonlinear_input_factor,
    regularization=1e-8,
)

assert nonlinear_input_factor.numerical_rank == 1
assert nonlinear_moments.method_id == "spherical-radial-cubature"
assert nonlinear_moments.regularization == 1e-8
```

The returned factor, `Cov[input, output]`, status, validity, method ID, and point count
make the approximation auditable. The requested output regularization is the only
added diagonal term. Invalid factors or nonfinite evaluations are reported; they do
not trigger covariance clipping, jitter, or a different transform. Use the guarded
unscented or Gauss--Hermite alternatives only when their dense-dimension or tensor
point limits fit the problem.

### 6b. Apply local curvature in one direction

Use a Gauss--Newton action when an optimizer or diagnostic needs `JᵀJ v`, not a dense
Jacobian or Hessian:

```python
local_parameters = jnp.asarray([1.0])
local_direction = jnp.asarray([1.0])
local_target = jnp.asarray([0.8, 2.1, 3.2])


def local_residual(value):
    return value[0] * jnp.asarray([1.0, 2.0, 3.0]) - local_target


local_curvature = phx.uq.gauss_newton_action(
    local_residual,
    local_parameters,
    local_direction,
    regularization=1e-3,
)

assert local_curvature.method_id == "jax_jvp_vjp"
assert local_curvature.regularization == 1e-3
```

The action uses one JVP and one transpose-VJP and records validity, status, operator,
method, approximation, and explicit regularization. Empirical observability and
controllability directions subsequently materialize a guarded dense action matrix;
experiment-design objectives likewise materialize action callables up to their
declared `max_dimension`. Neither API silently changes backend.

## 7. Infer a physical parameter with global/local MAP, NUTS, and Laplace

Define a normalized observation likelihood and a prior explicitly. Do not pass
`FunctionalSolver.loss()` as a posterior log density.

```python
sensor_x = jnp.linspace(0.05, 0.95, 24)
sensor_basis = 0.5 * sensor_x * (1.0 - sensor_x)
observations = 4.0 * sensor_basis
observation_likelihood = phx.uq.GaussianLikelihood(0.02)

posterior_variance = 1.0 / (1.0 / 3.0**2 + jnp.vdot(sensor_basis, sensor_basis) / 0.02**2)
posterior_mean = posterior_variance * jnp.vdot(sensor_basis, observations) / 0.02**2

parameter_space = phx.uq.ParameterSpace(
    {"source": posterior_mean},
    priors={"source": phx.uq.Normal(0.0, 3.0)},
)
posterior_problem = phx.uq.PosteriorProblem(
    parameter_space,
    lambda parameters: jnp.sum(
        observation_likelihood.log_prob(
            parameters["source"] * sensor_basis,
            observations,
        )
    ),
    predict=lambda parameters, x: cx.Field(
        parameters["source"] * 0.5 * x * (1.0 - x),
        dims=("x",),
    ),
)

map_search = phx.optim.DifferentialEvolutionSearch(
    16,
    20,
    design=phx.sampling.SobolDesign(scrambled=True),
)
global_map = phx.uq.search_map(
    posterior_problem,
    map_search,
    key=jr.key(50),
    position_bounds=(
        {"source": -6.0},
        {"source": 6.0},
    ),
)
map_result = phx.uq.find_map(
    posterior_problem,
    global_map.position,
    gradient_tolerance=1e-7,
)

nuts = phx.uq.sample_nuts(
    posterior_problem,
    key=jr.key(5),
    num_chains=4,
    num_warmup=1000,
    num_samples=1000,
    target_acceptance_rate=0.9,
    chain_method="interleaved",
)
assert nuts.diagnostics.max_rhat < 1.01
assert nuts.diagnostics.divergence_count == 0

posterior_query = jnp.linspace(0.0, 1.0, 65)
nuts_prediction = nuts.predict(posterior_query, batch_size=128)

dense_laplace = phx.uq.fit_laplace(
    posterior_problem,
    map_result.position,
)
laplace_prediction = dense_laplace.predict(
    jr.key(6),
    posterior_query,
    num_samples=512,
)

structured_laplace = phx.uq.fit_laplace(
    posterior_problem,
    map_result.position,
    curvature="diagonal",
)
```
laplace_linearized = dense_laplace.linearized_predict(posterior_query)
laplace_local_variance = laplace_linearized.exact_variance()

The MCMC result preserves distinct chain and draw dimensions and includes split
rank-normalized $\hat R$, bulk/tail ESS, acceptance, divergence, energy, depth, and
warmup diagnostics. Dense Laplace is a local Gaussian approximation; agreement on
this conjugate problem is a correctness check, not evidence that every nonlinear
posterior is Gaussian.

`chain_method="interleaved"` is the explicit accelerator-oriented scheduling
choice: chains can cross draw boundaries independently when NUTS trajectory lengths
differ. Prefer it for many chains with expensive posterior gradients. This conjugate
example is intentionally small; `"sequential"` or `"vectorized"` can be faster for
similarly cheap targets.

Structured Laplace automatically whitens declared Gaussian priors; transformed
physical covariance is available through `physical_covariance_vector_product`.
For normalized residual models, declare `gauss_newton_residual` on
`PosteriorProblem` and request `likelihood_curvature="ggn"`.

For larger explicitly selected subspaces, use `ParameterSubspace.from_leaf_paths`
or `ParameterSubspace.last_layer`, then choose `curvature="diagonal"`,
`"lanczos"`, or `"lobpcg"`. Keep NUTS as the reference whenever the selected
dimension permits.

`last_layer(...)` means the globally final array leaves, not the final layer of
every branch. For `SeparableMLP` and other branched models, pass each final module
path to `ParameterSubspace.from_subtree_paths(...)`; this includes all arrays in
those modules and avoids accidentally selecting only the last coordinate factor.
See the posterior-contract section of the UQ guide for the complete separable
selection pattern and its nonlinear joint-posterior interpretation.


### 7a. Infer with uncertain predictors and responses

```python
measured_inputs = jnp.linspace(0.2, 1.4, 20)[:, None]
measured_targets = 1.7 * measured_inputs[:, 0]

measurement_term = phx.uq.LinearizedGaussianMeasurementLikelihood(
    lambda parameters, value: parameters["slope"] * value[0],
    measured_inputs,
    measured_targets,
    input_covariance=jnp.asarray([[0.04**2]]),
    observation_covariance=jnp.asarray([[0.02**2]]),
)
measurement_space = phx.uq.ParameterSpace(
    {"slope": jnp.asarray(1.5)},
    priors={"slope": phx.uq.Normal(0.0, 3.0)},
)
measurement_problem = phx.uq.PosteriorProblem.from_terms(
    measurement_space,
    (measurement_term,),
)
measurement_map = phx.uq.find_map(measurement_problem)
```

The likelihood includes the parameter-dependent log determinant of the
effective covariance. For a stochastic-gradient run, put `inputs`, `targets`,
and `case_indices` in one `ArrayMinibatchSource` and call
`measurement_term.log_prob_cases(...)` from the factor function. Compare the
result against an explicit latent-input model when the predictor uncertainty is
not small enough for a local linearization.


## 8. Scale a factorized likelihood with fixed-step SG-MCMC

Keep the normalized per-observation likelihood from the exact reference problem,
but expose each observation as one factor. The source owns deterministic epoch
ordering and padded-tail masking:

```python
source = phx.uq.ArrayMinibatchSource(
    {
        "basis": sensor_basis,
        "observation": observations,
    },
    batch_size=8,
    seed=31,
)


def likelihood_factors(parameters, batch):
    return observation_likelihood.log_prob(
        parameters["source"] * batch.data["basis"],
        batch.data["observation"],
    )


stochastic_problem = phx.uq.MinibatchPosteriorProblem(
    parameter_space,
    likelihood_factors,
    num_factors=sensor_basis.size,
    full_log_likelihood=posterior_problem.log_likelihood,
    predict=lambda parameters, x: cx.Field(
        parameters["source"] * 0.5 * x * (1.0 - x),
        dims=("x",),
    ),
)
inspection = phx.uq.diagnose_minibatch_posterior(
    stochastic_problem,
    source,
)
assert inspection.passed, inspection.as_dict()
```

Build a difference-estimator control variate at the exact MAP and run independent
chains. `num_burnin` discards states; it does not tune the fixed step:

```python
control = phx.uq.build_sgmcmc_control_variate(
    stochastic_problem,
    source,
    map_result.position,
)
sgld = phx.uq.sample_sgld(
    stochastic_problem,
    source,
    key=jr.key(30),
    step_size=1e-4,
    num_chains=2,
    num_burnin=1,
    num_samples=4,
    steps_per_sample=1,
    control_variate=control,
)
sgld_refined = phx.uq.sample_sgld(
    stochastic_problem,
    source,
    key=jr.key(31),
    step_size=5e-5,
    num_chains=2,
    num_burnin=1,
    num_samples=4,
    steps_per_sample=1,
    control_variate=control,
)
sgnht = phx.uq.sample_sgnht(
    stochastic_problem,
    source,
    key=jr.key(32),
    step_size=5e-4,
    diffusion=0.01,
    num_chains=2,
    num_burnin=1,
    num_samples=4,
    steps_per_sample=1,
    control_variate=control,
)
```

Do not infer accuracy from rank diagnostics alone. Compare the base and halved-step
posterior means and variances, then compare both with the NUTS and Laplace references
already computed above:

```python
sgld_mean = jnp.mean(sgld.samples["source"])
refined_mean = jnp.mean(sgld_refined.samples["source"])
step_sensitivity = jnp.abs(sgld_mean - refined_mean)
reference_error = jnp.abs(refined_mean - jnp.mean(nuts.samples["source"]))

mixing = sgld_refined.mixing_report(
    max_rhat=1.05,
    min_bulk_ess=200,
    min_tail_ess=200,
)
```

The counts above are executable smoke settings, not an inference recommendation.
For production, increase burn-in and retained draws until independent chains pass
declared convergence thresholds. Report `step_sensitivity`, reference
moment/predictive discrepancies, stochastic-gradient variance, throughput, memory,
batch definition, and every fixed-step setting. SGLD and SGNHT are unadjusted
approximations: there is no Metropolis correction or automatic step-size adaptation.
Prefer exact NUTS or Laplace whenever full-data inference is feasible.

For neural operators, use `OperatorBatchObservationLikelihood` with
`OperatorMinibatchSource`; one complete physical case is one factor. Query points,
channels, masks, geometry, and quadrature remain coupled within the case. The
adapter intentionally does not subsample query anchors.

## 9. Choose local, represented-mode, or discovery inference

Pathfinder is a fast local approximation selected along an L-BFGS path:

```python
pathfinder = phx.uq.fit_pathfinder(
    posterior_problem,
    key=jr.key(8),
    num_samples=64,
)
pathfinder_prediction = pathfinder.predict(posterior_query)
```

Compare its moments and held-out scores against NUTS and dense Laplace. If a
multimodal posterior's important modes are already represented, use exact
flow-assisted NUTS and initialize chains across those modes:

```python
multimodal_problem = phx.uq.PosteriorProblem(
    phx.uq.ParameterSpace(
        jnp.asarray(0.0),
        priors=phx.uq.Normal(0.0, 4.0),
    ),
    lambda value: jnp.logaddexp(
        jnp.log(0.3) - 0.5 * ((value + 2.0) / 0.35) ** 2,
        jnp.log(0.7) - 0.5 * ((value - 2.0) / 0.35) ** 2,
    ),
)
flow_config = phx.uq.FlowNUTSConfig(
    num_adaptation_rounds=1,
    num_local_adaptation_steps=4,
    num_global_adaptation_steps=2,
    num_stabilization_steps=1,
    num_local_steps=1,
    num_global_steps=1,
    history_capacity_per_chain=4,
    history_thinning=1,
    flow_layers=1,
    num_knots=4,
    nn_width=8,
    nn_depth=1,
    max_epochs=2,
    max_patience=2,
    batch_size=2,
    validation_fraction=0.25,
)
flow_nuts = phx.uq.sample_flow_nuts(
    multimodal_problem,
    key=jr.key(9),
    num_chains=2,
    num_warmup=20,
    num_samples=8,
    initial_positions=jnp.asarray([-2.0, 2.0]),
    target_acceptance_rate=0.9,
    max_num_doublings=5,
    config=flow_config,
    chain_method="vectorized",
)
assert flow_nuts.log_density.shape == (2, 8)
assert flow_nuts.global_acceptance_rate.shape == (2, 8)
```

The counts above exercise the end-to-end contract; they are too small for convergence
or mode-occupancy claims.

The flow is trained only during adaptation. Every global transition uses the exact
asymmetric independence Metropolis--Hastings correction, and the frozen production
kernel alternates configured local NUTS and global flow steps. Inspect global
acceptance, proposal ESS, nonfinite counts, mode occupancy, and ordinary rank
diagnostics. This transports between represented modes; it does not certify that no
unrepresented mode exists.

Use `sample_tempered_smc` instead when low-dimensional mode discovery or a
log-evidence estimate is required. Its declared priors provide initial particles;
inspect the adaptive temperature schedule, ESS, divergences, and surviving
initial-particle count.

## 10. Represent omitted physics with a GP discrepancy

An ensemble cannot identify a physical mode that every member omits. Model that
failure explicitly as $y=u_\theta(x)+\delta(x)+\epsilon$. Keep observations separate
from the typed covariance/noise state:

```python
misspecified_observations = 4.0 * sensor_basis + 0.03 * jnp.sin(2.0 * jnp.pi * sensor_x)
discrepancy_model = phx.uq.ExactGaussianProcessDiscrepancy(
    sensor_x,
    misspecified_observations,
)
sparse_discrepancy = phx.uq.SparseGaussianProcessDiscrepancy.from_evenly_spaced_subset(
    sensor_x,
    misspecified_observations,
    num_inducing=8,
)
gp_state = phx.uq.GaussianProcessLikelihoodState(
    kernel=phx.kernels.AmplitudeKernel(
        phx.kernels.Matern32Kernel(length_scale=0.25),
        0.03,
    ),
    noise_scale=0.005,
)

conditioned_discrepancy = discrepancy_model.condition(
    4.0 * sensor_basis,
    posterior_query,
    state=gp_state,
    output_dim="x",
)
physical_mean = 4.0 * 0.5 * posterior_query * (1.0 - posterior_query)
discrepancy_prediction = conditioned_discrepancy.predictive_field(
    physical_mean,
    jr.key(7),
    num_samples=256,
    observation_variance=gp_state.noise_scale**2,
)
```

`log_marginal_likelihood` analytically integrates latent GP values. For fixed kernel
parameters, reuse `discrepancy_model.factor(state=gp_state)` inside every
physical-parameter evaluation. When amplitude, length scale, or noise are inferred,
build a `GaussianProcessLikelihoodState` from the current parameter PyTree:

```python
def inferred_gp_state(parameters):
    return phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.AmplitudeKernel(
            phx.kernels.Matern32Kernel(
                length_scale=parameters["length_scale"],
            ),
            parameters["amplitude"],
        ),
        noise_scale=parameters["noise_scale"],
    )


gp_term = phx.uq.GaussianProcessMarginalLikelihood(
    discrepancy_model,
    lambda parameters: parameters["source"] * sensor_basis,
    state=inferred_gp_state,
)
```

Put positive bijectors and informative priors on those leaves. Always compare against
a no-discrepancy model: the physical parameters and a flexible discrepancy may
otherwise explain the same observations.

### Use exact state space for supported temporal Matérn data

When the GP input is only scalar time and the covariance is Matérn-3/2 or
Matérn-5/2, the exact state-space path avoids dense observation-space storage:

```python
temporal_plan = phx.uq.compile_state_space_kernel(
    phx.kernels.ScaleKernel(
        phx.kernels.Matern52Kernel(length_scale=0.25),
        0.03**2,
    ),
    sensor_time,
    forecast_time,
    train_mask=sensor_available,
)
temporal_result = phx.uq.fit_state_space_gaussian_process(
    temporal_plan,
    sensor_residual,
    noise_scale=0.005,
)

latent_mean = temporal_result.posterior_mean
latent_variance = temporal_result.posterior_variance
future_observation_variance = temporal_result.predictive_variance
log_marginal_likelihood = temporal_result.log_marginal_likelihood
```

Times may be irregular and unsorted, and forecasts may extrapolate before or after
training. Supply a finite filler value wherever `sensor_available` is false; the
mask, not a large covariance, removes that value from the likelihood. Training times
must be unique. Repeated forecast times are allowed and restore repeated output
positions. Inspect `successful`, `status`, and `query_valid` before using the
marginals.

The kernel's `ScaleKernel.scale` is covariance variance, whereas `noise_scale` is
observation standard deviation. Kernel and schedule dtypes must agree. A transformed
fit recomputes the stationary prior, interval covariance, and dynamically earlier
initial time from the evaluated kernel. Concrete results distinguish prepared and
evaluated content IDs; traced results export the exact evaluated parameter arrays
when a host hash cannot be formed. Recompile when a new plan-level content identity
is required. The linear-storage result deliberately returns marginal variances rather
than a dense forecast covariance.
Use the dense scalar GP when a complete joint query
covariance, unsupported kernel algebra, multidimensional input, derivative
observation, or non-Gaussian likelihood is required.

The exact scalar GP is the correctness reference. Use explicit FITC only when dense
$O(n^3)$ conditioning is a measured bottleneck. Compare held-out scores and
`factor_storage_elements` against exact inference. The same kernel object can select
inducing points:

```python
selection = phx.uq.select_inducing_points(
    sensor_x[:, None],
    8,
    key=jr.key(8),
    kernel=gp_state.kernel,
)
sparse_factor = phx.uq.SparseGaussianProcessFactor(
    sensor_x,
    selection.points,
    state=gp_state,
)
```

Choose computation-aware inference instead of FITC when unresolved numerical
directions must remain in the posterior covariance. It consumes every observation
in full-batch kernel passes, but stores only action-projected geometry:

```python
computation_aware = phx.uq.ComputationAwareGaussianProcessDiscrepancy(
    sensor_x,
    misspecified_observations,
)
actions = phx.uq.BlockSparseGaussianProcessActionPolicy.from_random(
    jr.key(9),
    sensor_x.size,
    8,
)
factor = computation_aware.factor(state=gp_state, actions=actions)
residual = computation_aware.residual(4.0 * sensor_basis)
bound = factor.elbo(residual)
mean, variance = factor.latent_moments(residual, posterior_query)
```

The three paths answer different questions:

- exact or exact finite-feature inference is preferred whenever affordable;
- FITC changes the prior covariance through an inducing approximation;
- computation-aware inference preserves the prior and conditions on a budgeted
  subspace of linear observations.

The computation-aware objective is an ELBO, not exact evidence at reduced action
rank. Its covariance is conservative relative to the same exact GP at fixed
hyperparameters; that statement is not a calibration guarantee for a misspecified
kernel or actions learned from the same observations.

### 10a. Preserve missing correlated outputs

Do not impute absent channels or fit one independent GP per output. Encode the
observed point/channel rows and a PSD output covariance:

```python
output_names = ("velocity", "pressure")
observed_output_mask = jnp.ones((sensor_x.size, 2), dtype=bool)
observed_output_mask = observed_output_mask.at[::3, 1].set(False)
vector_physical_mean = jnp.stack((4.0 * sensor_basis, -2.0 * sensor_basis), axis=1)
vector_observations = vector_physical_mean + jnp.stack(
    (
        0.02 * jnp.sin(2.0 * jnp.pi * sensor_x),
        0.03 * jnp.cos(2.0 * jnp.pi * sensor_x),
    ),
    axis=1,
)
observation_design = phx.uq.MultiOutputDesign.from_dense(
    sensor_x,
    output_names=output_names,
    mask=observed_output_mask,
)
vector_gp = phx.uq.MultiOutputGaussianProcessDiscrepancy(
    observation_design,
    vector_observations,
)
coregionalization = phx.uq.Coregionalization(
    jnp.asarray([[0.8, 0.0], [-0.3, 0.6]]),
    jnp.asarray([0.1, 0.15]),
    output_names=output_names,
)
vector_state = phx.uq.MultiOutputGaussianProcessLikelihoodState(
    kernel=phx.uq.IntrinsicCoregionalizationKernel(
        phx.kernels.Matern52Kernel(length_scale=0.2),
        coregionalization,
    ),
    noise_scale=jnp.asarray([0.01, 0.02]),
)
query_design = phx.uq.MultiOutputDesign.from_dense(
    posterior_query,
    output_names=output_names,
)
vector_condition = vector_gp.condition(
    vector_physical_mean,
    query_design,
    state=vector_state,
)
vector_mean = vector_condition.dense_mean()
```

Use `LinearModelCoregionalizationKernel` when several latent spatial scales are
scientifically required. ICM and LMC both preserve cross-output covariance and
heterotopic row ordering.

### 10b. Condition on a differential operator

Represent values and PDE observations as blocks of one latent field. Dynamic
coefficients remain differentiable:

```python
value_points = jnp.linspace(0.05, 0.95, 8)[:, None]
interior_points = jnp.linspace(0.1, 0.9, 6)[:, None]
diffusion = jnp.asarray(0.2)
measured_values = jnp.sin(jnp.pi * value_points[:, 0])
measured_forcing = diffusion * jnp.pi**2 * jnp.sin(jnp.pi * interior_points[:, 0])
value_mean = jnp.zeros_like(measured_values)
forcing_mean = jnp.zeros_like(measured_forcing)

value = phx.uq.value_functional(1)
laplacian = phx.uq.laplacian_functional(1)


def functional_gp(diffusion):
    return phx.uq.FunctionalGaussianProcessDiscrepancy(
        (
            phx.uq.FunctionalObservationBlock(
                value_points,
                value,
                name="field-values",
            ),
            phx.uq.FunctionalObservationBlock(
                interior_points,
                -diffusion * laplacian,
                name="elliptic-operator",
            ),
        ),
        (measured_values, measured_forcing),
    )


functional_state = phx.uq.FunctionalGaussianProcessLikelihoodState(
    kernel=phx.kernels.SquaredExponentialKernel(length_scale=0.25),
    noise_scale=jnp.asarray([0.005, 0.02]),
)
operator_score = functional_gp(diffusion).log_marginal_likelihood(
    (value_mean, forcing_mean),
    state=functional_state,
)
```

The kernel's certified derivative order must support every functional. Add an
explicit value-functional `inducing_design` to `functional_state` only after exact
operator inference becomes the measured bottleneck.

Before release, run `discrepancy_identifiability_report` on repeated
no-discrepancy, fixed-GP, and jointly inferred-GP results. A passing report requires
reduced physical-parameter bias, improved held-out NLL and CRPS, adequate coverage,
and bounded physical/GP posterior correlation.
