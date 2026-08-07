# Uncertainty quantification

This recipe covers independently initialized ensembles, proper scoring, split
conformal calibration, uncertain-input propagation, explicit Bayesian physical
parameters, MAP/NUTS/flow-assisted NUTS/Laplace/Pathfinder inference, and
scalable Gaussian-process model discrepancy.

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
    return phx.nn.MLP(
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

## 7. Infer a physical parameter with NUTS and Laplace

Define a normalized observation likelihood and a prior explicitly. Do not pass
`FunctionalSolver.loss()` as a posterior log density.

```python
sensor_x = jnp.linspace(0.05, 0.95, 24)
sensor_basis = 0.5 * sensor_x * (1.0 - sensor_x)
observations = 4.0 * sensor_basis
observation_likelihood = phx.uq.GaussianLikelihood(0.02)

posterior_variance = 1.0 / (
    1.0 / 3.0**2 + jnp.vdot(sensor_basis, sensor_basis) / 0.02**2
)
posterior_mean = (
    posterior_variance
    * jnp.vdot(sensor_basis, observations)
    / 0.02**2
)

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

map_result = phx.uq.find_map(
    posterior_problem,
    gradient_tolerance=1e-7,
)

nuts = phx.uq.sample_nuts(
    posterior_problem,
    key=jr.key(5),
    num_chains=4,
    num_warmup=1000,
    num_samples=1000,
    target_acceptance_rate=0.9,
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

The MCMC result preserves distinct chain and draw dimensions and includes split
rank-normalized $\hat R$, bulk/tail ESS, acceptance, divergence, energy, depth, and
warmup diagnostics. Dense Laplace is a local Gaussian approximation; agreement on
this conjugate problem is a correctness check, not evidence that every nonlinear
posterior is Gaussian.

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

## 8. Choose local, represented-mode, or discovery inference

Pathfinder is a fast local approximation selected along an L-BFGS path:

```python
pathfinder = phx.uq.fit_pathfinder(
    posterior_problem,
    key=jr.key(8),
    num_samples=512,
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
    num_adaptation_rounds=3,
    num_local_adaptation_steps=80,
    num_global_adaptation_steps=20,
    num_local_steps=2,
    num_global_steps=1,
    history_capacity_per_chain=256,
    max_epochs=60,
)
flow_nuts = phx.uq.sample_flow_nuts(
    multimodal_problem,
    key=jr.key(9),
    num_chains=4,
    num_warmup=500,
    num_samples=1000,
    initial_positions=jnp.asarray([-2.0, -1.8, 1.8, 2.0]),
    target_acceptance_rate=0.9,
    config=flow_config,
    chain_method="vectorized",
)
assert flow_nuts.diagnostics.max_rhat < 1.05
assert jnp.mean(flow_nuts.global_acceptance_rate) > 0.0
```

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

## 9. Represent omitted physics with a GP discrepancy


An ensemble cannot identify a physical mode that every member omits. Model that
failure explicitly as $y=u_\theta(x)+\delta(x)+\epsilon$:

```python
misspecified_observations = (
    4.0 * sensor_basis + 0.03 * jnp.sin(2.0 * jnp.pi * sensor_x)
)
discrepancy_model = phx.uq.ExactGaussianProcessDiscrepancy(
    sensor_x,
    misspecified_observations,
    kernel="matern32",
)
sparse_discrepancy = phx.uq.SparseGaussianProcessDiscrepancy.from_evenly_spaced_subset(
    sensor_x,
    misspecified_observations,
    num_inducing=8,
    kernel="matern32",
)

conditioned_discrepancy = discrepancy_model.condition(
    4.0 * sensor_basis,
    posterior_query,
    amplitude=0.03,
    length_scale=0.25,
    noise_scale=0.005,
    output_dim="x",
)
physical_mean = 4.0 * 0.5 * posterior_query * (1.0 - posterior_query)
discrepancy_prediction = conditioned_discrepancy.predictive_field(
    physical_mean,
    jr.key(7),
    num_samples=256,
    observation_variance=0.005**2,
)
```

`log_marginal_likelihood` analytically integrates the latent GP values and can be
used inside `PosteriorProblem` to infer physical parameters, amplitude, length scale,
and noise scale jointly. Use positive bijectors and informative priors. Always compare
against a no-discrepancy model: physical parameters and a flexible discrepancy can
otherwise explain the same observations.

The exact scalar GP is the correctness reference. Use the explicit FITC
`SparseGaussianProcessDiscrepancy` only when dense $O(n^3)$ conditioning is a
measured bottleneck; compare its held-out scores and `factor_storage_elements`
against the exact model. For vector fields, `MultiOutputGaussianProcessDiscrepancy`
requires a positive-definite output covariance and produces correlated output draws.
It never assumes output independence.

Before releasing a jointly inferred physical/discrepancy model, call
`discrepancy_identifiability_report` with repeated no-discrepancy, fixed-GP, and
joint-GP results. A passing report requires reduced physical-parameter bias,
improved held-out NLL and CRPS, adequate coverage, and bounded physical/GP posterior
correlation.
