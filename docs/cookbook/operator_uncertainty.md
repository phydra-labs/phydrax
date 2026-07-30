# Neural-operator uncertainty quantification

This recipe keeps uncertainty attached to the same geometry-aware contract used by
neural operators. A predictive result retains physical case axes, tensor-grid or
point-cloud query geometry, quadrature, masks, and output channels. Stochastic axes
remain separately labeled as epistemic, input, observation, process, or numerical
uncertainty.

## 1. Build a source/query batch and ensemble

The compact ensemble below demonstrates evaluation. In a real workflow, train every
member independently on the training partition before stacking members. Different
initializations without independent fitting are not a deep ensemble.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

x = jnp.linspace(0.0, 1.0, 8, endpoint=False)
axis = phx.nn.OperatorAxis(
    "x",
    x,
    quadrature_weights=jnp.ones_like(x) / x.size,
    periodic=True,
)
case = jnp.arange(12, dtype=float)[:, None]
source_values = jnp.sin(2.0 * jnp.pi * x[None, :] + 0.13 * case)
source = phx.nn.FunctionSamples(values=source_values, axes=(axis,))
query = phx.nn.FunctionSamples(values=None, axes=(axis,))
full_batch = phx.nn.OperatorBatch(
    inputs={"state": source},
    queries={"query": query},
    case_axes=("case",),
)
calibration_batch = full_batch.take(jnp.arange(6))
test_batch = full_batch.take(jnp.arange(6, 12))
calibration_target = 0.8 * source_values[:6]
test_target = 0.8 * source_values[6:]

members = tuple(
    phx.nn.FNO(n_modes=(3,), width=4, depth=1, key=key)
    for key in jr.split(jr.key(0), 3)
)
ensemble = phx.uq.HomogeneousFunctionEnsemble.from_members(
    members,
    source_dim="ensemble_member",
)
calibration_prediction = ensemble.predict_operator(
    calibration_batch,
    key=jr.key(1),
    field_name="output",
    query_name="query",
)
test_prediction = ensemble.predict_operator(
    test_batch,
    key=jr.key(2),
    field_name="output",
    query_name="query",
)

assert test_prediction.predictive.samples.dims == (
    "ensemble_member",
    "case",
    "x",
)
assert test_prediction.mean().field("output").values.shape == (6, 8)
```

`HeterogeneousFunctionEnsemble.predict_operator` provides the same result contract
when architectures differ. It evaluates members in a Python loop and rejects output
specification, case-axis, shape, or query-geometry mismatches rather than silently
aligning unrelated fields.

## 2. Calibrate whole functions and report proper scores

Split by physical source-function case before training. Calibration rows must be
disjoint from training and test rows. `OperatorFunctionalConformal` treats one
complete field as one exchangeable calibration event. The default maximum score
therefore produces a simultaneous band, not independent pointwise intervals.

```python
calibrator = phx.uq.OperatorFunctionalConformal.calibrate(
    calibration_prediction.mean(),
    calibration_target,
    alpha=0.2,
    field_name="output",
)
interval = calibrator.interval(test_prediction.mean())

crps = phx.uq.operator_ensemble_crps(test_prediction, test_target)
energy = phx.uq.operator_energy_score(test_prediction, test_target)
pointwise_coverage = phx.uq.operator_interval_coverage(
    interval,
    test_target,
    mode="pointwise",
    field_name="output",
)
simultaneous_coverage = phx.uq.operator_interval_coverage(
    interval,
    test_target,
    mode="simultaneous",
    field_name="output",
)
width = phx.uq.operator_interval_width(interval, field_name="output")

assert jnp.all(jnp.isfinite(jnp.stack((crps, energy, width))))
assert 0.0 <= pointwise_coverage <= 1.0
assert 0.0 <= simultaneous_coverage <= 1.0
assert interval.simultaneous and interval.calibrated
```

Marginal CRPS scores each valid output component. The energy score treats each
quadrature-scaled output field as one multivariate event and detects some joint-field
failures that marginal CRPS cannot. Always report which score was used. Query masks
exclude padded locations, and quadrature weights prevent a denser region from
silently receiving more physical measure.

Set `score="l2"` during calibration for a quadrature-weighted functional norm ball.
That object has a calibrated radius but no pointwise interval representation; call
`radius(...)`, not `interval(...)`.

Finite-sample split-conformal coverage assumes exchangeable calibration and future
physical cases. It does not guarantee coverage after a resolution, geometry,
source-distribution, sensor, or rollout-horizon shift. Evaluate those shifts
separately.

## 3. Draw coherent MC-dropout functions

One root key identifies one complete output-function draw. FNO feature masks are
channel-wise and shared across the spatial grid; DeepONet branch and trunk networks
receive folded keys. Chunking changes memory use, not the sample/key mapping.

```python
stochastic_model = phx.nn.FNO(
    width=4,
    depth=1,
    n_modes=(3,),
    dropout=0.2,
    key=jr.key(3),
)
dropout_prediction = phx.uq.sample_operator_predictive(
    stochastic_model,
    test_batch,
    num_samples=4,
    key=jr.key(4),
    sample_batch_size=2,
    field_name="output",
    query_name="query",
)
assert dropout_prediction.predictive.samples.data.shape == (4, 6, 8)
```

Use `phx.nn.inference_mode(stochastic_model)` for deterministic deployment. MC
dropout yields a useful epistemic proxy only when dropout training and calibration
match its use; repeated stochastic evaluations alone do not make a Bayesian
posterior.

## 4. Keep uncertain inputs separate from epistemic members

Represent sampled source functions as named physical case axes, evaluate on one
common output query geometry, and then reclassify only those axes as input
uncertainty. Source values, source coordinates, masks, and quadrature may vary across
input draws. The output query must be shared because pointwise statistics at
unmatched coordinates are undefined.

```python
offsets = jnp.asarray([-0.1, 0.0, 0.1])[:, None, None]
uncertain_source = phx.nn.FunctionSamples(
    values=test_batch.input("state").values[None, ...] + offsets,
    axes=(axis,),
)
uncertain_batch = phx.nn.OperatorBatch(
    inputs={"state": uncertain_source},
    queries={"query": query},
    case_axes=("forcing_draw", "case"),
)
input_prediction = members[0].predict(uncertain_batch)
input_predictive = phx.uq.operator_input_predictive(
    input_prediction,
    input_sample_axes=("forcing_draw",),
    field_name="output",
)
crossed_prediction = ensemble.predict_operator(
    uncertain_batch,
    key=jr.key(5),
    input_sample_axes=("forcing_draw",),
    field_name="output",
    query_name="query",
)

assert input_predictive.predictive.sample_axes[0].source == "input"
assert crossed_prediction.predictive.samples.data.shape == (3, 3, 6, 8)
assert set(crossed_prediction.decompose_variance()) == {
    "epistemic",
    "input",
    "total",
}
```

The crossed ensemble/input design preserves conditional epistemic spread for each
input draw and conditional input spread for each member. `total_variance()` collapses
all declared sources only when that aggregate is actually wanted.

## 4a. Separate geometry uncertainty from weight uncertainty

For `GINO`, `RIGNO`, and `GAOT`, a mesh is part of the operator input, not an
anonymous batch layout. A sampled geometry realization must therefore carry its
source coordinates, values, quadrature weights, and mask together under an input
sample axis. Reusing values with independently shuffled coordinates is a
different physical input, not a valid uncertainty draw.

Output statistics also need a common support. Query coordinates may differ
between physical cases, but they must agree across epistemic and input draws of
the same case before computing pointwise means, quantiles, CRPS, or coverage.
If every draw has its own output mesh, interpolate to one declared reference
query and propagate the interpolation choice as part of the analysis. Do not
silently compare array positions.

Use these architecture-specific defaults:

| Architecture | First epistemic baseline | Optional stochastic path | Small posterior subspace |
| --- | --- | --- | --- |
| `GINO` | Independently trained deep ensemble | None by default | Explicit FNO/output projection or graph decoder |
| `RIGNO` | Independently trained deep ensemble | `processor_edge_dropout`, trained and sampled coherently | Graph decoder/readout |
| `GAOT` | Independently trained deep ensemble | Attention and feed-forward dropout, trained and sampled coherently | Transformer output projection or graph decoder |

One PRNG key passed to `sample_operator_predictive` identifies one complete
field draw, including every graph or transformer dropout decision. Chunking may
change memory use but must not change key-to-draw identity. Call
`phx.nn.inference_mode` before deterministic evaluation and before defining a
posterior density.

An ensemble conditioned on one observed geometry estimates weight uncertainty,
not shape uncertainty. Cross independently sampled geometry/input axes with
ensemble-member axes when both sources matter, and inspect
`decompose_variance()` before collapsing them into total variance. Functional
conformal coverage remains an in-distribution statement over complete physical
cases; it does not become a geometry-extrapolation guarantee.

## 5. Define a normalized operator observation likelihood

An operator training loss is not automatically a posterior likelihood. A finite
Gaussian sensor model is sum-normalized over observed query/channel elements and
physical cases. It does not include quadrature unless a different stochastic process
model explicitly derives such weights.

```python
baseline = members[0].predict(test_batch)
posterior_target = 0.8 * baseline.field("output").values


def scaled_prediction(parameters):
    field = baseline.field("output")
    return phx.nn.OperatorPrediction.from_field(
        "output",
        parameters["scale"] * field.values,
        field.query_name,
        baseline.query_geometry(field.query_name),
        spec=field.spec,
        case_axes=baseline.case_axes,
        case_shape=baseline.case_shape,
    )


likelihood_term = phx.uq.FixedOperatorObservationLikelihood(
    scaled_prediction,
    test_batch,
    posterior_target,
    phx.uq.GaussianLikelihood(0.05),
    output_spec=members[0].operator_output_specs["output"],
    field_name="output",
    query_name="query",
)
parameter_space = phx.uq.ParameterSpace(
    {"scale": jnp.asarray(1.0)},
    priors={"scale": phx.uq.Normal(1.0, 0.5)},
)
posterior = phx.uq.PosteriorProblem.from_terms(
    parameter_space,
    (likelihood_term,),
    predict=lambda parameters: phx.uq.operator_prediction_field(
        scaled_prediction(parameters),
        field_name="output",
    ),
    gauss_newton_residual=lambda parameters: likelihood_term.standardized_residual(
        parameters
    ),
)
mode = phx.uq.find_map(posterior)
approximation = phx.uq.fit_laplace(posterior, mode.position)
generic_posterior_prediction = approximation.predict(
    jr.key(6),
    num_samples=8,
    sample_dim="posterior_draw",
)
operator_posterior_prediction = phx.uq.OperatorPredictiveField.from_predictive(
    generic_posterior_prediction,
    test_batch,
    members[0].operator_output_specs["output"],
    field_name="output",
    query_name="query",
)
assert operator_posterior_prediction.predictive.samples.data.shape == (8, 6, 8)
```

For neural-network weight uncertainty, use `ParameterSubspace` to name exact leaves
or subtrees and freeze the complement. Architecture-aware choices include the FNO
projection, selected spectral blocks, all DeepONet branch/trunk heads, a local
operator decoder, or a graph readout. `last_layer()` follows global PyTree order and
is unsafe as a shorthand for multi-branch models. Disable dropout before constructing
a posterior density.

Full-weight NUTS/HMC is appropriate only for genuinely small selected subspaces.
Use exact dense Laplace as a low-dimensional reference, structured Laplace for a
measured larger-subspace bottleneck, and deep ensembles as the robust first baseline.
None of these repairs model-form error shared by every member; use an explicit
physical discrepancy model when omitted physics is identifiable from data.

The first operator-UQ implementation requires real physical outputs. Represent a
complex field as named real/imaginary channels or as a real observable before
constructing `OperatorPredictiveField`.

## 6. Run the reproducible benchmark

The UQ benchmark trains a four-member FNO ensemble on periodic Burgers and a
four-member DeepONet ensemble on an independent source/query point-cloud problem. It
uses a disjoint calibration split, evaluates resolution and rollout shifts, fits a
small final-projection Laplace posterior, and records CRPS, energy score, pointwise
and simultaneous coverage, interval width, parameter count, timing, and metadata.
It validates the operator-aware UQ contract, not the empirical calibration of
GINO, RIGNO, or GAOT. Geometry-operator uncertainty results require a separate
deformed-domain benchmark with disjoint training, calibration, and test
populations before they support an architecture-specific claim.

```console
python -m tools.operator_benchmarks --uq --quick \
  --seeds 0,1,2,3 --steps 5000 --learning-rate 0.003 \
  --validation-interval 50 --patience 20 --minimum-delta 1e-7 \
  --output artifacts/operator-uq-reference
```

JSON and Parquet artifacts are separate from deterministic architecture-ranking
artifacts. The checked-in reference is
`tools/operator_benchmarks/reference/converged/operator_uq_benchmarks.{json,parquet}`.
Shifted coverage is diagnostic; only held-out in-distribution simultaneous coverage
is compared with nominal coverage through a binomial confidence interval.
