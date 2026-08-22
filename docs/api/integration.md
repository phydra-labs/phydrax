# Integration

Integration separates the mathematical measure from its numerical realization:

- a target declares support, measure, and normalization;
- a plan declares deterministic or stochastic discretization;
- `IntegrationEstimate` returns value, status, diagnostics, provenance, and only
  method-valid error evidence.

```python
import jax.numpy as jnp
import phydrax as phx

x = phx.domain.ScalarInterval(-1.0, 2.0, label="x")
square = x.Function("x")(lambda value: value**2)

estimate = phx.integration.integrate(
    square,
    phx.integration.over(x.component()),
    phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(24)),
)
```

Randomized plans require `key=`; deterministic plans reject it. Use
`materialize(...)` followed by repeated `reduce(...)` calls when multiple
integrands must share exactly the same nodes or random design.

See [Integrals and measures](../guides_integrals.md) for target semantics,
normalization, method selection, uncertainty contracts, external weighted
measures, and composed space/time/stochastic reductions.

## Workflow

::: phydrax.integration.integrate

---

::: phydrax.integration.materialize

---

::: phydrax.integration.reduce

---

::: phydrax.integration.from_samples

---

::: phydrax.integration.IntegrationRealization

---
::: phydrax.integration.calibrate

---


::: phydrax.integration.compress

---


## Measure calibration

`calibrate` reweights an already materialized finite positive measure to exact or
quadratically reconciled normalized feature expectations. It preserves physical
mass and sample metadata, requires a successful core calibration, and appends
ordered transformation evidence before downstream reduction.

::: phydrax.integration.MeasureCalibrationDiagnostics

---

::: phydrax.integration.MeasureTransformationRecord

---

::: phydrax.integration.TransformedIntegrationDiagnostics

---

## Measure compression

`weighted_mmd`, `kernel_herd`, randomized pivoted Cholesky, and
`select_inducing_points` preserve every trailing axis declared by
`kernel.input_ndim`. Point designs therefore remain `(point, coordinate)`,
while a signature-kernel empirical measure is `(path, knot, channel)`.
Blockwise reductions slice only the leading measure axis. They never flatten a
path or infer ragged lengths; canonicalize padded suffixes first with
`phydrax.stochastic.repeat_last_path_padding`.

```python
observed_time = jnp.linspace(0.0, 1.0, 5)
observed_path = jnp.stack((observed_time, observed_time**2), axis=-1)
observed_paths = jnp.stack((observed_path, -observed_path, 0.5 * observed_path))
simulated_time = jnp.linspace(0.0, 1.0, 6)
simulated_path = jnp.stack((simulated_time, simulated_time**2), axis=-1)
simulated_paths = jnp.stack(
    (simulated_path, -simulated_path, 0.5 * simulated_path, 1.5 * simulated_path)
)
path_kernel = phx.kernels.SignaturePDEKernel(
    phx.kernels.LinearKernel(),
    polynomial_order=5,
)
distance = phx.coresets.weighted_mmd(
    observed_paths,
    simulated_paths,
    kernel=path_kernel,
)
selection = phx.coresets.kernel_herd(
    simulated_paths,
    phx.coresets.KernelHerding(3, kernel=path_kernel),
)
```

::: phydrax.coresets.MomentRecombination

---

::: phydrax.coresets.moment_recombine

---

::: phydrax.coresets.KernelHerding

---

::: phydrax.coresets.kernel_herd

---

::: phydrax.coresets.weighted_mmd

---

::: phydrax.integration.MeasureCompressionDiagnostics

---

## Term integration sources

Scalar penalties make realization ownership explicit:

- `per_step(target, plan)` materializes a fresh realization when a term
  evaluation is prepared;
- `fixed(realization)` reuses one materialized realization;
- `caller(target)` requires the evaluation caller to supply a compatible
  realization;
- `adaptive(target, initial_plan, policy)` delegates collocation refresh to
  `FunctionalSolver`.

Within one `FunctionalSolver` update, preparation occurs once per active term.
The resulting realization is reused across gradient, line-search, population,
curvature, and term-diagnostic evaluations for that update.

Use `mean_over(condition.on)` for normalized pointwise residual means and
`over(condition.on)` for physical or counting-measure integrals.

::: phydrax.integration.per_step

---

::: phydrax.integration.fixed

---

::: phydrax.integration.caller

---

::: phydrax.integration.adaptive

## Targets

::: phydrax.integration.over

---

::: phydrax.integration.mean_over

---

::: phydrax.integration.expectation

---

::: phydrax.integration.density

---

::: phydrax.integration.normalized_density

---

::: phydrax.integration.mapped

---

::: phydrax.integration.discrete

---

::: phydrax.integration.weighted

---

::: phydrax.integration.ComponentTarget

---

::: phydrax.integration.ProbabilityTarget

---

::: phydrax.integration.DensityTarget

---

::: phydrax.integration.MappedTarget

---

::: phydrax.integration.DiscreteMeasureTarget

---

::: phydrax.integration.WeightedSampleTarget

## Plans

::: phydrax.integration.FixedQuadraturePlan

---

::: phydrax.integration.AdaptiveQuadraturePlan

---

::: phydrax.integration.AdaptiveTrianglePlan

---

::: phydrax.integration.MonteCarloPlan

---

::: phydrax.integration.StratifiedMonteCarloPlan

---

::: phydrax.integration.QuasiMonteCarloPlan

---

::: phydrax.integration.ImportanceSamplingPlan

---

::: phydrax.integration.SparseGridPlan

---

::: phydrax.integration.CellQuadraturePlan

---

::: phydrax.integration.ProductIntegrationPlan

## Coupled multilevel estimation

A `MultilevelTarget` supplies paired fine/coarse samples from one validated stochastic
hierarchy. `MultilevelMonteCarloPlan` allocates work from measured correction variance
and cost, in batches, while retaining attempted and valid counts independently. The
estimator is resumable: initialization, advancement, and finalization expose the same
state used by the one-shot `integrate` call. Checkpoints and portable result archives
are checksummed and reject hierarchy, sampler, observable, or plan mismatches.

::: phydrax.integration.MultilevelMonteCarloPlan

---

::: phydrax.integration.MultilevelSampleBatch

---

::: phydrax.integration.MultilevelEstimatorState

---

::: phydrax.integration.MultilevelDiagnostics

---

::: phydrax.integration.initialize_multilevel

---

::: phydrax.integration.advance_multilevel

---

::: phydrax.integration.finalize_multilevel

---

::: phydrax.integration.write_multilevel_checkpoint

---

::: phydrax.integration.read_multilevel_checkpoint

## Rare-event splitting

Adaptive multilevel splitting consumes a canonical stochastic path event and two
ordinary solver callbacks: one initial population sampler and one continuation
sampler. Quantile levels, killed/branched ancestry, path scores, probability factors,
and stopping status remain explicit. Replication reports between-run uncertainty; a
single adaptively branched population is not mislabeled as IID Monte Carlo.

::: phydrax.integration.AdaptiveMultilevelSplittingPlan

---

::: phydrax.integration.adaptive_multilevel_splitting

---

::: phydrax.integration.replicate_adaptive_multilevel_splitting

---

::: phydrax.integration.AdaptiveMultilevelSplittingResult

## Smolyak control hierarchies

`SmolyakSurrogateHierarchyAdapter` pairs a deterministic sparse-grid surrogate with
the expensive model using a prefix-stable input sampler. Level zero estimates the
surrogate expectation; level one estimates the paired fine-minus-surrogate correction.
The adapter emits ordinary `MultilevelSampleBatch` objects, so variance and cost
allocation use the same multilevel estimator rather than a second control-variate
implementation.

::: phydrax.integration.SmolyakSurrogateHierarchyAdapter

---

::: phydrax.integration.smolyak_surrogate_expectation

## Sampling designs and estimators

::: phydrax.integration.IIDDesign

---

::: phydrax.integration.LatinHypercubeDesign

---

::: phydrax.integration.AntitheticDesign

---

::: phydrax.integration.RandomizedQMCDesign

---

::: phydrax.integration.StratifiedDesign

---

::: phydrax.integration.ControlVariateEstimator

---

::: phydrax.integration.SampleMeanEstimator

---

::: phydrax.integration.SelfNormalizedEstimator

## Rules

::: phydrax.integration.CubatureRule

---

::: phydrax.integration.GaussianCubatureRule

---

::: phydrax.integration.GaussLegendreRule

---

::: phydrax.integration.GaussKronrodRule

---

::: phydrax.integration.GaussHermiteRule

---

::: phydrax.integration.ClenshawCurtisRule

---

::: phydrax.integration.TanhSinhRule

---

::: phydrax.integration.ReferenceIntervalRule

---

::: phydrax.integration.ReferenceQuadrilateralRule

---

::: phydrax.integration.ReferenceTriangleRule

---

::: phydrax.integration.ReferenceHexahedronRule

---

::: phydrax.integration.ReferenceTetrahedronRule

---

::: phydrax.integration.interval_rule_data

---

::: phydrax.integration.reference_rule_data

## Estimates, status, and provenance

::: phydrax.integration.IntegrationEstimate

---

::: phydrax.integration.IntegrationStatus

---

::: phydrax.integration.status_message

---

::: phydrax.integration.IntegrationProvenance

## Diagnostics

::: phydrax.integration.FixedQuadratureDiagnostics

---

::: phydrax.integration.AdaptiveQuadratureDiagnostics

---

::: phydrax.integration.AdaptiveTriangleDiagnostics

---

::: phydrax.integration.AdaptiveTrianglePartition

---

::: phydrax.integration.MonteCarloDiagnostics

---

::: phydrax.integration.AntitheticDiagnostics

---

::: phydrax.integration.StratifiedDiagnostics

---

::: phydrax.integration.RandomizedQMCDiagnostics

---

::: phydrax.integration.WeightedSampleDiagnostics

---

::: phydrax.integration.SparseGridDiagnostics

---

::: phydrax.integration.MappedIntegrationDiagnostics

---

::: phydrax.integration.ProductIntegrationDiagnostics

## Realized batches

::: phydrax.integration.PointIntegrationBatch

---

::: phydrax.integration.SeparableIntegrationBatch

---

::: phydrax.integration.MappedIntegrationBatch

---

::: phydrax.integration.WeightedSampleBatch

---

::: phydrax.integration.ProductIntegrationRealization

---

::: phydrax.integration.SparseGridRealization
