# Integration

Measure-aware deterministic, adaptive, and stochastic integration. See
[Integrals and measures](../guides_integrals.md) for target semantics, normalization,
method selection, and uncertainty contracts.

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

::: phydrax.integration.GaussLegendreRule

---

::: phydrax.integration.GaussKronrodRule

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
