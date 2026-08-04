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
