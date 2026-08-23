# Metric measures and boundaries

`WeightedRiemannianMeasure` represents `exp(log_weight) dvol_g`. The weight is
relative to Riemannian volume, not to coordinate Lebesgue measure. Its coordinate
density can be attached to an interior domain component with
`with_weighted_riemannian_measure`.

::: phydrax.metrix.WeightedRiemannianMeasure

It supplies weighted divergence, weighted Laplacian, reversible drift,
coordinate score, and the Bakry–Émery Ricci tensor under the documented
`exp(log_weight) dvol_g` convention.

## Density validation

::: phydrax.metrix.VolumeDensityValidationReport

::: phydrax.metrix.validate_volume_density

## Intrinsic hypersurfaces

`RiemannianHypersurface` starts from an explicitly oriented conormal. It
normalizes with the inverse metric and exposes the corresponding unit normal and
tangent projector. This differs from an ambient Euclidean boundary normal.

::: phydrax.metrix.RiemannianHypersurface

::: phydrax.metrix.induced_boundary_metric

::: phydrax.metrix.induced_boundary_density

`ProbabilityFlux` accepts `boundary_geometry=` to pair a metric probability
current with the matching intrinsic unit normal. If both `metric=` and
`boundary_geometry=` are supplied, they must share one metric object.
