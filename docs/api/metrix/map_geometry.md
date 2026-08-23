# Riemannian maps and immersions

`Immersion` declares a differentiable map whose source dimension does not exceed
its target dimension. Rank is a sampled, explicit validation claim; construction
does not silently repair a rank-deficient Jacobian.

::: phydrax.metrix.Immersion

::: phydrax.metrix.validate_immersion

::: phydrax.metrix.ImmersionValidationReport

`pullback_metric` accepts an immersion candidate. The resulting
`RiemannianMetric` is valid where the map differential has full column rank and
the target metric is positive definite.

::: phydrax.metrix.RiemannianMapGeometry

The map geometry exposes pullback metric, Dirichlet energy density, tension
field, isometry and conformality residuals, volume distortion, and intrinsic
second fundamental and mean-curvature data. Source and target chart identities
must match their declared metrics.

Labeled `DomainFunction` adapters are available as
`riemannian_map_energy`, `riemannian_map_tension`,
`riemannian_map_isometry_residual`, and
`riemannian_map_conformality_residual`.
