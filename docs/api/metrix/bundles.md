# Vector-bundle and gauge connections

`VectorBundleConnection` represents local connection coefficients on a
fixed-rank trivialized real or complex vector bundle. Fiber and base dimensions
are static; topology and global transition cocycles remain atlas-level concerns.

::: phydrax.metrix.VectorBundleConnection

::: phydrax.metrix.bundle_covariant_derivative

::: phydrax.metrix.bundle_curvature

::: phydrax.metrix.gauge_transform_connection

::: phydrax.metrix.gauge_curvature_residual

The gauge convention is `A' = G⁻¹ A G + G⁻¹ dG`, corresponding to sections
transforming by `G⁻¹`. The curvature residual checks `F' = G⁻¹ F G` without
repairing either connection.

## Metric cochain assembly

`phydrax.exterior.metric_dual_hodges(bridge, metric, /, *, dual)` pairs
explicit primal/dual cell parameterizations and integrates their Riemannian
measures. Dual[k] has dimension n−k and count equal to primal degree k.
A dual-cell policy is never inferred from mesh shape; the result is a tuple of
metric-only diagonal Hodges, not another metric-assembly carrier.

::: phydrax.exterior.metric_dual_hodges
