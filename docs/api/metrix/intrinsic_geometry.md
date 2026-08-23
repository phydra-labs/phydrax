# Intrinsic endpoint geometry

`AbstractGeodesicManifold` is an optional strengthening of the optimization
manifold contract. It distinguishes exact exponential, logarithm, and squared
geodesic distance from a generic retraction.

::: phydrax.metrix.AbstractGeodesicManifold

Exact endpoint operations are supplied for Euclidean, sphere, Poincaré-ball,
hyperboloid, Fisher–Rao simplex, affine-invariant SPD, complex-projective,
unitary, special-unitary, and affine-invariant HPD geometry. Logarithms remain
local at cut loci and matrix logarithm branch boundaries.

## Coordinate metric integration

::: phydrax.metrix.MetricGeodesicResult

::: phydrax.metrix.integrate_metric_geodesic

The numerical service performs fixed-step RK4 integration of the existing
coordinate geodesic equations. It is a forward endpoint service, not a claim of
a globally minimizing logarithm.

## Intrinsic statistics

::: phydrax.metrix.frechet_objective

::: phydrax.metrix.frechet_mean

::: phydrax.metrix.FrechetMeanResult

The fixed Karcher iteration assumes all samples remain inside one convex normal
region.

## Consumers

`IntrinsicSquaredDistanceCost` plugs exact manifold distance into finite
transport. `RiemannianFlowMatchingMetric` evaluates velocity error with a
pointwise coordinate metric. A generic radial geodesic kernel is deliberately
not provided because positive definiteness is manifold- and profile-specific.
