# LiDAR measurements

Phydrax distinguishes acquisition rays, measured ranges, derived Cartesian points, and reconstructed geometry.

```text
ray support + range returns → Cartesian point product → optional surface reconstruction
```

## Scan assets

`LidarScan` requires a length-valued scalar `MeasurementAsset` on `RaySampleSupport`. Optional intensity and return-index/count channels must share the support and acquisition identity.

A ray support retains normalized direction, origin, bounds, acquisition time, validity, and stable sample identity. Return intensity is detector response, not inferred reflectance.

## Cartesian products

`cartesianize_lidar_scan` evaluates each valid point as origin plus range times normalized direction in the declared spatial frame. No-return samples remain invalid and cannot become points at the sensor origin. Range uncertainty remains an attribute; missing angular uncertainty is not invented.

`LidarPointProduct` records parent scan identity, derivation, source manifests, stable points, and re-bound point attributes.

## Surface prediction

`LidarSurfacePlan` uses the exact dynamic triangle query to produce predicted ranges, hit points, normals, primitive/entity IDs, incidence cosine, uniqueness margins, and capacity evidence. Observed and predicted ranges can enter `MeasurementComparisonPlan` only when quantity, unit, support, and sampling identities agree.

## LAS boundary

`LasPointProvider` is optional and bounded. LAS stores derived Cartesian point products, not authoritative pulse origins or time-of-flight. The adapter therefore returns `LidarPointProduct` and an `AdapterReport` declaring unavailable raw beam semantics. A coordinate contract is required explicitly; missing CRS is never guessed.
