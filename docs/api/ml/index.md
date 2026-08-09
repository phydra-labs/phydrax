# Machine learning API

`phydrax.ml` is the immutable, array-native classical machine-learning subsystem.
The core API defines fitting, data, schema, status, sparse-storage, and derivative
contracts. Family namespaces contain concrete recipes, fitted models, diagnostics,
and family-specific inspection functions.

A public fitting configuration ends in `Recipe` or uses the familiar estimator
name (`PCA`, `KMeans`, `DecisionTreeRegressor`, and similar). Calling
`phydrax.ml.fit(...)` or `recipe.fit_batch(...)` returns a `FitResult`; it never
mutates the recipe. `FitResult.model` is solver-frozen. Use
`FitResult.as_trainable()` only when the fitted arrays should enter a later
optimization partition.

## Reference groups

- [Core contracts](core.md)
- [Preprocessing and composition](workflows.md)
- [Linear and probabilistic supervision](supervised.md)
- [Kernels and neighbors](kernels_neighbors.md)
- [Covariance, clustering, manifolds, and outliers](unsupervised.md)
- [Trees, ensembles, and selection](trees_ensembles_selection.md)
- [Metrics and inspection](metrics_inspection.md)
- [Artifacts, conversion, and export](interop.md)

The module references below are generated from each namespace's public exports.
Internal numerical helpers in `phydrax.ml._numerics` are implementation details;
their behavior is exercised through the public recipes and models.
