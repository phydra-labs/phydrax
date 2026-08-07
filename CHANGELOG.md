# Changelog

## Unreleased

### Added

- Added `phydrax.optim.kfac(...)`, a Phydrax-native type-II generalized
  Gauss–Newton KFAC optimizer for pointwise `MLP` physics-informed models.
- Added derivative-aware first- and second-order residual curvature, including
  contracted Laplacians, hard-enforcement ansätze, coupled fields, and small exact
  blocks for inverse-problem parameters.
- Added per-constraint factor state, `expand` and `reduce` factorizations,
  matrix-free sums of Kronecker products, diagonal-preconditioned conjugate-gradient
  solves, scheduled factor refreshes, constraint subsampling, adaptive-collocation
  reuse, and frozen-batch Armijo search.
- Added KFAC diagnostics, TensorBoard metrics, a PINN benchmark campaign, API
  documentation, and end-to-end Poisson, heat, Burgers, coupled-field, and inverse
  problem coverage.

### Changed

- KFAC now exposes only its optimizer configuration through `phydrax.optim`; the
  `FunctionalSolver` adapter owns residual sampling, model-layout validation,
  derivative tracing, and training lifecycle behavior.
- The KFAC benchmark campaign now uses upstream `optax.lbfgs()` for its L-BFGS
  baseline.

### Removed

- Removed the unsupported custom `bfgs_sw`, `lbfgs_sw`, and `ssbroyden`
  transformations and the `phydrax.nn.optim` namespace. Use upstream Optax
  transformations, including `optax.lbfgs()`, through their native package.
