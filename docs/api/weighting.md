# Moment weighting

`phydrax.weighting` constructs normalized finite measures whose declared feature
expectations match external targets. It is the target-aware complement to
`phydrax.coresets`: calibration changes weights on an existing support; coreset
methods subsequently reduce that support.

For source points with prior probabilities `p`, feature matrix `F` with shape
`(source_points, moment_count)`, and target moments `t`, exact calibration solves

```text
minimize    sum_i w_i log(w_i / p_i)
subject to  w >= 0, sum(w) = 1, F.T @ w = t
```

`ExactMoments` certifies success only for a finite, regular relative-interior
solution. An affine-inconsistent or boundary-only target returns a typed failure
status; the solver never clips the target or reports a finite exponential tilt as
an exact boundary solution.

`QuadraticMoments(t, covariance=C)` instead solves the covariant soft problem

```text
dual penalty  0.5 * lambda.T @ C @ lambda
```

where `C` is a dense or PhydraX linear operator with constructive or verified
self-adjoint positive-semidefinite evidence. Diagonal uncertainty is expressed as
`covariance=diag(scale**2)`; there is no parallel `scale` keyword. Full and singular
PSD covariance operators retain cross-moment coupling without silently adding jitter.

`IntervalMoments`, `GroupMassConstraints`, `EqualWeightSubset`, and
`BoundaryFacePolicy` declare hard intervals, sparse group masses, fixed-cardinality
equal subsets, and certified finite faces. Their topology is fixed at preparation.
`MomentCalibrationExecutionPolicy` makes the efficient regular dual, canonical-conic,
or mixed-integer route explicit; incompatible structures fail rather than switching.
The canonical-conic route represents relative entropy with exponential-cone
epigraphs and appends interval/group rows without changing measure semantics.
The mixed-integer route lowers `EqualWeightSubset` to the shared bounded
`MixedIntegerProgram`. With `BoundaryFacePolicy`, bounded maximum-mass LPs certify
every forced-zero coordinate; `BoundaryFaceEvidence` stores those audited relaxations
and an averaged relative-interior witness. `implicit_calibrate_fixed_face` exposes
derivatives only after that face is fixed and certified.

```python
import jax.numpy as jnp
import phydrax as phx

points = jnp.linspace(-1.0, 1.0, 101)
features = jnp.stack((points, points**2), axis=1)
problem = phx.weighting.MomentCalibrationProblem(
    features,
    phx.weighting.ExactMoments(jnp.array([0.2, 0.45])),
)
result = phx.weighting.require_converged(
    phx.weighting.calibrate_moments(problem)
)
weights = result.weights
```

A dense feature array is interpreted as `(source_points, moment_count)` and lowered
to the operator `F.T`. `SparseLinearMap` and compatible one-dimensional
`AbstractLinearOperator` implementations are accepted directly. Geometry is formed
in moment space; neither path materializes a source-by-source matrix. Masks and
negative-infinite prior log weights retain exactly zero mass.

The solver removes constant and linearly redundant feature directions through an
audited covariance eigenspectrum. Results retain affine rank and consistency,
physical and scaled residuals, covariance regularity, optimizer evidence, KL
divergence, effective sample size, active support, weight extrema, and provenance.
Use `initial_dual=` to warm-start nearby targets.

`implicit_calibrate_moments` returns only normalized weights and differentiates the
regular stationarity equation rather than unrolling optimizer iterations. It raises
for affine-inconsistent, boundary, singular, non-finite, or unconverged exact
solutions. Gradients with respect to targets and finite prior logits are supported
on a fixed support and fixed numerical rank; masks, rank changes, and active-set
changes are not differentiable contracts.

For a materialized integration measure, use `phydrax.integration.calibrate` instead.
That adapter preserves physical mass, sample axes, masks, ancestry, support validity,
execution key, and ordered transformation provenance. See
[Integrals and measures](../guides_integrals.md#calibrate-a-reusable-finite-realization).

## Problem and target contracts

::: phydrax.weighting.MomentCalibrationProblem

---

::: phydrax.weighting.ExactMoments

---

::: phydrax.weighting.QuadraticMoments

---

::: phydrax.weighting.IntervalMoments

---

::: phydrax.weighting.GroupMassConstraints

---

::: phydrax.weighting.EqualWeightSubset

---

::: phydrax.weighting.BoundaryFacePolicy

---

::: phydrax.weighting.MomentCalibrationPolicy

## Solvers

::: phydrax.weighting.calibrate_moments

---

::: phydrax.weighting.implicit_calibrate_moments

---

::: phydrax.weighting.require_converged

## Results and evidence

::: phydrax.weighting.MomentCalibrationResult

---

::: phydrax.weighting.MomentCalibrationDiagnostics

---

::: phydrax.weighting.MomentCalibrationStatus

---

::: phydrax.weighting.MomentCalibrationProvenance

---

::: phydrax.weighting.moment_calibration_status_message

## Method provenance

The mathematical formulation follows Shane Barratt, Guillermo Angeris, and Stephen
Boyd, [“Optimal Representative Sample
Weighting”](https://stanford.edu/~boyd/papers/optimal_representative_sampling.html),
*Statistics and Computing* 31:19 (2021). The public
[`cvxgrp/rsw`](https://github.com/cvxgrp/rsw) implementation and Andrew Timm's
Apache-2.0 [`rswjax`](https://github.com/andytimm/rswjax) package motivated this
absorption study. The Phydrax implementation is independent: it does not import,
copy, or translate either package and instead uses Phydrax-native operator,
optimization, spectral, differentiation, diagnostics, and integration contracts.
