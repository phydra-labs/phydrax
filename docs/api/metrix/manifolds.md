# Array manifolds

`phydrax.metrix` supplies a small optimization-oriented manifold contract for
array-valued parameters. It is deliberately separate from the other geometry
contracts:

- `RiemannianMetric` is a coordinate metric field used by intrinsic differential
  operators.
- `AbstractStateGeometry` advances differential-equation states from local tangent
  coordinates and exposes pullbacks needed by geometric integrators.
- `AbstractRiemannianManifold` converts ambient autodiff cotangents to metric gradients,
  retracts ambient tangent steps, and transports optimizer state.

SO(n) and SPD(n) parameter manifolds delegate their retractions to the existing state
geometries. This shares the tested numerical kernels without merging the two APIs.

## Contract

A manifold's `point_shape` describes one point's trailing dimensions. Leading axes are
interpreted as a product of independent manifold points. Every operation preserves the
full ambient array shape, while `contains`, `constraint_residual`, `inner`, and `norm`
reduce the complete product to one scalar.

For an ambient autodiff cotangent `G`, `egrad_to_rgrad(x, G)` returns the tangent vector
`grad` satisfying

`g_x(grad, v) = Re <G, v>_F`

for every tangent `v`. This operation is not generally tangent projection. In
particular, affine-invariant SPD geometry uses `project_tangent(X, G) = sym(G)` but
`egrad_to_rgrad(X, G) = X sym(G) X`.

Retractions consume ambient tangents. This differs from `AbstractStateGeometry.retract`,
which consumes the state geometry's local coordinates. Transport metadata states what
is actually implemented: projection transport is not advertised as parallel or
isometric transport.

## Built-in manifolds

| Manifold | Metric | Retraction | Transport |
|---|---|---|---|
| `EuclideanManifold` | Euclidean/Hermitian product | Addition | Identity; exact and parallel |
| `SphereManifold` | Induced Euclidean | Normalize `x + v` | Destination tangent projection |
| `StiefelManifold` | Induced Frobenius | Reduced QR with deterministic signs | Destination tangent projection |
| `GrassmannManifold` | Induced quotient metric | Reduced QR representative | Destination horizontal projection |
| `ObliqueManifold` | Product of induced sphere metrics | Normalize each column | Destination tangent projection |
| `FixedRankManifold` | Embedded Frobenius | Truncated SVD | Destination tangent projection |
| `SpecialOrthogonalManifold` | Induced Frobenius | Existing exponential or Cayley SO state retraction | Destination tangent projection |
| `AffineInvariantSPDManifold` | `tr(X^-1 U X^-1 V)` | Existing congruence/exponential SPD state retraction | Exact affine geodesic parallel transport |

Stiefel, Grassmann, oblique, and fixed-rank manifolds currently support real matrices.
Grassmann points are orthonormal representatives; objectives must respect the
equivalence `X ~ XQ` for an orthogonal matrix `Q`. `FixedRankManifold` stores the
matrix directly, avoiding factor-gauge ambiguity. The SPD implementation uses linear
solves and matrix factorizations, never an explicit inverse or an `n^2` by `n^2`
dense metric.

## Choosing parameterization or manifold optimization

Use an unconstrained parameterization when feasibility is the only requirement and a
simple global map exists. Examples include a normalized quaternion for SO(3), a
sigmoid for an interval, or a Cholesky factor for an SPD matrix.

Use manifold optimization when the geometry or quotient structure is part of the
problem: Grassmann subspaces, orthonormal frames, affine-invariant covariance
objectives, or rotation-valued parameters whose updates and optimizer state must remain
tangent. Phydrax never infers a manifold from an array shape; binding parameters to
manifolds is explicit through [`ParameterGeometry`](../optim.md#parameter-pytree-binding).

::: phydrax.metrix.AbstractRiemannianManifold
    options:
        members:
            - contains
            - constraint_residual
            - project_tangent
            - egrad_to_rgrad
            - inner
            - norm
            - retract
            - transport

---

::: phydrax.metrix.EuclideanManifold

---

::: phydrax.metrix.SphereManifold

---

::: phydrax.metrix.StiefelManifold

---

::: phydrax.metrix.GrassmannManifold

---

::: phydrax.metrix.ObliqueManifold

---

::: phydrax.metrix.FixedRankManifold

---

::: phydrax.metrix.SpecialOrthogonalManifold

---

::: phydrax.metrix.AffineInvariantSPDManifold

---

::: phydrax.metrix.HyperboloidManifold

---

::: phydrax.metrix.PoincareBallManifold

---

::: phydrax.metrix.ProbabilitySimplexManifold

---

::: phydrax.metrix.validate_manifold
