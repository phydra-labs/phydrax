# Array state geometry

State geometry constrains array-valued differential-equation states without
changing the dynamics contract. Vector fields still return a state-shaped
ambient array. Geometric solvers project that array, express it in local
coordinates, and advance through a retraction.

`AbstractStateGeometry` is separate from coordinate-chart metrics: it describes
numerical state membership and local updates, not a second metric or domain
hierarchy. `LocalRetraction` binds a geometry to one validated base point and
records both a stable retraction ID and the resolved method.

`AbstractStateGeometry` is also separate from
[`AbstractRiemannianManifold`](manifolds.md). State retractions consume local
coordinates and expose differential pullbacks for geometric integration. Parameter
manifolds consume ambient tangents and supply a metric gradient plus optimizer-state
transport. The SO(n) and SPD(n) parameter manifolds delegate their retractions to the
state implementations documented here.

## Built-in geometries

- `EuclideanStateGeometry` is the identity/addition geometry. It is marked
  trivial, so ordinary Diffrax solvers remain supported.
- `EmbeddedStateGeometry` adapts explicit membership, tangent-projection, and
  retraction callables. Without an explicit inverse-retraction callable,
  `inverse_retract` and interpolation reject rather than substituting a
  projected chord. RKMK/SRKMK require both explicit inverse-retraction and
  pullback callables.
- `PointwiseStateGeometry` applies one geometry independently across leading
  point axes while preserving those axes and its wrapped capabilities.
- `SpecialOrthogonalStateGeometry` represents SO(n), using skew tangent
  coordinates and either an exponential or Cayley retraction. Its degree-63
  principal logarithm accepts Cayley spectral radius below 0.5. Cayley uses an
  analytic pullback; exponential solves each leading batch element
  independently with a differentiable matrix-free JVP solve, normalized
  right-hand sides, a checked relative residual, and fixed \(O(n^2)\) workspace
  per state. Neither depends on the logarithm. SO(n) also supplies
  the shared trivialization required by commutator-free tableaux.
- `SymmetricPositiveDefiniteStateGeometry` represents SPD(n), using symmetric
  tangent coordinates and a congruence/exponential retraction. It has exact
  local pullbacks but no shared group trivialization, so use RKMK rather than
  `CommutatorFreeSolver`.

Every geometry has a stable `geometry_id`, a `retraction_method`, and explicit
exact-pullback/shared-trivialization capability flags. Membership, projection,
retraction, dense interpolation, JIT compilation, and differentiation operate
on JAX arrays.

::: phydrax.metrix.AbstractStateGeometry
    options:
        members:
            - contains
            - project_tangent
            - to_local
            - from_local
            - retract
            - inverse_retract
            - pullback
            - local_retraction
            - interpolate

---

::: phydrax.metrix.LocalRetraction
    options:
        members:
            - __init__
            - __call__
            - evaluate
            - pullback

---

::: phydrax.metrix.EuclideanStateGeometry

---

::: phydrax.metrix.EmbeddedStateGeometry

---

::: phydrax.metrix.PointwiseStateGeometry

---

::: phydrax.metrix.SpecialOrthogonalStateGeometry

---

::: phydrax.metrix.SymmetricPositiveDefiniteStateGeometry
