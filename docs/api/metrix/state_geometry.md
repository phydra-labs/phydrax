# Array state geometry

State geometry constrains array-valued differential-equation states without
changing the dynamics contract. Vector fields still return a state-shaped
ambient array. Geometric solvers project that array, express it in local
coordinates, and advance through a retraction.

`AbstractStateGeometry` is separate from coordinate-chart metrics: it describes
numerical state membership and local updates, not a second metric or domain
hierarchy. `LocalRetraction` binds a geometry to one validated base point and
records both a stable retraction ID and the resolved method.

## Built-in geometries

- `EuclideanStateGeometry` is the identity/addition geometry. It is marked
  trivial, so ordinary Diffrax solvers remain supported.
- `EmbeddedStateGeometry` adapts explicit membership, tangent-projection, and
  retraction callables. Optional inverse-retraction and pullback callables refine
  higher-order local integration.
- `PointwiseStateGeometry` applies one geometry independently across leading
  point axes while preserving those axes.
- `SpecialOrthogonalStateGeometry` represents SO(n), using skew tangent
  coordinates and either an exponential or Cayley retraction.
- `SymmetricPositiveDefiniteStateGeometry` represents SPD(n), using symmetric
  tangent coordinates and a congruence/exponential retraction.

Every geometry has a stable `geometry_id` and a `retraction_method`. Membership,
projection, retraction, dense interpolation, JIT compilation, and differentiation
operate on JAX arrays.

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
