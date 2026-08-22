# Temporal and stochastic composition

Temporal output sampling, internal accepted steps, stochastic drivers, spatial-noise
rank, random-input quadrature, and ensemble realization are separate approximation
parts.

::: phydrax.discretization.TemporalMesh

---

::: phydrax.discretization.RealizedTemporalMesh

---

::: phydrax.stochastic.SpatialNoiseBasis

---

::: phydrax.stochastic.SpatialNoiseApproximation

---

::: phydrax.stochastic.StochasticLevelSpec

---

::: phydrax.stochastic.StochasticCouplingPlan

Adaptive DAE results expose a complete result bundle and the incoming bundle ID:

- `solution.source_discretization_bundle_id`
- `solution.discretization_bundle`
- `solution.discretization_bundle_id`
- `solution.temporal_mesh`

`SpatialNoiseBasis.field_space_id` identifies the exact finite field coordinates on
which its weighted modes act. Realization and coupling identity remain stochastic
contracts rather than field-space identity.
