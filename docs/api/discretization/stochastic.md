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

::: phydrax.stochastic.SpatialNoisePrecisionPolicy

---

::: phydrax.stochastic.SpatialNoiseApproximation

---

::: phydrax.stochastic.StochasticLevelSpec

---

::: phydrax.stochastic.StochasticCouplingPlan

---

::: phydrax.stochastic.NoiseCouplingWitness

---
Adaptive DAE results expose a complete result bundle and the incoming bundle ID:

- `solution.source_discretization_bundle_id`
- `solution.discretization_bundle`
- `solution.discretization_bundle_id`
- `solution.temporal_mesh`

`SpatialNoiseBasis.field_space_id` identifies the exact finite field coordinates on
which its weighted modes act. Realization and coupling identity remain stochastic
contracts rather than field-space identity.

`SpatialNoisePrecisionPolicy` separates covariance factorization, retained basis
storage, runtime diffusion, and orthogonality/residual certification. Every
factory accepts the policy; matrix-free Cholesky and Nyström construction use
construction precision, diagnostics use certification precision, and diffusion
casts operands before multiplication in runtime precision. A semidiscrete SPDE
retains a parent precision envelope with spatial-discretization and noise-basis
children.

Declaring adjacent levels as `noise_coupling="nested"` is not sufficient proof.
Both basis IDs and one shared noise-family ID are required, and the fine level
must carry a passing `NoiseCouplingWitness` with covariance and increment
projection residuals below its tolerance.
