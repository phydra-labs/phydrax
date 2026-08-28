# Discretization substrate

The discretization substrate binds continuum semantics to finite supports, field
spaces, measures, prepared numerical methods, transfers, and multi-part provenance.
It composes with `domain`, `geometry`, `integration`, `linalg`, `stochastic`,
`equations`, and `solver` without replacing their scientific contracts.

See [Guide → Discretization](../../guides_discretization.md) for lifecycle and method
examples, [Guide → Particle methods](../../guides_particle_methods.md) for material
entity and interaction contracts, and [Guide → SPH](../../guides_sph.md) for the
conservative barotropic workflow.

## Identity and lifecycle

::: phydrax.discretization.DiscretizationKey

---

::: phydrax.discretization.DiscretizationCapability

---

::: phydrax.discretization.AbstractDiscretizationPlan

---

::: phydrax.discretization.AbstractPreparedDiscretization

---

::: phydrax.discretization.PreparationReport

---

::: phydrax.discretization.DiscretizationBundle

---

::: phydrax.discretization.DiscretizationLevel

---

::: phydrax.discretization.DiscretizationHierarchy

---

::: phydrax.discretization.FieldTransfer

---
::: phydrax.discretization.AbstractRefinementTransfer

---


::: phydrax.discretization.TransferProperties
