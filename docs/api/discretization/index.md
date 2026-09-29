# Discretization substrate

The discretization substrate binds continuum semantics to finite supports, field
spaces, measures, prepared numerical methods, transfers, and multi-part provenance.
It composes with `domain`, `geometry`, `integration`, `linalg`, `stochastic`,
`equations`, and `solver` without replacing their scientific contracts.

See [Guide → Discretization](../../guides_discretization.md) for lifecycle and
method examples, [Guide → Explicit polygon H1](../../guides_explicit_polygon_h1.md)
for condensed point-value polygon bases,
[Guide → Virtual elements](../../guides_virtual_elements.md) for polygonal
projection spaces, [Guide → Particle methods](../../guides_particle_methods.md)
for material entity and interaction contracts,
[Guide → Particle-grid splatting](../../guides_particle_splatting.md) for
measure-aware particle/grid transfer, [Guide → SPH](../../guides_sph.md)
for conservative particle flow,
[Guide → Deformable contact](../../guides_deformable_contact.md) for exact-map
collision surfaces and fixed-capacity candidate/safety contracts, and
[Guide → Kinetic methods](../../guides_lattice_boltzmann.md) for local
lattice-Boltzmann and discrete-velocity methods.

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

## Prepared field queries and side actions

`PreparedFieldReconstruction.prepare_query(...)` locates fixed points once and
returns a `PreparedFieldQuery` whose route is reused for every coefficient
state; its `coverage` is `"complete"` or `"masked"` (`FieldQueryCoverage`) and
its `approximation` is `"exact"`, `"h1-projection"`, or `"l2-projection"`
(`FieldApproximation`). Side actions bind an owner's coefficient space to trace data on
selected facets with separate primal, dual-pullback, load-injection, and
Hilbert-adjoint maps; conormal fluxes and imposition provenance are published
by compiled physics owners.

::: phydrax.discretization.PreparedFieldQuery

---

::: phydrax.discretization.FieldQueryCoverage

---

::: phydrax.discretization.FieldApproximation

---

::: phydrax.discretization.FacetTraceRule

---

::: phydrax.discretization.FacetRuleFamily

---

::: phydrax.discretization.SideActionDescriptor

---

::: phydrax.discretization.SideTraceQuantity

---

::: phydrax.discretization.SideRepresentation

---

::: phydrax.discretization.SideApproximation

---

::: phydrax.discretization.SideOrientation

---

::: phydrax.discretization.AbstractSideRoute

---

::: phydrax.discretization.SideGatherRoute

---

::: phydrax.discretization.SideRouteMode

---

::: phydrax.discretization.PreparedTraceAction

---

::: phydrax.discretization.AbstractSideFluxEvaluator

---

::: phydrax.discretization.PreparedFluxAction

---

::: phydrax.discretization.TraceInverseEvidence

---

::: phydrax.discretization.certify_trace_inverse

---

::: phydrax.discretization.BoundaryImposition

---

::: phydrax.discretization.ImpositionKind

---

::: phydrax.discretization.SideTraceProvider

---

::: phydrax.discretization.ConormalFluxProvider

---

::: phydrax.discretization.BoundaryImpositionProvider

## Boundary trace spaces

Boundary-integral owners publish their boundary coefficient spaces as trace-space
capabilities: the trace quantity (`BoundaryTraceQuantity`), representation
(`BoundaryTraceRepresentation`), conformity (`BoundaryTraceConformity`), orientation
(`BoundaryTraceOrientation`), coefficient-carrying entity (`BoundaryEntityKind`),
the owner's native coordinate space, its physical Gram pairing and mass, and the
geometry revision. Scalar Dirichlet/Neumann pairs are published together with their
duality; tangential surface currents are never accepted as scalar Cauchy data.

::: phydrax.discretization.BoundaryTraceSpaceCapability

---

::: phydrax.discretization.CauchyTraceCapability

---

::: phydrax.discretization.BoundaryTraceQuantity

---

::: phydrax.discretization.BoundaryTraceRepresentation

---

::: phydrax.discretization.BoundaryTraceConformity

---

::: phydrax.discretization.BoundaryTraceOrientation

---

::: phydrax.discretization.BoundaryEntityKind
