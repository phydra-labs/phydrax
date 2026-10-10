# Boundary platform contracts

See [Boundary platform qualification](../../guides_boundary_platform.md) for the exact
support tuple, evidence ladder, Q0--Q3 maturity, fail-closed provider workflow, and
current-versus-planned support boundary.

These immutable contracts are metadata and validation values. They do not dispatch
providers, collect telemetry, approve licenses, serialize with pickle, or create a
global capability registry.

## Support envelopes

`BoundarySupportEnvelope` binds claims to exactly one
`geometry × trace × PDE/formulation × provider × precision × differentiation × platform`
tuple. `claims` is a local finite vocabulary for that envelope. Every
`unsupported_claims` entry names one of those claims and gives a reason. Unknown claims,
empty IDs, duplicate entries, and unbounded collections are rejected. The canonical
`envelope_id` includes unsupported declarations and stop-ship conditions.

::: phydrax.operators.integral.layer_potential.BoundarySupportEnvelope

## Qualification evidence

`BoundaryQualificationEvidence` records one envelope claim at one of `computed`,
`checked-discrete`, `quadrature-supported`, `continuum-qualified`, or
`continuum-certified`. Every level above `computed` needs prerequisite evidence and a
named finite nonnegative error bound. `maturity` is independently one of Q0--Q3.

Unsupported evidence is explicit: it has `supported=False`, a reason, and no numerical
error. It therefore cannot be confused with a supported zero error. Provenance,
operational, and artifact IDs are mandatory inputs to its `evidence_id`.

::: phydrax.operators.integral.layer_potential.BoundaryQualificationEvidence

## Product provenance

`BoundaryProductProvenance` records the source kind, source content, license,
clean-room record, provider, producer, and product/plan/result lineage. A clean-room
record ID identifies evidence; it is not a claim of legal approval. Parent collections
are canonicalized and self-parent product lineage is rejected.

::: phydrax.operators.integral.layer_potential.BoundaryProductProvenance

## Operational evidence

`BoundaryOperationalEvidence` records provider determinism, security and resource
preflights, byte limit/forecast/observation, plan/result lineage, and stop-ship reasons.
Both preflights must pass before a result can be recorded. Observed bytes require a
result; a resource-limit violation requires an explicit stop-ship reason. Failed
preflights never become implicit fallback dispatches.

::: phydrax.operators.integral.layer_potential.BoundaryOperationalEvidence

## Surface geometry

::: phydrax.geometry.SurfaceModel

---

::: phydrax.geometry.SurfaceAuditPolicy

---

::: phydrax.geometry.HighOrderSurfaceRealization

---

::: phydrax.geometry.import_surface

---

::: phydrax.geometry.export_surface

## Scalar boundary calculus

::: phydrax.operators.integral.layer_potential.prepare_scalar_calderon_dp0_3d

---

::: phydrax.operators.integral.layer_potential.prepare_scalar_screen_single_layer_dp0_3d

---

::: phydrax.solver.prepare_scalar_transmission_3d

## Periodic operators

::: phydrax.operators.integral.layer_potential.prepare_periodic_modified_helmholtz_single_layer_dp0_3d

---

::: phydrax.operators.integral.layer_potential.prepare_periodic_helmholtz_single_layer_dp0_3d

---

::: phydrax.operators.integral.layer_potential.prepare_periodic_maxwell_electric_field_action_3d

## Coupled and vector physics

The matching FEM–BEM products publish an owner derivative capability
(`derivative_capability`, a `phydrax.OwnerDerivativeCapability`) on the prepared
product and on every result, and a per-result `derivative_valid` flag separate
from primal `valid`. The linear policy's differentiation mode selects the route
at preparation; see the derivative support table in
[Boundary platform qualification](../../guides_boundary_platform.md#derivative-support).
The scalar product admits the volume source and DP0 transmission jumps under
`rhs-only`, and additionally the per-cell interior `conductivity` under
`mathematical`; the elasticity product admits its interior and boundary loads
under `rhs-only` only. Geometry, kernel, and quadrature derivatives are refused.

::: phydrax.solver.prepare_scalar_laplace_fem_bem_3d

---

::: phydrax.solver.solve_scalar_laplace_fem_bem_3d

---

::: phydrax.solver.PreparedScalarLaplaceFEMBEM3D

---

::: phydrax.solver.ScalarLaplaceFEMBEMResult3D

---

::: phydrax.solver.prepare_elasticity_fem_bem_3d

---

::: phydrax.solver.solve_elasticity_fem_bem_3d

---

::: phydrax.operators.integral.layer_potential.prepare_elasticity_single_layer_dp0_3d

---

::: phydrax.operators.integral.layer_potential.prepare_stokes_single_layer_dp0_3d

---

::: phydrax.operators.integral.layer_potential.prepare_maxwell_efie_3d

## Hydrodynamics and time

::: phydrax.operators.integral.layer_potential.prepare_free_surface_hydrodynamics_3d

---

::: phydrax.solver.solve_potential_flow_hydrodynamics_3d

---

::: phydrax.solver.solve_hydrodynamic_response_3d

---

::: phydrax.solver.prepare_convolution_quadrature

## Execution, adaptation, and persistence

::: phydrax.operators.integral.layer_potential.BEMExecutionEnvelope

---

::: phydrax.operators.integral.layer_potential.BoundaryMeshEpoch

---

::: phydrax.export.write_bem_array_archive

---

::: phydrax.export.read_bem_array_archive
