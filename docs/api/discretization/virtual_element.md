# Virtual elements

## Polygon mesh and geometry

::: phydrax.discretization.CellMesh.from_polygons

---

::: phydrax.discretization.PolygonAdmissibilityPolicy

---

::: phydrax.discretization.PolygonTriangulation

---

::: phydrax.discretization.PolygonGeometry

---

::: phydrax.discretization.PolygonGeometryEvidence

## Spaces and preparation

::: phydrax.discretization.VirtualElementSpec

`VirtualElementSpec` requires `value_spec=FormValueSpec(...)`; `value_shape` and
form semantics are derived. Planar H(div) is twisted degree1 flux and H(curl) is
untwisted degree1 circulation. Scalar H1/L2 fields remain untwisted zero-forms,
not top-degree density. Reconstruction ports declare the same value spec.
This is form typing of existing VEM families, not a VEM de Rham complex or a
claim of three-dimensional vector families.


---

::: phydrax.discretization.conforming_h1_virtual_element

---

::: phydrax.discretization.VirtualElementFieldSpec

---

::: phydrax.discretization.VirtualElementDofMap

---

::: phydrax.discretization.VirtualElementPlan

---

::: phydrax.discretization.VirtualElementDiscretization

---

::: phydrax.discretization.VirtualElementRuntimeData

---

::: phydrax.discretization.VirtualElementPrecisionPolicy

---

::: phydrax.discretization.VirtualElementResourceBudget

## Projections and stabilization

::: phydrax.discretization.VirtualElementProjectionData

---

::: phydrax.discretization.VirtualElementProjectionEvidence

---

::: phydrax.discretization.VirtualElementStabilizationPolicy

---

::: phydrax.discretization.VirtualElementStabilizationEvidence

## Constraints and forms

::: phydrax.discretization.VirtualElementDirichletConstraint

---

::: phydrax.discretization.virtual_element_dirichlet_constraint

---

::: phydrax.equations.VirtualElementForm

---

::: phydrax.equations.VirtualElementRobinAction

---

::: phydrax.equations.VirtualElementExecutionPolicy

---

::: phydrax.equations.VirtualElementExecutionContext

---

::: phydrax.equations.compile_virtual_element_problem

---

::: phydrax.equations.CompiledVirtualElementProblem

## Reconstruction

::: phydrax.equations.VirtualElementReconstruction

---

::: phydrax.equations.project_virtual_element_field

---

::: phydrax.equations.evaluate_virtual_element_reconstruction

---

::: phydrax.equations.evaluate_virtual_element_trace

## Prepared channels, traces, and fluxes

::: phydrax.equations.vem.VirtualElementReconstructionChannel

---

::: phydrax.equations.vem.prepare_virtual_element_field_reconstruction

---

::: phydrax.discretization.VirtualElementDiscretization.prepare_side_trace

---

::: phydrax.discretization.VirtualElementDiscretization.edge_trace_routes

---

::: phydrax.discretization.VirtualElementSpec.edge_trace_basis

---

::: phydrax.equations.CompiledVirtualElementProblem.prepare_conormal_flux

---

::: phydrax.equations.CompiledVirtualElementProblem.prepare_projected_flux

---

::: phydrax.equations.CompiledVirtualElementProblem.boundary_impositions
