# Isogeometric analysis

The scalar `IsogeometricPlan` S1 release API is intentionally limited to regular, untrimmed, full-dimensional
2D single patches with clamped isotropic fixed B-spline grids and one exactly
isoparametric scalar H1 field. See the
[isogeometric-analysis guide](../../guides_isogeometric_analysis.md) for the
support boundary and qualification semantics.

## Fixed spline topology and geometry state

::: phydrax.discretization.iga.BSplineGrid

---

::: phydrax.discretization.iga.NURBSGeometryState

---


::: phydrax.discretization.iga.IsogeometricQuadraturePolicy

---

::: phydrax.discretization.iga.IsogeometricH1QualificationPolicy

---

::: phydrax.discretization.iga.IsogeometricGeometryEvidence

## Plan, preparation, and runtime refresh

::: phydrax.discretization.iga.IsogeometricPlan

---

::: phydrax.discretization.iga.PreparedIsogeometricDiscretization

---

::: phydrax.discretization.iga.PreparedIsogeometricDiscretization.homogeneous_trace_constraint

---

::: phydrax.discretization.iga.IsogeometricRuntimeData

The prepared discretization is consumed by the existing finite-element form and
compiler API:

- [`FiniteElementForm`](finite_element.md#weak-forms-and-execution);
- `phydrax.equations.compile_finite_element_problem`;
- `phydrax.equations.FiniteElementExecutionPolicy` with
  `realization="matrix_free"` and `local_kernel="sum_factorized"`.

The qualification tool's independently assembled sparse parity path is private
to that tool. It checks public matrix-free execution and scatter; it is not an
independent numerical oracle or a public IGA sparse-realization contract.

## Prepared field queries and side traces

Physical-point reconstruction inverts the runtime NURBS map in JAX and exposes
the shared `PreparedFieldReconstruction.prepare_query` routes; side traces act on
patch-boundary faces and interior knot faces with physical facet measures and
outward normals. See the
[guide section](../../guides_isogeometric_analysis.md#prepared-field-queries-and-side-traces).

::: phydrax.discretization.iga.prepare_isogeometric_field_reconstruction

---

::: phydrax.discretization.iga.IsogeometricFieldReconstructionKernel

---

::: phydrax.discretization.iga.PreparedIsogeometricDiscretization.prepare_side_trace

---

::: phydrax.discretization.iga.PreparedIsogeometricDiscretization.integration_domain


## Public spline de Rham complex

`SplineDeRhamComplex(grids, /, *, periodic=None, geometry=None,
geometry_id=None, twist="untwisted", quadrature_degree=None, hodge_policy=None)`
implements the common cell de Rham protocol. Exact sparse d is paired with
Kronecker Gram blocks for identity tensor geometry; mapped/embedded geometry
uses declared metric quadrature and native solve preparation. `hodge_policy`
is the owning `LinearSolvePolicy`, not another selector vocabulary.

Geometry returns an ambient vector of dimension at least the intrinsic dimension.
Mapped Jacobians are checked finite and full tangent rank on native quadrature.
The default quadrature degree is `2 * max(axis_degree) + 4`; degrees below
`2 * max(axis_degree)` are refused so squared basis products are resolved.
The geometry callable receives actual knot-domain coordinates.
`BSplineGrid.open_uniform` defaults to `interval=(-1, 1)`; a unit-parameter recipe
such as X(u,v)=(1+u)(cos(2πv),sin(2πv)) requires grids prepared explicitly with
`interval=(0, 1)`. No implicit parameter rescaling changes the scalar grid default.



`interpolant(k, form_callable)` uses canonical tensor point/moment functionals;
`reconstruction(k, coefficients, parameter_points)` returns physical components.
`transfer(target, tolerance=1e-10)` and `trace(axis, side)` return `ComplexMap`.
`trace_complex_map(boundary_mask=None)` binds the owner's declared boundary faces
with outward parameterized signs. Absolute/relative Hilbert complexes use active
pairings and exact restricted degree maps.
Traces cover one through three axes; interval endpoints map to zero-dimensional
point Hilbert complexes rather than an invented lower-dimensional grid.

`refresh_geometry(geometry)` on a mapped complex reuses prepared quadrature/basis
queries, moment factors, sparse routes and binding ids while updating metric and
face pairings natively. `geometry_id` identifies a chart binding, not a floating
snapshot. An unmapped identity complex cannot refresh; prepare callable identity
geometry with explicit geometry_id when dynamic geometry is required.


`AssembledSplineDeRhamComplex` admits explicit native derivative/Gram operators
and validates d², degree dimensions and relative closure. These public complex
APIs do not automatically expand a qualified scalar S1/R1 release tuple.

::: phydrax.discretization.iga.SplineDeRhamComplex

::: phydrax.discretization.iga.AssembledSplineDeRhamComplex
