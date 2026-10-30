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


## ASTS2D T-spline surface H1 candidate

The shared `TMesh2D`/local-knot/extraction substrate admits one explicit
analysis-suitable profile: rectilinear bicubic T-meshes with topology-derived
local knot windows, simple interior knots, and no extraordinary vertices.
Arbitrary quadrilateral meshes are not bases and are never tensorized as a
fallback. This candidate route is separate from the tensor NURBS support tuple
above; it does not qualify shell or compatible-form spaces.

```python
from fractions import Fraction

import numpy as np
import phydrax as phx
from phydrax.geometry.brep import TSplineSurfacePatch

iga = phx.discretization.iga
breaks = tuple(Fraction(index, 4) for index in range(5))
mesh = iga.TMesh2D.tensor_product(breaks, breaks)
greville = tuple(anchor.local_knots.greville_parameter for anchor in mesh.anchors)
exact_control_points = tuple((u, v, Fraction(0)) for u, v in greville)
surface = TSplineSurfacePatch(
    mesh,
    np.asarray([[float(value) for value in row] for row in exact_control_points]),
    np.ones((mesh.coefficient_count,), dtype=np.float64),
    exact_control_points=exact_control_points,
    exact_weights=tuple(Fraction(1) for _ in mesh.anchors),
)
trim = iga.UVTrimmedSurface(
    iga.UVTrimLoop(((0, 0), (1, 0), (1, 1), (0, 1)))
)
plan = iga.TSplineIsogeometricPlan(
    surface,
    iga.TSplineH1FieldSpec(
        "u", weights_from_geometry=True, anchor_count=mesh.coefficient_count
    ),
    trim=trim,
    quadrature_policy=iga.IsogeometricQuadraturePolicy(4),
)
prepared = plan.prepare()
uv_points = np.asarray(((0.2, 0.3), (0.7, 0.8)), dtype=np.float64)
query_route = prepared.prepare_query_route(uv_points)
geometry_jet = prepared.evaluate_geometry_route(query_route)
```

The basis certificate proves exact Bernstein partition identities and cross-cell
C2 traces. `TSplineGeometryCertificate` separately checks positive rational
denominator, regularity, and global embedding using exact Bernstein interval
bounds on a strictly monotone physical-coordinate projection. A failed or
unresolved geometry proof refuses IGA admission. Geometry and independent field
weights remain raw. Refinement is an atomic discrete event; the returned
`TSplineIsogeometricRefinement` retains exact target-by-source transfer and
prolongs rational fields as `P(aq) / P(a)`. Exact homogeneous geometry uses
`(wX,w)` and preserves non-binary coefficients such as 1/3; its bounded numeric
JAX realization and conversion error remain separate.

The route integrates the supplied `UVTrimmedSurface` by cut-cell quadrature and
uses one-sided fixed cell IDs for JAX geometry/field queries. Concrete source
bank edits require an explicitly re-authored exact source revision; a rounded
target bank alone cannot retain the original exact-equivalence claim.

::: phydrax.discretization.iga.TMesh2D

---

::: phydrax.discretization.iga.TSplineGeometryState

---

::: phydrax.discretization.iga.TSplineH1FieldSpec

---

::: phydrax.discretization.iga.TSplineIsogeometricPlan

---

::: phydrax.discretization.iga.PreparedTSplineIsogeometricDiscretization

---

::: phydrax.discretization.iga.TSplineIsogeometricRefinement

---

::: phydrax.discretization.iga.TSplineQueryRoute


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
