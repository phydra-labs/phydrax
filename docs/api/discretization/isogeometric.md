# Isogeometric analysis

The S1 API is intentionally limited to regular, untrimmed, full-dimensional
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
