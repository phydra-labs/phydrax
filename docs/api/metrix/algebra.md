# Finite real algebras

::: phydrax.metrix.algebra.AbstractFiniteRealAlgebraSpec

---

::: phydrax.metrix.algebra.FiniteRealAlgebraSpec

---

::: phydrax.metrix.algebra.RealAlgebraSpec

---

::: phydrax.metrix.algebra.ComplexAlgebraSpec

---

::: phydrax.metrix.algebra.QuaternionAlgebraSpec

---

::: phydrax.metrix.algebra.OctonionAlgebraSpec

---

::: phydrax.metrix.algebra.CayleyDicksonAlgebraSpec

---

::: phydrax.metrix.algebra.MulticomplexAlgebraSpec

---

::: phydrax.metrix.algebra.AlgebraStructureTable

---

::: phydrax.metrix.algebra.AlgebraProductPlan


::: phydrax.metrix.algebra.AlgebraDerivationConstraint

---

::: phydrax.metrix.algebra.AlgebraSymmetryBudget

---

::: phydrax.metrix.algebra.AlgebraSymmetryResourceEvidence

---
---

::: phydrax.metrix.algebra.AlgebraPropertyEvidence

---

::: phydrax.metrix.algebra.AlgebraResourceBudget

---

::: phydrax.metrix.algebra.UnitComplexStateGeometry

---

::: phydrax.metrix.algebra.UnitQuaternionStateGeometry

## Nonassociative state, group, and matrix semantics

`UnitOctonionStateGeometry` is S7 tangent/retraction geometry, not a Lie group.
`MoufangLoopOperations` separately supplies left/right multiplication,
conjugate inverse, associator, and Moufang evidence.
`PreparedUnitOctonionEvolution` advances ordinary real coordinates with one
immutable `BracketingPlan`; it is not dispatched through RKMK.

`G2MatrixElement` certifies a 7 by 7 matrix by orthogonality, determinant, and
three-form preservation. `G2GroupOperations` provides composition, inverse,
action, adjoint, and exponentiation from a declared 14-dimensional derivation
basis.

`AlgebraMatrixProductPlan` defines the binary matrix product with the written
`A[i,k] * B[k,j]` order. Product chains require explicit parentheses.
`algebra_left_solve` and `algebra_right_solve` are distinct real regular-operator
solves. `AlgebraRegularSpectrum` reports the spectrum of that declared real
regular operator; it is not an algebra-valued eigenproblem.
