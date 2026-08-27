# Special holonomy structures

## Octonion bridge

::: phydrax.metrix.OctonionG2Bridge

`OctonionG2Bridge` owns the canonical flat identification between the seven ordered
imaginary octonion coordinates and a seven-dimensional chart. Its cross product and
associative three-form are derived from the exact `OctonionAlgebraSpec` structure table.
The bridge does not infer a moving octonion frame on a curved manifold.

---

## Local G2 structures

::: phydrax.metrix.LocalG2Structure

---

::: phydrax.metrix.LocalG2ValidationReport

---

::: phydrax.metrix.validate_local_g2_structure

A local G2 structure supplies both its Riemannian metric and degree-three associative
form. Validation checks their algebraic compatibility and reports closure, coclosure,
torsion freedom, and Ricci-flatness independently. It does not claim global holonomy,
completeness, compactness, or topology.

---

## Infinitesimal automorphisms

::: phydrax.metrix.G2DerivationInvarianceReport

---

::: phydrax.metrix.validate_g2_derivations

The derivation validator checks whether a prepared octonion derivation subspace fixes
the scalar coordinate, is skew with respect to the canonical metric, and
infinitesimally preserves the associative three-form. The prepared subspace is
numerical evidence derived from exact Leibniz constraints; it is not a full G2 group
implementation.

## Boundary

Unit octonions form a smooth Moufang loop, not a Lie group. This API neither enables
Lie-group integrators for unit octonions nor introduces Moufang-loop exponential,
logarithm, or composition methods. Unconstrained octonion states remain real coordinate
arrays, and unit-norm states can use `SphereManifold(8)` independently of multiplication.
