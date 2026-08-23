# Complex, Hermitian, and Kähler geometry

Metrix uses two explicit representations:

- complex array manifolds use complex JAX arrays and Hermitian pairings;
- coordinate complex geometry uses real `2n` coordinates plus an
  almost-complex endomorphism `J`.

Raw complex Hermitian matrices are not accepted by the real-coordinate
`RiemannianMetric` contract.

## Complex array manifolds

::: phydrax.metrix.ComplexProjectiveManifold

::: phydrax.metrix.UnitaryGroup

::: phydrax.metrix.SpecialUnitaryGroup

::: phydrax.metrix.UnitaryManifold

::: phydrax.metrix.SpecialUnitaryManifold

::: phydrax.metrix.AffineInvariantHPDManifold

Complex projective points are normalized homogeneous representatives. Every
metric and endpoint operation is invariant under global phase. The logarithm is
nonunique at orthogonal rays.

## Real-coordinate complex calculus

::: phydrax.metrix.ComplexCoordinateConvention

::: phydrax.metrix.AlmostComplexStructure

::: phydrax.metrix.validate_almost_complex_structure

::: phydrax.metrix.nijenhuis_tensor

::: phydrax.metrix.holomorphicity_residual

::: phydrax.metrix.wirtinger_derivatives

## Hermitian and Kähler composition

::: phydrax.metrix.HermitianStructure

::: phydrax.metrix.validate_hermitian_structure

::: phydrax.metrix.KahlerStructure

::: phydrax.metrix.validate_kahler_structure

The fundamental form is emitted as an ordinary `DifferentialForm`, so existing
exterior, Hodge, symplectic, Poisson, and graph/cochain operations apply without
a parallel form representation.

## Dolbeault and Chern calculus

::: phydrax.metrix.BigradedForm

::: phydrax.metrix.partial

::: phydrax.metrix.partial_bar

::: phydrax.metrix.ChernConnection

::: phydrax.metrix.HolomorphicBundleFrame

::: phydrax.metrix.KahlerPotentialGeometry

## Atlas and local SU(n) diagnostics

::: phydrax.metrix.CoordinateAtlas

::: phydrax.metrix.AtlasCover

::: phydrax.metrix.PatchwiseScalarField

::: phydrax.metrix.PatchwiseDifferentialForm

::: phydrax.metrix.ComplexAtlasStructure

::: phydrax.metrix.LocalSUNStructure

::: phydrax.metrix.validate_local_su_structure

`LocalSUNStructure` validates a local Ricci-flat Kähler candidate with an
explicitly declared `(n, 0)` complex volume form. It does not claim compactness,
completeness, global canonical-bundle trivialization, or global Calabi–Yau
topology.
