# Global projective and Ricci-flat Kähler workflows

`CoordinateAtlas` is a transition graph. `AtlasCover` strengthens it with explicit
chart supports and overlap predicates. Typed patchwise tensors, forms, metrics,
and densities retain their own transformation laws.

::: phydrax.metrix.AtlasCover

::: phydrax.metrix.AtlasOverlap

::: phydrax.metrix.ChartSupport

::: phydrax.metrix.PatchwiseTensorField

::: phydrax.metrix.PatchwiseDifferentialForm

::: phydrax.metrix.PatchwiseMetric

::: phydrax.metrix.PatchwiseDensity

`phydrax.integration.AtlasIntegrationTarget` combines fixed patch quadratures
with explicit ownership or partition weights. Unweighted sums over overlapping
charts are not treated as global integrals.

## Projective references

`phydrax.geometry.complex.ComplexProjectiveAtlas` provides affine CP^n charts,
projective-ratio transitions, overlap supports, and local Fubini–Study metrics
and Kähler structures.

`ProjectiveHypersurface` represents a homogeneous polynomial and exposes local
residual and smoothness margins. `fermat_hypersurface(N)` constructs the degree
`N + 1` Fermat hypersurface in CP^N.

## Kähler potentials

::: phydrax.metrix.KahlerPotentialGeometry

::: phydrax.operators.domain_monge_ampere_residual

::: phydrax.operators.domain_kahler_positivity_margin

::: phydrax.terms.ricci_flat_kahler_term

The potential correction remains in a fixed reference Kähler class. Positivity
is exposed separately from the Monge–Ampère residual. A local SU(n) validator is
not a global Calabi–Yau topology claim.
