# Projective Calabi–Yau metric campaigns

The campaign pipeline is explicit:

1. prepare a fixed smooth projective hypersurface;
2. sample intersections with complex projective lines;
3. select stable affine charts and hypersurface pivots;
4. evaluate the Poincaré residue volume;
5. train a projective-invariant Kähler potential;
6. globalize updates by metric positivity;
7. freeze a replayable metric artifact.

::: phydrax.geometry.complex.HomogeneousPolynomial

::: phydrax.geometry.complex.HypersurfacePatchGeometry

::: phydrax.geometry.complex.ResidueCanonicalSection

::: phydrax.geometry.complex.sample_projective_hypersurface

::: phydrax.geometry.complex.HypersurfaceKahlerGeometry

::: phydrax.solver.CalabiYauMetricProblem

::: phydrax.solver.solve_calabi_yau_metric

::: phydrax.solver.CalabiYauMetricArtifact

Reference factories cover CP1 calibration, a Fermat cubic elliptic curve, a
quartic K3 surface, and the Fermat quintic. These are fixed-complex-structure,
fixed-Kähler-class numerical candidates; sampled residuals do not prove global
topology or Yau's theorem.

## Precision

Campaign problems accept `GeometryPrecisionPolicy`. Sample weights, potential
gauge means, Monge--Ampère residual objectives, positivity decisions, gradient
norms, histories, and frozen artifacts retain the resolved policy and evidence.
