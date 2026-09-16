# Projective Calabi–Yau metric campaigns

The campaign pipeline is explicit:

1. prepare a fixed smooth projective hypersurface;
2. sample intersections with complex projective lines;
3. select stable affine charts and hypersurface pivots;
4. evaluate the Poincaré residue volume;
5. train a projective-invariant Kähler potential;
6. globalize updates by metric positivity;
7. evaluate the candidate on an independently identified held-out sample set;
8. freeze the metric together with raw residual, weight, ESS, batch, chart,
   pivot, positivity, precision, and optional Ricci evidence.

::: phydrax.geometry.complex.HomogeneousPolynomial

::: phydrax.geometry.complex.HypersurfacePatchGeometry

::: phydrax.geometry.complex.ResidueCanonicalSection

::: phydrax.geometry.complex.sample_projective_hypersurface

::: phydrax.geometry.complex.HypersurfaceKahlerGeometry

::: phydrax.solver.CalabiYauMetricProblem

::: phydrax.solver.solve_calabi_yau_metric

::: phydrax.solver.CalabiYauMetricArtifact

::: phydrax.solver.CalabiYauMetricEvidencePlan

::: phydrax.solver.CalabiYauMetricEvidence

`CalabiYauMetricEvidencePlan` rejects identical training/held-out sample
ancestry. Held-out weights are normalized only over valid retained points and
the result keeps raw Monge–Ampère and volume-ratio residuals, positivity,
weighted mean/RMS/max/quantiles, ESS, valid fraction, batch dispersion, and
chart/smoothness margins. Ricci evaluation is optional but must be supplied
explicitly when required; a Kähler-potential metric is Kähler by construction,
not by an unrelated threshold. `freeze_calabi_yau_result(..., evidence=...)`
binds this record into the canonical metric artifact.

Reference factories cover CP1 calibration, a Fermat cubic elliptic curve, a
quartic K3 surface, and the Fermat quintic. These are fixed-complex-structure,
fixed-Kähler-class numerical candidates; sampled residuals do not prove global
topology or Yau's theorem.

## Precision

Campaign problems accept `GeometryPrecisionPolicy`. Sample weights, potential
gauge means, Monge--Ampère residual objectives, positivity decisions, gradient
norms, histories, and frozen artifacts retain the resolved policy and evidence.

## Fixed-root moduli epochs and bounded certificates

`TrainableHomogeneousHypersurface` fixes monomial support and projective degree,
normalizes one declared nonzero coefficient pivot, and may impose a local
transverse slice to the infinitesimal PGL orbit. This is not a global moduli
quotient. `PreparedHypersurfaceEpoch` binds line seeds, simple-root ancestry,
charts, and pivots; collisions, discriminant/pivot/chart loss end the epoch.
`CalabiYauModuliProblem` backtracks rather than clipping across those boundaries.

`ComplexStructureFamilyPlan` adds a narrower exact algebraic contract:
fixed monomial support, normalized pivot, linearly independent coefficient
directions, and the existing transverse PGL slice. Its deformation values are
algebraic normal deformations, not harmonic Kodaira–Spencer representatives.

`PreparedCalabiYauModuliSamples` integrates caller-qualified representative
samples and symmetric Yukawa densities against one
`ProjectiveMeasureTarget`. It reports a sampled Gram metric, Yukawa tensor,
batch standard errors, Hermiticity, positivity, permutation symmetry, ESS, and
representative provenance. Only an explicitly `harmonic` representative kind
can produce authoritative Weil–Petersson/Yukawa evidence; algebraic inputs
remain proxies.

`phydrax.metrix.chern_character_form` computes the declared pointwise
Chern–Weil convention from matrix-valued curvature two-form coefficients.
`integrate_top_characteristic_form` is a sampled estimate with measure and
imaginary-part evidence, not an exact topological invariant.

`CalabiYauCertificate` gates adjunction, compactness, and Hopf--Rinow
completeness on exact degree/nonzero hypotheses, certified cellular cover and
gradient bounds, transition/residue consistency, and a positive global metric.
A nonzero Monge--Ampère residual remains an epsilon-candidate and never becomes
an exact Ricci-flat/Yau claim. Topology conclusions require separate certified
cells/maps.

The K3 and quintic reference constructors remain reproducible preparation,
solve, freeze, and evaluation workflows. No trained checkpoint, downloader, or
checkpoint registry is shipped: that qualification-only item is intentionally
excluded.

`tools/calabi_yau_qualification.py` emits one elliptic held-out metric record,
one explicitly non-authoritative algebraic moduli control, and one normalized
Chern–Weil control. `benchmarks/calabi_yau_evidence.py` separates training and
held-out sampling, metric solve, evidence preparation/evaluation, artifact
freeze, bytes, and scientific residuals.
