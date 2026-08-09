# Signed metrics and Lorentzian geometry

Signed metrics are nominally distinct from positive-definite Riemannian metrics.
`SemiRiemannianMetric` records a fixed nondegenerate signature.
`LorentzianMetric` additionally records a mostly-plus or mostly-minus convention;
`TimeOrientation` separately selects a future cone from a timelike reference field.
Degenerate metrics are rejected rather than repaired.

The distinction is enforced at operator boundaries. Index raising and lowering,
tensor self-contraction, Levi--Civita connections, covariant derivatives, curvature,
Hodge stars, codifferentials, and Hodge Laplacians are valid for every declared
nondegenerate signature. Positive norms, Brownian generators, and Riemannian
optimization still accept only positive-definite geometry. Causal classification,
proper time, and the d'Alembertian require `LorentzianMetric`.

```python
import jax.numpy as jnp
import phydrax as phx

chart = phx.metrix.CoordinateChart("spacetime", ("t", "x", "y", "z"))
metric = phx.metrix.minkowski_metric(chart)
point = jnp.zeros((4,))
velocity = jnp.array([1.0, 0.0, 0.0, 0.0])

assert phx.metrix.causal_character(metric, point, velocity) == -1
```

For labeled PDE functions, `intrinsic_dalembertian` preserves the ordinary
`DomainFunction` dependency and batching conventions.

The form convention includes the metric index in both Hodge-square and
codifferential signs. Consequently, for a scalar `f`,
`hodge_laplacian(DifferentialForm(f, ...), metric)` evaluates `−□f` on Lorentzian
spacetime. The labeled `domain_codifferential` and `domain_hodge_laplacian` adapters
preserve the same convention. See [Differential forms](forms.md) for the complete
sign identities.

## ADM decomposition and hypersurface geometry

`ADMParameterization` maps unconstrained callable fields to a positive lapse and
an SPD spatial metric through softplus and a triangular factor. Use it for
trainable spacetime geometry; `adm_metric` is appropriate when the caller already
guarantees those contracts. `decompose_adm_metric` reverses either construction
at one point or a leading batch of points.

`validate_adm_decomposition` is diagnostic only. It reports finiteness, lapse
positivity, spatial symmetry/positivity, Lorentzian signature, analytic-inverse
residual, and optional metric-reconstruction residual without modifying any
field. The future normal and extrinsic-curvature APIs use a time-first chart and
the convention
\(K_{ij}=-\tfrac12\mathcal L_n\gamma_{ij}\). Hamiltonian and momentum residuals
accept matter sources and an explicit Einstein coupling.

For an end-to-end parameter-identification example, including shared metric and
curvature observations, see
[Inverse spacetime geometry](../../cookbook/relativity_inverse.md).

::: phydrax.metrix.MetricSignature

::: phydrax.metrix.SemiRiemannianMetric

::: phydrax.metrix.LorentzianMetric

::: phydrax.metrix.minkowski_metric

::: phydrax.metrix.flrw_metric

::: phydrax.metrix.adm_metric

::: phydrax.metrix.ADMDecomposition

::: phydrax.metrix.ADMParameterization

::: phydrax.metrix.ADMValidationReport

::: phydrax.metrix.decompose_adm_metric

::: phydrax.metrix.parameterized_adm_metric

::: phydrax.metrix.validate_adm_decomposition

::: phydrax.metrix.ADMConstraintResiduals

::: phydrax.metrix.adm_normal_vector

::: phydrax.metrix.adm_normal_covector

::: phydrax.metrix.adm_spacetime_projector

::: phydrax.metrix.adm_extrinsic_curvature

::: phydrax.metrix.adm_hamiltonian_constraint

::: phydrax.metrix.adm_momentum_constraint

::: phydrax.metrix.adm_constraint_residuals

::: phydrax.metrix.validate_semi_riemannian_metric

::: phydrax.metrix.validate_lorentzian_metric

::: phydrax.metrix.causal_character

::: phydrax.metrix.proper_time_rate

::: phydrax.metrix.dalembertian

::: phydrax.operators.intrinsic_dalembertian
