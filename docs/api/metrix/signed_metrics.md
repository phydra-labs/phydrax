# Signed metrics and Lorentzian geometry

Signed metrics are nominally distinct from positive-definite Riemannian metrics.
`SemiRiemannianMetric` records a fixed nondegenerate signature;
`LorentzianMetric` additionally records a mostly-plus or mostly-minus convention and
a declared time orientation. Degenerate metrics are rejected rather than repaired.

The distinction is enforced at operator boundaries. Positive norms, Brownian generators,
and Riemannian optimization accept only `RiemannianMetric`. Causal classification,
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

::: phydrax.metrix.MetricSignature

::: phydrax.metrix.SemiRiemannianMetric

::: phydrax.metrix.LorentzianMetric

::: phydrax.metrix.minkowski_metric

::: phydrax.metrix.flrw_metric

::: phydrax.metrix.adm_metric

::: phydrax.metrix.validate_semi_riemannian_metric

::: phydrax.metrix.validate_lorentzian_metric

::: phydrax.metrix.causal_character

::: phydrax.metrix.proper_time_rate

::: phydrax.metrix.dalembertian

::: phydrax.operators.intrinsic_dalembertian
