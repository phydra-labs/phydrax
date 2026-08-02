# Metrics and metric jets

## Metric fields

A `RiemannianMetric` owns a pointwise callable returning one real metric matrix
and the chart in which its components are expressed. Leading coordinate axes are
batched automatically.

```python
import jax.numpy as jnp
import phydrax as phx

chart = phx.metrix.CoordinateChart("polar", ("r", "theta"))
metric = phx.metrix.diagonal_metric(
    lambda q: jnp.array([1.0, q[0] ** 2]),
    chart=chart,
)

q = jnp.array([2.0, 0.3])
g = metric(q)  # diag(1, 4)
g_inv = metric.inverse(q)  # diag(1, 1/4)
volume = metric.volume_density(q)  # 2
```

::: phydrax.metrix.RiemannianMetric
    options:
        members:
            - __init__
            - __call__
            - inverse
            - volume_density
            - log_volume_density
            - inner
            - norm_squared

## Constructors

::: phydrax.metrix.euclidean_metric

---

::: phydrax.metrix.diagonal_metric

---

::: phydrax.metrix.cholesky_metric

---

::: phydrax.metrix.pullback_metric

`cholesky_metric` is the recommended learned-SPD parameterization. A model emits
an unconstrained lower-triangular matrix; softplus-transformed diagonal entries
make the resulting metric positive definite by construction. A generic
`RiemannianMetric` remains useful for analytic geometry, but its callable is not
modified or repaired.

`pullback_metric(metric, transition)` implements `Jᵀ g J`, where the transition
maps source coordinates into the metric's target chart. The target chart must
match exactly.

## Reusable metric jets

`metric_jet` evaluates a metric, inverse, determinant, volume density, and
requested first/second coordinate derivatives at common points. Derivative axes
are appended on the right:

- `first_derivative[..., i, j, k] = ∂ₖ gᵢⱼ`,
- `second_derivative[..., i, j, k, l] = ∂ₗ∂ₖ gᵢⱼ`.

Use a jet when several geometric quantities need the same local derivatives;
this avoids independently differentiating the metric for each contraction.

::: phydrax.metrix.MetricJet

---

::: phydrax.metrix.metric_jet

## Explicit validation

Validation is opt-in and diagnostic. It reports finiteness, asymmetry, smallest
eigenvalue, and largest condition number over representative points. It never
changes the metric.

```python
report = phx.metrix.validate_metric(
    metric,
    jnp.array([[1.0, 0.0], [2.0, 0.5]]),
    maximum_condition_number=1e4,
)
```

::: phydrax.metrix.MetricValidationReport

---

::: phydrax.metrix.validate_metric

## Physical units

Metrix kernels consume nondimensional numerical coordinates. Physical-unit
conversion and nondimensionalization belong at the Phydrax input boundary. In a
heterogeneous chart, each coordinate may have a different reference scale; a
single homogeneous quantity cannot represent the full coordinate vector without
losing dense linear-algebra semantics.
