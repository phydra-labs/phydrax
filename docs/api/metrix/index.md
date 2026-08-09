# Metrix

`phydrax.metrix` is Phydrax's differentiable geometry layer. It represents charts,
differentiable maps, tensors and forms, positive and signed metrics, affine
connections, curvature, Lie groups, symplectic and Poisson structures, horizontal
cometrics, and array manifolds as ordinary JAX programs.

Metrix is intentionally below the domain and solver layers:

```text
charts and maps → tensors/forms → metric or cometric structure → named operators
                → affine connections and curvature
                → Lie-group / symplectic / Poisson structure
                → array manifolds and state geometry
```

A chart describes an ordered local coordinate representation, not a physical
region. Bounds, boundaries, periodicity, sampling, and measures remain domain
concerns. This separation lets one metric serve direct array calculations,
`DomainFunction` residuals, PINNs, operator models, and stochastic generators.

Array manifolds describe metrics, tangent updates, retractions, and optimizer-state
transport for selected parameter leaves. Array state geometry instead describes
membership, local updates, and pullbacks for a differential-equation solver state.
Neither introduces another domain hierarchy.

## Conventions

- Coordinates have shape `(..., dimension)`.
- Tensor component axes are trailing axes; sample/batch axes remain leading.
- A derivative axis is appended on the right. For example,
  `covariant_derivative(V, ...)` returns components `∇_j V^i` with shape
  `(..., i, j)`.
- Metrics are used exactly as supplied. Metrix never silently symmetrizes,
  regularizes, clips, or takes absolute eigenvalues.
- `RiemannianMetric` means real, symmetric, positive-definite geometry. Use
  `validate_metric` at representative points when that contract is not guaranteed
  by construction.
- Public kernels support eager execution, `jax.jit`, batching, and differentiation
  through array-valued metric parameters.

## Polar-coordinate PDE residual

```python
import jax.numpy as jnp
import phydrax as phx

chart = phx.metrix.CoordinateChart("polar", ("r", "theta"))
metric = phx.metrix.diagonal_metric(
    lambda q: jnp.array([1.0, q[0] ** 2]),
    chart=chart,
)

domain = phx.domain.GeometryDomain(
    phx.geometry.Square(center=(2.0, 0.0), side=1.0).compile()
)


@domain.Function("x")
def u(x):
    return x[0] ** 2


residual = phx.operators.laplace_beltrami(u, metric, var="x") - 4.0
```

The same metric can supply the intrinsic volume density to integral objectives:

```python
component = phx.domain.with_riemannian_measure(
    domain.component(),
    metric,
    var="x",
)
```

This changes integration weights by `sqrt(det(g))`; it does not change sampled
point coordinates or replace domain admissibility rules.

## Public areas

- [Charts and tensors](charts.md)
- [Differentiable maps](maps.md)
- [Metrics and metric jets](metrics.md)
- [Signed metrics](signed_metrics.md)
- [Connections and intrinsic operators](connections.md)
- [General affine connections](affine_connections.md)
- [Curvature](curvature.md)
- [Differential forms](forms.md)
- [Embedded geometry](embedded.md)
- [Array manifolds](manifolds.md)
- [Lie groups](lie_groups.md)
- [Symplectic and Poisson geometry](symplectic_poisson.md)
- [Sub-Riemannian geometry](subriemannian.md)
- [Stochastic geometry](stochastic.md)
- [Array state geometry](state_geometry.md)

## Benchmark

Run representative metric-jet, form, Poisson, Lorentzian, and horizontal kernels with:

```bash
python -m tools.geometric_benchmarks --smoke
```

The report separates compilation-plus-first-call time from steady execution time and
records output bytes, batch size, and coordinate dimension.
