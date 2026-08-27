# Metrix

`phydrax.metrix` is Phydrax's differentiable geometry layer. It represents charts,
atlases, differentiable maps and immersions, tensors and forms, positive and signed
metrics, affine and bundle connections, curvature, metric measures, Lie groups,
symplectic and Poisson structures, horizontal cometrics, finite real algebras,
complex/Kähler and G2 structures, and real or complex array manifolds as ordinary
JAX programs.

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
- Discrete Laplacian eigenfunctions are orthonormal under their declared probability
  measure. Their basis, measure, entity IDs, topology/metric/boundary provenance, and
  exact-versus-certified-tail status are immutable nontrainable state; only consuming
  multiplier parameters train.

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
- [Riemannian maps and immersions](map_geometry.md)
- [Metrics and metric jets](metrics.md)
- [Metric measures and boundaries](metric_measure.md)
- [Signed metrics](signed_metrics.md)
  The [inverse spacetime workflow](../../cookbook/relativity_inverse.md) combines
  safe ADM parameterization, curvature observations, and functional optimization.
- [Connections and intrinsic operators](connections.md)
- [General affine connections](affine_connections.md)
- [Curvature](curvature.md)
- [Differential forms](forms.md)
- [Finite real algebras](algebra.md)
- [Special holonomy and G2 geometry](special_holonomy.md)
- [Embedded geometry](embedded.md)
- [Array manifolds](manifolds.md)
- [Intrinsic endpoint geometry](intrinsic_geometry.md)
- [Complex, Hermitian, and Kähler geometry](complex_geometry.md)
- [Lie groups](lie_groups.md)
- [Symplectic and Poisson geometry](symplectic_poisson.md)
- [Sub-Riemannian geometry](subriemannian.md)
- [Stochastic geometry](stochastic.md)
- [Array state geometry](state_geometry.md)
- [Vector bundles and gauge geometry](bundles.md)

## Laplacian spectra

`phydrax.discretization.SpectralDecomposition` is the geometry-to-kernel spectral
contract. It pairs one canonical `ModalTransform` with a Laplacian
`OperatorSpectrum` and construction report. `product_laplacian_eigenbasis` selects
the lowest eigenvalue sums from compact factors without materializing their full Cartesian
product. `SphereLaplacianLevels` stores analytic eigenvalues and multiplicities
without choosing a spherical-harmonic basis.

::: phydrax.discretization.LaplacianEigenbasisReport

---

::: phydrax.discretization.SpectralDecomposition

---

::: phydrax.metrix.product_laplacian_eigenbasis

---

::: phydrax.metrix.SphereLaplacianLevels

## Benchmark

Run representative metric-jet, form, Poisson, Lorentzian, and horizontal kernels with:

```bash
python -m tools.geometric_benchmarks --smoke
```

Octonion-derived G2 cross products and local compatibility/torsion diagnostics are
included in:

```bash
python -m tools.exotic_geometry_benchmarks --smoke
```

The report separates compilation-plus-first-call time from steady execution time and
records output bytes, batch size, and coordinate dimension.
