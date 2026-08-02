# Stochastic geometry

Metrix separates coordinate Itô coefficients from geometric tensor fields. This
matters because an Itô drift does not transform as a vector under a nonlinear
coordinate change.

For a coordinate Stratonovich SDE

\[
dq^i = b_S^i\,dt + \sigma^i_a \circ dW^a,
\]

the coordinate Itô drift is

\[
b_I^i = b_S^i + \frac12 \sigma^j_a\,\partial_j\sigma^i_a.
\]

::: phydrax.metrix.coordinate_stratonovich_to_ito_drift

Given `a^{ij} = σ^i_a σ^j_a`, the vector drift used with a covariant Hessian is

\[
b_{\mathrm{cov}}^k = b_I^k + \frac12\Gamma^k_{ij}a^{ij}.
\]

::: phydrax.metrix.coordinate_to_covariant_drift

The two representations define the same scalar generator:

\[
b_I^i\partial_i u + \frac12a^{ij}\partial_i\partial_j u
=
b_{\mathrm{cov}}^i\nabla_i u
 + \frac12a^{ij}\nabla_i\nabla_j u.
\]

## Covariant backward and forward operators

::: phydrax.metrix.covariant_kolmogorov_generator

The covariant Fokker–Planck operator is the formal adjoint with respect to
Riemannian volume `dvol_g`:

\[
\mathcal L^*p
=-\nabla_i(b^i p)+\frac12\nabla_i\nabla_j(a^{ij}p).
\]

Consequently `p` is a scalar density relative to `dvol_g`, not a coordinate
Lebesgue density. This distinction is essential in curvilinear coordinates.

::: phydrax.metrix.covariant_fokker_planck_operator

## Brownian motion

Riemannian Brownian motion has generator one half of the Laplace–Beltrami
operator:

\[
\mathcal L_B u = \frac12\Delta_g u.
\]

::: phydrax.metrix.brownian_generator

## `DomainFunction` integration

The existing Phydrax stochastic operators accept an optional `metric=`:

```python
import jax.numpy as jnp
import phydrax as phx

domain = phx.domain.Square(center=(2.0, 0.0), side=1.0)
chart = phx.metrix.CoordinateChart("polar", ("r", "theta"))
metric = phx.metrix.diagonal_metric(
    lambda q: jnp.array([1.0, q[0] ** 2]),
    chart=chart,
)
observable = domain.Function("x")(lambda x: x[0] ** 2)
density = domain.Function("x")(lambda x: x[0] ** 2)
coordinate_drift = domain.Function("x")(lambda x: jnp.array([0.5 / x[0], 0.0]))
covariance = domain.Function("x")(lambda x: metric.inverse(x))

backward = phx.operators.kolmogorov_generator(
    observable,
    coordinate_drift,
    covariance=covariance,
    metric=metric,
    var="x",
)

forward = phx.operators.fokker_planck_operator(
    density,
    coordinate_drift,
    covariance=covariance,
    metric=metric,
    var="x",
)
```

Their contracts are:

- Without `metric`, drift and density retain the existing coordinate/Lebesgue
  semantics.
- With `metric`, the supplied Itô drift is interpreted as a coordinate drift,
  converted to the equivalent covariant vector drift, and the forward density is
  relative to `dvol_g`.
- With `interpretation="stratonovich"`, Phydrax first applies the coordinate
  Stratonovich-to-Itô correction. A diffusion factor is required because a
  covariance alone does not identify the diffusion vector fields.
- Time integration, path simulation, adaptive stepping, and Brownian-tree
  construction remain in Diffrax and Phydrax solver infrastructure. Metrix owns
  only the geometric kernels.

This boundary also keeps ordinary coordinate SDEs, constrained embedded SDEs,
and Riemannian Brownian motion distinct instead of hiding them behind one
ambiguous `manifold=True` switch.
