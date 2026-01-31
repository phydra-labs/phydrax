# Differential operators

This page documents Phydrax's core differential operators from a mathematical point of view
and explains how they interact with **labeled product domains** and **structured sampling**.

## Notation and conventions

Let the domain be a labeled product \(\Omega = \Omega_x \times \Omega_t \times \cdots\).
Phydrax represents functions as `DomainFunction` objects, which conceptually define a map

$$
u:\Omega\to\mathbb{R}^{m_1\times\cdots\times m_k}.
$$

For a *geometry* label (typically `"x"`) with spatial dimension \(d\), write
\(x=(x_1,\dots,x_d)\in\mathbb{R}^d\). For a *scalar* label (typically `"t"`), write
\(t\in\mathbb{R}\).

Phydrax adopts the convention that **derivative dimensions are appended on the right**:

- if \(u\) is scalar-valued, then \(\nabla_x u\) has a trailing axis of length \(d\);
- if \(u\) is vector-valued with trailing size \(m\), then \(\nabla_x u\) has trailing shape \((m,d)\);
- higher-rank tensor values are differentiated componentwise, appending derivative axes.

## Gradient / Jacobian

`phydrax.operators.grad(u, var="x")` constructs the gradient/Jacobian with respect to a labeled variable.

### Geometry variables

For a geometry variable \(x\in\mathbb{R}^d\):

If \(u:\Omega\to\mathbb{R}\) is scalar-valued, then

$$
\nabla_x u = \left(\frac{\partial u}{\partial x_1},\dots,\frac{\partial u}{\partial x_d}\right).
$$

If \(u:\Omega\to\mathbb{R}^m\) is vector-valued, then `grad` returns the Jacobian
\(J\in\mathbb{R}^{m\times d}\) with entries \(J_{ij}=\partial u_i/\partial x_j\).

### Scalar variables

For a scalar label \(t\), `grad(u, var="t")` reduces to the partial derivative \(\partial u/\partial t\).

## Hessian, Laplacian, Bi-Laplacian

### Hessian

`phydrax.operators.hessian(u, var="x")` returns the matrix of second derivatives.
For scalar-valued \(u\),

$$
H_{ij}(x)=\frac{\partial^2 u}{\partial x_i\,\partial x_j}.
$$

For vector-valued \(u\), the Hessian is taken componentwise, producing a trailing shape \((m,d,d)\).

### Laplacian

`phydrax.operators.laplacian(u, var="x")` computes

$$
\Delta u \;=\; \nabla\cdot\nabla u \;=\; \sum_{i=1}^{d}\frac{\partial^2 u}{\partial x_i^2}
        \;=\; \text{tr}(\nabla^2 u).
$$

### Bi-Laplacian

`phydrax.operators.bilaplacian(u, var="x")` computes the fourth-order operator

$$
\Delta^2 u \;=\; \Delta(\Delta u).
$$

## Divergence and curl

### Divergence

For a vector field \(v:\Omega\to\mathbb{R}^d\), `phydrax.operators.div(v, var="x")` computes

$$
\nabla\cdot v = \sum_{i=1}^{d}\frac{\partial v_i}{\partial x_i} = \text{tr}(\nabla v).
$$

If \(v\) has additional leading value axes (e.g. multiple vector fields stacked), divergence is applied
componentwise over those leading value axes.

### Curl (3D only)

For \(v:\Omega\to\mathbb{R}^3\), `phydrax.operators.curl(v, var="x")` computes

$$
\nabla\times v =
\begin{pmatrix}
  \partial_y v_z - \partial_z v_y \\
  \partial_z v_x - \partial_x v_z \\
  \partial_x v_y - \partial_y v_x
\end{pmatrix}.
$$

## Backends: autodiff, finite differences, spectral/basis

Many differential operators accept a `backend` keyword:

- `backend="ad"` uses autodiff and works for both point sampling and coord-separable sampling.
- `backend="jet"` uses Taylor-mode AD ("jets") for higher-order derivatives with respect to a single variable.
- `backend="fd"` uses finite differences on coord-separable grids (and falls back to autodiff for point inputs).
- `backend="basis"` uses basis-aware methods on coord-separable grids (and falls back to autodiff for point inputs).

### Jet backend (Taylor-mode / derivative jets)

The jet backend propagates a *truncated derivative jet* through the computation graph. Concretely, for a smooth
map \(f\) and a direction \(v\), it computes the derivatives of the 1D curve \(y(\epsilon)=f(x+\epsilon v)\) at
\(\epsilon=0\), i.e. the coefficients of the Taylor expansion

$$
f(x+\epsilon v)
= \sum_{k=0}^{K}\frac{\epsilon^k}{k!}\,D^k f(x)[v,\dots,v] + O(\epsilon^{K+1}).
$$

Under the hood, higher-order chain rules are governed by the Faà di Bruno formula. In one dimension, for a
composition \(f\circ g\),

$$
(f\circ g)^{(n)}(x)
= \sum_{k=1}^{n} f^{(k)}(g(x))\,B_{n,k}\bigl(g'(x),g''(x),\dots,g^{(n-k+1)}(x)\bigr),
$$

where \(B_{n,k}\) are the (partial) Bell polynomials. Jet-mode AD implements these combinatorics automatically,
which is why it can be more direct than nesting `jax.jacfwd`/`jax.jacrev` when you need \(n\ge 2\) derivatives
with respect to the *same* variable.

The `basis` keyword (used when `backend="basis"`) selects a 1D method along each coord-separable axis:

- `basis="fourier"`: FFT-based spectral derivatives on periodic grids;
- `basis="sine"` / `basis="cosine"`: FFT-based derivatives via odd/even extension;
- `basis="poly"`: polynomial (barycentric) differentiation on generic 1D grids.

!!! note
    FFT-based bases (`fourier`/`sine`/`cosine`) assume a uniformly-spaced coordinate axis.

## Coord-separable sampling and grid evaluation

When you sample a `CoordSeparableBatch`, selected geometry labels provide a **tuple of 1D coordinate axes**
instead of a point cloud. For a 2D geometry label `"x"`, the model/operator receives
\((x_{\text{axis}}, y_{\text{axis}})\) and evaluates on the implied Cartesian grid.

This is the preferred mode for spectral operators and neural operators (FNO/DeepONet).

!!! example
    Laplacian on a periodic 1D grid using the basis backend:

    ```python
    import jax.random as jr
    import jax.numpy as jnp
    import phydrax as phx

    geom = phx.domain.Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return jnp.sin(2.0 * jnp.pi * x[0])

    lap_u = phx.operators.laplacian(u, var="x", backend="basis", basis="fourier")

    batch = geom.component().sample_coord_separable({"x": phx.domain.FourierAxisSpec(64)}, key=jr.key(0))
    out = lap_u(batch)
    ```

## Surface differential operators

Several operators act on fields restricted to a geometry boundary (surface/curve). These use the
outward unit normal $n$ provided by the geometry and project ambient derivatives onto the tangent
space.

Let $\Gamma = \partial\Omega_x$ be a smooth boundary in $\mathbb{R}^d$ with unit normal
$n(x)\in\mathbb{R}^d$. Define the tangential projector

$$
P(x) = I - n(x)\,n(x)^\top.
$$

For a scalar field $u$, the **surface gradient** is

$$
\nabla_\Gamma u = P\,\nabla u.
$$

For a (tangent) vector field $v$, the **surface divergence** is

$$
\nabla_\Gamma\cdot v = \text{tr}\!\left(P\,\nabla v\right).
$$

The **Laplace–Beltrami** operator is the surface analogue of the Laplacian:

$$
\Delta_\Gamma u = \nabla_\Gamma\cdot(\nabla_\Gamma u).
$$

In Phydrax, these operators are exposed as `surface_grad`, `surface_div`, and
`laplace_beltrami` (see [API → Operators → Differential](api/operators/differential.md)).

!!! note
    Surface operators are intended to be evaluated on boundary components so that the geometry
    can supply consistent normals.

## Fractional operators

Phydrax includes a small set of fractional derivative operators, primarily for experimentation.

### Fractional Laplacian (integral estimator)

For $0<\alpha<2$, the fractional Laplacian in $\mathbb{R}^d$ can be written (up to a constant
$C_{d,\alpha}$) as a singular integral:

$$
(-\Delta)^{\alpha/2}u(x)
\propto \int_{\mathbb{R}^d}\frac{u(x)-u(y)}{\|x-y\|^{d+\alpha}}\,dy.
$$

`phydrax.operators.fractional_laplacian` implements a **truncated** ball estimator using offsets
$y=x+\xi$ with $\|\xi\|\le R$:

$$
\int_{B_R(0)} \frac{u(x)-u(x+\xi)}{\|\xi\|^{d+\alpha}}\,d\xi.
$$

The implementation excludes a small neighborhood $\|\xi\|\le\varepsilon$ to avoid the
singularity, and can optionally reduce variance for $\alpha>1$ by subtracting a first-order
correction involving $\nabla u$ (`desingularize=True`).

!!! warning
    The returned value is *not* normalized by $C_{d,\alpha}$, and the truncation radius $R$
    introduces a modeling choice. Use this operator with care and validate against known
    solutions.

### Grünwald–Letnikov (Monte Carlo / GMC)

For one-sided fractional derivatives (currently $\alpha\in(1,2)$), Phydrax provides a Monte Carlo
variant of a Grünwald–Letnikov discretization; see `fractional_derivative_gl_mc` and related
helpers on the API page.
