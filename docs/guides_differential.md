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

## Vector-field Lie bracket

For vector fields $X,Y:\Omega_x\to\mathbb R^d$,
`phydrax.operators.lie_bracket(X, Y, var="x")` constructs

$$
[X,Y]=D_XY-D_YX.
$$

Both outputs must be vectors whose size equals the dimension of the selected geometry
variable. This is the geometric Lie bracket of vector fields, not the matrix
`commutator` used for quantum operators. See
[Quantum operators and dynamics](guides_quantum.md) for the distinction among
vector-field, matrix, and canonical Poisson brackets.

## Stochastic generators and adjoints

For \(dX=b\,dt+\sigma\,dW\), set
\(a=\sigma\sigma^\mathsf T\). The backward generator and its density
adjoint are

$$
\mathcal Lu=b_i\partial_i u+\frac12a_{ij}\partial_{ij}u,
\qquad
\mathcal L^\ast p
=-\partial_i(b_i p)+\frac12\partial_i\partial_j(a_{ij}p).
$$

`kolmogorov_generator` and `fokker_planck_operator` differentiate with
respect to the selected state geometry while retaining time/parameter
dependencies. Observable outputs may be scalar or tensor-valued; the generator
acts componentwise. Densities must be scalar-valued.

For Stratonovich input, Phydrax applies the Euclidean drift correction

$$
b_i^I=b_i^S+\frac12\sum_{j,k}\sigma_{jk}\partial_j\sigma_{ik}.
$$

The diffusion factor is required in this mode; covariance alone does not
identify the correction. See
[API → Operators → Differential](api/operators/differential.md) for shapes and
[API → Constraints → Continuous](api/constraints/continuous.md) for stationary
and time-dependent residual constructors.

## Backends: autodiff, finite differences, spectral/basis

Many differential operators accept a `backend` keyword:

- `backend="ad"` uses autodiff and works for both point sampling and coord-separable sampling.
- `backend="jet"` uses Taylor-mode AD ("jets") for higher-order derivatives with respect to a single variable.
- `backend="fd"` uses finite differences on coord-separable grids (and falls back to autodiff for point inputs).
- `backend="basis"` uses basis-aware methods on coord-separable grids (and falls back to autodiff for point inputs).

!!! note
    For `LatentContractionModel` wrapped via `domain.Model(...)`, `partial`,
    `partial_n`, `dt_n`, and `laplacian` may take an exact latent-factor derivative
    contraction route under `backend="jet"`. For `backend="ad"`, Phydrax stays on
    AD derivatives (or directional JVP if `ad_engine="jvp"` is explicitly selected).
    The latent contraction route is an acceleration path (not an approximation): if
    preconditions are not met, Phydrax falls back to the generic derivative path and
    applies the model's configured fallback policy (`warn`, `error`, or `silent`).

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

When you sample a `CoordSeparableBatch`, selected labels provide a **tuple of 1D coordinate axes**
instead of a point cloud. For a 2D geometry label `"x"`, the model/operator receives
\((x_{\text{axis}}, y_{\text{axis}})\); for a scalar label (e.g. `"t"`), the tuple has one axis.

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

Here **surface** means one smooth boundary component of an embedded geometry.
This is distinct from the additive face collection returned by a product domain's
`boundary()` method and from signed simplicial/cochain boundary maps. Normal-based
surface operators require a single `DomainComponent`, not a
`DomainComponentUnion`.

Let \(\Gamma=\partial\Omega_x\) be embedded in \(\mathbb{R}^d\), with outward
unit normal \(n\) and tangent projector

$$
P = I - n n^\top.
$$

For an ambient scalar field \(u\) and vector field \(v\), Phydrax defines

$$
\nabla_\Gamma u = P\nabla u,
\qquad
\nabla_\Gamma\cdot v = \operatorname{tr}(P\nabla v).
$$

These are exposed as `surface_grad` and `surface_div`; use
`tangential_component` to apply \(P\) directly to a vector field. In three
ambient dimensions, the two surface-curl conventions are explicit:

$$
\operatorname{curl}_\Gamma u = n\times\nabla_\Gamma u,
\qquad
\operatorname{curl}_\Gamma v = n\cdot(\nabla\times v),
$$

implemented by `surface_curl_scalar` and `surface_curl_vector`.

Operator composition follows the normal provider's autodiff contract. Analytic
normal fields may contribute curvature derivatives; mesh-derived normals are
intentionally nondifferentiable. Use metric-coordinate calculus whenever an
exact curved-manifold identity is required.

`ambient_surface_hessian_trace(u, component)` computes
\(\operatorname{tr}(P(\nabla^2u)P)\). This is an **ambient,
extension-dependent contraction**. It equals the intrinsic
Laplace--Beltrami operator only when the ambient extension is compatible with
the surface, such as a closest-point extension or a flat surface.

For a field expressed in local manifold coordinates, the intrinsic operator is

$$
\Delta_g u =
\frac{1}{\sqrt{\lvert g\rvert}}
\partial_i\left(\sqrt{\lvert g\rvert}\,g^{ij}\partial_j u\right).
$$

Call `laplace_beltrami(u, metric)` with a
`phydrax.metrix.RiemannianMetric`. It does not accept a boundary component or
use ambient normals. See
[API → Metrix → Connections and intrinsic operators](api/metrix/connections.md)
and [API → Operators → Differential](api/operators/differential.md).

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
