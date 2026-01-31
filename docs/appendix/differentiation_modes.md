# Differentiation modes

This appendix gives a deeper mathematical explanation of the **differentiation modes** available in Phydrax’s
differential operators. Here “differentiation” refers to derivatives with respect to **labeled domain variables**
(e.g. space `"x"` or time `"t"`), as used by operators like `grad`, `partial_n`, `laplacian`, etc.

For the definitions and tensor shape conventions of the operators themselves, see
[Guides → Differential operators](../guides_differential.md).

## 0. Summary of user-facing knobs

Many differential operators share two primary keywords:

- `backend`: selects *how* the derivative is computed:
  - `"ad"`: automatic differentiation,
  - `"jet"`: Taylor-mode AD (derivative jets) for higher-order pure derivatives,
  - `"fd"`: finite differences on coord-separable grids (with AD fallback),
  - `"basis"`: basis-aware (spectral / interpolation) derivatives on coord-separable grids (with AD fallback).
- `mode`: selects the autodiff *direction* when `backend="ad"`:
  - `"reverse"`: reverse-mode (`jax.jacrev`),
  - `"forward"`: forward-mode (`jax.jacfwd`).

Some operators additionally accept:

- `basis ∈ {"poly","fourier","sine","cosine"}` when `backend="basis"`;
- `periodic ∈ {False,True}` when `backend="fd"`.

!!! note
    `backend="fd"` and `backend="basis"` require **coord-separable** (structured-grid) evaluation for the variable being
    differentiated. When given point batches, these backends fall back to `"ad"`.

## 1. Notation

Fix a labeled product domain \(\Omega=\Omega_x\times\Omega_t\times\cdots\). For a geometry label \(x\in\mathbb{R}^d\),
write \(x=(x_1,\dots,x_d)\). For a scalar label \(t\in\mathbb{R}\), write \(t\).

Locally, a `DomainFunction` can be viewed as a map

$$
u:\Omega\to \mathbb{R}^m,
$$

where \(m\) may itself represent a multi-axis tensor value; derivatives are taken componentwise with respect to the
domain variable(s).

For a smooth map \(f:\mathbb{R}^d\to\mathbb{R}^m\), let \(J_f(x)\in\mathbb{R}^{m\times d}\) denote the Jacobian, and
for a direction \(v\in\mathbb{R}^d\) define the \(k\)-th **directional derivative** along \(v\) by

$$
\partial_v^k f(x) \;:=\; \left.\frac{d^k}{d\epsilon^k} f(x+\epsilon v)\right|_{\epsilon=0}.
$$

For \(k=1\), \(\partial_v f(x)=J_f(x)\,v\). For \(v=e_i\) (a coordinate axis), \(\partial_{e_i}^k f\) reduces to the
pure partial derivative \(\partial^k f/\partial x_i^k\) (componentwise).

## 2. `backend="ad"`: automatic differentiation

The AD backend computes derivatives by applying the chain rule through the program that evaluates \(u\). It is the
default and works for both point sampling and coord-separable sampling.

### 2.1 Forward- vs reverse-mode AD (`mode="forward"` / `"reverse"`)

The core linear maps behind AD are:

**JVP** (Jacobian–vector product). For a tangent direction $v$,

$$
\mathrm{JVP}_f(x; v) = J_f(x)\,v.
$$

**VJP** (vector–Jacobian product). For a cotangent $w$,

$$
\mathrm{VJP}_f(x; w) = w^\top J_f(x)\quad(\text{equivalently }J_f(x)^\top w).
$$

Forward-mode AD is efficient when you need many JVPs (few input dimensions), while reverse-mode AD is efficient when
you need many VJPs (few output dimensions). When constructing full Jacobians, Phydrax exposes this choice as:

- `mode="forward"`: `jax.jacfwd` (conceptually “columns” via JVPs),
- `mode="reverse"`: `jax.jacrev` (conceptually “rows” via VJPs).

Rule of thumb for pointwise Jacobians \(J_f\in\mathbb{R}^{m\times d}\):

- prefer **reverse-mode** when \(m\ll d\) (e.g. scalar outputs),
- prefer **forward-mode** when \(d\ll m\).

### 2.2 Higher derivatives via nested transforms

Second derivatives can be formed by nesting Jacobian transforms. For example, for scalar-valued \(u\) and geometry
variable \(x\in\mathbb{R}^d\),

$$
\nabla^2 u(x) \;=\; \nabla_x(\nabla_x u(x)),
$$

and the Laplacian is the trace

$$
\Delta u(x)\;=\;\mathrm{tr}\bigl(\nabla^2 u(x)\bigr)=\sum_{i=1}^d \frac{\partial^2 u}{\partial x_i^2}(x).
$$

Nesting AD is exact up to floating-point and provides mixed partials \(\partial_{x_i x_j}u\) as well, but it can
become expensive when building large Hessians or higher-order derivatives.

### 2.3 Complex-valued outputs

For complex-valued outputs, Phydrax treats differentiation as real differentiation of the pair
\((\operatorname{Re}u,\operatorname{Im}u)\), i.e. it computes Jacobians for real and imaginary parts separately and
recombines them. This avoids relying on holomorphic assumptions and matches the “differentiate \(\mathbb{R}^2\)”
interpretation of complex numbers.

## 3. `backend="jet"`: Taylor-mode AD (derivative jets)

The jet backend computes **higher-order directional derivatives** by propagating a truncated Taylor series through
the computation.

Fix a smooth map \(f\) and a direction \(v\). Consider the 1D curve \(y(\epsilon)=f(x+\epsilon v)\). Its Taylor
expansion is

$$
f(x+\epsilon v)
= f(x) + \sum_{k=1}^{K}\frac{\epsilon^k}{k!}\,\partial_v^k f(x) + O(\epsilon^{K+1}).
$$

The jet backend returns the coefficients (directional derivatives)
\(\bigl(\partial_v^k f(x)\bigr)_{k=1}^K\) directly, using JAX’s jet machinery (whose internal chain rules are
governed by Faà di Bruno / Bell polynomial combinatorics).

### 3.1 Pure partial derivatives as directional derivatives

Pure \(n\)-th partial derivatives along a coordinate axis are directional derivatives with \(v=e_i\). For a geometry
variable \(x\in\mathbb{R}^d\),

$$
\frac{\partial^n f}{\partial x_i^n}(x) \;=\; \partial_{e_i}^n f(x).
$$

Similarly, for a scalar variable \(t\),

$$
\frac{d^n}{dt^n} f(t)\;=\;\left.\frac{d^n}{d\epsilon^n} f(t+\epsilon)\right|_{\epsilon=0}.
$$

This is exactly the situation targeted by `partial_n(..., backend="jet")`, which is why jets are a natural fit for
high-order derivatives with respect to a **single** labeled variable and a **single** axis.

### 3.2 What jets do (and do not) give you cheaply

Jets provide efficient access to *pure* derivatives like \(\partial_{x_i}^n u\). Mixed partials like
\(\partial_{x_i}\partial_{x_j}u\) with \(i\neq j\) are not directly obtained from a single 1D directional expansion;
they require genuinely multi-directional information (and are typically constructed via nested AD / Hessians).

## 4. `backend="fd"`: finite differences on coord-separable grids

The finite-difference backend treats the field as sampled on a structured grid and replaces derivatives by local
stencils. It is only used when the differentiated variable is provided as a tuple of 1D coordinate axes (coord-separable
evaluation).

### 4.1 First derivative (uniform grid)

Let \(x_i=x_0+i h\) for \(i=0,\dots,N-1\), and let \(u_i\approx u(x_i)\). A common second-order central difference is

$$
u'(x_i)\approx \frac{u_{i+1}-u_{i-1}}{2h}.
$$

For non-periodic boundaries, one-sided differences are used at endpoints, e.g.

$$
u'(x_0)\approx \frac{u_1-u_0}{h},\qquad
u'(x_{N-1})\approx \frac{u_{N-1}-u_{N-2}}{h}.
$$

With `periodic=True`, the stencil wraps around and the central formula applies at all indices (circular shifts).

### 4.2 Higher derivatives and multi-D tensor grids

Higher derivatives are obtained by repeated application of the 1D difference operator along the chosen axis.
On a tensor-product grid in \(d\) dimensions, “differentiate w.r.t. \(x_i\)” means “apply the 1D stencil along the
corresponding tensor axis while holding the other indices fixed”, i.e. a Kronecker-structured operator of the form

$$
D_{x_i}\;\approx\; I\otimes\cdots\otimes D \otimes\cdots\otimes I.
$$

The Laplacian is then approximated by the sum of second derivatives along each axis:

$$
\Delta u \;\approx\; \sum_{i=1}^{d} \frac{\partial^2 u}{\partial x_i^2}.
$$

!!! note
    The FD backend in Phydrax assumes **uniform spacing** along the differentiated axis and uses \(h=x_1-x_0\) from the
    provided coordinate array.

## 5. `backend="basis"`: basis-aware derivatives on structured grids

The basis backend differentiates an **implicit reconstruction** of the function along each coord-separable axis:

- spectral derivatives (Fourier / sine / cosine) via FFT on uniform grids, or
- polynomial interpolation derivatives via barycentric differentiation on generic 1D grids.

In all cases, the derivative along a chosen axis is a *linear map* applied to the sampled values along that axis.

### 5.1 Fourier spectral derivatives (`basis="fourier"`)

Assume a periodic interval of length \(L\) and expand

$$
u(x) = \sum_{k\in\mathbb{Z}} \hat u_k\,e^{i 2\pi k x/L}.
$$

Then

$$
u^{(n)}(x) = \sum_{k\in\mathbb{Z}} \left(i\frac{2\pi k}{L}\right)^n \hat u_k\,e^{i 2\pi k x/L}.
$$

On a uniform grid, the coefficients \(\hat u_k\) are approximated by the FFT, and differentiation becomes elementwise
multiplication by \(\left(i k\right)^n\) in frequency space (up to the scaling that depends on the physical grid spacing).

### 5.2 Sine/cosine bases (`basis="sine"` / `"cosine"`)

Sine and cosine bases can be understood as Fourier methods applied after symmetry extension:

- cosine: even extension (cosine series),
- sine: odd extension (sine series).

The implementation performs the extension to a periodic array, differentiates via FFT in the extended domain, and then
restricts back to the original grid.

### 5.3 Polynomial / barycentric differentiation (`basis="poly"`)

Let nodes \(x_0,\dots,x_{N-1}\) be distinct (not necessarily uniform), with values \(u_i=u(x_i)\). The barycentric
weights are

$$
w_i = \left(\prod_{j\ne i}(x_i-x_j)\right)^{-1}.
$$

Define the first-derivative differentiation matrix \(D\in\mathbb{R}^{N\times N}\) by

$$
D_{ij} =
\begin{cases}
\dfrac{w_j}{w_i(x_i-x_j)} & i\ne j,\\
-\sum\limits_{j\ne i} D_{ij} & i=j.
\end{cases}
$$

Then the interpolant derivative at nodes satisfies

$$
\bigl(p'(x_0),\dots,p'(x_{N-1})\bigr)^\top = D\,(u_0,\dots,u_{N-1})^\top,
$$

where \(p\) is the unique polynomial interpolant with \(p(x_i)=u_i\).
Higher derivatives can be obtained by repeated application of such differentiation matrices.

!!! note
    Polynomial differentiation can be sensitive to node placement for large \(N\). For non-periodic smooth problems on
    \([-1,1]\), Chebyshev-like nodes typically behave far better than uniform nodes.

## 6. Practical guidance (what to use when)

Common choices:

- **Point collocation (PINN-style)**: `backend="ad"` (often with `mode="reverse"` for scalar-valued PDE residuals).
- **Structured grids / neural operators**:
  - use `backend="basis"` for smooth fields matching the basis assumptions (especially periodic Fourier grids),
  - use `backend="fd"` for local, stencil-based discretizations or less-smooth signals.
- **High-order derivatives in one variable** (especially time derivatives of order \(n\ge 2\)): `backend="jet"` often
  avoids the overhead of deeply nested Jacobian constructions.
