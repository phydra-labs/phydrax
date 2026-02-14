# Differential operators

!!! note
    For a more detailed mathematical guide (notation, shapes, and backend behavior), see
    [Guides → Differential operators](../../guides_differential.md).

## Backends (AD / jet / FD / basis / MFD)

Many differential operators accept a `backend` keyword:

- `backend="ad"`: autodiff (works for point batches and coord-separable batches).
- `backend="jet"`: Taylor-mode AD ("jets") for higher-order derivatives with respect to a single variable.
  See the guide for mathematical details (including Faà di Bruno / Bell polynomial structure).
- `backend="fd"`: finite differences on coord-separable grids (falls back to AD on point batches).
  Use `periodic=True` for periodic stencils.
- `backend="basis"`: basis-aware methods on coord-separable grids (falls back to AD on point batches).
  The `basis` keyword selects a 1D method per axis:
  - `basis="fourier"`: FFT-based spectral derivatives on periodic grids,
  - `basis="sine"` / `basis="cosine"`: FFT-based spectral derivatives via odd/even extension
    (sine \(\leftrightarrow\) odd extension, cosine \(\leftrightarrow\) even extension),
  - `basis="poly"`: polynomial (barycentric) derivatives for generic 1D grids.
- `backend="mfd"`: meshless finite differences with two-sided interior stencils and
  explicit boundary closure modes (`hybrid`, `ghost`, `biased`).

!!! note
    FFT-based bases (`fourier`/`sine`/`cosine`) assume a *uniformly-spaced* coordinate axis.
    For non-uniform axes, prefer `basis="poly"` or `backend="ad"`.

## Common keyword arguments

Many operators share these keywords:

- `var`: label to differentiate with respect to (e.g. `"x"` or `"t"`). If omitted, the variable
  is inferred when possible.
- `mode`: autodiff mode (`"reverse"` uses `jax.jacrev`, `"forward"` uses `jax.jacfwd`) for AD-based paths.
- `backend`: `"ad"`, `"jet"`, `"fd"`, `"basis"`, or `"mfd"` (see above).
- `basis`: basis method used when `backend="basis"`.
- `periodic`: periodic treatment for FD/MFD stencils (used when `backend in {"fd","mfd"}`).
- `mfd_boundary_mode`: boundary closure policy when `backend="mfd"`:
  `"hybrid"` (default), `"ghost"`, or `"biased"`.
- `mfd_stencil_size`: local stencil/support size for MFD (evaluation-time knob).
- `mfd_step`: probe spacing for point-input MFD evaluation (evaluation-time knob).
- `mfd_mode`: point-input MFD mode, `"probe"` (default) or `"cloud"`.
- `mfd_cloud_plan`: fixed-cloud stencil plan built with
  `phydrax.operators.build_mfd_cloud_plan(...)` (required for `mfd_mode="cloud"`).

See the guide for operator shape conventions and for the math behind surface and fractional operators.

!!! note
    The `mfd_*` options are consumed at operator evaluation time (e.g. when calling the
    returned `DomainFunction`), so they can be tuned per-evaluation without rebuilding
    the symbolic operator.

!!! note
    `mfd_mode="cloud"` is currently a fixed-cloud point path for 1D variables
    (`var_dim == 1`). Tuple/coord-separable MFD remains unchanged.

## MFD cloud utilities

Use these helpers to precompute and reuse fixed-cloud stencils for point-input MFD.

::: phydrax.operators.build_mfd_cloud_plan

---

::: phydrax.operators.MFDCloudPlan

## Core derivatives

::: phydrax.operators.grad

---

::: phydrax.operators.div

---

::: phydrax.operators.curl

---

::: phydrax.operators.hessian

---

::: phydrax.operators.laplacian

---

::: phydrax.operators.bilaplacian

---

::: phydrax.operators.dt

---

::: phydrax.operators.dt_n

---

::: phydrax.operators.directional_derivative

---

::: phydrax.operators.material_derivative

---

::: phydrax.operators.partial_t

---

::: phydrax.operators.partial_x

---

::: phydrax.operators.partial_y

---

::: phydrax.operators.partial_z

---

::: phydrax.operators.partial_n

## Surface operators

::: phydrax.operators.surface_grad

---

::: phydrax.operators.surface_div

---

::: phydrax.operators.surface_curl_scalar

---

::: phydrax.operators.surface_curl_vector

---

::: phydrax.operators.tangential_component

---

::: phydrax.operators.laplace_beltrami

---

::: phydrax.operators.laplace_beltrami_divgrad

## Fractional derivatives

::: phydrax.operators.fractional_laplacian

---

::: phydrax.operators.fractional_derivative_gl_mc

---

::: phydrax.operators.riesz_fractional_derivative_gl_mc

---

::: phydrax.operators.caputo_time_fractional

---

::: phydrax.operators.caputo_time_fractional_dw

## Continuum mechanics

::: phydrax.operators.deformation_gradient

---

::: phydrax.operators.green_lagrange_strain

---

::: phydrax.operators.cauchy_strain

---

::: phydrax.operators.strain_rate

---

::: phydrax.operators.strain_rate_magnitude

---

::: phydrax.operators.pk1_from_pk2

---

::: phydrax.operators.cauchy_from_pk2

---

::: phydrax.operators.cauchy_stress

---

::: phydrax.operators.viscous_stress

---

::: phydrax.operators.von_mises_stress

---

::: phydrax.operators.hydrostatic_stress

---

::: phydrax.operators.hydrostatic_pressure

---

::: phydrax.operators.deviatoric_stress

---

::: phydrax.operators.maxwell_stress

---

::: phydrax.operators.neo_hookean_pk1

---

::: phydrax.operators.neo_hookean_cauchy

---

::: phydrax.operators.svk_pk2_stress

---

::: phydrax.operators.linear_elastic_cauchy_stress_2d

---

::: phydrax.operators.linear_elastic_orthotropic_stress_2d

---

::: phydrax.operators.div_tensor

---

::: phydrax.operators.div_cauchy_stress

## Navier–Stokes helpers

::: phydrax.operators.navier_stokes_stress

---

::: phydrax.operators.navier_stokes_divergence
