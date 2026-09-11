# Special functions

`phydrax.special` provides JAX-native named special functions and integrals with
fixed mathematical, domain, dtype, boundary, and differentiation contracts.
These kernels do not invoke `phydrax.integration`; use the integration API for
runtime-defined integrands and measures.

Float16 and bfloat16 real inputs are evaluated as float32. Float32 and float64
remain unchanged for real-valued outputs. Principal-branch helpers, Hankel,
Faddeeva, and spherical-harmonic outputs map those widths to complex64 and
complex128. Numerical array arguments broadcast by NumPy/JAX rules; arguments
identified below as structural remain static. Public functions are not
pre-jitted, so callers control compilation boundaries.

See [Special functions and named integrals](../guides_special_functions.md) for
parameter conventions, scaling definitions, derivative support, numerical
regimes, and application examples.

## Standard normal distribution

The normal helpers preserve real dtype promotion and remain JIT-, VMAP-, and
gradient-compatible. `normal_logcdf` and `normal_logsurvival` avoid destructive
tail subtraction; `normal_quantile` maps the closed probability endpoints to
signed infinity and returns NaN outside the probability domain.

::: phydrax.special.normal_pdf

---

::: phydrax.special.normal_logpdf

---

::: phydrax.special.normal_cdf

---

::: phydrax.special.normal_logcdf

---

::: phydrax.special.normal_survival

---

::: phydrax.special.normal_logsurvival

---

::: phydrax.special.normal_quantile

## Principal branch helpers

`principal_log` uses `Arg z` in `(-pi, pi]`; `principal_sqrt` is its
principal-square-root counterpart. Real inputs are deliberately promoted to a
complex dtype, so negative real values follow the principal continuation
rather than a real-domain `NaN`.

::: phydrax.special.principal_log
    options:
      show_root_heading: true

::: phydrax.special.principal_sqrt
    options:
      show_root_heading: true

## Carlson symmetric integrals

::: phydrax.special.elliprc
    options:
      show_root_heading: true

::: phydrax.special.elliprf
    options:
      show_root_heading: true

::: phydrax.special.elliprd
    options:
      show_root_heading: true

::: phydrax.special.elliprj
    options:
      show_root_heading: true

::: phydrax.special.elliprg
    options:
      show_root_heading: true

## Legendre elliptic integrals

All Legendre functions use the parameter `m = k**2`, not the modulus `k`.
Incomplete forms use an unwrapped amplitude `phi` in radians.

The complete third-kind integral is `ellippi(n, m)`. Its incomplete counterpart
is `ellippiinc(n, phi, m)`.

::: phydrax.special.ellipk
    options:
      show_root_heading: true

::: phydrax.special.ellipkm1
    options:
      show_root_heading: true

::: phydrax.special.ellipe
    options:
      show_root_heading: true

::: phydrax.special.ellipkinc
    options:
      show_root_heading: true

::: phydrax.special.ellipeinc
    options:
      show_root_heading: true

::: phydrax.special.ellippi
    options:
      show_root_heading: true

::: phydrax.special.ellippiinc
    options:
      show_root_heading: true

## Jacobi elliptic functions

::: phydrax.special.ellipj
    options:
      show_root_heading: true

::: phydrax.special.ellipam
    options:
      show_root_heading: true

## Airy functions

::: phydrax.special.airy
    options:
      show_root_heading: true

::: phydrax.special.airye
    options:
      show_root_heading: true

## Modified Bessel functions

The order `v` and argument `x` are real and nonnegative. Automatic
differentiation supports tangents in both `v` and `x`; the explicit
`*_order_derivative` functions expose the principal-continuation order
derivative directly, including stable exact- and near-integer limits.

::: phydrax.special.iv
    options:
      show_root_heading: true

::: phydrax.special.ive
    options:
      show_root_heading: true

::: phydrax.special.kv
    options:
      show_root_heading: true

::: phydrax.special.kve
    options:
      show_root_heading: true

::: phydrax.special.iv_order_derivative
    options:
      show_root_heading: true

::: phydrax.special.ive_order_derivative
    options:
      show_root_heading: true

::: phydrax.special.kv_order_derivative
    options:
      show_root_heading: true

::: phydrax.special.kve_order_derivative
    options:
      show_root_heading: true

## Cylindrical Bessel and Hankel functions

The order `v` is real and nonnegative. `jv` accepts `x >= 0`; `yv` and the
Hankel functions have a positive-argument interior with explicit zero limits.
Automatic differentiation supports tangents in both the order and argument.
`jv_order_derivative` and `yv_order_derivative` expose the
principal-continuation order derivatives directly, including integer-order
limits.

::: phydrax.special.jv
    options:
      show_root_heading: true

::: phydrax.special.yv
    options:
      show_root_heading: true

::: phydrax.special.hankel1
    options:
      show_root_heading: true

::: phydrax.special.hankel2
    options:
      show_root_heading: true

::: phydrax.special.jv_order_derivative
    options:
      show_root_heading: true

::: phydrax.special.yv_order_derivative
    options:
      show_root_heading: true

Fixed-order cylindrical aliases and all-order spherical-Bessel families are
not public APIs. Private radial consumers own those sequences without adding
`j0`/`j1`, `spherical_jn`, modified-spherical, or spherical-Hankel aliases.

## Gegenbauer polynomials

Degrees are static nonnegative integers. `alpha` is a real, broadcast
numerical argument with `alpha > -1/2`; `x` may be real or complex. The scalar
and Vandermonde forms use standard $C_n^{(\alpha)}$ normalization, including
the exact value and nonzero alpha-derivative limits at `alpha=0`. They do not
provide noninteger-degree or generalized-function continuation.

::: phydrax.special.gegenbauer_c
    options:
      show_root_heading: true

::: phydrax.special.gegenbauer_vander
    options:
      show_root_heading: true

::: phydrax.special.gegenbauer_alpha_derivative
    options:
      show_root_heading: true

## Zeta and polylogarithm families

`zeta` and `hurwitz_zeta` share an Euler--Maclaurin continuation and its
analytic derivatives. The Riemann pole and trivial zeros retain their
meromorphic derivative semantics. Hurwitz parameters require positive real
part.

`dilog` and `spence` use the principal complex branch. `polylog` returns
complex values and admits general order only on its documented bounded disk;
unsupported lanes are `NaN`, never a projected real value.

::: phydrax.special.zeta
    options:
      show_root_heading: true

::: phydrax.special.hurwitz_zeta
    options:
      show_root_heading: true

::: phydrax.special.dilog
    options:
      show_root_heading: true

::: phydrax.special.spence
    options:
      show_root_heading: true

::: phydrax.special.polylog
    options:
      show_root_heading: true

## Scalar spherical harmonics

These functions use orthonormal angular normalization with the Condon--Shortley
phase. `theta` is the polar angle and `phi` the azimuth. Degree `n` and order
`m` are static integral structure with `n >= 0` and `abs(m) <= n`; only the
numerical angle or direction arguments broadcast.

`sph_harm_y_cart` accepts real arrays ending in a three-vector. It stably
normalizes each vector, is invariant to positive rescaling, and returns a
lane-local complex `NaN` for a zero or nonfinite vector. Direct Cartesian
evaluation retains finite, correct derivatives on the z-axis instead of
differentiating through the singular angular chart. It remains an angular
harmonic, not the solid harmonic `r**n * Y_n^m`, and no public all-mode table is
provided.

Low-width real inputs are evaluated as float32.
`sph_legendre_p` returns float32 or float64; `sph_harm_y` and
`sph_harm_y_cart` return complex64 or complex128.

::: phydrax.special.sph_legendre_p
    options:
      show_root_heading: true

::: phydrax.special.sph_harm_y
    options:
      show_root_heading: true

::: phydrax.special.sph_harm_y_cart
    options:
      show_root_heading: true

## Solid harmonics

Regular and irregular solid harmonics share the scalar orthonormal
Condon--Shortley basis above. Degree and order are static; real Cartesian
vectors end in length three and broadcast over leading axes. Regular modes are
homogeneous Cartesian polynomials with their analytic origin values.
Irregular modes are singular at the origin and return lane-local complex
`NaN` for zero or nonfinite vectors. No public all-mode table or spin-valued
pairwise alias is provided.

::: phydrax.special.solid_harmonic_regular
    options:
      show_root_heading: true

::: phydrax.special.solid_harmonic_irregular
    options:
      show_root_heading: true

## Faddeeva, Dawson, and Voigt functions

::: phydrax.special.wofz
    options:
      show_root_heading: true

::: phydrax.special.dawsn
    options:
      show_root_heading: true

::: phydrax.special.voigt_profile
    options:
      show_root_heading: true
