# Special functions

`phydrax.special` provides JAX-native named special functions and integrals with
fixed mathematical, domain, dtype, boundary, and differentiation contracts.
These kernels do not invoke `phydrax.integration`; use the integration API for
runtime-defined integrands and measures.

Float16 and bfloat16 real inputs are evaluated as float32. Float32 and float64
remain unchanged. Hankel and Faddeeva outputs map those dtypes to complex64 and
complex128. Arguments broadcast by NumPy/JAX rules. Public functions are not
pre-jitted, so callers control compilation boundaries.

See [Special functions and named integrals](../guides_special_functions.md) for
parameter conventions, scaling definitions, derivative support, numerical
regimes, and application examples.

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

The order `v` and argument `x` are real and nonnegative. Differentiation is
supported with respect to `x`; an order tangent raises `TypeError`.

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

## Cylindrical Bessel and Hankel functions

The order `v` is real and nonnegative. `jv` accepts `x >= 0`; `yv` and the
Hankel functions have a positive-argument interior with explicit zero limits.
Differentiation is supported with respect to `x`; an order tangent raises
`TypeError`.

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
