# Special functions and named integrals

`phydrax.special` contains named numerical primitives with fixed mathematical
contracts. Use it when the function itself is the object of computation. Use
`phydrax.integration` when a user-defined integrand, measure, and numerical plan
must be composed at runtime.

Numerical array arguments accept Python scalars and JAX arrays, broadcast,
compose with `jax.jit` and `jax.vmap`, and use branch-safe fixed-iteration
kernels. Arguments identified below as structural remain static. Analytic
custom JVPs are used where differentiating an approximation branch would be
unstable.

```python
import jax
import jax.numpy as jnp
import phydrax as phx

m = jnp.array([0.0, 0.5, 0.9])
k = phx.special.ellipk(m)
dk = jax.vmap(jax.grad(phx.special.ellipk))(m)
```

## Conventions at a glance

| Family | Public functions | Convention and branch |
| --- | --- | --- |
| Principal branches | `principal_log`, `principal_sqrt` | principal logarithm and square root with complex promotion |
| Carlson | `elliprc`, `elliprf`, `elliprd`, `elliprj`, `elliprg` | nonnegative real domain or principal complex square-root continuation |
| Complete Legendre | `ellipk`, `ellipkm1`, `ellipe`, `ellippi` | parameter `m = k²`; principal complex continuation |
| Incomplete Legendre | `ellipkinc`, `ellipeinc`, `ellippiinc` | unwrapped real amplitude or principal complex Carlson continuation |
| Jacobi | `ellipj`, `ellipam` | fixed-depth descending AGM; principal complex square roots |
| Airy | `airy`, `airye` | entire Airy values; documented complex scaling for `airye` |
| Modified Bessel | `iv`, `ive`, `kv`, `kve` and their `*_order_derivative` functions | principal logarithm; `K` cut on the negative real axis |
| Cylindrical Bessel | `jv`, `yv`, `hankel1`, `hankel2`, `jv_order_derivative`, `yv_order_derivative` | principal logarithm; `Y`/Hankel cut on the negative real axis |
| Gegenbauer | `gegenbauer_c`, `gegenbauer_vander`, `gegenbauer_alpha_derivative` | standard $C_n^{(\alpha)}$ normalization for real $\alpha>-1/2$, including the exact $\alpha=0$ limit |
| Zeta and polylogarithms | `zeta`, `hurwitz_zeta`, `dilog`, `spence`, `polylog` | Euler--Maclaurin zeta continuation and principal complex polylogarithm branches |
| Spherical and solid harmonics | `sph_legendre_p`, `sph_harm_y`, `sph_harm_y_cart`, `solid_harmonic_regular`, `solid_harmonic_irregular` | orthonormal Condon--Shortley convention; polar `theta`, azimuthal `phi` |
| Faddeeva | `wofz`, `dawsn`, `voigt_profile` | complex Faddeeva/Dawson; `voigt_profile` remains real and nonholomorphic |

Complex64 and complex128 inputs retain their precision. Principal logarithm
uses `Arg z` in `(-pi, pi]`; signed zero selects the upper/lower lip of the
negative-real cut. Poles and cut crossings are not assigned fabricated finite
derivatives. Real invalid-domain lanes continue to return `NaN` independently.

## Carlson symmetric integrals

Carlson forms are the reusable elliptic core. The implementation uses scaled
symmetric duplication, preserving permutation symmetry and avoiding overflow
when all arguments share an extreme scale. Their homogeneity is

- `R_F(λx, λy, λz) = λ**(-1/2) R_F(x, y, z)`;
- `R_D` and `R_J` scale as `λ**(-3/2)`;
- `R_G` scales as `λ**(1/2)`.

```python
x = jnp.array([0.0, 0.2, 2.0])
rf = phx.special.elliprf(x, 1.0, 2.0)
rd = phx.special.elliprd(x, 1.0, 2.0)
```

At divergent nonnegative boundaries, the functions return positive infinity.
Negative arguments, a nonpositive `elliprc` second argument, and a nonpositive
`elliprj` fourth argument return `NaN`. Native JAX differentiation applies to
all admitted arguments.

## Legendre elliptic integrals

Phydrax follows SciPy's parameter convention:

```text
K(m), E(m), Pi(n | m), F(phi | m), E(phi | m), Pi(n; phi | m), where m = k².
```

`ellipkm1(p)` evaluates `K(1-p)` directly and switches to a logarithmic
expansion near `p = 0`, avoiding cancellation in `1-p`. Incomplete forms reduce
`phi` by whole periods, evaluate the principal segment through Carlson forms,
and restore exact complete-integral increments. This keeps large unwrapped
amplitudes meaningful for orbit and arc-length calculations.

```python
phi = jnp.linspace(-4.0 * jnp.pi, 4.0 * jnp.pi, 2048)
arc = phx.special.ellipeinc(phi, 0.7)
complete_third = phx.special.ellippi(0.2, 0.7)
third = phx.special.ellippiinc(0.2, phi, 0.7)
```

The complete first kind diverges at `m = 1`; the complete second kind equals
one there. `m > 1` is outside the real contract. Both `ellippi(n, m)` and
`ellippiinc(n, phi, m)` implement the pole-free real third-kind branch `n < 1`;
principal-value continuations across poles are not part of this API.

## Jacobi elliptic functions

`ellipj(u, m)` returns Jacobi `sn`, `cn`, `dn`, and the unwrapped amplitude
`am`. A fixed-depth descending Landen/AGM algorithm covers `0 < m < 1`;
parameter transformations cover `m < 0`; trigonometric and hyperbolic formulas
supply the exact `m = 0` and `m = 1` limits.

```python
sn, cn, dn, amplitude = phx.special.ellipj(jnp.linspace(0.0, 20.0, 1000), 0.8)
assert jnp.allclose(sn**2 + cn**2, 1.0)
assert jnp.allclose(dn**2 + 0.8 * sn**2, 1.0)
```

Both `u` and `m` are differentiable. The custom JVP uses the closed Jacobi
system for argument tangents and analytic parameter identities, including
finite limiting formulas at `m = 0` and `m = 1`.

## Airy functions

`airy(x)` returns `Ai(x)`, `Ai′(x)`, `Bi(x)`, and `Bi′(x)`. `airye(x)` uses
SciPy-compatible real-axis scaling. For positive `x`, with
`ζ = 2 x**(3/2) / 3`, it returns

```text
(exp(ζ) Ai, exp(ζ) Ai′, exp(-ζ) Bi, exp(-ζ) Bi′).
```

For nonpositive `x`, the scaled and ordinary functions agree. Central power
series, positive-axis scaled Chebyshev expansions, and oscillatory
negative-axis asymptotics are selected with continuity at their switches.
Analytic JVPs use the Airy equation, with cancellation-free asymptotic
derivative combinations for scaled positive arguments.

```python
x = jnp.linspace(-20.0, 20.0, 4096)
ai, aip, bi, bip = phx.special.airy(x)
wronskian = ai * bip - aip * bi  # 1 / pi
```

## Modified Bessel functions

The modified Bessel family provides ordinary and exponentially scaled values:

```text
ive(v, x) = exp(-x) I_v(x)
kve(v, x) = exp(x) K_v(x).
```

Small/moderate arguments use convergent series or Temme/continued-fraction
kernels. Large arguments use asymptotics, and large order uses an Olver uniform
expansion. Scaled forms should be preferred when an ordinary value would
underflow or overflow.

```python
v = jnp.array([0.0, 0.5, 10.0, 100.0])
x = jnp.array([1.0, 10.0, 100.0, 1000.0])
stable_i = phx.special.ive(v, x)
stable_k = phx.special.kve(v, x)
```

The real admitted contract remains `v >= 0`, `x >= 0`. Complex arguments use
the principal continuation. At zero, `I_0(0) = 1`, `I_v(0) = 0` for positive
real `v`, and `K_v(0) = +inf`. Argument derivatives use analytic recurrences.
`iv_order_derivative`, `ive_order_derivative`, `kv_order_derivative`, and
`kve_order_derivative` include stable exact/near-integer limits.

## Cylindrical Bessel and Hankel functions

`jv` and `yv` evaluate real `J_v(x)` and `Y_v(x)`. The Hankel functions are

```text
hankel1(v, x) = J_v(x) + i Y_v(x)
hankel2(v, x) = J_v(x) - i Y_v(x).
```

Power series, finite integral representations, stable order recurrence,
large-argument Hankel expansions, and large-order Airy-uniform expansions
cover the nonnegative real-order axis. Near the turning point `x ≈ v`, a
transition expansion prevents cancellation.

```python
radius = jnp.geomspace(1e-3, 100.0, 2048)
outgoing = phx.special.hankel1(0.0, radius)
radial_gradient = jax.vmap(jax.grad(lambda r: phx.special.jv(0.0, r)))(radius)
```

`J_0(0) = 1`, positive-real-order `J_v(0) = 0`, and `Y_v(0) = -inf` on the
real domain. Complex arguments follow the principal cut convention.
`jv_order_derivative` and `yv_order_derivative` support noninteger,
near-integer, exact-integer, negative, and mixed order/argument tangents.

Fixed-order cylindrical conveniences and all-order spherical-Bessel sequences
remain private radial-kernel implementation details. There are no public
`j0`/`j1`, `y0`/`y1`, `spherical_jn`, `spherical_yn`, modified-spherical, or
spherical-Hankel aliases; use the order-explicit public cylindrical functions
above unless a prepared application plan owns the radial sequence.

## Gegenbauer polynomials

`gegenbauer_c(n, alpha, x)` evaluates the standard Gegenbauer polynomial
$C_n^{(\alpha)}(x)$. The degree `n` is a static nonnegative integer. `alpha`
is real with `alpha > -1/2`; it broadcasts with real or complex `x`.
`gegenbauer_vander(alpha, x, degree)` returns all degrees from zero through the
static nonnegative `degree` on a final mode axis.

The value convention follows the generating function

```text
(1 - 2*x*t + t**2)**(-alpha)
  = sum(n >= 0, C_n^(alpha)(x) * t**n).
```

At `alpha=0` this generating function collapses exactly:
$C_0^{(0)}=1$ and $C_n^{(0)}=0$ for $n>0$. The parameter derivative does not
collapse with the value. `gegenbauer_alpha_derivative` and ordinary JAX
differentiation both return
$\left.\partial_\alpha C_n^{(\alpha)}(x)\right|_{\alpha=0}
=2T_n(x)/n$ for $n>0$ (and zero for `n=0`). The implementation propagates the
value and its alpha derivative through the same recurrence; it does not
differentiate a different normalization or substitute a nearby alpha.

The argument and alpha are numerical differentiable inputs. Noninteger degree,
`alpha <= -1/2`, complex alpha, and generalized Gegenbauer-function
continuations are outside the public contract. Spectral polynomial consumers
use private standard, monic, and orthonormal scalings plus explicitly prepared
quadrature and basis-connection operators; those internal operators do not add
a second scalar Gegenbauer API.

## Zeta, dilogarithm, and polylogarithm

`zeta(s)` and `hurwitz_zeta(s, a)` use one differentiated
Euler--Maclaurin substrate. `hurwitz_zeta` admits finite `a` with positive real
part; `zeta` adds the Riemann reflection formula on the negative half-plane.
The pole at `s=1` remains infinite, while the derivative at each trivial zero
is obtained from the reflected analytic expression rather than from a constant
zero branch. Exact host-side Bernoulli coefficients are private implementation
data.

`dilog(z)` is the principal complex `Li_2(z)` and `spence(z)` is exactly
`dilog(1-z)`. Signed imaginary zero selects the lip of the cut beginning at
`z=1`. Special values repair removable numerical cancellation without
replacing their analytic derivatives.

`polylog(s, z)` always returns a complex array. Its qualified general-order
envelope is

```text
abs(real(s)) <= 20, abs(imag(s)) <= 20, abs(z) <= 0.75.
```

`z=1` is additionally admitted when `real(s)>1`, where the value and
order derivative are inherited from `zeta`. The interior power series carries
value, order derivative, and argument derivative together. Unsupported lanes
return complex `NaN`; the function never projects a principal complex value
onto the real axis. `dilog` remains the wider-plane order-two API.

These functions are fixed-precision continuations, not arbitrary-precision
analytic-number-theory kernels. Complex Hurwitz parameters are restricted to
the positive-real-part half-plane, and general polylogarithm values outside the
qualified disk are deliberately refused.

## Scalar spherical harmonics

For static integral degree `n` and order `m`, with `n >= 0` and
`abs(m) <= n`, `sph_legendre_p` evaluates the orthonormal spherical Legendre
factor

```text
Pbar_n^m(cos(theta))
  = sqrt((2*n + 1)/(4*pi) * (n-m)!/(n+m)!) P_n^m(cos(theta))
```

for nonnegative `m`; negative orders use
`Pbar_n^(-m) = (-1)^m Pbar_n^m`. The associated Legendre function includes
the Condon--Shortley phase. Consequently,

```text
sph_harm_y(n, m, theta, phi)
  = Pbar_n^m(cos(theta)) * exp(1j*m*phi)
```

uses `theta` as the polar angle and `phi` as the azimuth, and satisfies
orthonormality over the unit sphere. `n` and `m` are structural rather than
array arguments: they must be known when JAX traces the call. The angle arrays
broadcast normally.

```python
theta = jnp.linspace(0.0, jnp.pi, 256)
zonal = phx.special.sph_harm_y(3, 0, theta, 0.0)
direction = jnp.array([2.0, -1.0, 4.0])
cartesian = phx.special.sph_harm_y_cart(3, 2, direction)
```

The angular recurrence forms integer powers of `sin(theta)`, so polar-axis
derivatives with respect to `theta` remain finite when the mathematical
derivative is finite. Converting Cartesian coordinates to `(theta, phi)` before
differentiating still introduces the chart singularity on the z-axis.
`sph_harm_y_cart` avoids that chart: it stably normalizes each trailing
three-vector and evaluates the harmonic from the normalized Cartesian
components, preserving the correct Cartesian axis derivatives. Positive
rescaling of a direction therefore does not change the value.

A zero vector or a vector containing a nonfinite component has no direction.
The Cartesian function refuses that lane with a complex `NaN` without
contaminating valid lanes. Float16 and bfloat16 inputs are evaluated as float32;
float32/float64 angular or Cartesian inputs produce complex64/complex128
harmonics, while `sph_legendre_p` returns the corresponding real dtype.

These are angular functions, not solid harmonics: `sph_harm_y_cart` discards
positive radial scale through normalization and does not compute
`r**n * Y_n^m`. Phydrax exposes pairwise scalar harmonics rather than a public
all-mode table, allowing callers and dynamic evaluators to accumulate only the
modes they need without materializing a basis table.

## Regular and irregular solid harmonics

For the same orthonormal Condon--Shortley $Y_n^m$ convention,
`solid_harmonic_regular(n, m, vector)` and
`solid_harmonic_irregular(n, m, vector)` evaluate

```text
R_n^m(vector) = r**n * Y_n^m(direction)
I_n^m(vector) = r**(-n-1) * Y_n^m(direction).
```

Degree and order are static integers with `n >= 0` and `abs(m) <= n`.
`vector` is a real numerical array ending in length three and broadcasts over
its leading axes. Both families satisfy
$H_n^{-m}=(-1)^m\overline{H_n^m}$.

The regular implementation is a homogeneous Cartesian recurrence, not angular
evaluation followed by a radial product. It therefore defines
$R_0^0(0)=1/\sqrt{4\pi}$ and every positive-degree mode as exactly zero at the
origin, with the derivatives of that same Cartesian polynomial. The irregular
family uses a scale-separated reciprocal radius and is admitted only for
finite nonzero vectors; a zero or nonfinite lane returns a lane-local complex
`NaN`. Coordinate AD is supported away from that singularity.

These pairwise functions do not expose an all-mode table, accept complex
vectors, or generalize to spin. For repeated synthesis of every active mode,
use `phydrax.discretization.SolidHarmonicPlan`, which preserves the shared
padded spherical coefficient layout without materializing a point-by-mode
table.

## Faddeeva and Dawson functions

The Faddeeva function is

```text
w(z) = exp(-z²) erfc(-i z),
w′(z) = -2 z w(z) + 2 i / sqrt(pi).
```

Phydrax evaluates it with Weideman's rational approximation in the upper
half-plane and the exact reflection identity below it. Dawson's integral uses
three real rational regimes and satisfies `F′(x) = 1 - 2 x F(x)`.

```python
z = jnp.array([0.0 + 0.5j, 1.0 + 0.5j])
w = phx.special.wofz(z)
dw = jax.jvp(phx.special.wofz, (z,), (jnp.ones_like(z),))[1]
```

The normalized Voigt profile uses `wofz` in its open domain. Exact Gaussian,
Cauchy, and zero-width limits define its scale boundaries. Negative Gaussian
or Cauchy scales return `NaN`.

## Differentiation support

| Family | Differentiable arguments |
| --- | --- |
| Carlson | every numerical argument away from poles/cuts |
| Complete/incomplete Legendre | all public numerical arguments on a fixed branch |
| Jacobi | argument `u` and parameter `m` inside one AGM branch |
| Airy | entire argument `x`; scaling factors follow their documented convention |
| Modified Bessel | argument and order, including integer-order limits |
| Cylindrical Bessel/Hankel | argument and order, including integer-order limits |
| Gegenbauer | argument and real parameter `alpha`, including the nonzero derivative at `alpha=0` |
| Zeta/Hurwitz zeta | order `s` and Hurwitz parameter `a` on the admitted meromorphic branch |
| Dilogarithm/polylogarithm | complex argument; polylogarithm order on its qualified interior envelope |
| Spin spherical synthesis | angles and declared frame angle; Cartesian direction and valid tangent frame away from topology changes |
| Solid harmonics | Cartesian vector away from the irregular origin singularity |
| Spherical harmonics | angular arguments; Cartesian directions away from zero/nonfinite lanes, including finite z-axis derivatives |
| Faddeeva/Dawson | complex argument |
| Voigt | admitted real arguments only; intentionally nonholomorphic |

At genuine poles, cut crossings, or nonsmooth scaling lips, derivatives remain
infinite or `NaN`; Phydrax does not clip them to finite substitutes. These are
fixed-precision principal-branch kernels, not arbitrary precision or an
all-Riemann-sheet API.

## Dtypes and numerical limits

- float16 and bfloat16 real inputs promote to float32;
- float32 remains float32 and maps to complex64 for complex-valued outputs;
- float64 remains float64 and maps to complex128;
- mixed numerical arguments use one common inexact dtype;
- NaNs propagate and invalid batch lanes are isolated;
- true overflow and underflow are retained.

These are fixed-precision kernels, not arbitrary-precision routines. Use a
high-precision reference such as mpmath when auditing isolated hard points.

## Provenance

The Faddeeva and Dawson kernels are adapted from JAX under Apache-2.0. The
modified Bessel regime structure is adapted from Numerax under MIT. Airy and
large-order cylindrical asymptotics are adapted from SciPy XSF, and the
cylindrical `jv`/`yv` kernels from XSF's bundled Cephes sources, under
BSD-3-Clause. The scalar spherical-harmonic recurrence is adapted from
JAXtronomy/spexial commit `6946494322d105edf84490529460553ac7c79b09`
under MIT. Spexial's change identifies GalacticDynamics/galax pull request
835 as the earlier development context; Phydrax does not redistribute the
Galax SCF potential or an all-mode table API. See `NOTICE` and the
corresponding files under `LICENSES/`.
