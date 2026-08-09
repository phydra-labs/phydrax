# Special functions and named integrals

`phydrax.special` contains named numerical primitives with fixed mathematical
contracts. Use it when the function itself is the object of computation. Use
`phydrax.integration` when a user-defined integrand, measure, and numerical plan
must be composed at runtime.

All functions accept Python scalars and JAX arrays, broadcast numerical
arguments, compose with `jax.jit` and `jax.vmap`, and use branch-safe fixed
iteration kernels. Analytic custom JVPs are used where differentiating an
approximation branch would be unstable.

```python
import jax
import jax.numpy as jnp
import phydrax as phx

m = jnp.array([0.0, 0.5, 0.9])
k = phx.special.ellipk(m)
dk = jax.vmap(jax.grad(phx.special.ellipk))(m)
```

## Conventions at a glance

| Family | Public functions | Convention and real domain |
| --- | --- | --- |
| Carlson | `elliprc`, `elliprf`, `elliprd`, `elliprj`, `elliprg` | nonnegative principal arguments; `elliprc` and the fourth argument of `elliprj` are positive |
| Complete Legendre | `ellipk`, `ellipkm1`, `ellipe`, `ellippi` | parameter `m = k²`; `m <= 1`; third-kind characteristic `n < 1`; `ellipkm1(p) = K(1-p)` for `p >= 0` |
| Incomplete Legendre | `ellipkinc`, `ellipeinc`, `ellippiinc` | unwrapped amplitude `phi`; parameter `m <= 1`; third-kind characteristic `n < 1` |
| Jacobi | `ellipj`, `ellipam` | real `u`, parameter `m <= 1`; `ellipj` returns `(sn, cn, dn, am)` |
| Airy | `airy`, `airye` | real argument; each returns `(Ai, Ai′, Bi, Bi′)` |
| Modified Bessel | `iv`, `ive`, `kv`, `kve` | nonnegative real order and argument |
| Cylindrical Bessel | `jv`, `yv`, `hankel1`, `hankel2` | nonnegative real order; nonnegative real argument with singular `Y`/Hankel zero limit |
| Faddeeva | `wofz`, `dawsn`, `voigt_profile` | complex Faddeeva argument; real Dawson/profile arguments |

A real-only function raises `TypeError` for a complex input rather than silently
discarding its imaginary part. Invalid real-domain lanes return `NaN`; valid
lanes in the same batch remain isolated.

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

The admitted contract is `v >= 0`, `x >= 0`. At zero, `I_0(0) = 1`,
`I_v(0) = 0` for positive `v`, and `K_v(0) = +inf`. Argument derivatives use
three-term recurrences and work in forward, reverse, and higher-order modes.
Differentiation with respect to order is deliberately unsupported and raises
`TypeError`; treating a floating order as a differentiable coordinate would
otherwise expose an unverified numerical derivative.

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

`J_0(0) = 1`, positive-order `J_v(0) = 0`, and `Y_v(0) = -inf` for admitted
orders. At positive infinity, the real Bessel values approach zero. As for the
modified family, argument differentiation is analytic and order
differentiation raises `TypeError`.

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
| Carlson | every numerical argument in the admitted real domain |
| Complete/incomplete Legendre | all public arguments in the admitted real domain |
| Jacobi | argument `u` and parameter `m` |
| Airy | argument `x` |
| Modified Bessel | argument `x`; order tangents raise `TypeError` |
| Cylindrical Bessel/Hankel | argument `x`; order tangents raise `TypeError` |
| Faddeeva/Dawson/Voigt | all admitted numerical arguments, with documented scale-boundary rules |

At genuine poles or nonsmooth endpoints, derivatives remain infinite or `NaN`;
Phydrax does not clip them to fabricated finite substitutes.

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
BSD-3-Clause. See `NOTICE` and the corresponding files under `LICENSES/`.
