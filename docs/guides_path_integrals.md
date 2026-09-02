# Euclidean path integrals and Feynman–Kac expectations

Phydrax provides finite-dimensional, time-sliced path operators for:

- Euclidean quantum propagation with fixed-endpoint Brownian bridges;
- Feynman–Kac expectations and discrete first-passage observables;
- positive-regulator, finite-slice real-time oscillatory estimates;
- finite periodic rings, compact U(1) lattice measures, and admitted exchange
  sectors.

Real-time results are values of the declared regulated finite integral. They do
not establish that a regulator-zero limit exists, extrapolate such a limit, or
claim field-theory/continuum universality.

## Uniform time slicing

`TemporalMesh.uniform(t0, t1, num_steps, role="path")` defines the nodes

$$
t_k=t_0+k\Delta t,
\qquad
\Delta t=\frac{t_1-t_0}{N}.
$$

Paths use shape `(..., num_paths, num_nodes, state_dim)`. Time and state are
internal path dimensions, not `SampleLayout` point-sampling axes. This avoids a
Cartesian product over every time slice.

## Euclidean fixed-endpoint kernels

For mass $m>0$, Planck constant $\hbar>0$, and real scalar potential $V$, the
Euclidean kernel is written as a free kernel times a Brownian-bridge expectation:

$$
K_E(x_1,T;x_0,0)
=
K_{E,0}(x_1,T;x_0,0)
\,\mathbb E_{\mathrm{bridge}}
\left[
\exp\left(-\frac{1}{\hbar}\int_0^T V(q(t),t)\,dt\right)
\right].
$$

`euclidean_kernel` samples the free bridge measure exactly and applies midpoint
quadrature only to the potential action. It therefore does not introduce arbitrary
bounds on every interior path coordinate.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

slicing = phx.discretization.TemporalMesh.uniform(0.0, 1.0, 32, role="path")
omega = 0.8
potential = lambda q, t: 0.5 * omega**2 * q[0] ** 2

estimate = phx.operators.euclidean_kernel(
    potential,
    jnp.array([0.0]),
    jnp.array([0.3]),
    slicing=slicing,
    mass=1.0,
    hbar=1.0,
    num_paths=2048,
    chunk_size=256,
    key=jr.key(0),
)

assert estimate.value > 0.0
assert estimate.standard_error > 0.0
```

A potential may also be a trainable `DomainFunction`. Set `position_var` and
`time_var` when its labels differ from the defaults `"q"` and `"t"`.
`euclidean_kernel_function` wraps the scalar kernel value as a `DomainFunction` over
two endpoint labels, so it can be sampled and composed with existing operators and
constraints. Diagnostics remain available from direct `euclidean_kernel` calls.

For the free particle, pass `potential=None`. `free_euclidean_kernel` evaluates the
analytic normalization directly, so its Monte Carlo standard error is exactly zero.

## Engineering Feynman–Kac expectations

For the Itô diffusion

$$
dX_t=b(X_t,t)\,dt+\sigma(X_t,t)\,dW_t,
$$

`feynman_kac_expectation` estimates the terminal form

$$
u(x,t_0)=
\mathbb E_{X_{t_0}=x}\left[
 g(X_{t_1},t_1)
 \exp\left(-\int_{t_0}^{t_1}c(X_s,s)\,ds\right)
\right].
$$

The explicit-noise diffusion sampler uses Euler–Maruyama as a reproducible
reference. Adaptive SDE paths use the canonical prepared stochastic path
ensemble and retain accepted/rejected-step, Wiener realization, replay, and
temporal evidence. General compiled geometry supports killed paths; specular
reflection requires an explicit velocity state and a regular certified normal.
Exact reflecting overdamped density kernels are restricted to prepared affine
interval image geometry.

```python
kappa = 0.35
wave_number = 1.1
heat_slicing = phx.discretization.TemporalMesh.uniform(0.0, 0.7, 24, role="path")

heat = phx.operators.feynman_kac_expectation(
    lambda x, t: jnp.cos(wave_number * x[0]),
    lambda x, t: jnp.zeros_like(x),
    jnp.sqrt(2.0 * kappa),
    jnp.array([0.25]),
    slicing=heat_slicing,
    num_paths=4096,
    key=jr.key(1),
)

analytic = jnp.cos(wave_number * 0.25) * jnp.exp(-kappa * wave_number**2 * 0.7)
assert jnp.abs(heat.value - analytic) < 6.0 * heat.standard_error
```

Use `feynman_kac_from_paths` when trajectories were generated separately or come from
an empirical ensemble.

### Source terms

`source_feynman_kac_from_paths` adds the Duhamel source integral on the same
paths as the terminal term. Left, trapezoid, and midpoint source quadrature are
explicit policy choices. `source_feynman_kac_from_stochastic_paths` consumes
the canonical adaptive ensemble without introducing another SDE integrator.
Sampling error, source-quadrature difference, temporal-solver evidence, and
boundary-event error remain separate.

### Regulated real-time estimates

`RealTimePathIntegralPlan` declares mass, Planck constant, positive regulator,
slice mesh, and finite population. The result contains the complex mean, real/
imaginary covariance, standard error, mean phase, and phase ESS. A phase below
policy threshold reports `unresolved_sign_problem`; it is never converted into
a positive-weight ESS. `RealTimeRegulatorContinuation` reports only paired
finite-regulator differences.

### Periodic, gauge, and exchange measures

`PeriodicPathPlan` requires a confining potential, finite periodic cell, or
fixed centroid to remove the improper free centroid mode. Absolute log
partition values require a named known reference and finite thermodynamic
integration schedule. `CompactU1GaugeMeasure` is finite Wilson U(1), not
non-Abelian or a continuum theory. `ExchangePathPlan` identifies complete
enumeration versus a restricted sector and reports fermionic average sign.

## First passage and reliability

`first_exit_index`, `first_exit_time`, and `survival_probability` accept an explicit
`inside(x) -> bool` callable. A crossing is detected at the first stored node outside
the region. A path that survives through `t1` receives index `-1` and exit time
`inf`.

This is a discrete crossing contract. Phydrax does not interpolate a crossing or
claim continuous-time first-passage accuracy. Refine `num_steps` and report the
remaining positive slice bias separately from the Bernoulli standard error.

## Diagnostics and convergence

`PathIntegralEstimate` contains:

- `value`;
- `standard_error` across independent sampled paths;
- `effective_sample_size` (ESS) of positive Euclidean or killing weights;
- `log_mean_weight` for stable inspection of weight scale;
- `num_paths`.

Four error axes must be reported independently:

1. path-population sampling error;
2. path time-slicing or adaptive temporal-solver error;
3. source/image/schedule quadrature or truncation error;
4. localized boundary-event error.

A small standard error does not diagnose time-slicing bias. A collapsing ESS means a
few paths dominate the weighted estimate even if the reported value looks smooth.

For reproducible optimization, keep the PRNG key fixed or pass explicit standard
normal arrays through `brownian_bridge_from_noise` and
`diffusion_paths_from_noise`. These common random numbers make parameter comparisons
and gradients deterministic. Resample keys when estimating out-of-sample uncertainty.

## Scope relative to FeynmaNN

`FeynmaNN` is an interference-inspired neural architecture. Its internal
sum-over-paths block is not a discretized physical trajectory measure and does not
compute the Euclidean or Feynman–Kac estimators described here.
