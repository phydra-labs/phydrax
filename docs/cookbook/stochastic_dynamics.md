# Stochastic dynamics: PINNs, SDEs, and semidiscrete SPDEs

This page covers two complementary stochastic workflows:

1. **learn an equation solution** with backward-Kolmogorov or Fokker--Planck residual constraints;
2. **simulate paths** after spatial semidiscretization with a finite-rank Wiener process and Diffrax.

They solve different problems. A stochastic PINN represents an observable or density as a
`DomainFunction`; a path solver returns realizations of a finite-dimensional SDE. Do not
interpret a path ensemble as a learned density, or a residual-trained density as sampled paths.

For latent-state estimation over those dynamics, use the
[filtering and smoothing recipe](filtering.md). For terminal-value stochastic
representations and semilinear PDE losses, use the
[backward stochastic equation recipe](bsde.md). Martingale, filtering, and
BSDE APIs share the same trajectory validity and realization-provenance
contracts.

## Backward Kolmogorov PINN

For the Itô diffusion

\[
dX_t=b(X_t,t)\,dt+\sigma(X_t,t)\,dW_t,
\]

the backward equation is

\[
\partial_tu+b_i\partial_i u+\frac12a_{ij}\partial_{ij}u=0,
\qquad a=\sigma\sigma^\mathsf T.
\]

A manufactured Brownian solution gives a direct residual check:

```python
import jax
import jax.scipy.special as jsp_special
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

sigma_value = 0.7
state = phx.domain.Interval1d(-2.0, 2.0)
time = phx.domain.TimeInterval(0.0, 1.0)
domain = state @ time

u = domain.Function("x", "t")(
    lambda x, t: x[0] ** 2 - sigma_value**2 * t
)
drift = domain.Function("x", "t")(
    lambda x, t: jnp.asarray([0.0])
)
diffusion = domain.Function("x", "t")(
    lambda x, t: jnp.asarray([[sigma_value]])
)

backward = phx.constraints.ContinuousKolmogorovConstraint(
    "u",
    domain.component(),
    drift=drift,
    diffusion=diffusion,
    evolution_var="t",
    num_points=64,
    structure=phx.domain.ProductStructure((("x", "t"),)),
    sampling_mode="fixed",
    fixed_batch_key=jr.key(0),
)
solver = phx.solver.FunctionalSolver(
    functions={"u": u},
    constraints=[backward],
)
assert solver.loss(key=jr.key(1)) < 1e-12
```

Replace the analytic `u` with `domain.Model("x", "t")(model)` for a PINN. Drift,
diffusion, and covariance may instead be names in `functions`; named coefficient fields
remain trainable. This is the inverse-problem path for learning a diffusion coefficient
jointly with the observable.

Set `evolution_var=None` for a stationary equation. For Stratonovich dynamics, pass
`interpretation="stratonovich"` and a diffusion factor. A covariance alone cannot recover
the Stratonovich drift correction.

## Fokker--Planck PINN

The forward density satisfies

\[
\partial_t p
=-\partial_i(b_i p)+\frac12\partial_i\partial_j(a_{ij}p).
\]

For the one-dimensional Ornstein--Uhlenbeck process

\[
dX_t=-\theta X_t\,dt+\sigma\,dW_t,
\]

the unnormalized stationary density

\[
p_\infty(x)\propto\exp\!\left(-\frac{\theta x^2}{\sigma^2}\right)
\]

has zero stationary residual:

```python
theta, sigma_value = 0.8, 0.6
state = phx.domain.Interval1d(-3.0, 3.0)
density = state.Function("x")(
    lambda x: jnp.exp(-theta * x[0] ** 2 / sigma_value**2)
)
drift = state.Function("x")(
    lambda x: jnp.asarray([-theta * x[0]])
)
diffusion = state.Function("x")(
    lambda x: jnp.asarray([[sigma_value]])
)

stationary = phx.constraints.ContinuousFokkerPlanckConstraint(
    "p",
    state.component(),
    drift=drift,
    diffusion=diffusion,
    evolution_var=None,
    num_points=64,
    structure=phx.domain.ProductStructure((("x",),)),
)
```

### Positivity, normalization, and data are explicit

The Fokker--Planck residual does not silently impose probability semantics.
Build those contracts explicitly:

```python
state = phx.domain.Interval1d(-2.0, 2.0)
time = phx.domain.TimeInterval(0.0, 1.0)
domain = state @ time

model = phx.nn.MLP(
    in_size=2,
    out_size="scalar",
    width_size=32,
    depth=3,
    final_activation=lambda raw: jax.nn.softplus(raw) + 1e-8,
    key=jr.key(3),
)
density = domain.Model("x", "t")(model)
drift = domain.Function("x", "t")(
    lambda x, t: jnp.asarray([-0.8 * x[0]])
)
diffusion = domain.Function("x", "t")(
    lambda x, t: jnp.asarray([[0.6]])
)

fokker_planck = phx.constraints.ContinuousFokkerPlanckConstraint(
    "p",
    domain.component(),
    drift=drift,
    diffusion=diffusion,
    evolution_var="t",
    num_points=128,
    structure=phx.domain.ProductStructure((("x", "t"),)),
)

# Partial integration retains the time quadrature weight. On [0, 1] with
# equal time weights, target 1 / num_t is equivalent to integral p(x, t) dx = 1.
num_x, num_t = 64, 16
normalization = phx.constraints.ContinuousIntegralInteriorConstraint(
    "p",
    domain,
    lambda p: p,
    num_points=(num_x, num_t),
    structure=phx.domain.ProductStructure((("x",), ("t",))),
    over="x",
    equal_to=jnp.full((num_t,), 1.0 / num_t),
)

normalizer = jnp.sqrt(jnp.pi) * jsp_special.erf(2.0)
initial = phx.constraints.ContinuousInitialFunctionConstraint(
    "p",
    domain,
    func=lambda x: jnp.exp(-x[0] ** 2) / normalizer,
    evolution_var="t",
    time_derivative_order=0,
    num_points=64,
    structure=phx.domain.ProductStructure((("x",),)),
)

density_solver = phx.solver.FunctionalSolver(
    functions={"p": density},
    constraints=[fokker_planck, normalization, initial],
)
```

The positive activation, per-time normalization, and initial data are separate
from the dynamics residual. Add absorbing, reflecting, or zero-flux boundary
constraints when the truncated state domain requires them. This separation
makes each failure diagnosable and still permits an unnormalized stationary
eigenfunction when normalization is not part of the task.

## Stochastic heat equation by method of lines

Consider

\[
dU_t=\kappa\Delta U_t\,dt+dW_t^Q.
\]

Choose the spatial discretization and retained covariance modes explicitly:

```python
axis = phx.domain.FourierAxisSpec(32).materialize(0.0, 1.0)
space = phx.solver.TensorGridDiscretization((axis,))

# q(lambda) is evaluated on low eigenvalues of -Delta_h.
noise = phx.solver.SpatialNoiseBasis.from_spectrum(
    space,
    lambda eigenvalue: 0.02 * jnp.exp(-0.05 * eigenvalue),
    rank=6,
)
initial = jnp.sin(2.0 * jnp.pi * axis.nodes)

heat = phx.solver.semidiscretize_reaction_diffusion(
    initial,
    space,
    t0=0.0,
    t1=0.2,
    kappa=0.01,
    noise_basis=noise,
    interpretation="ito",
)
realization = heat.wiener_realization(
    jr.key(2),
    sample_shape=(128,),
    tolerance=1e-4,
    label="stochastic-heat-0",
)
paths = phx.solver.solve_diffrax_ensemble(
    heat.problem,
    save_times=jnp.linspace(0.0, 0.2, 21),
    realization=realization,
    dt0=1e-3,
)
prediction = paths.to_predictive(
    sample_dim="path",
    time_dim="time",
    state_dims=("space",),
)
```

`prediction` labels the path axis as `process`. Reusing `realization` replays the
same global Brownian paths, even when a time horizon is split across solves. Changing
the root key changes paths; changing the grid, rank, spectrum, or modes changes
`noise_id`.

`TensorGridDiscretization` also supports periodic finite differences, sine bases with
homogeneous Dirichlet semantics, cosine bases with homogeneous Neumann semantics, and
multidimensional tensor grids. `SpectralSpatialDiscretization` reuses a precomputed
manifold `phydrax.nn.SpectralDiscretization` without changing its eigenbasis convention.

## Stochastic Allen--Cahn

The same objects produce

\[
dU_t=[\kappa\Delta U_t+U_t-U_t^3]dt+dW_t^Q.
\]

```python
allen_cahn = phx.solver.semidiscretize_reaction_diffusion(
    0.25 * jnp.cos(2.0 * jnp.pi * axis.nodes),
    space,
    t0=0.0,
    t1=0.2,
    kappa=0.01,
    reaction=lambda t, state, args: state - state**3,
    noise_basis=noise,
)
```

Pass `noise_amplitude` for multiplicative spatial noise. It may return a scalar, an array
with the exact state shape, or the full `state_shape + noise_shape` diffusion tensor.

## Density equations in reduced spectral coordinates

A full-grid Fokker--Planck density has one coordinate per spatial degree of freedom and
is generally intractable. For low-rank dynamics, project onto retained spectral modes,

\[
U_h(x,t)\approx\sum_{k=1}^r a_k(t)\phi_k(x),
\]

derive the finite-dimensional drift and diffusion for
`a = (a_1, ..., a_r)`, and define the density on a state domain such as
`HyperRectangle(lower, upper)`. Then apply `ContinuousFokkerPlanckConstraint` with
`state_var="x"` to that reduced coordinate. This is an ordinary finite-dimensional
Fokker--Planck PINN whose state dimension is `r`, while reconstruction uses the same
spectral modes as the semidiscrete solver.

This route is honest about dimensionality: it does not pretend to learn a probability
density over thousands of nodal values.

## Boundaries and verification

- Spatial sine/cosine discretizations encode homogeneous boundary extensions in the
  Laplacian. Nonhomogeneous or nonlinear boundary behavior still needs an explicit model.
- `SpatialNoiseBasis` is finite rank. Cylindrical/space--time white noise requires an
  explicit truncation before integration.
- A path ensemble measures process variation. Grid and time-step studies are separate
  numerical-uncertainty experiments.
- Validate linear semidiscrete systems against analytic mean/covariance evolution and
  validate Stratonovich models against either corrected Itô dynamics or known moments.

See [API → Differential operators](../api/operators/differential.md),
[API → Continuous constraints](../api/constraints/continuous.md), and
[API → Differential equation integration](../api/solver/differential.md).
