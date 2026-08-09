# Stochastic dynamics: PINNs, SDEs, and semidiscrete SPDEs

This page covers two complementary stochastic workflows:

1. **learn an equation solution** with backward-Kolmogorov or Fokker--Planck residual conditions;
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
import optax
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

backward_condition = phx.conditions.stochastic.Kolmogorov(
    "u",
    domain.component(),
    drift=drift,
    diffusion=diffusion,
    evolution_var="t",
)
backward_target = phx.integration.mean_over(backward_condition.on)
backward_realization = phx.integration.materialize(
    backward_target,
    phx.domain.PointSampling(
        64,
        layout=phx.domain.SampleLayout((("x", "t"),)),
    ),
    key=jr.key(0),
)
backward = phx.terms.ResidualPenalty(
    backward_condition,
    phx.integration.fixed(backward_realization),
)
solver = phx.solver.FunctionalSolver(
    functions={"u": u},
    terms=(backward,),
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

stationary_condition = phx.conditions.stochastic.FokkerPlanck(
    "p",
    state.component(),
    drift=drift,
    diffusion=diffusion,
    evolution_var=None,
)
stationary = phx.terms.ResidualPenalty(
    stationary_condition,
    phx.integration.per_step(
        phx.integration.mean_over(stationary_condition.on),
        phx.domain.PointSampling(
            64,
            layout=phx.domain.SampleLayout((("x",),)),
        ),
    ),
)
```

### Positivity, normalization, and data are explicit

The Fokker--Planck residual does not silently impose probability semantics.
Build those contracts explicitly:

```python
state = phx.domain.Interval1d(-2.0, 2.0)
time = phx.domain.TimeInterval(0.0, 1.0)
domain = state @ time

model = phx.nn.models.MLP(
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

fokker_planck_condition = phx.conditions.stochastic.FokkerPlanck(
    "p",
    domain.component(),
    drift=drift,
    diffusion=diffusion,
    evolution_var="t",
)
fokker_planck = phx.terms.ResidualPenalty(
    fokker_planck_condition,
    phx.integration.per_step(
        phx.integration.mean_over(fokker_planck_condition.on),
        phx.domain.PointSampling(
            128,
            layout=phx.domain.SampleLayout((("x", "t"),)),
        ),
    ),
)

# Partial integration leaves the time sites explicit. Multiplying both sides by
# their equal product-rule weight preserves the target \(1/\mathtt{num\_t}\).
num_x, num_t = 64, 16
normalization_condition = phx.conditions.Moment(
    "p",
    domain.component(),
    lambda p: p / num_t,
    target=jnp.full((num_t,), 1.0 / num_t),
)
normalization = phx.terms.MomentPenalty(
    normalization_condition,
    phx.integration.per_step(
        phx.integration.over(normalization_condition.on, axes="x"),
        phx.domain.PointSampling(
            (num_x, num_t),
            layout=phx.domain.SampleLayout((("x",), ("t",))),
        ),
    ),
)

normalizer = jnp.sqrt(jnp.pi) * jsp_special.erf(2.0)
initial_slice = domain.component({"t": phx.domain.FixedStart()})
initial_condition = phx.conditions.Initial(
    "p",
    initial_slice,
    target=lambda x: jnp.exp(-x[0] ** 2) / normalizer,
    evolution_var="t",
    order=0,
)
initial = phx.terms.ResidualPenalty(
    initial_condition,
    phx.integration.per_step(
        phx.integration.mean_over(initial_condition.on),
        phx.domain.PointSampling(
            64,
            layout=phx.domain.SampleLayout((("x",),)),
        ),
    ),
)

density_solver = phx.solver.FunctionalSolver(
    functions={"p": density},
    terms=(fokker_planck, normalization, initial),
)
```

The positive activation, per-time normalization, and initial data are separate
from the dynamics residual. Add absorbing, reflecting, or zero-flux boundary
conditions when the truncated state domain requires them. This separation
makes each failure diagnosable and still permits an unnormalized stationary
eigenfunction when normalization is not part of the task.

## Particle-first score matching in high dimension

Directly representing a normalized density over a high-dimensional state is often the
wrong target. If the SDE can be simulated, adapt its valid trajectory nodes to
state-time particles and learn

\[
s_\theta(t,x)\approx\nabla_x\log p_t(x)
\]

with the Hyvärinen objective

\[
\mathbb E_{p_t}\left[\tfrac12\|s_\theta(t,X_t)\|^2+
\nabla_x\!\cdot s_\theta(t,X_t)\right].
\]

The following executable example uses Gaussian particles at one saved time and a
trainable four-dimensional score field:

```python
score_dimension = 4
score_time = jnp.asarray(0.5)
particle_trajectory = phx.stochastic.StochasticTrajectory(
    jnp.asarray([score_time]),
    jr.normal(jr.key(4), (128, 1, score_dimension)),
    realization_axes=("path",),
    realization_shape=(128,),
    time_axis="saved_time",
    state_axes=("state",),
)
score_domain = phx.domain.HyperRectangle(
    jnp.full((score_dimension,), -5.0),
    jnp.full((score_dimension,), 5.0),
    label="x",
) @ phx.domain.TimeInterval(0.0, 1.0)
score_model = phx.nn.models.MLP(
    in_size=score_dimension + 1,
    out_size=score_dimension,
    width_size=32,
    depth=2,
    key=jr.key(5),
)
score = score_domain.Model("x", "t")(score_model)
particles = phx.stochastic.trajectory_state_time_samples(
    particle_trajectory,
    state_label="x",
    time_label="t",
)
score_term = phx.terms.ScoreMatchingTerm(
    "score",
    particles,
    policy=phx.terms.ScoreMatchingPolicy(
        "implicit",
        num_probes=8,
        distribution="rademacher",
    ),
    sampling_mode="fixed",
)
score_solver = phx.solver.FunctionalSolver(
    functions={"score": score},
    terms=(score_term,),
)
score_solver = score_solver.solve(
    num_iter=2000,
    optim=optax.adam(1e-3),
    keep_best=False,
)
diagnostics = score_term.diagnostics(
    score_solver.functions,
    key=jr.key(6),
)
```

`method="implicit"` estimates only the divergence with JVP probes; it does not form a
dense Jacobian. `method="exact"` is useful for small-dimensional validation.
`method="sliced"` uses projected score matching. In every mode the score output must
have exactly the state shape.

The adapter retains validity masks, state-time weights, path identities, independence
labels, and time coverage. Path-standard-error diagnostics reduce complete paths
rather than pretending that all time nodes are independent. For fresh simulation each
optimizer update, pass a callable returning a `StochasticTrajectory` and select
`sampling_mode="resample"`; the provider is called once outside differentiation.

This workflow returns a score field, not a normalized density. Density reconstruction,
likelihood evaluation, and sampling from a learned distribution require a separate
flow, transport, or reverse-time model and are not implied by score matching.

The `implicit-score-matching` entry in
`tools/high_dimensional_pde_benchmarks.py --suite methods` constructs
`TrajectoryStateTimeSamples`, evaluates `ScoreMatchingTerm` on the analytic
Ornstein--Uhlenbeck score, and compares the empirical Hyvärinen term with its
closed-form expectation. The record includes path-cluster standard error, score-field
RMSE, divergence error, valid fraction, runtime, and particle working-set size.

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

### Measure-aware path observables

Convert the solver result once, then compose deterministic space and time
quadrature with the empirical path measure:

```python
trajectory = paths.to_stochastic_trajectory(
    realization_axes=("path",),
    state_axes=("space",),
    discretization_id=space.discretization_id,
    basis_id=noise.basis_id,
)
path_measure = phx.stochastic.trajectory_measure(trajectory, mode="path")
time_measure = phx.stochastic.time_measure(trajectory, rule="trapezoid")
space_measure = phx.solver.spatial_measure(space, spatial_dims="space")

space_integrals = phx.integration.integrate(path_measure.samples, space_measure)
time_integrals = phx.integration.integrate(space_integrals.value, time_measure)
expected_space_time_integral = phx.integration.integrate(
    time_integrals.value,
    path_measure,
)
assert expected_space_time_integral.successful
```

The three stages have distinct semantics: physical spatial quadrature,
irregular saved-time quadrature per path, then empirical expectation over
complete paths. A failed path can produce `NO_VALID_SAMPLES` at the time stage
without poisoning the final expectation because the path measure excludes it.
Use `mode="marginal"` instead when the estimand is a time-indexed ensemble
mean and individual failed states should be excluded independently.

`prediction` labels the path axis as `process`. Reusing `realization` replays the
same global Brownian paths, even when a time horizon is split across solves. Changing
the root key changes paths; changing the grid, rank, spectrum, or modes changes
`noise_id`.

`TensorGridDiscretization` also supports periodic finite differences, sine bases
with homogeneous Dirichlet semantics, cosine bases with homogeneous Neumann
semantics, and multidimensional tensor grids. `SpectralSpatialDiscretization`
reuses a precomputed manifold `phydrax._spectral.SpectralDiscretization`
directly, without a provider or a second eigenbasis convention.

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
`HyperRectangle(lower, upper)`. Then apply
`phx.conditions.stochastic.FokkerPlanck(..., state_var="x")` to that reduced
coordinate. This is an ordinary finite-dimensional Fokker--Planck PINN whose
state dimension is \(r\), while reconstruction uses the same spectral modes as
the semidiscrete solver.

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
[API → Stochastic conditions](../api/conditions/stochastic.md), and
[API → Differential equation integration](../api/solver/differential.md).
