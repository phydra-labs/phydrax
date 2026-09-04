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
import coordax as cx
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

u = domain.Function("x", "t")(lambda x, t: x[0] ** 2 - sigma_value**2 * t)
drift = domain.Function("x", "t")(lambda x, t: jnp.asarray([0.0]))
diffusion = domain.Function("x", "t")(lambda x, t: jnp.asarray([[sigma_value]]))

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
density = state.Function("x")(lambda x: jnp.exp(-theta * x[0] ** 2 / sigma_value**2))
drift = state.Function("x")(lambda x: jnp.asarray([-theta * x[0]]))
diffusion = state.Function("x")(lambda x: jnp.asarray([[sigma_value]]))

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
drift = domain.Function("x", "t")(lambda x, t: jnp.asarray([-0.8 * x[0]]))
diffusion = domain.Function("x", "t")(lambda x, t: jnp.asarray([[0.6]]))

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
normalization = phx.terms.RandomizedMomentPenalty(
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
    target=lambda x: jnp.exp(-(x[0] ** 2)) / normalizer,
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

This workflow returns a score field, not a normalized density. For a prescribed VP or
VE process, `DenoisingScoreMatchingTerm` offers a derivative-free alternative when
the exact Gaussian transition score is known. A trained score becomes a stochastic
generator only after composition with `phx.transport.ReverseDiffusion`; it becomes a
deterministic density flow only after `probability_flow_system` is composed with
`DiffraxEvolution`, `ContinuousTransport`, and `ContinuousFlowLaw`. Neither
composition certifies model error, and a finite-time Gaussian terminal reference
remains explicitly exact, asymptotic, or external.

See [Gaussian score diffusions](../api/stochastic/diffusion.md) and
[score-based diffusion transport](../api/transport/diffusion.md).

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
axis = phx.discretization.FourierAxisSpec(32).materialize(0.0, 1.0)
space = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))

# q(lambda) is evaluated on low eigenvalues of -Delta_h.
noise = phx.stochastic.SpatialNoiseBasis.from_spectrum(
    space,
    lambda eigenvalue: 0.02 * jnp.exp(-0.05 * eigenvalue),
    rank=6,
)
initial = space.project(jnp.sin(2.0 * jnp.pi * axis.nodes))

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
    state_dims=("mode",),
)
```

### Measure-aware path observables

Convert the solver result once, then compose deterministic space and time
quadrature with the empirical path measure:

```python
modal_trajectory = paths.to_stochastic_trajectory(
    realization_axes=("path",),
    state_axes=("mode",),
    discretization_id=space.discretization_id,
    basis_id=noise.basis_id,
)
physical_values = cx.Field(
    jax.vmap(jax.vmap(space.reconstruct))(paths.states),
    dims=("path", "time", "space"),
)
path_measure = phx.stochastic.trajectory_measure(modal_trajectory, mode="path")
time_measure = phx.stochastic.time_measure(modal_trajectory, rule="trapezoid")
space_measure = phx.integration.spatial_measure(space, spatial_dims="space")

space_integrals = phx.integration.integrate(physical_values, space_measure)
time_integrals = phx.integration.integrate(space_integrals.value, time_measure)
expected_space_time_integral = phx.integration.integrate(
    time_integrals.value,
    path_measure,
)
assert expected_space_time_integral.successful
```

The solver trajectory is modal; reconstruct it before applying physical spatial
quadrature. The three reductions then have distinct semantics: physical spatial
quadrature, irregular saved-time quadrature per path, and empirical expectation over
complete paths. A failed path can produce `NO_VALID_SAMPLES` at the time stage
without poisoning the final expectation because the path measure excludes it.
Use `mode="marginal"` instead when the estimand is a time-indexed ensemble
mean and individual failed states should be excluded independently.

`prediction` labels the path axis as `process`. Reusing `realization` replays the
same global Brownian paths, even when a time horizon is split across solves. Changing
the root key changes paths; changing the grid, rank, spectrum, or modes changes
`noise_id`.

`TensorSpectralDiscretization` composes global tensor spectral bases. Its solver state
and `SpatialNoiseBasis.from_spectrum` modes are modal; use `project` and `reconstruct`
at physical boundaries. Fourier noise modes are projections of real
weighted-orthonormal eigenfunctions, so their complex columns preserve conjugate
symmetry under real Wiener coefficients.

The returned trajectories remain complex modal arrays. Internally, the standard
Diffrax backend stacks real and imaginary components along one leading size-two axis,
uses the same real Wiener increment for both, and removes that axis before returning
the solution. `solution.temporal_evidence.state_coordinates` records this realized
real-coordinate representation.

Uniform periodic finite differences use `periodic_finite_difference`; bounded FD2
operators can use `diagonalize_fd_laplacian`. `EigenbasisDiscretization` reuses a
precomputed `phydrax.discretization.SpectralDecomposition` while preserving
independent transform and operator-spectrum identities.

## Stochastic Allen--Cahn

The same objects produce

\[
dU_t=[\kappa\Delta U_t+U_t-U_t^3]dt+dW_t^Q.
\]

```python
allen_cahn_method = phx.discretization.PseudospectralMethodPlan(
    dealiasing=phx.discretization.PaddingDealiasingPlan(3),
).prepare(
    space,
    required_polynomial_degree=3,
    nonlinear=True,
)
allen_cahn = phx.solver.semidiscretize_reaction_diffusion(
    space.project(0.25 * jnp.cos(2.0 * jnp.pi * axis.nodes)),
    space,
    t0=0.0,
    t1=0.2,
    kappa=0.01,
    reaction=lambda t, coefficients, args: allen_cahn_method.nonlinear_action(
        coefficients,
        lambda values: values - values**3,
    ),
    reaction_id=(
        f"allen-cahn-cubic-modal-reaction-v1:{allen_cahn_method.prepared_id}"
    ),
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

## Evaluate stochastic decisions without changing their meaning

Use one identified prepared-noise bundle for a rollout. Use a disjoint bundle when
the result is meant to describe holdout behavior:

```python
training = phx.control.stochastic.evaluate_feedback_policy(
    controlled_problem,
    feedback_policy,
    training_noise,
    policy_id="frozen-feedback-policy",
    method="asymptotic-normal",
    sample_role="training",
)
holdout = phx.control.stochastic.evaluate_feedback_policy(
    controlled_problem,
    feedback_policy,
    holdout_noise,
    policy_id="frozen-feedback-policy",
    method="asymptotic-normal",
    sample_role="holdout",
)
if not bool(training.valid) or not bool(holdout.valid):
    raise RuntimeError(
        f"policy evaluation failed: training={training.status}, holdout={holdout.status}"
    )
```

`feedback_policy(context, state, args)` is called before the current noise
increment is exposed. `training.evidence` intentionally has no coverage claim.
`holdout.evidence` retains its requested assumptions and counts independent
clusters, not merely paths. An interval for this fixed policy is not an interval
for an optimal policy. `examples/stochastic_feedback_control.py` is a complete
runnable construction of `ControlledTransitionProblem` and
`PreparedControlledNoise`.

For paired comparison, pass the same prepared bundle to both policies through
`compare_feedback_policies`; the function reports right-minus-left return and
requires matching realization and coupling provenance.

## Select the exact LQ stochastic class

These solvers are exact only within different declared model classes:

| Model | Entry point | Information and noise contract |
|---|---|---|
| Additive, full-state, one controller | `finite_horizon_lqg_state_feedback` | Exogenous zero-mean additive noise; full state available to the policy |
| Additive, full-state, multiple minimizers | `finite_horizon_lqg_feedback_nash` | Same common additive process; explicit player cost axis and joint-control ownership |
| Multiplicative, one controller | `finite_horizon_multiplicative_lq_state_feedback` | Declared affine state/action noise channels and full channel covariance |
| Multiplicative, multiple minimizers | `finite_horizon_multiplicative_lq_feedback_nash` | Common multiplicative dynamics with player-specific expected quadratic costs |
| Centralized partial observation | `CentralizedLQGProblem` and `finite_horizon_centralized_lqg` | Gaussian prior; observation before action; policy input is a `GaussianBelief` |

Check `valid`, `status`, covariance evidence, curvature, rank, solve residuals,
stationarity, and Bellman evidence as applicable. The centralized belief solver
rejects singular active innovations and does not project or symmetrize a covariance.
It does not represent decentralized information, cross-correlated process and
observation noise, or action-dependent observations. The complete additive game
trace-correction workflow is `examples/lqg_feedback_game.py`.

## Fit a frozen policy, then inspect its BSDE view

First produce disjoint `ControlledPathBatch` values by rolling out the same identified
policy on training and holdout noise. Then fit only its value:

```python
fitted_problem = phx.control.stochastic.FittedBellmanProblem(
    training.paths,
    holdout.paths,
    feature_map,
    num_features=feature_count,
    feature_id="frozen-policy-features",
)
fitted_plan = phx.control.stochastic.FittedBellmanPlan(
    ridge=1.0e-8,
    plan_id="frozen-policy-bellman-plan",
    minimum_training_paths=feature_count,
)
fitted = phx.control.stochastic.fit_frozen_policy_bellman(
    fitted_problem, fitted_plan
)
if not bool(fitted.valid):
    raise RuntimeError(f"fitted Bellman evaluation failed: {fitted.status}")
```

Inspect training and holdout residuals separately. Ridge normal-equation evidence
does not replace the reported original normal-equation residual, and fitting never
performs policy improvement. `bridge_fitted_bellman_to_bsde` can then evaluate this
same frozen value on a selected current path batch. Its `physical_actions` remain
the policy outputs; its `martingale_integrands` are BSDE $Z$ values with
`z_shape`. They are never identified with one another.

## Keep SMP, dynamic programming, and SAA separate

| Workflow | Construct and evaluate | What a successful result means |
|---|---|---|
| Single-agent open-loop SMP | `StochasticMaximumPrincipleProblem`, then `evaluate_stochastic_maximum_principle` with supplied paths, adjoint values, martingale integrands, and pre-increment information labels | Pathwise forward/terminal/backward/measurability/conditional-stationarity evidence; a necessary condition unless separately declared convexity makes it sufficient |
| Multi-player open-loop SMP | `OpenLoopStochasticGameSMPProblem`, then `evaluate_open_loop_stochastic_game_smp` with one adjoint pair and information ID per player | Player-owned Hamiltonian-row evidence on supplied paths, not a constructed feedback strategy |
| HJB control reference | `DiscreteHJBProblem`, then `solve_discrete_hjb_reference` or `refine_discrete_hjb_reference` | Residual and nested-refinement evidence for one bounded scalar grid and finite action catalog |
| Zero-sum HJBI reference | `DiscreteZeroSumHJBIProblem`, then `solve_discrete_hjbi_reference` | Both declared action orders pass and their discrete Isaacs gap is within tolerance |
| Coupled all-minimizer HJB | `DiscreteCoupledHJBProblem` plus `CoupledHJBPolicyIterationPlan`, then `solve_coupled_hjb_reference` | One selected local feedback fixed-point branch for supplied starts, update order, damping, and bounded grid |
| Policy-game SAA | `StochasticPolicyGameProblem`, then the `plan_stochastic_policy_game` / `prepare_stochastic_policy_game` / `solve_prepared_stochastic_policy_game` lifecycle | Local stationarity of the frozen training empirical pseudo-gradient; the untouched holdout cluster costs remain evaluation evidence only |

The HJB, HJBI, and coupled-HJB results are finite-grid references, not continuum
viscosity-solution certificates. SMP does not solve a Bellman equation. SAA does not
turn a finite policy parameterization or empirical root into population or feedback
Nash. `examples/hjbi_reference_game.py` exercises the separate lower/upper HJBI
orders and all of their discrete gates.

## Build mean-field evidence in explicit stages

The mean-field layers are intentionally not one combined solver:

1. Build an `EmpiricalMeanField` with explicit particle/time axes, weights,
   `mean_field_id`, and `source_path_id`.
2. Adapt one supplied law with `adapt_mean_field_control_bsde`, then evaluate a
   `FrozenLawBestResponseProblem` using `solve_frozen_law_best_response`.
3. If law consistency is required, provide a genuinely new
   `induced_flow(response, args)` callback and run
   `solve_mean_field_game_fixed_point` on a `MeanFieldGameFixedPointProblem` and
   `MeanFieldGameFixedPointPlan`.
4. For finite common-noise support, keep one conditional law and public history per
   scenario in `CommonNoiseMeanFieldProblem`; run
   `solve_common_noise_mean_field_fixed_point` without mixing the scenarios first.
5. For individual or aggregate constraints, choose a
   `MeanFieldConstraintConcept`, the matching `GameMultiplierLayout`, and run
   `solve_constrained_mean_field_game`. Its result is sampled KKT evidence.
6. For a social planner, supply the mandatory `MeanFieldExternality` and use
   `evaluate_mean_field_control_planner`. Analytic Lions data and finite-particle
   adjoint data with a bias bound are distinct modes.
7. For a finite-$N$ statement, separately evaluate the finite population and every
   unilateral deviation through `evaluate_finite_population_continuation`. An MFG
   fixed point alone is insufficient.

`examples/mean_field_game.py` runs the frozen-law and independently induced-law
steps. A `FiniteStateCommonInformationGame` instead performs pure-prescription
Bayesian backward induction over a declared finite public state. A
`FiniteStateMasterEquationProblem` instead enumerates a finite physical-state and
exact empirical-law lattice. `solve_common_information_game` and
`solve_finite_state_master_equation_reference` therefore answer different questions;
the master result's neighbor-transfer differences are not Lions derivatives.

Every step checks its own result label, validity, status, evidence, and provenance.
There is no silent control clipping, covariance or law repair, active-set search,
conditional-law mixing, mixed-strategy substitution, or method fallback.

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
