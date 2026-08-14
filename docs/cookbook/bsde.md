# Backward stochastic equations and semilinear PDEs

This recipe builds one Brownian path ensemble, evaluates a Markovian BSDE, differentiates
the value model to obtain its control, and reuses the same residual as a trainable
term. It then shows the contracts for fully coupled forward-backward systems and
finite-activity jump compensation.

## 1. Generate forward paths once

```python
import jax.numpy as jnp
import jax.random as jr
import optax
import phydrax as phx

forward_problem = phx.solver.DifferentialProblem(
    lambda t, state, args: jnp.zeros_like(state),
    jnp.asarray([0.0]),
    t0=0.0,
    t1=1.0,
    wiener_terms=(
        phx.solver.WienerTerm(
            "brownian",
            lambda t, state, args: jnp.ones((1, 1)),
            (1,),
            structure="additive",
            basis_id="brownian",
        ),
    ),
)
realization = phx.stochastic.WienerRealization(
    jr.key(0),
    (1,),
    support=(0.0, 1.0),
    sample_shape=(256,),
    tolerance=1e-3,
    noise_id="brownian",
)
times = jnp.linspace(0.0, 1.0, 17)
forward_solution = phx.solver.solve_diffrax_ensemble(
    forward_problem,
    save_times=times,
    realization=realization,
    dt0=1e-2,
)
paths = phx.stochastic.bsde_paths_from_differential_solution(
    forward_solution,
    path_id="heat-forward-v1",
    process_id="brownian",
)
```

The conversion recovers Wiener increments from the same realization queried by the
solver. Do not regenerate increments from a second key: the backward stochastic
integral must be aligned with the forward path.

## 2. Declare the backward equation

For the terminal condition $g(x)=x^2$ and zero generator, the exact value is
$u(t,x)=x^2+1-t$ under unit Brownian diffusion.

```python
problem = phx.stochastic.BSDEProblem(
    lambda key: paths,
    lambda t, state, args: jnp.zeros_like(state),
    lambda t, state, args: jnp.ones((1, 1)),
    lambda t, state, value, control, args: jnp.zeros_like(value),
    lambda state, args: jnp.asarray([state[0] ** 2]),
    state_shape=(1,),
    noise_shape=(1,),
    output_shape=(1,),
    problem_id="backward-heat",
    process_id="brownian",
)

value = lambda t, state: jnp.asarray([state[0] ** 2 + 1.0 - t])
evaluation = phx.stochastic.evaluate_bsde(
    problem,
    paths,
    value,
    control_mode="autodiff",
    quadrature="left",
)
loss = phx.stochastic.bsde_objective_loss(
    evaluation,
    mode="joint",
)
diagnostics = phx.stochastic.bsde_diagnostics(evaluation)
```

`controls` has shape `sample_shape + (num_steps,) + output_shape + noise_shape`.
For this example, autodiff computes $Z=2X$. Local residuals need not vanish on a finite
Euler grid: the uncompensated fluctuation $(\Delta W)^2-\Delta t$ remains pathwise even
for the exact continuous-time solution. It converges under refinement and averages to
zero under independent sampling.

Check the corresponding PDE directly:

```python
pde_residual = phx.stochastic.semilinear_pde_residual(
    problem,
    value,
    jnp.asarray(0.3),
    jnp.asarray([0.4]),
)
assert jnp.allclose(pde_residual, 0.0)
```

## 3. Attach the residual to functional training

Use fixed paths for deterministic common-random-number optimization or resample through
the problem for fresh Monte Carlo batches.

```python
domain = phx.domain.Interval1d(-8.0, 8.0) @ phx.domain.TimeInterval(0.0, 1.0)


@domain.Function("t", "x")
def value_model(t, state):
    return jnp.asarray([state[0] ** 2 + 1.0 - t])


term = phx.terms.BSDETerm(
    problem,
    value_name="value",
    control_mode="autodiff",
    mode="joint",
    sampling_mode="fixed",
    fixed_paths=paths,
)
term_value = term.loss({"value": value_model})
```

The same term can be passed in the `terms` collection of a `FunctionalSolver`. The
value model remains a normal `DomainFunction`; the term owns path sampling and
stochastic-residual reduction rather than introducing a second training loop.

## 4. Generate global Feynman--Kac regression labels

The local BSDE residual is useful for diagnostics, but a global time-conditioned model
can also be trained from conditional-expectation labels. Reuse every valid node of the
existing path ensemble:

```python
label_plan = phx.stochastic.FeynmanKacSamplingPlan(
    terminal_time=1.0,
    sampling_mode="trajectory_nodes",
    quadrature="left",
    control_target_mode="martingale",
    time_weighting="trapezoid",
    refresh_mode="fixed",
)
labels = phx.stochastic.trajectory_node_feynman_kac_labels(
    problem,
    paths,
    label_plan,
    key=jr.key(2),
)
regression = phx.terms.FeynmanKacRegressionTerm(
    problem,
    label_plan,
    value_name="value",
    labels=labels,
)
label_diagnostics = phx.stochastic.feynman_kac_label_diagnostics(labels)
```

Every node on one forward path is correlated with every later node on that path.
`labels.cluster_ids` preserves that fact. Trajectory-node value labels have no
replicated conditional continuations, so their per-node standard errors are not
claimed. Use query-conditioned sampling when conditional Monte Carlo uncertainty is
required:

```python
query_plan = phx.stochastic.FeynmanKacSamplingPlan(
    terminal_time=1.0,
    sampling_mode="queries",
    num_paths_per_query=256,
    num_time_steps=16,
    antithetic=True,
    control_target_mode="martingale",
)
query_labels = phx.stochastic.query_feynman_kac_labels(
    problem,
    query_plan,
    query_times=jnp.asarray([0.0, 0.25, 0.75]),
    query_states=jnp.asarray([[0.0], [0.5], [-0.5]]),
    key=jr.key(3),
)
```

The query distribution is part of the numerical method: it defines where a global
field is trained. Increasing continuations only reduces conditional Monte Carlo error;
it does not repair poor query coverage or time discretization bias.

## 5. Run global Deep Picard iteration

For a semilinear generator, `solve_deep_picard` freezes the current value/control
field, generates Feynman--Kac targets, and trains the next global field through
`FunctionalSolver`. Here a small MLP supplies the trainable `DomainFunction`, while a
query-conditioned plan makes the training distribution explicit:

```python
trainable_model = phx.nn.models.MLP(
    in_size=2,
    out_size=1,
    width_size=32,
    depth=2,
    key=jr.key(4),
)
trainable_value = domain.Model("t", "x")(trainable_model)
picard_solver = phx.solver.FunctionalSolver(
    functions={"value": trainable_value},
    terms=(),
)
picard_plan = phx.stochastic.FeynmanKacSamplingPlan(
    terminal_time=1.0,
    sampling_mode="queries",
    num_paths_per_query=32,
    num_time_steps=4,
    refresh_mode="fixed",
)
picard_result = phx.solver.solve_deep_picard(
    picard_solver,
    problem,
    value_name="value",
    sampling_plan=picard_plan,
    num_picard_steps=1,
    inner_num_iter=500,
    optim=optax.adam(1e-3),
    query_times=jnp.asarray([0.0, 0.5]),
    query_states=jnp.asarray([[0.0], [0.5]]),
    target_damping=0.8,
    convergence_tolerance=1e-3,
    relative_tolerance=1e-3,
    seed=4,
)
if not picard_result.converged:
    print(picard_result.diagnostics.relative_target_rmse)
```

Picard convergence is an observed numerical property, not guaranteed by the API.
Report target contraction and terminal error, and validate on an independent query
distribution when using query mode.

Fully nonlinear frozen sources must be explicit and matrix-free:

```python
def source_builder(context):
    return phx.solver.StructuredPicardSource(
        lambda t, x, ctx, args: 0.1 * ctx.covariance_trace(t, x),
        source_id="trace-source-v1",
    )
```

The context also exposes `value`, `gradient`, `control`, and
`directional_hessian`. It never exposes or materializes a dense Hessian.

## 6. Train a localized Deep BSDE shooting solution

Deep BSDE learns \(Y_0\) and the pathwise control directly. A trainable constant
`Domain.Parameter` is the canonical pointwise initial value; the control may instead
be a time-conditioned network for realistic high-dimensional problems.

```python
shooting_solver = phx.solver.FunctionalSolver(
    functions={
        "initial": domain.Parameter(jnp.asarray([0.0])),
        "control": domain.Parameter(jnp.asarray([[0.0]])),
    },
    terms=(),
)
shooting_result = phx.solver.solve_deep_bsde(
    shooting_solver,
    problem,
    initial_value_name="initial",
    control_name="control",
    num_iter=20,
    optim=optax.adam(1e-2),
    sampling_mode="fixed",
    fixed_paths=paths,
    validation_paths=paths,
    keep_best=False,
)
assert shooting_result.diagnostics.finite
```

The terminal RMSE is the shooting objective on the explicit validation paths.
Inspect it together with control scale and valid fraction. This result estimates the
declared initial state or initial distribution; it does not claim a reusable global
\(u(t,x)\).

## 7. Train backward deep-splitting slices

Deep splitting preserves a value-field result but trains sequentially backward. This
small example coarsens the existing aligned Brownian paths to two transitions; a
production model should be expressive in `x`.

```python
coarse_indices = jnp.asarray([0, 8, 16])
coarse_states = paths.states[:, coarse_indices, :]
coarse_paths = phx.stochastic.BSDEPathBatch(
    paths.times[coarse_indices],
    coarse_states,
    jnp.diff(coarse_states, axis=1),
    sample_shape=paths.sample_shape,
    state_shape=paths.state_shape,
    noise_shape=paths.noise_shape,
    path_id="heat-forward-coarse-v1",
    process_id=paths.process_id,
    valid=paths.valid[:, coarse_indices],
)
splitting_solver = phx.solver.FunctionalSolver(
    functions={"value": domain.Parameter(jnp.asarray([0.0]))},
    terms=(),
)
splitting_result = phx.solver.solve_deep_splitting(
    splitting_solver,
    problem,
    value_name="value",
    inner_num_iter=10,
    optim=optax.adam(1e-2),
    sampling_mode="fixed",
    fixed_paths=coarse_paths,
    validation_paths=coarse_paths,
    keep_best=False,
)
slice_value = splitting_result.solution(
    jnp.asarray(0.25),
    jnp.asarray([0.0]),
)
assert jnp.all(jnp.isfinite(slice_value))
```

`solution.at_node(i, state)` avoids interpolation. `solution.control(t, state)`
differentiates the interpolated field and contracts it with the declared diffusion.
Held-out one-step RMSE contains transition noise, so validate the resulting slice field
against an independent solution error measure as well.

## 8. Fully coupled explicit forward-backward paths


When the forward coefficients depend on current value and control predictions, declare
a `CoupledFBSDEProblem` and supply both predictors:

```python
coupled = phx.solver.CoupledFBSDEProblem(
    jnp.linspace(0.0, 1.0, 33),
    jnp.asarray([0.0]),
    lambda t, x, y, z, args: 0.1 * y,
    lambda t, x, y, z, args: jnp.ones((1, 1)),
    lambda t, x, y, z, args: jnp.zeros((1,)),
    lambda x, args: x,
    state_shape=(1,),
    noise_shape=(1,),
    output_shape=(1,),
    num_paths=1024,
    problem_id="coupled-example",
    process_id="coupled-wiener",
)
coupled_result = phx.solver.solve_coupled_fbsde_explicit(
    jr.key(1),
    coupled,
    lambda t, x: x,
    lambda t, x: jnp.ones((1, 1)),
)
assert jnp.all(coupled_result.successful)
```

This is an explicit Euler--Maruyama coupling. It is not an implicit fixed-point solver.
Pass an existing `WienerRealization` to replay the same paths across model comparisons.

## 9. Add finite-activity jump compensation

A jump-control callable receives the process label, event time, pre-jump state, channel,
mark, problem arguments, and a deterministic event key:

```python
def jump_control(label, time, pre_state, channel, mark, args, *, key):
    del label, time, channel, mark, args, key
    return jnp.asarray([1.0])


jump_problem = phx.stochastic.JumpBSDEProblem(
    problem,
    lambda label, time, state, control, args: jnp.asarray([2.0]),
    {"counting": "poisson-rate-2"},
    problem_id="jump-backward",
)
```

The compensator callable above returns the rate integral $\int U(t,e)\nu_t(de)$; it
must not include the time-step width. `evaluate_jump_bsde` multiplies the rate by the
aligned interval width and subtracts the compensated event increment from the Brownian
residual.

The supplied `BSDEPathBatch` must contain `jump_events={"counting": event_batch}` with
pre-jump states. If it carries realization provenance, use a
`CompositeStochasticRealization` containing one matching Wiener component and a
`PoissonClockRealization` named `"counting"`. Event-capacity exhaustion remains a
failed path and is reported by `jump_bsde_diagnostics`.

## 10. Validation checklist

- Refine the time grid and report local and global residual convergence separately.
- Keep optimization paths fixed when comparing architectures or ablations.
- Use realization-independence labels in confidence intervals; antithetic paths are one
  dependence cluster, not two independent samples.
- Check the terminal residual independently of the interior equation.
- For Deep BSDE, validate terminal mismatch on paths not used by optimizer updates and
  report the learned control error when an analytic control is available.
- For deep splitting, separate stochastic one-step regression RMSE from global
  value-field error and report both across the full backward grid.
- For jump equations, validate event counts and compensator moments before interpreting
  the BSDE loss.
- Export evaluations with `phx.uq.export_result` when residual and path provenance must
  survive without the live model.

The repository benchmark invokes these contracts through public APIs:

```bash
python tools/high_dimensional_pde_benchmarks.py \
  --suite methods --dimensions 10,100 --include-training
```

`query-feynman-kac` reports value and control standard errors plus control error from
actual conditional path simulation. `deep-picard` reports held-out global-field,
gradient, fixed-point-target, and terminal errors. `deep-bsde` reports initial-value,
control, and terminal-shooting errors. `deep-splitting` reports interpolated field
error, gradient error, stochastic one-step RMSE, and terminal error. Compilation,
steady execution, and total training time are separate. Training methods run only at
dimensions declared by their entries in `HIGH_DIMENSIONAL_METHOD_MATRIX`.
