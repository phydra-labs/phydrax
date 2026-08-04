# Backward stochastic equations and semilinear PDEs

This recipe builds one Brownian path ensemble, evaluates a Markovian BSDE, differentiates
the value model to obtain its control, and reuses the same residual as a trainable
objective. It then shows the contracts for fully coupled forward-backward systems and
finite-activity jump compensation.

## 1. Generate forward paths once

```python
import jax.numpy as jnp
import jax.random as jr
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
    sample_shape=(2048,),
    tolerance=1e-3,
    noise_id="brownian",
)
times = jnp.linspace(0.0, 1.0, 33)
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

value = lambda t, state: jnp.asarray(
    [state[0] ** 2 + 1.0 - t]
)
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
domain = (
    phx.domain.Interval1d(-8.0, 8.0)
    @ phx.domain.TimeInterval(0.0, 1.0)
)

@domain.Function("t", "x")
def value_model(t, state):
    return jnp.asarray([state[0] ** 2 + 1.0 - t])

objective = phx.objectives.BSDEObjective(
    problem,
    value_name="value",
    control_mode="autodiff",
    mode="joint",
    sampling_mode="fixed",
    fixed_paths=paths,
)
objective_value = objective.loss({"value": value_model})
```

The same objective can be passed in the `objectives` collection of a
`FunctionalSolver`. The value model remains a normal `DomainFunction`; the objective
owns path sampling and stochastic-residual reduction rather than introducing a second
training loop.

## 4. Fully coupled explicit forward-backward paths

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

## 5. Add finite-activity jump compensation

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

## 6. Validation checklist

- Refine the time grid and report local and global residual convergence separately.
- Keep optimization paths fixed when comparing architectures or ablations.
- Use realization-independence labels in confidence intervals; antithetic paths are one
  dependence cluster, not two independent samples.
- Check the terminal residual independently of the interior equation.
- For jump equations, validate event counts and compensator moments before interpreting
  the BSDE loss.
- Export evaluations with `phx.uq.export_result` when residual and path provenance must
  survive without the live model.
