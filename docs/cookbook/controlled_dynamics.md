# Controlled dynamics

This recipe uses only the public Phydrax adapters for two related workflows:

1. integrate a differentiable controlled path, including exact landing at path-derivative breakpoints;
2. quantify numerical and declared model uncertainty for an ODE, then estimate the deterministic flow's finite-time Lyapunov spectrum.

A CDE and an RDE are not interchangeable. `solve_diffrax_cde` accepts a differentiable first-level `AbstractDifferentiableDrivingPath`. A rough control with second-level information belongs to `solve_rough_differential`. Do not replace the Phydrax adapter with an ordinary Diffrax `ControlTerm`: that would bypass path IDs, breakpoint masks, landing policy, and solution provenance.

## A sampled controlled differential equation

The control below has physical channels `(time, time²)`. Its masks preserve the physical sample and channel axes. `PiecewiseLinearDrivingPath.fit` returns both the path and explicit interpolation diagnostics.

```python
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import phydrax as phx

sample_times = jnp.asarray([0.0, 0.25, 0.6, 1.0])
control_values = jnp.stack((sample_times, sample_times**2), axis=-1)
time_mask = jnp.ones(sample_times.shape, dtype=bool)
value_mask = jnp.ones(control_values.shape, dtype=bool)

path, fit = phx.solver.PiecewiseLinearDrivingPath.fit(
    sample_times,
    control_values,
    time_mask=time_mask,
    value_mask=value_mask,
    path_id="time-and-square:piecewise-linear",
)


def controlled_field(
    time: jax.Array,
    state: jax.Array,
    context: Mapping[str, jax.Array],
) -> jax.Array:
    del time
    first = context["gain"] * (1.0 - state**2)
    second = jnp.ones_like(state)
    return jnp.stack((first, second), axis=-1)


problem = phx.solver.RoughDifferentialProblem(
    controlled_field,
    jnp.asarray([0.0]),
    driver_dimension=2,
    args={"gain": jnp.asarray(0.4)},
    problem_id="bounded-controlled-state",
)
solution = phx.solver.solve_diffrax_cde(
    problem,
    path,
    save_times=sample_times,
    rtol=1e-6,
    atol=1e-8,
)

assert bool(fit.valid)
assert fit.backend == "closed-form"
assert bool(solution.successful)
assert solution.states.shape == (sample_times.size, 1)
assert solution.path_id == path.path_id
```

`controlled_field(time, state, context)` is context-last and returns the exact physical shape `state_shape + (driver_dimension,)`, here `(1, 2)`. The active interior samples appear in `path.breakpoints` with an aligned Boolean `breakpoint_mask`. `solve_diffrax_cde` wraps the adaptive controller so every active derivative jump is both a step boundary and a jump boundary. It never smooths, shifts, or skips a knot.

For trainable controls, use `FixedBSplineDrivingPath`: its validated grid is non-trainable topology and its finite inexact coefficient array is a differentiable JAX leaf. Gradients preserve the leading spline-basis axis and all payload axes. For neural CDEs, `NeuralCDETrainingData` retains physical irregular times and validity per case, while `train_neural_cde` resumes only from a matching `NeuralCDETrainingState`; it does not reset optimizer or batch position.

## Probabilistic ODE and Lyapunov workflow

A probabilistic ODE posterior and a Lyapunov spectrum answer different questions. The first attributes uncertainty in a discretized deterministic trajectory. The second measures finite-time tangent growth of a declared deterministic map or flow. Sharing one drift keeps the physical model and parameter convention aligned, but the posterior covariance is not automatically propagated into the Lyapunov estimate.

```python
import jax
import jax.numpy as jnp
import phydrax as phx


def decay(
    time: jax.Array,
    state: jax.Array,
    rate: jax.Array,
) -> jax.Array:
    del time
    return rate * state


ode = phx.solver.DifferentialProblem(
    decay,
    jnp.asarray([1.0]),
    t0=0.0,
    t1=2.0,
    args=jnp.asarray(-0.4),
)
save_times = jnp.linspace(0.0, 2.0, 11)
posterior = phx.solver.solve_probabilistic_ode(
    ode,
    save_times=save_times,
    method=phx.solver.ProbabilisticODEMethod(
        order=2,
        update="ek1",
        num_steps=64,
        adaptive=True,
        diffusion_calibration="quasi_mle",
        covariance_output="matrix_free",
    ),
    initial_covariance=jnp.asarray([[1e-4]]),
    process_covariance=jnp.asarray([[2e-5]]),
    observation_covariance=jnp.asarray([[1e-6]]),
    parameter_covariance=jnp.asarray([[1e-4]]),
)

layout = phx.dynamics.StateLayout((1,), component_names=("state",))
flow = phx.dynamics.ContinuousSystem(
    decay,
    state_layout=layout,
    system_id="scalar-decay",
)
evolution = phx.dynamics.DiffraxEvolution(flow, rtol=1e-9, atol=1e-11)
lyapunov_grid = phx.dynamics.TimeGrid(
    jnp.linspace(0.0, 2.0, 201),
    time_id="scalar-decay-analysis",
)
spectrum = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
    evolution,
    jnp.asarray([1.0]),
    lyapunov_grid,
    args=jnp.asarray(-0.4),
    qr_interval=10,
    burn_in=20,
    accumulation_interval=50,
)

assert bool(posterior.successful)
assert posterior.means.shape == (save_times.size, 1)
assert posterior.uncertainty_sources == (
    "numerical",
    "process",
    "observation",
    "initial_condition",
    "parameter",
)
assert bool(spectrum.valid)
assert spectrum.full_spectrum
assert bool(spectrum.kaplan_yorke_valid)
```

Adaptation performs a uniform pilot pass and redistributes the fixed `num_steps`; it is not a variable-capacity fallback. Quasi-MLE calibration rescales only the numerical integrated-Wiener covariance. The other four source covariances retain their declared meaning and can be applied independently with `posterior.covariance_matvec(..., source=...)`. Dense covariance materialization remains guarded by `max_dense_dimension`; selecting block-diagonal factorization is an explicit approximation, never an automatic repair.

The Lyapunov result records the chosen evolution backend, discretization, grid, tangent
method, cadence, burn-in, checkpoint, validity, and status. Resume requires matching
provenance. Kaplan–Yorke dimension is valid only for a finite, valid full spectrum; a
leading-only estimate is marked noncertifying.

See [API → Differential equation integration](../api/solver/differential.md) for every
path, neural-CDE, and probabilistic ODE symbol. See
[API → Dynamical systems, identification, and chaos](../api/dynamics.md) for tangent and
Lyapunov analysis.
