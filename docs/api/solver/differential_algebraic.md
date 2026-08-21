# Differential-algebraic equation integration

`phydrax.solver` integrates regular finite-dimensional implicit systems of the form

\[
F(t, y, \dot y, a) = 0
\]

on a declared `TimeGrid`. The implementation is native Phydrax: consistency and every
BDF stage use the prepared nonlinear and linear-solve lifecycles from
`phydrax.nonlinear` and `phydrax.linalg`. It does not route through Optimistix or
Diffrax.

This path deliberately covers fixed-grid, regular index-one problems. It does not
infer a differentiation index, reduce a higher-index system, adapt the step size, or
handle events. Those capabilities require separate structural and trajectory
contracts rather than hidden heuristics in the integrator.

## System and structure

`DifferentialAlgebraicSystem` owns only the state-shaped residual and its invariant
metadata. Initialization and integration policy belong to the problem and solver.
`DAEStructure` declares differential/algebraic roles for variables and equations along
one component axis. For an unstructured state, use one role with
`component_axis=None`.

The residual, state, and state rate must all have `state_shape`. `state_scale`,
`state_rate_scale`, and `residual_scale` are independent positive, finite, real arrays
broadcast to that shape:

- state and state-rate scales define the coordinate weights used by consistency and
  BDF nonlinear solves;
- residual scale defines the equation weights used by convergence checks;
- none of the scales changes the raw value returned by `system(...)`.

Nontrivial state geometry is rejected by the fixed-grid BDF backend. The current BDF
formula combines ambient Euclidean states and therefore cannot honestly preserve a
manifold-valued state.

```python
import jax
import jax.numpy as jnp
import phydrax as phx


def residual(time, state, state_rate, decay):
    del time
    differential = state_rate[0] + decay * state[0]
    constraint = state[1] - state[0] ** 2
    return jnp.stack((differential, constraint))


system = phx.dynamics.DifferentialAlgebraicSystem(
    residual,
    state_shape=(2,),
    structure=phx.dynamics.DAEStructure(
        ("differential", "algebraic"),
    ),
    state_scale=jnp.asarray((1.0, 1.0)),
    state_rate_scale=jnp.asarray((1.0, 1.0)),
    residual_scale=jnp.asarray((1.0, 1.0)),
    system_id="decay-with-constraint",
)
```

`DifferentialAlgebraicSystem.from_mass_matrix` is a convenience constructor for
`M(t, y, a) @ ydot - f(t, y, a) = 0`. The caller still supplies the structural roles;
a singular mass matrix alone does not prove index-one regularity.

## Consistent initialization

`DifferentialAlgebraicProblem` stores guesses for both `y(0)` and `ydot(0)` and an
explicit `DAEInitializationSpec`. The default is `index_one()`:

- fix differential state components;
- solve algebraic state components;
- solve differential rate components;
- fix algebraic rate components while marking those rates semantically unavailable.

The other modes are explicit:

- `fixed_rate_state()` fixes the entire supplied state and solves the entire rate;
- `check_only()` changes nothing and only checks the supplied pair;
- `from_masks(fixed_state, fixed_rate)` defines custom fixed/free scalar masks.

A custom contract must expose exactly one free state-or-rate scalar per residual
scalar. Masks operate on flattened scalar coordinates after structural broadcasting;
they do not silently choose a least-squares initialization.

```python
problem = phx.solver.DifferentialAlgebraicProblem(
    system,
    jnp.asarray((1.0, 0.0)),
    initial_state_rate=jnp.zeros(2),
    args=jnp.asarray(0.5),
    initialization=phx.solver.DAEInitializationSpec.index_one(),
    problem_id="decay-with-constraint-ivp",
)

initialization = phx.solver.initialize_dae(problem, 0.0)
assert bool(initialization.valid)
```

`DAEInitializationResult` returns the consistent state and rate, their corrections,
fixed masks, componentwise rate validity, scaled residual threshold and norms, native
nonlinear status and diagnostics, and a stable initialization ID. A check-only result
has no nonlinear solve. Failed consistency is reported as evidence; the solver does
not replace a failed pair with the original guess.

## Fixed-grid BDF lifecycle

`DAESolvePolicy` selects `"bdf1"` or variable-step `"bdf2"`, native
`NewtonKrylov` or `NewtonTrustRegion` methods for initialization and stages,
termination contracts, a maximum adjacent-step ratio, and failure behavior.

BDF2 uses BDF1 for its first accepted step. Later stages use the exact coefficients
for the two adjacent grid intervals. A BDF2 grid whose adjacent interval ratio exceeds
`max_step_ratio` is rejected during planning; it is not silently downgraded. Every
accepted state is followed by an independent residual certification against the
configured nonlinear threshold.

Planning, preparation, and execution are separate:

1. `plan_dae` validates the grid, method, state contract, and static identities.
2. `prepare_dae` prepares reusable symbolic nonlinear/linear templates for the
   consistency problem and every BDF stage.
3. `solve_dae(prepared, ...)` solves consistency for the runtime state, rate, and
   model arguments, then refreshes numeric stage and Jacobian data while preserving
   both template identities.

```python
grid = phx.dynamics.TimeGrid(
    jnp.linspace(0.0, 1.0, 21),
    time_id="training-grid",
)
policy = phx.solver.DAESolvePolicy(
    integration_method="bdf2",
    nonlinear_method=phx.nonlinear.NewtonKrylov(),
    nonlinear_termination=phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-9,
        relative_residual=0.0,
        maximum_steps=20,
    ),
)
prepared = phx.solver.prepare_dae(problem, grid, policy=policy)
solution = phx.solver.solve_dae(prepared)
```

The prepared path supports JIT, JVP, VJP, and `vmap`. Each successful stage state has
implicit derivatives obtained from that stage's residual Jacobian; nonlinear
iteration history and diagnostics are nondifferentiable evidence. Dependencies on
previous accepted states are chained through the outer fixed-length scan. Grid times
are stop-gradient values: derivatives are for the declared discrete fixed-grid map,
not for a parameter-dependent time controller.

```python
def terminal_state(decay):
    solved = phx.solver.solve_dae(prepared, args=decay)
    return solved.states[-1, 0]

terminal_gradient = jax.jit(jax.grad(terminal_state))(jnp.asarray(0.5))
```

`DifferentialAlgebraicSolution` stores states, reconstructed BDF state rates, node and
step validity, orders, step sizes, residual and constraint norms, per-stage native
nonlinear statuses and work counts, initialization evidence, plan/method/linear-plan
provenance, and the original time identity. If initialization or one stage fails,
later nodes are `NOT_RUN`; no explicit fallback state is fabricated. Use
`failure="error"` when a non-successful solution must raise at the call boundary.

## Semidiscrete implicit PDE residuals

`compile_semidiscrete_dae` lowers validated `PDEProblemIR` into a
`CompiledSpatialResidual` and `DifferentialAlgebraicSystem`. It shares field layout,
coordinate, boundary-lift, parameter-binding, and expression validation with
`compile_semidiscrete_pde`, but retains temporal derivatives in the implicit
residual.

The required `equation_targets` mapping is a bijection from every equation name to
every field name. It determines residual component order and prevents positional
matching from silently changing the model. A differential target must contain exactly
one direct first temporal derivative of its target field. Algebraic targets contain
none. Temporal derivatives nested inside nonlinear or composite expressions are
rejected rather than approximated.

`SemidiscreteDAEStructuralReport` records targets, component roles, derivative counts,
and the explicit assumption `regular-index-1-required-unverified`. This is incidence
evidence, not a numerical rank certificate or an index claim. The compiled object's
`rate_jacobian` materializes the dense derivative of the residual with respect to the
state rate for diagnostics; the solver itself retains matrix-free native derivative
policies unless configured otherwise.

```python
compiled = phx.equations.compile_semidiscrete_dae(
    pde_problem,
    spatial_discretization,
    equation_targets={"momentum": "velocity", "constraint": "pressure"},
    parameter_values={"viscosity": 0.01},
)
dae_problem = phx.solver.DifferentialAlgebraicProblem(
    compiled.system,
    compiled.layout.pack(initial_fields),
    args=runtime_parameters,
)
```

Only semidiscrete states whose time derivative is representable by the PDE IR are
accepted. Unsupported temporal structure and a non-bijective residual layout fail at
compilation rather than becoming an invalid numerical solve.

## Trajectory interoperability

`phydrax.dynamics.identification.trajectory_data_from_differential_solution` accepts a
`DifferentialAlgebraicSolution`. It uses accepted BDF state rates as trajectory
derivatives, preserves componentwise rate validity, derives transition validity from
adjacent node validity, and retains the DAE plan in the source ID. The initialization
node is not labeled derivative-valid when the chosen consistency contract did not
determine all algebraic rates.

## API reference

::: phydrax.dynamics.DAEStructure

---

::: phydrax.dynamics.DifferentialAlgebraicSystem

---

::: phydrax.solver.DAEInitializationSpec

---

::: phydrax.solver.DAEInitializationResult

---

::: phydrax.solver.DAEInitializationStatus

---

::: phydrax.solver.DAESolvePolicy

---

::: phydrax.solver.DAESolvePlan

---

::: phydrax.solver.PreparedDAESolve

---

::: phydrax.solver.DifferentialAlgebraicProblem

---

::: phydrax.solver.DifferentialAlgebraicSolution

---

::: phydrax.solver.DAEStatus

---

::: phydrax.solver.initialize_dae

---

::: phydrax.solver.plan_dae

---

::: phydrax.solver.prepare_dae

---

::: phydrax.solver.solve_dae

---

::: phydrax.equations.SemidiscreteDAEStructuralReport

---

::: phydrax.equations.CompiledSpatialResidual

---

::: phydrax.equations.compile_semidiscrete_dae
