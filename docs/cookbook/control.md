# Control workflows

This workflow defines one discrete finite-horizon plant, audits a supplied control, solves
its unconstrained linear-quadratic form with LQR, compiles and solves the canonical QP,
runs receding-horizon MPC, and uses the QP trajectory to initialize iLQR. Every callback
uses the public context-last state-space signature.

## Define one physical problem

The example uses unit physical intervals so that the sampled left-rectangle running cost
and the linear-quadratic stage cost have the same scaling.

```python
import jax
import jax.numpy as jnp
import phydrax as phx

horizon = 6
time_grid = phx.dynamics.TimeGrid(
    jnp.arange(horizon + 1, dtype=float),
    time_id="integrator-time",
)

A = jnp.asarray([[1.0]])
B = jnp.asarray([[1.0]])
Q = jnp.asarray([[1.0]])
R = jnp.asarray([[0.2]])
Q_terminal = jnp.asarray([[4.0]])
context = {"A": A, "B": B, "Q": Q, "R": R, "Q_terminal": Q_terminal}


def transition(
    time: jax.Array,
    state: jax.Array,
    control: jax.Array,
    context: dict[str, jax.Array],
) -> jax.Array:
    del time
    return context["A"] @ state + context["B"] @ control


def running_cost(
    time: jax.Array,
    state: jax.Array,
    control: jax.Array,
    context: dict[str, jax.Array],
) -> jax.Array:
    del time
    return 0.5 * (
        state @ context["Q"] @ state
        + control @ context["R"] @ control
    )


def terminal_cost(
    time: jax.Array,
    state: jax.Array,
    context: dict[str, jax.Array],
) -> jax.Array:
    del time
    return 0.5 * state @ context["Q_terminal"] @ state


dynamics = phx.control.DiscreteControlDynamics(
    phx.dynamics.DiscreteSystem(
        transition,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id="scalar-integrator",
    )
)
problem = phx.control.ControlProblem(
    dynamics,
    time_grid,
    jnp.asarray([2.0]),
    running_cost=running_cost,
    terminal_cost=terminal_cost,
    args=context,
    problem_id="scalar-regulation",
)
open_loop = phx.control.PiecewiseConstantControlParameterization(
    time_grid,
    problem.control_shape,
    parameterization_id="zero-open-loop",
)
baseline = problem.evaluate(
    open_loop,
    jnp.zeros(open_loop.parameter_shape),
)

if not bool(baseline.successful):
    raise RuntimeError(f"baseline control failed with status {baseline.status}")
```

The trajectory axes are `(num_times, 1)` for states and `(num_steps, 1)` for controls;
a leading `case_shape` would precede both. `baseline.trajectory.valid` is per node,
`baseline.status` is per case, and `baseline.feasibility.certified` is `False` because a
grid-sampled nonlinear feasibility check is not a continuous-domain certificate. The
trajectory also records the problem, dynamics, control, backend, method,
discretization, and approximation IDs.

## Solve the finite-horizon LQR

LQR requires every stage array to carry its time axis explicitly. Its policy is state
feedback, so roll it out through the same `ControlProblem` rather than trying to sample it
without states.

```python
A_steps = jnp.broadcast_to(A, (horizon,) + A.shape)
B_steps = jnp.broadcast_to(B, (horizon,) + B.shape)
Q_steps = jnp.broadcast_to(Q, (horizon,) + Q.shape)
R_steps = jnp.broadcast_to(R, (horizon,) + R.shape)

lqr = phx.control.finite_horizon_lqr(
    A_steps,
    B_steps,
    Q_steps,
    R_steps,
    Q_terminal,
    time_grid=time_grid,
    policy_id="scalar-regulation-lqr",
)
if not bool(lqr.valid):
    raise RuntimeError(f"LQR failed with status {lqr.status}")

# AffineFeedbackPolicy stores its gains; its coefficient token is scalar-shaped.
lqr_evaluation = problem.evaluate(lqr.policy, jnp.zeros(()))
if not bool(lqr_evaluation.successful):
    raise RuntimeError(
        f"LQR policy rollout failed with status {lqr_evaluation.status}"
    )
```

Use `lqr.feedback_gain`, `lqr.feedforward`, and `lqr.value` for the policy and value
function. Before accepting the result, inspect `lqr.diagnostics.maximum_kkt_residual`,
`maximum_riccati_residual`, `maximum_control_condition_number`, and `converged`. LQR does
not silently regularize an invalid control Hessian.

## Compile and solve the canonical QP

`LinearQuadraticControlProblem` is the affine discrete contract used by both direct QP
control and MPC. The compiler preserves every state and control variable, dynamics row,
and constraint row; it does not condense, clip, or repair the problem.

```python
linear_quadratic = phx.control.LinearQuadraticControlProblem(
    A_steps,
    B_steps,
    problem.initial_state,
    Q_steps,
    R_steps,
    Q_terminal,
    time_grid=time_grid,
    problem_id=problem.problem_id,
    dynamics_id=dynamics.dynamics_id,
)

compilation = phx.control.compile_linear_quadratic_control(linear_quadratic)
qp_result = phx.optim.solve_quadratic_program(
    compilation.qp,
    method="dense-primal-dual",
)
qp_solution = phx.control.decode_linear_control_solution(compilation, qp_result)

if not bool(qp_solution.successful):
    raise RuntimeError(
        "control QP failed: "
        f"status={qp_solution.status}, backend={qp_solution.qp_result.backend}"
    )

# These are exact slices of the backend primal, not reconstructed or projected values.
states = qp_solution.states
controls = qp_solution.controls
first_control_slice = compilation.decision_layout.control_slice(0)
first_dynamics_rows = compilation.constraint_layout.dynamics_slices[0]
```

`solve_linear_quadratic_control(linear_quadratic)` is the convenience form of the same
compile, solve, and decode sequence. Use the explicit form above when you need the
canonical `QuadraticProgram`, decision slices, constraint provenance, or a separately
configured public QP call.

The default dense guard is 512 for
`num_variables + num_equalities + 2 * num_inequalities`. Raise it deliberately only when
the intended dense solve fits the available memory. `regularization=0.0` is the default;
a nonzero value is an explicit model choice. To use QPax, select
`method="qpax-implicit"`. QPax's explicit differentiation mode is not a supported control
backend.

## Run receding-horizon MPC

```python
mpc = phx.control.solve_receding_horizon_mpc(
    linear_quadratic,
    prediction_horizon=3,
    terminal_policy="global",
    method="dense-primal-dual",
)
if not bool(mpc.successful):
    raise RuntimeError(f"MPC failed with status {mpc.status}")
```

`terminal_policy="global"` applies terminal cost and constraints only to a prediction
window that reaches the original final node. Choose `"always"` to impose them at every
local endpoint or `"none"` to omit them. `mpc.subproblem_solutions`, `mpc.qp_results`,
`mpc.stage_valid`, `mpc.trajectory.backend_status`, and `mpc.method_id` preserve every
local solve and exact affine state handoff. Warm starts are not implemented; passing one
raises instead of being ignored.

## Initialize iLQR from the QP control

The same public `ControlProblem` can be given directly to iLQR. The QP control already
has the required shape `(num_steps,) + control_shape`.

```python
ilqr = phx.control.solve_ilqr(
    problem,
    qp_solution.controls,
    regularization=1e-6,
    max_iterations=50,
)
if not bool(ilqr.successful):
    raise RuntimeError(f"iLQR failed with status {ilqr.status}")

replayed = problem.rollout(ilqr.policy, jnp.empty((0,)))
if not bool(replayed.successful):
    raise RuntimeError(f"iLQR policy replay failed with status {replayed.status}")
```

The regularization is the fixed diagonal shift used by every backward pass. It is not
adapted after a curvature failure. Inspect `ilqr.diagnostics.status`, `failed_step`,
objective and gradient histories, step sizes, expected versus actual reductions, and
`converged`. iLQR accepts one unbatched, unconstrained case. Differential dynamics also
require an explicit `DifferentialControlFlow`; no integration method is selected or
retried behind the caller's back.

For one nonlinear constrained case, use `solve_multiple_shooting` instead. It is a dense
SQP method with explicit Hessian and QP regularization, dense guard, defect audit, and
failure statuses. Its nonlinear path constraints are still enforced only at the declared
nodes; neither it nor `ControlProblem.evaluate` certifies feasibility between nodes.

## B-spline controls and bounded global initialization

Choose a piecewise-constant parameterization for discrete interval controls and direct
local seeds, piecewise-linear controls for continuous nodal interpolation, or a B-spline
when smoothness and a lower-dimensional coefficient vector are useful. A B-spline's
convex-hull certificate covers continuous componentwise control bounds only.

```python
import jax.random as jr

coarse_grid = phx.nn.models.BSplineGrid(
    jnp.asarray([0.0, 0.0, 3.0, 6.0, 6.0]),
    1,
)
spline = phx.control.BSplineControlParameterization(
    coarse_grid,
    problem.control_shape,
    parameterization_id="global-bspline",
)
lower = -2.0 * jnp.ones(spline.parameter_shape)
upper = 2.0 * jnp.ones(spline.parameter_shape)
initial = jnp.zeros(spline.parameter_shape)

search = phx.optim.DifferentialEvolutionSearch(
    16,
    20,
    design=phx.sampling.SobolDesign(scrambled=True),
)
global_seed = phx.control.search_control(
    problem,
    spline,
    search,
    key=jr.key(0),
    coefficient_bounds=(lower, upper),
    initial_coefficients=initial,
)
if not bool(global_seed.successful):
    raise RuntimeError(
        "bounded search found no usable candidate: "
        f"reason={global_seed.termination_reason}, "
        f"invalid={global_seed.invalid_candidates}"
    )

control_bound = spline.bound_certificate(
    global_seed.coefficients,
    -2.0,
    2.0,
)
if not bool(control_bound.certified):
    raise RuntimeError("the continuous B-spline control bound was not certified")

fine_grid = phx.nn.models.BSplineGrid(
    jnp.asarray([0.0, 0.0, 1.5, 3.0, 4.5, 6.0, 6.0]),
    1,
)
refined = spline.refine(
    fine_grid,
    global_seed.coefficients,
    parameterization_id="global-bspline-refined",
    method="exact",
)

# A local single-case solver can start from the best-found sampled controls.
local_from_global = phx.control.solve_ilqr(problem, global_seed.controls)
```

Inspect `refined.transfer.method`, `condition_estimate`, and
`projection_error_bound`; an L2 transfer is an approximation, while the nested,
equal-degree refinement above is exact. The bounded search records its key, design
signature, population, best-objective history, invalid count, termination reason, and all
problem and approximation IDs. `population_converged` measures population dispersion.
Neither it nor the best objective proves global optimality or basin coverage.

## Failure and certification rules

- Never use a trajectory solely because arrays were returned. Check the solver's
  `successful` or `valid`, its explicit `status`, and backend diagnostics.
- `ControlResult.feasibility.feasible` is a sampled statement;
  `feasibility.certified` remains `False`. Do not report certified nonlinear feasibility.
- QP, MPC, iLQR, and multiple shooting do not clip, project, add slack, repair an iterate,
  or change methods after failure. Requested regularization remains explicit.
- iLQR and multiple shooting are single-case methods. Use explicit case axes only with
  APIs whose result contracts declare them.
- Dense Lyapunov and Gramian routines default to `max_dimension=128`; dense QP-based
  routines default to `max_dense_dimension=512`. Use matrix-free actions or a different
  formulation rather than bypassing a guard accidentally.
- Bounded differential evolution is an initializer or best-found finite search, not a
  globally optimal control certificate.
