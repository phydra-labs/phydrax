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
    return 0.5 * (state @ context["Q"] @ state + control @ context["R"] @ control)


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
    raise RuntimeError(f"LQR policy rollout failed with status {lqr_evaluation.status}")
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

compilation = phx.control.compile_linear_quadratic_control(
    linear_quadratic,
    compilation_policy=phx.control.LinearControlCompilationPolicy("sparse"),
)
qp_policy = phx.optim.ConvexSolvePolicy(phx.optim.DensePrimalDualQP())
prepared = phx.control.prepare_linear_quadratic_control(
    linear_quadratic,
    policy=qp_policy,
    compilation_policy=phx.control.LinearControlCompilationPolicy("sparse"),
)
qp_solution = phx.control.solve_prepared_linear_quadratic_control(prepared)

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

`solve_linear_quadratic_control(linear_quadratic, policy=qp_policy)` is the
one-shot form. The explicit lifecycle separates structural compilation, numeric
refresh, solve, and decode. Native state/control bounds stay outside user
polyhedral row axes, and their multipliers are retained separately.

The dense method keeps its explicit KKT dimension guard. The sparse compilation
policy emits shared-pattern sparse Hessian/equality/inequality operators without
changing the exact decision layout. QPax and optional MPAX remain explicitly
selected methods; no backend is chosen from problem size.

## Run receding-horizon MPC

```python
mpc = phx.control.solve_receding_horizon_mpc(
    linear_quadratic,
    prediction_horizon=3,
    terminal_policy="global",
    policy=qp_policy,
    warm_start_policy=phx.control.MPCWarmStartPolicy(
        terminal_control="hold",
        interior_margin=1e-7,
    ),
)
if not bool(mpc.successful):
    raise RuntimeError(f"MPC failed with status {mpc.status}")
```

`terminal_policy="global"` applies terminal cost and constraints only to a prediction
window that reaches the original final node. Choose `"always"` to impose them at every
local endpoint or `"none"` to omit them. MPC prepares one template per horizon/terminal
topology, refreshes compatible numeric windows, and shifts primal and dual state through
the declared layouts when `MPCWarmStartPolicy` is supplied. `mpc.subproblem_solutions`,
`mpc.qp_results`, `mpc.stage_valid`, and the exact affine state handoff remain visible.

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

## Optimize an implicit DAE by direct collocation

Multiple shooting repeatedly integrates explicit dynamics. Direct collocation instead
makes every state node and interval control an optimization variable and enforces an
implicit residual locally. This is the appropriate route when the natural model is a
regular DAE, when open-loop integration is fragile, or when shared physical parameters
must be optimized with the trajectory.

The example enforces the differential equation `ydot = u` and algebraic equation
`z = y`. It fixes the initial state, requires `y(1) = 1`, and minimizes control energy.
The midpoint solution is `y = z = t` and `u = 1`.

```python
dae = phx.dynamics.DifferentialAlgebraicSystem(
    lambda time, state, state_rate, control, args: jnp.asarray(
        (
            state_rate[0] - control[0],
            state[1] - state[0],
        )
    ),
    state_shape=(2,),
    structure=phx.dynamics.DAEStructure(("differential", "algebraic")),
    input_layout=phx.dynamics.InputLayout((1,), roles="control"),
    system_id="controlled-index-one-dae",
)
terminal = phx.control.BoundedTrajectoryConstraint(
    lambda trajectory, args: trajectory.final_state[0],
    lower=1.0,
    upper=1.0,
    constraint_id="unit-terminal-state",
)
trajectory_problem = phx.control.TrajectoryOptimizationProblem(
    dae,
    initial_state=jnp.asarray((0.0, 0.0)),
    running_cost=lambda time, state, control, args: 0.5 * control[0] ** 2,
    trajectory_constraints=(terminal,),
    problem_id="controlled-dae-energy",
)
collocation_mesh = phx.discretization.TemporalMesh(
    jnp.linspace(0.0, 1.0, 6),
    role="collocation",
    mesh_id="controlled-dae-collocation",
)
collocation_plan = phx.control.DirectCollocationPlan(
    collocation_mesh,
    method=phx.solver.ThetaMethod(0.5, endpoint=False),
    audit=phx.control.DirectCollocationAuditPolicy(
        defect_tolerance=1e-7,
        constraint_tolerance=1e-7,
        off_grid_points=2,
    ),
    plan_id="controlled-dae-midpoint",
)
initial_states = jnp.stack(
    (collocation_mesh.nodes, collocation_mesh.nodes),
    axis=-1,
)
initial_controls = jnp.ones((collocation_mesh.num_steps, 1))
collocated = phx.control.solve_direct_collocation(
    trajectory_problem,
    collocation_plan,
    initial_states,
    initial_controls,
    method=phx.optim.FilterInteriorPoint(max_dense_dimension=128),
    termination=phx.optim.OptimizationTermination(
        absolute_optimality=1e-8,
        relative_optimality=0.0,
        maximum_steps=80,
    ),
)
if not bool(collocated.successful):
    raise RuntimeError(
        "direct collocation failed: "
        f"status={collocated.status}, "
        f"optimizer={collocated.optimization_result.status}"
    )
if collocated.optimization_result.certificate is None:
    raise RuntimeError("direct collocation returned no KKT certificate")
```

Inspect both `maximum_defect` and `maximum_constraint_violation`. The separately sampled
`maximum_off_grid_defect` detects interpolation error but is not an enforced
continuous-time certificate. The result therefore records
`off_grid_certified=False`.

The native `FilterInteriorPoint` route remains dense and enforces its declared dimension
guard. For larger transcriptions, explicitly select `IpoptMinimize`; the structured path
supplies exact sparse constraint Jacobian values and topology. No backend is chosen by
problem size and no backend failure triggers a fallback.

## B-spline controls, finite catalogs, and bounded initialization

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

# Exact selection when the complete coefficient choices are known.
catalog = phx.optim.FiniteProductSpace(
    phx.optim.FiniteAxis(
        jnp.stack(
            (
                initial,
                0.5 * jnp.ones(spline.parameter_shape),
                -0.5 * jnp.ones(spline.parameter_shape),
            )
        )
    )
)
catalog_seed = phx.control.search_control_candidates(
    problem,
    spline,
    catalog,
    search=phx.optim.FiniteExhaustiveSearch(batch_size=2),
)
if not catalog_seed.valid:
    raise RuntimeError(catalog_seed.termination_reason)

# Differential evolution instead searches a bounded continuous coefficient box.

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

`catalog_seed` certifies the exact minimum only among the three declared coefficient
arrays. It records three search evaluations and one separate winner reconstruction;
it does not retain three trajectories. Inspect its invalid count, selected flat index,
candidate signature, and `total_control_evaluations`.

Inspect `refined.transfer.method`, `condition_estimate`, and
`projection_error_bound`; an L2 transfer is an approximation, while the nested,
equal-degree refinement above is exact. The bounded stochastic search records its key,
design signature, population, best-objective history, invalid count, termination
reason, and all problem and approximation IDs. `population_converged` measures
population dispersion. Neither it nor the best objective proves global optimality or
basin coverage.

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
- Bounded differential evolution is a stochastic initializer or best-found bounded
  search, not a globally optimal control certificate.
