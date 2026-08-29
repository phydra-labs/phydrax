"""Solve a controlled index-one DAE by JAX-native direct collocation."""

import jax.numpy as jnp

import phydrax as phx


system = phx.dynamics.DifferentialAlgebraicSystem(
    lambda time, state, state_rate, control, args: jnp.asarray(
        (
            state_rate[0] - control[0],
            state[1] - state[0],
        )
    ),
    state_shape=(2,),
    structure=phx.dynamics.DAEStructure(("differential", "algebraic")),
    input_layout=phx.dynamics.InputLayout((1,), roles="control"),
    system_id="example-controlled-dae",
)
terminal = phx.control.BoundedTrajectoryConstraint(
    lambda trajectory, args: trajectory.final_state[0],
    lower=1.0,
    upper=1.0,
    constraint_id="example-terminal-state",
)
problem = phx.control.TrajectoryOptimizationProblem(
    system,
    initial_state=jnp.asarray((0.0, 0.0)),
    running_cost=lambda time, state, control, args: 0.5 * control[0] ** 2,
    trajectory_constraints=(terminal,),
    problem_id="example-direct-collocation-dae",
)
mesh = phx.discretization.TemporalMesh(
    jnp.linspace(0.0, 1.0, 6),
    role="collocation",
    mesh_id="example-direct-collocation-mesh",
)
plan = phx.control.DirectCollocationPlan(
    mesh,
    method=phx.solver.ThetaMethod(0.5, endpoint=False),
    audit=phx.control.DirectCollocationAuditPolicy(
        defect_tolerance=1.0e-7,
        constraint_tolerance=1.0e-7,
        off_grid_points=2,
    ),
    plan_id="example-direct-collocation-plan",
)
initial_times = mesh.nodes
initial_states = jnp.stack((initial_times, initial_times), axis=-1)
initial_controls = jnp.ones((mesh.num_steps, 1))
result = phx.control.solve_direct_collocation(
    problem,
    plan,
    initial_states,
    initial_controls,
    method=phx.optim.PrimalDualInteriorPoint(
        mode="dense-filter", max_dense_dimension=128
    ),
    termination=phx.optim.OptimizationTermination(
        absolute_optimality=1.0e-8,
        relative_optimality=0.0,
        maximum_steps=80,
    ),
)
if not bool(result.successful):
    raise RuntimeError(
        "direct collocation failed: "
        f"status={result.status}, "
        f"optimizer_status={result.optimization_result.status}"
    )
if result.diagnostics.maximum_defect > plan.audit.defect_tolerance:
    raise RuntimeError("direct collocation returned an uncertified dynamics defect")
if result.diagnostics.maximum_constraint_violation > plan.audit.constraint_tolerance:
    raise RuntimeError("direct collocation returned an infeasible constraint")
if result.optimization_result.certificate is None:
    raise RuntimeError("direct collocation returned no KKT certificate")

print(
    {
        "objective": float(result.objective),
        "maximum_defect": float(result.diagnostics.maximum_defect),
        "maximum_constraint_violation": float(
            result.diagnostics.maximum_constraint_violation
        ),
        "jacobian_nonzeros": result.diagnostics.jacobian_nonzeros,
        "status": int(result.status),
    }
)
