#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Two frictional disks settling against an exact signed-distance container."""

import jax
import jax.numpy as jnp

import phydrax as phx


particles = phx.discretization.ParticleSetPlan(
    jnp.asarray([100, 200]),
    jnp.asarray([1.0, 1.0]),
    ambient_dimension=2,
).prepare()
spheres = phx.discretization.RigidSphereSetPlan(
    jnp.asarray([0.1, 0.1]),
    jnp.asarray([0, 0]),
)
materials = phx.equations.DEMMaterialTable(
    jnp.asarray([1.0e6]),
    jnp.asarray([0.25]),
    jnp.asarray([[0.7]]),
    jnp.asarray([[0.5]]),
)
barrier = phx.discretization.ImplicitDEMBarrier(
    phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile(),
    phx.discretization.DEMBarrierSide.INTERIOR,
    0,
    barrier_id="settling-container",
)
contact = phx.discretization.DEMContactModelPlan(
    phx.discretization.LinearSpringDashpotNormalPlan(1.0e4),
    tangential=phx.discretization.CundallStrackTangentialPlan(2.5e3),
)
method = phx.discretization.SoftSphereDEMMethodPlan(
    contact,
    maximum_overlap_fraction=0.2,
)
problem_ir = phx.equations.DiscreteElementProblemIR(
    "settling-disks",
    materials,
    gravity=jnp.asarray([0.0, -9.81]),
    barriers=(barrier,),
)
compiled = phx.equations.compile_discrete_element_problem(
    problem_ir,
    particles,
    spheres,
    method,
    neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
)
initial = compiled.initialize_state(
    0.0,
    jnp.asarray([[0.0, -0.91], [0.0, -0.72]]),
    jnp.zeros((2, 2)),
)
problem = phx.solver.FixedStepProblem(
    phx.solver.DEMFixedStepMethod(compiled.dynamics),
    initial,
    t0=0.0,
    t1=0.002,
    step_size=1.0e-4,
    state_geometry=compiled.dynamics.state_geometry,
    discretization_bundle=compiled.discretization_bundle,
)
solution = phx.solver.solve_fixed_step(problem)
final_state = jax.tree.map(lambda leaf: leaf[-1], solution.states)
diagnostics = compiled.diagnostics(solution.times[-1], final_state)

print(f"successful={bool(solution.successful)}")
print(f"active_contacts={int(diagnostics.active_contacts)}")
print(f"maximum_overlap_fraction={float(diagnostics.maximum_overlap_fraction):.6f}")
print(
    "contact_balance_loss="
    f"{float(diagnostics.energy.cumulative_contact_balance_loss):.6e}"
)
print(
    "relative_energy_residual="
    f"{float(diagnostics.energy.last_relative_energy_residual):.6e}"
)
