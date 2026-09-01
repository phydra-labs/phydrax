"""Build and audit the native mass-matrix heat DAE with virtual elements."""

import jax.numpy as jnp

import phydrax as phx


coordinates = jnp.asarray(
    (
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
        (1.0, 0.5),
        (0.0, 1.0),
        (0.5, 1.0),
        (1.0, 1.0),
    )
)
polygons = (
    (0, 1, 4, 3),
    (1, 2, 5, 4),
    (3, 4, 7, 6),
    (4, 5, 8, 7),
)
mesh = phx.discretization.CellMesh.from_polygons(coordinates, polygons)
field = phx.discretization.VirtualElementFieldSpec(
    "temperature",
    phx.discretization.conforming_h1_virtual_element(1),
)
space = phx.discretization.VirtualElementPlan(mesh, field).prepare()
constraint = phx.discretization.virtual_element_dirichlet_constraint(space, "temperature")
form = phx.equations.VirtualElementForm(
    "heat-diffusion",
    "temperature",
    (phx.equations.DiffusionAction("temperature", 1.0),),
)
compiled = phx.equations.compile_virtual_element_problem(
    form,
    space,
    constraint=constraint,
    dirichlet_values=0.0,
)
system = compiled.as_dae_system()
state = jnp.zeros(compiled.state_space.shape)
state_rate = jnp.zeros_like(state)
residual = system.evaluate(0.0, state, state_rate)
mass = compiled.mass_operator()
print(
    {
        "state_size": compiled.state_space.size,
        "dae_residual_norm": float(jnp.sqrt(jnp.sum(residual * residual))),
        "mass_action_norm": float(jnp.sqrt(jnp.sum(mass.mv(jnp.ones_like(state)) ** 2))),
        "system_id": system.system_id,
    }
)
