#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


# Runtime population and integer charge state.
grid = phx.discretization.TensorGridPlan(
    (phx.discretization.UniformCellAxisSpec(32, periodic=True),),
    axis_names=("x",),
).prepare(jnp.asarray([[0.0], [1.0]]))
particles = phx.discretization.ParticleSetPlan(
    jnp.arange(8), jnp.ones((8,)), ambient_dimension=1
).prepare()
population_plan = phx.discretization.ParticlePopulationPlan(particles)
population = population_plan.initialize()
charge_model = phx.discretization.pic.PICChargeModelPlan(
    -1.0,
    "electrons",
    minimum_charge_number=1,
    maximum_charge_number=1,
    initial_charge_number=1,
)
charge = charge_model.initialize(population)

# Conservative stochastic pair collisions.
velocity = jnp.stack(
    (
        jnp.linspace(-0.2, 0.2, 8),
        jnp.zeros((8,)),
        jnp.zeros((8,)),
    ),
    axis=-1,
)
collision = phx.discretization.pic.collisions.CoulombCollisionPlan(
    1.0, maximum_probability=0.2
).collide(
    velocity,
    population.mass,
    population.active,
    population.incarnation,
    jr.key(7),
    0.1,
)

# True 1D3V field/PIC coupling.
field_plan = phx.solver.CompatibleMaxwell1DPlan(grid)
field = field_plan.initialize()
transfer = phx.discretization.pic.ReducedPICTransferPlan(grid)
reduced = phx.solver.ReducedElectromagneticPICPlan(field_plan, transfer, -1.0)
state = phx.solver.ReducedElectromagneticPICState(
    phx.discretization.pic.PICParticleState(
        (jnp.arange(8, dtype=float)[:, None] + 0.5) / 8.0,
        collision.accepted_velocity,
    ),
    population,
    field,
    jnp.asarray(0.0),
    jnp.asarray(0, dtype=jnp.int32),
)
step = reduced.step(state, 1.0e-3)

# Simplicial ownership on an independent unstructured mesh.
mesh = phx.discretization.CellMesh(
    jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
    (phx.discretization.CellBlock("triangles", "triangle", jnp.asarray(((0, 1, 2),))),),
)
locator = phx.discretization.PreparedSimplicialCellLocator(mesh)
located = locator.locate(jnp.asarray(((0.2, 0.2), (0.4, 0.1))))

print(
    {
        "collision_successful": bool(collision.successful),
        "collision_momentum_defect": float(collision.momentum_defect),
        "reduced_pic_successful": bool(step.successful),
        "reduced_continuity_defect": float(step.continuity_defect),
        "unstructured_points_located": int(jnp.sum(located.inside)),
        "total_macrocharge": float(jnp.sum(charge_model.macrocharge(population, charge))),
    }
)
