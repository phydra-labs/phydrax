#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


axes = tuple(phx.discretization.UniformCellAxisSpec(16, periodic=True) for _ in range(2))
grid = phx.discretization.TensorGridPlan(axes, axis_names=("x", "y")).prepare(
    jnp.asarray([[0.0, 0.0], [1.0, 1.0]])
)
masses = jnp.asarray([1.0, 2.0, 1.5, 0.5])
particles = phx.discretization.ParticleSetPlan(
    jnp.asarray([11, 7, 23, 19]),
    masses,
    ambient_dimension=2,
).prepare()
position = jnp.asarray([[0.12, 0.18], [0.62, 0.25], [0.38, 0.78], [0.88, 0.72]])
velocity = jnp.asarray([[1.0, 0.0], [0.5, -1.0], [-0.25, 0.75], [1.5, 0.5]])
assignment = phx.discretization.TensorBSplineSplatAssignment(2)
prepared = phx.discretization.ParticleGridSplatPlan(grid, assignment=assignment).prepare(
    particles
)
state = prepared.build(position)
mass = prepared.deposit_content(state, masses)
momentum = prepared.deposit_content(state, masses[:, None] * velocity)
velocity_grid = prepared.reconstruct(state, velocity, masses)


def density_objective(current_position):
    current = prepared.build(current_position)
    density = prepared.deposit_content(current, masses).density
    return jnp.mean(density**2)


gradient = jax.jit(jax.grad(density_objective))(position)
print("particle mass", float(jnp.sum(masses)))
print("grid mass", float(jnp.sum(mass.content)))
print("balance defect", float(mass.balance.maximum_absolute_balance_defect))
print("partition defect", float(mass.balance.maximum_partition_defect))
print("covered grid entities", int(jnp.sum(velocity_grid.support)))
print("momentum", jnp.sum(momentum.content, axis=(0, 1)))
print("position-gradient norm squared", float(jnp.sum(gradient * gradient)))
print("assignment routes", prepared.route_count)
print("first-moment defect", float(jnp.max(jnp.abs(state.first_moments))))
print("gradient-sum defect", float(jnp.max(jnp.abs(state.gradient_sums))))
