"""Passive periodic Landau-de Gennes Q-tensor relaxation."""

import jax.numpy as jnp

import phydrax as phx


grid = phx.discretization.TensorGridPlan(
    (
        phx.discretization.UniformCellAxisSpec(32, periodic=True),
        phx.discretization.UniformCellAxisSpec(32, periodic=True),
    ),
    axis_names=("x", "y"),
).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
finite_difference = phx.discretization.periodic_finite_difference(grid)
basis = phx.equations.NematicTensorBasis(2)
dynamics = phx.solver.PreparedNematicDynamics(
    finite_difference,
    phx.equations.LandauDeGennesClosure(basis),
    phx.equations.LandauDeGennesParameters(-1.0, 0.0, 1.0, 0.02),
    phx.equations.BerisEdwardsParameters(0.5, 0.7),
)
x, y = jnp.meshgrid(
    (jnp.arange(32) + 0.5) / 32.0,
    (jnp.arange(32) + 0.5) / 32.0,
    indexing="ij",
)
state = jnp.stack(
    (
        0.05 * jnp.cos(2.0 * jnp.pi * x),
        0.05 * jnp.sin(2.0 * jnp.pi * y),
    ),
    axis=-1,
)
for _ in range(50):
    result = dynamics.step(state, 1.0e-4)
    state = result.compact_q

print("successful:", bool(result.successful))
print("free energy:", float(result.evaluation.total_free_energy))
