"""Conservative periodic binary-electrolyte relaxation."""

import jax.numpy as jnp

import phydrax as phx


grid = phx.discretization.TensorGridPlan(
    (phx.discretization.UniformCellAxisSpec(64, periodic=True),),
    axis_names=("x",),
).prepare(jnp.asarray(((0.0,), (1.0,))))
bridge = phx.discretization.StructuredCochainBridge(grid)
schema = phx.equations.ChemicalSpeciesSchema(
    ("cation", "anion"),
    (
        phx.equations.ChemicalPhaseKind.LIQUID,
        phx.equations.ChemicalPhaseKind.LIQUID,
    ),
    jnp.asarray((0.023, 0.035)),
    ("M", "X"),
    jnp.asarray(((1, 0), (0, 1)), dtype=jnp.int32),
    jnp.asarray((1, -1), dtype=jnp.int32),
)
parameters = phx.equations.ElectrolyteTransportParameters(
    schema,
    jnp.asarray((1.0e-3, 1.0e-3)),
    300.0,
    1.0e8,
)
electrostatic = phx.solver.CochainElectrostaticPlan(
    bridge,
    phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge),
    permittivity=parameters.permittivity,
)
dynamics = phx.solver.PoissonNernstPlanckPlan(
    electrostatic,
    phx.equations.IdealDiluteElectrochemicalClosure(schema),
    parameters,
)
coordinate = (jnp.arange(64) + 0.5) / 64.0
perturbation = 0.02 * jnp.sin(2.0 * jnp.pi * coordinate)
state = jnp.stack((1.0 + perturbation, 1.0 - perturbation), axis=-1)
for _ in range(20):
    evaluation = dynamics.evaluate(state)
    step = jnp.minimum(1.0e-4, 0.25 * evaluation.explicit_step_restriction)
    result = dynamics.step(state, step)
    state = result.concentrations

print("successful:", bool(result.successful))
print("free energy:", float(result.evaluation.total_free_energy))
