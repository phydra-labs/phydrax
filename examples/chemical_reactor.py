"""Stiff isothermal batch conversion through a prepared chemical mechanism."""

import jax.numpy as jnp

import phydrax as phx


schema = phx.equations.ChemicalSpeciesSchema(
    ("A", "B"),
    (
        phx.equations.ChemicalPhaseKind.GAS,
        phx.equations.ChemicalPhaseKind.GAS,
    ),
    jnp.asarray((0.028, 0.028)),
    ("X",),
    jnp.asarray(((1, 1),), dtype=jnp.int32),
    jnp.asarray((0, 0), dtype=jnp.int32),
)
thermodynamics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
    schema,
    jnp.asarray((20.0, 20.0)),
    jnp.asarray((0.0, 0.0)),
    reference_temperature=300.0,
    minimum_temperature=200.0,
    maximum_temperature=2000.0,
)
mechanism = phx.equations.ChemicalMechanismIR(
    "conversion",
    schema,
    thermodynamics,
    (
        phx.equations.ChemicalReactionSpec(
            "A->B",
            {"A": 1.0},
            {"B": 1.0},
            phx.equations.ArrheniusRatePlan(10.0),
        ),
    ),
).prepare()
reactor = phx.solver.ChemicalReactorPlan(
    mechanism,
    phx.solver.ChemicalReactorKind.ISOTHERMAL_CONSTANT_VOLUME,
    fixed_temperature=700.0,
    fixed_volume=1.0,
)
grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 1.0, 41), time_id="reactor")
solution = reactor.solve(jnp.asarray((1.0, 0.0)), grid)

print("successful:", bool(solution.successful))
print("final species amount:", solution.states[-1])
