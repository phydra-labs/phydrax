#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _mechanism(reference_energy=(0.0, 0.0)):
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        (
            phx.equations.ChemicalPhaseKind.GAS,
            phx.equations.ChemicalPhaseKind.GAS,
        ),
        jnp.asarray((1.0, 1.0)),
        ("X",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.asarray((0, 0), dtype=jnp.int32),
        gas_standard_pressure=101325.0,
    )
    thermodynamics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((10.0, 10.0)),
        jnp.asarray(reference_energy),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=1000.0,
    )
    return phx.equations.ChemicalMechanismIR(
        "reactor",
        schema,
        thermodynamics,
        (
            phx.equations.ChemicalReactionSpec(
                "A->B",
                {"A": 1.0},
                {"B": 1.0},
                phx.equations.ArrheniusRatePlan(1.0),
            ),
        ),
    ).prepare()


def test_isothermal_constant_volume_reactor_matches_analytic_decay():
    plan = phx.solver.ChemicalReactorPlan(
        _mechanism(),
        phx.solver.ChemicalReactorKind.ISOTHERMAL_CONSTANT_VOLUME,
        fixed_temperature=500.0,
        fixed_volume=1.0,
    )
    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 1.0, 17), time_id="chemistry")
    solution = plan.solve(
        jnp.asarray((1.0, 0.0)),
        grid,
        adaptive=phx.solver.RosenbrockAdaptivePolicy(
            relative_tolerance=1e-7,
            absolute_tolerance=1e-9,
            initial_step=0.02,
            maximum_accepted_steps=512,
            maximum_attempts=1024,
        ),
    )

    assert solution.successful
    np.testing.assert_allclose(solution.states[-1, 0], np.exp(-1.0), rtol=2e-4)
    np.testing.assert_allclose(jnp.sum(solution.states, axis=-1), 1.0, atol=2e-8)
    bdf = plan.solve_bdf(
        jnp.asarray((1.0, 0.0)),
        grid,
        maximum_order=2,
    )
    assert bdf.successful
    np.testing.assert_allclose(bdf.states[-1, 0], np.exp(-1.0), rtol=5e-3)


def test_adiabatic_constant_volume_reactor_conserves_extensive_energy():
    plan = phx.solver.ChemicalReactorPlan(
        _mechanism((0.0, -100.0)),
        phx.solver.ChemicalReactorKind.ADIABATIC_CONSTANT_VOLUME,
        fixed_volume=1.0,
    )
    initial = plan.initial_state(jnp.asarray((1.0, 0.0)), jnp.asarray(400.0))
    rate = plan.rate(jnp.asarray(0.0), initial)
    state = plan.evaluate(initial)

    np.testing.assert_allclose(rate[-1], 0.0)
    np.testing.assert_allclose(state.temperature, 400.0, rtol=1e-10)
    assert state.successful
