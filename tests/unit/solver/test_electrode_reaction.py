#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_butler_volmer_electrode_preserves_sites_and_current_charge_ledger():
    phases = (
        phx.equations.ChemicalPhaseSpec(
            "electrolyte", phx.equations.ChemicalPhaseKind.LIQUID, 3
        ),
        phx.equations.ChemicalPhaseSpec(
            "electrode", phx.equations.ChemicalPhaseKind.INERT, 3
        ),
        phx.equations.ChemicalPhaseSpec(
            "sites",
            phx.equations.ChemicalPhaseKind.SURFACE,
            2,
            site_density=10.0,
        ),
    )
    schema = phx.equations.ChemicalSpeciesSchema(
        ("Ion+", "electron", "vacancy", "adsorbed"),
        (
            phx.equations.ChemicalPhaseKind.LIQUID,
            phx.equations.ChemicalPhaseKind.INERT,
            phx.equations.ChemicalPhaseKind.SURFACE,
            phx.equations.ChemicalPhaseKind.SURFACE,
        ),
        jnp.asarray((0.02, 1.0e-6, 1.0, 1.02)),
        ("X", "Site"),
        jnp.asarray(((1, 0, 0, 1), (0, 0, 1, 1)), dtype=jnp.int32),
        jnp.asarray((1, -1, 0, 0), dtype=jnp.int32),
        phase_specs=phases,
    )
    thermodynamics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((10.0, 10.0, 10.0, 10.0)),
        jnp.zeros((4,)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=1000.0,
    )
    mechanism = phx.equations.ChemicalMechanismIR(
        "electrode",
        schema,
        thermodynamics,
        (
            phx.equations.ChemicalReactionSpec(
                "adsorption",
                {"Ion+": 1.0, "electron": 1.0, "vacancy": 1.0},
                {"adsorbed": 1.0},
                phx.equations.ButlerVolmerRatePlan(0.1, 0.5, 1),
                reverse_rate=phx.equations.ButlerVolmerRatePlan(
                    0.1, 0.5, 1, direction=-1
                ),
            ),
        ),
    ).prepare()
    plan = phx.solver.ReactiveElectrodePlan(
        mechanism,
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1.0,)),
        jnp.asarray((1,), dtype=jnp.int32),
        jnp.asarray(2.0),
    )
    surface_amount = jnp.asarray(((0.0, 0.0, 5.0, 0.0),))
    state = plan.initialize(surface_amount)
    concentration = jnp.asarray(((1.0, 1.0, 0.0, 0.0),))
    evaluation = plan.evaluate(
        concentration,
        state,
        jnp.asarray((0.0,)),
        jnp.asarray((0.05,)),
        jnp.asarray((300.0,)),
        jnp.asarray((101325.0,)),
    )
    result = plan.step(
        concentration,
        state,
        jnp.asarray((0.0,)),
        jnp.asarray((0.05,)),
        jnp.asarray((300.0,)),
        jnp.asarray((101325.0,)),
        jnp.asarray(1.0e-5),
    )

    assert evaluation.successful
    assert result.successful
    assert jnp.allclose(evaluation.charge_current_defect, 0.0)
    assert result.state.surface_amount[0, 2] < state.surface_amount[0, 2]
    assert result.state.surface_amount[0, 3] > state.surface_amount[0, 3]
