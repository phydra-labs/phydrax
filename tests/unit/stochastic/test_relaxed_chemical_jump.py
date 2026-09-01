#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_relaxed_chemical_jump_reports_continuous_bias_evidence():
    schema = phx.equations.ChemicalSpeciesSchema(
        ("A", "B"),
        (
            phx.equations.ChemicalPhaseKind.GAS,
            phx.equations.ChemicalPhaseKind.GAS,
        ),
        jnp.asarray((1.0, 1.0)),
        ("X",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.asarray((0, 0), dtype=jnp.int32),
    )
    thermodynamics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((10.0, 10.0)),
        jnp.asarray((0.0, 0.0)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=1000.0,
    )
    mechanism = phx.equations.ChemicalMechanismIR(
        "relaxed",
        schema,
        thermodynamics,
        (
            phx.equations.ChemicalReactionSpec(
                "A->B",
                {"A": 1.0},
                {"B": 1.0},
                phx.equations.ArrheniusRatePlan(0.5),
            ),
        ),
    ).prepare()
    exact = phx.stochastic.ChemicalJumpProcess(mechanism, 1.0)
    plan = phx.stochastic.RelaxedChemicalJumpPlan(
        exact,
        phx.stochastic.RelaxedChemicalJumpParameters(20.0, 20.0, 64, 1.0),
    )
    result = plan.simulate(
        jnp.asarray((20.0, 0.0)),
        phx.stochastic.ChemicalJumpRuntime(500.0, 101325.0),
        jr.key(4),
    )

    assert result.evidence.successful
    assert result.evidence.event_count > 0
    assert jnp.all(jnp.isfinite(result.final_state))
    assert jnp.isclose(jnp.sum(result.final_state), 20.0)
