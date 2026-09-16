#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.applications.reacting_flow import ChemicalExplosiveModePlan
from phydrax.equations import (
    ArrheniusRatePlan,
    ChemicalMechanismIR,
    ChemicalPhaseKind,
    ChemicalReactionSpec,
    ChemicalSpeciesSchema,
    PolynomialSpeciesThermodynamicsPlan,
)


def _autocatalytic_mechanism():
    schema = ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        (ChemicalPhaseKind.GAS,) * 2,
        jnp.asarray((0.01, 0.01)),
        ("E",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    thermodynamics = PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((20.0, 20.0)),
        jnp.zeros((2,)),
        minimum_temperature=200.0,
        maximum_temperature=3000.0,
    )
    return ChemicalMechanismIR(
        "autocatalytic",
        schema,
        thermodynamics,
        (
            ChemicalReactionSpec(
                "A+B->2B",
                {"A": 1.0, "B": 1.0},
                {"B": 2.0},
                ArrheniusRatePlan(1.0),
            ),
        ),
    ).prepare()


def test_cema_projects_conservation_modes_and_tracks_explosive_mode_atomically():
    plan = ChemicalExplosiveModePlan(
        _autocatalytic_mechanism(), contribution_labels=("transport",)
    )
    accepted = plan.initial_tracking_state()
    evaluated = plan.evaluate(
        jnp.asarray((0.9, 0.1)),
        1000.0,
        101325.0,
        tracking=accepted,
        source_contributions=jnp.asarray(((0.02, -0.02),)),
    )

    assert bool(evaluated.evidence.successful)
    assert float(jnp.real(evaluated.leading_eigenvalue)) > 0.0
    np.testing.assert_allclose(evaluated.reaction_participation, (1.0,), atol=1e-10)
    np.testing.assert_allclose(jnp.sum(evaluated.species_explosion_index), 1.0)
    np.testing.assert_allclose(evaluated.evidence.conservation_residual, 0.0, atol=1e-10)

    rejected = plan.commit_tracking(accepted, evaluated, False)
    assert not bool(rejected.initialized)
    committed = plan.commit_tracking(accepted, evaluated, True)
    assert bool(committed.initialized)

    tracked = plan.evaluate(
        jnp.asarray((0.85, 0.15)), 1000.0, 101325.0, tracking=committed
    )
    assert bool(tracked.evidence.successful)
    assert not bool(tracked.evidence.tracking_ambiguous)
