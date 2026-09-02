#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


R = phx.equations.UNIVERSAL_GAS_CONSTANT


def _ideal_model(reference_energy=(0.0, -5000.0)):
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        (phx.equations.ChemicalPhaseKind.GAS,) * 2,
        jnp.asarray((0.02, 0.02)),
        ("X",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    species = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray(((2.5 * R,), (2.5 * R,))),
        jnp.asarray(reference_energy),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=2000.0,
    )
    ideal = phx.equations.IdealGasReferenceHelmholtzTerm(schema, species)
    return phx.equations.HomogeneousHelmholtzPlan(
        ideal, phx.equations.ZeroResidualHelmholtzTerm(schema)
    )


def _pr_model():
    ideal_model = _ideal_model()
    parameters = phx.equations.PengRobinsonParameters(
        ideal_model.schema.catalog,
        jnp.asarray((190.564, 305.322)),
        jnp.asarray((4.5992e6, 4.8722e6)),
        jnp.asarray((0.01142, 0.0995)),
        jnp.zeros((2, 2)),
        provenance="public methane/ethane constants; zero interaction assumption",
    )
    residual = phx.equations.PengRobinsonResidualHelmholtzTerm(
        ideal_model.schema, parameters
    )
    return phx.equations.HomogeneousHelmholtzPlan(
        ideal_model.ideal,
        residual,
        maximum_molar_density=5.0e4,
    )


def test_ideal_gas_gibbs_equilibrium_conserves_elements_and_decreases_gibbs():
    plan = phx.solver.IdealGasGibbsEquilibriumPlan(_ideal_model())
    initial = jnp.asarray((0.9, 0.1))

    result = plan.solve(jnp.asarray(700.0), jnp.asarray(1.0e5), initial)

    assert bool(result.evidence.successful)
    np.testing.assert_allclose(jnp.sum(result.species_amount), 1.0, atol=1.0e-8)
    assert result.species_amount[1] > initial[1]
    assert result.evidence.gibbs_change <= 1.0e-8


def test_tpd_and_flash_return_fixed_shape_evidence():
    model = _pr_model()
    feed = jnp.asarray((0.5, 0.5))
    stability = phx.solver.TPDSearchPlan(model, maximum_steps=40, tolerance=1.0e-6).solve(
        jnp.asarray(180.0), jnp.asarray(1.0e6), feed
    )
    flash = phx.solver.FixedTwoPhaseTPFlashPlan(
        model, maximum_steps=40, tolerance=1.0e-6
    ).solve(jnp.asarray(180.0), jnp.asarray(1.0e6), feed)

    assert stability.trial_composition.shape == (2,)
    assert flash.phase_fraction.shape == (2,)
    assert flash.phase_composition.shape == (2, 2)
    assert flash.phase_molar_density.shape == (2,)
    assert bool(stability.unstable)
    assert stability.minimum_tpd < 0.0
    assert bool(flash.successful)
    assert int(flash.status) == int(phx.solver.PhaseEquilibriumStatus.SUCCESS_TWO_PHASE)
    assert bool(jnp.all(flash.phase_fraction > 0.0))
    np.testing.assert_allclose(flash.material_residual, 0.0, atol=1.0e-8)
    np.testing.assert_allclose(flash.fugacity_residual, 0.0, atol=1.0e-7)
