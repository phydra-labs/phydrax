#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _electrolyte():
    schema = phx.equations.ChemicalSpeciesSchema(
        ("cation", "anion"),
        (
            phx.equations.ChemicalPhaseKind.LIQUID,
            phx.equations.ChemicalPhaseKind.LIQUID,
        ),
        jnp.asarray((0.02, 0.03)),
        ("M", "X"),
        jnp.asarray(((1, 0), (0, 1)), dtype=jnp.int32),
        jnp.asarray((1, -1), dtype=jnp.int32),
    )
    parameters = phx.equations.ElectrolyteTransportParameters(
        schema,
        jnp.asarray((1e-3, 1e-3)),
        300.0,
        2.0,
    )
    return schema, parameters, phx.equations.IdealDiluteElectrochemicalClosure(schema)


def test_multiphase_electrolyte_derives_one_finite_stress_and_potential_set():
    _, electrolyte, electrochemical = _electrolyte()
    closure = phx.equations.MultiphaseElectrolyteClosure(
        phx.equations.BinaryPhaseThermodynamicClosure(), electrochemical
    )
    parameters = phx.equations.MultiphaseElectrolyteParameters(
        phx.equations.BinaryThermodynamicParameters(0.1, 0.05),
        electrolyte,
        jnp.asarray((0.2, -0.1)),
        1.0,
        2.0,
    )
    fields = closure.evaluate(
        jnp.zeros((4,)),
        jnp.zeros((4, 2)),
        jnp.zeros((4,)),
        jnp.ones((4, 2)),
        jnp.zeros((4,)),
        jnp.zeros((4, 2)),
        parameters,
    )

    assert fields.successful
    assert fields.total_stress.shape == (4, 2, 2)
    assert jnp.all(jnp.isfinite(fields.ionic_electrochemical_potential))


def test_electrolytic_nematic_composition_is_finite_and_dielectrically_positive():
    _, electrolyte, electrochemical = _electrolyte()
    basis = phx.equations.NematicTensorBasis(2)
    closure = phx.equations.ElectrolyticNematicClosure(
        phx.equations.LandauDeGennesClosure(basis), electrochemical
    )
    parameters = phx.equations.ElectrolyticNematicParameters(
        phx.equations.LandauDeGennesParameters(-1.0, 0.0, 1.0, 0.1),
        electrolyte,
        jnp.asarray((0.05, -0.02)),
        2.0,
        0.1,
    )
    fields = closure.evaluate(
        jnp.zeros((4, 2)),
        jnp.zeros((4, 2, 2)),
        jnp.zeros((4, 2)),
        jnp.ones((4, 2)),
        jnp.zeros((4,)),
        jnp.zeros((4, 2)),
        parameters,
    )

    assert fields.successful
    assert fields.permittivity_tensor.shape == (4, 2, 2)
    assert jnp.all(jnp.isfinite(fields.total_stress))
