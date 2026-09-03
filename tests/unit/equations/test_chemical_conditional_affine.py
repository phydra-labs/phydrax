#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _schema():
    return phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A", "B", "C"),
        (
            phx.equations.ChemicalPhaseKind.GAS,
            phx.equations.ChemicalPhaseKind.GAS,
            phx.equations.ChemicalPhaseKind.GAS,
        ),
        jnp.asarray((1.0, 1.0, 3.0)),
        ("X", "Y"),
        jnp.asarray(((1, 0, 2), (0, 1, 1)), dtype=jnp.int32),
        jnp.asarray((0, 0, 0), dtype=jnp.int32),
        gas_standard_pressure=101325.0,
    )


def _thermodynamics(schema):
    return phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((10.0, 10.0, 10.0)),
        jnp.asarray((0.0, 0.0, 0.0)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=2000.0,
    )


def _association_mechanism(*, forward_rate=None):
    schema = _schema()
    return phx.equations.ChemicalMechanismIR(
        "association",
        schema,
        _thermodynamics(schema),
        (
            phx.equations.ChemicalReactionSpec(
                "2A+B<->C",
                {"A": 2.0, "B": 1.0},
                {"C": 1.0},
                phx.equations.ArrheniusRatePlan(2.0)
                if forward_rate is None
                else forward_rate,
                reverse_rate=phx.equations.ArrheniusRatePlan(0.5),
            ),
        ),
    ).prepare()


def _drivers(a=2.0):
    return phx.equations.ChemicalConditionalAffineDrivers(
        jnp.asarray((a,), dtype=jnp.float64),
        jnp.asarray(500.0),
        jnp.asarray(101325.0),
    )


def test_compiler_certifies_directional_pivots_and_assembles_operator():
    mechanism = _association_mechanism()
    plan = phx.equations.ChemicalConditionalAffinePlan(("B", "C"), ("A",))
    certificate = plan.analyze(mechanism)
    prepared = plan.prepare(mechanism)

    assert certificate.certified
    assert certificate.pivot_species == ("B", "C")
    assembly = prepared.assemble(_drivers())

    np.testing.assert_allclose(assembly.directional_coefficients, (8.0, 0.5))
    np.testing.assert_allclose(
        assembly.operator,
        ((-8.0, 0.5), (8.0, -0.5)),
    )
    np.testing.assert_allclose(assembly.forcing, 0.0)
    assert assembly.successful


def test_reaction_multiplier_is_shared_by_forward_and_reverse_channels():
    prepared = phx.equations.ChemicalConditionalAffinePlan(("B", "C"), ("A",)).prepare(
        _association_mechanism()
    )

    assembly = prepared.assemble(_drivers(), reaction_multiplier=jnp.asarray((3.0,)))

    np.testing.assert_allclose(assembly.directional_coefficients, (24.0, 1.5))


def test_extent_reconstruction_preserves_complete_stoichiometric_state():
    prepared = phx.equations.ChemicalConditionalAffinePlan(("B", "C"), ("A",)).prepare(
        _association_mechanism()
    )
    state = jnp.asarray((2.0, 1.0, 0.0), dtype=jnp.float64)

    result = prepared.advance(state, _drivers(), jnp.asarray(0.01))
    increment = result.candidate_state - state

    assert result.successful
    np.testing.assert_allclose(result.element_residual, 0.0, atol=1e-13)
    np.testing.assert_allclose(result.charge_residual, 0.0, atol=1e-13)
    np.testing.assert_allclose(result.affine_consistency_residual, 0.0, atol=1e-13)
    np.testing.assert_allclose(increment[0], 2.0 * increment[1], atol=1e-13)
    np.testing.assert_allclose(increment[2], -increment[1], atol=1e-13)


def test_compiler_requires_every_nonpivot_and_rate_dependency_as_driver():
    mechanism = _association_mechanism()
    invalid = phx.equations.ChemicalConditionalAffinePlan(("C",), ("A",))
    certificate = invalid.analyze(mechanism)

    assert not certificate.certified
    assert "mass-action factors" in certificate.rejection_reasons[0]
    with pytest.raises(ValueError, match="certification failed"):
        invalid.prepare(mechanism)

    third_body = phx.equations.ThirdBodyRatePlan(
        phx.equations.ArrheniusRatePlan(2.0),
        jnp.asarray((1.0, 0.0, 1.0)),
    )
    dependency_mechanism = _association_mechanism(forward_rate=third_body)
    dependency_plan = phx.equations.ChemicalConditionalAffinePlan(("B", "C"), ("A",))
    dependency_certificate = dependency_plan.analyze(dependency_mechanism)

    assert not dependency_certificate.certified
    assert (
        "rate-plan concentration dependencies"
        in (dependency_certificate.rejection_reasons[0])
    )


def test_driver_only_channel_is_forcing_and_negative_state_fails_closed():
    prepared = phx.equations.ChemicalConditionalAffinePlan(("C",), ("A", "B")).prepare(
        _association_mechanism()
    )
    drivers = phx.equations.ChemicalConditionalAffineDrivers(
        jnp.asarray((2.0, 1.0), dtype=jnp.float64),
        jnp.asarray(500.0),
        jnp.asarray(101325.0),
    )
    assembly = prepared.assemble(drivers)

    np.testing.assert_allclose(assembly.forcing, (8.0,))
    failed = prepared.advance(
        jnp.asarray((2.0, 1.0, 0.0), dtype=jnp.float64),
        drivers,
        jnp.asarray(1.0),
    )

    assert not failed.successful
    assert failed.status == int(
        phx.equations.ChemicalConditionalAffineStatus.NEGATIVE_STATE
    )
    assert failed.minimum_species < 0.0
