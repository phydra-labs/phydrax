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
        jnp.asarray((1, 0, 2), dtype=jnp.int32),
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


def test_prepared_mechanism_preserves_elements_charge_and_zero_reactants():
    schema = _schema()
    mechanism = phx.equations.ChemicalMechanismIR(
        "association",
        schema,
        _thermodynamics(schema),
        (
            phx.equations.ChemicalReactionSpec(
                "2A+B->C",
                {"A": 2.0, "B": 1.0},
                {"C": 1.0},
                phx.equations.ArrheniusRatePlan(2.0),
            ),
        ),
    ).prepare()
    fields = mechanism.evaluate(
        jnp.asarray((2.0, 1.0, 0.0)),
        jnp.asarray(500.0),
        jnp.asarray(101325.0),
    )

    np.testing.assert_allclose(fields.forward_progress_rates, (8.0,))
    np.testing.assert_allclose(fields.species_amount_rate, (-16.0, -8.0, 8.0))
    np.testing.assert_allclose(fields.element_residual, 0.0)
    np.testing.assert_allclose(fields.charge_residual, 0.0)
    assert fields.successful

    blocked = mechanism.evaluate(
        jnp.asarray((0.0, 1.0, 0.0)),
        jnp.asarray(500.0),
        jnp.asarray(101325.0),
    )
    np.testing.assert_array_equal(blocked.forward_progress_rates, (0.0,))
    assert blocked.successful


def test_thermodynamically_reversible_equal_species_is_stationary():
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
        jnp.asarray((0.0, 0.0)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=2000.0,
    )
    mechanism = phx.equations.ChemicalMechanismIR(
        "reversible",
        schema,
        thermodynamics,
        (
            phx.equations.ChemicalReactionSpec(
                "A<->B",
                {"A": 1.0},
                {"B": 1.0},
                phx.equations.ArrheniusRatePlan(3.0),
                thermodynamic_reversible=True,
            ),
        ),
    ).prepare()
    fields = mechanism.evaluate(
        jnp.asarray((2.0, 2.0)),
        jnp.asarray(500.0),
        jnp.asarray(101325.0),
    )

    np.testing.assert_allclose(
        fields.forward_rate_constants, fields.reverse_rate_constants
    )
    np.testing.assert_allclose(fields.net_progress_rates, 0.0, atol=1e-12)
    np.testing.assert_allclose(fields.species_amount_rate, 0.0, atol=1e-12)


def test_mechanism_rejects_unbalanced_reaction():
    schema = _schema()
    mechanism = phx.equations.ChemicalMechanismIR(
        "invalid",
        schema,
        _thermodynamics(schema),
        (
            phx.equations.ChemicalReactionSpec(
                "A->B",
                {"A": 1.0},
                {"B": 1.0},
                phx.equations.ArrheniusRatePlan(1.0),
            ),
        ),
    )
    with pytest.raises(ValueError, match="element or charge"):
        mechanism.prepare()


def test_surface_coverage_and_sticking_rates_are_positive_and_finite():
    runtime = phx.equations.ChemicalRateRuntime()
    concentration = jnp.asarray((0.25, 1.0))
    coverage = phx.equations.SurfaceCoverageRatePlan(
        phx.equations.ArrheniusRatePlan(2.0),
        0,
        exponential_coefficient=0.1,
        power_exponent=1.0,
    )
    sticking = phx.equations.StickingRatePlan(0.2, 0.028)
    coverage_rate = coverage.evaluate(
        jnp.asarray(500.0),
        jnp.asarray(101325.0),
        concentration,
        runtime,
    )
    sticking_rate = sticking.evaluate(
        jnp.asarray(500.0),
        jnp.asarray(101325.0),
        concentration,
        runtime,
    )

    assert jnp.isfinite(coverage_rate) & (coverage_rate > 0.0)
    assert jnp.isfinite(sticking_rate) & (sticking_rate > 0.0)
