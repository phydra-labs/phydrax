import jax.numpy as jnp
import numpy as np

from phydrax.applications.semiconductor._high_field import (
    HighFieldDrivingForce,
    LocalImpactIonization,
    LocalVelocitySaturation,
)
from phydrax.applications.semiconductor._quantities import ELEMENTARY_CHARGE_SI as Q


def _saturation(force=HighFieldDrivingForce.ELECTRIC_FIELD):
    return LocalVelocitySaturation(
        1e5,
        2.0,
        reference_temperature=300.0,
        temperature_exponent=-0.5,
        maximum_force=1e10,
        temperature_range=(200.0, 1000.0),
        driving_force=force,
        orientation="synthetic scalar longitudinal direction",
        provenance="synthetic constitutive invariant fixture; not material calibration",
    )


def test_velocity_saturation_recovers_low_field_and_bounded_asymptote():
    law = _saturation()
    fields = jnp.array([0.0, 1e-2, 1e9])
    result = law.evaluate(0.1, fields, 300.0)
    assert np.all(result.successful)
    np.testing.assert_allclose(result.mobility[:2], 0.1, rtol=1e-12)
    np.testing.assert_allclose(result.drift_speed[-1], 1e5, rtol=1e-6)
    assert np.all(result.drift_speed <= result.saturation_velocity)
    assert float(result.drift_speed[0]) == 0.0
    warm = law.evaluate(0.1, 1e9, 600.0)
    np.testing.assert_allclose(warm.saturation_velocity, 1e5 / np.sqrt(2.0))


def test_driving_choice_distinguishes_builtin_and_electrochemical_fields():
    field_law = _saturation()
    fermi_law = _saturation(HighFieldDrivingForce.QUASI_FERMI_GRADIENT)
    electric = field_law.driving_force.from_fields(1e8, 0.0)
    electrochemical = fermi_law.driving_force.from_fields(1e8, 0.0)
    assert float(field_law.evaluate(0.1, electric, 300.0).reduction) < 0.02
    assert float(fermi_law.evaluate(0.1, electrochemical, 300.0).reduction) == 1.0
    np.testing.assert_allclose(fermi_law.driving_force.from_fields(0.0, -Q * 37.0), -37.0)
    invalid = field_law.evaluate(0.1, [2e10, 1.0], [300.0, 1100.0])
    assert not np.any(invalid.successful)
    assert np.all(np.isnan(invalid.mobility))


def _ionization():
    return LocalImpactIonization(
        2e6,
        7e5,
        1e7,
        2e7,
        electron_exponent=1.0,
        hole_exponent=2.0,
        electron_temperature_exponent=0.2,
        hole_temperature_exponent=-0.3,
        electron_birth_energy=0.1 * Q,
        hole_birth_energy=0.2 * Q,
        reference_temperature=300.0,
        temperature_range=(200.0, 800.0),
        maximum_field=1e9,
        orientation="synthetic scalar longitudinal direction",
        provenance="synthetic Chynoweth law; not an avalanche calibration",
    )


def test_ionization_pays_pair_creation_energy_and_preserves_charge():
    law = _ionization()
    field, gn, gp, gap = 1e7, -3e24, 2e24, 1.1 * Q
    result = law.evaluate(field, gn, gp, 300.0, gap)
    expected_n = 2e6 * np.exp(-1.0) * abs(gn)
    expected_p = 7e5 * np.exp(-4.0) * abs(gp)
    np.testing.assert_allclose(result.electron_source, expected_n + expected_p)
    np.testing.assert_allclose(result.hole_source, result.electron_source)
    np.testing.assert_allclose(result.electron_initiated_rate, expected_n)
    np.testing.assert_allclose(result.hole_initiated_rate, expected_p)
    np.testing.assert_allclose(
        result.electron_charge_source + result.hole_charge_source, 0.0, atol=0.0
    )
    total = (
        result.electron_energy_source
        + result.hole_energy_source
        + result.band_energy_source
        + result.lattice_energy_source
    )
    np.testing.assert_allclose(total / (gap * result.electron_source), 0.0, atol=1e-14)
    # A vanishing field gives exactly zero generation even with diffusion flux.
    zero = law.evaluate(0.0, gn, gp, 300.0, gap)
    assert float(zero.electron_source) == 0.0
    assert float(zero.electron_energy_source) == 0.0
