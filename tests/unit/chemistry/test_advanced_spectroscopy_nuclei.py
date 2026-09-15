import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_shared_voigt_profile_preserves_line_area():
    profile = phx.chemistry.SpectralProfilePlan(
        phx.chemistry.SpectralLineShape.VOIGT,
        -20.0,
        20.0,
        grid_size=20001,
        fwhm=0.4,
        lorentzian_fwhm=0.2,
        area_tolerance=5.0e-3,
    ).evaluate([0.0], [2.0])

    assert bool(profile.successful)
    np.testing.assert_allclose(profile.integrated_intensity, 2.0, rtol=5.0e-3)


def test_duschinsky_quadrature_and_herzberg_teller_emission_close_identical_modes():
    duschinsky = phx.chemistry.DuschinskyResult(
        [[1.0]],
        [0.0],
        0.0,
        0.0,
        True,
        "initial",
        "final",
    )
    result = phx.chemistry.DuschinskyFranckCondonPlan(
        [1.0],
        [1.0],
        duschinsky,
        1.0,
        quadrature_order=8,
    ).evaluate(
        [[0], [1], [2]],
        condon_dipole=[1.0, 0.0, 0.0],
        herzberg_teller_derivatives=[[0.0, 0.0, 0.0]],
        zero_zero_energy_hartree=2.0,
    )

    assert bool(result.successful)
    np.testing.assert_allclose(result.factors, [1.0, 0.0, 0.0], atol=2.0e-13)
    assert float(result.emission_rates[0]) > 0.0
    np.testing.assert_allclose(result.emission_rates[1:], 0.0, atol=1.0e-10)


def test_dynamic_resonance_raman_returns_finite_complex_response():
    result = phx.chemistry.ResonanceRamanPlan(
        [0.4, 0.7],
        [[1.0, 0.0, 0.0], [0.0, 0.8, 0.0]],
        [[[0.1, 0.0, 0.0]], [[0.0, 0.05, 0.0]]],
        [[0.02], [-0.01]],
        [0.05],
        0.35,
        [0.01, 0.02],
    ).evaluate()

    assert bool(result.successful)
    assert result.polarizability_derivatives.shape == (1, 3, 3)
    assert float(result.activities[0]) > 0.0
    np.testing.assert_allclose(result.relative_intensities, [1.0], atol=1.0e-14)


def test_quartic_force_field_gvpt2_vci_rotor_and_ensemble_are_self_consistent():
    energy = lambda coordinate: (
        0.5 * coordinate[0] ** 2
        + 0.06 * coordinate[0] ** 3 / 6.0
        + 0.12 * coordinate[0] ** 4 / 24.0
    )
    force_field = phx.chemistry.AnharmonicForceFieldPlan(energy, 1).evaluate()
    perturbation = phx.chemistry.VibrationalPerturbationPlan(
        phx.chemistry.VibrationalPerturbationKind.GVPT2,
        [1.0],
        force_field,
        maximum_quanta=5,
    ).evaluate([[0], [1]])
    configuration = phx.chemistry.VibrationalConfigurationPlan(
        [1.0],
        force_field,
        maximum_quanta=5,
        root_count=3,
    ).evaluate()
    rotor = phx.chemistry.HinderedRotorPlan(
        np.zeros(17),
        0.5,
        boltzmann_constant=1.0,
        temperature=1.0,
        root_count=5,
    ).evaluate()
    ensemble = phx.chemistry.ConformationalEnsemblePlan(
        [0.0, 1.0],
        temperature=1.0,
        boltzmann_constant=1.0,
    ).evaluate([[1.0, 0.0], [0.0, 1.0]])

    assert bool(force_field.successful)
    np.testing.assert_allclose(force_field.cubic, [[[0.06]]], atol=2.0e-14)
    np.testing.assert_allclose(force_field.quartic, [[[[0.12]]]], atol=2.0e-14)
    assert bool(perturbation.successful)
    assert bool(configuration.successful)
    np.testing.assert_allclose(
        configuration.vscf_energy,
        configuration.vci_energies[0],
        atol=2.0e-10,
    )
    assert bool(rotor.successful) and bool(ensemble.successful)
    np.testing.assert_allclose(rotor.energies[1:3], [0.5, 0.5], atol=2.0e-12)
    np.testing.assert_allclose(jnp.sum(ensemble.weights), 1.0, atol=1.0e-14)
