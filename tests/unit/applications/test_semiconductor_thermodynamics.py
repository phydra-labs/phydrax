import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.semiconductor._quantities import (
    BOLTZMANN_CONSTANT_SI as KB,
    ELEMENTARY_CHARGE_SI as Q,
)
from phydrax.applications.semiconductor._thermodynamics import (
    BandThermodynamics,
    IncompleteIonization,
)
from phydrax.units import CENTIMETER, derived_unit, ELECTRONVOLT


def _bands(statistics="fermi-dirac", **changes):
    parameters = dict(
        conduction_band_edge=-4.05 * Q,
        valence_band_edge=-5.17 * Q,
        conduction_density_of_states=2.8e25,
        valence_density_of_states=1.04e25,
        reference_temperature=300.0,
        temperature_range=(50.0, 600.0),
        energy_reference="synthetic aligned vacuum reference",
        provenance="Synthetic independently specified DOS and band edges; no process calibration.",
        statistics=statistics,
    )
    parameters.update(changes)
    return BandThermodynamics("synthetic-parabolic-bands", **parameters)


def _ionization():
    return IncompleteIonization(
        donor_binding_energy=0.045 * Q,
        acceptor_binding_energy=0.057 * Q,
        donor_degeneracy=2.0,
        acceptor_degeneracy=4.0,
        provenance="Synthetic isolated-level equilibrium assumptions, not calibrated defects.",
    )


@pytest.mark.parametrize("statistics", ["boltzmann", "fermi-dirac"])
def test_populations_and_inverse_are_invariant_under_electronic_gauge_shift(statistics):
    bands = _bands(statistics)
    temperature = jnp.asarray([150.0, 300.0, 500.0])
    potential = jnp.asarray([-0.13, 0.0, 0.27])
    ec, ev = bands.band_edges(potential, temperature)
    efn = ec + KB * temperature * jnp.asarray([-12.0, -1.0, 9.0])
    efp = ev - KB * temperature * jnp.asarray([-8.0, 0.2, 4.0])
    n, p = (
        bands.electron_density(potential, efn, temperature),
        bands.hole_density(potential, efp, temperature),
    )
    voltage_shift = 0.43
    np.testing.assert_allclose(
        bands.electron_density(
            potential + voltage_shift, efn - Q * voltage_shift, temperature
        ),
        n,
        rtol=3e-12,
    )
    np.testing.assert_allclose(
        bands.hole_density(
            potential + voltage_shift, efp - Q * voltage_shift, temperature
        ),
        p,
        rtol=3e-12,
    )
    np.testing.assert_allclose(
        bands.electron_fermi_energy(potential + voltage_shift, n, temperature),
        efn - Q * voltage_shift,
        rtol=0,
        atol=1e-31,
    )
    np.testing.assert_allclose(
        bands.hole_fermi_energy(potential + voltage_shift, p, temperature),
        efp - Q * voltage_shift,
        rtol=0,
        atol=1e-31,
    )
    offset = 0.8 * Q
    shifted = eqx.tree_at(
        lambda model: (model.conduction_band_edge, model.valence_band_edge),
        bands,
        (bands.conduction_band_edge + offset, bands.valence_band_edge + offset),
    )
    np.testing.assert_allclose(
        shifted.electron_density(potential, efn + offset, temperature), n, rtol=3e-12
    )
    np.testing.assert_allclose(
        shifted.hole_density(potential, efp + offset, temperature), p, rtol=3e-12
    )


def test_boltzmann_reduction_has_mass_action_einstein_and_classical_energy():
    bands = _bands("boltzmann")
    temperature = jnp.asarray([150.0, 300.0, 500.0])
    potential = 0.12
    ec, ev = bands.band_edges(potential, temperature)
    ef = 0.5 * (ec + ev) + KB * temperature
    nc, nv = bands.density_of_states(temperature)
    n = bands.electron_density(potential, ef, temperature)
    p = bands.hole_density(potential, ef, temperature)
    np.testing.assert_allclose(
        n * p / (nc * nv), jnp.exp((ev - ec) / (KB * temperature)), rtol=3e-13
    )
    np.testing.assert_allclose(
        bands.electron_einstein_ratio(n, temperature), KB * temperature / Q, rtol=2e-14
    )
    np.testing.assert_allclose(
        bands.hole_einstein_ratio(p, temperature), KB * temperature / Q, rtol=2e-14
    )
    np.testing.assert_allclose(
        bands.electron_energy_density(n, temperature),
        1.5 * n * KB * temperature,
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        bands.hole_energy_density(p, temperature), 1.5 * p * KB * temperature, rtol=2e-14
    )
    fd = _bands()
    eta = jnp.asarray([-32.0, -24.0, -18.0])
    np.testing.assert_allclose(fd.statistics_value(eta), jnp.exp(eta), rtol=6e-9)
    np.testing.assert_allclose(fd.statistics_derivative(eta), jnp.exp(eta), rtol=1.2e-8)


def test_fd_reference_degenerate_limit_and_derivative_are_one_statistics():
    bands = _bands()
    # F_j(0)=(1-2**(-j))*zeta(j+1), in the DLMF normalized convention.
    np.testing.assert_allclose(
        bands.statistics_value(0.0), 0.7651470246254079, rtol=3e-13
    )
    np.testing.assert_allclose(
        bands.statistics_derivative(0.0), 0.6048986434216304, rtol=3e-13
    )
    eta = jnp.asarray([-60.0, -8.0, 0.0, 6.0, 35.0, 79.0])
    density = bands.statistics_value(eta)
    derivative = bands.statistics_derivative(eta)
    assert np.all(np.diff(np.asarray(density)) > 0)
    assert np.all(np.asarray(derivative) > 0)
    np.testing.assert_allclose(
        jax.vmap(jax.grad(bands.statistics_value))(eta), derivative, rtol=3e-13
    )
    degenerate_eta = 80.0
    sommerfeld = (
        4
        * degenerate_eta**1.5
        / (3 * math.sqrt(math.pi))
        * (
            1
            + math.pi**2 / (8 * degenerate_eta**2)
            + 7 * math.pi**4 / (640 * degenerate_eta**4)
        )
    )
    np.testing.assert_allclose(
        bands.statistics_value(degenerate_eta), sommerfeld, rtol=2e-10
    )


def test_fd_inverse_has_implicit_forward_reverse_and_density_derivatives():
    bands = _bands()
    eta = jnp.asarray([-50.0, -4.0, 0.0, 7.0, 60.0])
    population = bands.statistics_value(eta)
    recovered = eqx.filter_jit(bands.inverse_statistics)(population)
    np.testing.assert_allclose(recovered, eta, rtol=0, atol=3e-10)
    direction = population * 0.13
    _, tangent = jax.jvp(bands.inverse_statistics, (population,), (direction,))
    np.testing.assert_allclose(
        tangent, direction / bands.statistics_derivative(eta), rtol=3e-10
    )
    adjoint = jax.grad(lambda values: jnp.sum(bands.inverse_statistics(values)))(
        population
    )
    np.testing.assert_allclose(
        adjoint * bands.statistics_derivative(eta), 1.0, rtol=3e-10
    )
    temperature, potential = 320.0, 0.1
    ec, ev = bands.band_edges(potential, temperature)
    efn, efp = ec + 5 * KB * temperature, ev - 3 * KB * temperature
    electron_derivative = jax.grad(
        lambda energy: bands.electron_density(potential, energy, temperature)
    )(efn)
    hole_derivative = jax.grad(
        lambda energy: bands.hole_density(potential, energy, temperature)
    )(efp)
    np.testing.assert_allclose(
        electron_derivative,
        bands.electron_compressibility(potential, efn, temperature),
        rtol=3e-13,
    )
    np.testing.assert_allclose(
        -hole_derivative,
        bands.hole_compressibility(potential, efp, temperature),
        rtol=3e-13,
    )
    n, p = (
        bands.electron_density(potential, efn, temperature),
        bands.hole_density(potential, efp, temperature),
    )
    np.testing.assert_allclose(
        bands.electron_einstein_ratio(n, temperature),
        n / (Q * electron_derivative),
        rtol=3e-10,
    )
    np.testing.assert_allclose(
        bands.hole_einstein_ratio(p, temperature), p / (-Q * hole_derivative), rtol=3e-10
    )


def test_log_scaled_fd_preserves_representable_density_below_normalized_underflow():
    bands = _bands()
    temperature = 300.0
    ec, _ = bands.band_edges(0.0, temperature)
    # exp(-750) underflows in float64, but Nc*exp(-750) is representable.
    efn = ec - 750 * KB * temperature
    n = bands.electron_density(0.0, efn, temperature)
    expected = jnp.exp(jnp.log(bands.conduction_density_of_states) - 750)
    assert float(n) > 0
    np.testing.assert_allclose(n, expected, rtol=3e-12, atol=0)
    recovered = bands.electron_fermi_energy(0.0, n, temperature)
    np.testing.assert_allclose(
        (recovered - ec) / (KB * temperature), -750.0, rtol=0, atol=4e-10
    )


@pytest.mark.parametrize("carrier", ["electron", "hole"])
def test_fd_kinetic_energy_obeys_thermodynamic_legendre_identity(carrier):
    bands = _bands()
    temperature = 300.0
    if carrier == "electron":
        energy_density = bands.electron_energy_density
        fermi_energy = bands.electron_fermi_energy
        edge_index, sign, dos = 0, 1, bands.conduction_density_of_states
    else:
        energy_density = bands.hole_energy_density
        fermi_energy = bands.hole_fermi_energy
        edge_index, sign, dos = 1, -1, bands.valence_density_of_states
    density = 4.2 * dos

    def free_energy(n, T):
        edge = bands.band_edges(0.0, T)[edge_index]
        kinetic_mu = sign * (fermi_energy(0.0, n, T) - edge)
        return n * kinetic_mu - (2 / 3) * energy_density(n, T)

    kinetic_mu = sign * (
        fermi_energy(0.0, density, temperature)
        - bands.band_edges(0.0, temperature)[edge_index]
    )
    np.testing.assert_allclose(
        jax.grad(free_energy, argnums=0)(density, temperature), kinetic_mu, rtol=3e-10
    )
    energy = energy_density(density, temperature)
    reconstructed = free_energy(density, temperature) - temperature * jax.grad(
        free_energy, argnums=1
    )(density, temperature)
    np.testing.assert_allclose(reconstructed, energy, rtol=3e-10)
    dilute_density = dos * jnp.exp(-25.0)
    np.testing.assert_allclose(
        energy_density(dilute_density, temperature),
        1.5 * dilute_density * KB * temperature,
        rtol=1e-11,
    )
    assert float(energy_density(0.0, temperature)) == 0.0
    np.testing.assert_allclose(
        jax.grad(lambda n: energy_density(n, temperature))(0.0),
        1.5 * KB * temperature,
        rtol=2e-14,
    )


@pytest.mark.parametrize("statistics", ["boltzmann", "fermi-dirac"])
@pytest.mark.parametrize("carrier", ["electron", "hole"])
def test_material_energy_includes_band_entropy_but_not_electrostatic_storage(
    statistics, carrier
):
    bands = _bands(
        statistics,
        conduction_temperature_coefficient=2e-4 * Q,
        gap_varshni_alpha=4.73e-4 * Q,
        gap_varshni_beta=636.0,
    )
    temperature, potential = 420.0, 0.23
    if carrier == "electron":
        free_energy = bands.electron_material_free_energy_density
        internal_energy = bands.electron_material_internal_energy_density
        kinetic_energy = bands.electron_energy_density
        fermi_energy = bands.electron_fermi_energy
        edge_index, sign, dos = 0, 1, bands.conduction_density_of_states
    else:
        free_energy = bands.hole_material_free_energy_density
        internal_energy = bands.hole_material_internal_energy_density
        kinetic_energy = bands.hole_energy_density
        fermi_energy = bands.hole_fermi_energy
        edge_index, sign, dos = 1, -1, bands.valence_density_of_states
    density = 0.7 * dos
    material_edge = bands.band_edges(0.0, temperature)[edge_index]
    edge_derivative = bands.material_band_temperature_derivatives(temperature)[edge_index]
    automatic_edge_derivative = jax.grad(lambda T: bands.band_edges(0.0, T)[edge_index])(
        temperature
    )
    np.testing.assert_allclose(edge_derivative, automatic_edge_derivative, rtol=3e-14)
    chemical_energy = sign * (
        fermi_energy(potential, density, temperature) + Q * potential
    )
    np.testing.assert_allclose(
        jax.grad(free_energy, argnums=0)(density, temperature),
        chemical_energy,
        rtol=3e-11,
    )
    helmholtz = free_energy(density, temperature)
    entropy = -jax.grad(free_energy, argnums=1)(density, temperature)
    material_internal = internal_energy(density, temperature)
    np.testing.assert_allclose(
        material_internal, helmholtz + temperature * entropy, rtol=3e-11
    )
    naive_band_plus_kinetic = (
        kinetic_energy(density, temperature) + sign * density * material_edge
    )
    np.testing.assert_allclose(
        material_internal - naive_band_plus_kinetic,
        -sign * density * temperature * edge_derivative,
        rtol=3e-11,
    )
    assert float(internal_energy(0.0, temperature)) == 0.0


def test_pair_material_energy_is_reference_independent_with_varshni_heat_capacity():
    bands = _bands(
        "boltzmann",
        conduction_temperature_coefficient=3e-4 * Q,
        gap_varshni_alpha=4.73e-4 * Q,
        gap_varshni_beta=636.0,
    )
    density, temperature = 2e22, 400.0

    def pair_energy(model, T):
        return model.electron_material_internal_energy_density(
            density, T
        ) + model.hole_material_internal_energy_density(density, T)

    ec, ev = bands.band_edges(0.0, temperature)
    dec, dev = bands.material_band_temperature_derivatives(temperature)
    expected = density * (3 * KB * temperature + ec - ev - temperature * (dec - dev))
    np.testing.assert_allclose(pair_energy(bands, temperature), expected, rtol=3e-14)
    offset = 0.9 * Q
    shifted = eqx.tree_at(
        lambda model: (model.conduction_band_edge, model.valence_band_edge),
        bands,
        (bands.conduction_band_edge + offset, bands.valence_band_edge + offset),
    )
    np.testing.assert_allclose(pair_energy(shifted, temperature), expected, rtol=3e-14)
    gap_second_derivative = (
        -2
        * bands.gap_varshni_alpha
        * bands.gap_varshni_beta**2
        / (temperature + bands.gap_varshni_beta) ** 3
    )
    expected_heat_capacity = density * (3 * KB - temperature * gap_second_derivative)
    np.testing.assert_allclose(
        jax.grad(lambda T: pair_energy(bands, T))(temperature),
        expected_heat_capacity,
        rtol=3e-13,
    )


def test_temperature_law_and_native_units_preserve_independent_dos():
    per_cm3 = derived_unit("1/cm3", ((CENTIMETER, -3),))
    bands = _bands(
        "boltzmann",
        conduction_band_edge=-4.05,
        valence_band_edge=-5.17,
        conduction_density_of_states=2.8e19,
        valence_density_of_states=1.04e19,
        energy_unit=ELECTRONVOLT,
        density_unit=per_cm3,
        conduction_temperature_coefficient=2e-5 * Q,
        gap_varshni_alpha=4.73e-4 * Q,
        gap_varshni_beta=636.0,
    )
    temperature = jnp.asarray([80.0, 300.0, 550.0])
    ec, ev = bands.band_edges(0.0, temperature)
    expected_gap = Q * (
        1.12 - 4.73e-4 * (temperature**2 / (temperature + 636) - 300**2 / 936)
    )
    np.testing.assert_allclose(ec - ev, expected_gap, rtol=3e-14)
    np.testing.assert_allclose(ec, -4.05 * Q + 2e-5 * Q * (temperature - 300), rtol=3e-14)
    nc, nv = bands.density_of_states(temperature)
    np.testing.assert_allclose(nc, 2.8e25 * (temperature / 300) ** 1.5, rtol=3e-14)
    np.testing.assert_allclose(nv, 1.04e25 * (temperature / 300) ** 1.5, rtol=3e-14)
    sensitivity = jax.grad(lambda T: bands.density_of_states(T)[0])(300.0)
    np.testing.assert_allclose(sensitivity, 1.5 * 2.8e25 / 300, rtol=3e-14)
    bands.admit_temperature(temperature)
    with pytest.raises(ValueError):
        bands.admit_temperature(601.0)
    with pytest.raises(ValueError):
        _bands(gap_varshni_alpha=0.1 * Q, gap_varshni_beta=10.0)


def test_explicit_impurity_degeneracies_occupancies_and_energy_share_levels():
    bands, ionization = _bands(), _ionization()
    ionization.admit(bands)
    temperature, potential = 260.0, 0.17
    nd, na = 3e21, 8e20
    ec, ev = bands.band_edges(potential, temperature)
    ed, ea = ec - ionization.donor_binding_energy, ev + ionization.acceptor_binding_energy
    donor_at_level, _ = ionization.ionized_densities(
        bands, potential, ed, temperature, nd, na
    )
    _, acceptor_at_level = ionization.ionized_densities(
        bands, potential, ea, temperature, nd, na
    )
    np.testing.assert_allclose(donor_at_level, nd / 3, rtol=3e-14)
    np.testing.assert_allclose(acceptor_at_level, na / 5, rtol=3e-14)
    ef = jnp.asarray([ea - 20 * KB * temperature, ea, ed, ed + 20 * KB * temperature])
    nd_plus, na_minus = ionization.ionized_densities(
        bands, potential, ef, temperature, nd, na
    )
    assert np.all((np.asarray(nd_plus) >= 0) & (np.asarray(nd_plus) <= nd))
    assert np.all((np.asarray(na_minus) >= 0) & (np.asarray(na_minus) <= na))
    assert np.all(np.diff(np.asarray(nd_plus)) <= 0)
    assert np.all(np.diff(np.asarray(na_minus)) >= 0)
    energy = ionization.bound_energy_density(bands, potential, ef, temperature, nd, na)
    np.testing.assert_allclose(
        energy, ed * (nd - nd_plus) + ea * na_minus, rtol=2e-13, atol=1e-12
    )
    shifted_energy = ionization.bound_energy_density(
        bands, potential + 0.3, ef - Q * 0.3, temperature, nd, na
    )
    np.testing.assert_allclose(
        shifted_energy - energy,
        -Q * 0.3 * (nd - nd_plus + na_minus),
        rtol=3e-12,
        atol=1e-12,
    )


def test_neutral_equilibrium_includes_freeze_out_compensation_and_gauge():
    bands, ionization = _bands(), _ionization()
    temperature = jnp.asarray([50.0, 300.0, 300.0])
    donors = jnp.asarray([1e21, 1e21, 2e21])
    acceptors = jnp.asarray([0.0, 0.0, 1.8e21])
    potential = jnp.asarray([0.0, 0.1, -0.12])
    ef = bands.equilibrium_fermi_energy(
        potential, temperature, donors, acceptors, ionization=ionization
    )
    n, p = (
        bands.electron_density(potential, ef, temperature),
        bands.hole_density(potential, ef, temperature),
    )
    nd, na = ionization.ionized_densities(
        bands, potential, ef, temperature, donors, acceptors
    )
    np.testing.assert_allclose((p + nd - n - na) / donors, 0.0, atol=8e-11)
    assert float(n[0]) < 0.6 * float(donors[0])
    assert float(n[1]) > 0.95 * float(donors[1])
    fully_ionized = bands.equilibrium_fermi_energy(
        potential[0], temperature[0], donors[0]
    )
    assert float(
        bands.electron_density(potential[0], fully_ionized, temperature[0])
    ) > 0.99 * float(donors[0])
    shifted = bands.equilibrium_fermi_energy(
        potential + 0.4, temperature, donors, acceptors, ionization=ionization
    )
    np.testing.assert_allclose(shifted, ef - Q * 0.4, rtol=0, atol=1e-31)


def test_neutral_equilibrium_differentiates_material_and_dopant_parameters():
    bands = _bands("boltzmann")
    donors, temperature = 3e21, 300.0

    def population(log_donors):
        concentration = jnp.exp(log_donors)
        ef = bands.equilibrium_fermi_energy(0.0, temperature, concentration)
        return bands.electron_density(0.0, ef, temperature)

    log_donors = jnp.log(donors)
    n = population(log_donors)
    ef = bands.equilibrium_fermi_energy(0.0, temperature, donors)
    p = bands.hole_density(0.0, ef, temperature)
    expected = n * donors / (n + p)
    np.testing.assert_allclose(jax.grad(population)(log_donors), expected, rtol=3e-10)

    def intrinsic_ef(log_nc):
        model = eqx.tree_at(
            lambda item: item.conduction_density_of_states, bands, jnp.exp(log_nc)
        )
        return model.equilibrium_fermi_energy(0.0, temperature)

    derivative = jax.grad(intrinsic_ef)(jnp.log(bands.conduction_density_of_states))
    np.testing.assert_allclose(derivative, -0.5 * KB * temperature, rtol=3e-11)


def test_runtime_domain_rejections_do_not_clip_or_extrapolate():
    bands = _bands()
    transformed_domain_error = (ValueError, RuntimeError, eqx.EquinoxRuntimeError)
    with pytest.raises(transformed_domain_error):
        bands.electron_fermi_energy(0.0, 0.0, 300.0)
    with pytest.raises(transformed_domain_error):
        bands.hole_energy_density(-1.0, 300.0)
    with pytest.raises(transformed_domain_error):
        bands.statistics_value(80.01)
    with pytest.raises(transformed_domain_error):
        bands.inverse_statistics(1.01 * bands.statistics_value(80.0))
    with pytest.raises(transformed_domain_error):
        bands.band_edges(0.0, 601.0)
    with pytest.raises(transformed_domain_error):
        bands.equilibrium_fermi_energy(0.0, 300.0, donors=1e30)
    bad_levels = eqx.tree_at(
        lambda item: item.donor_binding_energy, _ionization(), 2.0 * Q
    )
    with pytest.raises(ValueError):
        bad_levels.admit(bands)
