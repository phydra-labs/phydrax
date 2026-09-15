import jax.numpy as jnp
import numpy as np

import phydrax as phx


periodic = phx.chemistry.periodic


def test_supercell_force_constants_and_phonons_enforce_acoustic_invariants():
    equilibrium = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def forces(positions):
        displacement = positions[1] - positions[0] - jnp.asarray([1.0, 0.0, 0.0])
        return jnp.stack((displacement, -displacement))

    force_constants = periodic.SupercellForceConstantPlan(
        forces,
        equilibrium,
        symmetry_tolerance=1.0e-10,
    ).evaluate()
    blocks = jnp.zeros((3, 1, 3, 1, 3))
    blocks = blocks.at[0, 0, :, 0, :].set(-jnp.eye(3))
    blocks = blocks.at[1, 0, :, 0, :].set(2.0 * jnp.eye(3))
    blocks = blocks.at[2, 0, :, 0, :].set(-jnp.eye(3))
    phonons = periodic.PeriodicPhononPlan(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        blocks,
        [1.0],
        10.0,
    ).evaluate([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])

    assert bool(force_constants.successful) and bool(phonons.successful)
    np.testing.assert_allclose(
        force_constants.acoustic_sum_rule_residual, 0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(phonons.frequencies[0], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(phonons.frequencies[1], np.sqrt(2.0), atol=1.0e-12)


def test_lattice_thermodynamics_qha_and_rta_transport_remain_finite():
    thermal = periodic.lattice_thermodynamics([[1.0, 1.5, 2.0]], [1.0], 1.0, 1.0, 1.0)
    qha = periodic.quasi_harmonic_thermodynamics(
        [9.0, 10.0, 11.0],
        [0.1, 0.0, 0.1],
        [[[1.1, 1.6]], [[1.0, 1.5]], [[0.9, 1.4]]],
        [1.0],
        [0.5, 1.0, 1.5],
        1.0,
        1.0,
    )
    transport = periodic.anharmonic_rta_transport(
        [1.0, 0.6, 0.4],
        [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.2, 0.0, 0.0]],
        [1.0, 0.8, 0.5],
        np.ones((3, 3, 3)) * 0.01,
        10.0,
        0.05,
    )

    assert (
        bool(thermal.successful) and bool(qha.successful) and bool(transport.successful)
    )
    assert float(thermal.heat_capacity) > 0.0
    assert np.all(np.isfinite(np.asarray(qha.thermal_expansion)))
    assert float(transport.thermal_conductivity[0, 0]) > 0.0


def test_diagonal_gw_and_bse_close_solvable_one_transition_models():
    gw = periodic.DiagonalGWPlan(
        [-0.5, 0.3],
        [0.0, 0.0],
        [[-0.6, -0.2], [0.2, 0.6]],
        lambda index, energy: 0.1 + 0.0 * index + 0.0 * energy,
        "constant-self-energy",
        phx.units.HARTREE,
    ).evaluate()
    dipole_unit = phx.units.derived_unit(
        "e*bohr-periodic-bse",
        ((phx.units.ELEMENTARY_CHARGE, 1), (phx.units.BOHR, 1)),
    )
    bse = periodic.BetheSalpeterPlan(
        [0.5],
        [[0.1]],
        [[0.0]],
        [[1.0, 0.0, 0.0]],
        -1.0,
        phx.units.HARTREE,
        dipole_unit,
    )
    tda = bse.tda(1)
    full = bse.full(1)

    assert bool(gw.successful) and bool(tda.successful) and bool(full.successful)
    np.testing.assert_allclose(gw.quasiparticle_energies, [-0.4, 0.4], atol=1.0e-9)
    np.testing.assert_allclose(gw.renormalization_factors, 1.0, atol=1.0e-13)
    np.testing.assert_allclose(tda.excitation_energies, [0.4], atol=1.0e-12)
    np.testing.assert_allclose(full.excitation_energies, [0.4], atol=1.0e-12)
