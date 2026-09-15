import numpy as np
import pytest

import phydrax as phx


_H_EXPONENTS = [3.42525091, 0.62391373, 0.16885540]
_H_COEFFICIENTS = [0.15432897, 0.53532814, 0.44463454]


def _hydrogen_dimer():
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        [1, 1],
        [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]],
        [1.008, 1.008],
        units.scale,
        particle_ids=[11, 17],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0]
    )
    basis = phx.operators.quantum.gaussian.GaussianBasisPlan.from_contracted_s(
        [11, 17],
        [_H_EXPONENTS, _H_EXPONENTS],
        [_H_COEFFICIENTS, _H_COEFFICIENTS],
        source_id="sto-3g-hydrogen-fixture",
    ).prepare(system)
    return units, structure, system, basis


def test_contracted_gaussian_integrals_obey_normalization_and_permutation_symmetry():
    _, structure, system, basis = _hydrogen_dimer()
    positions_bohr = np.asarray(structure.positions) * float(
        phx.units.conversion_factor(system.units.scale.length_unit, phx.units.BOHR)
    )
    integrals = phx.operators.quantum.gaussian.molecular_integrals(
        basis, positions_bohr, [1.0, 1.0]
    )

    np.testing.assert_allclose(np.diag(integrals.overlap), 1.0, atol=2.0e-7)
    np.testing.assert_allclose(integrals.overlap, integrals.overlap.T, atol=1.0e-12)
    np.testing.assert_allclose(
        integrals.electron_repulsion,
        np.transpose(integrals.electron_repulsion, (1, 0, 2, 3)),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        integrals.electron_repulsion,
        np.transpose(integrals.electron_repulsion, (2, 3, 0, 1)),
        atol=1.0e-12,
    )


def test_native_rhf_closes_energy_force_and_excited_spectrum_chain():
    units, structure, system, basis = _hydrogen_dimer()
    rhf = phx.chemistry.NativeRHFPlan(
        system,
        basis,
        2,
        convergence_tolerance=1.0e-9,
        maximum_iterations=128,
        damping=0.2,
        force_displacement=2.0e-4,
    )
    positions_bohr = np.asarray(structure.positions) * float(
        phx.units.conversion_factor(units.scale.length_unit, phx.units.BOHR)
    )
    state = rhf.solve_atomic_units(positions_bohr)

    assert bool(state.converged)
    np.testing.assert_allclose(float(state.total_energy), -1.11676, atol=2.0e-4)
    kernel, evaluated_state = rhf.evaluate(structure.positions, dipole=True)
    assert bool(kernel.successful)
    np.testing.assert_allclose(
        np.sum(np.asarray(kernel.forces), axis=0), 0.0, atol=2.0e-5
    )
    np.testing.assert_allclose(kernel.dipole, 0.0, atol=1.0e-8)

    manifold_plan = phx.chemistry.ExcitedStateManifoldPlan(1)
    manifold = phx.chemistry.rhf_tamm_dancoff(
        rhf, structure.positions, evaluated_state, manifold_plan
    ).solve()
    spectrum = phx.chemistry.UVVisibleSpectrumPlan(
        phx.chemistry.SpectralAxis.ENERGY_EV,
        phx.chemistry.SpectralLineShape.GAUSSIAN,
        1.0,
        40.0,
        grid_size=4001,
        fwhm=0.2,
        area_tolerance=1.0e-3,
    ).evaluate(manifold)

    assert bool(manifold.successful)
    assert float(manifold.excitation_energies[0]) > 0.0
    assert bool(spectrum.successful)
    np.testing.assert_allclose(
        spectrum.integrated_strength,
        spectrum.expected_strength,
        rtol=1.0e-3,
        atol=1.0e-12,
    )


def test_native_rhf_tda_rejects_unsupported_spin_and_symmetry_sectors():
    units, structure, system, basis = _hydrogen_dimer()
    rhf = phx.chemistry.NativeRHFPlan(
        system,
        basis,
        2,
        convergence_tolerance=1.0e-9,
        damping=0.2,
    )
    positions_bohr = np.asarray(structure.positions) * float(
        phx.units.conversion_factor(units.scale.length_unit, phx.units.BOHR)
    )
    state = rhf.solve_atomic_units(positions_bohr)

    with pytest.raises(ValueError, match="singlet spin sector"):
        phx.chemistry.rhf_tamm_dancoff(
            rhf,
            structure.positions,
            state,
            phx.chemistry.ExcitedStateManifoldPlan(1, spin_sector="triplet"),
        )
    with pytest.raises(ValueError, match="symmetry-sector projection"):
        phx.chemistry.rhf_tamm_dancoff(
            rhf,
            structure.positions,
            state,
            phx.chemistry.ExcitedStateManifoldPlan(
                1,
                symmetry_sector="A1",
            ),
        )


def test_native_lda_produces_symmetric_static_polarizability():
    _, structure, system, basis = _hydrogen_dimer()
    grid = phx.chemistry.MolecularIntegrationGridPlan.cartesian_box(
        system,
        [0.0, 0.0, 0.0],
        half_width=4.0,
        points_per_axis=9,
    )
    lda = phx.chemistry.NativeLDAPlan(
        system,
        basis,
        grid,
        2,
        convergence_tolerance=1.0e-7,
        maximum_iterations=128,
        damping=0.3,
        field_displacement=2.0e-3,
    )
    result = lda.static_polarizability(structure.positions)

    assert bool(result.successful)
    assert float(result.symmetry_residual) <= 1.0e-10
    np.testing.assert_allclose(result.tensor, result.tensor.T, atol=1.0e-10)
    assert np.all(np.isfinite(np.asarray(result.tensor)))


def test_native_rhf_embedding_returns_conservative_point_charge_forces():
    units, structure, system, basis = _hydrogen_dimer()
    plan = phx.chemistry.NativeRHFPlan(
        system,
        basis,
        2,
        convergence_tolerance=1.0e-9,
        damping=0.2,
        force_displacement=5.0e-4,
    )
    provider = phx.chemistry.NativeRHFEmbeddedRegionProvider(plan)
    embedding = phx.chemistry.ElectrostaticEmbeddingState(
        [101],
        [[0.0, 0.0, 3.0]],
        [0.1],
        units,
    )
    result = provider.evaluate(structure.positions, embedding)

    assert bool(result.successful)
    total_force = np.sum(np.asarray(result.region_forces), axis=0) + np.sum(
        np.asarray(result.point_charge_forces), axis=0
    )
    np.testing.assert_allclose(total_force, 0.0, atol=2.0e-4)
