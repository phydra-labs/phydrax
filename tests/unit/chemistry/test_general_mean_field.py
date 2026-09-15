import numpy as np

import phydrax as phx


_EXPONENTS = [3.42525091, 0.62391373, 0.16885540]
_COEFFICIENTS = [0.15432897, 0.53532814, 0.44463454]


def _hydrogen_system(distance=0.74):
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        [1, 1],
        [[0.0, 0.0, -0.5 * distance], [0.0, 0.0, 0.5 * distance]],
        [1.0, 1.0],
        units.scale,
        particle_ids=[11, 17],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0]
    )
    basis = phx.operators.quantum.gaussian.GaussianBasisPlan.from_contracted_s(
        [11, 17],
        [_EXPONENTS, _EXPONENTS],
        [_COEFFICIENTS, _COEFFICIENTS],
        source_id="general-mean-field-hydrogen",
    ).prepare(system)
    positions = np.asarray(structure.positions) * float(
        phx.units.conversion_factor(units.scale.length_unit, phx.units.BOHR)
    )
    return structure, system, basis, positions


def test_general_rhf_and_analytic_lagrangian_force_close_against_energy_difference():
    _, system, basis, positions = _hydrogen_system()
    plan = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
    )
    state = plan.solve_atomic_units(positions)
    gradient = plan.analytic_gradient_atomic_units(positions, state)
    step = 2.0e-4
    displacement = np.zeros_like(positions)
    displacement[1, 2] = step
    plus = plan.solve_atomic_units(positions + displacement)
    minus = plan.solve_atomic_units(positions - displacement)
    finite = (float(plus.total_energy) - float(minus.total_energy)) / (2.0 * step)

    assert bool(state.evidence.successful)
    np.testing.assert_allclose(state.total_energy, -1.1167593, atol=3.0e-7)
    np.testing.assert_allclose(gradient.gradient[1, 2], finite, rtol=3.0e-5, atol=3.0e-7)
    np.testing.assert_allclose(
        np.sum(np.asarray(gradient.forces), axis=0), 0.0, atol=1.0e-10
    )


def test_open_shell_reference_states_share_the_one_electron_limit():
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        [1], [[0.0, 0.0, 0.0]], [1.0], units.scale, particle_ids=[5]
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0]
    )
    basis = phx.operators.quantum.gaussian.GaussianBasisPlan.from_contracted_s(
        [5], [[1.0]], [[1.0]], source_id="one-electron-reference"
    ).prepare(system)
    sector = phx.chemistry.MolecularElectronicSectorPlan(0, 2)
    energies = []
    states = []
    for reference in (
        phx.chemistry.ElectronicReferenceKind.UNRESTRICTED,
        phx.chemistry.ElectronicReferenceKind.RESTRICTED_OPEN_SHELL,
        phx.chemistry.ElectronicReferenceKind.GENERALIZED,
    ):
        state = phx.chemistry.MolecularHartreeFockPlan(
            system, basis, sector, reference
        ).solve_atomic_units(structure.positions)
        states.append(state)
        energies.append(float(state.total_energy))

    np.testing.assert_allclose(energies, energies[0], rtol=0.0, atol=2.0e-14)
    assert all(bool(state.evidence.converged) for state in states)
    np.testing.assert_allclose(
        states[1].alpha_coefficients,
        states[1].beta_coefficients,
        rtol=0.0,
        atol=0.0,
    )


def test_stretched_restricted_hydrogen_detects_external_instability():
    _, system, basis, positions = _hydrogen_system(distance=3.0)
    plan = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
    )
    state = plan.solve_atomic_units(positions)
    stability = plan.stability_analysis(positions, state)

    assert bool(stability.successful)
    assert bool(stability.internal_stable)
    assert not bool(stability.external_stable)
    assert float(np.min(np.asarray(stability.external_eigenvalues))) < 0.0


def test_implicit_cphf_polarizability_has_response_residual_and_axial_symmetry():
    _, system, basis, positions = _hydrogen_system()
    plan = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
    )
    state = plan.solve_atomic_units(positions)
    response = phx.chemistry.MeanFieldResponsePlan().electric_response(
        plan, positions, state
    )

    assert bool(response.successful)
    np.testing.assert_allclose(
        response.polarizability, np.asarray(response.polarizability).T, atol=1.0e-12
    )
    assert float(response.polarizability[2, 2]) > 0.0
    np.testing.assert_allclose(response.polarizability[:2], 0.0, atol=1.0e-12)


def test_moving_atom_centered_grid_translates_without_changing_weights():
    structure, system, _, positions = _hydrogen_system()
    grid = phx.chemistry.MolecularDFTGridPlan(
        system,
        phx.chemistry.AtomicRadialGridPlan(8),
        angular_degree=5,
    ).prepare()
    first = grid.evaluate(positions)
    translation = np.asarray([0.4, -0.2, 0.1])
    second = grid.evaluate(positions + translation)

    assert bool(first.successful) and bool(second.successful)
    np.testing.assert_allclose(
        second.points, np.asarray(first.points) + translation, atol=1.0e-13
    )
    np.testing.assert_allclose(second.weights, first.weights, rtol=2.0e-13, atol=2.0e-14)
    np.testing.assert_allclose(first.partition_sum_residual, 0.0, atol=1.0e-14)


def test_native_pbe_rks_closes_energy_density_and_commutator_residuals():
    _, system, basis, positions = _hydrogen_system()
    grid = phx.chemistry.MolecularDFTGridPlan(
        system,
        phx.chemistry.AtomicRadialGridPlan(10),
        angular_degree=5,
    )
    plan = phx.chemistry.MolecularKohnShamPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.DensityFunctionalPlan.pbe(),
        grid,
    )
    state = plan.solve_atomic_units(positions)

    assert bool(state.evidence.successful)
    np.testing.assert_allclose(state.evidence.electron_count_residual, 0.0, atol=2.0e-12)
    assert (
        float(state.evidence.commutator_residual) <= plan.convergence.commutator_tolerance
    )
    assert np.isfinite(float(state.total_energy))


def test_implicit_cphf_nuclear_hessian_is_symmetric_and_translationally_invariant():
    _, system, basis, positions = _hydrogen_system()
    plan = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
    )
    state = plan.solve_atomic_units(positions)
    response = phx.chemistry.MeanFieldResponsePlan().nuclear_hessian(
        plan, positions, state
    )
    matrix = np.asarray(response.hessian).reshape((positions.size, positions.size))

    assert bool(response.successful)
    np.testing.assert_allclose(matrix, matrix.T, atol=2.0e-10)
    np.testing.assert_allclose(
        np.sum(np.asarray(response.hessian), axis=2),
        0.0,
        atol=2.0e-8,
    )


def test_adiabatic_tddft_orbital_hessians_produce_positive_tda_and_full_roots():
    _, system, basis, positions = _hydrogen_system()
    grid = phx.chemistry.MolecularDFTGridPlan(
        system,
        phx.chemistry.AtomicRadialGridPlan(6),
        angular_degree=3,
    )
    plan = phx.chemistry.MolecularKohnShamPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.DensityFunctionalPlan.lda_pw92(),
        grid,
    )
    state = plan.solve_atomic_units(positions)
    response = phx.chemistry.KohnShamExcitedResponsePlan(plan, positions, state)
    tda = response.tda(1).solve()
    full = response.tddft(1).solve()

    assert bool(tda.successful) and bool(full.successful)
    assert float(tda.excitation_energies[0]) > 0.0
    assert float(full.excitation_energies[0]) > 0.0
    np.testing.assert_allclose(response.a_matrix, response.a_matrix.T, atol=1.0e-11)
    np.testing.assert_allclose(response.b_matrix, response.b_matrix.T, atol=1.0e-11)
