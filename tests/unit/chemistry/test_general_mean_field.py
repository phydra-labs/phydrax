from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import phydrax as phx
from phydrax.operators.quantum.gaussian import (
    FactorizedERITensor,
    PreparedDirectJK,
    PreparedGaussianBasis,
)


_EXPONENTS = [3.42525091, 0.62391373, 0.16885540]
_COEFFICIENTS = [0.15432897, 0.53532814, 0.44463454]


def _hydrogen_system(distance: Any = 0.74) -> Any:
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, -0.5 * distance], [0.0, 0.0, 0.5 * distance]],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units.scale,
        # ty: ignore[invalid-argument-type]
        particle_ids=[11, 17],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0]
    )
    basis = phx.operators.quantum.gaussian.GaussianBasisPlan.from_contracted_s(
        [11, 17],
        # ty: ignore[invalid-argument-type]
        [_EXPONENTS, _EXPONENTS],
        # ty: ignore[invalid-argument-type]
        [_COEFFICIENTS, _COEFFICIENTS],
        source_id="general-mean-field-hydrogen",
    ).prepare(system)
    positions = np.asarray(structure.positions) * float(
        phx.units.conversion_factor(units.scale.length_unit, phx.units.BOHR)
    )
    return structure, system, basis, positions


def test_general_mean_field_scenario_1() -> None:
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
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [1.0],
        units.scale,
        # ty: ignore[invalid-argument-type]
        particle_ids=[5],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0]
    )
    basis = phx.operators.quantum.gaussian.GaussianBasisPlan.from_contracted_s(
        [5],
        # ty: ignore[invalid-argument-type]
        [[1.0]],
        # ty: ignore[invalid-argument-type]
        [[1.0]],
        source_id="one-electron-reference",
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
        # ty: ignore[unresolved-attribute]
        states[1].alpha_coefficients,
        # ty: ignore[unresolved-attribute]
        states[1].beta_coefficients,
        rtol=0.0,
        atol=0.0,
    )
    _, system, basis, positions = _hydrogen_system(distance=3.0)
    plan = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
    )
    state = plan.solve_atomic_units(positions)
    # ty: ignore[invalid-argument-type]
    stability = plan.stability_analysis(positions, state)

    assert bool(stability.successful)
    assert bool(stability.internal_stable)
    assert not bool(stability.external_stable)
    assert float(np.min(np.asarray(stability.external_eigenvalues))) < 0.0


def test_general_mean_field_scenario_2() -> None:
    _, system, basis, positions = _hydrogen_system()
    plan = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
    )
    state = plan.solve_atomic_units(positions)
    response = phx.chemistry.MeanFieldResponsePlan().electric_response(
        plan,
        positions,
        # ty: ignore[invalid-argument-type]
        state,
    )

    assert bool(response.successful)
    np.testing.assert_allclose(
        response.polarizability, np.asarray(response.polarizability).T, atol=1.0e-12
    )
    assert float(response.polarizability[2, 2]) > 0.0
    np.testing.assert_allclose(response.polarizability[:2], 0.0, atol=1.0e-12)
    _, system, basis, positions = _hydrogen_system()
    plan = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
    )
    state = plan.solve_atomic_units(positions)
    response = phx.chemistry.MeanFieldResponsePlan().nuclear_hessian(
        plan,
        positions,
        # ty: ignore[invalid-argument-type]
        state,
    )
    matrix = np.asarray(response.hessian).reshape((positions.size, positions.size))

    assert bool(response.successful)
    np.testing.assert_allclose(matrix, matrix.T, atol=2.0e-10)
    np.testing.assert_allclose(
        np.sum(np.asarray(response.hessian), axis=2),
        0.0,
        atol=2.0e-8,
    )
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


def test_adiabatic_tddft_orbital_hessians_produce_positive_tda_and_full_roots() -> None:
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
    # ty: ignore[invalid-argument-type]
    response = phx.chemistry.KohnShamExcitedResponsePlan(plan, positions, state)
    tda = response.tda(1).solve()
    full = response.tddft(1).solve()

    assert bool(tda.successful) and bool(full.successful)
    assert float(tda.excitation_energies[0]) > 0.0
    assert float(full.excitation_energies[0]) > 0.0
    np.testing.assert_allclose(response.a_matrix, response.a_matrix.T, atol=1.0e-11)
    np.testing.assert_allclose(response.b_matrix, response.b_matrix.T, atol=1.0e-11)


def _approximate_integral_source(
    basis: PreparedGaussianBasis, positions: NDArray[np.float64], representation: str
) -> FactorizedERITensor | PreparedDirectJK:
    gaussian = phx.operators.quantum.gaussian
    if representation == "direct":
        return gaussian.DirectJKPlan(gaussian.GaussianScreeningPlan(1.0)).prepare(
            basis, positions
        )
    eri = gaussian.electron_repulsion_tensor(basis, positions)
    if representation == "pivoted-cholesky":
        return gaussian.PivotedCholeskyERIPlan(1.0).factorize(eri)
    if representation == "fixed":
        exact = gaussian.PivotedCholeskyERIPlan(1.0e-12).factorize(eri)
        return gaussian.FactorizedERITensor(
            0.5 * exact.factors,
            np.max(np.abs(np.asarray(eri) - 0.25 * np.asarray(exact.reconstruct()))),
            "deliberately-approximate-hydrogen-factors",
            "fixed",
        )
    raise ValueError(f"Unknown test integral representation: {representation}")


@pytest.mark.parametrize("representation", ("fixed", "pivoted-cholesky", "direct"))
def test_approximate_hf_refuses_exact_gradient_stability_and_response(
    representation: str,
) -> None:
    _, system, basis, positions = _hydrogen_system()
    gaussian = phx.operators.quantum.gaussian
    source = _approximate_integral_source(basis, positions, representation)
    integrals = gaussian.MolecularIntegralPlan(source)
    plan = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
        integrals=integrals,
    )
    exact = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
    ).solve_atomic_units(positions)
    state = plan.solve_atomic_units(positions)
    assert isinstance(state, phx.chemistry.RestrictedMeanFieldState)

    assert plan.integrals is integrals
    assert bool(state.evidence.successful)
    assert abs(float(state.total_energy - exact.total_energy)) > 1.0e-3
    assert state.integral_definition_id == integrals.definition_id(basis)
    assert state.integral_definition_id != exact.integral_definition_id
    if isinstance(source, PreparedDirectJK):
        assert (
            source.screening.retained_quartet_count < source.screening.total_quartet_count
        )
    else:
        assert float(source.residual_bound) > 1.0e-3

    with pytest.raises(NotImplementedError, match="integral definition"):
        plan.analytic_gradient_atomic_units(positions, state)
    with pytest.raises(NotImplementedError, match="integral definition"):
        plan.stability_analysis(positions, state)
    response = phx.chemistry.MeanFieldResponsePlan()
    with pytest.raises(NotImplementedError, match="exact integrals"):
        response.electric_response(plan, positions, state)
    with pytest.raises(NotImplementedError, match="exact integrals"):
        response.nuclear_hessian(plan, positions, state)
    with pytest.raises(NotImplementedError, match="exact integrals"):
        phx.chemistry.HartreeFockExcitedResponsePlan(plan, state, positions)


@pytest.mark.parametrize("representation", ("exact", "fixed", "direct"))
def test_rhf_returns_physical_fock_for_actual_returned_density(
    representation: str,
) -> None:
    _, system, basis, positions = _hydrogen_system()
    gaussian = phx.operators.quantum.gaussian
    source = (
        None
        if representation == "exact"
        else _approximate_integral_source(basis, positions, representation)
    )
    plan = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
        integrals=gaussian.MolecularIntegralPlan(source),
        convergence=phx.chemistry.SCFConvergencePlan(maximum_iterations=1),
        guess=phx.chemistry.InitialGuessPlan(phx.chemistry.InitialGuessKind.ZERO),
        acceleration=phx.chemistry.SCFAccelerationPlan(damping=0.4, level_shift=0.3),
    )
    state = plan.solve_atomic_units(positions)
    assert isinstance(state, phx.chemistry.RestrictedMeanFieldState)
    core = gaussian.kinetic_matrix(basis, positions) + gaussian.nuclear_attraction_matrix(
        basis, positions, np.asarray(system.atomic_numbers, dtype=np.float64)
    )
    density = np.asarray(state.density)
    if isinstance(source, PreparedDirectJK):
        result = source.evaluate(positions, density)
        coulomb, exchange = result.coulomb, result.exchange
    else:
        eri = np.asarray(
            gaussian.electron_repulsion_tensor(basis, positions)
            if source is None
            else source.reconstruct()
        )
        coulomb = np.einsum("cd,abcd->ab", density, eri)
        exchange = np.einsum("cd,acbd->ab", density, eri)
    physical = np.asarray(core + coulomb - 0.5 * exchange)

    assert not bool(state.evidence.converged)
    assert np.max(np.abs(density)) > 0.0
    np.testing.assert_allclose(state.fock, physical, rtol=2.0e-12, atol=2.0e-13)
    np.testing.assert_allclose(
        state.electronic_energy,
        0.5 * np.einsum("ab,ab->", density, np.asarray(core) + physical),
        rtol=2.0e-12,
        atol=2.0e-13,
    )
    commutator = physical @ density @ state.overlap - state.overlap @ density @ physical
    np.testing.assert_allclose(
        state.evidence.commutator_residual,
        np.max(np.abs(commutator)),
        rtol=2.0e-12,
        atol=2.0e-13,
    )


def test_nonconverged_rhf_force_reports_unsuccessful() -> None:
    _, system, basis, positions = _hydrogen_system()
    plan = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
        convergence=phx.chemistry.SCFConvergencePlan(maximum_iterations=1),
        guess=phx.chemistry.InitialGuessPlan(phx.chemistry.InitialGuessKind.ZERO),
    )
    state = plan.solve_atomic_units(positions)
    gradient = plan.analytic_gradient_atomic_units(positions, state)

    assert not bool(state.evidence.converged)
    assert not bool(state.evidence.successful)
    assert not bool(gradient.successful)
    assert np.all(np.isfinite(np.asarray(gradient.forces)))
    np.testing.assert_allclose(gradient.forces, -gradient.gradient, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("representation", ("exact", "fixed", "direct"))
def test_correlation_preserves_matching_integral_definition_and_rejects_substitution(
    representation: str,
) -> None:
    _, system, basis, positions = _hydrogen_system()
    gaussian = phx.operators.quantum.gaussian
    source = (
        None
        if representation == "exact"
        else _approximate_integral_source(basis, positions, representation)
    )
    state = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
        integrals=gaussian.MolecularIntegralPlan(source),
    ).solve_atomic_units(positions)
    assert isinstance(state, phx.chemistry.RestrictedMeanFieldState)
    assert bool(state.evidence.successful)
    correlation = phx.chemistry.electronic_structure.correlation
    partition = correlation.CorrelatedOrbitalPartition.from_restricted_state(state)
    transformation = correlation.MolecularIntegralTransformationPlan()
    charges = np.asarray(system.atomic_numbers, dtype=np.float64)

    if isinstance(source, PreparedDirectJK):
        with pytest.raises(ValueError, match="integral definition"):
            transformation.transform_restricted(
                basis, positions, charges, state, partition
            )
        return

    store = transformation.transform_restricted(
        basis, positions, charges, state, partition, factorized_ao=source
    )
    ao_eri = np.asarray(
        gaussian.electron_repulsion_tensor(basis, positions)
        if source is None
        else source.reconstruct()
    )
    coefficients = np.asarray(state.coefficients)
    expected = np.einsum(
        "pi,qj,rk,sl,pqrs->ijkl",
        coefficients.conj(),
        coefficients,
        coefficients.conj(),
        coefficients,
        ao_eri,
    )
    assert store.integral_definition_id == state.integral_definition_id
    assert store.source_basis_id == basis.prepared_id
    np.testing.assert_allclose(
        store.dense_two_body(), expected, rtol=2.0e-11, atol=2.0e-12
    )
    np.testing.assert_allclose(
        store.reference_energy, state.total_energy, rtol=0.0, atol=0.0
    )
    mp2 = correlation.MP2Plan().evaluate(store)
    assert bool(mp2.successful)
    np.testing.assert_allclose(
        mp2.total_energy, state.total_energy + mp2.correlation_energy
    )

    if source is None:
        different = _approximate_integral_source(basis, positions, "fixed")
        assert isinstance(different, FactorizedERITensor)
        with pytest.raises(ValueError, match="integral definition"):
            transformation.transform_restricted(
                basis, positions, charges, state, partition, factorized_ao=different
            )
    else:
        with pytest.raises(ValueError, match="integral definition"):
            transformation.transform_restricted(
                basis, positions, charges, state, partition
            )
        different = gaussian.FactorizedERITensor(
            0.9 * source.factors,
            source.residual_bound,
            source.source_id,
            source.representation,
        )
        with pytest.raises(ValueError, match="integral definition"):
            transformation.transform_restricted(
                basis, positions, charges, state, partition, factorized_ao=different
            )


def test_tiny_hydrogen_converges_with_rank_aware_diis_and_retained_evidence() -> None:
    _, system, basis, positions = _hydrogen_system()
    plan = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
        acceleration=phx.chemistry.SCFAccelerationPlan(diis_start=1),
    )
    state = plan.solve_atomic_units(positions)
    assert bool(state.evidence.successful)
    assert state.evidence.diis is not None
    assert bool(state.evidence.diis.successful)
    assert bool(state.evidence.diis.linear_solve.successful)
    assert int(state.evidence.diis.linear_solve.diagnostics.rank) >= 0
    assert float(state.evidence.diis.constraint_residual) <= float(
        state.evidence.diis.constraint_tolerance
    )
    np.testing.assert_allclose(state.total_energy, -1.1167593, atol=3.0e-7)
    gradient = plan.analytic_gradient_atomic_units(positions, state)
    assert bool(gradient.successful)
