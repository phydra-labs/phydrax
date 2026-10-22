"""Matched external DF-RHF energies and genuinely contracted analytic forces."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import phydrax as phx
from benchmarks.cuest_molecular import build_case, CuestCase, external_reference


pytestmark = [pytest.mark.cuda_native, pytest.mark.usefixtures("cuest_runtime")]


@pytest.fixture
def case(cuest_runtime: None) -> CuestCase:
    pytest.importorskip(
        "pyscf", reason="independent matched DF-RHF integration oracle requires PySCF"
    )
    case = build_case()
    try:
        one = case.integrals.one_electron(case.basis, case.positions, np.ones(2))
        jax.block_until_ready(one)
    except (ImportError, OSError, RuntimeError) as error:
        pytest.fail(f"cuEST provider NOT ADMITTED: {type(error).__name__}: {error}")
    assert bool(one.successful), "cuEST provider NOT ADMITTED"
    return case


def solve(
    case: CuestCase, positions: np.ndarray
) -> phx.chemistry.RestrictedMeanFieldState:
    state = case.hf.solve_atomic_units(positions)
    assert isinstance(state, phx.chemistry.RestrictedMeanFieldState)
    assert bool(state.evidence.successful), "native cuEST DF-RHF did not converge"
    assert state.integral_definition_id == case.integrals.definition_id(case.basis)
    return state


def test_native_scf_matches_independent_df_energy_density_and_stationarity(
    case: CuestCase,
) -> None:
    state = solve(case, case.positions)
    reference = external_reference(case)
    expected_energy = reference.kernel()
    assert reference.converged
    np.testing.assert_allclose(state.total_energy, expected_energy, rtol=0.0, atol=2.0e-9)
    permutation = case.pyscf_to_native
    np.testing.assert_allclose(
        state.density,
        reference.make_rdm1()[np.ix_(permutation, permutation)],
        rtol=2.0e-7,
        atol=2.0e-8,
    )
    density = np.asarray(state.density, dtype=np.float64)
    overlap = np.asarray(state.overlap, dtype=np.float64)
    fock = np.asarray(state.fock, dtype=np.float64)
    np.testing.assert_allclose(np.trace(density @ overlap), 2.0, atol=2.0e-9)
    np.testing.assert_allclose(
        fock @ density @ overlap - overlap @ density @ fock, 0.0, atol=2.0e-9
    )
    bound = case.source.prepare(case.basis).bind_geometry(
        case.positions, np.ones(len(case.positions))
    )
    jk = bound.jk(state.density)
    one = bound.one_electron()
    np.testing.assert_allclose(
        fock, one.core_hamiltonian + jk.coulomb - 0.5 * jk.exchange, atol=2.0e-9
    )
    reconstructed_energy = 0.5 * np.einsum(
        "ij,ij->", density, np.asarray(one.core_hamiltonian) + fock
    ) + float(one.nuclear_repulsion)
    np.testing.assert_allclose(state.total_energy, reconstructed_energy, atol=2.0e-10)


@pytest.mark.parametrize("step", (1.0e-3, 3.0e-4, 1.0e-4))
def test_analytic_forces_match_same_definition_and_external_df_energy_differences(
    case: CuestCase, step: float
) -> None:
    positions = case.positions.copy()
    # A tilted bond exercises all Cartesian force components, not just symmetry zeros.
    positions[0] += np.asarray([-0.12, 0.09, -0.03])
    positions[1] += np.asarray([0.12, -0.09, 0.03])
    state = solve(case, positions)
    gradient = case.hf.analytic_gradient_atomic_units(positions, state)
    assert bool(gradient.successful)
    np.testing.assert_allclose(
        gradient.gradient,
        gradient.one_electron + gradient.two_electron + gradient.pulay + gradient.nuclear,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        gradient.forces, -np.asarray(gradient.gradient), atol=2.0e-13
    )
    np.testing.assert_allclose(np.sum(gradient.forces, axis=0), 0.0, atol=2.0e-8)
    np.testing.assert_allclose(
        np.sum(np.cross(positions, gradient.forces), axis=0), 0.0, atol=2.0e-8
    )
    for atom, axis in ((0, 0), (1, 1), (1, 2)):
        shift = np.zeros_like(positions)
        shift[atom, axis] = step
        plus, minus = solve(case, positions + shift), solve(case, positions - shift)
        native_finite = (float(plus.total_energy) - float(minus.total_energy)) / (
            2 * step
        )
        external_energies = []
        for displaced in (positions + shift, positions - shift):
            reference = external_reference(case, displaced)
            external_energies.append(reference.kernel())
            assert reference.converged
        external_finite = (external_energies[0] - external_energies[1]) / (2 * step)
        np.testing.assert_allclose(
            gradient.gradient[atom, axis], native_finite, rtol=4.0e-5, atol=8.0e-7
        )
        np.testing.assert_allclose(
            gradient.gradient[atom, axis], external_finite, rtol=4.0e-5, atol=8.0e-7
        )


def test_user_unit_evaluate_and_moving_geometry_are_consistent(case: CuestCase) -> None:
    state = solve(case, case.positions)
    gradient = case.hf.analytic_gradient_atomic_units(case.positions, state)
    evaluated, forces = case.hf.evaluate(case.structure.positions)
    length = float(
        phx.units.conversion_factor(case.system.units.scale.length_unit, phx.units.BOHR)
    )
    energy = float(
        phx.units.conversion_factor(
            phx.units.HARTREE, case.system.units.scale.energy_unit
        )
    )
    np.testing.assert_allclose(evaluated.total_energy, state.total_energy, atol=2.0e-11)
    np.testing.assert_allclose(
        forces, np.asarray(gradient.forces) * energy * length, rtol=2.0e-9, atol=2.0e-9
    )
    moved = case.positions.copy()
    moved[1, 2] += 0.14
    changed = solve(case, moved)
    reference = external_reference(case, moved)
    expected = reference.kernel()
    assert reference.converged
    np.testing.assert_allclose(changed.total_energy, expected, atol=2.0e-9)
    assert abs(float(changed.total_energy - state.total_energy)) > 1.0e-5
    with pytest.raises(ValueError, match="(?i)(geometry|another|owner)"):
        case.hf.analytic_gradient_atomic_units(moved, state)
    restored = solve(case, case.positions)
    np.testing.assert_allclose(restored.total_energy, state.total_energy, atol=2.0e-11)


def test_hessian_and_hf_autodiff_do_not_silently_use_exact_integrals(
    case: CuestCase,
) -> None:
    state = solve(case, case.positions)
    with pytest.raises(
        NotImplementedError, match="(?i)(exact|integral definition|Hessian)"
    ):
        phx.chemistry.MeanFieldResponsePlan().nuclear_hessian(
            case.hf, case.positions, state
        )
    with pytest.raises(NotImplementedError, match="(?i)(AD|autodiff|differentiat|JVP)"):
        jax.jvp(
            lambda positions: case.hf.solve_atomic_units(positions).total_energy,
            (case.positions,),
            (np.ones_like(case.positions),),
        )
