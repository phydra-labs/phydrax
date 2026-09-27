from typing import Any

import numpy as np
import pytest

import phydrax as phx


_EXPONENTS = [3.42525091, 0.62391373, 0.16885540]
_COEFFICIENTS = [0.15432897, 0.53532814, 0.44463454]


def _h2_store() -> Any:
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units.scale,
        # ty: ignore[invalid-argument-type]
        particle_ids=[1, 2],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0]
    )
    gaussian = phx.operators.quantum.gaussian
    basis = gaussian.GaussianBasisPlan.from_contracted_s(
        [1, 2],
        # ty: ignore[invalid-argument-type]
        [_EXPONENTS, _EXPONENTS],
        # ty: ignore[invalid-argument-type]
        [_COEFFICIENTS, _COEFFICIENTS],
        source_id="correlation-hydrogen",
    ).prepare(system)
    positions = np.asarray(structure.positions) * float(
        phx.units.conversion_factor(units.scale.length_unit, phx.units.BOHR)
    )
    state = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
    ).solve_atomic_units(positions)
    correlation = phx.chemistry.electronic_structure.correlation
    # ty: ignore[invalid-argument-type]
    partition = correlation.CorrelatedOrbitalPartition.from_restricted_state(state)
    store = correlation.MolecularIntegralTransformationPlan().transform_restricted(
        basis,
        positions,
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        # ty: ignore[invalid-argument-type]
        state,
        partition,
    )
    return state, store


def test_correlation_scenario_1() -> None:
    state, store = _h2_store()
    correlation = phx.chemistry.electronic_structure.correlation
    mp2 = correlation.MP2Plan().evaluate(store)
    fci = correlation.FCIPlan(1, 1).evaluate(store)

    assert bool(mp2.successful) and bool(fci.successful)
    assert float(mp2.total_energy) < float(state.total_energy)
    assert float(fci.energies[0]) < float(mp2.total_energy)
    np.testing.assert_allclose(
        mp2.total_energy, state.total_energy + mp2.correlation_energy
    )
    np.testing.assert_allclose(fci.energies[0], -1.13728383, atol=5.0e-8)
    _, store = _h2_store()
    correlation = phx.chemistry.electronic_structure.correlation
    casci = correlation.CASCIPlan((0, 1), 1, 1)
    casscf = correlation.CASSCFPlan(casci).evaluate(store)

    assert bool(casscf.successful)
    np.testing.assert_allclose(casscf.orbital_rotation, np.eye(2), atol=0.0)
    np.testing.assert_allclose(casscf.orbital_gradient_norms, [0.0], atol=0.0)
    np.testing.assert_allclose(casscf.casci.energies[0], -1.13728383, atol=5.0e-8)


def test_pyscf_ccsd_closes_right_and_lambda_residuals_and_reaches_fci_for_two_electrons() -> (
    None
):
    pytest.importorskip("pyscf")
    _, store = _h2_store()
    correlation = phx.chemistry.electronic_structure.correlation
    provider = phx.chemistry.interchange.PySCFCoupledClusterProvider()
    plan = correlation.CoupledClusterPlan(convergence_tolerance=1.0e-9)
    result = provider.evaluate(plan, store)
    resumed = provider.evaluate(plan, store, checkpoint=result.checkpoint())
    fci = correlation.FCIPlan(1, 1).evaluate(store)

    assert bool(result.successful)
    assert float(result.amplitude_residual) <= 1.0e-9
    assert float(result.lambda_residual) <= 1.0e-9
    np.testing.assert_allclose(result.total_energy, fci.energies[0], atol=2.0e-9)
    assert bool(resumed.successful)
    assert int(resumed.iterations) >= int(result.iterations)
    np.testing.assert_allclose(resumed.total_energy, result.total_energy, atol=2.0e-12)
