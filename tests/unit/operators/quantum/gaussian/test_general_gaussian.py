import jax
import numpy as np

import phydrax as phx


def _system(numbers, positions, particle_ids):
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        numbers,
        positions,
        np.ones((len(numbers),)),
        units.scale,
        particle_ids=particle_ids,
    )
    return structure, phx.atomistic.AtomisticSystemPlan.from_structure(
        structure,
        units,
        molecule_ids=np.zeros((len(numbers),), dtype=np.int32),
    )


def test_boys_derivative_matches_next_order_across_numerical_regimes():
    gaussian = phx.operators.quantum.gaussian
    arguments = np.asarray([0.0, 1.0e-9, 0.2, 20.0, 100.0])
    values = gaussian.boys_values(1, arguments)
    derivative = jax.vmap(jax.grad(gaussian.boys0))(arguments)

    np.testing.assert_allclose(
        derivative, -np.asarray(values)[:, 1], rtol=2.0e-11, atol=2.0e-13
    )
    np.testing.assert_allclose(values[0], [1.0, 1.0 / 3.0], rtol=0.0, atol=1.0e-14)


def test_real_spherical_shells_are_metric_orthonormal_through_f():
    structure, system = _system([1], [[0.0, 0.0, 0.0]], [11])
    gaussian = phx.operators.quantum.gaussian
    shells = tuple(
        gaussian.GaussianShellPlan(
            11,
            angular,
            [0.7],
            [1.0],
            representation=gaussian.GaussianShellRepresentation.REAL_SPHERICAL,
        )
        for angular in range(4)
    )
    basis = gaussian.GaussianBasisPlan(
        shells, source_id="spherical-orthonormality"
    ).prepare(system)
    overlap = gaussian.overlap_matrix(basis, structure.positions)

    assert basis.cartesian_basis_function_count == 20
    assert basis.basis_function_count == 16
    np.testing.assert_allclose(overlap, np.eye(16), rtol=1.0e-12, atol=2.0e-12)


def test_ao_spatial_gradient_matches_centered_difference_for_p_shell():
    structure, system = _system([1], [[0.0, 0.0, 0.0]], [17])
    gaussian = phx.operators.quantum.gaussian
    basis = gaussian.GaussianBasisPlan(
        [gaussian.GaussianShellPlan(17, 1, [0.8, 0.25], [0.7, 0.3])],
        source_id="p-gradient",
    ).prepare(system)
    point = np.asarray([[0.2, -0.1, 0.3]])
    gradient = np.asarray(gaussian.ao_gradients(basis, structure.positions, point))[0]
    step = 1.0e-5
    finite = np.empty_like(gradient)
    for axis in range(3):
        displacement = np.zeros_like(point)
        displacement[0, axis] = step
        plus = gaussian.ao_values(basis, structure.positions, point + displacement)[0]
        minus = gaussian.ao_values(basis, structure.positions, point - displacement)[0]
        finite[:, axis] = (np.asarray(plus) - np.asarray(minus)) / (2.0 * step)

    np.testing.assert_allclose(gradient, finite, rtol=2.0e-8, atol=2.0e-10)


def test_direct_and_cholesky_routes_reproduce_dense_p_shell_eris():
    structure, system = _system([1], [[0.0, 0.0, 0.0]], [23])
    gaussian = phx.operators.quantum.gaussian
    basis = gaussian.GaussianBasisPlan(
        [gaussian.GaussianShellPlan(23, 1, [0.6], [1.0])],
        source_id="p-factorization",
    ).prepare(system)
    eri = gaussian.electron_repulsion_tensor(basis, structure.positions)
    density = np.asarray([[1.0, 0.1, -0.2], [0.1, 0.8, 0.05], [-0.2, 0.05, 0.6]])
    direct = (
        gaussian.DirectJKPlan()
        .prepare(basis, structure.positions)
        .evaluate(structure.positions, density)
    )
    expected_j = np.einsum("cd,abcd->ab", density, np.asarray(eri))
    expected_k = np.einsum("cd,acbd->ab", density, np.asarray(eri))
    factorized = gaussian.PivotedCholeskyERIPlan(1.0e-12).factorize(eri)

    assert bool(direct.successful)
    np.testing.assert_allclose(direct.coulomb, expected_j, rtol=2.0e-11, atol=2.0e-12)
    np.testing.assert_allclose(direct.exchange, expected_k, rtol=2.0e-11, atol=2.0e-12)
    np.testing.assert_allclose(factorized.reconstruct(), eri, rtol=2.0e-11, atol=2.0e-12)


def test_basis_exchange_record_preserves_general_contractions_and_artifact_identity():
    _, system = _system([8], [[0.0, 0.0, 0.0]], [31])
    gaussian = phx.operators.quantum.gaussian
    record = {
        "elements": {
            "8": {
                "electron_shells": [
                    {
                        "angular_momentum": [0, 1],
                        "exponents": ["2.0", "0.5"],
                        "coefficients": [["0.7", "0.3"], ["0.6", "0.4"]],
                    }
                ]
            }
        }
    }
    plan = gaussian.GaussianBasisPlan.from_basis_exchange_record(
        [31],
        [8],
        record,
        source_id="synthetic-basis-record",
        source_artifact_id="basis-artifact",
    )
    basis = plan.prepare(system)

    assert plan.source_artifact_id == "basis-artifact"
    assert tuple(shell.angular_momentum for shell in plan.shells) == (0, 1)
    assert basis.basis_function_count == 4


def test_ecp_core_count_is_bound_to_nuclear_charge():
    _, system = _system([6], [[0.0, 0.0, 0.0]], [41])
    gaussian = phx.operators.quantum.gaussian
    channel = gaussian.ECPChannelPlan(
        0,
        [gaussian.ECPGaussianTerm(2, 1.0, -2.0)],
    )
    ecp = gaussian.EffectiveCorePotentialPlan(
        41,
        2,
        0,
        [channel],
        "synthetic-ecp-artifact",
    ).prepare(system)

    assert ecp.center_index == 0
    assert ecp.plan.core_electron_count == 2
