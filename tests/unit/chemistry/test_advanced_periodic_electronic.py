import jax.numpy as jnp
import numpy as np

import phydrax as phx


periodic = phx.chemistry.periodic


def _cell():
    return phx.discretization.PeriodicCell(
        5.0 * np.eye(3), periodic_axes=(True, True, True)
    )


def test_ewald_gth_and_analytic_periodic_derivatives_are_finite_and_conservative():
    cell = jnp.asarray([[5.0, 0.0, 0.0], [1.0, 4.8, 0.0], [0.5, 0.2, 5.2]])
    positions = jnp.asarray([[0.2, 0.3, 0.4], [0.7, 0.3, 0.4]]) @ cell
    ewald_plan = periodic.PeriodicEwaldPlan(0.8, real_shell=3, reciprocal_shell=3)
    ewald = ewald_plan.evaluate(positions, [1.0, -1.0], cell)
    shifted = ewald_plan.evaluate(positions.at[1].add(cell[0]), [1.0, -1.0], cell)
    gth = periodic.GTHPseudopotentialPlan(
        1.0, 0.3, [0.1, -0.02, 0.0, 0.0]
    ).local_reciprocal([0.0, 1.0, 4.0])

    def pair_energy(coordinates, vectors):
        fractional = jnp.linalg.solve(vectors.T, (coordinates[1] - coordinates[0]))
        displacement = fractional @ vectors
        return 0.5 * jnp.sum(displacement**2)

    derivative = periodic.PeriodicEnergyDerivativePlan(
        pair_energy, "periodic-pair"
    ).evaluate(positions, cell)

    assert bool(ewald.successful) and bool(derivative.successful)
    np.testing.assert_allclose(np.sum(ewald.forces, axis=0), 0.0, atol=2.0e-12)
    np.testing.assert_allclose(shifted.energy, ewald.energy, atol=2.0e-11)
    np.testing.assert_allclose(shifted.forces, ewald.forces, atol=2.0e-11)
    np.testing.assert_allclose(np.sum(derivative.forces, axis=0), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(ewald.stress, np.asarray(ewald.stress).T, atol=1.0e-12)
    assert np.all(np.isfinite(np.asarray(gth)))


def test_gamma_fftdf_gdf_and_spin_kpoint_scf_close_electron_counts():
    fftdf = periodic.GammaFFTDFPlan(
        4.0 * np.eye(3),
        np.zeros((2, 1, 1)),
        2.0,
        0.0,
        phx.units.HARTREE,
        convergence_tolerance=1.0e-8,
    ).evaluate()
    factors = phx.operators.quantum.gaussian.FactorizedERITensor(
        [[[0.5]]], 0.0, "gamma-test", "gdf"
    )
    gdf = periodic.GammaGDFPlan(
        [[-1.0]],
        [[1.0]],
        factors,
        2.0,
        0.0,
        phx.units.HARTREE,
    ).evaluate()
    model = periodic.PeriodicAOModelPlan(
        [[0, 0, 0]],
        [[[-1.0]]],
        [[[1.0]]],
        [0.2],
        [1.0],
        0.0,
        phx.units.HARTREE,
    )
    mesh = periodic.KPointMeshPlan.monkhorst_pack((2, 1, 1))
    spin = periodic.SpinPeriodicSCFPlan(
        _cell(),
        mesh,
        periodic.PeriodicElectronicSectorPlan(1.0, spin_magnetization=1.0),
        model,
        smearing_energy=0.02,
    ).evaluate()

    assert bool(fftdf.successful) and bool(gdf.successful) and bool(spin.successful)
    grid_weight = 64.0 / 2.0
    np.testing.assert_allclose(np.sum(fftdf.density) * grid_weight, 2.0, atol=2.0e-8)
    np.testing.assert_allclose(gdf.density[0, 0], 2.0, atol=1.0e-10)
    weighted_count = np.sum(
        np.asarray(mesh.weights)[None, :, None] * np.asarray(spin.occupations),
        axis=(1, 2),
    )
    np.testing.assert_allclose(weighted_count, [1.0, 0.0], atol=2.0e-10)


def test_bands_berry_wannier_defects_and_reference_provider_preserve_identities():
    model = periodic.PeriodicAOModelPlan(
        [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
        [[[-0.2]], [[-1.0]], [[-0.2]]],
        [[[0.0]], [[1.0]], [[0.0]]],
        [0.0],
        [2.0],
        0.0,
        phx.units.HARTREE,
    )
    bands = periodic.BandStructurePlan(
        model,
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]],
        2.0 * np.pi / 5.0 * np.eye(3),
    ).evaluate()
    berry = periodic.berry_wannier_from_neighbor_overlaps(
        np.exp(0.2j) * np.ones((4, 1, 1))
    )
    defect = periodic.defect_formation_energy(
        -10.0, -9.0, [1.0], [-0.5], 1.0, 0.2, 0.1, 0.05
    )
    task = periodic.PeriodicElectronicTaskPlan(
        "external-hybrid", ("energy", "bands"), "reference-v1"
    )

    def evaluate(request, positions, cell):
        del positions, cell
        return periodic.PeriodicElectronicReferenceResult(
            -1.0,
            0.0,
            True,
            phx.units.HARTREE,
            "external-periodic",
            request.task_id,
            band_energies=bands.energies,
        )

    provider = periodic.CallablePeriodicReferenceProvider(
        evaluate, "external-periodic", "reference-v1"
    )
    reference = provider.evaluate(task, np.zeros((1, 3)), 5.0 * np.eye(3))

    assert bool(bands.successful) and bool(berry.successful)
    assert bool(defect.successful) and bool(reference.successful)
    np.testing.assert_allclose(berry.berry_phase, 0.8, atol=1.0e-13)
    np.testing.assert_allclose(defect.formation_energy, -0.15, atol=1.0e-14)
    assert reference.task_id == task.task_id


def test_candidate_qualification_campaigns_and_support_remain_unreleased_and_bounded():
    campaigns = phx.chemistry.candidate_chemistry_qualification_campaigns()
    registry = phx.chemistry.candidate_complete_chemistry_support_registry()

    assert len(campaigns) == 4
    assert all(
        type(campaign).from_record(campaign.to_record()).campaign_id
        == campaign.campaign_id
        for campaign in campaigns
    )
    assert len(registry.support_tuples) == 8
    assert all(
        support.capability.startswith("chemistry.candidate.")
        for support in registry.support_tuples
    )
