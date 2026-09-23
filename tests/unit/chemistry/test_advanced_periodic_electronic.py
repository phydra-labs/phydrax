import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


periodic = phx.chemistry.periodic


def _cell():
    return phx.discretization.PeriodicCell(
        np.asarray([[5.0, 0.0, 0.0], [0.6, 4.8, 0.0], [0.3, 0.2, 5.2]]),
        periodic_axes=(True, True, True),
    )


def _units():
    return phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()


def _manifest(source_id):
    return periodic.PeriodicProvenanceManifest.for_bytes(
        f"{source_id}-fixture".encode(), source_id, "test-redistributable"
    )


def _pencil(*, generalized=False, reference_electron_count=2.0):
    cell = _cell()
    basis = periodic.PeriodicOrbitalBasisPlan(
        cell,
        ("lower", "upper"),
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
        phx.units.ANGSTROM,
        periodic.PeriodicBlochGauge("lattice"),
    )
    h_blocks = np.asarray([[[[[-1.0], [0.0]]], [[[0.0], [0.8]]]]]).reshape(
        (1, 2, 1, 2, 1)
    )
    h = phx.operators.periodic.periodic_translation_family_from_dense_blocks(
        [[0, 0, 0]], h_blocks
    )
    if generalized:
        s_blocks = np.asarray([[[[[1.0], [0.12]]], [[[0.12], [1.0]]]]]).reshape(
            (1, 2, 1, 2, 1)
        )
        s = phx.operators.periodic.periodic_translation_family_from_dense_blocks(
            [[0, 0, 0]], s_blocks
        )
        pencil = periodic.PeriodicOrbitalPencilPlan(
            basis,
            h.plan,
            h.state,
            s.plan,
            s.state,
            phx.units.ELECTRONVOLT,
        ).prepare()
    else:
        pencil = periodic.PeriodicOrbitalPencilPlan.orthonormal(
            basis, h.plan, h.state, phx.units.ELECTRONVOLT
        ).prepare()
    mean_field = periodic.PeriodicHubbardMeanFieldPlan(
        basis,
        [0.0, 0.0],
        [reference_electron_count, 0.0],
        0.0,
        phx.units.ELECTRONVOLT,
    )
    mesh = phx.discretization.ReciprocalMeshPlan.monkhorst_pack(cell, (2, 1, 1))
    return mesh, pencil, mean_field


def test_prepared_ewald_enforces_neutrality_and_reports_background_evidence():
    cell = _cell()
    positions = np.asarray([[0.2, 0.3, 0.4], [1.2, 0.3, 0.4]])
    neutral = periodic.PeriodicEwaldPlan(
        cell,
        _units(),
        0.8,
        real_shell=2,
        reciprocal_shell=2,
    ).prepare()
    result = neutral.evaluate(positions, [1.0, -1.0])

    assert bool(result.successful)
    assert not bool(result.evidence.background_applied)
    assert result.energy_unit == _units().scale.energy_unit
    assert result.force_unit == _units().scale.force_unit
    np.testing.assert_allclose(np.sum(result.forces, axis=0), 0.0, atol=2.0e-11)
    np.testing.assert_allclose(result.energy_ledger.closure_residual, 0.0, atol=1.0e-12)
    with pytest.raises(ValueError, match="uniform-background"):
        neutral.evaluate(positions, [1.0, 0.0])

    background = (
        periodic.PeriodicEwaldPlan(
            cell,
            _units(),
            0.8,
            real_shell=2,
            reciprocal_shell=2,
            neutrality="uniform-background",
        )
        .prepare()
        .evaluate(positions, [1.0, 0.0])
    )
    assert bool(background.successful)
    assert bool(background.evidence.background_applied)
    assert background.energy_ledger.component_names[-1] == "uniform-background"
    assert float(background.energy_ledger.components[-1]) < 0.0


def test_governed_gth_local_and_nonlocal_components_retain_units_and_source():
    channel = periodic.GTHProjectorChannel(0, 0.4, [[0.25, 0.0], [0.0, 0.1]])
    manifest = _manifest("gth-fixture")
    plan = periodic.GTHPseudopotentialPlan(
        2.0,
        0.3,
        [0.1, -0.02, 0.0, 0.0],
        phx.units.ELECTRONVOLT,
        phx.units.ANGSTROM,
        manifest,
        channels=(channel,),
    )
    local = plan.local_reciprocal([0.0, 1.0, 4.0])
    nonlocal_ = plan.nonlocal_energy(([[1.0, 0.0], [0.0, 1.0]],))

    assert bool(local.successful) and bool(nonlocal_.successful)
    assert local.evidence.source_manifest_id == manifest.manifest_id
    assert nonlocal_.energy_unit == phx.units.ELECTRONVOLT
    np.testing.assert_allclose(nonlocal_.energy, 0.35, atol=1.0e-12)
    assert np.all(np.isfinite(np.asarray(local.values)))


def test_spin_scf_closes_restricted_generalized_insulator_evidence():
    mesh, pencil, mean_field = _pencil(generalized=True)
    result = periodic.SpinPeriodicSCFPlan(
        mesh,
        phx.chemistry.PeriodicElectronicSectorPlan(2.0),
        pencil,
        mean_field,
        reference_kind="restricted",
    ).evaluate()

    assert bool(result.successful)
    assert result.reference_kind == "restricted"
    np.testing.assert_allclose(result.occupations[:, :, 0], 1.0, atol=1.0e-12)
    np.testing.assert_allclose(result.evidence.electron_count_residual, 0.0, atol=1.0e-12)
    assert float(result.evidence.commutator_residual) <= 1.0e-9
    assert float(result.evidence.eigenpair_residual) <= 1.0e-9
    np.testing.assert_allclose(result.energy_ledger.closure_residual, 0.0, atol=1.0e-12)


def test_spin_scf_closes_collinear_finite_temperature_metal_counts():
    mesh, pencil, mean_field = _pencil(reference_electron_count=1.0)
    result = periodic.SpinPeriodicSCFPlan(
        mesh,
        phx.chemistry.PeriodicElectronicSectorPlan(1.0, spin_magnetization=0.4),
        pencil,
        mean_field,
        reference_kind="collinear",
        reference_spin_populations=[[0.7, 0.0], [0.3, 0.0]],
        smearing_energy=0.15,
    ).evaluate()
    weighted = np.sum(
        np.asarray(mesh.weights)[None, :, None] * np.asarray(result.occupations),
        axis=(1, 2),
    )

    assert bool(result.successful)
    np.testing.assert_allclose(weighted, [0.7, 0.3], atol=1.0e-10)
    assert float(result.entropy) > 0.0
    assert float(result.free_energy) < float(result.energy)
    assert float(result.evidence.free_energy_residual) <= 1.0e-12


def test_governed_gamma_gdf_is_production_while_local_gth_fftdf_is_candidate():
    manifest = _manifest("gamma-gdf-integrals")
    factors = phx.operators.quantum.gaussian.FactorizedERITensor(
        [[[0.5]]], 0.0, manifest.source_id, "gdf"
    )
    gdf = periodic.GammaGDFPlan(
        [[-1.0]],
        [[1.0]],
        factors,
        2.0,
        0.0,
        phx.units.ELECTRONVOLT,
        manifest,
    ).evaluate()

    gth_manifest = _manifest("local-gth-candidate")
    gth = periodic.GTHPseudopotentialPlan(
        2.0,
        0.3,
        [0.1, -0.02, 0.0, 0.0],
        phx.units.ELECTRONVOLT,
        phx.units.ANGSTROM,
        gth_manifest,
    )
    fftdf = periodic.GammaFFTDFPlan(
        _cell(),
        (2, 2, 2),
        [[0.0, 0.0, 0.0]],
        (gth,),
        2.0,
        0.0,
        phx.units.ELECTRONVOLT,
        phx.units.ANGSTROM,
        smearing_energy=0.1,
        convergence_tolerance=1.0e-7,
        maximum_iterations=300,
        damping=0.5,
    ).evaluate()

    assert bool(gdf.successful)
    assert gdf.classification == "production-supplied-gdf-rhf"
    assert gdf.source_manifest_ids == (manifest.manifest_id,)
    np.testing.assert_allclose(gdf.evidence.electron_count_residual, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(gdf.evidence.factorization_residual, 0.0, atol=1.0e-12)
    assert fftdf.classification == "candidate-local-gth-lda-x"
    assert fftdf.source_manifest_ids == (gth_manifest.manifest_id,)
    np.testing.assert_allclose(
        np.sum(fftdf.density) * abs(np.linalg.det(np.asarray(_cell().vectors))) / 8.0,
        2.0,
        atol=2.0e-6,
    )


def test_stationary_periodic_derivatives_close_complete_force_stress_ledger():
    cell = _cell()

    def hellmann_feynman(positions, vectors):
        return 0.5 * jnp.sum(positions**2) + 0.02 * jnp.linalg.det(vectors)

    def pulay(positions, vectors):
        return 0.1 * jnp.sum(positions) + 0.0 * jnp.sum(vectors)

    def zero(positions, vectors):
        return 0.0 * (jnp.sum(positions) + jnp.sum(vectors))

    components = (
        periodic.PeriodicStationaryEnergyComponent(
            "orbital", "hellmann-feynman", hellmann_feynman, "analytic-orbital"
        ),
        periodic.PeriodicStationaryEnergyComponent(
            "metric", "pulay", pulay, "analytic-pulay"
        ),
        periodic.PeriodicStationaryEnergyComponent(
            "smearing", "entropy", zero, "zero-temperature-entropy"
        ),
        periodic.PeriodicStationaryEnergyComponent(
            "projectors", "nonlocal", zero, "local-model-no-projectors"
        ),
        periodic.PeriodicStationaryEnergyComponent(
            "ions", "ionic", zero, "analytic-zero-ion-term"
        ),
    )
    positions = np.asarray([[0.2, 0.3, 0.4], [0.8, 0.1, 0.5]])
    result = periodic.PeriodicStationaryDerivativePlan(
        cell,
        components,
        phx.units.ELECTRONVOLT,
        phx.units.ANGSTROM,
        energy_kind="free-energy",
        directional_tolerance=2.0e-8,
    ).evaluate(
        positions,
        0.0,
        [[1.0, 0.0, 0.0], [-0.5, 0.2, 0.0]],
        [[0.2, 0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.0, 0.05]],
    )

    assert bool(result.successful)
    assert set(result.ledger.component_roles) == {
        "hellmann-feynman",
        "pulay",
        "entropy",
        "nonlocal",
        "ionic",
    }
    assert float(result.ledger.force_closure_residual) <= 1.0e-12
    assert float(result.evidence.force_directional_residual) <= 2.0e-8
    assert float(result.evidence.stress_directional_residual) <= 2.0e-8


def test_generic_periodic_provider_binds_method_task_result_and_provenance():
    units = _units()
    cell = _cell()
    system = phx.atomistic.AtomisticSystemPlan(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 1], dtype=np.int32),
        np.asarray([1.008, 1.008]),
        units,
        molecule_ids=np.zeros(2, dtype=np.int32),
        cell=cell,
    )
    method = phx.chemistry.ExternalElectronicMethodPlan(
        "provider-scalar-relativistic",
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
        definition_ids=("scalar-relativistic", "gth-fixture"),
    )
    model = phx.chemistry.ElectronicModelChemistryPlan(
        method,
        basis=phx.chemistry.BasisSetReference("periodic-fixture", "test"),
    )
    task = phx.chemistry.BandStructureTaskPlan([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    calculation = phx.chemistry.ElectronicCalculationPlan(
        system,
        phx.chemistry.PeriodicElectronicSectorPlan(2.0),
        model,
        task,
    )
    provenance = _manifest("external-periodic-provider")
    capabilities = phx.chemistry.ElectronicProviderCapabilities(
        phx.chemistry.ElectronicTheoryCapabilities(
            (phx.chemistry.ElectronicMethodFamily.CUSTOM_EXTERNAL,),
            (phx.chemistry.ElectronicReferenceKind.RESTRICTED,),
        ),
        phx.chemistry.ElectronicGeometryCapabilities(
            finite=False, periodic_ranks=(3,), variable_cell=True
        ),
        phx.chemistry.ElectronicObservableCapabilities(
            (task.task_kind,),
            (phx.chemistry.ElectronicProperty.BAND_ENERGIES,),
            derivative_orders=(0,),
        ),
    )

    def evaluate(plan, positions, cell_vectors):
        return phx.chemistry.make_electronic_evaluation(
            plan,
            "external-periodic",
            positions,
            -1.0,
            band_energies=[[-1.0, 0.5], [-0.8, 0.7]],
            cell_vectors=cell_vectors,
            artifact_ids=(provenance.manifest_id,),
            convergence=phx.chemistry.ElectronicConvergenceEvidence(
                True, energy_residual=0.0, density_residual=0.0
            ),
        )

    result = (
        phx.chemistry.CallableElectronicProvider(
            evaluate, "external-periodic", capabilities
        )
        .prepare(calculation)
        .evaluate(np.zeros((2, 3)), np.asarray(cell.vectors))
    )

    assert isinstance(result, phx.chemistry.ElectronicPeriodicEvaluation)
    assert bool(result.successful)
    assert result.header.provider_id == "external-periodic"
    assert result.header.task_id == task.task_id
    assert result.header.artifact_ids == (provenance.manifest_id,)
    np.testing.assert_allclose(result.band_energies, [[-1.0, 0.5], [-0.8, 0.7]])
