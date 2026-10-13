from itertools import product
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import erfc

import phydrax as phx


periodic = phx.chemistry.periodic


def _cell() -> Any:
    return phx.discretization.PeriodicCell(
        np.asarray([[5.0, 0.0, 0.0], [0.6, 4.8, 0.0], [0.3, 0.2, 5.2]]),
        periodic_axes=(True, True, True),
    )


def _units() -> Any:
    return phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()


def _manifest(source_id: Any) -> Any:
    return periodic.PeriodicProvenanceManifest.for_bytes(
        f"{source_id}-fixture".encode(), source_id, "test-redistributable"
    )


def _pencil(*, generalized: Any = False, reference_electron_count: Any = 2.0) -> Any:
    cell = _cell()
    basis = phx.operators.periodic.PeriodicOrbitalBasisPlan(
        cell,
        ("lower", "upper"),
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
        phx.units.ANGSTROM,
        phx.operators.periodic.PeriodicBlochGauge("lattice"),
    )
    h_blocks = np.asarray([[[[[-1.0], [0.0]]], [[[0.0], [0.8]]]]]).reshape(
        (1, 2, 1, 2, 1)
    )
    h = phx.operators.periodic.periodic_translation_family_from_dense_blocks(
        [[0, 0, 0]],
        h_blocks,
    )
    if generalized:
        s_blocks = np.asarray([[[[[1.0], [0.12]]], [[[0.12], [1.0]]]]]).reshape(
            (1, 2, 1, 2, 1)
        )
        s = phx.operators.periodic.periodic_translation_family_from_dense_blocks(
            [[0, 0, 0]],
            s_blocks,
        )
        pencil = phx.operators.periodic.PeriodicOrbitalPencilPlan(
            basis,
            h.plan,
            h.state,
            s.plan,
            s.state,
            phx.units.ELECTRONVOLT,
        ).prepare()
    else:
        pencil = phx.operators.periodic.PeriodicOrbitalPencilPlan.orthonormal(
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


def test_advanced_periodic_electronic_scenario_1() -> None:
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
    manifest = _manifest("gamma-gdf-integrals")
    factors = phx.operators.quantum.gaussian.FactorizedERITensor(
        [[[0.5]]],
        0.0,
        manifest.source_id,
        "gdf",
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


def test_stationary_periodic_derivatives_close_complete_force_stress_ledger() -> None:
    cell = _cell()

    def hellmann_feynman(positions: Any, vectors: Any) -> Any:
        return 0.5 * jnp.sum(positions**2) + 0.02 * jnp.linalg.det(vectors)

    def pulay(positions: Any, vectors: Any) -> Any:
        return 0.1 * jnp.sum(positions) + 0.0 * jnp.sum(vectors)

    def zero(positions: Any, vectors: Any) -> Any:
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


def test_generic_periodic_provider_binds_method_task_result_and_provenance() -> None:
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
    # ty: ignore[invalid-argument-type]
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

    def evaluate(plan: Any, positions: Any, cell_vectors: Any) -> Any:
        return phx.chemistry.make_electronic_evaluation(
            plan,
            "external-periodic",
            positions,
            -1.0,
            # ty: ignore[invalid-argument-type]
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
    # ty: ignore[no-matching-overload]
    np.testing.assert_allclose(result.band_energies, [[-1.0, 0.5], [-0.8, 0.7]])


def _dense_ewald_components(
    positions: NDArray[np.float64],
    charges: NDArray[np.float64],
    cell: NDArray[np.float64],
    shell: int,
    background: bool,
) -> NDArray[np.float64]:
    """Independent finite-shell double sum, not the prepared Ewald route."""
    alpha = 0.8
    constant = _units().coulomb_constant
    indices = np.asarray(
        list(product(range(-shell, shell + 1), repeat=3)), dtype=np.float64
    )
    displacements = (
        positions[:, None, None, :]
        - positions[None, :, None, :]
        + (indices @ cell)[None, None, :, :]
    )
    included = ~(
        (np.arange(charges.size)[:, None] == np.arange(charges.size)[None, :])[:, :, None]
        & np.all(indices == 0, axis=1)[None, None, :]
    )
    distance = np.sqrt(np.where(included, np.sum(displacements**2, axis=-1), 1.0))
    real = (
        0.5
        * constant
        * np.sum(
            np.where(
                included,
                charges[:, None, None]
                * charges[None, :, None]
                * erfc(alpha * distance)
                / distance,
                0.0,
            )
        )
    )
    volume = abs(np.linalg.det(cell))
    reciprocal_indices = indices[np.any(indices != 0, axis=1)]
    wavevectors = 2.0 * np.pi * reciprocal_indices @ np.linalg.inv(cell).T
    squared = np.sum(wavevectors**2, axis=1)
    structure = np.sum(
        charges[:, None] * np.exp(1.0j * (positions @ wavevectors.T)), axis=0
    )
    reciprocal = (
        2.0
        * np.pi
        * constant
        / volume
        * np.sum(np.exp(-squared / (4.0 * alpha**2)) / squared * abs(structure) ** 2)
    )
    self_energy = -constant * alpha / np.sqrt(np.pi) * np.sum(charges**2)
    uniform = (
        -np.pi * constant * np.sum(charges) ** 2 / (2.0 * alpha**2 * volume)
        if background
        else 0.0
    )
    return np.asarray((real, reciprocal, self_energy, uniform), dtype=np.float64)


@pytest.mark.parametrize("count", [2, 5, 8])
@pytest.mark.parametrize("shell", [1, 2])
@pytest.mark.parametrize("background", [False, True])
def test_ewald_tiles_match_independent_dense_energy_force_and_stress(
    count: int, shell: int, background: bool
) -> None:
    cell = np.asarray(_cell().vectors, dtype=np.float64)
    index = np.arange(count, dtype=np.float64)
    fractional = np.stack(
        (
            (0.13 + 0.19 * index) % 1.0,
            (0.21 + 0.27 * index) % 1.0,
            (0.31 + 0.11 * index) % 1.0,
        ),
        axis=1,
    )
    positions = fractional @ cell
    charges = np.linspace(-0.7, 0.9, count, dtype=np.float64)
    charges -= np.mean(charges)
    if background:
        charges[0] += 0.4
    plan = periodic.PeriodicEwaldPlan(
        _cell(),
        _units(),
        0.8,
        real_shell=shell,
        reciprocal_shell=shell,
        neutrality="uniform-background" if background else "require-neutral",
        pair_tile_size=3,
        translation_tile_size=7,
    )
    result = plan.prepare().evaluate(positions, charges)
    expected = _dense_ewald_components(positions, charges, cell, shell, background)
    np.testing.assert_allclose(
        result.energy_ledger.components, expected, rtol=1.0e-10, atol=1.0e-10
    )
    np.testing.assert_allclose(
        result.energy, np.sum(expected), rtol=1.0e-10, atol=1.0e-10
    )
    step = 2.0e-5
    forces = np.empty_like(positions)
    for site in range(count):
        for axis in range(3):
            perturbation = np.zeros_like(positions)
            perturbation[site, axis] = step
            positive = np.sum(
                _dense_ewald_components(
                    positions + perturbation, charges, cell, shell, background
                )
            )
            negative = np.sum(
                _dense_ewald_components(
                    positions - perturbation, charges, cell, shell, background
                )
            )
            forces[site, axis] = -(positive - negative) / (2.0 * step)
    strain_gradient = np.empty_like(cell)
    for row in range(3):
        for column in range(3):
            strain = np.zeros_like(cell)
            strain[row, column] = step
            positive_deformation = np.eye(3, dtype=np.float64) + strain
            negative_deformation = np.eye(3, dtype=np.float64) - strain
            positive = np.sum(
                _dense_ewald_components(
                    positions @ positive_deformation.T,
                    charges,
                    cell @ positive_deformation.T,
                    shell,
                    background,
                )
            )
            negative = np.sum(
                _dense_ewald_components(
                    positions @ negative_deformation.T,
                    charges,
                    cell @ negative_deformation.T,
                    shell,
                    background,
                )
            )
            strain_gradient[row, column] = (positive - negative) / (2.0 * step)
    stress = (strain_gradient + strain_gradient.T) / (2.0 * abs(np.linalg.det(cell)))
    np.testing.assert_allclose(result.forces, forces, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(result.stress, stress, rtol=1.0e-5, atol=1.0e-5)
    difference = fractional[:, None, :] - fractional[None, :, :]
    difference -= np.rint(difference)
    distances = np.linalg.norm(difference @ cell, axis=-1)
    np.fill_diagonal(distances, np.inf)
    np.testing.assert_allclose(
        result.evidence.minimum_pair_distance, np.min(distances), rtol=1.0e-12
    )
    assert bool(result.successful) and bool(result.evidence.successful)
    assert bool(result.evidence.background_applied) == background
    np.testing.assert_allclose(result.evidence.net_charge, np.sum(charges), atol=1.0e-12)
    np.testing.assert_allclose(
        result.evidence.charge_residual, abs(np.sum(charges)), atol=1.0e-12
    )
    assert result.evidence.neutrality == plan.neutrality
    assert float(result.evidence.energy_closure_residual) <= 1.0e-9
    assert float(result.evidence.force_balance_residual) <= 1.0e-9
    assert float(result.evidence.stress_symmetry_residual) <= 1.0e-9
    assert result.energy_ledger.component_names == (
        "real-space",
        "reciprocal-space",
        "self",
        "uniform-background",
    )
    assert result.plan_id == plan.plan_id


def test_ewald_self_exclusion_and_background_are_required() -> None:
    positions = np.asarray([[0.2, 0.3, 0.4], [1.2, 0.5, 0.6]], dtype=np.float64)
    charges = np.asarray([1.0, 0.4], dtype=np.float64)
    plan = periodic.PeriodicEwaldPlan(
        _cell(),
        _units(),
        0.8,
        real_shell=1,
        reciprocal_shell=1,
        neutrality="uniform-background",
        pair_tile_size=1,
        translation_tile_size=1,
    )
    result = plan.prepare().evaluate(positions, charges)
    expected = _dense_ewald_components(
        positions, charges, np.asarray(_cell().vectors), 1, True
    )
    np.testing.assert_allclose(
        result.energy_ledger.components, expected, atol=1.0e-10, rtol=1.0e-10
    )


def test_ewald_tile_policy_identity_and_minimum_distance_refusal() -> None:
    first = periodic.PeriodicEwaldPlan(
        _cell(), _units(), 0.8, pair_tile_size=2, translation_tile_size=3
    )
    second = periodic.PeriodicEwaldPlan(
        _cell(), _units(), 0.8, pair_tile_size=4, translation_tile_size=3
    )
    third = periodic.PeriodicEwaldPlan(
        _cell(), _units(), 0.8, pair_tile_size=2, translation_tile_size=5
    )
    assert len({first.plan_id, second.plan_id, third.plan_id}) == 3
    assert (
        len(
            {
                first.prepare().prepared_id,
                second.prepare().prepared_id,
                third.prepare().prepared_id,
            }
        )
        == 3
    )
    assert first.pair_tile_size == 2 and first.translation_tile_size == 3
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="tile"):
            periodic.PeriodicEwaldPlan(_cell(), _units(), 0.8, pair_tile_size=invalid)
        with pytest.raises(ValueError, match="tile"):
            periodic.PeriodicEwaldPlan(
                _cell(), _units(), 0.8, translation_tile_size=invalid
            )
    prepared = first.prepare()
    cell = np.asarray(_cell().vectors)
    positions = np.asarray([[0.2, 0.3, 0.4], [1.2, 0.5, 0.6]], dtype=np.float64)
    with pytest.raises(ValueError, match="minimum_distance"):
        prepared.evaluate(np.stack((positions[0], positions[0] + cell[0])), [1.0, -1.0])
    near = np.stack((positions[0], positions[0] + np.asarray([0.5e-10, 0.0, 0.0])))
    with pytest.raises(ValueError, match="minimum_distance"):
        prepared.evaluate(near, [1.0, -1.0])
