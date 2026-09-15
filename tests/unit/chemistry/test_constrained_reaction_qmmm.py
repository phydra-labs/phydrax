import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _units():
    return phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()


def _surface(system, evaluator, provider_id):
    def wrapped(positions, _cell):
        energy, forces = evaluator(jnp.asarray(positions))
        return phx.chemistry.PotentialEnergySurfaceEvaluation(
            energy,
            forces,
            None,
            True,
            provider_id=provider_id,
            source_result_id=phx.chemistry.electronic_geometry_id(system, positions),
        )

    return phx.chemistry.CallablePotentialEnergySurface(
        wrapped,
        system.system_id,
        system.units,
        provider_id,
        phx.chemistry.PotentialEnergySurfaceCapabilities(),
    )


def test_distance_constrained_lagrangian_modes_remove_one_internal_degree():
    units = _units()
    positions = np.asarray([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]])
    structure = phx.atomistic.AtomicStructure(
        [8, 1, 1],
        positions,
        [15.999, 1.008, 1.008],
        units.scale,
        particle_ids=[2, 3, 5],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0, 0]
    )
    raw = np.eye(9).reshape((3, 3, 3, 3))
    hessian = phx.chemistry.MolecularHessianResult(
        raw,
        raw,
        0.0,
        1,
        True,
        units,
        ("analytic-hessian",),
        "analytic-hessian-plan",
    )
    constraints = phx.chemistry.MolecularConstraintSetPlan.distances(
        system, [(2, 3)], [0.95]
    )
    result = phx.chemistry.ConstrainedVibrationalAnalysisPlan(
        system,
        constraints,
        stationarity_tolerance=1.0e-8,
    ).evaluate(structure, hessian, np.zeros_like(positions))

    assert bool(result.successful)
    assert result.constraint_rank == 1
    assert result.rigid_mode_rank == 6
    assert result.vibration.internal_mode_count == 2
    assert result.vibration.stationary_point is phx.chemistry.StationaryPointKind.MINIMUM


def test_anchored_cartesian_restraint_removes_only_tangent_rigid_modes():
    units = _units()
    positions = np.asarray([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]])
    structure = phx.atomistic.AtomicStructure(
        [8, 1, 1],
        positions,
        [15.999, 1.008, 1.008],
        units.scale,
        particle_ids=[2, 3, 5],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure,
        units,
        molecule_ids=[0, 0, 0],
    )
    raw = np.eye(9).reshape((3, 3, 3, 3))
    hessian = phx.chemistry.MolecularHessianResult(
        raw,
        raw,
        0.0,
        1,
        True,
        units,
        ("anchored-cartesian-hessian",),
        "anchored-cartesian-hessian-plan",
    )
    constraints = phx.chemistry.MolecularConstraintSetPlan.cartesian(
        system,
        [(2, 0), (2, 1), (2, 2)],
        [0.0, 0.0, 0.0],
    )
    result = phx.chemistry.ConstrainedVibrationalAnalysisPlan(
        system,
        constraints,
        stationarity_tolerance=1.0e-8,
    ).evaluate(structure, hessian, np.zeros_like(positions))

    assert bool(result.successful)
    assert result.constraint_rank == 3
    assert result.rigid_mode_rank == 3
    assert result.vibration.internal_mode_count == 3


def test_named_constraint_constructors_share_zero_residual_convention():
    units = _units()
    positions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    structure = phx.atomistic.AtomicStructure(
        [1, 1, 1, 1],
        positions,
        [1.0, 1.0, 1.0, 1.0],
        units.scale,
        particle_ids=[1, 2, 3, 4],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure,
        units,
        molecule_ids=[0, 0, 0, 0],
    )
    constraints = phx.chemistry.MolecularConstraintSetPlan.combine(
        (
            phx.chemistry.MolecularConstraintSetPlan.cartesian(
                system,
                [(1, 0)],
                [1.0],
            ),
            phx.chemistry.MolecularConstraintSetPlan.angles(
                system,
                [(1, 2, 3)],
                [0.5 * np.pi],
            ),
            phx.chemistry.MolecularConstraintSetPlan.dihedrals(
                system,
                [(1, 2, 3, 4)],
                [0.5 * np.pi],
            ),
        )
    )

    np.testing.assert_allclose(constraints.residual(positions), 0.0, atol=1.0e-12)


def test_climbing_image_neb_finds_double_well_saddle():
    units = _units()
    reactant = phx.atomistic.AtomicStructure(
        [1], [[-1.0, 0.0, 0.0]], [1.0], units.scale, particle_ids=[7]
    )
    product = phx.atomistic.AtomicStructure(
        [1], [[1.0, 0.0, 0.0]], [1.0], units.scale, particle_ids=[7]
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        reactant,
        units,
        molecule_ids=[0],
    )

    def double_well(positions):
        x, y, z = positions[0]
        energy = (x * x - 1.0) ** 2 + 0.5 * y * y + 0.5 * z * z
        forces = jnp.asarray([[-4.0 * x * (x * x - 1.0), -y, -z]])
        return energy, forces

    surface = _surface(system, double_well, "double-well")
    path = phx.chemistry.NudgedElasticBandPlan(
        system,
        surface,
        image_count=5,
        climbing_start=0,
        force_tolerance=1.0e-8,
        maximum_steps=10,
    ).run(reactant, product)

    assert bool(path.successful)
    np.testing.assert_allclose(
        path.transition_state_positions,
        [[0.0, 0.0, 0.0]],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        path.energies[path.climbing_image],
        1.0,
        atol=1.0e-12,
    )
    saddle_vibration = phx.chemistry.VibrationalAnalysisResult(
        [-1.0],
        [-1.0],
        [-100.0],
        [True],
        [[[1.0], [0.0], [0.0]]],
        [1.0],
        external_mode_count=0,
        external_projection_residual=0.0,
        successful=True,
        stationary_point=phx.chemistry.StationaryPointKind.FIRST_ORDER_SADDLE,
        units=units,
        plan_id="double-well-saddle",
    )
    qualification = phx.chemistry.ReactionPathQualificationResult(
        path,
        saddle_vibration,
        [1.0],
        minimum_overlap=0.99,
    )
    assert bool(qualification.successful)
    transition_state = phx.atomistic.AtomicStructure(
        [1],
        path.transition_state_positions,
        [1.0],
        units.scale,
        particle_ids=[7],
    )
    irc = phx.chemistry.IntrinsicReactionCoordinatePlan(
        system,
        surface,
        step_size=0.05,
        maximum_steps=50,
        force_tolerance=1.0e-8,
    ).run(transition_state, [[1.0, 0.0, 0.0]])
    assert bool(irc.successful)
    np.testing.assert_allclose(irc.forward[-1, 0, 0], 1.0, atol=1.0e-12)
    np.testing.assert_allclose(irc.reverse[-1, 0, 0], -1.0, atol=1.0e-12)


def test_neb_masks_inactive_nan_padding_and_accepts_padded_endpoints():
    units = _units()
    reactant = phx.atomistic.AtomicStructure(
        [1, 0],
        [[-1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
        [1.0, 0.0],
        units.scale,
        particle_ids=[71, 73],
        active_mask=[True, False],
    )
    product = phx.atomistic.AtomicStructure(
        [1, 0],
        [[1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
        [1.0, 0.0],
        units.scale,
        particle_ids=[71, 73],
        active_mask=[True, False],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        reactant,
        units,
        molecule_ids=[0, -1],
    )

    def double_well(positions):
        x = positions[0, 0]
        energy = (x * x - 1.0) ** 2
        forces = jnp.zeros_like(positions).at[0, 0].set(-4.0 * x * (x * x - 1.0))
        return energy, forces

    initial_images = np.full((5, 2, 3), np.nan)
    initial_images[:, 0, :] = 0.0
    initial_images[:, 0, 0] = np.linspace(-1.0, 1.0, 5)
    path = phx.chemistry.NudgedElasticBandPlan(
        system,
        _surface(system, double_well, "padded-double-well"),
        image_count=5,
        climbing_start=0,
        force_tolerance=1.0e-8,
        maximum_steps=10,
    ).run(reactant, product, initial_images=initial_images)

    assert bool(path.successful)
    assert np.all(np.isnan(np.asarray(path.images)[:, 1]))
    assert np.all(np.isfinite(np.asarray(path.neb_forces)[:, 0]))


def test_subtractive_qmmm_energy_and_link_force_pullback_close_exactly():
    units = _units()
    structure = phx.atomistic.AtomicStructure(
        [1, 1],
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]],
        [1.0, 1.0],
        units.scale,
        particle_ids=[10, 20],
    )
    full_system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0]
    )
    region = phx.chemistry.QuantumRegionPlan(
        full_system,
        [10],
        boundary_bonds=[(10, 20)],
        total_charge=0,
        spin_multiplicity=1,
    ).prepare()

    def quadratic(scale):
        return lambda positions: (
            0.5 * scale * jnp.sum(positions**2),
            -scale * positions,
        )

    full = _surface(full_system, quadratic(1.0), "full-mm")
    low = _surface(region.region_system, quadratic(1.0), "model-mm")
    high = _surface(region.region_system, quadratic(2.0), "model-qm")
    surface = phx.chemistry.SubtractiveQMMMSurface(region, full, high, low)
    result = surface.evaluate_components(structure.positions)
    region_positions = np.asarray(region.realize(structure.positions))
    expected_energy = 0.5 * np.sum(np.asarray(structure.positions) ** 2) + 0.5 * np.sum(
        region_positions**2
    )

    assert bool(result.successful)
    np.testing.assert_allclose(result.total_energy, expected_energy)
    finite_difference = np.zeros_like(np.asarray(structure.positions))
    step = 1.0e-5
    for atom in range(2):
        for component in range(3):
            shift = np.zeros_like(finite_difference)
            shift[atom, component] = step
            plus = surface.evaluate(np.asarray(structure.positions) + shift).energy
            minus = surface.evaluate(np.asarray(structure.positions) - shift).energy
            finite_difference[atom, component] = -(float(plus) - float(minus)) / (
                2.0 * step
            )
    np.testing.assert_allclose(result.forces, finite_difference, atol=1.0e-6)


def test_electrostatic_embedding_includes_point_charge_force_pullback():
    units = _units()
    structure = phx.atomistic.AtomicStructure(
        [1, 1],
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        [1.0, 1.0],
        units.scale,
        particle_ids=[31, 37],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure,
        units,
        charges=[0.0, 0.5],
        molecule_ids=[0, 0],
    )
    region = phx.chemistry.QuantumRegionPlan(
        system,
        [31],
        boundary_bonds=[(31, 37)],
    ).prepare()

    def zero_classical(positions):
        return jnp.asarray(0.0), jnp.zeros_like(positions)

    classical = _surface(system, zero_classical, "partitioned-mm")

    def embedded(region_positions, embedding):
        energy = 0.5 * jnp.sum(region_positions**2) + 0.5 * jnp.sum(
            embedding.positions**2
        )
        return phx.chemistry.EmbeddedRegionEvaluation(
            energy,
            -region_positions,
            -embedding.positions,
            True,
            "embedded-analytic",
        )

    provider = phx.chemistry.CallableEmbeddedRegionProvider(
        embedded,
        "embedded-analytic",
        region.region_system.system_id,
        units,
    )
    surface = phx.chemistry.ElectrostaticEmbeddingQMMMSurface(
        region,
        classical,
        provider,
        "explicit-zero-duplicate-partition",
    )
    wrong_region_provider = phx.chemistry.CallableEmbeddedRegionProvider(
        embedded,
        "embedded-analytic",
        "same-shape-wrong-region",
        units,
    )
    with pytest.raises(ValueError, match="another quantum-region system"):
        phx.chemistry.ElectrostaticEmbeddingQMMMSurface(
            region,
            classical,
            wrong_region_provider,
            "explicit-zero-duplicate-partition",
        )
    wrong_unit_provider = phx.chemistry.CallableEmbeddedRegionProvider(
        embedded,
        "embedded-analytic",
        region.region_system.system_id,
        phx.atomistic.AtomisticUnitSystem.reduced(),
    )
    with pytest.raises(ValueError, match="unit systems differ"):
        phx.chemistry.ElectrostaticEmbeddingQMMMSurface(
            region,
            classical,
            wrong_unit_provider,
            "explicit-zero-duplicate-partition",
        )
    result = surface.evaluate(structure.positions)
    finite_difference = np.zeros_like(np.asarray(structure.positions))
    step = 1.0e-5
    for atom in range(2):
        for component in range(3):
            shift = np.zeros_like(finite_difference)
            shift[atom, component] = step
            plus = surface.evaluate(np.asarray(structure.positions) + shift).energy
            minus = surface.evaluate(np.asarray(structure.positions) - shift).energy
            finite_difference[atom, component] = -(float(plus) - float(minus)) / (
                2.0 * step
            )

    assert bool(result.successful)
    np.testing.assert_allclose(result.forces, finite_difference, atol=1.0e-6)
