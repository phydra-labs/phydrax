from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic._relaxation import AtomisticRelaxationStatus
from phydrax.chemistry.reaction import (
    nudged_elastic_band_forces,
    ReactionPathBranch,
    resolve_reaction_path_branch,
)


def _units() -> Any:
    return phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()


def _surface(system: Any, evaluator: Any, provider_id: Any) -> Any:
    def wrapped(positions: Any, _cell: Any) -> Any:
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


def test_constrained_reaction_qmmm_scenario_1() -> None:
    units = _units()
    positions = np.asarray([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]])
    structure = phx.atomistic.AtomicStructure(
        np.asarray([8, 1, 1]),
        positions,
        np.asarray([15.999, 1.008, 1.008]),
        units.scale,
        particle_ids=np.asarray([2, 3, 5]),
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
        system.system_id,
        structure.structure_id,
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
    units = _units()
    positions = np.asarray([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]])
    structure = phx.atomistic.AtomicStructure(
        np.asarray([8, 1, 1]),
        positions,
        np.asarray([15.999, 1.008, 1.008]),
        units.scale,
        particle_ids=np.asarray([2, 3, 5]),
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
        system.system_id,
        structure.structure_id,
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
        np.asarray([1, 1, 1, 1]),
        positions,
        np.asarray([1.0, 1.0, 1.0, 1.0]),
        units.scale,
        particle_ids=np.asarray([1, 2, 3, 4]),
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


def test_climbing_image_neb_finds_double_well_saddle() -> None:
    units = _units()
    reactant = phx.atomistic.AtomicStructure(
        np.asarray([1]),
        np.asarray([[-1.0, 0.0, 0.0]]),
        np.asarray([1.0]),
        units.scale,
        particle_ids=np.asarray([7]),
    )
    product = phx.atomistic.AtomicStructure(
        np.asarray([1]),
        np.asarray([[1.0, 0.0, 0.0]]),
        np.asarray([1.0]),
        units.scale,
        particle_ids=np.asarray([7]),
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        reactant,
        units,
        molecule_ids=[0],
    )

    def double_well(positions: Any) -> Any:
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
        np.asarray([-1.0]),
        np.asarray([-1.0]),
        np.asarray([-100.0]),
        np.asarray([True]),
        np.asarray([[[1.0], [0.0], [0.0]]]),
        np.asarray([1.0]),
        external_mode_count=0,
        external_projection_residual=0.0,
        successful=True,
        stationary_point=phx.chemistry.StationaryPointKind.FIRST_ORDER_SADDLE,
        units=units,
        source_system_id=system.system_id,
        source_geometry_id=reactant.structure_id,
        plan_id="double-well-saddle",
    )
    qualification = phx.chemistry.ReactionPathQualificationResult(
        path,
        saddle_vibration,
        np.asarray([1.0]),
        minimum_overlap=0.99,
    )
    assert bool(qualification.successful)
    transition_state = phx.atomistic.AtomicStructure(
        np.asarray([1]),
        path.transition_state_positions,
        np.asarray([1.0]),
        units.scale,
        particle_ids=np.asarray([7]),
    )
    irc = phx.chemistry.IntrinsicReactionCoordinatePlan(
        system,
        surface,
        step_size=0.05,
        maximum_steps=50,
        force_tolerance=1.0e-8,
    ).run(transition_state, np.asarray([[1.0, 0.0, 0.0]]))
    assert bool(irc.successful)
    np.testing.assert_allclose(irc.forward[-1, 0, 0], 1.0, atol=1.0e-12)
    np.testing.assert_allclose(irc.reverse[-1, 0, 0], -1.0, atol=1.0e-12)


def test_neb_masks_inactive_nan_padding_and_accepts_padded_endpoints() -> None:
    units = _units()
    reactant = phx.atomistic.AtomicStructure(
        np.asarray([1, 0]),
        np.asarray([[-1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]]),
        np.asarray([1.0, 0.0]),
        units.scale,
        particle_ids=np.asarray([71, 73]),
        active_mask=np.asarray([True, False]),
    )
    product = phx.atomistic.AtomicStructure(
        np.asarray([1, 0]),
        np.asarray([[1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]]),
        np.asarray([1.0, 0.0]),
        units.scale,
        particle_ids=np.asarray([71, 73]),
        active_mask=np.asarray([True, False]),
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        reactant,
        units,
        molecule_ids=[0, -1],
    )

    def double_well(positions: Any) -> Any:
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


def _curved_saddle(positions: Any) -> Any:
    x, y, z = positions[0]
    valley = y - 0.5 * x * x
    energy = (x * x - 1.0) ** 2 + 2.0 * valley * valley + 0.5 * z * z
    forces = jnp.asarray(
        [[-(4.0 * x * (x * x - 1.0) - 4.0 * x * valley), -4.0 * valley, -z]]
    )
    return energy, forces


def _hydrogen(units: Any, position: Any, *, cell: Any = None) -> Any:
    return phx.atomistic.AtomicStructure(
        np.asarray([1]),
        np.asarray([position]),
        np.asarray([1.0]),
        units.scale,
        particle_ids=np.asarray([7]),
        cell=cell,
        periodic_axes=None if cell is None else np.asarray([True, True, True]),
    )


def test_climbing_neb_finds_an_independent_curved_two_dimensional_saddle() -> None:
    units = _units()
    reactant = _hydrogen(units, [-1.0, 0.5, 0.0])
    product = _hydrogen(units, [1.0, 0.5, 0.0])
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        reactant, units, molecule_ids=[0]
    )
    plan = phx.chemistry.NudgedElasticBandPlan(
        system,
        _surface(system, _curved_saddle, "curved-saddle"),
        image_count=7,
        climbing_start=0,
        force_tolerance=1.0e-7,
        maximum_steps=4000,
    )
    path = plan.run(reactant, product)

    assert bool(path.successful)
    assert path.climbing_active
    assert int(path.status) == int(AtomisticRelaxationStatus.CONVERGED)
    np.testing.assert_allclose(
        path.transition_state_positions, [[0.0, 0.0, 0.0]], atol=1e-6
    )
    np.testing.assert_allclose(path.energies[path.climbing_image], 1.0, atol=1e-10)
    # One accepted iterate: reported energies/forces are those of the images.
    for image, energy in zip(path.images, path.energies, strict=True):
        expected_energy, _ = _curved_saddle(jnp.asarray(image))
        np.testing.assert_allclose(energy, expected_energy, atol=1e-12)
    for image in range(1, path.images.shape[0] - 1):
        _, expected_force = _curved_saddle(jnp.asarray(path.images[image]))
        np.testing.assert_allclose(path.forces[image], expected_force, atol=1e-12)
    projection = nudged_elastic_band_forces(
        path.images,
        path.energies,
        path.forces,
        np.asarray([True]),
        spring_constant=plan.spring_constant,
        climbing=path.climbing_active,
    )
    np.testing.assert_allclose(projection.forces, path.neb_forces, atol=1e-14)
    assert float(np.max(np.asarray(path.image_residual))) <= 1.0e-7


def test_neb_climbing_transition_and_band_atomic_failure() -> None:
    units = _units()
    reactant = _hydrogen(units, [-1.0, 0.5, 0.0])
    product = _hydrogen(units, [1.0, 0.5, 0.0])
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        reactant, units, molecule_ids=[0]
    )
    waiting = phx.chemistry.NudgedElasticBandPlan(
        system,
        _surface(system, _curved_saddle, "curved-saddle-late-climb"),
        image_count=5,
        climbing_start=1000,
        force_tolerance=1.0e-7,
        maximum_steps=20,
    ).run(reactant, product)
    assert not bool(waiting.successful)
    assert not waiting.climbing_active
    assert int(waiting.status) == int(AtomisticRelaxationStatus.EVALUATION_LIMIT)

    def failing(positions: Any, _cell: Any) -> Any:
        energy, forces = _curved_saddle(jnp.asarray(positions))
        ok = bool(abs(float(positions[0][0])) > 0.2)
        return phx.chemistry.PotentialEnergySurfaceEvaluation(
            energy if ok else jnp.nan,
            forces if ok else jnp.full_like(forces, jnp.nan),
            None,
            ok,
            provider_id="failing-saddle",
            source_result_id=phx.chemistry.electronic_geometry_id(system, positions),
        )

    surface = phx.chemistry.CallablePotentialEnergySurface(
        failing,
        system.system_id,
        system.units,
        "failing-saddle",
        phx.chemistry.PotentialEnergySurfaceCapabilities(),
    )
    failed = phx.chemistry.NudgedElasticBandPlan(
        system, surface, image_count=5, climbing_start=0, maximum_steps=20
    ).run(reactant, product)
    assert not bool(failed.successful)
    assert int(failed.status) == int(AtomisticRelaxationStatus.SCIENTIFIC_FAILURE)
    # The whole band fails together on its first evaluation: 2 endpoints + 3 images.
    assert int(failed.provider_evaluations) == 5


def test_periodic_neb_uses_a_declared_winding_branch() -> None:
    units = _units()
    length = 4.0
    cell = length * np.eye(3)
    reactant = _hydrogen(units, [0.5, 0.0, 0.0], cell=cell)
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        reactant, units, molecule_ids=[0]
    )
    shortest, certificate = resolve_reaction_path_branch(
        system,
        [[0.5, 0.0, 0.0]],
        [[3.5, 0.0, 0.0]],
        policy=ReactionPathBranch.SHORTEST_IMAGE,
    )
    np.testing.assert_allclose(shortest, [[-0.5, 0.0, 0.0]], atol=1e-12)
    np.testing.assert_array_equal(certificate.image_shifts, [[-1, 0, 0]])
    assert float(certificate.margins[0]) > 1.0
    explicit, _ = resolve_reaction_path_branch(
        system, [[0.5, 0.0, 0.0]], [[3.5, 0.0, 0.0]]
    )
    np.testing.assert_allclose(explicit, [[3.5, 0.0, 0.0]])
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_reaction_path_branch(
            system,
            [[0.5, 0.0, 0.0]],
            [[2.5, 0.0, 0.0]],
            policy=ReactionPathBranch.SHORTEST_IMAGE,
        )

    def lattice(positions: Any) -> Any:
        x, y, z = positions[0]
        phase = 2.0 * np.pi * x / length
        energy = -jnp.cos(phase) + 0.5 * y * y + 0.5 * z * z
        forces = jnp.asarray([[-(2.0 * np.pi / length) * jnp.sin(phase), -y, -z]])
        return energy, forces

    origin = _hydrogen(units, [0.0, 0.0, 0.0], cell=cell)
    lattice_system = phx.atomistic.AtomisticSystemPlan.from_structure(
        origin, units, molecule_ids=[0]
    )
    plan = phx.chemistry.NudgedElasticBandPlan(
        lattice_system,
        _surface(lattice_system, lattice, "periodic-cosine"),
        image_count=5,
        climbing_start=0,
        force_tolerance=1.0e-9,
        maximum_steps=50,
    )
    # Identical wrapped endpoints: only the declared winding defines a path.
    path = plan.run(origin, origin, product_image_shifts=[[1, 0, 0]])
    assert bool(path.successful)
    np.testing.assert_allclose(path.images[-1], [[length, 0.0, 0.0]])
    np.testing.assert_allclose(
        path.transition_state_positions, [[2.0, 0.0, 0.0]], atol=1e-12
    )
    np.testing.assert_allclose(path.energies[path.climbing_image], 1.0, atol=1e-12)
    degenerate = plan.run(origin, origin)
    assert not bool(degenerate.successful)
    assert int(degenerate.status) == int(AtomisticRelaxationStatus.SCIENTIFIC_FAILURE)


def test_ambiguous_atom_correspondence_is_refused() -> None:
    units = _units()

    def structure(positions: Any, ids: Any, numbers: Any) -> Any:
        return phx.atomistic.AtomicStructure(
            numbers,
            positions,
            np.asarray([1.0, 4.0]),
            units.scale,
            particle_ids=ids,
        )

    reactant = structure([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [1, 2], [1, 2])
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        reactant, units, molecule_ids=[0, 0]
    )

    def pair(positions: Any) -> Any:
        return 0.5 * jnp.sum(positions * positions), -positions

    plan = phx.chemistry.NudgedElasticBandPlan(
        system, _surface(system, pair, "pair"), image_count=3, maximum_steps=2
    )
    relabeled = structure([[2.5, 0.0, 0.0], [0.5, 0.0, 0.0]], [2, 1], [2, 1])
    with pytest.raises(ValueError, match="stable particle order"):
        plan.run(reactant, relabeled)
    with pytest.raises(ValueError, match="permutation"):
        plan.run(reactant, relabeled, correspondence=[0, 0])
    with pytest.raises(ValueError, match="different elements"):
        plan.run(reactant, relabeled, correspondence=[0, 1])
    mapped = plan.run(reactant, relabeled, correspondence=[1, 0])
    np.testing.assert_allclose(mapped.images[-1], [[0.5, 0.0, 0.0], [2.5, 0.0, 0.0]])


def test_subtractive_qmmm_energy_and_link_force_pullback_close_exactly() -> None:
    units = _units()
    structure = phx.atomistic.AtomicStructure(
        np.asarray([1, 1]),
        np.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        np.asarray([1.0, 1.0]),
        units.scale,
        particle_ids=np.asarray([10, 20]),
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

    def quadratic(scale: Any) -> Any:
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


def test_electrostatic_embedding_includes_point_charge_force_pullback() -> None:
    units = _units()
    structure = phx.atomistic.AtomicStructure(
        np.asarray([1, 1]),
        np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        np.asarray([1.0, 1.0]),
        units.scale,
        particle_ids=np.asarray([31, 37]),
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

    def zero_classical(positions: Any) -> Any:
        return jnp.asarray(0.0), jnp.zeros_like(positions)

    classical = _surface(system, zero_classical, "partitioned-mm")

    def embedded(region_positions: Any, embedding: Any) -> Any:
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
