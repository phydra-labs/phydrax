from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic._alchemical import (
    AlchemicalControlKind,
    AlchemicalControlSchedulePlan,
    AlchemicalInteractionPartitionPlan,
    ControlledHamiltonianPlan,
)
from phydrax.atomistic._force_field import (
    AtomisticForceFieldPlan,
    AtomisticForceFieldProvenance,
    AtomisticNonbondedPolicy,
)
from phydrax.atomistic._potential import AtomisticStressConvention


_TRICLINIC = np.asarray([[2.6, 0.0, 0.0], [0.5, 2.4, 0.0], [0.3, 0.2, 2.7]])


def _learned_image_case(
    atomic_numbers: list[int], vectors: np.ndarray, cutoff: float
) -> tuple[Any, Any, Any]:
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    cell = phx.discretization.PeriodicCell(vectors)
    count = len(atomic_numbers)
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        list(range(count)),
        # ty: ignore[invalid-argument-type]
        atomic_numbers,
        # ty: ignore[invalid-argument-type]
        [1.0] * count,
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=atomic_numbers,
        cell=cell,
    ).prepare()
    model = phx.nn.atomistic.PaiNNPotential(
        units.scale,
        cutoff=cutoff,
        feature_count=4,
        interaction_count=1,
        radial_basis_count=3,
        key=jr.key(3),
    )
    program = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LearnedGraphPotentialTerm(model, allow_periodic=True)]
    ).prepare(
        system,
        graph_execution=phx.atomistic.AtomisticGraphExecutionPlan(
            128, backend="particle"
        ),
    )
    capacity = phx.discretization.ParticleImageCapacity(
        maximum_particles_per_cell=count,
        maximum_edges=1024,
        maximum_degree=512,
        maximum_images=512,
    )
    neighborhood = phx.discretization.DenseParticleImageNeighborhoodPlan(
        cutoff, cell, capacity, maximum_dense_routes=8192
    ).prepare(system.particles)
    return system, program, neighborhood


def _independent_strain_derivative(
    energy: Any, positions: Any, cell: Any, step: float = 1.0e-5
) -> np.ndarray:
    """Central differences of independently deformed Cartesian geometry."""
    origin = np.asarray(cell.origin)
    vectors = np.asarray(cell.vectors)
    result = np.zeros((3, 3))
    for row in range(3):
        for column in range(3):
            values = []
            for sign in (1.0, -1.0):
                deformation = np.eye(3)
                deformation[row, column] += sign * step
                deformed_positions = origin + (np.asarray(positions) - origin) @ (
                    deformation.T
                )
                deformed_vectors = vectors @ deformation.T
                values.append(
                    energy(jnp.asarray(deformed_positions), jnp.asarray(deformed_vectors))
                )
            result[row, column] = (values[0] - values[1]) / (2.0 * step)
    return result


def test_learned_image_stress_matches_independent_triclinic_strain() -> None:
    system, program, neighborhood = _learned_image_case([1, 6], _TRICLINIC, 3.0)
    cell = system.cell
    positions = cell.cartesian(jnp.asarray([[0.1, 0.2, 0.3], [0.6, 0.5, 0.45]]))
    species = system.plan.atomic_numbers
    evaluation = program.evaluate(
        positions, neighborhood.build(positions), compute_stress=True, species=species
    )

    def energy(deformed_positions: Any, deformed_vectors: Any) -> float:
        rebuilt = neighborhood.build(deformed_positions, cell_vectors=deformed_vectors)
        value, (_, _, successful, _) = program.energy(
            deformed_positions, rebuilt, species=species, cell_vectors=deformed_vectors
        )
        assert bool(successful)
        return float(value)

    assert bool(evaluation.successful)
    assert (
        evaluation.stress_convention is AtomisticStressConvention.CAUCHY_TENSION_POSITIVE
    )
    reference = _independent_strain_derivative(energy, positions, cell)
    np.testing.assert_allclose(
        evaluation.strain_derivative, reference, rtol=1.0e-6, atol=1.0e-8
    )
    np.testing.assert_allclose(
        evaluation.stress,
        0.5 * (reference + reference.T) / abs(np.linalg.det(_TRICLINIC)),
        rtol=1.0e-6,
        atol=1.0e-9,
    )
    unstressed = program.evaluate(
        positions, neighborhood.build(positions), species=species
    )
    assert unstressed.stress is None and unstressed.strain_derivative is None
    np.testing.assert_allclose(unstressed.forces, evaluation.forces, rtol=1.0e-12)


def test_single_atom_self_images_have_zero_force_and_nonzero_cell_response() -> None:
    vectors = 2.2 * np.eye(3) + np.asarray(
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.2, 0.0]]
    )
    system, program, neighborhood = _learned_image_case([6], vectors, 3.0)
    positions = jnp.asarray([[0.0, 0.0, 0.0]])
    species = system.plan.atomic_numbers
    evaluation = program.evaluate(
        positions, neighborhood.build(positions), compute_stress=True, species=species
    )

    def energy(deformed_positions: Any, deformed_vectors: Any) -> float:
        rebuilt = neighborhood.build(deformed_positions, cell_vectors=deformed_vectors)
        return float(
            program.energy(
                deformed_positions,
                rebuilt,
                species=species,
                cell_vectors=deformed_vectors,
            )[0]
        )

    assert bool(evaluation.successful)
    np.testing.assert_allclose(evaluation.forces, 0.0, atol=1.0e-12)
    assert float(jnp.max(jnp.abs(evaluation.stress))) > 1.0e-6
    np.testing.assert_allclose(
        evaluation.strain_derivative,
        _independent_strain_derivative(energy, positions, system.cell),
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_fixed_image_integers_make_wrapped_and_unwrapped_representations_agree() -> None:
    system, program, neighborhood = _learned_image_case([1, 6], _TRICLINIC, 3.0)
    cell = system.cell
    species = system.plan.atomic_numbers
    wrapped = cell.cartesian(jnp.asarray([[0.1, 0.2, 0.3], [0.6, 0.5, 0.45]]))
    shifted = wrapped.at[1].add(jnp.asarray([1.0, -1.0, 2.0]) @ cell.vectors)
    inside = program.evaluate(
        wrapped, neighborhood.build(wrapped), compute_stress=True, species=species
    )
    outside = program.evaluate(
        shifted, neighborhood.build(shifted), compute_stress=True, species=species
    )
    assert bool(inside.successful) and bool(outside.successful)
    np.testing.assert_allclose(outside.energy, inside.energy, rtol=1.0e-12)
    np.testing.assert_allclose(outside.forces, inside.forces, atol=1.0e-10)
    np.testing.assert_allclose(outside.stress, inside.stress, atol=1.0e-10)


def _lennard_jones_case(
    cutoff: float = 1.2,
    periodic_axes: tuple[bool, bool, bool] = (True, True, True),
) -> tuple[Any, Any, Any]:
    cell = phx.discretization.PeriodicCell(
        # ty: ignore[invalid-argument-type]
        [[3.0, 0.0, 0.0], [0.4, 2.8, 0.0], [0.2, 0.1, 3.1]],
        periodic_axes=periodic_axes,
    )
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1, 2],
        # ty: ignore[invalid-argument-type]
        [1, 1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0, 0],
        cell=cell,
    ).prepare()
    program = phx.atomistic.AtomisticPotentialProgram(
        # ty: ignore[invalid-argument-type]
        [phx.atomistic.LennardJonesPotential([1.0], [1.0], cutoff)]
    ).prepare(system)
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(3, box=cell).prepare(
        system.particles
    )
    return system, program, neighborhood


@pytest.mark.parametrize(
    ("periodic_axes", "convention"),
    [
        pytest.param(
            (True, True, True),
            AtomisticStressConvention.CAUCHY_TENSION_POSITIVE,
            id="bulk-volume",
        ),
        pytest.param(
            (True, True, False),
            AtomisticStressConvention.CAUCHY_TENSION_POSITIVE_EMBEDDING_VOLUME,
            id="slab-embedding-volume",
        ),
    ],
)
def test_classical_runtime_shear_strain_matches_independent_deformation(
    periodic_axes: tuple[bool, bool, bool], convention: Any
) -> None:
    system, program, neighborhood = _lennard_jones_case(periodic_axes=periodic_axes)
    cell = system.cell
    positions = cell.cartesian(
        jnp.asarray([[0.1, 0.1, 0.1], [0.4, 0.15, 0.1], [0.95, 0.5, 0.9]])
    )
    evaluation = program.evaluate(
        positions, neighborhood.build(positions), compute_stress=True
    )

    def energy(deformed_positions: Any, deformed_vectors: Any) -> float:
        value, (_, _, successful, _) = program.energy(
            deformed_positions,
            neighborhood.build(deformed_positions),
            cell_vectors=deformed_vectors,
        )
        assert bool(successful)
        return float(value)

    assert bool(evaluation.successful)
    assert evaluation.stress_convention is convention
    reference = _independent_strain_derivative(energy, positions, cell)
    np.testing.assert_allclose(
        evaluation.strain_derivative, reference, rtol=1.0e-6, atol=1.0e-8
    )
    np.testing.assert_allclose(
        evaluation.stress,
        0.5 * (reference + reference.T) / abs(np.linalg.det(np.asarray(cell.vectors))),
        rtol=1.0e-6,
        atol=1.0e-9,
    )


def test_requested_stress_refuses_lattice_without_embedding_volume() -> None:
    cell = phx.discretization.PeriodicCell(
        # ty: ignore[invalid-argument-type]
        [[3.0, 0.0, 0.0], [0.4, 2.8, 0.0]]
    )
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0],
        cell=cell,
    ).prepare()
    program = phx.atomistic.AtomisticPotentialProgram(
        # ty: ignore[invalid-argument-type]
        [phx.atomistic.LennardJonesPotential([1.0], [1.0], 1.2)]
    ).prepare(system)
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1, box=cell).prepare(
        system.particles
    )
    positions = jnp.asarray([[0.1, 0.1, 0.0], [1.0, 0.2, 0.3]])
    relation = neighborhood.build(positions)
    assert bool(program.evaluate(positions, relation).successful)
    with pytest.raises(ValueError, match="no physical or embedding volume"):
        program.evaluate(positions, relation, compute_stress=True)


def test_classical_runtime_cell_outside_unique_image_certificate_fails_closed() -> None:
    system, program, neighborhood = _lennard_jones_case()
    cell = system.cell
    positions = cell.cartesian(
        jnp.asarray([[0.1, 0.1, 0.1], [0.4, 0.15, 0.1], [0.7, 0.5, 0.6]])
    )
    shrunk = 0.7 * cell.vectors
    evaluation = program.evaluate(
        positions,
        neighborhood.build(positions),
        compute_stress=True,
        cell_vectors=shrunk,
    )
    assert not bool(evaluation.successful)
    assert not bool(evaluation.neighborhood_successful)
    assert bool(jnp.isnan(evaluation.energy))
    assert bool(jnp.all(jnp.isnan(evaluation.stress)))


def test_mixed_classical_program_keeps_pair_once_guards() -> None:
    system, _, classical = _lennard_jones_case()
    cell = system.cell
    units = system.plan.units
    model = phx.nn.atomistic.PaiNNPotential(
        units.scale,
        cutoff=2.0,
        feature_count=4,
        interaction_count=1,
        radial_basis_count=3,
        key=jr.key(5),
    )
    mixed = phx.atomistic.AtomisticPotentialProgram(
        [
            # ty: ignore[invalid-argument-type]
            phx.atomistic.LennardJonesPotential([1.0], [1.0], 1.2),
            phx.atomistic.LearnedGraphPotentialTerm(
                model, allow_periodic=True, name="graph"
            ),
        ]
    ).prepare(
        system,
        graph_execution=phx.atomistic.AtomisticGraphExecutionPlan(64, backend="particle"),
    )
    image = phx.discretization.DenseParticleImageNeighborhoodPlan(
        2.0,
        cell,
        phx.discretization.ParticleImageCapacity(
            maximum_particles_per_cell=3,
            maximum_edges=512,
            maximum_degree=256,
            maximum_images=512,
        ),
        maximum_dense_routes=4096,
    ).prepare(system.particles)
    positions = cell.cartesian(
        jnp.asarray([[0.1, 0.1, 0.1], [0.4, 0.15, 0.1], [0.7, 0.5, 0.6]])
    )
    with pytest.raises(ValueError, match="Pair-once"):
        mixed.evaluate(positions, image.build(positions))
    with pytest.raises(ValueError, match="unique-image"):
        mixed.evaluate(positions, classical.build(positions))


def test_requested_stress_refuses_terms_without_cell_derivative() -> None:
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[1, 1],
    ).prepare()
    model = phx.nn.atomistic.PaiNNPotential(
        units.scale,
        cutoff=2.0,
        feature_count=4,
        interaction_count=1,
        radial_basis_count=3,
        key=jr.key(7),
    )
    program = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LearnedGraphPotentialTerm(model)]
    ).prepare(
        system,
        graph_execution=phx.atomistic.AtomisticGraphExecutionPlan(4, backend="particle"),
    )
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
        system.particles
    )
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="cell derivative"):
        program.evaluate(positions, neighborhood.build(positions), compute_stress=True)


def test_controlled_cell_stress_matches_controlled_energy_strain() -> None:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    cell = phx.discretization.PeriodicCell(
        # ty: ignore[invalid-argument-type]
        [[7.0, 0.0, 0.0], [0.8, 6.6, 0.0], [0.3, 0.4, 7.2]]
    )
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [10, 20],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0],
        cell=cell,
    )
    force_field = AtomisticForceFieldPlan(
        system,
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([0.5], [1.0], 3.0)]
        ),
        AtomisticNonbondedPolicy(3.0, electrostatics="direct"),
        AtomisticForceFieldProvenance("native", ("cell",), "test", "controlled-cell"),
    ).prepare()
    schedule = AlchemicalControlSchedulePlan(
        ("on", "off"),
        ("sterics",),
        (AlchemicalControlKind.STERICS,),
        jnp.asarray([[1.0], [0.0]]),
    )
    controlled = ControlledHamiltonianPlan(
        force_field,
        schedule,
        # ty: ignore[invalid-argument-type]
        AlchemicalInteractionPartitionPlan(schedule.control_ids, ([10],)),
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1, box=cell).prepare(
        force_field.system.particles
    )
    fractional = jnp.asarray([[0.1, 0.1, 0.1], [0.25, 0.12, 0.08]])
    positions = cell.cartesian(fractional)
    control = jnp.asarray([0.4])
    result = phx.atomistic.atomistic_cell_energy_and_stress(
        controlled, fractional, neighborhood.build(positions), control_values=control
    )
    direct = controlled.evaluate(
        positions,
        neighborhood.build(positions),
        compute_stress=True,
        control_values=control,
    )

    def energy(deformed_positions: Any, deformed_vectors: Any) -> float:
        value, (_, _, successful, _) = controlled.energy(
            deformed_positions,
            neighborhood.build(deformed_positions),
            control_values=control,
            cell_vectors=deformed_vectors,
        )
        assert bool(successful)
        return float(value)

    assert bool(result.successful) and bool(direct.successful)
    assert result.stress is not None and direct.stress is not None
    assert direct.dU_dcontrols is not None
    np.testing.assert_allclose(result.energy, direct.energy, rtol=1.0e-12)
    np.testing.assert_allclose(result.stress, direct.stress, rtol=1.0e-10)
    np.testing.assert_allclose(
        result.control_derivatives, direct.dU_dcontrols, rtol=1.0e-10
    )
    np.testing.assert_allclose(
        result.cell_gradient,
        _independent_strain_derivative(energy, positions, cell),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
