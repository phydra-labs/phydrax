from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.atomistic import (
    AtomisticDenseImageSearch,
    AtomisticPotentialProgram,
    PreparedAtomisticPotentialProgram,
    PreparedAtomisticSystem,
)
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
from phydrax.discretization import PeriodicCell
from tests._support.atomistic_bundles import (
    bundle_state,
    neighborhood_bundle_plan,
    role_capacity,
    search_capacity,
)


_TRICLINIC = np.asarray([[2.6, 0.0, 0.0], [0.5, 2.4, 0.0], [0.3, 0.2, 2.7]])
_LENNARD_JONES_CELL = np.asarray([[3.0, 0.0, 0.0], [0.4, 2.8, 0.0], [0.2, 0.1, 3.1]])
# The dense image search is the former dense neighborhood reference.
_SEARCH = AtomisticDenseImageSearch(maximum_dense_routes=8192)
_SKIN = 0.3


def _prepare(
    program: AtomisticPotentialProgram,
    system: PreparedAtomisticSystem,
    /,
    *,
    particles: int,
    routes: int,
    degree: int,
    graph_execution: phx.atomistic.AtomisticGraphExecutionPlan | None = None,
    frame: PeriodicCell | None = None,
) -> PreparedAtomisticPotentialProgram:
    """Prepare ``program`` with the explicit dense image bundle of its requirements."""
    bundle = neighborhood_bundle_plan(
        program.requirements.neighborhoods,
        search=_SEARCH,
        skin=_SKIN,
        search_capacities=(
            search_capacity(1024, particles_per_cell=particles, images=512),
        ),
        role_capacities=(role_capacity(routes, degree),),
        frame=frame,
    )
    return program.prepare(
        system, neighborhood_bundle=bundle, graph_execution=graph_execution
    )


def _system(
    particle_ids: list[int],
    atomic_numbers: list[int],
    atom_type_ids: list[int],
    units: phx.atomistic.AtomisticUnitSystem,
    cell: PeriodicCell | None,
) -> phx.atomistic.AtomisticSystemPlan:
    return phx.atomistic.AtomisticSystemPlan(
        np.asarray(particle_ids, dtype=np.int64),
        np.asarray(atomic_numbers, dtype=np.int64),
        np.ones(len(particle_ids), dtype=np.float64),
        units,
        atom_type_ids=np.asarray(atom_type_ids, dtype=np.int64),
        cell=cell,
    )


def _lennard_jones(
    cutoff: float, epsilon: float = 1.0
) -> phx.atomistic.LennardJonesPotential:
    return phx.atomistic.LennardJonesPotential(
        np.asarray([epsilon], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
        cutoff,
    )


def _learned_image_case(
    atomic_numbers: list[int], vectors: np.ndarray, cutoff: float
) -> tuple[PreparedAtomisticSystem, PreparedAtomisticPotentialProgram, PeriodicCell]:
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    cell = PeriodicCell(vectors)
    count = len(atomic_numbers)
    system = _system(
        list(range(count)), atomic_numbers, atomic_numbers, units, cell
    ).prepare()
    model = phx.nn.atomistic.PaiNNPotential(
        units.scale,
        cutoff=cutoff,
        feature_count=4,
        interaction_count=1,
        radial_basis_count=3,
        key=jr.key(3),
    )
    program = _prepare(
        AtomisticPotentialProgram(
            [phx.atomistic.LearnedGraphPotentialTerm(model, allow_periodic=True)]
        ),
        system,
        particles=count,
        routes=1024,
        degree=128,
        graph_execution=phx.atomistic.AtomisticGraphExecutionPlan(
            128, backend="particle"
        ),
    )
    return system, program, cell


def _independent_strain_derivative(
    energy: Callable[[Array, Array], float],
    positions: Array,
    cell: PeriodicCell,
    step: float = 1.0e-5,
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


def _deformed_energy(
    program: PreparedAtomisticPotentialProgram, species: Array | None = None
) -> Callable[[Array, Array], float]:
    """Energy of a bundle state certified at each deformed geometry."""

    def energy(deformed_positions: Array, deformed_vectors: Array) -> float:
        state = bundle_state(program, deformed_positions, cell_vectors=deformed_vectors)
        value, auxiliary = program.energy(
            deformed_positions, state, species=species, cell_vectors=deformed_vectors
        )
        assert bool(auxiliary.successful)
        return float(value)

    return energy


def test_learned_image_stress_matches_independent_triclinic_strain() -> None:
    system, program, cell = _learned_image_case([1, 6], _TRICLINIC, 3.0)
    positions = cell.cartesian(jnp.asarray([[0.1, 0.2, 0.3], [0.6, 0.5, 0.45]]))
    species = system.plan.atomic_numbers
    state = bundle_state(program, positions)
    evaluation = program.evaluate(positions, state, compute_stress=True, species=species)

    assert bool(evaluation.successful)
    assert (
        evaluation.stress_convention is AtomisticStressConvention.CAUCHY_TENSION_POSITIVE
    )
    assert evaluation.stress is not None and evaluation.strain_derivative is not None
    reference = _independent_strain_derivative(
        _deformed_energy(program, species), positions, cell
    )
    np.testing.assert_allclose(
        evaluation.strain_derivative, reference, rtol=1.0e-6, atol=1.0e-8
    )
    np.testing.assert_allclose(
        evaluation.stress,
        0.5 * (reference + reference.T) / abs(np.linalg.det(_TRICLINIC)),
        rtol=1.0e-6,
        atol=1.0e-9,
    )
    unstressed = program.evaluate(positions, state, species=species)
    assert unstressed.stress is None and unstressed.strain_derivative is None
    np.testing.assert_allclose(unstressed.forces, evaluation.forces, rtol=1.0e-12)


def test_single_atom_self_images_have_zero_force_and_nonzero_cell_response() -> None:
    vectors = 2.2 * np.eye(3) + np.asarray(
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.2, 0.0]]
    )
    system, program, cell = _learned_image_case([6], vectors, 3.0)
    positions = jnp.asarray([[0.0, 0.0, 0.0]])
    species = system.plan.atomic_numbers
    evaluation = program.evaluate(
        positions, bundle_state(program, positions), compute_stress=True, species=species
    )

    assert bool(evaluation.successful)
    assert evaluation.stress is not None and evaluation.strain_derivative is not None
    np.testing.assert_allclose(evaluation.forces, 0.0, atol=1.0e-12)
    assert float(jnp.max(jnp.abs(evaluation.stress))) > 1.0e-6
    np.testing.assert_allclose(
        evaluation.strain_derivative,
        _independent_strain_derivative(
            _deformed_energy(program, species), positions, cell
        ),
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_fixed_image_integers_make_wrapped_and_unwrapped_representations_agree() -> None:
    system, program, cell = _learned_image_case([1, 6], _TRICLINIC, 3.0)
    species = system.plan.atomic_numbers
    wrapped = cell.cartesian(jnp.asarray([[0.1, 0.2, 0.3], [0.6, 0.5, 0.45]]))
    images = np.asarray([[0, 0, 0], [1, -1, 2]], dtype=np.int32)
    shifted = wrapped + jnp.asarray(images, dtype=wrapped.dtype) @ cell.vectors
    inside = program.evaluate(
        wrapped, bundle_state(program, wrapped), compute_stress=True, species=species
    )
    outside = program.evaluate(
        wrapped,
        bundle_state(program, wrapped, image_counts=images),
        compute_stress=True,
        species=species,
        image_counts=jnp.asarray(images),
        unwrapped_positions=shifted,
    )
    assert bool(inside.successful) and bool(outside.successful)
    assert inside.stress is not None and outside.stress is not None
    np.testing.assert_allclose(outside.energy, inside.energy, rtol=1.0e-12)
    np.testing.assert_allclose(outside.forces, inside.forces, atol=1.0e-10)
    np.testing.assert_allclose(outside.stress, inside.stress, atol=1.0e-10)


def _lennard_jones_case(
    cutoff: float = 1.2,
    periodic_axes: tuple[bool, bool, bool] = (True, True, True),
) -> tuple[PreparedAtomisticSystem, PreparedAtomisticPotentialProgram, PeriodicCell]:
    cell = PeriodicCell(_LENNARD_JONES_CELL, periodic_axes=periodic_axes)
    system = _system(
        [0, 1, 2], [1, 1, 1], [0, 0, 0], phx.atomistic.AtomisticUnitSystem.reduced(), cell
    ).prepare()
    program = _prepare(
        AtomisticPotentialProgram([_lennard_jones(cutoff)]),
        system,
        particles=3,
        routes=8,
        degree=4,
    )
    return system, program, cell


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
    periodic_axes: tuple[bool, bool, bool], convention: AtomisticStressConvention
) -> None:
    _, program, cell = _lennard_jones_case(periodic_axes=periodic_axes)
    positions = cell.cartesian(
        jnp.asarray([[0.1, 0.1, 0.1], [0.4, 0.15, 0.1], [0.95, 0.5, 0.9]])
    )
    evaluation = program.evaluate(
        positions, bundle_state(program, positions), compute_stress=True
    )

    assert bool(evaluation.successful)
    assert evaluation.stress_convention is convention
    assert evaluation.stress is not None and evaluation.strain_derivative is not None
    reference = _independent_strain_derivative(_deformed_energy(program), positions, cell)
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
    cell = PeriodicCell(np.asarray([[3.0, 0.0, 0.0], [0.4, 2.8, 0.0]]))
    system = _system(
        [0, 1], [1, 1], [0, 0], phx.atomistic.AtomisticUnitSystem.reduced(), cell
    ).prepare()
    program = _prepare(
        AtomisticPotentialProgram([_lennard_jones(1.2)]),
        system,
        particles=2,
        routes=4,
        degree=2,
    )
    positions = jnp.asarray([[0.1, 0.1, 0.0], [1.0, 0.2, 0.3]])
    state = bundle_state(program, positions)
    assert bool(program.evaluate(positions, state).successful)
    with pytest.raises(ValueError, match="no physical or embedding volume"):
        program.evaluate(positions, state, compute_stress=True)


def test_classical_runtime_cell_outside_unique_image_certificate_fails_closed() -> None:
    _, program, cell = _lennard_jones_case()
    positions = cell.cartesian(
        jnp.asarray([[0.1, 0.1, 0.1], [0.4, 0.15, 0.1], [0.7, 0.5, 0.6]])
    )
    shrunk = 0.7 * cell.vectors
    # The state is certified at the shrunk lattice: the refusal is the
    # pair-once unique-image admission of that lattice, not a stale geometry.
    evaluation = program.evaluate(
        positions,
        bundle_state(program, positions, cell_vectors=shrunk),
        compute_stress=True,
        cell_vectors=shrunk,
    )
    assert not bool(evaluation.successful)
    assert not bool(evaluation.neighborhood_successful)
    assert bool(jnp.isnan(evaluation.energy))
    assert evaluation.stress is not None
    assert bool(jnp.all(jnp.isnan(evaluation.stress)))


def test_mixed_classical_program_keeps_pair_once_guards() -> None:
    system, classical, cell = _lennard_jones_case()
    units = system.plan.units
    model = phx.nn.atomistic.PaiNNPotential(
        units.scale,
        cutoff=2.0,
        feature_count=4,
        interaction_count=1,
        radial_basis_count=3,
        key=jr.key(5),
    )
    execution = phx.atomistic.AtomisticGraphExecutionPlan(64, backend="particle")

    def graph_term() -> phx.atomistic.LearnedGraphPotentialTerm:
        return phx.atomistic.LearnedGraphPotentialTerm(
            model, allow_periodic=True, name="graph"
        )

    # One image source serves both the classical minimum-image pairs and the
    # directed image graph whose cutoff exceeds the unique-image radius.
    mixed = _prepare(
        AtomisticPotentialProgram([_lennard_jones(1.2), graph_term()]),
        system,
        particles=3,
        routes=512,
        degree=64,
        graph_execution=execution,
    )
    graph = _prepare(
        AtomisticPotentialProgram([graph_term()]),
        system,
        particles=3,
        routes=512,
        degree=64,
        graph_execution=execution,
    )
    positions = cell.cartesian(
        jnp.asarray([[0.1, 0.1, 0.1], [0.4, 0.15, 0.1], [0.7, 0.5, 0.6]])
    )
    combined = mixed.evaluate(positions, bundle_state(mixed, positions))
    pairs = classical.evaluate(positions, bundle_state(classical, positions))
    images = graph.evaluate(positions, bundle_state(graph, positions))
    assert (
        bool(combined.successful) and bool(pairs.successful) and bool(images.successful)
    )
    # The classical term still counts each pair once (no image double count).
    np.testing.assert_allclose(
        combined.energy, pairs.energy + images.energy, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        combined.forces, pairs.forces + images.forces, rtol=1.0e-10, atol=1.0e-12
    )
    # A classical pair-once cutoff beyond the unique-image radius is refused.
    with pytest.raises(ValueError, match="unique-image"):
        _prepare(
            AtomisticPotentialProgram([_lennard_jones(2.0)]),
            system,
            particles=3,
            routes=8,
            degree=4,
        )


def test_requested_stress_refuses_terms_without_cell_derivative() -> None:
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    system = _system([0, 1], [1, 1], [1, 1], units, None).prepare()
    model = phx.nn.atomistic.PaiNNPotential(
        units.scale,
        cutoff=2.0,
        feature_count=4,
        interaction_count=1,
        radial_basis_count=3,
        key=jr.key(7),
    )
    program = _prepare(
        AtomisticPotentialProgram([phx.atomistic.LearnedGraphPotentialTerm(model)]),
        system,
        particles=2,
        routes=2,
        degree=1,
        graph_execution=phx.atomistic.AtomisticGraphExecutionPlan(4, backend="particle"),
        frame=PeriodicCell(
            20.0 * np.eye(3),
            origin=np.full((3,), -10.0),
            periodic_axes=(False, False, False),
        ),
    )
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="cell derivative"):
        program.evaluate(positions, bundle_state(program, positions), compute_stress=True)


def test_controlled_cell_stress_matches_controlled_energy_strain() -> None:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    cell = PeriodicCell(np.asarray([[7.0, 0.0, 0.0], [0.8, 6.6, 0.0], [0.3, 0.4, 7.2]]))
    plan = AtomisticForceFieldPlan(
        _system([10, 20], [1, 1], [0, 0], units, cell),
        AtomisticPotentialProgram([_lennard_jones(3.0, epsilon=0.5)]),
        AtomisticNonbondedPolicy(3.0, electrostatics="direct"),
        AtomisticForceFieldProvenance("native", ("cell",), "test", "controlled-cell"),
    )
    force_field = plan.prepare(
        neighborhood_bundle=neighborhood_bundle_plan(
            plan.potential.requirements.neighborhoods,
            search=_SEARCH,
            skin=_SKIN,
            search_capacities=(search_capacity(1024, particles_per_cell=2, images=512),),
            role_capacities=(role_capacity(4, 2),),
        )
    )
    schedule = AlchemicalControlSchedulePlan(
        ("on", "off"),
        ("sterics",),
        (AlchemicalControlKind.STERICS,),
        jnp.asarray([[1.0], [0.0]]),
    )
    controlled = ControlledHamiltonianPlan(
        force_field,
        schedule,
        AlchemicalInteractionPartitionPlan(
            schedule.control_ids, (np.asarray([10], dtype=np.int64),)
        ),
    ).prepare()
    fractional = jnp.asarray([[0.1, 0.1, 0.1], [0.25, 0.12, 0.08]])
    positions = cell.cartesian(fractional)
    control = jnp.asarray([0.4])
    state = bundle_state(controlled, positions)
    result = phx.atomistic.atomistic_cell_energy_and_stress(
        controlled, fractional, state, control_values=control
    )
    direct = controlled.evaluate(
        positions, state, compute_stress=True, control_values=control
    )

    def energy(deformed_positions: Array, deformed_vectors: Array) -> float:
        value, auxiliary = controlled.energy(
            deformed_positions,
            bundle_state(controlled, deformed_positions, cell_vectors=deformed_vectors),
            control_values=control,
            cell_vectors=deformed_vectors,
        )
        assert bool(auxiliary.successful)
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
