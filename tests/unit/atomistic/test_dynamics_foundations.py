from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.units import (
    ANGSTROM,
    conversion_factor,
    ELECTRONVOLT,
    FREQUENCY,
    KILOCALORIE_PER_MOLE,
    PRESSURE,
    VELOCITY,
)
from tests._support.atomistic_bundles import (
    prepare_program,
    role_capacity,
    search_capacity,
)


# Explicit resources of the two-atom fixtures: the finite system is searched in
# a nonperiodic frame enclosing every trajectory, and one pair (one directed
# route per receiver) is the complete pair storage of two atoms.
_FRAME = phx.discretization.PeriodicCell(
    20.0 * jnp.eye(3), origin=jnp.full((3,), -10.0), periodic_axes=(False,) * 3
)
_RESOURCES: dict[str, Any] = {
    "search": phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
    "skin": 0.5,
    "search_capacities": (search_capacity(8, particles_per_cell=2, images=27),),
    "role_capacities": (role_capacity(1, 1),),
}


def _resources(cell: Any) -> dict[str, Any]:
    return {**_RESOURCES, "frame": _FRAME if cell is None else None}


def _runtime(*, step_size: Any = 1.0e-3, cell: Any = None, topology: Any = None) -> Any:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    plan = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [10, 20],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0],
        topology=topology,
        cell=cell,
    )
    system = plan.prepare()
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([1.0], [1.0], 2.5, switch_distance=2.0)]
        ),
        system,
        **_resources(cell),
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        phx.atomistic.VelocityVerletPlan(step_size),
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system), ensemble="nve"
    ).prepare(dynamics)
    return units, system, dynamics, thermodynamic


def test_dynamics_foundations_scenario_1() -> None:
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    assert units.scale.length_unit == ANGSTROM
    assert units.scale.energy_unit == ELECTRONVOLT
    assert units.scale.energy_semantics == "single-simulated-system"
    assert units.constant_set_id == "codata-2018"
    assert units.pressure_unit.dimension == PRESSURE
    assert units.velocity_unit.dimension == VELOCITY
    assert units.frequency_unit.dimension == FREQUENCY
    np.testing.assert_allclose(units.kinetic_to_energy, 103.64269652680505)
    np.testing.assert_allclose(units.boltzmann_constant, 8.617333262145e-5)
    np.testing.assert_allclose(units.coulomb_constant, 14.399645478425668)
    np.testing.assert_allclose(units.reduced_planck_constant, 0.6582119569509067)
    assert units.force_to_momentum_rate == 1.0 / units.kinetic_to_energy
    restored = phx.atomistic.AtomisticUnitSystem.from_dict(units.to_dict())
    assert restored.unit_system_id == units.unit_system_id
    ambiguous = units.to_dict()
    ambiguous["kinetic_to_energy"] = units.kinetic_to_energy
    with pytest.raises(ValueError, match="canonical fields"):
        phx.atomistic.AtomisticUnitSystem.from_dict(ambiguous)
    with pytest.raises(ValueError, match="ordinary ENERGY"):
        phx.atomistic.AtomisticScaleContract(ANGSTROM, KILOCALORIE_PER_MOLE)
    reduced = phx.atomistic.AtomisticUnitSystem.reduced()
    with pytest.raises(ValueError, match="reference system"):
        conversion_factor(reduced.scale.length_unit, ANGSTROM)
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    structure_a = phx.atomistic.AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units.scale,
    )
    structure_b = phx.atomistic.AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units.scale,
    )
    first = phx.atomistic.AtomisticSystemPlan.from_structure(structure_a, units)
    second = phx.atomistic.AtomisticSystemPlan.from_structure(structure_b, units)
    assert first.system_id == second.system_id
    assert structure_a.structure_id != structure_b.structure_id


def _bound_energy(dynamics: Any, state: Any, positions: Any) -> Any:
    """Energy at a trial point bound to the accepted epochs, outside the scalar.

    The accepted bundle is certified at the accepted geometry; a trial point
    first binds that frozen topology at its own coordinates (same cell and
    image representation) and must be certified there.
    """
    geometry = phx.atomistic.bind_atomistic_bundle_geometry(
        dynamics.bundle,
        dynamics.system,
        positions,
        cell_vectors=state.cell_vectors if dynamics.periodic else None,
        image_counts=state.kinematics.image_counts if dynamics.periodic else None,
    )
    neighborhoods = dynamics.bundle.bind(state.neighborhoods, geometry)
    assert bool(jnp.all(neighborhoods.successful))
    energy, auxiliary = dynamics.potential.energy(positions, neighborhoods)
    assert bool(auxiliary.successful)
    return energy


def test_topology_exception_scales_follow_admitted_classical_pair_keys() -> None:
    topology = phx.atomistic.MolecularTopologyPlan(
        bonds=[[20, 10]],
        pair_exceptions=[[10, 20]],
        lennard_jones_scales=[0.5],
        electrostatic_scales=[0.0],
    )
    _, system, dynamics, thermodynamic = _runtime(topology=topology)
    np.testing.assert_array_equal(system.topology.bond_indices, [[0, 1]])
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]])
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(0),
    )
    # The physical pair keys come from the bundle's admitted classical
    # pair-once relation, the relation the Lennard-Jones term consumes.
    requirement = dynamics.potential.plan.requirements.neighborhood(
        phx.atomistic.AtomisticNeighborhoodRole.CLASSICAL_MINIMUM_IMAGE_PAIRS
    )
    view = dynamics.bundle.view(state.neighborhoods, requirement)
    pairs = view.pair_relation
    assert pairs is not None
    keys = system.pair_key_space.keys(pairs)
    assert bool(keys.successful)
    valid = np.asarray(pairs.valid & keys.valid)
    assert int(np.count_nonzero(valid)) == 1
    lj, electrostatic = system.topology.pair_scales(keys.keys)
    np.testing.assert_allclose(np.asarray(lj)[valid], [0.5])
    np.testing.assert_allclose(np.asarray(electrostatic)[valid], [0.0])
    assert view.lennard_jones_scales is not None
    assert view.electrostatic_scales is not None
    np.testing.assert_allclose(np.asarray(view.lennard_jones_scales)[valid], [0.5])
    np.testing.assert_allclose(np.asarray(view.electrostatic_scales)[valid], [0.0])

    # The exception weight scales the excepted pair's interaction exactly.
    _, _, plain_dynamics, plain_thermodynamic = _runtime()
    plain = plain_dynamics.initialize_state(
        positions,
        plain_thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(0),
    )
    assert float(plain.force.potential_energy) != 0.0
    np.testing.assert_allclose(
        state.force.potential_energy, 0.5 * plain.force.potential_energy, rtol=1e-12
    )
    np.testing.assert_allclose(
        state.force.forces, 0.5 * plain.force.forces, rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("periodic", [False, True], ids=["finite", "periodic"])
def test_pair_force_matches_bound_central_difference(periodic: bool) -> None:
    cell = phx.discretization.PeriodicCell(6.0 * jnp.eye(3)) if periodic else None
    _, _, dynamics, thermodynamic = _runtime(cell=cell)
    positions = jnp.asarray([[5.6, 0.0, 0.0], [6.8, 0.0, 0.0]])
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(0),
    )
    direction = jnp.zeros_like(positions).at[1, 0].set(1.0)
    step = 1.0e-5
    # The accepted bundle routes are bound to the wrapped representation.
    wrapped = state.kinematics.positions
    plus = _bound_energy(dynamics, state, wrapped + step * direction)
    minus = _bound_energy(dynamics, state, wrapped - step * direction)
    finite_difference = -(plus - minus) / (2.0 * step)
    np.testing.assert_allclose(
        finite_difference, state.force.forces[1, 0], rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(jnp.sum(state.force.forces, axis=0), 0.0, atol=1e-12)


def test_velocity_verlet_step_conserves_linear_momentum() -> None:
    _, _, dynamics, thermodynamic = _runtime(step_size=1.0e-4)
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]])
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.asarray([[0.0, 0.05, 0.0], [0.0, -0.05, 0.0]]),
        key=jax.random.key(1),
    )
    step = eqx.filter_jit(dynamics.step_detailed)(state, thermodynamic)
    assert bool(step.successful)
    assert int(step.accepted_state.step_index) == 1
    assert bool(jnp.all(jnp.isfinite(step.diagnostics.total_energy)))
    np.testing.assert_allclose(step.diagnostics.total_linear_momentum, 0.0, atol=1e-12)


def test_coordinate_representations_preserve_bond_force_and_curvature() -> None:
    for periodic in [False, True]:
        cell = phx.discretization.PeriodicCell(6.0 * jnp.eye(3)) if periodic else None
        topology = phx.atomistic.MolecularTopologyPlan(bonds=[[10, 20]])
        _, system, _, _ = _runtime(cell=cell, topology=topology)
        potential = prepare_program(
            phx.atomistic.AtomisticPotentialProgram(
                # ty: ignore[invalid-argument-type]
                [phx.atomistic.HarmonicBondPotential([4.0], [1.0])]
            ),
            system,
            **_resources(cell),
        )
        dynamics = phx.atomistic.AtomisticDynamicsPlan(
            system, potential, phx.atomistic.VelocityVerletPlan(1.0e-3)
        ).prepare()
        thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
            phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system), ensemble="nve"
        ).prepare(dynamics)
        unwrapped = jnp.asarray([[5.6, 0.0, 0.0], [6.8, 0.0, 0.0]])
        state = dynamics.initialize_state(
            unwrapped,
            thermodynamic,
            velocity=jnp.zeros_like(unwrapped),
            key=jax.random.key(19),
        )
        np.testing.assert_allclose(
            state.force.forces, [[0.8, 0, 0], [-0.8, 0, 0]], atol=1e-12
        )
        position = state.kinematics.positions
        kwargs: dict[str, Any] = {"unwrapped_positions": unwrapped}
        if cell is not None:
            kwargs["cell_vectors"] = cell.vectors
            kwargs["image_counts"] = state.kinematics.image_counts

        def energy(coordinates: Any) -> Any:
            return potential.evaluate(coordinates, state.neighborhoods, **kwargs).energy

        def force(coordinates: Any) -> Any:
            return potential.evaluate(coordinates, state.neighborhoods, **kwargs).forces[
                1, 0
            ]

        np.testing.assert_allclose(
            -eqx.filter_jit(jax.grad(energy))(position), state.force.forces, atol=1e-12
        )
        np.testing.assert_allclose(
            eqx.filter_jit(jax.grad(force))(position),
            [[4.0, 0, 0], [-4.0, 0, 0]],
            atol=1e-12,
        )
        stepped = eqx.filter_jit(dynamics.step_detailed)(state, thermodynamic)
        assert bool(stepped.successful)
        assert float(stepped.accepted_state.kinematics.momenta[1, 0]) < 0.0

        if cell is not None:
            fractional = cell.fractional(position)
            images = state.kinematics.image_counts

            def cell_energy(vectors: Any) -> Any:
                coordinates = cell.cartesian_with_vectors(fractional, vectors)
                whole = cell.cartesian_with_vectors(fractional + images, vectors)
                return potential.evaluate(
                    coordinates,
                    state.neighborhoods,
                    unwrapped_positions=whole,
                    cell_vectors=vectors,
                    image_counts=images,
                ).energy

            expected = jnp.zeros((3, 3)).at[0, 0].set(0.16)
            np.testing.assert_allclose(
                jax.grad(cell_energy)(cell.vectors), expected, atol=1e-12
            )
