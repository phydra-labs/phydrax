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


def _runtime(*, step_size=1.0e-3, cell=None, topology=None):
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    plan = phx.atomistic.AtomisticSystemPlan(
        [10, 20],
        [1, 1],
        [1.0, 1.0],
        units,
        atom_type_ids=[0, 0],
        topology=topology,
        cell=cell,
    )
    system = plan.prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1, box=cell).prepare(
        system.particles
    )
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([1.0], [1.0], 2.5, switch_distance=2.0)]
    ).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.VelocityVerletPlan(step_size),
    ).prepare()
    return units, system, dynamics


def test_unit_system_is_complete_and_derived_from_unit_definitions():
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


def test_scale_rejects_molar_energy_and_reduced_units_are_not_si_convertible():
    with pytest.raises(ValueError, match="ordinary ENERGY"):
        phx.atomistic.AtomisticScaleContract(ANGSTROM, KILOCALORIE_PER_MOLE)
    reduced = phx.atomistic.AtomisticUnitSystem.reduced()
    with pytest.raises(ValueError, match="reference system"):
        conversion_factor(reduced.scale.length_unit, ANGSTROM)


def test_system_identity_is_independent_of_initial_positions():
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    structure_a = phx.atomistic.AtomicStructure(
        [1, 1], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [1.0, 1.0], units.scale
    )
    structure_b = phx.atomistic.AtomicStructure(
        [1, 1], [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], [1.0, 1.0], units.scale
    )
    first = phx.atomistic.AtomisticSystemPlan.from_structure(structure_a, units)
    second = phx.atomistic.AtomisticSystemPlan.from_structure(structure_b, units)
    assert first.system_id == second.system_id
    assert structure_a.structure_id != structure_b.structure_id


def test_topology_resolves_stable_ids_and_sparse_pair_exceptions():
    topology = phx.atomistic.MolecularTopologyPlan(
        bonds=[[20, 10]],
        pair_exceptions=[[10, 20]],
        lennard_jones_scales=[0.5],
        electrostatic_scales=[0.0],
    )
    _, system, _ = _runtime(topology=topology)
    neighborhood = (
        phx.discretization.DenseParticleNeighborhoodPlan(1)
        .prepare(system.particles)
        .build(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    )
    keys = system.pair_key_space.keys(neighborhood.pair_relation)
    lj, electrostatic = system.topology.pair_scales(keys.keys)
    np.testing.assert_allclose(lj, [0.5])
    np.testing.assert_allclose(electrostatic, [0.0])
    np.testing.assert_array_equal(system.topology.bond_indices, [[0, 1]])


@pytest.mark.parametrize("periodic", [False, True])
def test_lennard_jones_force_is_negative_energy_gradient(periodic):
    cell = phx.discretization.PeriodicCell(6.0 * jnp.eye(3)) if periodic else None
    _, _, dynamics = _runtime(cell=cell)
    positions = jnp.asarray([[5.6, 0.0, 0.0], [6.8, 0.0, 0.0]])
    state = dynamics.initialize_state(
        positions, velocity=jnp.zeros_like(positions), key=jax.random.key(0)
    )
    direction = jnp.zeros_like(positions).at[1, 0].set(1.0)
    step = 1.0e-5
    neighborhood = state.neighborhood
    plus = dynamics.potential.energy(positions + step * direction, neighborhood)[0]
    minus = dynamics.potential.energy(positions - step * direction, neighborhood)[0]
    finite_difference = -(plus - minus) / (2.0 * step)
    np.testing.assert_allclose(
        finite_difference, state.force.forces[1, 0], rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(jnp.sum(state.force.forces, axis=0), 0.0, atol=1e-12)


@pytest.mark.parametrize("periodic", [False, True])
def test_coordinate_representations_preserve_bond_force_and_curvature(periodic):
    cell = phx.discretization.PeriodicCell(6.0 * jnp.eye(3)) if periodic else None
    topology = phx.atomistic.MolecularTopologyPlan(bonds=[[10, 20]])
    _, system, _ = _runtime(cell=cell, topology=topology)
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1, box=cell).prepare(
        system.particles
    )
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.HarmonicBondPotential([4.0], [1.0])]
    ).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system, potential, neighborhood, phx.atomistic.VelocityVerletPlan(1.0e-3)
    ).prepare()
    unwrapped = jnp.asarray([[5.6, 0.0, 0.0], [6.8, 0.0, 0.0]])
    state = dynamics.initialize_state(
        unwrapped, velocity=jnp.zeros_like(unwrapped), key=jax.random.key(19)
    )
    np.testing.assert_allclose(
        state.force.forces, [[0.8, 0, 0], [-0.8, 0, 0]], atol=1e-12
    )
    position = state.kinematics.positions
    kwargs = {"unwrapped_positions": unwrapped}
    if cell is not None:
        kwargs["fractional_positions"] = cell.fractional(position)
        kwargs["cell_vectors"] = cell.vectors

    def energy(coordinates):
        return potential.evaluate(coordinates, state.neighborhood, **kwargs).energy

    def force(coordinates):
        return potential.evaluate(coordinates, state.neighborhood, **kwargs).forces[1, 0]

    np.testing.assert_allclose(
        -eqx.filter_jit(jax.grad(energy))(position), state.force.forces, atol=1e-12
    )
    np.testing.assert_allclose(
        eqx.filter_jit(jax.grad(force))(position),
        [[4.0, 0, 0], [-4.0, 0, 0]],
        atol=1e-12,
    )
    stepped = eqx.filter_jit(dynamics.step_detailed)(state)
    assert bool(stepped.successful)
    assert float(stepped.accepted_state.kinematics.momenta[1, 0]) < 0.0

    if cell is not None:
        fractional = cell.fractional(position)
        images = state.kinematics.image_counts

        def cell_energy(vectors):
            coordinates = cell.cartesian_with_vectors(fractional, vectors)
            whole = cell.cartesian_with_vectors(fractional + images, vectors)
            return potential.evaluate(
                coordinates,
                state.neighborhood,
                unwrapped_positions=whole,
                fractional_positions=fractional,
                cell_vectors=vectors,
            ).energy

        expected = jnp.zeros((3, 3)).at[0, 0].set(0.16)
        np.testing.assert_allclose(
            jax.grad(cell_energy)(cell.vectors), expected, atol=1e-12
        )


def test_velocity_verlet_is_reversible_to_second_order_and_jittable():
    _, _, dynamics = _runtime(step_size=1.0e-4)
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]])
    state = dynamics.initialize_state(
        positions,
        velocity=jnp.asarray([[0.0, 0.05, 0.0], [0.0, -0.05, 0.0]]),
        key=jax.random.key(1),
    )
    step = eqx.filter_jit(dynamics.step_detailed)(state)
    assert bool(step.successful)
    assert int(step.accepted_state.step_index) == 1
    assert bool(jnp.all(jnp.isfinite(step.diagnostics.total_energy)))
    np.testing.assert_allclose(step.diagnostics.total_linear_momentum, 0.0, atol=1e-12)
