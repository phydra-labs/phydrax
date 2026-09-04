from itertools import product

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax.units import COULOMB, KELVIN, KILOGRAM, SECOND


def _cell():
    return phx.discretization.PeriodicCell(
        [[3.0, 0.0, 0.0], [0.4, 2.8, 0.0], [0.2, 0.1, 3.1]]
    )


def test_triclinic_minimum_image_matches_brute_lattice_enumeration():
    cell = _cell()
    displacement = jnp.asarray([2.7, 1.9, -2.2])
    observed = cell.minimum_image(displacement)
    vectors = np.asarray(cell.vectors)
    candidates = np.asarray(
        [
            np.asarray(displacement) - np.asarray(shift) @ vectors
            for shift in product(range(-3, 4), repeat=3)
        ]
    )
    expected = candidates[np.argmin(np.sum(candidates * candidates, axis=1))]
    np.testing.assert_allclose(observed, expected, atol=1.0e-12)


def test_metric_cell_list_matches_dense_physical_pairs():
    cell = _cell()
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        [0, 1, 2, 3],
        [1, 1, 1, 1],
        [1.0, 1.0, 1.0, 1.0],
        units,
        atom_type_ids=[0, 0, 0, 0],
        cell=cell,
    ).prepare()
    fractional = jnp.asarray(
        [[0.05, 0.05, 0.05], [0.95, 0.05, 0.05], [0.5, 0.5, 0.5], [0.6, 0.5, 0.5]]
    )
    positions = cell.cartesian(fractional)
    metric = phx.discretization.MetricCellListParticleNeighborhoodPlan(
        0.7, 4, 6, cell
    ).prepare(system.particles)
    dense = phx.discretization.DenseParticleNeighborhoodPlan(6, box=cell).prepare(
        system.particles
    )
    metric_state = metric.build(positions)
    dense_state = dense.build(positions)
    dense_geometry = phx.discretization.particle_pair_geometry(
        positions, dense_state.pair_relation, box=cell
    )
    dense_pairs = {
        tuple(sorted((int(left), int(right))))
        for left, right, distance in zip(
            np.asarray(dense_state.pair_relation.left_particle_ids),
            np.asarray(dense_state.pair_relation.right_particle_ids),
            np.asarray(dense_geometry.distance),
            strict=True,
        )
        if distance < 0.7
    }
    metric_pairs = {
        tuple(sorted((int(left), int(right))))
        for left, right, valid in zip(
            np.asarray(metric_state.pair_relation.left_particle_ids),
            np.asarray(metric_state.pair_relation.right_particle_ids),
            np.asarray(metric_state.pair_relation.valid),
            strict=True,
        )
        if valid
    }
    assert metric_pairs == dense_pairs


def test_verlet_cell_deformation_enters_rebuild_certificate():
    cell = _cell()
    particles = phx.discretization.ParticleSetPlan(
        [0, 1], [1.0, 1.0], ambient_dimension=3
    ).prepare()
    base = phx.discretization.DenseParticleNeighborhoodPlan(1, box=cell)
    verlet = phx.discretization.VerletParticleNeighborhoodPlan(base, 0.8, 0.2).prepare(
        particles
    )
    positions = cell.cartesian(jnp.asarray([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]]))
    state = verlet.initialize(positions, cell_vectors=cell.vectors)
    deformed = cell.vectors.at[0, 0].add(0.15)
    updated = verlet.update(positions, state, cell_vectors=deformed)
    assert bool(updated.rebuilt)
    assert float(updated.maximum_cell_deformation) > 0.1


def test_cell_stress_is_finite_symmetric_energy_derivative():
    cell = _cell()
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        [0, 1], [1, 1], [1.0, 1.0], units, atom_type_ids=[0, 0], cell=cell
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1, box=cell).prepare(
        system.particles
    )
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([1.0], [1.0], 1.2)]
    ).prepare(system)
    fractional = jnp.asarray([[0.1, 0.1, 0.1], [0.4, 0.1, 0.1]])
    positions = cell.cartesian(fractional)
    relation = neighborhood.build(positions)
    result = phx.atomistic.atomistic_cell_energy_and_stress(
        potential, fractional, relation
    )
    assert bool(result.successful)
    assert bool(jnp.all(jnp.isfinite(result.stress)))
    np.testing.assert_allclose(result.stress, result.stress.T, atol=1.0e-12)


def test_periodic_learned_graph_execution_is_explicit_and_finite():
    model_units = (
        phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    )
    units = phx.atomistic.AtomisticUnitSystem(
        model_units.scale,
        mass_unit=KILOGRAM,
        time_unit=SECOND,
        charge_unit=COULOMB,
        temperature_unit=KELVIN,
        constant_set_id="codata-2018",
    )
    cell = phx.discretization.PeriodicCell(5.0 * jnp.eye(3))
    system = phx.atomistic.AtomisticSystemPlan(
        [0, 1],
        [1, 1],
        [1.0, 1.0],
        units,
        atom_type_ids=[1, 1],
        cell=cell,
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1, box=cell).prepare(
        system.particles
    )
    model = phx.nn.atomistic.PaiNNPotential(
        model_units.scale,
        cutoff=2.0,
        feature_count=4,
        interaction_count=1,
        radial_basis_count=3,
        key=jr.key(77),
    )
    program = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LearnedGraphPotentialTerm(model, allow_periodic=True)]
    ).prepare(
        system,
        graph_execution=phx.atomistic.AtomisticGraphExecutionPlan(1, backend="particle"),
    )
    positions = jnp.asarray([[0.2, 0.2, 0.2], [4.4, 0.2, 0.2]])
    relation = neighborhood.build(positions)
    result = program.evaluate(positions, relation, species=system.plan.atomic_numbers)
    assert bool(result.successful)
    assert bool(jnp.isfinite(result.energy))
