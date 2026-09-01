import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def test_periodic_fof_unbinding_properties_and_merger_matching():
    ids = jnp.arange(6)
    positions = jnp.asarray(
        [
            [0.10, 0.10, 0.10],
            [0.12, 0.10, 0.10],
            [0.14, 0.10, 0.10],
            [0.70, 0.70, 0.70],
            [0.72, 0.70, 0.70],
            [0.74, 0.70, 0.70],
        ]
    )
    velocities = jnp.zeros_like(positions)
    masses = jnp.ones((6,))
    fof = cosmology.PeriodicFoFFinderPlan((1.0, 1.0, 1.0), 0.05, 4).find(
        ids, positions, velocities, masses, jnp.ones((6,), dtype=bool)
    )
    assert bool(fof.successful)
    assert int(jnp.sum(fof.group_active)) == 2
    np.testing.assert_array_equal(jnp.sort(fof.group_counts[fof.group_active]), [3, 3])

    mask = fof.group_labels == fof.group_labels[0]
    unbound = cosmology.DirectHaloUnbindingPlan(1.0, softening=0.01).unbind(
        positions, velocities, masses, mask
    )
    assert bool(unbound.successful)
    assert jnp.sum(unbound.bound_mask) == 3
    properties = cosmology.HaloPropertyPlan(1.0).evaluate(
        positions,
        velocities,
        masses,
        unbound.bound_mask,
        fof.group_positions[0],
    )
    assert bool(properties.successful)

    substructure = cosmology.DensityPeakSubstructurePlan(2).identify(
        positions, masses, mask
    )
    assert bool(substructure.successful)
    source = jnp.asarray([[1, 2, 3, -1], [4, 5, 6, -1]])
    rank = jnp.asarray([[0, 1, 2, 99], [0, 1, 2, 99]])
    target = jnp.asarray([[1, 2, 7, -1], [4, 5, 8, -1]])
    match = cosmology.ParticleCoreOverlapTreePlan(3, 2).match(source, rank, target)
    np.testing.assert_array_equal(match.descendant_indices, [0, 1])


def test_star_formation_and_stochastic_feedback_are_replayable_and_conservative():
    population_plan = cosmology.CosmologicalPopulationPlan(4, 3)
    population = population_plan.empty()
    star_formation = cosmology.StochasticStarFormationPlan(
        star_mass=0.5, maximum_events=2
    )
    gas_mass = jnp.asarray([1.0, 1.0])
    gas_momentum = jnp.asarray([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    gas_energy = jnp.asarray([2.0, 2.0])
    result = star_formation.apply(
        population,
        gas_mass,
        gas_momentum,
        gas_energy,
        jnp.asarray([0.01, 0.02]),
        jnp.asarray([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]]),
        jnp.asarray([True, False]),
        0.5,
        jax.random.key(7),
        3,
    )
    replay = star_formation.apply(
        population,
        gas_mass,
        gas_momentum,
        gas_energy,
        jnp.asarray([0.01, 0.02]),
        jnp.asarray([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]]),
        jnp.asarray([True, False]),
        0.5,
        jax.random.key(7),
        3,
    )
    assert bool(result.successful)
    np.testing.assert_allclose(result.gas_masses, replay.gas_masses)
    np.testing.assert_allclose(result.mass_defect, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.momentum_defect, 0.0, atol=1e-12)

    population_with_energy = eqx.tree_at(
        lambda state: state.energy_reservoirs,
        result.population,
        jnp.where(result.population.active_mask, 2.0, 0.0),
    )
    feedback = cosmology.StochasticThermalFeedbackPlan(
        heating_energy_per_mass=0.5, maximum_events=2
    ).apply(
        population_with_energy,
        result.gas_masses,
        result.gas_energies,
        jnp.asarray([[0, 1], [0, 1], [0, 1], [0, 1]]),
        jax.random.key(11),
        4,
    )
    assert bool(feedback.successful)
    np.testing.assert_allclose(feedback.energy_defect, 0.0, atol=1e-12)
