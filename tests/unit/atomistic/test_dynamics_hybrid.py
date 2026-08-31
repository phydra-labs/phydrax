import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _program():
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        [10, 20],
        [1, 1],
        [1.0, 2.0],
        units,
        atom_type_ids=[0, 1],
        region_ids=[1, 0],
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
        system.particles
    )
    lj = phx.atomistic.LennardJonesPotential(
        [0.5, 1.0], [1.0, 1.2], 2.5, name="lj", force_group=0
    )
    program = phx.atomistic.AtomisticPotentialProgram([lj]).prepare(system)
    return system, neighborhood, lj, program


def test_alchemical_and_region_masked_composition_are_energy_derived():
    system, neighborhood, lj, program = _program()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]])
    relation = neighborhood.build(positions)
    full = program.evaluate(positions, relation, species=system.plan.atom_type_ids)
    alchemical = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.AlchemicalScaledPotential(lj)]
    ).prepare(system)
    half = alchemical.evaluate(
        positions,
        relation,
        species=system.plan.atom_type_ids,
        alchemical_lambda=0.5,
    )
    np.testing.assert_allclose(half.energy, 0.5 * full.energy, atol=1.0e-12)
    masked = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.RegionMaskedPotential(lj, 1)]
    ).prepare(system)
    masked_value = masked.evaluate(positions, relation, species=system.plan.atom_type_ids)
    assert bool(masked_value.successful)
    assert bool(jnp.isfinite(masked_value.energy))


def test_force_group_and_semigrand_transition_are_typed():
    system, neighborhood, _, program = _program()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]])
    relation = neighborhood.build(positions)
    group = phx.atomistic.evaluate_force_group(
        program,
        0,
        positions,
        relation,
        species=system.plan.atom_type_ids,
    )
    assert bool(group.successful)
    transition = phx.atomistic.variance_constrained_semigrand_step(
        program,
        positions,
        relation,
        system.plan.atom_type_ids,
        jax.random.key_data(jax.random.key(12)),
        0,
        phx.atomistic.VarianceConstrainedSemiGrandPlan(1.0, [0.0, 0.0], [0.5, 0.5], 1.0),
    )
    assert bool(transition.successful)
    assert transition.species.shape == (2,)


def test_one_bead_ring_polymer_has_zero_spring_and_finite_step():
    system, neighborhood, _, program = _program()
    ring = phx.atomistic.PreparedRingPolymerDynamics(
        phx.atomistic.RingPolymerPlan(1, 1.0, 1.0e-4),
        program,
        neighborhood,
    )
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]])
    state = ring.initialize_state(
        positions, velocity=jnp.zeros_like(positions), key=jax.random.key(13)
    )
    np.testing.assert_allclose(state.spring_energy, 0.0, atol=0.0)
    step = ring.step(state)
    assert bool(step.successful)
    assert bool(jnp.all(jnp.isfinite(step.estimators.centroid)))
    assert float(step.estimators.radius_of_gyration) == 0.0


def test_born_oppenheimer_provider_boundary_advances_conservative_state():
    system, _, _, _ = _program()

    def evaluator(prepared, positions, cell_vectors):
        del prepared, cell_vectors
        energy = 0.5 * jnp.sum(positions * positions)
        return phx.atomistic.ExternalAtomisticEvaluation(
            energy,
            -positions,
            None,
            jnp.asarray(True),
            "harmonic-electronic-surface",
        )

    provider = phx.atomistic.CallableBornOppenheimerProvider(
        evaluator, "harmonic-electronic-surface"
    )
    dynamics = phx.atomistic.BornOppenheimerVelocityVerletPlan(system, provider, 1.0e-3)
    positions = jnp.asarray([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    state = dynamics.initialize(positions, velocity=jnp.zeros_like(positions))
    step = dynamics.step(state)
    assert bool(step.successful)
    assert int(step.state.step_index) == 1
    assert bool(jnp.isfinite(step.state.energy))
