import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _system(*, cell=None):
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    return phx.atomistic.AtomisticSystemPlan(
        [0, 1],
        [1, 1],
        [1.0, 1.0],
        units,
        atom_type_ids=[0, 0],
        charges=[1.0, -1.0],
        cell=cell,
    ).prepare()


def test_direct_coulomb_matches_two_charge_reference():
    system = _system()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
        system.particles
    )
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.DirectCoulombPotential()]
    ).prepare(system)
    positions = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    state = neighborhood.build(positions)
    evaluation = potential.evaluate(positions, state)
    np.testing.assert_allclose(evaluation.energy, -0.5, atol=1.0e-12)
    np.testing.assert_allclose(jnp.sum(evaluation.forces, axis=0), 0.0, atol=1.0e-12)


def test_pme_is_finite_and_tracks_direct_ewald_reference():
    cell = phx.discretization.ParticleCell(6.0 * jnp.eye(3))
    system = _system(cell=cell)
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1, box=cell).prepare(
        system.particles
    )
    positions = jnp.asarray([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
    relation = neighborhood.build(positions)
    reference = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.EwaldReferencePotential(0.8, 2.5, 5)]
    ).prepare(system)
    pme = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.ParticleMeshEwaldPotential(0.8, 2.5, (16, 16, 16))]
    ).prepare(system)
    expected = reference.evaluate(positions, relation)
    observed = pme.evaluate(positions, relation)
    assert bool(expected.successful)
    assert bool(observed.successful)
    np.testing.assert_allclose(observed.energy, expected.energy, rtol=2e-1, atol=2e-1)
    np.testing.assert_allclose(jnp.sum(observed.forces, axis=0), 0.0, atol=1.0e-3)


def test_isotropic_barostat_produces_typed_detailed_balance_move():
    cell = phx.discretization.ParticleCell(5.0 * jnp.eye(3))
    system = _system(cell=cell)
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
        system.particles
    )
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.0)]
    ).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.VelocityVerletPlan(1.0e-3),
    ).prepare()
    positions = jnp.asarray([[1.0, 1.0, 1.0], [2.2, 1.0, 1.0]])
    state = dynamics.initialize_state(
        positions, velocity=jnp.zeros_like(positions), key=jax.random.key(11)
    )
    move = phx.atomistic.apply_isotropic_monte_carlo_barostat(
        dynamics,
        state,
        phx.atomistic.IsotropicMonteCarloBarostatPlan(0.1, 1.0, 0.05),
        0,
    )
    assert bool(move.successful)
    assert bool(jnp.isfinite(move.log_acceptance_probability))
    assert float(move.volume_after) > 0.0
    assert move.accepted_state.cell_vectors.shape == (3, 3)


def test_pme_supports_isotropic_npt_energy_re_evaluation():
    cell = phx.discretization.ParticleCell(6.0 * jnp.eye(3))
    system = _system(cell=cell)
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
        system.particles
    )
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.ParticleMeshEwaldPotential(0.8, 2.5, (8, 8, 8))]
    ).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.VelocityVerletPlan(1.0e-3),
    ).prepare()
    positions = jnp.asarray([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
    state = dynamics.initialize_state(
        positions, velocity=jnp.zeros_like(positions), key=jax.random.key(14)
    )
    move = phx.atomistic.apply_isotropic_monte_carlo_barostat(
        dynamics,
        state,
        phx.atomistic.IsotropicMonteCarloBarostatPlan(0.0, 1.0, 0.01),
        0,
    )
    assert bool(move.successful)
    assert bool(jnp.isfinite(move.energy_after))
