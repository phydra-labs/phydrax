#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _particles():
    return phx.discretization.ParticleSetPlan(
        [0, 1, -1],
        [1.0, 2.0, np.nan],
        ambient_dimension=2,
        active_mask=[True, True, False],
    ).prepare()


def test_summation_density_state_layout_round_trips_and_masks_padding():
    layout = phx.discretization.WeaklyCompressibleSPHStateLayout(
        _particles(), density_evolved=False
    )
    position = jnp.asarray([[0.0, 0.1], [0.2, 0.3], [jnp.nan, jnp.nan]])
    velocity = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [jnp.nan, jnp.nan]])
    state = layout.pack(position, velocity)
    position_, velocity_, density_ = layout.unpack(state)

    assert layout.shape == (3, 4)
    assert state.shape == layout.shape
    assert jnp.allclose(position_[:2], position[:2])
    assert jnp.allclose(velocity_[:2], velocity[:2])
    assert jnp.array_equal(position_[2], jnp.zeros((2,)))
    assert jnp.array_equal(velocity_[2], jnp.zeros((2,)))
    assert density_ is None
    assert layout.state_geometry_id.startswith("state-geometry:wcsph:")
    with pytest.raises(ValueError, match="does not accept density"):
        layout.pack(position, velocity, jnp.ones((3,)))
    with pytest.raises(ValueError, match="no density component"):
        layout.density(state)


def test_continuity_density_state_layout_round_trips_density_and_rates():
    layout = phx.discretization.WeaklyCompressibleSPHStateLayout(
        _particles(), density_evolved=True
    )
    position = jnp.asarray([[0.0, 0.1], [0.2, 0.3], [jnp.nan, jnp.nan]])
    velocity = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [jnp.nan, jnp.nan]])
    density = jnp.asarray([1.0, 1.2, jnp.nan])
    state = layout.pack(position, velocity, density)
    position_, velocity_, density_ = layout.unpack(state)
    rate = layout.pack_rate(
        velocity_,
        jnp.zeros_like(velocity_),
        jnp.asarray([0.1, -0.2, jnp.nan]),
    )

    assert layout.shape == (3, 5)
    assert jnp.allclose(density_[:2], density[:2])
    assert density_[2] == pytest.approx(1.0)
    assert jnp.array_equal(rate[2], jnp.zeros((5,)))
    assert jnp.allclose(layout.validate(state), state)
    with pytest.raises(ValueError, match="requires density"):
        layout.pack(position, velocity)
    with pytest.raises(Exception, match="finite and positive"):
        layout.pack(
            position, velocity, jnp.asarray([1.0, 0.0, jnp.nan])
        ).block_until_ready()


def test_wcsph_state_layout_rejects_wrong_shapes():
    layout = phx.discretization.WeaklyCompressibleSPHStateLayout(
        _particles(), density_evolved=True
    )
    with pytest.raises(ValueError, match="position must have shape"):
        layout.pack(jnp.zeros((2, 2)), jnp.zeros((3, 2)), jnp.ones((3,)))
    with pytest.raises(ValueError, match="WCSPH state must have shape"):
        layout.unpack(jnp.zeros((3, 4)))
    with pytest.raises(ValueError, match="density rate"):
        layout.pack_rate(jnp.zeros((3, 2)), jnp.zeros((3, 2)), jnp.zeros((2,)))


def test_runtime_sph_source_emits_atomically_with_exact_ledgers():
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3),
        jnp.ones((3,)),
        ambient_dimension=2,
    ).prepare()
    population_plan = phx.discretization.ParticlePopulationPlan(particles)
    source = phx.discretization.SPHParticleSourcePlan(
        population_plan,
        jnp.asarray(((0.0, 0.0),)),
        jnp.asarray(((1.0, 0.0),)),
        jnp.asarray((1.0,)),
        jnp.asarray((((0.5, 0.0),),)),
        jnp.asarray((1.0,)),
        domain_lower=jnp.asarray((-1.0, -1.0)),
        domain_upper=jnp.asarray((2.0, 1.0)),
        minimum_source_clearance=0.0,
        minimum_particle_clearance=0.25,
        minimum_wall_clearance=0.0,
        maximum_emissions_per_site=1,
        density_initialization="continuity",
        replay_policy=phx.solver.HybridReplayPolicy(1),
    )
    population = population_plan.initialize(
        active_mask=jnp.asarray((True, False, False)),
        masses=jnp.asarray((1.0, 0.0, 0.0)),
    )
    runtime = source.initialize_runtime(
        jnp.asarray(((-0.5, 0.0), (0.0, 0.0), (0.0, 0.0))),
        jnp.zeros((3, 2)),
        population=population,
        evolved_density=jnp.asarray((1.0, 0.0, 0.0)),
    )
    result = phx.discretization.emit_sph_particles(
        source,
        runtime,
        phx.discretization.SPHEmissionRequest(
            jnp.asarray((1.0,)),
            jnp.asarray(((2.0, 0.0),)),
            jnp.asarray((1.5,)),
            jnp.asarray((3.0,)),
            jnp.asarray((True,)),
        ),
        jnp.asarray(1.0),
    )

    assert bool(result.successful)
    assert int(result.inserted_count) == 1
    assert int(result.event_tape.event_count) == 1
    assert jnp.isclose(result.accepted_state.source.cumulative_injected_mass, 1.0)
    assert jnp.allclose(
        result.accepted_state.source.cumulative_injected_momentum,
        jnp.asarray((2.0, 0.0)),
    )
    assert jnp.isclose(result.accepted_state.source.cumulative_barotropic_energy, 3.0)
