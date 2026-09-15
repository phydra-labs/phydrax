import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def _workflow():
    positions = jnp.asarray(
        [(x, y, z) for x in (0.25, 0.75) for y in (0.25, 0.75) for z in (0.25, 0.75)]
    )
    count = positions.shape[0]
    mass = jnp.full((count,), 1.0 / count)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), mass, ambient_dimension=3
    ).prepare()
    axes = tuple(
        phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(3)
    )
    grid = phx.discretization.TensorGridPlan(axes, axis_names=("x", "y", "z")).prepare(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    )
    system = phx.equations.EulerSystem(3)
    finite_volume = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "homogeneous-sidm-workflow",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(grid.axis_names),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        finite_volume,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.HLLCFluxPlan(),
        ),
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.3, maximum_retries=0),
    )
    gravity = phx.solver.ParticleMeshGravityPlan(
        phx.solver.NewtonianSelfGravityPlan(0.01).prepare(
            phx.solver.prepare_balance_law_transport(runtime)
        ),
        phx.discretization.ParticleGridSplatPlan(grid).prepare(particles),
    )
    kdk = cosmology.CosmologicalKDKPlan(particles, (1.0, 1.0, 1.0))
    particle_mesh = cosmology.CosmologicalParticleMeshPlan(kdk, gravity, (0.5, 0.55))
    box = phx.discretization.ParticleBox(
        jnp.zeros((3,)), jnp.ones((3,)), periodic_axes=(True, True, True)
    )
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        count * (count - 1) // 2, box=box
    ).prepare(particles)
    sidm = cosmology.CosmologicalSIDMPlan(
        particle_mesh,
        neighborhood,
        phx.discretization.CoupledSummationSmoothingLengthPlan(
            1.0,
            1.0e-3,
            2.0,
            maximum_iterations=80,
            tolerance=1.0e-6,
            relaxation=0.7,
        ),
        phx.discretization.WendlandC2SPHKernel(3),
        cosmology.SIDMCrossSectionPlan(0.01),
        cosmology.SIDMCollisionPolicy(
            maximum_pair_probability=0.1,
            maximum_particle_probability=0.25,
            minimum_knudsen_number=1.0e-3,
            maximum_events_per_half_step=count // 2,
        ),
    )
    velocity = jnp.asarray(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (0.5, -0.5, 0.0),
            (-0.5, 0.5, 0.0),
        )
    )
    state = kdk.initialize(positions, mass[:, None] * 0.5 * velocity, 0.5)
    return sidm, state


def test_homogeneous_rare_scattering_rate_and_split_rollout_are_conservative():
    sidm, state = _workflow()
    sample_count = 256
    keys = jr.split(jr.key(2026), sample_count)
    epochs = jnp.arange(sample_count, dtype=jnp.int32)

    def sample(local_key, epoch):
        collision = sidm.collide(state, local_key, epoch, 1.0)
        return (
            collision.diagnostics.event_count,
            collision.diagnostics.total_momentum_defect,
            collision.diagnostics.total_kinetic_energy_defect,
            collision.successful,
        )

    counts, momentum_defects, energy_defects, successful = jax.jit(jax.vmap(sample))(
        keys, epochs
    )
    reference = sidm.collide(state, jr.key(0), 0, 1.0)
    expected = sample_count * jnp.sum(reference.diagnostics.pair_probability)
    observed = jnp.sum(counts)

    assert jnp.all(successful)
    assert expected > 5.0
    assert jnp.abs(observed - expected) <= 6.0 * jnp.sqrt(expected) + 2.0
    np.testing.assert_allclose(momentum_defects, 0.0, atol=2e-14)
    np.testing.assert_allclose(energy_defects, 0.0, atol=2e-14)

    evolved = sidm.rollout(cosmology.FLRWBackground(1.0, 0.3), state, jr.key(99))
    assert bool(evolved.successful)
    assert int(evolved.diagnostics.accepted_steps) == 1
    total_before = jnp.sum(state.canonical_momenta, axis=0)
    total_after = jnp.sum(evolved.state.canonical_momenta, axis=0)
    np.testing.assert_allclose(total_after, total_before, atol=2e-12)
