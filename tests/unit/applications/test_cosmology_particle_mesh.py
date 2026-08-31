import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


cosmology = phx.applications.cosmology


def _gravity(count, particles):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem(1)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "cosmology-pm-test",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(grid.axis_names),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
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
    gravity = phx.solver.NewtonianSelfGravityPlan(0.01).prepare(
        phx.solver.prepare_balance_law_transport(runtime)
    )
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    return phx.solver.ParticleMeshGravityPlan(gravity, transfer)


def _case(count=8):
    scale = cosmology.CosmologyScaleContract("L", "M", "T")
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count),
        jnp.full((count,), 1.0 / count),
        ambient_dimension=1,
    ).prepare()
    gravity = _gravity(count, particles)
    kdk = cosmology.CosmologicalKDKPlan(particles, (1.0,), scale=scale)
    rollout = cosmology.CosmologicalParticleMeshPlan(
        kdk, gravity, jnp.asarray([0.5, 0.55, 0.6])
    )
    background = cosmology.FLRWBackground(1.0, 0.3, scale=scale)
    positions = ((jnp.arange(count) + 0.5) / count)[:, None]
    state = kdk.initialize(positions, jnp.zeros_like(positions), 0.5)
    return background, particles, gravity, kdk, rollout, state


def test_uniform_lattice_has_zero_force_and_completed_rollout():
    background, _, gravity, _, rollout, state = _case()
    force = gravity.acceleration(state.positions)
    assert isinstance(force, phx.solver.ParticleMeshGravityForceResult)
    np.testing.assert_allclose(force.acceleration, 0.0, atol=1e-12)
    assert bool(force.successful)
    result = rollout.rollout(background, state)
    assert bool(result.successful)
    assert int(result.diagnostics.accepted_steps) == 2
    np.testing.assert_allclose(result.state.positions, state.positions, atol=1e-12)
    np.testing.assert_allclose(result.state.canonical_momenta, 0.0, atol=1e-12)
    assert result.diagnostics.maximum_mass_balance_defect < 1e-12
    assert result.diagnostics.maximum_net_force_norm < 1e-12


def test_rollout_is_piecewise_differentiable_away_from_cell_boundaries():
    background, _, _, kdk, rollout, state = _case()
    pattern = jnp.sin(2.0 * jnp.pi * state.positions)

    def objective(amplitude):
        displaced = jnp.mod(state.positions + amplitude * pattern, 1.0)
        initial = kdk.initialize(displaced, jnp.zeros_like(displaced), 0.5)
        final = rollout.rollout(background, initial).state.positions
        return jnp.sum(final**2)

    amplitude = jnp.asarray(1.0e-3)
    value, tangent = jax.jvp(objective, (amplitude,), (jnp.asarray(1.0),))
    epsilon = 1.0e-5
    finite_difference = (
        objective(amplitude + epsilon) - objective(amplitude - epsilon)
    ) / (2.0 * epsilon)
    assert jnp.isfinite(value)
    np.testing.assert_allclose(tangent, finite_difference, rtol=5e-3, atol=1e-6)


def test_kdk_wraps_periodically_and_uses_particle_mass_authority():
    background, particles, _, kdk, _, _ = _case(count=2)
    positions = jnp.asarray([[0.99], [0.25]])
    momentum = particles.safe_masses[:, None] * jnp.asarray([[1.0], [0.0]])
    state = kdk.initialize(positions, momentum, 0.5)
    advanced, diagnostics = kdk.advance(
        background,
        state,
        0.6,
        jnp.zeros_like(positions),
        jnp.zeros_like(positions),
    )
    assert bool(diagnostics.successful)
    assert jnp.all((advanced.positions >= 0.0) & (advanced.positions < 1.0))
    assert "masses" not in type(advanced).__annotations__


def test_cosmological_pm_rejects_dual_particle_or_geometry_authority():
    _, particles, gravity, _, _, _ = _case()
    other = phx.discretization.ParticleSetPlan(
        jnp.arange(particles.capacity),
        jnp.full((particles.capacity,), 1.0 / particles.capacity),
        ambient_dimension=1,
        plan_id="different-particles",
    ).prepare()
    other_kdk = cosmology.CosmologicalKDKPlan(other, (1.0,))
    with pytest.raises(ValueError, match="share one particle support"):
        cosmology.CosmologicalParticleMeshPlan(other_kdk, gravity, [0.5, 0.6])
    with pytest.raises(ValueError, match="increasing"):
        cosmology.CosmologicalParticleMeshPlan(
            cosmology.CosmologicalKDKPlan(particles, (1.0,)),
            gravity,
            [0.5, 0.5],
        )


def test_periodic_pm_rejects_spatial_curvature():
    background, _, _, _, rollout, state = _case()
    curved = cosmology.FLRWBackground(
        background.hubble_constant,
        background.matter_density,
        curvature_density=0.01,
        scale=background.scale,
    )
    with pytest.raises((ValueError, RuntimeError), match="zero spatial curvature"):
        jax.block_until_ready(rollout.rollout(curved, state).state.positions)
