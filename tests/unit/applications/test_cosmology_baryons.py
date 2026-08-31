import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def _case(count=8):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem(1)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "cosmology-baryon-test",
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
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count),
        jnp.full((count,), 1.0 / count),
        ambient_dimension=1,
    ).prepare()
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    particle_gravity = phx.solver.ParticleMeshGravityPlan(gravity, transfer)
    scale = cosmology.CosmologyScaleContract("L", "M", "T")
    kdk = cosmology.CosmologicalKDKPlan(particles, (1.0,), scale=scale)
    gas = cosmology.ComovingEulerPlan(
        dynamics,
        adiabatic_index=5.0 / 3.0,
        expansion_dimension=3,
        substeps=8,
    )
    plan = cosmology.CosmologicalGasParticleGravityPlan(
        gas, kdk, particle_gravity, [0.5, 0.51]
    )
    background = cosmology.FLRWBackground(1.0, 1.0, scale=scale)
    gas_average = jnp.zeros((count, 3)).at[:, 0].set(1.0).at[:, 2].set(1.0)
    gas_state = gas.initialize(gas_average, 0.5)
    positions = ((jnp.arange(count) + 0.5) / count)[:, None]
    particle_state = kdk.initialize(positions, jnp.zeros_like(positions), 0.5)
    state = cosmology.CosmologicalGasParticleState(gas_state, particle_state)
    return background, gas, plan, state


def test_uniform_comoving_euler_has_adiabatic_expansion_scaling():
    background, gas, _, state = _case()
    zero = jnp.zeros((state.gas.cell_average.shape[0], 1))
    evolved, diagnostics = gas.advance(background, state.gas, 0.51, zero, zero)
    expected_energy = (0.51 / 0.5) ** -2.0
    assert bool(diagnostics.successful)
    np.testing.assert_allclose(evolved.cell_average[:, 0], 1.0, rtol=1e-12)
    np.testing.assert_allclose(evolved.cell_average[:, 2], expected_energy, rtol=2e-5)


def test_uniform_gas_particle_epoch_uses_shared_zero_force_and_commits_atomically():
    background, _, plan, state = _case()
    shared = plan.shared_gravity(state)
    assert bool(shared.successful)
    np.testing.assert_allclose(shared.cell_acceleration, 0.0, atol=1e-12)
    np.testing.assert_allclose(shared.particle_acceleration, 0.0, atol=1e-12)
    result = plan.rollout(background, state)
    assert bool(result.successful)
    assert int(result.diagnostics.accepted_steps) == 1
    np.testing.assert_allclose(
        result.state.particles.positions, state.particles.positions, atol=1e-12
    )
    assert result.diagnostics.mass_balance_defect[0] < 1e-12
