import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.cosmology._coupled import ComovingEulerPlan
from phydrax.applications.cosmology._mixed_matter import (
    WaveParticleGasCosmologyPlan,
    WaveParticleGasCosmologyState,
)
from phydrax.applications.cosmology._particles import CosmologicalKDKPlan
from phydrax.applications.cosmology._wave_dark_matter import (
    WaveDarkMatterPlan,
    WaveDarkMatterStepPolicy,
)


def _linear_mixed_workflow(count=16):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem(1)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "linear-mixed-workflow",
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
    gravity_owner = phx.solver.NewtonianSelfGravityPlan(0.02).prepare(
        phx.solver.prepare_balance_law_transport(runtime)
    )
    particle_support = phx.discretization.ParticleSetPlan(
        jnp.arange(count),
        jnp.full((count,), 1.0 / count),
        ambient_dimension=1,
    ).prepare()
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(particle_support)
    particle_gravity = phx.solver.ParticleMeshGravityPlan(gravity_owner, transfer)
    kdk = CosmologicalKDKPlan(particle_support, (1.0,))

    half_cell = 0.5 / count
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
        axis_names=("x",),
        field_name="psi",
    ).prepare((phx.discretization.AxisDomain.periodic(half_cell, 1.0 + half_cell),))
    background = phx.applications.cosmology.FLRWBackground(1.0, 1.0)
    schedule = jnp.asarray([0.5, 0.50001, 0.50002])
    wave = WaveDarkMatterPlan(
        1.0,
        schedule,
        gravitational_constant=0.02,
        reduced_planck_constant=0.03,
        step_policy=WaveDarkMatterStepPolicy(
            maximum_phase_radians=2.0,
            minimum_de_broglie_cells=2.0,
            norm_relative_tolerance=1.0e-8,
        ),
    ).prepare(space, background)
    gas = ComovingEulerPlan(
        dynamics,
        adiabatic_index=5.0 / 3.0,
        expansion_dimension=3,
        substeps=8,
    )
    prepared = WaveParticleGasCosmologyPlan(wave, kdk, gas, particle_gravity).prepare()

    x = space.axes[0].nodes
    amplitude = jnp.asarray(1.0e-3)
    wave_state = wave.initialize(
        jnp.sqrt(1.0 + amplitude * jnp.cos(2.0 * jnp.pi * x)).astype(jnp.complex128)
    )
    particle_positions = (x - amplitude * jnp.sin(2.0 * jnp.pi * x) / (2.0 * jnp.pi))[
        :, None
    ]
    particle_state = kdk.initialize(
        particle_positions,
        jnp.zeros_like(particle_positions),
        schedule[0],
    )
    gas_density = 1.0 + amplitude * jnp.cos(2.0 * jnp.pi * x)
    gas_average = jnp.stack(
        (gas_density, jnp.zeros_like(gas_density), gas_density), axis=-1
    )
    gas_state = gas.initialize(gas_average, schedule[0])
    state = WaveParticleGasCosmologyState(wave_state, particle_state, gas_state)
    return prepared, state


def test_linear_mixed_workflow_closes_shared_source_force_work_and_time_levels():
    prepared, state = _linear_mixed_workflow()
    initial_assembly = prepared.density.assemble(state)

    result = prepared.rollout(state)

    assert bool(result.successful)
    assert int(result.diagnostics.accepted_steps) == 2
    assert bool(jnp.all(result.diagnostics.accepted))
    assert bool(jnp.all(result.diagnostics.time_level_consistent))
    assert bool(jnp.all(result.diagnostics.density_nonnegative))
    assert bool(jnp.all(result.diagnostics.gas_density_positive))
    assert bool(jnp.all(result.diagnostics.gas_pressure_positive))
    assert bool(jnp.all(result.diagnostics.wave_norm_conserved))
    assert jnp.max(jnp.abs(result.diagnostics.source_integral)) < 2e-12
    assert jnp.max(result.diagnostics.poisson_relative_residual) < 2e-10
    assert jnp.max(result.diagnostics.gauge_defect) < 2e-12
    assert jnp.max(jnp.abs(result.diagnostics.mass_balance_defect)) < 2e-11
    assert jnp.all(jnp.isfinite(result.diagnostics.component_force))
    assert jnp.all(jnp.isfinite(result.diagnostics.component_gravity_work))
    assert jnp.all(jnp.isfinite(result.diagnostics.total_gravity_work))
    final_assembly = prepared.density.assemble(result.state)
    np.testing.assert_allclose(
        final_assembly.component_mass,
        initial_assembly.component_mass,
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        result.state.wave.scale_factor,
        prepared.plan.wave.scale_factors[-1],
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.state.particles.scale_factor,
        prepared.plan.wave.scale_factors[-1],
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.state.gas.scale_factor,
        prepared.plan.wave.scale_factors[-1],
        atol=0.0,
    )
