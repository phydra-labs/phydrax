#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _wcsph(name, ids, positions, *, density_reference=1.0):
    count = len(ids)
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray(ids),
        jnp.full((count,), density_reference / count),
        ambient_dimension=1,
        name=name,
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    method = phx.discretization.WeaklyCompressibleSPHMethodPlan(
        phx.discretization.WendlandC2SPHKernel(1),
        1.25 / count,
        density=phx.discretization.ContinuityDensityPlan(),
    )
    compiled = phx.equations.compile_weakly_compressible_sph_problem(
        phx.equations.WeaklyCompressibleFluidProblemIR(
            name,
            phx.equations.TaitBarotropicMaterial(density_reference, 2.0),
        ),
        particles,
        method,
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(
            count * (count - 1) // 2, box=box
        ),
    )
    return compiled, jnp.asarray(positions)[:, None], box


def test_transport_velocity_runs_through_fixed_step_substrate():
    compiled, position, _ = _wcsph("transport", range(6), (jnp.arange(6) + 0.5) / 6.0)
    transport = phx.discretization.PreparedTransportVelocityDynamics(
        compiled.dynamics,
        phx.discretization.TransportVelocitySPHMethodPlan(1.0),
    )
    initial = transport.initialize_state(position, jnp.zeros_like(position))
    method = phx.solver.TransportVelocityFixedStepMethod(transport)
    solution = phx.solver.solve_fixed_step(
        phx.solver.FixedStepProblem(
            method,
            initial,
            t0=0.0,
            t1=0.002,
            step_size=0.001,
        )
    )
    diagnostics = transport.diagnostics(0.0, solution.states[-1], None)

    assert solution.successful
    assert jnp.all(jnp.isfinite(solution.states))
    assert jnp.isfinite(diagnostics.background_acceleration_norm)


def test_multiphase_interface_is_reciprocal_and_compiles_flat_state():
    first, first_position, box = _wcsph("phase-a", range(4), (jnp.arange(4) + 0.25) / 4.0)
    second, second_position, _ = _wcsph("phase-b", range(4), (jnp.arange(4) + 0.75) / 4.0)
    phase_a = phx.discretization.PhaseDefinition("phase-a", first.dynamics)
    phase_b = phx.discretization.PhaseDefinition("phase-b", second.dynamics)
    relation = phx.discretization.DenseBipartiteParticleNeighborhoodPlan(16).prepare(
        first.dynamics.particles,
        second.dynamics.particles,
        target_population_id=phase_a.phase_id,
        source_population_id=phase_b.phase_id,
    )
    dynamics = phx.discretization.PreparedMultiphaseWCSPHDynamics(
        phase_a,
        phase_b,
        phx.discretization.MultiphaseWCSPHPlan(surface_tension=0.01),
        relation,
        box=box,
    )
    state_a = first.initialize_state(first_position, jnp.zeros_like(first_position))
    state_b = second.initialize_state(second_position, jnp.zeros_like(second_position))
    state = dynamics.pack(state_a, state_b)
    rate = dynamics(0.0, state, None)
    diagnostics = dynamics.diagnostics(0.0, state, None)

    assert rate.shape == state.shape
    assert jnp.allclose(diagnostics.total_momentum_rate, 0.0, atol=1e-13)
    assert diagnostics.interface_pair_count > 0


def test_iisph_and_dfsph_fixed_steps_return_complete_projection_status():
    count = 6
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), spacing), ambient_dimension=1
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        count * (count - 1) // 2, box=box
    ).prepare(particles)
    kernel = phx.discretization.WendlandC2SPHKernel(1)
    position = (jnp.arange(count, dtype=float) + 0.5)[:, None] * spacing
    velocity = jnp.zeros_like(position)

    iisph = phx.discretization.PreparedIISPH(
        particles,
        neighborhood,
        kernel,
        1.25 * spacing,
        phx.discretization.IISPHMethodPlan(1.0, maximum_iterations=2, tolerance=1.0),
    )
    iisph_state = iisph.initialize_state(position, velocity)
    iisph_result = iisph.step_detailed(0.0, iisph_state, 0.001, None)
    iisph_solution = phx.solver.solve_fixed_step(
        phx.solver.FixedStepProblem(
            phx.solver.IISPHFixedStepMethod(iisph),
            iisph_state,
            t0=0.0,
            t1=0.001,
            step_size=0.001,
        )
    )

    dfsph = phx.discretization.PreparedDFSPH(
        particles,
        neighborhood,
        kernel,
        1.25 * spacing,
        phx.discretization.DFSPHMethodPlan(
            1.0,
            divergence_iterations=2,
            density_iterations=2,
            divergence_tolerance=1.0,
            density_tolerance=1.0,
        ),
    )
    dfsph_state = dfsph.initialize_state(position, velocity)
    dfsph_result = dfsph.step_detailed(0.0, dfsph_state, 0.001, None)
    dfsph_solution = phx.solver.solve_fixed_step(
        phx.solver.FixedStepProblem(
            phx.solver.DFSPHFixedStepMethod(dfsph),
            dfsph_state,
            t0=0.0,
            t1=0.001,
            step_size=0.001,
        )
    )

    assert iisph_result.iterations == 2
    assert jnp.isfinite(iisph_result.residual)
    assert iisph_solution.successful
    assert dfsph_result.divergence_iterations == 2
    assert dfsph_result.density_iterations == 2
    assert jnp.isfinite(dfsph_result.divergence_residual)
    assert jnp.isfinite(dfsph_result.density_residual)
    assert dfsph_solution.successful
