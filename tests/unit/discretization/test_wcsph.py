#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _compiled(*, continuity, backend="dense", viscosity=None, acceleration=None):
    count = 8
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), spacing), ambient_dimension=1
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    density = (
        phx.discretization.ContinuityDensityPlan()
        if continuity
        else phx.discretization.SummationDensityPlan()
    )
    method = phx.discretization.WeaklyCompressibleSPHMethodPlan(phx.discretization.WendlandC2SPHKernel(1),
    1.25 * spacing,
    density=density, physical_viscosity=viscosity, )
    neighborhood = (
        phx.discretization.DenseParticleNeighborhoodPlan(
            count * (count - 1) // 2, box=box
        )
        if backend == "dense"
        else phx.discretization.CellListParticleNeighborhoodPlan(
            method.kernel.support_factor * method.smoothing_length,
            4,
            4 * count,
            box,
        )
    )
    problem = phx.equations.WeaklyCompressibleFluidProblemIR(
        "wcsph-fluid",
        phx.equations.TaitBarotropicMaterial(1.0, 1.0),
        external_acceleration=acceleration,
        external_acceleration_id=None if acceleration is None else "acceleration:test",
    )
    return phx.equations.compile_weakly_compressible_sph_problem(
        problem, particles, method, neighborhood=neighborhood
    )


def _initial(compiled):
    count = compiled.dynamics.particles.capacity
    position = (jnp.arange(count, dtype=float) + 0.5)[:, None] / count
    position = position + 0.002 * jnp.sin(2.0 * jnp.pi * position)
    velocity = 0.03 * jnp.cos(2.0 * jnp.pi * position)
    return compiled.initialize_state(position, velocity)


def test_wcsph_density_formulations_have_explicit_state_and_drift_layouts():
    summation = _compiled(continuity=False)
    continuity = _compiled(continuity=True)
    summation_state = _initial(summation)
    continuity_state = _initial(continuity)

    assert summation_state.shape == (8, 2)
    assert continuity_state.shape == (8, 3)
    assert summation.dynamics(0.0, summation_state, None).shape == summation_state.shape
    assert (
        continuity.dynamics(0.0, continuity_state, None).shape == continuity_state.shape
    )
    assert not summation.dynamics.state_layout.density_evolved
    assert continuity.dynamics.state_layout.density_evolved
    with pytest.raises(ValueError, match="does not accept density"):
        summation.initialize_state(
            summation_state[:, :1], summation_state[:, 1:2], jnp.ones((8,))
        )


def test_continuity_density_pressure_work_is_semidiscretely_energy_balanced():
    compiled = _compiled(continuity=True)
    state = _initial(compiled)
    diagnostics = compiled.dynamics.diagnostics(0.0, state, None)

    def total_energy(value):
        position, velocity, density = compiled.dynamics.state_layout.unpack(value)
        masses = compiled.dynamics.particles.safe_masses
        return jnp.sum(
            0.5 * masses * jnp.sum(velocity * velocity, axis=-1)
            + masses * compiled.problem.material.specific_internal_energy(density)
        )

    directional_rate = jnp.vdot(
        jax.grad(total_energy)(state), compiled.dynamics(0.0, state, None)
    )

    assert jnp.allclose(diagnostics.pressure_energy_balance_defect, 0.0, atol=2e-13)
    assert jnp.allclose(directional_rate, 0.0, atol=2e-13)
    assert jnp.allclose(diagnostics.total_energy_rate, 0.0, atol=2e-13)


def test_summation_wcsph_pressure_acceleration_matches_barotropic_reference():
    compiled = _compiled(continuity=False)
    state = _initial(compiled)
    position, velocity, _ = compiled.dynamics.state_layout.unpack(state)
    barotropic = phx.equations.compile_barotropic_sph_problem(
        phx.equations.BarotropicFluidProblemIR("reference", compiled.problem.material),
        compiled.dynamics.particles,
        phx.discretization.BarotropicSPHMethodPlan(
            compiled.dynamics.method.kernel,
            compiled.dynamics.method.smoothing_length,
        ),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(
            28, box=compiled.dynamics.neighborhood.box
        ),
    )
    rate = compiled.dynamics(0.0, state, None)

    assert jnp.allclose(rate[:, :1], velocity)
    assert jnp.allclose(
        rate[:, 1:2],
        barotropic.dynamics.acceleration(0.0, position, None),
        rtol=2e-12,
        atol=2e-13,
    )


def test_wcsph_external_acceleration_and_power_are_explicit():
    def gravity(time, position, velocity, density, scale):
        del time, velocity, density
        return jnp.ones_like(position) * scale

    compiled = _compiled(continuity=True, acceleration=gravity)
    state = _initial(compiled)
    scale = jnp.asarray(0.2)
    diagnostics = compiled.dynamics.diagnostics(0.0, state, scale)
    rate = compiled.dynamics(0.0, state, scale)

    assert jnp.allclose(diagnostics.external_force, jnp.asarray([0.2]))
    assert jnp.allclose(
        rate[:, 1],
        _compiled(continuity=True).dynamics(0.0, state, None)[:, 1] + scale,
        rtol=2e-12,
        atol=2e-13,
    )
    assert jnp.isfinite(diagnostics.external_power)


def test_wcsph_step_graph_and_linearization_contracts():
    compiled = _compiled(
        continuity=True,
        backend="cell",
        viscosity=phx.discretization.MorrisViscosityPlan(0.01),
    )
    state = _initial(compiled)
    restriction = compiled.dynamics.stable_step(0.0, state, None)
    graph = compiled.dynamics.graph_view(0.0, state, None, directed=True)
    value, jvp, vjp = compiled.dynamics.linearize(0.0, state, None)
    direction = jnp.cos(3.0 * state)
    cotangent = jnp.sin(5.0 * state)

    assert restriction.selected > 0.0
    assert restriction.selected <= restriction.acoustic
    assert restriction.selected <= restriction.force
    assert restriction.selected <= restriction.viscous
    assert graph.n_node[0] == 8
    assert int(jnp.sum(graph.edge_mask)) > 0
    assert value.shape == state.shape
    assert jnp.allclose(
        jnp.vdot(jvp(direction), cotangent),
        jnp.vdot(direction, vjp(cotangent)[0]),
        rtol=2e-9,
        atol=2e-10,
    )
