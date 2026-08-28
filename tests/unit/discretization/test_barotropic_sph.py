#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _periodic_problem(count=8, *, external_potential=None, external_potential_id=None):
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count),
        jnp.full((count,), spacing),
        ambient_dimension=1,
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        count * (count - 1) // 2,
        box=box,
    )
    method = phx.discretization.BarotropicSPHMethodPlan(
        phx.discretization.WendlandC2SPHKernel(1),
        1.25 * spacing,
    )
    material = phx.equations.TaitBarotropicMaterial(1.0, 1.0)
    problem = phx.equations.BarotropicFluidProblemIR(
        "periodic-fluid",
        material,
        external_potential=external_potential,
        external_potential_id=external_potential_id,
    )
    return phx.equations.compile_barotropic_sph_problem(
        problem,
        particles,
        method,
        neighborhood=neighborhood,
    )


def _positions(count=8):
    spacing = 1.0 / count
    lattice = (jnp.arange(count, dtype=float) + 0.5)[:, None] * spacing
    perturbation = 0.015 * spacing * jnp.sin(2.0 * jnp.pi * lattice)
    return lattice + perturbation


def test_barotropic_sph_density_matches_direct_periodic_sum():
    compiled = _periodic_problem()
    position = _positions()
    displacement = position[:, None, :] - position[None, :, :]
    displacement = displacement - jnp.round(displacement)
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    kernel = compiled.dynamics.method.kernel
    direct = jnp.sum(
        compiled.dynamics.particles.masses[None, :]
        * kernel.value(distance, compiled.dynamics.method.smoothing_length),
        axis=1,
    )

    assert jnp.allclose(compiled.dynamics.density(position), direct, atol=2e-14)


def test_analytic_pressure_gradient_is_the_discrete_energy_gradient():
    compiled = _periodic_problem()
    position = _positions()
    analytic = compiled.dynamics.potential_gradient(0.0, position, None)
    reference = jax.grad(
        lambda configuration: compiled.dynamics.potential_energy(0.0, configuration, None)
    )(position)

    assert jnp.allclose(analytic, reference, rtol=2e-11, atol=2e-13)
    assert jnp.allclose(jnp.sum(-analytic, axis=0), 0.0, atol=2e-14)


def test_barotropic_sph_is_translation_invariant_and_rotation_covariant():
    compiled = _periodic_problem()
    position = _positions()
    reference = compiled.dynamics.internal_potential_gradient(position)
    translated = compiled.dynamics.internal_potential_gradient(position + 0.137)
    assert jnp.allclose(translated, reference, atol=2e-13)

    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4), jnp.full((4,), 0.25), ambient_dimension=2
    ).prepare()
    method = phx.discretization.BarotropicSPHMethodPlan(
        phx.discretization.WendlandC2SPHKernel(2), 0.45
    )
    problem = phx.equations.BarotropicFluidProblemIR(
        "planar-fluid", phx.equations.TaitBarotropicMaterial(1.0, 1.0)
    )
    planar = phx.equations.compile_barotropic_sph_problem(
        problem,
        particles,
        method,
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(6),
    )
    square = jnp.asarray(
        [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]]
    ) + jnp.asarray([[0.01, 0.0], [0.0, 0.0], [0.0, -0.02], [0.0, 0.0]])
    rotation = jnp.asarray([[0.0, -1.0], [1.0, 0.0]])
    gradient = planar.dynamics.internal_potential_gradient(square)
    rotated = planar.dynamics.internal_potential_gradient(square @ rotation.T)
    assert jnp.allclose(rotated, gradient @ rotation.T, rtol=2e-11, atol=2e-12)


def test_external_potential_and_linearization_preserve_discrete_ad_contract():
    def harmonic(time, position, stiffness):
        del time
        return 0.5 * stiffness * jnp.sum(position * position)

    compiled = _periodic_problem(
        external_potential=harmonic,
        external_potential_id="potential:harmonic",
    )
    position = _positions()
    stiffness = jnp.asarray(0.4)
    analytic, jvp, vjp = compiled.dynamics.linearize(0.0, position, stiffness)
    reference = jax.grad(
        lambda configuration: compiled.dynamics.potential_energy(
            0.0, configuration, stiffness
        )
    )(position)
    direction = jnp.cos(3.0 * position)
    cotangent = jnp.sin(5.0 * position)

    assert jnp.allclose(analytic, reference, rtol=2e-11, atol=2e-13)
    assert jnp.allclose(
        jnp.vdot(jvp(direction), cotangent),
        jnp.vdot(direction, vjp(cotangent)[0]),
        rtol=2e-11,
        atol=2e-13,
    )


def test_barotropic_sph_phase_layout_diagnostics_and_step_restriction():
    compiled = _periodic_problem()
    position = _positions()
    velocity = 0.01 * jnp.cos(2.0 * jnp.pi * position)
    phase = compiled.dynamics.pack_phase_state(position, velocity)
    unpacked_position, momentum, unpacked_velocity = compiled.dynamics.unpack_phase_state(
        phase
    )
    diagnostics = compiled.dynamics.diagnostics(0.0, position, momentum, None)
    restriction = compiled.dynamics.stable_step(0.0, position, momentum, None)

    assert phase.shape == (8, 2)
    assert jnp.allclose(unpacked_position, position)
    assert jnp.allclose(unpacked_velocity, velocity)
    assert diagnostics.total_mass == pytest.approx(1.0)
    assert diagnostics.admissible
    assert jnp.allclose(diagnostics.net_internal_force, 0.0, atol=2e-14)
    assert jnp.allclose(diagnostics.net_internal_torque, 0.0, atol=2e-14)
    assert diagnostics.active_pairs > 0
    assert jnp.isfinite(restriction.acoustic)
    assert jnp.isfinite(restriction.force)
    assert restriction.selected > 0.0


def test_cell_list_matches_dense_force_energy_and_linearization():
    dense = _periodic_problem()
    particles = dense.dynamics.particles
    method = dense.dynamics.method
    box = dense.dynamics.neighborhood.box
    cell = phx.equations.compile_barotropic_sph_problem(
        dense.problem,
        particles,
        method,
        neighborhood=phx.discretization.CellListParticleNeighborhoodPlan(
            method.kernel.support_factor * method.smoothing_length,
            4,
            24,
            box,
        ),
    )
    position = _positions()
    direction = jnp.cos(3.0 * position)
    cotangent = jnp.sin(5.0 * position)
    dense_gradient = dense.dynamics.potential_gradient(0.0, position, None)
    cell_gradient = cell.dynamics.potential_gradient(0.0, position, None)
    _, dense_jvp = jax.jvp(
        lambda configuration: dense.dynamics.potential_gradient(0.0, configuration, None),
        (position,),
        (direction,),
    )
    _, cell_jvp = jax.jvp(
        lambda configuration: cell.dynamics.potential_gradient(0.0, configuration, None),
        (position,),
        (direction,),
    )
    _, dense_vjp = jax.vjp(
        lambda configuration: dense.dynamics.potential_gradient(0.0, configuration, None),
        position,
    )
    _, cell_vjp = jax.vjp(
        lambda configuration: cell.dynamics.potential_gradient(0.0, configuration, None),
        position,
    )

    assert jnp.allclose(
        cell.dynamics.density(position), dense.dynamics.density(position), atol=2e-14
    )
    assert jnp.allclose(
        cell.dynamics.potential_energy(0.0, position, None),
        dense.dynamics.potential_energy(0.0, position, None),
        atol=2e-14,
    )
    assert jnp.allclose(cell_gradient, dense_gradient, rtol=2e-12, atol=2e-13)
    assert jnp.allclose(cell_jvp, dense_jvp, rtol=2e-11, atol=2e-12)
    assert jnp.allclose(
        cell_vjp(cotangent)[0],
        dense_vjp(cotangent)[0],
        rtol=2e-11,
        atol=2e-12,
    )
    assert cell.dynamics.neighborhood_state(position).successful
    assert (
        cell.discretization_bundle.record(cell.dynamics.neighborhood.key).artifact_kind
        == "cell-list-particle-neighborhood"
    )


def test_cell_list_compilation_validates_search_and_realization_contracts():
    dense = _periodic_problem()
    method = dense.dynamics.method
    box = dense.dynamics.neighborhood.box
    too_short = phx.discretization.CellListParticleNeighborhoodPlan(
        0.99 * method.kernel.support_factor * method.smoothing_length,
        4,
        24,
        box,
    )
    with pytest.raises(ValueError, match="cover the SPH kernel support"):
        phx.equations.compile_barotropic_sph_problem(
            dense.problem,
            dense.dynamics.particles,
            method,
            neighborhood=too_short,
        )
    with pytest.raises(ValueError, match="does not match"):
        phx.equations.compile_barotropic_sph_problem(
            dense.problem,
            dense.dynamics.particles,
            method,
            neighborhood=phx.discretization.CellListParticleNeighborhoodPlan(
                method.kernel.support_factor * method.smoothing_length,
                4,
                24,
                box,
            ),
            execution=phx.discretization.ParticleExecutionPolicy(
                realization="dense_pairs"
            ),
        )
