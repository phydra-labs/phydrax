#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _compiled(viscosity, *, backend="dense"):
    count = 8
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), spacing), ambient_dimension=1
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    method = phx.discretization.WeaklyCompressibleSPHMethodPlan(phx.discretization.WendlandC2SPHKernel(1),
    1.25 * spacing,
    density=phx.discretization.SummationDensityPlan(), physical_viscosity=phx.discretization.MorrisViscosityPlan(viscosity), )
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
    return phx.equations.compile_weakly_compressible_sph_problem(
        phx.equations.WeaklyCompressibleFluidProblemIR(
            "viscous-fluid", phx.equations.TaitBarotropicMaterial(1.0, 1.0)
        ),
        particles,
        method,
        neighborhood=neighborhood,
    )


def _state(compiled, velocity):
    count = compiled.dynamics.particles.capacity
    position = (jnp.arange(count, dtype=float) + 0.5)[:, None] / count
    return compiled.initialize_state(position, velocity(position))


def test_morris_viscosity_is_zero_for_uniform_translation():
    compiled = _compiled(0.02)
    state = _state(compiled, lambda position: jnp.full_like(position, 0.3))
    diagnostics = compiled.dynamics.diagnostics(0.0, state, None)

    assert jnp.array_equal(diagnostics.net_viscous_force, jnp.zeros((1,)))
    assert diagnostics.viscous_dissipation_rate == pytest.approx(0.0)
    assert diagnostics.viscous_positive_power_defect == pytest.approx(0.0)


def test_morris_viscosity_is_pairwise_momentum_conservative_and_dissipative():
    compiled = _compiled(0.02)
    state = _state(compiled, lambda position: 0.1 * jnp.sin(2.0 * jnp.pi * position))
    diagnostics = compiled.dynamics.diagnostics(0.0, state, None)

    assert jnp.allclose(diagnostics.net_viscous_force, 0.0, atol=2e-14)
    assert diagnostics.viscous_power < 0.0
    assert diagnostics.viscous_dissipation_rate > 0.0
    assert diagnostics.viscous_positive_power_defect == pytest.approx(0.0, abs=2e-15)
    assert jnp.allclose(
        diagnostics.viscous_power,
        -diagnostics.viscous_dissipation_rate,
        atol=2e-14,
    )


def test_zero_morris_viscosity_produces_zero_viscous_rate():
    compiled = _compiled(0.0)
    state = _state(compiled, lambda position: 0.1 * jnp.sin(2.0 * jnp.pi * position))
    diagnostics = compiled.dynamics.diagnostics(0.0, state, None)

    assert diagnostics.viscous_dissipation_rate == pytest.approx(0.0)
    assert jnp.array_equal(diagnostics.net_viscous_force, jnp.zeros((1,)))
    assert jnp.isinf(compiled.dynamics.stable_step(0.0, state, None).viscous)


def test_morris_viscosity_dense_cell_and_derivative_parity():
    dense = _compiled(0.02, backend="dense")
    cell = _compiled(0.02, backend="cell")
    state = _state(dense, lambda position: 0.1 * jnp.sin(2.0 * jnp.pi * position))
    direction = jnp.cos(3.0 * state)
    dense_rate, dense_jvp = jax.jvp(
        lambda value: dense.dynamics(0.0, value, None),
        (state,),
        (direction,),
    )
    cell_rate, cell_jvp = jax.jvp(
        lambda value: cell.dynamics(0.0, value, None),
        (state,),
        (direction,),
    )

    assert jnp.allclose(cell_rate, dense_rate, rtol=2e-12, atol=2e-13)
    assert jnp.allclose(cell_jvp, dense_jvp, rtol=2e-10, atol=2e-11)
