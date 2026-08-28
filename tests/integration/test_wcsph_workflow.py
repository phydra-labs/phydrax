#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _compiled(*, backend="cell", viscosity=0.0):
    count = 8
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), spacing), ambient_dimension=1
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    method = phx.discretization.WeaklyCompressibleSPHMethodPlan(phx.discretization.WendlandC2SPHKernel(1),
    1.25 * spacing,
    density=phx.discretization.ContinuityDensityPlan(), physical_viscosity=phx.discretization.MorrisViscosityPlan(viscosity), )
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
            "workflow-fluid", phx.equations.TaitBarotropicMaterial(1.0, 1.0)
        ),
        particles,
        method,
        neighborhood=neighborhood,
    )


def _initial(amplitude=0.002):
    position = (jnp.arange(8, dtype=float) + 0.5)[:, None] / 8.0
    position = position + amplitude * jnp.sin(2.0 * jnp.pi * position)
    velocity = jnp.zeros_like(position)
    return position, velocity


@pytest.mark.parametrize("solver", [phx.solver.SSPRK33(), phx.solver.SSPRK54()])
def test_wcsph_solves_through_native_ssprk_methods(solver):
    compiled = _compiled(viscosity=0.01)
    position, velocity = _initial()
    problem = compiled.as_differential_problem(position, velocity, t0=0.0, t1=0.002)
    solution = phx.solver.solve_diffrax(
        problem,
        save_times=jnp.asarray([0.0, 0.001, 0.002]),
        solver=solver,
        dt0=2.0e-4,
        max_steps=32,
    )

    assert solution.backend_successful
    assert solution.states.shape == (3, 8, 3)
    assert jnp.all(solution.valid)
    assert jnp.all(jnp.isfinite(solution.states))
    assert solution.discretization_bundle_id == compiled.discretization_bundle.bundle_id


def test_dense_and_cell_wcsph_trajectories_match():
    dense = _compiled(backend="dense", viscosity=0.01)
    cell = _compiled(backend="cell", viscosity=0.01)
    position, velocity = _initial()

    def solve(compiled):
        return phx.solver.solve_diffrax(
            compiled.as_differential_problem(position, velocity, t0=0.0, t1=0.002),
            save_times=jnp.asarray([0.0, 0.001, 0.002]),
            solver=phx.solver.SSPRK33(),
            dt0=2.0e-4,
            max_steps=32,
        )

    dense_solution = solve(dense)
    cell_solution = solve(cell)
    assert jnp.allclose(
        cell_solution.states,
        dense_solution.states,
        rtol=2e-11,
        atol=2e-12,
    )


def test_short_wcsph_trajectory_has_fixed_discrete_gradient():
    compiled = _compiled(viscosity=0.0)

    def terminal(amplitude):
        position, velocity = _initial(amplitude)
        solution = phx.solver.solve_diffrax(
            compiled.as_differential_problem(position, velocity, t0=0.0, t1=0.001),
            save_times=jnp.asarray([0.001]),
            solver=phx.solver.SSPRK33(),
            dt0=5.0e-4,
            max_steps=4,
        )
        return solution.states[-1, 0, 0]

    amplitude = jnp.asarray(0.001)
    derivative = jax.grad(terminal)(amplitude)
    step = jnp.asarray(1.0e-5)
    finite_difference = (terminal(amplitude + step) - terminal(amplitude - step)) / (
        2.0 * step
    )

    assert jnp.isfinite(derivative)
    assert jnp.allclose(derivative, finite_difference, rtol=3e-5, atol=3e-7)
