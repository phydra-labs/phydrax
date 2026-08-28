#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _compiled_problem(count=8, *, backend="dense"):
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count),
        jnp.full((count,), spacing),
        ambient_dimension=1,
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    method = phx.discretization.BarotropicSPHMethodPlan(
        phx.discretization.WendlandC2SPHKernel(1),
        1.25 * spacing,
    )
    if backend == "dense":
        neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
            count * (count - 1) // 2,
            box=box,
        )
    elif backend == "cell":
        neighborhood = phx.discretization.CellListParticleNeighborhoodPlan(
            method.kernel.support_factor * method.smoothing_length,
            4,
            4 * count,
            box,
        )
    else:
        raise ValueError("Unknown test particle backend.")
    return phx.equations.compile_barotropic_sph_problem(
        phx.equations.BarotropicFluidProblemIR(
            "sound-wave",
            phx.equations.TaitBarotropicMaterial(1.0, 1.0),
        ),
        particles,
        method,
        neighborhood=neighborhood,
    )


def _initial_state(count, amplitude):
    spacing = 1.0 / count
    lattice = (jnp.arange(count, dtype=float) + 0.5)[:, None] * spacing
    mode = jnp.sin(2.0 * jnp.pi * lattice)
    return lattice + amplitude * mode, jnp.zeros_like(lattice)


def test_barotropic_sph_compiles_and_solves_with_stormer_verlet():
    compiled = _compiled_problem()
    position, velocity = _initial_state(8, 2.0e-3)
    ivp = compiled.as_differential_problem(
        position,
        velocity,
        t0=0.0,
        t1=0.004,
    )
    initial = compiled.dynamics.diagnostics(
        0.0,
        position,
        compiled.dynamics.pack_phase_state(position, velocity)[:, 1:],
        None,
    )
    solution = phx.solver.solve_diffrax(
        ivp,
        save_times=jnp.asarray([0.0, 0.002, 0.004]),
        solver=phx.solver.StormerVerlet(1),
        dt0=2.0e-4,
        max_steps=32,
    )
    final_position, final_momentum, _ = compiled.dynamics.unpack_phase_state(
        solution.states[-1]
    )
    final = compiled.dynamics.diagnostics(
        solution.times[-1],
        final_position,
        final_momentum,
        None,
    )

    assert solution.states.shape == (3, 8, 2)
    assert solution.backend_successful
    assert jnp.all(solution.valid)
    assert solution.discretization_bundle_id == compiled.discretization_bundle.bundle_id
    assert solution.state_geometry_id == "state-geometry:canonical-phase"
    assert solution.solver_id == "solver:stormer-verlet:canonical"
    assert len(solution.discretization_bundle.records) == 3
    assert jnp.all(jnp.isfinite(solution.states))
    assert jnp.allclose(final.linear_momentum, initial.linear_momentum, atol=2e-13)
    assert jnp.allclose(final.total_energy, initial.total_energy, rtol=2e-8, atol=2e-12)


def test_short_sph_trajectory_has_the_fixed_discrete_gradient():
    compiled = _compiled_problem(count=6)
    final_time = 0.001

    def terminal(amplitude):
        position, velocity = _initial_state(6, amplitude)
        ivp = compiled.as_differential_problem(
            position,
            velocity,
            t0=0.0,
            t1=final_time,
        )
        solution = phx.solver.solve_diffrax(
            ivp,
            save_times=jnp.asarray([final_time]),
            solver=phx.solver.StormerVerlet(1),
            dt0=5.0e-4,
            max_steps=4,
        )
        return solution.states[-1, 0, 0]

    amplitude = jnp.asarray(1.0e-3)
    derivative = jax.grad(terminal)(amplitude)
    step = jnp.asarray(1.0e-5)
    finite_difference = (terminal(amplitude + step) - terminal(amplitude - step)) / (
        2.0 * step
    )

    assert jnp.isfinite(derivative)
    assert jnp.allclose(derivative, finite_difference, rtol=2e-5, atol=2e-7)


def test_cell_list_and_dense_backends_produce_the_same_trajectory():
    dense = _compiled_problem(backend="dense")
    cell = _compiled_problem(backend="cell")
    position, velocity = _initial_state(8, 2.0e-3)

    def solve(compiled):
        return phx.solver.solve_diffrax(
            compiled.as_differential_problem(
                position,
                velocity,
                t0=0.0,
                t1=0.004,
            ),
            save_times=jnp.asarray([0.0, 0.002, 0.004]),
            solver=phx.solver.StormerVerlet(1),
            dt0=2.0e-4,
            max_steps=32,
        )

    dense_solution = solve(dense)
    cell_solution = solve(cell)

    assert dense_solution.backend_successful
    assert cell_solution.backend_successful
    assert jnp.allclose(
        cell_solution.states,
        dense_solution.states,
        rtol=2e-12,
        atol=2e-13,
    )
    assert (
        cell.discretization_bundle.record(cell.dynamics.neighborhood.key).artifact_kind
        == "cell-list-particle-neighborhood"
    )
