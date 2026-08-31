from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


fd = phx.applications.solid_mechanics


def test_differentiable_cable_net_inverse_design_workflow():
    edges = jnp.asarray(((4, 0), (4, 1), (4, 2), (4, 3)), dtype=jnp.int32)
    structure = fd.ForceDensityStructure.from_edges(
        edges,
        5,
        3,
        fixed_nodes=(0, 1, 2, 3),
    )
    reference = jnp.asarray(
        (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    )
    loads = jnp.zeros((5, 3)).at[4, 2].set(-2.0)
    sample = fd.ForceDensityInputs(
        jnp.ones((4,)),
        structure.prescribed_values(reference),
        loads,
    )
    equilibrium_problem = fd.ForceDensityProblem(
        structure,
        sign_mode="tension",
        problem_id="square-cable-net",
    )
    plan = fd.plan_force_density(equilibrium_problem, sample)

    def decode(design, external_load):
        load = loads.at[4, 2].set(external_load)
        return fd.ForceDensityInputs(
            jnp.repeat(design.reshape(()), 4),
            sample.prescribed_values,
            load,
        )

    design_problem = fd.ForceDensityDesignProblem(
        plan,
        decode,
        lambda state, design, _: (state.positions[4, 2] + 0.25) ** 2,
        design_bounds=phx.optim.Bounds(0.25, 8.0),
        problem_id="square-cable-net-target-height",
    )
    result = fd.solve_force_density_design(
        design_problem,
        jnp.asarray(1.0),
        args=jnp.asarray(-2.0),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1.0e-6,
            relative_optimality=0.0,
            maximum_steps=400,
        ),
    )

    assert result.successful
    assert result.equilibrium.state.positions[4, 2] == pytest.approx(-0.25, abs=5e-4)
    assert (
        jnp.max(result.equilibrium.state.axial_forces)
        - jnp.min(result.equilibrium.state.axial_forces)
        <= 1.0e-10
    )
    assert result.equilibrium.diagnostics.global_balance_norm <= 1.0e-9

    optimized_q = result.inputs.force_densities

    def center_height(external_load):
        inputs = fd.ForceDensityInputs(
            optimized_q,
            sample.prescribed_values,
            loads.at[4, 2].set(external_load),
        )
        solved = fd.solve_force_density(fd.prepare_force_density(plan, inputs))
        return solved.state.positions[4, 2]

    sensitivity = jax.grad(center_height)(jnp.asarray(-2.0))
    assert jnp.isfinite(sensitivity)
    assert sensitivity > 0.0
