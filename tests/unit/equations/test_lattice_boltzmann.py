#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _compiled_flow(*, collision=None, acceleration=None, fluid_mask=None):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(12, periodic=True),
            phx.discretization.UniformCellAxisSpec(12, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D2Q9()
    ).prepare()
    geometry = (
        None
        if fluid_mask is None
        else phx.discretization.LatticeBoltzmannGeometrySnapshot(
            discretization, fluid_mask
        )
    )
    method = phx.discretization.LatticeBoltzmannMethodPlan(
        phx.discretization.BGKCollisionPlan() if collision is None else collision,
        forcing=None if acceleration is None else phx.discretization.GuoForcingPlan(),
    )
    problem = phx.equations.LatticeBoltzmannProblem(
        "periodic-flow",
        2,
        acceleration=acceleration,
        acceleration_id=None if acceleration is None else "test-acceleration",
    )
    return phx.equations.compile_lattice_boltzmann_problem(
        problem,
        discretization,
        method,
        phx.discretization.LatticeBoltzmannBoundaryPlan(geometry=geometry),
        time_step=0.01,
    )


def test_compiler_initializes_and_reconstructs_forced_macroscopic_state():
    compiled = _compiled_flow(
        acceleration=lambda time, coordinates, parameters: parameters
    )
    parameters = phx.discretization.LatticeBoltzmannRuntimeParameters(
        0.04,
        force_parameters=jnp.asarray((0.02, -0.01)),
    )
    state = compiled.initialize_state(1.0, jnp.asarray((0.0, 0.0)), parameters)
    macros = compiled.macroscopic_state(0.0, state, parameters)

    assert state.shape == (12, 12, 9)
    np.testing.assert_allclose(macros.density, 1.0, atol=1e-13)
    np.testing.assert_allclose(macros.velocity, 0.0, atol=1e-13)
    assert compiled.discretization_bundle.record(compiled.discretization.key)


def test_uniform_guo_acceleration_preserves_mass_and_updates_momentum():
    acceleration = jnp.asarray((0.005, 0.0))
    compiled = _compiled_flow(
        collision=phx.discretization.TRTCollisionPlan(),
        acceleration=lambda time, coordinates, parameters: parameters,
    )
    parameters = phx.discretization.LatticeBoltzmannRuntimeParameters(
        0.04,
        force_parameters=acceleration,
    )
    initial = compiled.initialize_state(1.0, jnp.asarray((0.0, 0.0)), parameters)
    fixed = phx.solver.FixedStepProblem(
        phx.solver.LatticeBoltzmannFixedStepMethod(compiled.dynamics),
        initial,
        t0=0.0,
        t1=0.04,
        step_size=0.01,
        args=parameters,
        discretization_bundle=compiled.discretization_bundle,
    )
    result = phx.solver.FixedStepRolloutPlan(retention="final").rollout(fixed)
    macros = compiled.macroscopic_state(0.04, result.final_state, parameters)

    assert result.successful
    np.testing.assert_allclose(macros.density, 1.0, atol=2e-12)
    np.testing.assert_allclose(
        jnp.mean(macros.velocity, axis=(0, 1)),
        0.04 * acceleration,
        rtol=2e-10,
        atol=2e-12,
    )


def test_compiled_dynamics_keeps_frozen_solid_populations_inert():
    fluid = np.ones((12, 12), dtype=bool)
    fluid[4, 5] = False
    compiled = _compiled_flow(fluid_mask=fluid)
    parameters = phx.discretization.LatticeBoltzmannRuntimeParameters(0.04)
    state = compiled.initialize_state(1.0, jnp.zeros((2,)), parameters)
    sentinel = jnp.linspace(0.02, 0.18, 9)
    state = state.at[4, 5].set(sentinel)
    result = compiled.dynamics.step_detailed(
        jnp.asarray(0),
        jnp.asarray(0.0),
        state,
        jnp.asarray(0.01),
        parameters,
    )

    assert result.successful
    np.testing.assert_array_equal(result.accepted_state[4, 5], sentinel)
