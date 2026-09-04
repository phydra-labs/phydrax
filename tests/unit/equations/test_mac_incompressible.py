#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _compiled(count=8, *, forcing=None, forcing_id=None, viscosity=0.01):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [2.0 * jnp.pi, 2.0 * jnp.pi]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    projection = phx.solver.MACPressureProjectionPlan(
        operators, solve_method="transform", tolerance=1e-10
    )
    problem = phx.equations.IncompressibleFlowProblem(
        2,
        viscosity,
        forcing=forcing,
        forcing_id=forcing_id,
    )
    compiled = phx.equations.compile_mac_incompressible_flow(
        problem, momentum, projection
    )
    return finite_volume, operators, compiled


def _taylor_green(discretization):
    x_faces = discretization.face_centers[0]
    y_faces = discretization.face_centers[1]
    return (
        jnp.sin(x_faces[..., 0]) * jnp.cos(x_faces[..., 1]),
        -jnp.cos(y_faces[..., 0]) * jnp.sin(y_faces[..., 1]),
    )


def test_mac_compiler_projects_packs_and_diagnoses_physical_velocity():
    discretization, operators, compiled = _compiled()
    velocity = _taylor_green(discretization)
    state = compiled.project_state(velocity)
    physical = compiled.physical_state(0.0, state)
    diagnostics = compiled.diagnostics(0.0, state)
    restriction = compiled.step_restriction(0.0, state)

    assert state.shape == compiled.state_shape
    assert compiled.resolved_method == "mac-symmetry-preserving-projected"
    assert diagnostics.projection_converged
    assert diagnostics.divergence_norm < 2e-10
    assert diagnostics.pressure_gauge_residual < 2e-10
    assert jnp.abs(diagnostics.nonlinear_energy_rate) < 2e-9
    assert restriction.advective > 0.0
    assert restriction.molecular > 0.0
    assert jnp.isinf(restriction.sgs)
    assert restriction.combined > 0.0
    np.testing.assert_allclose(
        operators.velocity_space.flatten(physical), state, atol=2e-10
    )


def test_mac_compiler_advances_one_projected_ssprk_step():
    discretization, operators, compiled = _compiled()
    state = compiled.project_state(_taylor_green(discretization))
    method = phx.solver.SSPRK33FixedStepMethod(compiled)

    result = eqx.filter_jit(method.step)(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-3),
        None,
    )
    velocity = compiled.unpack_velocity(result.accepted_state)

    assert result.successful
    assert jnp.linalg.norm(operators.divergence(velocity)) < 2e-9
    assert jnp.linalg.norm(result.accepted_state - state) > 0.0


def test_mac_compiler_short_horizon_forcing_gradient_is_finite():
    def forcing(_time, velocity, amplitude):
        return (jnp.ones_like(velocity[0]) * amplitude, jnp.zeros_like(velocity[1]))

    discretization, _, compiled = _compiled(
        forcing=forcing,
        forcing_id="uniform-x-force",
        viscosity=0.0,
    )
    state = compiled.project_state(_taylor_green(discretization))
    method = phx.solver.SSPRK33FixedStepMethod(compiled)

    def terminal_energy(amplitude):
        result = method.step(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0),
            state,
            jnp.asarray(5.0e-4),
            amplitude,
        )
        velocity = compiled.unpack_velocity(result.accepted_state)
        return 0.5 * compiled.momentum.operators.velocity_space.inner(velocity, velocity)

    gradient = jax.jit(jax.grad(terminal_energy))(jnp.asarray(0.2))

    assert jnp.isfinite(gradient)


def test_mac_compiler_rejects_nonunit_density_projection():
    discretization, operators, compiled = _compiled()
    projection = phx.solver.MACPressureProjectionPlan(operators, density=2.0)

    with pytest.raises(ValueError, match="unit density"):
        phx.equations.compile_mac_incompressible_flow(
            compiled.problem,
            compiled.momentum,
            projection,
        )
    assert (
        compiled.project_state(_taylor_green(discretization)).shape
        == compiled.state_shape
    )


def test_bounded_mac_compiler_preserves_exact_couette_equilibrium():
    count = 8
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide(
                "y",
                "lower",
                "no-slip",
                provider=phx.discretization.MACBoundaryProvider(jnp.zeros(2)),
            ),
            phx.discretization.MACBoundarySide(
                "y",
                "upper",
                "no-slip",
                provider=phx.discretization.MACBoundaryProvider(jnp.asarray([1.0, 0.0])),
            ),
        ),
    )
    momentum = phx.discretization.MACMomentumPlan(
        operators, boundaries=boundaries
    ).prepare()
    compiled = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 0.1),
        momentum,
        phx.solver.MACPressureProjectionPlan(
            operators,
            boundaries=boundaries,
            solve_method="transform",
            tolerance=1e-10,
        ),
    )
    velocity = (
        finite_volume.face_centers[0][..., 1],
        jnp.zeros(finite_volume.face_layouts[1].shape),
    )
    state = compiled.project_state(velocity)
    rate = compiled(0.0, state, None)
    diagnostics = compiled.diagnostics(0.0, state)

    assert diagnostics.projection_converged
    assert diagnostics.divergence_norm < 2e-10
    assert diagnostics.boundary_defect < 2e-12
    assert jnp.max(jnp.abs(rate)) < 2e-10
