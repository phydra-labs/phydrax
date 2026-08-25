#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _quadrilateral_grid(nx=4, ny=4):
    vertices = np.asarray(
        [(i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)]
    )
    quadrilaterals = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx + 1
            upper_right = upper_left + 1
            quadrilaterals.append((lower_left, lower_right, upper_right, upper_left))
    return vertices, np.asarray(quadrilaterals, dtype=np.int32)


def _scalar_system(velocity):
    speed = tuple(float(value) for value in velocity)
    return phx.equations.ScalarConservationSystem(
        len(speed),
        lambda state, axis, args: jnp.asarray(speed[axis], dtype=state.dtype) * state,
        lambda left, right, axis, args: jnp.full(
            left.shape[:-1],
            jnp.abs(jnp.asarray(speed[axis], dtype=left.dtype)),
            dtype=left.dtype,
        ),
        system_id="implicit-unstructured-advection",
    )


def _compiled_dynamics(system, reconstruction_factory, *, precision=None):
    vertices, quadrilaterals = _quadrilateral_grid()
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=quadrilaterals,
        component_names=system.component_names,
    ).prepare()
    reconstruction = reconstruction_factory(discretization)
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        reconstruction, phx.discretization.RusanovFluxPlan()
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    problem = phx.equations.ConservationProblemIR(
        "implicit-unstructured", "state", system, boundaries
    )
    return phx.equations.compile_conservation_problem(
        problem, discretization, method, precision=precision
    ).dynamics


def test_backward_euler_solves_affine_advection_and_refreshes_symbolic_plan():
    system = _scalar_system((0.4, -0.3))
    dynamics = _compiled_dynamics(
        system,
        lambda discretization: phx.discretization.CellPolynomialReconstructionPlan(
            1
        ).prepare(discretization),
    )
    centers = dynamics.discretization.cell_centers
    previous = (0.9 + 0.2 * centers[:, 0] - 0.1 * centers[:, 1])[:, None]
    step_size = jnp.asarray(0.04)
    derivative = 0.4 * 0.2 + (-0.3) * (-0.1)
    expected = previous - step_size * derivative
    plan = phx.solver.FiniteVolumeBackwardEulerPlan(
        dynamics,
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-11,
            relative_residual=1e-11,
            maximum_steps=8,
        ),
    )
    prepared = plan.prepare(previous, jnp.asarray(0.0), step_size)
    template_id = prepared.nonlinear.linear_template_id
    result = prepared.solve()

    assert result.successful
    np.testing.assert_allclose(result.state, expected, rtol=2e-9, atol=2e-9)
    assert result.nonlinear.diagnostics.final_residual_norm < 1e-10
    assert result.accepted_step_size == step_size
    assert result.time == step_size

    refreshed = prepared.refresh(
        result.state,
        result.time,
        step_size,
        initial_guess=result.state,
    )
    assert refreshed.nonlinear.linear_template_id == template_id
    assert refreshed.nonlinear.numeric_version == prepared.nonlinear.numeric_version + 1
    second = refreshed.solve()
    assert second.successful
    np.testing.assert_allclose(
        second.state,
        previous - 2.0 * step_size * derivative,
        rtol=2e-9,
        atol=2e-9,
    )


def test_backward_euler_residual_jvp_and_vjp_are_adjoint():
    system = _scalar_system((0.4, -0.3))
    dynamics = _compiled_dynamics(
        system,
        lambda discretization: phx.discretization.CellPolynomialReconstructionPlan(
            1
        ).prepare(discretization),
    )
    centers = dynamics.discretization.cell_centers
    previous = (0.9 + 0.2 * centers[:, 0] - 0.1 * centers[:, 1])[:, None]
    plan = phx.solver.FiniteVolumeBackwardEulerPlan(dynamics)
    prepared = plan.prepare(previous, jnp.asarray(0.0), jnp.asarray(0.03))
    direction = jnp.sin(jnp.arange(previous.size)).reshape(previous.shape)
    cotangent = jnp.cos(jnp.arange(previous.size)).reshape(previous.shape)

    def residual(state):
        return plan.residual_operator(state, prepared.stage)

    _, tangent = jax.jvp(residual, (previous,), (direction,))
    _, pullback = jax.vjp(residual, previous)
    adjoint = pullback(cotangent)[0]
    np.testing.assert_allclose(
        jnp.vdot(tangent, cotangent),
        jnp.vdot(direction, adjoint),
        rtol=2e-10,
        atol=2e-10,
    )


def test_backward_euler_compressible_state_is_admissible_and_float32_explicit():
    system = phx.equations.EulerSystem(2)
    precision = phx.discretization.FiniteVolumePrecisionPolicy("float32")
    dynamics = _compiled_dynamics(
        system,
        lambda discretization: phx.discretization.PiecewiseConstantReconstruction(),
        precision=precision,
    )
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.1, -0.05, 1.0), dtype=jnp.float32),
        dynamics.discretization.state_shape,
    )
    previous = system.primitive_to_conserved(primitive).astype(jnp.float32)
    result = phx.solver.FiniteVolumeBackwardEulerPlan(
        dynamics,
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-6,
            relative_residual=1e-6,
            maximum_steps=4,
        ),
    ).advance(previous, jnp.asarray(0.0), jnp.asarray(0.01))

    assert result.successful
    assert result.state.dtype == jnp.float32
    assert result.nonlinear.residual.dtype == jnp.float32
    np.testing.assert_allclose(result.state, previous, rtol=2e-6, atol=2e-6)
    assert jnp.all(system.admissible(result.state))


def test_backward_euler_advances_nonuniform_compressible_flow():
    system = phx.equations.EulerSystem(2)
    dynamics = _compiled_dynamics(
        system,
        lambda discretization: phx.discretization.PiecewiseConstantReconstruction(),
    )
    centers = dynamics.discretization.cell_centers
    density = 1.0 + 0.05 * jnp.sin(2.0 * jnp.pi * centers[:, 0])
    primitive = jnp.stack(
        (
            density,
            jnp.full_like(density, 0.2),
            jnp.zeros_like(density),
            jnp.ones_like(density),
        ),
        axis=-1,
    )
    previous = system.primitive_to_conserved(primitive)
    result = phx.solver.FiniteVolumeBackwardEulerPlan(
        dynamics,
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-9,
            relative_residual=1e-9,
            maximum_steps=8,
        ),
    ).advance(previous, jnp.asarray(0.0), jnp.asarray(1e-3))

    assert result.successful
    assert result.nonlinear.diagnostics.final_residual_norm < 1e-8
    assert jnp.max(jnp.abs(result.state - previous)) > 0.0
    assert jnp.all(system.admissible(result.state))


def test_backward_euler_plan_identity_includes_nonlinear_termination():
    system = _scalar_system((0.4, -0.3))
    dynamics = _compiled_dynamics(
        system,
        lambda discretization: phx.discretization.PiecewiseConstantReconstruction(),
    )
    baseline_termination = phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-8,
        relative_residual=2e-8,
        absolute_step=3e-12,
        relative_step=4e-10,
        maximum_steps=6,
        maximum_evaluations=30,
        maximum_linear_iterations=50,
        divergence_factor=1e6,
    )
    baseline = phx.solver.FiniteVolumeBackwardEulerPlan(
        dynamics, termination=baseline_termination
    )
    identical = phx.solver.FiniteVolumeBackwardEulerPlan(
        dynamics,
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-8,
            relative_residual=2e-8,
            absolute_step=3e-12,
            relative_step=4e-10,
            maximum_steps=6,
            maximum_evaluations=30,
            maximum_linear_iterations=50,
            divergence_factor=1e6,
        ),
    )
    changed_steps = phx.solver.FiniteVolumeBackwardEulerPlan(
        dynamics,
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-8,
            relative_residual=2e-8,
            absolute_step=3e-12,
            relative_step=4e-10,
            maximum_steps=7,
            maximum_evaluations=30,
            maximum_linear_iterations=50,
            divergence_factor=1e6,
        ),
    )
    changed_tolerance = phx.solver.FiniteVolumeBackwardEulerPlan(
        dynamics,
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=5e-8,
            relative_residual=2e-8,
            absolute_step=3e-12,
            relative_step=4e-10,
            maximum_steps=6,
            maximum_evaluations=30,
            maximum_linear_iterations=50,
            divergence_factor=1e6,
        ),
    )

    assert identical.plan_id == baseline.plan_id
    assert changed_steps.plan_id != baseline.plan_id
    assert changed_tolerance.plan_id != baseline.plan_id


def test_backward_euler_refresh_rejects_broadcastable_previous_state_shape():
    system = _scalar_system((0.4, -0.3))
    dynamics = _compiled_dynamics(
        system,
        lambda discretization: phx.discretization.PiecewiseConstantReconstruction(),
    )
    previous = jnp.ones(dynamics.discretization.state_shape)
    plan = phx.solver.FiniteVolumeBackwardEulerPlan(dynamics)
    prepared = plan.prepare(previous, jnp.asarray(0.0), jnp.asarray(0.01))
    malformed_previous = previous[0]

    with pytest.raises(ValueError, match="previous state must have shape"):
        prepared.refresh(
            malformed_previous,
            jnp.asarray(0.01),
            jnp.asarray(0.01),
            initial_guess=previous,
        )
