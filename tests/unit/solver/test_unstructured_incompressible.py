#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _collocated_operators(nx=6, ny=5):
    logical = np.asarray([(i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)])
    x = logical[:, 0]
    y = logical[:, 1]
    vertices = np.stack((x + 0.12 * x * y, y + 0.06 * x * (1.0 - y)), axis=-1)
    quadrilaterals = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx + 1
            upper_right = upper_left + 1
            quadrilaterals.append((lower_left, lower_right, upper_right, upper_left))
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(quadrilaterals),
    ).prepare()
    gradient = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(
        discretization
    )
    return phx.discretization.PreparedUnstructuredCollocatedOperators(
        discretization, gradient
    )


def test_collocated_gradient_divergence_gauge_and_rhie_chow_contracts():
    operators = _collocated_operators()
    discretization = operators.discretization
    centers = discretization.cell_centers
    pressure = 0.7 + 0.3 * centers[:, 0] - 0.2 * centers[:, 1]
    gradient = operators.cell_gradient(pressure)
    np.testing.assert_allclose(
        gradient,
        jnp.broadcast_to(jnp.asarray((0.3, -0.2)), gradient.shape),
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        operators.face_normal_gradient(pressure + 13.0),
        operators.face_normal_gradient(pressure),
        atol=2e-11,
    )
    np.testing.assert_allclose(
        operators.positive_gauged_laplacian(operators.gauge_project(pressure + 13.0)),
        operators.positive_gauged_laplacian(operators.gauge_project(pressure)),
        atol=2e-11,
    )
    projected = operators.gauge_project(pressure)
    np.testing.assert_allclose(
        jnp.sum(discretization.cell_volumes * projected), 0.0, atol=2e-12
    )

    face_velocity = jnp.sin(jnp.arange(discretization.face_measures.size))
    divergence = operators.divergence(face_velocity)
    boundary = ~operators.interior_faces
    expected_boundary_flux = jnp.sum(
        face_velocity[boundary] * discretization.face_measures[boundary]
    )
    np.testing.assert_allclose(
        jnp.sum(discretization.cell_volumes * divergence),
        expected_boundary_flux,
        atol=2e-12,
    )

    nx = 6
    checkerboard = jnp.asarray(
        [(-1.0) ** (cell % nx + cell // nx) for cell in range(discretization.cell_count)]
    )
    zero_velocity = jnp.zeros((discretization.cell_count, discretization.cell_dimension))
    arithmetic = operators.interpolate_normal_velocity(zero_velocity)
    rhie_chow = operators.rhie_chow_face_velocity(
        zero_velocity,
        checkerboard,
        jnp.ones((discretization.cell_count,)),
    )
    np.testing.assert_allclose(arithmetic, 0.0)
    assert jnp.max(jnp.abs(rhie_chow[operators.interior_faces])) > 0.0


def test_pressure_projection_recovers_discrete_gradient_and_removes_divergence():
    operators = _collocated_operators()
    discretization = operators.discretization
    projection = phx.solver.UnstructuredPressureProjectionPlan(
        operators, tolerance=1e-10, maximum_iterations=500
    )
    centers = discretization.cell_centers
    potential = (
        jnp.sin(1.3 * centers[:, 0])
        + 0.4 * jnp.cos(0.7 * centers[:, 1])
        + 0.2 * centers[:, 0] * centers[:, 1]
    )
    step_size = jnp.asarray(0.07)
    face_velocity = step_size * operators.face_normal_gradient(potential)
    boundary_velocity = jnp.zeros_like(face_velocity)
    cell_velocity = jnp.zeros((discretization.cell_count, discretization.cell_dimension))
    result = projection.project(
        cell_velocity,
        step_size,
        face_normal_velocity=face_velocity,
        boundary_normal_velocity=boundary_velocity,
    )

    assert result.linear.successful
    assert result.converged
    np.testing.assert_allclose(
        result.pressure_increment,
        operators.gauge_project(potential),
        rtol=2e-8,
        atol=2e-8,
    )
    assert jnp.linalg.norm(result.divergence_after) < 1e-8
    assert jnp.linalg.norm(result.divergence_after) < 1e-7 * jnp.linalg.norm(
        result.divergence_before
    )
    np.testing.assert_allclose(result.gauge_defect, 0.0, atol=2e-11)
    jitted = eqx.filter_jit(projection.project)(
        cell_velocity,
        step_size,
        face_normal_velocity=face_velocity,
        boundary_normal_velocity=boundary_velocity,
    )
    assert jitted.converged
    np.testing.assert_allclose(
        jitted.face_normal_velocity,
        result.face_normal_velocity,
        rtol=2e-9,
        atol=2e-9,
    )


def test_pressure_projection_refreshes_nonuniform_momentum_inverse():
    operators = _collocated_operators()
    discretization = operators.discretization
    projection = phx.solver.UnstructuredPressureProjectionPlan(
        operators, tolerance=1e-10, maximum_iterations=500
    )
    centers = discretization.cell_centers
    potential = (
        0.3 * jnp.sin(1.1 * centers[:, 0])
        - 0.2 * jnp.cos(0.9 * centers[:, 1])
        + 0.1 * centers[:, 0] * centers[:, 1]
    )
    inverse = 0.02 * (1.0 + 0.8 * centers[:, 0] + 0.3 * centers[:, 1])
    face_inverse = operators.interpolate_inverse_momentum(inverse)
    face_velocity = face_inverse * operators.face_normal_gradient(potential)
    boundary_velocity = jnp.zeros_like(face_velocity)
    cell_velocity = jnp.zeros((discretization.cell_count, discretization.cell_dimension))
    result = projection.project(
        cell_velocity,
        jnp.asarray(0.05),
        inverse_momentum_diagonal=inverse,
        face_normal_velocity=face_velocity,
        boundary_normal_velocity=boundary_velocity,
    )

    assert result.linear.successful
    assert result.converged
    np.testing.assert_allclose(
        result.face_inverse_momentum, face_inverse, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        result.pressure_increment,
        operators.gauge_project(potential),
        rtol=3e-8,
        atol=3e-8,
    )
    assert jnp.linalg.norm(result.divergence_after) < 1e-8


def test_fixed_pressure_corrections_are_jittable_and_report_linear_status():
    operators = _collocated_operators()
    discretization = operators.discretization
    projection = phx.solver.UnstructuredPressureProjectionPlan(
        operators, tolerance=1e-9, maximum_iterations=400
    )
    correction = phx.solver.UnstructuredPressureCorrectionPlan(projection, 2)
    centers = discretization.cell_centers
    velocity = jnp.stack((centers[:, 0], -0.3 * centers[:, 1]), axis=-1)
    pressure = jnp.zeros((discretization.cell_count,))
    boundary_velocity = jnp.zeros((discretization.face_measures.size,))

    def predictor(time, current_velocity, args):
        del time, args
        return current_velocity

    result = eqx.filter_jit(correction.advance)(
        jnp.asarray(0.0),
        velocity,
        pressure,
        jnp.asarray(0.05),
        predictor,
        boundary_normal_velocity=boundary_velocity,
    )
    assert result.converged
    assert jnp.all(result.linear_status_history == 0)
    assert result.divergence_history[-1] <= result.divergence_history[0] + 1e-10
    np.testing.assert_allclose(
        jnp.sum(discretization.cell_volumes * result.pressure), 0.0, atol=2e-10
    )


def test_pressure_correction_normalizes_mixed_input_dtypes_eager_and_jit():
    operators = _collocated_operators()
    discretization = operators.discretization
    projection = phx.solver.UnstructuredPressureProjectionPlan(
        operators, tolerance=1e-9, maximum_iterations=400
    )
    correction = phx.solver.UnstructuredPressureCorrectionPlan(projection, 2)
    centers = discretization.cell_centers
    velocity = jnp.stack((centers[:, 0], -0.3 * centers[:, 1]), axis=-1).astype(
        jnp.float32
    )
    pressure = jnp.zeros((discretization.cell_count,), dtype=jnp.float32)
    inverse = jnp.full((discretization.cell_count,), 0.05, dtype=jnp.float32)
    boundary_velocity = jnp.zeros((discretization.face_measures.size,), dtype=jnp.float32)

    def predictor(time, current_velocity, args):
        del time, args
        return (0.95 * current_velocity).astype(jnp.float32)

    arguments = (
        jnp.asarray(0.0, dtype=jnp.float32),
        velocity,
        pressure,
        jnp.asarray(0.05, dtype=jnp.float32),
        predictor,
    )
    eager = correction.advance(
        *arguments,
        inverse_momentum_diagonal=inverse,
        boundary_normal_velocity=boundary_velocity,
    )
    jitted = eqx.filter_jit(correction.advance)(
        *arguments,
        inverse_momentum_diagonal=inverse,
        boundary_normal_velocity=boundary_velocity,
    )

    assert eager.velocity.dtype == projection.pressure_space.dtype
    assert eager.face_normal_velocity.dtype == projection.pressure_space.dtype
    assert eager.pressure.dtype == projection.pressure_space.dtype
    assert eager.divergence_history.dtype == projection.pressure_space.dtype
    assert jitted.velocity.dtype == projection.pressure_space.dtype
    assert jitted.pressure.dtype == projection.pressure_space.dtype
    assert eager.converged & jitted.converged
    np.testing.assert_allclose(jitted.velocity, eager.velocity)
    np.testing.assert_allclose(jitted.pressure, eager.pressure)
    np.testing.assert_allclose(jitted.divergence_history, eager.divergence_history)

    projection32 = phx.solver.UnstructuredPressureProjectionPlan(
        operators,
        tolerance=1e-6,
        maximum_iterations=400,
        dtype=jnp.float32,
    )
    correction32 = phx.solver.UnstructuredPressureCorrectionPlan(projection32, 2)

    def predictor64(time, current_velocity, args):
        del time, args
        return (0.95 * current_velocity).astype(jnp.float64)

    arguments64 = (
        jnp.asarray(0.0, dtype=jnp.float64),
        velocity.astype(jnp.float64),
        pressure.astype(jnp.float64),
        jnp.asarray(0.05, dtype=jnp.float64),
        predictor64,
    )
    eager32 = correction32.advance(
        *arguments64,
        inverse_momentum_diagonal=inverse.astype(jnp.float64),
        boundary_normal_velocity=boundary_velocity.astype(jnp.float64),
    )
    jitted32 = eqx.filter_jit(correction32.advance)(
        *arguments64,
        inverse_momentum_diagonal=inverse.astype(jnp.float64),
        boundary_normal_velocity=boundary_velocity.astype(jnp.float64),
    )
    assert eager32.velocity.dtype == jnp.float32
    assert eager32.pressure.dtype == jnp.float32
    assert eager32.divergence_history.dtype == jnp.float32
    assert jitted32.velocity.dtype == jnp.float32
    assert jitted32.pressure.dtype == jnp.float32
    assert eager32.converged == jitted32.converged
    assert jnp.all(jnp.isfinite(eager32.velocity))
    assert jnp.all(jnp.isfinite(jitted32.velocity))
    np.testing.assert_array_equal(
        jitted32.linear_status_history, eager32.linear_status_history
    )
    np.testing.assert_allclose(jitted32.velocity, eager32.velocity)
    np.testing.assert_allclose(jitted32.pressure, eager32.pressure)
