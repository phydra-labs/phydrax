#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _triangulated_grid(nx=3, ny=3):
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    vertices = np.asarray([(xi, yi) for yi in y for xi in x])
    triangles = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx + 1
            upper_right = upper_left + 1
            triangles.append((lower_left, lower_right, upper_right))
            triangles.append((lower_left, upper_right, upper_left))
    return vertices, np.asarray(triangles, dtype=np.int32)


def _scalar_system():
    velocity = jnp.asarray([0.7, -0.2])
    return phx.equations.ScalarConservationSystem(
        2,
        lambda state, axis, args: velocity[axis] * state,
        lambda left, right, axis, args: jnp.full(
            left.shape[:-1], jnp.abs(velocity[axis])
        ),
        system_id="triangle-advection",
    )


def _boundaries(discretization):
    return phx.discretization.TriangleFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )


def test_triangle_geometry_has_oriented_closed_control_volumes():
    vertices, triangles = _triangulated_grid(2, 2)
    discretization = phx.discretization.TriangleFiniteVolumePlan(
        vertices, triangles
    ).prepare()

    np.testing.assert_allclose(jnp.sum(discretization.cell_volumes), 1.0)
    assert discretization.cell_count == 8
    assert jnp.all(discretization.cell_volumes > 0.0)
    assert jnp.all(discretization.face_measures > 0.0)
    assert discretization.quality.maximum_closure_residual < 1e-12
    owner_vector = (
        discretization.face_centers
        - discretization.cell_centers[discretization.owner_cells]
    )
    assert jnp.all(jnp.sum(owner_vector * discretization.area_vectors, axis=-1) > 0.0)


def test_triangle_first_order_residual_preserves_constant_and_global_balance():
    vertices, triangles = _triangulated_grid(3, 3)
    discretization = phx.discretization.TriangleFiniteVolumePlan(
        vertices, triangles
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "triangle-advection",
        "state",
        _scalar_system(),
        _boundaries(discretization),
    )
    method = phx.discretization.TriangleFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    state = jnp.ones(discretization.state_shape)
    residual, diagnostics = compiled.residual_with_diagnostics(jnp.asarray(0.0), state)

    np.testing.assert_allclose(residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(diagnostics.conservation_defect, 0.0, atol=1e-12)
    assert compiled.stable_step(state) > 0.0


def test_triangle_hllc_euler_constant_state_and_ssprk_step():
    vertices, triangles = _triangulated_grid(2, 2)
    system = phx.equations.EulerSystem(2)
    discretization = phx.discretization.TriangleFiniteVolumePlan(
        vertices, triangles, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "triangle-euler", "state", system, _boundaries(discretization)
    )
    method = phx.discretization.TriangleFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    primitive = jnp.broadcast_to(
        jnp.asarray([1.0, 0.2, -0.1, 1.0]),
        (discretization.cell_count, 4),
    )
    state = system.primitive_to_conserved(primitive)
    result = phx.solver.UnsplitFiniteVolumeSSPRK3Plan(compiled.dynamics).advance(
        0.0, state, 0.001
    )

    np.testing.assert_allclose(result.state, state, atol=2e-12)
    assert jnp.all(system.admissible(result.state))


def test_triangle_wlsq_is_affine_exact_and_muscl_preserves_face_values():
    vertices, triangles = _triangulated_grid(4, 4)
    discretization = phx.discretization.TriangleFiniteVolumePlan(
        vertices, triangles
    ).prepare()
    wlsq = phx.discretization.PreparedTriangleWLSQ(discretization)
    centers = discretization.cell_centers
    values = (1.2 + 2.0 * centers[:, 0] - 0.7 * centers[:, 1])[:, None]
    gradient = wlsq.gradient(values)

    np.testing.assert_allclose(
        gradient[..., 0, :],
        jnp.broadcast_to(jnp.asarray([2.0, -0.7]), gradient[..., 0, :].shape),
        rtol=1e-11,
        atol=1e-11,
    )
    reconstruction = phx.discretization.TriangleMUSCLReconstructionPlan(
        wlsq, limiter="unlimited"
    )
    left, right = reconstruction.reconstruct(values)
    exact = (
        1.2
        + 2.0 * discretization.face_centers[:, 0]
        - 0.7 * discretization.face_centers[:, 1]
    )[:, None]
    np.testing.assert_allclose(left, exact, rtol=1e-11, atol=1e-11)
    interior = discretization.neighbour_cells >= 0
    np.testing.assert_allclose(right[interior], exact[interior], rtol=1e-11, atol=1e-11)


def test_triangle_mesh_archive_roundtrip(tmp_path):
    vertices, triangles = _triangulated_grid(2, 2)
    plan = phx.discretization.TriangleFiniteVolumePlan(vertices, triangles)
    path = tmp_path / "triangle-fv.mesh"
    phx.discretization.write_triangle_fv_archive(path, plan)
    restored = phx.discretization.read_triangle_fv_archive(path)

    assert restored.plan_id == plan.plan_id
    np.testing.assert_array_equal(restored.vertices, plan.vertices)
    np.testing.assert_array_equal(restored.triangles, plan.triangles)


def test_triangle_geometry_is_differentiable_at_fixed_topology():
    vertices, triangles = _triangulated_grid(2, 2)
    plan = phx.discretization.TriangleFiniteVolumePlan(vertices, triangles)
    prepared = plan.prepare()
    connectivity = prepared.connectivity
    owner = prepared.owner_cells
    owner_sign = prepared.owner_signs

    def total_area(scale):
        scaled = jnp.asarray(vertices).at[:, 0].multiply(scale)
        area, *_ = phx.discretization.evaluate_triangle_fv_geometry(
            scaled, triangles, connectivity, owner, owner_sign
        )
        return jnp.sum(area)

    value, tangent = jax.jvp(total_area, (jnp.asarray(1.3),), (jnp.asarray(1.0),))
    np.testing.assert_allclose(value, 1.3, rtol=1e-12)
    np.testing.assert_allclose(tangent, 1.0, rtol=1e-12)


def test_triangle_plan_rejects_duplicate_and_orientation_inconsistent_cells():
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    with pytest.raises(ValueError, match="duplicate"):
        phx.discretization.TriangleFiniteVolumePlan(
            vertices, jnp.asarray([[0, 1, 2], [2, 1, 0]])
        )
    with pytest.raises(ValueError, match="opposite orientation"):
        phx.discretization.TriangleFiniteVolumePlan(
            vertices, jnp.asarray([[0, 1, 2], [0, 1, 3]])
        ).prepare()


def test_triangle_compiler_threads_precision_and_rejects_unsupported_fields():
    vertices, triangles = _triangulated_grid(2, 2)
    discretization = phx.discretization.TriangleFiniteVolumePlan(
        vertices, triangles
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "triangle-precision",
        "state",
        _scalar_system(),
        _boundaries(discretization),
    )
    method = phx.discretization.TriangleFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    precision = phx.discretization.FiniteVolumePrecisionPolicy("float32")
    compiled = phx.equations.compile_conservation_problem(
        problem, discretization, method, precision=precision
    )
    state = jnp.ones(discretization.state_shape, dtype=jnp.float32)
    residual, diagnostics = compiled.residual_with_diagnostics(jnp.asarray(0.0), state)

    assert residual.dtype == jnp.float32
    assert dict(diagnostics.precision_evidence.observed)["compute"] == "float32"
    with pytest.raises(ValueError, match="capacity"):
        phx.equations.compile_conservation_problem(
            problem,
            discretization,
            method,
            capacity=jnp.ones((discretization.cell_count,)),
        )
    with pytest.raises(ValueError, match="bathymetry"):
        phx.equations.compile_conservation_problem(
            problem,
            discretization,
            method,
            bathymetry=jnp.ones((discretization.cell_count,)),
        )


def test_triangle_boundaries_reject_axis_only_and_direct_flux_policies():
    vertices, triangles = _triangulated_grid(1, 1)
    discretization = phx.discretization.TriangleFiniteVolumePlan(
        vertices, triangles
    ).prepare()
    patch = discretization.boundary_patch_names[0]
    with pytest.raises(TypeError, match="normal-oriented"):
        phx.discretization.TriangleFiniteVolumeBoundarySet(
            discretization.boundary_patch_names,
            {patch: phx.discretization.ReflectiveBoundary()},
        )

    def normal_flux(time, interior, coordinates, normal, args):
        del time, coordinates, normal, args
        return jnp.zeros_like(interior)

    with pytest.raises(TypeError, match="normal-oriented"):
        phx.discretization.TriangleFiniteVolumeBoundarySet(
            discretization.boundary_patch_names,
            {
                patch: phx.discretization.PrescribedNormalFluxBoundary(
                    normal_flux, boundary_id="direct-flux"
                )
            },
        )


def test_triangle_slip_wall_reflects_oblique_normal_velocity():
    system = phx.equations.EulerSystem(2)
    primitive = jnp.asarray([[1.0, 1.0, 0.0, 1.0]])
    state = system.primitive_to_conserved(primitive)
    normal = jnp.asarray([[2.0**-0.5, 2.0**-0.5]])
    reflected = phx.discretization.SlipWallBoundary().exterior_state(
        system,
        0.0,
        state,
        jnp.zeros((1, 2)),
        normal,
        0,
        None,
    )
    velocity_before = system.conserved_to_primitive(state)[..., 1:-1]
    velocity_after = system.conserved_to_primitive(reflected)[..., 1:-1]
    np.testing.assert_allclose(
        jnp.sum(velocity_after * normal, axis=-1),
        -jnp.sum(velocity_before * normal, axis=-1),
        atol=1e-12,
    )


def test_euler_normal_hllc_is_rotation_covariant():
    system = phx.equations.EulerSystem(2)
    solver = phx.discretization.HLLCFluxPlan()
    left_primitive = jnp.asarray([[1.0, 0.4, -0.1, 1.0]])
    right_primitive = jnp.asarray([[0.8, -0.2, 0.3, 0.7]])
    left = system.primitive_to_conserved(left_primitive)
    right = system.primitive_to_conserved(right_primitive)
    angle = 0.63
    rotation = jnp.asarray(
        [
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ]
    )
    normal = jnp.asarray([[0.6, 0.8]])
    rotated_normal = normal @ rotation.T

    def rotate_state(state):
        primitive = system.conserved_to_primitive(state)
        velocity = primitive[..., 1:-1] @ rotation.T
        return system.primitive_to_conserved(
            jnp.concatenate(
                (
                    primitive[..., :1],
                    velocity,
                    primitive[..., -1:],
                ),
                axis=-1,
            )
        )

    original = solver.normal_face_flux(system, left, right, normal)
    rotated = solver.normal_face_flux(
        system,
        rotate_state(left),
        rotate_state(right),
        rotated_normal,
    )
    expected_flux = original.normal_flux.at[..., 1:3].set(
        original.normal_flux[..., 1:3] @ rotation.T
    )

    np.testing.assert_allclose(rotated.normal_flux, expected_flux, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        rotated.max_speed, original.max_speed, rtol=2e-12, atol=2e-12
    )


def test_triangle_compiler_rejects_hllc_for_scalar_system():
    vertices, triangles = _triangulated_grid(2, 2)
    discretization = phx.discretization.TriangleFiniteVolumePlan(
        vertices, triangles
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "invalid-scalar-hllc",
        "state",
        _scalar_system(),
        _boundaries(discretization),
    )
    method = phx.discretization.TriangleFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    with pytest.raises(ValueError, match="Euler-compatible"):
        phx.equations.compile_conservation_problem(problem, discretization, method)


def test_triangle_compiler_rejects_muscl_from_different_geometry():
    vertices, triangles = _triangulated_grid(3, 3)
    first = phx.discretization.TriangleFiniteVolumePlan(vertices, triangles).prepare()
    shifted = np.asarray(vertices).copy()
    shifted[:, 0] *= 1.1
    second = phx.discretization.TriangleFiniteVolumePlan(shifted, triangles).prepare()
    reconstruction = phx.discretization.TriangleMUSCLReconstructionPlan(
        phx.discretization.PreparedTriangleWLSQ(first)
    )
    method = phx.discretization.TriangleFiniteVolumeMethodPlan(
        reconstruction,
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "mismatched-muscl",
        "state",
        _scalar_system(),
        _boundaries(second),
    )
    with pytest.raises(ValueError, match="different geometry"):
        phx.equations.compile_conservation_problem(problem, second, method)


def test_triangle_wlsq_is_affine_exact_on_skew_distorted_mesh():
    vertices, triangles = _triangulated_grid(5, 4)
    distorted = np.asarray(vertices).copy()
    interior = (
        (distorted[:, 0] > 0.0)
        & (distorted[:, 0] < 1.0)
        & (distorted[:, 1] > 0.0)
        & (distorted[:, 1] < 1.0)
    )
    distorted[interior, 0] += (
        0.035
        * np.sin(3.0 * np.pi * distorted[interior, 1])
        * np.sin(np.pi * distorted[interior, 0])
    )
    distorted[interior, 1] += (
        0.025
        * np.cos(2.0 * np.pi * distorted[interior, 0])
        * np.sin(np.pi * distorted[interior, 1])
    )
    discretization = phx.discretization.TriangleFiniteVolumePlan(
        distorted, triangles
    ).prepare()
    wlsq = phx.discretization.PreparedTriangleWLSQ(discretization)
    centers = discretization.cell_centers
    values = (-0.3 + 1.7 * centers[:, 0] + 0.45 * centers[:, 1])[:, None]
    gradient = wlsq.gradient(values)[..., 0, :]
    expected = jnp.broadcast_to(jnp.asarray([1.7, 0.45]), gradient.shape)

    np.testing.assert_allclose(gradient, expected, rtol=2e-11, atol=2e-11)
    assert wlsq.report.maximum_condition_number < 1e6


def test_triangle_muscl_reports_distorted_mesh_residual_order_and_conservation():
    errors = []
    conservation_defects = []
    velocity = jnp.asarray([0.8, -0.35])
    for width in (8, 16, 32):
        vertices, triangles = _triangulated_grid(width, width)
        distorted = np.asarray(vertices).copy()
        spacing = 1.0 / width
        interior_vertices = (
            (distorted[:, 0] > 0.0)
            & (distorted[:, 0] < 1.0)
            & (distorted[:, 1] > 0.0)
            & (distorted[:, 1] < 1.0)
        )
        distorted[interior_vertices, 0] += (
            0.12 * spacing * np.sin(3.0 * np.pi * distorted[interior_vertices, 1])
        )
        distorted[interior_vertices, 1] += (
            0.10 * spacing * np.cos(2.0 * np.pi * distorted[interior_vertices, 0])
        )
        discretization = phx.discretization.TriangleFiniteVolumePlan(
            distorted, triangles
        ).prepare()
        wlsq = phx.discretization.PreparedTriangleWLSQ(discretization)
        method = phx.discretization.TriangleFiniteVolumeMethodPlan(
            phx.discretization.TriangleMUSCLReconstructionPlan(wlsq, limiter="unlimited"),
            phx.discretization.RusanovFluxPlan(),
        )
        system = phx.equations.ScalarConservationSystem(
            2,
            lambda state, axis, args: velocity[axis] * state,
            lambda left, right, axis, args: jnp.full(
                left.shape[:-1], jnp.abs(velocity[axis])
            ),
            system_id=f"triangle-quadratic-advection-{width}",
        )
        problem = phx.equations.ConservationProblemIR(
            "triangle-quadratic-advection",
            "state",
            system,
            _boundaries(discretization),
        )
        compiled = phx.equations.compile_conservation_problem(
            problem, discretization, method
        )
        triangle_points = jnp.asarray(distorted)[jnp.asarray(triangles)]

        def average_square(coordinate):
            pair_sum = (
                coordinate[:, 0] * coordinate[:, 1]
                + coordinate[:, 0] * coordinate[:, 2]
                + coordinate[:, 1] * coordinate[:, 2]
            )
            return (jnp.sum(coordinate**2, axis=1) + pair_sum) / 6.0

        x = triangle_points[..., 0]
        y = triangle_points[..., 1]
        average_xy = (
            2.0 * jnp.sum(x * y, axis=1)
            + x[:, 0] * y[:, 1]
            + x[:, 1] * y[:, 0]
            + x[:, 0] * y[:, 2]
            + x[:, 2] * y[:, 0]
            + x[:, 1] * y[:, 2]
            + x[:, 2] * y[:, 1]
        ) / 12.0
        state = (average_square(x) + 0.5 * average_xy + 0.7 * average_square(y))[:, None]
        centers = discretization.cell_centers
        exact = -(
            velocity[0] * (2.0 * centers[:, 0] + 0.5 * centers[:, 1])
            + velocity[1] * (0.5 * centers[:, 0] + 1.4 * centers[:, 1])
        )
        residual, diagnostics = compiled.residual_with_diagnostics(
            jnp.asarray(0.0), state
        )
        flux, _ = compiled.face_fluxes(jnp.asarray(0.0), state)
        balance_terms = np.asarray(discretization.cell_volumes[:, None] * residual)
        integrated = np.asarray(flux * discretization.face_measures[:, None])
        boundary_terms = np.where(
            np.asarray(discretization.neighbour_cells < 0)[:, None],
            integrated,
            0.0,
        )
        expected_defect = math.fsum(
            balance_terms[:, 0].tolist() + boundary_terms[:, 0].tolist()
        )
        assert float(diagnostics.conservation_defect[0]) == expected_defect
        boundary_edges = np.asarray(discretization.connectivity.boundary_edges)
        cell_edges = np.asarray(discretization.connectivity.cell_edges)
        interior_cells = ~np.any(boundary_edges[cell_edges], axis=1)
        error = residual[:, 0] - exact
        weights = discretization.cell_volumes[interior_cells]
        errors.append(
            float(
                jnp.sqrt(jnp.sum(weights * error[interior_cells] ** 2) / jnp.sum(weights))
            )
        )
        conservation_defects.append(
            float(jnp.max(jnp.abs(diagnostics.conservation_defect)))
        )

    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert 0.85 < orders[-1] < 1.25
    assert max(conservation_defects) < 2e-11


def test_triangle_k_exact_reconstructs_true_quadratic_cell_averages():
    vertices, triangles = _triangulated_grid(5, 5)
    distorted = np.asarray(vertices).copy()
    interior = (
        (distorted[:, 0] > 0.0)
        & (distorted[:, 0] < 1.0)
        & (distorted[:, 1] > 0.0)
        & (distorted[:, 1] < 1.0)
    )
    distorted[interior, 0] += 0.02 * np.sin(3.0 * np.pi * distorted[interior, 1])
    distorted[interior, 1] += 0.015 * np.cos(2.0 * np.pi * distorted[interior, 0])
    discretization = phx.discretization.TriangleFiniteVolumePlan(
        distorted, triangles
    ).prepare()
    prepared = phx.discretization.PreparedTriangleQuadratic(discretization)
    moments = prepared.moments
    centers = discretization.cell_centers
    state = (
        1.0
        + 2.0 * centers[:, 0]
        - 0.7 * centers[:, 1]
        + 0.3 * (centers[:, 0] ** 2 + moments[:, 0])
        + 0.2 * (centers[:, 0] * centers[:, 1] + moments[:, 1])
        - 0.4 * (centers[:, 1] ** 2 + moments[:, 2])
    )[:, None]
    reconstruction = phx.discretization.TriangleKExactReconstructionPlan(prepared)
    left, right = reconstruction.reconstruct(state)
    face = discretization.face_centers
    exact = (
        1.0
        + 2.0 * face[:, 0]
        - 0.7 * face[:, 1]
        + 0.3 * face[:, 0] ** 2
        + 0.2 * face[:, 0] * face[:, 1]
        - 0.4 * face[:, 1] ** 2
    )[:, None]

    np.testing.assert_allclose(left, exact, rtol=2e-10, atol=2e-10)
    interior_faces = discretization.neighbour_cells >= 0
    np.testing.assert_allclose(
        right[interior_faces],
        exact[interior_faces],
        rtol=2e-10,
        atol=2e-10,
    )


def test_triangle_quadratic_conditioning_is_scale_invariant():
    vertices, triangles = _triangulated_grid(4, 4)
    conditions = []
    for scale in (1e-6, 1.0, 1e6):
        discretization = phx.discretization.TriangleFiniteVolumePlan(
            scale * vertices, triangles
        ).prepare()
        prepared = phx.discretization.PreparedTriangleQuadratic(discretization)
        conditions.append(float(prepared.report.maximum_condition_number))

    np.testing.assert_allclose(
        conditions,
        np.full((3,), conditions[1]),
        rtol=1e-10,
        atol=1e-10,
    )


def test_triangle_k_exact_cubic_residual_converges_second_order():
    errors = []
    velocity = jnp.asarray([0.8, -0.35])
    for width in (8, 16, 32):
        vertices, triangles = _triangulated_grid(width, width)
        distorted = np.asarray(vertices).copy()
        spacing = 1.0 / width
        interior = (
            (distorted[:, 0] > 0.0)
            & (distorted[:, 0] < 1.0)
            & (distorted[:, 1] > 0.0)
            & (distorted[:, 1] < 1.0)
        )
        distorted[interior, 0] += (
            0.08 * spacing * np.sin(3.0 * np.pi * distorted[interior, 1])
        )
        distorted[interior, 1] += (
            0.06 * spacing * np.cos(2.0 * np.pi * distorted[interior, 0])
        )
        discretization = phx.discretization.TriangleFiniteVolumePlan(
            distorted, triangles
        ).prepare()
        reconstruction = phx.discretization.TriangleKExactReconstructionPlan(
            phx.discretization.PreparedTriangleQuadratic(discretization)
        )
        method = phx.discretization.TriangleFiniteVolumeMethodPlan(
            reconstruction,
            phx.discretization.RusanovFluxPlan(),
        )
        system = phx.equations.ScalarConservationSystem(
            2,
            lambda state, axis, args: velocity[axis] * state,
            lambda left, right, axis, args: jnp.full(
                left.shape[:-1], jnp.abs(velocity[axis])
            ),
            system_id=f"triangle-cubic-advection-{width}",
        )
        problem = phx.equations.ConservationProblemIR(
            "triangle-cubic-advection",
            "state",
            system,
            _boundaries(discretization),
        )
        compiled = phx.equations.compile_conservation_problem(
            problem, discretization, method
        )
        points = jnp.asarray(distorted)[jnp.asarray(triangles)]

        def average_cube(coordinate):
            ordered = sum(
                coordinate[:, i] ** 2 * coordinate[:, j]
                for i in range(3)
                for j in range(3)
                if i != j
            )
            return (
                jnp.sum(coordinate**3, axis=1)
                + ordered
                + coordinate[:, 0] * coordinate[:, 1] * coordinate[:, 2]
            ) / 10.0

        x = points[..., 0]
        y = points[..., 1]
        state = (average_cube(x) + average_cube(y))[:, None]
        moments = phx.discretization.evaluate_triangle_second_moments(
            distorted, triangles
        )
        centers = discretization.cell_centers
        average_xx = centers[:, 0] ** 2 + moments[:, 0]
        average_yy = centers[:, 1] ** 2 + moments[:, 2]
        exact = -3.0 * (velocity[0] * average_xx + velocity[1] * average_yy)
        residual = compiled(jnp.asarray(0.0), state)[:, 0]
        interior_cells = (
            (centers[:, 0] > 0.25)
            & (centers[:, 0] < 0.75)
            & (centers[:, 1] > 0.25)
            & (centers[:, 1] < 0.75)
        )
        weights = discretization.cell_volumes[interior_cells]
        errors.append(
            float(
                jnp.sqrt(
                    jnp.sum(
                        weights * (residual[interior_cells] - exact[interior_cells]) ** 2
                    )
                    / jnp.sum(weights)
                )
            )
        )

    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert orders[-1] > 1.75, (errors, orders)


def test_triangle_k_exact_uses_shared_positivity_retry_runtime():
    vertices, triangles = _triangulated_grid(4, 4)
    system = phx.equations.EulerSystem(2)
    discretization = phx.discretization.TriangleFiniteVolumePlan(
        vertices,
        triangles,
        component_names=system.component_names,
    ).prepare()
    reconstruction = phx.discretization.TriangleKExactReconstructionPlan(
        phx.discretization.PreparedTriangleQuadratic(discretization)
    )
    method = phx.discretization.TriangleFiniteVolumeMethodPlan(
        reconstruction,
        phx.discretization.HLLCFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "triangle-runtime-euler",
        "state",
        system,
        _boundaries(discretization),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
    )
    primitive = jnp.broadcast_to(
        jnp.asarray([1.0, 0.1, -0.05, 1.0]),
        (discretization.cell_count, 4),
    )
    average = system.primitive_to_conserved(primitive)
    initial = runtime.initialize_state(average, 0.0, 0.001)
    np.testing.assert_allclose(initial.cell_average(), average)
    np.testing.assert_array_equal(
        initial.content_state.effective_cell_volumes,
        discretization.cell_volumes,
    )
    assert initial.content_state.topology_epoch_id == runtime.topology_epoch_id
    assert initial.topology_journal.current_epoch_id == runtime.topology_epoch_id
    result = runtime.advance(initial)

    assert result.accepted
    assert result.positivity.limited_state_valid
    ledger = result.accepted_flux_integrals
    assert ledger.units == "content"
    assert len(ledger.blocks) == 1
    assert ledger.blocks[0].flux_integral.shape == (
        discretization.face_measures.size,
        system.component_count,
    )
    assert ledger.blocks[0].block_id == discretization.face_block.block_id
    content_change = (
        result.runtime_state.content_state.conservative_content
        - initial.content_state.conservative_content
    )
    np.testing.assert_allclose(
        ledger.scatter_content_integral(),
        content_change,
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(ledger.source_integral, 0.0, atol=2e-11)
    source_sum, boundary_sum, net_cell_sum = ledger.conservation_sums()
    np.testing.assert_allclose(
        source_sum - boundary_sum,
        net_cell_sum,
        rtol=2e-11,
        atol=2e-11,
    )
    second = runtime.advance(result.runtime_state)
    assert tuple((block.block_id, block.route_id) for block in ledger.blocks) == tuple(
        (block.block_id, block.route_id)
        for block in second.accepted_flux_integrals.blocks
    )
    assert "accepted_" + "integrated_fluxes" not in vars(result)


def test_triangle_quadratic_moments_are_stable_under_large_translation():
    vertices, triangles = _triangulated_grid(4, 4)
    reference = phx.discretization.evaluate_triangle_second_moments(vertices, triangles)
    translated = phx.discretization.evaluate_triangle_second_moments(
        vertices + jnp.asarray([1e9, -1e9]), triangles
    )

    np.testing.assert_allclose(translated, reference, rtol=2e-6, atol=2e-9)


def _channel_triangle_plan(width=6):
    vertices, triangles = _triangulated_grid(width, width)
    edge_counts = {}
    for triangle in triangles:
        for start, stop in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(start), int(stop))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary_edges = np.asarray(
        [edge for edge, count in edge_counts.items() if count == 1],
        dtype=np.int32,
    )
    points = np.asarray(vertices)
    patches = {"bottom": [], "top": [], "sides": []}
    for edge in boundary_edges:
        coordinates = points[edge]
        if np.allclose(coordinates[:, 1], 0.0):
            patches["bottom"].append(edge)
        elif np.allclose(coordinates[:, 1], 1.0):
            patches["top"].append(edge)
        else:
            patches["sides"].append(edge)
    system = phx.equations.CompressibleNavierStokesSystem(
        phx.equations.ConstantTransport(0.2, 0.1), 2
    )
    return phx.discretization.TriangleFiniteVolumePlan(
        vertices,
        triangles,
        boundary_patches={
            name: np.asarray(edges, dtype=np.int32) for name, edges in patches.items()
        },
        component_names=system.component_names,
    )


def test_triangle_viscous_flux_recovers_affine_couette_stress():
    discretization = _channel_triangle_plan().prepare()
    system = phx.equations.CompressibleNavierStokesSystem(
        phx.equations.ConstantTransport(0.2, 0.1), 2
    )
    gradient = phx.discretization.PreparedTriangleWLSQ(discretization)
    viscous = phx.discretization.TriangleViscousFluxPlan(gradient)
    method = phx.discretization.TriangleFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
        viscous=viscous,
    )
    boundaries = phx.discretization.TriangleFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            "bottom": phx.discretization.NoSlipAdiabaticWallBoundary(
                jnp.asarray([0.0, 0.0])
            ),
            "top": phx.discretization.NoSlipAdiabaticWallBoundary(
                jnp.asarray([1.0, 0.0])
            ),
            "sides": phx.discretization.ExtrapolationBoundary(),
        },
    )
    centers = discretization.cell_centers
    primitive = jnp.stack(
        (
            jnp.ones(discretization.cell_count),
            centers[:, 1],
            jnp.zeros(discretization.cell_count),
            jnp.ones(discretization.cell_count),
        ),
        axis=-1,
    )
    state = system.primitive_to_conserved(primitive)
    flux = viscous.face_fluxes(system, 0.0, state, discretization, boundaries)
    owner_centers = centers[discretization.owner_cells]
    neighbour_centers = centers[jnp.maximum(discretization.neighbour_cells, 0)]
    interior = (
        (discretization.neighbour_cells >= 0)
        & jnp.all((owner_centers > 0.2) & (owner_centers < 0.8), axis=-1)
        & jnp.all(
            (neighbour_centers > 0.2) & (neighbour_centers < 0.8),
            axis=-1,
        )
    )
    normal = discretization.area_vectors / discretization.face_measures[:, None]
    expected_traction = 0.2 * jnp.stack((normal[:, 1], normal[:, 0]), axis=-1)
    np.testing.assert_allclose(
        flux[interior, 1:3],
        expected_traction[interior],
        rtol=2e-10,
        atol=2e-10,
    )
    report = viscous.stability_report(system, state, discretization)
    assert report.selected_step > 0.0
    problem = phx.equations.ConservationProblemIR(
        "triangle-couette", "state", system, boundaries
    )
    wall_patch_ids = jnp.asarray(
        [
            discretization.boundary_patch_names.index("bottom"),
            discretization.boundary_patch_names.index("top"),
        ]
    )
    wall_faces = jnp.isin(discretization.boundary_patch_ids, wall_patch_ids)
    np.testing.assert_allclose(
        flux[wall_faces, 1:3],
        expected_traction[wall_faces],
        rtol=2e-10,
        atol=2e-10,
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    assert compiled.stable_step(state) > 0.0


def test_triangle_thermal_wall_requires_viscous_closure_and_sets_heat_flux():
    discretization = _channel_triangle_plan(4).prepare()
    system = phx.equations.CompressibleNavierStokesSystem(
        phx.equations.ConstantTransport(0.1, 0.3), 2
    )
    isothermal = phx.discretization.NoSlipIsothermalWallBoundary(jnp.zeros((2,)), 2.0)
    boundaries = phx.discretization.TriangleFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {name: isothermal for name in discretization.boundary_patch_names},
    )
    inviscid_method = phx.discretization.TriangleFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "triangle-isothermal", "state", system, boundaries
    )
    with pytest.raises(ValueError, match="require viscous"):
        phx.equations.compile_conservation_problem(
            problem, discretization, inviscid_method
        )
    gradient = phx.discretization.PreparedTriangleWLSQ(discretization)
    viscous = phx.discretization.TriangleViscousFluxPlan(gradient)
    method = phx.discretization.TriangleFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
        viscous=viscous,
    )
    primitive = jnp.broadcast_to(
        jnp.asarray([1.0, 0.0, 0.0, 1.0]),
        (discretization.cell_count, 4),
    )
    state = system.primitive_to_conserved(primitive)
    flux = viscous.face_fluxes(system, 0.0, state, discretization, boundaries)
    boundary_faces = discretization.neighbour_cells < 0
    assert jnp.max(jnp.abs(flux[boundary_faces, -1])) > 0.0
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    normal = discretization.area_vectors / discretization.face_measures[:, None]
    owner_to_face = (
        discretization.face_centers
        - discretization.cell_centers[discretization.owner_cells]
    )
    projection = jnp.sum(owner_to_face * normal, axis=-1)
    expected_heat = 0.3 / projection
    np.testing.assert_allclose(
        flux[boundary_faces, -1],
        expected_heat[boundary_faces],
        rtol=2e-10,
        atol=2e-10,
    )
    assert jnp.all(jnp.isfinite(compiled(jnp.asarray(0.0), state)))


def test_triangle_reconstruction_precision_is_explicit_float32():
    vertices, triangles = _triangulated_grid(4, 4)
    system = phx.equations.EulerSystem(2)
    discretization = phx.discretization.TriangleFiniteVolumePlan(
        vertices,
        triangles,
        component_names=system.component_names,
    ).prepare()
    primitive = jnp.broadcast_to(
        jnp.asarray([1.0, 0.1, -0.05, 1.0], dtype=jnp.float32),
        (discretization.cell_count, 4),
    )
    state = system.primitive_to_conserved(primitive).astype(jnp.float32)
    wlsq = phx.discretization.PreparedTriangleWLSQ(discretization)
    gradient = wlsq.gradient(state)
    gradient_jit = eqx.filter_jit(wlsq.gradient)(state)
    assert gradient.dtype == jnp.float32
    assert gradient_jit.dtype == jnp.float32

    muscl = phx.discretization.TriangleMUSCLReconstructionPlan(wlsq, limiter="unlimited")
    muscl_left, muscl_right = muscl.reconstruct(state)
    muscl_left_jit, muscl_right_jit = eqx.filter_jit(muscl.reconstruct)(state)
    assert muscl_left.dtype == muscl_right.dtype == jnp.float32
    assert muscl_left_jit.dtype == muscl_right_jit.dtype == jnp.float32

    quadratic = phx.discretization.PreparedTriangleQuadratic(discretization)
    reconstruction = phx.discretization.TriangleKExactReconstructionPlan(quadratic)
    left, right = reconstruction.reconstruct_at(
        state, discretization.face_quadrature_points
    )
    left_jit, right_jit = eqx.filter_jit(reconstruction.reconstruct_at)(
        state, discretization.face_quadrature_points
    )
    assert left.dtype == right.dtype == jnp.float32
    assert left_jit.dtype == right_jit.dtype == jnp.float32

    method = phx.discretization.TriangleFiniteVolumeMethodPlan(
        reconstruction,
        phx.discretization.HLLCFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "triangle-float32",
        "state",
        system,
        _boundaries(discretization),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
        precision=phx.discretization.FiniteVolumePrecisionPolicy("float32"),
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
    )
    runtime_state = runtime.initialize_state(
        state,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1e-3, dtype=jnp.float32),
    )
    eager_result = runtime.advance(runtime_state)
    result = eqx.filter_jit(runtime.advance)(runtime_state)
    assert (
        eager_result.runtime_state.content_state.conservative_content.dtype == jnp.float32
    )
    assert eager_result.runtime_state.cell_average().dtype == jnp.float32
    assert eager_result.accepted_flux_integrals.blocks[0].flux_integral.dtype == (
        jnp.float32
    )
    assert eager_result.accepted_flux_integrals.source_integral.dtype == jnp.float32
    assert result.runtime_state.content_state.conservative_content.dtype == jnp.float32
    assert result.runtime_state.cell_average().dtype == jnp.float32
    assert result.accepted_flux_integrals.blocks[0].flux_integral.dtype == jnp.float32
    assert result.accepted_flux_integrals.source_integral.dtype == jnp.float32

    ns_system = phx.equations.CompressibleNavierStokesSystem(
        phx.equations.ConstantTransport(0.1, 0.2), 2
    )
    ns_state = ns_system.primitive_to_conserved(primitive).astype(jnp.float32)
    viscous = phx.discretization.TriangleViscousFluxPlan(wlsq)
    viscous_flux = viscous.face_fluxes(
        ns_system,
        jnp.asarray(0.0, dtype=jnp.float32),
        ns_state,
        discretization,
        _boundaries(discretization),
    )
    assert viscous_flux.dtype == jnp.float32
    viscous_flux_jit = eqx.filter_jit(viscous.face_fluxes)(
        ns_system,
        jnp.asarray(0.0, dtype=jnp.float32),
        ns_state,
        discretization,
        _boundaries(discretization),
    )
    assert viscous_flux_jit.dtype == jnp.float32


def test_finite_volume_geometry_protocols_do_not_force_tensor_connectivity():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4),
            phx.discretization.UniformCellAxisSpec(3),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    structured = phx.discretization.FiniteVolumePlan(grid).prepare()
    mapped = phx.discretization.MappedFiniteVolumePlan(
        structured,
        lambda point: point,
        mapping_id="protocol-identity",
    ).prepare()
    vertices, triangles = _triangulated_grid(2, 2)
    explicit = phx.discretization.TriangleFiniteVolumePlan(vertices, triangles).prepare()

    assert isinstance(structured, phx.discretization.PreparedFiniteVolumeGeometry)
    assert isinstance(mapped, phx.discretization.PreparedFiniteVolumeGeometry)
    assert not isinstance(structured, phx.discretization.ExplicitFaceBlockGeometry)
    assert not isinstance(mapped, phx.discretization.ExplicitFaceBlockGeometry)
    assert isinstance(explicit, phx.discretization.ExplicitFaceBlockGeometry)
