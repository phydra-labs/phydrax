#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry import RigidFrame
from phydrax.optics.geometric import (
    PlanarRefractiveStack,
    SequentialOpticsStatus,
    trace_planar_refractive_stack,
)
from phydrax.velocimetry.camera import (
    BrownConradyDistortion,
    calibrate_camera_rig,
    CameraCalibrationPlan,
    CameraCalibrationProblem,
    CameraCalibrationStatus,
    CameraIntrinsics,
    CameraModel,
    CameraPose,
    CameraRig,
    pixels_to_rays,
    project_points,
    ProjectionStatus,
    RayStatus,
    triangulate_weighted_rays,
    TriangulationStatus,
)


def _camera(*, distortion=None, pose=None, refractive_stack=None):
    return CameraModel(
        CameraIntrinsics(
            (100.0, 200.0),
            (50.0, 60.0),
            image_shape=(101, 201),
        ),
        distortion=distortion,
        pose=pose,
        refractive_stack=refractive_stack,
    )


def test_zero_distortion_reduces_to_pinhole_and_rays_round_trip():
    camera = _camera(distortion=BrownConradyDistortion())
    point = jnp.asarray((1.0, 2.0, 10.0))

    projection = project_points(camera, point)
    ray = pixels_to_rays(camera, projection.pixels)

    np.testing.assert_allclose(projection.pixels, (70.0, 80.0), rtol=1e-6)
    np.testing.assert_allclose(
        ray.directions,
        point / jnp.sqrt(jnp.sum(point * point)),
        rtol=1e-6,
    )
    assert bool(projection.valid)
    assert bool(ray.valid)
    assert int(ray.iterations) == 0


def test_projection_uses_camera_to_world_pose_and_rejects_negative_depth():
    pose = CameraPose(RigidFrame(np.eye(3), np.asarray((1.0, 2.0, 3.0))))
    camera = _camera(pose=pose)

    centered = project_points(camera, jnp.asarray((1.0, 2.0, 5.0)))
    behind = project_points(camera, jnp.asarray((1.0, 2.0, 2.0)))

    np.testing.assert_allclose(centered.pixels, camera.intrinsics.principal_point)
    assert bool(centered.valid)
    assert not bool(behind.valid)
    assert int(behind.status) == int(ProjectionStatus.BEHIND_CAMERA)


def test_brown_conrady_projection_unprojection_is_consistent():
    camera = _camera(
        distortion=BrownConradyDistortion(
            radial=(0.08, -0.01, 0.002),
            tangential=(0.001, -0.002),
        )
    )
    direction = jnp.asarray((0.16, -0.11, 1.0))
    projection = project_points(camera, 4.0 * direction)
    ray = pixels_to_rays(camera, projection.pixels)

    expected = direction / jnp.sqrt(jnp.sum(direction * direction))
    np.testing.assert_allclose(ray.directions, expected, rtol=2e-5, atol=2e-6)
    assert int(ray.status) == int(RayStatus.SUCCESS)


def test_refraction_reports_total_internal_reflection_and_parallel_failure():
    stack = PlanarRefractiveStack(
        [[0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0]],
        [1.5, 1.0],
    )
    angle = np.deg2rad(60.0)
    tir = trace_planar_refractive_stack(
        stack,
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((np.sin(angle), 0.0, np.cos(angle))),
    )
    parallel = trace_planar_refractive_stack(
        stack,
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((1.0, 0.0, 0.0)),
    )

    assert not bool(tir.valid)
    assert int(tir.status) == int(SequentialOpticsStatus.TOTAL_INTERNAL_REFLECTION)
    assert not bool(parallel.valid)
    assert int(parallel.status) == int(SequentialOpticsStatus.PARALLEL)


def test_refractive_projection_reports_nonconvergence_without_a_fallback():
    stack = PlanarRefractiveStack(
        [[0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0]],
        [1.0, 1.33],
    )
    camera = _camera(refractive_stack=stack)

    result = project_points(
        camera,
        jnp.asarray((1.0, 0.0, 3.0)),
        refraction_maximum_iterations=0,
    )

    assert not bool(result.valid)
    assert int(result.status) == int(ProjectionStatus.REFRACTION_NONCONVERGENCE)


def test_refractive_camera_round_trip_and_explicit_status_mapping():
    stack = PlanarRefractiveStack(
        [[0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0]],
        [1.0, 1.33],
    )
    camera = _camera(refractive_stack=stack)
    point = jnp.asarray((0.25, -0.1, 3.0))

    projection = project_points(camera, point)
    ray = pixels_to_rays(camera, projection.pixels)
    offset = point - ray.origins

    assert bool(projection.valid)
    assert bool(ray.valid)
    np.testing.assert_allclose(
        jnp.cross(ray.directions, offset), jnp.zeros((3,)), atol=2e-6
    )
    assert float(jnp.sum(ray.directions * offset)) > 0.0

    tir_camera = _camera(
        refractive_stack=PlanarRefractiveStack(
            [[0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0]],
            [2.0, 1.0],
        )
    )
    tir = pixels_to_rays(tir_camera, jnp.asarray((100.0, 200.0)))
    assert not bool(tir.valid)
    assert int(tir.status) == int(RayStatus.TOTAL_INTERNAL_REFLECTION)

    parallel_camera = _camera(
        refractive_stack=PlanarRefractiveStack(
            [[1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [1.0, 1.2],
        )
    )
    parallel = pixels_to_rays(parallel_camera, parallel_camera.intrinsics.principal_point)
    assert not bool(parallel.valid)
    assert int(parallel.status) == int(RayStatus.PARALLEL_INTERFACE)


def test_all_ray_triangulation_and_parallel_ray_degeneracy():
    point = jnp.asarray((0.25, -0.5, 3.0))
    origins = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    directions = point[None, :] - origins
    solved = triangulate_weighted_rays(
        origins,
        directions,
        jnp.ones((3,), dtype=bool),
        jnp.ones((3,)),
    )
    degenerate = triangulate_weighted_rays(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        jnp.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))),
        jnp.ones((2,), dtype=bool),
        jnp.ones((2,)),
    )

    np.testing.assert_allclose(solved.point, point, rtol=1e-5, atol=1e-5)
    assert bool(solved.valid)
    assert int(solved.rank) == 3
    assert not bool(degenerate.valid)
    assert int(degenerate.rank) == 2
    assert int(degenerate.status) == int(TriangulationStatus.RANK_DEFICIENT)
    assert bool(jnp.all(jnp.isnan(degenerate.point)))


def test_calibration_reports_unobservable_free_focal_lengths():
    camera = _camera()
    rig = CameraRig((camera,))
    points = jnp.asarray(
        ((0.0, 0.0, 2.0), (0.0, 0.0, 3.0), (0.0, 0.0, 4.0), (0.0, 0.0, 5.0))
    )
    pixels = project_points(camera, points).pixels[None, ...]
    problem = CameraCalibrationProblem(
        rig,
        points,
        pixels,
        jnp.ones((1, 4), dtype=bool),
    )
    free = np.zeros((1, 16), dtype=bool)
    free[0, 0:2] = True
    result = calibrate_camera_rig(problem, CameraCalibrationPlan(free))

    assert not bool(result.valid)
    assert not bool(result.diagnostics.observable)
    assert int(result.diagnostics.rank) == 0
    assert int(result.status) == int(CameraCalibrationStatus.UNOBSERVABLE)
    assert result.optimization is None


def test_calibration_updates_preserve_the_refractive_stack():
    stack = PlanarRefractiveStack(
        [[0.0, 0.0, 0.5]],
        [[0.0, 0.0, 1.0]],
        [1.0, 1.0],
    )
    camera = _camera(refractive_stack=stack)
    rig = CameraRig((camera,))
    points = jnp.asarray(
        (
            (-0.2, -0.1, 2.0),
            (0.3, -0.15, 2.5),
            (-0.25, 0.2, 3.0),
            (0.2, 0.25, 3.5),
        )
    )
    problem = CameraCalibrationProblem(
        rig,
        points,
        project_points(camera, points).pixels[None, ...],
        jnp.ones((1, 4), dtype=bool),
    )
    free = np.zeros((1, 16), dtype=bool)
    free[0, 0] = True

    result = calibrate_camera_rig(problem, CameraCalibrationPlan(free))

    assert bool(result.diagnostics.observable)
    retained = result.rig.cameras[0].refractive_stack
    assert retained is not None
    assert retained.stack_id == stack.stack_id
    assert retained.capacity == stack.capacity
    assert retained.active_count == stack.active_count
    np.testing.assert_allclose(retained.interface_points, stack.interface_points)
    np.testing.assert_allclose(retained.interface_normals, stack.interface_normals)
    np.testing.assert_allclose(retained.refractive_indices, stack.refractive_indices)
    np.testing.assert_array_equal(retained.interface_active, stack.interface_active)


def test_reference_camera_gauge_requires_fixed_reference_pose():
    free = np.zeros((2, 16), dtype=bool)
    free[:, 0] = True
    free[0, 10] = True

    with pytest.raises(ValueError, match="reference camera pose"):
        CameraCalibrationPlan(
            free,
            gauge="reference-camera",
            reference_camera=0,
        )
