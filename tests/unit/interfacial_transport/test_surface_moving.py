#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.interfacial_transport as it
from phydrax.discretization import TopologyEpoch
from phydrax.geometry.simplicial import TriangleMesh
from tests._support.film_meshes import icosphere, planar_grid


def _wavy_plane(cells: int) -> TriangleMesh:
    """Unit grid on ``z = 0.1 sin(2 pi x)``: planar strips between x-grid lines."""
    mesh = planar_grid(cells, cells)
    points = np.asarray(mesh.vertices).copy()
    points[:, 2] = 0.1 * np.sin(2.0 * np.pi * points[:, 0])
    return TriangleMesh(points, np.asarray(mesh.topology.faces))


def _smooth_field(points: np.ndarray) -> np.ndarray:
    """A smooth non-homothetic, non-rigid displacement field with normal parts."""
    x, y, z = points.T
    return np.stack(
        (
            0.3 * x * y + 0.05 * np.sin(3.0 * y),
            0.2 * z**2 - 0.1 * x,
            0.15 * np.sin(2.0 * x) + 0.1 * y * z,
        ),
        axis=1,
    )


def test_tangential_mesh_motion_of_a_fixed_surface_keeps_uniform_content_uniform() -> (
    None
):
    surface = it.prepare_film_surface(_wavy_plane(16))
    points = np.asarray(surface.coordinates)
    # Vertices slide along the straight x-grid lines of a ruled surface, so the
    # discrete surface is unchanged while the mesh deforms non-homothetically.
    slide = 0.02 * np.sin(np.pi * points[:, 1]) * (1.0 + 0.5 * np.cos(3.0 * points[:, 0]))
    target = points + np.stack((0.0 * slide, slide, 0.0 * slide), axis=1)
    motion = it.SurfaceMeshMotion(surface, target, 0.1)
    assert int(motion.evidence.status) == it.SurfaceMotionStatus.ACCEPTED
    assert float(motion.evidence.maximum_relative_gcl_residual) < 1e-14
    np.testing.assert_allclose(motion.normal_velocity, 0.0, atol=1e-14)
    thickness = 2e-6
    at_rest = motion.material_velocity(jnp.zeros_like(target))
    moved = motion.transport(thickness * surface.vertex_area, at_rest)
    assert bool(moved.accepted)
    np.testing.assert_allclose(
        moved.content / motion.target.vertex_area, thickness, rtol=1e-13
    )
    assert abs(float(moved.content_residual)) <= 1e-14 * thickness
    assert int(motion.target.geometry_revision) == int(surface.geometry_revision) + 1


@pytest.mark.parametrize("surface_kind", ["sphere", "wavy-plane"])
def test_arbitrary_motion_obeys_the_exact_finite_step_content_area_relation(
    surface_kind: str,
) -> None:
    mesh = icosphere(2) if surface_kind == "sphere" else _wavy_plane(12)
    surface = it.prepare_film_surface(mesh)
    points = np.asarray(surface.coordinates)
    step_size = 0.2
    motion = it.SurfaceMeshMotion(
        surface, points + step_size * _smooth_field(points), step_size
    )
    evidence = motion.evidence
    assert int(evidence.status) == it.SurfaceMotionStatus.ACCEPTED
    assert float(evidence.maximum_relative_gcl_residual) < 1e-14
    np.testing.assert_allclose(
        motion.target.vertex_area,
        surface.vertex_area + step_size * motion.area_rate_m2_s,
        rtol=1e-14,
    )
    # A Lagrangian film keeps each cell's content; its density follows the
    # exactly integrated dual-area rate.
    density = 1e-6 * (1.0 + 0.3 * points[:, 0])
    content = density * surface.vertex_area
    carried = motion.transport(content, motion.mesh_velocity)
    assert bool(carried.accepted)
    np.testing.assert_allclose(
        carried.content / motion.target.vertex_area,
        content / (surface.vertex_area + step_size * motion.area_rate_m2_s),
        rtol=1e-13,
    )
    # Material moving tangentially relative to the mesh exchanges content
    # conservatively.
    swirl = 0.1 * np.stack((-points[:, 1], points[:, 0], 0.0 * points[:, 0]), axis=1)
    relative = motion.tangential_mesh_velocity + swirl
    moved = motion.transport(content, motion.material_velocity(relative))
    assert bool(moved.accepted)
    assert float(jnp.max(jnp.abs(moved.content - content))) > 1e-3 * float(
        np.max(content)
    )
    assert abs(float(moved.content_residual)) <= 1e-14 * float(np.sum(content))


def test_expanding_closed_surface_conserves_total_content() -> None:
    surface = it.prepare_film_surface(icosphere(2))
    rng = np.random.default_rng(11)
    count = surface.topology.num_vertices
    content = surface.vertex_area * 1e-6 * rng.uniform(0.5, 1.5, count)
    motion = it.SurfaceMeshMotion(surface, 1.1 * surface.coordinates, 0.2)
    assert float(motion.evidence.maximum_relative_gcl_residual) < 1e-14
    np.testing.assert_allclose(motion.normal_velocity, 0.5, rtol=1e-3)
    eulerian = motion.material_velocity(jnp.zeros((count, 3)))
    moved = motion.transport(content, eulerian)
    assert abs(float(moved.content_residual)) <= 1e-15 * float(np.sum(content))
    lagrangian = motion.transport(1e-6 * surface.vertex_area, motion.mesh_velocity)
    np.testing.assert_allclose(
        lagrangian.content / motion.target.vertex_area, 1e-6 / 1.1**2, rtol=1e-12
    )
    shrink = it.SurfaceMeshMotion(motion.target, surface.coordinates, 0.2)
    back = shrink.transport(
        moved.content, shrink.material_velocity(jnp.zeros((count, 3)))
    )
    np.testing.assert_allclose(
        float(jnp.sum(back.content)), float(np.sum(content)), rtol=1e-15
    )


def test_invalid_motions_fail_closed() -> None:
    sphere = it.prepare_film_surface(icosphere(1))
    content = 1e-6 * sphere.vertex_area
    # Far translation cancels the coordinate differences of the target: its
    # dual measures no longer match the exactly integrated area rate.
    far = it.SurfaceMeshMotion(sphere, sphere.coordinates + 1e9, 1.0)
    assert int(far.evidence.status) == it.SurfaceMotionStatus.GCL_VIOLATED
    assert not bool(far.evidence.valid)
    refused = far.transport(content, far.mesh_velocity)
    assert int(refused.status) == it.SurfaceMotionStatus.GCL_VIOLATED
    np.testing.assert_array_equal(refused.content, content)
    mirrored = it.SurfaceMeshMotion(
        sphere, sphere.coordinates * jnp.asarray((-1.0, 1.0, 1.0)), 1.0
    )
    assert int(mirrored.evidence.status) == it.SurfaceMotionStatus.ORIENTATION_REVERSED
    plane = it.prepare_film_surface(planar_grid(8, 8))
    points = np.asarray(plane.coordinates)
    bubble = np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])
    slide = it.SurfaceMeshMotion(
        plane, points + 0.25 * np.stack((bubble, 0.0 * bubble, 0.0 * bubble), 1), 0.1
    )
    assert bool(slide.evidence.valid)
    uniform = 1e-6 * plane.vertex_area
    fast = slide.transport(uniform, jnp.zeros_like(slide.mesh_velocity))
    assert int(fast.status) == it.SurfaceMotionStatus.COURANT_LIMIT
    assert float(fast.courant_number) > 1.0
    np.testing.assert_array_equal(fast.content, uniform)


def test_material_velocity_equal_to_mesh_velocity_is_lagrangian() -> None:
    surface = it.prepare_film_surface(planar_grid(10, 10))
    points = np.asarray(surface.coordinates)
    bubble = np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])
    target = points + 0.03 * np.stack(
        (bubble, np.zeros_like(bubble), np.zeros_like(bubble)), 1
    )
    motion = it.SurfaceMeshMotion(surface, target, 0.5)
    content = 1e-6 * surface.vertex_area * (1.0 + points[:, 0])
    moved = motion.transport(content, motion.mesh_velocity)
    np.testing.assert_allclose(moved.content, content, rtol=1e-14)


def test_fixed_topology_motion_is_differentiable() -> None:
    surface = it.prepare_film_surface(planar_grid(6, 6))
    points = np.asarray(surface.coordinates)
    bubble = np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])
    direction = jnp.asarray(np.stack((bubble, bubble, np.zeros_like(bubble)), 1))
    content = 1e-6 * surface.vertex_area * (1.0 + points[:, 0])

    def moved_total_left(scale: jax.Array) -> jax.Array:
        motion = it.SurfaceMeshMotion(
            surface, surface.coordinates + scale * direction, 0.1
        )
        moved = motion.transport(content, jnp.zeros_like(direction)).content
        return jnp.sum(jnp.where(points[:, 0] < 0.5, moved, 0.0))

    gradient = jax.grad(moved_total_left)(jnp.asarray(0.01))
    step = jnp.asarray(1e-6, dtype=jnp.float64)
    finite = (moved_total_left(0.01 + step) - moved_total_left(0.01 - step)) / (2 * step)
    np.testing.assert_allclose(gradient, finite, rtol=1e-6)


def test_epoch_transfer_conserves_content_and_reports_no_derivative() -> None:
    source = TopologyEpoch(0, "geometry-a", "topology-a", "whole")
    target = TopologyEpoch(1, "geometry-b", "topology-b", "whole")
    transfer = it.SurfaceEpochTransfer(
        source,
        target,
        jnp.asarray((0, 1, 1, 2), dtype=jnp.int32),
        jnp.asarray((0, 0, 1, 2), dtype=jnp.int32),
        jnp.asarray((0.25, 0.75, 1.0, 1.0), dtype=jnp.float64),
        source_size=3,
        target_size=3,
    )
    result = transfer.apply(jnp.asarray((4.0, 2.0, 1.0)))
    np.testing.assert_allclose(result.content, (1.0, 5.0, 1.0))
    assert float(result.conservation_residual) == 0.0
    assert not bool(result.derivative_available)
    with pytest.raises(ValueError, match="all of its content"):
        it.SurfaceEpochTransfer(
            source,
            target,
            jnp.asarray((0,), dtype=jnp.int32),
            jnp.asarray((0,), dtype=jnp.int32),
            jnp.asarray((0.5,), dtype=jnp.float64),
            source_size=1,
            target_size=1,
        )
