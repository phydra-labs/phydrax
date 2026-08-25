#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _polygon_centroid(vertices):
    points = np.asarray(vertices)
    following = np.roll(points, -1, axis=0)
    cross = points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    area = 0.5 * np.sum(cross)
    center = np.sum((points + following) * cross[:, None], axis=0) / (6.0 * area)
    return area, center


def test_mixed_triangle_quadrilateral_geometry_has_one_exact_cell_complex():
    vertices = np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
        )
    )
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        triangles=np.asarray(((1, 2, 5), (1, 5, 4))),
        quadrilaterals=np.asarray(((0, 1, 4, 3),)),
    ).prepare()

    assert discretization.cell_count == 3
    assert discretization.topology.dimension == 2
    assert isinstance(discretization, phx.discretization.PreparedFiniteVolumeGeometry)
    assert isinstance(discretization, phx.discretization.ExplicitFaceBlockGeometry)
    np.testing.assert_allclose(discretization.cell_volumes, (0.5, 0.5, 1.0))
    np.testing.assert_allclose(jnp.sum(discretization.cell_volumes), 2.0)
    np.testing.assert_allclose(
        jnp.sum(discretization.face_quadrature_weights, axis=1),
        discretization.face_measures,
    )
    assert discretization.quality.maximum_closure_residual < 1e-12
    for lower, upper in zip(
        discretization.topology.incidences[:-1],
        discretization.topology.incidences[1:],
        strict=True,
    ):
        assert (lower.scipy_boundary() @ upper.scipy_boundary()).nnz == 0
    owner_vector = (
        discretization.face_centers
        - discretization.cell_centers[discretization.owner_cells]
    )
    assert jnp.all(jnp.sum(owner_vector * discretization.area_vectors, axis=-1) > 0.0)
    shared = discretization.neighbour_cells >= 0
    assert jnp.sum(shared) == 2


def test_skewed_quadrilateral_uses_mapped_area_and_physical_centroid():
    vertices = np.asarray(((0.0, 0.0), (2.0, 0.0), (1.5, 1.0), (0.0, 1.0)))
    expected_area, expected_center = _polygon_centroid(vertices)
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(((0, 1, 2, 3),)),
    ).prepare()
    reversed_discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(((0, 3, 2, 1),)),
    ).prepare()

    np.testing.assert_allclose(discretization.cell_volumes[0], expected_area)
    np.testing.assert_allclose(discretization.cell_centers[0], expected_center)
    np.testing.assert_allclose(
        reversed_discretization.cell_volumes, discretization.cell_volumes
    )
    np.testing.assert_allclose(
        reversed_discretization.cell_centers, discretization.cell_centers
    )
    assert discretization.quality.maximum_closure_residual < 1e-12


@pytest.mark.parametrize(
    "vertices",
    (
        np.asarray(((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0))),
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.2, 0.2), (0.0, 1.0))),
    ),
)
def test_invalid_bilinear_quadrilaterals_are_rejected(vertices):
    with pytest.raises(ValueError, match="Quadrilateral"):
        phx.discretization.UnstructuredFiniteVolumePlan(
            vertices,
            quadrilaterals=np.asarray(((0, 1, 2, 3),)),
        )


def test_tetrahedral_geometry_has_exact_chain_orientation_and_face_closure():
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
    )
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        tetrahedra=np.asarray(((0, 1, 2, 3), (0, 2, 1, 4))),
    ).prepare()

    assert discretization.cell_count == 2
    assert discretization.topology.dimension == 3
    assert discretization.face_measures.size == 7
    np.testing.assert_allclose(discretization.cell_volumes, (1.0 / 6.0, 1.0 / 6.0))
    np.testing.assert_allclose(jnp.sum(discretization.cell_volumes), 1.0 / 3.0)
    np.testing.assert_allclose(
        jnp.sum(discretization.face_quadrature_weights, axis=1),
        discretization.face_measures,
    )
    assert discretization.quality.maximum_closure_residual < 1e-12
    for lower, upper in zip(
        discretization.topology.incidences[:-1],
        discretization.topology.incidences[1:],
        strict=True,
    ):
        product = lower.scipy_boundary() @ upper.scipy_boundary()
        product.eliminate_zeros()
        assert product.nnz == 0
    owner_vector = (
        discretization.face_centers
        - discretization.cell_centers[discretization.owner_cells]
    )
    assert jnp.all(jnp.sum(owner_vector * discretization.area_vectors, axis=-1) > 0.0)
    shared = discretization.neighbour_cells >= 0
    assert jnp.sum(shared) == 1


def test_tetrahedral_orientation_is_normalized_but_degeneracy_is_rejected():
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    forward = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, tetrahedra=np.asarray(((0, 1, 2, 3),))
    ).prepare()
    reversed_discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, tetrahedra=np.asarray(((0, 2, 1, 3),))
    ).prepare()
    np.testing.assert_allclose(reversed_discretization.cell_volumes, forward.cell_volumes)

    flat_vertices = vertices.copy()
    flat_vertices[3] = (0.25, 0.25, 0.0)
    with pytest.raises(ValueError, match="Tetrahedral"):
        phx.discretization.UnstructuredFiniteVolumePlan(
            flat_vertices, tetrahedra=np.asarray(((0, 1, 2, 3),))
        )


def test_global_ids_are_lossless_or_rejected_for_the_active_jax_width(tmp_path):
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    quadrilaterals = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    int32_max = np.iinfo(np.int32).max
    large_plan = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=quadrilaterals,
        vertex_global_ids=np.arange(int32_max + 1, int32_max + 5, dtype=np.int64),
        cell_global_ids=np.asarray((int32_max + 5,), dtype=np.int64),
    )
    assert large_plan.vertex_global_ids.dtype == jnp.int64
    assert large_plan.cell_global_ids.dtype == jnp.int64
    np.testing.assert_array_equal(
        large_plan.vertex_global_ids,
        np.arange(int32_max + 1, int32_max + 5, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="signed int64"):
        phx.discretization.UnstructuredFiniteVolumePlan(
            vertices,
            quadrilaterals=quadrilaterals,
            vertex_global_ids=np.asarray(
                (0, 1, 2, np.iinfo(np.int64).max + 1), dtype=np.uint64
            ),
        )

    boundary_plan = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=quadrilaterals,
        vertex_global_ids=np.arange(int32_max - 3, int32_max + 1, dtype=np.int64),
        cell_global_ids=np.asarray((int32_max,), dtype=np.int64),
    )
    boundary_archive = tmp_path / "int32-boundary.fvmesh"
    overflow_archive = tmp_path / "int32-overflow.fvmesh"
    phx.discretization.write_unstructured_fv_archive(boundary_archive, boundary_plan)
    phx.discretization.write_unstructured_fv_archive(overflow_archive, large_plan)
    script = r"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._array_archive import read_array_archive

jax.config.update("jax_enable_x64", False)
assert not bool(jax.config.read("jax_enable_x64"))
int32_max = np.iinfo(np.int32).max
vertices = np.asarray(
    ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
)
quadrilaterals = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
boundary = phx.discretization.UnstructuredFiniteVolumePlan(
    vertices,
    quadrilaterals=quadrilaterals,
    vertex_global_ids=np.arange(int32_max - 3, int32_max + 1, dtype=np.int64),
    cell_global_ids=np.asarray((int32_max,), dtype=np.int64),
)
assert boundary.vertex_global_ids.dtype == jnp.int32
assert boundary.cell_global_ids.dtype == jnp.int32
assert int(boundary.vertex_global_ids[-1]) == int32_max
try:
    phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=quadrilaterals,
        vertex_global_ids=np.arange(
            int32_max + 1, int32_max + 5, dtype=np.int64
        ),
    )
except ValueError as error:
    assert "signed int32" in str(error)
else:
    raise AssertionError("x64-disabled construction accepted an overflowing global ID")

_, canonical_arrays = read_array_archive(sys.argv[1])
assert canonical_arrays["vertex_global_ids"].dtype == np.dtype(np.int64)
assert canonical_arrays["cell_global_ids"].dtype == np.dtype(np.int64)
restored = phx.discretization.read_unstructured_fv_archive(sys.argv[1])
assert restored.vertex_global_ids.dtype == jnp.int32
assert restored.cell_global_ids.dtype == jnp.int32
assert int(restored.vertex_global_ids[-1]) == int32_max
try:
    phx.discretization.read_unstructured_fv_archive(sys.argv[2])
except ValueError as error:
    assert "signed int32" in str(error)
else:
    raise AssertionError("x64-disabled archive loading accepted an overflowing global ID")
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(boundary_archive),
            str(overflow_archive),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_tetrahedral_face_quadrature_is_degree_four_exact():
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, tetrahedra=np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    ).prepare()
    assert discretization.face_quadrature_points.shape == (4, 6, 3)
    assert discretization.face_quadrature_weights.shape == (4, 6)
    assert jnp.all(discretization.face_quadrature_weights > 0.0)
    static_metrics = phx.discretization.lower_static_unstructured_stage_metrics(
        discretization
    )
    assert static_metrics.face_blocks[0].layout.quadrature_count == 6

    faces = np.asarray(discretization.connectivity.faces)
    face = int(np.flatnonzero(np.all(faces == (0, 1, 2), axis=1))[0])
    points = np.asarray(discretization.face_quadrature_points[face])
    weights = np.asarray(discretization.face_quadrature_weights[face])
    for x_degree in range(5):
        for y_degree in range(5 - x_degree):
            observed = np.sum(
                weights * points[:, 0] ** x_degree * points[:, 1] ** y_degree
            )
            expected = (
                math.factorial(x_degree)
                * math.factorial(y_degree)
                / math.factorial(x_degree + y_degree + 2)
            )
            np.testing.assert_allclose(observed, expected, rtol=2e-13, atol=2e-14)


def test_tetrahedral_face_quadrature_commutes_with_rotation_and_scale():
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    tetrahedra = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    reference = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, tetrahedra=tetrahedra
    ).prepare()
    theta = 0.61
    phi = -0.37
    rotation_z = np.asarray(
        (
            (np.cos(theta), -np.sin(theta), 0.0),
            (np.sin(theta), np.cos(theta), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    rotation_x = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, np.cos(phi), -np.sin(phi)),
            (0.0, np.sin(phi), np.cos(phi)),
        )
    )
    rotation = rotation_z @ rotation_x
    scale = 2.75
    translation = np.asarray((0.8, -1.1, 0.35))
    transformed_vertices = translation + scale * (vertices @ rotation.T)
    transformed = phx.discretization.UnstructuredFiniteVolumePlan(
        transformed_vertices, tetrahedra=tetrahedra
    ).prepare()

    expected_points = translation + scale * (
        np.asarray(reference.face_quadrature_points) @ rotation.T
    )
    np.testing.assert_allclose(
        transformed.face_quadrature_points,
        expected_points,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        transformed.face_quadrature_weights,
        scale**2 * reference.face_quadrature_weights,
        rtol=2e-13,
        atol=2e-13,
    )


def test_moving_tetrahedral_quadrature_matches_static_six_point_rule():
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    plan = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, tetrahedra=np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    reference = plan.prepare()
    velocity = jnp.asarray((0.3, -0.2, 0.1))

    def translation(time, initial_vertices, args):
        del args
        return initial_vertices + time * velocity

    motion = phx.discretization.FixedConnectivityMotionPlan(
        plan, translation, mapping_id="tetra-six-point-translation"
    )
    step = motion.prepare_ssprk33_step(
        0.2,
        0.1,
        "tetra-six-point-epoch",
        0,
        0,
        None,
        prior_effective_cell_volumes=reference.cell_volumes,
    )
    assert bool(step.passed)

    stage_times = (
        (step.stage_1, 0.2),
        (step.stage_2, 0.3),
        (step.stage_3, 0.25),
        (step.accepted_geometry, 0.3),
    )
    expected_normal_velocity = (
        jnp.sum(velocity[None, :] * reference.area_vectors, axis=1)
        / reference.face_measures
    )
    for stage, time in stage_times:
        block = stage.face_blocks[0]
        assert block.layout.quadrature_count == 6
        assert block.layout.block_id == motion.face_layout.block_id
        np.testing.assert_allclose(
            block.quadrature_points,
            reference.face_quadrature_points + time * velocity,
            rtol=2e-13,
            atol=2e-13,
        )
        np.testing.assert_allclose(
            block.quadrature_weights,
            reference.face_quadrature_weights,
            rtol=2e-13,
            atol=2e-13,
        )
        np.testing.assert_allclose(
            block.quadrature_grid_normal_velocity,
            jnp.broadcast_to(
                expected_normal_velocity[:, None],
                block.quadrature_grid_normal_velocity.shape,
            ),
            rtol=2e-13,
            atol=2e-13,
        )
