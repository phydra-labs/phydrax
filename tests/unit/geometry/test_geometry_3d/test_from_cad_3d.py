#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import numpy as np
import pytest
import trimesh
from jax import numpy as jnp

from phydrax.domain.geometry3d import Geometry3DFromCAD


@pytest.fixture
def simple_cube_mesh():
    # Create a simple cube mesh using trimesh
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


@pytest.fixture
def geometry_from_cube(simple_cube_mesh):
    # Initialize Geometry3DFromCAD with the cube mesh
    return Geometry3DFromCAD(mesh=simple_cube_mesh, recenter=False)


def test_initialization(geometry_from_cube):
    geom = geometry_from_cube
    assert geom.mesh is not None
    assert geom.mesh_vertices.shape[1] == 3
    assert geom.mesh_faces.shape[1] == 3


def test_volume_property(geometry_from_cube):
    geom = geometry_from_cube
    expected_volume = 1.0  # Cube with side length 1m
    computed_volume = float(geom.volume)
    assert np.isclose(computed_volume, expected_volume, atol=1e-6)


def test_bounds_property(geometry_from_cube):
    geom = geometry_from_cube
    bounds = np.asarray(geom.bounds, dtype=float)
    expected_bounds = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    assert np.allclose(bounds, expected_bounds, atol=1e-6)


def test_contains_method(geometry_from_cube):
    geom = geometry_from_cube
    inside_point = jnp.array([[0.0, 0.0, 0.0]], dtype=float)
    outside_point = jnp.array([[2.0, 2.0, 2.0]], dtype=float)
    assert geom._contains(inside_point)[0]
    assert ~geom._contains(outside_point)[0]


def test_adf_batched_matches_vmap(geometry_from_cube):
    geom = geometry_from_cube
    key = jax.random.key(0)
    pts = jax.random.uniform(
        key,
        shape=(128, 3),
        minval=-0.75,
        maxval=0.75,
        dtype=float,
    )
    sdf_batched = geom.adf(pts)
    sdf_vmap = jax.vmap(geom.adf)(pts)
    assert np.allclose(np.asarray(sdf_batched), np.asarray(sdf_vmap), atol=1e-6)


def test_adf_jvp_batched_matches_vmap(geometry_from_cube):
    geom = geometry_from_cube
    key0, key1 = jax.random.split(jax.random.key(1), 2)
    pts = jax.random.uniform(
        key0,
        shape=(32, 3),
        minval=-0.75,
        maxval=0.75,
        dtype=float,
    )
    t_pts = jax.random.normal(key1, shape=(32, 3), dtype=float)

    _, tval_batched = jax.jvp(geom.adf, (pts,), (t_pts,))
    tval_vmap = jax.vmap(lambda p, tp: jax.jvp(geom.adf, (p,), (tp,))[1])(pts, t_pts)
    assert np.allclose(np.asarray(tval_batched), np.asarray(tval_vmap), atol=1e-6)


def test_mesh_sdf_fast_sign_matches_exact(geometry_from_cube):
    from phydrax.domain.geometry3d._sdf import make_mesh_sdf_fast

    geom = geometry_from_cube
    sdf_fast = make_mesh_sdf_fast(geom, beam_width=8, leaf_size=8)
    sdf_exact = geom._make_mesh_sdf()

    pts = jax.random.uniform(
        jax.random.key(2),
        shape=(512, 3),
        minval=-0.75,
        maxval=0.75,
        dtype=float,
    )
    ref = np.asarray(sdf_exact(pts))
    out = np.asarray(sdf_fast(pts))
    mask = np.abs(ref) > 1e-4
    assert np.all((out[mask] < 0.0) == (ref[mask] < 0.0))


def test_on_boundary_method(geometry_from_cube):
    geom = geometry_from_cube
    boundary_point = jnp.array([[0.5, 0.0, 0.0]], dtype=float)
    interior_point = jnp.array([[0.0, 0.0, 0.0]], dtype=float)
    assert geom._on_boundary(boundary_point)[0]
    assert ~geom._on_boundary(interior_point)[0]


def test_sample_boundary(geometry_from_cube):
    geom = geometry_from_cube
    num_points = 100
    sampled_points = geom.sample_boundary(num_points=num_points)
    assert sampled_points.shape == (num_points, 3)
    # Check if points are on boundary

    distances = jax.vmap(geom.adf_orig)(sampled_points)
    assert np.allclose(distances, 0.0, atol=1e-7)


def test_sample_interior(geometry_from_cube):
    geom = geometry_from_cube
    num_points = 100
    sampled_points = geom.sample_interior(num_points=num_points)
    assert sampled_points.shape == (num_points, 3)
    # Check if points are inside
    distances = jax.vmap(geom.adf_orig)(sampled_points)
    assert np.all(distances <= 0.0)


def test_geometry_from_cad_file(tmp_path):
    # Test initialization from a mesh file
    mesh = trimesh.creation.icosphere(radius=1.0)
    mesh_file = tmp_path / "sphere.stl"
    mesh.export(mesh_file)

    geom = Geometry3DFromCAD(mesh=mesh_file)
    assert geom.mesh is not None
    assert np.isclose(float(geom.volume), mesh.volume, atol=1e-6)


def test_boundary_normals(geometry_from_cube):
    geom = geometry_from_cube
    boundary_points = jnp.array(
        [
            [0.5, 0.0, 0.0],  # +X face
            [-0.5, 0.0, 0.0],  # -X face
            [0.0, 0.5, 0.0],  # +Y face
            [0.0, -0.5, 0.0],  # -Y face
            [0.0, 0.0, 0.5],  # +Z face
            [0.0, 0.0, -0.5],  # -Z face
        ],
        dtype=float,
    )

    expected_normals = np.array(
        [
            [1.0, 0.0, 0.0],  # +X face normal
            [-1.0, 0.0, 0.0],  # -X face normal
            [0.0, 1.0, 0.0],  # +Y face normal
            [0.0, -1.0, 0.0],  # -Y face normal
            [0.0, 0.0, 1.0],  # +Z face normal
            [0.0, 0.0, -1.0],  # -Z face normal
        ]
    )

    computed_normals = geom._boundary_normals(boundary_points)
    assert np.allclose(computed_normals, expected_normals, atol=1e-6)


def test_boundary_normals_edge_and_vertex(geometry_from_cube):
    geom = geometry_from_cube
    points = jnp.array(
        [
            [0.5, 0.5, 0.0],  # edge between +X/+Y faces
            [0.5, 0.5, 0.5],  # +X/+Y/+Z corner
        ],
        dtype=float,
    )
    normals = np.asarray(geom._boundary_normals(points), dtype=float)
    expected = np.asarray(
        [
            [1.0, 1.0, 0.0] / np.sqrt(2.0),
            [1.0, 1.0, 1.0] / np.sqrt(3.0),
        ],
        dtype=float,
    )
    assert np.allclose(normals, expected, atol=1e-6)


def test_boundary_normals_no_grad(geometry_from_cube):
    geom = geometry_from_cube
    point = jnp.array([0.6, 0.0, 0.0], dtype=float)

    def f(p):
        return jnp.sum(geom._boundary_normals(p))

    grad = jax.grad(f)(point)
    assert np.allclose(np.asarray(grad), 0.0, atol=1e-10)


def test_boundary_normals_jittable_batched(geometry_from_cube):
    geom = geometry_from_cube
    points = jnp.array(
        [
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.6, 0.2, -0.1],
        ],
        dtype=float,
    )

    normals = jax.jit(lambda p: geom._boundary_normals(p))(points)
    assert normals.shape == points.shape
    assert np.all(np.isfinite(np.asarray(normals)))


def test_sample_interior_separable(geometry_from_cube):
    """Test the _sample_interior_separable method of Geometry3DFromCAD."""
    import jax.random as jr
    import numpy as np

    key = jr.key(42)
    num_points = (100, 100, 100)
    sampled, mask = geometry_from_cube._sample_interior_separable(
        num_points, sampler="uniform", key=key
    )

    # Check that the returned values have the expected structure
    assert len(sampled) == 3  # Should return (x, y, z) coordinates
    sampled_x, sampled_y, sampled_z = sampled

    # Check that the mask has the expected shape
    assert mask.ndim == 3
    assert mask.shape == (sampled_x.shape[0], sampled_y.shape[0], sampled_z.shape[0])

    # Check that at least some points are inside the mesh
    assert np.any(mask)

    # Test with explicit dimensions for num_points
    key = jr.key(43)
    num_points_explicit = (10, 15, 20)
    sampled_explicit, mask_explicit = geometry_from_cube._sample_interior_separable(
        num_points_explicit, sampler="uniform", key=key
    )

    # Check that the dimensions match what we specified
    assert sampled_explicit[0].shape[0] == num_points_explicit[0]
    assert sampled_explicit[1].shape[0] == num_points_explicit[1]
    assert sampled_explicit[2].shape[0] == num_points_explicit[2]
    assert mask_explicit.shape == num_points_explicit

    # Test with where condition
    key = jr.key(44)

    def where_condition(point):
        # Only include points in the positive octant
        return (point[0] > 0) & (point[1] > 0) & (point[2] > 0)

    sampled_with_where, mask_with_where = geometry_from_cube._sample_interior_separable(
        num_points_explicit, where=where_condition, sampler="uniform", key=key
    )

    # Check that the mask respects the where condition
    # Find indices where all coordinates are positive
    positive_indices = np.where(
        (np.asarray(sampled_with_where[0])[:, np.newaxis, np.newaxis] > 0)
        & (np.asarray(sampled_with_where[1])[np.newaxis, :, np.newaxis] > 0)
        & (np.asarray(sampled_with_where[2])[np.newaxis, np.newaxis, :] > 0)
    )

    # For all positive indices, the mask should be True only if the point is inside the mesh
    for i, j, k in zip(*positive_indices):
        if mask_with_where[i, j, k]:
            # If the mask is True, the point should be inside the mesh
            point = np.array(
                [
                    float(sampled_with_where[0][i]),
                    float(sampled_with_where[1][j]),
                    float(sampled_with_where[2][k]),
                ]
            )
            # The point should be in the positive octant
            assert np.all(point > 0)
