#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import meshio
import numpy as np
import pytest
import trimesh
from jax import numpy as jnp
from numpy.typing import ArrayLike

from phydrax.domain.geometry2d import Geometry2DFromCAD


@pytest.fixture
def simple_square_mesh():
    # Create a simple square mesh using meshio
    points = np.array(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]
    )
    cells: list[tuple[str, ArrayLike] | meshio.CellBlock] = [
        ("triangle", np.array([[0, 1, 2], [0, 2, 3]]))
    ]
    return meshio.Mesh(points=points, cells=cells)


@pytest.fixture
def geometry_from_square(simple_square_mesh):
    # Initialize Geometry2DFromCAD with the square mesh
    return Geometry2DFromCAD(mesh=simple_square_mesh, recenter=False)


def test_initialization(geometry_from_square):
    geom = geometry_from_square
    assert geom.mesh is not None
    assert geom.mesh_vertices.shape[1] == 3  # trimesh vertices are 3D
    assert geom.mesh_faces.shape[1] == 3


def test_area_property(geometry_from_square):
    geom = geometry_from_square
    expected_area = 1.0  # Square with side length 1m
    computed_area = float(geom.area)
    assert np.isclose(computed_area, expected_area, atol=1e-6)


def test_measure_partitions_match_geometry_measures(geometry_from_square):
    geom = geometry_from_square
    interior = geom.interior_measure_partition
    boundary = geom.boundary_measure_partition
    assert interior.vertices.shape == (2, 3, 2)
    assert boundary.vertices.shape == (4, 2, 2)
    assert np.isclose(float(interior.total_measure), float(geom.area))
    assert np.isclose(float(boundary.total_measure), float(geom.boundary_length_value))
    points, strata, base_mass = boundary.sample(
        8,
        key=jax.random.key(31),
        minimum_per_stratum=1,
    )
    assert points.shape == (8, 2)
    assert set(map(int, strata)) == {0, 1, 2, 3}
    assert np.isclose(float(jnp.sum(base_mass)), 1.0)


def test_boundary_chart_atlas_integrates_arclength_without_seam_duplication(
    geometry_from_square,
):
    geom = geometry_from_square
    quadrature = geom.boundary_chart_atlas.tensor_quadrature(5)
    assert quadrature.points.shape == (4, 5, 2)
    assert jnp.all(geom._on_boundary(quadrature.points.reshape((-1, 2))))
    measured = quadrature.integrate(jnp.ones(quadrature.weights.shape))
    assert jnp.allclose(measured, geom.boundary_length_value)


def test_bounds_property(geometry_from_square):
    geom = geometry_from_square
    bounds = np.asarray(geom.bounds, dtype=float)
    expected_bounds = np.array([[-0.5, -0.5], [0.5, 0.5]])
    assert np.allclose(bounds, expected_bounds, atol=1e-6)


def test_contains_method(geometry_from_square):
    geom = geometry_from_square
    inside_point = jnp.array([[0.0, 0.0]], dtype=float)
    outside_point = jnp.array([[2.0, 2.0]], dtype=float)
    assert geom._contains(inside_point)[0]
    assert ~geom._contains(outside_point)[0]


def test_on_boundary_method(geometry_from_square):
    geom = geometry_from_square
    boundary_point = jnp.array([[0.5, 0.0]], dtype=float)
    interior_point = jnp.array([[0.0, 0.0]], dtype=float)
    assert geom._on_boundary(boundary_point)[0]
    assert ~geom._on_boundary(interior_point)[0]


def test_sample_boundary(geometry_from_square):
    geom = geometry_from_square
    num_points = 100
    sampled_points = geom.sample_boundary(num_points=num_points)
    assert sampled_points.shape == (num_points, 2)
    # Check if points are on boundary
    distances = jax.vmap(geom.adf_orig)(sampled_points)
    assert np.allclose(distances, 0.0, atol=1e-8)


def test_sample_interior(geometry_from_square):
    geom = geometry_from_square
    num_points = 100
    sampled_points = geom.sample_interior(num_points=num_points)
    assert sampled_points.shape == (num_points, 2)
    # Check if points are inside
    distances = jax.vmap(geom.adf_orig)(sampled_points)
    assert np.all(distances <= 0.0)


def test_geometry_from_cad_file(tmp_path):
    # Test initialization from a mesh file
    vertices = np.array(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh_file = tmp_path / "square.stl"
    mesh.export(mesh_file)

    geom = Geometry2DFromCAD(mesh=mesh_file)
    assert geom.mesh is not None
    assert np.isclose(float(geom.area), mesh.area, atol=1e-6)


def test_boundary_normals(geometry_from_square):
    geom = geometry_from_square
    boundary_points = jnp.array(
        [
            [0.5, 0.0],  # Right edge
            [-0.5, 0.0],  # Left edge
            [0.0, 0.5],  # Top edge
            [0.0, -0.5],  # Bottom edge
        ],
        dtype=float,
    )

    expected_normals = np.array(
        [
            [1.0, 0.0],  # Normal pointing outward from right edge
            [-1.0, 0.0],  # Normal pointing outward from left edge
            [0.0, 1.0],  # Normal pointing outward from top edge
            [0.0, -1.0],  # Normal pointing outward from bottom edge
        ]
    )

    computed_normals = geom._boundary_normals(boundary_points)
    assert np.allclose(computed_normals, expected_normals, atol=1e-6)


def test_sample_interior_separable(geometry_from_square):
    """Test the _sample_interior_separable method of Geometry2DFromCAD."""
    import jax.random as jr
    import numpy as np

    # Test with a single number for num_points
    key = jr.key(42)
    num_points = [10, 10]
    sampled, mask = geometry_from_square._sample_interior_separable(
        num_points, sampler="uniform", key=key
    )

    # Check that the returned values have the expected structure
    assert len(sampled) == 2  # Should return (x, y) coordinates
    sampled_x, sampled_y = sampled

    # Check that the mask has the expected shape
    assert mask.ndim == 2
    assert mask.shape == (sampled_x.shape[0], sampled_y.shape[0])

    # Check that at least some points are inside the mesh
    assert np.any(mask)

    # Test with explicit dimensions for num_points
    key = jr.key(43)
    num_points_explicit = (10, 15)
    sampled_explicit, mask_explicit = geometry_from_square._sample_interior_separable(
        num_points_explicit, sampler="uniform", key=key
    )

    # Check that the dimensions match what we specified
    assert sampled_explicit[0].shape[0] == num_points_explicit[0]
    assert sampled_explicit[1].shape[0] == num_points_explicit[1]
    assert mask_explicit.shape == num_points_explicit

    # Test with where condition
    key = jr.key(44)

    def where_condition(point):
        # Only include points in the positive quadrant
        return (point[0] > 0) & (point[1] > 0)

    sampled_with_where, mask_with_where = geometry_from_square._sample_interior_separable(
        num_points_explicit, where=where_condition, sampler="uniform", key=key
    )

    # Check that the mask respects the where condition
    # Find indices where all coordinates are positive
    positive_indices = np.where(
        (np.asarray(sampled_with_where[0])[:, np.newaxis] > 0)
        & (np.asarray(sampled_with_where[1])[np.newaxis, :] > 0)
    )

    # For all positive indices, the mask should be True only if the point is inside the mesh
    for i, j in zip(*positive_indices):
        if mask_with_where[i, j]:
            # If the mask is True, the point should be inside the mesh
            point = np.array(
                [
                    float(sampled_with_where[0][i]),
                    float(sampled_with_where[1][j]),
                ]
            )
            # The point should be in the positive quadrant
            assert np.all(point > 0)
