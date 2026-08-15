#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import numpy as np
import pytest
import trimesh
from jax import numpy as jnp

import phydrax as phx


@pytest.fixture
def simple_cube_mesh():
    # Create a simple cube mesh using trimesh
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


@pytest.fixture
def geometry_from_cube(simple_cube_mesh):
    # Compile the mesh source and adapt it to the domain algebra.
    return phx.domain.GeometryDomain(
        phx.geometry.mesh_region_from_source(simple_cube_mesh, recenter=False).compile()
    )


def test_initialization(geometry_from_cube):
    assert isinstance(geometry_from_cube, phx.domain.GeometryDomain)
    assert geometry_from_cube.geometry.kind is phx.geometry.GeometryKind.REGION
    assert geometry_from_cube.geometry.has_capability(
        phx.geometry.GeometryCapability.REGION_QUERY
    )


def test_initialization_rejects_open_surface_mesh():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.update_faces(np.arange(mesh.faces.shape[0]) != 0)
    mesh.remove_unreferenced_vertices()

    with pytest.raises(ValueError, match="watertight"):
        phx.domain.GeometryDomain(
            phx.geometry.mesh_region_from_source(mesh, recenter=False).compile()
        )


def test_initialization_rejects_nonfinite_vertices():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    vertices = np.asarray(mesh.vertices).copy()
    vertices[0, 0] = np.nan
    invalid = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)

    with pytest.raises(ValueError, match="finite"):
        phx.domain.GeometryDomain(
            phx.geometry.mesh_region_from_source(invalid, recenter=False).compile()
        )


def test_volume_property(geometry_from_cube):
    geom = geometry_from_cube
    expected_volume = 1.0  # Cube with side length 1m
    computed_volume = float(geom.volume)
    assert np.isclose(computed_volume, expected_volume, atol=1e-6)


def test_boundary_partition_matches_surface_measure(geometry_from_cube):
    geom = geometry_from_cube
    partition = phx.geometry.BoundaryAtlasPartition(geom.boundary_atlas)
    assert partition.num_strata == 12
    assert np.isclose(float(partition.total_measure), float(geom.surface_area_value))
    points, strata, base_mass = partition.sample(
        24,
        key=jax.random.key(32),
        minimum_per_stratum=1,
    )
    assert points.shape == (24, 3)
    assert len(set(map(int, strata))) == partition.num_strata
    assert np.isclose(float(jnp.sum(base_mass)), 1.0)


def test_surface_chart_lowering_integrates_faces_without_seam_duplication(
    geometry_from_cube,
):
    geom = geometry_from_cube
    component = geom.component({"x": phx.domain.Boundary()})
    target = phx.integration.over(component)
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(4))
    realization = phx.integration.materialize(target, plan)
    points = realization.batch.points["x"].data
    surface_measure = phx.integration.reduce(1.0, realization)
    x_coordinate = geom.Function("x")(lambda x: x[0])
    x_moment = phx.integration.reduce(x_coordinate, realization)

    assert points.shape == (192, 3)
    assert jnp.all(geom._on_boundary(points))
    assert jnp.allclose(jnp.asarray(surface_measure.value.data), geom.surface_area_value)
    assert jnp.allclose(jnp.asarray(x_moment.value.data), 0.0, atol=1e-12)


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


def test_compiled_mesh_region_field_has_correct_sign(geometry_from_cube):
    points = jnp.asarray([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0], [0.5, 0.0, 0.0]])
    values = jax.jit(lambda value: geometry_from_cube.geometry.boundary_field(value))(
        points
    )
    assert values[0] < 0.0
    assert values[1] > 0.0
    assert values[2] == pytest.approx(0.0)


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

    distances = jax.vmap(geom.adf)(sampled_points)
    assert np.allclose(distances, 0.0, atol=1e-7)


def test_sample_interior(geometry_from_cube):
    geom = geometry_from_cube
    num_points = 100
    sampled_points = geom.sample_interior(num_points=num_points)
    assert sampled_points.shape == (num_points, 3)
    # Check if points are inside
    distances = jax.vmap(geom.adf)(sampled_points)
    assert np.all(distances <= 0.0)


def test_geometry_from_cad_file(tmp_path):
    # Test initialization from a mesh file
    mesh = trimesh.creation.icosphere(radius=1.0)
    mesh_file = tmp_path / "sphere.stl"
    mesh.export(mesh_file)

    geom = phx.domain.GeometryDomain(
        phx.geometry.mesh_region_from_source(mesh_file).compile()
    )
    assert isinstance(geom, phx.domain.GeometryDomain)
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


def test_boundary_normals_at_nonsmooth_features_are_valid_subgradients(
    geometry_from_cube,
):
    points = jnp.array(
        [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=float,
    )
    normals = np.asarray(geometry_from_cube._boundary_normals(points), dtype=float)
    assert np.allclose(np.linalg.norm(normals, axis=-1), 1.0)
    assert np.all(normals * np.asarray(points) >= -1e-12)
    assert np.all(np.sum(normals * np.asarray(points), axis=-1) > 0.0)


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
    """Test separable interior sampling through the geometry domain adapter."""
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


def test_boundary_factor_is_scale_covariant_with_unit_face_gradient():
    scales = (1e-7, 1.0)
    normalized_points = jnp.array(
        [
            [0.5, 0.0, 0.0],
            [0.49, 0.1, -0.1],
            [0.4, -0.15, 0.15],
            [0.0, 0.0, 0.0],
            [0.75, 0.1, -0.1],
        ]
    )
    normalized_values = []
    normalized_ansatz_values = []
    normalized_gate_values = []
    normalized_gate_gradients = []
    normalized_gate_midpoints = []

    for scale in scales:
        geometry = phx.domain.GeometryDomain(
            phx.geometry.mesh_region_from_source(
                trimesh.creation.box(extents=(scale, scale, scale)),
                recenter=False,
            ).compile()
        )
        boundary_point = jnp.array([0.5 * scale, 0.0, 0.0])
        assert abs(float(geometry.adf(boundary_point))) <= 1e-12 * scale
        assert jnp.allclose(
            jax.grad(geometry.adf)(boundary_point),
            jnp.array([1.0, 0.0, 0.0]),
            atol=1e-12,
            rtol=0.0,
        )
        ansatz_factor = geometry.boundary_ansatz_factor
        assert abs(float(ansatz_factor(boundary_point))) <= 1e-12 * scale
        assert jnp.allclose(
            jax.grad(ansatz_factor)(boundary_point),
            jnp.array([1.0, 0.0, 0.0]),
            atol=1e-10,
            rtol=0.0,
        )
        normalized_ansatz_values.append(ansatz_factor(scale * normalized_points) / scale)
        gate = geometry.make_enforcement_gate()
        normalized_gate_values.append(gate(scale * normalized_points))
        normalized_gate_gradients.append(scale * jax.grad(gate)(boundary_point))
        normalized_gate_midpoints.append(gate(jnp.array([0.25 * scale, 0.0, 0.0])))
        normalized_values.append(geometry.adf(scale * normalized_points) / scale)

    assert jnp.allclose(
        normalized_values[0],
        normalized_values[1],
        atol=1e-10,
        rtol=1e-10,
    )
    assert jnp.allclose(
        normalized_ansatz_values[0],
        normalized_ansatz_values[1],
        atol=1e-10,
        rtol=1e-10,
    )
    assert jnp.allclose(
        normalized_gate_values[0],
        normalized_gate_values[1],
        atol=1e-10,
        rtol=1e-10,
    )
    assert jnp.allclose(
        normalized_gate_gradients[0],
        normalized_gate_gradients[1],
        atol=1e-10,
        rtol=1e-10,
    )
    assert jnp.allclose(
        normalized_gate_midpoints[0],
        normalized_gate_midpoints[1],
        atol=1e-10,
        rtol=1e-10,
    )
    assert float(normalized_gate_midpoints[0]) > 0.5
    assert jnp.allclose(normalized_gate_values[0][0], 0.0, atol=1e-10)
    assert float(normalized_gate_values[0][3]) > 0.9
