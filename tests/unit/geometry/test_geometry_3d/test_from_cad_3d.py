#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import numpy as np
import pytest
import trimesh
from jax import numpy as jnp

import phydrax as phx


@pytest.fixture
def simple_cube_mesh() -> Any:
    # Create a simple cube mesh using trimesh
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


@pytest.fixture
def geometry_from_cube(simple_cube_mesh: Any) -> Any:
    # Compile the mesh source and adapt it to the domain algebra.
    return phx.domain.GeometryDomain(
        phx.geometry.mesh_region_from_source(
            (np.asarray(simple_cube_mesh.vertices), np.asarray(simple_cube_mesh.faces)),
            recenter=False,
        ).compile()
    )


def test_mesh_region_geometry_preserves_measure_bounds_membership_and_compiled_sign(
    geometry_from_cube: Any,
) -> None:
    geometry = geometry_from_cube
    assert isinstance(geometry, phx.domain.GeometryDomain)
    assert geometry.geometry.kind is phx.geometry.GeometryKind.REGION
    assert geometry.geometry.has_capability(phx.geometry.GeometryCapability.REGION_QUERY)
    assert np.isclose(float(geometry.volume), 1.0, atol=1e-6)
    assert np.allclose(
        np.asarray(geometry.bounds, dtype=np.float64),
        np.asarray([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]),
        atol=1e-6,
    )
    assert bool(geometry._contains(jnp.asarray([[0.0, 0.0, 0.0]]))[0])
    assert not bool(geometry._contains(jnp.asarray([[2.0, 2.0, 2.0]]))[0])
    assert bool(geometry._on_boundary(jnp.asarray([[0.5, 0.0, 0.0]]))[0])
    assert not bool(geometry._on_boundary(jnp.asarray([[0.0, 0.0, 0.0]]))[0])
    field = jax.jit(lambda value: geometry.geometry.boundary_field(value))(
        jnp.asarray([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0], [0.5, 0.0, 0.0]])
    )
    assert field[0] < 0.0
    assert field[1] > 0.0
    assert field[2] == pytest.approx(0.0)


def test_from_cad_3d_scenario_1() -> None:
    open_mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    open_mesh.update_faces(np.arange(open_mesh.faces.shape[0]) != 0)
    open_mesh.remove_unreferenced_vertices()
    with pytest.raises(ValueError, match="watertight"):
        phx.domain.GeometryDomain(
            phx.geometry.mesh_region_from_source(
                (np.asarray(open_mesh.vertices), np.asarray(open_mesh.faces)),
                recenter=False,
            ).compile()
        )

    nonfinite_mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    vertices = np.asarray(nonfinite_mesh.vertices).copy()
    vertices[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        phx.domain.GeometryDomain(
            phx.geometry.mesh_region_from_source(
                (vertices, np.asarray(nonfinite_mesh.faces)),
                recenter=False,
            ).compile()
        )
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
                (
                    np.asarray(
                        trimesh.creation.box(extents=(scale, scale, scale)).vertices
                    ),
                    np.asarray(trimesh.creation.box(extents=(scale, scale, scale)).faces),
                ),
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


def test_boundary_partition_and_surface_chart_match_the_surface_measure(
    geometry_from_cube: Any,
) -> None:
    geometry = geometry_from_cube
    partition = phx.geometry.BoundaryAtlasPartition(geometry.boundary_atlas)
    assert partition.num_strata == 12
    assert np.isclose(
        float(partition.total_measure),
        float(geometry.surface_area_value),
    )
    sampled, strata, base_mass = partition.sample(
        24,
        key=jax.random.key(32),
        minimum_per_stratum=1,
    )
    assert sampled.shape == (24, 3)
    assert len(set(map(int, strata))) == partition.num_strata
    assert np.isclose(float(jnp.sum(base_mass)), 1.0)

    component = geometry.component({"x": phx.domain.Boundary()})
    realization = phx.integration.materialize(
        phx.integration.over(component),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(4)),
    )
    points = realization.batch.points["x"].data
    surface_measure = phx.integration.reduce(1.0, realization)
    x_coordinate = geometry.Function("x")(lambda x: x[0])
    x_moment = phx.integration.reduce(x_coordinate, realization)
    assert points.shape == (192, 3)
    assert jnp.all(geometry._on_boundary(points))
    assert jnp.allclose(
        jnp.asarray(surface_measure.value.data),
        geometry.surface_area_value,
    )
    assert jnp.allclose(jnp.asarray(x_moment.value.data), 0.0, atol=1e-12)


def test_batched_adf_and_jvp_match_scalar_vmap_references(
    geometry_from_cube: Any,
) -> None:
    geometry = geometry_from_cube
    key0, key1 = jax.random.split(jax.random.key(1), 2)
    points = jax.random.uniform(
        key0,
        shape=(128, 3),
        minval=-0.75,
        maxval=0.75,
        dtype=jnp.float64,
    )
    np.testing.assert_allclose(
        geometry.adf(points),
        jax.vmap(geometry.adf)(points),
        atol=1e-6,
    )

    tangent_points = points[:32]
    tangents = jax.random.normal(key1, shape=(32, 3), dtype=jnp.float64)
    _, batched = jax.jvp(geometry.adf, (tangent_points,), (tangents,))
    scalar = jax.vmap(
        lambda point, tangent: jax.jvp(
            geometry.adf,
            (point,),
            (tangent,),
        )[1]
    )(tangent_points, tangents)
    np.testing.assert_allclose(batched, scalar, atol=1e-6)


def test_mesh_region_sampling_respects_boundary_and_interior_support(
    geometry_from_cube: Any,
) -> None:
    geometry = geometry_from_cube
    boundary = geometry.sample_boundary(num_points=100)
    interior = geometry.sample_interior(num_points=100)
    assert boundary.shape == interior.shape == (100, 3)
    assert np.allclose(jax.vmap(geometry.adf)(boundary), 0.0, atol=1e-7)
    assert np.all(jax.vmap(geometry.adf)(interior) <= 0.0)


def test_geometry_from_cad_file(tmp_path: Any) -> None:
    # Test initialization from a mesh file
    mesh = trimesh.creation.icosphere(radius=1.0)
    mesh_file = tmp_path / "sphere.stl"
    mesh.export(mesh_file)

    geom = phx.domain.GeometryDomain(
        phx.geometry.mesh_region_from_source(mesh_file).compile()
    )
    assert isinstance(geom, phx.domain.GeometryDomain)
    assert np.isclose(float(geom.volume), mesh.volume, atol=1e-6)


def test_boundary_normals_cover_faces_features_gradients_and_compilation(
    geometry_from_cube: Any,
) -> None:
    geometry = geometry_from_cube
    face_points = jnp.asarray(
        [
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, -0.5],
        ],
        dtype=jnp.float64,
    )
    expected = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    np.testing.assert_allclose(
        geometry._boundary_normals(face_points),
        expected,
        atol=1e-6,
    )

    features = jnp.asarray(
        [[0.5, 0.5, 0.0], [0.5, 0.5, 0.5]],
        dtype=jnp.float64,
    )
    feature_normals = np.asarray(
        geometry._boundary_normals(features),
        dtype=np.float64,
    )
    assert np.allclose(np.linalg.norm(feature_normals, axis=-1), 1.0)
    assert np.all(feature_normals * np.asarray(features) >= -1e-12)
    assert np.all(np.sum(feature_normals * np.asarray(features), axis=-1) > 0.0)

    gradient = jax.grad(lambda point: jnp.sum(geometry._boundary_normals(point)))(
        jnp.asarray([0.6, 0.0, 0.0], dtype=jnp.float64)
    )
    np.testing.assert_allclose(gradient, 0.0, atol=1e-10)
    compiled_points = jnp.asarray(
        [
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.6, 0.2, -0.1],
        ],
        dtype=jnp.float64,
    )
    compiled = jax.jit(lambda points: geometry._boundary_normals(points))(compiled_points)
    assert compiled.shape == compiled_points.shape
    assert np.all(np.isfinite(np.asarray(compiled)))


def test_sample_interior_separable(geometry_from_cube: Any) -> None:
    """Test separable interior sampling through the geometry domain adapter."""
    import jax.random as jr

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

    def where_condition(point: Any) -> Any:
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
