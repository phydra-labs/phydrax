#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import build123d as bd
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _tetrahedron():
    vertices = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = jnp.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    return vertices, faces


def test_halfedge_topology_and_exact_bvh_query():
    vertices, faces = _tetrahedron()
    mesh = phx.geometry.TriangleMesh(vertices, faces, source_id="tetrahedron")
    assert mesh.topology.watertight
    assert mesh.topology.euler_characteristic == 2
    assert mesh.topology.num_halfedges == 12
    assert mesh.topology.num_boundary_loops == 0

    points = jnp.asarray([[0.2, 0.2, 0.2], [2.0, 0.0, 0.0]])
    index = mesh.query_index()
    bvh_result = jax.jit(lambda value: index.query(value))(points)
    assert np.allclose(np.asarray(bvh_result.distance), [0.2, 1.0])


def test_topology_rejects_nonmanifold_and_inconsistent_orientation():
    with pytest.raises(ValueError, match="inconsistent orientation"):
        phx.geometry.TriangleTopology([[0, 1, 2], [0, 1, 3]])
    with pytest.raises(ValueError, match="non-manifold"):
        phx.geometry.TriangleTopology([[0, 1, 2], [1, 0, 3], [0, 1, 4]])


def test_mesh_and_planar_regions_have_explicit_signed_queries():
    vertices, faces = _tetrahedron()
    region = phx.geometry.MeshRegion(vertices, faces, feature_id="tet").compile()
    query = jax.jit(lambda value: region.boundary_field(value))(
        jnp.asarray([[0.1, 0.1, 0.1], [1.0, 1.0, 1.0]])
    )
    assert query[0] < 0.0
    assert query[1] > 0.0
    assert float(region.measure) == pytest.approx(1.0 / 6.0)

    planar = phx.geometry.PlanarMeshRegion(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [0.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
            [1.5, 0.5],
        ],
        ((0, 1, 2, 3), (4, 5, 6, 7)),
        feature_id="with-hole",
    ).compile()
    assert float(planar.measure) == pytest.approx(3.0)
    assert np.array_equal(
        np.asarray(planar.contains(jnp.asarray([[0.25, 0.25], [1.0, 1.0]]))),
        [True, False],
    )


def test_boundary_atlas_frames_selection_and_sampling_metadata():
    box = phx.geometry.Box((0.0, 0.0, 0.0), (2.0, 3.0, 4.0)).compile()
    selected = box.boundary_atlas.select(tags=("x_min",))
    assert selected.num_charts == 1
    frame = jax.jit(lambda index, ref: selected.frame(index, ref))(
        jnp.asarray([0]), jnp.asarray([[0.5, 0.5]])
    )
    assert np.allclose(np.asarray(frame.normal), [[-1.0, 0.0, 0.0]])
    samples = phx.geometry.sample_boundary_atlas(selected, 32, key=jax.random.key(4))
    assert bool(samples.report.complete)
    assert float(jnp.sum(samples.weights)) == pytest.approx(1.0)
    assert np.all(np.asarray(samples.strata) == 0)
    assert np.allclose(np.asarray(samples.points[:, 0]), -1.0)


def test_matrix_free_ddg_linear_precision():
    mesh = phx.geometry.TriangleMesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        [[0, 1, 2], [0, 2, 3]],
    )
    operators = phx.geometry.discrete_operators(mesh)
    values = mesh.vertices[:, 0] + 2.0 * mesh.vertices[:, 1]
    gradient = operators.gradient(values)
    assert np.allclose(np.asarray(gradient), [[1.0, 2.0, 0.0]] * 2)
    assert np.all(np.asarray(operators.vertex_mass) > 0.0)


def test_occt_brep_import_preserves_topology_patches_and_boundary_identity():
    model = phx.geometry.model_from_occt_shape(
        bd.Box(1.0, 2.0, 3.0).wrapped,
        linear_deflection=0.1,
        angular_deflection=0.3,
    )
    assert model.topology.num_faces == 6
    assert model.topology.num_edges == 12
    assert len(model.patches) == model.topology.num_faces
    assert model.triangle_face_ids.shape == model.mesh_faces.shape[:1]

    geometry = phx.geometry.BRepSource(model).compile()
    assert float(geometry.measure) == pytest.approx(6.0)
    assert geometry.boundary_atlas.num_charts == model.topology.num_faces
    selected = geometry.boundary_atlas.select(entity_ids=(0,))
    assert selected.source_id == model.source_id
    assert np.array_equal(np.asarray(selected.source_entity_ids), [0])
    face_point = jnp.asarray([0.5, 0.0, 0.0])
    direct_gradient = jax.grad(geometry.boundary_field)(face_point)
    assert np.all(np.isfinite(np.asarray(direct_gradient)))
    assert float(jnp.linalg.norm(direct_gradient)) == pytest.approx(1.0)
    direct_normal_jacobian = jax.jacrev(geometry.boundary_normal)(face_point)
    assert np.all(np.isfinite(np.asarray(direct_normal_jacobian)))
    assert np.allclose(np.asarray(direct_normal_jacobian), 0.0)

    differentiable = phx.geometry.FixedTopologyBRepSource(model).compile()
    differentiable_gradient = jax.grad(differentiable.boundary_field)(face_point)
    assert np.all(np.isfinite(np.asarray(differentiable_gradient)))
    assert float(jnp.linalg.norm(differentiable_gradient)) == pytest.approx(1.0)
    assert np.array_equal(
        np.asarray(geometry.contains(jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))),
        [True, False],
    )


def test_rational_bspline_surface_is_jax_differentiable():
    control_points = jnp.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ]
    )
    knots = jnp.asarray([0.0, 0.0, 1.0, 1.0])

    def height(corner_height):
        points = control_points.at[1, 1, 2].set(corner_height)
        patch = phx.geometry.BSplineSurfacePatch(
            points,
            jnp.ones((2, 2)),
            knots,
            knots,
            1,
            1,
        )
        return patch.evaluate(jnp.asarray([0.5, 0.5]))[2]

    patch = phx.geometry.BSplineSurfacePatch(
        control_points,
        jnp.ones((2, 2)),
        knots,
        knots,
        1,
        1,
    )

    assert np.allclose(
        np.asarray(patch.evaluate(jnp.asarray([0.5, 0.5]))),
        [0.5, 0.5, 0.25],
    )
    assert float(jax.grad(height)(jnp.asarray(1.0))) == pytest.approx(0.25)
    assert float(
        phx.geometry.surface_jacobian(patch, jnp.asarray([0.5, 0.5]))
    ) == pytest.approx(np.sqrt(1.5))


def test_level_set_domain_ansatz_factor_has_unit_boundary_jet():
    domain = phx.domain.GeometryDomain(
        phx.geometry.Ellipse((0.0, 0.0), (2.0, 1.0)).compile()
    )
    factor = domain.boundary_ansatz_factor
    boundary_point = jnp.asarray([2.0, 0.0])
    assert float(factor(boundary_point)) == pytest.approx(0.0, abs=1e-12)
    gradient = jax.grad(factor)(boundary_point)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.linalg.norm(gradient)) == pytest.approx(1.0)


def test_fixed_topology_brep_preserves_connectivity_and_shape_gradients():
    model = phx.geometry.model_from_occt_shape(
        bd.Sphere(1.0).wrapped,
        linear_deflection=0.15,
        angular_deflection=0.3,
    )
    geometry = phx.geometry.FixedTopologyBRepSource(model).compile()
    radius_index = next(
        index
        for index, spec in enumerate(geometry.schema.specs)
        if spec.parameter_id.name == "radius"
    )

    def volume(radius):
        state = geometry.state.replace_at(radius_index, radius)
        return geometry.kernel.measure(state)

    initial = volume(jnp.asarray(1.0))
    derivative = jax.grad(volume)(jnp.asarray(1.0))
    realization = geometry.kernel.realize(geometry.state)
    assert np.array_equal(np.asarray(realization.faces), np.asarray(model.mesh_faces))
    assert float(realization.seam_residual) < 1e-10
    assert float(derivative) == pytest.approx(3.0 * float(initial), rel=1e-10)
    assert float(volume(jnp.asarray(1.2))) > float(initial)


def test_sketch_and_design_constraint_solvers_lower_to_geometry():
    sketch = phx.geometry.Sketch(
        jnp.asarray([[0.0, 0.0], [1.2, 0.1], [1.1, 1.0], [0.0, 0.9]]),
        lines=jnp.asarray([[0, 1], [1, 2], [2, 3], [3, 0]]),
        constraints=(
            phx.geometry.FixedPoint(0, (0.0, 0.0)),
            phx.geometry.Horizontal(0),
            phx.geometry.Vertical(1),
            phx.geometry.Horizontal(2),
            phx.geometry.Vertical(3),
            phx.geometry.PointDistance(0, 1, 1.0),
            phx.geometry.PointDistance(1, 2, 1.0),
        ),
    )
    sketch_solution = sketch.solve()
    assert bool(sketch_solution.converged)
    assert float(sketch_solution.residual_norm) < 1e-8
    assert float(sketch.to_source(sketch_solution).compile().measure) == pytest.approx(
        1.0, abs=1e-8
    )

    sphere = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0).compile()
    target_radius = 1.5
    target_volume = 4.0 * np.pi * target_radius**3 / 3.0
    system = phx.geometry.DesignConstraintSystem(
        sphere,
        (phx.geometry.MeasureTarget(target_volume),),
    )
    design_solution = system.solve()
    radius_id = next(
        parameter_id
        for parameter_id in sphere.schema.parameter_ids
        if parameter_id.name == "radius"
    )
    assert bool(design_solution.converged)
    assert float(design_solution.state.values[sphere.schema.index(radius_id)]) == (
        pytest.approx(target_radius)
    )


def test_affine_geometry_preserves_boundary_tags_and_exact_atlas_measure():
    transformed = (
        phx.geometry.Box((0.0, 0.0, 0.0), (2.0, 3.0, 4.0))
        .rotated((0.0, 0.0, 1.0), 0.3)
        .scaled(2.0)
        .translated((1.0, -2.0, 0.5))
        .compile()
    )
    atlas = transformed.boundary_atlas
    assert atlas.physical_tags == (
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "z_min",
        "z_max",
    )
    partition = phx.geometry.BoundaryAtlasPartition(atlas)
    assert float(partition.total_measure) == pytest.approx(
        float(transformed.boundary_measure), rel=1e-10
    )
