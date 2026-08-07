#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import build123d as bd
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.gp import gp_Pnt, gp_Vec
from scipy.interpolate import BSpline as SciPyBSpline

import phydrax as phx
from phydrax.geometry.brep import BRepBoundaryMap


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


def test_rational_bspline_surface_preserves_endpoint_differentials():
    control_points = jnp.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )
    knots = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    patch = phx.geometry.BSplineSurfacePatch(
        control_points,
        jnp.ones((2, 2)),
        knots,
        knots,
        1,
        1,
    )
    parameters = jnp.asarray([[0.5, 0.5], [1.0 - 1e-6, 0.5], [1.0, 0.5]])

    values = patch.evaluate(parameters)
    differential = phx.geometry.surface_differential(patch, parameters)
    jacobian = phx.geometry.surface_jacobian(patch, parameters)
    normal = phx.geometry.surface_normal(patch, parameters)
    expected_differential = jnp.broadcast_to(
        jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        differential.shape,
    )

    expected_values = jnp.concatenate(
        (parameters, jnp.zeros(parameters.shape[:-1] + (1,))),
        axis=-1,
    )
    assert np.allclose(np.asarray(values), np.asarray(expected_values))
    assert np.allclose(np.asarray(differential), np.asarray(expected_differential))
    assert np.allclose(np.asarray(jacobian), 1.0)
    assert np.allclose(np.asarray(normal), [0.0, 0.0, 1.0])

    boundary_map = BRepBoundaryMap(
        (patch,),
        jnp.asarray([[[0.0, 0.0], [1.0, 1.0]]]),
    )
    assert np.allclose(
        np.asarray(
            boundary_map.jacobian(
                jnp.asarray([0]),
                jnp.asarray([[1.0, 0.5]]),
            )
        ),
        1.0,
    )


def test_rational_bspline_surface_matches_tensor_product_oracle():
    u_degree = 2
    v_degree = 3
    u_knots = np.asarray([0.0, 0.0, 0.0, 0.3, 0.5, 0.5, 1.0, 1.0, 1.0])
    v_knots = np.asarray([0.0, 0.0, 0.0, 0.0, 0.35, 0.8, 1.0, 1.0, 1.0, 1.0])
    control_points = np.asarray(
        [
            [
                [
                    float(u_index) / 5.0,
                    float(v_index) / 5.0,
                    0.04 * u_index**2 + 0.03 * u_index * v_index,
                ]
                for v_index in range(6)
            ]
            for u_index in range(6)
        ]
    )
    weights = np.asarray(
        [
            [1.0 + 0.05 * u_index + 0.03 * v_index for v_index in range(6)]
            for u_index in range(6)
        ]
    )
    parameters = np.asarray(
        [
            [0.0, 0.0],
            [0.3, 0.35],
            [0.5, 0.8],
            [0.72, 1.0],
            [1.0, 0.55],
            [1.0, 1.0],
        ]
    )

    u_oracle = SciPyBSpline(u_knots, np.eye(6), u_degree)
    v_oracle = SciPyBSpline(v_knots, np.eye(6), v_degree)
    u_basis = u_oracle(parameters[:, 0])
    v_basis = v_oracle(parameters[:, 1])
    du_basis = u_oracle(parameters[:, 0], nu=1)
    dv_basis = v_oracle(parameters[:, 1], nu=1)
    coefficients = np.einsum("qi,qj,ij->qij", u_basis, v_basis, weights)
    u_coefficients = np.einsum("qi,qj,ij->qij", du_basis, v_basis, weights)
    v_coefficients = np.einsum("qi,qj,ij->qij", u_basis, dv_basis, weights)
    denominator = np.sum(coefficients, axis=(1, 2))
    numerator = np.einsum("qij,ijc->qc", coefficients, control_points)
    expected_values = numerator / denominator[:, None]

    u_numerator = np.einsum("qij,ijc->qc", u_coefficients, control_points)
    v_numerator = np.einsum("qij,ijc->qc", v_coefficients, control_points)
    u_denominator = np.sum(u_coefficients, axis=(1, 2))
    v_denominator = np.sum(v_coefficients, axis=(1, 2))
    expected_u = (
        u_numerator * denominator[:, None] - numerator * u_denominator[:, None]
    ) / denominator[:, None] ** 2
    expected_v = (
        v_numerator * denominator[:, None] - numerator * v_denominator[:, None]
    ) / denominator[:, None] ** 2
    expected_differential = np.stack((expected_u, expected_v), axis=-1)

    patch = phx.geometry.BSplineSurfacePatch(
        control_points,
        weights,
        u_knots,
        v_knots,
        u_degree,
        v_degree,
    )
    actual_values = jax.jit(lambda value: patch.evaluate(value))(jnp.asarray(parameters))
    actual_differential = phx.geometry.surface_differential(
        patch,
        jnp.asarray(parameters),
    )
    assert np.allclose(
        np.asarray(actual_values),
        expected_values,
        rtol=1e-9,
        atol=1e-9,
    )
    assert np.allclose(
        np.asarray(actual_differential),
        expected_differential,
        rtol=1e-9,
        atol=1e-9,
    )

    def height(center_weight):
        dynamic_weights = jnp.asarray(weights).at[2, 3].set(center_weight)
        dynamic_patch = phx.geometry.BSplineSurfacePatch(
            control_points,
            dynamic_weights,
            u_knots,
            v_knots,
            u_degree,
            v_degree,
        )
        return dynamic_patch.evaluate(jnp.asarray([0.43, 0.57]))[2]

    center_weight = weights[2, 3]
    epsilon = 1e-5
    finite_difference = (
        float(height(center_weight + epsilon)) - float(height(center_weight - epsilon))
    ) / (2.0 * epsilon)
    assert float(jax.grad(height)(jnp.asarray(center_weight))) == pytest.approx(
        finite_difference,
        rel=1e-7,
        abs=1e-9,
    )


def test_rational_bspline_curve_matches_scipy_and_endpoint_derivative():
    degree = 3
    knots = np.asarray([0.0, 0.0, 0.0, 0.0, 0.4, 0.7, 1.0, 1.0, 1.0, 1.0])
    control_points = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.5],
            [0.4, -0.1],
            [0.7, 0.8],
            [0.9, 0.3],
            [1.0, 1.0],
        ]
    )
    weights = np.asarray([1.0, 1.2, 0.8, 1.1, 0.9, 1.3])
    query = np.asarray([0.0, 0.23, 0.7, 1.0])
    basis_oracle = SciPyBSpline(knots, np.eye(6), degree)
    basis = basis_oracle(query)
    derivative_basis = basis_oracle(query, nu=1)
    coefficients = basis * weights
    derivative_coefficients = derivative_basis * weights
    denominator = np.sum(coefficients, axis=-1)
    derivative_denominator = np.sum(derivative_coefficients, axis=-1)
    numerator = coefficients @ control_points
    derivative_numerator = derivative_coefficients @ control_points
    expected_values = numerator / denominator[:, None]
    expected_derivative = (
        derivative_numerator * denominator[:, None]
        - numerator * derivative_denominator[:, None]
    ) / denominator[:, None] ** 2

    curve = phx.geometry.BSplineCurve(control_points, weights, knots, degree)
    actual_values = curve.evaluate(jnp.asarray(query))
    actual_derivative = jax.vmap(jax.jacfwd(lambda parameter: curve.evaluate(parameter)))(
        jnp.asarray(query)
    )
    assert np.allclose(np.asarray(actual_values), expected_values, atol=1e-10)
    assert np.allclose(
        np.asarray(actual_derivative),
        expected_derivative,
        rtol=1e-9,
        atol=1e-9,
    )

    singular = phx.geometry.BSplineCurve(
        [[0.0, 0.0], [1.0, 0.0]],
        [1.0, -1.0],
        [0.0, 0.0, 1.0, 1.0],
        1,
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="denominator"):
        jax.block_until_ready(singular.evaluate(jnp.asarray(0.5)))


def test_occt_bspline_import_matches_native_surface_differential():
    coordinates = np.linspace(0.0, 1.0, 5)
    face = bd.Face.make_surface_from_array_of_points(
        [
            [
                (
                    float(u),
                    float(v),
                    0.15 * float(u) * float(v)
                    + 0.05 * float(u) ** 2
                    - 0.03 * float(v) ** 2,
                )
                for u in coordinates
            ]
            for v in coordinates
        ],
        tol=1e-6,
        min_deg=2,
        max_deg=3,
    )
    model = phx.geometry.model_from_occt_shape(
        face.wrapped,
        linear_deflection=0.05,
        angular_deflection=0.2,
    )
    assert len(model.patches) == 1
    assert isinstance(model.patches[0], phx.geometry.BSplineSurfacePatch)
    assert model.report.converted_surface_count == 0

    normalized = np.asarray([[0.0, 0.0], [0.2, 0.3], [0.5, 0.5], [1.0, 0.6], [0.4, 1.0]])
    bounds = np.asarray(model.parameter_bounds[0])
    parameters = bounds[0] + normalized * (bounds[1] - bounds[0])
    native_surface = BRepAdaptor_Surface(face.wrapped, True).BSpline()
    expected_values = []
    expected_differentials = []
    for u, v in parameters:
        point = gp_Pnt()
        u_tangent = gp_Vec()
        v_tangent = gp_Vec()
        native_surface.D1(
            float(u),
            float(v),
            point,
            u_tangent,
            v_tangent,
        )
        expected_values.append([point.X(), point.Y(), point.Z()])
        expected_differentials.append(
            [
                [u_tangent.X(), v_tangent.X()],
                [u_tangent.Y(), v_tangent.Y()],
                [u_tangent.Z(), v_tangent.Z()],
            ]
        )

    patch = model.patches[0]
    actual_values = patch.evaluate(jnp.asarray(parameters))
    actual_differentials = phx.geometry.surface_differential(
        patch,
        jnp.asarray(parameters),
    )
    assert np.allclose(np.asarray(actual_values), expected_values, atol=1e-10)
    assert np.allclose(
        np.asarray(actual_differentials),
        expected_differentials,
        rtol=1e-9,
        atol=1e-9,
    )

    frame = model.boundary_atlas.frame(
        jnp.zeros((normalized.shape[0],), dtype=jnp.int32),
        jnp.asarray(normalized),
    )
    assert np.all(np.isfinite(np.asarray(frame.origin)))
    assert np.all(np.isfinite(np.asarray(frame.tangents)))
    assert np.all(np.isfinite(np.asarray(frame.normal)))
    assert np.all(np.asarray(frame.jacobian) > 0.0)


def test_fixed_topology_bspline_loft_preserves_mixed_patch_dispatch():
    wires = [
        bd.Wire.make_circle(
            1.0,
            plane=bd.Plane(origin=(0.0, 0.0, 0.0)),
        ),
        bd.Wire.make_circle(
            1.25,
            plane=bd.Plane(origin=(0.15, -0.05, 0.8)),
        ),
        bd.Wire.make_circle(
            0.9,
            plane=bd.Plane(origin=(-0.1, 0.1, 1.7)),
        ),
    ]
    model = phx.geometry.model_from_occt_shape(
        bd.Solid.make_loft(wires).wrapped,
        linear_deflection=0.1,
        angular_deflection=0.25,
    )
    assert tuple(type(patch).__name__ for patch in model.patches) == (
        "BSplineSurfacePatch",
        "PlanePatch",
        "PlanePatch",
    )

    geometry = phx.geometry.BRepSource(model).compile()
    assert float(geometry.measure) > 0.0
    references = jnp.asarray([[0.0, 0.0], [0.5, 0.5], [1.0, 0.4], [0.3, 1.0]])
    indices = jnp.zeros((references.shape[0],), dtype=jnp.int32)
    frame = geometry.boundary_atlas.frame(indices, references)
    assert np.all(np.isfinite(np.asarray(frame.normal)))
    assert np.all(np.asarray(frame.jacobian) > 0.0)

    differentiable = phx.geometry.FixedTopologyBRepSource(model).compile()
    control_index = next(
        index
        for index, spec in enumerate(differentiable.schema.specs)
        if spec.parameter_id.name == "control_points"
    )
    weight_index = next(
        index
        for index, spec in enumerate(differentiable.schema.specs)
        if spec.parameter_id.name == "weights"
    )
    control_points = differentiable.state.values[control_index]
    weights = differentiable.state.values[weight_index]
    state = differentiable.state.replace_at(
        control_index,
        control_points.at[0, 0, 2].add(1e-3),
    ).replace_at(
        weight_index,
        weights.at[0, 0].multiply(1.01),
    )
    realization = differentiable.kernel.realize(state)
    realized_frame = realization.atlas.frame(indices, references)
    assert np.array_equal(np.asarray(realization.faces), np.asarray(model.mesh_faces))
    assert np.all(np.isfinite(np.asarray(realization.vertices)))
    assert np.all(np.isfinite(np.asarray(realized_frame.normal)))
    assert np.all(np.asarray(realized_frame.jacobian) > 0.0)
    assert np.isfinite(float(realization.seam_residual))


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
    radius_id = next(
        parameter_id
        for parameter_id in sphere.schema.parameter_ids
        if parameter_id.name == "radius"
    )
    center_id = next(
        parameter_id
        for parameter_id in sphere.schema.parameter_ids
        if parameter_id.name == "center"
    )
    global_solution = system.search(
        phx.geometry.DifferentialEvolutionSearch(
            8,
            4,
            relative_tolerance=0.0,
            absolute_tolerance=0.0,
        ),
        key=jax.random.key(12),
        bounds={
            center_id: (-0.25, 0.25),
            radius_id: (0.25, 2.5),
        },
    )
    design_solution = system.solve(initial_state=global_solution.state)
    assert bool(design_solution.converged)
    assert float(design_solution.state.values[sphere.schema.index(radius_id)]) == (
        pytest.approx(target_radius)
    )
    optimized = sphere.with_state(design_solution.state)
    domain = phx.domain.GeometryDomain(optimized)
    assert float(domain.measure) == pytest.approx(target_volume)


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
