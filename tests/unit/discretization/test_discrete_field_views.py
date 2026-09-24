#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._interpolation import apply_gather_stencil, rectilinear_stencil
from phydrax.discretization import (
    DiscreteFieldFunctionView,
    FieldQueryStatus,
    prepare_point_cloud_field_reconstruction,
)
from phydrax.discretization.finite_difference import (
    BSplineGridInterpolation,
    MultilinearGridInterpolation,
    prepare_finite_difference_field_reconstruction,
)
from phydrax.discretization.finite_volume import (
    prepare_finite_volume_field_reconstruction,
)


jax.config.update("jax_enable_x64", True)

_VALID = int(FieldQueryStatus.VALID)
_OUTSIDE = int(FieldQueryStatus.OUTSIDE_SUPPORT)
_SIDE_REQUIRED = int(FieldQueryStatus.SIDE_REQUIRED)


def _view(reconstruction, coefficients):
    domain = phx.domain.GeometryDomain(reconstruction.support_geometry, label="x")
    return DiscreteFieldFunctionView(reconstruction, coefficients, domain, variable="x")


def _structured_fv(shape=(4, 2)):
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(count) for count in shape),
        axis_names=tuple("xyz"[: len(shape)]),
    ).prepare(jnp.stack((jnp.zeros(len(shape)), jnp.ones(len(shape)))))
    return phx.discretization.FiniteVolumePlan(
        grid, component_names=("density", "energy")
    ).prepare()


def _triangulated_square(resolution=4):
    vertices = np.asarray(
        [
            (i / resolution, j / resolution)
            for j in range(resolution + 1)
            for i in range(resolution + 1)
        ]
    )
    triangles = []
    for j in range(resolution):
        for i in range(resolution):
            corner = j * (resolution + 1) + i
            triangles.append((corner, corner + 1, corner + resolution + 2))
            triangles.append((corner, corner + resolution + 2, corner + resolution + 1))
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, triangles=np.asarray(triangles, dtype=np.int32)
    ).prepare()


def _cell_averages(discretization, function):
    values = function(discretization.cell_quadrature_points)
    averages = (
        jnp.sum(discretization.cell_quadrature_weights * values, axis=1)
        / discretization.cell_volumes
    )
    return averages[:, None]


def _quadratic(points):
    x, y = points[..., 0], points[..., 1]
    return 1.0 + 2.0 * x - y + 0.5 * x * y + y**2


def test_cell_average_view_returns_the_containing_cell_average():
    discretization = _structured_fv()
    reconstruction = prepare_finite_volume_field_reconstruction(
        discretization, phx.discretization.PiecewiseConstantReconstruction()
    )
    averages = jnp.arange(16.0).reshape((4, 2, 2))
    centers = discretization.cell_centers.reshape((-1, 2))

    result = reconstruction.evaluate(averages, centers)

    assert result.evidence.status.tolist() == [_VALID] * 8
    np.testing.assert_allclose(result.values, averages.reshape((-1, 2)))
    field = _view(reconstruction, averages).as_domain_function()
    np.testing.assert_allclose(field.func(jnp.asarray((0.6, 0.7))), averages[2, 1])
    outside = reconstruction.validity(jnp.asarray(((1.2, 0.5), (0.25, 0.3))))
    assert outside.status.tolist() == [_OUTSIDE, _SIDE_REQUIRED]
    assert outside.support_count.tolist() == [0, 2]

    unstructured = _triangulated_square()
    state = _cell_averages(unstructured, _quadratic)
    located = prepare_finite_volume_field_reconstruction(
        unstructured, phx.discretization.PiecewiseConstantReconstruction()
    )
    np.testing.assert_allclose(
        located.evaluate(state, unstructured.cell_centers).values, state, atol=1e-14
    )


def test_cell_average_views_refuse_coordinate_derivatives():
    discretization = _structured_fv()
    reconstruction = prepare_finite_volume_field_reconstruction(
        discretization, phx.discretization.PiecewiseConstantReconstruction()
    )
    averages = jnp.ones((4, 2, 2))

    assert reconstruction.regularity == phx.DerivativeRegularity.piecewise_polynomial(
        continuity=-1, degree_bound=0
    )
    with pytest.raises(ValueError, match="maximum_derivative_order=0"):
        reconstruction.derivative(averages, jnp.asarray(((0.1, 0.2),)), (1, 0))
    with pytest.raises(ValueError, match="degenerate"):
        phx.operators.grad(_view(reconstruction, averages).as_domain_function(), var="x")
    with pytest.raises(ValueError, match="face-trace plans"):
        prepare_finite_volume_field_reconstruction(
            discretization, phx.discretization.MUSCLReconstruction()
        )


def test_structured_face_traces_follow_positive_axis_orientation():
    discretization = _structured_fv()
    reconstruction = prepare_finite_volume_field_reconstruction(
        discretization, phx.discretization.PiecewiseConstantReconstruction()
    )
    averages = jnp.arange(16.0).reshape((4, 2, 2))
    view = _view(reconstruction, averages)
    face = jnp.asarray(((0.25, 0.3),))

    # The face x = 1/4 separates cell (0, 0) (owner, lower) and (1, 0).
    for side, expected in (
        ("owner", averages[0, 0]),
        ("neighbor", averages[1, 0]),
        ("average", 0.5 * (averages[0, 0] + averages[1, 0])),
    ):
        np.testing.assert_allclose(view.trace(face, side=side).func(face[0]), expected)
    explicit = view.trace(face, side="neighbor", cell_ids=jnp.asarray((0,)))
    np.testing.assert_allclose(explicit.func(face[0]), averages[0, 0])
    with pytest.raises(ValueError, match="require explicit cell_ids"):
        view.trace(jnp.asarray(((0.1, 0.0),)), side="neighbor")
    with pytest.raises(ValueError, match="containing each trace site"):
        view.trace(face, side="owner", cell_ids=jnp.asarray((5,)))
    with pytest.raises(ValueError, match="lie in the finite-volume grid"):
        view.trace(jnp.asarray(((1.5, 0.5),)), side="average")


def test_unstructured_face_traces_use_the_mesh_face_orientation():
    discretization = _triangulated_square(2)
    reconstruction = prepare_finite_volume_field_reconstruction(
        discretization, phx.discretization.PiecewiseConstantReconstruction()
    )
    state = jnp.arange(float(discretization.cell_count))[:, None]
    view = _view(reconstruction, state)
    owner = np.asarray(discretization.owner_cells)
    neighbor = np.asarray(discretization.neighbor_cells)
    face = int(np.flatnonzero(neighbor >= 0)[0])
    site = discretization.face_centers[face][None, :]

    assert reconstruction.validity(site).status.tolist() == [_SIDE_REQUIRED]
    np.testing.assert_allclose(
        view.trace(site, side="owner").func(site[0]), state[owner[face]]
    )
    np.testing.assert_allclose(
        view.trace(site, side="neighbor").func(site[0]), state[neighbor[face]]
    )
    np.testing.assert_allclose(
        view.trace(site, side="average").func(site[0]),
        0.5 * (state[owner[face]] + state[neighbor[face]]),
    )


def test_cell_polynomial_view_is_k_exact_with_exact_derivatives_and_transpose():
    discretization = _triangulated_square()
    polynomial = phx.discretization.CellPolynomialReconstructionPlan(2).prepare(
        discretization
    )
    reconstruction = prepare_finite_volume_field_reconstruction(
        discretization, polynomial
    )
    state = _cell_averages(discretization, _quadratic)
    points = jnp.asarray(((0.3, 0.1), (0.62, 0.41), (0.13, 0.77)))
    y = points[:, 1]

    np.testing.assert_allclose(
        reconstruction.evaluate(state, points).values[:, 0],
        _quadratic(points),
        atol=1e-11,
    )
    np.testing.assert_allclose(
        reconstruction.derivative(state, points, (1, 0)).values[:, 0],
        2.0 + 0.5 * y,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        reconstruction.derivative(state, points, (0, 2)).values[:, 0], 2.0, atol=1e-9
    )
    with pytest.raises(ValueError, match="maximum_derivative_order=2"):
        reconstruction.derivative(state, points, (2, 1))
    # The view evaluates the owner's reconstruction in the containing cell.
    centers = discretization.cell_centers
    routes = jnp.arange(discretization.cell_count)
    np.testing.assert_allclose(
        reconstruction.evaluate(state, centers).values,
        polynomial.evaluate(state, routes, centers[:, None, :])[:, 0],
        atol=1e-12,
    )
    cotangent = jnp.asarray(((1.0,), (-2.0,), (0.5,)))
    assert bool(reconstruction.duality_evidence(state, points, cotangent).valid)
    _, pullback = jax.vjp(lambda c: reconstruction.evaluate(c, points).values, state)
    np.testing.assert_allclose(
        pullback(cotangent)[0], reconstruction.transpose(points, cotangent), atol=1e-12
    )


def test_weno_view_is_nonlinear_and_refuses_an_algebraic_transpose():
    discretization = _triangulated_square()
    weno = phx.discretization.UnstructuredWENOZReconstructionPlan(
        2, limiter="none"
    ).prepare(discretization)
    reconstruction = prepare_finite_volume_field_reconstruction(discretization, weno)
    state = _cell_averages(discretization, _quadratic)
    centers = discretization.cell_centers

    assert not reconstruction.coefficient_linear
    assert reconstruction.regularity.continuity == -1
    assert reconstruction.regularity.conditions
    routes = jnp.arange(discretization.cell_count)
    owner = weno.optimal.evaluate_coefficients(
        state, weno.coefficients(state), routes, centers[:, None, :]
    )
    np.testing.assert_allclose(
        reconstruction.evaluate(state, centers).values, owner[:, 0], atol=1e-12
    )
    with pytest.raises(ValueError, match="nonlinear"):
        reconstruction.duality_evidence(state, centers, jnp.ones_like(state))
    gradient = jax.grad(
        lambda c: jnp.sum(reconstruction.evaluate(c, centers[:3]).values)
    )(state)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    limited = phx.discretization.UnstructuredWENOZReconstructionPlan(2).prepare(
        discretization
    )
    with pytest.raises(ValueError, match="limiter='none'"):
        prepare_finite_volume_field_reconstruction(discretization, limited)


def test_structured_views_bind_only_to_an_equivalent_box():
    discretization = _structured_fv()
    square = phx.geometry.Rectangle((0.5, 0.5), (1.0, 1.0)).compile()
    reconstruction = prepare_finite_volume_field_reconstruction(
        discretization,
        phx.discretization.PiecewiseConstantReconstruction(),
        support_geometry=square,
    )
    averages = jnp.arange(16.0).reshape((4, 2, 2))
    view = DiscreteFieldFunctionView(
        reconstruction,
        averages,
        phx.domain.GeometryDomain(square, label="x"),
        variable="x",
    )

    np.testing.assert_allclose(
        view.as_domain_function().func(jnp.asarray((0.9, 0.9))), averages[3, 1]
    )
    with pytest.raises(ValueError, match="bounds differ"):
        prepare_finite_volume_field_reconstruction(
            discretization,
            phx.discretization.PiecewiseConstantReconstruction(),
            support_geometry=phx.geometry.Rectangle((0.5, 0.5), (1.0, 2.0)).compile(),
        )


def _finite_difference(shape=(9, 7), periodic=(False, False)):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(count, periodic=flag, endpoint=not flag)
            for count, flag in zip(shape, periodic, strict=True)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 2.0))))
    request = phx.discretization.DerivativeRequest(
        "dx",
        grid,
        "x",
        derivative_order=1,
        accuracy_order=2,
        boundary="periodic" if periodic[0] else "one_sided",
    )
    return phx.discretization.FiniteDifferencePlan(
        grid, (request,), field_name="u"
    ).prepare()


def _nodal(discretization, function):
    x, y = discretization.grid.primary_entity_layout.coordinates_by_axis
    return function(x[:, None], y[None, :])


_FD_POINTS = jnp.asarray(((0.13, 0.37), (0.5, 1.9), (0.99, 0.01)))


def test_multilinear_grid_view_matches_the_native_rectilinear_interpolant():
    discretization = _finite_difference()
    reconstruction = prepare_finite_difference_field_reconstruction(
        discretization, interpolation=MultilinearGridInterpolation()
    )
    values = _nodal(discretization, lambda x, y: jnp.sin(3.0 * x) * jnp.cos(y))
    native = apply_gather_stencil(
        values.reshape((-1,)),
        rectilinear_stencil(
            discretization.grid.primary_entity_layout.coordinates_by_axis,
            _FD_POINTS,
            boundary=("constant", "constant"),
        ),
    )

    np.testing.assert_allclose(
        reconstruction.evaluate(values, _FD_POINTS).values, native.values, atol=1e-14
    )
    bilinear = _nodal(discretization, lambda x, y: 1.0 + x - 2.0 * y + 3.0 * x * y)
    x, y = _FD_POINTS[:, 0], _FD_POINTS[:, 1]
    np.testing.assert_allclose(
        reconstruction.evaluate(bilinear, _FD_POINTS).values,
        1.0 + x - 2.0 * y + 3.0 * x * y,
        atol=1e-13,
    )
    assert reconstruction.maximum_derivative_order == 0
    outside = reconstruction.validity(jnp.asarray(((1.2, 0.3), (jnp.nan, 0.2))))
    assert outside.status.tolist() == [_OUTSIDE, int(FieldQueryStatus.NONFINITE)]
    assert bool(
        reconstruction.duality_evidence(values, _FD_POINTS, jnp.arange(3.0)).valid
    )


def test_bspline_grid_view_interpolates_and_reproduces_tensor_polynomials():
    discretization = _finite_difference()
    reconstruction = prepare_finite_difference_field_reconstruction(
        discretization, interpolation=BSplineGridInterpolation(3)
    )

    def cubic(x, y):
        return 1.0 + x - 2.0 * y + 3.0 * x * y + x**2 * y - 0.5 * y**3 + x**3

    values = _nodal(discretization, cubic)
    x, y = _FD_POINTS[:, 0], _FD_POINTS[:, 1]

    assert reconstruction.regularity == phx.DerivativeRegularity.piecewise_polynomial(
        continuity=2, degree_bound=6
    )
    np.testing.assert_allclose(
        reconstruction.evaluate(values, _FD_POINTS).values, cubic(x, y), atol=1e-12
    )
    field = _view(reconstruction, values).as_domain_function()
    np.testing.assert_allclose(
        jax.vmap(phx.operators.grad(field, var="x").func)(_FD_POINTS),
        jnp.stack(
            (
                1.0 + 3.0 * y + 2.0 * x * y + 3.0 * x**2,
                -2.0 + 3.0 * x + x**2 - 1.5 * y**2,
            ),
            axis=-1,
        ),
        atol=1e-11,
    )
    np.testing.assert_allclose(
        reconstruction.derivative(values, _FD_POINTS, (1, 1)).values,
        3.0 + 2.0 * x,
        atol=1e-10,
    )
    with pytest.raises(ValueError, match="maximum_derivative_order=2"):
        phx.operators.partial_n(field, var="x", axis=0, order=3)
    # Nodal values of arbitrary data are interpolated exactly.
    rough = _nodal(discretization, lambda x, y: jnp.sign(jnp.sin(7.0 * x + 3.0 * y)))
    nodes = discretization.grid.points
    np.testing.assert_allclose(
        reconstruction.evaluate(rough, nodes).values, rough.reshape((-1,)), atol=1e-12
    )
    cotangent = jnp.asarray((1.0, -2.0, 0.5))
    assert bool(reconstruction.duality_evidence(rough, _FD_POINTS, cotangent).valid)
    _, pullback = jax.vjp(lambda c: reconstruction.evaluate(c, _FD_POINTS).values, rough)
    np.testing.assert_allclose(
        pullback(cotangent)[0],
        reconstruction.transpose(_FD_POINTS, cotangent),
        atol=1e-12,
    )
    assert reconstruction.validity(jnp.asarray(((0.5, 2.1),))).status.tolist() == [
        _OUTSIDE
    ]
    with pytest.raises(ValueError, match="bounded axes"):
        prepare_finite_difference_field_reconstruction(
            _finite_difference(periodic=(True, False)),
            interpolation=BSplineGridInterpolation(3),
        )


def test_periodic_multilinear_view_wraps_inside_the_periodic_cell_only():
    discretization = _finite_difference((8, 5), periodic=(True, False))
    reconstruction = prepare_finite_difference_field_reconstruction(
        discretization, interpolation=MultilinearGridInterpolation()
    )
    values = _nodal(discretization, lambda x, y: jnp.cos(2.0 * jnp.pi * x) + 0.0 * y)

    np.testing.assert_allclose(
        np.asarray(reconstruction.support_geometry.bounds), ((0.0, 0.0), (1.0, 2.0))
    )
    # Between the last node x = 7/8 and the periodic image of x = 0.
    expected = np.cos(2.0 * np.pi * 7.0 / 8.0) * 0.4 + 1.0 * 0.6
    result = reconstruction.evaluate(values, jnp.asarray(((0.95, 1.0), (1.01, 1.0))))
    np.testing.assert_allclose(result.values[0], expected, atol=1e-14)
    assert result.evidence.status.tolist() == [_VALID, _OUTSIDE]


def _point_cloud(points):
    weights = np.full(points.shape[0], 1.0 / points.shape[0])
    return phx.discretization.PointCloudPlan(points, weights, degree=2).prepare()


_UNIT_SQUARE = phx.geometry.Rectangle((0.5, 0.5), (1.0, 1.0))


def test_point_cloud_view_reports_conditioning_and_support_evidence():
    random = np.random.default_rng(0)
    scattered = random.uniform(0.0, 1.0, (200, 2))
    # Only collinear points lie within the radius of (0.17, 0.1); the cloud's own
    # nearest-neighbor stencils still reach the scattered points.
    scattered = scattered[(scattered[:, 0] > 0.6) | (scattered[:, 1] > 0.6)]
    line = np.stack((np.linspace(0.05, 0.29, 7), np.full(7, 0.1)), axis=-1)
    points = np.concatenate((scattered, line))
    discretization = _point_cloud(points)
    square = _UNIT_SQUARE.compile()
    reconstruction = prepare_point_cloud_field_reconstruction(
        discretization,
        support_geometry=square,
        radius=0.15,
        capacity=points.shape[0],
    )
    values = jnp.asarray(_quadratic(points))
    queries = jnp.asarray(((0.7, 0.7), (0.55, 0.85), (0.17, 0.1), (0.5, 1.7)))

    result = DiscreteFieldFunctionView(
        reconstruction, values, phx.domain.GeometryDomain(square, label="x"), variable="x"
    ).query(queries)
    status = result.evidence.status.tolist()

    assert status == [
        _VALID,
        _VALID,
        int(FieldQueryStatus.ILL_CONDITIONED),
        _OUTSIDE,
    ]
    conditioning = np.asarray(result.evidence.conditioning)
    assert np.all(np.isfinite(conditioning[:2])) and np.all(conditioning[:2] >= 1.0)
    assert not np.isfinite(conditioning[2])
    assert result.evidence.support_count.tolist()[3] == 0
    np.testing.assert_allclose(result.values[:2], _quadratic(queries[:2]), atol=1e-11)
    np.testing.assert_allclose(result.values[2:], 0.0)
    cotangent = jnp.asarray((1.0, -2.0, 0.5, 3.0))
    assert bool(reconstruction.duality_evidence(values, queries, cotangent).valid)

    bounded = prepare_point_cloud_field_reconstruction(
        discretization, support_geometry=square, radius=0.15, capacity=8
    )
    assert bounded.validity(queries[:1]).status.tolist() == [
        int(FieldQueryStatus.LOCATION_FAILED)
    ]


def test_point_cloud_views_refuse_domain_functions_without_coverage():
    points = np.random.default_rng(1).uniform(0.0, 1.0, (40, 2))
    discretization = _point_cloud(points)
    square = _UNIT_SQUARE.compile()
    reconstruction = prepare_point_cloud_field_reconstruction(
        discretization, support_geometry=square, radius=0.4, capacity=40
    )
    view = DiscreteFieldFunctionView(
        reconstruction,
        jnp.asarray(_quadratic(points)),
        phx.domain.GeometryDomain(square, label="x"),
        variable="x",
    )

    with pytest.raises(ValueError, match="every point"):
        view.as_domain_function()
    with pytest.raises(ValueError, match="maximum_derivative_order=0"):
        view.query(jnp.asarray(((0.5, 0.5),)), derivative=(1, 0))
    with pytest.raises(ValueError, match="outside the support geometry"):
        prepare_point_cloud_field_reconstruction(
            discretization,
            support_geometry=phx.geometry.Rectangle((0.5, 0.5), (0.5, 0.5)).compile(),
            radius=0.4,
            capacity=40,
        )
