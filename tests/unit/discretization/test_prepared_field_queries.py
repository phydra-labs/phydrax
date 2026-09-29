#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared field queries: fixed routes reused across coefficient refreshes."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import (
    AbstractFieldReconstructionKernel,
    FieldQueryStatus,
    FieldSideBinding,
    PreparedFieldReconstruction,
    SimplicialLocationPolicy,
)
from phydrax.discretization.fem import prepare_finite_element_field_reconstruction
from phydrax.discretization.finite_volume import (
    prepare_finite_volume_field_reconstruction,
)
from phydrax.linalg import (
    ArraySpace,
    DenseCholesky,
    DenseLinearOperator,
    DiagonalPairing,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    OperatorPairing,
    OperatorProperties,
    prepare,
)


jax.config.update("jax_enable_x64", True)

_SQUARE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_DIAGONAL_CELLS = ((0, 1, 2), (0, 2, 3))


def _fe_discretization(degree: int, *, field: str = "u") -> Any:
    mesh = phx.discretization.CellMesh(
        jnp.asarray(_SQUARE),
        (
            phx.discretization.CellBlock(
                "cells", "triangle", jnp.asarray(_DIAGONAL_CELLS)
            ),
        ),
    )
    spec = phx.discretization.lagrange_element("triangle", degree)
    return phx.discretization.FiniteElementPlan(
        mesh, phx.discretization.FiniteElementFieldSpec(field, spec)
    ).prepare()


def _fe_nodal(discretization: Any, function: Any) -> Any:
    coordinates = np.asarray(discretization.dof_maps[0].dof_coordinates)
    return jnp.asarray(function(coordinates[:, 0], coordinates[:, 1]))


def _fe_grid(degree: int, resolution: int) -> Any:
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
    mesh = phx.discretization.CellMesh(
        jnp.asarray(vertices),
        (
            phx.discretization.CellBlock(
                "cells", "triangle", jnp.asarray(np.asarray(triangles, np.int32))
            ),
        ),
    )
    spec = phx.discretization.lagrange_element("triangle", degree)
    return phx.discretization.FiniteElementPlan(
        mesh, phx.discretization.FiniteElementFieldSpec("u", spec)
    ).prepare()


_LOCATE_CALLS: list[int] = []


class _CountingKernel(AbstractFieldReconstructionKernel):
    """A kernel that records every point location of its wrapped owner kernel."""

    inner: AbstractFieldReconstructionKernel

    def __init__(self, inner: AbstractFieldReconstructionKernel) -> None:
        self.inner = inner

    @property
    def kernel_id(self) -> str:
        return f"counting:{self.inner.kernel_id}"

    @property
    def cell_count(self) -> int:
        return self.inner.cell_count

    @property
    def support_coverage(self) -> Any:
        return self.inner.support_coverage

    def locate(
        self, points: Any, derivative: tuple[int, ...], side: FieldSideBinding | None
    ) -> Any:
        _LOCATE_CALLS.append(points.shape[0])
        return self.inner.locate(points, derivative, side)

    def apply(self, route: Any, coefficients: Any) -> Any:
        return self.inner.apply(route, coefficients)

    def transpose(self, route: Any, cotangent: Any) -> Any:
        return self.inner.transpose(route, cotangent)

    def bind_side(self, sites: Any, side: Any, cell_ids: Any) -> Any:
        return self.inner.bind_side(sites, side, cell_ids)


def _counted(reconstruction: PreparedFieldReconstruction) -> PreparedFieldReconstruction:
    return PreparedFieldReconstruction(
        _CountingKernel(reconstruction.kernel),
        support_geometry=reconstruction.support_geometry,
        value_port=reconstruction.value_port,
        regularity=reconstruction.regularity,
        trace_policy=reconstruction.trace_policy,
        coefficient_shape=reconstruction.coefficient_shape,
        physical_dimension=reconstruction.physical_dimension,
        maximum_derivative_order=reconstruction.maximum_derivative_order,
        field_space_id=reconstruction.field_space_id,
        support_id=reconstruction.support_id,
    )


def _quadratic(coefficients: Any) -> Any:
    a, b, c, d, e, f = coefficients
    return lambda x, y: a + b * x + c * y + d * x**2 + e * x * y + f * y**2


def _quadratic_dx(coefficients: Any) -> Any:
    _, b, _, d, e, _ = coefficients
    return lambda x, y: b + 2.0 * d * x + e * y


_POINTS = np.asarray(((0.2, 0.1), (0.7, 0.3), (0.3, 0.8), (0.1, 0.6), (0.55, 0.45)))


def test_prepared_query_reuses_its_route_across_coefficient_refreshes() -> None:
    discretization = _fe_discretization(2)
    reconstruction = _counted(
        prepare_finite_element_field_reconstruction(discretization, "u")
    )
    _LOCATE_CALLS.clear()
    values_query = reconstruction.prepare_query(_POINTS)
    gradient_query = reconstruction.prepare_query(_POINTS, derivative=(1, 0))
    assert _LOCATE_CALLS == [_POINTS.shape[0], _POINTS.shape[0]]

    evaluate = eqx.filter_jit(lambda query, state: query.apply(state))
    for coefficients in (
        (1.0, 2.0, -1.0, 0.5, 3.0, -2.0),
        (0.0, -1.0, 4.0, 2.0, 0.0, 1.5),
        (3.0, 0.0, 0.0, -1.0, 1.0, 0.25),
    ):
        state = _fe_nodal(discretization, _quadratic(coefficients))
        np.testing.assert_allclose(
            evaluate(values_query, state),
            _quadratic(coefficients)(_POINTS[:, 0], _POINTS[:, 1]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            evaluate(gradient_query, state),
            _quadratic_dx(coefficients)(_POINTS[:, 0], _POINTS[:, 1]),
            atol=1e-11,
        )
    assert _LOCATE_CALLS == [_POINTS.shape[0], _POINTS.shape[0]]
    assert values_query.complete and values_query.approximation == "exact"
    assert values_query.output_shape == (_POINTS.shape[0],)


def test_masked_query_admits_valid_points_and_keeps_every_status() -> None:
    discretization = _fe_discretization(1)
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    points = np.concatenate((_POINTS[:2], ((1.5, 0.5),), _POINTS[2:3], ((-0.2, 0.4),)))
    state = _fe_nodal(discretization, lambda x, y: 1.0 + 2.0 * x - y)

    query = reconstruction.prepare_query(points, coverage="masked")

    assert not query.complete
    assert query.admitted.tolist() == [0, 1, 3]
    assert query.evidence.status.tolist() == [
        int(FieldQueryStatus.VALID),
        int(FieldQueryStatus.VALID),
        int(FieldQueryStatus.OUTSIDE_SUPPORT),
        int(FieldQueryStatus.VALID),
        int(FieldQueryStatus.OUTSIDE_SUPPORT),
    ]
    admitted = points[[0, 1, 3]]
    np.testing.assert_allclose(
        query.apply(state), 1.0 + 2.0 * admitted[:, 0] - admitted[:, 1], atol=1e-12
    )
    with pytest.raises(ValueError, match="OUTSIDE_SUPPORT"):
        reconstruction.prepare_query(points)
    with pytest.raises(ValueError, match="No prepared query point is valid"):
        reconstruction.prepare_query(np.asarray(((2.0, 2.0),)), coverage="masked")


def _p1_mass() -> np.ndarray:
    """Independent P1 mass of the two-triangle square (area / 12 local stencil)."""
    mass = np.zeros((4, 4))
    local = (0.5 / 12.0) * np.asarray(((2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0)))
    for cell in _DIAGONAL_CELLS:
        mass[np.ix_(cell, cell)] += local
    return mass


def test_query_transpose_is_the_coordinate_dual_and_adjoint_uses_declared_pairings() -> (
    None
):
    discretization = _fe_discretization(1)
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    query = reconstruction.prepare_query(_POINTS)
    rng = np.random.default_rng(7)
    state = jnp.asarray(rng.normal(size=4))
    cotangent = jnp.asarray(rng.normal(size=_POINTS.shape[0]))
    measure = jnp.asarray(rng.uniform(0.5, 2.0, size=_POINTS.shape[0]))
    mass = _p1_mass()
    euclidean = ArraySpace((4,), dtype=np.float64)
    riesz = DenseLinearOperator(
        jnp.asarray(mass),
        source=euclidean,
        target=euclidean,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
    )
    pairing = OperatorPairing(
        riesz,
        prepared_inverse=prepare(
            LinearSystem(riesz),
            LinearSolvePolicy(DenseCholesky(), failure=FailurePolicy("error")),
        ),
    )

    evidence = query.duality_evidence(state, cotangent)
    operator = query.as_linear_operator(
        coefficient_pairing=pairing, value_pairing=DiagonalPairing(measure)
    )
    transpose = operator.transpose_mv(cotangent)
    adjoint = operator.adjoint_mv(cotangent)

    assert bool(evidence.valid)
    np.testing.assert_allclose(
        jnp.vdot(query.apply(state), cotangent), jnp.vdot(state, transpose), atol=1e-12
    )
    np.testing.assert_allclose(
        jnp.vdot(query.apply(state), measure * cotangent),
        state @ mass @ adjoint,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        adjoint, np.linalg.solve(mass, query.transpose(measure * cotangent)), atol=1e-10
    )
    assert not np.allclose(adjoint, transpose)


def test_prepared_query_route_size_is_independent_of_the_coefficient_count() -> None:
    policy = SimplicialLocationPolicy(64, 16, 1)
    retained = []
    for resolution in (8, 16):
        discretization = _fe_grid(2, resolution)
        reconstruction = prepare_finite_element_field_reconstruction(
            discretization, "u", location_policy=policy
        )
        query = reconstruction.prepare_query(_POINTS)
        state = _fe_nodal(discretization, lambda x, y: x**2 - 2.0 * x * y + y)
        np.testing.assert_allclose(
            query.apply(state),
            _POINTS[:, 0] ** 2 - 2.0 * _POINTS[:, 0] * _POINTS[:, 1] + _POINTS[:, 1],
            atol=1e-11,
        )
        retained.append(sum(leaf.size for leaf in jax.tree.leaves(query.route)))

    # Quadrupling the coefficient count leaves the located route unchanged.
    assert retained[0] == retained[1]


def test_one_sided_queries_bind_the_requested_trace_side() -> None:
    discretization = _fe_discretization(1)
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    # Nodal values of x - y below the diagonal and zero above it.
    state = jnp.asarray((0.0, 1.0, 0.0, 0.0))
    diagonal = np.asarray(((0.3, 0.3), (0.6, 0.6)))

    lower = reconstruction.prepare_query(
        diagonal, derivative=(1, 0), side="owner", cell_ids=np.zeros(2, np.int32)
    )
    upper = reconstruction.prepare_query(
        diagonal, derivative=(1, 0), side="owner", cell_ids=np.ones(2, np.int32)
    )

    np.testing.assert_allclose(lower.apply(state), (1.0, 1.0), atol=1e-12)
    np.testing.assert_allclose(upper.apply(state), (0.0, 0.0), atol=1e-12)
    with pytest.raises(ValueError, match="SIDE_REQUIRED"):
        reconstruction.prepare_query(diagonal, derivative=(1, 0))
    with pytest.raises(ValueError, match="require a side"):
        reconstruction.prepare_query(diagonal, cell_ids=np.zeros(2, np.int32))
    with pytest.raises(ValueError, match="maximum_derivative_order"):
        reconstruction.prepare_query(diagonal, derivative=(1, 1))


def test_query_refuses_a_refreshed_geometry_revision() -> None:
    discretization = _fe_discretization(1)
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    query = reconstruction.prepare_query(_POINTS[:2])
    moved = discretization.prepare_runtime(
        np.asarray(_SQUARE) * 1.5, numeric_version="moved"
    )
    refreshed = prepare_finite_element_field_reconstruction(
        discretization, "u", runtime=moved
    )

    query.require_reconstruction(reconstruction)
    with pytest.raises(ValueError, match="another reconstruction revision"):
        query.require_reconstruction(refreshed)


def _weno_square() -> Any:
    resolution = 4
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


def test_nonlinear_query_exposes_a_linearization_instead_of_a_transpose() -> None:
    discretization = _weno_square()
    weno = phx.discretization.UnstructuredWENOZReconstructionPlan(
        2, limiter="none"
    ).prepare(discretization)
    reconstruction = prepare_finite_volume_field_reconstruction(discretization, weno)
    points = np.asarray(discretization.cell_centers)[:4] + 0.01
    x = discretization.cell_quadrature_points[..., 0]
    y = discretization.cell_quadrature_points[..., 1]
    values = jnp.sin(3.0 * x) * jnp.cos(2.0 * y) + x * y
    state = (
        jnp.sum(discretization.cell_quadrature_weights * values, axis=1)
        / discretization.cell_volumes
    )[:, None]
    direction = jnp.asarray(np.random.default_rng(3).normal(size=state.shape))
    query = reconstruction.prepare_query(points)

    linearization = query.linearize(state)
    step = 1.0e-6
    central = (
        query.apply(state + step * direction) - query.apply(state - step * direction)
    ) / (2.0 * step)

    assert not query.coefficient_linear
    np.testing.assert_allclose(
        linearization.primal,
        reconstruction.evaluate(state, points).values,
        atol=1e-13,
    )
    np.testing.assert_allclose(linearization.jvp(direction), central, atol=1e-7)
    cotangent = jnp.ones(query.output_shape)
    np.testing.assert_allclose(
        jnp.vdot(linearization.vjp(cotangent), direction),
        jnp.vdot(cotangent, linearization.jvp(direction)),
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="nonlinear"):
        query.transpose(cotangent)
    with pytest.raises(ValueError, match="nonlinear"):
        query.as_linear_operator()


# --- Explicit polygon H1 and virtual elements ---

_POLYGON_COORDINATES = np.asarray(
    (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.5, 0.0),
        (0.55, 0.5),
        (0.5, 1.0),
    )
)
# A pentagon, a triangle, and a quadrilateral meeting at the interior vertex 5.
_POLYGON_CELLS = tuple(
    np.asarray(loop, dtype=np.int32)
    for loop in ((0, 4, 5, 6, 3), (4, 1, 5), (1, 2, 6, 5))
)
_POLYGON_POINTS = np.asarray(
    ((0.2, 0.3), (0.7, 0.1), (0.8, 0.7), (0.3, 0.9), (0.62, 0.52))
)


def _polygon_mesh() -> Any:
    return phx.discretization.CellMesh.from_polygons(
        jnp.asarray(_POLYGON_COORDINATES), _POLYGON_CELLS
    )


def _polygon_h1_reconstruction() -> tuple[Any, Any]:
    space = phx.discretization.ExplicitPolygonH1Plan(
        _polygon_mesh(), phx.discretization.ExplicitPolygonH1FieldSpec("u")
    ).prepare()
    module = phx.discretization.explicit_polygon_h1
    return space, module.prepare_explicit_polygon_h1_field_reconstruction(space)


def _vem_space(element: Any) -> Any:
    return phx.discretization.VirtualElementPlan(
        _polygon_mesh(), phx.discretization.VirtualElementFieldSpec("u", element)
    ).prepare()


def _convex_polygon_mean(vertices: np.ndarray, function: Any) -> float:
    """Mean over a convex polygon; the edge-midpoint rule is exact for quadratics."""
    total = 0.0
    area = 0.0
    for index in range(1, vertices.shape[0] - 1):
        a, b, c = vertices[0], vertices[index], vertices[index + 1]
        measure = 0.5 * abs((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0])
        midpoints = np.stack((0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)))
        total += measure * np.mean(function(midpoints[:, 0], midpoints[:, 1]))
        area += measure
    return total / area


def _vem_quadratic_dofs(mesh: Any, function: Any) -> Any:
    """Degree-2 H1 VEM DOFs: vertex values, edge midpoints, and cell means."""
    points = _POLYGON_COORDINATES
    edges = np.asarray(mesh.connectivity.edges)
    midpoints = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    means = [
        _convex_polygon_mean(points[loop], function)
        for block in mesh.blocks
        for loop in np.asarray(block.vertices)
    ]
    return jnp.asarray(
        np.concatenate(
            (
                function(points[:, 0], points[:, 1]),
                function(midpoints[:, 0], midpoints[:, 1]),
                np.asarray(means),
            )
        )
    )


def test_explicit_polygon_h1_query_is_exact_for_affine_fields() -> None:
    space, reconstruction = _polygon_h1_reconstruction()
    affine = jnp.asarray(
        0.5 + 2.0 * _POLYGON_COORDINATES[:, 0] - 3.0 * _POLYGON_COORDINATES[:, 1]
    )
    values = reconstruction.prepare_query(_POLYGON_POINTS)
    slopes = (
        reconstruction.prepare_query(_POLYGON_POINTS, derivative=(1, 0)),
        reconstruction.prepare_query(_POLYGON_POINTS, derivative=(0, 1)),
    )

    assert reconstruction.approximation == "exact"
    assert values.complete
    np.testing.assert_allclose(
        values.apply(affine),
        0.5 + 2.0 * _POLYGON_POINTS[:, 0] - 3.0 * _POLYGON_POINTS[:, 1],
        atol=1e-12,
    )
    np.testing.assert_allclose(slopes[0].apply(affine), 2.0, atol=1e-11)
    np.testing.assert_allclose(slopes[1].apply(affine), -3.0, atol=1e-11)

    # A non-affine vertex state agrees with the direct fan reconstruction.
    state = jnp.asarray(np.random.default_rng(11).normal(size=7))
    direct = phx.discretization.prepare_explicit_polygon_h1_reconstruction(space, state)
    # One point per block (triangle, quadrilateral, pentagon), each block one cell.
    block_points = _POLYGON_POINTS[[1, 2, 0]]
    for block, point in enumerate(block_points):
        value, gradient = phx.discretization.evaluate_explicit_polygon_h1_reconstruction(
            direct, space, block, point[None, None, :]
        )
        np.testing.assert_allclose(
            reconstruction.prepare_query(point).apply(state), value[0], atol=1e-12
        )
        for axis, derivative in enumerate(((1, 0), (0, 1))):
            np.testing.assert_allclose(
                reconstruction.prepare_query(point, derivative=derivative).apply(state),
                gradient[0, :, axis],
                atol=1e-11,
            )


def test_explicit_polygon_h1_query_masks_outside_points_and_fan_edge_gradients() -> None:
    space, reconstruction = _polygon_h1_reconstruction()
    state = jnp.asarray(np.random.default_rng(5).normal(size=7))
    points = np.asarray(((0.2, 0.3), (1.2, 0.5), (0.8, 0.7), (-0.1, -0.1)))

    masked = reconstruction.prepare_query(points, coverage="masked")

    assert not masked.complete
    assert masked.admitted.tolist() == [0, 2]
    assert masked.evidence.status.tolist() == [
        int(FieldQueryStatus.VALID),
        int(FieldQueryStatus.OUTSIDE_SUPPORT),
        int(FieldQueryStatus.VALID),
        int(FieldQueryStatus.OUTSIDE_SUPPORT),
    ]
    np.testing.assert_allclose(
        masked.apply(state),
        reconstruction.prepare_query(points[[0, 2]]).apply(state),
        atol=1e-14,
    )

    # The triangle (block 0) is fanned from its witness; a point on the fan
    # edge towards vertex 4 has a continuous value but a two-valued gradient.
    witness = np.asarray(space.default_runtime.bases[0].witness[0])
    fan_edge = 0.5 * (witness + _POLYGON_COORDINATES[4])[None, :]
    vertex = _POLYGON_COORDINATES[5][None, :]
    for point in (fan_edge, vertex):
        assert reconstruction.validity(point).status.tolist() == [
            int(FieldQueryStatus.VALID)
        ]
        assert reconstruction.validity(point, derivative=(1, 0)).status.tolist() == [
            int(FieldQueryStatus.SIDE_UNRESOLVED)
        ]
    with pytest.raises(ValueError, match="SIDE_UNRESOLVED"):
        reconstruction.prepare_query(fan_edge, derivative=(0, 1), side="owner")

    # On a shared cell edge the gradient needs a side; the side cell resolves it.
    shared = np.asarray(((0.525, 0.25),))
    assert reconstruction.validity(shared, derivative=(1, 0)).status.tolist() == [
        int(FieldQueryStatus.SIDE_REQUIRED)
    ]
    affine = jnp.asarray(
        1.0 - _POLYGON_COORDINATES[:, 0] + 4.0 * _POLYGON_COORDINATES[:, 1]
    )
    for cell in (0, 2):
        one_sided = reconstruction.prepare_query(
            shared, derivative=(0, 1), side="owner", cell_ids=np.asarray((cell,))
        )
        np.testing.assert_allclose(one_sided.apply(affine), (4.0,), atol=1e-11)


def _quadratic_dy(coefficients: Any) -> Any:
    _, _, c, _, e, f = coefficients
    return lambda x, y: c + e * x + 2.0 * f * y


def test_virtual_element_projected_query_reuses_its_route_across_refreshes() -> None:
    space = _vem_space(phx.discretization.conforming_h1_virtual_element(2))
    base = phx.equations.vem.prepare_virtual_element_field_reconstruction(
        space, channel="h1-projection"
    )
    reconstruction = PreparedFieldReconstruction(
        _CountingKernel(base.kernel),
        support_geometry=base.support_geometry,
        value_port=base.value_port,
        regularity=base.regularity,
        trace_policy=base.trace_policy,
        coefficient_shape=base.coefficient_shape,
        physical_dimension=base.physical_dimension,
        maximum_derivative_order=base.maximum_derivative_order,
        field_space_id=base.field_space_id,
        support_id=base.support_id,
        approximation=base.approximation,
    )
    _LOCATE_CALLS.clear()
    values = reconstruction.prepare_query(_POLYGON_POINTS)
    slopes = reconstruction.prepare_query(_POLYGON_POINTS, derivative=(0, 1))
    evaluate = eqx.filter_jit(lambda query, state: query.apply(state))
    x, y = _POLYGON_POINTS[:, 0], _POLYGON_POINTS[:, 1]

    # The energy projection reproduces every quadratic from its DOFs.
    for coefficients in (
        (1.0, 2.0, -1.0, 0.5, 3.0, -2.0),
        (0.0, -1.0, 4.0, 2.0, 0.0, 1.5),
        (3.0, 0.0, 0.0, -1.0, 1.0, 0.25),
    ):
        state = _vem_quadratic_dofs(space.mesh, _quadratic(coefficients))
        np.testing.assert_allclose(
            evaluate(values, state), _quadratic(coefficients)(x, y), atol=1e-10
        )
        np.testing.assert_allclose(
            evaluate(slopes, state), _quadratic_dy(coefficients)(x, y), atol=1e-9
        )
    assert _LOCATE_CALLS == [_POLYGON_POINTS.shape[0], _POLYGON_POINTS.shape[0]]
    assert values.approximation == "h1-projection"
    assert bool(values.duality_evidence(state, jnp.arange(5.0)).valid)


def test_virtual_element_channels_are_labeled_and_family_specific() -> None:
    prepare = phx.equations.vem.prepare_virtual_element_field_reconstruction
    hdiv = _vem_space(phx.discretization.conforming_hdiv_virtual_element(1))
    discontinuous = _vem_space(phx.discretization.discontinuous_l2_virtual_element(1))

    with pytest.raises(ValueError, match="scalar ConformingH1"):
        prepare(hdiv, channel="h1-projection")
    with pytest.raises(ValueError, match="scalar ConformingH1"):
        prepare(discontinuous, channel="h1-projection")
    with pytest.raises(ValueError, match="channel"):
        prepare(hdiv, channel="exact")  # ty: ignore[invalid-argument-type]

    # H(div) moments of a constant field: the canonical normal flux mean on each
    # edge (lower to higher vertex, normal rotated clockwise) and cell means.
    field = np.asarray((0.7, -1.3))
    edges = np.asarray(hdiv.mesh.connectivity.edges)
    tangents = _POLYGON_COORDINATES[edges[:, 1]] - _POLYGON_COORDINATES[edges[:, 0]]
    normals = np.stack((tangents[:, 1], -tangents[:, 0]), axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    edge_moments = np.stack((normals @ field, np.zeros(edges.shape[0])), axis=1)
    cell_moments = np.tile(field, (3, 1))
    state = jnp.asarray(np.concatenate((edge_moments.ravel(), cell_moments.ravel())))

    reconstruction = prepare(hdiv, channel="l2-projection")
    query = reconstruction.prepare_query(_POLYGON_POINTS)

    assert reconstruction.approximation == "l2-projection"
    assert query.output_shape == (_POLYGON_POINTS.shape[0], 2)
    np.testing.assert_allclose(query.apply(state), np.tile(field, (5, 1)), atol=1e-11)


# --- Point clouds ---
# Degree-2 moving least squares reproduces quadratics wherever the local fit is
# admitted; references are host polynomial evaluations and host neighbor counts.
# A point cloud publishes masked point queries with support and conditioning
# evidence; it publishes no facet trace.

_CLOUD_RADIUS = 0.15
_CLOUD_QUERIES = np.asarray(((0.7, 0.7), (0.55, 0.85), (0.17, 0.1), (0.3, 0.4)))


def _point_cloud_points() -> np.ndarray:
    """Scattered points outside the lower-left square plus one collinear stencil."""
    rng = np.random.default_rng(0)
    scattered = rng.uniform(0.0, 1.0, (200, 2))
    scattered = scattered[(scattered[:, 0] > 0.6) | (scattered[:, 1] > 0.6)]
    line = np.stack((np.linspace(0.05, 0.29, 7), np.full(7, 0.1)), axis=-1)
    return np.concatenate((scattered, line))


def _point_cloud_reconstruction(points: np.ndarray, quadrature: np.ndarray) -> Any:
    discretization = phx.discretization.PointCloudPlan(points, quadrature, degree=2)
    return phx.discretization.prepare_point_cloud_field_reconstruction(
        discretization.prepare(),
        support_geometry=phx.geometry.Rectangle((0.5, 0.5), (1.0, 1.0)).compile(),
        radius=_CLOUD_RADIUS,
        capacity=points.shape[0],
    )


def test_point_cloud_masked_query_keeps_partial_support_and_conditioning_evidence() -> (
    None
):
    points = _point_cloud_points()
    reconstruction = _point_cloud_reconstruction(
        points, np.full(points.shape[0], 1.0 / points.shape[0])
    )
    neighbors = np.sum(
        np.linalg.norm(points[None, :, :] - _CLOUD_QUERIES[:, None, :], axis=-1)
        < _CLOUD_RADIUS,
        axis=1,
    )

    query = reconstruction.prepare_query(_CLOUD_QUERIES, coverage="masked")

    # The collinear stencil has neighbors but no unisolvent quadratic fit, and
    # the lower-left hole inside the support geometry has no neighbors at all.
    assert reconstruction.kernel.support_coverage == "partial"
    assert not query.complete
    assert query.admitted.tolist() == [0, 1]
    assert query.evidence.status.tolist() == [
        int(FieldQueryStatus.VALID),
        int(FieldQueryStatus.VALID),
        int(FieldQueryStatus.ILL_CONDITIONED),
        int(FieldQueryStatus.OUTSIDE_SUPPORT),
    ]
    assert query.evidence.support_count.tolist() == neighbors.tolist()
    assert neighbors[2] == 7 and neighbors[3] == 0
    conditioning = np.asarray(query.evidence.conditioning)
    assert np.all(np.isfinite(conditioning[:2])) and np.all(conditioning[:2] >= 1.0)
    assert not np.isfinite(conditioning[2])
    admitted = _CLOUD_QUERIES[:2]
    for coefficients in (
        (0.3, -1.0, 2.0, 0.5, -0.7, 1.1),
        (1.0, 0.0, 0.0, 0.0, 2.0, 0.0),
    ):
        field = _quadratic(coefficients)
        state = jnp.asarray(field(points[:, 0], points[:, 1]))
        np.testing.assert_allclose(
            query.apply(state), field(admitted[:, 0], admitted[:, 1]), atol=1e-10
        )
    with pytest.raises(ValueError, match=r"ILL_CONDITIONED, OUTSIDE_SUPPORT"):
        reconstruction.prepare_query(_CLOUD_QUERIES)
    with pytest.raises(ValueError, match="maximum_derivative_order=0"):
        reconstruction.prepare_query(admitted, derivative=(1, 0))


def test_point_cloud_query_adjoint_uses_the_cloud_quadrature_pairing() -> None:
    points = _point_cloud_points()
    rng = np.random.default_rng(11)
    quadrature = rng.uniform(0.5, 2.0, points.shape[0])
    reconstruction = _point_cloud_reconstruction(points, quadrature)
    state = jnp.asarray(rng.normal(size=points.shape[0]))
    cotangent = jnp.asarray(rng.normal(size=2))
    query = reconstruction.prepare_query(_CLOUD_QUERIES[:2])

    operator = query.as_linear_operator(
        coefficient_pairing=DiagonalPairing(jnp.asarray(quadrature))
    )
    transpose = np.asarray(operator.transpose_mv(cotangent))
    adjoint = np.asarray(operator.adjoint_mv(cotangent))

    assert bool(query.duality_evidence(state, cotangent).valid)
    # Only cloud points inside an admitted neighborhood carry transpose weight.
    distances = np.linalg.norm(
        points[None, :, :] - _CLOUD_QUERIES[:2, None, :], axis=-1
    ).min(axis=0)
    assert np.all(transpose[distances >= _CLOUD_RADIUS] == 0.0)
    np.testing.assert_allclose(
        jnp.vdot(query.apply(state), cotangent),
        np.sum(quadrature * np.asarray(state) * adjoint),
        atol=1e-12,
    )
    np.testing.assert_allclose(adjoint, transpose / quadrature, atol=1e-12)
    assert not np.allclose(adjoint, transpose)


# --- Finite differences and global spectral ---
# Uniform tensor grids and tensor spectral spaces; references are host numpy
# polynomial and trigonometric evaluations of the sampled fields.


def _tensor_fd(shape: tuple[int, ...], bounds: Any) -> Any:
    names = ("x", "y", "z")[: len(shape)]
    d = phx.discretization
    grid = d.TensorGridPlan(
        tuple(d.UniformAxisSpec(count) for count in shape), axis_names=names
    ).prepare(jnp.asarray(bounds, dtype=jnp.float64))
    requests = tuple(
        d.DerivativeRequest(f"d{name}", grid, name, derivative_order=1, accuracy_order=2)
        for name in names
    )
    return d.FiniteDifferencePlan(grid, requests, field_name="u").prepare()


def _fd_nodes(discretization: Any) -> tuple[np.ndarray, ...]:
    layout = discretization.grid.primary_entity_layout
    axes = tuple(np.asarray(values) for values in layout.coordinates_by_axis)
    return tuple(np.meshgrid(*axes, indexing="ij"))


def _tensor_cubic(scale: float) -> Any:
    def field(x: Any, y: Any) -> Any:
        return 1.0 + scale * x - 2.0 * y + x**2 * y - 0.5 * x**3 + scale * x * y**3

    def dx(x: Any, y: Any) -> Any:
        return scale + 2.0 * x * y - 1.5 * x**2 + scale * y**3

    def dyy(x: Any, y: Any) -> Any:
        return 6.0 * scale * x * y

    return field, dx, dyy


def test_fd_bspline_query_reproduces_tensor_cubics_across_refreshes() -> None:
    discretization = _tensor_fd((9, 7), ((0.0, -1.0), (2.0, 1.0)))
    reconstruction = phx.discretization.prepare_finite_difference_field_reconstruction(
        discretization, interpolation=phx.discretization.BSplineGridInterpolation(3)
    )
    x, y = _fd_nodes(discretization)
    points = np.random.default_rng(3).uniform((0.0, -1.0), (2.0, 1.0), (12, 2))
    values = reconstruction.prepare_query(points)
    slopes = reconstruction.prepare_query(points, derivative=(1, 0))
    curvatures = reconstruction.prepare_query(points, derivative=(0, 2))

    for scale in (0.7, -1.3):
        field, dx, dyy = _tensor_cubic(scale)
        nodal = jnp.asarray(field(x, y))
        np.testing.assert_allclose(
            values.apply(nodal), field(points[:, 0], points[:, 1]), atol=1e-11
        )
        np.testing.assert_allclose(
            slopes.apply(nodal), dx(points[:, 0], points[:, 1]), atol=1e-10
        )
        np.testing.assert_allclose(
            curvatures.apply(nodal), dyy(points[:, 0], points[:, 1]), atol=1e-9
        )
    assert values.complete and values.coefficient_shape == (9, 7)
    with pytest.raises(ValueError, match="maximum_derivative_order"):
        reconstruction.prepare_query(points, derivative=(0, 3))


def test_fd_query_transpose_is_dual_and_adjoint_uses_the_sbp_norm() -> None:
    discretization = _tensor_fd((13, 15), ((0.0, 0.0), (1.0, 2.0)))
    grid = discretization.grid
    norm = phx.discretization.SBPGridNorm(
        tuple(
            phx.discretization.SBPDerivativePlan(grid, name, interior_order=4).prepare()
            for name in ("x", "y")
        )
    )
    reconstruction = phx.discretization.prepare_finite_difference_field_reconstruction(
        discretization, interpolation=phx.discretization.MultilinearGridInterpolation()
    )
    rng = np.random.default_rng(7)
    points = rng.uniform((0.0, 0.0), (1.0, 2.0), (6, 2))
    outside = np.concatenate((points, ((1.5, 0.5), (0.5, -0.1))))
    x, y = _fd_nodes(discretization)
    query = reconstruction.prepare_query(points)
    state = jnp.asarray(rng.normal(size=grid.shape))
    cotangent = jnp.asarray(rng.normal(size=6))
    point_weights = jnp.asarray(rng.uniform(0.5, 2.0, 6))

    bilinear = 0.5 + 2.0 * x - y + 3.0 * x * y
    operator = query.as_linear_operator(
        coefficient_pairing=norm.pairing(),
        value_pairing=DiagonalPairing(point_weights),
    )
    adjoint = operator.adjoint_mv(cotangent)
    masked = reconstruction.prepare_query(outside, coverage="masked")

    np.testing.assert_allclose(
        query.apply(jnp.asarray(bilinear)),
        0.5 + 2.0 * points[:, 0] - points[:, 1] + 3.0 * points[:, 0] * points[:, 1],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jnp.vdot(query.apply(state), cotangent),
        jnp.vdot(state, query.transpose(cotangent)),
        atol=1e-12,
    )
    # Hilbert adjoint in the tensor SBP norm H and a weighted point pairing.
    np.testing.assert_allclose(
        jnp.sum(point_weights * query.apply(state) * cotangent),
        jnp.sum(norm.weights * state * adjoint),
        atol=1e-12,
    )
    assert not np.allclose(adjoint, query.transpose(cotangent))
    assert masked.admitted.tolist() == list(range(6)) and not masked.complete
    with pytest.raises(ValueError, match="OUTSIDE_SUPPORT"):
        reconstruction.prepare_query(outside)
    with pytest.raises(ValueError, match="maximum_derivative_order=0"):
        reconstruction.prepare_query(points, derivative=(1, 0))


def _fourier_chebyshev(field: str = "u") -> Any:
    d = phx.discretization
    return d.TensorSpectralPlan(
        (d.FourierBasisPlan(16), d.ChebyshevBasisPlan(10)),
        axis_names=("x", "y"),
        field_name=field,
    ).prepare((d.AxisDomain.periodic(0.0, 1.0), d.AxisDomain.interval(-1.0, 1.0)))


def _spectral_nodes(space: Any) -> tuple[np.ndarray, ...]:
    axes = tuple(np.asarray(axis.nodes) for axis in space.axes)
    return tuple(np.meshgrid(*axes, indexing="ij"))


def _trig_polynomial(x: Any, y: Any) -> Any:
    tau = 2.0 * np.pi
    return np.sin(tau * x) * (y**3 - y) + np.cos(2.0 * tau * x) * y**2 + 0.5


def _trig_polynomial_dx(x: Any, y: Any) -> Any:
    tau = 2.0 * np.pi
    return tau * np.cos(tau * x) * (y**3 - y) - 2.0 * tau * np.sin(2.0 * tau * x) * y**2


def _trig_polynomial_dxdy(x: Any, y: Any) -> Any:
    tau = 2.0 * np.pi
    return (
        tau * np.cos(tau * x) * (3.0 * y**2 - 1.0) - 4.0 * tau * np.sin(2.0 * tau * x) * y
    )


def test_spectral_query_is_exact_for_trigonometric_polynomial_fields() -> None:
    space = _fourier_chebyshev()
    reconstruction = phx.discretization.prepare_spectral_field_reconstruction(space)
    x, y = _spectral_nodes(space)
    rng = np.random.default_rng(13)
    points = rng.uniform((0.0, -1.0), (1.0, 1.0), (10, 2))
    values = reconstruction.prepare_query(points)
    slopes = reconstruction.prepare_query(points, derivative=(1, 0))
    mixed = reconstruction.prepare_query(points, derivative=(1, 1))
    coefficients = space.project(jnp.asarray(_trig_polynomial(x, y)))
    cotangent = jnp.asarray(rng.normal(size=10))

    for scale in (1.0, -2.5):
        state = scale * coefficients
        np.testing.assert_allclose(
            values.apply(state),
            scale * _trig_polynomial(points[:, 0], points[:, 1]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            slopes.apply(state),
            scale * _trig_polynomial_dx(points[:, 0], points[:, 1]),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            mixed.apply(state),
            scale * _trig_polynomial_dxdy(points[:, 0], points[:, 1]),
            atol=1e-9,
        )
    # Real synthesis of complex modes: <R c, w> = Re(sum(c * R^T w)).
    np.testing.assert_allclose(
        jnp.sum(values.apply(coefficients) * cotangent),
        jnp.real(jnp.sum(coefficients * values.transpose(cotangent))),
        atol=1e-12,
    )
    outside = np.concatenate((points[:2], ((0.5, 1.2),)))
    masked = reconstruction.prepare_query(outside, coverage="masked")
    assert masked.admitted.tolist() == [0, 1]
    assert masked.evidence.status.tolist()[2] == int(FieldQueryStatus.OUTSIDE_SUPPORT)
    with pytest.raises(ValueError, match="OUTSIDE_SUPPORT"):
        reconstruction.prepare_query(outside)
    with pytest.raises(ValueError, match="maximum_derivative_order"):
        reconstruction.prepare_query(points, derivative=(3, 0))


def test_real_spectral_query_operator_acts_on_realified_complex_modes() -> None:
    space = _fourier_chebyshev()
    reconstruction = phx.discretization.prepare_spectral_field_reconstruction(space)
    x, y = _spectral_nodes(space)
    rng = np.random.default_rng(29)
    points = rng.uniform((0.0, -1.0), (1.0, 1.0), (5, 2))
    query = reconstruction.prepare_query(points)
    modes = np.asarray(space.project(jnp.asarray(_trig_polynomial(x, y))))
    cotangent = rng.normal(size=5)

    operator = query.as_linear_operator()
    assert isinstance(operator.source, ArraySpace)
    shape = operator.source.shape
    assert shape == (*reconstruction.coefficient_shape, 2)
    assert operator.source.dtype == jnp.float64
    np.testing.assert_allclose(
        operator.mv(jnp.asarray(np.stack((modes.real, modes.imag), axis=-1))),
        _trig_polynomial(points[:, 0], points[:, 1]),
        atol=1e-12,
    )
    # Real matrix of the R-linear synthesis, probed column by column through mv.
    size = operator.source.size
    basis = jnp.asarray(np.eye(size).reshape((size, *shape)))
    matrix = np.asarray(jax.vmap(operator.mv)(basis)).T
    assert np.max(np.abs(matrix[:, 1::2])) > 0.1  # imaginary parts contribute
    np.testing.assert_allclose(
        np.asarray(operator.transpose_mv(jnp.asarray(cotangent))).reshape(-1),
        matrix.T @ cotangent,
        atol=1e-12,
    )
    weights = rng.uniform(0.5, 2.0, size=shape)
    measure = rng.uniform(0.5, 2.0, size=5)
    weighted = query.as_linear_operator(
        coefficient_pairing=DiagonalPairing(jnp.asarray(weights)),
        value_pairing=DiagonalPairing(jnp.asarray(measure)),
    )
    np.testing.assert_allclose(
        np.asarray(weighted.adjoint_mv(jnp.asarray(cotangent))).reshape(-1),
        matrix.T @ (measure * cotangent) / weights.reshape(-1),
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="native coefficient dtype"):
        query.as_linear_operator(dtype=jnp.float64)


# --- Isogeometric analysis ---
# Single NURBS patches: an exact rational quarter annulus 1 <= r <= 2 in the
# first quadrant and affine boxes. References are host polynomial evaluations
# and host (scipy) B-spline collocation; physical-linear fields are exact in
# the isoparametric rational space (coefficients are affine images of the
# control points).

_IGA_HALF = np.sqrt(0.5)


def _iga_insert_knot(
    knots: np.ndarray, controls: np.ndarray, weights: np.ndarray, value: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boehm insertion of one knot into a quadratic NURBS curve."""
    homogeneous = np.concatenate((controls * weights[:, None], weights[:, None]), axis=1)
    span = np.searchsorted(knots, value, side="right") - 1
    refined = []
    for index in range(homogeneous.shape[0] + 1):
        if index <= span - 2:
            refined.append(homogeneous[index])
        elif index > span:
            refined.append(homogeneous[index - 1])
        else:
            alpha = (value - knots[index]) / (knots[index + 2] - knots[index])
            refined.append(
                alpha * homogeneous[index] + (1 - alpha) * homogeneous[index - 1]
            )
    points = np.asarray(refined)
    return np.sort(np.append(knots, value)), points[:, :2] / points[:, 2:], points[:, 2]


def _iga_quarter_annulus(scale: float = 1.0) -> tuple[Any, Any, np.ndarray]:
    """Plan, prepared patch, and control points of a 2 x 2 span quarter annulus."""
    iga = phx.discretization.iga
    knots, arc, arc_weights = _iga_insert_knot(
        np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0)),
        np.asarray(((1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.asarray((1.0, _IGA_HALF, 1.0)),
        0.5,
    )
    radii = 1.0 + np.asarray((0.0, 0.25, 0.75, 1.0))
    controls = scale * radii[:, None, None] * arc[None, :, :]
    plan = iga.IsogeometricPlan.isoparametric(
        (
            iga.BSplineGrid(jnp.asarray((0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0)), 2),
            iga.BSplineGrid(jnp.asarray(knots), 2),
        ),
        iga.NURBSGeometryState(
            jnp.asarray(controls), jnp.asarray(np.ones(4)[:, None] * arc_weights)
        ),
        axis_names=("r", "theta"),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(3),
    )
    return plan, plan.prepare(numeric_version="annulus"), controls


def _iga_annulus_region() -> Any:
    geometry = phx.geometry
    ring = geometry.Ball((0.0, 0.0), 2.0) - geometry.Ball((0.0, 0.0), 1.0)
    return (ring & geometry.Orthotope((1.0, 1.0), (2.0, 2.0))).compile()


def _iga_annulus_points(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    radius = rng.uniform(1.0, 2.0, count)
    angle = rng.uniform(0.0, 0.5 * np.pi, count)
    # One point on each interior knot line (r = 1.5 and theta = pi / 4).
    radius = np.append(radius, (1.5, 1.3))
    angle = np.append(angle, (0.3, 0.25 * np.pi))
    return np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)


def _iga_box(degree: int) -> tuple[Any, np.ndarray]:
    """Axis-aligned affine patch [0, 2] x [0, 1] with one interior knot per axis."""
    iga = phx.discretization.iga
    knots = np.concatenate((np.zeros(degree + 1), (0.5,), np.ones(degree + 1)))
    grid = iga.BSplineGrid(jnp.asarray(knots), degree)
    greville = np.asarray(grid.greville_abscissae)
    xx, yy = np.meshgrid(2.0 * greville, greville, indexing="ij")
    plan = iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        iga.NURBSGeometryState(
            jnp.asarray(np.stack((xx, yy), axis=-1)), jnp.ones(xx.shape)
        ),
        axis_names=("xi", "eta"),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(degree + 1),
    )
    return plan.prepare(numeric_version="box"), knots


def test_isogeometric_query_reproduces_physical_linear_fields_on_a_curved_patch() -> None:
    _, prepared, controls = _iga_quarter_annulus()
    reconstruction = phx.discretization.iga.prepare_isogeometric_field_reconstruction(
        prepared, "u", support_geometry=_iga_annulus_region()
    )
    points = _iga_annulus_points(10, 3)
    values = reconstruction.prepare_query(points)
    slopes = (
        reconstruction.prepare_query(points, derivative=(1, 0)),
        reconstruction.prepare_query(points, derivative=(0, 1)),
    )
    evaluate = eqx.filter_jit(lambda query, state: query.apply(state))

    for a, b, c in ((0.5, 2.0, -3.0), (-1.0, 0.25, 4.0)):
        state = jnp.asarray(a + b * controls[..., 0] + c * controls[..., 1])
        np.testing.assert_allclose(
            evaluate(values, state), a + b * points[:, 0] + c * points[:, 1], atol=1e-12
        )
        np.testing.assert_allclose(evaluate(slopes[0], state), b, atol=1e-11)
        np.testing.assert_allclose(evaluate(slopes[1], state), c, atol=1e-11)
    assert values.complete and values.approximation == "exact"
    # Rational pieces with C^1 continuity across both simple interior knots.
    assert reconstruction.regularity.continuity == 1
    assert reconstruction.regularity.pieces == "smooth"
    # Inside the hole, beyond the outer arc, and inside the patch.
    probes = np.asarray(((0.3, 0.3), (1.8, 1.8), (1.2, 0.3)))
    masked = reconstruction.prepare_query(probes, coverage="masked")
    assert masked.evidence.status.tolist() == [
        int(FieldQueryStatus.OUTSIDE_SUPPORT),
        int(FieldQueryStatus.OUTSIDE_SUPPORT),
        int(FieldQueryStatus.VALID),
    ]
    with pytest.raises(ValueError, match="OUTSIDE_SUPPORT"):
        reconstruction.prepare_query(probes)


def test_isogeometric_query_reproduces_tensor_quadratics_on_a_box_patch() -> None:
    from scipy.interpolate import BSpline

    prepared, knots = _iga_box(2)
    reconstruction = phx.discretization.iga.prepare_isogeometric_field_reconstruction(
        prepared, "u"
    )

    def field(x: Any, y: Any) -> Any:
        return 1.0 + x - 2.0 * y + 0.5 * x**2 + x * y**2 - y**2 + x**2 * y**2

    greville = np.asarray((0.0, 0.25, 0.75, 1.0))
    collocation = BSpline.design_matrix(greville, knots, 2).toarray()
    xx, yy = np.meshgrid(2.0 * greville, greville, indexing="ij")
    solved = np.linalg.solve(collocation, field(xx, yy))
    state = jnp.asarray(np.linalg.solve(collocation, solved.T).T)
    rng = np.random.default_rng(5)
    points = np.concatenate(
        (rng.uniform((0.0, 0.0), (2.0, 1.0), (8, 2)), ((1.0, 0.3), (0.4, 0.5)))
    )
    x, y = points[:, 0], points[:, 1]

    np.testing.assert_allclose(
        reconstruction.prepare_query(points).apply(state), field(x, y), atol=1e-12
    )
    np.testing.assert_allclose(
        reconstruction.prepare_query(points, derivative=(1, 0)).apply(state),
        1.0 + x + y**2 + 2.0 * x * y**2,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        reconstruction.prepare_query(points, derivative=(0, 1)).apply(state),
        -2.0 + 2.0 * x * y - 2.0 * y + 2.0 * x**2 * y,
        atol=1e-11,
    )
    assert reconstruction.regularity.pieces == "polynomial"
    np.testing.assert_allclose(
        np.asarray(reconstruction.support_geometry.bounds), ((0.0, 0.0), (2.0, 1.0))
    )


def test_isogeometric_query_transpose_and_adjoint_use_declared_pairings() -> None:
    _, prepared, controls = _iga_quarter_annulus()
    reconstruction = phx.discretization.iga.prepare_isogeometric_field_reconstruction(
        prepared, "u", support_geometry=_iga_annulus_region()
    )
    points = _iga_annulus_points(6, 9)
    query = reconstruction.prepare_query(points, derivative=(0, 1))
    rng = np.random.default_rng(2)
    state = jnp.asarray(rng.normal(size=controls.shape[:2]))
    cotangent = jnp.asarray(rng.normal(size=points.shape[0]))
    measure = jnp.asarray(rng.uniform(0.5, 2.0, points.shape[0]))
    coefficient_weights = jnp.asarray(rng.uniform(0.5, 2.0, controls.shape[:2]))

    operator = query.as_linear_operator(
        coefficient_pairing=DiagonalPairing(coefficient_weights),
        value_pairing=DiagonalPairing(measure),
    )
    transpose = operator.transpose_mv(cotangent)
    adjoint = operator.adjoint_mv(cotangent)

    assert bool(query.duality_evidence(state, cotangent).valid)
    np.testing.assert_allclose(
        jnp.vdot(query.apply(state), cotangent), jnp.vdot(state, transpose), atol=1e-12
    )
    np.testing.assert_allclose(
        jnp.vdot(query.apply(state), measure * cotangent),
        jnp.vdot(state, coefficient_weights * adjoint),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        adjoint, query.transpose(measure * cotangent) / coefficient_weights, atol=1e-12
    )
    assert not np.allclose(adjoint, transpose)


def test_isogeometric_query_binds_sides_on_a_c0_knot_line() -> None:
    prepared, _ = _iga_box(1)
    reconstruction = phx.discretization.iga.prepare_isogeometric_field_reconstruction(
        prepared, "u"
    )
    # Hat in x with its kink on the knot line x = 1: u = x, then 2 - x.
    state = jnp.asarray(np.outer((0.0, 1.0, 0.0), np.ones(3)))
    kink = np.asarray(((1.0, 0.25), (1.0, 0.7)))
    # Overlay cells are C-ordered (xi interval, eta interval).
    left = reconstruction.prepare_query(
        kink, derivative=(1, 0), side="owner", cell_ids=np.asarray((0, 1))
    )
    right = reconstruction.prepare_query(
        kink, derivative=(1, 0), side="owner", cell_ids=np.asarray((2, 3))
    )

    np.testing.assert_allclose(left.apply(state), (1.0, 1.0), atol=1e-12)
    np.testing.assert_allclose(right.apply(state), (-1.0, -1.0), atol=1e-12)
    np.testing.assert_allclose(
        reconstruction.prepare_query(kink).apply(state), (1.0, 1.0), atol=1e-12
    )
    assert reconstruction.regularity.continuity == 0
    with pytest.raises(ValueError, match="SIDE_REQUIRED"):
        reconstruction.prepare_query(kink, derivative=(1, 0))
    with pytest.raises(ValueError, match="containing each trace site"):
        reconstruction.prepare_query(kink, side="owner", cell_ids=np.asarray((1, 0)))


def test_isogeometric_query_refuses_undeclared_supports_orders_and_revisions() -> None:
    plan, prepared, _ = _iga_quarter_annulus()
    prepare = phx.discretization.iga.prepare_isogeometric_field_reconstruction
    quarter_disk = (
        phx.geometry.Ball((0.0, 0.0), 2.0)
        & phx.geometry.Orthotope((1.0, 1.0), (2.0, 2.0))
    ).compile()
    reconstruction = prepare(prepared, "u", support_geometry=_iga_annulus_region())
    query = reconstruction.prepare_query(_iga_annulus_points(2, 1))
    moved = prepared.prepare_runtime(
        phx.discretization.iga.NURBSGeometryState(
            1.5 * plan.geometry.control_points, plan.geometry.weights
        ),
        numeric_version="moved",
    )

    with pytest.raises(ValueError, match="not derived"):
        prepare(prepared, "u")
    with pytest.raises(ValueError, match="does not lie on the support boundary"):
        prepare(prepared, "u", support_geometry=quarter_disk)
    with pytest.raises(ValueError, match="maximum_derivative_order"):
        reconstruction.prepare_query(_iga_annulus_points(2, 1), derivative=(1, 1))
    with pytest.raises(KeyError, match="Unknown isogeometric field"):
        prepare(prepared, "p", support_geometry=_iga_annulus_region())
    scaled_region = (
        (phx.geometry.Ball((0.0, 0.0), 3.0) - phx.geometry.Ball((0.0, 0.0), 1.5))
        & phx.geometry.Orthotope((1.5, 1.5), (3.0, 3.0))
    ).compile()
    refreshed = prepare(prepared, "u", runtime=moved, support_geometry=scaled_region)
    query.require_reconstruction(reconstruction)
    with pytest.raises(ValueError, match="another reconstruction revision"):
        query.require_reconstruction(refreshed)


# --- Finite volumes ---


def _fv_skewed_triangles() -> tuple[np.ndarray, np.ndarray]:
    """Skewed right-diagonal triangulation of the unit square."""
    resolution = 4
    axis = np.linspace(0.0, 1.0, resolution + 1)
    vertices = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1).reshape((-1, 2))
    inside = np.all((vertices > 0.0) & (vertices < 1.0), axis=1)
    vertices[inside] += 0.04 * np.sin(5.0 * vertices[inside][:, ::-1] + 1.0)
    triangles = []
    for j in range(resolution):
        for i in range(resolution):
            corner = j * (resolution + 1) + i
            triangles.append((corner, corner + 1, corner + resolution + 2))
            triangles.append((corner, corner + resolution + 2, corner + resolution + 1))
    return vertices, np.asarray(triangles, dtype=np.int32)


def _fv_edge_midpoint_averages(
    vertices: np.ndarray, triangles: np.ndarray, function: Any
) -> Any:
    """Exact triangle averages of polynomials through degree two."""
    corners = vertices[triangles]
    midpoints = 0.5 * (corners + np.roll(corners, -1, axis=1))
    return jnp.asarray(np.mean(function(midpoints), axis=1)[:, None])


def _fv_quadratic_derivatives(points: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    x, y = points[..., 0], points[..., 1]
    return {
        (0, 0): 0.2 + x - 2.0 * y + 1.5 * x * x - 0.5 * x * y + 0.8 * y * y,
        (1, 0): 1.0 + 3.0 * x - 0.5 * y,
        (0, 1): -2.0 - 0.5 * x + 1.6 * y,
        (2, 0): np.full_like(x, 3.0),
        (1, 1): np.full_like(x, -0.5),
        (0, 2): np.full_like(x, 1.6),
    }


def test_fv_triangle_k_exact_query_reproduces_quadratics_and_their_derivatives() -> None:
    d = phx.discretization
    vertices, triangles = _fv_skewed_triangles()
    owner = d.TriangleFiniteVolumePlan(vertices, triangles).prepare()
    plan = d.TriangleKExactReconstructionPlan(d.PreparedTriangleQuadratic(owner))
    reconstruction = prepare_finite_volume_field_reconstruction(owner, plan)
    averages = _fv_edge_midpoint_averages(
        vertices, triangles, lambda points: _fv_quadratic_derivatives(points)[(0, 0)]
    )
    points = np.einsum("v,cvd->cd", np.asarray((0.2, 0.3, 0.5)), vertices[triangles])
    reference = _fv_quadratic_derivatives(points)
    for derivative, expected in reference.items():
        query = reconstruction.prepare_query(points, derivative=derivative)
        np.testing.assert_allclose(query.apply(averages)[:, 0], expected, atol=1e-9)
    assert reconstruction.maximum_derivative_order == 2
    assert reconstruction.coefficient_linear
    cotangent = jnp.asarray(np.random.default_rng(4).normal(size=query.output_shape))
    assert bool(query.duality_evidence(averages, cotangent).valid)
    trace = owner.prepare_side_trace(
        owner.cell_space.name,
        owner.integration_domain("exterior_facet"),
        rule=phx.discretization.FacetTraceRule(points=2),
        reconstruction=plan,
    )
    assert trace.descriptor.field_space_id == reconstruction.field_space_id


def test_fv_triangle_limited_muscl_query_is_nonlinear_with_a_local_linearization() -> (
    None
):
    d = phx.discretization
    vertices, triangles = _fv_skewed_triangles()
    owner = d.TriangleFiniteVolumePlan(vertices, triangles).prepare()
    wlsq = d.PreparedTriangleWLSQ(owner)
    state = _fv_edge_midpoint_averages(
        vertices,
        triangles,
        lambda points: np.tanh(8.0 * (points[..., 0] - 0.45)) + 0.3 * points[..., 1],
    )
    exterior = np.flatnonzero(np.asarray(owner.neighbor_cells) < 0)
    # Boundary edge midpoints lie in exactly one cell: the owner's limited face
    # state of the owner reconstruction is the queried value.
    points = np.asarray(owner.face_centers)[exterior]
    for limiter in ("unlimited", "venkatakrishnan"):
        plan = d.TriangleMUSCLReconstructionPlan(wlsq, limiter=limiter)
        reconstruction = prepare_finite_volume_field_reconstruction(owner, plan)
        query = reconstruction.prepare_query(points)
        left, _ = plan.reconstruct(state)
        np.testing.assert_allclose(
            query.apply(state), np.asarray(left)[exterior], atol=1e-12
        )
        assert reconstruction.coefficient_linear == (limiter == "unlimited")
    with pytest.raises(ValueError, match="nonlinear"):
        query.transpose(jnp.ones(query.output_shape))
    rng = np.random.default_rng(12)
    tangent = jnp.asarray(rng.normal(size=state.shape))
    step = 1e-6
    difference = (
        query.apply(state + step * tangent) - query.apply(state - step * tangent)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        query.linearize(state).jvp(tangent), difference, rtol=1e-5, atol=1e-6
    )
    # The limited plane is affine inside each cell, so its x-derivative query
    # equals the central difference of two value queries in the same cell.
    inside = np.einsum("v,cvd->cd", np.asarray((0.2, 0.3, 0.5)), vertices[triangles])
    shift = np.asarray((1e-4, 0.0))
    slope = reconstruction.prepare_query(inside, derivative=(1, 0)).apply(state)
    ahead = reconstruction.prepare_query(inside + shift).apply(state)
    behind = reconstruction.prepare_query(inside - shift).apply(state)
    np.testing.assert_allclose(slope, (ahead - behind) / 2e-4, rtol=1e-8, atol=1e-8)
