#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import (
    CellLocationStatus,
    DiscreteFieldFunctionView,
    FieldQueryStatus,
)
from phydrax.discretization.fem import (
    FiniteElementFieldReconstructionKernel,
    prepare_finite_element_field_reconstruction,
    prepare_finite_element_point_interpolation,
    PreparedFiniteElementCellMap,
)


jax.config.update("jax_enable_x64", True)

_SQUARE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_DIAGONAL_CELLS = ((0, 1, 2), (0, 2, 3))


def _discretization(
    degree: Any,
    *,
    element: Any = None,
    cell_kind: Any = "triangle",
    cells: Any = _DIAGONAL_CELLS,
) -> Any:
    mesh = phx.discretization.CellMesh(
        jnp.asarray(_SQUARE),
        (phx.discretization.CellBlock("cells", cell_kind, jnp.asarray(cells)),),
    )
    spec = (
        phx.discretization.lagrange_element(cell_kind, degree)
        if element is None
        else element
    )
    return phx.discretization.FiniteElementPlan(
        mesh, phx.discretization.FiniteElementFieldSpec("u", spec)
    ).prepare()


def _nodal(discretization: Any, function: Any) -> Any:
    coordinates = np.asarray(discretization.dof_maps[0].dof_coordinates)
    return jnp.asarray(function(coordinates[:, 0], coordinates[:, 1]))


def _view(reconstruction: Any, coefficients: Any) -> Any:
    domain = phx.domain.GeometryDomain(reconstruction.support_geometry, label="x")
    return DiscreteFieldFunctionView(reconstruction, coefficients, domain, variable="x")


_INTERIOR = jnp.asarray(((0.2, 0.1), (0.7, 0.3), (0.3, 0.8), (0.1, 0.6)))


def test_finite_element_field_view_scenario_1() -> None:
    discretization = _discretization(2)
    coefficients = _nodal(discretization, lambda x, y: np.sin(3.0 * x) + x * y**2)
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    cells = jnp.asarray((0, 0, 1, 1))
    reference = jnp.asarray(((0.2, 0.1), (0.6, 0.3), (0.1, 0.7), (0.3, 0.3)))
    native = prepare_finite_element_point_interpolation(
        discretization, "u", "cells", cells, reference
    )
    native_dx = prepare_finite_element_point_interpolation(
        discretization, "u", "cells", cells, reference, derivative_axis=0
    )
    points = native.reference_positions

    values = reconstruction.evaluate(coefficients, points)
    gradient = reconstruction.derivative(coefficients, points, (1, 0))

    assert bool(jnp.all(values.valid)) and bool(jnp.all(gradient.valid))
    np.testing.assert_allclose(
        values.values, native.interpolate(coefficients), atol=1e-12
    )
    np.testing.assert_allclose(
        gradient.values, native_dx.interpolate(coefficients), atol=1e-12
    )
    field = _view(reconstruction, coefficients).as_domain_function()
    np.testing.assert_allclose(
        jax.vmap(field.func)(points), native.interpolate(coefficients), atol=1e-12
    )
    discretization = _discretization(2)
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    coefficients = _nodal(discretization, lambda x, y: 1.0 + 2.0 * x - y + x * y + y**2)
    field = _view(reconstruction, coefficients).as_domain_function()
    gradient = phx.operators.grad(field, var="x")
    x, y = _INTERIOR[:, 0], _INTERIOR[:, 1]

    assert reconstruction.regularity == phx.DerivativeRegularity.piecewise_polynomial(
        continuity=0, degree_bound=2
    )
    np.testing.assert_allclose(
        jax.vmap(field.func)(_INTERIOR), 1.0 + 2.0 * x - y + x * y + y**2, atol=1e-12
    )
    np.testing.assert_allclose(
        jax.vmap(gradient.func)(_INTERIOR),
        jnp.stack((2.0 + y, -1.0 + x + 2.0 * y), axis=-1),
        atol=1e-11,
    )
    # Native tabulation provides first derivatives only; second orders refuse.
    with pytest.raises(ValueError, match="maximum_derivative_order=1"):
        phx.operators.partial_n(field, var="x", axis=0, order=2)
    reconstruction, coefficients, view = _kinked()
    gradient = phx.operators.grad(view.as_domain_function(), var="x")
    facet = jnp.asarray(((0.5, 0.5), (0.25, 0.25)))

    evidence = reconstruction.validity(facet, derivative=(1, 0))
    assert evidence.status.tolist() == [int(FieldQueryStatus.SIDE_REQUIRED)] * 2
    assert evidence.support_count.tolist() == [2, 2]
    # Values are single-valued on the facet; gradients are not.
    np.testing.assert_allclose(reconstruction.evaluate(coefficients, facet).values, 0.0)
    with pytest.raises(ValueError, match="SIDE_REQUIRED"):
        gradient.func(facet[0])
    np.testing.assert_allclose(gradient.func(jnp.asarray((0.7, 0.2))), (1.0, -1.0))

    owner = view.trace(facet, side="owner", cell_ids=jnp.asarray((0, 0)))
    neighbor = view.trace(facet, side="neighbor", cell_ids=jnp.asarray((1, 1)))
    average = view.trace(facet, side="average")
    for trace, expected in (
        (owner, (1.0, -1.0)),
        (neighbor, (0.0, 0.0)),
        (average, (0.5, -0.5)),
    ):
        traced = phx.operators.grad(trace, var="x")
        np.testing.assert_allclose(
            jax.vmap(traced.func)(facet), [expected] * 2, atol=1e-12
        )


def _kinked() -> Any:
    # u = max(x - y, 0): gradient (1, -1) in cell 0 and (0, 0) in cell 1.
    discretization = _discretization(1)
    coefficients = _nodal(discretization, lambda x, y: np.maximum(x - y, 0.0))
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    return reconstruction, coefficients, _view(reconstruction, coefficients)


def test_finite_element_field_view_scenario_2() -> None:
    _, _, view = _kinked()
    facet = jnp.asarray(((0.5, 0.5),))

    with pytest.raises(ValueError, match="require explicit cell_ids"):
        view.trace(facet, side="owner")
    with pytest.raises(ValueError, match="containing each trace site"):
        view.trace(jnp.asarray(((0.7, 0.2),)), side="owner", cell_ids=jnp.asarray((1,)))
    with pytest.raises(ValueError, match="in the FE support"):
        view.trace(jnp.asarray(((1.5, 0.5),)), side="average")
    # A boundary site lies in one cell: its owner side is derived.
    boundary = view.trace(jnp.asarray(((0.5, 0.0),)), side="owner")
    np.testing.assert_allclose(boundary.func(jnp.asarray((0.5, 0.0))), 0.5)
    reconstruction, coefficients, view = _kinked()
    outside = jnp.asarray(((1.2, 0.5), (0.5, 0.5)))

    evidence = reconstruction.validity(outside)
    assert evidence.status.tolist() == [int(FieldQueryStatus.OUTSIDE_SUPPORT), 0]
    with pytest.raises(ValueError, match="OUTSIDE_SUPPORT"):
        view.as_domain_function().func(outside)
    with pytest.raises(Exception, match="invalid"):
        eqx.filter_jit(view.as_domain_function().func)(outside).block_until_ready()
    discretization = _discretization(2)
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    coefficients = _nodal(discretization, lambda x, y: np.cos(x) * y)
    cotangent = jnp.asarray((1.0, -2.0, 0.5, 3.0))

    evidence = reconstruction.duality_evidence(coefficients, _INTERIOR, cotangent)
    assert bool(evidence.valid)
    _, pullback = jax.vjp(
        lambda c: reconstruction.evaluate(c, _INTERIOR).values, coefficients
    )
    np.testing.assert_allclose(
        pullback(cotangent)[0], reconstruction.transpose(_INTERIOR, cotangent), atol=1e-13
    )
    # Coordinate derivatives of a view never differentiate the reconstruction data.
    field = _view(reconstruction, coefficients).as_domain_function()
    with pytest.raises(ValueError, match="FIXED"):
        eqx.filter_grad(lambda tree: jnp.sum(tree.func(_INTERIOR[0])))(field)


def test_finite_element_field_view_scenario_3() -> None:
    reconstruction, coefficients, _ = _kinked()
    square = phx.geometry.Rectangle((0.5, 0.5), (1.0, 1.0)).compile()

    with pytest.raises(TypeError, match="GeometryDomain"):
        DiscreteFieldFunctionView(
            reconstruction,
            coefficients,
            # ty: ignore[invalid-argument-type]
            phx.domain.Interval1d(0.0, 1.0),
            variable="x",
        )
    with pytest.raises(ValueError, match="not equivalent"):
        DiscreteFieldFunctionView(
            reconstruction,
            coefficients,
            phx.domain.GeometryDomain(square, label="x"),
            variable="x",
        )
    # An explicit analytic support is admitted once the mesh evidences coverage.
    explicit = prepare_finite_element_field_reconstruction(
        _discretization(1), "u", support_geometry=square
    )
    view = DiscreteFieldFunctionView(
        explicit, coefficients, phx.domain.GeometryDomain(square, label="x"), variable="x"
    )
    np.testing.assert_allclose(
        view.as_domain_function().func(jnp.asarray((0.7, 0.2))), 0.5
    )
    with pytest.raises(ValueError, match="does not cover"):
        prepare_finite_element_field_reconstruction(
            _discretization(1),
            "u",
            support_geometry=phx.geometry.Rectangle((1.0, 0.5), (2.0, 1.0)).compile(),
        )
    quadrilateral = _discretization(1, cell_kind="quadrilateral", cells=((0, 1, 2, 3),))

    # Own-point evaluation from native tabulation remains available.
    native = prepare_finite_element_point_interpolation(
        quadrilateral, "u", "cells", jnp.asarray((0,)), jnp.asarray(((0.25, 0.5),))
    )
    coefficients = _nodal(quadrilateral, lambda x, y: x + 2.0 * y)
    np.testing.assert_allclose(native.interpolate(coefficients), (1.25,), atol=1e-12)
    mapped = prepare_finite_element_field_reconstruction(quadrilateral, "u")
    np.testing.assert_allclose(
        mapped.evaluate(coefficients, native.reference_positions).values,
        (1.25,),
        atol=1e-12,
    )
    dual = jnp.asarray((2.5,))
    assert bool(
        mapped.duality_evidence(coefficients, native.reference_positions, dual).valid
    )
    discretization = _discretization(
        0, element=phx.discretization.discontinuous_element("triangle", 0)
    )
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    coefficients = jnp.asarray((1.0, 3.0))
    view = _view(reconstruction, coefficients)
    facet = jnp.asarray(((0.5, 0.5),))

    assert reconstruction.maximum_derivative_order == 0
    assert reconstruction.validity(facet).status.tolist() == [
        int(FieldQueryStatus.SIDE_REQUIRED)
    ]
    with pytest.raises(ValueError, match="degenerate"):
        phx.operators.grad(view.as_domain_function(), var="x")
    average = view.trace(facet, side="average")
    np.testing.assert_allclose(average.func(facet[0]), 2.0)


def test_default_location_finds_generic_interior_points_of_small_meshes() -> None:
    # 3x3 right-triangle mesh of the unit square: 18 cells span several BVH leaves
    # whose boxes overlap, while only a few cell boxes contain any given point.
    count = 3
    grid = np.linspace(0.0, 1.0, count + 1)
    vertices = np.stack(np.meshgrid(grid, grid, indexing="xy"), axis=-1).reshape(-1, 2)
    corner = np.arange((count + 1) ** 2).reshape(count + 1, count + 1)[:-1, :-1].ravel()
    a, b, c, d = corner, corner + 1, corner + count + 2, corner + count + 1
    cells = np.concatenate((np.stack((a, b, c), -1), np.stack((a, c, d), -1)))
    mesh = phx.discretization.CellMesh(
        jnp.asarray(vertices),
        (phx.discretization.CellBlock("cells", "triangle", jnp.asarray(cells)),),
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    points = jnp.asarray(
        np.concatenate(
            (
                ((0.41, 0.23),),
                np.random.default_rng(7).uniform(0.02, 0.98, size=(63, 2)),
            )
        )
    )

    assert (
        reconstruction.validity(points).status.tolist()
        == [int(FieldQueryStatus.VALID)] * points.shape[0]
    )
    kernel = reconstruction.kernel
    assert isinstance(kernel, FiniteElementFieldReconstructionKernel)
    located = kernel.locator.locate(points)
    cell_map = kernel.locator.cell_map
    assert isinstance(cell_map, PreparedFiniteElementCellMap)
    # Barycentric weights of the located cell's vertices must rebuild each point.
    corners = np.asarray(kernel.locator.coordinates)[
        np.asarray(cell_map.coordinate_dofs)[np.asarray(located.cell_ids)]
    ]
    barycentric = np.asarray(located.barycentric)
    assert np.all(barycentric >= -1e-12)
    np.testing.assert_allclose(
        np.einsum("pv,pvd->pd", barycentric, corners), points, atol=1e-12
    )
    coefficients = _nodal(discretization, lambda x, y: 1.0 + 2.0 * x - 3.0 * y)
    np.testing.assert_allclose(
        reconstruction.evaluate(coefficients, points).values,
        1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1],
        atol=1e-12,
    )


def test_embedded_triangle_field_preserves_physical_chart_and_transpose() -> None:
    # Parallel disconnected sheets have overlapping AABBs but disjoint images.
    vertices = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 2.0),
            (0.0, 0.0, 0.4),
            (1.0, 0.0, 1.4),
            (0.0, 1.0, 2.4),
        )
    )
    mesh = phx.discretization.CellMesh(
        vertices,
        (
            phx.discretization.CellBlock(
                "sheet", "triangle", jnp.asarray(((0, 1, 2), (3, 4, 5)))
            ),
        ),
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.lagrange_element("triangle", 1),
        ),
    ).prepare()
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    points = jnp.asarray(((0.2, 0.3, 0.8), (0.5, 0.1, 0.7)))
    coefficients = 2.0 + discretization.dof_maps[0].dof_coordinates @ jnp.asarray(
        (1.0, 2.0, 3.0)
    )
    evaluated = reconstruction.evaluate(coefficients, points)
    np.testing.assert_allclose(
        evaluated.values,
        2.0 + points @ jnp.asarray((1.0, 2.0, 3.0)),
        atol=1e-12,
    )
    assert bool(jnp.all(evaluated.valid))
    normal = jnp.asarray((-1.0, -2.0, 1.0))
    off_sheet = reconstruction.evaluate(coefficients, points + 0.01 * normal)
    assert not bool(jnp.any(off_sheet.valid))
    assert bool(
        jnp.all(off_sheet.evidence.status == int(FieldQueryStatus.OUTSIDE_SUPPORT))
    )
    cotangent = jnp.asarray((1.3, -0.7))
    np.testing.assert_allclose(
        jnp.vdot(evaluated.values, cotangent),
        jnp.vdot(coefficients, reconstruction.transpose(points, cotangent)),
        atol=1e-12,
    )
    function = _view(reconstruction, coefficients).as_domain_function()
    np.testing.assert_allclose(
        jax.vmap(function.func)(points), evaluated.values, atol=1e-12
    )
    np.testing.assert_allclose(
        jax.vmap(phx.operators.grad(function, var="x").func)(points),
        jnp.broadcast_to(jnp.asarray((2.0 / 3.0, 4.0 / 3.0, 10.0 / 3.0)), (2, 3)),
        atol=1e-12,
    )


def test_location_finds_points_where_curved_cells_bulge_past_their_nodes() -> None:
    # P2 cell 0 is the image of (xi, eta) -> (xi, eta + xi^2 - 0.4 xi): its
    # bottom edge dips to y = -0.04 below every coordinate node (all y >= 0).
    # Cell 1 is a straight P2 triangle sharing the BVH leaf.
    def curve(xi: Any, eta: Any) -> Any:
        return np.stack((xi, eta + xi**2 - 0.4 * xi), axis=-1)

    nodes = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5))
    )
    coordinates = np.concatenate(
        (curve(nodes[:, 0], nodes[:, 1]), nodes + np.asarray((2.0, 0.0)))
    )
    vertices = coordinates[[0, 1, 2, 6, 7, 8]]
    mesh = phx.discretization.CellMesh.from_triangles(
        jnp.asarray(vertices), jnp.asarray(((0, 1, 2), (3, 4, 5)), dtype=jnp.int32)
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
        coordinate_spec=phx.discretization.CellGeometrySpec(
            {"triangles": phx.discretization.lagrange_element("triangle", 2)},
            {"triangles": jnp.arange(12, dtype=jnp.int32).reshape(2, 6)},
            jnp.asarray(coordinates),
        ),
    ).prepare()
    locator = phx.discretization.PreparedSimplicialCellLocator(
        phx.discretization.prepare_finite_element_cell_map(discretization, 0),
        discretization.default_runtime.coordinates,
        phx.discretization.SimplicialLocationPolicy(2, 32, 1),
    )
    reference = np.asarray(((0.2, 0.01), (0.3, 0.005), (0.25, 0.3), (0.6, 0.02)))
    straight = np.asarray(((2.3, 0.2),))
    below_curve = np.asarray(((0.5, 0.0),))
    points = np.concatenate((curve(reference[:, 0], reference[:, 1]), straight))
    assert np.all(points[:2, 1] < 0.0)

    located = locator.locate(jnp.asarray(points))
    outside = locator.locate(jnp.asarray(below_curve))

    assert located.status.tolist() == [int(CellLocationStatus.LOCATED)] * 5
    assert located.cell_ids.tolist() == [0, 0, 0, 0, 1]
    np.testing.assert_allclose(
        np.asarray(located.reference_coordinates),
        np.concatenate((reference, straight - np.asarray((2.0, 0.0)))),
        atol=1e-10,
    )
    assert outside.status.tolist() == [int(CellLocationStatus.OUTSIDE)]


class _TemperatureModel(phx.AbstractArrayModel):
    weight: jax.Array
    output_port: phx.ValuePort = eqx.field(static=True)
    input_port: phx.ValuePort = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, input_port: Any, output_port: Any) -> None:
        self.weight = jnp.asarray((0.5, -0.25))
        self.input_port = input_port
        self.output_port = output_port
        self.in_size = 2
        self.out_size = "scalar"

    def __call__(self, x: Any, /, *, key: Any = None) -> Any:
        return jnp.tanh(self.weight @ x)

    def model_ports(self) -> Any:
        return phx.ModelPorts(inputs=(self.input_port,), outputs=(self.output_port,))


def _temperature_port(dimension: Any) -> Any:
    return phx.ValuePort(
        "temperature",
        event_shape=(),
        component_ids=("T",),
        representation="scalar-field",
        dimensions=(dimension,),
    )


def _bound_model(domain: Any, output_port: Any) -> Any:
    x_port = domain.value_port("x")
    model = _TemperatureModel(x_port, output_port)
    return domain.Model(
        "x", port_mapping=phx.PortMapping(inputs=[(x_port.port_id, x_port.port_id)])
    )(model)


def test_fe_plus_network_requires_compatible_support_units_and_ports() -> None:
    kelvin = phx.units.DimensionSignature({"temperature": 1})
    meter = phx.units.DimensionSignature({"length": 1})
    discretization = _discretization(1)
    reconstruction = prepare_finite_element_field_reconstruction(
        discretization, "u", value_port=_temperature_port(kelvin)
    )
    coefficients = _nodal(discretization, lambda x, y: x + y)
    view = _view(reconstruction, coefficients)
    u_fe = view.as_domain_function()

    u_nn = _bound_model(view.domain, _temperature_port(kelvin))
    total = u_fe + u_nn
    point = jnp.asarray((0.7, 0.2))
    np.testing.assert_allclose(total.func(point), 0.9 + jnp.tanh(0.35 - 0.05), atol=1e-12)
    np.testing.assert_allclose(
        phx.operators.grad(total, var="x").func(point),
        jnp.asarray((1.0, 1.0)) + (1.0 - jnp.tanh(0.3) ** 2) * jnp.asarray((0.5, -0.25)),
        atol=1e-12,
    )
    assert (u_fe * 2.0 - 1.0).func(point) == pytest.approx(0.8)

    with pytest.raises(ValueError, match="units"):
        u_fe + _bound_model(view.domain, _temperature_port(meter))
    unported = view.domain.Model("x", binding=phx.domain.ModelBinding())(
        phx.nn.models.MLP(
            in_size=2, out_size="scalar", width_size=4, depth=1, key=jax.random.key(0)
        )
    )
    with pytest.raises(ValueError, match="declare a value port"):
        u_fe + unported
    other_support = phx.domain.GeometryDomain(
        phx.geometry.Rectangle((1.0, 0.5), (2.0, 1.0)).compile(), label="x"
    )
    with pytest.raises(ValueError, match="Label collision"):
        u_fe + _bound_model(other_support, _temperature_port(kelvin))


def test_fe_view_minus_network_on_a_box_composes_under_jit() -> None:
    kelvin = phx.units.DimensionSignature({"temperature": 1})
    discretization = _discretization(1)
    reconstruction = prepare_finite_element_field_reconstruction(
        discretization, "u", value_port=_temperature_port(kelvin)
    )
    coefficients = _nodal(discretization, lambda x, y: x + y)
    box = phx.domain.HyperRectangle(np.zeros(2), np.ones(2), label="x")
    target = phx.domain.DomainFunction(
        domain=box,
        deps=("x",),
        func=_view(reconstruction, coefficients).as_domain_function().func,
    )
    prediction = _bound_model(box, _temperature_port(kelvin))
    point = jnp.asarray((0.7, 0.2))

    @eqx.filter_jit
    def residual(prediction: Any, target: Any, point: Any) -> Any:
        return (prediction - target).func(point)

    np.testing.assert_allclose(
        residual(prediction, target, point), jnp.tanh(0.35 - 0.05) - 0.9, atol=1e-12
    )
