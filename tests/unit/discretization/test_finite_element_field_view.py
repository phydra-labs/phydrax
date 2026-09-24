#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import (
    DiscreteFieldFunctionView,
    FieldQueryStatus,
)
from phydrax.discretization.fem import (
    prepare_finite_element_field_reconstruction,
    prepare_finite_element_point_interpolation,
)


jax.config.update("jax_enable_x64", True)

_SQUARE = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_DIAGONAL_CELLS = ((0, 1, 2), (0, 2, 3))


def _discretization(degree, *, element=None, cell_kind="triangle", cells=_DIAGONAL_CELLS):
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


def _nodal(discretization, function):
    coordinates = np.asarray(discretization.dof_maps[0].dof_coordinates)
    return jnp.asarray(function(coordinates[:, 0], coordinates[:, 1]))


def _view(reconstruction, coefficients):
    domain = phx.domain.GeometryDomain(reconstruction.support_geometry, label="x")
    return DiscreteFieldFunctionView(reconstruction, coefficients, domain, variable="x")


_INTERIOR = jnp.asarray(((0.2, 0.1), (0.7, 0.3), (0.3, 0.8), (0.1, 0.6)))


def test_view_evaluation_matches_native_point_interpolation_values_and_gradients():
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


def test_quadratic_elements_reproduce_quadratics_and_exact_gradients_at_arbitrary_points():
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


def _kinked():
    # u = max(x - y, 0): gradient (1, -1) in cell 0 and (0, 0) in cell 1.
    discretization = _discretization(1)
    coefficients = _nodal(discretization, lambda x, y: np.maximum(x - y, 0.0))
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    return reconstruction, coefficients, _view(reconstruction, coefficients)


def test_c0_facet_gradient_needs_an_explicit_trace_side():
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


def test_trace_sides_are_validated_against_the_cells_containing_each_site():
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


def test_outside_support_queries_fail_closed_with_evidence():
    reconstruction, coefficients, view = _kinked()
    outside = jnp.asarray(((1.2, 0.5), (0.5, 0.5)))

    evidence = reconstruction.validity(outside)
    assert evidence.status.tolist() == [int(FieldQueryStatus.OUTSIDE_SUPPORT), 0]
    with pytest.raises(ValueError, match="OUTSIDE_SUPPORT"):
        view.as_domain_function().func(outside)
    with pytest.raises(Exception, match="invalid"):
        eqx.filter_jit(view.as_domain_function().func)(outside).block_until_ready()


def test_coefficient_adjoint_is_the_exact_transpose_scatter():
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


def test_views_require_an_equivalent_explicit_geometry_domain():
    reconstruction, coefficients, _ = _kinked()
    square = phx.geometry.Rectangle((0.5, 0.5), (1.0, 1.0)).compile()

    with pytest.raises(TypeError, match="GeometryDomain"):
        DiscreteFieldFunctionView(
            reconstruction, coefficients, phx.domain.Interval1d(0.0, 1.0), variable="x"
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


def test_non_simplicial_cells_need_an_explicit_inverse_provider():
    quadrilateral = _discretization(1, cell_kind="quadrilateral", cells=((0, 1, 2, 3),))

    with pytest.raises(ValueError, match="explicit AbstractCellLocator"):
        prepare_finite_element_field_reconstruction(quadrilateral, "u")
    # Own-point evaluation from native tabulation remains available.
    native = prepare_finite_element_point_interpolation(
        quadrilateral, "u", "cells", jnp.asarray((0,)), jnp.asarray(((0.25, 0.5),))
    )
    coefficients = _nodal(quadrilateral, lambda x, y: x + 2.0 * y)
    np.testing.assert_allclose(native.interpolate(coefficients), (1.25,), atol=1e-12)


def test_discontinuous_fields_need_a_side_for_facet_values():
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


class _TemperatureModel(phx.AbstractArrayModel):
    weight: jax.Array
    output_port: phx.ValuePort = eqx.field(static=True)
    input_port: phx.ValuePort = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, input_port, output_port):
        self.weight = jnp.asarray((0.5, -0.25))
        self.input_port = input_port
        self.output_port = output_port
        self.in_size = 2
        self.out_size = "scalar"

    def __call__(self, x, /, *, key=None):
        return jnp.tanh(self.weight @ x)

    def model_ports(self):
        return phx.ModelPorts(inputs=(self.input_port,), outputs=(self.output_port,))


def _temperature_port(dimension):
    return phx.ValuePort(
        "temperature",
        event_shape=(),
        component_ids=("T",),
        representation="scalar-field",
        dimensions=(dimension,),
    )


def _bound_model(domain, output_port):
    x_port = domain.value_port("x")
    model = _TemperatureModel(x_port, output_port)
    return domain.Model(
        "x", port_mapping=phx.PortMapping(inputs=[(x_port.port_id, x_port.port_id)])
    )(model)


def test_fe_plus_network_requires_compatible_support_units_and_ports():
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
