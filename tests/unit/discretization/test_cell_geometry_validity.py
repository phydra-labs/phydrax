#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.discretization._cell_geometry import (
    CellGeometrySpec,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    SplineCellGeometryElement,
)
from phydrax.discretization._coordinate_enclosure import (
    CoordinateSourceBank,
    evaluate_expression,
    Expression,
    normalize_pyramid_reference_derivative,
    normalize_pyramid_reference_expression,
    RationalPolynomial,
)
from phydrax.discretization._exact_plc_geometry import ExactPlcCellGeometrySource


D = phx.discretization
G = phx.geometry
Status = D.CellValidityStatus


def _polygon_status(points: Any, policy: Any = None) -> Any:
    points = np.asarray(points, dtype=np.float64)
    loop = np.arange(points.shape[0], dtype=np.int32)[None]
    mesh = D.CellMesh(points, (D.CellBlock("cells", "polygon", loop),))
    certificate = D.certify_cell_geometry_validity(mesh, policy=policy)
    return Status(int(certificate.status[0]))


def _p3_triangle(delta: float) -> tuple[Any, Any]:
    mesh = D.CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        (D.CellBlock("cells", "triangle", np.asarray(((0, 1, 2),))),),
    )
    element = D.fem.lagrange_element("triangle", 3)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    u, v = nodes.T
    coordinates = np.column_stack((((u - 0.5) ** 3) / 3.0 + delta * u, v))
    geometry = D.CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(nodes.shape[0], dtype=np.int64)[None, :]},
        coordinates,
    )
    return mesh, geometry


def _scalar_chart_geometry(chart: Expression) -> CellGeometrySpec:
    source = D.fem.lagrange_element("interval", 1)
    element = (
        RationalComposedCellGeometryElement(source, "interval", (chart,))
        if isinstance(chart, RationalPolynomial)
        else PolynomialComposedCellGeometryElement(source, "interval", (chart,))
    )
    return CellGeometrySpec(
        {"cells": element},
        {"cells": np.asarray(((0, 1),), dtype=np.int64)},
        np.asarray(((0.0,), (1.0,)), dtype=np.float64),
    )


def _pyramid_chart_geometry(
    chart: tuple[RationalPolynomial, ...],
) -> tuple[CellGeometrySpec, RationalComposedCellGeometryElement]:
    source = D.fem.lagrange_element("hexahedron", 1)
    element = RationalComposedCellGeometryElement(source, "pyramid", chart)
    coordinates = np.asarray(source.reference_nodes, dtype=np.float64)
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(coordinates.shape[0], dtype=np.int64)[None, :]},
        coordinates,
    )
    return geometry, element


@dataclass
class _MutableCoordinateProvider:
    bank: CoordinateSourceBank
    source_id: str = "source-bank-fixture"
    source_revision: str = "source-bank-state"
    source_map_id: str = "source-bank-provider"

    def source_coordinates(self) -> CoordinateSourceBank:
        return self.bank


def test_exact_coordinate_bank_is_an_immutable_snapshot() -> None:
    mesh = D.CellMesh(
        np.asarray(((0.0,), (1.0,)), dtype=np.float64),
        (D.CellBlock("cells", "interval", np.asarray(((0, 1),), dtype=np.int32)),),
    )
    original = ((Fraction(0),), (Fraction(1),))
    provider = _MutableCoordinateProvider(original)
    geometry = CellGeometrySpec(
        {"cells": D.fem.lagrange_element("interval", 1)},
        {"cells": np.asarray(((0, 1),), dtype=np.int64)},
        mesh.coordinates,
        source_mesh=mesh,
        source_map=provider,
    )
    provider.bank = ((Fraction(2),), (Fraction(3),))
    assert geometry.source_coordinates() == original
    assert geometry.source_coordinates(np.asarray((1,), dtype=np.int64)) == (original[1],)


def test_exact_coordinate_bank_rejects_inconsistent_execution_coefficients() -> None:
    mesh = D.CellMesh(
        np.asarray(((0.0,), (1.0,)), dtype=np.float64),
        (D.CellBlock("cells", "interval", np.asarray(((0, 1),), dtype=np.int32)),),
    )
    provider = _MutableCoordinateProvider(((Fraction(0),), (Fraction(1),)))
    with pytest.raises(ValueError):
        CellGeometrySpec(
            {"cells": D.fem.lagrange_element("interval", 1)},
            {"cells": np.asarray(((0, 1),), dtype=np.int64)},
            np.asarray(((0.0,), (-1.0,)), dtype=np.float64),
            source_mesh=mesh,
            source_map=provider,
        )


def test_exact_positive_interval_rejects_collapsed_execution_map() -> None:
    exact = ((Fraction(1),), (Fraction(1) + Fraction(1, 2**54),))
    mesh = D.CellMesh(
        np.asarray(((1.0,), (1.0,)), dtype=np.float64),
        (D.CellBlock("cells", "interval", np.asarray(((0, 1),), dtype=np.int32)),),
    )
    source = ExactPlcCellGeometrySource(
        "subfloat-interval",
        "subfloat-interval-state",
        np.asarray((0, 1), dtype=np.int64),
        exact,
        vertex_source_entity_ids=("left", "right"),
        vertex_occurrence_paths=((), ()),
    )
    geometry = CellGeometrySpec(
        {"cells": D.fem.lagrange_element("interval", 1)},
        {"cells": np.asarray(((0, 1),), dtype=np.int64)},
        mesh.coordinates,
        source_mesh=mesh,
        source_map=source,
    )
    certificate = D.certify_cell_geometry_validity(geometry, mesh=mesh)
    assert int(certificate.source_status[0]) == Status.CERTIFIED_VALID
    assert int(certificate.execution_status[0]) == Status.INVALID
    assert int(certificate.status[0]) == Status.INVALID
    assert not certificate.all_certified
    assert float(certificate.source_determinant_lower[0]) > 0.0
    assert float(certificate.determinant_lower[0]) == 0.0
    assert float(certificate.maximum_coefficient_rounding_error) >= float(
        Fraction(1, 2**54)
    )
    assert float(certificate.maximum_element_coefficient_rounding_error) == 0.0
    certificate.require_current(geometry, mesh=mesh)


def test_spline_execution_certificate_uses_binary64_extraction_shadow() -> None:
    large = 2**53
    extraction = (
        (Fraction(-large), Fraction(-large), Fraction(-large - 1), Fraction(-large - 1)),
        (Fraction(large), Fraction(large), Fraction(large + 1), Fraction(large + 1)),
        (Fraction(1), Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0), Fraction(1)),
    )
    element = SplineCellGeometryElement(
        (0, 1, 2, 3),
        (Fraction(0), Fraction(1), Fraction(0), Fraction(1)),
        extraction,
        np.ones((4,), dtype=np.float64),
        "collapsed-execution-spline",
        "collapsed-execution-spline-state",
        degrees=(1, 1),
    )
    coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 1.0)),
        dtype=np.float64,
    )
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(4, dtype=np.int32)[None, :]},
        coordinates,
    )

    values, _ = element.tabulate(np.asarray(((0.25, 0.5),)))
    execution_map = np.asarray(values) @ coordinates
    assert execution_map[0, 0] == float(large)
    assert execution_map[0, 1] == 0.5

    certificate = D.certify_cell_geometry_validity(geometry)
    assert int(certificate.source_status[0]) == Status.CERTIFIED_VALID
    assert int(certificate.execution_status[0]) == Status.INVALID
    assert int(certificate.status[0]) == Status.INVALID
    assert float(certificate.maximum_coefficient_rounding_error) == 0.0
    assert float(certificate.maximum_element_coefficient_rounding_error) >= 1.0
    certificate.require_current(geometry)


def test_spline_weight_shadow_rounding_is_dimensionless_and_separate() -> None:
    extraction = tuple(
        tuple(Fraction(int(row == column)) for column in range(4)) for row in range(4)
    )
    exact_weights = (
        Fraction(1),
        Fraction(1) + Fraction(1, 2**54),
        Fraction(1),
        Fraction(1),
    )
    element = SplineCellGeometryElement(
        (0, 1, 2, 3),
        (Fraction(0), Fraction(1), Fraction(0), Fraction(1)),
        extraction,
        np.ones((4,), dtype=np.float64),
        "rounded-spline-weight",
        "rounded-spline-weight-state",
        degrees=(1, 1),
        exact_weights=exact_weights,
    )
    coordinates = np.asarray(
        ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)),
        dtype=np.float64,
    )
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(4, dtype=np.int32)[None, :]},
        coordinates,
    )

    certificate = D.certify_cell_geometry_validity(geometry)
    assert int(certificate.source_status[0]) == Status.CERTIFIED_VALID
    assert int(certificate.execution_status[0]) == Status.CERTIFIED_VALID
    assert float(certificate.maximum_coefficient_rounding_error) == 0.0
    assert float(certificate.maximum_element_coefficient_rounding_error) > 0.0


def test_chart_coefficient_shadows_are_reported_separately_from_coordinates() -> None:
    polynomial = _scalar_chart_geometry({(1,): Fraction(1), (0,): Fraction(1, 3)})
    polynomial_certificate = D.certify_cell_geometry_validity(polynomial)
    assert int(polynomial_certificate.source_status[0]) == Status.CERTIFIED_VALID
    assert int(polynomial_certificate.execution_status[0]) == Status.CERTIFIED_VALID
    assert float(polynomial_certificate.maximum_coefficient_rounding_error) == 0.0
    assert float(polynomial_certificate.maximum_element_coefficient_rounding_error) > 0.0

    rational = _scalar_chart_geometry(
        RationalPolynomial(
            {(1,): Fraction(1)},
            {(0,): Fraction(1), (1,): Fraction(1) + Fraction(1, 2**54)},
        )
    )
    rational_certificate = D.certify_cell_geometry_validity(rational)
    assert int(rational_certificate.source_status[0]) == Status.CERTIFIED_VALID
    assert int(rational_certificate.execution_status[0]) == Status.CERTIFIED_VALID
    assert float(rational_certificate.maximum_coefficient_rounding_error) == 0.0
    assert float(rational_certificate.maximum_element_coefficient_rounding_error) > 0.0


def test_transverse_constant_singularity_terminates_without_certification() -> None:
    source = D.fem.lagrange_element("hexahedron", 1)
    element = PolynomialComposedCellGeometryElement(
        source,
        "hexahedron",
        (
            {
                (3, 0, 0): Fraction(1, 3),
                (2, 0, 0): Fraction(-1, 3),
                (1, 0, 0): Fraction(1, 9),
            },
            {(0, 1, 0): Fraction(1)},
            {(0, 0, 1): Fraction(1)},
        ),
    )
    points = np.asarray(source.reference_nodes, dtype=np.float64)
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(points.shape[0], dtype=np.int64)[None, :]},
        points,
    )
    certificate = D.certify_cell_geometry_validity(
        geometry, policy=D.CellValidityPolicy(relative_determinant_floor=0.0)
    )
    assert int(certificate.status[0]) in (Status.INVALID, Status.UNRESOLVED)
    assert not certificate.all_certified
    assert float(certificate.determinant_lower[0]) <= 0.0
    assert float(certificate.determinant_upper[0]) >= 0.0


@pytest.mark.parametrize(
    ("chart", "expected_status", "expected_bounds"),
    (
        pytest.param(
            {(1,): Fraction(1), (2,): Fraction(1, 4)},
            Status.CERTIFIED_VALID,
            (1.0, 1.5),
            id="polynomial",
        ),
        pytest.param(
            RationalPolynomial(
                {(1,): Fraction(1)}, {(0,): Fraction(1), (1,): Fraction(1)}
            ),
            Status.CERTIFIED_VALID,
            (0.25, 1.0),
            id="rational",
        ),
        pytest.param(
            RationalPolynomial(
                {(1,): Fraction(1)}, {(0,): Fraction(1), (1,): Fraction(-2)}
            ),
            Status.UNRESOLVED,
            None,
            id="interior-pole",
        ),
    ),
)
def test_source_chart_validity_replays_whole_map_proof(
    chart: Expression,
    expected_status: D.CellValidityStatus,
    expected_bounds: tuple[float, float] | None,
) -> None:
    geometry = _scalar_chart_geometry(chart)
    certificate = D.certify_cell_geometry_validity(geometry)
    assert int(certificate.status[0]) == expected_status
    certificate.require_current(geometry)
    if expected_bounds is not None:
        assert 0.0 < float(certificate.determinant_lower[0]) <= expected_bounds[0]
        assert float(certificate.determinant_upper[0]) >= expected_bounds[1]


def test_source_poles_do_not_alias_a_positive_binary64_shadow() -> None:
    linear = Fraction(1) + Fraction(1, 2**53)
    constant = Fraction(1, 4) + Fraction(1, 2**54)
    denominator: dict[tuple[int, ...], Fraction] = {
        (2,): Fraction(1),
        (1,): -linear,
        (0,): constant,
    }
    assert evaluate_expression(denominator, (Fraction(1, 2),)) == 0
    assert evaluate_expression(denominator, (Fraction(1, 2) + Fraction(1, 2**53),)) == 0
    shadow_linear = Fraction(float(linear))
    shadow_constant = Fraction(float(constant))
    assert shadow_linear == Fraction(1)
    assert Fraction(1, 4) - shadow_linear / 2 + shadow_constant > 0
    chart = RationalPolynomial(
        {(3,): Fraction(1), (2,): -linear, (1,): constant},
        denominator,
    )

    geometry = _scalar_chart_geometry(chart)
    certificate = D.certify_cell_geometry_validity(geometry)

    assert int(certificate.source_status[0]) == Status.UNRESOLVED
    assert int(certificate.execution_status[0]) == Status.CERTIFIED_VALID
    assert int(certificate.status[0]) == Status.UNRESOLVED
    assert float(certificate.maximum_element_coefficient_rounding_error) > 0.0


def test_pyramid_validity_matches_runtime_apex_derivative_support() -> None:
    # dF_x/dx = 1 + (x - 1/2)/(1 - z) varies with the collapsed horizontal
    # coordinate at the apex, while F_x itself has the unique apex value 1/2.
    # Its Jacobian determinant remains positive throughout the interior.
    denominator: dict[tuple[int, ...], Fraction] = {
        (0, 0, 0): Fraction(1),
        (0, 0, 1): Fraction(-1),
    }
    bad_x = RationalPolynomial(
        {
            (0, 0, 0): Fraction(1, 8),
            (1, 0, 0): Fraction(1, 2),
            (2, 0, 0): Fraction(1, 2),
            (1, 0, 1): Fraction(-1),
        },
        denominator,
    )
    normalized_value = normalize_pyramid_reference_expression(bad_x)
    normalized_dx = normalize_pyramid_reference_derivative(bad_x, 0)
    assert normalized_value.apex_value == Fraction(1, 2)
    assert normalized_dx.apex_value is None
    assert evaluate_expression(
        normalized_dx.expression,
        (Fraction(0), Fraction(0), Fraction(1)),
    ) == Fraction(1, 2)
    assert evaluate_expression(
        normalized_dx.expression,
        (Fraction(1), Fraction(0), Fraction(1)),
    ) == Fraction(3, 2)
    unit_denominator: dict[tuple[int, ...], Fraction] = {(0, 0, 0): Fraction(1)}
    y = RationalPolynomial({(0, 1, 0): Fraction(1)}, unit_denominator)
    z = RationalPolynomial({(0, 0, 1): Fraction(1)}, unit_denominator)
    geometry, element = _pyramid_chart_geometry((bad_x, y, z))

    certificate = D.certify_cell_geometry_validity(
        geometry, policy=D.CellValidityPolicy(relative_determinant_floor=0.0)
    )
    assert int(certificate.source_status[0]) == Status.UNRESOLVED
    assert int(certificate.execution_status[0]) == Status.UNRESOLVED
    with pytest.raises(ValueError, match="pyramid-apex continuation"):
        element.tabulate(np.asarray(((0.5, 0.5, 1.0),), dtype=np.float64))


def test_pyramid_nonaffine_chart_has_a_supported_apex_jacobian() -> None:
    unit_denominator: dict[tuple[int, ...], Fraction] = {(0, 0, 0): Fraction(1)}
    # F_x = x + 1/2 (x - 1/2)^2 has det(J) = x + 1/2 > 0 on the pyramid.
    chart = (
        RationalPolynomial(
            {
                (0, 0, 0): Fraction(1, 8),
                (1, 0, 0): Fraction(1, 2),
                (2, 0, 0): Fraction(1, 2),
            },
            unit_denominator,
        ),
        RationalPolynomial({(0, 1, 0): Fraction(1)}, unit_denominator),
        RationalPolynomial({(0, 0, 1): Fraction(1)}, unit_denominator),
    )
    geometry, element = _pyramid_chart_geometry(chart)

    certificate = D.certify_cell_geometry_validity(
        geometry, policy=D.CellValidityPolicy(relative_determinant_floor=0.0)
    )
    assert int(certificate.status[0]) == Status.CERTIFIED_VALID
    values, gradients = element.tabulate(np.asarray(((0.5, 0.5, 1.0),), dtype=np.float64))
    local_coordinates = np.asarray(geometry.coordinates)[
        np.asarray(geometry.geometry_dofs[0])[0]
    ]
    mapped = np.asarray(values) @ local_coordinates
    mapped_jacobian = np.einsum("pnd,nc->pdc", np.asarray(gradients), local_coordinates)
    assert np.allclose(mapped[0], (0.5, 0.5, 1.0))
    assert np.allclose(mapped_jacobian[0], np.eye(3))


@pytest.mark.parametrize("kind", ("simplex:2", "simplex:3", "tensor:2", "tensor:3"))
@pytest.mark.parametrize(
    "scale", (1.0, -1.0, 0.0), ids=("positive", "inverted", "collapsed")
)
def test_qualified_scalar_form_geometry_retains_exact_affine_validity(
    kind: str, scale: float
) -> None:
    topology = D.reference_cell_topology(kind)
    coordinates = np.asarray(topology.vertices, dtype=np.float64)
    coordinates[:, 0] *= scale
    mesh = D.CellMesh(
        coordinates,
        (
            D.CellBlock(
                "root",
                kind,
                np.arange(coordinates.shape[0], dtype=np.int32)[None, :],
            ),
        ),
    )
    certificate = D.certify_cell_geometry_validity(mesh)

    expected = Status.CERTIFIED_VALID if scale > 0.0 else Status.INVALID
    np.testing.assert_array_equal(certificate.status, (expected,))
    np.testing.assert_array_equal(certificate.determinant_lower, (scale,))
    np.testing.assert_array_equal(certificate.determinant_upper, (scale,))
    assert certificate.unsupported_block_names == ()
    assert mesh.blocks[0].cell_kind == kind


def test_constant_coordinate_map_has_exact_invalid_determinant() -> None:
    mesh = D.CellMesh(
        np.zeros((3, 2), dtype=np.float64),
        (D.CellBlock("collapsed", "triangle", np.asarray(((0, 1, 2),), dtype=np.int64)),),
    )
    certificate = D.certify_cell_geometry_validity(mesh)

    np.testing.assert_array_equal(certificate.status, (Status.INVALID,))
    np.testing.assert_array_equal(certificate.determinant_lower, (0.0,))
    np.testing.assert_array_equal(certificate.determinant_upper, (0.0,))


def test_relative_determinant_floor_requires_strict_margin() -> None:
    mesh = D.CellMesh(
        np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
            dtype=np.float64,
        ),
        (D.CellBlock("cells", "triangle", np.asarray(((0, 1, 2),), dtype=np.int64)),),
    )
    geometry = D.CellGeometrySpec.affine(mesh)
    certificates = tuple(
        D.certify_cell_geometry_validity(
            geometry,
            mesh=mesh,
            policy=D.CellValidityPolicy(relative_determinant_floor=floor),
        )
        for floor in (0.49, 0.5, 0.51)
    )

    assert int(certificates[0].status[0]) == Status.CERTIFIED_VALID
    assert int(certificates[1].status[0]) == Status.INVALID
    assert int(certificates[2].status[0]) == Status.INVALID


@pytest.mark.parametrize("ambient_dimension", (2, 3), ids=("square-map", "embedded-map"))
@pytest.mark.parametrize(
    ("floor", "square_status", "embedded_status"),
    (
        (0.015625, Status.CERTIFIED_VALID, Status.CERTIFIED_VALID),
        (0.25, Status.CERTIFIED_VALID, Status.CERTIFIED_VALID),
        (0.5, Status.CERTIFIED_VALID, Status.INVALID),
        (0.75, Status.INVALID, Status.INVALID),
    ),
    ids=("wide-margin", "moderate-margin", "embedded-equality", "violated-margin"),
)
def test_nonlinear_source_relative_margin_preserves_strict_threshold(
    ambient_dimension: int,
    floor: float,
    square_status: D.CellValidityStatus,
    embedded_status: D.CellValidityStatus,
) -> None:
    source = D.fem.lagrange_element("quadrilateral", 1)
    element = PolynomialComposedCellGeometryElement(
        source,
        "quadrilateral",
        (
            {(1, 0): Fraction(1)},
            {(0, 1): Fraction(1), (0, 2): Fraction(1)},
        ),
    )
    coordinates = np.zeros((source.local_dof_count, ambient_dimension), dtype=np.float64)
    coordinates[:, :2] = np.asarray(source.reference_nodes, dtype=np.float64)
    coordinates[:, 1] += coordinates[:, 0]
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(source.local_dof_count, dtype=np.int64)[None, :]},
        coordinates,
    )
    certificate = D.certify_cell_geometry_validity(
        geometry, policy=D.CellValidityPolicy(relative_determinant_floor=floor)
    )
    # The squared determinant/Hadamard ratio is exactly one half everywhere.
    expected_status = square_status if ambient_dimension == 2 else embedded_status
    assert int(certificate.source_status[0]) == expected_status
    assert int(certificate.execution_status[0]) == expected_status


def test_tangent_jacobian_zero_is_rejected_by_exact_contact() -> None:
    mesh = D.CellMesh(
        np.asarray(((0.0,), (1.0,)), dtype=np.float64),
        (D.CellBlock("tangent", "interval", np.asarray(((0, 1),))),),
    )
    element = D.fem.ReferenceNodalFamily(
        "interval", 4, node_set="equispaced"
    ).finite_element()
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    coordinates = (216.0 * (nodes[:, 0] - 0.5) ** 3)[:, None]
    geometry = D.CellGeometrySpec(
        {"tangent": element},
        {"tangent": np.arange(nodes.shape[0], dtype=np.int64)[None, :]},
        coordinates,
    )
    certificate = D.certify_cell_geometry_validity(
        geometry,
        mesh=mesh,
        policy=D.CellValidityPolicy(relative_determinant_floor=0.0),
    )

    assert int(certificate.status[0]) == Status.INVALID
    assert float(certificate.determinant_lower[0]) <= 0.0
    assert float(certificate.determinant_upper[0]) >= 0.0


def test_exact_source_contact_remains_independent_of_execution_shadows() -> None:
    geometry = _scalar_chart_geometry(
        {(1,): Fraction(1, 4), (3,): Fraction(-1, 3), (5,): Fraction(1, 5)}
    )
    certificate = D.certify_cell_geometry_validity(
        geometry, policy=D.CellValidityPolicy(relative_determinant_floor=0.0)
    )
    assert int(certificate.status[0]) == Status.UNRESOLVED
    assert int(certificate.source_status[0]) == Status.UNRESOLVED
    assert float(certificate.source_determinant_lower[0]) <= 0.0
    assert float(certificate.source_determinant_upper[0]) >= 0.0
    assert float(certificate.determinant_lower[0]) > 0.0
    assert float(certificate.maximum_element_coefficient_rounding_error) > 0.0


def test_negative_jacobian_sample_is_rejected_exactly() -> None:
    mesh, geometry = _p3_triangle(-1.0e-7)
    certificate = D.certify_cell_geometry_validity(
        geometry,
        mesh=mesh,
        policy=D.CellValidityPolicy(relative_determinant_floor=0.0),
    )

    assert int(certificate.status[0]) == Status.INVALID
    assert float(certificate.determinant_lower[0]) < 0.0


def test_thin_positive_jacobian_remains_certifiable() -> None:
    mesh, geometry = _p3_triangle(1.0e-7)
    certificate = D.certify_cell_geometry_validity(geometry, mesh=mesh)

    assert int(certificate.status[0]) == Status.CERTIFIED_VALID
    assert float(certificate.determinant_lower[0]) > 0.0


def _pentagram() -> Any:
    angles = np.pi / 2.0 + 4.0 * np.pi * np.arange(5) / 5.0
    return np.column_stack((np.cos(angles), np.sin(angles)))


def _embedded(points: Any, *, bend: Any = 0.0) -> Any:
    planar = np.column_stack((np.asarray(points, dtype=np.float64), np.zeros(5)))
    planar[3, 2] = bend
    angle = 0.4
    rotation = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, np.cos(angle), -np.sin(angle)),
            (0.0, np.sin(angle), np.cos(angle)),
        )
    )
    return planar @ rotation.T


CONVEX = ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 2.0), (0.0, 1.0))
# Notched square whose vertex-zero fan leaves the cell.
CONCAVE = ((0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (1.5, 1.0), (0.0, 3.0))


def test_cell_geometry_validity_scenario_1() -> None:
    for points, expected in (
        (CONVEX, Status.CERTIFIED_VALID),
        (CONCAVE, Status.CERTIFIED_VALID),
        (CONVEX[::-1], Status.INVALID),
        # Bow tie with a crossing between non-adjacent edges.
        (((0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (1.0, -1.0), (0.0, 2.0)), Status.INVALID),
        (_pentagram(), Status.INVALID),
        # Repeated vertex coordinates behind distinct vertex ids.
        (((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 0.0), (0.0, 2.0)), Status.INVALID),
        # A vertex touching a non-adjacent edge.
        (((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 0.0), (0.0, 4.0)), Status.INVALID),
        # Adjacent edges doubling back along one line.
        (((0.0, 0.0), (4.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0)), Status.INVALID),
    ):
        assert _polygon_status(points) == expected
    points = np.asarray(((0.0, 0.0), (1.0e-10, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    policy = D.CellValidityPolicy(relative_determinant_floor=1.0e-8)

    assert _polygon_status(points, policy) == Status.INVALID
    assert _polygon_status(_embedded(CONCAVE)) == Status.CERTIFIED_VALID
    bent = _embedded(CONCAVE, bend=0.1)
    loose = D.CellValidityPolicy(relative_planarity_tolerance=0.5)

    assert _polygon_status(bent) == Status.INVALID
    assert _polygon_status(bent, loose) == Status.CERTIFIED_VALID


def _spike(offset: Any) -> Any:
    # Vertex 3 approaches edge 0 (on the line y = x) from above.
    tip = (0.3, 0.3 if offset == 0 else np.nextafter(0.3, 1.0))
    return np.asarray(((0.1, 0.1), (0.7, 0.7), (0.1, 0.9), tip, (0.0, 0.5)))


@pytest.mark.skipif(not meshcore_available(), reason="phydrax-meshcore unavailable")
def test_filtered_simplicity_is_uncertain_where_exact_decides() -> None:
    for offset, exact in (
        (0, G.PolygonSimplicityStatus.SELF_INTERSECTING),
        (1, G.PolygonSimplicityStatus.SIMPLE),
    ):
        points = _spike(offset)
        filtered = G.polygon_simplicity_2d(points, mode=G.PredicateMode.FILTERED)
        resolved = G.polygon_simplicity_2d(points, mode=G.PredicateMode.EXACT)

        assert int(filtered.status) == G.PolygonSimplicityStatus.UNCERTAIN
        assert int(filtered.uncertain_pairs) > 0
        assert int(resolved.status) == exact
        assert int(resolved.uncertain_pairs) == 0
        expected = Status.INVALID if offset == 0 else Status.CERTIFIED_VALID
        assert _polygon_status(points) == expected


def test_cell_geometry_validity_scenario_2() -> None:
    a = np.asarray(((0.0, 0.0),) * 5)
    b = np.asarray(((2.0, 2.0), (1.0, 0.0), (2.0, 0.0), (1.0, 1.0), (1.0, 0.0)))
    c = np.asarray(((0.0, 2.0), (1.0, 0.0), (1.0, 0.0), (2.0, 2.0), (2.0, 0.0)))
    d = np.asarray(((2.0, 0.0), (1.0, 1.0), (3.0, 0.0), (3.0, 3.0), (3.0, 0.0)))
    S = G.SegmentIntersectionStatus

    result = G.segment_intersections_2d(a, b, c, d, mode=G.PredicateMode.FILTERED)

    assert tuple(int(value) for value in result.status) == (
        S.PROPER_CROSSING,
        S.ENDPOINT_CONTACT,
        S.COLLINEAR_OVERLAP,
        S.DISJOINT,
        S.DISJOINT,
    )
    with pytest.raises(ValueError):
        G.segment_intersections_2d(a, b, c, d, mode=G.PredicateMode.FILTERED_DEVICE)
    points = np.asarray(((0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (1.0, -1.0), (0.0, 2.0)))
    mesh = D.CellMesh(
        points, (D.CellBlock("cells", "polygon", np.arange(5, dtype=np.int32)[None]),)
    )

    with pytest.raises(phx.meshing.MeshingFailure, match="self_intersection"):
        phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


@pytest.mark.parametrize("ambient", (2, 3))
def test_polygon_duplicate_vertices_match_dense_reference(ambient: int) -> None:
    angles = np.arange(9, dtype=np.float64) * (2.0 * np.pi / 9.0)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    if ambient == 3:
        points = np.column_stack((points, np.zeros(9, dtype=np.float64)))
    for duplicate in (None, 3, 5, 7):
        candidate = points.copy()
        if duplicate is not None:
            candidate[duplicate] = candidate[0]
        coincident = np.all(candidate[:, None] == candidate[None, :], axis=-1)
        repeated = np.any(coincident & ~np.eye(9, dtype=np.bool_))
        expected = Status.INVALID if repeated else Status.CERTIFIED_VALID
        assert _polygon_status(candidate) == expected
