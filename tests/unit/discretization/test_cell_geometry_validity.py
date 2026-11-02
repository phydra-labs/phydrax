#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from dataclasses import dataclass, fields
from fractions import Fraction
from typing import Any

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
import phydrax.discretization._cell_geometry as cell_geometry_module
import phydrax.discretization._cell_geometry_validity as validity_module
from phydrax._meshcore import meshcore_available
from phydrax.discretization._cell_geometry import (
    CellGeometrySpec,
    PerCellRestrictedCellGeometryElement,
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
from tests._support.validity_reuse import assert_incremental_matches_full


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


def test_validity_revalidate_reproves_and_require_current_checks_binding() -> None:
    mesh, geometry = _p3_triangle(0.5)
    certificate = D.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate.require_current(geometry, mesh=mesh)
    certificate.revalidate(geometry, mesh=mesh)
    public = D.CellValidityCertificate(geometry, mesh=mesh, policy=certificate.policy)
    assert public.certificate_id == certificate.certificate_id
    with pytest.raises(TypeError):
        D.CellValidityCertificate(
            geometry,
            mesh=mesh,
            status=certificate.status,  # ty: ignore[unknown-argument]
        )
    summary = certificate.stage_summary()
    for proof in ("execution", "source"):
        assert sum(summary[proof].values()) == certificate.status.shape[0]

    lower = np.asarray(certificate.determinant_lower) * 0.5
    fake_status = np.where(
        np.asarray(certificate.execution_status) == Status.CERTIFIED_VALID,
        Status.INVALID,
        Status.CERTIFIED_VALID,
    )
    # Simulate an untrusted stored record, bypassing the computing public constructor.
    forged = validity_module.CellValidityCertificate._from_computed(
        fake_status,
        lower,
        certificate.determinant_upper,
        certificate.depth,
        source_proof=validity_module._BlockCertificate(
            fake_status,
            lower,
            np.asarray(certificate.source_determinant_upper),
            np.asarray(certificate.source_depth),
        ),
        maximum_coefficient_rounding_error=float(
            certificate.maximum_coefficient_rounding_error
        ),
        maximum_element_coefficient_rounding_error=float(
            certificate.maximum_element_coefficient_rounding_error
        ),
        apex_derivative_unique=np.asarray(certificate.apex_derivative_unique),
        geometry=geometry,
        mesh=mesh,
        policy=certificate.policy,
        block_names=certificate.block_names,
        block_offsets=certificate.block_offsets,
        unsupported_block_names=certificate.unsupported_block_names,
        geometry_id=certificate.geometry_id,
        geometry_layout_id=certificate.geometry_layout_id,
        cell_map_digests=certificate.cell_map_digests,
        policy_id=certificate.policy_id,
    )
    # require_current checks the binding and stored claims only; revalidate
    # recomputes the proof and refuses the forged determinant bounds.
    forged.require_current(geometry, mesh=mesh)
    with pytest.raises(ValueError, match="does not reproduce the current mathematical"):
        forged.revalidate(geometry, mesh=mesh)
    transformed = eqx.tree_at(
        lambda value: value.status,
        forged,
        forged.status.copy(),
    )
    with pytest.raises(ValueError, match="does not reproduce the current mathematical"):
        D.certify_cell_geometry_validity(geometry, mesh=mesh, parent=transformed)
    stale_identity = eqx.tree_at(
        lambda value: value.determinant_lower,
        certificate,
        certificate.determinant_lower * 0.5,
    )
    assert stale_identity.certificate_id == certificate.certificate_id
    with pytest.raises(ValueError, match="does not reproduce the current mathematical"):
        D.certify_cell_geometry_validity(geometry, mesh=mesh, parent=stale_identity)
    wrong_dtype = eqx.tree_at(
        lambda value: value.depth,
        certificate,
        np.asarray(certificate.depth, dtype=np.int16),
    )
    assert wrong_dtype.certificate_id == certificate.certificate_id
    np.testing.assert_array_equal(wrong_dtype.depth, certificate.depth)
    with pytest.raises(ValueError, match="does not reproduce the current mathematical"):
        D.certify_cell_geometry_validity(geometry, mesh=mesh, parent=wrong_dtype)

    _, other_geometry = _p3_triangle(0.25)
    with pytest.raises(ValueError, match="bound to these exact scientific inputs"):
        certificate.require_current(other_geometry, mesh=mesh)
    with pytest.raises(ValueError, match="bound to these exact scientific inputs"):
        certificate.revalidate(other_geometry, mesh=mesh)
    object.__setattr__(public, "cell_map_digests", bytes(len(public.cell_map_digests)))
    assert public.certificate_id == certificate.certificate_id
    with pytest.raises(ValueError, match="does not reproduce the current mathematical"):
        public.revalidate(geometry, mesh=mesh)


def test_coordinate_source_basis_is_formed_once_per_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cell_geometry_module, "_SOURCE_BASES", {})
    original = cell_geometry_module._root_source_basis
    calls: list[str] = []

    def counted(element: Any, /) -> Any:
        calls.append(element.element_id)
        return original(element)

    monkeypatch.setattr(cell_geometry_module, "_root_source_basis", counted)
    element = D.fem.lagrange_element("triangle", 2)
    first = cell_geometry_module.coordinate_source_basis(element)
    second = cell_geometry_module.coordinate_source_basis(element)

    assert calls == [element.element_id]
    assert first == second
    assert all(left is not right for left, right in zip(first, second, strict=True))
    first[0].clear()
    assert cell_geometry_module.coordinate_source_basis(element) == second


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
    # Whole-cell validity holds on the closed collapsed cube; only pointwise apex
    # tabulation is direction-dependent and refused at runtime.
    assert int(certificate.source_status[0]) == Status.CERTIFIED_VALID
    assert int(certificate.execution_status[0]) == Status.CERTIFIED_VALID
    assert not bool(np.asarray(certificate.apex_derivative_unique)[0])
    with pytest.raises(ValueError, match="pyramid-apex continuation"):
        element.tabulate(np.asarray(((0.5, 0.5, 1.0),), dtype=np.float64))


def _p1_pyramid_mesh(base: tuple[tuple[float, float], ...]) -> D.CellMesh:
    points = np.asarray(
        [(x, y, 0.0) for x, y in base] + [(0.25, 0.25, -0.25)], dtype=np.float64
    )
    return D.CellMesh(
        points,
        (D.CellBlock("pyramids", "pyramid", np.asarray(((0, 1, 2, 3, 4),), np.int32)),),
    )


def test_kite_pyramid_is_valid_with_direction_dependent_apex_jacobian() -> None:
    # Tet-to-hex closure pyramid: tet corner, two edge midpoints, face centroid.
    mesh = _p1_pyramid_mesh(((0.0, 0.0), (0.0, 0.5), (1 / 3, 1 / 3), (0.5, 0.0)))
    certificate = D.certify_cell_geometry_validity(
        D.CellGeometrySpec.affine(mesh),
        mesh=mesh,
        policy=D.CellValidityPolicy(relative_determinant_floor=0.0),
    )
    assert int(certificate.status[0]) == Status.CERTIFIED_VALID
    assert not bool(np.asarray(certificate.apex_derivative_unique)[0])


def test_folded_kite_pyramid_is_invalid() -> None:
    # The third base corner crosses the opposite diagonal, folding the base.
    mesh = _p1_pyramid_mesh(((0.0, 0.0), (0.0, 0.5), (0.1, 0.1), (0.5, 0.0)))
    certificate = D.certify_cell_geometry_validity(
        D.CellGeometrySpec.affine(mesh),
        mesh=mesh,
        policy=D.CellValidityPolicy(relative_determinant_floor=0.0),
    )
    assert int(certificate.status[0]) == Status.INVALID


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


def test_per_cell_pyramid_chart_accepts_only_removable_apex_poles() -> None:
    # F_x = x - (x - z/2)(y - z/2) / (4(1 - z)) is u(1 - w)(1 - v/4) + w/2 in
    # collapsed-cube form: the 1 - z pole cancels and the apex image is 1/2.
    pole: dict[tuple[int, ...], Fraction] = {
        (0, 0, 0): Fraction(1),
        (0, 0, 1): Fraction(-1),
    }
    removable_x = RationalPolynomial(
        {
            (1, 0, 0): Fraction(1),
            (1, 0, 1): Fraction(-7, 8),
            (1, 1, 0): Fraction(-1, 4),
            (0, 1, 1): Fraction(1, 8),
            (0, 0, 2): Fraction(-1, 16),
        },
        pole,
    )
    y: dict[tuple[int, ...], Fraction] = {(0, 1, 0): Fraction(1)}
    z: dict[tuple[int, ...], Fraction] = {(0, 0, 1): Fraction(1)}
    source = D.fem.lagrange_element("hexahedron", 1)
    element = PerCellRestrictedCellGeometryElement(
        source, "pyramid", (7,), (3,), ((removable_x, y, z),)
    )
    assert normalize_pyramid_reference_expression(removable_x).apex_value == Fraction(
        1, 2
    )

    point = (Fraction(1, 4), Fraction(1, 2), Fraction(1, 4))
    values, gradients = element.tabulate_cell(
        0, np.asarray([[float(value) for value in point]], dtype=np.float64)
    )
    nodes = np.asarray(source.reference_nodes, dtype=np.float64)
    mapped = np.asarray(values)[0] @ nodes
    expected_x = evaluate_expression(removable_x, point)
    np.testing.assert_allclose(
        mapped, (float(expected_x), 0.5, 0.25), rtol=0.0, atol=1e-14
    )
    assert np.all(np.isfinite(np.asarray(gradients)))

    mesh = D.CellMesh(
        np.asarray(
            ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, 1)),
            dtype=np.float64,
        ),
        (
            D.CellBlock(
                "pyramids",
                "pyramid",
                np.arange(5, dtype=np.int32)[None, :],
                global_ids=np.asarray((7,), dtype=np.int64),
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"pyramids": element},
        {"pyramids": np.arange(nodes.shape[0], dtype=np.int64)[None, :]},
        nodes,
    )
    certificate = D.certify_cell_geometry_validity(
        geometry,
        mesh=mesh,
        policy=D.CellValidityPolicy(relative_determinant_floor=0.0),
    )
    assert int(certificate.status[0]) == Status.CERTIFIED_VALID
    # dF_x/dx = 1 - v/4 depends on the collapsed direction at the apex.
    assert not bool(np.asarray(certificate.apex_derivative_unique)[0])
    with pytest.raises(ValueError, match="pyramid-apex continuation"):
        element.tabulate_cell(0, np.asarray(((0.5, 0.5, 1.0),), dtype=np.float64))

    # (x - z/2) / (1 - z) is the collapsed coordinate u: no unique apex image.
    direction_x = RationalPolynomial(
        {(1, 0, 0): Fraction(1), (0, 0, 1): Fraction(-1, 2)}, pole
    )
    # x / (1 - z) keeps a genuine pole at the apex.
    pole_x = RationalPolynomial({(1, 0, 0): Fraction(1)}, pole)
    for refused in (direction_x, pole_x):
        with pytest.raises(ValueError, match="denominator is not proved nonzero"):
            PerCellRestrictedCellGeometryElement(
                source, "pyramid", (7,), (3,), ((refused, y, z),)
            )


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


def _flat_tetrahedron_points(relative: float) -> np.ndarray:
    # Jacobian columns e1, e2 and (1, 1, h): det = h and the Hadamard bound is
    # sqrt(2 + h^2), so the relative determinant is h / sqrt(2 + h^2).
    height = relative * np.sqrt(2.0 / (1.0 - relative * relative))
    return np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, height)),
        dtype=np.float64,
    )


@pytest.mark.parametrize("orientation", (1.0, -1.0), ids=("positive", "reflected"))
def test_near_flat_full_dimensional_jacobian_keeps_its_sign(orientation: float) -> None:
    # Relative determinant 1e-9: its square (1e-18) is below binary64
    # resolution of the metric (Gram) determinant, so the sign must come from
    # det(J) itself.
    points = _flat_tetrahedron_points(1.0e-9)
    jacobian = (points[1:] - points[0]).T
    jacobian[:, 2] *= orientation
    height = points[3, 2]
    inverse, _, measure, determinant, successful = (
        validity_module.cell_jacobian_inverse_and_measure(jacobian)
    )
    assert bool(successful)
    assert np.sign(float(determinant)) == orientation
    np.testing.assert_allclose(float(determinant), orientation * height, rtol=1.0e-9)
    np.testing.assert_allclose(float(measure), height, rtol=1.0e-9)
    np.testing.assert_allclose(np.asarray(inverse) @ jacobian, np.eye(3), atol=1.0e-6)


def test_finite_element_preparation_accepts_a_near_flat_positive_tetrahedron() -> None:
    # Relative determinant 1e-9: positive, but the binary64 metric (Gram)
    # determinant of this cell has no reliable sign. Finite-element
    # preparation and its cell map must evaluate det(J) and accept it.
    points = _flat_tetrahedron_points(1.0e-9)
    mesh = D.CellMesh(
        points,
        (D.CellBlock("tet", "tetrahedron", np.asarray(((0, 1, 2, 3),))),),
    )
    discretization = D.FiniteElementPlan(
        mesh, D.FiniteElementFieldSpec("u", D.lagrange_element("tetrahedron", 1))
    ).prepare()
    cell_map = D.fem.prepare_finite_element_cell_map(discretization, 0)
    evaluation = cell_map.evaluate(
        discretization.default_runtime.coordinates,
        np.zeros((1,), dtype=np.int32),
        np.full((1, 3), 0.25, dtype=np.float64),
    )
    assert bool(np.asarray(evaluation.valid)[0])
    np.testing.assert_allclose(
        np.asarray(evaluation.determinant)[0], points[3, 2], rtol=1.0e-9
    )
    assert float(np.asarray(evaluation.minimum_metric_eigenvalue)[0]) > 0.0


def _flat_triangle_points(measure: float) -> np.ndarray:
    # Embedded triangle with Jacobian columns (1, 0, 0) and (1, h, 0): its
    # Gram-relative determinant, the policy's measure of embedded maps, is
    # r^2 = h^2 / (1 + h^2).
    height = np.sqrt(measure / (1.0 - measure))
    return np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, height, 0.0)), dtype=np.float64
    )


def _affine_status(points: np.ndarray, kind: str, policy: Any = None) -> int:
    mesh = D.CellMesh(
        points,
        (D.CellBlock("cells", kind, np.arange(points.shape[0])[None, :]),),
    )
    geometry = D.CellGeometrySpec.affine(mesh)
    certificate = D.certify_cell_geometry_validity(geometry, mesh=mesh, policy=policy)
    return int(certificate.status[0])


@pytest.mark.parametrize(
    ("kind", "points"),
    (("tetrahedron", _flat_tetrahedron_points), ("triangle", _flat_triangle_points)),
    ids=("full-dimensional", "embedded"),
)
def test_validity_floor_refuses_exactly_what_jacobian_consumers_cannot_evaluate(
    kind: str, points: Any
) -> None:
    floor = D.CellValidityPolicy().determinant_floor()
    assert _affine_status(points(0.5 * floor), kind) == Status.INVALID
    assert _affine_status(points(2.0 * floor), kind) == Status.CERTIFIED_VALID
    # A requested floor below the consumers' bound does not lower it.
    lenient = D.CellValidityPolicy(relative_determinant_floor=0.0)
    assert _affine_status(points(0.5 * floor), kind, lenient) == Status.INVALID
    # Every certified cell is evaluable by the consumers.
    cell = points(floor * (1.0 + 1.0e-6))
    jacobian = (cell[1:] - cell[0]).T
    *_, successful = validity_module.cell_jacobian_inverse_and_measure(jacobian)
    assert bool(successful)


def test_thin_layer_polygon_is_valid_and_an_unresolvable_edge_is_not() -> None:
    # A 1e-6-thick layer polygon: thinness is an aspect ratio, not a degeneracy.
    thin = ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0e-6), (0.0, 1.0e-6))
    assert _polygon_status(thin) == Status.CERTIFIED_VALID
    # An edge of one unit in the last place of 0.5 has no resolvable direction.
    degenerate = (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.0, np.nextafter(0.5, 1.0)),
        (0.0, 0.5),
    )
    assert _polygon_status(degenerate) == Status.INVALID


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


def _prism_layer_points(levels: int) -> np.ndarray:
    wall = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)), dtype=np.float64)
    # A varying layer thickness makes every prism map bilinear (nonaffine).
    thickness = 0.1 + 0.03 * wall[:, 0] + 0.02 * wall[:, 1]
    return np.concatenate(
        [np.column_stack((wall, level * thickness)) for level in range(levels + 1)]
    )


def _prism_layer(level: int) -> np.ndarray:
    triangles = np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int32)
    return np.concatenate((4 * level + triangles, 4 * (level + 1) + triangles), axis=1)


def test_layer_assembly_reuses_layer_proofs_by_exact_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = _prism_layer_points(2)
    # The accepted first layer is certified on its own, renumbered vertex set.
    order = np.arange(8, dtype=np.int32)[::-1]
    first = D.CellMesh(
        points[:8][order],
        (D.CellBlock("cells", "prism", order[_prism_layer(0)]),),
    )
    layer_validity = D.certify_cell_geometry_validity(first)
    assert layer_validity.all_certified
    assembled = D.CellMesh(
        points,
        (
            D.CellBlock(
                "cells", "prism", np.concatenate((_prism_layer(0), _prism_layer(1)))
            ),
        ),
    )

    full, proofs = assert_incremental_matches_full(monkeypatch, assembled, layer_validity)

    assert full.all_certified
    assert proofs == 2


def _curved_triangle_pair(
    interior_shift: float,
) -> tuple[Any, CellGeometrySpec, int]:
    mesh = D.CellMesh(
        np.asarray(
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 0.0), (3.0, 0.0), (2.0, 1.0)),
            dtype=np.float64,
        ),
        (D.CellBlock("cells", "triangle", np.asarray(((0, 1, 2), (3, 4, 5)))),),
    )
    element = D.fem.lagrange_element("triangle", 3)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    u, v = nodes.T
    cell = np.column_stack((((u - 0.5) ** 3) / 3.0 + u, v))
    coordinates = np.concatenate((cell, cell + np.asarray((2.0, 0.0))))
    interior = nodes.shape[0] + int(np.flatnonzero((u > 0) & (v > 0) & (u + v < 1))[0])
    coordinates[interior, 1] += interior_shift
    routes = np.arange(2 * nodes.shape[0], dtype=np.int64).reshape(2, -1)
    return (
        mesh,
        D.CellGeometrySpec({"cells": element}, {"cells": routes}, coordinates),
        interior,
    )


def _cell_maps(certificate: Any, cell: int) -> set[bytes]:
    """The execution and source map digests of one cell."""
    table = np.frombuffer(certificate.cell_map_digests, dtype=np.uint8).reshape(-1, 2, 16)
    return {row.tobytes() for row in table[cell]}


def test_relaxation_round_reproves_only_moved_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh, previous, _ = _curved_triangle_pair(0.0)
    previous_validity = D.certify_cell_geometry_validity(previous, mesh=mesh)
    _, relaxed, _ = _curved_triangle_pair(1.0e-3)

    full, proofs = assert_incremental_matches_full(
        monkeypatch, relaxed, previous_validity, mesh=mesh
    )

    assert full.all_certified
    # The rounded P3 basis gives each cell distinct execution and source maps.
    assert proofs == len(_cell_maps(full, 1)) == 2
    assert _cell_maps(full, 0) == _cell_maps(previous_validity, 0)


def test_one_ulp_coefficient_change_is_reproved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh, geometry, interior = _curved_triangle_pair(0.0)
    parent = D.certify_cell_geometry_validity(geometry, mesh=mesh)
    coordinates = np.array(geometry.coordinates, dtype=np.float64)
    coordinates[interior, 0] = np.nextafter(coordinates[interior, 0], np.inf)
    nudged = D.CellGeometrySpec(
        {"cells": geometry.elements[0]},
        {"cells": np.asarray(geometry.geometry_dofs[0])},
        coordinates,
    )

    full, proofs = assert_incremental_matches_full(monkeypatch, nudged, parent, mesh=mesh)

    assert proofs == len(_cell_maps(full, 1))
    assert _cell_maps(full, 0) == _cell_maps(parent, 0)
    assert not _cell_maps(full, 1) & _cell_maps(parent, 1)


def test_validated_parent_replays_once_before_target_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh, geometry, _ = _curved_triangle_pair(0.0)
    parent = D.certify_cell_geometry_validity(geometry, mesh=mesh)
    calls = [0]
    original = validity_module.CellValidityCertificate.revalidate

    def counted(self: Any, *args: Any, **kwargs: Any) -> None:
        calls[0] += 1
        original(self, *args, **kwargs)

    monkeypatch.setattr(validity_module.CellValidityCertificate, "revalidate", counted)
    validated = validity_module.ValidatedCellValidityParent(parent)
    assert calls == [1]
    reused = D.certify_cell_geometry_validity(geometry, mesh=mesh, parent=validated)
    assert reused.certificate_id == parent.certificate_id
    assert calls == [1]


@pytest.mark.parametrize(
    "field_name",
    (
        "certified_valid_count",
        "invalid_count",
        "unresolved_count",
        "block_names",
        "block_offsets",
        "unsupported_block_names",
    ),
)
def test_stored_validity_refuses_forged_static_claim(field_name: str) -> None:
    from phydrax._strict import reconstruct_strict

    mesh, geometry, _ = _curved_triangle_pair(0.0)
    parent = D.certify_cell_geometry_validity(geometry, mesh=mesh)
    values = {field.name: getattr(parent, field.name) for field in fields(parent)}
    old = values[field_name]
    values[field_name] = old + 1 if isinstance(old, int) else old + ("forged",)
    forged = reconstruct_strict(type(parent), values)
    assert forged.certificate_id == parent.certificate_id
    with pytest.raises(ValueError):
        D.certify_cell_geometry_validity(geometry, mesh=mesh, parent=forged)


def test_stored_validity_refuses_policy_with_stale_identity() -> None:
    from phydrax._model._structure import _restore_dataclass
    from phydrax._strict import reconstruct_strict

    mesh, geometry, _ = _curved_triangle_pair(0.0)
    parent = D.certify_cell_geometry_validity(geometry, mesh=mesh)
    policy_fields = {
        field.name: getattr(parent.policy, field.name) for field in fields(parent.policy)
    }
    policy_fields["relative_planarity_tolerance"] = 0.25
    forged_policy = _restore_dataclass(type(parent.policy), policy_fields)
    values = {field.name: getattr(parent, field.name) for field in fields(parent)}
    values["policy"] = forged_policy
    forged = reconstruct_strict(type(parent), values)
    with pytest.raises(ValueError, match="policy does not reproduce"):
        D.certify_cell_geometry_validity(geometry, mesh=mesh, parent=forged)


def test_parent_validity_requires_the_same_policy() -> None:
    mesh, geometry, _ = _curved_triangle_pair(0.0)
    parent = D.certify_cell_geometry_validity(
        geometry,
        mesh=mesh,
        policy=D.CellValidityPolicy(relative_determinant_floor=1.0e-6),
    )
    with pytest.raises(ValueError, match="same validity policy"):
        D.certify_cell_geometry_validity(geometry, mesh=mesh, parent=parent)


def _assert_embedded_affine_minor_matches_gram(
    jacobian: tuple[tuple[validity_module.ExactScalar, ...], ...], label: str, /
) -> None:
    policy = D.CellValidityPolicy()
    gram = tuple(
        tuple(
            sum(
                (jacobian[axis][left] * jacobian[axis][right] for axis in range(3)),
                Fraction(0),
            )
            for right in range(2)
        )
        for left in range(2)
    )
    expected = validity_module.determinant_exact(gram)
    status, actual = validity_module._affine_quality(jacobian, 2, policy)
    assert actual == expected, label
    squared_scale = gram[0][0] * gram[1][1]
    expected_status = (
        Status.CERTIFIED_VALID
        if expected > 0
        and expected - Fraction(policy.determinant_floor()) * squared_scale > 0
        else Status.INVALID
    )
    assert status == expected_status, label


def test_embedded_affine_minor_matches_gram_for_dyadic_samples() -> None:
    rng = np.random.default_rng(8291)
    matrices = rng.uniform(-2.0, 2.0, size=(80, 3, 2))
    for index in range(matrices.shape[0]):
        jacobian = tuple(
            tuple(Fraction(float(matrices[index, row, column])) for column in range(2))
            for row in range(3)
        )
        _assert_embedded_affine_minor_matches_gram(jacobian, f"sample {index}")


@pytest.mark.parametrize("case", ("near-degenerate", "singular", "radical"))
def test_embedded_affine_minor_matches_gram_for_adversarial_cases(case: str) -> None:
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement

    if case == "near-degenerate":
        jacobian = (
            (Fraction(1), Fraction(1)),
            (Fraction(0), Fraction(1, 2**100)),
            (Fraction(0), Fraction(0)),
        )
    elif case == "singular":
        jacobian = (
            (Fraction(1), Fraction(2)),
            (Fraction(2), Fraction(4)),
            (Fraction(3), Fraction(6)),
        )
    else:
        radical = RadicalElement.sqrt(2)
        jacobian = (
            (radical, Fraction(1)),
            (Fraction(1), radical),
            (Fraction(0), Fraction(1)),
        )
    _assert_embedded_affine_minor_matches_gram(jacobian, case)
