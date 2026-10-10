#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transfer import (
    _certified_cell_measures,
    _certified_sqrt_polynomial_integral,
    _embedded_squared_density,
    _prepare_mapped_edge_arc_length,
    CellGeometryTransitionError,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_enclosure import (
    constant,
    CoordinateEnclosureBudget,
    Polynomial,
    RationalPolynomial,
)
from phydrax.meshing._contracts import MeshingLimits
from phydrax.meshing._quad_generation import extract_surface_quads


def _cylinder() -> tuple[
    CellMesh, CellGeometrySpec, SplineCellGeometryElement, NDArray[np.float64]
]:
    controls = np.asarray(
        [[(x, 1.0, 0.0), (x, 1.0, 1.0), (x, 0.0, 1.0)] for x in (0.0, 1.0)],
        dtype=np.float64,
    ).reshape(-1, 3)
    u = np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64)
    v = np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=np.float64)
    weights = np.tile(np.asarray((1.0, math.sqrt(0.5), 1.0), dtype=np.float64), (2, 1))
    element = SplineCellGeometryElement(
        u, v, weights, 1, 2, (1, 2), "spline", "original-r1"
    )
    corners = np.asarray(
        ((0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        corners,
        (
            CellBlock(
                "root", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int64)
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"root": element}, {"root": np.arange(6, dtype=np.int64)[None]}, controls
    )
    return mesh, geometry, element, controls


def _six_quads(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    root: SplineCellGeometryElement,
    controls: NDArray[np.float64],
) -> tuple[CellMesh, CellGeometrySpec]:
    triangles = CellMesh(
        mesh.coordinates,
        (
            CellBlock(
                "a",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int64),
                global_ids=np.asarray((0,), dtype=np.int64),
            ),
            CellBlock(
                "b",
                "triangle",
                np.asarray(((0, 2, 3),), dtype=np.int64),
                global_ids=np.asarray((1,), dtype=np.int64),
            ),
        ),
    )
    elements = {
        "a": RestrictedCellGeometryElement(
            root,
            "triangle",
            np.asarray(((1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        ),
        "b": RestrictedCellGeometryElement(
            root,
            "triangle",
            np.asarray(((1.0, 0.0), (1.0, 1.0)), dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        ),
    }
    ancestry = CellGeometryRestrictionSource(
        cell_geometry_id(geometry),
        mesh.topology_id,
        {name: np.asarray((0,), dtype=np.int64) for name in elements},
        {name: np.asarray(((0, 1, 2, 3),), dtype=np.int64) for name in elements},
    )
    charts = CellGeometrySpec(
        elements,
        {name: np.arange(6, dtype=np.int64)[None] for name in elements},
        controls,
        restriction_source=ancestry,
    )
    result = extract_surface_quads(triangles, MeshingLimits(), source_geometry=charts)
    if result.geometry is None:
        raise ValueError(
            "Actual spline quad extraction did not publish its coordinate source."
        )
    return result.mesh, result.geometry


def test_original_and_six_rational_quads_enclose_the_cylinder_not_the_carrier() -> None:
    mesh, geometry, root, controls = _cylinder()
    quads, restricted = _six_quads(mesh, geometry, root, controls)
    values, errors, exact = _certified_cell_measures(mesh, geometry, relative_tolerance=0)
    pieces, bounds, child_exact = _certified_cell_measures(
        quads, restricted, relative_tolerance=0
    )
    assert not exact and not child_exact
    assert abs(values.sum() - math.pi / 2) <= errors.sum() + 2e-15
    assert abs(pieces.sum() - math.pi / 2) <= bounds.sum() + 2e-15
    assert abs(pieces.sum() - values.sum()) <= bounds.sum() + errors.sum()
    assert abs(values.sum() - math.sqrt(2)) > 0.1
    np.testing.assert_array_equal(geometry.coordinates, controls)
    gram = _embedded_squared_density(root, controls)
    weighted, error = _certified_sqrt_polynomial_integral(
        gram, {(1, 0): Fraction(1)}, "quadrilateral", relative_tolerance=0
    )
    assert abs(weighted - math.pi / 4) <= error + 2e-15


def test_full_source_arc_enclosure_and_restricted_edges_agree() -> None:
    _, _, root, controls = _cylinder()
    start, end = (Fraction(0), Fraction(0)), (Fraction(0), Fraction(1))
    prepared = _prepare_mapped_edge_arc_length(root, controls, start, end)
    result = prepared.integrate(relative_tolerance=0)
    cold = _prepare_mapped_edge_arc_length(root, controls, start, end).integrate(
        relative_tolerance=0
    )
    assert result == cold
    assert result.lower <= math.pi / 2 <= result.upper
    assert result.lower > math.sqrt(2)
    straight = _prepare_mapped_edge_arc_length(
        root, controls, start, (Fraction(1), Fraction(0))
    ).integrate(relative_tolerance=0)
    assert straight.value == straight.lower == straight.upper == 1 and straight.error == 0
    pieces = []
    for offset in (0.0, 0.5):
        element = RestrictedCellGeometryElement(
            root,
            "quadrilateral",
            np.diag((1.0, 0.5)),
            np.asarray((0.0, offset), dtype=np.float64),
        )
        pieces.append(
            _prepare_mapped_edge_arc_length(element, controls, start, end).integrate(
                relative_tolerance=0
            )
        )
    assert (
        abs(sum(piece.value for piece in pieces) - result.value)
        <= sum(piece.error for piece in pieces) + result.error
    )


def test_signed_rational_weight_retains_whole_integral_relative_error() -> None:
    weight = RationalPolynomial(
        {(1, 0): Fraction(1), (0, 0): Fraction(-1, 2)},
        {(0, 0): Fraction(1), (1, 0): Fraction(1)},
    )
    value, error = _certified_sqrt_polynomial_integral(
        constant(2, 2),
        weight,
        "quadrilateral",
        absolute_tolerance=0,
        relative_tolerance=1e-10,
    )
    oracle = math.sqrt(2) * (1 - 1.5 * math.log(2))
    assert abs(value - oracle) <= error + 3e-16
    assert error <= 1e-10 * (abs(value) - error)
    bad = RationalPolynomial(constant(1, 2), {(0, 0): Fraction(1), (1, 0): Fraction(-2)})
    with pytest.raises(ValueError, match="denominator"):
        _certified_sqrt_polynomial_integral(bad, constant(1, 2), "quadrilateral")


def test_rational_measure_one_under_work_and_insufficient_series_refuse_atomically() -> (
    None
):
    mesh, geometry, root, controls = _cylinder()
    ledger = CoordinateEnclosureBudget(100_000_000, 100_000_000)
    with ledger.activate():
        _certified_cell_measures(mesh, geometry, relative_tolerance=0)
    with pytest.raises(CellGeometryTransitionError) as refusal:
        _certified_cell_measures(
            mesh, geometry, relative_tolerance=0, maximum_work=ledger.work_units - 1
        )
    assert refusal.value.reason == "resource_limit"
    assert refusal.value.measured > refusal.value.limit
    gram = _embedded_squared_density(root, controls)
    for limits in (
        {"maximum_subcells": 1},
        {"maximum_binomial_terms": 1, "maximum_subcells": 2},
    ):
        with pytest.raises(CellGeometryTransitionError) as exhausted:
            _certified_sqrt_polynomial_integral(
                gram, constant(1, 2), "quadrilateral", relative_tolerance=0, **limits
            )
        assert exhausted.value.reason == "resource_limit"
    np.testing.assert_array_equal(geometry.coordinates, controls)


def test_rational_removable_corner_uses_exact_cone_and_refuses_unbounded_poles() -> None:
    denominator: Polynomial = {
        (2, 0): Fraction(1),
        (1, 1): Fraction(2),
        (0, 2): Fraction(1),
    }
    gram = RationalPolynomial({**denominator, (3, 0): Fraction(1)}, denominator)
    value, error = _certified_sqrt_polynomial_integral(
        gram, constant(1, 2), "triangle", relative_tolerance=0
    )
    points, weights = np.polynomial.legendre.leggauss(64)
    u, v = np.meshgrid((points + 1) / 2, (points + 1) / 2, indexing="ij")
    oracle = np.sum(weights[:, None] * weights[None, :] * v * np.sqrt(1 + u**3 * v)) / 4
    assert abs(value - oracle) <= error + 2e-15
    pole = RationalPolynomial(constant(1, 2), denominator)
    with pytest.raises(ValueError):
        _certified_sqrt_polynomial_integral(pole, constant(1, 2), "triangle")


def test_bounded_directional_apex_gram_is_integrated_without_faking_a_corner_value() -> (
    None
):
    gram = RationalPolynomial(
        {(2, 0): Fraction(1), (0, 2): Fraction(1)},
        {(2, 0): Fraction(1), (1, 1): Fraction(2), (0, 2): Fraction(1)},
    )
    value, error = _certified_sqrt_polynomial_integral(
        gram, constant(1, 2), "triangle", relative_tolerance=0
    )
    oracle = (math.sqrt(2) + math.asinh(1)) / (4 * math.sqrt(2))
    assert abs(value - oracle) <= error + 2e-15


def test_actual_nonlinear_six_quad_dg_content_and_component_adjoint() -> None:
    from phydrax.discretization.fem import (
        discontinuous_element,
        FiniteElementFieldSpec,
        FiniteElementPlan,
    )
    from phydrax.discretization.fem._topology_transfer import (
        _prepare_mapped_nested_dg_transfer,
    )

    mesh, geometry, root, controls = _cylinder()
    quads, restricted = _six_quads(mesh, geometry, root, controls)
    source = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "u", {"root": discontinuous_element("quadrilateral", 0)}, component_shape=(2,)
        ),
        coordinate_spec=geometry,
    ).prepare()
    target = FiniteElementPlan(
        quads,
        FiniteElementFieldSpec(
            "u",
            {
                block.name: discontinuous_element("quadrilateral", 0)
                for block in quads.blocks
            },
            component_shape=(2,),
        ),
        coordinate_spec=restricted,
    ).prepare()
    prepared = _prepare_mapped_nested_dg_transfer(
        source,
        target,
        geometry,
        restricted,
        field_name="u",
        parent_cells=np.zeros(6, dtype=np.int64),
    )
    coefficients = np.asarray(((2.7, -0.8),), dtype=np.float64)
    transferred = np.asarray(prepared.transfer.apply(coefficients))
    dual = np.asarray(
        ((0.3, -0.2), (-0.7, 0.6), (0.4, 0.9), (-0.8, 0.1), (0.2, -0.5), (0.6, 0.8)),
        dtype=np.float64,
    )
    if prepared.target_measures is None:
        raise ValueError(
            "The accepted DG transfer omitted its physical inventory functional."
        )
    np.testing.assert_allclose(
        np.asarray(prepared.target_measures) @ transferred,
        (math.pi / 2) * coefficients[0],
        rtol=0.0,
        atol=3e-13,
    )
    np.testing.assert_allclose(
        np.vdot(transferred, dual),
        np.vdot(coefficients, prepared.transfer.pullback(dual)),
        rtol=0.0,
        atol=2e-14,
    )
    assert prepared.evidence.passed


def test_rational_surface_and_projective_chart_content_keeps_the_original_source() -> (
    None
):
    from phydrax.discretization._coordinate_enclosure import add, axes
    from phydrax.discretization.fem._sphere_chart_transfer import (
        _projective_form,
        _projective_rational_density,
    )
    from phydrax.discretization.fem._surface_chart_transfer import (
        _integral,
        _IntegrationBudgets,
        _restricted,
    )

    _, _, root, controls = _cylinder()
    gram = _embedded_squared_density(root, controls)
    reference = np.asarray(
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
        dtype=object,
    )
    restricted, _ = _restricted(gram, reference)
    assert isinstance(restricted, RationalPolynomial)
    budgets: _IntegrationBudgets = dict(
        absolute_tolerance=1e-10,
        relative_tolerance=0.0,
        maximum_work=100_000_000,
        maximum_subcells=10000,
        maximum_binomial_terms=32,
    )
    value, error = _integral(restricted, constant(1, 2), budgets)
    assert abs(value - math.pi / 4) <= error + 2e-15
    u, v = axes(2)
    denominator = add(constant(1, 2), u)
    density, measure_weight, exponent = _projective_rational_density(
        restricted, (u, v), denominator, Fraction(1)
    )
    values, errors = _projective_form(
        (measure_weight,), (constant(1, 2),), density, denominator, exponent, budgets
    )
    # (u,v)/(1+u) maps this triangle to 2*u+v<=1. Its physical
    # cylinder content is half the original chart, not a corner-chord area.
    assert abs(values[0, 0] - math.pi / 8) <= errors[0, 0] + 2e-15
    assert errors[0, 0] <= budgets["absolute_tolerance"]
