from fractions import Fraction
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    coordinate_source_basis,
    RationalComposedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import (
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._coordinate_enclosure import (
    evaluate_expression,
    normalize_pyramid_reference_expression,
    Polynomial,
    RationalPolynomial,
)
from phydrax.discretization._nested_reference_types import (
    exact_reference_cell_vertices,
    ReferenceCellMapWitnesses,
)
from phydrax.discretization.fem._reference import lagrange_element


def _rational_chart() -> RationalComposedCellGeometryElement:
    return RationalComposedCellGeometryElement(
        lagrange_element("interval", 1),
        "interval",
        (
            RationalPolynomial(
                {(1,): Fraction(1)}, {(0,): Fraction(1), (1,): Fraction(-1)}
            ),
        ),
    )


def test_rational_chart_compilation_preserves_values_and_chain_derivatives() -> None:
    element = _rational_chart()
    points = jnp.asarray(((0.25,),), dtype=jnp.float64)
    tabulate = RationalComposedCellGeometryElement.tabulate
    values, gradients = eqx.filter_jit(tabulate)(element, points)
    automatic, _ = eqx.filter_jit(jax.jacfwd(tabulate, argnums=1))(element, points)

    np.testing.assert_allclose(values, ((2.0 / 3.0, 1.0 / 3.0),), atol=1.0e-14)
    expected = np.asarray((((-16.0 / 9.0,), (16.0 / 9.0,)),), dtype=np.float64)
    np.testing.assert_allclose(gradients, expected, atol=1.0e-14)
    np.testing.assert_allclose(automatic[:, :, :, 0], expected, atol=1.0e-14)


def test_rational_hex_identity_continues_at_pyramid_apex_with_exact_jvp() -> None:
    source = lagrange_element("hexahedron", 1)
    one_minus_z: Polynomial = {
        (0, 0, 0): Fraction(1),
        (0, 0, 1): Fraction(-1),
    }
    chart = (
        RationalPolynomial(
            {(1, 0, 0): Fraction(1), (1, 0, 1): Fraction(-1)}, one_minus_z
        ),
        RationalPolynomial(
            {(0, 1, 0): Fraction(1), (0, 1, 1): Fraction(-1)}, one_minus_z
        ),
        RationalPolynomial(
            {(0, 0, 1): Fraction(1), (0, 0, 2): Fraction(-1)}, one_minus_z
        ),
    )
    element = RationalComposedCellGeometryElement(source, "pyramid", chart)
    points = jnp.asarray(((0.5, 0.5, 1.0),), dtype=jnp.float64)
    tabulate = RationalComposedCellGeometryElement.tabulate

    values, gradients = eqx.filter_jit(tabulate)(element, points)
    automatic, _ = eqx.filter_jit(jax.jacfwd(tabulate, argnums=1))(element, points)
    expected_values, expected_gradients = source.tabulate(points)

    np.testing.assert_allclose(values, expected_values, atol=1.0e-14)
    np.testing.assert_allclose(gradients, expected_gradients, atol=1.0e-14)
    np.testing.assert_allclose(automatic[:, :, 0, :], expected_gradients, atol=1.0e-14)


def test_pyramid_unique_value_evaluates_without_nonunique_exact_gradient() -> None:
    source = lagrange_element("hexahedron", 1)
    unit: Polynomial = {(0, 0, 0): Fraction(1)}
    one_minus_z: Polynomial = {
        (0, 0, 0): Fraction(1),
        (0, 0, 1): Fraction(-1),
    }
    chart = (
        RationalPolynomial({(1, 0, 0): Fraction(1)}, unit),
        RationalPolynomial({(0, 1, 0): Fraction(1)}, unit),
        RationalPolynomial(
            {
                (0, 0, 0): Fraction(1, 4),
                (1, 0, 0): Fraction(-1),
                (2, 0, 0): Fraction(1),
                (0, 0, 1): Fraction(1),
                (0, 0, 2): Fraction(-1),
            },
            one_minus_z,
        ),
    )
    element = RationalComposedCellGeometryElement(source, "pyramid", chart)
    pyramid_vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.5, 1.0),
        ),
        dtype=np.float64,
    )
    mesh = CellMesh(
        pyramid_vertices,
        (
            CellBlock(
                "pyramids",
                "pyramid",
                np.asarray(((0, 1, 2, 3, 4),), dtype=np.int32),
                global_ids=np.asarray((31,), dtype=np.int64),
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"pyramids": element},
        {"pyramids": np.arange(8, dtype=np.int32)[None, :]},
        np.asarray(source.reference_nodes, dtype=np.float64),
    )
    apex = (Fraction(1, 2), Fraction(1, 2), Fraction(1))

    actual = geometry.evaluate(
        mesh,
        np.asarray((31,), dtype=np.int64),
        np.asarray(((0.5, 0.5, 1.0),)),
    )
    exact = geometry.evaluate_exact(
        mesh, np.asarray((31,), dtype=np.int64), ((apex,),)
    )

    np.testing.assert_allclose(actual, (((0.5, 0.5, 1.0),),), atol=1.0e-14)
    assert exact == ((apex,),)
    with pytest.raises(ValueError, match="pyramid-apex continuation"):
        element.tabulate(
            jnp.asarray(((0.5, 0.5, 1.0),), dtype=jnp.float64)
        )


def test_hybrid_pyramid_source_basis_has_exact_nodal_conversion() -> None:
    element = lagrange_element("pyramid", 1)
    basis = coordinate_source_basis(element)
    nodes = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(element.reference_nodes)
    )

    def evaluate_basis(
        expression: Any, point: tuple[Fraction, ...]
    ) -> Any:
        if isinstance(expression, RationalPolynomial) and not evaluate_expression(
            expression.denominator, point
        ):
            normalized = normalize_pyramid_reference_expression(expression)
            assert normalized.apex_value is not None
            return normalized.apex_value
        return evaluate_expression(expression, point)

    for node_index, node in enumerate(nodes):
        values = tuple(evaluate_basis(expression, node) for expression in basis)
        assert values == tuple(
            Fraction(int(basis_index == node_index))
            for basis_index in range(len(basis))
        )



def test_rational_reference_composition_retains_exact_geometry_and_validity() -> None:
    inverse = RationalPolynomial(
        {(1,): Fraction(1)}, {(0,): Fraction(1), (1,): Fraction(1)}
    )
    element = RationalComposedCellGeometryElement(
        _rational_chart(), "interval", (inverse,)
    )
    coordinates = np.asarray(((0.0,), (1.0,)), dtype=np.float64)
    cell_ids = np.asarray((37,), dtype=np.int64)
    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                "cells",
                "interval",
                np.asarray(((0, 1),), dtype=np.int64),
                global_ids=cell_ids,
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"cells": element},
        {"cells": np.asarray(((0, 1),), dtype=np.int64)},
        coordinates,
    )

    values = geometry.evaluate_exact(mesh, cell_ids, (((Fraction(1, 3),),),))
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)

    assert values == (((Fraction(1, 3),),),)
    np.testing.assert_array_equal(validity.status, (CellValidityStatus.CERTIFIED_VALID,))
    np.testing.assert_allclose(validity.determinant_lower, (1.0,), atol=1.0e-14)
    np.testing.assert_allclose(validity.determinant_upper, (1.0,), atol=1.0e-14)


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "compiled"))
def test_rational_chart_pole_refuses_gradient_only_consumption(compiled: bool) -> None:
    element = _rational_chart()
    points = jnp.asarray(((1.0,),), dtype=jnp.float64)

    def gradient(element: RationalComposedCellGeometryElement, points: Array) -> Array:
        return element.tabulate(points)[1]

    evaluate = eqx.filter_jit(gradient) if compiled else gradient
    error = eqx.EquinoxRuntimeError if compiled else ValueError
    with pytest.raises(error):
        evaluate(element, points).block_until_ready()


def test_interior_restriction_uses_exact_cell_map_without_vertex_parentage() -> None:
    corners = (
        (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)),
        (Fraction(1, 2), Fraction(1, 3), Fraction(1, 3)),
        (Fraction(1, 3), Fraction(1, 2), Fraction(1, 3)),
        (Fraction(1, 3), Fraction(1, 3), Fraction(1, 2)),
    )
    cells = {"child": np.asarray((17,), dtype=np.int64)}
    vertices = {"child": np.full((1, 4), -1, dtype=np.int64)}
    witness = ReferenceCellMapWitnesses(
        np.asarray((0,), dtype=np.int64),
        cells["child"],
        "tetrahedron",
        "hexahedron",
        np.asarray((corners,), dtype=np.float64),
        (corners,),
    )
    source = CellGeometryRestrictionSource(
        "source-geometry",
        "source-topology",
        cells,
        vertices,
        block_parent_reference_maps={"child": witness},
    )
    retained = source.parent_reference_maps[0]
    assert retained is not None
    assert exact_reference_cell_vertices(retained) == (corners,)
    np.testing.assert_array_equal(source.parent_vertex_ids[0], ((-1, -1, -1, -1),))
    with pytest.raises(ValueError):
        CellGeometryRestrictionSource(
            "source-geometry", "source-topology", cells, vertices
        )
