from fractions import Fraction

import numpy as np

from phydrax.discretization._cell_geometry import (
    CellGeometrySpec,
    SplineCellGeometryElement,
)
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._coordinate_enclosure import (
    add,
    affine_arguments,
    axes,
    compose,
    constant,
    coordinate_source_basis,
    evaluate_expression,
    expression_equal,
    expression_sequence_equal,
    polynomial_bounds,
    RationalPolynomial,
    subtract,
)
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.discretization.fem._reference import FiniteElementSpec


def test_exact_bernstein_bounds_tighten_under_source_chart_restriction() -> None:
    polynomial = add(
        {
            (0,): Fraction(1, 4),
            (1,): Fraction(-1),
            (2,): Fraction(1),
        },
        constant(Fraction(1, 100), 1),
    )
    root_bounds = polynomial_bounds(polynomial, "box", 1)
    assert root_bounds[0] < 0.0

    arguments = affine_arguments(
        np.asarray((0.4375,), dtype=np.float64),
        np.asarray(((0.125,),), dtype=np.float64),
    )
    restricted = compose(polynomial, arguments)
    lower, upper = polynomial_bounds(restricted, "box", 1)

    assert lower > 0.0
    assert lower <= upper
    assert evaluate_expression(restricted, (Fraction(1, 2),)) == Fraction(1, 100)


def test_rational_expression_equality_is_cross_multiplication() -> None:
    (u,) = axes(1)
    denominator = add(constant(1, 1), u)
    quotient = RationalPolynomial(u, denominator)
    equivalent = subtract(constant(1, 1), RationalPolynomial(constant(1, 1), denominator))

    assert expression_equal(quotient, equivalent)
    lower, upper = polynomial_bounds(quotient, "box", 1)
    assert np.isfinite(lower) and np.isfinite(upper)
    assert lower <= 0.0 <= 0.5 <= upper


def test_rational_pole_returns_unbounded_enclosure() -> None:
    (u,) = axes(1)
    denominator = subtract(u, constant(Fraction(1, 2), 1))

    lower, upper = polynomial_bounds(
        RationalPolynomial(constant(1, 1), denominator), "box", 1
    )

    assert lower == float("-inf")
    assert upper == float("inf")


def test_spline_coordinate_basis_uses_retained_exact_weight_bank() -> None:
    root_two = RadicalElement.sqrt(Fraction(2))
    shadow = float(root_two)
    anchors = (11, 12)
    extraction = (
        (Fraction(1), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
    )

    algebraic = SplineCellGeometryElement(
        anchors,
        (Fraction(0), Fraction(1), Fraction(0), Fraction(1)),
        extraction,
        (shadow, 1.0),
        "source",
        "revision",
        degrees=(1, 1),
        exact_weights=(root_two, Fraction(1)),
    )
    dyadic = SplineCellGeometryElement(
        anchors,
        (Fraction(0), Fraction(1), Fraction(0), Fraction(1)),
        extraction,
        (shadow, 1.0),
        "source",
        "revision",
        degrees=(1, 1),
        exact_weights=(Fraction(shadow), Fraction(1)),
    )

    assert algebraic.exact_weights == (root_two, Fraction(1))
    assert algebraic.element_id != dyadic.element_id
    assert not expression_sequence_equal(
        coordinate_source_basis(algebraic), coordinate_source_basis(dyadic)
    )


def test_affine_trimmed_tetra_uses_canonical_zero_form_basis() -> None:
    mesh = CellMesh.from_simplices(
        np.asarray(
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)),
            dtype=np.float64,
        ),
        np.asarray(((0, 1, 2, 3),), dtype=np.int64),
        dimension=3,
    )
    geometry = CellGeometrySpec.affine(mesh)
    element = geometry.elements[0]
    assert isinstance(element, FiniteElementSpec)
    assert element.family == "trimmed"
    assert element.form_basis is not None
    assert element.form_basis.form_degree == 0

    references = (
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
            (Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)),
        ),
    )
    cell_id = int(mesh.blocks[0].global_ids[0])
    actual = geometry.evaluate_exact(
        mesh, np.asarray((cell_id,), dtype=np.int64), references
    )
    expected = (
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(2), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(3), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(4)),
            (Fraction(1, 2), Fraction(3, 4), Fraction(1)),
        ),
    )

    assert actual == expected
